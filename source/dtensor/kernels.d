/**
 * Device-side kernels for dtensor operations.
 */
@compute(CompileFor.deviceOnly)
module dtensor.kernels;

pragma(LDC_no_moduleinfo);

import ldc.dcompute;
import dcompute.std.index;
import dcompute.std.sync : barrier;
import dcompute.std.ndview;
import std.meta : AliasSeq;

pragma(LDC_inline_ir)
    R inlineIR(string s, R, P...)(P);

/// Returns a pointer to dynamic shared memory at offset 0.
T* getSharedBase(T)()
{
    return inlineIR!(`
        %gen = call i64 asm "cvta.shared.u64 $0, $1;", "=l,l"(i64 0)
        %ptr = inttoptr i64 %gen to ptr
        ret ptr %ptr
    `, T*)();
}

/// Identity value for tree reduction out-of-bounds padding.
template identityValue(string op, T)
{
    static if (op == "sum")
        enum T identityValue = cast(T) 0;
    else static if (op == "max")
    {
        static if (is(T == float))
            enum T identityValue = -float.infinity;
        else static if (is(T == double))
            enum T identityValue = -double.infinity;
        else static if (is(T == int))
            enum T identityValue = int.min;
        else
            static assert(false, "unsupported type for max: " ~ T.stringof);
    }
    else static if (op == "min")
    {
        static if (is(T == float))
            enum T identityValue = float.infinity;
        else static if (is(T == double))
            enum T identityValue = double.infinity;
        else static if (is(T == int))
            enum T identityValue = int.max;
        else
            static assert(false, "unsupported type for min: " ~ T.stringof);
    }
    else
        static assert(false, "unsupported op: " ~ op);
}

/// Generic elementwise binary kernel templated on `op` ("+", "-", "*", "/").
/// Supports tensor-tensor (both NdView!(T, N)), tensor-scalar, and scalar-tensor.
@kernel() void elementwiseOp(string op, T, size_t N, Lhs, Rhs)(NdView!(T, N) res, Lhs a, Rhs b)
{
    auto i = GlobalIndex.x;
    if (i >= res.length) return;

    static if (is(Lhs == NdView!(T, N)) && is(Rhs == NdView!(T, N)))
        res.atAssign(i, mixin("a.at(i) " ~ op ~ " b.at(i)"));
    else static if (is(Lhs == NdView!(T, N)) && is(Rhs == T))
        res.atAssign(i, mixin("a.at(i) " ~ op ~ " b"));
    else static if (is(Lhs == T) && is(Rhs == NdView!(T, N)))
        res.atAssign(i, mixin("a " ~ op ~ " b.at(i)"));
    else
        static assert(false, "unsupported operand types");
}

/// Tree reduction kernel across a thread block using shared memory.
/// Each block of 256 threads reduces up to 256 elements of `input` and writes the
/// block result to `res.atAssign(GroupIndex.x, ...)`.
@kernel() void reduceBlock(string op, T, size_t N)(NdView!(T, 1) res, NdView!(T, N) input)
{
    T* sdata = getSharedBase!T();
    auto tid = SharedIndex.x;
    auto gid = GlobalIndex.x;

    sdata[tid] = (gid < input.length) ? input.at(gid) : identityValue!(op, T);
    barrier();

    for (size_t s = 128; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            static if (op == "sum")
                sdata[tid] += sdata[tid + s];
            else static if (op == "max")
            {
                if (sdata[tid + s] > sdata[tid])
                    sdata[tid] = sdata[tid + s];
            }
            else static if (op == "min")
            {
                if (sdata[tid + s] < sdata[tid])
                    sdata[tid] = sdata[tid + s];
            }
            else
                static assert(false, "unsupported op: " ~ op);
        }
        barrier();
    }

    if (tid == 0)
    {
        res.atAssign(GroupIndex.x, sdata[0]);
    }
}

/// Reduction along a single axis of an N-dimensional tensor.
/// One thread per output element loops over `input.shape[axis]`.
@kernel() void reduceAxis(string op, T, size_t InRank, size_t OutRank)(NdView!(T, OutRank) res, NdView!(T, InRank) input, size_t axis)
{
    auto i = GlobalIndex.x;
    if (i >= res.length) return;

    size_t[OutRank] outCoords;
    size_t rem = i;
    static foreach_reverse (d; 1 .. OutRank)
    {
        outCoords[d] = rem % res.shape[d];
        rem /= res.shape[d];
    }
    outCoords[0] = rem;

    size_t baseOffset = 0;
    size_t outDim = 0;
    foreach (d; 0 .. InRank)
    {
        if (d != axis)
        {
            baseOffset += outCoords[outDim] * input.strides[d];
            outDim++;
        }
    }

    size_t count = input.shape[axis];
    if (count == 0) return;
    size_t stride = input.strides[axis];
    size_t offset = baseOffset;

    T acc = input.ptr[offset];
    offset += stride;

    foreach (k; 1 .. count)
    {
        T val = input.ptr[offset];
        static if (op == "sum" || op == "mean")
            acc += val;
        else static if (op == "max")
        {
            if (val > acc) acc = val;
        }
        else static if (op == "min")
        {
            if (val < acc) acc = val;
        }
        else
            static assert(false, "unsupported reduction op: " ~ op);
        offset += stride;
    }

    static if (op == "mean")
        acc = cast(T)(acc / cast(T) count);

    res.atAssign(i, acc);
}

alias Ops = AliasSeq!("+", "-", "*", "/");
alias OpNames = AliasSeq!("add", "sub", "mul", "div");

template InstantiateKernels(string name, string op, T, size_t N)
{
    alias V = NdView!(T, N);
    mixin("alias " ~ name ~ "_tt = elementwiseOp!(op, T, N, V, V);\n"
        ~ "alias " ~ name ~ "_ts = elementwiseOp!(op, T, N, V, T);\n"
        ~ "alias " ~ name ~ "_st = elementwiseOp!(op, T, N, T, V);\n");
}

// Instantiate common elementwise kernels at build time so LDC emits PTX entries for them:
static foreach (i, op; Ops)
{
    static foreach (T; AliasSeq!(float, double, int))
    {
        static foreach (N; 1 .. 5)
        {
            mixin InstantiateKernels!("_k_" ~ OpNames[i] ~ "_" ~ T.stringof ~ "_" ~ N.stringof, op, T, N);
        }
    }
}

alias ReductionOps = AliasSeq!("sum", "max", "min");
alias AxisOps = AliasSeq!("sum", "max", "min", "mean");

// Pre-instantiate block reduction kernels:
static foreach (op; ReductionOps)
{
    static foreach (T; AliasSeq!(float, double, int))
    {
        static foreach (N; 1 .. 5)
        {
            mixin("alias _k_reduceBlock_" ~ op ~ "_" ~ T.stringof ~ "_" ~ N.stringof ~
                  " = reduceBlock!(op, T, N);");
        }
    }
}

// Pre-instantiate axis reduction kernels:
static foreach (op; AxisOps)
{
    static foreach (T; AliasSeq!(float, double, int))
    {
        static foreach (InRank; 2 .. 5)
        {
            mixin("alias _k_reduceAxis_" ~ op ~ "_" ~ T.stringof ~ "_" ~ InRank.stringof ~
                  " = reduceAxis!(op, T, InRank, InRank - 1);");
        }
    }
}
