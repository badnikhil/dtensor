/**
 * Device-side kernels for dtensor operations.
 */
@compute(CompileFor.deviceOnly)
module dtensor.kernels;

pragma(LDC_no_moduleinfo);

import ldc.dcompute;
import dcompute.std.index;
import dcompute.std.ndview;
import std.meta : AliasSeq;

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

alias Ops = AliasSeq!("+", "-", "*", "/");
alias OpNames = AliasSeq!("add", "sub", "mul", "div");

template InstantiateKernels(string name, string op, T, size_t N)
{
    alias V = NdView!(T, N);
    mixin("alias " ~ name ~ "_tt = elementwiseOp!(op, T, N, V, V);\n"
        ~ "alias " ~ name ~ "_ts = elementwiseOp!(op, T, N, V, T);\n"
        ~ "alias " ~ name ~ "_st = elementwiseOp!(op, T, N, T, V);\n");
}

// Instantiate common kernels at build time so LDC emits PTX entries for them:
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
