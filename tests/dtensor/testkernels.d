/**
 * Device-side test kernels for verifying `deviceView()` interop with dcompute.
 */
@compute(CompileFor.deviceOnly)
module dtensor.testkernels;

pragma(LDC_no_moduleinfo);

import ldc.dcompute;
import dcompute.std.index;
import dcompute.std.ndview;

/// Scales each element of a 2-D view: y = x * factor.
@kernel() void tensorScale2D(NdView!(float, 2) y, NdView!(float, 2) x, float factor)
{
    auto i = GlobalIndex.x;
    if (i >= x.length) return;
    y.atAssign(i, x.at(i) * factor);
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

// Instantiate common kernels at build time so LDC emits PTX entries for them:
template TypeTuple(T...) { alias TypeTuple = T; }

static foreach (op; ["+", "-", "*", "/"])
{
    static foreach (T; TypeTuple!(float, double, int))
    {
        static foreach (N; 1 .. 5)
        {
            mixin("alias _k_" ~ (op == "+" ? "add" : op == "-" ? "sub" : op == "*" ? "mul" : "div")
                ~ "_" ~ T.stringof ~ "_" ~ N.stringof ~ "_tt = elementwiseOp!(op, T, N, NdView!(T, N), NdView!(T, N));");
            mixin("alias _k_" ~ (op == "+" ? "add" : op == "-" ? "sub" : op == "*" ? "mul" : "div")
                ~ "_" ~ T.stringof ~ "_" ~ N.stringof ~ "_ts = elementwiseOp!(op, T, N, NdView!(T, N), T);");
            mixin("alias _k_" ~ (op == "+" ? "add" : op == "-" ? "sub" : op == "*" ? "mul" : "div")
                ~ "_" ~ T.stringof ~ "_" ~ N.stringof ~ "_st = elementwiseOp!(op, T, N, T, NdView!(T, N));");
        }
    }
}
