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
