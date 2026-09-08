/**
 * dtensor.tensor — the host-side `Tensor` container.
 *
 * `Tensor!(T, N)` is a host handle onto an N-dimensional block of *device*
 * memory. It owns nothing of its own: the allocation lives in dcompute's
 * `Buffer!T` and its lifetime is `Buffer`'s (see `Lifetime` below).
 *
 * Conventions locked in here — the rest of the tensor stack depends on them:
 *
 * $(UL
 * $(LI **Strides are counted in ELEMENTS, never bytes.** This matches
 *      `mir.ndslice` and is the contract dcompute's forthcoming
 *      `NdView!(T, N)` uses.)
 * $(LI **Row-major (C order) is the default layout**, as produced by
 *      `contiguousStrides`. Nothing forbids other layouts — `transpose`
 *      produces one — but a freshly allocated `Tensor` is C-contiguous.)
 * $(LI **`reshape` and `transpose` are pure stride views.** They never touch
 *      device memory; they hand back a new `Tensor` sharing the same
 *      `Buffer`.)
 * )
 *
 * $(H3 Lifetime)
 *
 * There is exactly ONE ownership scheme in this package, and it is not this
 * package's: dcompute's `Buffer!T` refcounts its `CUdeviceptr` (postblit
 * retains, destructor releases, last owner calls `cuMemFree`) — upstream PR
 * #106 / branch `harden/raii-destructors-v2`. `Tensor` therefore has no
 * destructor, no `release()`, and no refcount of its own: it stores a
 * `Buffer!T` by value and lets the compiler-generated field destructor do the
 * work. Views made by `reshape`/`transpose` copy the `Buffer`, which bumps its
 * refcount, so the allocation outlives every view of it and is freed exactly
 * once.
 *
 * Building against a dcompute that predates PR #106 leaves the allocation
 * un-freed. That is an upstream gap, deliberately not papered over here — a
 * second ownership scheme layered on top is how double-frees are born.
 *
 * $(H3 Where dcompute's NdView plugs in)
 *
 * `Tensor` is the host side; kernels need a POD, `@compute`-legal device-side
 * view. When dcompute lands `NdView!(T, N)` — `{GlobalPointer!T ptr;
 * size_t[N] shape, strides;}` — the seam is `Tensor.deviceView()` below: the
 * three fields it needs are already in exactly the right units. Nothing else
 * in this module changes.
 */
module dtensor.tensor;

import dcompute.driver.cuda;

import mir.ndslice.slice : Slice, SliceKind, Universal;

/// Row-major (C-order) strides for `shape`, in ELEMENTS.
size_t[N] contiguousStrides(size_t N)(const size_t[N] shape) @safe pure nothrow @nogc
{
    size_t[N] s;
    size_t acc = 1;
    foreach_reverse (i; 0 .. N)
    {
        s[i] = acc;
        acc *= shape[i];
    }
    return s;
}

/// Number of elements described by `shape`.
size_t elementCount(size_t N)(const size_t[N] shape) @safe pure nothrow @nogc
{
    size_t n = 1;
    foreach (d; shape)
        n *= d;
    return n;
}

@safe pure nothrow @nogc unittest
{
    assert(contiguousStrides([2LU, 3LU, 4LU]) == [12LU, 4LU, 1LU]);
    assert(contiguousStrides([5LU]) == [1LU]);
    assert(elementCount([2LU, 3LU, 4LU]) == 24);
}

/**
 * An N-dimensional tensor whose elements live in device memory.
 *
 * Params:
 *   T = element type
 *   N = rank (compile-time; see TENSOR_ROADMAP §6 — dtype and rank are in the
 *       type system, not type-erased)
 */
struct Tensor(T, size_t N)
if (N >= 1)
{
    /// The device allocation. Owns it; see the module-level `Lifetime` note.
    Buffer!T store;
    /// Extent of each axis.
    size_t[N] shape;
    /// Distance between consecutive elements along each axis, **in elements**.
    size_t[N] strides;

    /// Allocates `shape` worth of device memory, C-contiguous.
    this(size_t[N] shape)
    {
        // ponytail: plain cuMemAlloc per Tensor. A caching pool allocator is the
        // known ceiling — allocation-heavy workloads (per-op temporaries in a
        // fused expression, training loops) will serialise on the driver, which
        // synchronises on every cuMemAlloc/cuMemFree. Upgrade path is a
        // size-bucketed free-list, or dcompute exposing cuMemAllocAsync +
        // cuMemPool* (ROADMAP B12); neither changes this signature.
        ensureInit();
        this.shape = shape;
        this.strides = contiguousStrides(shape);
        this.store = Buffer!T(elementCount(shape));
    }

    /// Total number of elements.
    @property size_t length() const @safe pure nothrow @nogc
    {
        return elementCount(shape);
    }

    /// True when the strides are the row-major ones for this shape.
    @property bool contiguous() const @safe pure nothrow @nogc
    {
        return strides == contiguousStrides(shape);
    }

    /// Offset of `idx`, in elements, from the start of the allocation.
    size_t offsetOf(const size_t[N] idx) const @safe pure nothrow @nogc
    {
        size_t o = 0;
        foreach (i; 0 .. N)
        {
            assert(idx[i] < shape[i], "index out of bounds");
            o += idx[i] * strides[i];
        }
        return o;
    }

    /**
     * Reinterprets the same device memory with a new shape. Pure stride
     * arithmetic — no allocation, no copy, no kernel.
     *
     * Only defined for contiguous tensors: reshaping a strided view generally
     * requires materialising a copy, which is a data-movement operation and
     * does not belong behind a name that promises a view.
     */
    Tensor!(T, M) reshape(size_t M)(size_t[M] newShape)
    {
        assert(contiguous,
            "reshape of a non-contiguous view would need a copy; make it contiguous first");
        assert(elementCount(newShape) == length, "reshape must preserve the element count");
        Tensor!(T, M) r;
        r.store = store; // shares the allocation; Buffer's refcount keeps it alive
        r.shape = newShape;
        r.strides = contiguousStrides(newShape);
        return r;
    }

    /**
     * Permutes the axes. Pure stride arithmetic — no data movement.
     *
     * `perm[i]` names the axis of `this` that becomes axis `i` of the result.
     */
    Tensor transpose(const size_t[N] perm)
    {
        version (assert)
        {
            bool[N] seen;
            foreach (p; perm)
            {
                assert(p < N, "permutation axis out of range");
                assert(!seen[p], "permutation repeats an axis");
                seen[p] = true;
            }
        }
        Tensor r;
        r.store = store;
        foreach (i; 0 .. N)
        {
            r.shape[i] = shape[perm[i]];
            r.strides[i] = strides[perm[i]];
        }
        return r;
    }

    /// NumPy's `.T` — reverses every axis.
    Tensor transpose()
    {
        size_t[N] p;
        foreach (i; 0 .. N)
            p[i] = N - 1 - i;
        return transpose(p);
    }

    /**
     * Host view of the tensor's contents.
     *
     * Copies the whole allocation device→host once and returns a `Universal`
     * ndslice carrying this tensor's shape and strides, so a transposed or
     * reshaped view round-trips with the right element order and no device-side
     * gather. The returned slice points at freshly allocated GC memory and is
     * a snapshot: later writes to the device are not reflected.
     */
    Slice!(T*, N, Universal) toNdslice()
    {
        auto host = new T[length];
        store.hostMemory = host;
        store.copy!(Copy.deviceToHost);

        Slice!(T*, N, Universal) s;
        s._structure[0] = shape;
        foreach (i; 0 .. N)
            s._structure[1][i] = cast(ptrdiff_t) strides[i];
        s._iterator = host.ptr;
        return s;
    }

    // ---- SEAM: dcompute's NdView!(T, N) plugs in here ----------------------
    //
    // NdView is device-side substrate and lives in dcompute (TENSOR_ROADMAP
    // §2), not here — a second {GlobalPointer!T ptr; size_t[N] shape, strides;}
    // defined in dtensor would be a competing type. When dcompute lands it,
    // this method is the whole change on our side:
    //
    //     NdView!(T, N) deviceView()
    //     {
    //         return NdView!(T, N)(cast(GlobalPointer!T) store.raw,
    //                              shape, strides);
    //     }
    //
    // The three fields are already in the units NdView wants — strides in
    // ELEMENTS — so nothing else in this module moves.
    // ------------------------------------------------------------------------
}

/**
 * Uploads a host ndslice of any kind into a fresh C-contiguous `Tensor`.
 *
 * The source is read in row-major order, so a `Universal` slice with exotic
 * strides is normalised on the way in.
 */
auto fromNdslice(Iterator, size_t N, SliceKind kind)(Slice!(Iterator, N, kind) src)
{
    import mir.ndslice.topology : flattened;
    import std.traits : Unqual;

    alias T = Unqual!(Slice!(Iterator, N, kind).DeepElement);

    auto t = Tensor!(T, N)(src.shape);
    auto host = new T[t.length];
    size_t i;
    foreach (e; src.flattened)
        host[i++] = e;
    assert(i == host.length);

    t.store.hostMemory = host;
    t.store.copy!(Copy.hostToDevice);
    return t;
}

// ---------------------------------------------------------------------------
// Tests
//
// Everything below needs a CUDA *driver* and a device, because a Tensor owns
// device memory. None of it needs the embedded-PTX / `launch!` pipeline: no
// kernel is compiled or launched here, so these run even while the on-GPU
// kernel path is unverified. The pure shape/stride unittest above needs no GPU
// at all.
// ---------------------------------------------------------------------------
version (unittest)
{
    import mir.ndslice.slice : sliced;
    import mir.ndslice.topology : iota, as;
    import mir.ndslice.allocation : slice;
}

/// Tensor <-> ndslice round-trip preserves values.
unittest
{
    auto src = iota(3, 4).as!float.slice;   // 0 .. 11, contiguous host data
    auto t = fromNdslice(src);

    static assert(is(typeof(t) == Tensor!(float, 2)));
    assert(t.shape == [3LU, 4LU]);
    assert(t.strides == [4LU, 1LU]);
    assert(t.contiguous);

    auto back = t.toNdslice();
    foreach (i; 0 .. 3)
        foreach (j; 0 .. 4)
            assert(back[i, j] == src[i, j]);
}

/// reshape is a pure stride view: right shape, right strides, same allocation.
unittest
{
    auto t = fromNdslice(iota(2, 6).as!float.slice);
    immutable base = t.store.raw;

    auto r = t.reshape([3LU, 2LU, 2LU]);
    assert(r.shape == [3LU, 2LU, 2LU]);
    assert(r.strides == [4LU, 2LU, 1LU]);
    assert(r.store.raw == base, "reshape must not move data");
    assert(t.store.raw == base, "reshape must not disturb the source");

    // Same 12 elements, same row-major order, just re-shaped.
    auto flat = t.toNdslice();
    auto cube = r.toNdslice();
    foreach (k; 0 .. 12)
        assert(cube[k / 4, (k % 4) / 2, k % 2] == flat[k / 6, k % 6]);
}

/// transpose is a pure stride view: axes and strides permute, data does not.
unittest
{
    auto src = iota(2, 3).as!float.slice;
    auto t = fromNdslice(src);
    immutable base = t.store.raw;

    auto tr = t.transpose();                 // NumPy .T
    assert(tr.shape == [3LU, 2LU]);
    assert(tr.strides == [1LU, 3LU]);        // permuted, NOT recomputed
    assert(!tr.contiguous);
    assert(tr.store.raw == base, "transpose must not move data");

    auto back = tr.toNdslice();
    foreach (i; 0 .. 3)
        foreach (j; 0 .. 2)
            assert(back[i, j] == src[j, i]);

    // Explicit permutation, and the offset model that goes with it.
    auto same = t.transpose([1LU, 0LU]);
    assert(same.shape == tr.shape && same.strides == tr.strides);
    assert(tr.offsetOf([2LU, 1LU]) == 2 * 1 + 1 * 3);

    // A rank-3 permutation, to prove nothing is hard-coded for N == 2.
    auto c = fromNdslice(iota(2, 3, 4).as!float.slice);
    auto p = c.transpose([2LU, 0LU, 1LU]);
    assert(p.shape == [4LU, 2LU, 3LU]);
    assert(p.strides == [1LU, 12LU, 4LU]);
}

/// Allocation and destruction neither leak nor double-free.
unittest
{
    import std.traits : hasElaborateDestructor;

    // Views share one allocation; the copies must not each free it.
    {
        auto t = Tensor!(float, 2)([4LU, 4LU]);
        auto a = t.reshape([16LU]);
        auto b = t.transpose();
        assert(a.store.raw == t.store.raw && b.store.raw == t.store.raw);
    } // a, b, t all die here — exactly one cuMemFree must happen

    // Still usable afterwards: a double-free would have poisoned the context.
    auto probe = Tensor!(float, 1)([8LU]);
    assert(probe.store.raw != 0);

    static if (hasElaborateDestructor!(Buffer!float))
    {
        // 64 x 64 MiB = 4 GiB of churn through a 4 GiB card. Leak even one and
        // cuMemAlloc throws outOfMemory long before the loop ends.
        enum elems = 16 * 1024 * 1024;      // 64 MiB of float
        foreach (i; 0 .. 64)
        {
            auto t = Tensor!(float, 1)([cast(size_t) elems]);
            assert(t.store.raw != 0);
        }
    }
    else
    {
        pragma(msg, "dtensor: this dcompute has no Buffer destructor "
            ~ "(pre-PR #106) — the leak-churn test is skipped, and Tensor "
            ~ "leaks device memory until #106 lands.");
    }
}
