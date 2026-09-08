/**
 * Host-side `Tensor` container over dcompute device memory.
 *
 * Strides are in elements, not bytes. Layout is row-major by default.
 * `reshape` and `transpose` are stride views and never move data.
 * Lifetime is dcompute's: `Buffer!T` refcounts the allocation, so `Tensor`
 * has no destructor of its own.
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

/// An N-dimensional tensor whose elements live in device memory.
struct Tensor(T, size_t N)
if (N >= 1)
{
    /// The device allocation.
    Buffer!T store;
    /// Extent of each axis.
    size_t[N] shape;
    /// Distance between consecutive elements along each axis, in elements.
    size_t[N] strides;

    /// Allocates `shape` worth of device memory, C-contiguous.
    this(size_t[N] shape)
    {
        // One cuMemAlloc per Tensor. A size-bucketed pool, or cuMemAllocAsync
        // once dcompute exposes it, would avoid serialising on the driver.
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

    /// Reinterprets the same memory with a new shape. Contiguous tensors only.
    Tensor!(T, M) reshape(size_t M)(size_t[M] newShape)
    {
        assert(contiguous,
            "reshape of a non-contiguous view would need a copy; make it contiguous first");
        assert(elementCount(newShape) == length, "reshape must preserve the element count");
        Tensor!(T, M) r;
        r.store = store;
        r.shape = newShape;
        r.strides = contiguousStrides(newShape);
        return r;
    }

    /// Permutes the axes. `perm[i]` is the axis of `this` becoming axis `i`.
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

    /// Reverses every axis.
    Tensor transpose()
    {
        size_t[N] p;
        foreach (i; 0 .. N)
            p[i] = N - 1 - i;
        return transpose(p);
    }

    /// Copies the allocation to the host and returns it as an ndslice
    /// carrying this tensor's shape and strides.
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

}

/// Uploads a host ndslice into a fresh C-contiguous `Tensor`, reading the
/// source in row-major order.
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

version (unittest)
{
    import mir.ndslice.slice : sliced;
    import mir.ndslice.topology : iota, as;
    import mir.ndslice.allocation : slice;
}

unittest
{
    auto src = iota(3, 4).as!float.slice;
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

unittest
{
    auto t = fromNdslice(iota(2, 6).as!float.slice);
    immutable base = t.store.raw;

    auto r = t.reshape([3LU, 2LU, 2LU]);
    assert(r.shape == [3LU, 2LU, 2LU]);
    assert(r.strides == [4LU, 2LU, 1LU]);
    assert(r.store.raw == base, "reshape must not move data");
    assert(t.store.raw == base, "reshape must not disturb the source");

    auto flat = t.toNdslice();
    auto cube = r.toNdslice();
    foreach (k; 0 .. 12)
        assert(cube[k / 4, (k % 4) / 2, k % 2] == flat[k / 6, k % 6]);
}

unittest
{
    auto src = iota(2, 3).as!float.slice;
    auto t = fromNdslice(src);
    immutable base = t.store.raw;

    auto tr = t.transpose();
    assert(tr.shape == [3LU, 2LU]);
    assert(tr.strides == [1LU, 3LU]);
    assert(!tr.contiguous);
    assert(tr.store.raw == base, "transpose must not move data");

    auto back = tr.toNdslice();
    foreach (i; 0 .. 3)
        foreach (j; 0 .. 2)
            assert(back[i, j] == src[j, i]);

    auto same = t.transpose([1LU, 0LU]);
    assert(same.shape == tr.shape && same.strides == tr.strides);
    assert(tr.offsetOf([2LU, 1LU]) == 2 * 1 + 1 * 3);

    auto c = fromNdslice(iota(2, 3, 4).as!float.slice);
    auto p = c.transpose([2LU, 0LU, 1LU]);
    assert(p.shape == [4LU, 2LU, 3LU]);
    assert(p.strides == [1LU, 12LU, 4LU]);
}

unittest
{
    import std.traits : hasElaborateDestructor;

    {
        auto t = Tensor!(float, 2)([4LU, 4LU]);
        auto a = t.reshape([16LU]);
        auto b = t.transpose();
        assert(a.store.raw == t.store.raw && b.store.raw == t.store.raw);
    }

    auto probe = Tensor!(float, 1)([8LU]);
    assert(probe.store.raw != 0);

    static if (hasElaborateDestructor!(Buffer!float))
    {
        // 4 GiB of churn through a 4 GiB card: a leak throws outOfMemory.
        enum elems = 16 * 1024 * 1024;
        foreach (i; 0 .. 64)
        {
            auto t = Tensor!(float, 1)([cast(size_t) elems]);
            assert(t.store.raw != 0);
        }
    }
    else
    {
        pragma(msg, "dtensor: this dcompute has no Buffer destructor, "
            ~ "so the churn test is skipped and Tensor leaks device memory.");
    }
}
