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
import dcompute.driver.cuda.rc : Buffer;
public import dcompute.std.ndview : NdView;
import ldc.dcompute : GlobalPointer;

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

/// Thrown when tensor shapes are incompatible for broadcasting.
class ShapeMismatchException : Exception
{
    this(string msg, string file = __FILE__, size_t line = __LINE__) pure nothrow @safe
    {
        super(msg, file, line);
    }
}

/// Computes the broadcast output shape for two tensor shapes according to NumPy rules.
/// Shapes are right-aligned. Dimensions are compatible if they are equal or one of them is 1.
/// Throws `ShapeMismatchException` if shapes cannot be broadcast.
size_t[(Na >= Nb ? Na : Nb)] broadcastShapes(size_t Na, size_t Nb)(const size_t[Na] shapeA, const size_t[Nb] shapeB)
{
    enum size_t N = Na >= Nb ? Na : Nb;
    size_t[N] outShape;
    foreach (k; 0 .. N)
    {
        size_t dimA = (k < N - Na) ? 1 : shapeA[k - (N - Na)];
        size_t dimB = (k < N - Nb) ? 1 : shapeB[k - (N - Nb)];

        if (dimA == dimB)
            outShape[k] = dimA;
        else if (dimA == 1)
            outShape[k] = dimB;
        else if (dimB == 1)
            outShape[k] = dimA;
        else
        {
            import std.format : format;
            throw new ShapeMismatchException(
                format("Cannot broadcast shapes %s and %s: incompatible dimensions %s and %s at axis %s",
                    shapeA, shapeB, dimA, dimB, k));
        }
    }
    return outShape;
}

/// Computes broadcast strides of rank `N` for a tensor of rank `Na` (`Na <= N`).
/// Size-1 dimensions and prepended dimensions use stride 0 so each index aliases the same element.
size_t[N] broadcastStrides(size_t Na, size_t N)(const size_t[Na] shapeA, const size_t[Na] stridesA) @safe pure nothrow @nogc
{
    size_t[N] outStrides;
    foreach (k; 0 .. N)
    {
        if (k < N - Na)
            outStrides[k] = 0;
        else
        {
            size_t orig = k - (N - Na);
            if (shapeA[orig] == 1)
                outStrides[k] = 0;
            else
                outStrides[k] = stridesA[orig];
        }
    }
    return outStrides;
}

/// Helper template detecting whether `X` is an instantiation of `Tensor`.
template isTensor(X)
{
    static if (is(X == Tensor!(U, M), U, size_t M))
        enum isTensor = true;
    else
        enum isTensor = false;
}

/// Returns an `NdView!(T, N)` with broadcast strides matching `outShape`.
NdView!(T, N) broadcastView(size_t N, T, size_t Na)(auto ref const Tensor!(T, Na) t, const size_t[N] outShape) pure nothrow @nogc
{
    size_t[N] bStrides = broadcastStrides!(Na, N)(t.shape, t.strides);
    return NdView!(T, N)(GlobalPointer!T(cast(T*) t.store.raw), outShape, bStrides);
}

@safe pure unittest
{
    assert(broadcastShapes([2LU, 3LU], [2LU, 3LU]) == [2LU, 3LU]);
    assert(broadcastShapes([3LU, 1LU], [1LU, 4LU]) == [3LU, 4LU]);
    assert(broadcastShapes([2LU, 3LU], [3LU]) == [2LU, 3LU]);
    assert(broadcastShapes([3LU], [2LU, 3LU]) == [2LU, 3LU]);
    assert(broadcastShapes([2LU, 1LU, 4LU], [3LU, 1LU]) == [2LU, 3LU, 4LU]);

    assert(broadcastStrides!(2, 2)([3LU, 1LU], [1LU, 1LU]) == [1LU, 0LU]);
    assert(broadcastStrides!(2, 2)([1LU, 4LU], [4LU, 1LU]) == [0LU, 1LU]);
    assert(broadcastStrides!(1, 2)([3LU], [1LU]) == [0LU, 1LU]);
}

unittest
{
    import std.exception : assertThrown;
    assertThrown!ShapeMismatchException(broadcastShapes([2LU, 3LU], [2LU, 4LU]));
    assertThrown!ShapeMismatchException(broadcastShapes([2LU, 3LU], [4LU]));
}

/// Helper to launch a kernel with dynamic shared memory.
private void launchKernel(alias k, Args...)(uint[3] grid, uint[3] block, uint sharedMem, Args args)
{
    if (Program.globalProgram.raw is null)
    {
        import std.traits : moduleName;
        ensureInit();
        Program.globalProgram = Program.fromModule!(moduleName!(__traits(parent, k)))();
    }
    defaultQueue().enqueue!k(grid, block, sharedMem)(args);
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

    /// Returns an `NdView!(T, N)` pointing to this tensor's device memory
    /// with the same shape and strides, suitable for passing to a `@kernel`.
    NdView!(T, N) deviceView() const pure nothrow @nogc
    {
        return NdView!(T, N)(GlobalPointer!T(cast(T*) store.raw), shape, strides);
    }

    /// Elementwise binary operations between two tensors (+, -, *, /) with NumPy broadcasting.
    Tensor!(T, (N >= Nb ? N : Nb)) opBinary(string op, size_t Nb)(auto ref const Tensor!(T, Nb) rhs) const
    if (op == "+" || op == "-" || op == "*" || op == "/")
    {
        import dtensor.kernels : elementwiseOp;

        enum size_t OutRank = N >= Nb ? N : Nb;
        const size_t[OutRank] outShape = broadcastShapes!(N, Nb)(this.shape, rhs.shape);
        auto res = Tensor!(T, OutRank)(outShape);

        auto vA = broadcastView!OutRank(this, outShape);
        auto vB = broadcastView!OutRank(rhs, outShape);

        enum uint block = 256;
        uint grid = cast(uint)((res.length + block - 1) / block);
        if (grid == 0) grid = 1;

        alias K = elementwiseOp!(op, T, OutRank, NdView!(T, OutRank), NdView!(T, OutRank));
        launch!K([grid, 1, 1], [block, 1, 1], res.deviceView(), vA, vB);

        return res;
    }

    /// Elementwise binary operations between tensor and scalar (+, -, *, /).
    Tensor!(T, N) opBinary(string op, S)(S rhs) const
    if ((op == "+" || op == "-" || op == "*" || op == "/") &&
        !isTensor!S &&
        (is(S : T) || is(typeof(cast(T) rhs))))
    {
        import dtensor.kernels : elementwiseOp;

        auto res = Tensor!(T, N)(this.shape);
        auto vA = this.deviceView();

        enum uint block = 256;
        uint grid = cast(uint)((res.length + block - 1) / block);
        if (grid == 0) grid = 1;

        alias K = elementwiseOp!(op, T, N, NdView!(T, N), T);
        launch!K([grid, 1, 1], [block, 1, 1], res.deviceView(), vA, cast(T) rhs);

        return res;
    }

    /// Elementwise binary operations between scalar and tensor (+, -, *, /).
    Tensor!(T, N) opBinaryRight(string op, S)(S lhs) const
    if ((op == "+" || op == "-" || op == "*" || op == "/") &&
        !isTensor!S &&
        (is(S : T) || is(typeof(cast(T) lhs))))
    {
        import dtensor.kernels : elementwiseOp;

        auto res = Tensor!(T, N)(this.shape);
        auto vB = this.deviceView();

        enum uint block = 256;
        uint grid = cast(uint)((res.length + block - 1) / block);
        if (grid == 0) grid = 1;

        alias K = elementwiseOp!(op, T, N, T, NdView!(T, N));
        launch!K([grid, 1, 1], [block, 1, 1], res.deviceView(), cast(T) lhs, vB);

        return res;
    }

    private T reduceFullImpl(string op)() const
    {
        import dtensor.kernels : reduceBlock;

        assert(this.length > 0, "cannot reduce empty tensor");

        enum uint BLOCK_SIZE = 256;
        uint grid = cast(uint)((this.length + BLOCK_SIZE - 1) / BLOCK_SIZE);
        if (grid == 0) grid = 1;

        auto partial = Tensor!(T, 1)([grid]);
        alias K1 = reduceBlock!(op, T, N);
        launchKernel!K1([grid, 1, 1], [BLOCK_SIZE, 1, 1], cast(uint)(BLOCK_SIZE * T.sizeof),
                        partial.deviceView(), this.deviceView());

        size_t currentLen = grid;
        auto currentTensor = partial;

        while (currentLen > 1)
        {
            uint nextGrid = cast(uint)((currentLen + BLOCK_SIZE - 1) / BLOCK_SIZE);
            if (nextGrid == 0) nextGrid = 1;

            auto nextTensor = Tensor!(T, 1)([nextGrid]);
            alias K2 = reduceBlock!(op, T, 1);
            launchKernel!K2([nextGrid, 1, 1], [BLOCK_SIZE, 1, 1], cast(uint)(BLOCK_SIZE * T.sizeof),
                            nextTensor.deviceView(), currentTensor.deviceView());

            currentTensor = nextTensor;
            currentLen = nextGrid;
        }

        return currentTensor.toNdslice()[0];
    }

    private Tensor!(T, N - 1) reduceAxisImpl(string op)(size_t axis) const
    if (N > 1)
    {
        import dtensor.kernels : reduceAxis;

        assert(axis < N, "axis out of bounds");
        assert(this.shape[axis] > 0, "cannot reduce along zero-sized axis");

        size_t[N - 1] outShape;
        size_t j = 0;
        foreach (d; 0 .. N)
        {
            if (d != axis)
                outShape[j++] = this.shape[d];
        }

        auto res = Tensor!(T, N - 1)(outShape);

        enum uint block = 256;
        uint grid = cast(uint)((res.length + block - 1) / block);
        if (grid == 0) grid = 1;

        alias K = reduceAxis!(op, T, N, N - 1);
        launch!K([grid, 1, 1], [block, 1, 1], res.deviceView(), this.deviceView(), axis);

        return res;
    }

    /// Full reduction: sum of all elements.
    T sum() const
    {
        return reduceFullImpl!"sum"();
    }

    /// Full reduction: maximum of all elements.
    T max() const
    {
        return reduceFullImpl!"max"();
    }

    /// Full reduction: minimum of all elements.
    T min() const
    {
        return reduceFullImpl!"min"();
    }

    /// Full reduction: arithmetic mean of all elements.
    /// For integer tensors, computes integer division `sum / count`.
    T mean() const
    {
        assert(this.length > 0, "cannot compute mean of empty tensor");
        return cast(T)(this.sum() / cast(T) this.length);
    }

    static if (N > 1)
    {
        /// Sum reduction along `axis`, returning a tensor of rank N-1.
        Tensor!(T, N - 1) sum(size_t axis) const
        {
            return reduceAxisImpl!"sum"(axis);
        }

        /// Maximum reduction along `axis`, returning a tensor of rank N-1.
        Tensor!(T, N - 1) max(size_t axis) const
        {
            return reduceAxisImpl!"max"(axis);
        }

        /// Minimum reduction along `axis`, returning a tensor of rank N-1.
        Tensor!(T, N - 1) min(size_t axis) const
        {
            return reduceAxisImpl!"min"(axis);
        }

        /// Arithmetic mean reduction along `axis`, returning a tensor of rank N-1.
        /// For integer tensors, computes integer division along the axis.
        Tensor!(T, N - 1) mean(size_t axis) const
        {
            return reduceAxisImpl!"mean"(axis);
        }
    }
    else static if (N == 1)
    {
        /// Reduction of a 1-D tensor along its single axis (0) returns a scalar.
        T sum(size_t axis) const
        {
            assert(axis == 0, "axis out of bounds for 1-D tensor");
            return this.sum();
        }

        /// Reduction of a 1-D tensor along its single axis (0) returns a scalar.
        T max(size_t axis) const
        {
            assert(axis == 0, "axis out of bounds for 1-D tensor");
            return this.max();
        }

        /// Reduction of a 1-D tensor along its single axis (0) returns a scalar.
        T min(size_t axis) const
        {
            assert(axis == 0, "axis out of bounds for 1-D tensor");
            return this.min();
        }

        /// Reduction of a 1-D tensor along its single axis (0) returns a scalar.
        T mean(size_t axis) const
        {
            assert(axis == 0, "axis out of bounds for 1-D tensor");
            return this.mean();
        }
    }
}

/// Full sum reduction: `sum(tensor)`.
auto sum(T, size_t N)(auto ref const Tensor!(T, N) t)
{
    return t.sum();
}

/// Sum reduction along `axis`: `sum(tensor, axis)`.
auto sum(T, size_t N)(auto ref const Tensor!(T, N) t, size_t axis)
{
    return t.sum(axis);
}

/// Full max reduction: `max(tensor)`.
auto max(T, size_t N)(auto ref const Tensor!(T, N) t)
{
    return t.max();
}

/// Max reduction along `axis`: `max(tensor, axis)`.
auto max(T, size_t N)(auto ref const Tensor!(T, N) t, size_t axis)
{
    return t.max(axis);
}

/// Full min reduction: `min(tensor)`.
auto min(T, size_t N)(auto ref const Tensor!(T, N) t)
{
    return t.min();
}

/// Min reduction along `axis`: `min(tensor, axis)`.
auto min(T, size_t N)(auto ref const Tensor!(T, N) t, size_t axis)
{
    return t.min(axis);
}

/// Full mean reduction: `mean(tensor)`.
auto mean(T, size_t N)(auto ref const Tensor!(T, N) t)
{
    return t.mean();
}

/// Mean reduction along `axis`: `mean(tensor, axis)`.
auto mean(T, size_t N)(auto ref const Tensor!(T, N) t, size_t axis)
{
    return t.mean(axis);
}

/// Elementwise addition: `a + b`.
auto add(A, B)(auto ref A a, auto ref B b)
{
    return a + b;
}

/// Elementwise subtraction: `a - b`.
auto sub(A, B)(auto ref A a, auto ref B b)
{
    return a - b;
}

/// Elementwise multiplication: `a * b`.
auto mul(A, B)(auto ref A a, auto ref B b)
{
    return a * b;
}

/// Elementwise division: `a / b`.
auto div(A, B)(auto ref A a, auto ref B b)
{
    return a / b;
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

unittest
{
    auto t = Tensor!(float, 2)([3LU, 4LU]);
    auto v = t.deviceView();
    static assert(is(typeof(v) == NdView!(float, 2)));
    assert(cast(size_t) v.ptr.ptr == t.store.raw);
    assert(v.shape == [3LU, 4LU]);
    assert(v.strides == [4LU, 1LU]);

    auto tr = t.transpose();
    auto vtr = tr.deviceView();
    assert(cast(size_t) vtr.ptr.ptr == t.store.raw);
    assert(vtr.shape == [4LU, 3LU]);
    assert(vtr.strides == [1LU, 4LU]);
}

unittest
{
    import dtensor.testkernels : tensorScale2D;

    auto src = iota(3, 4).as!float.slice;
    auto x = fromNdslice(src);
    auto y = Tensor!(float, 2)([3LU, 4LU]);

    enum uint block = 64;
    immutable uint grid = cast(uint)((x.length + block - 1) / block);

    launch!tensorScale2D([grid, 1, 1], [block, 1, 1], y.deviceView(), x.deviceView(), 2.5f);

    auto res = y.toNdslice();
    foreach (i; 0 .. 3)
        foreach (j; 0 .. 4)
            assert(res[i, j] == src[i, j] * 2.5f);

    // Also verify over a transposed view:
    auto xt = x.transpose();
    auto yt = Tensor!(float, 2)([4LU, 3LU]);
    immutable uint gridT = cast(uint)((xt.length + block - 1) / block);

    launch!tensorScale2D([gridT, 1, 1], [block, 1, 1], yt.deviceView(), xt.deviceView(), 3.0f);

    auto resT = yt.toNdslice();
    foreach (i; 0 .. 4)
        foreach (j; 0 .. 3)
            assert(resT[i, j] == src[j, i] * 3.0f);
}

unittest
{
    // Same-shape elementwise ops between two tensors
    auto sa = iota(2, 3).as!float.slice;
    auto sb = iota([2, 3], 10).as!float.slice;

    auto a = fromNdslice(sa);
    auto b = fromNdslice(sb);

    // Addition
    auto cAdd = a + b;
    static assert(is(typeof(cAdd) == Tensor!(float, 2)));
    assert(cAdd.shape == [2LU, 3LU]);
    auto resAdd = cAdd.toNdslice();
    foreach (i; 0 .. 2)
        foreach (j; 0 .. 3)
            assert(resAdd[i, j] == sa[i, j] + sb[i, j]);

    // Subtraction
    auto cSub = a - b;
    auto resSub = cSub.toNdslice();
    foreach (i; 0 .. 2)
        foreach (j; 0 .. 3)
            assert(resSub[i, j] == sa[i, j] - sb[i, j]);

    // Multiplication
    auto cMul = a * b;
    auto resMul = cMul.toNdslice();
    foreach (i; 0 .. 2)
        foreach (j; 0 .. 3)
            assert(resMul[i, j] == sa[i, j] * sb[i, j]);

    // Division
    auto cDiv = b / (a + 1.0f);
    auto resDiv = cDiv.toNdslice();
    foreach (i; 0 .. 2)
        foreach (j; 0 .. 3)
            assert(resDiv[i, j] == sb[i, j] / (sa[i, j] + 1.0f));

    // Also check function API: add, sub, mul, div
    foreach (i; 0 .. 2)
        foreach (j; 0 .. 3)
        {
            assert(add(a, b).toNdslice()[i, j] == resAdd[i, j]);
            assert(sub(a, b).toNdslice()[i, j] == resSub[i, j]);
            assert(mul(a, b).toNdslice()[i, j] == resMul[i, j]);
        }
}

unittest
{
    // Scalar operations
    auto sa = iota(2, 3).as!float.slice;
    auto a = fromNdslice(sa);

    // Tensor + scalar
    auto cAdd = a + 2.5f;
    auto resAdd = cAdd.toNdslice();
    foreach (i; 0 .. 2)
        foreach (j; 0 .. 3)
            assert(resAdd[i, j] == sa[i, j] + 2.5f);

    // Tensor - scalar
    auto cSub = a - 1.0f;
    auto resSub = cSub.toNdslice();
    foreach (i; 0 .. 2)
        foreach (j; 0 .. 3)
            assert(resSub[i, j] == sa[i, j] - 1.0f);

    // Tensor * scalar
    auto cMul = a * 3.0f;
    auto resMul = cMul.toNdslice();
    foreach (i; 0 .. 2)
        foreach (j; 0 .. 3)
            assert(resMul[i, j] == sa[i, j] * 3.0f);

    // Tensor / scalar
    auto cDiv = a / 2.0f;
    auto resDiv = cDiv.toNdslice();
    foreach (i; 0 .. 2)
        foreach (j; 0 .. 3)
            assert(resDiv[i, j] == sa[i, j] / 2.0f);

    // Scalar - tensor
    auto cRsub = 10.0f - a;
    auto resRsub = cRsub.toNdslice();
    foreach (i; 0 .. 2)
        foreach (j; 0 .. 3)
            assert(resRsub[i, j] == 10.0f - sa[i, j]);

    // Scalar / tensor
    auto cRdiv = 12.0f / (a + 1.0f);
    auto resRdiv = cRdiv.toNdslice();
    foreach (i; 0 .. 2)
        foreach (j; 0 .. 3)
            assert(resRdiv[i, j] == 12.0f / (sa[i, j] + 1.0f));

    // Functions add, sub, mul, div with scalar
    foreach (i; 0 .. 2)
        foreach (j; 0 .. 3)
        {
            assert(add(a, 2.5f).toNdslice()[i, j] == resAdd[i, j]);
            assert(sub(10.0f, a).toNdslice()[i, j] == resRsub[i, j]);
            assert(mul(3.0f, a).toNdslice()[i, j] == resMul[i, j]);
        }
}

unittest
{
    // Broadcasting
    // Case 1: [3, 1] + [1, 4] -> [3, 4]
    auto sa = iota([3, 1], 1).as!float.slice;
    auto sb = iota([1, 4], 10).as!float.slice;

    auto a = fromNdslice(sa);
    auto b = fromNdslice(sb);

    auto c = a + b;
    static assert(is(typeof(c) == Tensor!(float, 2)));
    assert(c.shape == [3LU, 4LU]);
    auto res = c.toNdslice();
    foreach (i; 0 .. 3)
        foreach (j; 0 .. 4)
            assert(res[i, j] == sa[i, 0] + sb[0, j]);

    auto cSub = a - b;
    auto resSub = cSub.toNdslice();
    foreach (i; 0 .. 3)
        foreach (j; 0 .. 4)
            assert(resSub[i, j] == sa[i, 0] - sb[0, j]);

    auto cMul = a * b;
    auto resMul = cMul.toNdslice();
    foreach (i; 0 .. 3)
        foreach (j; 0 .. 4)
            assert(resMul[i, j] == sa[i, 0] * sb[0, j]);

    // Case 2: [2, 3] + [3] -> [2, 3] (different ranks)
    auto s23 = iota(2, 3).as!float.slice;
    auto s3 = iota([3], 100).as!float.slice;

    auto t23 = fromNdslice(s23);
    auto t3 = fromNdslice(s3);

    auto c23 = t23 + t3;
    static assert(is(typeof(c23) == Tensor!(float, 2)));
    assert(c23.shape == [2LU, 3LU]);
    auto res23 = c23.toNdslice();
    foreach (i; 0 .. 2)
        foreach (j; 0 .. 3)
            assert(res23[i, j] == s23[i, j] + s3[j]);

    // Right-hand broadcast with different ranks: [3] + [2, 3] -> [2, 3]
    auto c32 = t3 + t23;
    auto res32 = c32.toNdslice();
    foreach (i; 0 .. 2)
        foreach (j; 0 .. 3)
            assert(res32[i, j] == res23[i, j]);

    // Case 3: 1D broadcast [1] + [5] -> [5]
    auto s1 = iota([1], 5).as!float.slice;
    auto s5 = iota([5], 1).as!float.slice;
    auto t1 = fromNdslice(s1);
    auto t5 = fromNdslice(s5);
    auto c15 = t1 + t5;
    static assert(is(typeof(c15) == Tensor!(float, 1)));
    assert(c15.shape == [5LU]);
    auto res15 = c15.toNdslice();
    foreach (i; 0 .. 5)
        assert(res15[i] == s1[0] + s5[i]);
}

unittest
{
    // Transposed input
    // [3, 2] transposed to [2, 3] + contiguous [2, 3]
    auto sa = iota(3, 2).as!float.slice;
    auto sb = iota([2, 3], 20).as!float.slice;

    auto a = fromNdslice(sa);
    auto b = fromNdslice(sb);

    auto at = a.transpose();
    assert(at.shape == [2LU, 3LU]);
    assert(!at.contiguous);

    auto c = at + b;
    assert(c.shape == [2LU, 3LU]);
    auto res = c.toNdslice();
    foreach (i; 0 .. 2)
        foreach (j; 0 .. 3)
            assert(res[i, j] == sa[j, i] + sb[i, j]);

    // Transposed input with scalar
    auto cScalar = at * 2.0f;
    assert(cScalar.shape == [2LU, 3LU]);
    auto resScalar = cScalar.toNdslice();
    foreach (i; 0 .. 2)
        foreach (j; 0 .. 3)
            assert(resScalar[i, j] == sa[j, i] * 2.0f);

    // Transposed input + broadcast [3]
    auto s3 = iota([3], 50).as!float.slice;
    auto t3 = fromNdslice(s3);
    auto cBc = at + t3;
    assert(cBc.shape == [2LU, 3LU]);
    auto resBc = cBc.toNdslice();
    foreach (i; 0 .. 2)
        foreach (j; 0 .. 3)
            assert(resBc[i, j] == sa[j, i] + s3[j]);
}

unittest
{
    // Incompatible shapes error
    import std.exception : assertThrown;

    auto a = Tensor!(float, 2)([2LU, 3LU]);
    auto b = Tensor!(float, 2)([2LU, 4LU]);
    auto c = Tensor!(float, 1)([4LU]);

    assertThrown!ShapeMismatchException(a + b);
    assertThrown!ShapeMismatchException(a - b);
    assertThrown!ShapeMismatchException(a * c);
    assertThrown!ShapeMismatchException(a / c);
}

unittest
{
    // Loop churn test: allocate and free 2000 x 1MB tensors and verify
    // cuMemGetInfo returns to baseline within a small tolerance.
    import bindbc.cuda : cuMemGetInfo;
    import std.stdio : writefln;

    ensureInit();

    // Warm-up allocation to prime CUDA driver memory structures
    {
        auto warm = Tensor!(float, 1)([256 * 1024LU]);
    }

    size_t freeBefore, totalBefore;
    immutable errBefore = cuMemGetInfo(&freeBefore, &totalBefore);
    assert(errBefore == 0, "cuMemGetInfo failed before loop");

    enum size_t count = 2000;
    enum size_t elems = 256 * 1024LU; // 256K * 4B = 1 MiB

    foreach (i; 0 .. count)
    {
        auto t = Tensor!(float, 1)([elems]);
        assert(t.store.raw != 0);
    }

    size_t freeAfter, totalAfter;
    immutable errAfter = cuMemGetInfo(&freeAfter, &totalAfter);
    assert(errAfter == 0, "cuMemGetInfo failed after loop");

    writefln("cuMemGetInfo baseline: %d bytes free (%d MB), after 2000x1MB: %d bytes free (%d MB), delta: %d bytes",
        freeBefore, freeBefore / (1024 * 1024),
        freeAfter, freeAfter / (1024 * 1024),
        cast(long)freeBefore - cast(long)freeAfter);

    // Tolerance of 2 MiB (against 2000 MiB churn)
    enum size_t tolerance = 2 * 1024 * 1024;
    immutable long delta = cast(long)freeBefore - cast(long)freeAfter;
    assert(delta >= -cast(long)tolerance && delta <= cast(long)tolerance,
        "Device memory leak detected: free memory did not return to baseline");
}

unittest
{
    // View outlives parent test:
    // Verify that views (transpose, reshape) share ownership of the device buffer,
    // data remains valid and accessible after the parent is destroyed,
    // device memory is NOT freed while any view is alive, and IS freed when the last view dies.
    import bindbc.cuda : cuMemGetInfo;
    import std.stdio : writefln;

    ensureInit();

    // Warm-up
    {
        auto warm = Tensor!(float, 1)([256 * 1024LU]);
    }

    size_t freeBaseline, totalBaseline;
    assert(cuMemGetInfo(&freeBaseline, &totalBaseline) == 0);

    enum size_t rows = 1024, cols = 1024; // 1M floats = 4 MiB
    enum size_t elems = rows * cols;
    enum size_t bytes = elems * float.sizeof;

    // Part 1: Data correctness when view outlives parent
    {
        Tensor!(float, 2) viewTransposed;
        Tensor!(float, 1) viewReshaped;
        size_t parentRaw;

        {
            auto src = iota(3, 4).as!float.slice;
            auto parent = fromNdslice(src);
            parentRaw = parent.store.raw;
            assert(parentRaw != 0);

            viewTransposed = parent.transpose();
            viewReshaped = parent.reshape([12LU]);

            assert(viewTransposed.store.raw == parentRaw);
            assert(viewReshaped.store.raw == parentRaw);

            // parent is destroyed here
        }

        // Parent is dead, views are alive
        assert(viewTransposed.store.raw == parentRaw);
        assert(viewReshaped.store.raw == parentRaw);

        auto sliceT = viewTransposed.toNdslice();
        foreach (i; 0 .. 4)
            foreach (j; 0 .. 3)
                assert(sliceT[i, j] == cast(float)(j * 4 + i));

        auto sliceR = viewReshaped.toNdslice();
        foreach (k; 0 .. 12)
            assert(sliceR[k] == cast(float) k);
    }

    // Part 2: Device memory lifetime when view outlives parent
    {
        size_t freeBefore;
        assert(cuMemGetInfo(&freeBefore, &totalBaseline) == 0);

        Tensor!(float, 2) viewTransposed;
        Tensor!(float, 1) viewReshaped;
        size_t parentRaw;

        {
            auto parent = Tensor!(float, 2)([rows, cols]);
            parentRaw = parent.store.raw;
            assert(parentRaw != 0);

            size_t freeWithParent;
            assert(cuMemGetInfo(&freeWithParent, &totalBaseline) == 0);
            assert(freeBefore >= freeWithParent + bytes, "Parent device memory was not allocated");

            viewTransposed = parent.transpose();
            viewReshaped = parent.reshape([elems]);

            assert(viewTransposed.store.raw == parentRaw);
            assert(viewReshaped.store.raw == parentRaw);

            // parent is destroyed here
        }

        // Parent is destroyed, but viewTransposed and viewReshaped still hold ownership
        size_t freeWithViews;
        assert(cuMemGetInfo(&freeWithViews, &totalBaseline) == 0);
        assert(freeBefore >= freeWithViews + bytes, "Device memory freed prematurely while views alive");
        assert(viewTransposed.store.raw == parentRaw);
        assert(viewReshaped.store.raw == parentRaw);

        // Destroy one view
        viewReshaped = Tensor!(float, 1).init;
        size_t freeWithOneView;
        assert(cuMemGetInfo(&freeWithOneView, &totalBaseline) == 0);
        assert(freeBefore >= freeWithOneView + bytes, "Device memory freed while last view alive");

        // Destroy final view
        viewTransposed = Tensor!(float, 2).init;
        size_t freeAfterViews;
        assert(cuMemGetInfo(&freeAfterViews, &totalBaseline) == 0);

        enum size_t tolerance = 1024 * 1024; // 1 MiB tolerance
        immutable long diff = cast(long)freeBefore - cast(long)freeAfterViews;
        assert(diff >= -cast(long)tolerance && diff <= cast(long)tolerance,
            "Device memory was not freed after all views died");

        writefln("View-outlives-parent memory: baseline=%d bytes, with-views=%d bytes, after-views=%d bytes",
            freeBefore, freeWithViews, freeAfterViews);
    }
}

unittest
{
    // Full reductions on large non-power-of-2 1-D tensor (> 1 block: 1_000_003 elements)
    import mir.ndslice.topology : iota, map, flattened;
    import mir.ndslice.allocation : slice;
    import mir.math.sum : sum;
    import mir.math.stat : mean;
    import mir.algorithm.iteration : reduce;
    import std.math : isClose, abs;

    ensureInit();

    enum size_t N = 1_000_003;
    // Values alternating or scaled so sum doesn't overflow float
    auto host = new float[N];
    foreach (i; 0 .. N)
        host[i] = ((i % 100) - 50) * 0.1f;

    auto hostSlice = host.sliced;
    auto t = fromNdslice(hostSlice);

    float expSum = hostSlice.sum;
    float expMean = hostSlice.mean;
    float expMax = reduce!"a > b ? a : b"(-float.infinity, hostSlice);
    float expMin = reduce!"a < b ? a : b"(float.infinity, hostSlice);

    float devSum = t.sum();
    float devMean = t.mean();
    float devMax = t.max();
    float devMin = t.min();

    // Verify float tolerance (tree reduce rounding vs host summation)
    assert(isClose(devSum, expSum, 1e-4, 1e-4), "1_000_003 sum mismatch");
    assert(isClose(devMean, expMean, 1e-4, 1e-4), "1_000_003 mean mismatch");
    assert(devMax == expMax, "1_000_003 max mismatch");
    assert(devMin == expMin, "1_000_003 min mismatch");

    // Also test free functions
    assert(isClose(dtensor.tensor.sum(t), expSum, 1e-4, 1e-4));
    assert(isClose(dtensor.tensor.mean(t), expMean, 1e-4, 1e-4));
    assert(dtensor.tensor.max(t) == expMax);
    assert(dtensor.tensor.min(t) == expMin);

    // Rank-1 axis reduction returns scalar
    assert(isClose(t.sum(0), expSum, 1e-4, 1e-4));
    assert(isClose(t.mean(0), expMean, 1e-4, 1e-4));
    assert(t.max(0) == expMax);
    assert(t.min(0) == expMin);
}

unittest
{
    // Reductions along each axis of a 3-D tensor (sum, max, min, mean)
    import mir.ndslice.topology : iota, alongDim, map, as;
    import mir.ndslice.allocation : slice;
    import mir.math.sum : sum;
    import mir.math.stat : mean;
    import mir.algorithm.iteration : reduce;
    import std.math : isClose;

    ensureInit();

    // Shape [3, 5, 7] - non-power-of-2 dimensions
    auto hostFloat = iota(3, 5, 7).as!float.slice;
    auto tFloat = fromNdslice(hostFloat);

    // Axis 0
    {
        auto expSum = hostFloat.alongDim!0.map!sum.slice;
        auto expMean = hostFloat.alongDim!0.map!mean.slice;
        auto expMax = hostFloat.alongDim!0.map!(s => reduce!"a > b ? a : b"(-float.infinity, s)).slice;
        auto expMin = hostFloat.alongDim!0.map!(s => reduce!"a < b ? a : b"(float.infinity, s)).slice;

        auto devSum = tFloat.sum(0).toNdslice();
        auto devMean = tFloat.mean(0).toNdslice();
        auto devMax = tFloat.max(0).toNdslice();
        auto devMin = tFloat.min(0).toNdslice();

        assert(devSum.shape == [5LU, 7LU]);
        foreach (i; 0 .. 5)
            foreach (j; 0 .. 7)
            {
                assert(isClose(devSum[i, j], expSum[i, j], 1e-5, 1e-5));
                assert(isClose(devMean[i, j], expMean[i, j], 1e-5, 1e-5));
                assert(devMax[i, j] == expMax[i, j]);
                assert(devMin[i, j] == expMin[i, j]);
            }
    }

    // Axis 1
    {
        auto expSum = hostFloat.alongDim!1.map!sum.slice;
        auto expMean = hostFloat.alongDim!1.map!mean.slice;
        auto expMax = hostFloat.alongDim!1.map!(s => reduce!"a > b ? a : b"(-float.infinity, s)).slice;
        auto expMin = hostFloat.alongDim!1.map!(s => reduce!"a < b ? a : b"(float.infinity, s)).slice;

        auto devSum = tFloat.sum(1).toNdslice();
        auto devMean = tFloat.mean(1).toNdslice();
        auto devMax = tFloat.max(1).toNdslice();
        auto devMin = tFloat.min(1).toNdslice();

        assert(devSum.shape == [3LU, 7LU]);
        foreach (i; 0 .. 3)
            foreach (j; 0 .. 7)
            {
                assert(isClose(devSum[i, j], expSum[i, j], 1e-5, 1e-5));
                assert(isClose(devMean[i, j], expMean[i, j], 1e-5, 1e-5));
                assert(devMax[i, j] == expMax[i, j]);
                assert(devMin[i, j] == expMin[i, j]);
            }
    }

    // Axis 2
    {
        auto expSum = hostFloat.alongDim!2.map!sum.slice;
        auto expMean = hostFloat.alongDim!2.map!mean.slice;
        auto expMax = hostFloat.alongDim!2.map!(s => reduce!"a > b ? a : b"(-float.infinity, s)).slice;
        auto expMin = hostFloat.alongDim!2.map!(s => reduce!"a < b ? a : b"(float.infinity, s)).slice;

        auto devSum = tFloat.sum(2).toNdslice();
        auto devMean = tFloat.mean(2).toNdslice();
        auto devMax = tFloat.max(2).toNdslice();
        auto devMin = tFloat.min(2).toNdslice();

        assert(devSum.shape == [3LU, 5LU]);
        foreach (i; 0 .. 3)
            foreach (j; 0 .. 5)
            {
                assert(isClose(devSum[i, j], expSum[i, j], 1e-5, 1e-5));
                assert(isClose(devMean[i, j], expMean[i, j], 1e-5, 1e-5));
                assert(devMax[i, j] == expMax[i, j]);
                assert(devMin[i, j] == expMin[i, j]);
            }
    }
}

unittest
{
    // Transposed inputs: non-contiguous layout reductions
    import mir.ndslice.topology : iota, alongDim, map, as;
    import mir.ndslice.dynamic : transposed;
    import mir.ndslice.allocation : slice;
    import mir.math.sum : sum;
    import mir.math.stat : mean;
    import mir.algorithm.iteration : reduce;
    import std.math : isClose;

    ensureInit();

    // 2-D transposed [3, 4] -> [4, 3]
    auto host2D = iota(3, 4).as!double.slice;
    auto t2D = fromNdslice(host2D);
    auto t2DT = t2D.transpose();
    assert(!t2DT.contiguous);

    auto host2DT = host2D.transposed;

    // Full reductions on transposed 2-D
    assert(isClose(t2DT.sum(), host2DT.sum, 1e-9, 1e-9));
    assert(isClose(t2DT.mean(), host2DT.mean, 1e-9, 1e-9));
    assert(t2DT.max() == reduce!"a > b ? a : b"(-double.infinity, host2DT));
    assert(t2DT.min() == reduce!"a < b ? a : b"(double.infinity, host2DT));

    // Axis reductions on transposed 2-D
    auto devSum0 = t2DT.sum(0).toNdslice();
    auto expSum0 = host2DT.alongDim!0.map!sum.slice;
    foreach (j; 0 .. 3)
        assert(isClose(devSum0[j], expSum0[j], 1e-9, 1e-9));

    auto devSum1 = t2DT.sum(1).toNdslice();
    auto expSum1 = host2DT.alongDim!1.map!sum.slice;
    foreach (i; 0 .. 4)
        assert(isClose(devSum1[i], expSum1[i], 1e-9, 1e-9));

    // 3-D transposed [2, 3, 4] with permutation [2, 0, 1] -> [4, 2, 3]
    auto host3D = iota(2, 3, 4).as!float.slice;
    auto t3D = fromNdslice(host3D);
    auto t3DT = t3D.transpose([2LU, 0LU, 1LU]);
    assert(!t3DT.contiguous);
    auto host3DT = host3D.transposed!(2, 0, 1);

    auto devMaxAxis1 = t3DT.max(1).toNdslice();
    auto expMaxAxis1 = host3DT.alongDim!1.map!(s => reduce!"a > b ? a : b"(-float.infinity, s)).slice;
    assert(devMaxAxis1.shape == [4LU, 3LU]);
    foreach (i; 0 .. 4)
        foreach (j; 0 .. 3)
            assert(devMaxAxis1[i, j] == expMaxAxis1[i, j]);
}

unittest
{
    // Integer reductions: int types, negative numbers, integer division for mean
    import mir.ndslice.topology : iota, alongDim, map, as;
    import mir.ndslice.allocation : slice;
    import mir.math.sum : sum;
    import mir.algorithm.iteration : reduce;

    ensureInit();

    // Negative and positive ints: [-15 .. 14] in [5, 6]
    auto hostInt = iota([5, 6], -15).as!int.slice;
    auto tInt = fromNdslice(hostInt);

    int expSum = cast(int) hostInt.sum;
    int expMax = reduce!"a > b ? a : b"(int.min, hostInt);
    int expMin = reduce!"a < b ? a : b"(int.max, hostInt);
    int expMean = expSum / 30; // documented integer division

    assert(tInt.sum() == expSum);
    assert(tInt.max() == expMax);
    assert(tInt.min() == expMin);
    assert(tInt.mean() == expMean);

    // Axis 1 integer mean
    auto devMeanAxis1 = tInt.mean(1).toNdslice();
    auto expSumAxis1 = hostInt.alongDim!1.map!sum.slice;
    foreach (i; 0 .. 5)
    {
        int rowMean = cast(int)(expSumAxis1[i] / 6);
        assert(devMeanAxis1[i] == rowMean);
    }
}

unittest
{
    // Double precision and 4-D tensor reduction
    import mir.ndslice.topology : iota, alongDim, map, as;
    import mir.ndslice.allocation : slice;
    import mir.math.sum : sum;
    import mir.math.stat : mean;
    import std.math : isClose;

    ensureInit();

    // 4-D tensor shape [2, 3, 2, 4] = 48 doubles
    auto host4D = iota(2, 3, 2, 4).as!double.slice;
    auto t4D = fromNdslice(host4D);

    // Full reduction on 4-D double
    assert(isClose(t4D.sum(), host4D.sum, 1e-12, 1e-12));
    assert(isClose(t4D.mean(), host4D.mean, 1e-12, 1e-12));
    assert(t4D.max() == 47.0);
    assert(t4D.min() == 0.0);

    // Axis 2 reduction on 4-D double -> 3-D tensor shape [2, 3, 4]
    auto devSumAxis2 = t4D.sum(2).toNdslice();
    auto expSumAxis2 = host4D.alongDim!2.map!sum.slice;
    assert(devSumAxis2.shape == [2LU, 3LU, 4LU]);
    foreach (i; 0 .. 2)
        foreach (j; 0 .. 3)
            foreach (k; 0 .. 4)
                assert(isClose(devSumAxis2[i, j, k], expSumAxis2[i, j, k], 1e-12, 1e-12));
}
