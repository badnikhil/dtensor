# dtensor

GPU-accelerated tensor programming model for D, built on
[dcompute](https://github.com/libmir/dcompute).

`dtensor` is the **host-side framework layer**: the `Tensor` container, shape/stride
bookkeeping, `mir.ndslice` interop, and (later) broadcasting, kernel fusion, reductions
and GEMM. The device substrate it sits on — address-space pointers, `@kernel`, the N-d
device view, launchers, pitched memory — lives in dcompute.

Design doc: `dcompute/docs/TENSOR_ROADMAP.md`.

## Status

Skeleton. What exists today:

| Piece | State |
|---|---|
| `Tensor!(T, N)` — device-memory-owning host container | ✅ |
| Shape + strides (**strides in elements**) | ✅ |
| `reshape` / `transpose` as pure stride views (no data movement) | ✅ |
| `toNdslice` / `fromNdslice` round-trip | ✅ |
| Lifetime via dcompute `Buffer!T` RAII | ✅ |
| Broadcasting, fusion, reductions, GEMM, dtype, autodiff | ❌ later milestones |
| Pool allocator | ❌ deferred — plain alloc/free for now |

## Requirements

- **LDC ≥ 1.43** (embedded-PTX `launch!` is gated on `__VERSION__ >= 2113`)
- CUDA driver + a CUDA-capable device
- dcompute (currently consumed as a local path dependency while both move together)

## Build

dcompute's `dcompute.std` package `static assert`s unless the compiler is invoked with
`-mdcompute-targets`, and dub does not push a dependent's `dflags` down into a dependency —
so the flag has to reach *every* package via `DFLAGS`, and the build type has to be given
explicitly (setting `DFLAGS` otherwise makes dub switch to its `$DFLAGS` build type and drop
`-unittest`):

```sh
export DC=~/dlang/ldc-1.43.0/bin/ldc2

DFLAGS="-mdcompute-targets=cuda-800" nice -19 ionice -c3 \
    dub build --build=release --compiler=$DC

DFLAGS="-mdcompute-targets=cuda-800" nice -19 ionice -c3 \
    dub test --build=unittest --compiler=$DC
```

sm_80 rather than the RTX 2050's native sm_86: LDC's valid-target list rejects `cuda-860`,
and the driver JITs 8.0 PTX up to 8.6.

The tests allocate device memory and copy host↔device, so they need a CUDA driver and a
device — but they compile and launch no kernels, so they do not depend on the embedded-PTX
`launch!` path. The pure shape/stride unittest needs no GPU at all.

## Conventions

- **Strides are in elements, never bytes.** This matches `mir.ndslice` and is the contract
  dcompute's `NdView!(T,N)` will use.
- **A `Tensor` owns its device memory through dcompute's `Buffer!T`.** There is no second
  ownership scheme here; views produced by `reshape`/`transpose` share the buffer.

## License

BSL-1.0 — see `LICENSE.txt`.
