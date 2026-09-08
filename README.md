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

```sh
nice -19 ionice -c3 dub build --compiler=ldc2 -j2
nice -19 ionice -c3 dub test --compiler=ldc2 -j2      # needs a CUDA device
```

The host-only logic (strides, shapes, reshape/transpose) is additionally testable with
no GPU present:

```sh
nice -19 ionice -c3 dub run :hosttest --compiler=ldc2 -j2
```

## Conventions

- **Strides are in elements, never bytes.** This matches `mir.ndslice` and is the contract
  dcompute's `NdView!(T,N)` will use.
- **A `Tensor` owns its device memory through dcompute's `Buffer!T`.** There is no second
  ownership scheme here; views produced by `reshape`/`transpose` share the buffer.

## License

BSL-1.0 — see `LICENSE.txt`.
