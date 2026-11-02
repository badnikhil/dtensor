# dtensor

GPU tensors for D built on dcompute. Kernels are compiled at build time by LDC with no runtime JIT compilation. The library currently supports CUDA only and is in early development.

## Requirements

- LDC 1.43.0+
- NVIDIA GPU with CUDA driver installed
- dcompute `tensor` branch of github.com/badnikhil/dcompute checked out at `../dcompute`

## Usage

```d
import std.stdio;
import mir.ndslice.allocation : slice;
import mir.ndslice.topology : iota, as;
import dtensor;

void main()
{
    auto a = fromNdslice(iota([3, 1], 1).as!float.slice);
    auto b = fromNdslice(iota([1, 4], 10).as!float.slice);

    auto c = a + b;
    writeln("Result:\n", c.toNdslice());
}
```

## Running Tests

Run the test suite with LDC in PATH:

```sh
PATH=/home/nikhil/dlang/ldc-1.43.0/bin:$PATH nice -n 19 ionice -c3 dub test --force
```

An executable broadcasting example is available under `examples/elementwise/` and can be run with:

```sh
cd examples/elementwise
PATH=/home/nikhil/dlang/ldc-1.43.0/bin:$PATH nice -n 19 ionice -c3 dub run
```

## License

BSL-1.0
