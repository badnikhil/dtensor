import std.stdio;
import mir.ndslice.allocation : slice;
import mir.ndslice.topology : iota, as;
import dtensor;

void main()
{
    // Broadcast add: [3, 1] + [1, 4] -> [3, 4]
    auto a = fromNdslice(iota([3, 1], 1).as!float.slice);
    auto b = fromNdslice(iota([1, 4], 10).as!float.slice);

    auto c = a + b;

    writeln("a shape: ", a.shape);
    writeln("b shape: ", b.shape);
    writeln("c shape: ", c.shape);
    writeln("c = a + b:\n", c.toNdslice());
}
