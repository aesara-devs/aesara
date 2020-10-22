from theano import tensor
from theano.gof.fg import FunctionGraph
from theano.gof.graph import Apply, Variable
from theano.gof.op import Op
from theano.gof.toolbox import NodeFinder, is_same_graph
from theano.gof.type import Type


class TestNodeFinder:
    def test_straightforward(self):
        class MyType(Type):
            def __init__(self, name):
                self.name = name

            def __str__(self):
                return self.name

            def __repr__(self):
                return self.name

            def __eq__(self, other):
                return isinstance(other, MyType)

        class MyOp(Op):

            __props__ = ("nin", "name")

            def __init__(self, nin, name):
                self.nin = nin
                self.name = name

            def make_node(self, *inputs):
                def as_variable(x):
                    assert isinstance(x, Variable)
                    return x

                assert len(inputs) == self.nin
                inputs = list(map(as_variable, inputs))
                for input in inputs:
                    if not isinstance(input.type, MyType):
                        raise Exception("Error 1")
                outputs = [MyType(self.name + "_R")()]
                return Apply(self, inputs, outputs)

            def __str__(self):
                return self.name

        sigmoid = MyOp(1, "Sigmoid")
        add = MyOp(2, "Add")
        dot = MyOp(2, "Dot")

        def MyVariable(name):
            return Variable(MyType(name), None, None)

        def inputs():
            x = MyVariable("x")
            y = MyVariable("y")
            z = MyVariable("z")
            return x, y, z

        x, y, z = inputs()
        e0 = dot(y, z)
        e = add(add(sigmoid(x), sigmoid(sigmoid(z))), dot(add(x, y), e0))
        g = FunctionGraph([x, y, z], [e], clone=False)
        g.attach_feature(NodeFinder())

        assert hasattr(g, "get_nodes")
        for type, num in ((add, 3), (sigmoid, 3), (dot, 2)):
            if not len([t for t in g.get_nodes(type)]) == num:
                raise Exception("Expected: %i times %s" % (num, type))
        new_e0 = add(y, z)
        assert e0.owner in g.get_nodes(dot)
        assert new_e0.owner not in g.get_nodes(add)
        g.replace(e0, new_e0)
        assert e0.owner not in g.get_nodes(dot)
        assert new_e0.owner in g.get_nodes(add)
        for type, num in ((add, 4), (sigmoid, 3), (dot, 1)):
            if not len([t for t in g.get_nodes(type)]) == num:
                raise Exception("Expected: %i times %s" % (num, type))


class TestIsSameGraph:
    def check(self, expected):
        """
        Core function to perform comparison.

        :param expected: A list of tuples (v1, v2, ((g1, o1), ..., (gN, oN)))
        with:
            - `v1` and `v2` two Variables (the graphs to be compared)
            - `gj` a `givens` dictionary to give as input to `is_same_graph`
            - `oj` the expected output of `is_same_graph(v1, v2, givens=gj)`

        This function also tries to call `is_same_graph` by inverting `v1` and
        `v2`, and ensures the output remains the same.
        """
        for v1, v2, go in expected:
            for gj, oj in go:
                r1 = is_same_graph(v1, v2, givens=gj)
                assert r1 == oj
                r2 = is_same_graph(v2, v1, givens=gj)
                assert r2 == oj

    def test_single_var(self):
        # Test `is_same_graph` with some trivial graphs (one Variable).

        x, y, z = tensor.vectors("x", "y", "z")
        self.check(
            [
                (x, x, (({}, True),)),
                (
                    x,
                    y,
                    (
                        ({}, False),
                        ({y: x}, True),
                    ),
                ),
                (x, tensor.neg(x), (({}, False),)),
                (x, tensor.neg(y), (({}, False),)),
            ]
        )

    def test_full_graph(self):
        # Test `is_same_graph` with more complex graphs.

        x, y, z = tensor.vectors("x", "y", "z")
        t = x * y
        self.check(
            [
                (x * 2, x * 2, (({}, True),)),
                (
                    x * 2,
                    y * 2,
                    (
                        ({}, False),
                        ({y: x}, True),
                    ),
                ),
                (
                    x * 2,
                    y * 2,
                    (
                        ({}, False),
                        ({x: y}, True),
                    ),
                ),
                (
                    x * 2,
                    y * 3,
                    (
                        ({}, False),
                        ({y: x}, False),
                    ),
                ),
                (
                    t * 2,
                    z * 2,
                    (
                        ({}, False),
                        ({t: z}, True),
                    ),
                ),
                (
                    t * 2,
                    z * 2,
                    (
                        ({}, False),
                        ({z: t}, True),
                    ),
                ),
                (x * (y * z), (x * y) * z, (({}, False),)),
            ]
        )

    def test_merge_only(self):
        # Test `is_same_graph` when `equal_computations` cannot be used.

        x, y, z = tensor.vectors("x", "y", "z")
        t = x * y
        self.check(
            [
                (x, t, (({}, False), ({t: x}, True))),
                (
                    t * 2,
                    x * 2,
                    (
                        ({}, False),
                        ({t: x}, True),
                    ),
                ),
                (
                    x * x,
                    x * y,
                    (
                        ({}, False),
                        ({y: x}, True),
                    ),
                ),
                (
                    x * x,
                    x * y,
                    (
                        ({}, False),
                        ({y: x}, True),
                    ),
                ),
                (
                    x * x + z,
                    x * y + t,
                    (({}, False), ({y: x}, False), ({y: x, t: z}, True)),
                ),
            ],
        )
