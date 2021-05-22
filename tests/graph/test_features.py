from aesara.graph.basic import Apply, Variable
from aesara.graph.features import NodeFinder
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import Op
from aesara.graph.type import Type


class TestNodeFinder:
    def test_straightforward(self):
        class MyType(Type):
            def __init__(self, name):
                self.name = name

            def filter(self, *args, **kwargs):
                raise NotImplementedError()

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

            def perform(self, *args, **kwargs):
                raise NotImplementedError()

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
