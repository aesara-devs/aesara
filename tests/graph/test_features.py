import pytest

from aesara.graph.basic import Apply, Variable
from aesara.graph.features import (
    Feature,
    InnerGraphWatcher,
    NodeFinder,
    ReplaceValidate,
)
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import Op
from aesara.graph.type import Type
from tests.graph.utils import MyInnerGraphOp, MyVariable, op1, op2


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
            if len([t for t in g.get_nodes(type)]) != num:
                raise Exception("Expected: %i times %s" % (num, type))
        new_e0 = add(y, z)
        assert e0.owner in g.get_nodes(dot)
        assert new_e0.owner not in g.get_nodes(add)
        g.replace(e0, new_e0)
        assert e0.owner not in g.get_nodes(dot)
        assert new_e0.owner in g.get_nodes(add)
        for type, num in ((add, 4), (sigmoid, 3), (dot, 1)):
            if len([t for t in g.get_nodes(type)]) != num:
                raise Exception("Expected: %i times %s" % (num, type))


class TestReplaceValidate:
    def test_verbose(self, capsys):
        var1 = MyVariable("var1")
        var2 = MyVariable("var2")
        var3 = op1(var2, var1)
        fg = FunctionGraph([var1, var2], [var3], clone=False)

        rv_feature = ReplaceValidate()
        fg.attach_feature(rv_feature)
        rv_feature.replace_all_validate(
            fg, [(var3, var1)], reason="test-reason", verbose=True
        )

        capres = capsys.readouterr()
        assert capres.err == ""
        assert (
            "rewriting: rewrite test-reason replaces Op1.0 of Op1(var2, var1) with var1 of None"
            in capres.out
        )

        class TestFeature(Feature):
            def validate(self, *args):
                raise Exception()

        fg.attach_feature(TestFeature())

        with pytest.raises(Exception):
            rv_feature.replace_all_validate(
                fg, [(var3, var1)], reason="test-reason", verbose=True
            )

        capres = capsys.readouterr()
        assert "rewriting: validate failed on node Op1.0" in capres.out


class TestInnerGraphWatcher:
    def test_basic(self):
        var1 = MyVariable("var1")
        var2 = MyVariable("var2")
        var3 = MyVariable("var3")

        igo = MyInnerGraphOp([var1, var2], [op2(var1, var2)])

        igo_outer = igo(var2, var3)
        igo_outer.name = "igo_outer"
        outer_out = op1(var1, igo_outer)
        fg = FunctionGraph([var1, var2, var3], [outer_out], clone=False)

        igw = InnerGraphWatcher()
        fg.attach_feature(igw)

        assert hasattr(fg, "_nodes_to_inner_graphs")
        assert igo_outer.owner in fg._nodes_to_inner_graphs
        assert fg._parent_graph is None
        assert igo.fgraph._parent_graph is fg

        var4 = MyVariable("var4")
        new_igo_outer = igo(var1, var4)
        new_igo_outer.name = "new_igo_outer"
        # This adds a new inner-graph
        fg.replace(var3, new_igo_outer, import_missing=True)

        assert set(fg._nodes_to_inner_graphs.keys()) == {
            new_igo_outer.owner,
            igo_outer.owner,
        }

        # This undoes the last inner-graph addition
        fg.replace(new_igo_outer, var3, import_missing=True)
        assert set(fg._nodes_to_inner_graphs.keys()) == {igo_outer.owner}

        # This should remove all inner-graphs
        fg.replace(outer_out, op1(var1, var2), import_missing=True)
        assert not fg._nodes_to_inner_graphs
        assert len(fg._nodes_to_inner_graphs.maps) == 1

    def test_nested(self):
        """Make sure that inner-graphs created within inner-graphs are tracked."""
        var1 = MyVariable("var1")
        var2 = MyVariable("var2")

        igo1 = MyInnerGraphOp([var1], [op1(var1)])
        igo1.name = "igo1"
        igo2 = MyInnerGraphOp([var1], [igo1(var1)])
        igo2.name = "igo2"

        igo2_out = igo2(var1)
        out = op2(var2, igo2_out)

        fg = FunctionGraph([var1, var2], [out], clone=False)

        igw = InnerGraphWatcher()
        fg.attach_feature(igw)

        assert hasattr(fg, "_nodes_to_inner_graphs")
        assert igo2_out.owner in fg._nodes_to_inner_graphs

        assert fg._parent_graph is None
        assert igo1.fgraph._parent_graph is fg
        assert igo2.fgraph._parent_graph is fg
        assert len(fg._nodes_to_inner_graphs.maps) == 2

        # This is the "outer" graph within `fg`
        outer_fg = fg._nodes_to_inner_graphs[igo2_out.owner]
        igo1_out = outer_fg.outputs[0]
        assert igo1_out.owner.op == igo1

        assert igo1_out.owner in fg._nodes_to_inner_graphs
        inner_fg = fg._nodes_to_inner_graphs[igo1_out.owner]
        op1_out = inner_fg.outputs[0]
        assert op1_out.owner.op == op1

        assert igo1_out.owner in fg._nodes_to_inner_graphs

        # Remove the inner-inner-graph by replacing it in the inner-graph of `fg`
        outer_fg.replace(igo1_out, outer_fg.inputs[0], import_missing=True)
        assert igo1_out.owner not in fg._nodes_to_inner_graphs
