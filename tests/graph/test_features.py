import pickle

import pytest

from aesara.graph.basic import Apply, Variable
from aesara.graph.features import Feature, History, NodeFinder, ReplaceValidate
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import Op
from aesara.graph.type import Type
from tests.graph.utils import MyVariable, MyVariable2, op1, op2


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

        x, y, z = MyVariable("x"), MyVariable("y"), MyVariable("z")
        e0 = dot(y, z)
        e = add(add(sigmoid(x), sigmoid(sigmoid(z))), dot(add(x, y), e0))
        g = FunctionGraph([x, y, z], [e], clone=False)
        nf = NodeFinder()
        g.attach_feature(nf)

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

        g.remove_feature(nf)
        assert not hasattr(g, "get_nodes")
        assert not hasattr(g, "_finder_ops_to_nodes")


class TestReplaceValidate:
    def test_basic(self):
        var1 = MyVariable("var1")
        var2 = MyVariable("var2")
        var3 = op1(var2, var1)
        fg = FunctionGraph([var1, var2], [var3], clone=False)

        # One should already be attached
        (rv_feature,) = fg._features
        assert isinstance(rv_feature, ReplaceValidate)

        fg.attach_feature(ReplaceValidate())

        assert hasattr(fg, "_replace_nodes_removed")
        assert hasattr(fg, "_replace_validate_failed")

        rv_feature.replace_all_validate(fg, [(var3, var1)])
        assert var3 not in fg.variables

        # This `Variable` has a different `Type`
        var4 = MyVariable2("var4")
        with pytest.raises(TypeError):
            rv_feature.replace_all_validate(fg, [(var1, var4)])

        fg.remove_feature(rv_feature)
        assert not hasattr(fg, "_replace_nodes_removed")
        assert not hasattr(fg, "_replace_validate_failed")

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

    def test_pickle(self):
        var1 = MyVariable("var1")
        var2 = MyVariable("var2")
        var3 = op1(var2, var1)
        fg = FunctionGraph([var1, var2], [var3], clone=False)

        rv_feature = ReplaceValidate()
        fg.attach_feature(rv_feature)

        fg_pkld = pickle.dumps(fg)
        fg_unpkld = pickle.loads(fg_pkld)

        assert ReplaceValidate in {type(ft) for ft in fg_unpkld._features}
        assert all(
            hasattr(fg, attr)
            for attr in (
                "replace_validate",
                "replace_all_validate",
                "replace_all_validate_remove",
                "checkpoint",
                "revert",
                "validate",
                "consistent",
            )
        )


class TestHistory:
    def test_basic(self):
        var1 = MyVariable("var1")
        var2 = MyVariable("var2")
        var3 = op1(var2, var1)
        fg = FunctionGraph([var1, var2], [var3], clone=False)

        hf = History()
        fg.attach_feature(hf)

        assert hasattr(fg, "_history_is_reverting")
        assert hasattr(fg, "_history_history")

        chkpnt = fg.checkpoint()

        fg.replace_all([(var3, op2(var2, var1))])
        assert var3 not in fg.variables

        assert fg._history_history
        fg.revert(chkpnt)
        assert not fg._history_is_reverting

        assert not fg._history_history
        assert var3 in fg.variables

    def test_pickle(self):
        var1 = MyVariable("var1")
        var2 = MyVariable("var2")
        var3 = op1(var2, var1)
        fg = FunctionGraph([var1, var2], [var3], clone=False)

        hf = History()
        fg.attach_feature(hf)

        fg_pkld = pickle.dumps(fg)
        fg_unpkld = pickle.loads(fg_pkld)

        assert any(isinstance(ft, History) for ft in fg_unpkld._features)
        assert all(
            hasattr(fg, attr)
            for attr in (
                "checkpoint",
                "revert",
                "validate",
                "consistent",
            )
        )
