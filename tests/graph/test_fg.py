import pickle

import numpy as np
import pytest

from aesara.configdefaults import config
from aesara.graph.fg import FunctionGraph, MissingInputError
from tests.graph.utils import MyConstant, MyVariable, MyVariable2, op1, op2, op3


class TestFunctionGraph:
    def test_pickle(self):
        var1 = op1()
        var2 = op2()
        var3 = op1(var1)
        var4 = op2(var3, var2)
        func = FunctionGraph([var1, var2], [var4])

        s = pickle.dumps(func)
        new_func = pickle.loads(s)

        assert all(type(a) == type(b) for a, b in zip(func.inputs, new_func.inputs))
        assert all(type(a) == type(b) for a, b in zip(func.outputs, new_func.outputs))
        assert all(
            type(a.op) is type(b.op)  # noqa: E721
            for a, b in zip(func.apply_nodes, new_func.apply_nodes)
        )
        assert all(a.type == b.type for a, b in zip(func.variables, new_func.variables))

    def test_validate_inputs(self):
        var1 = op1()
        var2 = op2()

        with pytest.raises(TypeError):
            FunctionGraph(var1, [var2])

        with pytest.raises(TypeError):
            FunctionGraph([var1], var2)

        with pytest.raises(ValueError):
            var3 = op1(var1)
            FunctionGraph([var3], [var2], clone=False)

        with pytest.raises(ValueError):
            var3 = op1(var1)
            FunctionGraph([var3], clone=False)

    def test_init(self):
        var1 = MyVariable("var1")
        var2 = MyVariable("var2")
        var3 = op1(var1)
        var4 = op2(var3, var2)
        fg = FunctionGraph([var1, var2], [var3, var4], clone=False)
        assert fg.inputs == [var1, var2]
        assert fg.outputs == [var3, var4]
        assert fg.apply_nodes == {var3.owner, var4.owner}
        assert fg.update_mapping is None
        assert fg.check_integrity() is None
        assert fg.variables == {var1, var2, var3, var4}
        assert fg.get_clients(var1) == [(var3.owner, 0)]
        assert fg.get_clients(var2) == [(var4.owner, 1)]
        assert fg.get_clients(var3) == [(var4.owner, 0), ("output", 0)]
        assert fg.get_clients(var4) == [("output", 1)]

        varC = MyConstant("varC")
        var5 = op1(var1, varC)
        fg = FunctionGraph(outputs=[var3, var4, var5], clone=False)
        assert fg.inputs == [var1, var2]

        memo = {}
        fg = FunctionGraph(outputs=[var3, var4], clone=True, memo=memo)

        assert memo[var1].type == var1.type
        assert memo[var1].name == var1.name
        assert memo[var2].type == var2.type
        assert memo[var2].name == var2.name
        assert var3 in memo
        assert var4 in memo

    def test_remove_client(self):
        var1 = MyVariable("var1")
        var2 = MyVariable("var2")
        var3 = op1(var2, var1)
        var4 = op2(var3, var2)
        var5 = op3(var4, var2, var2)
        fg = FunctionGraph([var1, var2], [var3, var5], clone=False)

        assert fg.variables == {var1, var2, var3, var4, var5}
        assert fg.get_clients(var2) == [
            (var3.owner, 0),
            (var4.owner, 1),
            (var5.owner, 1),
            (var5.owner, 2),
        ]

        fg.remove_client(var2, (var4.owner, 1))

        assert fg.get_clients(var2) == [
            (var3.owner, 0),
            (var5.owner, 1),
            (var5.owner, 2),
        ]

        fg.remove_client(var1, (var3.owner, 1))

        assert fg.get_clients(var1) == []

        assert var4.owner in fg.apply_nodes

        # This next `remove_client` should trigger a complete removal of `var4`'s
        # variables and `Apply` node from the `FunctionGraph`.
        #
        # Also, notice that we already removed `var4` from `var2`'s client list
        # above, so, when we completely remove `var4`, `fg.remove_client` will
        # attempt to remove `(var4.owner, 1)` from `var2`'s client list again.
        # This attempt would previously raise a `ValueError` exception, because
        # the entry was not in the list.
        fg.remove_client(var4, (var5.owner, 0), reason="testing")

        assert var4.owner not in fg.apply_nodes
        assert var4.owner.tag.removed_by == ["testing"]
        assert not any(o in fg.variables for o in var4.owner.outputs)

    def test_import_node(self):

        var1 = MyVariable("var1")
        var2 = MyVariable("var2")
        var3 = op1(var2, var1)
        var4 = op2(var3, var2)
        var5 = op3(var4, var2, var2)
        fg = FunctionGraph([var1, var2], [var3, var5], clone=False)

        var8 = MyVariable("var8")
        var6 = op2(var8)

        with pytest.raises(MissingInputError):
            fg.import_node(var6.owner)

        assert var8 not in fg.variables

        fg.import_node(var6.owner, import_missing=True)
        assert var8 in fg.inputs
        assert var6.owner in fg.apply_nodes

        var7 = op2(var2)
        assert not hasattr(var7.owner.tag, "imported_by")
        fg.import_node(var7.owner)

        assert hasattr(var7.owner.tag, "imported_by")
        assert var7 in fg.variables
        assert var7.owner in fg.apply_nodes
        assert (var7.owner, 0) in fg.get_clients(var2)

    def test_import_var(self):

        var1 = MyVariable("var1")
        var2 = MyVariable("var2")
        var3 = op1(var2, var1)
        var4 = op2(var3, var2)
        var5 = op3(var4, var2, var2)
        fg = FunctionGraph([var1, var2], [var3, var5], clone=False)

        var0 = MyVariable("var0")

        with pytest.raises(MissingInputError):
            # We can't import a new `FunctionGraph` input (i.e. something
            # without an owner), at least not without setting `import_missing`
            fg.import_var(var0, "testing")

        fg.import_var(var0, import_missing=True)

        assert var0 in fg.inputs

        var5 = op2()
        # We can import variables with owners
        fg.import_var(var5, "testing")
        assert var5 in fg.variables
        assert var5.owner in fg.apply_nodes

        with pytest.raises(TypeError, match="Computation graph contains.*"):
            from aesara.graph.null_type import NullType

            fg.import_var(NullType()(), "testing")

    def test_change_input(self):

        var1 = MyVariable("var1")
        var2 = MyVariable("var2")
        var3 = op1(var2, var1)
        var4 = op2(var3, var2)
        var5 = op3(var4, var2, var2)
        fg = FunctionGraph([var1, var2], [var3, var5], clone=False)

        var6 = MyVariable2("var6")
        with pytest.raises(TypeError):
            fg.change_node_input("output", 1, var6)

        with pytest.raises(TypeError):
            fg.change_node_input(var5.owner, 1, var6)

        old_apply_nodes = set(fg.apply_nodes)
        old_variables = set(fg.variables)
        old_var5_clients = list(fg.get_clients(var5))

        # We're replacing with the same variable, so nothing should happen
        fg.change_node_input(var5.owner, 1, var2)

        assert old_apply_nodes == fg.apply_nodes
        assert old_variables == fg.variables
        assert old_var5_clients == fg.get_clients(var5)

        # Perform a valid `Apply` node input change
        fg.change_node_input(var5.owner, 1, var1)

        assert var5.owner.inputs[1] is var1
        assert (var5.owner, 1) not in fg.get_clients(var2)

    @config.change_flags(compute_test_value="raise")
    def test_replace_test_value(self):

        var1 = MyVariable("var1")
        var1.tag.test_value = 1
        var2 = MyVariable("var2")
        var2.tag.test_value = 2
        var3 = op1(var2, var1)
        var4 = op2(var3, var2)
        var4.tag.test_value = np.array([1, 2])
        var5 = op3(var4, var2, var2)
        fg = FunctionGraph([var1, var2], [var3, var5], clone=False)

        var6 = op3()
        var6.tag.test_value = np.array(0)

        assert var6.tag.test_value.shape != var4.tag.test_value.shape

        with pytest.raises(AssertionError, match="The replacement.*"):
            fg.replace(var4, var6)

    def test_replace(self):

        var1 = MyVariable("var1")
        var2 = MyVariable("var2")
        var3 = op1(var2, var1)
        var4 = op2(var3, var2)
        var5 = op3(var4, var2, var2)
        fg = FunctionGraph([var1, var2], [var3, var5], clone=False)

        with pytest.raises(TypeError):
            var0 = MyVariable2("var0")
            # The types don't match and one cannot be converted to the other
            fg.replace(var3, var0)

        # Test a basic replacement
        fg.replace_all([(var3, var1)])
        assert var3 not in fg.variables
        assert fg.apply_nodes == {var4.owner, var5.owner}
        assert var4.owner.inputs == [var1, var2]

    def test_replace_verbose(self, capsys):

        var1 = MyVariable("var1")
        var2 = MyVariable("var2")
        var3 = op1(var2, var1)
        fg = FunctionGraph([var1, var2], [var3], clone=False)

        fg.replace(var3, var1, reason="test-reason", verbose=True)

        capres = capsys.readouterr()
        assert capres.err == ""
        assert (
            "optimizer: rewrite test-reason replaces Op1.0 of Op1(var2, var1) with var1 of None"
            in capres.out
        )

    def test_replace_circular(self):
        """`FunctionGraph` allows cycles--for better or worse."""

        var1 = MyVariable("var1")
        var2 = MyVariable("var2")
        var3 = op1(var2, var1)
        var4 = op2(var3, var2)
        var5 = op3(var4, var2, var2)
        fg = FunctionGraph([var1, var2], [var3, var5], clone=False)

        fg.replace_all([(var3, var4)])

        # The following works (and is kind of gross), because `var4` has been
        # mutated in-place
        assert fg.apply_nodes == {var4.owner, var5.owner}
        assert var4.owner.inputs == [var4, var2]

    def test_replace_bad_state(self):

        var1 = MyVariable("var1")
        var2 = MyVariable("var2")
        var3 = op1(var2, var1)
        var4 = op2(var3, var2)
        var5 = op3(var4, var2, var2)
        fg = FunctionGraph([var1, var2], [var3, var5], clone=False)

        with pytest.raises(MissingInputError):
            var0 = MyVariable("var0")

            # FIXME TODO XXX: This breaks the state of the `FunctionGraph`,
            # because it doesn't check for validity of the replacement *first*.
            fg.replace(var1, var0, verbose=True)

    def test_check_integrity(self):

        var1 = MyVariable("var1")
        var2 = MyVariable("var2")
        var3 = op1(var2, var1)
        var4 = op2(var3, var2)
        var5 = op3(var4, var2, var2)
        fg = FunctionGraph([var1, var2], [var3, var5], clone=False)

        with pytest.raises(Exception, match="The nodes are .*"):
            fg.apply_nodes.remove(var5.owner)

            fg.check_integrity()

        with pytest.raises(Exception, match="Inconsistent clients.*"):
            fg.apply_nodes.add(var5.owner)
            fg.remove_client(var2, (var5.owner, 1))

            fg.check_integrity()

        fg.add_client(var2, (var5.owner, 1))

        with pytest.raises(Exception, match="The variables are.*"):
            fg.variables.remove(var4)

            fg.check_integrity()

        fg.variables.add(var4)

        with pytest.raises(Exception, match="Undeclared input.*"):
            var6 = MyVariable2("var6")
            fg.clients[var6] = [(var5.owner, 3)]
            fg.variables.add(var6)
            var5.owner.inputs.append(var6)

            fg.check_integrity()

        fg.variables.remove(var6)
        var5.owner.inputs.remove(var6)

        # TODO: What if the index value is greater than 1?  It will throw an
        # `IndexError`, but that doesn't sound like anything we'd want.
        with pytest.raises(Exception, match="Inconsistent clients list.*"):
            fg.add_client(var4, ("output", 1))

            fg.check_integrity()

        fg.remove_client(var4, ("output", 1))

        with pytest.raises(Exception, match="Client not in FunctionGraph.*"):
            fg.add_client(var4, (var6.owner, 0))

            fg.check_integrity()

        fg.remove_client(var4, (var6.owner, 0))

        with pytest.raises(Exception, match="Inconsistent clients list.*"):
            fg.add_client(var4, (var3.owner, 0))

            fg.check_integrity()

    def test_contains(self):

        var1 = MyVariable("var1")
        var2 = MyVariable("var2")
        var3 = op1(var2, var1)
        var4 = op2(var3, var2)
        var5 = op3(var4, var2, var2)
        fg = FunctionGraph([var1, var2], [var3, var5], clone=False)

        assert var1 in fg
        assert var3 in fg
        assert var3.owner in fg
        assert var5 in fg
        assert var5.owner in fg
