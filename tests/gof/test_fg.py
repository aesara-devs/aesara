import pickle

import numpy as np
import pytest

from tests.gof.utils import MyVariable, MyVariable2, op1, op2, op3
from theano import change_flags
from theano.gof.fg import FunctionGraph, MissingInputError
from theano.gof.toolbox import BadOptimization


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
        assert fg.clients(var1) == [(var3.owner, 0)]
        assert fg.clients(var2) == [(var4.owner, 1)]
        assert fg.clients(var3) == [(var4.owner, 0), ("output", 0)]
        assert fg.clients(var4) == [("output", 1)]

    def test_remove_client(self):
        var1 = MyVariable("var1")
        var2 = MyVariable("var2")
        var3 = op1(var2, var1)
        var4 = op2(var3, var2)
        var5 = op3(var4, var2, var2)
        fg = FunctionGraph([var1, var2], [var3, var5], clone=False)

        assert fg.variables == {var1, var2, var3, var4, var5}
        assert fg.clients(var2) == [
            (var3.owner, 0),
            (var4.owner, 1),
            (var5.owner, 1),
            (var5.owner, 2),
        ]

        fg.remove_client(var2, (var4.owner, 1))

        assert fg.clients(var2) == [
            (var3.owner, 0),
            (var5.owner, 1),
            (var5.owner, 2),
        ]

        fg.remove_client(var1, (var3.owner, 1))

        assert fg.clients(var1) == []

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

        var5 = MyVariable("var5")
        var6 = op2(var5)

        with pytest.raises(MissingInputError):
            fg.import_node(var6.owner)

        var6 = op2(var2)
        assert not hasattr(var6.owner.tag, "imported_by")
        fg.import_node(var6.owner)

        assert hasattr(var6.owner.tag, "imported_by")
        assert var6 in fg.variables
        assert var6.owner in fg.apply_nodes
        assert (var6.owner, 0) in var2.clients

    def test_import_var(self):

        var1 = MyVariable("var1")
        var2 = MyVariable("var2")
        var3 = op1(var2, var1)
        var4 = op2(var3, var2)
        var5 = op3(var4, var2, var2)
        fg = FunctionGraph([var1, var2], [var3, var5], clone=False)

        with pytest.raises(MissingInputError):
            var0 = MyVariable("var0")
            # We can't import a new `FunctionGraph` input (i.e. something
            # without an owner)
            fg.import_var(var0, "testing")

        var5 = op2()
        # We can import variables with owners
        fg.import_var(var5, "testing")
        assert var5 in fg.variables
        assert var5.owner in fg.apply_nodes

        with pytest.raises(TypeError, match="Computation graph contains.*"):
            from theano.gof.null_type import NullType

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
            fg.change_input("output", 1, var6)

        with pytest.raises(TypeError):
            fg.change_input(var5.owner, 1, var6)

        old_apply_nodes = set(fg.apply_nodes)
        old_variables = set(fg.variables)
        old_var5_clients = list(var5.clients)

        # We're replacing with the same variable, so nothing should happen
        fg.change_input(var5.owner, 1, var2)

        assert old_apply_nodes == fg.apply_nodes
        assert old_variables == fg.variables
        assert old_var5_clients == var5.clients

        # Perform a valid `Apply` node input change
        fg.change_input(var5.owner, 1, var1)

        assert var5.owner.inputs[1] is var1
        assert (var5.owner, 1) not in var2.clients

    @change_flags(compute_test_value="raise")
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

        with pytest.raises(Exception, match="Cannot replace.*"):
            var4.fgraph = object()
            # Trigger a `FunctionGraph` ownership error
            fg.replace(var4, var1, verbose=True)

        var4.fgraph = fg

        with pytest.raises(BadOptimization):
            var0 = MyVariable2("var0")
            # The types don't match and one cannot be converted to the other
            fg.replace(var3, var0)

        # Test a basic replacement
        fg.replace_all([(var3, var1)])
        assert var3 not in fg.variables
        assert fg.apply_nodes == {var4.owner, var5.owner}
        assert var4.owner.inputs == [var1, var2]

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
            var0.fgraph = object()

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
            var2.clients.remove((var5.owner, 1))

            fg.check_integrity()

        var2.clients.append((var5.owner, 1))

        with pytest.raises(Exception, match="The variables are.*"):
            fg.variables.remove(var4)

            fg.check_integrity()

        fg.variables.add(var4)

        with pytest.raises(Exception, match="Undeclared input.*"):
            var6 = MyVariable2("var6")
            var6.fgraph = fg
            var6.clients = [(var5.owner, 3)]
            fg.variables.add(var6)
            var5.owner.inputs.append(var6)

            fg.check_integrity()

        fg.variables.remove(var6)
        var5.owner.inputs.remove(var6)

        # TODO: What if the index value is greater than 1?  It will throw an
        # `IndexError`, but that doesn't sound like anything we'd want.
        with pytest.raises(Exception, match="Inconsistent clients list.*"):
            var4.clients.append(("output", 1))

            fg.check_integrity()

        var4.clients.remove(("output", 1))

        with pytest.raises(Exception, match="Client not in FunctionGraph.*"):
            var4.clients.append((var6.owner, 0))

            fg.check_integrity()

        var4.clients.remove((var6.owner, 0))

        with pytest.raises(Exception, match="Inconsistent clients list.*"):
            var4.clients.append((var3.owner, 0))

            fg.check_integrity()
