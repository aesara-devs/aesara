import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
from typing_extensions import Literal

from aesara.configdefaults import config
from aesara.graph.basic import Apply, NominalVariable
from aesara.graph.features import Feature
from aesara.graph.fg import FunctionGraph
from aesara.graph.utils import MissingInputError
from tests.graph.utils import (
    MyConstant,
    MyOp,
    MyType,
    MyVariable,
    MyVariable2,
    op1,
    op2,
    op3,
)


class CallbackTracker(Feature):
    def __init__(self):
        self.callback_history: List[
            Tuple[
                Literal["attach", "detach", "import", "change_input", "prune"],
                Tuple[Any, ...],
                Dict[Any, Any],
            ]
        ] = []

    def on_attach(self, *args, **kwargs):
        self.callback_history.append(("attach", args, kwargs))

    def on_detach(self, *args, **kwargs):
        self.callback_history.append(("detach", args, kwargs))

    def on_import(self, *args, **kwargs):
        self.callback_history.append(("import", args, kwargs))

    def on_change_input(self, *args, **kwargs):
        self.callback_history.append(("change_input", args, kwargs))

    def on_prune(self, *args, **kwargs):
        self.callback_history.append(("prune", args, kwargs))


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
        cb_tracker = CallbackTracker()
        fg = FunctionGraph(
            [var1, var2], [var3, var4], clone=False, features=[cb_tracker]
        )

        assert fg.inputs == [var1, var2]
        assert fg.outputs == [var3, var4]
        assert fg.apply_nodes == {var3.owner, var4.owner}
        assert fg.update_mapping is None
        assert fg.check_integrity() is None
        assert fg.variables == {var1, var2, var3, var4}
        assert fg.get_clients(var1) == [(var3.owner, 0)]
        assert fg.get_clients(var2) == [(var4.owner, 1)]
        assert fg.get_clients(var3) == [("output", 0), (var4.owner, 0)]
        assert fg.get_clients(var4) == [("output", 1)]

        assert len(cb_tracker.callback_history) == 3
        assert cb_tracker.callback_history[0] == ("attach", (fg,), {})
        assert cb_tracker.callback_history[1] == (
            "import",
            (fg, var3.owner, "init"),
            {},
        )
        assert cb_tracker.callback_history[2] == (
            "import",
            (fg, var4.owner, "init"),
            {},
        )

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
        cb_tracker = CallbackTracker()
        fg = FunctionGraph(
            [var1, var2], [var3, var5], clone=False, features=[cb_tracker]
        )

        assert fg.variables == {var1, var2, var3, var4, var5}
        assert fg.get_clients(var2) == [
            (var3.owner, 0),
            (var4.owner, 1),
            (var5.owner, 1),
            (var5.owner, 2),
        ]

        assert len(cb_tracker.callback_history) == 4
        assert cb_tracker.callback_history[0] == ("attach", (fg,), {})
        assert cb_tracker.callback_history[1] == (
            "import",
            (fg, var3.owner, "init"),
            {},
        )
        assert cb_tracker.callback_history[2] == (
            "import",
            (fg, var4.owner, "init"),
            {},
        )
        assert cb_tracker.callback_history[3] == (
            "import",
            (fg, var5.owner, "init"),
            {},
        )
        cb_tracker.callback_history.clear()

        fg.remove_client(var2, (var4.owner, 1))

        assert fg.get_clients(var2) == [
            (var3.owner, 0),
            (var5.owner, 1),
            (var5.owner, 2),
        ]

        assert len(cb_tracker.callback_history) == 0

        fg.remove_client(var1, (var3.owner, 1))

        assert fg.get_clients(var1) == []

        assert var4.owner in fg.apply_nodes

        assert len(cb_tracker.callback_history) == 0

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

        assert len(cb_tracker.callback_history) == 1
        assert cb_tracker.callback_history[0] == (
            "prune",
            (fg, var4.owner, "testing"),
            {},
        )

    def test_import_node(self):

        var1 = MyVariable("var1")
        var2 = MyVariable("var2")
        var3 = op1(var2, var1)
        var4 = op2(var3, var2)
        var5 = op3(var4, var2, var2)
        cb_tracker = CallbackTracker()
        fg = FunctionGraph(
            [var1, var2], [var3, var5], clone=False, features=[cb_tracker]
        )

        assert len(cb_tracker.callback_history) == 4
        assert cb_tracker.callback_history[0] == ("attach", (fg,), {})
        assert cb_tracker.callback_history[1] == (
            "import",
            (fg, var3.owner, "init"),
            {},
        )
        assert cb_tracker.callback_history[2] == (
            "import",
            (fg, var4.owner, "init"),
            {},
        )
        assert cb_tracker.callback_history[3] == (
            "import",
            (fg, var5.owner, "init"),
            {},
        )
        cb_tracker.callback_history.clear()

        var8 = MyVariable("var8")
        var6 = op2(var8)

        with pytest.raises(MissingInputError):
            fg.import_node(var6.owner)

        assert var8 not in fg.variables
        assert len(cb_tracker.callback_history) == 0

        fg.import_node(var6.owner, import_missing=True)
        assert var8 in fg.inputs
        assert var6.owner in fg.apply_nodes

        assert len(cb_tracker.callback_history) == 1
        assert cb_tracker.callback_history[0] == ("import", (fg, var6.owner, None), {})
        cb_tracker.callback_history.clear()

        var7 = op2(var2)
        assert not hasattr(var7.owner.tag, "imported_by")
        fg.import_node(var7.owner)

        assert hasattr(var7.owner.tag, "imported_by")
        assert var7 in fg.variables
        assert var7.owner in fg.apply_nodes
        assert (var7.owner, 0) in fg.get_clients(var2)

        assert len(cb_tracker.callback_history) == 1
        assert cb_tracker.callback_history[0] == ("import", (fg, var7.owner, None), {})

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
        var3.name = "var3"
        var4 = op2(var3, var2)
        var4.name = "var4"
        var5 = op3(var4, var2, var2)
        var5.name = "var5"
        cb_tracker = CallbackTracker()
        fg = FunctionGraph(
            [var1, var2], [var3, var5], clone=False, features=[cb_tracker]
        )

        assert len(cb_tracker.callback_history) == 4
        assert cb_tracker.callback_history[0] == ("attach", (fg,), {})
        assert cb_tracker.callback_history[1] == (
            "import",
            (fg, var3.owner, "init"),
            {},
        )
        assert cb_tracker.callback_history[2] == (
            "import",
            (fg, var4.owner, "init"),
            {},
        )
        assert cb_tracker.callback_history[3] == (
            "import",
            (fg, var5.owner, "init"),
            {},
        )
        cb_tracker.callback_history.clear()

        var6 = MyVariable2("var6")
        with pytest.raises(TypeError):
            fg.change_node_input("output", 1, var6)

        with pytest.raises(TypeError):
            fg.change_node_input(var5.owner, 1, var6)

        assert len(cb_tracker.callback_history) == 0

        old_apply_nodes = set(fg.apply_nodes)
        old_variables = set(fg.variables)
        old_var5_clients = list(fg.get_clients(var5))
        old_var5_node = var5.owner

        # We're replacing with the same variable, so nothing should happen
        fg.change_node_input(var5.owner, 1, var2)

        assert len(cb_tracker.callback_history) == 0

        assert old_apply_nodes == fg.apply_nodes
        assert old_variables == fg.variables
        assert old_var5_clients == fg.get_clients(var5)

        # Perform a valid `Apply` node input change
        fg.change_node_input(var5.owner, 1, var1)

        assert var5.owner.inputs == [var4, var1, var2]
        assert fg.outputs[1].owner == var5.owner
        assert (var5.owner, 1) not in fg.get_clients(var2)

        assert len(cb_tracker.callback_history) == 3
        assert cb_tracker.callback_history == [
            ("prune", (fg, old_var5_node, None), {}),
            ("import", (fg, var5.owner, None), {}),
            (
                "change_input",
                (fg, old_var5_node, var5.owner, 1, var2, var1),
                {"reason": None},
            ),
        ]
        cb_tracker.callback_history.clear()

        old_var5_node = var5.owner

        # Perform a valid `Apply` node input change that results in a
        # node removal (i.e. `var4.owner`)
        fg.change_node_input(var5.owner, 0, var1)

        assert var5.owner.inputs[0] is var1
        assert var4 not in fg.clients
        assert var4.owner not in fg.apply_nodes
        assert var4 not in fg.variables

        assert len(cb_tracker.callback_history) == 4
        assert cb_tracker.callback_history[0] == ("prune", (fg, var4.owner, None), {})
        assert cb_tracker.callback_history[1] == (
            "prune",
            (fg, old_var5_node, None),
            {},
        )
        assert cb_tracker.callback_history[2] == ("import", (fg, var5.owner, None), {})
        assert cb_tracker.callback_history[3] == (
            "change_input",
            (fg, old_var5_node, var5.owner, 0, var4, var1),
            {"reason": None},
        )

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
        cb_tracker = CallbackTracker()
        fg = FunctionGraph(
            [var1, var2], [var3, var5], clone=False, features=[cb_tracker]
        )

        assert len(cb_tracker.callback_history) == 4
        assert cb_tracker.callback_history[0] == ("attach", (fg,), {})
        assert cb_tracker.callback_history[1] == (
            "import",
            (fg, var3.owner, "init"),
            {},
        )
        assert cb_tracker.callback_history[2] == (
            "import",
            (fg, var4.owner, "init"),
            {},
        )
        assert cb_tracker.callback_history[3] == (
            "import",
            (fg, var5.owner, "init"),
            {},
        )
        cb_tracker.callback_history.clear()

        with pytest.raises(TypeError):
            var0 = MyVariable2("var0")
            # The types don't match and one cannot be converted to the other
            fg.replace(var3, var0)

        assert len(cb_tracker.callback_history) == 0

        old_var4_node = var4.owner

        # Test a basic replacement
        fg.replace_all([(var3, var1)])
        assert var3 not in fg.variables

        assert fg.apply_nodes == {var4.owner, var5.owner}
        assert var4.owner.inputs == [var1, var2]
        assert fg.outputs == [var1, var5]

        assert len(cb_tracker.callback_history) == 5
        assert cb_tracker.callback_history[0] == (
            "change_input",
            (fg, "output", "output", 0, var3, var1),
            {"reason": None},
        )
        assert cb_tracker.callback_history[1] == ("prune", (fg, var3.owner, None), {})
        assert cb_tracker.callback_history[2] == (
            "prune",
            (fg, old_var4_node, None),
            {},
        )
        assert cb_tracker.callback_history[3] == ("import", (fg, var4.owner, None), {})
        assert cb_tracker.callback_history[4] == (
            "change_input",
            (fg, old_var4_node, var4.owner, 0, var3, var1),
            {"reason": None},
        )

        var3 = op1(var1)
        var4 = op2(var3)
        var5 = op3(var4)
        cb_tracker = CallbackTracker()
        fg = FunctionGraph([var1], [var5], clone=False, features=[cb_tracker])

        assert cb_tracker.callback_history == [
            ("attach", (fg,), {}),
            ("import", (fg, var3.owner, "init"), {}),
            ("import", (fg, var4.owner, "init"), {}),
            ("import", (fg, var5.owner, "init"), {}),
        ]
        cb_tracker.callback_history.clear()

        old_var5_node = var5.owner

        # Test a replacement that would remove the replacement variable
        # (i.e. `var3`) from the graph when the variable to be replaced
        # (i.e. `var4`) is removed
        fg.replace_all([(var4, var3)])

        assert fg.apply_nodes == {var3.owner, var5.owner}
        assert fg.inputs == [var1]
        assert fg.outputs == [var5]
        assert fg.variables == {var1, var3, var5}

        assert cb_tracker.callback_history == [
            ("prune", (fg, var4.owner, None), {}),
            ("prune", (fg, old_var5_node, None), {}),
            ("import", (fg, var5.owner, None), {}),
            (
                "change_input",
                (fg, old_var5_node, var5.owner, 0, var4, var3),
                {"reason": None},
            ),
        ]

        var3 = op1(var1)
        var4 = op2(var3)
        var5 = op3(var4, var4)
        cb_tracker = CallbackTracker()
        fg = FunctionGraph([var1], [var5], clone=False, features=[cb_tracker])

        assert cb_tracker.callback_history == [
            ("attach", (fg,), {}),
            ("import", (fg, var3.owner, "init"), {}),
            ("import", (fg, var4.owner, "init"), {}),
            ("import", (fg, var5.owner, "init"), {}),
        ]
        cb_tracker.callback_history.clear()

        old_var5_node = var5.owner

        # Test multiple `change_node_input` calls on the same node
        fg.replace_all([(var4, var3)])

        assert fg.apply_nodes == {var3.owner, var5.owner}
        assert fg.inputs == [var1]
        assert fg.outputs == [var5]
        assert fg.variables == {var1, var3, var5}

        tmp_var5_node = Apply(op3, [var3, var4], [MyVariable("var5_tmp")])

        assert cb_tracker.callback_history == [
            ("prune", (fg, old_var5_node, None), {}),
            ("import", (fg, tmp_var5_node, None), {}),
            (
                "change_input",
                (fg, old_var5_node, tmp_var5_node, 0, var4, var3),
                {"reason": None},
            ),
            ("prune", (fg, var4.owner, None), {}),
            ("prune", (fg, tmp_var5_node, None), {}),
            ("import", (fg, var5.owner, None), {}),
            (
                "change_input",
                (fg, tmp_var5_node, var5.owner, 1, var4, var3),
                {"reason": None},
            ),
        ]

    def test_replace_outputs(self):
        var1 = MyVariable("var1")
        var2 = MyVariable("var2")
        var3 = op1(var1)
        var4 = op2(var2)
        cb_tracker = CallbackTracker()
        fg = FunctionGraph(
            [var1, var2], [var3, var4, var3], clone=False, features=[cb_tracker]
        )

        fg.replace_all([(var3, var1)])
        assert var3 not in fg.variables

        assert fg.apply_nodes == {var4.owner}
        assert fg.outputs == [var1, var4, var1]

        assert cb_tracker.callback_history == [
            ("attach", (fg,), {}),
            ("import", (fg, var3.owner, "init"), {}),
            ("import", (fg, var4.owner, "init"), {}),
            ("change_input", (fg, "output", "output", 0, var3, var1), {"reason": None}),
            ("prune", (fg, var3.owner, None), {}),
            ("change_input", (fg, "output", "output", 2, var3, var1), {"reason": None}),
        ]

    def test_replace_contract(self):
        x = MyVariable("x")
        v1 = op1(x)
        v2 = op1(v1)
        v3 = op1(v2)
        v4 = op1(v3)

        v1.name = "v1"
        v2.name = "v2"
        v3.name = "v3"
        v4.name = "v4"

        cb_tracker = CallbackTracker()
        fg = FunctionGraph([x], [v4], clone=False, features=[cb_tracker])

        assert cb_tracker.callback_history == [
            ("attach", (fg,), {}),
            ("import", (fg, v1.owner, "init"), {}),
            ("import", (fg, v2.owner, "init"), {}),
            ("import", (fg, v3.owner, "init"), {}),
            ("import", (fg, v4.owner, "init"), {}),
        ]
        cb_tracker.callback_history.clear()

        old_v3_node = v3.owner
        old_v4_node = v4.owner

        # This replacement should produce a new `Apply` node that's equivalent
        # to `v2` and try to replace `v3`'s node with that one.  In other
        # words, the replacement creates a new node that's already in the
        # `FunctionGraph`.
        # The end result is `v3 = v2`.
        fg.replace_all([(v2, v1)])

        assert v2 not in fg.variables
        assert fg.clients == {
            x: [(v1.owner, 0)],
            v1: [(v3.owner, 0)],
            # v2: [],
            v3: [(v4.owner, 0)],
            v4: [("output", 0)],
        }
        assert fg.apply_nodes == {v4.owner, v3.owner, v1.owner}
        assert v2 not in set(sum((n.outputs for n in fg.apply_nodes), []))

        assert cb_tracker.callback_history == [
            ("prune", (fg, old_v3_node, None), {}),
            ("import", (fg, v3.owner, None), {}),
            ("change_input", (fg, old_v3_node, v3.owner, 0, v2, v1), {"reason": None}),
        ]

        # Let's try the same thing at a different point in the chain
        x = MyVariable("x")
        v1 = op1(x)
        v2 = op1(v1)
        v3 = op1(v2)
        v4 = op1(v3)

        v1.name = "v1"
        v2.name = "v2"
        v3.name = "v3"
        v4.name = "v4"

        cb_tracker = CallbackTracker()
        fg = FunctionGraph([x], [v4], clone=False, features=[cb_tracker])

        assert cb_tracker.callback_history == [
            ("attach", (fg,), {}),
            ("import", (fg, v1.owner, "init"), {}),
            ("import", (fg, v2.owner, "init"), {}),
            ("import", (fg, v3.owner, "init"), {}),
            ("import", (fg, v4.owner, "init"), {}),
        ]
        cb_tracker.callback_history.clear()

        old_v4_node = v4.owner

        fg.replace_all([(v3, v2)])

        assert v3 not in fg.variables
        assert fg.clients == {
            x: [(v1.owner, 0)],
            v1: [(v2.owner, 0)],
            v2: [(v4.owner, 0)],
            # v3: [],
            v4: [("output", 0)],
        }
        assert fg.apply_nodes == {v4.owner, v2.owner, v1.owner}
        assert v3 not in set(sum((n.outputs for n in fg.apply_nodes), []))

        exp_res = [
            ("prune", (fg, old_v4_node, None), {}),
            ("import", (fg, v4.owner, None), {}),
            ("change_input", (fg, old_v4_node, v4.owner, 0, v3, v2), {"reason": None}),
        ]
        assert cb_tracker.callback_history == exp_res

    def test_replace_verbose(self, capsys):

        var1 = MyVariable("var1")
        var2 = MyVariable("var2")
        var3 = op1(var2, var1)
        fg = FunctionGraph([var1, var2], [var3], clone=False)

        fg.replace(var3, var1, reason="test-reason", verbose=True)

        capres = capsys.readouterr()
        assert capres.err == ""
        assert capres.out.startswith("rewriting: rewrite test-reason replaces")

    def test_replace_circular(self):
        """`FunctionGraph` allows cycles--for better or worse."""

        var1 = MyVariable("var1")
        var2 = MyVariable("var2")
        var3 = op1(var2, var1)
        var4 = op2(var3, var2)
        var5 = op3(var4, var2, var2)
        cb_tracker = CallbackTracker()
        fg = FunctionGraph(
            [var1, var2], [var3, var5], clone=False, features=[cb_tracker]
        )

        assert len(cb_tracker.callback_history) == 4
        assert cb_tracker.callback_history[0] == ("attach", (fg,), {})
        assert cb_tracker.callback_history[1] == (
            "import",
            (fg, var3.owner, "init"),
            {},
        )
        assert cb_tracker.callback_history[2] == (
            "import",
            (fg, var4.owner, "init"),
            {},
        )
        assert cb_tracker.callback_history[3] == (
            "import",
            (fg, var5.owner, "init"),
            {},
        )
        cb_tracker.callback_history.clear()

        old_var4_owner = var4.owner

        fg.replace_all([(var3, var4)])

        # The following works (and is kind of gross), because `var4` has been
        # mutated in-place
        assert var4.owner.inputs == [var4, var2]
        assert fg.apply_nodes == {var4.owner, var5.owner}
        assert fg.outputs == [var4, var5]

        assert cb_tracker.callback_history == [
            (
                "change_input",
                (fg, "output", "output", 0, var3, var4),
                {"reason": None},
            ),
            ("prune", (fg, var3.owner, None), {}),
            ("prune", (fg, old_var4_owner, None), {}),
            ("import", (fg, var4.owner, None), {}),
            (
                "change_input",
                (fg, old_var4_owner, var4.owner, 0, var3, var4),
                {"reason": None},
            ),
        ]

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
            fg.replace(var1, var0)

    def test_check_integrity(self):

        var1 = MyVariable("var1")
        var2 = MyVariable("var2")
        var3 = op1(var2, var1)
        var3.name = "var3"
        var4 = op2(var3, var2)
        var4.name = "var4"
        var5 = op3(var4, var2, var2)
        var5.name = "var5"
        fg = FunctionGraph([var1, var2], [var3, var5], clone=False)

        with pytest.raises(Exception, match="The following nodes are .*"):
            fg.apply_nodes.remove(var5.owner)

            fg.check_integrity()

        with pytest.raises(Exception, match="Inconsistent clients.*"):
            fg.apply_nodes.add(var5.owner)
            fg.remove_client(var2, (var5.owner, 1))

            fg.check_integrity()

        fg.add_client(var2, (var5.owner, 1))

        with pytest.raises(Exception, match="The following variables are.*"):
            fg.variables.remove(var4)

            fg.check_integrity()

        fg.variables.add(var4)

        with pytest.raises(Exception, match="Undeclared input.*"):
            var6 = MyVariable("var6")
            var7 = op1(var6)
            var7.name = "var7"
            fg.clients[var6] = [(var7.owner, 0)]
            fg.variables.add(var6)
            fg.clients[var7] = [("output", 2)]
            fg.variables.add(var7)
            fg.outputs.append(var7)
            fg.apply_nodes.add(var7.owner)

            fg.check_integrity()

        fg.variables.remove(var6)
        fg.variables.remove(var7)
        del fg.clients[var6]
        del fg.clients[var7]
        fg.outputs.remove(var7)
        fg.apply_nodes.remove(var7.owner)

        # TODO: What if the index value is greater than 1?  It will throw an
        # `IndexError`, but that doesn't sound like anything we'd want.
        with pytest.raises(Exception, match="Inconsistent clients list.*"):
            fg.add_client(var4, ("output", 1))

            fg.check_integrity()

        fg.remove_client(var4, ("output", 1))

        with pytest.raises(TypeError, match="The first entry of.*"):
            fg.add_client(var4, (None, 0))

        var7 = op1(var4)

        with pytest.raises(Exception, match="Client not in FunctionGraph.*"):
            fg.add_client(var4, (var7.owner, 0))

            fg.check_integrity()

        fg.remove_client(var4, (var7.owner, 0))

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

    def test_remove_node(self):
        var1 = MyVariable("var1")
        var2 = MyVariable("var2")
        node1_out = op1(var1)
        node2_out = op2(var2, node1_out)
        node3_out = op3(node2_out)
        fg = FunctionGraph([var1, var2], [node3_out], clone=False)

        fg.remove_node(node3_out.owner)
        fg.check_integrity()

        assert not fg.apply_nodes

        fg = FunctionGraph([var1, var2], [node2_out, node3_out], clone=False)

        fg.remove_node(node3_out.owner)
        fg.check_integrity()

        assert fg.apply_nodes == {node1_out.owner, node2_out.owner}

        fg = FunctionGraph([var1, var2], [node2_out, node3_out], clone=False)

        fg.remove_node(node2_out.owner)
        fg.check_integrity()

        assert not fg.apply_nodes

    def test_remove_output(self):
        var1 = MyVariable("var1")
        var2 = MyVariable("var2")
        node1_out = op1(var1)
        node2_out = op2(var2, node1_out)
        node3_out = op3(node2_out)

        fg = FunctionGraph([var1, var2], [node2_out, node3_out], clone=False)

        fg.remove_output(0)
        fg.check_integrity()

        assert fg.apply_nodes == {node1_out.owner, node2_out.owner, node3_out.owner}
        assert fg.inputs == [var1, var2]
        assert fg.outputs == [node3_out]

        fg = FunctionGraph([var1, var2], [node2_out, node3_out], clone=False)

        fg.remove_output(1)
        fg.check_integrity()

        assert fg.apply_nodes == {node1_out.owner, node2_out.owner}
        assert fg.inputs == [var1, var2]
        assert fg.outputs == [node2_out]

        fg = FunctionGraph([var1, var2], [node2_out, node3_out, var1], clone=False)

        fg.remove_output(2)
        fg.check_integrity()

        assert fg.apply_nodes == {node1_out.owner, node2_out.owner, node3_out.owner}
        assert fg.inputs == [var1, var2]
        assert fg.outputs == [node2_out, node3_out]

        fg = FunctionGraph([var1, var2], [var1], clone=False)

        fg.remove_output(0)
        fg.check_integrity()

        assert fg.inputs == [var1, var2]
        assert fg.outputs == []

    def test_remove_output_2(self):

        var0 = MyVariable("var0")
        var1 = MyVariable("var1")
        var2 = MyVariable("var2")
        var3 = MyVariable("var3")
        var4 = MyVariable("var4")
        op1_out = op1(var1, var0)

        out0 = op2(op1_out, var2)
        out1 = op1(var3, var4)
        out1.name = "out1"
        out2 = op1(out1, var0)
        out2.name = "out2"
        out3 = out1

        fg = FunctionGraph(
            [var0, var1, var2, var3, var4],
            [out0, out1, out2, out3],
            clone=False,
        )

        fg.remove_output(1)
        fg.check_integrity()

        assert fg.outputs == [out0, out2, out3]

        fg = FunctionGraph(
            [var0, var1, var2, var3, var4],
            [out0, out1, out2, var4, var4],
            clone=False,
        )

        fg.remove_output(3)
        fg.check_integrity()

        assert fg.inputs == [var0, var1, var2, var3, var4]
        assert fg.outputs == [out0, out1, out2, var4]

    def test_remove_output_3(self):

        var0 = MyVariable("var0")
        var1 = MyVariable("var1")
        var2 = MyVariable("var2")
        var3 = MyVariable("var3")
        var4 = MyVariable("var4")
        var5 = MyVariable("var5")
        var6 = MyVariable("var6")
        op1_out = op1(var1, var0)
        out0 = op2(op1_out, var2)
        out1 = op1(var3, var4)
        out1.name = "out1"
        out2 = op1(op1_out, var5)
        out2.name = "out2"
        out3 = op1(var3, var6)
        out3.name = "out3"
        out4 = op1_out
        out5 = var3
        fg = FunctionGraph(
            [var0, var1, var2, var3, var4, var5, var6],
            [out0, out1, out2, out3, out4, out5],
            clone=False,
        )

        fg.remove_output(1)
        fg.check_integrity()

        assert fg.inputs == [var0, var1, var2, var3, var4, var5, var6]
        assert fg.outputs == [out0, out2, out3, out4, out5]
        assert out1 not in fg.clients

    def test_remove_input(self):

        var0 = MyVariable("var0")
        var1 = MyVariable("var1")
        var2 = MyVariable("var2")
        var3 = MyVariable("var3")
        var4 = MyVariable("var4")

        op1_out = op1(var1, var0)
        out0 = op2(op1_out, var2)
        out1 = op1(var3, var4)
        out1.name = "out1"
        out2 = op1(out1, var0)
        out2.name = "out2"
        out3 = out1

        fg = FunctionGraph(
            [var0, var1, var2, var3, var4],
            [out0, out1, out2, out3],
            clone=False,
        )

        fg.remove_input(4)
        fg.check_integrity()

        assert fg.inputs == [var0, var1, var2, var3]
        assert fg.outputs == [out0]

    def test_remove_in_and_out(self):
        var1 = MyVariable("var1")
        var2 = MyVariable("var2")
        op1_out = op1(var2, var1)
        op2_out = op2(op1_out, var2)
        op3_out = op3(op2_out, var2, var2)
        fg = FunctionGraph([var1, var2], [op1_out, op3_out], clone=False)

        # Remove an output
        fg.remove_output(1)
        fg.check_integrity()

        assert fg.outputs == [op1_out]
        assert op3_out not in fg.clients
        assert not any(
            op3_out.owner in clients for clients in sum(fg.clients.values(), [])
        )

        # Remove an input
        fg.remove_input(0)
        fg.check_integrity()

        assert var1 not in fg.variables
        assert fg.inputs == [var2]
        assert fg.outputs == []
        assert not any(
            op1_out.owner in clients for clients in sum(fg.clients.values(), [])
        )

    def test_remove_duplicates(self):
        var1 = MyVariable("var1")
        var2 = MyVariable("var2")
        op1_out = op1(var2, var1)
        op2_out = op2(op1_out, var2)
        op3_out = op3(op2_out, var2, var2)
        fg = FunctionGraph([var1, var1, var2], [op1_out, op3_out, op3_out], clone=False)

        fg.remove_output(2)
        fg.check_integrity()

        assert fg.outputs == [op1_out, op3_out]

        fg.remove_input(0)
        fg.check_integrity()

        assert var1 not in fg.variables
        assert fg.inputs == [var1, var2]
        assert fg.outputs == []

    def test_remove_output_empty(self):

        var1 = MyVariable("var1")
        var2 = MyVariable("var2")
        op1_out = op1(var1)
        op3_out = op3(op1_out, var2)
        fg = FunctionGraph([var1, var2], [op3_out], clone=False)

        fg.remove_output(0)
        fg.check_integrity()

        assert fg.inputs == [var1, var2]
        assert not fg.apply_nodes
        assert op1_out not in fg.clients
        assert not any(
            op1_out.owner in clients for clients in sum(fg.clients.values(), [])
        )
        assert not any(
            op3_out.owner in clients for clients in sum(fg.clients.values(), [])
        )

    def test_remove_node_multi_out(self):
        var1 = MyVariable("var1")
        var2 = MyVariable("var2")
        multi_op = MyOp("mop", n_outs=2)
        op1_out = op1(var1)
        mop_out_1, mop_out_2 = multi_op(op1_out, var2)
        op3_out = op3(mop_out_2)

        fg = FunctionGraph([var1, var2], [mop_out_1, op3_out], clone=False)

        fg.remove_node(mop_out_1.owner)
        fg.check_integrity()

        assert fg.inputs == [var1, var2]
        assert fg.outputs == []
        assert mop_out_1 not in fg.clients
        assert mop_out_2 not in fg.clients
        assert mop_out_1 not in fg.variables
        assert mop_out_2 not in fg.variables

        mop1_out_1, mop1_out_2 = multi_op(var1)
        op2_out = op2(mop1_out_1)
        op3_out = op3(mop1_out_1, mop1_out_2)

        fg = FunctionGraph([var1], [op2_out, op3_out], clone=False)

        fg.remove_node(op3_out.owner)
        fg.check_integrity()

        assert fg.inputs == [var1]
        assert fg.outputs == [op2_out]
        # If we only want to track "active" variables in the graphs, the
        # following would need to be true, as well
        # assert mop1_out_2 not in fg.clients
        # assert mop1_out_2 not in fg.variables

        fg = FunctionGraph([var1], [op2_out, op3_out, mop1_out_2], clone=False)

        fg.remove_node(op3_out.owner)
        fg.check_integrity()

        assert fg.inputs == [var1]
        assert fg.outputs == [op2_out, mop1_out_2]
        assert mop1_out_2 in fg.clients
        assert mop1_out_2 in fg.variables
        assert mop1_out_2 in fg.outputs

    def test_empty(self):
        var1 = MyVariable("var1")
        var2 = MyVariable("var2")
        fg = FunctionGraph([var1, var2], [], clone=False)
        fg.check_integrity()

        assert fg.inputs == [var1, var2]
        assert fg.outputs == []
        assert not fg.variables
        assert not fg.apply_nodes
        assert fg.clients == {var1: [], var2: []}

    def test_nominals(self):
        t1 = MyType()

        nm = NominalVariable(1, t1)
        nm2 = NominalVariable(2, t1)

        v1 = op1(nm, nm2)

        fg = FunctionGraph(outputs=[v1], clone=False)

        assert nm not in fg.inputs
        assert nm2 not in fg.inputs
        assert nm in fg.variables
        assert nm2 in fg.variables
