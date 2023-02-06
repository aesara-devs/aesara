import sys

import pytest

from aesara.configdefaults import config
from aesara.graph.basic import Apply, Constant, equal_computations
from aesara.graph.features import Feature
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import Op
from aesara.graph.rewriting.basic import (
    EquilibriumGraphRewriter,
    MergeOptimizer,
    OpKeyGraphRewriter,
    OpToRewriterTracker,
    PatternNodeRewriter,
    SequentialNodeRewriter,
    SubstitutionNodeRewriter,
    WalkingGraphRewriter,
    in2out,
    node_rewriter,
)
from aesara.raise_op import assert_op
from aesara.tensor.math import Dot, add, dot
from aesara.tensor.type import matrix, values_eq_approx_always_true
from tests.graph.utils import (
    MyOp,
    MyType,
    MyVariable,
    op1,
    op2,
    op3,
    op4,
    op5,
    op6,
    op_cast_type2,
    op_multiple_outputs,
    op_y,
    op_z,
)


class AssertNoChanges(Feature):
    """A `Feature` that raises an error when nodes are changed in a graph."""

    def on_change_input(
        self, fgraph, old_node, new_node, i, old_var, new_var, reason=None
    ):
        raise AssertionError()


def OpKeyPatternNodeRewriter(p1, p2, ign=False):
    return OpKeyGraphRewriter(PatternNodeRewriter(p1, p2), ignore_newtrees=ign)


def WalkingPatternNodeRewriter(p1, p2, ign=True):
    return WalkingGraphRewriter(PatternNodeRewriter(p1, p2), ignore_newtrees=ign)


class TestPatternNodeRewriter:
    def test_replace_output(self):
        # replacing the whole graph
        x, y, z = MyVariable("x"), MyVariable("y"), MyVariable("z")
        e = op1(op2(x, y), z)
        g = FunctionGraph([x, y, z], [e], clone=False)
        OpKeyPatternNodeRewriter((op1, (op2, "1", "2"), "3"), (op4, "3", "2")).rewrite(
            g
        )
        assert equal_computations(g.outputs, [op4(z, y)])

    def test_nested_out_pattern(self):
        x, y, z = MyVariable("x"), MyVariable("y"), MyVariable("z")
        e = op1(x, y)
        g = FunctionGraph([x, y, z], [e], clone=False)
        OpKeyPatternNodeRewriter(
            (op1, "1", "2"), (op4, (op1, "1"), (op2, "2"), (op3, "1", "2"))
        ).rewrite(g)
        assert equal_computations(g.outputs, [op4(op1(x), op2(y), op3(x, y))])

    def test_unification_1(self):
        x, y, z = MyVariable("x"), MyVariable("y"), MyVariable("z")
        e = op1(op2(x, x), z)  # the arguments to op2 are the same
        g = FunctionGraph([x, y, z], [e], clone=False)
        OpKeyPatternNodeRewriter(
            (op1, (op2, "1", "1"), "2"),  # they are the same in the pattern
            (op4, "2", "1"),
        ).rewrite(g)
        # So the replacement should occur
        assert equal_computations(g.outputs, [op4(z, x)])

    def test_unification_2(self):
        x, y, z = MyVariable("x"), MyVariable("y"), MyVariable("z")
        e = op1(op2(x, y), z)  # the arguments to op2 are different
        g = FunctionGraph([x, y, z], [e], clone=False)
        OpKeyPatternNodeRewriter(
            (op1, (op2, "1", "1"), "2"),  # they are the same in the pattern
            (op4, "2", "1"),
        ).rewrite(g)
        # The replacement should NOT occur
        assert equal_computations(g.outputs, [op1(op2(x, y), z)])

    def test_replace_subgraph(self):
        # replacing inside the graph
        x, y, z = MyVariable("x"), MyVariable("y"), MyVariable("z")
        e = op1(op2(x, y), z)
        g = FunctionGraph([x, y, z], [e], clone=False)
        OpKeyPatternNodeRewriter((op2, "1", "2"), (op1, "2", "1")).rewrite(g)
        assert equal_computations(g.outputs, [op1(op1(y, x), z)])

    def test_no_recurse(self):
        # if the out pattern is an acceptable in pattern
        # and that the ignore_newtrees flag is True,
        # it should do the replacement and stop
        x, y, z = MyVariable("x"), MyVariable("y"), MyVariable("z")
        e = op1(op2(x, y), z)
        g = FunctionGraph([x, y, z], [e], clone=False)
        OpKeyPatternNodeRewriter((op2, "1", "2"), (op2, "2", "1"), ign=True).rewrite(g)
        assert equal_computations(g.outputs, [op1(op2(y, x), z)])

    def test_multiple(self):
        # it should replace all occurrences of the pattern
        x, y, z = MyVariable("x"), MyVariable("y"), MyVariable("z")
        e = op1(op2(x, y), op2(y, x), op2(y, z))
        g = FunctionGraph([x, y, z], [e], clone=False)
        OpKeyPatternNodeRewriter((op2, "1", "2"), (op4, "1")).rewrite(g)
        assert equal_computations(g.outputs, [op1(op4(x), op4(y), op4(y))])

    def test_nested_even(self):
        # regardless of the order in which we rewrite, this
        # should work
        x, y, z = MyVariable("x"), MyVariable("y"), MyVariable("z")
        e = op1(op1(op1(op1(x))))
        g = FunctionGraph([x, y, z], [e], clone=False)
        OpKeyPatternNodeRewriter((op1, (op1, "1")), "1").rewrite(g)
        assert equal_computations(g.outputs, [x])

    def test_nested_odd(self):
        x, y, z = MyVariable("x"), MyVariable("y"), MyVariable("z")
        e = op1(op1(op1(op1(op1(x)))))
        g = FunctionGraph([x, y, z], [e], clone=False)
        OpKeyPatternNodeRewriter((op1, (op1, "1")), "1").rewrite(g)
        assert equal_computations(g.outputs, [op1(x)])

    def test_expand(self):
        x, y, z = MyVariable("x"), MyVariable("y"), MyVariable("z")
        e = op1(op1(op1(x)))
        g = FunctionGraph([x, y, z], [e], clone=False)
        OpKeyPatternNodeRewriter((op1, "1"), (op2, (op1, "1")), ign=True).rewrite(g)
        assert equal_computations(g.outputs, [op2(op1(op2(op1(op2(op1(x))))))])

    def test_ambiguous(self):
        # this test should always work with WalkingGraphRewriter and the
        # ignore_newtrees flag set to False. Behavior with ignore_newtrees
        # = True or with other NodeProcessingGraphRewriters may differ.
        x, y, z = MyVariable("x"), MyVariable("y"), MyVariable("z")
        e = op1(op1(op1(op1(op1(x)))))
        g = FunctionGraph([x, y, z], [e], clone=False)
        WalkingPatternNodeRewriter((op1, (op1, "1")), (op1, "1"), ign=False).rewrite(g)
        assert equal_computations(g.outputs, [op1(x)])

    def test_constant(self):
        x = Constant(MyType(), 2, name="x")
        y = MyVariable("y")
        z = Constant(MyType(), 2, name="z")
        e = op1(op1(x, y), y)
        g = FunctionGraph([y], [e], clone=False)
        OpKeyPatternNodeRewriter((op1, z, "1"), (op2, "1", z)).rewrite(g)
        assert equal_computations(g.outputs, [op1(op2(y, z), y)])

    def test_constraints(self):
        x, y, z = MyVariable("x"), MyVariable("y"), MyVariable("z")
        e = op4(op1(op2(x, y)), op1(op1(x, y)))
        g = FunctionGraph([x, y, z], [e], clone=False)

        def constraint(r):
            # Only replacing if the input is an instance of Op2
            return r.owner.op == op2

        OpKeyPatternNodeRewriter(
            (op1, {"pattern": "1", "constraint": constraint}), (op3, "1")
        ).rewrite(g)
        assert equal_computations(g.outputs, [op4(op3(op2(x, y)), op1(op1(x, y)))])

    def test_match_same(self):
        x, y, z = MyVariable("x"), MyVariable("y"), MyVariable("z")
        e = op1(x, x)
        g = FunctionGraph([x, y, z], [e], clone=False)
        OpKeyPatternNodeRewriter((op1, "x", "y"), (op3, "x", "y")).rewrite(g)
        assert equal_computations(g.outputs, [op3(x, x)])

    @pytest.mark.xfail(
        reason="This pattern & constraint case isn't used and doesn't make much sense."
    )
    def test_match_same_illegal(self):
        x, y, z = MyVariable("x"), MyVariable("y"), MyVariable("z")
        e = op2(op1(x, x), op1(x, y))
        g = FunctionGraph([x, y, z], [e], clone=False)

        def constraint(r):
            # Only replacing if the input is an instance of Op2
            return r.owner.inputs[0] is not r.owner.inputs[1]

        OpKeyPatternNodeRewriter(
            {"pattern": (op1, "x", "y"), "constraint": constraint}, (op3, "x", "y")
        ).rewrite(g)
        assert equal_computations(g.outputs, [op2(op1(x, x), op3(x, y))])

    def test_allow_multiple_clients(self):
        x, y, z = MyVariable("x"), MyVariable("y"), MyVariable("z")
        e0 = op1(x, y)
        # `e0` has multiple clients (i.e. the `op4` and `op3` nodes)
        e = op3(op4(e0), e0)
        g = FunctionGraph([x, y, z], [e], clone=False)
        OpKeyPatternNodeRewriter((op4, (op1, "x", "y")), (op3, "x", "y")).rewrite(g)
        assert equal_computations(g.outputs, [op3(op4(op1(x, y)), op1(x, y))])

    def test_eq(self):
        # replacing the whole graph
        x, y, z = MyVariable("x"), MyVariable("y"), MyVariable("z")
        e = op1(op_y(x, y), z)
        g = FunctionGraph([x, y, z], [e], clone=False)
        OpKeyPatternNodeRewriter((op1, (op_z, "1", "2"), "3"), (op4, "3", "2")).rewrite(
            g
        )
        assert equal_computations(g.outputs, [op4(z, y)])


def KeyedSubstitutionNodeRewriter(op1, op2):
    return OpKeyGraphRewriter(SubstitutionNodeRewriter(op1, op2))


class TestSubstitutionNodeRewriter:
    def test_straightforward(self):
        x, y, z = MyVariable("x"), MyVariable("y"), MyVariable("z")
        e = op1(op1(op1(op1(op1(x)))))
        g = FunctionGraph([x, y, z], [e], clone=False)
        KeyedSubstitutionNodeRewriter(op1, op2).rewrite(g)
        assert equal_computations(g.outputs, [op2(op2(op2(op2(op2(x)))))])

    def test_straightforward_2(self):
        x, y, z = MyVariable("x"), MyVariable("y"), MyVariable("z")
        e = op1(op2(x), op3(y), op4(z))
        g = FunctionGraph([x, y, z], [e], clone=False)
        KeyedSubstitutionNodeRewriter(op3, op4).rewrite(g)
        assert equal_computations(g.outputs, [op1(op2(x), op4(y), op4(z))])


class NoInputOp(Op):
    __props__ = ("param",)

    def __init__(self, param):
        self.param = param

    def make_node(self):
        return Apply(self, [], [MyType()()])

    def perform(self, node, inputs, output_storage):
        output_storage[0][0] = self.param


class TestMergeOptimizer:
    def test_straightforward(self):
        x, y, z = MyVariable("x"), MyVariable("y"), MyVariable("z")
        e = op1(op2(x, y), op2(x, y), op2(x, z))
        g = FunctionGraph([x, y, z], [e], clone=False)
        MergeOptimizer().rewrite(g)
        out_var = g.outputs[0]
        var_1, var_2, var_3 = out_var.owner.inputs
        assert var_1 is var_2
        assert var_1 is not var_3

    def test_constant_merging(self):
        x = MyVariable("x")
        y = Constant(MyType(), 2, name="y")
        z = Constant(MyType(), 2, name="z")
        e = op1(op2(x, y), op2(x, y), op2(x, z))
        g = FunctionGraph([x, y, z], [e], clone=False)
        MergeOptimizer().rewrite(g)
        out_var = g.outputs[0]
        var_1, var_2, var_3 = out_var.owner.inputs
        assert var_1 is var_2
        assert var_2 is var_3

    def test_deep_merge(self):
        x, y, z = MyVariable("x"), MyVariable("y"), MyVariable("z")
        e = op1(op3(op2(x, y), z), op4(op3(op2(x, y), z)))
        g = FunctionGraph([x, y, z], [e], clone=False)
        MergeOptimizer().rewrite(g)
        out_var = g.outputs[0]
        var_1, var_2 = out_var.owner.inputs
        assert var_2.owner.inputs[0] is var_1

    def test_no_merge(self):
        x, y, z = MyVariable("x"), MyVariable("y"), MyVariable("z")
        e = op1(op3(op2(x, y)), op3(op2(y, x)))
        g = FunctionGraph([x, y, z], [e])
        g.attach_feature(AssertNoChanges())
        MergeOptimizer().rewrite(g)

    def test_merge_outputs(self):
        x, y, z = MyVariable("x"), MyVariable("y"), MyVariable("z")
        e1 = op3(op2(x, y))
        e2 = op3(op2(x, y))
        g = FunctionGraph([x, y, z], [e1, e2], clone=False)
        MergeOptimizer().rewrite(g)
        assert g.outputs[0] is g.outputs[1]

    def test_identical_constant_args(self):
        x = MyVariable("x")
        y = Constant(MyType(), 2, name="y")
        z = Constant(MyType(), 2, name="z")
        e1 = op1(y, z)
        g = FunctionGraph([x, y, z], [e1], clone=False)
        MergeOptimizer().rewrite(g)

        assert g.outputs[0].owner.op == op1
        input_1 = g.outputs[0].owner.inputs[0]
        assert input_1 is g.outputs[0].owner.inputs[1]

    @pytest.mark.skip(reason="This was disabled for some unknown reason")
    def test_one_assert_merge(self):
        """Merge two nodes, one has assert, the other not."""
        x1 = matrix("x1")
        x2 = matrix("x2")
        e = dot(x1, x2) + dot(assert_op(x1, (x1 > x2).all()), x2)
        g = FunctionGraph([x1, x2], [e], clone=False)
        MergeOptimizer().rewrite(g)

        assert g.outputs[0].owner.op == add
        add_inputs = g.outputs[0].owner.inputs
        assert isinstance(add_inputs[0].owner.op, Dot)
        # Confirm that the `Assert`s are correct
        assert_var = add_inputs[0].owner.inputs[0]
        assert_ref = assert_op(x1, (x1 > x2).all())
        assert equal_computations([assert_var], [assert_ref])
        # Confirm the merge
        assert add_inputs[0] is add_inputs[1]

    def test_both_assert_merge_identical(self):
        """Merge two nodes, both have `Assert`s on the same node with the same conditions."""
        x1 = matrix("x1")
        x2 = matrix("x2")
        e = dot(assert_op(x1, (x1 > x2).all()), x2) + dot(
            assert_op(x1, (x1 > x2).all()), x2
        )
        g = FunctionGraph([x1, x2], [e], clone=False)
        MergeOptimizer().rewrite(g)

        assert g.outputs[0].owner.op == add
        add_inputs = g.outputs[0].owner.inputs
        assert isinstance(add_inputs[0].owner.op, Dot)
        # Confirm that the `Assert`s are correct
        assert_var = add_inputs[0].owner.inputs[0]
        assert_ref = assert_op(x1, (x1 > x2).all())
        assert equal_computations([assert_var], [assert_ref])
        # Confirm the merge
        assert add_inputs[0] is add_inputs[1]

    @pytest.mark.skip(reason="Advanced `Assert` merging is disabled")
    def test_both_assert_merge_1(self):
        # Merge two nodes, both have assert on the same node
        # with different conditions.
        x1 = matrix("x1")
        x2 = matrix("x2")
        x3 = matrix("x3")
        e = dot(assert_op(x1, (x1 > x3).all()), x2) + dot(
            assert_op(x1, (x1 > x2).all()), x2
        )
        g = FunctionGraph([x1, x2, x3], [e], clone=False)
        MergeOptimizer().rewrite(g)

        assert g.outputs[0].owner.op == add
        add_inputs = g.outputs[0].owner.inputs
        assert isinstance(add_inputs[0].owner.op, Dot)
        # Confirm that the `Assert`s are correct
        assert_var = add_inputs[0].owner.inputs[0]
        assert_ref = assert_op(x1, (x1 > x3).all(), (x1 > x2).all())
        assert equal_computations([assert_var], [assert_ref])
        # Confirm the merge
        assert add_inputs[0] is add_inputs[1]

    @pytest.mark.skip(reason="Advanced `Assert` merging is disabled")
    def test_both_assert_merge_2(self):
        # Merge two nodes, both have assert on different node
        x1 = matrix("x1")
        x2 = matrix("x2")
        x3 = matrix("x3")
        e = dot(assert_op(x1, (x1 > x3).all()), x2) + dot(
            x1, assert_op(x2, (x2 > x3).all())
        )
        g = FunctionGraph([x1, x2, x3], [e], clone=False)
        MergeOptimizer().rewrite(g)

        assert g.outputs[0].owner.op == add
        add_inputs = g.outputs[0].owner.inputs
        assert isinstance(add_inputs[0].owner.op, Dot)
        # Confirm that the `Assert`s are correct
        assert_var_1, assert_var_2 = add_inputs[0].owner.inputs
        assert_ref_1 = assert_op(x1, (x1 > x3).all())
        assert equal_computations([assert_var_1], [assert_ref_1])
        assert_ref_2 = assert_op(x2, (x2 > x3).all())
        assert equal_computations([assert_var_2], [assert_ref_2])
        # Confirm the merge
        assert add_inputs[0] is add_inputs[1]

    @pytest.mark.skip(reason="Advanced `Assert` merging is disabled")
    def test_both_assert_merge_2_reverse(self):
        # Test case "test_both_assert_merge_2" but in reverse order
        x1 = matrix("x1")
        x2 = matrix("x2")
        x3 = matrix("x3")
        e = dot(x1, assert_op(x2, (x2 > x3).all())) + dot(
            assert_op(x1, (x1 > x3).all()), x2
        )
        g = FunctionGraph([x1, x2, x3], [e], clone=False)
        MergeOptimizer().rewrite(g)

        assert g.outputs[0].owner.op == add
        add_inputs = g.outputs[0].owner.inputs
        assert isinstance(add_inputs[0].owner.op, Dot)
        # Confirm that the `Assert`s are correct
        assert_var_1, assert_var_2 = add_inputs[0].owner.inputs
        assert_ref_1 = assert_op(x2, (x2 > x3).all())
        assert equal_computations([assert_var_1], [assert_ref_1])
        assert_ref_2 = assert_op(x1, (x1 > x3).all())
        assert equal_computations([assert_var_2], [assert_ref_2])
        # Confirm the merge
        assert add_inputs[0] is add_inputs[1]

    def test_merge_noinput(self):
        """Check that identical Apply nodes without inputs will be merged."""
        x = NoInputOp(param=0)()
        y = NoInputOp(param=0)()
        z = NoInputOp(param=1)()

        fg = FunctionGraph([], [x, y, z], clone=False)
        MergeOptimizer().rewrite(fg)

        assert fg.outputs[0] is fg.outputs[1]
        assert fg.outputs[0] is not fg.outputs[2]


class TestEquilibrium:
    def test_1(self):
        x, y, z = map(MyVariable, "xyz")
        # TODO FIXME: These `Op`s don't have matching/consistent `__prop__`s
        # and `__init__`s, so they can't be `etuplized` correctly
        e = op3(op4(x, y))
        g = FunctionGraph([x, y, z], [e], clone=False)
        rewriter = EquilibriumGraphRewriter(
            [
                PatternNodeRewriter((op1, "x", "y"), (op2, "x", "y")),
                PatternNodeRewriter((op4, "x", "y"), (op1, "x", "y")),
                PatternNodeRewriter((op3, (op2, "x", "y")), (op4, "x", "y")),
            ],
            max_use_ratio=10,
        )
        rewriter.rewrite(g)
        assert equal_computations(g.outputs, [op2(x, y)])

    def test_2(self):
        x, y, z = map(MyVariable, "xyz")
        e = op1(op1(op3(x, y)))
        g = FunctionGraph([x, y, z], [e], clone=False)
        rewriter = EquilibriumGraphRewriter(
            [
                PatternNodeRewriter((op1, (op2, "x", "y")), (op4, "x", "y")),
                PatternNodeRewriter((op3, "x", "y"), (op4, "x", "y")),
                PatternNodeRewriter((op4, "x", "y"), (op5, "x", "y")),
                PatternNodeRewriter((op5, "x", "y"), (op6, "x", "y")),
                PatternNodeRewriter((op6, "x", "y"), (op2, "x", "y")),
            ],
            max_use_ratio=10,
        )
        rewriter.rewrite(g)
        assert equal_computations(g.outputs, [op2(x, y)])

    @config.change_flags(on_opt_error="raise")
    def test_low_use_ratio(self):
        x, y, z = map(MyVariable, "xyz")
        e = op3(op4(x, y))
        g = FunctionGraph([x, y, z], [e], clone=False)
        rewriter = EquilibriumGraphRewriter(
            [
                PatternNodeRewriter((op1, "x", "y"), (op2, "x", "y")),
                PatternNodeRewriter((op4, "x", "y"), (op1, "x", "y")),
                PatternNodeRewriter((op3, (op2, "x", "y")), (op4, "x", "y")),
            ],
            max_use_ratio=1.0 / len(g.apply_nodes),
        )
        with pytest.raises(AssertionError):
            rewriter.rewrite(g)
        assert equal_computations(g.outputs, [op1(x, y)])


@pytest.mark.parametrize("tracks", [True, False])
@pytest.mark.parametrize("out_pattern", [(op2, "x"), "x", 1.0])
def test_patternsub_values_eq_approx(out_pattern, tracks):
    # PatternNodeRewriter would fail when `values_eq_approx` and `get_nodes` were specified
    x = MyVariable("x")
    e = op1(x)
    fg = FunctionGraph([x], [e], clone=False)

    rewriter = EquilibriumGraphRewriter(
        [
            PatternNodeRewriter(
                (op1, "x"),
                out_pattern,
                tracks=[op1] if tracks else (),
                get_nodes=(lambda fgraph, node: [node]) if tracks else None,
                values_eq_approx=values_eq_approx_always_true,
            )
        ],
        max_use_ratio=1,
    )
    rewriter.rewrite(fg)
    output = fg.outputs[0]
    if isinstance(out_pattern, tuple):
        assert output.owner.op == op2
        assert output.tag.values_eq_approx is values_eq_approx_always_true
    elif out_pattern == "x":
        assert output is x
        assert output.tag.values_eq_approx is values_eq_approx_always_true
    else:
        # The replacement types do not match, so the substitution should've
        # failed
        assert output is e


@pytest.mark.parametrize("out_pattern", [(op1, "x"), "x"])
def test_patternsub_invalid_dtype(out_pattern):
    # PatternNodeRewriter would wrongly return output of different dtype as the original node
    x = MyVariable("x")
    e = op_cast_type2(x)
    fg = FunctionGraph([x], [e])

    rewriter = EquilibriumGraphRewriter(
        [
            PatternNodeRewriter(
                (op_cast_type2, "x"),
                out_pattern,
            )
        ],
        max_use_ratio=1,
    )
    rewriter.rewrite(fg)
    assert e.type.is_super(fg.outputs[0].type)


def test_patternsub_different_output_lengths():
    # Test that PatternNodeRewriter won't replace nodes with different numbers of outputs
    ps = PatternNodeRewriter(
        (op1, "x"),
        ("x"),
        name="ps",
    )
    rewriter = in2out(ps)

    x = MyVariable("x")
    e1, e2 = op_multiple_outputs(x)
    o = op1(e1)

    fgraph = FunctionGraph(inputs=[x], outputs=[o])
    rewriter.rewrite(fgraph)
    assert fgraph.outputs[0].owner.op == op1


class TestSequentialNodeRewriter:
    def test_optimizer_verbose(self, capsys):

        x = MyVariable("x")
        y = MyVariable("y")
        o1 = op1(x, y)

        fgraph = FunctionGraph([x, y], [o1], clone=False)

        @node_rewriter(None)
        def local_rewrite_1(fgraph, node):
            if node.inputs[0] == x:
                res = op2(y, *node.inputs[1:])
                return [res]

        @node_rewriter(None)
        def local_rewrite_2(fgraph, node):
            if node.inputs[0] == y:
                res = op2(x, *node.inputs[1:])
                return [res]

        seq_rewriter = SequentialNodeRewriter(local_rewrite_1, local_rewrite_2)

        with config.change_flags(optimizer_verbose=True):
            (new_res,) = seq_rewriter.transform(fgraph, o1.owner)
            _ = seq_rewriter.transform(fgraph, new_res.owner)

        capres = capsys.readouterr()
        assert capres.err == ""
        out1, out2 = capres.out.split("\n", maxsplit=1)
        assert out1.startswith("rewriting: rewrite local_rewrite_1 replaces")
        assert out2.startswith("rewriting: rewrite local_rewrite_2 replaces")


def test_node_rewriter_str():
    @node_rewriter([op1, MyOp])
    def local_rewriter_1(fgraph, node):
        pass

    assert str(local_rewriter_1) == "local_rewriter_1"
    res = repr(local_rewriter_1)
    assert res.startswith("FromFunctionNodeRewriter(")
    assert "Op1" in res
    assert "local_rewriter_1" in res


def test_node_rewriter():

    with pytest.raises(ValueError):

        @node_rewriter([])
        def local_bad_1(fgraph, node):
            return node.outputs

    with pytest.raises(TypeError):

        @node_rewriter([None])
        def local_bad_2(fgraph, node):
            return node.outputs

    x = MyVariable("x")
    y = MyVariable("y")

    o1 = op1(x, y)

    class MyNewOp(MyOp):
        pass

    o2 = MyNewOp("MyNewOp")(x, y)

    class MyNewOp2(MyOp):
        pass

    o3 = MyNewOp2("MyNewOp2")(x, y)

    fgraph = FunctionGraph([x, y], [o1, o2, o3], clone=False)

    hits = [0]

    @node_rewriter([op1, MyNewOp])
    def local_rewriter_1(fgraph, node, hits=hits):
        hits[0] += 1
        return node.outputs

    # This is allowed by the `op1` in `tracks`
    local_rewriter_1.transform(fgraph, fgraph.outputs[0].owner)
    assert hits[0] == 1

    # This is allowed by the `MyOp` in `tracks`
    local_rewriter_1.transform(fgraph, fgraph.outputs[1].owner)
    assert hits[0] == 2

    # This is not allowed by `tracks`
    local_rewriter_1.transform(fgraph, fgraph.outputs[2].owner)
    assert hits[0] == 2


def test_OpToRewriterTracker():
    @node_rewriter(None)
    def local_rewriter_1(fgraph, node):
        pass

    @node_rewriter([op1])
    def local_rewriter_2(fgraph, node):
        pass

    @node_rewriter([Op])
    def local_rewriter_3(fgraph, node):
        pass

    @node_rewriter([MyOp])
    def local_rewriter_4(fgraph, node):
        pass

    @node_rewriter([MyOp])
    def local_rewriter_5(fgraph, node):
        pass

    tracker = OpToRewriterTracker()
    tracker.add_tracker(local_rewriter_1)
    tracker.add_tracker(local_rewriter_2)
    tracker.add_tracker(local_rewriter_3)
    tracker.add_tracker(local_rewriter_4)
    tracker.add_tracker(local_rewriter_5)

    assert tracker.tracked_instances == {op1: [local_rewriter_2]}
    assert tracker.tracked_types == {
        Op: [local_rewriter_3],
        MyOp: [local_rewriter_4, local_rewriter_5],
    }
    assert tracker.untracked_rewrites == [local_rewriter_1]

    res = tracker.get_trackers(op1)
    assert res == [
        local_rewriter_4,
        local_rewriter_5,
        local_rewriter_3,
        local_rewriter_2,
        local_rewriter_1,
    ]

    class MyNewOp(Op):
        def perform(self, *args):
            pass

    new_op = MyNewOp()

    res = tracker.get_trackers(new_op)
    assert res == [local_rewriter_3, local_rewriter_1]

    assert list(tracker.get_rewriters()) == [
        local_rewriter_3,
        local_rewriter_4,
        local_rewriter_5,
        local_rewriter_2,
        local_rewriter_1,
    ]


def test_deprecations():
    """Make sure we can import deprecated classes from current and deprecated modules."""
    with pytest.deprecated_call():
        from aesara.graph.rewriting.basic import GlobalOptimizer

    with pytest.deprecated_call():
        from aesara.graph.opt import GlobalOptimizer, LocalOptimizer  # noqa: F401 F811

    del sys.modules["aesara.graph.opt"]

    with pytest.deprecated_call():
        from aesara.graph.opt import GraphRewriter  # noqa: F401
