import pytest

from aesara.assert_op import assert_op
from aesara.configdefaults import config
from aesara.graph.basic import Apply, Constant
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import Op
from aesara.graph.opt import (
    EquilibriumOptimizer,
    MergeOptimizer,
    OpKeyOptimizer,
    OpSub,
    PatternSub,
    TopoOptimizer,
    aesara,
    logging,
    pre_constant_merge,
    pre_greedy_local_optimizer,
)
from aesara.tensor.basic_opt import constant_folding
from aesara.tensor.math import dot
from aesara.tensor.subtensor import AdvancedSubtensor
from aesara.tensor.type import matrix, values_eq_approx_always_true
from aesara.tensor.type_other import MakeSlice, SliceConstant, slicetype
from tests.graph.utils import (
    MyType,
    MyVariable,
    op1,
    op2,
    op3,
    op4,
    op5,
    op6,
    op_cast_type2,
    op_y,
    op_z,
)


def inputs():
    x = MyVariable("x")
    y = MyVariable("y")
    z = MyVariable("z")
    return x, y, z


def PatternOptimizer(p1, p2, ign=False):
    return OpKeyOptimizer(PatternSub(p1, p2), ignore_newtrees=ign)


def TopoPatternOptimizer(p1, p2, ign=True):
    return TopoOptimizer(PatternSub(p1, p2), ignore_newtrees=ign)


class TestPatternOptimizer:
    def test_replace_output(self):
        # replacing the whole graph
        x, y, z = inputs()
        e = op1(op2(x, y), z)
        g = FunctionGraph([x, y, z], [e])
        PatternOptimizer((op1, (op2, "1", "2"), "3"), (op4, "3", "2")).optimize(g)
        assert str(g) == "FunctionGraph(Op4(z, y))"

    def test_nested_out_pattern(self):
        x, y, z = inputs()
        e = op1(x, y)
        g = FunctionGraph([x, y, z], [e])
        PatternOptimizer(
            (op1, "1", "2"), (op4, (op1, "1"), (op2, "2"), (op3, "1", "2"))
        ).optimize(g)
        assert str(g) == "FunctionGraph(Op4(Op1(x), Op2(y), Op3(x, y)))"

    def test_unification_1(self):
        x, y, z = inputs()
        e = op1(op2(x, x), z)  # the arguments to op2 are the same
        g = FunctionGraph([x, y, z], [e])
        PatternOptimizer(
            (op1, (op2, "1", "1"), "2"),  # they are the same in the pattern
            (op4, "2", "1"),
        ).optimize(g)
        # So the replacement should occur
        assert str(g) == "FunctionGraph(Op4(z, x))"

    def test_unification_2(self):
        x, y, z = inputs()
        e = op1(op2(x, y), z)  # the arguments to op2 are different
        g = FunctionGraph([x, y, z], [e])
        PatternOptimizer(
            (op1, (op2, "1", "1"), "2"),  # they are the same in the pattern
            (op4, "2", "1"),
        ).optimize(g)
        # The replacement should NOT occur
        assert str(g) == "FunctionGraph(Op1(Op2(x, y), z))"

    def test_replace_subgraph(self):
        # replacing inside the graph
        x, y, z = inputs()
        e = op1(op2(x, y), z)
        g = FunctionGraph([x, y, z], [e])
        PatternOptimizer((op2, "1", "2"), (op1, "2", "1")).optimize(g)
        assert str(g) == "FunctionGraph(Op1(Op1(y, x), z))"

    def test_no_recurse(self):
        # if the out pattern is an acceptable in pattern
        # and that the ignore_newtrees flag is True,
        # it should do the replacement and stop
        x, y, z = inputs()
        e = op1(op2(x, y), z)
        g = FunctionGraph([x, y, z], [e])
        PatternOptimizer((op2, "1", "2"), (op2, "2", "1"), ign=True).optimize(g)
        assert str(g) == "FunctionGraph(Op1(Op2(y, x), z))"

    def test_multiple(self):
        # it should replace all occurrences of the pattern
        x, y, z = inputs()
        e = op1(op2(x, y), op2(x, y), op2(y, z))
        g = FunctionGraph([x, y, z], [e])
        PatternOptimizer((op2, "1", "2"), (op4, "1")).optimize(g)
        assert str(g) == "FunctionGraph(Op1(Op4(x), Op4(x), Op4(y)))"

    def test_nested_even(self):
        # regardless of the order in which we optimize, this
        # should work
        x, y, z = inputs()
        e = op1(op1(op1(op1(x))))
        g = FunctionGraph([x, y, z], [e])
        PatternOptimizer((op1, (op1, "1")), "1").optimize(g)
        assert str(g) == "FunctionGraph(x)"

    def test_nested_odd(self):
        x, y, z = inputs()
        e = op1(op1(op1(op1(op1(x)))))
        g = FunctionGraph([x, y, z], [e])
        PatternOptimizer((op1, (op1, "1")), "1").optimize(g)
        assert str(g) == "FunctionGraph(Op1(x))"

    def test_expand(self):
        x, y, z = inputs()
        e = op1(op1(op1(x)))
        g = FunctionGraph([x, y, z], [e])
        PatternOptimizer((op1, "1"), (op2, (op1, "1")), ign=True).optimize(g)
        assert str(g) == "FunctionGraph(Op2(Op1(Op2(Op1(Op2(Op1(x)))))))"

    def test_ambiguous(self):
        # this test should always work with TopoOptimizer and the
        # ignore_newtrees flag set to False. Behavior with ignore_newtrees
        # = True or with other NavigatorOptimizers may differ.
        x, y, z = inputs()
        e = op1(op1(op1(op1(op1(x)))))
        g = FunctionGraph([x, y, z], [e])
        TopoPatternOptimizer((op1, (op1, "1")), (op1, "1"), ign=False).optimize(g)
        assert str(g) == "FunctionGraph(Op1(x))"

    def test_constant_unification(self):
        x = Constant(MyType(), 2, name="x")
        y = MyVariable("y")
        z = Constant(MyType(), 2, name="z")
        e = op1(op1(x, y), y)
        g = FunctionGraph([y], [e])
        PatternOptimizer((op1, z, "1"), (op2, "1", z)).optimize(g)
        assert str(g) == "FunctionGraph(Op1(Op2(y, z), y))"

    def test_constraints(self):
        x, y, z = inputs()
        e = op4(op1(op2(x, y)), op1(op1(x, y)))
        g = FunctionGraph([x, y, z], [e])

        def constraint(r):
            # Only replacing if the input is an instance of Op2
            return r.owner.op == op2

        PatternOptimizer(
            (op1, {"pattern": "1", "constraint": constraint}), (op3, "1")
        ).optimize(g)
        assert str(g) == "FunctionGraph(Op4(Op3(Op2(x, y)), Op1(Op1(x, y))))"

    def test_match_same(self):
        x, y, z = inputs()
        e = op1(x, x)
        g = FunctionGraph([x, y, z], [e])
        PatternOptimizer((op1, "x", "y"), (op3, "x", "y")).optimize(g)
        assert str(g) == "FunctionGraph(Op3(x, x))"

    def test_match_same_illegal(self):
        x, y, z = inputs()
        e = op2(op1(x, x), op1(x, y))
        g = FunctionGraph([x, y, z], [e])

        def constraint(r):
            # Only replacing if the input is an instance of Op2
            return r.owner.inputs[0] is not r.owner.inputs[1]

        PatternOptimizer(
            {"pattern": (op1, "x", "y"), "constraint": constraint}, (op3, "x", "y")
        ).optimize(g)
        assert str(g) == "FunctionGraph(Op2(Op1(x, x), Op3(x, y)))"

    def test_multi(self):
        x, y, z = inputs()
        e0 = op1(x, y)
        e = op3(op4(e0), e0)
        g = FunctionGraph([x, y, z], [e])
        PatternOptimizer((op4, (op1, "x", "y")), (op3, "x", "y")).optimize(g)
        assert str(g) == "FunctionGraph(Op3(Op4(*1 -> Op1(x, y)), *1))"

    def test_eq(self):
        # replacing the whole graph
        x, y, z = inputs()
        e = op1(op_y(x, y), z)
        g = FunctionGraph([x, y, z], [e])
        PatternOptimizer((op1, (op_z, "1", "2"), "3"), (op4, "3", "2")).optimize(g)
        str_g = str(g)
        assert str_g == "FunctionGraph(Op4(z, y))"


#     def test_multi_ingraph(self):
#         # known to fail
#         x, y, z = inputs()
#         e0 = op1(x, y)
#         e = op4(e0, e0)
#         g = FunctionGraph([x, y, z], [e])
#         PatternOptimizer((op4, (op1, 'x', 'y'), (op1, 'x', 'y')),
#                          (op3, 'x', 'y')).optimize(g)
#         assert str(g) == "FunctionGraph(Op3(x, y))"


def OpSubOptimizer(op1, op2):
    return OpKeyOptimizer(OpSub(op1, op2))


class TestOpSubOptimizer:
    def test_straightforward(self):
        x, y, z = inputs()
        e = op1(op1(op1(op1(op1(x)))))
        g = FunctionGraph([x, y, z], [e])
        OpSubOptimizer(op1, op2).optimize(g)
        assert str(g) == "FunctionGraph(Op2(Op2(Op2(Op2(Op2(x))))))"

    def test_straightforward_2(self):
        x, y, z = inputs()
        e = op1(op2(x), op3(y), op4(z))
        g = FunctionGraph([x, y, z], [e])
        OpSubOptimizer(op3, op4).optimize(g)
        assert str(g) == "FunctionGraph(Op1(Op2(x), Op4(y), Op4(z)))"


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
        x, y, z = inputs()
        e = op1(op2(x, y), op2(x, y), op2(x, z))
        g = FunctionGraph([x, y, z], [e])
        MergeOptimizer().optimize(g)
        assert str(g) == "FunctionGraph(Op1(*1 -> Op2(x, y), *1, Op2(x, z)))"

    def test_constant_merging(self):
        x = MyVariable("x")
        y = Constant(MyType(), 2, name="y")
        z = Constant(MyType(), 2, name="z")
        e = op1(op2(x, y), op2(x, y), op2(x, z))
        g = FunctionGraph([x, y, z], [e])
        MergeOptimizer().optimize(g)
        strg = str(g)
        assert (
            strg == "FunctionGraph(Op1(*1 -> Op2(x, y), *1, *1))"
            or strg == "FunctionGraph(Op1(*1 -> Op2(x, z), *1, *1))"
        )

    def test_deep_merge(self):
        x, y, z = inputs()
        e = op1(op3(op2(x, y), z), op4(op3(op2(x, y), z)))
        g = FunctionGraph([x, y, z], [e])
        MergeOptimizer().optimize(g)
        assert str(g) == "FunctionGraph(Op1(*1 -> Op3(Op2(x, y), z), Op4(*1)))"

    def test_no_merge(self):
        x, y, z = inputs()
        e = op1(op3(op2(x, y)), op3(op2(y, x)))
        g = FunctionGraph([x, y, z], [e])
        MergeOptimizer().optimize(g)
        assert str(g) == "FunctionGraph(Op1(Op3(Op2(x, y)), Op3(Op2(y, x))))"

    def test_merge_outputs(self):
        x, y, z = inputs()
        e1 = op3(op2(x, y))
        e2 = op3(op2(x, y))
        g = FunctionGraph([x, y, z], [e1, e2])
        MergeOptimizer().optimize(g)
        assert str(g) == "FunctionGraph(*1 -> Op3(Op2(x, y)), *1)"

    def test_multiple_merges(self):
        x, y, z = inputs()
        e1 = op1(x, y)
        e2 = op2(op3(x), y, z)
        e = op1(e1, op4(e2, e1), op1(e2))
        g = FunctionGraph([x, y, z], [e])
        MergeOptimizer().optimize(g)
        strg = str(g)
        # note: graph.as_string can only produce the following two possibilities, but if
        # the implementation was to change there are 6 other acceptable answers.
        assert (
            strg
            == "FunctionGraph(Op1(*1 -> Op1(x, y), Op4(*2 -> Op2(Op3(x), y, z), *1), Op1(*2)))"
            or strg
            == "FunctionGraph(Op1(*2 -> Op1(x, y), Op4(*1 -> Op2(Op3(x), y, z), *2), Op1(*1)))"
        )

    def test_identical_constant_args(self):
        x = MyVariable("x")
        y = Constant(MyType(), 2, name="y")
        z = Constant(MyType(), 2, name="z")
        with config.change_flags(compute_test_value="off"):
            e1 = op1(y, z)
        g = FunctionGraph([x, y, z], [e1])
        MergeOptimizer().optimize(g)
        strg = str(g)
        assert strg == "FunctionGraph(Op1(y, y))" or strg == "FunctionGraph(Op1(z, z))"

    @pytest.mark.skip(reason="This was disabled for some unknown reason")
    def test_one_assert_merge(self):
        # Merge two nodes, one has assert, the other not.
        x1 = matrix("x1")
        x2 = matrix("x2")
        e = dot(x1, x2) + dot(assert_op(x1, (x1 > x2).all()), x2)
        g = FunctionGraph([x1, x2], [e])
        MergeOptimizer().optimize(g)
        strg = aesara.printing.debugprint(g, file="str")
        strref = """Elemwise{add,no_inplace} [id A] ''   4
 |dot [id B] ''   3
 | |Assert{msg='Aesara Assert failed!'} [id C] ''   2
 | | |x1 [id D]
 | | |All [id E] ''   1
 | |   |Elemwise{gt,no_inplace} [id F] ''   0
 | |     |x1 [id D]
 | |     |x2 [id G]
 | |x2 [id G]
 |dot [id B] ''   3
"""
        assert strg == strref, (strg, strref)

    def test_both_assert_merge_identical(self):
        # Merge two nodes, both have assert on the same node
        # with the same conditions.
        x1 = matrix("x1")
        x2 = matrix("x2")
        e = dot(assert_op(x1, (x1 > x2).all()), x2) + dot(
            assert_op(x1, (x1 > x2).all()), x2
        )
        g = FunctionGraph([x1, x2], [e])
        MergeOptimizer().optimize(g)
        strg = aesara.printing.debugprint(g, file="str")
        strref = """Elemwise{add,no_inplace} [id A] ''   4
 |dot [id B] ''   3
 | |Assert{msg='Aesara Assert failed!'} [id C] ''   2
 | | |x1 [id D]
 | | |All [id E] ''   1
 | |   |Elemwise{gt,no_inplace} [id F] ''   0
 | |     |x1 [id D]
 | |     |x2 [id G]
 | |x2 [id G]
 |dot [id B] ''   3
"""
        # print(strg)
        assert strg == strref, (strg, strref)

    @pytest.mark.skip(reason="This was disabled for some unknown reason")
    def test_both_assert_merge_1(self):
        # Merge two nodes, both have assert on the same node
        # with different conditions.
        x1 = matrix("x1")
        x2 = matrix("x2")
        x3 = matrix("x3")
        e = dot(assert_op(x1, (x1 > x3).all()), x2) + dot(
            assert_op(x1, (x1 > x2).all()), x2
        )
        g = FunctionGraph([x1, x2, x3], [e])
        MergeOptimizer().optimize(g)
        strg = aesara.printing.debugprint(g, file="str")
        strref1 = """Elemwise{add,no_inplace} [id A] ''   6
 |dot [id B] ''   5
 | |Assert{msg='Aesara Assert failed!'} [id C] ''   4
 | | |x1 [id D]
 | | |All [id E] ''   3
 | | | |Elemwise{gt,no_inplace} [id F] ''   1
 | | |   |x1 [id D]
 | | |   |x3 [id G]
 | | |All [id H] ''   2
 | |   |Elemwise{gt,no_inplace} [id I] ''   0
 | |     |x1 [id D]
 | |     |x2 [id J]
 | |x2 [id J]
 |dot [id B] ''   5
"""
        strref2 = """Elemwise{add,no_inplace} [id A] ''   6
 |dot [id B] ''   5
 | |Assert{msg='Aesara Assert failed!'} [id C] ''   4
 | | |x1 [id D]
 | | |All [id E] ''   3
 | | | |Elemwise{gt,no_inplace} [id F] ''   1
 | | |   |x1 [id D]
 | | |   |x2 [id G]
 | | |All [id H] ''   2
 | |   |Elemwise{gt,no_inplace} [id I] ''   0
 | |     |x1 [id D]
 | |     |x3 [id J]
 | |x2 [id G]
 |dot [id B] ''   5
"""
        # print(strg)
        assert strg == strref1 or strg == strref2, (strg, strref1, strref2)

    @pytest.mark.skip(reason="This was disabled for some unknown reason")
    def test_both_assert_merge_2(self):
        # Merge two nodes, both have assert on different node
        x1 = matrix("x1")
        x2 = matrix("x2")
        x3 = matrix("x3")
        e = dot(assert_op(x1, (x1 > x3).all()), x2) + dot(
            x1, assert_op(x2, (x2 > x3).all())
        )
        g = FunctionGraph([x1, x2, x3], [e])
        MergeOptimizer().optimize(g)
        strg = aesara.printing.debugprint(g, file="str")
        strref = """Elemwise{add,no_inplace} [id A] ''   7
 |dot [id B] ''   6
 | |Assert{msg='Aesara Assert failed!'} [id C] ''   5
 | | |x1 [id D]
 | | |All [id E] ''   3
 | |   |Elemwise{gt,no_inplace} [id F] ''   1
 | |     |x1 [id D]
 | |     |x3 [id G]
 | |Assert{msg='Aesara Assert failed!'} [id H] ''   4
 |   |x2 [id I]
 |   |All [id J] ''   2
 |     |Elemwise{gt,no_inplace} [id K] ''   0
 |       |x2 [id I]
 |       |x3 [id G]
 |dot [id B] ''   6
"""
        # print(strg)
        assert strg == strref, (strg, strref)

    @pytest.mark.skip(reason="This was disabled for some unknown reason")
    def test_both_assert_merge_2_reverse(self):
        # Test case "test_both_assert_merge_2" but in reverse order
        x1 = matrix("x1")
        x2 = matrix("x2")
        x3 = matrix("x3")
        e = dot(x1, assert_op(x2, (x2 > x3).all())) + dot(
            assert_op(x1, (x1 > x3).all()), x2
        )
        g = FunctionGraph([x1, x2, x3], [e])
        MergeOptimizer().optimize(g)
        strg = aesara.printing.debugprint(g, file="str")
        strref = """Elemwise{add,no_inplace} [id A] ''   7
 |dot [id B] ''   6
 | |Assert{msg='Aesara Assert failed!'} [id C] ''   5
 | | |x1 [id D]
 | | |All [id E] ''   3
 | |   |Elemwise{gt,no_inplace} [id F] ''   1
 | |     |x1 [id D]
 | |     |x3 [id G]
 | |Assert{msg='Aesara Assert failed!'} [id H] ''   4
 |   |x2 [id I]
 |   |All [id J] ''   2
 |     |Elemwise{gt,no_inplace} [id K] ''   0
 |       |x2 [id I]
 |       |x3 [id G]
 |dot [id B] ''   6
"""
        print(strg)
        assert strg == strref, (strg, strref)

    def test_merge_noinput(self):
        # Check that identical Apply nodes without inputs will be merged
        x = NoInputOp(param=0)()
        y = NoInputOp(param=0)()
        z = NoInputOp(param=1)()

        fg = FunctionGraph([], [x, y, z])
        MergeOptimizer().optimize(fg)
        no_input_ops = [n for n in fg.apply_nodes if isinstance(n.op, NoInputOp)]
        assert len(no_input_ops) == 2, fg.apply_nodes


class TestEquilibrium:
    def test_1(self):
        x, y, z = map(MyVariable, "xyz")
        e = op3(op4(x, y))
        g = FunctionGraph([x, y, z], [e])
        # print g
        opt = EquilibriumOptimizer(
            [
                PatternSub((op1, "x", "y"), (op2, "x", "y")),
                PatternSub((op4, "x", "y"), (op1, "x", "y")),
                PatternSub((op3, (op2, "x", "y")), (op4, "x", "y")),
            ],
            max_use_ratio=10,
        )
        opt.optimize(g)
        # print g
        assert str(g) == "FunctionGraph(Op2(x, y))"

    def test_2(self):
        x, y, z = map(MyVariable, "xyz")
        e = op1(op1(op3(x, y)))
        g = FunctionGraph([x, y, z], [e])
        # print g
        opt = EquilibriumOptimizer(
            [
                PatternSub((op1, (op2, "x", "y")), (op4, "x", "y")),
                PatternSub((op3, "x", "y"), (op4, "x", "y")),
                PatternSub((op4, "x", "y"), (op5, "x", "y")),
                PatternSub((op5, "x", "y"), (op6, "x", "y")),
                PatternSub((op6, "x", "y"), (op2, "x", "y")),
            ],
            max_use_ratio=10,
        )
        opt.optimize(g)
        assert str(g) == "FunctionGraph(Op2(x, y))"

    @config.change_flags(on_opt_error="ignore")
    def test_low_use_ratio(self):
        x, y, z = map(MyVariable, "xyz")
        e = op3(op4(x, y))
        g = FunctionGraph([x, y, z], [e])
        # print 'before', g
        # display pesky warnings along with stdout
        # also silence logger for 'aesara.graph.opt'
        _logger = logging.getLogger("aesara.graph.opt")
        oldlevel = _logger.level
        _logger.setLevel(logging.CRITICAL)
        try:
            opt = EquilibriumOptimizer(
                [
                    PatternSub((op1, "x", "y"), (op2, "x", "y")),
                    PatternSub((op4, "x", "y"), (op1, "x", "y")),
                    PatternSub((op3, (op2, "x", "y")), (op4, "x", "y")),
                ],
                max_use_ratio=1.0 / len(g.apply_nodes),
            )  # each opt can only be applied once
            opt.optimize(g)
        finally:
            _logger.setLevel(oldlevel)
        # print 'after', g
        assert str(g) == "FunctionGraph(Op1(x, y))"


def test_pre_constant_merge():

    empty_fgraph = FunctionGraph([], [])

    x = MyVariable("x")
    y = MyVariable("y")
    c1 = Constant(MyType(), 1, "c1")
    c2 = Constant(MyType(), 1, "c1")
    o1 = op2(c1, x)
    o2 = op1(o1, y, c2)

    assert c1 is not c2

    res = pre_constant_merge(empty_fgraph, [o2])

    assert [o2] == res
    assert o2.owner.inputs[2] is c1

    o2 = op1(o1, y, c2)
    fg = FunctionGraph([x, y], [o2], clone=False)

    assert o2.owner in fg.apply_nodes

    res = pre_constant_merge(fg, [o2])

    assert res == [o2]
    assert o2.owner.inputs[2] is c2

    # What is this supposed to test?
    ms = MakeSlice()(1)
    res = pre_constant_merge(empty_fgraph, [ms])

    assert res == [ms]

    const_slice = SliceConstant(type=slicetype, data=slice(1, None, 2))

    assert isinstance(const_slice, Constant)

    adv = AdvancedSubtensor()(matrix(), [2, 3], const_slice)

    res = pre_constant_merge(empty_fgraph, adv)
    assert res == [adv]


def test_pre_greedy_local_optimizer():

    empty_fgraph = FunctionGraph([], [])

    x = MyVariable("x")
    y = MyVariable("y")
    c1 = Constant(MyType(), 1, "c1")
    c2 = Constant(MyType(), 2, "c2")
    o1 = op2(c1, c2)
    o3 = op1(c1, y)
    o2 = op1(o1, c2, x, o3, o1)

    assert o2.owner.inputs[0].owner is not None
    assert o2.owner.inputs[4].owner is not None

    # This should fold `o1`, because it has only `Constant` arguments, and
    # replace it with the `Constant` result
    cst = pre_greedy_local_optimizer(empty_fgraph, [constant_folding], o2)

    assert cst.owner.inputs[0].owner is None
    assert cst.owner.inputs[1] is c2
    assert cst.owner.inputs[2] is x
    assert cst.owner.inputs[3] is o3
    assert cst.owner.inputs[4] is cst.owner.inputs[0]

    # We're going to do it again, except this time `o1` is
    # in the `fgraph`, so it shouldn't be folded
    fg = FunctionGraph([], [o1], clone=False)
    o2 = op1(o1, c2, x, o3, o1)

    cst = pre_greedy_local_optimizer(fg, [constant_folding], o2)

    assert cst.owner.inputs[0] is o1
    assert cst.owner.inputs[4] is cst.owner.inputs[0]

    # What exactly is this supposed to test?
    ms = MakeSlice()(1)
    cst = pre_greedy_local_optimizer(empty_fgraph, [constant_folding], ms)

    assert isinstance(cst, SliceConstant)

    # Make sure constant of slice signature is hashable.
    assert isinstance(hash(cst.signature()), int)


@pytest.mark.parametrize("tracks", [True, False])
@pytest.mark.parametrize("out_pattern", [(op2, "x"), "x", 1.0])
def test_patternsub_values_eq_approx(out_pattern, tracks):
    # PatternSub would fail when `values_eq_approx` and `get_nodes` were specified
    x = MyVariable("x")
    e = op1(x)
    fg = FunctionGraph([x], [e], clone=False)

    opt = EquilibriumOptimizer(
        [
            PatternSub(
                (op1, "x"),
                out_pattern,
                tracks=[op1] if tracks else (),
                get_nodes=(lambda fgraph, node: [node]) if tracks else None,
                values_eq_approx=values_eq_approx_always_true,
            )
        ],
        max_use_ratio=1,
    )
    opt.optimize(fg)
    output = fg.outputs[0]
    if isinstance(out_pattern, tuple):
        assert output.owner.op == op2
        assert output.tag.values_eq_approx is values_eq_approx_always_true
    elif out_pattern == "x":
        assert output is x
        assert output.tag.values_eq_approx is values_eq_approx_always_true
    else:
        assert isinstance(output, Constant)
        assert not hasattr(output.tag, "value_eq_approx")


@pytest.mark.parametrize("out_pattern", [(op1, "x"), "x"])
def test_patternsub_invalid_dtype(out_pattern):
    # PatternSub would wrongly return output of different dtype as the original node
    x = MyVariable("x")
    e = op_cast_type2(x)
    fg = FunctionGraph([x], [e])

    opt = EquilibriumOptimizer(
        [
            PatternSub(
                (op_cast_type2, "x"),
                out_pattern,
            )
        ],
        max_use_ratio=1,
    )
    opt.optimize(fg)
    assert fg.apply_nodes.pop().op == op_cast_type2
