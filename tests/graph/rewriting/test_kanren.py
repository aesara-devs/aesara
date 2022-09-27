from copy import copy

import numpy as np
import pytest
from etuples import etuple
from kanren import eq, fact, run
from kanren.assoccomm import associative, commutative, eq_assoccomm
from kanren.core import lall
from unification import var, vars

import aesara.tensor as at
from aesara.graph.basic import Apply
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import Op
from aesara.graph.rewriting.basic import EquilibriumGraphRewriter
from aesara.graph.rewriting.kanren import KanrenRelationSub
from aesara.graph.rewriting.unify import eval_if_etuple
from aesara.graph.rewriting.utils import rewrite_graph
from aesara.tensor.math import Dot, _dot
from tests.graph.utils import MyType, MyVariable


@pytest.fixture(autouse=True)
def clear_assoccomm():
    old_commutative_index = copy(commutative.index)
    old_commutative_facts = copy(commutative.facts)
    old_associative_index = copy(associative.index)
    old_associative_facts = copy(associative.facts)
    try:
        yield
    finally:
        commutative.index = old_commutative_index
        commutative.facts = old_commutative_facts
        associative.index = old_associative_index
        associative.facts = old_associative_facts


def test_kanren_basic():
    A_at = at.matrix("A")
    x_at = at.vector("x")

    y_at = at.dot(A_at, x_at)

    q = var()
    res = list(run(None, q, eq(y_at, etuple(_dot, q, x_at))))

    assert res == [A_at]


def test_KanrenRelationSub_filters():
    x_at = at.vector("x")
    y_at = at.vector("y")
    z_at = at.vector("z")
    A_at = at.matrix("A")

    fact(commutative, _dot)
    fact(commutative, at.add)
    fact(associative, at.add)

    Z_at = A_at.dot((x_at + y_at) + z_at)

    fgraph = FunctionGraph(outputs=[Z_at], clone=False)

    def distributes(in_lv, out_lv):
        A_lv, x_lv, y_lv, z_lv = vars(4)
        return lall(
            # lhs == A * (x + y + z)
            eq_assoccomm(
                etuple(_dot, A_lv, etuple(at.add, x_lv, etuple(at.add, y_lv, z_lv))),
                in_lv,
            ),
            # This relation does nothing but provide us with a means of
            # generating associative-commutative matches in the `kanren`
            # output.
            eq((A_lv, x_lv, y_lv, z_lv), out_lv),
        )

    def results_filter(results):
        _results = [eval_if_etuple(v) for v in results]

        # Make sure that at least a couple permutations are present
        assert (A_at, x_at, y_at, z_at) in _results
        assert (A_at, y_at, x_at, z_at) in _results
        assert (A_at, z_at, x_at, y_at) in _results

        return None

    _ = KanrenRelationSub(distributes, results_filter=results_filter).transform(
        fgraph, fgraph.outputs[0].owner
    )

    res = KanrenRelationSub(distributes, node_filter=lambda x: False).transform(
        fgraph, fgraph.outputs[0].owner
    )
    assert res is False


def test_KanrenRelationSub_multiout():
    class MyMultiOutOp(Op):
        def make_node(self, *inputs):
            outputs = [MyType()(), MyType()()]
            return Apply(self, list(inputs), outputs)

        def perform(self, node, inputs, outputs):
            outputs[0] = np.array(inputs[0])
            outputs[1] = np.array(inputs[0])

    x = MyVariable("x")
    y = MyVariable("y")
    multi_op = MyMultiOutOp()
    o1, o2 = multi_op(x, y)
    fgraph = FunctionGraph([x, y], [o1], clone=False)

    def relation(in_lv, out_lv):
        return eq(in_lv, out_lv)

    res = KanrenRelationSub(relation).transform(fgraph, fgraph.outputs[0].owner)

    assert res == [o1, o2]


def test_KanrenRelationSub_dot():
    """Make sure we can run miniKanren "optimizations" over a graph until a fixed-point/normal-form is reached."""
    x_at = at.vector("x")
    c_at = at.vector("c")
    d_at = at.vector("d")
    A_at = at.matrix("A")
    B_at = at.matrix("B")

    Z_at = A_at.dot(x_at + B_at.dot(c_at + d_at))

    fgraph = FunctionGraph(outputs=[Z_at], clone=False)

    assert isinstance(fgraph.outputs[0].owner.op, Dot)

    def distributes(in_lv, out_lv):
        return lall(
            # lhs == A * (x + b)
            eq(
                etuple(_dot, var("A"), etuple(at.add, var("x"), var("b"))),
                in_lv,
            ),
            # rhs == A * x + A * b
            eq(
                etuple(
                    at.add,
                    etuple(_dot, var("A"), var("x")),
                    etuple(_dot, var("A"), var("b")),
                ),
                out_lv,
            ),
        )

    distribute_opt = EquilibriumGraphRewriter(
        [KanrenRelationSub(distributes)], max_use_ratio=10
    )

    fgraph_opt = rewrite_graph(fgraph, custom_rewrite=distribute_opt)
    (expr_opt,) = fgraph_opt.outputs

    assert expr_opt.owner.op == at.add
    assert isinstance(expr_opt.owner.inputs[0].owner.op, Dot)
    assert fgraph_opt.inputs[0] is A_at
    assert expr_opt.owner.inputs[0].owner.inputs[0].name == "A"
    assert expr_opt.owner.inputs[1].owner.op == at.add
    assert isinstance(expr_opt.owner.inputs[1].owner.inputs[0].owner.op, Dot)
    assert isinstance(expr_opt.owner.inputs[1].owner.inputs[1].owner.op, Dot)


def test_deprecations():
    """Make sure we can import deprecated classes from current and deprecated modules."""
    with pytest.deprecated_call():
        from aesara.graph.kanren import KanrenRelationSub  # noqa: F401 F811
