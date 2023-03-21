""" This file don't test everything. It only test one past crash error."""

import aesara
from aesara import as_symbolic
from aesara.graph.basic import Constant
from aesara.tensor.math import argmax
from aesara.tensor.type import iscalar, vector
from aesara.tensor.type_other import (
    MakeSlice,
    NoneConst,
    NoneTypeT,
    SliceConstant,
    SliceType,
    make_slice,
)


def test_SliceType():
    st = SliceType()
    assert st == st.clone()


def test_make_slice_merge():
    # In the past, this was crahsing during compilation.
    i = iscalar()
    s1 = make_slice(0, i)
    s2 = make_slice(0, i)
    f = aesara.function([i], [s1, s2])
    nodes = f.maker.fgraph.apply_nodes
    assert len([n for n in nodes if isinstance(n.op, MakeSlice)]) == 1


def test_none_Constant():
    # FIXME: This is a poor test.

    # Tests equals
    # We had an error in the past with unpickling

    o1 = Constant(NoneTypeT(), None, name="NoneConst")
    o2 = Constant(NoneTypeT(), None, name="NoneConst")
    assert o1.equals(o2)
    assert NoneConst.equals(o1)
    assert o1.equals(NoneConst)
    assert NoneConst.equals(o2)
    assert o2.equals(NoneConst)

    # This trigger equals that returned the wrong answer in the past.
    import pickle

    import aesara

    x = vector("x")
    y = argmax(x)
    kwargs = {}
    # We can't pickle DebugMode
    if aesara.config.mode in ["DebugMode", "DEBUG_MODE"]:
        kwargs = {"mode": "FAST_RUN"}
    f = aesara.function([x], [y], **kwargs)
    pickle.loads(pickle.dumps(f))


def test_as_symbolic():
    res = as_symbolic(None)
    assert res is NoneConst

    res = as_symbolic(slice(iscalar()))
    assert res.owner.op == make_slice

    res = as_symbolic(slice(1, 2))
    assert isinstance(res, SliceConstant)
