""" This file don't test everything. It only test one past crash error."""

import aesara
from aesara.gof import Constant
from aesara.tensor.type_other import MakeSlice, NoneConst, NoneTypeT, make_slice


def test_make_slice_merge():
    # In the past, this was crahsing during compilation.
    i = aesara.tensor.iscalar()
    s1 = make_slice(0, i)
    s2 = make_slice(0, i)
    f = aesara.function([i], [s1, s2])
    nodes = f.maker.fgraph.apply_nodes
    assert len([n for n in nodes if isinstance(n.op, MakeSlice)]) == 1
    aesara.printing.debugprint(f)


def test_none_Constant():
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
    from aesara import tensor

    x = tensor.vector("x")
    y = tensor.argmax(x)
    kwargs = {}
    # We can't pickle DebugMode
    if aesara.config.mode in ["DebugMode", "DEBUG_MODE"]:
        kwargs = {"mode": "FAST_RUN"}
    f = aesara.function([x], [y], **kwargs)
    pickle.loads(pickle.dumps(f))
