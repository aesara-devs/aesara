from functools import singledispatch

import numpy as np

from aesara import config
from aesara.graph.basic import Apply
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import Op
from aesara.link.utils import fgraph_to_python
from aesara.scalar.basic import Add
from aesara.tensor.elemwise import Elemwise
from aesara.tensor.type import scalar, vector
from aesara.tensor.type_other import NoneConst


@singledispatch
def to_python(op, **kwargs):
    raise NotImplementedError()


@to_python.register(Elemwise)
def to_python_Elemwise(op, **kwargs):
    scalar_op = op.scalar_op
    return to_python(scalar_op, **kwargs)


@to_python.register(Add)
def to_python_Add(op, **kwargs):
    def add(*args):
        return np.add(*args)

    return add


def test_fgraph_to_python_names():
    import inspect

    x = scalar("1x")
    y = scalar("_")
    z = scalar()
    q = scalar("def")
    r = NoneConst

    out_fg = FunctionGraph([x, y, z, q, r], [x, y, z, q, r], clone=False)
    out_jx = fgraph_to_python(out_fg, to_python)

    sig = inspect.signature(out_jx)
    assert (x.auto_name, "_", z.auto_name, q.auto_name, r.name) == tuple(
        sig.parameters.keys()
    )
    assert (1, 2, 3, 4, 5) == out_jx(1, 2, 3, 4, 5)


def test_fgraph_to_python_once():
    """Make sure that an output is only computed once when it's referenced multiple times."""

    x = vector("x")
    y = vector("y")

    class TestOp(Op):
        def __init__(self):
            self.called = 0

        def make_node(self, *args):
            return Apply(self, list(args), [x.type() for x in args])

        def perform(self, inputs, outputs):
            for i, inp in enumerate(inputs):
                outputs[i][0] = inp[0]

    @to_python.register(TestOp)
    def to_python_TestOp(op, **kwargs):
        def func(*args, op=op):
            op.called += 1
            return list(args)

        return func

    op1 = TestOp()
    op2 = TestOp()

    q, r = op1(x, y)
    outs = op2(q + r, q + r)

    out_fg = FunctionGraph([x, y], outs, clone=False)
    assert len(out_fg.outputs) == 2

    out_py = fgraph_to_python(out_fg, to_python)

    x_val = np.r_[1, 2].astype(config.floatX)
    y_val = np.r_[2, 3].astype(config.floatX)

    res = out_py(x_val, y_val)
    assert len(res) == 2
    assert op1.called == 1
    assert op2.called == 1

    res = out_py(x_val, y_val)
    assert len(res) == 2
    assert op1.called == 2
    assert op2.called == 2
