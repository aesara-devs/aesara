import inspect
from functools import singledispatch

import numpy as np

from aesara import config
from aesara.graph.basic import Apply
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import Op
from aesara.link.utils import (
    fgraph_to_python,
    get_name_for_object,
    unique_name_generator,
)
from aesara.scalar.basic import Add, float64
from aesara.tensor import constant
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
    z = float64()
    q = scalar("def")
    r = NoneConst

    out_fg = FunctionGraph([x, y, z, q, r], [x, y, z, q, r], clone=False)
    out_jx = fgraph_to_python(out_fg, to_python)

    sig = inspect.signature(out_jx)
    assert (
        "tensor_variable",
        "_",
        "scalar_variable",
        "tensor_variable_1",
        r.name,
    ) == tuple(sig.parameters.keys())
    assert (1, 2, 3, 4, 5) == out_jx(1, 2, 3, 4, 5)

    obj = object()
    assert get_name_for_object(obj) == type(obj).__name__


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


def test_fgraph_to_python_multiline_str():
    """Make sure that multiline `__str__` values are supported by `fgraph_to_python`."""

    x = vector("x")
    y = vector("y")

    class TestOp(Op):
        def __init__(self):
            super().__init__()

        def make_node(self, *args):
            return Apply(self, list(args), [x.type() for x in args])

        def perform(self, inputs, outputs):
            for i, inp in enumerate(inputs):
                outputs[i][0] = inp[0]

        def __str__(self):
            return "Test\nOp()"

    @to_python.register(TestOp)
    def to_python_TestOp(op, **kwargs):
        def func(*args, op=op):
            return list(args)

        return func

    op1 = TestOp()
    op2 = TestOp()

    q, r = op1(x, y)
    outs = op2(q + r, q + r)

    out_fg = FunctionGraph([x, y], outs, clone=False)
    assert len(out_fg.outputs) == 2

    out_py = fgraph_to_python(out_fg, to_python)

    out_py_src = inspect.getsource(out_py)

    assert (
        """
    # Elemwise{add,no_inplace}(Test
    # Op().0, Test
    # Op().1)
    """
        in out_py_src
    )


def test_fgraph_to_python_constant_outputs():
    """Make sure that constant outputs are handled properly."""

    y = constant(1)

    out_fg = FunctionGraph([], [y], clone=False)

    out_py = fgraph_to_python(out_fg, to_python)

    assert out_py()[0] is y.data


def test_fgraph_to_python_constant_inputs():
    x = constant([1.0])
    y = vector("y")

    out = x + y
    out_fg = FunctionGraph(outputs=[out], clone=False)

    out_py = fgraph_to_python(out_fg, to_python, storage_map=None)

    res = out_py(2.0)
    assert res == (3.0,)

    storage_map = {out: [None], x: [np.r_[2.0]], y: [None]}
    out_py = fgraph_to_python(out_fg, to_python, storage_map=storage_map)

    res = out_py(2.0)
    assert res == (4.0,)


def test_unique_name_generator():
    unique_names = unique_name_generator(["blah"], suffix_sep="_")

    x = vector("blah")
    x_name = unique_names(x)
    assert x_name == "blah_1"

    y = vector("blah_1")
    y_name = unique_names(y)
    assert y_name == "blah_1_1"

    # Make sure that the old name associations are still good
    x_name = unique_names(x)
    assert x_name == "blah_1"
    y_name = unique_names(y)
    assert y_name == "blah_1_1"

    # Try a name that overlaps with the original name
    z = vector("blah")
    z_name = unique_names(z)
    assert z_name == "blah_2"

    # Try a name that overlaps with an extended name
    w = vector("blah_1")
    w_name = unique_names(w)
    assert w_name == "blah_1_2"

    q = vector()
    q_name_1 = unique_names(q)
    q_name_2 = unique_names(q)

    assert q_name_1 == q_name_2 == "tensor_variable"

    unique_names = unique_name_generator()

    r = vector()
    r_name_1 = unique_names(r)
    r_name_2 = unique_names(r, force_unique=True)

    assert r_name_1 != r_name_2

    r_name_3 = unique_names(r)
    assert r_name_2 == r_name_3
