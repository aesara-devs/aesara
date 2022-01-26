from functools import singledispatch

import numpy as np

import aesara.tensor as at
from aesara import config
from aesara.graph.basic import Apply, NominalVariable
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import Op
from aesara.link.basic import Linker
from aesara.link.utils import (
    fgraph_to_python,
    get_name_for_object,
    map_storage,
    unique_name_generator,
)
from aesara.scalar.basic import Add
from aesara.tensor.elemwise import Elemwise
from aesara.tensor.type import TensorType, scalar, vector
from aesara.tensor.type_other import NoneConst
from tests.link.test_link import OpWithIgnoredInput


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

    assert q_name_1 == q_name_2 == q.auto_name

    unique_names = unique_name_generator()

    r = vector()
    r_name_1 = unique_names(r)
    r_name_2 = unique_names(r, force_unique=True)

    assert r_name_1 != r_name_2

    r_name_3 = unique_names(r)
    assert r_name_2 == r_name_3


def test_map_storage_uncomputed_inputs():

    # This will not be considered an input by `FunctionGraph`
    a = NominalVariable(0, TensorType("floatX", (None,)))
    a.name = "a"
    b = at.mul(a, at.as_tensor(2.0))
    x = at.vector("x")
    c = at.as_tensor(3.0)
    y = c * x
    op = OpWithIgnoredInput()
    z = op(b, y)

    fgraph = FunctionGraph(outputs=[z], clone=False)
    order = Linker.toposort(fgraph)

    input_storage, output_storage, storage_map = map_storage(fgraph, order, None, None)

    assert len(storage_map[b]) == 0
    assert storage_map[x] == [None]
    assert storage_map[y] == [None]
    assert storage_map[c] == [np.array(3.0, dtype=config.floatX)]
    assert storage_map[z] == [None]
