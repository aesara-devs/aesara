import numpy as np
import pytest

import aesara.scalar as aes
import aesara.scalar.basic as aesb
import aesara.tensor as at
from aesara import config
from aesara.compile.sharedvalue import SharedVariable
from aesara.graph.basic import Constant
from aesara.graph.fg import FunctionGraph
from aesara.scalar.basic import Composite
from aesara.tensor.elemwise import Elemwise
from tests.link.numba.test_basic import compare_numba_and_py, set_test_value


rng = np.random.default_rng(42849)


@pytest.mark.parametrize(
    "x, y",
    [
        (
            set_test_value(at.lvector(), np.arange(4, dtype="int64")),
            set_test_value(at.dvector(), np.arange(4, dtype="float64")),
        ),
        (
            set_test_value(at.dmatrix(), np.arange(4, dtype="float64").reshape((2, 2))),
            set_test_value(at.lscalar(), np.array(4, dtype="int64")),
        ),
    ],
)
def test_Second(x, y):
    # We use the `Elemwise`-wrapped version of `Second`
    g = at.second(x, y)
    g_fg = FunctionGraph(outputs=[g])
    compare_numba_and_py(
        g_fg,
        [
            i.tag.test_value
            for i in g_fg.inputs
            if not isinstance(i, (SharedVariable, Constant))
        ],
    )


@pytest.mark.parametrize(
    "v, min, max",
    [
        (set_test_value(at.scalar(), np.array(10, dtype=config.floatX)), 3.0, 7.0),
        (set_test_value(at.scalar(), np.array(1, dtype=config.floatX)), 3.0, 7.0),
        (set_test_value(at.scalar(), np.array(10, dtype=config.floatX)), 7.0, 3.0),
    ],
)
def test_Clip(v, min, max):
    g = aes.clip(v, min, max)
    g_fg = FunctionGraph(outputs=[g])

    compare_numba_and_py(
        g_fg,
        [
            i.tag.test_value
            for i in g_fg.inputs
            if not isinstance(i, (SharedVariable, Constant))
        ],
    )


@pytest.mark.parametrize(
    "inputs, input_values, scalar_fn",
    [
        (
            [at.scalar("x"), at.scalar("y"), at.scalar("z")],
            [
                np.array(10, dtype=config.floatX),
                np.array(20, dtype=config.floatX),
                np.array(30, dtype=config.floatX),
            ],
            lambda x, y, z: aes.add(x, y, z),
        ),
        (
            [at.scalar("x"), at.scalar("y"), at.scalar("z")],
            [
                np.array(10, dtype=config.floatX),
                np.array(20, dtype=config.floatX),
                np.array(30, dtype=config.floatX),
            ],
            lambda x, y, z: aes.mul(x, y, z),
        ),
        (
            [at.scalar("x"), at.scalar("y")],
            [
                np.array(10, dtype=config.floatX),
                np.array(20, dtype=config.floatX),
            ],
            lambda x, y: x + y * 2 + aes.exp(x - y),
        ),
    ],
)
def test_Composite(inputs, input_values, scalar_fn):
    composite_inputs = [aes.float64(i.name) for i in inputs]
    comp_op = Elemwise(Composite(composite_inputs, [scalar_fn(*composite_inputs)]))
    out_fg = FunctionGraph(inputs, [comp_op(*inputs)])
    compare_numba_and_py(out_fg, input_values)


@pytest.mark.parametrize(
    "v, dtype",
    [
        (set_test_value(at.fscalar(), np.array(1.0, dtype="float32")), aesb.float64),
        (set_test_value(at.dscalar(), np.array(1.0, dtype="float64")), aesb.float32),
    ],
)
def test_Cast(v, dtype):
    g = aesb.Cast(dtype)(v)
    g_fg = FunctionGraph(outputs=[g])
    compare_numba_and_py(
        g_fg,
        [
            i.tag.test_value
            for i in g_fg.inputs
            if not isinstance(i, (SharedVariable, Constant))
        ],
    )


@pytest.mark.parametrize(
    "v, dtype",
    [
        (set_test_value(at.iscalar(), np.array(10, dtype="int32")), aesb.float64),
    ],
)
def test_reciprocal(v, dtype):
    g = aesb.reciprocal(v)
    g_fg = FunctionGraph(outputs=[g])
    compare_numba_and_py(
        g_fg,
        [
            i.tag.test_value
            for i in g_fg.inputs
            if not isinstance(i, (SharedVariable, Constant))
        ],
    )
