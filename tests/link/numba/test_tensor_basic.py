import numpy as np
import pytest

import aesara.scalar as aes
import aesara.tensor as at
import aesara.tensor.basic as atb
from aesara import config
from aesara.compile.sharedvalue import SharedVariable
from aesara.graph.basic import Constant
from aesara.graph.fg import FunctionGraph
from aesara.tensor.shape import Unbroadcast
from tests.link.numba.test_basic import (
    compare_numba_and_py,
    compare_shape_dtype,
    set_test_value,
)


rng = np.random.default_rng(42849)


@pytest.mark.parametrize(
    "v, shape",
    [
        (0.0, (2, 3)),
        (1.1, (2, 3)),
        (set_test_value(at.scalar("a"), np.array(10.0, dtype=config.floatX)), (20,)),
        (set_test_value(at.vector("a"), np.ones(10, dtype=config.floatX)), (20, 10)),
    ],
)
def test_Alloc(v, shape):
    g = at.alloc(v, *shape)
    g_fg = FunctionGraph(outputs=[g])

    (_, numba_res) = compare_numba_and_py(
        g_fg,
        [
            i.tag.test_value
            for i in g_fg.inputs
            if not isinstance(i, (SharedVariable, Constant))
        ],
    )

    assert numba_res[0].shape == shape


def test_AllocEmpty():

    x = at.empty((2, 3), dtype="float32")
    x_fg = FunctionGraph([], [x])

    # We cannot compare the values in the arrays, only the shapes and dtypes
    compare_numba_and_py(x_fg, [], assert_fn=compare_shape_dtype)


@pytest.mark.parametrize(
    "v, offset",
    [
        (set_test_value(at.vector(), np.arange(10, dtype=config.floatX)), 0),
        (set_test_value(at.vector(), np.arange(10, dtype=config.floatX)), 1),
        (set_test_value(at.vector(), np.arange(10, dtype=config.floatX)), -1),
    ],
)
def test_AllocDiag(v, offset):
    g = atb.AllocDiag(offset=offset)(v)
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
    "v", [set_test_value(aes.float64(), np.array(1.0, dtype="float64"))]
)
def test_TensorFromScalar(v):
    g = atb.TensorFromScalar()(v)
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
    "v",
    [
        set_test_value(at.scalar(), np.array(1.0, dtype=config.floatX)),
    ],
)
def test_ScalarFromTensor(v):
    g = atb.ScalarFromTensor()(v)
    g_fg = FunctionGraph(outputs=[g])
    compare_numba_and_py(
        g_fg,
        [
            i.tag.test_value
            for i in g_fg.inputs
            if not isinstance(i, (SharedVariable, Constant))
        ],
    )


def test_Unbroadcast():
    v = set_test_value(at.row(), np.array([[1.0, 2.0]], dtype=config.floatX))
    g = Unbroadcast(0)(v)
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
    "vals, dtype",
    [
        (
            (
                set_test_value(at.scalar(), np.array(1, dtype=config.floatX)),
                set_test_value(at.scalar(), np.array(2, dtype=config.floatX)),
                set_test_value(at.scalar(), np.array(3, dtype=config.floatX)),
            ),
            config.floatX,
        ),
        (
            (
                set_test_value(at.dscalar(), np.array(1, dtype=np.float64)),
                set_test_value(at.lscalar(), np.array(3, dtype=np.int32)),
            ),
            "float64",
        ),
        (
            (set_test_value(at.iscalar(), np.array(1, dtype=np.int32)),),
            "float64",
        ),
        (
            (set_test_value(at.scalar(dtype=bool), True),),
            bool,
        ),
    ],
)
def test_MakeVector(vals, dtype):
    g = atb.MakeVector(dtype)(*vals)
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
    "start, stop, step, dtype",
    [
        (
            set_test_value(at.lscalar(), np.array(1)),
            set_test_value(at.lscalar(), np.array(10)),
            set_test_value(at.lscalar(), np.array(3)),
            config.floatX,
        ),
    ],
)
def test_ARange(start, stop, step, dtype):
    g = atb.ARange(dtype)(start, stop, step)
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
    "vals, axis",
    [
        (
            (
                set_test_value(
                    at.matrix(), rng.normal(size=(1, 2)).astype(config.floatX)
                ),
                set_test_value(
                    at.matrix(), rng.normal(size=(1, 2)).astype(config.floatX)
                ),
            ),
            0,
        ),
        (
            (
                set_test_value(
                    at.matrix(), rng.normal(size=(2, 1)).astype(config.floatX)
                ),
                set_test_value(
                    at.matrix(), rng.normal(size=(3, 1)).astype(config.floatX)
                ),
            ),
            0,
        ),
        (
            (
                set_test_value(
                    at.matrix(), rng.normal(size=(1, 2)).astype(config.floatX)
                ),
                set_test_value(
                    at.matrix(), rng.normal(size=(1, 2)).astype(config.floatX)
                ),
            ),
            1,
        ),
        (
            (
                set_test_value(
                    at.matrix(), rng.normal(size=(2, 2)).astype(config.floatX)
                ),
                set_test_value(
                    at.matrix(), rng.normal(size=(2, 1)).astype(config.floatX)
                ),
            ),
            1,
        ),
    ],
)
def test_Join(vals, axis):
    g = at.join(axis, *vals)
    g_fg = FunctionGraph(outputs=[g])

    compare_numba_and_py(
        g_fg,
        [
            i.tag.test_value
            for i in g_fg.inputs
            if not isinstance(i, (SharedVariable, Constant))
        ],
    )


def test_Join_view():
    vals = (
        set_test_value(at.matrix(), rng.normal(size=(2, 2)).astype(config.floatX)),
        set_test_value(at.matrix(), rng.normal(size=(2, 2)).astype(config.floatX)),
    )
    g = atb.Join(view=1)(1, *vals)
    g_fg = FunctionGraph(outputs=[g])

    with pytest.raises(NotImplementedError):
        compare_numba_and_py(
            g_fg,
            [
                i.tag.test_value
                for i in g_fg.inputs
                if not isinstance(i, (SharedVariable, Constant))
            ],
        )


@pytest.mark.parametrize(
    "n_splits, axis, values, sizes",
    [
        (
            0,
            0,
            set_test_value(at.vector(), rng.normal(size=20).astype(config.floatX)),
            set_test_value(at.vector(dtype="int64"), []),
        ),
        (
            5,
            0,
            set_test_value(at.vector(), rng.normal(size=5).astype(config.floatX)),
            set_test_value(
                at.vector(dtype="int64"), rng.multinomial(5, np.ones(5) / 5)
            ),
        ),
        (
            5,
            0,
            set_test_value(at.vector(), rng.normal(size=10).astype(config.floatX)),
            set_test_value(
                at.vector(dtype="int64"), rng.multinomial(10, np.ones(5) / 5)
            ),
        ),
        (
            5,
            -1,
            set_test_value(at.matrix(), rng.normal(size=(11, 7)).astype(config.floatX)),
            set_test_value(
                at.vector(dtype="int64"), rng.multinomial(7, np.ones(5) / 5)
            ),
        ),
        (
            5,
            -2,
            set_test_value(at.matrix(), rng.normal(size=(11, 7)).astype(config.floatX)),
            set_test_value(
                at.vector(dtype="int64"), rng.multinomial(11, np.ones(5) / 5)
            ),
        ),
    ],
)
def test_Split(n_splits, axis, values, sizes):
    g = at.split(values, sizes, n_splits, axis=axis)
    assert len(g) == n_splits
    if n_splits == 0:
        return
    g_fg = FunctionGraph(outputs=[g] if n_splits == 1 else g)

    compare_numba_and_py(
        g_fg,
        [
            i.tag.test_value
            for i in g_fg.inputs
            if not isinstance(i, (SharedVariable, Constant))
        ],
    )


@pytest.mark.parametrize(
    "val, offset",
    [
        (
            set_test_value(
                at.matrix(), np.arange(10 * 10, dtype=config.floatX).reshape((10, 10))
            ),
            0,
        ),
        (
            set_test_value(at.vector(), np.arange(10, dtype=config.floatX)),
            0,
        ),
    ],
)
def test_ExtractDiag(val, offset):
    g = at.diag(val, offset)
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
    "n, m, k, dtype",
    [
        (set_test_value(at.lscalar(), np.array(1, dtype=np.int64)), None, 0, None),
        (
            set_test_value(at.lscalar(), np.array(1, dtype=np.int64)),
            set_test_value(at.lscalar(), np.array(2, dtype=np.int64)),
            0,
            "float32",
        ),
        (
            set_test_value(at.lscalar(), np.array(1, dtype=np.int64)),
            set_test_value(at.lscalar(), np.array(2, dtype=np.int64)),
            1,
            "int64",
        ),
    ],
)
def test_Eye(n, m, k, dtype):
    g = at.eye(n, m, k, dtype=dtype)
    g_fg = FunctionGraph(outputs=[g])

    compare_numba_and_py(
        g_fg,
        [
            i.tag.test_value
            for i in g_fg.inputs
            if not isinstance(i, (SharedVariable, Constant))
        ],
    )
