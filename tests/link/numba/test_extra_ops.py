import contextlib

import numpy as np
import pytest

import aesara.tensor as at
from aesara import config
from aesara.compile.sharedvalue import SharedVariable
from aesara.graph.basic import Constant
from aesara.graph.fg import FunctionGraph
from aesara.tensor import extra_ops
from tests.link.numba.test_basic import compare_numba_and_py, set_test_value


rng = np.random.default_rng(42849)


@pytest.mark.parametrize(
    "val",
    [
        set_test_value(at.lscalar(), np.array(6, dtype="int64")),
    ],
)
def test_Bartlett(val):
    g = extra_ops.bartlett(val)
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
    "x, shape",
    [
        (
            set_test_value(at.vector(), rng.random(size=(2,)).astype(config.floatX)),
            [set_test_value(at.lscalar(), np.array(v)) for v in [3, 2]],
        ),
        (
            set_test_value(at.vector(), rng.random(size=(2,)).astype(config.floatX)),
            [at.as_tensor(3, dtype=np.int64), at.as_tensor(2, dtype=np.int64)],
        ),
        (
            set_test_value(at.vector(), rng.random(size=(2,)).astype(config.floatX)),
            at.as_tensor([set_test_value(at.lscalar(), np.array(v)) for v in [3, 2]]),
        ),
        (
            set_test_value(at.vector(), rng.random(size=(2,)).astype(config.floatX)),
            [at.as_tensor(3, dtype=np.int8), at.as_tensor(2, dtype=np.int64)],
        ),
    ],
)
def test_BroadcastTo(x, shape):
    g = extra_ops.BroadcastTo()(x, shape)
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
    "val, axis, mode",
    [
        (
            set_test_value(
                at.matrix(), np.arange(3, dtype=config.floatX).reshape((3, 1))
            ),
            1,
            "add",
        ),
        (
            set_test_value(
                at.matrix(), np.arange(6, dtype=config.floatX).reshape((3, 2))
            ),
            0,
            "add",
        ),
        (
            set_test_value(
                at.matrix(), np.arange(6, dtype=config.floatX).reshape((3, 2))
            ),
            1,
            "add",
        ),
        (
            set_test_value(
                at.matrix(), np.arange(6, dtype=config.floatX).reshape((3, 2))
            ),
            0,
            "mul",
        ),
        (
            set_test_value(
                at.matrix(), np.arange(6, dtype=config.floatX).reshape((3, 2))
            ),
            1,
            "mul",
        ),
    ],
)
def test_CumOp(val, axis, mode):
    g = extra_ops.CumOp(axis=axis, mode=mode)(val)
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
    "a, val",
    [
        (
            set_test_value(at.lmatrix(), np.zeros((10, 2), dtype="int64")),
            set_test_value(at.lscalar(), np.array(1, dtype="int64")),
        )
    ],
)
def test_FillDiagonal(a, val):
    g = extra_ops.FillDiagonal()(a, val)
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
    "a, val, offset",
    [
        (
            set_test_value(at.lmatrix(), np.zeros((10, 2), dtype="int64")),
            set_test_value(at.lscalar(), np.array(1, dtype="int64")),
            set_test_value(at.lscalar(), np.array(-1, dtype="int64")),
        ),
        (
            set_test_value(at.lmatrix(), np.zeros((10, 2), dtype="int64")),
            set_test_value(at.lscalar(), np.array(1, dtype="int64")),
            set_test_value(at.lscalar(), np.array(0, dtype="int64")),
        ),
        (
            set_test_value(at.lmatrix(), np.zeros((10, 3), dtype="int64")),
            set_test_value(at.lscalar(), np.array(1, dtype="int64")),
            set_test_value(at.lscalar(), np.array(1, dtype="int64")),
        ),
    ],
)
def test_FillDiagonalOffset(a, val, offset):
    g = extra_ops.FillDiagonalOffset()(a, val, offset)
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
    "arr, shape, mode, order, exc",
    [
        (
            tuple(set_test_value(at.lscalar(), v) for v in np.array([0])),
            set_test_value(at.lvector(), np.array([2])),
            "raise",
            "C",
            None,
        ),
        (
            tuple(set_test_value(at.lscalar(), v) for v in np.array([0, 0, 3])),
            set_test_value(at.lvector(), np.array([2, 3, 4])),
            "raise",
            "C",
            None,
        ),
        (
            tuple(
                set_test_value(at.lvector(), v)
                for v in np.array([[0, 1], [2, 0], [1, 3]])
            ),
            set_test_value(at.lvector(), np.array([2, 3, 4])),
            "raise",
            "C",
            None,
        ),
        (
            tuple(
                set_test_value(at.lvector(), v)
                for v in np.array([[0, 1], [2, 0], [1, 3]])
            ),
            set_test_value(at.lvector(), np.array([2, 3, 4])),
            "raise",
            "F",
            NotImplementedError,
        ),
        (
            tuple(
                set_test_value(at.lvector(), v)
                for v in np.array([[0, 1, 2], [2, 0, 3], [1, 3, 5]])
            ),
            set_test_value(at.lvector(), np.array([2, 3, 4])),
            "raise",
            "C",
            ValueError,
        ),
        (
            tuple(
                set_test_value(at.lvector(), v)
                for v in np.array([[0, 1, 2], [2, 0, 3], [1, 3, 5]])
            ),
            set_test_value(at.lvector(), np.array([2, 3, 4])),
            "wrap",
            "C",
            None,
        ),
        (
            tuple(
                set_test_value(at.lvector(), v)
                for v in np.array([[0, 1, 2], [2, 0, 3], [1, 3, 5]])
            ),
            set_test_value(at.lvector(), np.array([2, 3, 4])),
            "clip",
            "C",
            None,
        ),
    ],
)
def test_RavelMultiIndex(arr, shape, mode, order, exc):
    g = extra_ops.RavelMultiIndex(mode, order)(*(arr + (shape,)))
    g_fg = FunctionGraph(outputs=[g])

    cm = contextlib.suppress() if exc is None else pytest.raises(exc)
    with cm:
        compare_numba_and_py(
            g_fg,
            [
                i.tag.test_value
                for i in g_fg.inputs
                if not isinstance(i, (SharedVariable, Constant))
            ],
        )


@pytest.mark.parametrize(
    "x, repeats, axis, exc",
    [
        (
            set_test_value(at.lscalar(), np.array(1, dtype="int64")),
            set_test_value(at.lscalar(), np.array(0, dtype="int64")),
            None,
            None,
        ),
        (
            set_test_value(at.lmatrix(), np.zeros((2, 2), dtype="int64")),
            set_test_value(at.lscalar(), np.array(1, dtype="int64")),
            None,
            None,
        ),
        (
            set_test_value(at.lvector(), np.arange(2, dtype="int64")),
            set_test_value(at.lvector(), np.array([1, 1], dtype="int64")),
            None,
            None,
        ),
        (
            set_test_value(at.lmatrix(), np.zeros((2, 2), dtype="int64")),
            set_test_value(at.lscalar(), np.array(1, dtype="int64")),
            0,
            UserWarning,
        ),
    ],
)
def test_Repeat(x, repeats, axis, exc):
    g = extra_ops.Repeat(axis)(x, repeats)
    g_fg = FunctionGraph(outputs=[g])

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            g_fg,
            [
                i.tag.test_value
                for i in g_fg.inputs
                if not isinstance(i, (SharedVariable, Constant))
            ],
        )


@pytest.mark.parametrize(
    "x, axis, return_index, return_inverse, return_counts, exc",
    [
        (
            set_test_value(at.lscalar(), np.array(1, dtype="int64")),
            None,
            False,
            False,
            False,
            None,
        ),
        (
            set_test_value(at.lvector(), np.array([1, 1, 2], dtype="int64")),
            None,
            False,
            False,
            False,
            None,
        ),
        (
            set_test_value(at.lmatrix(), np.array([[1, 1], [2, 2]], dtype="int64")),
            None,
            False,
            False,
            False,
            None,
        ),
        (
            set_test_value(
                at.lmatrix(), np.array([[1, 1], [1, 1], [2, 2]], dtype="int64")
            ),
            0,
            False,
            False,
            False,
            UserWarning,
        ),
        (
            set_test_value(
                at.lmatrix(), np.array([[1, 1], [1, 1], [2, 2]], dtype="int64")
            ),
            0,
            True,
            True,
            True,
            UserWarning,
        ),
    ],
)
def test_Unique(x, axis, return_index, return_inverse, return_counts, exc):
    g = extra_ops.Unique(return_index, return_inverse, return_counts, axis)(x)

    if isinstance(g, list):
        g_fg = FunctionGraph(outputs=g)
    else:
        g_fg = FunctionGraph(outputs=[g])

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            g_fg,
            [
                i.tag.test_value
                for i in g_fg.inputs
                if not isinstance(i, (SharedVariable, Constant))
            ],
        )


@pytest.mark.parametrize(
    "arr, shape, order, exc",
    [
        (
            set_test_value(at.lvector(), np.array([9, 15, 1], dtype="int64")),
            at.as_tensor([2, 3, 4]),
            "C",
            None,
        ),
        (
            set_test_value(at.lvector(), np.array([1, 0], dtype="int64")),
            at.as_tensor([2]),
            "C",
            None,
        ),
        (
            set_test_value(at.lvector(), np.array([9, 15, 1], dtype="int64")),
            at.as_tensor([2, 3, 4]),
            "F",
            NotImplementedError,
        ),
    ],
)
def test_UnravelIndex(arr, shape, order, exc):
    g = extra_ops.UnravelIndex(order)(arr, shape)

    if isinstance(g, list):
        g_fg = FunctionGraph(outputs=g)
    else:
        g_fg = FunctionGraph(outputs=[g])

    cm = contextlib.suppress() if exc is None else pytest.raises(exc)
    with cm:
        compare_numba_and_py(
            g_fg,
            [
                i.tag.test_value
                for i in g_fg.inputs
                if not isinstance(i, (SharedVariable, Constant))
            ],
        )


@pytest.mark.parametrize(
    "a, v, side, sorter, exc",
    [
        (
            set_test_value(at.vector(), np.array([1.0, 2.0, 3.0], dtype=config.floatX)),
            set_test_value(at.matrix(), rng.random((3, 2)).astype(config.floatX)),
            "left",
            None,
            None,
        ),
        pytest.param(
            set_test_value(
                at.vector(),
                np.array([0.29769574, 0.71649186, 0.20475563]).astype(config.floatX),
            ),
            set_test_value(
                at.matrix(),
                np.array(
                    [
                        [0.18847123, 0.39659508],
                        [0.56220006, 0.57428752],
                        [0.86720994, 0.44522637],
                    ]
                ).astype(config.floatX),
            ),
            "left",
            None,
            None,
            marks=pytest.mark.xfail(
                reason="This won't work until https://github.com/numba/numba/pull/7005 is merged"
            ),
        ),
        (
            set_test_value(at.vector(), np.array([1.0, 2.0, 3.0], dtype=config.floatX)),
            set_test_value(at.matrix(), rng.random((3, 2)).astype(config.floatX)),
            "right",
            set_test_value(at.lvector(), np.array([0, 2, 1])),
            UserWarning,
        ),
    ],
)
def test_Searchsorted(a, v, side, sorter, exc):
    g = extra_ops.SearchsortedOp(side)(a, v, sorter)
    g_fg = FunctionGraph(outputs=[g])

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            g_fg,
            [
                i.tag.test_value
                for i in g_fg.inputs
                if not isinstance(i, (SharedVariable, Constant))
            ],
        )
