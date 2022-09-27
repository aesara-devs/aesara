import contextlib

import numpy as np
import pytest

import aesara.tensor as at
from aesara.compile.sharedvalue import SharedVariable
from aesara.graph.basic import Constant
from aesara.graph.fg import FunctionGraph
from aesara.tensor import nlinalg, slinalg
from tests.link.numba.test_basic import compare_numba_and_py, set_test_value


rng = np.random.default_rng(42849)


@pytest.mark.parametrize(
    "x, lower, exc",
    [
        (
            set_test_value(
                at.dmatrix(),
                (lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64")),
            ),
            True,
            None,
        ),
        (
            set_test_value(
                at.lmatrix(),
                (lambda x: x.T.dot(x))(
                    rng.integers(1, 10, size=(3, 3)).astype("int64")
                ),
            ),
            True,
            None,
        ),
        (
            set_test_value(
                at.dmatrix(),
                (lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64")),
            ),
            False,
            UserWarning,
        ),
    ],
)
def test_Cholesky(x, lower, exc):
    g = slinalg.Cholesky(lower)(x)

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
    "A, x, lower, exc",
    [
        (
            set_test_value(
                at.dmatrix(),
                (lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64")),
            ),
            set_test_value(at.dvector(), rng.random(size=(3,)).astype("float64")),
            "gen",
            None,
        ),
        (
            set_test_value(
                at.lmatrix(),
                (lambda x: x.T.dot(x))(
                    rng.integers(1, 10, size=(3, 3)).astype("int64")
                ),
            ),
            set_test_value(at.dvector(), rng.random(size=(3,)).astype("float64")),
            "gen",
            None,
        ),
    ],
)
def test_Solve(A, x, lower, exc):
    g = slinalg.Solve(lower)(A, x)

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
    "A, x, lower, exc",
    [
        (
            set_test_value(
                at.dmatrix(),
                (lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64")),
            ),
            set_test_value(at.dvector(), rng.random(size=(3,)).astype("float64")),
            "sym",
            UserWarning,
        ),
    ],
)
def test_SolveTriangular(A, x, lower, exc):
    g = slinalg.SolveTriangular(lower)(A, x)

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
    "x, exc",
    [
        (
            set_test_value(
                at.dmatrix(),
                (lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64")),
            ),
            None,
        ),
        (
            set_test_value(
                at.lmatrix(),
                (lambda x: x.T.dot(x))(rng.poisson(size=(3, 3)).astype("int64")),
            ),
            None,
        ),
    ],
)
def test_Det(x, exc):
    g = nlinalg.Det()(x)
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


# We were seeing some weird results in CI where the following two almost
# sign-swapped results were being return from Numba and Python, respectively.
# The issue might be related to https://github.com/numba/numba/issues/4519.
# Regardless, I was not able to reproduce anything like it locally after
# extensive testing.
x = np.array(
    [
        [-0.60407637, -0.71177603, -0.35842241],
        [-0.07735968, 0.50000561, -0.86256007],
        [-0.7931628, 0.49332471, 0.35710434],
    ],
    dtype=np.float64,
)

y = np.array(
    [
        [0.60407637, 0.71177603, -0.35842241],
        [0.07735968, -0.50000561, -0.86256007],
        [0.7931628, -0.49332471, 0.35710434],
    ],
    dtype=np.float64,
)


@pytest.mark.parametrize(
    "x, exc",
    [
        (
            set_test_value(
                at.dmatrix(),
                (lambda x: x.T.dot(x))(x),
            ),
            None,
        ),
        (
            set_test_value(
                at.dmatrix(),
                (lambda x: x.T.dot(x))(y),
            ),
            None,
        ),
        (
            set_test_value(
                at.lmatrix(),
                (lambda x: x.T.dot(x))(
                    rng.integers(1, 10, size=(3, 3)).astype("int64")
                ),
            ),
            None,
        ),
    ],
)
def test_Eig(x, exc):
    g = nlinalg.Eig()(x)

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
    "x, uplo, exc",
    [
        (
            set_test_value(
                at.dmatrix(),
                (lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64")),
            ),
            "L",
            None,
        ),
        (
            set_test_value(
                at.lmatrix(),
                (lambda x: x.T.dot(x))(
                    rng.integers(1, 10, size=(3, 3)).astype("int64")
                ),
            ),
            "U",
            UserWarning,
        ),
    ],
)
def test_Eigh(x, uplo, exc):
    g = nlinalg.Eigh(uplo)(x)

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
    "op, x, exc, op_args",
    [
        (
            nlinalg.MatrixInverse,
            set_test_value(
                at.dmatrix(),
                (lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64")),
            ),
            None,
            (),
        ),
        (
            nlinalg.MatrixInverse,
            set_test_value(
                at.lmatrix(),
                (lambda x: x.T.dot(x))(
                    rng.integers(1, 10, size=(3, 3)).astype("int64")
                ),
            ),
            None,
            (),
        ),
        (
            nlinalg.Inv,
            set_test_value(
                at.dmatrix(),
                (lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64")),
            ),
            None,
            (),
        ),
        (
            nlinalg.Inv,
            set_test_value(
                at.lmatrix(),
                (lambda x: x.T.dot(x))(
                    rng.integers(1, 10, size=(3, 3)).astype("int64")
                ),
            ),
            None,
            (),
        ),
        (
            nlinalg.MatrixPinv,
            set_test_value(
                at.dmatrix(),
                (lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64")),
            ),
            None,
            (True,),
        ),
        (
            nlinalg.MatrixPinv,
            set_test_value(
                at.lmatrix(),
                (lambda x: x.T.dot(x))(
                    rng.integers(1, 10, size=(3, 3)).astype("int64")
                ),
            ),
            None,
            (False,),
        ),
    ],
)
def test_matrix_inverses(op, x, exc, op_args):
    g = op(*op_args)(x)
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
    "x, mode, exc",
    [
        (
            set_test_value(
                at.dmatrix(),
                (lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64")),
            ),
            "reduced",
            None,
        ),
        (
            set_test_value(
                at.dmatrix(),
                (lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64")),
            ),
            "r",
            None,
        ),
        (
            set_test_value(
                at.lmatrix(),
                (lambda x: x.T.dot(x))(
                    rng.integers(1, 10, size=(3, 3)).astype("int64")
                ),
            ),
            "reduced",
            None,
        ),
        (
            set_test_value(
                at.lmatrix(),
                (lambda x: x.T.dot(x))(
                    rng.integers(1, 10, size=(3, 3)).astype("int64")
                ),
            ),
            "complete",
            UserWarning,
        ),
    ],
)
def test_QRFull(x, mode, exc):
    g = nlinalg.QRFull(mode)(x)

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
    "x, full_matrices, compute_uv, exc",
    [
        (
            set_test_value(
                at.dmatrix(),
                (lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64")),
            ),
            True,
            True,
            None,
        ),
        (
            set_test_value(
                at.dmatrix(),
                (lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64")),
            ),
            False,
            True,
            None,
        ),
        (
            set_test_value(
                at.lmatrix(),
                (lambda x: x.T.dot(x))(
                    rng.integers(1, 10, size=(3, 3)).astype("int64")
                ),
            ),
            True,
            True,
            None,
        ),
        (
            set_test_value(
                at.lmatrix(),
                (lambda x: x.T.dot(x))(
                    rng.integers(1, 10, size=(3, 3)).astype("int64")
                ),
            ),
            True,
            False,
            UserWarning,
        ),
    ],
)
def test_SVD(x, full_matrices, compute_uv, exc):
    g = nlinalg.SVD(full_matrices, compute_uv)(x)

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
