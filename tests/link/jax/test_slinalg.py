import numpy as np
import pytest

import aesara.tensor as at
from aesara.configdefaults import config
from aesara.graph.fg import FunctionGraph
from aesara.tensor import nlinalg as at_nlinalg
from aesara.tensor import slinalg as at_slinalg
from aesara.tensor import subtensor as at_subtensor
from aesara.tensor.math import clip, cosh
from aesara.tensor.type import matrix, vector
from tests.link.jax.test_basic import compare_jax_and_py


def test_jax_basic():
    rng = np.random.default_rng(28494)

    x = matrix("x")
    y = matrix("y")
    b = vector("b")

    # `ScalarOp`
    z = cosh(x**2 + y / 3.0)

    # `[Inc]Subtensor`
    out = at_subtensor.set_subtensor(z[0], -10.0)
    out = at_subtensor.inc_subtensor(out[0, 1], 2.0)
    out = out[:5, :3]

    out_fg = FunctionGraph([x, y], [out])

    test_input_vals = [
        np.tile(np.arange(10), (10, 1)).astype(config.floatX),
        np.tile(np.arange(10, 20), (10, 1)).astype(config.floatX),
    ]
    (jax_res,) = compare_jax_and_py(out_fg, test_input_vals)

    # Confirm that the `Subtensor` slice operations are correct
    assert jax_res.shape == (5, 3)

    # Confirm that the `IncSubtensor` operations are correct
    assert jax_res[0, 0] == -10.0
    assert jax_res[0, 1] == -8.0

    out = clip(x, y, 5)
    out_fg = FunctionGraph([x, y], [out])
    compare_jax_and_py(out_fg, test_input_vals)

    out = at.diagonal(x, 0)
    out_fg = FunctionGraph([x], [out])
    compare_jax_and_py(
        out_fg, [np.arange(10 * 10).reshape((10, 10)).astype(config.floatX)]
    )

    out = at_slinalg.cholesky(x)
    out_fg = FunctionGraph([x], [out])
    compare_jax_and_py(
        out_fg,
        [
            (np.eye(10) + rng.standard_normal(size=(10, 10)) * 0.01).astype(
                config.floatX
            )
        ],
    )

    # not sure why this isn't working yet with lower=False
    out = at_slinalg.Cholesky(lower=False)(x)
    out_fg = FunctionGraph([x], [out])
    compare_jax_and_py(
        out_fg,
        [
            (np.eye(10) + rng.standard_normal(size=(10, 10)) * 0.01).astype(
                config.floatX
            )
        ],
    )

    out = at_slinalg.solve(x, b)
    out_fg = FunctionGraph([x, b], [out])
    compare_jax_and_py(
        out_fg,
        [
            np.eye(10).astype(config.floatX),
            np.arange(10).astype(config.floatX),
        ],
    )

    out = at.diag(b)
    out_fg = FunctionGraph([b], [out])
    compare_jax_and_py(out_fg, [np.arange(10).astype(config.floatX)])

    out = at_nlinalg.det(x)
    out_fg = FunctionGraph([x], [out])
    compare_jax_and_py(
        out_fg, [np.arange(10 * 10).reshape((10, 10)).astype(config.floatX)]
    )

    out = at_nlinalg.matrix_inverse(x)
    out_fg = FunctionGraph([x], [out])
    compare_jax_and_py(
        out_fg,
        [
            (np.eye(10) + rng.standard_normal(size=(10, 10)) * 0.01).astype(
                config.floatX
            )
        ],
    )


@pytest.mark.parametrize("check_finite", [False, True])
@pytest.mark.parametrize("lower", [False, True])
@pytest.mark.parametrize("trans", [0, 1, 2])
def test_jax_SolveTriangular(trans, lower, check_finite):
    x = matrix("x")
    b = vector("b")

    out = at_slinalg.solve_triangular(
        x,
        b,
        trans=trans,
        lower=lower,
        check_finite=check_finite,
    )
    out_fg = FunctionGraph([x, b], [out])
    compare_jax_and_py(
        out_fg,
        [
            np.eye(10).astype(config.floatX),
            np.arange(10).astype(config.floatX),
        ],
    )
