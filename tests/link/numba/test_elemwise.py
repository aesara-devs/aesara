import contextlib

import numpy as np
import pytest
import scipy.special

import aesara
import aesara.tensor as at
import aesara.tensor.inplace as ati
import aesara.tensor.math as aem
from aesara import config
from aesara.compile.ops import deep_copy_op
from aesara.compile.sharedvalue import SharedVariable
from aesara.graph.basic import Constant
from aesara.graph.fg import FunctionGraph
from aesara.tensor import elemwise as at_elemwise
from aesara.tensor.math import All, Any, Max, Mean, Min, Prod, ProdWithoutZeros, Sum
from aesara.tensor.special import LogSoftmax, Softmax, SoftmaxGrad
from tests.link.numba.test_basic import (
    compare_numba_and_py,
    my_multi_out,
    set_test_value,
)


rng = np.random.default_rng(42849)


@pytest.mark.parametrize(
    "inputs, input_vals, output_fn, exc",
    [
        (
            [at.vector()],
            [rng.uniform(size=100).astype(config.floatX)],
            lambda x: at.gammaln(x),
            None,
        ),
        (
            [at.vector()],
            [rng.standard_normal(100).astype(config.floatX)],
            lambda x: at.sigmoid(x),
            None,
        ),
        (
            [at.vector()],
            [rng.standard_normal(100).astype(config.floatX)],
            lambda x: at.log1mexp(x),
            None,
        ),
        (
            [at.vector()],
            [rng.standard_normal(100).astype(config.floatX)],
            lambda x: at.erf(x),
            None,
        ),
        (
            [at.vector()],
            [rng.standard_normal(100).astype(config.floatX)],
            lambda x: at.erfc(x),
            None,
        ),
        (
            [at.vector() for i in range(4)],
            [rng.standard_normal(100).astype(config.floatX) for i in range(4)],
            lambda x, y, x1, y1: (x + y) * (x1 + y1) * y,
            None,
        ),
        (
            [at.matrix(), at.scalar()],
            [rng.normal(size=(2, 2)).astype(config.floatX), 0.0],
            lambda a, b: at.switch(a, b, a),
            None,
        ),
        (
            [at.scalar(), at.scalar()],
            [
                np.array(1.0, dtype=config.floatX),
                np.array(1.0, dtype=config.floatX),
            ],
            lambda x, y: ati.add_inplace(deep_copy_op(x), deep_copy_op(y)),
            None,
        ),
        (
            [at.vector(), at.vector()],
            [
                rng.standard_normal(100).astype(config.floatX),
                rng.standard_normal(100).astype(config.floatX),
            ],
            lambda x, y: ati.add_inplace(deep_copy_op(x), deep_copy_op(y)),
            None,
        ),
        (
            [at.vector(), at.vector()],
            [
                rng.standard_normal(100).astype(config.floatX),
                rng.standard_normal(100).astype(config.floatX),
            ],
            lambda x, y: my_multi_out(x, y),
            NotImplementedError,
        ),
    ],
)
def test_Elemwise(inputs, input_vals, output_fn, exc):

    outputs = output_fn(*inputs)

    out_fg = FunctionGraph(
        outputs=[outputs] if not isinstance(outputs, list) else outputs
    )

    cm = contextlib.suppress() if exc is None else pytest.raises(exc)
    with cm:
        compare_numba_and_py(out_fg, input_vals)


@pytest.mark.parametrize(
    "v, new_order",
    [
        # `{'drop': [], 'shuffle': [], 'augment': [0, 1]}`
        (
            set_test_value(
                at.lscalar(name="a"),
                np.array(1, dtype=np.int64),
            ),
            ("x", "x"),
        ),
        # I.e. `a_at.T`
        # `{'drop': [], 'shuffle': [1, 0], 'augment': []}`
        (
            set_test_value(
                at.matrix("a"), np.array([[1.0, 2.0], [3.0, 4.0]], dtype=config.floatX)
            ),
            (1, 0),
        ),
        # `{'drop': [], 'shuffle': [0, 1], 'augment': [2]}`
        (
            set_test_value(
                at.matrix("a"), np.array([[1.0, 2.0], [3.0, 4.0]], dtype=config.floatX)
            ),
            (1, 0, "x"),
        ),
        # `{'drop': [1], 'shuffle': [2, 0], 'augment': [0, 2, 4]}`
        (
            set_test_value(
                at.tensor(config.floatX, shape=(None, 1, None), name="a"),
                np.array([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=config.floatX),
            ),
            ("x", 2, "x", 0, "x"),
        ),
        # I.e. `a_at.dimshuffle((0,))`
        # `{'drop': [1], 'shuffle': [0], 'augment': []}`
        (
            set_test_value(
                at.tensor(config.floatX, shape=(None, 1), name="a"),
                np.array([[1.0], [2.0], [3.0], [4.0]], dtype=config.floatX),
            ),
            (0,),
        ),
        (
            set_test_value(
                at.tensor(config.floatX, shape=(None, 1), name="a"),
                np.array([[1.0], [2.0], [3.0], [4.0]], dtype=config.floatX),
            ),
            (0,),
        ),
        (
            set_test_value(
                at.tensor(config.floatX, shape=(1, 1, 1), name="a"),
                np.array([[[1.0]]], dtype=config.floatX),
            ),
            (),
        ),
    ],
)
def test_Dimshuffle(v, new_order):
    g = at_elemwise.DimShuffle(v.broadcastable, new_order)(v)
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
    "careduce_fn, axis, v",
    [
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Sum(
                axis=axis, dtype=dtype, acc_dtype=acc_dtype
            )(x),
            0,
            set_test_value(at.vector(), np.arange(3, dtype=config.floatX)),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: All(axis)(x),
            0,
            set_test_value(at.vector(), np.arange(3, dtype=config.floatX)),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Any(axis)(x),
            0,
            set_test_value(at.vector(), np.arange(3, dtype=config.floatX)),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Mean(axis)(x),
            0,
            set_test_value(at.vector(), np.arange(3, dtype=config.floatX)),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Mean(axis)(x),
            0,
            set_test_value(
                at.matrix(), np.arange(3 * 2, dtype=config.floatX).reshape((3, 2))
            ),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Sum(
                axis=axis, dtype=dtype, acc_dtype=acc_dtype
            )(x),
            0,
            set_test_value(
                at.matrix(), np.arange(3 * 2, dtype=config.floatX).reshape((3, 2))
            ),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Sum(
                axis=axis, dtype=dtype, acc_dtype=acc_dtype
            )(x),
            (0, 1),
            set_test_value(
                at.matrix(), np.arange(3 * 2, dtype=config.floatX).reshape((3, 2))
            ),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Sum(
                axis=axis, dtype=dtype, acc_dtype=acc_dtype
            )(x),
            (1, 0),
            set_test_value(
                at.matrix(), np.arange(3 * 2, dtype=config.floatX).reshape((3, 2))
            ),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Sum(
                axis=axis, dtype=dtype, acc_dtype=acc_dtype
            )(x),
            None,
            set_test_value(
                at.matrix(), np.arange(3 * 2, dtype=config.floatX).reshape((3, 2))
            ),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Sum(
                axis=axis, dtype=dtype, acc_dtype=acc_dtype
            )(x),
            1,
            set_test_value(
                at.matrix(), np.arange(3 * 2, dtype=config.floatX).reshape((3, 2))
            ),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Prod(
                axis=axis, dtype=dtype, acc_dtype=acc_dtype
            )(x),
            0,
            set_test_value(at.vector(), np.arange(3, dtype=config.floatX)),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: ProdWithoutZeros(
                axis=axis, dtype=dtype, acc_dtype=acc_dtype
            )(x),
            0,
            set_test_value(at.vector(), np.arange(3, dtype=config.floatX)),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Prod(
                axis=axis, dtype=dtype, acc_dtype=acc_dtype
            )(x),
            0,
            set_test_value(
                at.matrix(), np.arange(3 * 2, dtype=config.floatX).reshape((3, 2))
            ),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Prod(
                axis=axis, dtype=dtype, acc_dtype=acc_dtype
            )(x),
            1,
            set_test_value(
                at.matrix(), np.arange(3 * 2, dtype=config.floatX).reshape((3, 2))
            ),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Max(axis)(x),
            None,
            set_test_value(
                at.matrix(), np.arange(3 * 2, dtype=config.floatX).reshape((3, 2))
            ),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Max(axis)(x),
            None,
            set_test_value(
                at.lmatrix(), np.arange(3 * 2, dtype=np.int64).reshape((3, 2))
            ),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Min(axis)(x),
            None,
            set_test_value(
                at.matrix(), np.arange(3 * 2, dtype=config.floatX).reshape((3, 2))
            ),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Min(axis)(x),
            None,
            set_test_value(
                at.lmatrix(), np.arange(3 * 2, dtype=np.int64).reshape((3, 2))
            ),
        ),
    ],
)
def test_CAReduce(careduce_fn, axis, v):
    g = careduce_fn(v, axis=axis)
    g_fg = FunctionGraph(outputs=[g])

    compare_numba_and_py(
        g_fg,
        [
            i.tag.test_value
            for i in g_fg.inputs
            if not isinstance(i, (SharedVariable, Constant))
        ],
    )


def test_scalar_Elemwise_Clip():
    a = at.scalar("a")
    b = at.scalar("b")

    z = at.switch(1, a, b)
    c = at.clip(z, 1, 3)
    c_fg = FunctionGraph(outputs=[c])

    compare_numba_and_py(c_fg, [1, 1])


@pytest.mark.parametrize(
    "dy, sm, axis, exc",
    [
        (
            set_test_value(
                at.matrix(), np.array([[1, 1, 1], [0, 0, 0]], dtype=config.floatX)
            ),
            set_test_value(at.matrix(), rng.random(size=(2, 3)).astype(config.floatX)),
            None,
            None,
        ),
        (
            set_test_value(
                at.matrix(), np.array([[1, 1, 1], [0, 0, 0]], dtype=config.floatX)
            ),
            set_test_value(at.matrix(), rng.random(size=(2, 3)).astype(config.floatX)),
            0,
            None,
        ),
        (
            set_test_value(
                at.matrix(), np.array([[1, 1, 1], [0, 0, 0]], dtype=config.floatX)
            ),
            set_test_value(at.matrix(), rng.random(size=(2, 3)).astype(config.floatX)),
            1,
            None,
        ),
    ],
)
def test_SoftmaxGrad(dy, sm, axis, exc):
    g = SoftmaxGrad(axis=axis)(dy, sm)
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
    "x, axis, exc",
    [
        (
            set_test_value(at.vector(), rng.random(size=(2,)).astype(config.floatX)),
            None,
            None,
        ),
        (
            set_test_value(at.matrix(), rng.random(size=(2, 3)).astype(config.floatX)),
            None,
            None,
        ),
        (
            set_test_value(at.matrix(), rng.random(size=(2, 3)).astype(config.floatX)),
            0,
            None,
        ),
    ],
)
def test_Softmax(x, axis, exc):
    g = Softmax(axis=axis)(x)
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
    "x, axis, exc",
    [
        (
            set_test_value(at.vector(), rng.random(size=(2,)).astype(config.floatX)),
            None,
            None,
        ),
        (
            set_test_value(at.matrix(), rng.random(size=(2, 3)).astype(config.floatX)),
            0,
            None,
        ),
        (
            set_test_value(at.matrix(), rng.random(size=(2, 3)).astype(config.floatX)),
            1,
            None,
        ),
    ],
)
def test_LogSoftmax(x, axis, exc):
    g = LogSoftmax(axis=axis)(x)
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
    "x, axes, exc",
    [
        (
            set_test_value(at.dscalar(), np.array(0.0, dtype="float64")),
            [],
            None,
        ),
        (
            set_test_value(at.dvector(), rng.random(size=(3,)).astype("float64")),
            [0],
            None,
        ),
        (
            set_test_value(at.dmatrix(), rng.random(size=(3, 2)).astype("float64")),
            [0],
            None,
        ),
        (
            set_test_value(at.dmatrix(), rng.random(size=(3, 2)).astype("float64")),
            [0, 1],
            None,
        ),
    ],
)
def test_MaxAndArgmax(x, axes, exc):
    g = aem.MaxAndArgmax(axes)(x)

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


def test_sum_broadcast_to():
    """Make sure that we handle the writability of `BroadcastTo` results correctly."""

    x = at.vector("x")
    out = at.broadcast_to(x, (2, 2)).sum()

    x_val = np.array([1, 2], dtype=config.floatX)
    compare_numba_and_py(((x,), (out,)), [x_val])


@pytest.mark.parametrize("size", [(10, 10), (1000, 1000), (10000, 10000)])
@pytest.mark.parametrize("axis", [0, 1])
def test_logsumexp_benchmark(size, axis, benchmark):
    X = at.matrix("X")
    X_max = at.max(X, axis=axis, keepdims=True)
    X_max = at.switch(at.isinf(X_max), 0, X_max)
    X_lse = at.log(at.sum(at.exp(X - X_max), axis=axis, keepdims=True)) + X_max

    X_val = np.random.normal(size=size)

    X_lse_fn = aesara.function([X], X_lse, mode="NUMBA")

    # JIT compile first
    _ = X_lse_fn(X_val)

    res = benchmark(X_lse_fn, X_val)

    exp_res = scipy.special.logsumexp(X_val, axis=axis, keepdims=True)
    np.testing.assert_array_almost_equal(res, exp_res)
