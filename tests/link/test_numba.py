import contextlib
from unittest import mock

import numpy as np
import pytest

import aesara.scalar as aes
import aesara.scalar.basic as aesb
import aesara.tensor as aet
import aesara.tensor.basic as aetb
from aesara import config
from aesara.compile.function import function
from aesara.compile.mode import Mode
from aesara.compile.ops import ViewOp
from aesara.compile.sharedvalue import SharedVariable
from aesara.graph.basic import Constant
from aesara.graph.fg import FunctionGraph
from aesara.graph.optdb import Query
from aesara.link.numba.linker import NumbaLinker
from aesara.scalar.basic import Composite
from aesara.tensor import elemwise as aet_elemwise
from aesara.tensor import subtensor as aet_subtensor
from aesara.tensor.elemwise import Elemwise
from aesara.tensor.shape import Reshape, Shape, Shape_i, SpecifyShape


opts = Query(include=[None], exclude=["cxx_only", "BlasOpt"])
numba_mode = Mode(NumbaLinker(), opts)
py_mode = Mode("py", opts)


def set_test_value(x, v):
    x.tag.test_value = v
    return x


def compare_shape_dtype(x, y):
    (x,) = x
    (y,) = y
    return x.shape == y.shape and x.dtype == y.dtype


def compare_numba_and_py(
    fgraph,
    inputs,
    assert_fn=None,
):
    """Function to compare python graph output and Numba compiled output for testing equality

    In the tests below computational graphs are defined in Aesara. These graphs are then passed to
    this function which then compiles the graphs in both Numba and python, runs the calculation
    in both and checks if the results are the same

    Parameters
    ----------
    fgraph: FunctionGraph
        Aesara function Graph object
    inputs: iter
        Inputs for function graph
    assert_fn: func, opt
        Assert function used to check for equality between python and Numba. If not
        provided uses np.testing.assert_allclose

    """
    if assert_fn is None:

        def assert_fn(x, y):
            return np.testing.assert_allclose(x, y, rtol=1e-4) and compare_shape_dtype(
                x, y
            )

    fn_inputs = [i for i in fgraph.inputs if not isinstance(i, SharedVariable)]

    aesara_py_fn = function(
        fn_inputs, fgraph.outputs, mode=py_mode, accept_inplace=True
    )
    py_res = aesara_py_fn(*inputs)

    aesara_numba_fn = function(
        fn_inputs,
        fgraph.outputs,
        mode=numba_mode,
        accept_inplace=True,
    )
    numba_res = aesara_numba_fn(*inputs)

    # We evaluate the Numba implementation in pure Python for coverage
    # purposes.
    def py_tuple_setitem(t, i, v):
        l = list(t)
        l[i] = v
        return tuple(l)

    def py_to_scalar(x):
        if isinstance(x, np.ndarray):
            return x.item()
        else:
            return x

    with mock.patch("aesara.link.numba.dispatch.numba.njit", lambda x: x), mock.patch(
        "aesara.link.numba.dispatch.numba.vectorize", lambda x: x
    ), mock.patch(
        "aesara.link.numba.dispatch.tuple_setitem", py_tuple_setitem
    ), mock.patch(
        "aesara.link.numba.dispatch.direct_cast", lambda x, dtype: x
    ), mock.patch(
        "aesara.link.numba.dispatch.numba.np.numpy_support.from_dtype",
        lambda dtype: dtype,
    ), mock.patch(
        "aesara.link.numba.dispatch.to_scalar", py_to_scalar
    ):
        aesara_numba_fn = function(
            fn_inputs,
            fgraph.outputs,
            mode=numba_mode,
            accept_inplace=True,
        )
        _ = aesara_numba_fn(*inputs)

    if len(fgraph.outputs) > 1:
        for j, p in zip(numba_res, py_res):
            assert_fn(j, p)
    else:
        assert_fn(numba_res, py_res)

    return numba_res


@pytest.mark.parametrize(
    "inputs, input_vals, output_fn",
    [
        (
            [aet.vector()],
            [np.random.randn(100).astype(config.floatX)],
            lambda x: aet.nnet.sigmoid(x),
        ),
        (
            [aet.vector() for i in range(4)],
            [np.random.randn(100).astype(config.floatX) for i in range(4)],
            lambda x, y, x1, y1: (x + y) * (x1 + y1) * y,
        ),
    ],
)
def test_Elemwise(inputs, input_vals, output_fn):
    out_fg = FunctionGraph(inputs, [output_fn(*inputs)])
    compare_numba_and_py(out_fg, input_vals)


@pytest.mark.parametrize(
    "inputs, input_values",
    [
        (
            [aet.scalar("x"), aet.scalar("y")],
            [np.array(10).astype(config.floatX), np.array(20).astype(config.floatX)],
        ),
    ],
)
def test_numba_Composite(inputs, input_values):
    x_s = aes.float64("x")
    y_s = aes.float64("y")
    comp_op = Elemwise(Composite([x_s, y_s], [x_s + y_s * 2 + aes.exp(x_s - y_s)]))
    out_fg = FunctionGraph(inputs, [comp_op(*inputs)])
    compare_numba_and_py(out_fg, input_values)


@pytest.mark.parametrize(
    "x, indices",
    [
        (aet.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))), (1,)),
        (
            aet.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            (slice(None)),
        ),
        (aet.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))), (1, 2, 0)),
        (
            aet.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            (slice(1, 2), 1, slice(None)),
        ),
    ],
)
def test_Subtensor(x, indices):
    """Test NumPy's basic indexing."""
    out_aet = x[indices]
    assert isinstance(out_aet.owner.op, aet_subtensor.Subtensor)
    out_fg = FunctionGraph([], [out_aet])
    compare_numba_and_py(out_fg, [])


@pytest.mark.parametrize(
    "x, indices",
    [
        (aet.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))), ([1, 2],)),
        (
            aet.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            ([1, 2], slice(None)),
        ),
    ],
)
def test_AdvancedSubtensor1(x, indices):
    """Test NumPy's advanced indexing in one dimension."""
    out_aet = x[[1, 2]]
    assert isinstance(out_aet.owner.op, aet_subtensor.AdvancedSubtensor1)
    out_fg = FunctionGraph([], [out_aet])
    compare_numba_and_py(out_fg, [])


@pytest.mark.parametrize(
    "x, indices",
    [
        (aet.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))), ([1, 2], [2, 3])),
        (
            aet.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            ([1, 2], slice(None), [3, 4]),
        ),
    ],
)
def test_AdvancedSubtensor(x, indices):
    """Test NumPy's advanced indexing in more than one dimension."""
    out_aet = x[indices]
    assert isinstance(out_aet.owner.op, aet_subtensor.AdvancedSubtensor)
    out_fg = FunctionGraph([], [out_aet])
    compare_numba_and_py(out_fg, [])


@pytest.mark.parametrize(
    "x, y, indices",
    [
        (
            aet.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            aet.as_tensor(np.array(10)),
            (1,),
        ),
        (
            aet.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            aet.as_tensor(np.random.poisson(size=(4, 5))),
            (slice(None)),
        ),
        (
            aet.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            aet.as_tensor(np.array(10)),
            (1, 2, 0),
        ),
        (
            aet.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            aet.as_tensor(np.random.poisson(size=(1, 5))),
            (slice(1, 2), 1, slice(None)),
        ),
    ],
)
def test_IncSubtensor(x, y, indices):
    out_aet = aet.set_subtensor(x[indices], y)
    assert isinstance(out_aet.owner.op, aet_subtensor.IncSubtensor)
    out_fg = FunctionGraph([], [out_aet])
    compare_numba_and_py(out_fg, [])

    out_aet = aet.inc_subtensor(x[indices], y)
    assert isinstance(out_aet.owner.op, aet_subtensor.IncSubtensor)
    out_fg = FunctionGraph([], [out_aet])
    compare_numba_and_py(out_fg, [])

    x_at = x.type()
    out_aet = aet.set_subtensor(x_at[indices], y, inplace=True)
    assert isinstance(out_aet.owner.op, aet_subtensor.IncSubtensor)
    out_fg = FunctionGraph([x_at], [out_aet])
    compare_numba_and_py(out_fg, [x.data])


@pytest.mark.parametrize(
    "x, y, indices",
    [
        (
            aet.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            aet.as_tensor(np.random.poisson(size=(2, 4, 5))),
            ([1, 2],),
        ),
        (
            aet.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            aet.as_tensor(np.random.poisson(size=(2, 4, 5))),
            ([1, 2], slice(None)),
        ),
    ],
)
def test_AdvancedIncSubtensor1(x, y, indices):
    out_aet = aet.set_subtensor(x[indices], y)
    assert isinstance(out_aet.owner.op, aet_subtensor.AdvancedIncSubtensor1)
    out_fg = FunctionGraph([], [out_aet])
    compare_numba_and_py(out_fg, [])

    out_aet = aet.inc_subtensor(x[indices], y)
    assert isinstance(out_aet.owner.op, aet_subtensor.AdvancedIncSubtensor1)
    out_fg = FunctionGraph([], [out_aet])
    compare_numba_and_py(out_fg, [])

    x_at = x.type()
    out_aet = aet.set_subtensor(x_at[indices], y, inplace=True)
    assert isinstance(out_aet.owner.op, aet_subtensor.AdvancedIncSubtensor1)
    out_fg = FunctionGraph([x_at], [out_aet])
    compare_numba_and_py(out_fg, [x.data])


@pytest.mark.parametrize(
    "x, y, indices",
    [
        (
            aet.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            aet.as_tensor(np.random.poisson(size=(2, 5))),
            ([1, 2], [2, 3]),
        ),
        (
            aet.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            aet.as_tensor(np.random.poisson(size=(2, 4))),
            ([1, 2], slice(None), [3, 4]),
        ),
    ],
)
def test_AdvancedIncSubtensor(x, y, indices):
    out_aet = aet.set_subtensor(x[indices], y)
    assert isinstance(out_aet.owner.op, aet_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([], [out_aet])
    compare_numba_and_py(out_fg, [])

    out_aet = aet.inc_subtensor(x[indices], y)
    assert isinstance(out_aet.owner.op, aet_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([], [out_aet])
    compare_numba_and_py(out_fg, [])

    x_at = x.type()
    out_aet = aet.set_subtensor(x_at[indices], y)
    # Inplace isn't really implemented for `AdvancedIncSubtensor`, so we just
    # hack it on here
    out_aet.owner.op.inplace = True
    assert isinstance(out_aet.owner.op, aet_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([x_at], [out_aet])
    compare_numba_and_py(out_fg, [x.data])


@pytest.mark.parametrize(
    "x, i",
    [
        (np.zeros((20, 3)), 1),
    ],
)
def test_Shape(x, i):
    g = Shape()(aet.as_tensor_variable(x))
    g_fg = FunctionGraph([], [g])

    compare_numba_and_py(g_fg, [])

    g = Shape_i(i)(aet.as_tensor_variable(x))
    g_fg = FunctionGraph([], [g])

    compare_numba_and_py(g_fg, [])


@pytest.mark.parametrize(
    "v, shape",
    [
        (0.0, (2, 3)),
        (1.1, (2, 3)),
        (set_test_value(aet.scalar("a"), np.array(10.0, dtype=config.floatX)), (20,)),
        (set_test_value(aet.vector("a"), np.ones(10, dtype=config.floatX)), (20, 10)),
    ],
)
def test_Alloc(v, shape):
    g = aet.alloc(v, *shape)
    g_fg = FunctionGraph(outputs=[g])

    (numba_res,) = compare_numba_and_py(
        g_fg,
        [
            i.tag.test_value
            for i in g_fg.inputs
            if not isinstance(i, (SharedVariable, Constant))
        ],
    )

    assert numba_res.shape == shape


def test_AllocEmpty():

    x = aet.empty((2, 3), dtype="float32")
    x_fg = FunctionGraph([], [x])

    # We need cannot compare the values in the arrays, only the shapes and
    # dtypes
    compare_numba_and_py(x_fg, [], assert_fn=compare_shape_dtype)


@pytest.mark.parametrize(
    "v, offset",
    [
        (set_test_value(aet.vector(), np.arange(10, dtype=config.floatX)), 0),
        (set_test_value(aet.vector(), np.arange(10, dtype=config.floatX)), 1),
        (set_test_value(aet.vector(), np.arange(10, dtype=config.floatX)), -1),
    ],
)
def test_AllocDiag(v, offset):
    g = aetb.AllocDiag(offset=offset)(v)
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
    "v, new_order, inplace",
    [
        # `{'drop': [], 'shuffle': [], 'augment': [0, 1]}`
        (
            set_test_value(
                aet.lscalar(name="a"),
                np.array(1, dtype=np.int64),
            ),
            ("x", "x"),
            True,
        ),
        # I.e. `a_aet.T`
        # `{'drop': [], 'shuffle': [1, 0], 'augment': []}`
        (
            set_test_value(
                aet.matrix("a"), np.array([[1.0, 2.0], [3.0, 4.0]], dtype=config.floatX)
            ),
            (1, 0),
            True,
        ),
        # `{'drop': [], 'shuffle': [0, 1], 'augment': [2]}`
        (
            set_test_value(
                aet.matrix("a"), np.array([[1.0, 2.0], [3.0, 4.0]], dtype=config.floatX)
            ),
            (1, 0, "x"),
            True,
        ),
        # `{'drop': [1], 'shuffle': [2, 0], 'augment': [0, 2, 4]}`
        (
            set_test_value(
                aet.tensor(config.floatX, [False, True, False], name="a"),
                np.array([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=config.floatX),
            ),
            ("x", 2, "x", 0, "x"),
            True,
        ),
        # I.e. `a_aet.dimshuffle((0,))`
        # `{'drop': [1], 'shuffle': [0], 'augment': []}`
        (
            set_test_value(
                aet.tensor(config.floatX, [False, True], name="a"),
                np.array([[1.0], [2.0], [3.0], [4.0]], dtype=config.floatX),
            ),
            (0,),
            True,
        ),
        (
            set_test_value(
                aet.tensor(config.floatX, [False, True], name="a"),
                np.array([[1.0], [2.0], [3.0], [4.0]], dtype=config.floatX),
            ),
            (0,),
            True,
        ),
    ],
)
def test_Dimshuffle(v, new_order, inplace):
    g = aet_elemwise.DimShuffle(v.broadcastable, new_order, inplace=inplace)(v)
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
    g = aetb.TensorFromScalar()(v)
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
        set_test_value(aet.scalar(), np.array(1.0, dtype=config.floatX)),
    ],
)
def test_ScalarFromTensor(v):
    g = aetb.ScalarFromTensor()(v)
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
    "v, axis, fails",
    [
        (
            set_test_value(aet.matrix(), np.array([[1.0]], dtype=config.floatX)),
            [(0, True), (1, True)],
            False,
        ),
        (
            set_test_value(aet.matrix(), np.array([[1.0, 2.0]], dtype=config.floatX)),
            [(0, True), (1, False)],
            False,
        ),
        (
            set_test_value(aet.matrix(), np.array([[1.0, 2.0]], dtype=config.floatX)),
            [(0, True), (1, True)],
            True,
        ),
    ],
)
def test_Rebroadcast(v, axis, fails):
    g = aetb.Rebroadcast(*axis)(v)
    g_fg = FunctionGraph(outputs=[g])
    cm = contextlib.suppress() if not fails else pytest.raises(ValueError)
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
    "v, dtype",
    [
        (set_test_value(aet.fscalar(), np.array(1.0, dtype="float32")), aesb.float64),
        (set_test_value(aet.dscalar(), np.array(1.0, dtype="float64")), aesb.float32),
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
    "v, shape, ndim",
    [
        (set_test_value(aet.vector(), np.arange(4, dtype=config.floatX)), (2, 2), 2),
        (
            set_test_value(aet.vector(), np.arange(4, dtype=config.floatX)),
            set_test_value(aet.lvector(), np.array([2, 2], dtype="int64")),
            2,
        ),
    ],
)
def test_Reshape(v, shape, ndim):
    g = Reshape(ndim)(v, shape)
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
    "v, shape, fails",
    [
        (
            set_test_value(aet.matrix(), np.array([[1.0]], dtype=config.floatX)),
            (1, 1),
            False,
        ),
        (
            set_test_value(aet.matrix(), np.array([[1.0, 2.0]], dtype=config.floatX)),
            (1, 1),
            True,
        ),
    ],
)
def test_SpecifyShape(v, shape, fails):
    g = SpecifyShape()(v, shape)
    g_fg = FunctionGraph(outputs=[g])
    cm = contextlib.suppress() if not fails else pytest.raises(AssertionError)
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
    "v",
    [
        set_test_value(aet.vector(), np.arange(4, dtype=config.floatX)),
    ],
)
def test_ViewOp(v):
    g = ViewOp()(v)
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
    "x, y",
    [
        (
            set_test_value(aet.lvector(), np.arange(4, dtype="int64")),
            set_test_value(aet.dvector(), np.arange(4, dtype="float64")),
        ),
        (
            set_test_value(
                aet.dmatrix(), np.arange(4, dtype="float64").reshape((2, 2))
            ),
            set_test_value(aet.lscalar(), np.array(4, dtype="int64")),
        ),
    ],
)
def test_Second(x, y):
    # We use the `Elemwise`-wrapped version of `Second`
    g = aet.second(x, y)
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
        (set_test_value(aet.scalar(), np.array(10, dtype=config.floatX)), 3.0, 7.0),
        (set_test_value(aet.scalar(), np.array(1, dtype=config.floatX)), 3.0, 7.0),
        (set_test_value(aet.scalar(), np.array(10, dtype=config.floatX)), 7.0, 3.0),
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
    "vals, dtype",
    [
        (
            (
                set_test_value(aet.scalar(), np.array(1, dtype=config.floatX)),
                set_test_value(aet.scalar(), np.array(2, dtype=config.floatX)),
                set_test_value(aet.scalar(), np.array(3, dtype=config.floatX)),
            ),
            config.floatX,
        ),
    ],
)
def test_MakeVector(vals, dtype):
    g = aetb.MakeVector(dtype)(*vals)
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
            set_test_value(aet.lscalar(), np.array(1)),
            set_test_value(aet.lscalar(), np.array(10)),
            set_test_value(aet.lscalar(), np.array(3)),
            config.floatX,
        ),
    ],
)
def test_ARange(start, stop, step, dtype):
    g = aetb.ARange(dtype)(start, stop, step)
    g_fg = FunctionGraph(outputs=[g])

    compare_numba_and_py(
        g_fg,
        [
            i.tag.test_value
            for i in g_fg.inputs
            if not isinstance(i, (SharedVariable, Constant))
        ],
    )
