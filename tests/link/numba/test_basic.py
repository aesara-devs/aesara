import contextlib
import inspect
from unittest import mock

import numba
import numpy as np
import pytest

import aesara.scalar as aes
import aesara.scalar.math as aesm
import aesara.tensor as at
import aesara.tensor.basic as atb
import aesara.tensor.math as aem
from aesara import config, shared
from aesara.compile.function import function
from aesara.compile.mode import Mode
from aesara.compile.ops import ViewOp
from aesara.compile.sharedvalue import SharedVariable
from aesara.graph.basic import Apply, Constant
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import Op, get_test_value
from aesara.graph.rewriting.db import RewriteDatabaseQuery
from aesara.graph.type import Type
from aesara.ifelse import ifelse
from aesara.link.numba.dispatch import basic as numba_basic
from aesara.link.numba.dispatch import numba_typify
from aesara.link.numba.linker import NumbaLinker
from aesara.raise_op import assert_op
from aesara.tensor import blas
from aesara.tensor import subtensor as at_subtensor
from aesara.tensor.elemwise import Elemwise
from aesara.tensor.shape import Reshape, Shape, Shape_i, SpecifyShape, Unbroadcast


class MyType(Type):
    def filter(self, data):
        return data

    def __eq__(self, other):
        return isinstance(other, MyType)

    def __hash__(self):
        return hash(MyType)


class MyOp(Op):
    def perform(self, *args):
        pass


class MySingleOut(Op):
    def make_node(self, a, b):
        return Apply(self, [a, b], [a.type()])

    def perform(self, node, inputs, outputs):
        res = (inputs[0] + inputs[1]).astype(inputs[0][0].dtype)
        outputs[0][0] = res


class MyMultiOut(Op):
    nin = 2
    nout = 2

    @staticmethod
    def impl(a, b):
        res1 = 2 * a
        res2 = 2 * b
        return [res1, res2]

    def make_node(self, a, b):
        return Apply(self, [a, b], [a.type(), b.type()])

    def perform(self, node, inputs, outputs):
        res1, res2 = self.impl(inputs[0], inputs[1])
        outputs[0][0] = res1
        outputs[1][0] = res2


my_multi_out = Elemwise(MyMultiOut())
my_multi_out.ufunc = MyMultiOut.impl
my_multi_out.ufunc.nin = 2
my_multi_out.ufunc.nout = 2

opts = RewriteDatabaseQuery(include=[None], exclude=["cxx_only", "BlasOpt"])
numba_mode = Mode(NumbaLinker(), opts)
py_mode = Mode("py", opts)

rng = np.random.default_rng(42849)


def set_test_value(x, v):
    x.tag.test_value = v
    return x


def compare_shape_dtype(x, y):
    (x,) = x
    (y,) = y
    return x.shape == y.shape and x.dtype == y.dtype


def eval_python_only(fn_inputs, fgraph, inputs):
    """Evaluate the Numba implementation in pure Python for coverage purposes."""

    def py_tuple_setitem(t, i, v):
        ll = list(t)
        ll[i] = v
        return tuple(ll)

    def py_to_scalar(x):
        if isinstance(x, np.ndarray):
            return x.item()
        else:
            return x

    def njit_noop(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        else:
            return lambda x: x

    def vectorize_noop(*args, **kwargs):
        def wrap(fn):
            # `numba.vectorize` allows an `out` positional argument.  We need
            # to account for that
            sig = inspect.signature(fn)
            nparams = len(sig.parameters)

            def inner_vec(*args):
                if len(args) > nparams:
                    # An `out` argument has been specified for an in-place
                    # operation
                    out = args[-1]
                    out[...] = np.vectorize(fn)(*args[:nparams])
                    return out
                else:
                    return np.vectorize(fn)(*args)

            return inner_vec

        if len(args) == 1 and callable(args[0]):
            return wrap(args[0], **kwargs)
        else:
            return wrap

    mocks = [
        mock.patch("numba.njit", njit_noop),
        mock.patch("numba.vectorize", vectorize_noop),
        mock.patch("aesara.link.numba.dispatch.basic.tuple_setitem", py_tuple_setitem),
        mock.patch("aesara.link.numba.dispatch.basic.numba_njit", njit_noop),
        mock.patch("aesara.link.numba.dispatch.basic.numba_vectorize", vectorize_noop),
        mock.patch("aesara.link.numba.dispatch.basic.direct_cast", lambda x, dtype: x),
        mock.patch("aesara.link.numba.dispatch.basic.to_scalar", py_to_scalar),
        mock.patch(
            "aesara.link.numba.dispatch.basic.numba.np.numpy_support.from_dtype",
            lambda dtype: dtype,
        ),
        mock.patch("numba.np.unsafe.ndarray.to_fixed_tuple", lambda x, n: tuple(x)),
    ]

    with contextlib.ExitStack() as stack:
        for ctx in mocks:
            stack.enter_context(ctx)

        aesara_numba_fn = function(
            fn_inputs,
            fgraph.outputs,
            mode=numba_mode,
            accept_inplace=True,
        )
        _ = aesara_numba_fn(*inputs)


def compare_numba_and_py(fgraph, inputs, assert_fn=None):
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

    # Get some coverage
    eval_python_only(fn_inputs, fgraph, inputs)

    if len(fgraph.outputs) > 1:
        for j, p in zip(numba_res, py_res):
            assert_fn(j, p)
    else:
        assert_fn(numba_res, py_res)

    return numba_res


@pytest.mark.parametrize(
    "v, expected, force_scalar, not_implemented",
    [
        (MyType(), None, False, True),
        (aes.float32, numba.types.float32, False, False),
        (at.fscalar, numba.types.Array(numba.types.float32, 0, "A"), False, False),
        (at.fscalar, numba.types.float32, True, False),
        (at.lvector, numba.types.int64[:], False, False),
        (at.dmatrix, numba.types.float64[:, :], False, False),
        (at.dmatrix, numba.types.float64, True, False),
    ],
)
def test_get_numba_type(v, expected, force_scalar, not_implemented):
    cm = (
        contextlib.suppress()
        if not not_implemented
        else pytest.raises(NotImplementedError)
    )
    with cm:
        res = numba_basic.get_numba_type(v, force_scalar=force_scalar)
        assert res == expected


@pytest.mark.parametrize(
    "v, expected, force_scalar",
    [
        (Apply(MyOp(), [], []), numba.types.void(), False),
        (Apply(MyOp(), [], []), numba.types.void(), True),
        (
            Apply(MyOp(), [at.lvector()], []),
            numba.types.void(numba.types.int64[:]),
            False,
        ),
        (Apply(MyOp(), [at.lvector()], []), numba.types.void(numba.types.int64), True),
        (
            Apply(MyOp(), [at.dmatrix(), aes.float32()], [at.dmatrix()]),
            numba.types.float64[:, :](numba.types.float64[:, :], numba.types.float32),
            False,
        ),
        (
            Apply(MyOp(), [at.dmatrix(), aes.float32()], [at.dmatrix()]),
            numba.types.float64(numba.types.float64, numba.types.float32),
            True,
        ),
        (
            Apply(MyOp(), [at.dmatrix(), aes.float32()], [at.dmatrix(), aes.int32()]),
            numba.types.Tuple([numba.types.float64[:, :], numba.types.int32])(
                numba.types.float64[:, :], numba.types.float32
            ),
            False,
        ),
        (
            Apply(MyOp(), [at.dmatrix(), aes.float32()], [at.dmatrix(), aes.int32()]),
            numba.types.Tuple([numba.types.float64, numba.types.int32])(
                numba.types.float64, numba.types.float32
            ),
            True,
        ),
    ],
)
def test_create_numba_signature(v, expected, force_scalar):
    res = numba_basic.create_numba_signature(v, force_scalar=force_scalar)
    assert res == expected


@pytest.mark.parametrize(
    "input, wrapper_fn, check_fn",
    [
        (
            np.random.RandomState(1),
            numba_typify,
            lambda x, y: np.all(x.get_state()[1] == y.get_state()[1]),
        )
    ],
)
def test_box_unbox(input, wrapper_fn, check_fn):
    input = wrapper_fn(input)

    pass_through = numba.njit(lambda x: x)
    res = pass_through(input)

    assert isinstance(res, type(input))
    assert check_fn(res, input)


@pytest.mark.parametrize(
    "x, indices",
    [
        (at.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))), (1,)),
        (
            at.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            (slice(None)),
        ),
        (at.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))), (1, 2, 0)),
        (
            at.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            (slice(1, 2), 1, slice(None)),
        ),
    ],
)
def test_Subtensor(x, indices):
    """Test NumPy's basic indexing."""
    out_at = x[indices]
    assert isinstance(out_at.owner.op, at_subtensor.Subtensor)
    out_fg = FunctionGraph([], [out_at])
    compare_numba_and_py(out_fg, [])


@pytest.mark.parametrize(
    "x, indices",
    [
        (at.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))), ([1, 2],)),
    ],
)
def test_AdvancedSubtensor1(x, indices):
    """Test NumPy's advanced indexing in one dimension."""
    out_at = at_subtensor.advanced_subtensor1(x, *indices)
    assert isinstance(out_at.owner.op, at_subtensor.AdvancedSubtensor1)
    out_fg = FunctionGraph([], [out_at])
    compare_numba_and_py(out_fg, [])


@pytest.mark.parametrize(
    "x, indices",
    [
        (at.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))), ([1, 2], [2, 3])),
        (
            at.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            ([1, 2], slice(None), [3, 4]),
        ),
    ],
)
def test_AdvancedSubtensor(x, indices):
    """Test NumPy's advanced indexing in more than one dimension."""
    out_at = x[indices]
    assert isinstance(out_at.owner.op, at_subtensor.AdvancedSubtensor)
    out_fg = FunctionGraph([], [out_at])
    compare_numba_and_py(out_fg, [])


@pytest.mark.parametrize(
    "x, y, indices",
    [
        (
            at.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            at.as_tensor(np.array(10)),
            (1,),
        ),
        (
            at.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            at.as_tensor(rng.poisson(size=(4, 5))),
            (slice(None)),
        ),
        (
            at.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            at.as_tensor(np.array(10)),
            (1, 2, 0),
        ),
        (
            at.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            at.as_tensor(rng.poisson(size=(1, 5))),
            (slice(1, 2), 1, slice(None)),
        ),
    ],
)
def test_IncSubtensor(x, y, indices):
    out_at = at.set_subtensor(x[indices], y)
    assert isinstance(out_at.owner.op, at_subtensor.IncSubtensor)
    out_fg = FunctionGraph([], [out_at])
    compare_numba_and_py(out_fg, [])

    out_at = at.inc_subtensor(x[indices], y)
    assert isinstance(out_at.owner.op, at_subtensor.IncSubtensor)
    out_fg = FunctionGraph([], [out_at])
    compare_numba_and_py(out_fg, [])

    x_at = x.type()
    out_at = at.set_subtensor(x_at[indices], y, inplace=True)
    assert isinstance(out_at.owner.op, at_subtensor.IncSubtensor)
    out_fg = FunctionGraph([x_at], [out_at])
    compare_numba_and_py(out_fg, [x.data])


@pytest.mark.parametrize(
    "x, y, indices",
    [
        (
            at.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            at.as_tensor(rng.poisson(size=(2, 4, 5))),
            ([1, 2],),
        ),
        (
            at.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            at.as_tensor(rng.poisson(size=(2, 4, 5))),
            ([1, 1],),
        ),
    ],
)
def test_AdvancedIncSubtensor1(x, y, indices):
    out_at = at_subtensor.advanced_set_subtensor1(x, y, *indices)
    assert isinstance(out_at.owner.op, at_subtensor.AdvancedIncSubtensor1)
    out_fg = FunctionGraph([], [out_at])
    compare_numba_and_py(out_fg, [])

    out_at = at_subtensor.advanced_inc_subtensor1(x, y, *indices)
    assert isinstance(out_at.owner.op, at_subtensor.AdvancedIncSubtensor1)
    out_fg = FunctionGraph([], [out_at])
    compare_numba_and_py(out_fg, [])

    x_at = x.type()
    out_at = at_subtensor.AdvancedIncSubtensor1(inplace=True)(x_at, y, *indices)
    assert isinstance(out_at.owner.op, at_subtensor.AdvancedIncSubtensor1)
    out_fg = FunctionGraph([x_at], [out_at])
    compare_numba_and_py(out_fg, [x.data])


@pytest.mark.parametrize(
    "x, y, indices",
    [
        (
            at.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            at.as_tensor(rng.poisson(size=(2, 5))),
            ([1, 2], [2, 3]),
        ),
        (
            at.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            at.as_tensor(rng.poisson(size=(2, 4))),
            ([1, 2], slice(None), [3, 4]),
        ),
        pytest.param(
            at.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            at.as_tensor(rng.poisson(size=(2, 5))),
            ([1, 1], [2, 2]),
            marks=pytest.mark.xfail(
                reason="Duplicate index handling hasn't been implemented, yet."
            ),
        ),
    ],
)
def test_AdvancedIncSubtensor(x, y, indices):
    out_at = at.set_subtensor(x[indices], y)
    assert isinstance(out_at.owner.op, at_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([], [out_at])
    compare_numba_and_py(out_fg, [])

    out_at = at.inc_subtensor(x[indices], y)
    assert isinstance(out_at.owner.op, at_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([], [out_at])
    compare_numba_and_py(out_fg, [])

    x_at = x.type()
    out_at = at.set_subtensor(x_at[indices], y)
    # Inplace isn't really implemented for `AdvancedIncSubtensor`, so we just
    # hack it on here
    out_at.owner.op.inplace = True
    assert isinstance(out_at.owner.op, at_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([x_at], [out_at])
    compare_numba_and_py(out_fg, [x.data])


@pytest.mark.parametrize(
    "x, i",
    [
        (np.zeros((20, 3)), 1),
    ],
)
def test_Shape(x, i):
    g = Shape()(at.as_tensor_variable(x))
    g_fg = FunctionGraph([], [g])

    compare_numba_and_py(g_fg, [])

    g = Shape_i(i)(at.as_tensor_variable(x))
    g_fg = FunctionGraph([], [g])

    compare_numba_and_py(g_fg, [])


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
    "v, shape, ndim",
    [
        (set_test_value(at.vector(), np.array([4], dtype=config.floatX)), (), 0),
        (set_test_value(at.vector(), np.arange(4, dtype=config.floatX)), (2, 2), 2),
        (
            set_test_value(at.vector(), np.arange(4, dtype=config.floatX)),
            set_test_value(at.lvector(), np.array([2, 2], dtype="int64")),
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


def test_Reshape_scalar():
    v = at.vector()
    v.tag.test_value = np.array([1.0], dtype=config.floatX)
    g = Reshape(1)(v[0], (1,))
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
            set_test_value(at.matrix(), np.array([[1.0]], dtype=config.floatX)),
            (1, 1),
            False,
        ),
        (
            set_test_value(at.matrix(), np.array([[1.0, 2.0]], dtype=config.floatX)),
            (1, 1),
            True,
        ),
        (
            set_test_value(at.matrix(), np.array([[1.0, 2.0]], dtype=config.floatX)),
            (1, None),
            False,
        ),
    ],
)
def test_SpecifyShape(v, shape, fails):
    g = SpecifyShape()(v, *shape)
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
        set_test_value(at.vector(), np.arange(4, dtype=config.floatX)),
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


@pytest.mark.parametrize(
    "inputs, op, exc",
    [
        (
            [
                set_test_value(
                    at.matrix(), rng.random(size=(2, 3)).astype(config.floatX)
                ),
                set_test_value(at.lmatrix(), rng.poisson(size=(2, 3))),
            ],
            MySingleOut,
            UserWarning,
        ),
        (
            [
                set_test_value(
                    at.matrix(), rng.random(size=(2, 3)).astype(config.floatX)
                ),
                set_test_value(at.lmatrix(), rng.poisson(size=(2, 3))),
            ],
            MyMultiOut,
            UserWarning,
        ),
    ],
)
def test_perform(inputs, op, exc):

    g = op()(*inputs)

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


def test_perform_params():
    """This tests for `Op.perform` implementations that require the `params` arguments."""

    x = at.vector()
    x.tag.test_value = np.array([1.0, 2.0], dtype=config.floatX)

    out = assert_op(x, np.array(True))

    if not isinstance(out, (list, tuple)):
        out = [out]

    out_fg = FunctionGraph([x], out)

    with pytest.warns(UserWarning, match=".*object mode.*"):
        compare_numba_and_py(out_fg, [get_test_value(i) for i in out_fg.inputs])


def test_perform_type_convert():
    """This tests the use of `Type.filter` in `objmode`.

    The `Op.perform` takes a single input that it returns as-is, but it gets a
    native scalar and it's supposed to return an `np.ndarray`.
    """

    x = at.vector()
    x.tag.test_value = np.array([1.0, 2.0], dtype=config.floatX)

    out = assert_op(x.sum(), np.array(True))

    if not isinstance(out, (list, tuple)):
        out = [out]

    out_fg = FunctionGraph([x], out)

    with pytest.warns(UserWarning, match=".*object mode.*"):
        compare_numba_and_py(out_fg, [get_test_value(i) for i in out_fg.inputs])


@pytest.mark.parametrize(
    "x, y, exc",
    [
        (
            set_test_value(at.matrix(), rng.random(size=(3, 2)).astype(config.floatX)),
            set_test_value(at.vector(), rng.random(size=(2,)).astype(config.floatX)),
            None,
        ),
        (
            set_test_value(
                at.matrix(dtype="float64"), rng.random(size=(3, 2)).astype("float64")
            ),
            set_test_value(
                at.vector(dtype="float32"), rng.random(size=(2,)).astype("float32")
            ),
            None,
        ),
        (
            set_test_value(at.lmatrix(), rng.poisson(size=(3, 2))),
            set_test_value(at.fvector(), rng.random(size=(2,)).astype("float32")),
            None,
        ),
        (
            set_test_value(at.lvector(), rng.random(size=(2,)).astype(np.int64)),
            set_test_value(at.lvector(), rng.random(size=(2,)).astype(np.int64)),
            None,
        ),
    ],
)
def test_Dot(x, y, exc):
    g = aem.Dot()(x, y)
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
            set_test_value(aes.float64(), np.array(0.0, dtype="float64")),
            None,
        ),
        (
            set_test_value(aes.float64(), np.array(-32.0, dtype="float64")),
            None,
        ),
        (
            set_test_value(aes.float64(), np.array(-40.0, dtype="float64")),
            None,
        ),
        (
            set_test_value(aes.float64(), np.array(32.0, dtype="float64")),
            None,
        ),
        (
            set_test_value(aes.float64(), np.array(40.0, dtype="float64")),
            None,
        ),
        (
            set_test_value(aes.int64(), np.array(32, dtype="int64")),
            None,
        ),
    ],
)
def test_Softplus(x, exc):
    g = aesm.Softplus(aes.upgrade_to_float)(x)
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
    "x, y, exc",
    [
        (
            set_test_value(
                at.dmatrix(),
                rng.random(size=(3, 3)).astype("float64"),
            ),
            set_test_value(
                at.dmatrix(),
                rng.random(size=(3, 3)).astype("float64"),
            ),
            None,
        ),
        (
            set_test_value(
                at.dmatrix(),
                rng.random(size=(3, 3)).astype("float64"),
            ),
            set_test_value(
                at.lmatrix(),
                rng.poisson(size=(3, 3)).astype("int64"),
            ),
            None,
        ),
    ],
)
def test_BatchedDot(x, y, exc):
    g = blas.BatchedDot()(x, y)

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


def test_shared():
    a = shared(np.array([1, 2, 3], dtype=config.floatX))

    aesara_numba_fn = function([], a, mode="NUMBA")
    numba_res = aesara_numba_fn()

    np.testing.assert_allclose(numba_res, a.get_value())

    aesara_numba_fn = function([], a * 2, mode="NUMBA")
    numba_res = aesara_numba_fn()

    np.testing.assert_allclose(numba_res, a.get_value() * 2)

    # Changed the shared value and make sure that the Numba-compiled function
    # also changes.
    new_a_value = np.array([3, 4, 5], dtype=config.floatX)
    a.set_value(new_a_value)

    numba_res = aesara_numba_fn()
    np.testing.assert_allclose(numba_res, new_a_value * 2)


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
    "inputs, cond_fn, true_vals, false_vals",
    [
        ([], lambda: np.array(True), np.r_[1, 2, 3], np.r_[-1, -2, -3]),
        (
            [set_test_value(at.dscalar(), np.array(0.2, dtype=np.float64))],
            lambda x: x < 0.5,
            np.r_[1, 2, 3],
            np.r_[-1, -2, -3],
        ),
        (
            [
                set_test_value(at.dscalar(), np.array(0.3, dtype=np.float64)),
                set_test_value(at.dscalar(), np.array(0.5, dtype=np.float64)),
            ],
            lambda x, y: x > y,
            x,
            y,
        ),
        (
            [
                set_test_value(at.dvector(), np.array([0.3, 0.1], dtype=np.float64)),
                set_test_value(at.dvector(), np.array([0.5, 0.9], dtype=np.float64)),
            ],
            lambda x, y: at.all(x > y),
            x,
            y,
        ),
        (
            [
                set_test_value(at.dvector(), np.array([0.3, 0.1], dtype=np.float64)),
                set_test_value(at.dvector(), np.array([0.5, 0.9], dtype=np.float64)),
            ],
            lambda x, y: at.all(x > y),
            [x, 2 * x],
            [y, 3 * y],
        ),
        (
            [
                set_test_value(at.dvector(), np.array([0.5, 0.9], dtype=np.float64)),
                set_test_value(at.dvector(), np.array([0.3, 0.1], dtype=np.float64)),
            ],
            lambda x, y: at.all(x > y),
            [x, 2 * x],
            [y, 3 * y],
        ),
    ],
)
def test_IfElse(inputs, cond_fn, true_vals, false_vals):

    out = ifelse(cond_fn(*inputs), true_vals, false_vals)

    if not isinstance(out, list):
        out = [out]

    out_fg = FunctionGraph(inputs, out)

    compare_numba_and_py(out_fg, [get_test_value(i) for i in out_fg.inputs])


@pytest.mark.xfail(reason="https://github.com/numba/numba/issues/7409")
def test_config_options_parallel():
    x = at.dvector()

    with config.change_flags(numba__vectorize_target="parallel"):
        aesara_numba_fn = function([x], x * 2, mode=numba_mode)
        numba_mul_fn = aesara_numba_fn.vm.jit_fn.py_func.__globals__["mul"]
        assert numba_mul_fn.targetoptions["parallel"] is True


def test_config_options_fastmath():
    x = at.dvector()

    with config.change_flags(numba__fastmath=True):
        aesara_numba_fn = function([x], x * 2, mode=numba_mode)
        numba_mul_fn = aesara_numba_fn.vm.jit_fn.py_func.__globals__["mul"]
        assert numba_mul_fn.targetoptions["fastmath"] is True


def test_config_options_cached():
    x = at.dvector()

    with config.change_flags(numba__cache=True):
        aesara_numba_fn = function([x], x * 2, mode=numba_mode)
        numba_mul_fn = aesara_numba_fn.vm.jit_fn.py_func.__globals__["mul"]
        assert not isinstance(
            numba_mul_fn._dispatcher.cache, numba.core.caching.NullCache
        )

    with config.change_flags(numba__cache=False):
        aesara_numba_fn = function([x], x * 2, mode=numba_mode)
        numba_mul_fn = aesara_numba_fn.vm.jit_fn.py_func.__globals__["mul"]
        assert isinstance(numba_mul_fn._dispatcher.cache, numba.core.caching.NullCache)


def test_scalar_return_value_conversion():
    r"""Make sure that we convert \"native\" scalars to `ndarray`\s in the graph outputs."""
    x = at.scalar(name="x")
    x_fn = function(
        [x],
        2 * x,
        mode=numba_mode,
    )
    assert isinstance(x_fn(1.0), np.ndarray)
