from copy import copy

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_equal, assert_string_equal

import aesara
import tests.unittest_tools as utt
from aesara.compile.mode import get_default_mode
from aesara.graph.basic import Constant, equal_computations
from aesara.tensor import get_vector_length
from aesara.tensor.basic import constant
from aesara.tensor.elemwise import DimShuffle
from aesara.tensor.math import dot, eq
from aesara.tensor.subtensor import AdvancedSubtensor, Subtensor
from aesara.tensor.type import (
    TensorType,
    cscalar,
    dmatrix,
    dscalar,
    dvector,
    iscalar,
    ivector,
    matrices,
    matrix,
    scalar,
    tensor3,
)
from aesara.tensor.type_other import MakeSlice, NoneConst
from aesara.tensor.var import (
    DenseTensorConstant,
    DenseTensorVariable,
    TensorConstant,
    TensorVariable,
)
from tests.tensor.utils import random


pytestmark = pytest.mark.filterwarnings("error")


@pytest.mark.parametrize(
    "fct, value",
    [
        (np.arccos, 0.5),
        (np.arccosh, 1.0),
        (np.arcsin, 0.5),
        (np.arcsinh, 0.5),
        (np.arctan, 0.5),
        (np.arctanh, 0.5),
        (np.cos, 0.5),
        (np.cosh, 0.5),
        (np.deg2rad, 0.5),
        (np.exp, 0.5),
        (np.exp2, 0.5),
        (np.expm1, 0.5),
        (np.log, 0.5),
        (np.log10, 0.5),
        (np.log1p, 0.5),
        (np.log2, 0.5),
        (np.rad2deg, 0.5),
        (np.sin, 0.5),
        (np.sinh, 0.5),
        (np.sqrt, 0.5),
        (np.tan, 0.5),
        (np.tanh, 0.5),
    ],
)
def test_numpy_method(fct, value):
    x = dscalar("x")
    y = fct(x)
    f = aesara.function([x], y)
    utt.assert_allclose(np.nan_to_num(f(value)), np.nan_to_num(fct(value)))


def test_infix_dot_method():
    X = dmatrix("X")
    y = dvector("y")

    res = X @ y
    exp_res = X.dot(y)
    assert equal_computations([res], [exp_res])

    X_val = np.arange(2 * 3).reshape((2, 3))
    res = X_val @ y
    exp_res = dot(X_val, y)
    assert equal_computations([res], [exp_res])


def test_empty_list_indexing():
    ynp = np.zeros((2, 2))[:, []]
    znp = np.zeros((2, 2))[:, ()]
    data = [[0, 0], [0, 0]]
    x = dmatrix("x")
    y = x[:, []]
    z = x[:, ()]
    fy = aesara.function([x], y)
    fz = aesara.function([x], z)
    assert_equal(fy(data).shape, ynp.shape)
    assert_equal(fz(data).shape, znp.shape)


def test_copy():
    x = dmatrix("x")
    data = np.random.random((5, 5))
    y = x.copy(name="y")
    f = aesara.function([x], y)
    assert_equal(f(data), data)
    assert_string_equal(y.name, "y")


def test__getitem__Subtensor():
    # Make sure we get `Subtensor`s for basic indexing operations
    x = matrix("x")
    i = iscalar("i")

    z = x[i]
    op_types = [type(node.op) for node in aesara.graph.basic.io_toposort([x, i], [z])]
    assert op_types[-1] == Subtensor

    # This should ultimately do nothing (i.e. just return `x`)
    z = x[()]
    assert len(z.owner.op.idx_list) == 0
    # assert z is x

    # This is a poorly placed optimization that produces a `DimShuffle`
    # It lands in the `full_slices` condition in
    # `_tensor_py_operators.__getitem__`
    z = x[..., None]
    op_types = [type(node.op) for node in aesara.graph.basic.io_toposort([x, i], [z])]
    assert all(op_type == DimShuffle for op_type in op_types)

    z = x[None, :, None, :]
    op_types = [type(node.op) for node in aesara.graph.basic.io_toposort([x, i], [z])]
    assert all(op_type == DimShuffle for op_type in op_types)

    # This one lands in the non-`full_slices` condition in
    # `_tensor_py_operators.__getitem__`
    z = x[:i, :, None]
    op_types = [type(node.op) for node in aesara.graph.basic.io_toposort([x, i], [z])]
    assert op_types[1:] == [DimShuffle, Subtensor]

    z = x[:]
    op_types = [type(node.op) for node in aesara.graph.basic.io_toposort([x, i], [z])]
    assert op_types[-1] == Subtensor

    z = x[..., :]
    op_types = [type(node.op) for node in aesara.graph.basic.io_toposort([x, i], [z])]
    assert op_types[-1] == Subtensor

    z = x[..., i, :]
    op_types = [type(node.op) for node in aesara.graph.basic.io_toposort([x, i], [z])]
    assert op_types[-1] == Subtensor


def test__getitem__AdvancedSubtensor_bool():
    x = matrix("x")
    i = TensorType("bool", shape=(None, None))("i")

    z = x[i]
    op_types = [type(node.op) for node in aesara.graph.basic.io_toposort([x, i], [z])]
    assert op_types[-1] == AdvancedSubtensor

    i = TensorType("bool", shape=(None,))("i")
    z = x[:, i]
    op_types = [type(node.op) for node in aesara.graph.basic.io_toposort([x, i], [z])]
    assert op_types[-1] == AdvancedSubtensor

    i = TensorType("bool", shape=(None,))("i")
    z = x[..., i]
    op_types = [type(node.op) for node in aesara.graph.basic.io_toposort([x, i], [z])]
    assert op_types[-1] == AdvancedSubtensor

    with pytest.raises(TypeError):
        z = x[[True, False], i]

    z = x[ivector("b"), i]
    op_types = [type(node.op) for node in aesara.graph.basic.io_toposort([x, i], [z])]
    assert op_types[-1] == AdvancedSubtensor


def test__getitem__AdvancedSubtensor():
    # Make sure we get `AdvancedSubtensor`s for basic indexing operations
    x = matrix("x")
    i = ivector("i")

    # This is a `__getitem__` call that's redirected to `_tensor_py_operators.take`
    z = x[i]
    op_types = [type(node.op) for node in aesara.graph.basic.io_toposort([x, i], [z])]
    assert op_types[-1] == AdvancedSubtensor

    # This should index nothing (i.e. return an empty copy of `x`)
    # We check that the index is empty
    z = x[[]]
    op_types = [type(node.op) for node in aesara.graph.basic.io_toposort([x, i], [z])]
    assert op_types == [AdvancedSubtensor]
    assert isinstance(z.owner.inputs[1], TensorConstant)

    z = x[:, i]
    op_types = [type(node.op) for node in aesara.graph.basic.io_toposort([x, i], [z])]
    assert op_types == [MakeSlice, AdvancedSubtensor]

    z = x[..., i, None]
    op_types = [type(node.op) for node in aesara.graph.basic.io_toposort([x, i], [z])]
    assert op_types == [MakeSlice, AdvancedSubtensor]

    z = x[i, None]
    op_types = [type(node.op) for node in aesara.graph.basic.io_toposort([x, i], [z])]
    assert op_types[-1] == AdvancedSubtensor


def test_print_constant():
    c = aesara.tensor.constant(1, name="const")
    assert str(c) == "const{1}"
    d = aesara.tensor.constant(1)
    assert str(d) == "TensorConstant{1}"


@pytest.mark.parametrize(
    "x, indices, new_order",
    [
        (tensor3(), (np.newaxis, slice(None), np.newaxis), ("x", 0, "x", 1, 2)),
        (cscalar(), (np.newaxis,), ("x",)),
        (cscalar(), (NoneConst,), ("x",)),
        (matrix(), (np.newaxis,), ("x", 0, 1)),
        (matrix(), (np.newaxis, np.newaxis), ("x", "x", 0, 1)),
        (matrix(), (np.newaxis, slice(None)), ("x", 0, 1)),
        (matrix(), (np.newaxis, slice(None), slice(None)), ("x", 0, 1)),
        (matrix(), (np.newaxis, np.newaxis, slice(None)), ("x", "x", 0, 1)),
        (matrix(), (slice(None), np.newaxis), (0, "x", 1)),
        (matrix(), (slice(None), slice(None), np.newaxis), (0, 1, "x")),
        (
            matrix(),
            (np.newaxis, slice(None), np.newaxis, slice(None), np.newaxis),
            ("x", 0, "x", 1, "x"),
        ),
    ],
)
def test__getitem__newaxis(x, indices, new_order):
    res = x[indices]
    assert isinstance(res.owner.op, DimShuffle)
    assert res.broadcastable == tuple(i == "x" for i in new_order)
    assert res.owner.op.new_order == new_order


def test_fixed_shape_variable_basic():
    x = TensorVariable(TensorType("int64", shape=(4,)), None)
    assert isinstance(x.shape, Constant)
    assert np.array_equal(x.shape.data, (4,))

    x = TensorConstant(
        TensorType("int64", shape=(None, None)), np.array([[1, 2], [2, 3]])
    )
    assert x.type.shape == (2, 2)

    with pytest.raises(ValueError):
        TensorConstant(TensorType("int64", shape=(1, None)), np.array([[1, 2], [2, 3]]))


def test_get_vector_length():
    x = TensorVariable(TensorType("int64", shape=(4,)), None)
    res = get_vector_length(x)
    assert res == 4

    x = TensorVariable(TensorType("int64", shape=(None,)), None)
    with pytest.raises(ValueError):
        get_vector_length(x)


def test_dense_types():
    x = matrix()
    assert isinstance(x, DenseTensorVariable)
    assert not isinstance(x, DenseTensorConstant)

    x = constant(1)
    assert not isinstance(x, DenseTensorVariable)
    assert isinstance(x, DenseTensorConstant)


class TestTensorConstantSignature:
    vals = [
        [np.nan, np.inf, 0, 1],
        [np.nan, np.inf, -np.inf, 1],
        [0, np.inf, -np.inf, 1],
        [0, 3, -np.inf, 1],
        [0, 3, np.inf, 1],
        [np.nan, 3, 4, 1],
        [0, 3, 4, 1],
        np.nan,
        np.inf,
        -np.inf,
        0,
        1,
    ]

    @pytest.mark.parametrize("val_1", vals)
    @pytest.mark.parametrize("val_2", vals)
    def test_nan_inf_constant_signature(self, val_1, val_2):
        # Test that the signature of a constant tensor containing NaN and Inf
        # values is correct.
        # We verify that signatures of two rows i, j in the matrix above are
        # equal if and only if i == j.
        x = constant(val_1)
        y = constant(val_2)
        assert (x.signature() == y.signature()) == (val_1 is val_2)

    def test_nan_nan(self):
        # Also test that nan !=0 and nan != nan.
        x = scalar()
        mode = get_default_mode()
        if isinstance(mode, aesara.compile.debugmode.DebugMode):
            # Disable the check preventing usage of NaN / Inf values.
            # We first do a copy of the mode to avoid side effects on other tests.
            mode = copy(mode)
            mode.check_isfinite = False
        f = aesara.function([x], eq(x, np.nan), mode=mode)

        assert f(0) == 0
        assert f(np.nan) == 0

    def test_empty_hash(self):
        x = constant(np.array([], dtype=np.int64))
        y = constant(np.array([], dtype=np.int64))

        x_sig = x.signature()
        y_sig = y.signature()

        assert hash(x_sig) == hash(y_sig)


class TestTensorInstanceMethods:
    def setup_method(self):
        self.vars = matrices("X", "Y")
        self.vals = [
            m.astype(aesara.config.floatX) for m in [random(2, 2), random(2, 2)]
        ]

    def test_repeat(self):
        X, _ = self.vars
        x, _ = self.vals
        assert_array_equal(X.repeat(2).eval({X: x}), x.repeat(2))

    def test_trace(self):
        X, _ = self.vars
        x, _ = self.vals
        assert_array_equal(X.trace().eval({X: x}), x.trace())

    def test_ravel(self):
        X, _ = self.vars
        x, _ = self.vals
        assert_array_equal(X.ravel().eval({X: x}), x.ravel())

    def test_diagonal(self):
        X, _ = self.vars
        x, _ = self.vals
        assert_array_equal(X.diagonal().eval({X: x}), x.diagonal())
        assert_array_equal(X.diagonal(1).eval({X: x}), x.diagonal(1))
        assert_array_equal(X.diagonal(-1).eval({X: x}), x.diagonal(-1))
        for offset, axis1, axis2 in [(1, 0, 1), (-1, 0, 1), (0, 1, 0), (-2, 1, 0)]:
            assert_array_equal(
                X.diagonal(offset, axis1, axis2).eval({X: x}),
                x.diagonal(offset, axis1, axis2),
            )

    def test_take(self):
        X, _ = self.vars
        x, _ = self.vals
        indices = [1, 0, 3]
        assert_array_equal(X.take(indices).eval({X: x}), x.take(indices))
        indices = [1, 0, 1]
        assert_array_equal(X.take(indices, 1).eval({X: x}), x.take(indices, 1))
        indices = np.array([-10, 5, 12], dtype="int32")
        assert_array_equal(
            X.take(indices, 1, mode="wrap").eval({X: x}),
            x.take(indices, 1, mode="wrap"),
        )
        assert_array_equal(
            X.take(indices, -1, mode="wrap").eval({X: x}),
            x.take(indices, -1, mode="wrap"),
        )
        assert_array_equal(
            X.take(indices, 1, mode="clip").eval({X: x}),
            x.take(indices, 1, mode="clip"),
        )
        assert_array_equal(
            X.take(indices, -1, mode="clip").eval({X: x}),
            x.take(indices, -1, mode="clip"),
        )
        # Test error handling
        with pytest.raises(IndexError):
            X.take(indices).eval({X: x})
        with pytest.raises(IndexError):
            (2 * X.take(indices)).eval({X: x})
        with pytest.raises(TypeError):
            X.take([0.0])
        indices = [[1, 0, 1], [0, 1, 1]]
        assert_array_equal(X.take(indices, 1).eval({X: x}), x.take(indices, 1))
        # Test equivalent advanced indexing
        assert_array_equal(X[:, indices].eval({X: x}), x[:, indices])
