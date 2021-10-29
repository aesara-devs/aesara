import numpy as np
import pytest
from numpy.testing import assert_equal, assert_string_equal

import aesara
import tests.unittest_tools as utt
from aesara.graph.basic import equal_computations
from aesara.tensor.elemwise import DimShuffle
from aesara.tensor.math import dot
from aesara.tensor.subtensor import AdvancedSubtensor, Subtensor
from aesara.tensor.type import (
    TensorType,
    cscalar,
    dmatrix,
    dvector,
    iscalar,
    ivector,
    matrix,
    tensor3,
)
from aesara.tensor.type_other import MakeSlice
from aesara.tensor.var import TensorConstant


@pytest.mark.parametrize(
    "fct",
    [
        np.arccos,
        np.arccosh,
        np.arcsin,
        np.arcsinh,
        np.arctan,
        np.arctanh,
        # np.ceil, np.floor, np.trunc,
        np.cos,
        np.cosh,
        np.deg2rad,
        np.exp,
        np.exp2,
        np.expm1,
        np.log,
        np.log10,
        np.log1p,
        np.log2,
        np.rad2deg,
        np.sin,
        np.sinh,
        np.sqrt,
        np.tan,
        np.tanh,
    ],
)
def test_numpy_method(fct):
    # This type of code is used frequently by PyMC3 users
    x = dmatrix("x")
    data = np.random.rand(5, 5)
    x.tag.test_value = data
    y = fct(x)
    f = aesara.function([x], y)
    utt.assert_allclose(np.nan_to_num(f(data)), np.nan_to_num(fct(data)))


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
    data = np.random.rand(5, 5)
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
    i = TensorType("bool", (False, False))("i")

    z = x[i]
    op_types = [type(node.op) for node in aesara.graph.basic.io_toposort([x, i], [z])]
    assert op_types[-1] == AdvancedSubtensor

    i = TensorType("bool", (False,))("i")
    z = x[:, i]
    op_types = [type(node.op) for node in aesara.graph.basic.io_toposort([x, i], [z])]
    assert op_types[-1] == AdvancedSubtensor

    i = TensorType("bool", (False,))("i")
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


@pytest.mark.parametrize(
    "x, indices, new_order",
    [
        (tensor3(), (np.newaxis, slice(None), np.newaxis), ("x", 0, "x", 1, 2)),
        (cscalar(), (np.newaxis,), ("x",)),
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
