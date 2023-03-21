import numpy as np
import pytest

import aesara.tensor as at
from aesara import config, function
from aesara.gradient import NullTypeGradError, grad
from aesara.raise_op import Assert
from aesara.tensor.math import eq
from aesara.tensor.random.op import (
    RandomState,
    RandomVariable,
    default_rng,
    default_supp_shape_from_params,
)
from aesara.tensor.shape import specify_shape
from aesara.tensor.type import all_dtypes, iscalar, tensor


@pytest.fixture(scope="module", autouse=True)
def set_aesara_flags():
    with config.change_flags(cxx="", compute_test_value="raise"):
        yield


def test_default_supp_shape_from_params():
    with pytest.raises(ValueError, match="^ndim_supp*"):
        default_supp_shape_from_params(0, (np.array([1, 2]), 0))

    res = default_supp_shape_from_params(
        1, (np.array([1, 2]), np.eye(2)), rep_param_idx=0
    )
    assert res == (2,)

    res = default_supp_shape_from_params(
        1, (np.array([1, 2]), 0), param_shapes=((2,), ())
    )
    assert res == (2,)

    with pytest.raises(ValueError, match="^Reference parameter*"):
        default_supp_shape_from_params(1, (np.array(1),), rep_param_idx=0)

    res = default_supp_shape_from_params(
        2, (np.array([1, 2]), np.ones((2, 3, 4))), rep_param_idx=1
    )
    assert res == (3, 4)


def test_RandomVariable_basics():
    str_res = str(
        RandomVariable(
            "normal",
            0,
            [0, 0],
            "float32",
            inplace=True,
        )
    )

    assert str_res == "normal_rv{0, (0, 0), float32, True}"

    # `ndims_params` should be a `Sequence` type
    with pytest.raises(TypeError, match="^Parameter ndims_params*"):
        RandomVariable(
            "normal",
            0,
            0,
            config.floatX,
            inplace=True,
        )

    # `size` should be a `Sequence` type
    with pytest.raises(TypeError, match="^Parameter size*"):
        RandomVariable(
            "normal",
            0,
            [0, 0],
            config.floatX,
            inplace=True,
        )(0, 1, size={1, 2})

    # No dtype
    with pytest.raises(TypeError, match="^dtype*"):
        RandomVariable(
            "normal",
            0,
            [0, 0],
            inplace=True,
        )(0, 1)

    # Confirm that `inplace` works
    rv = RandomVariable(
        "normal",
        0,
        [0, 0],
        "normal",
        inplace=True,
    )

    assert rv.inplace
    assert rv.destroy_map == {0: [0]}

    # A no-params `RandomVariable`
    rv = RandomVariable(name="test_rv", ndim_supp=0, ndims_params=())

    with pytest.raises(TypeError):
        rv.make_node(rng=1)

    # `RandomVariable._infer_shape` should handle no parameters
    rv_shape = rv._infer_shape(at.constant([]), (), [])
    assert rv_shape.equals(at.constant([], dtype="int64"))

    # Integer-specificed `dtype`
    dtype_1 = all_dtypes[1]
    rv_node = rv.make_node(None, None, 1)
    rv_out = rv_node.outputs[1]
    rv_out.tag.test_value = 1

    assert rv_out.dtype == dtype_1

    with pytest.raises(NullTypeGradError):
        grad(rv_out, [rv_node.inputs[0]])


def test_RandomVariable_bcast():
    rv = RandomVariable("normal", 0, [0, 0], config.floatX, inplace=True)

    mu = tensor(config.floatX, shape=(1, None, None))
    mu.tag.test_value = np.zeros((1, 2, 3)).astype(config.floatX)
    sd = tensor(config.floatX, shape=(None, None))
    sd.tag.test_value = np.ones((2, 3)).astype(config.floatX)

    s1 = iscalar()
    s1.tag.test_value = 1
    s2 = iscalar()
    s2.tag.test_value = 2
    s3 = iscalar()
    s3.tag.test_value = 3
    s3 = Assert("testing")(s3, eq(s1, 1))

    res = rv(mu, sd, size=(s1, s2, s3))
    assert res.broadcastable == (False,) * 3

    size = at.as_tensor((1, 2, 3), dtype=np.int32).astype(np.int64)
    res = rv(mu, sd, size=size)
    assert res.broadcastable == (True, False, False)

    res = rv(0, 1, size=at.as_tensor(1, dtype=np.int64))
    assert res.broadcastable == (True,)


def test_RandomVariable_bcast_specify_shape():
    rv = RandomVariable("normal", 0, [0, 0], config.floatX, inplace=True)

    s1 = at.as_tensor(1, dtype=np.int64)
    s2 = iscalar()
    s2.tag.test_value = 2
    s3 = iscalar()
    s3.tag.test_value = 3
    s3 = Assert("testing")(s3, eq(s1, 1))

    size = specify_shape(at.as_tensor([s1, s3, s2, s2, s1]), (5,))
    mu = tensor(config.floatX, shape=(None, None, 1))
    mu.tag.test_value = np.random.normal(size=(2, 2, 1)).astype(config.floatX)

    std = tensor(config.floatX, shape=(None, 1, 1))
    std.tag.test_value = np.ones((2, 1, 1)).astype(config.floatX)

    res = rv(mu, std, size=size)
    assert res.type.shape == (1, None, None, None, 1)


def test_RandomVariable_floatX():
    test_rv_op = RandomVariable(
        "normal",
        0,
        [0, 0],
        "floatX",
        inplace=True,
    )

    assert test_rv_op.dtype == "floatX"

    assert test_rv_op(0, 1).dtype == config.floatX

    new_floatX = "float64" if config.floatX == "float32" else "float32"

    with config.change_flags(floatX=new_floatX):
        assert test_rv_op(0, 1).dtype == new_floatX


@pytest.mark.parametrize(
    "seed, maker_op, numpy_res",
    [
        (3, RandomState, np.random.RandomState(3)),
        (3, default_rng, np.random.default_rng(3)),
    ],
)
def test_random_maker_op(seed, maker_op, numpy_res):
    seed = at.as_tensor_variable(seed)
    z = function(inputs=[], outputs=[maker_op(seed)])()
    aes_res = z[0]
    assert maker_op.random_type.values_eq(aes_res, numpy_res)


def test_random_maker_ops_no_seed():
    # Testing the initialization when seed=None
    # Since internal states randomly generated,
    # we just check the output classes
    z = function(inputs=[], outputs=[RandomState()])()
    aes_res = z[0]
    assert isinstance(aes_res, np.random.RandomState)

    z = function(inputs=[], outputs=[default_rng()])()
    aes_res = z[0]
    assert isinstance(aes_res, np.random.Generator)
