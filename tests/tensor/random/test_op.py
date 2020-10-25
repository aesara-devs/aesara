import numpy as np
from pytest import fixture, raises

import theano.tensor as tt
from theano import change_flags, config
from theano.gradient import NullTypeGradError
from theano.tensor.opt import Assert
from theano.tensor.random.basic import normal
from theano.tensor.random.op import RandomVariable, default_shape_from_params, observed
from theano.tensor.type_other import NoneTypeT


@fixture(scope="module", autouse=True)
def set_theano_flags():
    with change_flags(cxx="", compute_test_value="raise"):
        yield


def test_default_shape_from_params():
    with raises(ValueError, match="^ndim_supp*"):
        default_shape_from_params(0, (np.array([1, 2]), 0))

    res = default_shape_from_params(1, (np.array([1, 2]), np.eye(2)), rep_param_idx=0)
    assert res == (2,)

    res = default_shape_from_params(1, (np.array([1, 2]), 0), param_shapes=((2,), ()))
    assert res == (2,)

    with raises(ValueError, match="^Reference parameter*"):
        default_shape_from_params(1, (np.array(1),), rep_param_idx=0)

    res = default_shape_from_params(
        2, (np.array([1, 2]), np.ones((2, 3, 4))), rep_param_idx=1
    )
    assert res == (3, 4)


def test_RandomVariable():

    str_res = str(
        RandomVariable(
            "normal",
            0,
            [0, 0],
            "normal",
            inplace=True,
        )
    )

    assert str_res == "normal_rv"

    # `ndims_params` should be a `Sequence` type
    with raises(TypeError, match="^Parameter ndims_params*"):
        RandomVariable(
            "normal",
            0,
            0,
            config.floatX,
            inplace=True,
        )

    # `size` should be a `Sequence` type
    with raises(TypeError, match="^Parameter size*"):
        RandomVariable(
            "normal",
            0,
            [0, 0],
            config.floatX,
            inplace=True,
        )(0, 1, size={1, 2})

    # No dtype
    with raises(TypeError, match="^dtype*"):
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
    assert rv.destroy_map == {0: [3]}

    # A no-params `RandomVariable`
    rv = RandomVariable(name="test_rv", ndim_supp=0, ndims_params=())

    with raises(TypeError):
        rv.make_node(rng=1)

    # `RandomVariable._infer_shape` should handle no parameters
    rv_shape = rv._infer_shape(tt.constant([]), (), [])
    assert rv_shape.equals(tt.constant([], dtype="int64"))

    # Integer-specificed `dtype`
    dtype_1 = tt.all_dtypes[1]
    rv_node = rv.make_node(None, None, 1)
    rv_out = rv_node.outputs[1]
    rv_out.tag.test_value = 1

    assert rv_out.dtype == dtype_1

    with raises(NullTypeGradError):
        tt.grad(rv_out, [rv_node.inputs[0]])

    rv = RandomVariable("normal", 0, [0, 0], config.floatX, inplace=True)

    mu = tt.tensor(config.floatX, [True, False, False])
    mu.tag.test_value = np.zeros((1, 2, 3)).astype(config.floatX)
    sd = tt.tensor(config.floatX, [False, False])
    sd.tag.test_value = np.ones((2, 3)).astype(config.floatX)

    s1 = tt.iscalar()
    s1.tag.test_value = 1
    s2 = tt.iscalar()
    s2.tag.test_value = 2
    s3 = tt.iscalar()
    s3.tag.test_value = 3
    s3 = Assert("testing")(s3, tt.eq(s1, 1))

    res = rv.compute_bcast([mu, sd], (s1, s2, s3))
    assert res == [False] * 3


def test_observed():
    rv_var = normal(0, 1, size=3)
    obs_var = observed(rv_var, np.array([0.2, 0.1, -2.4], dtype=config.floatX))

    assert obs_var.owner.inputs[0] is rv_var

    with raises(TypeError):
        observed(rv_var, np.array([1, 2], dtype=int))

    with raises(TypeError):
        observed(rv_var, np.array([[1.0, 2.0]], dtype=rv_var.dtype))

    obs_rv = observed(None, np.array([0.2, 0.1, -2.4], dtype=config.floatX))

    assert isinstance(obs_rv.owner.inputs[0].type, NoneTypeT)

    rv_val = tt.vector()
    rv_val.tag.test_value = np.array([0.2, 0.1, -2.4], dtype=config.floatX)

    obs_var = observed(rv_var, rv_val)

    with raises(NullTypeGradError):
        tt.grad(obs_var.sum(), [rv_val])
