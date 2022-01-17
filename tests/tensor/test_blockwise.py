import numpy as np
import pytest

import aesara
import aesara.tensor as at
from aesara.tensor.blockwise import (
    Blockwise,
    _calculate_shapes,
    _parse_input_dimensions,
    _update_dim_sizes,
)
from aesara.tensor.math import Dot
from aesara.tensor.type import TensorType
from tests.unittest_tools import check_infer_shape


class DotBW(Dot):
    gufunc_sig = ((("m", "n"), ("n", "p")), (("m", "p"),))


dot_bw = DotBW()


def test_update_dim_sizes():
    with pytest.raises(ValueError, match=".*dimensional argument.*"):
        _update_dim_sizes({}, at.tensor("float64", ()), ("m",))


@pytest.mark.parametrize(
    "args, arg_vals, input_core_dims, output_core_dims",
    [
        (
            (
                at.tensor("float64", (None, None, None)),
                at.tensor("float64", (None, None, None)),
            ),
            (np.zeros((5, 3, 2)), np.zeros((5, 2, 4))),
            (("m", "n"), ("n", "p")),
            (("m", "p"),),
        ),
    ],
)
def test_parse_input_dimensions(args, arg_vals, input_core_dims, output_core_dims):
    bcast_shape, dim_sizes = _parse_input_dimensions(args, input_core_dims)

    res_fn = aesara.function(
        args, list(bcast_shape) + list(dim_sizes.values()), on_unused_input="ignore"
    )

    exp_bcast_shape, exp_dim_sizes = np.lib.function_base._parse_input_dimensions(
        arg_vals, input_core_dims
    )

    res = res_fn(*arg_vals)
    bcast_shape_res = res[: len(bcast_shape)]
    dim_sizes_res = res[len(bcast_shape) :]

    assert tuple(bcast_shape_res) == exp_bcast_shape
    assert dict(zip(dim_sizes.keys(), dim_sizes_res)) == exp_dim_sizes

    # Also test `_calculate_shapes`, since it's nearly trivial and we're
    # already set up for it.
    # First, compute the output shape
    (shapes_at,) = _calculate_shapes(bcast_shape, dim_sizes, output_core_dims)
    shape_fn = aesara.function(args, list(shapes_at), on_unused_input="ignore")

    exp_shape_res = np.lib.function_base._calculate_shapes(
        exp_bcast_shape, exp_dim_sizes, output_core_dims
    )
    shape_res = shape_fn(*arg_vals)
    assert tuple(shape_res) == exp_shape_res

    # TODO: Second, compute the input broadcast shapes
    # input_bcast_shapes, = _calculate_shapes(bcast_shape, dim_sizes, input_core_dims)


@pytest.mark.parametrize(
    "op, args, arg_vals, np_fn",
    [
        (
            dot_bw,
            (
                at.tensor("float64", (None, None, None)),
                at.tensor("float64", (None, None, None)),
            ),
            (np.zeros((5, 3, 2)), np.zeros((5, 2, 4))),
            lambda x, y: np.dot(x, y),
        ),
    ],
)
def test_Blockwise_perform(op, args, arg_vals, np_fn):
    x = Blockwise(op)(*args)
    x_fn = aesara.function(args, x)

    res = x_fn(arg_vals)

    sig = op.gufunc_sig
    sig = f"({','.join(sig[0])})->({','.join(sig[1])})"

    np_fn = np.vectorize(np_fn, signature=sig)
    exp_res = np_fn(*arg_vals)

    assert np.array_equal(res, exp_res)


@pytest.mark.parametrize(
    "op, s_left, s_right",
    [
        (dot_bw, (3, 5, 6), (3, 6, 7)),
        (dot_bw, (3, 1, 2), (3, 2, 1)),
        (
            dot_bw,
            (3, 1, 1),
            (
                3,
                1,
            ),
        ),
    ],
)
def test_Blockwise_infer_shape(op, s_left, s_right):
    dtype = aesara.config.floatX
    t_left = TensorType(dtype, [(entry == 1) for entry in s_left])()
    t_right = TensorType(dtype, [(entry == 1) for entry in s_right])()
    t_left_val = np.zeros(s_left, dtype=dtype)
    t_right_val = np.zeros(s_right, dtype=dtype)
    check_infer_shape(
        [t_left, t_right],
        [Blockwise(op)(t_left, t_right)],
        [t_left_val, t_right_val],
        Blockwise,
    )


def test_Blockwise_grad():
    raise AssertionError()
