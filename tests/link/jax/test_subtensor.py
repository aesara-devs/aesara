import numpy as np
import pytest

import aesara.tensor as at
from aesara.configdefaults import config
from aesara.graph.fg import FunctionGraph
from aesara.tensor import subtensor as at_subtensor
from tests.link.jax.test_basic import compare_jax_and_py


def test_jax_Subtensor_constant():
    # Basic indices
    x_at = at.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5)))
    out_at = x_at[1, 2, 0]
    assert isinstance(out_at.owner.op, at_subtensor.Subtensor)
    out_fg = FunctionGraph([], [out_at])
    compare_jax_and_py(out_fg, [])

    out_at = x_at[1:, 1, :]
    assert isinstance(out_at.owner.op, at_subtensor.Subtensor)
    out_fg = FunctionGraph([], [out_at])
    compare_jax_and_py(out_fg, [])

    out_at = x_at[:2, 1, :]
    assert isinstance(out_at.owner.op, at_subtensor.Subtensor)
    out_fg = FunctionGraph([], [out_at])
    compare_jax_and_py(out_fg, [])

    out_at = x_at[1:2, 1, :]
    assert isinstance(out_at.owner.op, at_subtensor.Subtensor)
    out_fg = FunctionGraph([], [out_at])
    compare_jax_and_py(out_fg, [])

    # Advanced indexing
    out_at = at_subtensor.advanced_subtensor1(x_at, [1, 2])
    assert isinstance(out_at.owner.op, at_subtensor.AdvancedSubtensor1)
    out_fg = FunctionGraph([], [out_at])
    compare_jax_and_py(out_fg, [])

    out_at = x_at[[1, 2], [2, 3]]
    assert isinstance(out_at.owner.op, at_subtensor.AdvancedSubtensor)
    out_fg = FunctionGraph([], [out_at])
    compare_jax_and_py(out_fg, [])

    # Advanced and basic indexing
    out_at = x_at[[1, 2], :]
    assert isinstance(out_at.owner.op, at_subtensor.AdvancedSubtensor)
    out_fg = FunctionGraph([], [out_at])
    compare_jax_and_py(out_fg, [])

    out_at = x_at[[1, 2], :, [3, 4]]
    assert isinstance(out_at.owner.op, at_subtensor.AdvancedSubtensor)
    out_fg = FunctionGraph([], [out_at])
    compare_jax_and_py(out_fg, [])

    # Flipping
    out_at = x_at[::-1]
    out_fg = FunctionGraph([], [out_at])
    compare_jax_and_py(out_fg, [])


@pytest.mark.xfail(reason="`a` should be specified as static when JIT-compiling")
def test_jax_Subtensor_dynamic():
    a = at.iscalar("a")
    x = at.arange(3)
    out_at = x[:a]
    assert isinstance(out_at.owner.op, at_subtensor.Subtensor)
    out_fg = FunctionGraph([a], [out_at])
    compare_jax_and_py(out_fg, [1])


def test_jax_Subtensor_boolean_mask():
    """JAX does not support resizing arrays with boolean masks."""
    x_at = at.arange(-5, 5)
    out_at = x_at[x_at < 0]
    assert isinstance(out_at.owner.op, at_subtensor.AdvancedSubtensor)

    with pytest.raises(NotImplementedError, match="resizing arrays with boolean"):
        out_fg = FunctionGraph([], [out_at])
        compare_jax_and_py(out_fg, [])


def test_jax_Subtensor_boolean_mask_reexpressible():
    """Summing values with boolean indexing.

    This test ensures that the sum of an `AdvancedSubtensor` `Op`s with boolean
    indexing is replaced with the sum of an equivalent `Switch` `Op`, using the
    `jax_boolean_indexing_sum` rewrite.

    JAX forces users to re-express this logic manually, so this is an
    improvement over its user interface.

    """
    x_at = at.vector("x")
    out_at = x_at[x_at < 0].sum()
    out_fg = FunctionGraph([x_at], [out_at])
    compare_jax_and_py(out_fg, [np.arange(-5, 5).astype(config.floatX)])


def test_jax_IncSubtensor():
    rng = np.random.default_rng(213234)

    x_np = rng.uniform(-1, 1, size=(3, 4, 5)).astype(config.floatX)
    x_at = at.constant(np.arange(3 * 4 * 5).reshape((3, 4, 5)).astype(config.floatX))

    # "Set" basic indices
    st_at = at.as_tensor_variable(np.array(-10.0, dtype=config.floatX))
    out_at = at_subtensor.set_subtensor(x_at[1, 2, 3], st_at)
    assert isinstance(out_at.owner.op, at_subtensor.IncSubtensor)
    out_fg = FunctionGraph([], [out_at])
    compare_jax_and_py(out_fg, [])

    st_at = at.as_tensor_variable(np.r_[-1.0, 0.0].astype(config.floatX))
    out_at = at_subtensor.set_subtensor(x_at[:2, 0, 0], st_at)
    assert isinstance(out_at.owner.op, at_subtensor.IncSubtensor)
    out_fg = FunctionGraph([], [out_at])
    compare_jax_and_py(out_fg, [])

    out_at = at_subtensor.set_subtensor(x_at[0, 1:3, 0], st_at)
    assert isinstance(out_at.owner.op, at_subtensor.IncSubtensor)
    out_fg = FunctionGraph([], [out_at])
    compare_jax_and_py(out_fg, [])

    # "Set" advanced indices
    st_at = at.as_tensor_variable(
        rng.uniform(-1, 1, size=(2, 4, 5)).astype(config.floatX)
    )
    out_at = at_subtensor.set_subtensor(x_at[np.r_[0, 2]], st_at)
    assert isinstance(out_at.owner.op, at_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([], [out_at])
    compare_jax_and_py(out_fg, [])

    st_at = at.as_tensor_variable(np.r_[-1.0, 0.0].astype(config.floatX))
    out_at = at_subtensor.set_subtensor(x_at[[0, 2], 0, 0], st_at)
    assert isinstance(out_at.owner.op, at_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([], [out_at])
    compare_jax_and_py(out_fg, [])

    # "Set" boolean indices
    mask_at = at.constant(x_np > 0)
    out_at = at_subtensor.set_subtensor(x_at[mask_at], 0.0)
    assert isinstance(out_at.owner.op, at_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([], [out_at])
    compare_jax_and_py(out_fg, [])

    # "Increment" basic indices
    st_at = at.as_tensor_variable(np.array(-10.0, dtype=config.floatX))
    out_at = at_subtensor.inc_subtensor(x_at[1, 2, 3], st_at)
    assert isinstance(out_at.owner.op, at_subtensor.IncSubtensor)
    out_fg = FunctionGraph([], [out_at])
    compare_jax_and_py(out_fg, [])

    st_at = at.as_tensor_variable(np.r_[-1.0, 0.0].astype(config.floatX))
    out_at = at_subtensor.inc_subtensor(x_at[:2, 0, 0], st_at)
    assert isinstance(out_at.owner.op, at_subtensor.IncSubtensor)
    out_fg = FunctionGraph([], [out_at])
    compare_jax_and_py(out_fg, [])

    out_at = at_subtensor.set_subtensor(x_at[0, 1:3, 0], st_at)
    assert isinstance(out_at.owner.op, at_subtensor.IncSubtensor)
    out_fg = FunctionGraph([], [out_at])
    compare_jax_and_py(out_fg, [])

    # "Increment" advanced indices
    st_at = at.as_tensor_variable(
        rng.uniform(-1, 1, size=(2, 4, 5)).astype(config.floatX)
    )
    out_at = at_subtensor.inc_subtensor(x_at[np.r_[0, 2]], st_at)
    assert isinstance(out_at.owner.op, at_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([], [out_at])
    compare_jax_and_py(out_fg, [])

    st_at = at.as_tensor_variable(np.r_[-1.0, 0.0].astype(config.floatX))
    out_at = at_subtensor.inc_subtensor(x_at[[0, 2], 0, 0], st_at)
    assert isinstance(out_at.owner.op, at_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([], [out_at])
    compare_jax_and_py(out_fg, [])

    # "Increment" boolean indices
    mask_at = at.constant(x_np > 0)
    out_at = at_subtensor.set_subtensor(x_at[mask_at], 1.0)
    assert isinstance(out_at.owner.op, at_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([], [out_at])
    compare_jax_and_py(out_fg, [])

    st_at = at.as_tensor_variable(x_np[[0, 2], 0, :3])
    out_at = at_subtensor.set_subtensor(x_at[[0, 2], 0, :3], st_at)
    assert isinstance(out_at.owner.op, at_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([], [out_at])
    compare_jax_and_py(out_fg, [])

    st_at = at.as_tensor_variable(x_np[[0, 2], 0, :3])
    out_at = at_subtensor.inc_subtensor(x_at[[0, 2], 0, :3], st_at)
    assert isinstance(out_at.owner.op, at_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([], [out_at])
    compare_jax_and_py(out_fg, [])


def test_jax_IncSubtensor_boolean_indexing_reexpressible():
    """Setting or incrementing values with boolean indexing.

    This test ensures that `AdvancedIncSubtensor` `Op`s with boolean indexing is
    replaced with an equivalent `Switch` `Op`, using the
    `jax_boolean_indexing_set_of_inc` rewrite.

    JAX forces users to re-express this logic manually, so this is an
    improvement over its user interface.

    """
    rng = np.random.default_rng(213234)
    x_np = rng.uniform(-1, 1, size=(4, 5)).astype(config.floatX)

    x_at = at.matrix("x")
    mask_at = at.as_tensor(x_at) > 0
    out_at = at_subtensor.set_subtensor(x_at[mask_at], 0.0)
    assert isinstance(out_at.owner.op, at_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([x_at], [out_at])
    compare_jax_and_py(out_fg, [x_np])

    mask_at = at.as_tensor(x_at) > 0
    out_at = at_subtensor.inc_subtensor(x_at[mask_at], 1.0)
    assert isinstance(out_at.owner.op, at_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([x_at], [out_at])
    compare_jax_and_py(out_fg, [x_np])
