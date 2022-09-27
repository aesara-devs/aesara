import jax
import numpy as np
import pytest
from jax._src.errors import NonConcreteBooleanIndexError
from packaging.version import parse as version_parse

import aesara.tensor as at
from aesara.configdefaults import config
from aesara.graph.fg import FunctionGraph
from aesara.tensor import subtensor as at_subtensor
from tests.link.jax.test_basic import compare_jax_and_py


def test_jax_Subtensors():
    # Basic indices
    x_at = at.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5)))
    out_at = x_at[1, 2, 0]
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


@pytest.mark.xfail(
    version_parse(jax.__version__) >= version_parse("0.2.12"),
    reason="Omnistaging cannot be disabled",
)
def test_jax_Subtensors_omni():
    x_at = at.arange(3 * 4 * 5).reshape((3, 4, 5))

    # Boolean indices
    out_at = x_at[x_at < 0]
    assert isinstance(out_at.owner.op, at_subtensor.AdvancedSubtensor)
    out_fg = FunctionGraph([], [out_at])
    compare_jax_and_py(out_fg, [])


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


def test_jax_IncSubtensors_unsupported():
    rng = np.random.default_rng(213234)
    x_np = rng.uniform(-1, 1, size=(3, 4, 5)).astype(config.floatX)
    x_at = at.constant(np.arange(3 * 4 * 5).reshape((3, 4, 5)).astype(config.floatX))

    mask_at = at.as_tensor(x_np) > 0
    out_at = at_subtensor.set_subtensor(x_at[mask_at], 0.0)
    assert isinstance(out_at.owner.op, at_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([], [out_at])
    with pytest.raises(
        NonConcreteBooleanIndexError, match="Array boolean indices must be concrete"
    ):
        compare_jax_and_py(out_fg, [])

    mask_at = at.as_tensor_variable(x_np) > 0
    out_at = at_subtensor.set_subtensor(x_at[mask_at], 1.0)
    assert isinstance(out_at.owner.op, at_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([], [out_at])
    with pytest.raises(
        NonConcreteBooleanIndexError, match="Array boolean indices must be concrete"
    ):
        compare_jax_and_py(out_fg, [])

    st_at = at.as_tensor_variable(x_np[[0, 2], 0, :3])
    out_at = at_subtensor.set_subtensor(x_at[[0, 2], 0, :3], st_at)
    assert isinstance(out_at.owner.op, at_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([], [out_at])
    with pytest.raises(IndexError, match="Array slice indices must have static"):
        compare_jax_and_py(out_fg, [])

    st_at = at.as_tensor_variable(x_np[[0, 2], 0, :3])
    out_at = at_subtensor.inc_subtensor(x_at[[0, 2], 0, :3], st_at)
    assert isinstance(out_at.owner.op, at_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([], [out_at])
    with pytest.raises(IndexError, match="Array slice indices must have static"):
        compare_jax_and_py(out_fg, [])
