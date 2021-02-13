import numpy as np
import pytest

import aesara.tensor as aet
from aesara import config, shared
from aesara.compile.function import function
from aesara.compile.mode import Mode
from aesara.graph.basic import Constant
from aesara.graph.fg import FunctionGraph
from aesara.graph.opt import EquilibriumOptimizer
from aesara.graph.optdb import Query
from aesara.tensor.elemwise import DimShuffle
from aesara.tensor.random.basic import (
    dirichlet,
    multivariate_normal,
    normal,
    poisson,
    uniform,
)
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.random.opt import (
    lift_rv_shapes,
    local_dimshuffle_rv_lift,
    local_subtensor_rv_lift,
)
from aesara.tensor.subtensor import AdvancedSubtensor, AdvancedSubtensor1, Subtensor
from aesara.tensor.type import iscalar, vector


inplace_mode = Mode("py", Query(include=["random_make_inplace"], exclude=[]))
no_mode = Mode("py", Query(include=[], exclude=[]))


def test_inplace_optimization():

    out = normal(0, 1)

    assert out.owner.op.inplace is False

    f = function(
        [],
        out,
        mode=inplace_mode,
    )

    (new_out,) = f.maker.fgraph.outputs
    assert new_out.type == out.type
    assert isinstance(new_out.owner.op, type(out.owner.op))
    assert new_out.owner.op.inplace is True
    assert all(
        np.array_equal(a.data, b.data)
        for a, b in zip(new_out.owner.inputs[1:], out.owner.inputs[1:])
    )


def check_shape_lifted_rv(rv, params, size, rng):
    aet_params = []
    for p in params:
        p_tt = aet.as_tensor(p)
        p_tt = p_tt.type()
        p_tt.tag.test_value = p
        aet_params.append(p_tt)

    aet_size = []
    for s in size:
        s_tt = aet.as_tensor(s)
        s_tt = s_tt.type()
        s_tt.tag.test_value = s
        aet_size.append(s_tt)

    rv = rv(*aet_params, size=aet_size, rng=rng)
    rv_lifted = lift_rv_shapes(rv.owner)

    # Make sure the size input is empty
    assert np.array_equal(rv_lifted.inputs[1].data, [])

    f_ref = function(
        aet_params + aet_size,
        rv,
        mode=no_mode,
    )
    f_lifted = function(
        aet_params + aet_size,
        rv_lifted.outputs[1],
        mode=no_mode,
    )
    f_ref_val = f_ref(*(params + size))
    f_lifted_val = f_lifted(*(params + size))
    assert np.array_equal(f_ref_val, f_lifted_val)


@config.change_flags(compute_test_value="raise")
def test_lift_rv_shapes():

    rng = shared(np.random.RandomState(1233532), borrow=False)

    test_params = [
        np.array(1.0, dtype=config.floatX),
        np.array(5.0, dtype=config.floatX),
    ]
    test_size = []
    check_shape_lifted_rv(normal, test_params, test_size, rng)

    test_params = [
        np.array([0.0, 1.0], dtype=config.floatX),
        np.array(5.0, dtype=config.floatX),
    ]
    test_size = [3, 2]
    check_shape_lifted_rv(normal, test_params, test_size, rng)

    test_params = [
        np.array([[0], [10], [100]], dtype=config.floatX),
        np.diag(np.array([1e-6], dtype=config.floatX)),
    ]
    test_size = [2, 3]
    check_shape_lifted_rv(multivariate_normal, test_params, test_size, rng)

    test_params = [
        np.array([[100, 1, 1], [1, 100, 1], [1, 1, 100]], dtype=config.floatX)
    ]
    test_size = [2, 3]
    check_shape_lifted_rv(dirichlet, test_params, test_size, rng)


@pytest.mark.parametrize(
    "ds_order, lifted, dist_op, dist_params, size, rtol",
    [
        (
            (1, 0, 2),
            True,
            normal,
            (
                np.arange(2 * 2 * 2).reshape((2, 2, 2)).astype(config.floatX),
                np.array(1e-6).astype(config.floatX),
            ),
            (),
            1e-3,
        ),
        (
            (0, 1, 2),
            True,
            normal,
            (np.array(0).astype(config.floatX), np.array(1e-6).astype(config.floatX)),
            (2, 1, 2),
            1e-3,
        ),
        (
            (0, 2, 1),
            True,
            normal,
            (np.array(0).astype(config.floatX), np.array(1e-6).astype(config.floatX)),
            (2, 1, 2),
            1e-3,
        ),
        (
            (1, 0, 2),
            True,
            normal,
            (np.array(0).astype(config.floatX), np.array(1e-6).astype(config.floatX)),
            (2, 1, 2),
            1e-3,
        ),
        (
            (0, 2, 1),
            True,
            normal,
            (
                np.array([[-1, 20], [300, -4000]], dtype=config.floatX),
                np.array([[1e-6, 2e-6]], dtype=config.floatX),
            ),
            (3, 2, 2),
            1e-3,
        ),
        (
            ("x", 0, 2, 1, "x"),
            True,
            normal,
            (
                np.array([[-1, 20], [300, -4000]], dtype=config.floatX),
                np.array([[1e-6, 2e-6]], dtype=config.floatX),
            ),
            (3, 2, 2),
            1e-3,
        ),
        (
            ("x", 0, "x", 2, "x", 1, "x"),
            True,
            normal,
            (
                np.array([[-1, 20], [300, -4000]], dtype=config.floatX),
                np.array([[1e-6, 2e-6]], dtype=config.floatX),
            ),
            (3, 2, 2),
            1e-3,
        ),
        (
            ("x", 0, 2, 1, "x"),
            True,
            normal,
            (
                np.array([[-1, 20], [300, -4000]], dtype=config.floatX),
                np.array([[1e-6, 2e-6]], dtype=config.floatX),
            ),
            (3, 2, 2),
            1e-3,
        ),
        (
            ("x", 1, 0, 2, "x"),
            False,
            normal,
            (
                np.array([[-1, 20], [300, -4000]], dtype=config.floatX),
                np.array([[1e-6, 2e-6]], dtype=config.floatX),
            ),
            (3, 2, 2),
            1e-3,
        ),
        # Only one distribution parameter
        (
            (0, 2, 1),
            True,
            poisson,
            (np.array([[10, 50], [100, 150]], dtype=config.floatX),),
            (3, 2, 2),
            1,
        ),
        # A multi-dimensional case
        (
            (0, 2, 1),
            False,
            multivariate_normal,
            (
                np.array([[-1, 20], [300, -4000]], dtype=config.floatX),
                np.eye(2).astype(config.floatX) * 1e-6,
            ),
            (3,),
            1e-3,
        ),
    ],
)
@config.change_flags(compute_test_value_opt="raise", compute_test_value="raise")
def test_DimShuffle_lift(ds_order, lifted, dist_op, dist_params, size, rtol):

    rng = shared(np.random.RandomState(1233532), borrow=False)

    dist_params_tt = []
    for p in dist_params:
        p_tt = aet.as_tensor(p).type()
        p_tt.tag.test_value = p
        dist_params_tt.append(p_tt)

    size_tt = []
    for s in size:
        s_tt = iscalar()
        s_tt.tag.test_value = s
        size_tt.append(s_tt)

    dist_st = dist_op(*dist_params_tt, size=size_tt, rng=rng).dimshuffle(ds_order)

    f_inputs = [
        p for p in dist_params_tt + size_tt if not isinstance(p, (slice, Constant))
    ]

    mode = Mode(
        "py", EquilibriumOptimizer([local_dimshuffle_rv_lift], max_use_ratio=100)
    )

    f_opt = function(
        f_inputs,
        dist_st,
        mode=mode,
    )

    (new_out,) = f_opt.maker.fgraph.outputs

    if lifted:
        assert new_out.owner.op == dist_op
        assert all(
            isinstance(i.owner.op, DimShuffle)
            for i in new_out.owner.inputs[3:]
            if i.owner
        )
    else:
        assert isinstance(new_out.owner.op, DimShuffle)
        return

    f_base = function(
        f_inputs,
        dist_st,
        mode=no_mode,
    )

    arg_values = [p.get_test_value() for p in f_inputs]
    res_base = f_base(*arg_values)
    res_opt = f_opt(*arg_values)

    np.testing.assert_allclose(res_base, res_opt, rtol=rtol)


@pytest.mark.parametrize(
    "indices, lifted, dist_op, dist_params, size",
    [
        (
            # `size`-less advanced boolean indexing
            (np.r_[True, False, False, True],),
            True,
            uniform,
            (
                (0.1 - 1e-5) * np.arange(4).astype(dtype=config.floatX),
                0.1 * np.arange(4).astype(dtype=config.floatX),
            ),
            (),
        ),
        (
            # `size`-only advanced boolean indexing
            (np.r_[True, False, False, True],),
            True,
            uniform,
            (
                np.array(0.9 - 1e-5, dtype=config.floatX),
                np.array(0.9, dtype=config.floatX),
            ),
            (4,),
        ),
        (
            # `size`-only slice
            (slice(4, -6, -1),),
            True,
            uniform,
            (
                np.array(0.9 - 1e-5, dtype=config.floatX),
                np.array(0.9, dtype=config.floatX),
            ),
            (5, 2),
        ),
        (
            (slice(1, None), [0, 2]),
            True,
            normal,
            (
                np.array([1, 10, 100], dtype=config.floatX),
                np.array([1e-5, 2e-5, 3e-5], dtype=config.floatX),
            ),
            (4, 3),
        ),
        (
            (np.array([1]), 0),
            True,
            normal,
            (
                np.array([[-1, 20], [300, -4000]], dtype=config.floatX),
                np.array([[1e-6, 2e-6]], dtype=config.floatX),
            ),
            (3, 2, 2),
        ),
        # A multi-dimensional case
        (
            (np.array([1]), 0),
            False,
            multivariate_normal,
            (
                np.array([[-1, 20], [300, -4000]], dtype=config.floatX),
                np.eye(2).astype(config.floatX) * 1e-6,
            ),
            (),
        ),
        # Only one distribution parameter
        (
            (0,),
            True,
            poisson,
            (np.array([[1, 2], [3, 4]], dtype=config.floatX),),
            (3, 2, 2),
        ),
    ],
)
@config.change_flags(compute_test_value_opt="raise", compute_test_value="raise")
def test_Subtensor_lift(indices, lifted, dist_op, dist_params, size):

    rng = shared(np.random.RandomState(1233532), borrow=False)

    dist_params_tt = []
    for p in dist_params:
        p_tt = aet.as_tensor(p).type()
        p_tt.tag.test_value = p
        dist_params_tt.append(p_tt)

    size_tt = []
    for s in size:
        s_tt = iscalar()
        s_tt.tag.test_value = s
        size_tt.append(s_tt)

    from aesara.tensor.subtensor import as_index_constant

    indices_tt = ()
    for i in indices:
        i_tt = as_index_constant(i)
        if not isinstance(i_tt, slice):
            i_tt.tag.test_value = i
        indices_tt += (i_tt,)

    dist_st = dist_op(*dist_params_tt, size=size_tt, rng=rng)[indices_tt]

    f_inputs = [
        p
        for p in dist_params_tt + size_tt + list(indices_tt)
        if not isinstance(p, (slice, Constant))
    ]

    mode = Mode(
        "py", EquilibriumOptimizer([local_subtensor_rv_lift], max_use_ratio=100)
    )

    f_opt = function(
        f_inputs,
        dist_st,
        mode=mode,
    )

    (new_out,) = f_opt.maker.fgraph.outputs

    if lifted:
        assert isinstance(new_out.owner.op, RandomVariable)
        assert all(
            isinstance(i.owner.op, (AdvancedSubtensor, AdvancedSubtensor1, Subtensor))
            for i in new_out.owner.inputs[3:]
            if i.owner
        )
    else:
        assert isinstance(
            new_out.owner.op, (AdvancedSubtensor, AdvancedSubtensor1, Subtensor)
        )
        return

    f_base = function(
        f_inputs,
        dist_st,
        mode=no_mode,
    )

    arg_values = [p.get_test_value() for p in f_inputs]
    res_base = f_base(*arg_values)
    res_opt = f_opt(*arg_values)

    np.testing.assert_allclose(res_base, res_opt, rtol=1e-3)


def test_Subtensor_lift_restrictions():
    rng = shared(np.random.RandomState(1233532), borrow=False)

    std = vector("std")
    std.tag.test_value = np.array([1e-5, 2e-5, 3e-5], dtype=config.floatX)
    x = normal(aet.arange(2), aet.ones(2), rng=rng)
    y = x[1]
    # The non-`Subtensor` client depends on the RNG state, so we can't perform
    # the lift
    z = x - y

    fg = FunctionGraph([rng], [z], clone=False)
    _ = EquilibriumOptimizer([local_subtensor_rv_lift], max_use_ratio=100).apply(fg)

    subtensor_node = fg.outputs[0].owner.inputs[1].owner.inputs[0].owner
    assert subtensor_node == y.owner
    assert isinstance(subtensor_node.op, Subtensor)
    assert subtensor_node.inputs[0].owner.op == normal

    # The non-`Subtensor` client doesn't depend on the RNG state, so we can
    # perform the lift
    z = aet.ones(x.shape) - x[1]

    fg = FunctionGraph([rng], [z], clone=False)
    EquilibriumOptimizer([local_subtensor_rv_lift], max_use_ratio=100).apply(fg)

    rv_node = fg.outputs[0].owner.inputs[1].owner.inputs[0].owner
    assert rv_node.op == normal
    assert isinstance(rv_node.inputs[-1].owner.op, Subtensor)
    assert isinstance(rv_node.inputs[-2].owner.op, Subtensor)


def test_Dimshuffle_lift_restrictions():
    rng = shared(np.random.RandomState(1233532), borrow=False)

    x = normal(aet.arange(2).reshape((2,)), 100, size=(2, 2, 2), rng=rng)
    y = x.dimshuffle(1, 0, 2)
    # The non-`Dimshuffle` client depends on the RNG state, so we can't
    # perform the lift
    z = x - y

    fg = FunctionGraph([rng], [z], clone=False)
    _ = EquilibriumOptimizer([local_dimshuffle_rv_lift], max_use_ratio=100).apply(fg)

    dimshuffle_node = fg.outputs[0].owner.inputs[1].owner
    assert dimshuffle_node == y.owner
    assert isinstance(dimshuffle_node.op, DimShuffle)
    assert dimshuffle_node.inputs[0].owner.op == normal

    # The non-`Dimshuffle` client doesn't depend on the RNG state, so we can
    # perform the lift
    z = aet.ones(x.shape) - y

    fg = FunctionGraph([rng], [z], clone=False)
    EquilibriumOptimizer([local_dimshuffle_rv_lift], max_use_ratio=100).apply(fg)

    rv_node = fg.outputs[0].owner.inputs[1].owner
    assert rv_node.op == normal
    assert isinstance(rv_node.inputs[-1].owner.op, DimShuffle)
    assert isinstance(rv_node.inputs[-2].owner.op, DimShuffle)
