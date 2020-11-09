import numpy as np
import pytest

import theano.tensor as tt
from theano import change_flags, config, shared
from theano.compile.function import function
from theano.compile.mode import Mode
from theano.gof.fg import FunctionGraph
from theano.gof.graph import Constant
from theano.gof.opt import EquilibriumOptimizer
from theano.gof.optdb import Query
from theano.tensor.elemwise import DimShuffle
from theano.tensor.random.basic import dirichlet, multivariate_normal, normal, poisson
from theano.tensor.random.opt import lift_rv_shapes, local_dimshuffle_rv_lift


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
    tt_params = []
    for p in params:
        p_tt = tt.as_tensor(p)
        p_tt = p_tt.type()
        p_tt.tag.test_value = p
        tt_params.append(p_tt)

    tt_size = []
    for s in size:
        s_tt = tt.as_tensor(s)
        s_tt = s_tt.type()
        s_tt.tag.test_value = s
        tt_size.append(s_tt)

    rv = rv(*tt_params, size=tt_size, rng=rng)
    rv_lifted = lift_rv_shapes(rv.owner)

    # Make sure the size input is empty
    assert np.array_equal(rv_lifted.inputs[1].data, [])

    f_ref = function(
        tt_params + tt_size,
        rv,
        mode=no_mode,
    )
    f_lifted = function(
        tt_params + tt_size,
        rv_lifted.outputs[1],
        mode=no_mode,
    )
    f_ref_val = f_ref(*(params + size))
    f_lifted_val = f_lifted(*(params + size))
    assert np.array_equal(f_ref_val, f_lifted_val)


@change_flags(compute_test_value="raise")
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
@change_flags(compute_test_value_opt="raise", compute_test_value="raise")
def test_DimShuffle_lift(ds_order, lifted, dist_op, dist_params, size, rtol):

    rng = shared(np.random.RandomState(1233532), borrow=False)

    dist_params_tt = []
    for p in dist_params:
        p_tt = tt.as_tensor(p).type()
        p_tt.tag.test_value = p
        dist_params_tt.append(p_tt)

    size_tt = []
    for s in size:
        s_tt = tt.iscalar()
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


def test_Dimshuffle_lift_restrictions():
    rng = shared(np.random.RandomState(1233532), borrow=False)

    x = normal(tt.arange(2).reshape((2,)), 100, size=(2, 2, 2), rng=rng)
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
    z = tt.ones(x.shape) - y

    fg = FunctionGraph([rng], [z], clone=False)
    EquilibriumOptimizer([local_dimshuffle_rv_lift], max_use_ratio=100).apply(fg)

    rv_node = fg.outputs[0].owner.inputs[1].owner
    assert rv_node.op == normal
    assert isinstance(rv_node.inputs[-1].owner.op, DimShuffle)
    assert isinstance(rv_node.inputs[-2].owner.op, DimShuffle)
