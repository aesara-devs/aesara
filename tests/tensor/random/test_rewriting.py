import numpy as np
import pytest

import aesara.tensor as at
from aesara import config, shared
from aesara.compile.function import function
from aesara.compile.mode import Mode
from aesara.graph.basic import Constant
from aesara.graph.fg import FunctionGraph
from aesara.graph.rewriting.basic import EquilibriumGraphRewriter
from aesara.graph.rewriting.db import RewriteDatabaseQuery
from aesara.tensor.elemwise import DimShuffle
from aesara.tensor.random.basic import (
    dirichlet,
    multinomial,
    multivariate_normal,
    normal,
    poisson,
    uniform,
)
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.random.rewriting import (
    local_dimshuffle_rv_lift,
    local_rv_size_lift,
    local_subtensor_rv_lift,
)
from aesara.tensor.subtensor import AdvancedSubtensor, AdvancedSubtensor1, Subtensor
from aesara.tensor.type import iscalar, vector


no_mode = Mode("py", RewriteDatabaseQuery(include=[], exclude=[]))


def apply_local_rewrite_to_rv(
    rewrite, op_fn, dist_op, dist_params, size, rng, name=None
):
    dist_params_at = []
    for p in dist_params:
        p_at = at.as_tensor(p).type()
        p_at.tag.test_value = p
        dist_params_at.append(p_at)

    size_at = []
    for s in size:
        s_at = iscalar()
        s_at.tag.test_value = s
        size_at.append(s_at)

    dist_st = op_fn(dist_op(*dist_params_at, size=size_at, rng=rng, name=name))

    f_inputs = [
        p for p in dist_params_at + size_at if not isinstance(p, (slice, Constant))
    ]

    mode = Mode("py", EquilibriumGraphRewriter([rewrite], max_use_ratio=100))

    f_rewritten = function(
        f_inputs,
        dist_st,
        mode=mode,
    )

    (new_out,) = f_rewritten.maker.fgraph.outputs

    return new_out, f_inputs, dist_st, f_rewritten


def test_inplace_rewrites():
    out = normal(0, 1)
    out.owner.inputs[0].default_update = out.owner.outputs[0]

    assert out.owner.op.inplace is False

    f = function(
        [],
        out,
        mode="FAST_RUN",
    )

    (new_out, new_rng) = f.maker.fgraph.outputs
    assert new_out.type == out.type
    assert isinstance(new_out.owner.op, type(out.owner.op))
    assert new_out.owner.op.inplace is True
    assert all(
        np.array_equal(a.data, b.data)
        for a, b in zip(new_out.owner.inputs[2:], out.owner.inputs[2:])
    )
    assert np.array_equal(new_out.owner.inputs[1].data, [])


def test_inplace_rewrites_extra_props():
    class Test(RandomVariable):
        name = "test"
        ndim_supp = 0
        ndims_params = [0]
        __props__ = ("name", "ndim_supp", "ndims_params", "dtype", "inplace", "extra")
        dtype = "floatX"
        _print_name = ("Test", "\\operatorname{Test}")

        def __init__(self, extra, *args, **kwargs):
            self.extra = extra
            super().__init__(*args, **kwargs)

        def make_node(self, rng, size, dtype, sigma):
            return super().make_node(rng, size, dtype, sigma)

        def rng_fn(self, rng, sigma, size):
            return rng.normal(scale=sigma, size=size)

    out = Test(extra="some value")(1)
    out.owner.inputs[0].default_update = out.owner.outputs[0]

    assert out.owner.op.inplace is False

    f = function(
        [],
        out,
        mode="FAST_RUN",
    )

    (new_out, new_rng) = f.maker.fgraph.outputs
    assert new_out.type == out.type
    assert isinstance(new_out.owner.op, type(out.owner.op))
    assert new_out.owner.op.inplace is True
    assert new_out.owner.op.extra == out.owner.op.extra
    assert all(
        np.array_equal(a.data, b.data)
        for a, b in zip(new_out.owner.inputs[2:], out.owner.inputs[2:])
    )
    assert np.array_equal(new_out.owner.inputs[1].data, [])


@config.change_flags(compute_test_value="raise")
@pytest.mark.parametrize(
    "dist_op, dist_params, size",
    [
        (
            normal,
            [
                np.array(1.0, dtype=config.floatX),
                np.array(5.0, dtype=config.floatX),
            ],
            [],
        ),
        (
            normal,
            [
                np.array([0.0, 1.0], dtype=config.floatX),
                np.array(5.0, dtype=config.floatX),
            ],
            [],
        ),
        (
            normal,
            [
                np.array([0.0, 1.0], dtype=config.floatX),
                np.array(5.0, dtype=config.floatX),
            ],
            [3, 2],
        ),
        (
            multivariate_normal,
            [
                np.array([[0], [10], [100]], dtype=config.floatX),
                np.diag(np.array([1e-6], dtype=config.floatX)),
            ],
            [2, 3, 3],
        ),
        (
            dirichlet,
            [np.array([[100, 1, 1], [1, 100, 1], [1, 1, 100]], dtype=config.floatX)],
            [2, 3, 3],
        ),
        (
            multinomial,
            [
                np.array([10, 20], dtype="int64"),
                np.array([[0.999, 0.001], [0.001, 0.999]], dtype=config.floatX),
            ],
            [3, 2],
        ),
    ],
)
def test_local_rv_size_lift(dist_op, dist_params, size):
    rng = shared(np.random.default_rng(1233532), borrow=False)

    new_out, f_inputs, dist_st, f_rewritten = apply_local_rewrite_to_rv(
        local_rv_size_lift,
        lambda rv: rv,
        dist_op,
        dist_params,
        size,
        rng,
    )

    assert at.get_vector_length(new_out.owner.inputs[1]) == 0


@pytest.mark.parametrize(
    "ds_order, lifted, dist_op, dist_params, size, rtol",
    [
        (
            ("x", 0),
            True,
            normal,
            (
                np.array([0.0, -100.0], dtype=np.float64),
                np.array(1e-6, dtype=np.float64),
            ),
            (),
            1e-7,
        ),
        (
            ("x",),
            True,
            normal,
            (
                np.array(-10.0, dtype=np.float64),
                np.array(1e-6, dtype=np.float64),
            ),
            (),
            1e-7,
        ),
        (
            ("x", "x", "x"),
            True,
            normal,
            (
                np.array(-10.0, dtype=np.float64),
                np.array(1e-6, dtype=np.float64),
            ),
            (),
            1e-7,
        ),
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
            (3, 2),
            1e-3,
        ),
    ],
)
@config.change_flags(compute_test_value_opt="raise", compute_test_value="raise")
def test_DimShuffle_lift(ds_order, lifted, dist_op, dist_params, size, rtol):
    rng = shared(np.random.default_rng(1233532), borrow=False)

    new_out, f_inputs, dist_st, f_rewritten = apply_local_rewrite_to_rv(
        local_dimshuffle_rv_lift,
        lambda rv: rv.dimshuffle(ds_order),
        dist_op,
        dist_params,
        size,
        rng,
    )

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
    res_rewritten = f_rewritten(*arg_values)

    np.testing.assert_allclose(res_base, res_rewritten, rtol=rtol)


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
    from aesara.tensor.subtensor import as_index_constant

    rng = shared(np.random.default_rng(1233532), borrow=False)

    indices_at = ()
    for i in indices:
        i_at = as_index_constant(i)
        if not isinstance(i_at, slice):
            i_at.tag.test_value = i
        indices_at += (i_at,)

    new_out, f_inputs, dist_st, f_rewritten = apply_local_rewrite_to_rv(
        local_subtensor_rv_lift,
        lambda rv: rv[indices_at],
        dist_op,
        dist_params,
        size,
        rng,
    )

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
    res_rewritten = f_rewritten(*arg_values)

    np.testing.assert_allclose(res_base, res_rewritten, rtol=1e-3)


def test_Subtensor_lift_restrictions():
    rng = shared(np.random.default_rng(1233532), borrow=False)

    std = vector("std")
    std.tag.test_value = np.array([1e-5, 2e-5, 3e-5], dtype=config.floatX)
    x = normal(at.arange(2), at.ones(2), rng=rng)
    y = x[1]
    # The non-`Subtensor` client depends on the RNG state, so we can't perform
    # the lift
    z = x - y

    fg = FunctionGraph([rng], [z], clone=False)
    _ = EquilibriumGraphRewriter([local_subtensor_rv_lift], max_use_ratio=100).apply(fg)

    subtensor_node = fg.outputs[0].owner.inputs[1].owner.inputs[0].owner
    assert subtensor_node == y.owner
    assert isinstance(subtensor_node.op, Subtensor)
    assert subtensor_node.inputs[0].owner.op == normal

    z = at.ones(x.shape) - x[1]

    # We add `x` as an output to make sure that `is_rv_used_in_graph` handles
    # `"output"` "nodes" correctly.
    fg = FunctionGraph([rng], [z, x], clone=False)
    EquilibriumGraphRewriter([local_subtensor_rv_lift], max_use_ratio=100).apply(fg)

    assert fg.outputs[0] == z
    assert fg.outputs[1] == x

    # The non-`Subtensor` client doesn't depend on the RNG state, so we can
    # perform the lift
    fg = FunctionGraph([rng], [z], clone=False)
    EquilibriumGraphRewriter([local_subtensor_rv_lift], max_use_ratio=100).apply(fg)

    rv_node = fg.outputs[0].owner.inputs[1].owner.inputs[0].owner
    assert rv_node.op == normal
    assert isinstance(rv_node.inputs[-1].owner.op, Subtensor)
    assert isinstance(rv_node.inputs[-2].owner.op, Subtensor)


def test_Dimshuffle_lift_restrictions():
    rng = shared(np.random.default_rng(1233532), borrow=False)

    x = normal(at.arange(2).reshape((2,)), 100, size=(2, 2, 2), rng=rng)
    y = x.dimshuffle(1, 0, 2)
    # The non-`Dimshuffle` client depends on the RNG state, so we can't
    # perform the lift
    z = x - y

    fg = FunctionGraph([rng], [z, y], clone=False)
    _ = EquilibriumGraphRewriter([local_dimshuffle_rv_lift], max_use_ratio=100).apply(
        fg
    )

    dimshuffle_node = fg.outputs[0].owner.inputs[1].owner
    assert dimshuffle_node == y.owner
    assert isinstance(dimshuffle_node.op, DimShuffle)
    assert dimshuffle_node.inputs[0].owner.op == normal

    z = at.ones(x.shape) - y

    # We add `x` as an output to make sure that `is_rv_used_in_graph` handles
    # `"output"` "nodes" correctly.
    fg = FunctionGraph([rng], [z, x], clone=False)
    EquilibriumGraphRewriter([local_dimshuffle_rv_lift], max_use_ratio=100).apply(fg)

    assert fg.outputs[0] == z
    assert fg.outputs[1] == x

    # The non-`Dimshuffle` client doesn't depend on the RNG state, so we can
    # perform the lift
    fg = FunctionGraph([rng], [z], clone=False)
    EquilibriumGraphRewriter([local_dimshuffle_rv_lift], max_use_ratio=100).apply(fg)

    rv_node = fg.outputs[0].owner.inputs[1].owner
    assert rv_node.op == normal
    assert isinstance(rv_node.inputs[-1].owner.op, DimShuffle)
    assert isinstance(rv_node.inputs[-2].owner.op, DimShuffle)


@pytest.mark.parametrize(
    "ds_order, lifted, dist_op, dist_params, size, rtol",
    [
        (
            ("x",),
            True,
            normal,
            (
                np.array(-10.0, dtype=np.float64),
                np.array(1e-6, dtype=np.float64),
            ),
            (),
            1e-7,
        ),
        (
            (0, 1, 2),
            True,
            normal,
            (np.array(0).astype(config.floatX), np.array(1e-6).astype(config.floatX)),
            (2, 1, 2),
            1e-3,
        ),
    ],
)
def test_Dimshuffle_lift_rename(ds_order, lifted, dist_op, dist_params, size, rtol):
    rng = shared(np.random.default_rng(1233532), borrow=False)

    new_out, *_ = apply_local_rewrite_to_rv(
        local_dimshuffle_rv_lift,
        lambda rv: rv.dimshuffle(ds_order),
        dist_op,
        dist_params,
        size,
        rng,
        name="test_name",
    )

    assert new_out.name == "test_name_lifted"
