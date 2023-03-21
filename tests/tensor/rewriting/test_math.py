import copy
import logging
import time
from io import StringIO

import numpy as np
import pytest

import aesara
import aesara.scalar as aes
import aesara.tensor as at
from aesara import pprint, shared
from aesara.compile import optdb
from aesara.compile.debugmode import DebugMode
from aesara.compile.function import function
from aesara.compile.mode import Mode, get_default_mode, get_mode
from aesara.compile.ops import DeepCopyOp, deep_copy_op
from aesara.configdefaults import config
from aesara.graph.basic import Apply, Constant, equal_computations
from aesara.graph.fg import FunctionGraph
from aesara.graph.rewriting.basic import (
    SequentialNodeRewriter,
    WalkingGraphRewriter,
    check_stack_trace,
    in2out,
    out2in,
)
from aesara.graph.rewriting.db import RewriteDatabaseQuery
from aesara.graph.rewriting.utils import is_same_graph, rewrite_graph
from aesara.misc.safe_asarray import _asarray
from aesara.tensor import inplace
from aesara.tensor.basic import Alloc, join, switch
from aesara.tensor.blas import Dot22, Gemv
from aesara.tensor.blas_c import CGemv
from aesara.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from aesara.tensor.math import Dot, MaxAndArgmax, Prod, Sum, _conj
from aesara.tensor.math import abs as at_abs
from aesara.tensor.math import add
from aesara.tensor.math import all as at_all
from aesara.tensor.math import any as at_any
from aesara.tensor.math import (
    arccosh,
    arcsinh,
    arctanh,
    bitwise_and,
    bitwise_or,
    bitwise_xor,
    conj,
    cos,
    cosh,
    deg2rad,
    dot,
    eq,
    erf,
    erfc,
    exp,
    expm1,
    floor_divide,
    ge,
    gt,
    invert,
    iround,
    le,
    log,
    log1mexp,
    log1p,
    log2,
    log10,
    lt,
)
from aesara.tensor.math import max as at_max
from aesara.tensor.math import maximum
from aesara.tensor.math import min as at_min
from aesara.tensor.math import minimum, mul, neg, neq
from aesara.tensor.math import pow as at_pow
from aesara.tensor.math import prod, rad2deg, reciprocal
from aesara.tensor.math import round as at_round
from aesara.tensor.math import sgn, sigmoid, sin, sinh, softplus, sqrt, square, sub
from aesara.tensor.math import sum as at_sum
from aesara.tensor.math import tan, tanh, true_divide, xor
from aesara.tensor.rewriting.elemwise import local_dimshuffle_lift
from aesara.tensor.rewriting.math import (
    compute_mul,
    is_1pexp,
    local_grad_log_erfc_neg,
    local_greedy_distributor,
    local_mul_canonizer,
    mul_canonizer,
    parse_mul_tree,
    perform_sigm_times_exp,
    simplify_mul,
)
from aesara.tensor.shape import Reshape, Shape_i
from aesara.tensor.type import (
    TensorType,
    cmatrix,
    dmatrices,
    dmatrix,
    dscalar,
    dtensor3,
    dvector,
    fmatrices,
    fmatrix,
    fscalar,
    ftensor4,
    fvector,
    imatrices,
    imatrix,
    iscalar,
    ivector,
    lscalar,
    matrices,
    matrix,
    scalar,
    tensor,
    tensor3,
    tensor4,
    values_eq_approx_remove_nan,
    vector,
    vectors,
    zscalar,
)
from aesara.tensor.var import TensorConstant
from tests import unittest_tools as utt


rewrite_mode = config.mode
if rewrite_mode == "FAST_COMPILE":
    rewrite_mode = "FAST_RUN"
rewrite_mode = get_mode(rewrite_mode)

dimshuffle_lift = out2in(local_dimshuffle_lift)

_stablize_rewrites = RewriteDatabaseQuery(include=["fast_run"])
_stablize_rewrites.position_cutoff = 1.51
_stablize_rewrites = optdb.query(_stablize_rewrites)

_specialize_rewrites = RewriteDatabaseQuery(include=["fast_run"])
_specialize_rewrites.position_cutoff = 2.01
_specialize_rewrites = optdb.query(_specialize_rewrites)

_fast_run_rewrites = RewriteDatabaseQuery(include=["fast_run"])
_fast_run_rewrites = optdb.query(_fast_run_rewrites)


def ds(x, y):
    return DimShuffle(x.type.broadcastable, y)(x)


def rewrite(g, level="fast_run"):
    if level == "fast_run":
        _fast_run_rewrites.rewrite(g)
    elif level == "specialize":
        _specialize_rewrites.rewrite(g)
    elif level == "stabilize":
        _stablize_rewrites.rewrite(g)
    else:
        raise ValueError(level)
    return g


def inputs(xbc=(0, 0), ybc=(0, 0), zbc=(0, 0)):
    x = TensorType(dtype="float64", shape=xbc)("x")
    y = TensorType(dtype="float64", shape=ybc)("y")
    z = TensorType(dtype="float64", shape=zbc)("z")
    return x, y, z


def test_add_canonizer_problem0():
    n_segments = 10
    label = lscalar("label")
    segment_labels = label + _asarray([0] * n_segments, dtype="int64")

    r = segment_labels * 5
    f = function([label], r)
    f(3)

    # This was crashing in the past.
    c0 = at.constant([True])
    c1 = at.constant([True])
    function([], c0 + c1)


class TestGreedyDistribute:
    def test_main(self):
        a, b, c, d, x, y, z = matrices("abcdxyz")

        # 1. ((a/x + b/y) * x * y) --> a*y + b*x
        e = (a / z + b / x) * x * z
        g = FunctionGraph([a, b, c, d, x, y, z], [e])
        mul_canonizer.rewrite(g)
        WalkingGraphRewriter(
            SequentialNodeRewriter(local_greedy_distributor), order="out_to_in"
        ).rewrite(g)
        assert str(pprint(g.outputs[0])) == "((a * x) + (b * z))"

        # 2. ((a/x + b) * x) --> a + b*x
        e = (a / x + b) * x
        g = FunctionGraph([a, b, x], [e])
        mul_canonizer.rewrite(g)
        WalkingGraphRewriter(
            SequentialNodeRewriter(local_greedy_distributor), order="out_to_in"
        ).rewrite(g)
        assert str(pprint(g.outputs[0])) == "(a + (b * x))"

    def test_kording_bug(self):
        x, y = vectors("xy")
        eps = scalar("eps")
        s = scalar("s")

        # r = mul(at.fill(x, 2.*a), x/a , (y+z) , a)
        # r = mul((x/a+y) , a, z)
        r = mul(s - 1, eps + x / s, eps + y / s, s)

        f = function([s, eps, x, y], r**2)

        s_val = np.asarray(4, dtype=config.floatX)
        eps_val = np.asarray(1.0e-6, dtype=config.floatX)
        x_val = np.asarray([1.5, 2], dtype=config.floatX)
        y_val = np.asarray([2.3, 3.1], dtype=config.floatX)

        r0 = f(s_val, eps_val, x_val, y_val)
        r1 = f(s_val, eps_val, x_val, y_val)
        r2 = f(s_val, eps_val, x_val, y_val)

        assert np.all(r0 == r1)
        assert np.all(r0 == r2)


class TestAlgebraicCanonizer:
    x, y, z = matrices("xyz")

    @pytest.mark.parametrize(
        "e, exp_g",
        [
            # ((2.0 * x) / (2.0 * y), None),
            # ((2.0 * x) / (4.0 * y), None),
            # (x / (y / z), None),
            # ((x * y) / x, None),
            # ((x / y) * (y / z) * (z / x), None),
            # ((a / b) * (b / c) * (c / d), None),
            # ((a * b) / (b * c) / (c * d), None),
            # (2 * x / 2, None),
            # (x / y / x, None),
            # ((x / x) * (y / y), None),
            (
                (-1 * x) / y / (-2 * z),
                (at.as_tensor([[0.5]], dtype="floatX") * x) / (y * z),
            ),
        ],
    )
    def test_muldiv(self, e, exp_g):
        g_rewritten = rewrite_graph(e, custom_rewrite=mul_canonizer)
        assert equal_computations([g_rewritten], [exp_g])

    def test_elemwise_multiple_inputs_rewrites(self):
        """Verify that the `AlgebraicCanonizer` merges sequential ``Elemwise({mul,add})``."""
        # Test with and without DimShuffle
        shp = (5, 5)
        fx, fy, fz = fmatrices("xyz")
        dx, dy, dz = dmatrices("xyz")
        # fv = fvector('r').dimshuffle('x', 0)
        # dv = dvector('s').dimshuffle('x', 0)
        fxv = _asarray(np.random.random(shp), dtype="float32")
        fyv = _asarray(np.random.random(shp), dtype="float32")
        fzv = _asarray(np.random.random(shp), dtype="float32")
        # fvv = _asarray(np.random.random((shp[0]), dtype='float32').reshape(1, shp[0])
        # dxv = _asarray(np.random.random((*shp), dtype='float64')
        # dyv = _asarray(np.random.random((*shp), dtype='float64')
        # dzv = _asarray(np.random.random((*shp), dtype='float64')
        # dvv = _asarray(np.random.random((shp[0]), dtype='float64').reshape(1, shp[0])
        cases = [
            (fx + fy, (fx, fy), (fxv, fyv), 1, "float32"),
            (fx * fy, (fx, fy), (fxv, fyv), 1, "float32"),
            (fx + fy + fz, (fx, fy, fz), (fxv, fyv, fzv), 1, "float32"),
            # (dx+dy+dz,(dx,dy,dz),(dxv,dyv,dzv),1,'float64'),
            (fx * fy * fz, (fx, fy, fz), (fxv, fyv, fzv), 1, "float32"),
            # (dx*dy*dz,(dx,dy,dz),(dxv,dyv,dzv),1,'float64'),
            # (fx*fy*(fx+fy+fz),(fx,fy,fz),(fxv,fyv,fzv),2,'float32'),
            # (dx*dy*(dx+dy+dz),(dx,dy,dz),(dxv,dyv,dzv),2,'float64'),
            # (fx*fy*(fx+fy+dz),(fx,fy,dz),(dxv,dyv,dzv),2,'float64'),  # check mixed type add
            # (dz*fy*(fx+fy),(fx,fy,dz),(dxv,dyv,dzv),2,'float64'),  # check mixed type mul
            # check with dimshuffle of constant
            (
                fx + fy + fz + 2,
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                {
                    "custom": "float32",
                    "numpy+floatX": config.floatX,
                    "numpy": "float64",
                },
            ),
            (
                fx * fy * fz * 2,
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                {
                    "custom": "float32",
                    "numpy+floatX": config.floatX,
                    "numpy": "float64",
                },
            ),
            # (2+fx+fy+fz,(fx,fy,fz),(fxv,fyv,fzv),1,'float32'),
            # (2*fx*fy*fz,(fx,fy,fz),(fxv,fyv,fzv),1,'float32'),
            (
                2 + fx + fy + fz + 2,
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                {
                    "custom": "float32",
                    "numpy+floatX": config.floatX,
                    "numpy": "float64",
                },
            ),
            (
                2 * fx * fy * fz * 2,
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                {
                    "custom": "float32",
                    "numpy+floatX": config.floatX,
                    "numpy": "float64",
                },
            ),
            # (fx*fy*2*(fx+fy+fz),(fx,fy,fz),(fxv,fyv,fzv),2,'float32'),
            # (fx*fy*(2+fx+fy+fz),(fx,fy,fz),(fxv,fyv,fzv),2,'float32'),
            (
                fx * fy * 2 * (fx + fy + fz + 2),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                2,
                {
                    "custom": "float32",
                    "numpy+floatX": config.floatX,
                    "numpy": "float64",
                },
            ),
            # check with broadcast of row
            # (fx+fy+fz+fv,(fx,fy,fz,fv),(fxv,fyv,fzv,fvv),1,'float32'),
            # (fx*fy*fz*fv,(fx,fy,fz,fv),(fxv,fyv,fzv,fvv),1,'float32'),
            # (fv+fx+fy+fz,(fx,fy,fz,fv),(fxv,fyv,fzv,fvv),1,'float32'),
            # (fv*fx*fy*fz,(fx,fy,fz,fv),(fxv,fyv,fzv,fvv),1,'float32'),
            # (fx*fy*fv*(fx+fy+fz),(fx,fy,fz,fv),(fxv,fyv,fzv,fvv),2,'float32'),
            # (fx*fy*(fv+fx+fy+fz),(fx,fy,fz,fv),(fxv,fyv,fzv,fvv),2,'float32'),
            # (fx*fy*fv*(fv+fx+fy+fz),(fx,fy,fz,fv),(fxv,fyv,fzv,fvv),2,'float32'),
            # (dx+dy+dz+dv,(dx,dy,dz,dv),(dxv,dyv,dzv,dvv),1,'float64'),
            # (dx*dy*dz*dv,(dx,dy,dz,dv),(dxv,dyv,dzv,dvv),1,'float64'),
            # (dv+dx+dy+dz,(dx,dy,dz,dv),(dxv,dyv,dzv,dvv),1,'float64'),
            # (dv*dx*dy*dz,(dx,dy,dz,dv),(dxv,dyv,dzv,dvv),1,'float64'),
            # (dx*dy*dv*(dx+dy+dz),(dx,dy,dz,dv),(dxv,dyv,dzv,dvv),2,'float64'),
            # (dx*dy*(dv+dx+dy+dz),(dx,dy,dz,dv),(dxv,dyv,dzv,dvv),2,'float64'),
            # (dx*dy*dv*(dv+dx+dy+dz),(dx,dy,dz,dv),(dxv,dyv,dzv,dvv),2,'float64'),
        ]  # [10:11]
        # print cases

        # We must be sure that the `AlgebraicCanonizer` is working, but that we don't have other
        # rewrites that could hide bug in the `AlgebraicCanonizer` as `local_elemwise_fusion`
        mode = get_default_mode()
        rewrites = RewriteDatabaseQuery(["canonicalize"])
        rewrites = rewrites.excluding("local_elemwise_fusion")
        mode = mode.__class__(linker=mode.linker, optimizer=rewrites)
        for id, [g, sym_inputs, val_inputs, nb_elemwise, out_dtype] in enumerate(cases):
            if isinstance(out_dtype, dict):
                out_dtype = out_dtype[config.cast_policy]
            f = function(
                list(sym_inputs),
                g,
                mode=mode,
            )

            out = f(*val_inputs)
            assert len(f.maker.fgraph.toposort()) == nb_elemwise
            assert out_dtype == out.dtype

    @pytest.mark.skip(
        reason="Current implementation of AlgebraicCanonizer does not implement all cases."
    )
    def test_elemwise_multiple_inputs_rewrites_2(self):
        """Verify that the `AlgebraicCanonizer` merges sequential ``Elemwise({mul,add})``.

        This part are that case that should have been done, but that are not implemented.
        """

        # Test with and without `DimShuffle`
        shp = (5, 5)
        fx, fy, fz = fmatrices("xyz")
        dx, dy, dz = dmatrices("xyz")
        fv = fvector("r").dimshuffle("x", 0)
        dv = dvector("s").dimshuffle("x", 0)
        fxv = _asarray(np.random.random(shp), dtype="float32")
        fyv = _asarray(np.random.random(shp), dtype="float32")
        fzv = _asarray(np.random.random(shp), dtype="float32")
        fvv = _asarray(np.random.random(shp[0]), dtype="float32").reshape(1, shp[0])
        dxv = _asarray(np.random.random(shp), dtype="float64")
        dyv = _asarray(np.random.random(shp), dtype="float64")
        dzv = _asarray(np.random.random(shp), dtype="float64")
        dvv = _asarray(np.random.random(shp[0]), dtype="float64").reshape(1, shp[0])
        cases = [
            (fx + fy, (fx, fy), (fxv, fyv), 1, "float32"),
            (fx * fy, (fx, fy), (fxv, fyv), 1, "float32"),
            (fx + fy + fz, (fx, fy, fz), (fxv, fyv, fzv), 1, "float32"),
            (dx + dy + dz, (dx, dy, dz), (dxv, dyv, dzv), 1, "float64"),
            (fx * fy * fz, (fx, fy, fz), (fxv, fyv, fzv), 1, "float32"),
            (dx * dy * dz, (dx, dy, dz), (dxv, dyv, dzv), 1, "float64"),
            (fx * fy * (fx + fy + fz), (fx, fy, fz), (fxv, fyv, fzv), 2, "float32"),
            (dx * dy * (dx + dy + dz), (dx, dy, dz), (dxv, dyv, dzv), 2, "float64"),
            (
                fx * fy * (fx + fy + dz),
                (fx, fy, dz),
                (dxv, dyv, dzv),
                2,
                "float64",
            ),  # check mixed type add
            (
                dz * fy * (fx + fy),
                (fx, fy, dz),
                (dxv, dyv, dzv),
                2,
                "float64",
            ),  # check mixed type mul
            # check with dimshuffle of constant
            (fx + fy + fz + 2, (fx, fy, fz), (fxv, fyv, fzv), 1, "float32"),
            (fx * fy * fz * 2, (fx, fy, fz), (fxv, fyv, fzv), 1, "float32"),
            (2 + fx + fy + fz, (fx, fy, fz), (fxv, fyv, fzv), 1, "float32"),
            (2 * fx * fy * fz, (fx, fy, fz), (fxv, fyv, fzv), 1, "float32"),
            (2 + fx + fy + fz + 2, (fx, fy, fz), (fxv, fyv, fzv), 1, "float32"),
            (2 * fx * fy * fz * 2, (fx, fy, fz), (fxv, fyv, fzv), 1, "float32"),
            (fx * fy * 2 * (fx + fy + fz), (fx, fy, fz), (fxv, fyv, fzv), 2, "float32"),
            (fx * fy * (2 + fx + fy + fz), (fx, fy, fz), (fxv, fyv, fzv), 2, "float32"),
            (
                fx * fy * 2 * (fx + fy + fz + 2),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                2,
                "float32",
            ),
            # check with broadcast of row
            (fx + fy + fz + fv, (fx, fy, fz, fv), (fxv, fyv, fzv, fvv), 1, "float32"),
            (fx * fy * fz * fv, (fx, fy, fz, fv), (fxv, fyv, fzv, fvv), 1, "float32"),
            (fv + fx + fy + fz, (fx, fy, fz, fv), (fxv, fyv, fzv, fvv), 1, "float32"),
            (fv * fx * fy * fz, (fx, fy, fz, fv), (fxv, fyv, fzv, fvv), 1, "float32"),
            (
                fx * fy * fv * (fx + fy + fz),
                (fx, fy, fz, fv),
                (fxv, fyv, fzv, fvv),
                2,
                "float32",
            ),
            (
                fx * fy * (fv + fx + fy + fz),
                (fx, fy, fz, fv),
                (fxv, fyv, fzv, fvv),
                2,
                "float32",
            ),
            (
                fx * fy * fv * (fv + fx + fy + fz),
                (fx, fy, fz, fv),
                (fxv, fyv, fzv, fvv),
                2,
                "float32",
            ),
            (dx + dy + dz + dv, (dx, dy, dz, dv), (dxv, dyv, dzv, dvv), 1, "float64"),
            (dx * dy * dz * dv, (dx, dy, dz, dv), (dxv, dyv, dzv, dvv), 1, "float64"),
            (dv + dx + dy + dz, (dx, dy, dz, dv), (dxv, dyv, dzv, dvv), 1, "float64"),
            (dv * dx * dy * dz, (dx, dy, dz, dv), (dxv, dyv, dzv, dvv), 1, "float64"),
            (
                dx * dy * dv * (dx + dy + dz),
                (dx, dy, dz, dv),
                (dxv, dyv, dzv, dvv),
                2,
                "float64",
            ),
            (
                dx * dy * (dv + dx + dy + dz),
                (dx, dy, dz, dv),
                (dxv, dyv, dzv, dvv),
                2,
                "float64",
            ),
            (
                dx * dy * dv * (dv + dx + dy + dz),
                (dx, dy, dz, dv),
                (dxv, dyv, dzv, dvv),
                2,
                "float64",
            ),
        ]  # [10:11]
        # print cases

        # We must be sure that the AlgebraicCanonizer is working, but that we don't have other
        # rewrites that could hide bugs in the `AlgebraicCanonizer` as `local_elemwise_fusion`
        mode = get_default_mode()
        mode._optimizer = RewriteDatabaseQuery(["canonicalize"])
        mode._optimizer = mode._optimizer.excluding("local_elemwise_fusion")
        for id, [g, sym_inputs, val_inputs, nb_elemwise, out_dtype] in enumerate(cases):
            f = function(
                list(sym_inputs),
                g,
                mode=mode,
            )

            out = f(*val_inputs)
            assert len(f.maker.fgraph.toposort()) == nb_elemwise
            assert out_dtype == out.dtype

    def test_mul_div_cases(self):
        """
        TODO

            x / x -> 1
            (x * y) / x -> y
            x / y / x -> 1 / y
            x / y / z -> x / (y * z)
            x / (y / z) -> (x * z) / y
            (a / b) * (b / c) * (c / d) -> a / d
            (2.0 * x) / (4.0 * y) -> (0.5 * x) / y
            2 * x / 2 -> x

        """
        # with and without DimShuffle
        # TODO: with DimShuffle

        shp = (3, 3)
        fx, fy, fz, fw = fmatrices("xyzw")
        dx, dy, dz, dw = dmatrices("xyzw")
        fv = fvector("r").dimshuffle("x", 0)
        dv = dvector("s").dimshuffle("x", 0)
        fxv = _asarray(np.random.random(shp), dtype="float32")
        fyv = _asarray(np.random.random(shp), dtype="float32")
        fzv = _asarray(np.random.random(shp), dtype="float32")
        fwv = _asarray(np.random.random(shp), dtype="float32")
        fvv = _asarray(np.random.random(shp[0]), dtype="float32").reshape(1, shp[0])
        dxv = _asarray(np.random.random(shp), dtype="float64")
        dyv = _asarray(np.random.random(shp), dtype="float64")
        dzv = _asarray(np.random.random(shp), dtype="float64")
        dwv = _asarray(np.random.random(shp), dtype="float64")
        dvv = _asarray(np.random.random(shp[0]), dtype="float64").reshape(1, shp[0])

        # We must be sure that the `AlgebraicCanonizer` is working, but that we don't have other
        # rewrites that could hide bugs in the `AlgebraicCanonizer` as `local_elemwise_fusion`
        mode = get_default_mode()

        rewrite_query = RewriteDatabaseQuery(["canonicalize"])
        rewrite_query = rewrite_query.including("ShapeOpt", "local_fill_to_alloc")
        rewrite_query = rewrite_query.excluding("local_elemwise_fusion")
        mode = mode.__class__(linker=mode.linker, optimizer=rewrite_query)
        # test x / x -> 1
        for id, (g, sym_inputs, val_inputs, out_dtype) in enumerate(
            [
                (fx / fx, [fx], [fxv], "float32"),
                (dx / dx, [dx], [dxv], "float64"),
                (fv / fv, [fv], [fvv], "float32"),
                (dv / dv, [dv], [dvv], "float64"),
            ]
        ):
            f = function(list(sym_inputs), g, mode=mode)
            out = f(*val_inputs)
            assert (out == np.ones(shp, dtype=out_dtype)).all()
            topo = f.maker.fgraph.toposort()
            if sym_inputs[0].broadcastable[0]:
                assert len(topo) == 2
                assert isinstance(topo[0].op, Shape_i)
                assert isinstance(topo[1].op, Alloc)
            else:
                assert len(topo) == 3
                assert isinstance(topo[0].op, Shape_i)
                assert isinstance(topo[1].op, Shape_i)
                assert isinstance(topo[2].op, Alloc)
            assert out_dtype == out.dtype

        # test (x * y) / x -> y
        for id, (g, sym_inputs, val_inputs, nb_elemwise, out_dtype) in enumerate(
            [
                ((dx * dy) / dx, [dx, dy], [dxv, dyv], 0, "float64"),
                ((fx * fy) / fx, [fx, fy], [fxv, fyv], 0, "float32"),
                ((dv * dy) / dv, [dv, dy], [dvv, dyv], 0, "float64"),
                ((fv * fy) / fv, [fv, fy], [fvv, fyv], 0, "float32"),
                # must broadcast as there is a dimshuffle in the computation
                ((dx * dv) / dx, [dx, dv], [dxv, dvv], 1, "float64"),
                # topo: [Elemwise{second,no_inplace}(x, <TensorType(float64, row)>)]
                ((fx * fv) / fx, [fx, fv], [fxv, fvv], 1, "float32")
                # topo: [Elemwise{second,no_inplace}(x, <TensorType(float32, row)>)]
            ]
        ):
            f = function(list(sym_inputs), g, mode=mode)
            out = f(*val_inputs)
            assert out_dtype == out.dtype
            utt.assert_allclose(out, val_inputs[1])
            topo = f.maker.fgraph.toposort()
            assert not any(node.op == at.true_divide for node in topo)

        # test x / y / x -> 1 / y
        for id, (g, sym_inputs, val_inputs, nb_elemwise, out_dtype) in enumerate(
            [
                ((dx / dy) / dx, [dx, dy], [dxv, dyv], 1, "float64"),
                ((fx / fy) / fx, [fx, fy], [fxv, fyv], 1, "float32"),
                ((dv / dy) / dv, [dv, dy], [dvv, dyv], 1, "float64"),
                ((fv / fy) / fv, [fv, fy], [fvv, fyv], 1, "float32"),
                # must broadcast as their is a dimshuffle in the computation
                ((dx / dv) / dx, [dx, dv], [dxv, dvv], 1, "float64"),
                # topo: [Shape_i, Shape_i, Elemwise{reciprocal,no_inplace}(<TensorType(float64, row)>), Alloc]
                ((fx / fv) / fx, [fx, fv], [fxv, fvv], 1, "float32"),
                # topo: [Shape_i, Shape_i, Elemwise{reciprocal,no_inplace}(<TensorType(float32, row)>), Alloc]
            ]
        ):
            f = function(list(sym_inputs), g, mode=mode)
            out = f(*val_inputs)
            utt.assert_allclose(out, (1 / val_inputs[1]))
            topo = f.maker.fgraph.toposort()
            elem = [t for t in topo if isinstance(t.op, Elemwise)]
            assert len(elem) == nb_elemwise
            assert isinstance(elem[0].op, (Elemwise,))
            assert isinstance(
                elem[0].op.scalar_op,
                (aes.basic.Reciprocal, aes.basic.TrueDivide),
            )
            assert out_dtype == out.dtype

        # test (a / b) * (b / c) * (c / d) -> a / d
        for id, (g, sym_inputs, val_inputs, out_dtype) in enumerate(
            [
                (
                    (dx / dy) * (dy / dz) * (dz / dw),
                    [dx, dy, dz, dw],
                    [dxv, dyv, dzv, dwv],
                    "float64",
                ),
                (
                    (fx / fy) * (fy / fz) * (fz / fw),
                    [fx, fy, fz, fw],
                    [fxv, fyv, fzv, fwv],
                    "float32",
                ),
                (
                    (dv / dy) * (dy / dz) * (dz / dw),
                    [dv, dy, dz, dw],
                    [dvv, dyv, dzv, dwv],
                    "float64",
                ),
                (
                    (fv / fy) * (fy / fz) * (fz / fw),
                    [fv, fy, fz, fw],
                    [fvv, fyv, fzv, fwv],
                    "float32",
                ),
                (
                    (dx / dv) * (dv / dz) * (dz / dw),
                    [dx, dv, dz, dw],
                    [dxv, dvv, dzv, dwv],
                    "float64",
                ),
                (
                    (fx / fv) * (fv / fz) * (fz / fw),
                    [fx, fv, fz, fw],
                    [fxv, fvv, fzv, fwv],
                    "float32",
                ),
                (
                    (dx / dy) * (dy / dv) * (dv / dw),
                    [dx, dy, dv, dw],
                    [dxv, dyv, dvv, dwv],
                    "float64",
                ),
                (
                    (fx / fy) * (fy / fv) * (fv / fw),
                    [fx, fy, fv, fw],
                    [fxv, fyv, fvv, fwv],
                    "float32",
                ),
                (
                    (dx / dy) * (dy / dz) * (dz / dv),
                    [dx, dy, dz, dv],
                    [dxv, dyv, dzv, dvv],
                    "float64",
                ),
                (
                    (fx / fy) * (fy / fz) * (fz / fv),
                    [fx, fy, fz, fv],
                    [fxv, fyv, fzv, fvv],
                    "float32",
                ),
            ]
        ):
            f = function(list(sym_inputs), g, mode=mode)
            out = f(*val_inputs)
            utt.assert_allclose(out, (val_inputs[0] / val_inputs[3]))
            topo = f.maker.fgraph.toposort()
            assert len(topo) == 1
            assert isinstance(topo[0].op, (Elemwise,))
            assert isinstance(topo[0].op.scalar_op, aes.basic.TrueDivide)
            assert len(topo[0].inputs) == 2
            assert out_dtype == out.dtype

        # test (2.0 * x) / (4.0 * y) -> (0.5 * x) / y
        for id, (g, sym_inputs, val_inputs, out_dtype) in enumerate(
            [
                (((2.0 * dx) / (4.0 * dy)), [dx, dy], [dxv, dyv], "float64"),
                (
                    ((2.0 * fx) / (4.0 * fy)),
                    [fx, fy],
                    [fxv, fyv],
                    {
                        "custom": "float32",
                        "numpy+floatX": config.floatX,
                        "numpy": "float64",
                    },
                ),
                (((2.0 * dv) / (4.0 * dy)), [dv, dy], [dvv, dyv], "float64"),
                (
                    ((2.0 * fv) / (4.0 * fy)),
                    [fv, fy],
                    [fvv, fyv],
                    {
                        "custom": "float32",
                        "numpy+floatX": config.floatX,
                        "numpy": "float64",
                    },
                ),
                (((2.0 * dx) / (4.0 * dv)), [dx, dv], [dxv, dvv], "float64"),
                (
                    ((2.0 * fx) / (4.0 * fv)),
                    [fx, fv],
                    [fxv, fvv],
                    {
                        "custom": "float32",
                        "numpy+floatX": config.floatX,
                        "numpy": "float64",
                    },
                ),
            ]
        ):
            if isinstance(out_dtype, dict):
                out_dtype = out_dtype[config.cast_policy]
            f = function(list(sym_inputs), g, mode=mode)
            out = f(*val_inputs)
            utt.assert_allclose(out, (0.5 * val_inputs[0] / val_inputs[1]))
            topo = f.maker.fgraph.toposort()
            assert len(topo) == 2
            assert isinstance(topo[0].op, (Elemwise,))
            assert isinstance(topo[0].op.scalar_op, aes.basic.Mul)
            assert len(topo[0].inputs) == 2
            assert isinstance(topo[1].op, (Elemwise,))
            assert isinstance(topo[1].op.scalar_op, aes.basic.TrueDivide)
            assert len(topo[1].inputs) == 2
            assert out_dtype == out.dtype

        # test 2 * x / 2 -> x
        for id, (g, sym_inputs, val_inputs, out_dtype) in enumerate(
            [
                ((2 * dx) / 2, [dx], [dxv], "float64"),
                (
                    (2 * fx) / 2,
                    [fx],
                    [fxv],
                    {
                        "custom": "float32",
                        "numpy+floatX": config.floatX,
                        "numpy": "float64",
                    },
                ),
                ((2 * dv) / 2, [dv], [dvv], "float64"),
                (
                    (2 * fv) / 2,
                    [fv],
                    [fvv],
                    {
                        "custom": "float32",
                        "numpy+floatX": config.floatX,
                        "numpy": "float64",
                    },
                ),
            ]
        ):
            if isinstance(out_dtype, dict):
                out_dtype = out_dtype[config.cast_policy]
            f = function(list(sym_inputs), g, mode=mode)
            out = f(*val_inputs)
            utt.assert_allclose(out, val_inputs[0])
            topo = f.maker.fgraph.toposort()
            assert len(topo) == 1
            topo[0].op == deep_copy_op
            assert out_dtype == out.dtype

        # test x / abs(x) -> sign(x)
        for id, (g, sym_inputs, val_inputs, out_dtype) in enumerate(
            [
                (dx / abs(dx), [dx], [0.5 - dxv], "float64"),
                (fx / abs(fx), [fx], [0.5 - fxv], "float32"),
                (dx / abs(dx), [dx], [0.1 * dxv], "float64"),
                (fx / abs(fx), [fx], [0.1 * fxv], "float32"),
                (dv / abs(dv), [dv], [0.5 - dvv], "float64"),
                (fv / abs(fv), [fv], [0.5 - fvv], "float32"),
            ]
        ):
            f = function(list(sym_inputs), g, mode=mode)
            out = f(*val_inputs)
            assert np.all(np.isfinite(out))
            utt.assert_allclose(out, np.sign(val_inputs[0]))
            assert out_dtype == out.dtype
            assert len(f.maker.fgraph.toposort()) == 1

        # test (2*x) / (3*abs(x)) -> sign(x)
        for id, (g, sym_inputs, val_inputs, out_dtype) in enumerate(
            [
                ((2 * dx) / (3 * abs(dx)), [dx], [0.5 - dxv], "float64"),
                (
                    (2 * fx) / (3 * abs(fx)),
                    [fx],
                    [0.5 - fxv],
                    {
                        "custom": "float32",
                        "numpy+floatX": config.floatX,
                        "numpy": "float64",
                    },
                ),
                ((2 * dx) / (3 * abs(dx)), [dx], [0.1 * dxv], "float64"),
                (
                    (2 * fx) / (3 * abs(fx)),
                    [fx],
                    [0.1 * fxv],
                    {
                        "custom": "float32",
                        "numpy+floatX": config.floatX,
                        "numpy": "float64",
                    },
                ),
                ((2 * dv) / (3 * abs(dv)), [dv], [0.5 - dvv], "float64"),
                (
                    (2 * fv) / (3 * abs(fv)),
                    [fv],
                    [0.5 - fvv],
                    {
                        "custom": "float32",
                        "numpy+floatX": config.floatX,
                        "numpy": "float64",
                    },
                ),
            ]
        ):
            if isinstance(out_dtype, dict):
                out_dtype = out_dtype[config.cast_policy]
            f = function(list(sym_inputs), g, mode=mode)
            topo = f.maker.fgraph.toposort()
            out = f(*val_inputs)
            assert np.all(np.isfinite(out))
            utt.assert_allclose(out, np.sign(val_inputs[0]) * 2 / 3)
            assert out_dtype == out.dtype

    def test_abs_mul_div(self):
        """Test that ``4 * x / abs(2*x)`` gets "simplified" during canonicalization."""

        x = dscalar()
        # a = at.at_abs(x)

        if config.mode == "FAST_COMPILE":
            mode = get_mode("FAST_RUN").excluding("local_elemwise_fusion")
        else:
            mode = get_default_mode().excluding("local_elemwise_fusion")

        f = function([x], [(4 * x) / abs(2 * x)], mode=mode)
        f(0.1)
        f(-1)
        # Some stabilization rewrites make the output finite instead of NaN.
        # `debug_mode` will raise an error when he see NaN
        if not isinstance(mode, DebugMode):
            assert np.isfinite(f(0))

        assert len(f.maker.fgraph.toposort()) == 2
        assert f.maker.fgraph.toposort()[0].op == sgn

        f = function([x], [(4 * x) / abs(x / 2)], mode=mode)
        f(0.1)
        f(-1)
        if not isinstance(mode, DebugMode):
            assert np.isfinite(f(0))

        assert len(f.maker.fgraph.toposort()) == 2
        assert f.maker.fgraph.toposort()[0].op == sgn

    @pytest.mark.skip(
        reason="Current implementation of AlgebraicCanonizer does not "
        "implement all cases. Skip the corresponding test."
    )
    def test_multiple_case_that_fail(self):
        shp = (4, 4)
        fx, fy, fz = fmatrices("xyz")
        dx, dy, dz = dmatrices("xyz")
        fxv = _asarray(np.random.random(shp), dtype="float32")
        fyv = _asarray(np.random.random(shp), dtype="float32")
        fzv = _asarray(np.random.random(shp), dtype="float32")
        dxv = _asarray(np.random.random(shp), dtype="float32")
        dyv = _asarray(np.random.random(shp), dtype="float32")
        dzv = _asarray(np.random.random(shp), dtype="float32")
        # fvv = _asarray(np.random.random((shp[0]), dtype='float32').reshape(1, shp[0])

        mode = get_default_mode()

        rewrites = RewriteDatabaseQuery(["canonicalize"])
        rewrites = rewrites.excluding("local_elemwise_fusion")
        mode = mode.__class__(linker=mode.linker, optimizer=rewrites)
        # test fail!
        # test x / y / z -> x / (y * z)
        for g, sym_inputs, val_inputs, out_dtype in [
            ((dx / dy) / dz, [dx, dy, dz], [dxv, dyv, dzv], "float64"),
            ((fx / fy) / fz, [fx, fy, fz], [fxv, fyv, fzv], "float32"),
        ]:
            f = function(list(sym_inputs), g, mode=mode)
            out = f(*val_inputs)
            utt.assert_allclose(out, val_inputs[0] / val_inputs[1] / val_inputs[2])
            topo = f.maker.fgraph.toposort()
            assert len(topo) == 2
            assert isinstance(topo[0].op, (Elemwise,))
            assert isinstance(topo[0].op.scalar_op, aes.basic.Reciprocal)
            assert len(topo[0].inputs) == 1
            assert out_dtype == out.dtype

        # test x / (y / z) -> (x * z) / y
        for g, sym_inputs, val_inputs, out_dtype in [
            (dx / (dy / dz), [dx, dy, dz], [dxv, dyv, dzv], "float64"),
            (fx / (fy / fz), [fx, fy, fz], [fxv, fyv, fzv], "float32"),
        ]:
            f = function(list(sym_inputs), g, mode=mode)
            out = f(*val_inputs)
            utt.assert_allclose(out, val_inputs[0] / (val_inputs[1] / val_inputs[2]))
            topo = f.maker.fgraph.toposort()
            assert len(topo) == 2
            assert isinstance(topo[0].op, (Elemwise,))
            assert isinstance(topo[0].op.scalar_op, aes.basic.Reciprocal)
            assert len(topo[0].inputs) == 1
            assert out_dtype == out.dtype

    def test_canonicalize_nan(self):
        # Regression test for bug in canonicalization of NaN values.
        # This bug caused an infinite loop which was caught by the equilibrium
        # rewriter, resulting in an error log message.

        sio = StringIO()
        handler = logging.StreamHandler(sio)
        handler.setLevel(logging.ERROR)
        logging.getLogger("aesara.graph.rewriting.basic").addHandler(handler)
        try:
            x = vector()
            function([x], x + np.nan)
        finally:
            logging.getLogger("aesara.graph.rewriting.basic").removeHandler(handler)
        # Ideally this test would only catch the maxed out equilibrium
        # rewriter error message, but to be safe in case this message
        # is modified in the future, we assert that there is no error
        # at all.
        assert not sio.getvalue()

    def test_mismatching_types(self):
        a = at.as_tensor([[0.0]], dtype=np.float64)
        b = tensor("float64", shape=(None,)).dimshuffle("x", 0)
        z = add(a, b)
        # Construct a node with the wrong output `Type`
        z = Apply(
            z.owner.op, z.owner.inputs, [tensor("float64", shape=(None, None))]
        ).outputs[0]

        z_rewritten = rewrite_graph(
            z, custom_rewrite=in2out(local_mul_canonizer, name="blah")
        )
        # No rewrite was applied
        assert z_rewritten is z


def test_local_merge_abs():
    x, y, z = matrices("xyz")
    x_val = np.random.random((5, 5)).astype(config.floatX)
    y_val = np.random.random((5, 5)).astype(config.floatX)
    z_val = np.random.random((5, 5)).astype(config.floatX)
    mode = config.mode
    if mode == "FAST_COMPILE":
        mode = "FAST_RUN"
    mode = get_mode(mode).excluding("local_elemwise_fusion")

    f = function([y, z], (abs(y * z * -2)), mode=mode)
    f(y_val, z_val)
    assert isinstance(f.maker.fgraph.toposort()[1].op.scalar_op, aes.Abs)
    assert len(f.maker.fgraph.toposort()) == 2

    f = function([x, y], abs(x / y), mode=mode)
    f(x_val, y_val)
    assert isinstance(f.maker.fgraph.toposort()[1].op.scalar_op, aes.Abs)
    assert len(f.maker.fgraph.toposort()) == 2


def test_merge_abs_bugfix():
    """
    See https://groups.google.com/d/topic/theano-users/TaXfqXP2Mj0/discussion
    """
    input = matrix()
    # normalize on cols
    step1 = input / input.sum(0)
    # normalize on rows
    step2 = step1 / step1.sum(1)
    # get l1 norm
    l1_norm = at_abs(step2).sum()
    function([input], aesara.gradient.grad(l1_norm, input))


def test_mixeddiv():
    # Test that int division is preserved
    i = iscalar()
    d = dscalar()
    assert 0 == function([i, d], d * (i // (i + 1)))(3, 1.0)


def test_const_type_in_mul_canonizer():
    input = dmatrix()
    w = dmatrix()
    visb = dvector()
    hidb = dvector()
    betas = dvector()
    a = dvector()

    def sigm(x):
        return 1.0 / (1 + exp(-x))

    hid = sigm((dot(w, input) + hidb) * betas)

    vis_gauss1 = (dot(w.T, hid) + visb) * betas / (2 * a * a)
    vis_gauss2 = (dot(w.T, hid) + visb) * betas / (2.0 * a * a)

    f1 = function([input, w, visb, hidb, betas, a], vis_gauss1)
    f2 = function([input, w, visb, hidb, betas, a], vis_gauss2)

    ival = np.random.random((5, 5))
    wval = np.random.random((5, 5))
    visbval = np.random.random(5)
    hidbval = np.random.random(5)
    betaval = np.random.random(5)
    aval = np.random.random(5)

    utt.assert_allclose(
        f2(ival, wval, visbval, hidbval, betaval, aval),
        f1(ival, wval, visbval, hidbval, betaval, aval),
    )


def test_cast_in_mul_canonizer():
    x, y = vectors("xy")
    m = minimum(x, y)
    o = m.sum()
    go = at.fill(o, 1)
    e = eq(go, x)
    o1 = (1 - e) * go
    o2 = e * go
    mode = get_default_mode().excluding("fusion").including("fast_run")
    f = function([x, y], [o1, o2], mode=mode)
    nodes = f.maker.fgraph.apply_nodes
    assert (
        len(
            [
                n
                for n in nodes
                if isinstance(getattr(n.op, "scalar_op", None), aes.Identity)
            ]
        )
        == 0
    )
    assert len([n for n in nodes if isinstance(n.op.scalar_op, aes.Cast)]) == 1
    f([1], [1])


class TestFusion:
    rewrites = RewriteDatabaseQuery(
        include=[
            "local_elemwise_fusion",
            "composite_elemwise_fusion",
            "canonicalize",
            "inplace",
        ],
        exclude=["cxx_only", "BlasOpt"],
    )
    mode = Mode(get_default_mode().linker, rewrites)
    _shared = staticmethod(shared)
    topo_exclude = ()

    def do(self, mode, shared_fn, shp, nb_repeat=1, assert_len_topo=True, slice=None):
        """
        param shared_fn: if None, will use function
        verify that the elemwise fusion work
        Test with and without DimShuffle
        """

        # TODO: disable the canonizer?
        def my_init(shp, dtype="float64", num=0):
            ret = np.zeros(shp, dtype=dtype) + num
            return ret

        fw, fx, fy, fz = (
            tensor(dtype="float32", shape=(None,) * len(shp), name=n) for n in "wxyz"
        )
        dw, dx, dy, dz = (
            tensor(dtype="float64", shape=(None,) * len(shp), name=n) for n in "wxyz"
        )
        ix, iy, iz = (
            tensor(dtype="int32", shape=(None,) * len(shp), name=n) for n in "xyz"
        )
        fv = fvector("v")
        fs = fscalar("s")

        fwv = my_init(shp, "float32", 1)
        fxv = my_init(shp, "float32", 2)
        fyv = my_init(shp, "float32", 3)
        fzv = my_init(shp, "float32", 4)
        fvv = _asarray(np.random.random(shp[0]), dtype="float32")
        fsv = np.asarray(np.random.random(), dtype="float32")
        dwv = my_init(shp, "float64", 5)
        ixv = _asarray(my_init(shp, num=60), dtype="int32")
        iyv = _asarray(my_init(shp, num=70), dtype="int32")
        izv = _asarray(my_init(shp, num=70), dtype="int32")
        fwx = fw + fx
        ftanx = tan(fx)
        cases = [
            (
                fx + fy + fz,
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv + fyv + fzv,
                "float32",
            ),  # 0
            (
                fx * fy * fz,
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv * fyv * fzv,
                "float32",
            ),  # 1
            (
                fx + fy * fz,
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv + fyv * fzv,
                "float32",
            ),  # 2
            (
                fx * fy + fz,
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv * fyv + fzv,
                "float32",
            ),  # 3
            (
                fw + fx + fy + fz,
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv + fxv + fyv + fzv,
                "float32",
            ),
            (
                (fw + fx) + (fy + fz),
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv + fxv + fyv + fzv,
                "float32",
            ),  # 5
            (
                ((fw + fx) + fy) + fz,
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv + fxv + fyv + fzv,
                "float32",
            ),
            (
                (fw + (fx + fy)) + fz,
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv + fxv + fyv + fzv,
                "float32",
            ),
            (
                (fw + (fx + fy) + fz),
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv + fxv + fyv + fzv,
                "float32",
            ),
            (
                fw + (fx + (fy + fz)),
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv + fxv + fyv + fzv,
                "float32",
            ),
            (
                (fw + fx) + (fy + fz),
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv + fxv + fyv + fzv,
                "float32",
            ),  # 10
            (
                fw * fx * fy * fz,
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv * fxv * fyv * fzv,
                "float32",
            ),
            (
                fw + fx * fy * fz,
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv + fxv * fyv * fzv,
                "float32",
            ),
            (
                fx + fy * fz * fx,
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv + fyv * fzv * fxv,
                "float32",
            ),
            (
                fx * fy + fz + fy,
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv * fyv + fzv + fyv,
                "float32",
            ),
            (
                fx * fy * fz * fw + fx + fy + fz + fw,
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fxv * fyv * fzv * fwv + fxv + fyv + fzv + fwv,
                "float32",
            ),  # 15
            # test with constant
            (
                (fw + fx) + (fy + fz) + 2.0,
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv + fxv + fyv + fzv + 2,
                "float32",
            ),
            (
                ((fw + fx) + 2.0 + fy) + fz,
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv + fxv + fyv + fzv + 2,
                "float32",
            ),
            (
                (fw + (fx + 2.0 + fy)) + fz,
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv + fxv + fyv + fzv + 2,
                "float32",
            ),
            (
                (fw + (fx + fy) + 2 + fz),
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv + fxv + fyv + fzv + 2,
                "float32",
            ),
            (
                fw + (fx + (fy + fz) + 2.0),
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv + fxv + fyv + fzv + 2,
                "float32",
            ),  # 20
            (
                2 + (fw + fx) + (fy + fz),
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv + fxv + fyv + fzv + 2,
                "float32",
            ),
            # mix float32 and float64
            (
                2 + (dw + fx) + (fy + fz),
                (dw, fx, fy, fz),
                (dwv, fxv, fyv, fzv),
                1,
                dwv + fxv + fyv + fzv + 2,
                "float64",
            ),
            (
                2 + (fw + dw) + (fy + fz),
                (fw, dw, fy, fz),
                (fwv, dwv, fyv, fzv),
                1,
                fwv + dwv + fyv + fzv + 2,
                "float64",
            ),
            (
                2 + (fw + fx) + (dw + fz),
                (fw, fx, dw, fz),
                (fwv, fxv, dwv, fzv),
                1,
                fwv + fxv + dwv + fzv + 2,
                "float64",
            ),
            (
                2 + (fw + fx) + (fy + dw),
                (fw, fx, fy, dw),
                (fwv, fxv, fyv, dwv),
                1,
                fwv + fxv + fyv + dwv + 2,
                "float64",
            ),  # 25
            # test when their is other op then elemwise.
            (
                (fwx.sum()) + (fwx) + (fy + fz),
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                4,
                (fwv + fxv).sum() + fwv + fxv + fyv + fzv,
                "float32",
            ),
            # test other elemwise op
            (
                fx + fy + cos(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv + fyv + np.cos(fzv),
                "float32",
            ),
            (
                fx + fy + cosh(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv + fyv + np.cosh(fzv),
                "float32",
            ),
            (
                fx + fy + abs(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv + fyv + np.absolute(fzv),
                "float32",
            ),
            (
                ix + iy + abs(iz),
                (ix, iy, iz),
                (ixv, iyv, izv),
                1,
                ixv + iyv + np.absolute(izv),
                "int32",
            ),  # 30
            (
                fx + fy + log(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv + fyv + np.log(fzv),
                "float32",
            ),
            (
                fx + fy + log2(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv + fyv + np.log2(fzv),
                "float32",
            ),
            (
                fx + fy + log10(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv + fyv + np.log10(fzv),
                "float32",
            ),
            (
                fx + fy**fz,
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv + fyv**fzv,
                "float32",
            ),  # pow
            (
                fx + fy + exp(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv + fyv + np.exp(fzv),
                "float32",
            ),  # 35
            (
                fx - fy - fz,
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - fyv - fzv,
                "float32",
            ),
            (
                fx - (fy / fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - (fyv / fzv),
                "float32",
            ),
            (
                fx - true_divide(fy, 2),
                (fx, fy),
                (fxv, fyv),
                1,
                fxv - (fyv / 2),
                "float32",
            ),
            (
                fx - true_divide(fy, fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - (fyv / fzv),
                "float32",
            ),
            (
                fx - floor_divide(ix * 100, iy * 1000),
                (fx, ix, iy),
                (fxv, ixv, iyv),
                1,
                fxv - ((ixv * 100) // (iyv * 1000)),
                {
                    "custom": "float64",
                    "numpy + floatX": config.floatX,
                    "numpy": "float64",
                },
            ),  # 40
            (fx - (fy / 2), (fx, fy), (fxv, fyv), 1, fxv - (fyv / 2), "float32"),
            (
                fx - (fy % fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - (fyv % fzv),
                "float32",
            ),
            (
                fx - (fy > fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - (fyv > fzv),
                "float32",
            ),
            (
                fx - (fy >= fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - (fyv >= fzv),
                "float32",
            ),
            (
                fx - (fy < fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - (fyv < fzv),
                "float32",
            ),  # 45
            (
                fx - (fy <= fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - (fyv <= fzv),
                "float32",
            ),
            (
                fx - eq(fy, fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - (fyv == fzv),
                "float32",
            ),
            (
                fx - neq(fy, fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - (fyv != fzv),
                "float32",
            ),
            (
                fx - fy + tan(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - fyv + np.tan(fzv),
                "float32",
            ),
            (
                fx - fy + tanh(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - fyv + np.tanh(fzv),
                "float32",
            ),  # 50
            (
                fx - fy + sin(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - fyv + np.sin(fzv),
                "float32",
            ),
            (
                fx - fy + sinh(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - fyv + np.sinh(fzv),
                "float32",
            ),
            (
                fx - fy + square(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - fyv + (fzv * fzv),
                "float32",
            ),
            (
                fx - fy + sqrt(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - fyv + np.sqrt(fzv),
                "float32",
            ),
            (
                fx - fy + reciprocal(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - fyv + (1 / fzv),
                "float32",
            ),  # 55
            (
                fx - fy + neg(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - fyv + (-fzv),
                "float32",
            ),
            (
                fx - fy + at_round(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - fyv + np.round(fzv),
                "float32",
            ),
            (
                ix - iy + iround(fz),
                (ix, iy, fz),
                (ixv, iyv, fzv),
                1,
                ixv - iyv + np.round(fzv),
                "int64",
            ),
            # Bit op
            (
                fx - bitwise_or(iy, iz),
                (fx, iy, iz),
                (fxv, iyv, izv),
                1,
                fxv - (iyv | izv),
                {
                    "custom": "float64",
                    "numpy + floatX": config.floatX,
                    "numpy": "float64",
                },
            ),
            (
                fx - xor(iy, iz),
                (fx, iy, iz),
                (fxv, iyv, izv),
                1,
                fxv - (iyv ^ izv),
                {
                    "custom": "float64",
                    "numpy + floatX": config.floatX,
                    "numpy": "float64",
                },
            ),  # 60
            (
                fx - bitwise_and(iy, iz),
                (fx, iy, iz),
                (fxv, iyv, izv),
                1,
                fxv - (iyv & izv),
                {
                    "custom": "float64",
                    "numpy + floatX": config.floatX,
                    "numpy": "float64",
                },
            ),
            (
                fx - invert(iy),
                (fx, iy),
                (fxv, iyv),
                1,
                fxv - (~iyv),
                {
                    "custom": "float64",
                    "numpy + floatX": config.floatX,
                    "numpy": "float64",
                },
            ),
            (
                fx - at.cast(fy, dtype="float64"),
                (fx, fy),
                (fxv, fyv),
                1,
                fxv - np.asarray(fyv, "float64"),
                "float64",
            ),
            (
                at_pow(fx * fy + fz, fx * fy),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                np.power(fxv * fyv + fzv, fxv * fyv),
                "float32",
            ),
            (
                fv + fy**fz,
                (fv, fy, fz),
                (fvv, fyv, fzv),
                2,
                fvv + fyv**fzv,
                "float32",
            ),  # fused with a dimshuffle #65
            (
                fv - fy + tanh(fz),
                (fv, fy, fz),
                (fvv, fyv, fzv),
                2,
                fvv - fyv + np.tanh(fzv),
                "float32",
            ),  # fused with a dimshuffle
            # Cases where the same input is reused many times.
            (
                mul(fx, fx, fx, fx),
                (fx,),
                (fxv,),
                1,
                fxv * fxv * fxv * fxv,
                "float32",
            ),
            (
                mul(fx, ftanx, ftanx),
                (fx,),
                (fxv,),
                1,
                fxv * np.tan(fxv) * np.tan(fxv),
                "float32",
            ),
            (
                mul(fx, ftanx, ftanx, fx),
                (fx,),
                (fxv,),
                1,
                fxv * np.tan(fxv) * np.tan(fxv) * fxv,
                "float32",
            ),
            (
                mul(ftanx, ftanx, fx + fy),
                (fx, fy),
                (fxv, fyv),
                1,
                np.tan(fxv) * np.tan(fxv) * (fxv + fyv),
                "float32",
            ),  # 70
            # Cases with different broadcast pattern. They should not
            # be merged as this would duplicate computation
            # The graph should have 2 elemwise and 1 dimshuffle
            (
                fx * sin(fs),
                (fx, fs),
                (fxv, fsv),
                3,
                fxv * np.sin(fsv),
                "float32",
            ),
        ]
        if slice:
            cases = cases[slice]
        times = np.zeros(len(cases))
        fail1 = []
        fail2 = []
        fail3 = []
        fail4 = []
        for (
            id,
            [g, sym_inputs, val_inputs, nb_elemwise, answer, out_dtype],
        ) in enumerate(cases):
            if isinstance(out_dtype, dict):
                out_dtype = out_dtype[config.cast_policy]

            if shared_fn is None:
                f = function(list(sym_inputs), g, mode=mode)
                for x in range(nb_repeat):
                    out = f(*val_inputs)
                t1 = time.perf_counter()
            else:
                out = shared_fn(np.zeros(shp, dtype=out_dtype), "out")
                assert out.dtype == g.dtype
                f = function(sym_inputs, [], updates=[(out, g)], mode=mode)
                t0 = time.perf_counter()
                for x in range(nb_repeat):
                    f(*val_inputs)
                t1 = time.perf_counter()
                out = out.get_value()

            times[id] = t1 - t0
            atol = 1e-8
            if out_dtype == "float32":
                atol = 1e-6
            if not np.allclose(out, answer * nb_repeat, atol=atol):
                fail1.append(id)
            topo = f.maker.fgraph.toposort()
            topo_ = [n for n in topo if not isinstance(n.op, self.topo_exclude)]
            if assert_len_topo:
                if len(topo_) != nb_elemwise:
                    fail3.append((id, topo_, nb_elemwise))
                if nb_elemwise == 1:
                    # if no variable appears multiple times in the
                    # input of g,
                    # check that the number of input to the Composite
                    # Elemwise is ok
                    if len(set(g.owner.inputs)) == len(g.owner.inputs):
                        expected_len_sym_inputs = sum(
                            not isinstance(x, Constant) for x in topo_[0].inputs
                        )
                        assert expected_len_sym_inputs == len(sym_inputs)

            if out_dtype != out.dtype:
                fail4.append((id, out_dtype, out.dtype))

        assert len(fail1 + fail2 + fail3 + fail4) == 0

        return times

    def test_add_mul_fusion_inplace(self):
        rewrites_query = RewriteDatabaseQuery(
            include=[
                "local_elemwise_fusion",
                "composite_elemwise_fusion",
                "canonicalize",
                "inplace",
            ],
            exclude=["cxx_only", "BlasOpt"],
        )

        mode = Mode(self.mode.linker, rewrites_query)

        x, y, z = dmatrices("xyz")
        out = dot(x, y) + x + y + z
        f = function([x, y, z], out, mode=mode)
        topo = [n for n in f.maker.fgraph.toposort()]
        assert len(topo) == 2
        assert topo[-1].op.inplace_pattern

        new_out = f.maker.fgraph.outputs[0]
        assert isinstance(new_out.owner.op, Elemwise)
        assert isinstance(new_out.owner.op.scalar_op, aes.basic.Add)
        assert len(new_out.owner.inputs) == 4

        # TODO: Do we really need to do this?
        _ = f(
            np.random.random((5, 5)), np.random.random((5, 5)), np.random.random((5, 5))
        )


@utt.assertFailure_fast
def test_log1p():
    m = config.mode
    if m == "FAST_COMPILE":
        m = "FAST_RUN"
    m = get_mode(m)
    m = m.excluding("fusion")
    # check some basic cases
    x = dvector()
    f = function([x], log(1 + (x)), mode=m)
    assert [node.op for node in f.maker.fgraph.toposort()] == [log1p]
    f = function([x], log(1 + (-x)), mode=m)
    assert [node.op for node in f.maker.fgraph.toposort()] == [
        neg,
        inplace.log1p_inplace,
    ]
    f = function([x], -log(1 + (-x)), mode=m)
    assert [node.op for node in f.maker.fgraph.toposort()] == [
        neg,
        inplace.log1p_inplace,
        inplace.neg_inplace,
    ]

    # check trickier cases (and use different dtype)
    y = fmatrix()
    f = function([x, y], log(at.fill(y, 1) + (x)), mode=m)
    # the first three ops are Shape_i, Shape_i, and Dimshuffle
    topo = f.maker.fgraph.toposort()
    assert topo[-1].op == at.alloc
    assert log1p in [node.op for node in topo]

    f = function([x, y], log(0 + (x) + at.fill(y, 1.0)), mode=m)
    topo = f.maker.fgraph.toposort()
    assert topo[-1].op == at.alloc
    assert log1p in [node.op for node in topo]

    f = function([x, y], log(2 + (x) - at.fill(y, 1.0)), mode=m)
    topo = f.maker.fgraph.toposort()
    assert topo[-1].op == at.alloc
    assert log1p in [node.op for node in topo]

    f([1e-7, 10], [[0, 0], [0, 0]])  # debugmode will verify values

    # should work for int
    z = imatrix()
    f = function([z], log(1 + (z)), mode=m)
    assert [node.op for node in f.maker.fgraph.toposort()] == [log1p]


def test_local_log_add_exp():
    m = config.mode
    if m == "FAST_COMPILE":
        m = "FAST_RUN"
    m = get_mode(m)
    m = m.excluding("fusion")
    m = copy.copy(m)
    # No need to put them back as we have a new object
    m.check_isfinite = False

    # check some basic cases
    x = dvector()
    y = dvector()
    f = function([x, y], log(exp(x) + exp(y)), mode=m)

    # test that it gives the correct result when it doesn't overflow
    f([10], [10])  # doesn't causes overflow
    utt.assert_allclose(f([10], [10]), 10 + np.log1p(1))

    assert np.isfinite(f([10000], [10000]))  # causes overflow if handled incorrectly
    utt.assert_allclose(f([10000], [10000]), 10000 + np.log1p(1))

    # test that when max = +-inf, rewritten output still works correctly
    assert f([-np.inf], [-np.inf]) == -np.inf
    assert f([np.inf], [np.inf]) == np.inf
    assert f([np.inf], [-np.inf]) == np.inf

    # test that it also works with more than two args
    x = dvector()
    y = dvector()
    f = function([x, y], log(exp(x) + exp(y) + exp(x - y) + exp(x + y)), mode=m)

    assert np.isfinite(f([10000], [10000]))  # causes overflow if handled incorrectly
    utt.assert_allclose(f([10000], [10000]), 20000)

    # TODO: test that the rewrite works in the presence of broadcasting.


def test_local_subtensor_of_dot():
    m1 = matrix()
    m2 = matrix()
    d1 = np.arange(6).reshape((3, 2)).astype(config.floatX)
    d2 = np.arange(8).reshape((2, 4)).astype(config.floatX) + 10
    mode = get_default_mode().including("local_subtensor_of_dot")

    def test_equality(a, b):
        return a.shape == b.shape and np.allclose(a, b)

    # [cst]
    f = function([m1, m2], aesara.tensor.dot(m1, m2)[1], mode=mode)
    topo = f.maker.fgraph.toposort()
    assert test_equality(f(d1, d2), np.dot(d1, d2)[1])
    # DimShuffle happen in FAST_COMPILE
    assert isinstance(topo[-1].op, (CGemv, Gemv, DimShuffle))

    # slice
    f = function([m1, m2], aesara.tensor.dot(m1, m2)[1:2], mode=mode)
    topo = f.maker.fgraph.toposort()
    assert test_equality(f(d1, d2), np.dot(d1, d2)[1:2])
    assert isinstance(topo[-1].op, Dot22)

    m1 = tensor3()
    m2 = tensor3()
    idx = iscalar()
    d1 = np.arange(30).reshape(2, 5, 3).astype(config.floatX)
    d2 = np.arange(72).reshape(4, 3, 6).astype(config.floatX) + 100

    f = function([m1, m2, idx], aesara.tensor.dot(m1, m2)[idx, 1:4, :, idx:], mode=mode)
    assert test_equality(f(d1, d2, 1), np.dot(d1, d2)[1, 1:4, :, 1:])
    # if we return the gradients. We need to use same mode as before.
    assert check_stack_trace(f, ops_to_check="last")

    f = function([m1, m2, idx], aesara.tensor.dot(m1, m2)[1:4, :, idx:, idx], mode=mode)
    assert test_equality(f(d1, d2, 1), np.dot(d1, d2)[1:4, :, 1:, 1])

    # Now test that the stack trace is copied over properly,
    # if we return the gradients. We need to use same mode as before.
    assert check_stack_trace(f, ops_to_check="last")


def test_local_elemwise_sub_zeros():
    scal = scalar()
    vect = vector()
    mat = matrix()

    rng = np.random.default_rng(seed=utt.fetch_seed())
    scalar_val = rng.random(1).astype(config.floatX)[0]
    vect_val = rng.random(5).astype(config.floatX)
    mat_val = rng.random((3, 2)).astype(config.floatX)

    mode = (
        get_default_mode()
        .excluding(
            "canonicalize",
            "uncanonicalize",
            "ShapeOpt",
            "local_fill_to_alloc",
            "local_elemwise_alloc",
        )
        .including("local_elemwise_sub_zeros")
    )

    # Test scalar minus scalar
    f = function([scal], scal - scal, mode=mode)
    assert isinstance(f.maker.fgraph.toposort()[0].op, Elemwise)
    assert isinstance(f.maker.fgraph.toposort()[0].op.scalar_op, aes.Second)
    assert isinstance(
        f.maker.fgraph.toposort()[0].inputs[1], TensorConstant
    ) or isinstance(f.maker.fgraph.toposort()[0].inputs[1], TensorConstant)
    utt.assert_allclose(f(scalar_val), 0.0)
    assert check_stack_trace(f, ops_to_check="all")

    # Test vector minus vector
    f = function([vect], vect - vect, mode=mode)
    assert isinstance(f.maker.fgraph.toposort()[0].op, Elemwise)
    assert isinstance(f.maker.fgraph.toposort()[0].op.scalar_op, aes.Second)
    assert isinstance(
        f.maker.fgraph.toposort()[0].inputs[1], TensorConstant
    ) or isinstance(f.maker.fgraph.toposort()[0].inputs[1], TensorConstant)
    utt.assert_allclose(f(vect_val), np.zeros(vect_val.shape))
    assert check_stack_trace(f, ops_to_check="all")

    # Test vector minus vector
    f = function([mat], mat - mat, mode=mode)
    assert isinstance(f.maker.fgraph.toposort()[0].op, Elemwise)
    assert isinstance(f.maker.fgraph.toposort()[0].op.scalar_op, aes.Second)
    assert isinstance(
        f.maker.fgraph.toposort()[0].inputs[1], TensorConstant
    ) or isinstance(f.maker.fgraph.toposort()[0].inputs[1], TensorConstant)
    utt.assert_allclose(f(mat_val), np.zeros(mat_val.shape))
    assert check_stack_trace(f, ops_to_check="all")


class TestLocalUselessElemwiseComparison:
    def setup_method(self):
        self.rng = np.random.default_rng(utt.fetch_seed())

    def test_local_useless_elemwise_comparison(self):
        # TODO FIXME: This is not a real test!
        # TODO: test each case individually.
        # The following case is what made me discover those cases.
        X = matrix("X")
        Y = vector("Y")
        X_sum, updates = aesara.scan(
            fn=lambda x: x.sum(), outputs_info=None, sequences=[X], non_sequences=None
        )
        Z = X_sum + Y
        # aesara.printing.debugprint(Z)
        # here is the output for the debug print:
        """
        Elemwise{add,no_inplace} [id A] ''
         |for{cpu,scan_fn} [id B] ''
         | |Subtensor{int64} [id C] ''
         | | |Shape [id D] ''
         | | | |Subtensor{int64::} [id E] 'X[0:]'
         | | |   |X [id F]
         | | |   |Constant{0} [id G]
         | | |Constant{0} [id H]
         | |Subtensor{:int64:} [id I] ''
         | | |Subtensor{int64::} [id E] 'X[0:]'
         | | |ScalarFromTensor [id J] ''
         | |   |Subtensor{int64} [id C] ''
         | |Subtensor{int64} [id C] ''
         |Y [id K]

        Inner graphs:

        for{cpu,scan_fn} [id B] ''
         >Sum{acc_dtype=float64} [id L] ''
         > |X[t] [id M] -> [id I]
        """

        mode = get_default_mode().excluding("fusion")
        f = function([X, Y], Z, mode=mode)
        f(
            self.rng.random((2, 3)).astype(config.floatX),
            self.rng.random(2).astype(config.floatX),
        )
        # aesara.printing.debugprint(f, print_type=True)
        # here is the output for the debug print:
        """
        Elemwise{Add}[(0, 0)] [id A] <TensorType(float64, vector)> ''   7
         |for{cpu,scan_fn} [id B] <TensorType(float64, vector)> ''   6
         | |Shape_i{0} [id C] <TensorType(int64, scalar)> ''   0
         | | |X [id D] <TensorType(float64, matrix)>
         | |Subtensor{int64:int64:int8} [id E] <TensorType(float64, matrix)> ''   5
         | | |X [id D] <TensorType(float64, matrix)>
         | | |ScalarFromTensor [id F] <int64> ''   4
         | | | |Elemwise{switch,no_inplace} [id G] <TensorType(int64, scalar)> ''   3
         | | |   |Elemwise{le,no_inplace} [id H] <TensorType(int8, scalar)> ''   2
         | | |   | |Shape_i{0} [id C] <TensorType(int64, scalar)> ''   0
         | | |   | |TensorConstant{0} [id I] <TensorType(int8, scalar)>
         | | |   |TensorConstant{0} [id I] <TensorType(int8, scalar)>
         | | |   |TensorConstant{0} [id J] <TensorType(int64, scalar)>
         | | |ScalarFromTensor [id K] <int64> ''   1
         | | | |Shape_i{0} [id C] <TensorType(int64, scalar)> ''   0
         | | |Constant{1} [id L] <int8>
         | |Shape_i{0} [id C] <TensorType(int64, scalar)> ''   0
         |Y [id M] <TensorType(float64, vector)>

        Inner graphs:

        for{cpu,scan_fn} [id B] <TensorType(float64, vector)> ''
         >Sum{acc_dtype=float64} [id N] <TensorType(float64, scalar)> ''
         > |X[t] [id O] <TensorType(float64, vector)> -> [id E]
        """

    def assert_eqs_const(self, f, val, op=deep_copy_op):
        topo = f.maker.fgraph.toposort()
        elem = topo[0]
        assert len(topo) == 1, topo
        assert elem.op == op, elem.op
        if op == deep_copy_op:
            assert len(elem.inputs) == 1, elem.inputs
            assert isinstance(elem.inputs[0], TensorConstant), elem
            assert at.extract_constant(elem.inputs[0]) == val, val
        else:
            assert len(elem.inputs) == 2, elem.inputs
            assert isinstance(elem.inputs[0], TensorConstant), elem
            assert at.extract_constant(elem.inputs[0]) == val, val

    def assert_identity(self, f):
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert topo[0].op == deep_copy_op
        if f.outputs[0].variable.dtype == "bool":
            x_vals = [0, 1]
        else:
            x_vals = [0, 1, 10]
        for x_val in x_vals:
            assert f(x_val) == x_val

    def test_inequality_with_self(self):
        x = scalar("x", dtype=config.floatX)
        mode = get_default_mode().including("local_useless_elemwise_comparison")

        f = function([x], lt(x, x), mode=mode)
        self.assert_eqs_const(f, 0)

        f = function([x], le(x, x), mode=mode)
        self.assert_eqs_const(f, 1)

        f = function([x], gt(x, x), mode=mode)
        self.assert_eqs_const(f, 0)

        f = function([x], ge(x, x), mode=mode)
        self.assert_eqs_const(f, 1)

        f = function([x], minimum(x, x), mode=mode)
        self.assert_identity(f)

        f = function([x], maximum(x, x), mode=mode)
        self.assert_identity(f)

    def test_shape_inequality_with_self(self):
        x = vector("x", dtype=config.floatX)
        mode = get_default_mode().including(
            "local_useless_elemwise_comparison",
            "local_shape_to_shape_i",
            "local_track_shape_i",
            "local_subtensor_make_vector",
            "local_subtensor_remove_broadcastable_index",
            "local_useless_dimshuffle_makevector",
        )
        f = function([x], lt(x.shape[0], 0), mode=mode)
        self.assert_eqs_const(f, 0)

        f = function([x], ge(x.shape[0], 0), mode=mode)
        self.assert_eqs_const(f, 1)

        f = function([x], maximum(x.shape[0], 0), mode=mode)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert isinstance(topo[0].op, Shape_i), topo[0].op
        x_val = np.ones(100, dtype=config.floatX)
        assert f(x_val) == x_val.shape[0]

        f = function([x], maximum(0, x.shape[0]), mode=mode)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert isinstance(topo[0].op, Shape_i), topo[0].op
        x_val = np.ones(100, dtype=config.floatX)
        assert f(x_val) == x_val.shape[0]

        f = function([x], minimum(x.shape[0], 0), mode=mode)
        self.assert_eqs_const(f, 0)
        assert f(x_val) == 0

        f = function([x], minimum(0, x.shape[0]), mode=mode)
        self.assert_eqs_const(f, 0)
        assert f(x_val) == 0
        f = function([x], minimum([0, 0], x.shape[0]), mode=mode)
        # This case isn't rewritten.
        # self.assert_eqs_const(f, 0)
        utt.assert_allclose(f(x_val), [0, 0])

    def test_shape_add_inequality(self):
        x = vector("x", dtype=config.floatX)
        mode = get_default_mode().including(
            "local_useless_elemwise_comparison",
            "local_shape_to_shape_i",
            "local_track_shape_i",
            "local_subtensor_make_vector",
        )

        y = vector("y", dtype=config.floatX)

        f = function([x, y], lt(x.shape[0] + y.shape[0], 0), mode=mode)
        self.assert_eqs_const(f, 0)

        f = function([x, y], ge(x.shape[0] + y.shape[0], 0), mode=mode)
        self.assert_eqs_const(f, 1)

    @pytest.mark.skipif(
        config.mode == "FAST_COMPILE",
        reason="This rewrite is disabled.",
    )
    def test_equality_shapes(self):
        # Test equality where one sides contain only shapes related
        # stuff.
        x = vector("x", dtype=config.floatX)
        for g in [x.shape[0], Shape_i(0)(x)]:
            f = function([x], eq(g, 0))
            assert f([3, 3]) == 0
            assert f([]) == 1

            f = function([x], eq(g, -1))
            self.assert_eqs_const(f, 0)
            assert f([3, 3]) == 0

        g = join(0, x.shape[0:], x.shape[0:1])  # todo test reshape, dimshuffle
        f = function([x], eq(g, 0))
        assert (f([3, 3]) == 0).all()
        assert (f([]) == 1).all()

        f = function([x], eq(g, -1))
        self.assert_eqs_const(f, 0, op=at.alloc)
        assert (f([3, 3]) == 0).all()

    def test_and(self):
        # bitwise "and" with 0 should give 0 for both bool and int
        # bitwise "and" with 1 should only simplify for bool
        mode = get_default_mode().including("canonicalize")
        for dtype, zero, one in [
            ("bool", np.array(False), np.array(True)),
            ("int8", np.int8(0), np.int8(1)),
            ("int8", 0, 1),
        ]:
            x = scalar("x", dtype=dtype)

            f = function([x], bitwise_and(x, zero), mode=mode)
            self.assert_eqs_const(f, 0)

            f = function([x], bitwise_and(zero, x), mode=mode)
            self.assert_eqs_const(f, 0)

            f = function([x], bitwise_and(x, one), mode=mode)
            if dtype == "bool":
                self.assert_identity(f)

            f = function([x], bitwise_and(one, x), mode=mode)
            if dtype == "bool":
                self.assert_identity(f)

    def test_and_int(self):
        # Test that bitwise "and" is correctly computed on int constants.
        f = function([], bitwise_and(5, 6))
        assert f() == 4

    def test_or(self):
        # bitwise "or" with 0 should simplify for both bool and int
        # bitwise "or" with 1 should only give 1 for bool
        mode = get_default_mode().including("canonicalize")
        for dtype, zero, one in [
            ("bool", np.array(False), np.array(True)),
            ("int8", np.int8(0), np.int8(1)),
            ("int8", 0, 1),
        ]:
            x = scalar("x", dtype=dtype)

            f = function([x], bitwise_or(x, one), mode=mode)
            if dtype == "bool":
                self.assert_eqs_const(f, 1)

            f = function([x], bitwise_or(one, x), mode=mode)
            if dtype == "bool":
                self.assert_eqs_const(f, 1)

            f = function([x], bitwise_or(x, zero), mode=mode)
            self.assert_identity(f)

            f = function([x], bitwise_or(zero, x), mode=mode)
            self.assert_identity(f)

    def test_or_int(self):
        # Test that bitwise "or" is correctly computed on int constants.
        f = function([], bitwise_or(5, 6))
        assert f() == 7

    def test_xor(self):
        # bitwise "xor" with itself should always give 0 for both bool and int.
        mode = get_default_mode().including("canonicalize")
        for dtype in ("bool", "int8"):
            x = scalar("x", dtype=dtype)

            f = function([x], xor(x, x), mode=mode)
            self.assert_eqs_const(f, 0)

    def test_stacktrace(self):
        mode = get_default_mode().including("local_useless_elemwise_comparison")

        x = vector("x", dtype=config.floatX)
        f = function([x], gt(x, x), mode=mode)
        assert check_stack_trace(f, ops_to_check="last")

        f = function([x], le(x, x), mode=mode)
        assert check_stack_trace(f, ops_to_check="last")


def test_local_mul_specialize():
    mode = config.mode
    if mode == "FAST_COMPILE":
        mode = "FAST_RUN"
    mode = get_mode(mode)
    mode = mode.excluding("fusion")

    v = vector()
    m = vector()

    f = function([v], v * 1, mode=mode)
    nodes = [node.op for node in f.maker.fgraph.toposort()]
    nodes == [deep_copy_op]

    f = function([v], v * 0, mode=mode)
    nodes = [node.op for node in f.maker.fgraph.toposort()]
    assert nodes == [Shape_i(0), at.alloc]

    f = function([v], v * (-1), mode=mode)
    nodes = [node.op for node in f.maker.fgraph.toposort()]
    assert nodes == [neg]

    f = function([v, m], v * 1 * (-m), mode=mode)
    nodes = [node.op for node in f.maker.fgraph.toposort()]
    assert nodes == [mul]

    f = function([v, m], v * 0 * (-m), mode=mode)
    nodes = [node.op for node in f.maker.fgraph.toposort()]
    assert nodes == [Shape_i(0), at.alloc]

    f = function([v, m], v * (-1) * (-m), mode=mode)
    nodes = [node.op for node in f.maker.fgraph.toposort()]
    assert nodes == [mul]

    f = function([v, m], v * (-1) * m, mode=mode)
    nodes = [node.op for node in f.maker.fgraph.toposort()]
    assert nodes == [mul]


def speed_local_pow_specialize_range():
    val = np.random.random(1e7)
    v = vector()
    mode = get_default_mode()
    mode_without_pow_rewrite = mode.excluding("local_pow_specialize")
    for i in range(500, 513):
        f1 = function([v], v**i, mode=mode)
        f2 = function([v], v**i, mode=mode_without_pow_rewrite)
        assert len(f1.maker.fgraph.toposort()) == 1
        t1 = time.perf_counter()
        f1(val)
        t2 = time.perf_counter()
        f2(val)
        t3 = time.perf_counter()
        print(i, t2 - t1, t3 - t2, t2 - t1 < t3 - t2)
        if not t2 - t1 < t3 - t2:
            print("WARNING WE ARE SLOWER")
    for i in range(-3, -1500, -1):
        f1 = function([v], v**i, mode=mode)
        f2 = function([v], v**i, mode=mode_without_pow_rewrite)
        assert len(f1.maker.fgraph.toposort()) == 1
        t1 = time.perf_counter()
        f1(val)
        t2 = time.perf_counter()
        f2(val)
        t3 = time.perf_counter()
        print(i, t2 - t1, t3 - t2, t2 - t1 < t3 - t2)
        if not t2 - t1 < t3 - t2:
            print("WARNING WE ARE SLOWER")


def test_local_pow_specialize():
    mode = config.mode
    if mode == "FAST_COMPILE":
        mode = "FAST_RUN"
    mode = get_mode(mode)
    mode = mode.excluding("fusion")

    v = vector()
    val = np.arange(10, dtype=config.floatX)
    val_no0 = np.arange(1, 10, dtype=config.floatX)

    f = function([v], v**0, mode=mode)
    nodes = [node.op for node in f.maker.fgraph.toposort()]
    assert nodes == [Shape_i(0), at.alloc]
    utt.assert_allclose(f(val), val**0)

    f = function([v], v**1, mode=mode)
    nodes = [node.op for node in f.maker.fgraph.toposort()]
    nodes == [deep_copy_op]
    utt.assert_allclose(f(val), val**1)

    f = function([v], v ** (-1), mode=mode)
    nodes = [node.op for node in f.maker.fgraph.toposort()]
    assert nodes == [reciprocal]
    utt.assert_allclose(f(val_no0), val_no0 ** (-1))

    f = function([v], v**2, mode=mode)
    nodes = [node.op for node in f.maker.fgraph.toposort()]
    assert nodes == [square]
    utt.assert_allclose(f(val), val**2)

    f = function([v], v ** (-2), mode=mode)
    nodes = [node.op for node in f.maker.fgraph.toposort()]
    assert len(nodes) == 2
    assert nodes[0] == square
    assert isinstance(nodes[1].scalar_op, aes.basic.Reciprocal)
    utt.assert_allclose(f(val_no0), val_no0 ** (-2))

    f = function([v], v ** (0.5), mode=mode)
    nodes = [node.op for node in f.maker.fgraph.toposort()]
    assert nodes == [sqrt]
    utt.assert_allclose(f(val), val ** (0.5))

    f = function([v], v ** (-0.5), mode=mode)
    nodes = [node.op for node in f.maker.fgraph.toposort()]
    assert len(nodes) == 2
    assert nodes[0] == sqrt
    assert isinstance(nodes[1].scalar_op, aes.basic.Reciprocal)
    utt.assert_allclose(f(val_no0), val_no0 ** (-0.5))


def test_local_pow_specialize_device_more_aggressive_on_cpu():
    mode = config.mode
    if mode == "FAST_COMPILE":
        mode = "FAST_RUN"
    mode = get_mode(mode)
    mode = mode.excluding("fusion").excluding("gpu")

    v = vector()
    val = np.arange(10, dtype=config.floatX)
    val_no0 = np.arange(1, 10, dtype=config.floatX)
    f = function([v], v ** (15), mode=mode)
    nodes = [node.op for node in f.maker.fgraph.toposort()]
    assert len(nodes) == 1
    assert len(f.maker.fgraph.toposort()[0].op.scalar_op.fgraph.apply_nodes) == 6
    assert isinstance(nodes[0].scalar_op, aes.Composite)
    utt.assert_allclose(f(val), val**15)

    f = function([v], v ** (-15), mode=mode)
    nodes = [node.op for node in f.maker.fgraph.toposort()]
    assert len(nodes) == 2
    assert len(f.maker.fgraph.toposort()[0].op.scalar_op.fgraph.apply_nodes) == 6
    assert isinstance(nodes[0].scalar_op, aes.Composite)
    assert isinstance(nodes[-1].scalar_op, aes.basic.Reciprocal)
    utt.assert_allclose(f(val_no0), val_no0 ** (-15))

    f = function([v], v ** (16), mode=mode)
    nodes = [node.op for node in f.maker.fgraph.toposort()]
    assert len(nodes) == 1
    assert len(f.maker.fgraph.toposort()[0].op.scalar_op.fgraph.apply_nodes) == 4
    assert isinstance(nodes[0].scalar_op, aes.Composite)
    utt.assert_allclose(f(val), val**16)

    f = function([v], v ** (-16), mode=mode)
    nodes = [node.op for node in f.maker.fgraph.toposort()]
    assert len(nodes) == 2
    assert len(f.maker.fgraph.toposort()[0].op.scalar_op.fgraph.apply_nodes) == 4
    assert isinstance(nodes[0].scalar_op, aes.Composite)
    assert isinstance(nodes[-1].scalar_op, aes.basic.Reciprocal)
    utt.assert_allclose(f(val_no0), val_no0 ** (-16))


class TestFuncInverse:
    def setup_method(self):
        mode = get_default_mode()
        self.mode = mode.including("local_func_inv")

    def assert_func_pair_rewritten(
        self, func1, func2, data, should_copy=True, is_complex=False
    ):
        """Check that a pair of functions are rewritten properly."""

        x = cmatrix() if is_complex else fmatrix()
        o = func2(func1(x))
        f = function([x], o, mode=self.mode)
        delta = f(data) - data
        topo = f.maker.fgraph.toposort()

        if should_copy:
            acceptable_topo_lens = [1]
        else:
            # The 2 funcs can be split apart if they are not inverses
            acceptable_topo_lens = [1, 2]

        if should_copy:
            delta_condition = np.all(delta == 0)
        else:
            delta_condition = np.all(delta != 0)

        assert len(topo) in acceptable_topo_lens
        assert delta_condition
        assert (
            isinstance(topo[0].op, DeepCopyOp) == should_copy
        ), "Inverse functions not removed!"

    def test(self):
        """Test rewrites for consecutive functional inverses."""

        dx = np.random.random((5, 4)).astype("float32")
        self.assert_func_pair_rewritten(deg2rad, rad2deg, dx)
        dx = np.random.random((5, 4)).astype("float32") * 180
        self.assert_func_pair_rewritten(rad2deg, deg2rad, dx)

        # Test the other functional inverses
        dx = np.random.random((5, 4)).astype("float32")
        self.assert_func_pair_rewritten(cosh, arccosh, dx)
        self.assert_func_pair_rewritten(arcsinh, sinh, dx)
        self.assert_func_pair_rewritten(arctanh, tanh, dx)
        self.assert_func_pair_rewritten(reciprocal, reciprocal, dx)
        self.assert_func_pair_rewritten(neg, neg, dx)
        cx = dx + complex(0, 1) * (dx + 0.01)
        self.assert_func_pair_rewritten(conj, conj, cx, is_complex=True)

        # Test that non-inverse functions are ran normally
        self.assert_func_pair_rewritten(
            conj, neg, cx, should_copy=False, is_complex=True
        )
        dx = np.random.random((5, 4)).astype("float32") + 0.01
        self.assert_func_pair_rewritten(rad2deg, rad2deg, dx, should_copy=False)
        self.assert_func_pair_rewritten(rad2deg, cosh, dx, should_copy=False)

    def test_integer_upcast(self):
        """
        All invertible methods (except for `Neg`) can upgrade their input to float.
        Here we test that the rewrite works with just one pair of methods
        """
        x = ivector("x")
        f = function([x], deg2rad(rad2deg(x)), mode=self.mode)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1


class TestExpLog:
    def setup_method(self):
        mode = get_default_mode()
        self.mode = mode.including(
            "local_exp_log",
            "local_exp_log_nan_switch",
        ).excluding("fusion")

    def test_log_exp(self):
        # log(exp(x)) -> x
        data = np.random.random((4, 3)).astype("float32")
        x = fmatrix()
        f = function([x], log(exp(x)), mode=self.mode)
        graph = f.maker.fgraph.toposort()
        ops_graph = [
            node
            for node in graph
            if isinstance(node.op, Elemwise)
            and isinstance(node.op.scalar_op, (aes.Log, aes.Exp))
        ]
        assert len(ops_graph) == 0
        np.testing.assert_array_equal(f(data), data)

    def test_log_exp_integer_upcast(self):
        x = ivector("x")
        f = function([x], log(exp(x)), mode=self.mode)
        ops_graph = [
            node
            for node in f.maker.fgraph.toposort()
            if isinstance(node.op, Elemwise)
            and isinstance(node.op.scalar_op, (aes.Log, aes.Exp))
        ]
        assert len(ops_graph) == 0

    @pytest.mark.parametrize("dtype", ["float32", "int32"])
    def test_log1p_expm1(self, dtype):
        # log1p(expm1(x)) -> x
        data = (np.random.random((4, 3)) * 100).astype(dtype)
        x = matrix(dtype=dtype)
        f = function([x], log1p(expm1(x)), mode=self.mode, allow_input_downcast=True)
        graph = f.maker.fgraph.toposort()
        ops_graph = [
            node
            for node in graph
            if isinstance(node.op, Elemwise)
            and isinstance(node.op.scalar_op, (aes.Log, aes.Exp, aes.Log1p, aes.Expm1))
        ]
        assert len(ops_graph) == 0
        np.testing.assert_array_equal(f(data), data)

    @pytest.mark.parametrize("exp_op", [exp, expm1])
    def test_exp_log(self, exp_op):
        # exp(log(x)) -> switch(x >= 0, x, nan)
        # expm1(log(x)) -> switch(x >= 0, x - 1, nan)
        data_valid = np.random.random((4, 3)).astype("float32")
        data_valid[0, 0] = 0  # edge case
        data_invalid = data_valid - 1

        x = fmatrix()
        f = function([x], exp_op(log(x)), mode=self.mode)
        graph = f.maker.fgraph.toposort()
        ops_graph = [
            node
            for node in graph
            if isinstance(node.op, Elemwise)
            and isinstance(node.op.scalar_op, (aes.Log, aes.Log1p, aes.Exp, aes.Expm1))
        ]
        assert len(ops_graph) == 0

        if exp_op == exp:
            expected = data_valid
        else:
            expected = data_valid - 1
        np.testing.assert_almost_equal(f(data_valid), expected)
        assert np.all(np.isnan(f(data_invalid)))

    @pytest.mark.parametrize("exp_op", [exp, expm1])
    def test_exp_log1p(self, exp_op):
        # exp(log1p(x)) -> switch(x >= -1, x + 1, nan)
        # expm1(log1p(x)) -> switch(x >= -1, x, nan)
        data_valid = np.random.random((4, 3)).astype("float32") * 2 - 1
        data_valid[0, 0] = -1  # edge case
        data_invalid = data_valid - 2

        x = fmatrix()
        f = function([x], exp_op(log1p(x)), mode=self.mode)
        graph = f.maker.fgraph.toposort()
        ops_graph = [
            node
            for node in graph
            if isinstance(node.op, Elemwise)
            and isinstance(node.op.scalar_op, (aes.Log, aes.Log1p, aes.Exp, aes.Expm1))
        ]
        assert len(ops_graph) == 0

        if exp_op == exp:
            expected = data_valid + 1
        else:
            expected = data_valid
        np.testing.assert_almost_equal(f(data_valid), expected)
        assert np.all(np.isnan(f(data_invalid)))

    @pytest.mark.parametrize("exp_op", [exp, expm1])
    def test_exp_log1mexp(self, exp_op):
        # exp(log1mexp(x)) -> switch(x <= 0, 1 - exp(x), nan)
        # expm1(log1mexp(x)) -> switch(x <= 0, - exp(x), nan)
        data_valid = -np.random.random((4, 3)).astype("float32")
        data_valid[0, 0] = 0  # edge case
        data_invalid = data_valid + 1

        x = fmatrix()
        f = function([x], exp_op(log1mexp(x)), mode=self.mode)
        graph = f.maker.fgraph.toposort()
        ops_graph = [
            node
            for node in graph
            if isinstance(node.op, Elemwise)
            and isinstance(
                node.op.scalar_op, (aes.Log, aes.Log1p, aes.Log1mexp, aes.Expm1)
            )
        ]
        assert len(ops_graph) == 0

        if exp_op == exp:
            expected = 1 - np.exp(data_valid)
        else:
            expected = -np.exp(data_valid)
        np.testing.assert_almost_equal(f(data_valid), expected)
        assert np.all(np.isnan(f(data_invalid)))

    @pytest.mark.parametrize("exp_op", [exp, expm1])
    def test_exp_softplus(self, exp_op):
        # exp(softplus(x)) -> 1 + exp(x)
        # expm1(softplus(x)) -> exp(x)
        data_valid = np.random.random((4, 3)).astype("float32") * 2 - 1

        x = fmatrix()
        f = function([x], exp_op(softplus(x)), mode=self.mode)
        graph = f.maker.fgraph.toposort()
        ops_graph = [
            node
            for node in graph
            if isinstance(node.op, Elemwise)
            and isinstance(
                node.op.scalar_op,
                (aes.Log, aes.Log1p, aes.Softplus, aes.Expm1, aes.Switch),
            )
        ]
        assert len(ops_graph) == 0

        if exp_op == exp:
            expected = 1 + np.exp(data_valid)
        else:
            expected = np.exp(data_valid)
        np.testing.assert_almost_equal(
            f(data_valid),
            expected,
            decimal=6,
        )

    @pytest.mark.parametrize(
        ["nested_expression", "expected_switches"],
        [
            (lambda x: exp(log(exp(log(exp(x))))), 0),
            (lambda x: exp(log(exp(log(x)))), 1),
        ],
    )
    def test_exp_log_nested(self, nested_expression, expected_switches):
        # Make sure nested exp-log graphs have as little `nan` switches as necessary
        x = fvector()
        f = function([x], nested_expression(x), mode=self.mode)
        graph = f.maker.fgraph.toposort()
        ops_graph = [
            node
            for node in graph
            if isinstance(node.op, Elemwise)
            and isinstance(node.op.scalar_op, aes.Switch)
        ]
        assert len(ops_graph) == expected_switches


class TestLocalSwitchSink:
    def setup_method(self):
        # condition values
        self.condm = np.asarray([[0.1, 0, 1, -1], [0.0, 0.0, 0.0, 0.0], [1, 1, 1, 1]])
        self.condv = np.asarray([0.1, 0, 1, -1])
        self.conds = [0.1, 0, 1, -1]

        # x values
        self.xm = np.ones((3, 4))
        self.xv = np.ones((4,))
        self.xs = 1.0

        # expected results
        self.resm = (
            [np.asarray([[1, 0, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1]])] * 3
            + [np.asarray([[1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]])]
            + 2 * [np.asarray([[1, 0, 1, 0]])]
            + [[np.ones((3, 4)), np.zeros((3, 4)), np.ones((3, 4)), np.zeros((3, 4))]]
            + [[np.ones((4,)), np.zeros((4,)), np.ones((4,)), np.zeros((4,))]]
            + [[np.asarray(1.0), np.asarray(0.0), np.asarray(1.0), np.asarray(0.0)]]
        )

        self.mode = (
            get_default_mode()
            .including("canonicalize", "fast_run")
            .excluding("gpu", "fusion")
        )
        self.mode = copy.copy(self.mode)
        self.mode.check_isfinite = False

    def function_remove_nan(self, *args, **kwargs):
        """Wrapper around function for this test.

        It disables checking for NaNs removed by rewrites in `DebugMode`
        (it has false positives in that case).
        """
        f = function(*args, **kwargs)

        def wrapped_f(*args, **kwargs):
            # This is a bit ugly since it changes the global value of
            # TensorType.values_eq_approx.
            old_values_eq_approx = staticmethod(TensorType.values_eq_approx)
            TensorType.values_eq_approx = staticmethod(values_eq_approx_remove_nan)
            try:
                out = f(*args, **kwargs)
            finally:
                TensorType.values_eq_approx = old_values_eq_approx
            return out

        return wrapped_f

    def test_local_mul_switch_sink(self):
        c = dscalar()
        idx = 0
        for condition in [
            (dmatrix("cond"), self.condm),
            (dvector("cond"), self.condv),
            (dscalar("cond"), self.conds),
        ]:
            for x in [
                (dmatrix("x"), self.xm),
                (dvector("x"), self.xv),
                (dscalar("x"), self.xs),
            ]:
                y = mul(
                    at.switch(condition[0] > 0, 1.0 * x[0], 0.0 * x[0]),
                    at.switch(condition[0] > 0, 1.0 * x[0], log(c) * x[0]),
                )
                f = self.function_remove_nan(
                    [condition[0], x[0], c], [y], mode=self.mode
                )
                if type(condition[1]) is list:
                    for i in range(len(condition[1])):
                        res = f(condition[1][i], x[1], -1)
                        assert (
                            res == np.asarray(self.resm[idx][i])
                        ).sum() == self.resm[idx][i].size
                else:
                    res = f(condition[1], x[1], -1)
                    assert (res == np.asarray(self.resm[idx])).sum() == self.resm[
                        idx
                    ].size
                idx += 1

        # This case prevented a rewrite from being applied in the past
        x = dscalar("x")
        y = at.switch(x < 7, x, sqrt(x - 7))
        f = self.function_remove_nan([x], aesara.gradient.grad(y, x), self.mode)
        assert f(5) == 1, f(5)

    @pytest.mark.slow
    def test_local_div_switch_sink(self):
        c = dscalar()
        idx = 0
        for condition in [
            (dmatrix("cond"), self.condm),
            (dvector("cond"), self.condv),
            (dscalar("cond"), self.conds),
        ]:
            for x in [
                (dmatrix("x"), self.xm),
                (dvector("x"), self.xv),
                (dscalar("x"), self.xs),
            ]:
                y = true_divide(
                    at.switch(condition[0] > 0, 1.0 * x[0], 0.0 * x[0]),
                    at.switch(condition[0] > 0, 1.0 * x[0], log(c) * x[0]),
                )
                f = self.function_remove_nan(
                    [condition[0], x[0], c], [y], mode=self.mode
                )
                if type(condition[1]) is list:
                    for i in range(len(condition[1])):
                        res = f(condition[1][i], x[1], -1)
                        assert (
                            res == np.asarray(self.resm[idx][i])
                        ).sum() == self.resm[idx][i].size
                else:
                    res = f(condition[1], x[1], -1)
                    assert (res == np.asarray(self.resm[idx])).sum() == self.resm[
                        idx
                    ].size
                idx += 1


@pytest.mark.skipif(
    config.cxx == "",
    reason="erf need a c++ compiler or scipy",
)
class TestLocalErf:
    def setup_method(self):
        self.mode = (
            get_default_mode()
            .including("canonicalize", "fast_run")
            .excluding("gpu", "fusion", "inplace")
        )

    def test_local_one_plus_erf(self):
        val = np.asarray([-30, -3, -2, -1, 0, 1, 2, 3, 30], dtype=config.floatX)
        x = vector()

        f = function([x], 1 + erf(x), mode=self.mode)
        assert [n.op for n in f.maker.fgraph.toposort()] == [neg, erfc]
        f(val)

        f = function([x], erf(x) + 1, mode=self.mode)
        assert [n.op for n in f.maker.fgraph.toposort()] == [neg, erfc]
        f(val)

        f = function([x], erf(x) + 2, mode=self.mode)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 2
        assert topo[0].op == erf
        assert isinstance(topo[1].op, Elemwise)
        assert isinstance(topo[1].op.scalar_op, aes.Add)
        f(val)

    def test_local_one_minus_erf(self):
        val = np.asarray([-30, -3, -2, -1, 0, 1, 2, 3, 30], dtype=config.floatX)
        x = vector()

        f = function([x], 1 - erf(x), mode=self.mode)
        assert [n.op for n in f.maker.fgraph.toposort()] == [erfc]
        f(val)

        f = function([x], 1 + (-erf(x)), mode=self.mode)
        assert [n.op for n in f.maker.fgraph.toposort()] == [erfc]

        f = function([x], (-erf(x)) + 1, mode=self.mode)
        assert [n.op for n in f.maker.fgraph.toposort()] == [erfc]

        f = function([x], (-1.0 * erf(x)) + 1, mode=self.mode)
        assert [n.op for n in f.maker.fgraph.toposort()] == [erfc]

        f = function([x], 2 - erf(x), mode=self.mode)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 2
        assert topo[0].op == erf
        assert isinstance(topo[1].op, Elemwise)
        assert isinstance(topo[1].op.scalar_op, aes.Add) or isinstance(
            topo[1].op.scalar_op, aes.Sub
        )

    def test_local_erf_minus_one(self):
        val = np.asarray([-30, -3, -2, -1, 0, 1, 2, 3, 30], dtype=config.floatX)
        x = vector()

        f = function([x], erf(x) - 1, mode=self.mode)
        assert [n.op for n in f.maker.fgraph.toposort()] == [erfc, neg]
        f(val)

        f = function([x], erf(x) + (-1), mode=self.mode)
        assert [n.op for n in f.maker.fgraph.toposort()] == [erfc, neg]

        f = function([x], -1 + erf(x), mode=self.mode)
        assert [n.op for n in f.maker.fgraph.toposort()] == [erfc, neg]

        f = function([x], erf(x) - 2, mode=self.mode)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 2
        assert topo[0].op == erf
        assert isinstance(topo[1].op, Elemwise)
        assert isinstance(topo[1].op.scalar_op, aes.Add) or isinstance(
            topo[1].op.scalar_op, aes.Sub
        )


@pytest.mark.skipif(
    config.cxx == "",
    reason="erf need a c++ compiler or scipy",
)
class TestLocalErfc:
    def setup_method(self):
        self.mode_fusion = (
            get_default_mode()
            .including("canonicalize", "fast_run")
            .excluding("gpu", "inplace")
        )
        self.mode = self.mode_fusion.excluding("fusion")

    def test_local_one_minus_erfc(self):
        """Test the rewrites ``1 - erfc(x) -> erf(x)`` and ``-erfc(x) + 1 -> erf(x)``."""

        val = np.asarray([-30, -3, -2, -1, 0, 1, 2, 3, 30], dtype=config.floatX)
        x = vector("x")

        f = function([x], 1 - erfc(x), mode=self.mode)
        assert [n.op for n in f.maker.fgraph.toposort()] == [erf]
        f(val)

        f = function([x], (-erfc(x)) + 1, mode=self.mode)
        assert [n.op for n in f.maker.fgraph.toposort()] == [erf]

        f = function([x], (-1.0 * erfc(x)) + 1, mode=self.mode)
        assert [n.op for n in f.maker.fgraph.toposort()] == [erf]

        f = function([x], 2 - erfc(x), mode=self.mode)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 2
        assert topo[0].op == erfc
        assert isinstance(topo[1].op, Elemwise)
        assert isinstance(topo[1].op.scalar_op, aes.Sub)

    def test_local_erf_neg_minus_one(self):
        """Test the rewrite ``-1 + erfc(-x) -> erf(x)``."""
        val = np.asarray([-30, -3, -2, -1, 0, 1, 2, 3, 30], dtype=config.floatX)
        x = vector("x")

        f = function([x], -1 + erfc(-x), mode=self.mode)
        assert [n.op for n in f.maker.fgraph.toposort()] == [erf]
        f(val)

        f = function([x], erfc(-x) - 1, mode=self.mode)
        assert [n.op for n in f.maker.fgraph.toposort()] == [erf]

        f = function([x], erfc(-x) + (-1), mode=self.mode)
        assert [n.op for n in f.maker.fgraph.toposort()] == [erf]

        f = function([x], erfc(-1.0 * x) + (-1), mode=self.mode)
        assert [n.op for n in f.maker.fgraph.toposort()] == [erf]

    def test_local_log_erfc(self):
        val = [-30, -27, -26, -11, -10, -3, -2, -1, 0, 1, 2, 3, 10, 11, 26, 27, 28, 30]
        if config.mode in ["DebugMode", "DEBUG_MODE", "FAST_COMPILE"]:
            # python mode doesn't like the reciprocal(0)
            val.remove(0)
        val = np.asarray(val, dtype=config.floatX)
        x = vector("x")

        # their are some `nan`s that will appear in the graph due to the logs
        # of negatives values
        mode = copy.copy(self.mode)
        mode.check_isfinite = False
        mode_fusion = copy.copy(self.mode_fusion)
        mode_fusion.check_isfinite = False

        f = function([x], log(erfc(x)), mode=mode)
        assert len(f.maker.fgraph.apply_nodes) == 22
        assert f.maker.fgraph.outputs[0].dtype == config.floatX
        assert all(np.isfinite(f(val)))

        f = function([x], log(erfc(-x)), mode=mode)
        assert len(f.maker.fgraph.apply_nodes) == 23
        assert f.maker.fgraph.outputs[0].dtype == config.floatX
        assert all(np.isfinite(f(-val)))

        f = function([x], log(erfc(x)), mode=mode_fusion)
        assert len(f.maker.fgraph.apply_nodes) == 1
        assert f.maker.fgraph.outputs[0].dtype == config.floatX
        assert len(f.maker.fgraph.toposort()[0].op.scalar_op.fgraph.apply_nodes) == 22

        # TODO: fix this problem: The python code upcast somewhere internally
        #  some value of float32 to python float for part of its computation.
        #  That makes the c and python code generate sligtly different values
        if not (
            config.floatX == "float32" and config.mode in ["DebugMode", "DEBUG_MODE"]
        ):
            assert all(np.isfinite(f(val)))

    @np.errstate(divide="ignore", invalid="ignore")
    def test_local_grad_log_erfc_neg(self):
        # TODO: This evaluation is questionable; is the transform's math not
        # already established?  It doesn't look like these tests are preforming
        # a real numerical evaluation of the underlying math.  Instead, it
        # looks like they're being used as an extremely poor way of validating
        # the transform results.  It would be better to remove these numerical
        # evaluations and confirm the transform output directly and exactly.
        val = [
            -100,
            -30,
            -27,
            -26.4,
            -26.2,
            -26,
            -11,
            -10,
            -9,
            -3,
            -2,
            -1,
            0,
            1,
            2,
            3,
            9,
            10,
            11,
            27,
            26.4,
            26.2,
            26,
            28,
            30,
            100,
        ]
        val = np.asarray(val, dtype=config.floatX)
        x = vector("x")
        y = vector("y")

        # Test cases for which the requisite form isn't present
        no_matches = [
            ([x, y], exp(square(x)) / erfc(y)),
            ([x, y], exp(neg(x)) / erfc(y)),
            ([x, y], exp(x * 1) / erfc(y)),
            ([x, y], exp(neg(square(x))) / erfc(y)),
            ([x], mul(1.0, 2.0, x) / erfc(x)),
        ]
        for inputs, no_match in no_matches:
            fg = FunctionGraph(inputs, [no_match], clone=False)

            WalkingGraphRewriter(
                SequentialNodeRewriter(local_grad_log_erfc_neg), order="out_to_in"
            ).rewrite(fg)

            # Make sure that the graph hasn't been changed
            assert fg.outputs[0] is no_match

        # Some `nan`s will appear in the graph for the log of negatives values
        mode = Mode("py", self.mode.optimizer)
        mode.check_isfinite = False

        # Make sure that we catch our target graph in a way that it's naturally
        # produced
        log_erfc_grad = aesara.gradient.grad(log(erfc(x)).sum(), x)
        f = function([x], log_erfc_grad, mode=mode)

        # The resulting graph should be `mul(switch(...), y)`
        assert f.maker.fgraph.outputs[0].owner.op == mul
        assert f.maker.fgraph.outputs[0].owner.inputs[0].owner.op == switch
        assert all(np.isfinite(f(val)))
        assert f.maker.fgraph.outputs[0].dtype == config.floatX

        # Test with a different `mul` and `constant`
        f = function([x], mul(exp(neg(square(x))), -10.12837917) / erfc(x), mode=mode)

        assert f.maker.fgraph.outputs[0].owner.op == mul
        assert f.maker.fgraph.outputs[0].owner.inputs[0].owner.op == switch
        assert f.maker.fgraph.outputs[0].dtype == config.floatX
        assert all(np.isfinite(f(val)))

        # Test it works without the `mul`
        f = function([x], exp(neg(square(x))) / erfc(x), mode=mode)

        assert f.maker.fgraph.outputs[0].owner.op == switch
        assert f.maker.fgraph.outputs[0].dtype == config.floatX
        assert all(np.isfinite(f(val)))

        # Test that it works without the `square` and `neg`
        f = function([x], exp(mul(-1, x, x)) / erfc(x), mode=mode)

        assert f.maker.fgraph.outputs[0].owner.op == switch
        assert f.maker.fgraph.outputs[0].dtype == config.floatX
        assert all(np.isfinite(f(val)))

        # Test that it works correctly when `x` is multiplied by a constant
        f = function([x], aesara.gradient.grad(log(erfc(2 * x)).sum(), x), mode=mode)

        assert f.maker.fgraph.outputs[0].owner.op == mul
        assert f.maker.fgraph.outputs[0].owner.inputs[0].owner.op == switch
        assert np.isfinite(f(val)).all()
        assert f.maker.fgraph.outputs[0].dtype == config.floatX

        # I suppose this tests whether or not the transform is applied before
        # fusion?
        mode_fusion = copy.copy(self.mode_fusion)
        mode_fusion.check_isfinite = False

        f = function([x], aesara.gradient.grad(log(erfc(x)).sum(), x), mode=mode_fusion)

        assert len(f.maker.fgraph.apply_nodes) == 1, len(f.maker.fgraph.apply_nodes)
        assert f.maker.fgraph.outputs[0].dtype == config.floatX

    def speed_local_log_erfc(self):
        val = np.random.random(1e6)
        x = vector()
        mode = get_mode("FAST_RUN")
        f1 = function([x], log(erfc(x)), mode=mode.excluding("local_log_erfc"))
        f2 = function([x], log(erfc(x)), mode=mode)
        print(f1.maker.fgraph.toposort())
        print(f2.maker.fgraph.toposort())
        t0 = time.perf_counter()
        f1(val)
        t1 = time.perf_counter()
        f2(val)
        t2 = time.perf_counter()
        print(t1 - t0, t2 - t1)


class TestLocalMergeSwitchSameCond:
    def test_elemwise(self):
        # float Ops
        mats = matrices("cabxy")
        c, a, b, x, y = mats
        s1 = at.switch(c, a, b)
        s2 = at.switch(c, x, y)
        for op in (
            add,
            sub,
            mul,
            true_divide,
            floor_divide,
            minimum,
            maximum,
            gt,
            lt,
            ge,
            le,
            eq,
            neq,
            at_pow,
        ):
            g = rewrite(FunctionGraph(mats, [op(s1, s2)]))
            assert str(g).count("Switch") == 1
        # integer Ops
        mats = imatrices("cabxy")
        c, a, b, x, y = mats
        s1 = at.switch(c, a, b)
        s2 = at.switch(c, x, y)
        for op in (
            bitwise_and,
            bitwise_or,
            bitwise_xor,
        ):
            g = rewrite(FunctionGraph(mats, [op(s1, s2)]))
            assert str(g).count("Switch") == 1
        # add/mul with more than two inputs
        u, v = matrices("uv")
        s3 = at.switch(c, u, v)
        for op in (add, mul):
            g = rewrite(FunctionGraph(mats + [u, v], [op(s1, s2, s3)]))
            assert str(g).count("Switch") == 1


class TestLocalSumProd:
    """Test sum/prod rewrites."""

    def setup_method(self):
        self.mode = get_default_mode().including("canonicalize", "specialize")

    def test_local_sum_prod_mul_by_scalar(self):
        # Test the rewrite `local_sum_prod_mul_by_scalar` for both Sum and
        # Prod ops in six cases each :
        # 1-the inputs to the mul contain a scalar and no non-scalar
        # 2-the inputs to the mul contain a scalar and one non-scalar
        # 3-the inputs to the mul contain a scalar and two non-scalars
        # 4-the inputs to the mul contain two scalars and no non-scalar
        # 5-the inputs to the mul contain two scalars and one non-scalar
        # 6-the inputs to the mul contain two scalars and two non-scalars

        vect = dvector()
        mat = dmatrix()
        scalar1 = dscalar()
        scalar2 = dscalar()

        v_val = np.random.random(2)
        m_val = np.random.random((2, 2))
        s1_val = np.random.random()
        s2_val = np.random.random()

        def test_reduction_rewrite(
            inputs, inputs_val, reduction_op, expected_output, nb_expected_sum_nodes
        ):
            mul_out = mul(*inputs)
            f = function(inputs, reduction_op()(mul_out), mode=self.mode)
            out = f(*inputs_val)
            utt.assert_allclose(out, expected_output)

            # Ensure that the rewrite has been applied properly by
            # ensuring that the rewritten graph contains the expected number
            # of apply nodes for the sum op
            prod_nodes = [
                n for n in f.maker.fgraph.toposort() if isinstance(n.op, reduction_op)
            ]
            assert len(prod_nodes) == nb_expected_sum_nodes

        # Test sum

        # Case 1
        test_reduction_rewrite([scalar1], [s1_val], Sum, s1_val, 0)

        # Case 2
        test_reduction_rewrite(
            [vect, scalar1], [v_val, s1_val], Sum, s1_val * v_val.sum(), 1
        )

        # Case 3
        test_reduction_rewrite(
            [vect, mat, scalar1],
            [v_val, m_val, s1_val],
            Sum,
            s1_val * (v_val * m_val).sum(),
            1,
        )

        # Case 4
        test_reduction_rewrite(
            [scalar1, scalar2], [s1_val, s2_val], Sum, s1_val * s2_val, 0
        )

        # Case 5
        test_reduction_rewrite(
            [vect, scalar1, scalar2],
            [v_val, s1_val, s2_val],
            Sum,
            s1_val * s2_val * v_val.sum(),
            1,
        )

        # Case 6
        test_reduction_rewrite(
            [vect, mat, scalar1, scalar2],
            [v_val, m_val, s1_val, s2_val],
            Sum,
            s1_val * s2_val * (v_val * m_val).sum(),
            1,
        )

        # Test prod

        # Case 1
        test_reduction_rewrite([scalar1], [s1_val], Prod, s1_val, 0)

        # Case 2
        test_reduction_rewrite(
            [vect, scalar1],
            [v_val, s1_val],
            Prod,
            (s1_val * v_val).prod(),
            1,
        )

        # Case 3
        test_reduction_rewrite(
            [vect, mat, scalar1],
            [v_val, m_val, s1_val],
            Prod,
            (s1_val * v_val * m_val).prod(),
            2,
        )

        # Case 4
        test_reduction_rewrite(
            [scalar1, scalar2], [s1_val, s2_val], Prod, s1_val * s2_val, 0
        )

        # Case 5
        test_reduction_rewrite(
            [vect, scalar1, scalar2],
            [v_val, s1_val, s2_val],
            Prod,
            (s1_val * s2_val * v_val).prod(),
            1,
        )

        # Case 6
        test_reduction_rewrite(
            [vect, mat, scalar1, scalar2],
            [v_val, m_val, s1_val, s2_val],
            Prod,
            (s1_val * s2_val * v_val * m_val).prod(),
            2,
        )

    def test_local_sum_prod_all_to_none(self):
        a = tensor3()
        input = np.arange(3 * 4 * 5, dtype=config.floatX).reshape(3, 4, 5)
        # test sum
        f = function([a], a.sum(), mode=self.mode)
        assert len(f.maker.fgraph.apply_nodes) == 1
        utt.assert_allclose(f(input), input.sum())
        # test prod
        f = function([a], a.prod(), mode=self.mode)
        assert len(f.maker.fgraph.apply_nodes) == 1
        utt.assert_allclose(f(input), input.prod())
        # test sum
        f = function([a], a.sum([0, 1, 2]), mode=self.mode)
        assert len(f.maker.fgraph.apply_nodes) == 1
        utt.assert_allclose(f(input), input.sum())
        # test prod
        f = function([a], a.prod([0, 1, 2]), mode=self.mode)
        assert len(f.maker.fgraph.apply_nodes) == 1
        utt.assert_allclose(f(input), input.prod())

        f = function([a], a.sum(0).sum(0).sum(0), mode=self.mode)
        assert len(f.maker.fgraph.apply_nodes) == 1
        utt.assert_allclose(f(input), input.sum())

    def test_local_sum_sum_prod_prod(self):
        a = tensor3()
        input = np.arange(3 * 4 * 5, dtype=config.floatX).reshape(3, 4, 5)
        dims = [
            (0, 0),
            (1, 0),
            (2, 0),
            (0, 1),
            (1, 1),
            (2, 1),
            ((0, 1), 0),
            ((1, 2), 0),
            (0, (0, 1)),
            (1, (0, 1)),
            (2, (0, 1)),
        ]

        def my_prod(data, d, dd):
            # This prod when d or dd is a tuple of 2 dimensions.
            if not isinstance(d, tuple) and not isinstance(dd, tuple):
                return data.prod(d).prod(dd)
            if isinstance(d, tuple):
                d = sorted(d)
                return data.prod(d[1]).prod(d[0]).prod(dd)
            else:
                dd = sorted(dd)
                return data.prod(d).prod(dd[1]).prod(dd[0])

        def my_sum(data, d, dd):
            # This sum when d or dd is a tuple of 2 dimensions.
            if not isinstance(d, tuple) and not isinstance(dd, tuple):
                return data.sum(d).sum(dd)
            if isinstance(d, tuple):
                d = sorted(d)
                return data.sum(d[1]).sum(d[0]).sum(dd)
            else:
                dd = sorted(dd)
                return data.sum(d).sum(dd[1]).sum(dd[0])

        def my_sum_prod(data, d, dd):
            # This sum when d or dd is a tuple of 2 dimensions.
            if not isinstance(d, tuple) and not isinstance(dd, tuple):
                return data.sum(d).prod(dd)
            if isinstance(d, tuple):
                d = sorted(d)
                return data.sum(d[1]).sum(d[0]).prod(dd)
            else:
                dd = sorted(dd)
                return data.sum(d).prod(dd[1]).prod(dd[0])

        for d, dd in dims:
            expected = my_sum(input, d, dd)
            f = function([a], a.sum(d).sum(dd), mode=self.mode)
            utt.assert_allclose(f(input), expected)
            assert len(f.maker.fgraph.apply_nodes) == 1
        for d, dd in dims[:6]:
            f = function([a], a.sum(d).sum(dd).sum(0), mode=self.mode)
            utt.assert_allclose(f(input), input.sum(d).sum(dd).sum(0))
            assert len(f.maker.fgraph.apply_nodes) == 1
        for d in [0, 1, 2]:
            f = function([a], a.sum(d).sum(None), mode=self.mode)
            utt.assert_allclose(f(input), input.sum(d).sum())
            assert len(f.maker.fgraph.apply_nodes) == 1
        f = function([a], a.sum(None).sum(), mode=self.mode)
        utt.assert_allclose(f(input), input.sum())
        assert len(f.maker.fgraph.apply_nodes) == 1

        # test prod
        for d, dd in dims:
            expected = my_prod(input, d, dd)
            f = function([a], a.prod(d).prod(dd), mode=self.mode)
            utt.assert_allclose(f(input), expected)
            assert len(f.maker.fgraph.apply_nodes) == 1
        for d, dd in dims[:6]:
            f = function([a], a.prod(d).prod(dd).prod(0), mode=self.mode)
            utt.assert_allclose(f(input), input.prod(d).prod(dd).prod(0))
            assert len(f.maker.fgraph.apply_nodes) == 1
        for d in [0, 1, 2]:
            f = function([a], a.prod(d).prod(None), mode=self.mode)
            utt.assert_allclose(f(input), input.prod(d).prod())
            assert len(f.maker.fgraph.apply_nodes) == 1
        f = function([a], a.prod(None).prod(), mode=self.mode)
        utt.assert_allclose(f(input), input.prod())
        assert len(f.maker.fgraph.apply_nodes) == 1

        # Test that sum prod didn't get rewritten.
        for d, dd in dims:
            expected = my_sum_prod(input, d, dd)
            f = function([a], a.sum(d).prod(dd), mode=self.mode)
            utt.assert_allclose(f(input), expected)
            assert len(f.maker.fgraph.apply_nodes) == 2
        for d, dd in dims[:6]:
            f = function([a], a.sum(d).prod(dd).prod(0), mode=self.mode)
            utt.assert_allclose(f(input), input.sum(d).prod(dd).prod(0))
            assert len(f.maker.fgraph.apply_nodes) == 2
        for d in [0, 1, 2]:
            f = function([a], a.sum(d).prod(None), mode=self.mode)
            utt.assert_allclose(f(input), input.sum(d).prod())
            assert len(f.maker.fgraph.apply_nodes) == 2
        f = function([a], a.sum(None).prod(), mode=self.mode)
        utt.assert_allclose(f(input), input.sum())
        assert len(f.maker.fgraph.apply_nodes) == 1

    def test_local_sum_prod_alloc(self):
        a = dtensor3()
        input = np.asarray(np.arange(2 * 3 * 4).reshape(2, 3, 4), dtype="float64")
        mode = self.mode.including("specialize").excluding("fusion")

        for t_like, n_like, nb_nodes in [
            (at.zeros_like, np.zeros_like, (1, 3, 3, 2)),
            (at.ones_like, np.ones_like, (5, 5, 5, 6)),
        ]:
            # test sum
            f = function([a], t_like(a).sum(None), mode=mode)
            utt.assert_allclose(f(input), n_like(input).sum())
            assert len(f.maker.fgraph.apply_nodes) == nb_nodes[0]

            f = function([a], t_like(a).sum([0, 1, 2]), mode=mode)
            utt.assert_allclose(f(input), n_like(input).sum())
            assert len(f.maker.fgraph.apply_nodes) == nb_nodes[0]

            for d in range(3):
                f = function([a], t_like(a).sum(d), mode=mode)
                utt.assert_allclose(f(input), n_like(input).sum(d))
                assert len(f.maker.fgraph.apply_nodes) == nb_nodes[1]
                topo = f.maker.fgraph.toposort()
                assert topo[-1].op == at.alloc
                assert not any(isinstance(node.op, Sum) for node in topo)
            for i in range(3):
                f = function([a], t_like(a).sum(i), mode=mode)
                utt.assert_allclose(f(input), n_like(input).sum(i))
                assert len(f.maker.fgraph.apply_nodes) == nb_nodes[2]
                topo = f.maker.fgraph.toposort()
                assert topo[-1].op == at.alloc
                assert not any(isinstance(node.op, Sum) for node in topo)

            # test prod
            f = function([a], t_like(a).prod(None), mode=mode)
            utt.assert_allclose(f(input), n_like(input).prod())
            # assert len(f.maker.fgraph.apply_nodes) == nb_nodes[0]

            f = function([a], t_like(a).prod([0, 1, 2]), mode=mode)
            utt.assert_allclose(f(input), n_like(input).prod())
            # assert len(f.maker.fgraph.apply_nodes) == nb_nodes[0]

            for d in range(3):
                f = function([a], t_like(a).prod(d), mode=mode)
                utt.assert_allclose(f(input), n_like(input).prod(d))
                # assert len(f.maker.fgraph.apply_nodes) == nb_nodes[1]
                topo = f.maker.fgraph.toposort()
                assert topo[-1].op == at.alloc
                assert not any(isinstance(node.op, Prod) for node in topo)
            for i in range(3):
                f = function([a], t_like(a).prod(i), mode=mode)
                utt.assert_allclose(f(input), n_like(input).prod(i))
                # assert len(f.maker.fgraph.apply_nodes) == nb_nodes[2]
                topo = f.maker.fgraph.toposort()
                assert topo[-1].op == at.alloc
                assert not any(isinstance(node.op, Prod) for node in topo)

            for d, dd in [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]:
                f = function([a], t_like(a).sum(d).sum(dd), mode=mode)
                utt.assert_allclose(f(input), n_like(input).sum(d).sum(dd))
                assert len(f.maker.fgraph.apply_nodes) == nb_nodes[3]
                topo = f.maker.fgraph.toposort()
                assert topo[-1].op == at.alloc
                assert not any(isinstance(node.op, Sum) for node in topo)

    def test_local_sum_sum_int8(self):
        """Test that `local_sum_sum` works when combining two sums on an int8 array.

        This is a regression test for ticket gh-356.
        """

        x = tensor3(dtype="int8")
        y = x.sum(axis=0).sum(axis=1)

        with config.change_flags(on_opt_error="raise"):
            # This compilation would fail prior to fix.
            function([x], y)

    def test_local_sum_sum_dtype(self):
        """Test that `local_sum_sum` works when specifying dtypes manually."""

        x = tensor3(dtype="int8")
        y = x.sum(axis=0, dtype="int32").sum(axis=1, dtype="int64")

        with config.change_flags(on_opt_error="raise"):
            # This compilation would fail prior to fix.
            function([x], y)

    def test_local_sum_prod_mul_by_scalar_stack_trace(self):
        """Test that stack trace is copied over correctly for `local_sum_prod_mul_by_scalar`."""
        m0 = (
            get_default_mode()
            .excluding("inplace_elemwise_opt")
            .including("canonicalize", "specialize")
        )

        vect = dvector()
        mat = dmatrix()
        ds = dscalar()

        f = function([vect, ds], at_sum(vect * ds), mode=m0)
        assert check_stack_trace(f, ops_to_check="all")

        f = function([vect], at_sum(-vect), mode=m0)
        assert check_stack_trace(f, ops_to_check=[Sum])

        f = function([vect, ds], Prod()(vect * ds), mode=m0)
        assert check_stack_trace(f, ops_to_check=[Prod])

        f = function([vect], Prod()(-vect), mode=m0)
        assert check_stack_trace(f, ops_to_check=[Prod])

        f = function([mat, ds], at_sum(mat * ds), mode=m0)
        assert check_stack_trace(f, ops_to_check="all")

        f = function([mat], at_sum(-mat), mode=m0)
        assert check_stack_trace(f, ops_to_check=[Sum])


class TestLocalReduce:
    def setup_method(self):
        self.mode = get_default_mode().including(
            "canonicalize", "specialize", "uncanonicalize", "local_max_and_argmax"
        )

    def test_local_reduce_broadcast_all_0(self):
        for fct in [
            at_sum,
            at_all,
            at_any,
            prod,
            at_max,
            at_min,
        ]:
            x = TensorType("int64", shape=(1, 1, 1))()
            f = function([x], [fct(x)], mode=self.mode)
            assert not any(
                isinstance(node.op, CAReduce) for node in f.maker.fgraph.toposort()
            )

    def test_local_reduce_broadcast_all_1(self):
        for fct in [
            at_sum,
            at_all,
            at_any,
            prod,
            at_max,
            at_min,
        ]:
            x = TensorType("int64", shape=(1, 1))()
            f = function([x], [fct(x, axis=[0, 1])], mode=self.mode)
            assert not any(
                isinstance(node.op, CAReduce) for node in f.maker.fgraph.toposort()
            )

    def test_local_reduce_broadcast_some_0(self):
        for fct in [
            at_sum,
            at_all,
            at_any,
            prod,
            at_max,
            at_min,
        ]:
            x = TensorType("int64", shape=(1, None, 1))()
            f = function([x], [fct(x, axis=[0, 1])], mode=self.mode)

            order = f.maker.fgraph.toposort()
            assert 1 == sum(isinstance(node.op, CAReduce) for node in order)

            node = [node for node in order if isinstance(node.op, CAReduce)][0]

            op = node.op
            assert isinstance(op, CAReduce)
            # The leading broadcastable dimension has been dropped by the
            # `local_reduce_broadcastable` rewrite.  Now, summation is over
            # the original `x`'s dimension 1.
            assert node.inputs[0].ndim == 2, node
            assert op.axis == (0,), op.axis

    def test_local_reduce_broadcast_some_1(self):
        for fct in [
            at_sum,
            at_all,
            at_any,
            prod,
            at_max,
            at_min,
        ]:
            x = TensorType("int64", shape=(1, 1, 1))()
            f = function([x], [fct(x, axis=[0, 2])], mode=self.mode)
            assert not any(
                isinstance(node.op, CAReduce) for node in f.maker.fgraph.toposort()
            )

    def test_local_reduce_join(self):
        vx = matrix()
        vy = matrix()
        vz = matrix()
        x = np.asarray([[1, 0], [3, 4]], dtype=config.floatX)
        y = np.asarray([[4, 0], [2, 1]], dtype=config.floatX)
        z = np.asarray([[5, 0], [1, 2]], dtype=config.floatX)
        # Test different reduction scalar operation
        for out, res in [
            (at_max((vx, vy), 0), np.max((x, y), 0)),
            (at_min((vx, vy), 0), np.min((x, y), 0)),
            (at_sum((vx, vy, vz), 0), np.sum((x, y, z), 0)),
            (prod((vx, vy, vz), 0), np.prod((x, y, z), 0)),
            (prod((vx, vy.T, vz), 0), np.prod((x, y.T, z), 0)),
        ]:
            f = function([vx, vy, vz], out, on_unused_input="ignore", mode=self.mode)
            assert (f(x, y, z) == res).all(), out
            topo = f.maker.fgraph.toposort()
            assert len(topo) <= 2, out
            assert isinstance(topo[-1].op, Elemwise), out

        # Test different axis for the join and the reduction
        # We must force the dtype, of otherwise, this tests will fail
        # on 32 bit systems
        A = shared(np.array([1, 2, 3, 4, 5], dtype="int64"))

        f = function([], at_sum(at.stack([A, A]), axis=0), mode=self.mode)
        utt.assert_allclose(f(), [2, 4, 6, 8, 10])
        topo = f.maker.fgraph.toposort()
        assert isinstance(topo[-1].op, Elemwise)

        # Test a case that was bugged in a old Aesara bug
        f = function([], at_sum(at.stack([A, A]), axis=1), mode=self.mode)

        utt.assert_allclose(f(), [15, 15])
        topo = f.maker.fgraph.toposort()
        assert not isinstance(topo[-1].op, Elemwise)

        # This case could be rewritten
        A = shared(np.array([1, 2, 3, 4, 5]).reshape(5, 1))
        f = function([], at_sum(at.concatenate((A, A), axis=1), axis=1), mode=self.mode)
        utt.assert_allclose(f(), [2, 4, 6, 8, 10])
        topo = f.maker.fgraph.toposort()
        assert not isinstance(topo[-1].op, Elemwise)

        A = shared(np.array([1, 2, 3, 4, 5]).reshape(5, 1))
        f = function([], at_sum(at.concatenate((A, A), axis=1), axis=0), mode=self.mode)
        utt.assert_allclose(f(), [15, 15])
        topo = f.maker.fgraph.toposort()
        assert not isinstance(topo[-1].op, Elemwise)

        # Test that the rewrite does not crash in one case where it
        # is not applied.  Reported at
        # https://groups.google.com/d/topic/theano-users/EDgyCU00fFA/discussion
        out = at_sum([vx, vy, vz], axis=None)
        f = function([vx, vy, vz], out)


class TestLocalSumProdDimshuffle:
    def setup_method(self):
        self.mode = get_default_mode().including("canonicalize")

    def test_local_sum_div_dimshuffle(self):
        a = matrix("a")
        b = vector("b")
        c = tensor3("c")
        d = scalar("d")
        sum = at_sum
        sums = [
            sum(a / d),
            sum(a / d.dimshuffle("x", "x")),
            sum(a / d.dimshuffle("x", "x"), axis=0),
            sum(a / d.dimshuffle("x", "x"), axis=1),
            sum(b / d),
            sum(b / d.dimshuffle("x")),
            sum(c / d),
            sum(c / d.dimshuffle("x", "x", "x")),
            sum(c / d.dimshuffle("x", "x", "x"), axis=0),
            sum(c / d.dimshuffle("x", "x", "x"), axis=1),
            sum(c / d.dimshuffle("x", "x", "x"), axis=2),
            sum(a / b, axis=0),
            sum(a / b.dimshuffle(0, "x"), axis=1),
            sum(a.dimshuffle(0, 1) / b.dimshuffle(0, "x"), axis=1),
            sum(a.dimshuffle(1, 0) / b.dimshuffle(0, "x"), axis=1),
            sum(c / a, axis=0),
            sum(c / a.dimshuffle(1, 0), axis=0),
            sum(c / a.dimshuffle(0, "x", 1), axis=1),
            sum(c / a.dimshuffle(1, "x", 0), axis=1),
            sum(c / a.dimshuffle(0, 1, "x"), axis=2),
            sum(c / a.dimshuffle(1, 0, "x"), axis=2),
            sum(c / b, axis=0),
            sum(c / b, axis=1),
            sum(c / b, axis=(0, 1)),
            sum(c / b.dimshuffle(0, "x"), axis=0),
            sum(c / b.dimshuffle(0, "x"), axis=2),
            sum(c / b.dimshuffle(0, "x"), axis=(0, 2)),
            sum(c / b.dimshuffle(0, "x", "x"), axis=1),
            sum(c / b.dimshuffle(0, "x", "x"), axis=2),
            sum(c / b.dimshuffle(0, "x", "x"), axis=(1, 2)),
            sum(sum(c, axis=0) / b, axis=0),
            sum(sum(c, axis=1) / b, axis=0),
        ]

        rng = np.random.default_rng(utt.fetch_seed())
        a_val = rng.standard_normal((2, 2)).astype(config.floatX)
        b_val = rng.standard_normal(2).astype(config.floatX)
        c_val = rng.standard_normal((2, 2, 2)).astype(config.floatX)
        d_val = np.asarray(rng.standard_normal(), config.floatX)

        for i, s in enumerate(sums):
            f = function([a, b, c, d], s, mode=self.mode, on_unused_input="ignore")
            g = f.maker.fgraph.toposort()
            assert isinstance(g[-1].op.scalar_op, aes.basic.TrueDivide)
            f(a_val, b_val, c_val, d_val)

    def test_local_prod_div_dimshuffle(self):
        a = matrix("a")
        b = vector("b")
        c = tensor3("c")
        e = matrix("e")
        d = scalar("d")
        prods = [
            prod(a / d),
            prod(a / d.dimshuffle("x", "x")),
            prod(a / d.dimshuffle("x", "x"), axis=0),
            prod(a / d.dimshuffle("x", "x"), axis=1),
            prod(b / d),
            prod(b / d.dimshuffle("x")),
            prod(c / d),
            prod(c / d.dimshuffle("x", "x", "x")),
            prod(c / d.dimshuffle("x", "x", "x"), axis=0),
            prod(c / d.dimshuffle("x", "x", "x"), axis=1),
            prod(c / d.dimshuffle("x", "x", "x"), axis=2),
            prod(a / b, axis=0),
            prod(a / b.dimshuffle(0, "x"), axis=1),
            prod(a.dimshuffle(0, 1) / b.dimshuffle(0, "x"), axis=1),
            prod(a.dimshuffle(1, 0) / b.dimshuffle(0, "x"), axis=1),
            prod(c / a, axis=0),
            prod(c / a.dimshuffle(1, 0), axis=0),
            prod(c / a.dimshuffle(0, "x", 1), axis=1),
            prod(c / a.dimshuffle(1, "x", 0), axis=1),
            prod(c / a.dimshuffle(0, 1, "x"), axis=2),
            prod(c / a.dimshuffle(1, 0, "x"), axis=2),
            prod(c / b, axis=0),
            prod(c / b, axis=1),
            prod(c / b, axis=(0, 1)),
            prod(c / b.dimshuffle(0, "x"), axis=0),
            prod(c / b.dimshuffle(0, "x"), axis=2),
            prod(c / b.dimshuffle(0, "x"), axis=(0, 2)),
            prod(c / b.dimshuffle(0, "x", "x"), axis=1),
            prod(c / b.dimshuffle(0, "x", "x"), axis=2),
            prod(c / b.dimshuffle(0, "x", "x"), axis=(1, 2)),
            prod(c / b.dimshuffle(0, "x", "x"), axis=(0, 1)),
            prod(c / b.dimshuffle(0, "x", "x"), axis=(1, 0)),
            prod(prod(c, axis=0) / b, axis=0),
            prod(prod(c, axis=1) / b, axis=0),
        ]

        rng = np.random.default_rng(utt.fetch_seed())
        a_val = rng.standard_normal((2, 2)).astype(config.floatX)
        b_val = rng.standard_normal(2).astype(config.floatX)
        c_val = rng.standard_normal((2, 2, 2)).astype(config.floatX)
        d_val = np.asarray(rng.standard_normal(), config.floatX)

        default_mode = get_default_mode()
        # `FusionOptimizer` is included to make sure that `expected_outer_operator`
        # remains the same for all rewrite modes.
        mode_with_rewrite = default_mode.including(
            "local_sum_prod_div_dimshuffle", "FusionOptimizer"
        )
        mode_without_rewrite = default_mode.excluding("local_sum_prod_div_dimshuffle")

        # Numerical tests: tests whether the numerical values with and without
        # rewrites are equal or not.
        for i, s in enumerate(prods):
            f = function(
                [a, b, c, d], s, on_unused_input="ignore", mode=mode_without_rewrite
            )
            g = function(
                [a, b, c, d], s, on_unused_input="ignore", mode=mode_with_rewrite
            )

            utt.assert_allclose(
                f(a_val, b_val, c_val, d_val), g(a_val, b_val, c_val, d_val)
            )

        # Logical tests: tests whether the rewrite has been appplied or not
        # by checking graph structure.
        prods = [
            prod(a / e),
            prod(a / d),
            prod(a / d.dimshuffle("x", "x")),
            prod(c / d.dimshuffle("x", "x", "x"), axis=1),
            prod(a.dimshuffle(1, 0) / b.dimshuffle(0, "x"), axis=1),
            prod(c / b.dimshuffle(0, "x", "x"), axis=(1, 0)),
            prod(prod(c, axis=1) / b, axis=0),
            prod(prod(c, axis=(1, 2)) / b, axis=0),
        ]

        expected_outer_operator = [
            aes.basic.Mul,
            aes.basic.Composite,
            aes.basic.Composite,
            aes.basic.TrueDivide,
            aes.basic.Composite,
            aes.basic.Mul,
            aes.basic.Composite,
            aes.basic.Mul,
        ]

        for i, s in enumerate(prods):
            g = function(
                [a, b, c, d, e], s, on_unused_input="ignore", mode=mode_with_rewrite
            )
            assert isinstance(
                g.maker.fgraph.toposort()[-1].op.scalar_op, expected_outer_operator[i]
            )

    # TODO:
    # test_local_sum_prod_dimshuffle (a * b * c)
    # test_local_sum_divprod_dimshuffle ((a * b) / (c * d))


def test_local_useless_adds():
    default_mode = get_default_mode()

    # Test for all zeros
    a = scalar()
    s = add(at.zeros_like(a))
    mode_with_rewrite = default_mode.including("canonicalization", "local_useless_fill")
    f = function([a], s, mode=mode_with_rewrite)
    assert not any(node.op == add for node in f.maker.fgraph.apply_nodes)

    # test of non-zero dimension
    a = vector()
    s = add(at.zeros_like(a))
    mode_with_rewrite = default_mode.including(
        "canonicalization", "local_useless_elemwise"
    )
    f = function([a], s, mode=mode_with_rewrite)
    assert not any(node.op == add for node in f.maker.fgraph.apply_nodes)

    # test of 0-d
    a = scalar()
    s = add(at.zeros_like(a))
    mode_with_rewrite = default_mode.including(
        "canonicalization", "local_useless_fill", "local_useless_elemwise"
    )
    f = function([a], s, mode=mode_with_rewrite)
    assert not any(node.op == add for node in f.maker.fgraph.apply_nodes)

    # Test when the 0 input is forcing upcasting
    a = at.constant(0, dtype="int64")
    b = at.constant(1, dtype="int32")
    s = a + b
    mode_with_rewrite = default_mode.including(
        "canonicalization", "local_add_canonizer"
    )
    f = function([], s, mode=mode_with_rewrite)
    transformed = f.maker.fgraph.outputs[0]
    assert not any(node.op == add for node in f.maker.fgraph.apply_nodes)
    assert transformed.type == s.type


def test_local_div_to_reciprocal():
    # XXX TODO: This does *not* test `local_div_to_reciprocal`!
    num_len_s = lscalar("num_len")
    denom_s = scalar("denom")

    num_v = at.alloc(1, num_len_s)
    denom_m = denom_s.dimshuffle("x", "x")

    out = num_v / denom_m
    assert out.broadcastable == (True, False)

    f = function([num_len_s, denom_s], out)
    out_val = f(3, 2.0)
    assert out_val.shape == (1, 3)
    utt.assert_allclose(out_val, 0.5)


class TestIntDivByOne:
    def setup_method(self):
        self.mode = get_default_mode()
        self.mode = self.mode.including("local_intdiv_by_one")

    def test_remove_floor(self):
        """Tests removing the extra floor_divide by 1 introduced by `local_subtensor_merge` rewrite."""

        y = tensor4("y")
        self.mode = self.mode.excluding("fusion")
        f = function([y], y[::-1][::-1], mode=self.mode)

        graph = f.maker.fgraph.toposort()
        divs = [
            node
            for node in graph
            if isinstance(node.op, Elemwise)
            and isinstance(node.op.scalar_op, aes.FloorDivide)
        ]
        assert len(divs) == 0

    def test2(self):
        # Simple test case for removing dividing by 1
        y = tensor4("y")
        z = y // 1
        f = function([y], z, mode=self.mode)
        graph = f.maker.fgraph.toposort()
        divs = [
            node
            for node in graph
            if isinstance(node.op, Elemwise)
            and isinstance(node.op.scalar_op, aes.FloorDivide)
        ]
        assert len(divs) == 0

    def test3(self):
        # Simple test case for removing dividing by a tensor of ones
        y = tensor4("y")
        z = y // np.ones((2, 2, 2, 2))
        f = function([y], z, mode=self.mode)
        graph = f.maker.fgraph.toposort()
        divs = [
            node
            for node in graph
            if isinstance(node.op, Elemwise)
            and isinstance(node.op.scalar_op, aes.FloorDivide)
        ]
        assert len(divs) == 0


@pytest.mark.parametrize("t", [scalar, ivector, ftensor4])
@pytest.mark.parametrize("op", [floor_divide, true_divide])
def test_local_zero_div(t, op):
    """Test the canonicalization ``0/x -> 0``."""
    x = t("x")
    y = op(0, x)
    g = rewrite(FunctionGraph([x], [y]))
    # The division should be gone
    divs = [
        node
        for node in g.toposort()
        if isinstance(node.op, Elemwise)
        and isinstance(node.op.scalar_op, type(op.scalar_op))
    ]
    assert len(divs) == 0
    # The output type should match the un-rewritten one
    output = g.outputs[0]
    assert output.ndim == y.ndim
    assert output.type == y.type
    # The output should be zero
    if output.owner and isinstance(output.owner.op, Alloc):
        out_var = output.owner.inputs[0]
    else:
        out_var = output

    assert out_var.data == 0


def test_local_sumsqr2dot():
    G = matrix("G")
    W = matrix("W")

    y = square(W.dimshuffle("x", 0, 1) * G.dimshuffle(0, "x", 1)).sum(axis=(1, 2))
    MODE = get_default_mode().including("local_sumsqr2dot")

    f = function([W, G], y, mode=MODE)

    w_val = np.random.random((4, 3)).astype(config.floatX)
    g_val = np.random.random((5, 3)).astype(config.floatX)

    f_val = f(w_val, g_val)
    f_test = np.dot(np.square(g_val), np.square(w_val).sum(axis=0))

    utt.assert_allclose(f_val, f_test)
    assert any(
        isinstance(
            n.op,
            (
                Dot,
                Dot22,
                Gemv,
                CGemv,
            ),
        )
        for n in f.maker.fgraph.toposort()
    )


def test_local_expm1():
    x = matrix("x")
    u = scalar("u")

    y = exp(x) - 1.0
    z = exp(x) - 2.0
    t = exp(x) - x
    s = exp(u) - np.ones((4, 3)).astype(config.floatX)
    MODE = get_default_mode().including("local_expm1")
    f = function([x], y, mode=MODE)
    g = function([x], z, mode=MODE)
    h = function([x], t, mode=MODE)
    r = function([u], s, mode=MODE)
    x_val = np.random.random((4, 3)).astype(config.floatX)
    f_val = f(x_val)
    f_test = function([x], expm1(x), mode=MODE)

    utt.assert_allclose(f_val, f_test(x_val))

    assert any(
        isinstance(n.op, Elemwise) and isinstance(n.op.scalar_op, aes.basic.Expm1)
        for n in f.maker.fgraph.toposort()
    )

    assert not any(
        isinstance(n.op, Elemwise) and isinstance(n.op.scalar_op, aes.basic.Expm1)
        for n in g.maker.fgraph.toposort()
    )

    assert not any(
        isinstance(n.op, Elemwise) and isinstance(n.op.scalar_op, aes.basic.Expm1)
        for n in h.maker.fgraph.toposort()
    )

    # This rewrite works when `local_add_neg_to_sub` specialization rewrite is invoked
    expect_rewrite = config.mode != "FAST_COMPILE"
    assert (
        any(
            isinstance(n.op, Elemwise) and isinstance(n.op.scalar_op, aes.basic.Expm1)
            for n in r.maker.fgraph.toposort()
        )
        == expect_rewrite
    )


def compile_graph_log_sum_exp(x, axis, dimshuffle_op=None):
    sum_exp = at_sum(exp(x), axis=axis)
    if dimshuffle_op:
        sum_exp = dimshuffle_op(sum_exp)
    y = log(sum_exp)
    MODE = get_default_mode().including("local_log_sum_exp")
    return function([x], y, mode=MODE)


def check_max_log_sum_exp(x, axis, dimshuffle_op=None):
    f = compile_graph_log_sum_exp(x, axis, dimshuffle_op)

    fgraph = f.maker.fgraph.toposort()
    for node in fgraph:
        if (
            hasattr(node.op, "scalar_op")
            and node.op.scalar_op == aes.basic.scalar_maximum
        ):
            return

        # In mode FAST_COMPILE, the rewrites don't replace the
        # `MaxAndArgmax` `Op`.
        if isinstance(node.op, MaxAndArgmax):
            return

    # TODO FIXME: Refactor this test so that it makes a direct assertion and
    # nothing more.
    raise AssertionError("No maximum detected after log_sum_exp rewrite")


def test_local_log_sum_exp_maximum():
    """Test that the rewrite is applied by checking the presence of the maximum."""
    x = tensor3("x")
    check_max_log_sum_exp(x, axis=(0,), dimshuffle_op=None)
    check_max_log_sum_exp(x, axis=(1,), dimshuffle_op=None)
    check_max_log_sum_exp(x, axis=(2,), dimshuffle_op=None)
    check_max_log_sum_exp(x, axis=(0, 1), dimshuffle_op=None)
    check_max_log_sum_exp(x, axis=(0, 1, 2), dimshuffle_op=None)

    # If a transpose is applied to the sum
    transpose_op = DimShuffle((False, False), (1, 0))
    check_max_log_sum_exp(x, axis=2, dimshuffle_op=transpose_op)

    # If the sum is performed with keepdims=True
    x = TensorType(dtype="floatX", shape=(None, 1, None))("x")
    sum_keepdims_op = x.sum(axis=(0, 1), keepdims=True).owner.op
    check_max_log_sum_exp(x, axis=(0, 1), dimshuffle_op=sum_keepdims_op)


def test_local_log_sum_exp_near_one():
    """Test that the rewritten result is correct around 1.0."""

    x = tensor3("x")
    x_val = 1.0 + np.random.random((4, 3, 2)).astype(config.floatX) / 10.0

    f = compile_graph_log_sum_exp(x, axis=(1,))
    naive_ret = np.log(np.sum(np.exp(x_val), axis=1))
    rewritten_ret = f(x_val)
    assert np.allclose(naive_ret, rewritten_ret)

    # If a transpose is applied
    transpose_op = DimShuffle((False, False), (1, 0))
    f = compile_graph_log_sum_exp(x, axis=(1,), dimshuffle_op=transpose_op)
    naive_ret = np.log(np.sum(np.exp(x_val), axis=1).T)
    rewritten_ret = f(x_val)
    assert np.allclose(naive_ret, rewritten_ret)


def test_local_log_sum_exp_large():
    """Test that the rewrite result is correct for extreme value 100."""
    x = vector("x")
    f = compile_graph_log_sum_exp(x, axis=0)

    x_val = np.array([-100.0, 100.0]).astype(config.floatX)

    rewritten_ret = f(x_val)
    assert np.allclose(rewritten_ret, 100.0)


def test_local_log_sum_exp_inf():
    """Test that when max = +-inf, the rewritten output still works correctly."""
    x = vector("x")
    f = compile_graph_log_sum_exp(x, axis=0)

    assert f([-np.inf, -np.inf]) == -np.inf
    assert f([np.inf, np.inf]) == np.inf
    assert f([-np.inf, np.inf]) == np.inf


def test_local_reciprocal_1_plus_exp():
    x = vector("x")
    y = at.reciprocal(1 + exp(x))
    z = rewrite_graph(y, include=["canonicalization", "stabilize", "specialize"])
    assert z.owner.op == sigmoid


class TestSigmoidRewrites:
    def get_mode(self, excluding=None):
        """
        Return appropriate mode for the tests.

        Parameters
        ----------
        excluding
            List of rewrites to exclude.

        Returns
        -------
        The current default mode unless the `config.mode` option is
        set to 'FAST_COMPILE' (in which case it is replaced by the 'FAST_RUN'
        mode), without the rewrites specified in `excluding`.
        """
        if excluding is None:
            excluding = []
        m = config.mode
        if m == "FAST_COMPILE":
            mode = aesara.compile.mode.get_mode("FAST_RUN")
        else:
            mode = aesara.compile.mode.get_default_mode()
        if excluding:
            return mode.excluding(*excluding)
        else:
            return mode

    def test_exp_over_1_plus_exp(self):
        m = self.get_mode(excluding=["local_elemwise_fusion"])

        x = vector()
        data = np.random.random(54).astype(config.floatX)

        # tests exp_over_1_plus_exp
        f = aesara.function([x], exp(x) / (1 + exp(x)), mode=m)
        assert [node.op for node in f.maker.fgraph.toposort()] == [sigmoid]
        f(data)
        f = aesara.function([x], exp(x) / (2 + exp(x)), mode=m)
        assert [node.op for node in f.maker.fgraph.toposort()] != [sigmoid]
        f(data)
        f = aesara.function([x], exp(x) / (1 - exp(x)), mode=m)
        assert [node.op for node in f.maker.fgraph.toposort()] != [sigmoid]
        f(data)
        f = aesara.function([x], exp(x + 1) / (1 + exp(x)), mode=m)
        assert [node.op for node in f.maker.fgraph.toposort()] != [sigmoid]
        f(data)

        # tests inv_1_plus_exp
        f = aesara.function([x], at.fill(x, 1.0) / (1 + exp(-x)), mode=m)
        # todo: solve issue #4589 first
        # assert check_stack_trace(f, ops_to_check=sigmoid)
        assert [node.op for node in f.maker.fgraph.toposort()] == [sigmoid]
        f(data)
        f = aesara.function([x], at.fill(x, 1.0) / (2 + exp(-x)), mode=m)
        assert [node.op for node in f.maker.fgraph.toposort()] != [sigmoid]
        f(data)
        f = aesara.function([x], at.fill(x, 1.0) / (1 - exp(-x)), mode=m)
        assert [node.op for node in f.maker.fgraph.toposort()] != [sigmoid]
        f(data)
        f = aesara.function([x], at.fill(x, 1.1) / (1 + exp(-x)), mode=m)
        assert [node.op for node in f.maker.fgraph.toposort()] != [sigmoid]
        f(data)

        # tests inv_1_plus_exp with neg
        f = aesara.function([x], at.fill(x, -1.0) / (1 + exp(-x)), mode=m)
        # todo: solve issue #4589 first
        # assert check_stack_trace(
        #     f, ops_to_check=[sigmoid, neg_inplace])
        assert [node.op for node in f.maker.fgraph.toposort()] == [
            sigmoid,
            inplace.neg_inplace,
        ]
        f(data)
        f = aesara.function([x], at.fill(x, -1.0) / (1 - exp(-x)), mode=m)
        assert [node.op for node in f.maker.fgraph.toposort()] != [
            sigmoid,
            inplace.neg_inplace,
        ]
        f(data)
        f = aesara.function([x], at.fill(x, -1.0) / (2 + exp(-x)), mode=m)
        assert [node.op for node in f.maker.fgraph.toposort()] != [
            sigmoid,
            inplace.neg_inplace,
        ]
        f(data)
        f = aesara.function([x], at.fill(x, -1.1) / (1 + exp(-x)), mode=m)
        assert [node.op for node in f.maker.fgraph.toposort()] != [
            sigmoid,
            inplace.neg_inplace,
        ]
        f(data)

        # tests double inv_1_plus_exp with neg
        # (-1)(exp(x)) / (1+exp(x))(1+exp(-x))
        # = (-1)/(1+exp(-x)) * exp(x)/(1+exp(x))
        # = - (sigm(x) * sigm(x))
        f = aesara.function(
            [x],
            (at.fill(x, -1.0) * exp(x)) / ((1 + exp(x)) * (1 + exp(-x))),
            mode=m,
        )
        # todo: solve issue #4589 first
        # assert check_stack_trace(f, ops_to_check=[sigmoid, mul])
        assert [node.op for node in f.maker.fgraph.toposort()] == [sigmoid, mul]
        f(data)
        f = aesara.function(
            [x],
            (at.fill(x, -1.1) * exp(x)) / ((1 + exp(x)) * (1 + exp(-x))),
            mode=m,
        )
        assert [node.op for node in f.maker.fgraph.toposort()] != [
            sigmoid,
            mul,
            inplace.neg_inplace,
        ]
        f(data)
        f = aesara.function(
            [x],
            (at.fill(x, -1.0) * exp(x)) / ((2 + exp(x)) * (1 + exp(-x))),
            mode=m,
        )
        assert [node.op for node in f.maker.fgraph.toposort()] != [
            sigmoid,
            mul,
            inplace.neg_inplace,
        ]
        f(data)
        f = aesara.function(
            [x],
            (at.fill(x, -1.0) * exp(x)) / ((1 + exp(x)) * (2 + exp(-x))),
            mode=m,
        )
        assert [node.op for node in f.maker.fgraph.toposort()] != [
            sigmoid,
            mul,
            inplace.neg_inplace,
        ]
        f(data)
        f = aesara.function(
            [x],
            (at.fill(x, -1.0) * exp(x)) / ((1 + exp(x)) * (1 + exp(x))),
            mode=m,
        )
        assert [node.op for node in f.maker.fgraph.toposort()] != [
            sigmoid,
            mul,
            inplace.neg_inplace,
        ]
        f(data)
        f = aesara.function(
            [x],
            (at.fill(x, -1.0) * exp(x)) / ((1 + exp(x)) * (2 + exp(-x))),
            mode=m,
        )
        assert [node.op for node in f.maker.fgraph.toposort()] != [
            sigmoid,
            mul,
            inplace.neg_inplace,
        ]
        f(data)

    def test_local_1msigmoid(self):
        m = self.get_mode(excluding=["fusion", "inplace"])
        x = fmatrix()

        # Test `exp_over_1_plus_exp`
        f = aesara.function([x], 1 - exp(x) / (1 + exp(x)), mode=m)
        # FIXME: PatternNodeRewriter does not copy stack trace
        #  (see https://github.com/Theano/Theano/issues/4581)
        # assert check_stack_trace(f, ops_to_check=[neg, sigmoid])
        assert [node.op for node in f.maker.fgraph.toposort()] == [neg, sigmoid]

        # Test `inv_1_plus_exp`
        f = aesara.function([x], 1 - at.fill(x, 1.0) / (1 + exp(-x)), mode=m)
        # assert check_stack_trace(f, ops_to_check=[neg, sigmoid])
        assert [node.op for node in f.maker.fgraph.toposort()] == [neg, sigmoid]

        # Test float constant
        f = aesara.function(
            [x], np.array(1.000001, dtype="float32") - sigmoid(x), mode=m
        )
        assert [node.op for node in f.maker.fgraph.toposort()] == [neg, sigmoid]

    def test_local_sigm_times_exp(self):
        """
        exp(x) * sigm(-x) -> sigm(x)
        exp(-x) * sigm(x) -> sigm(-x)
        """

        def match(func, ops):
            # print [node.op.scalar_op for node in func.maker.fgraph.toposort()]
            assert [node.op for node in func.maker.fgraph.toposort()] == ops

        m = self.get_mode(excluding=["local_elemwise_fusion", "inplace"])
        x, y = vectors("x", "y")

        f = aesara.function([x], sigmoid(-x) * exp(x), mode=m)
        match(f, [sigmoid])
        assert check_stack_trace(f, ops_to_check=sigmoid)

        f = aesara.function([x], sigmoid(x) * exp(-x), mode=m)
        match(f, [neg, sigmoid])
        assert check_stack_trace(f, ops_to_check=sigmoid)

        f = aesara.function([x], -(-(-(sigmoid(x)))) * exp(-x), mode=m)
        match(f, [neg, sigmoid, neg])
        # assert check_stack_trace(f, ops_to_check=sigmoid)

        f = aesara.function(
            [x, y],
            (sigmoid(x) * sigmoid(-y) * -exp(-x) * exp(x * y) * exp(y)),
            mode=m,
        )
        topo = f.maker.fgraph.toposort()
        for op, nb in [(sigmoid, 2), (mul, 2), (neg, 1), (exp, 1)]:
            assert sum(n.op == op for n in topo) == nb
        # assert check_stack_trace(f, ops_to_check=[sigmoid, mul,
        #                                           exp])

    def test_perform_sigm_times_exp(self):
        """Test the core function doing the `sigm_times_exp` rewrite.

        It is easier to test different graph scenarios this way than by
        compiling an Aesara function.
        """

        x, y, z, t = vectors("x", "y", "z", "t")
        exp_op = exp

        def check(expr1, expr2):
            trees = [parse_mul_tree(e) for e in (expr1, expr2)]
            perform_sigm_times_exp(trees[0])
            trees[0] = simplify_mul(trees[0])
            good = is_same_graph(compute_mul(trees[0]), compute_mul(trees[1]))
            if not good:
                print(trees[0])
                print(trees[1])
                print("***")
                aesara.printing.debugprint(compute_mul(trees[0]))
                print("***")
                aesara.printing.debugprint(compute_mul(trees[1]))
            assert good

        check(sigmoid(x) * exp_op(-x), sigmoid(-x))
        check(
            -x * sigmoid(x) * (y * (-1 * z) * exp_op(-x)),
            -x * sigmoid(-x) * (y * (-1 * z)),
        )
        check(
            -sigmoid(-x)
            * (
                exp_op(y)
                * (-exp_op(-z) * 3 * -exp_op(x))
                * (y * 2 * (-sigmoid(-y) * (z + t) * exp_op(z)) * sigmoid(z))
            )
            * -sigmoid(x),
            sigmoid(x)
            * (-sigmoid(y) * (-sigmoid(-z) * 3) * (y * 2 * ((z + t) * exp_op(z))))
            * (-sigmoid(x)),
        )
        check(
            exp_op(-x) * -exp_op(-x) * (-sigmoid(x) * -sigmoid(x)),
            -sigmoid(-x) * sigmoid(-x),
        )
        check(-exp_op(x) * -sigmoid(-x) * -exp_op(-x), -sigmoid(-x))

    def test_grad_log1msigm(self):
        # At some point, this returned nan, because (1 - sigm(x)) was
        # on both the numerator and the denominator of a fraction,
        # but the two nodes in question had not been merged.
        x = matrix("x")
        lr = scalar("lr")

        s = sigmoid(x)
        l = log(1 - s)
        c = l.mean()
        ux = x - lr * aesara.grad(c, x)

        # Before the rewriting, inf and NaN will be produced in the graph,
        # and DebugMode will complain. Everything is fine afterwards.
        mode = self.get_mode()
        if not isinstance(mode, aesara.compile.debugmode.DebugMode):
            f = aesara.function([x, lr], ux, mode=mode)
            ux_v = f([[50]], 0.1)
            assert not np.isnan(ux_v)


class TestSoftplusRewrites:
    def setup_method(self):
        if aesara.config.mode == "FAST_COMPILE":
            m = aesara.compile.mode.get_mode("FAST_RUN").excluding(
                "local_elemwise_fusion"
            )
        else:
            m = aesara.compile.mode.get_default_mode().excluding(
                "local_elemwise_fusion"
            )
        self.m = m

    def test_logsigm_to_softplus(self):
        x = vector()

        out = log(sigmoid(x))
        f = aesara.function([x], out, mode=self.m)

        # Fix ticket #4581 first
        # assert check_stack_trace(
        #     f, ops_to_check=(aesara.scalar.Neg,
        #                      ScalarSoftplus))
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 3
        assert isinstance(topo[0].op.scalar_op, aesara.scalar.Neg)
        assert isinstance(topo[1].op.scalar_op, aesara.scalar.Softplus)
        assert isinstance(topo[2].op.scalar_op, aesara.scalar.Neg)
        f(np.random.random(54).astype(config.floatX))

    def test_log1msigm_to_softplus(self):
        x = matrix()

        out = log(1 - sigmoid(x))
        f = aesara.function([x], out, mode=self.m)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 2
        assert isinstance(topo[0].op.scalar_op, aesara.scalar.Softplus)
        assert isinstance(topo[1].op.scalar_op, aesara.scalar.Neg)
        # assert check_stack_trace(f, ops_to_check='all')
        f(np.random.random((54, 11)).astype(config.floatX))

        # Test close to 1
        out = log(1.000001 - sigmoid(x))
        f = aesara.function([x], out, mode=self.m)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 2
        assert isinstance(topo[0].op.scalar_op, aesara.scalar.Softplus)
        assert isinstance(topo[1].op.scalar_op, aesara.scalar.Neg)

        # Same test with a flatten
        out = log(1 - at.flatten(sigmoid(x)))
        f = aesara.function([x], out, mode=self.m)

        # assert check_stack_trace(f, ops_to_check='all')
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 3
        assert at.is_flat(topo[0].outputs[0])
        assert isinstance(topo[1].op.scalar_op, aesara.scalar.Softplus)
        assert isinstance(topo[2].op.scalar_op, aesara.scalar.Neg)
        f(np.random.random((54, 11)).astype(config.floatX))

        # Same test with a reshape
        out = log(1 - sigmoid(x).reshape([x.size]))
        f = aesara.function([x], out, mode=self.m)
        topo = f.maker.fgraph.toposort()
        # assert len(topo) == 3
        assert any(isinstance(node.op, Reshape) for node in topo)
        assert any(
            isinstance(
                getattr(node.op, "scalar_op", None),
                aesara.scalar.Softplus,
            )
            for node in topo
        )
        f(np.random.random((54, 11)).astype(config.floatX))

    def test_log1pexp_to_softplus(self):
        m = aesara.config.mode
        if m == "FAST_COMPILE":
            m = "FAST_RUN"

        x = vector()

        out = log(1 + exp(x))
        f = aesara.function([x], out, mode=self.m)

        # Fix ticket #4581 first
        # assert check_stack_trace(f, ops_to_check='all')
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert isinstance(topo[0].op.scalar_op, aesara.scalar.Softplus)
        f(np.random.random(54).astype(config.floatX))

    def test_log1p_neg_sigmoid_to_softpuls(self):
        x = scalar()
        out = log1p(-sigmoid(x))
        f = aesara.function([x], out, mode=self.m)

        topo = f.maker.fgraph.toposort()
        assert len(topo) == 2
        assert isinstance(topo[0].op.scalar_op, aesara.scalar.Softplus)
        assert isinstance(topo[1].op.scalar_op, aesara.scalar.Neg)

        # This value would underflow to -inf without rewrite
        assert np.isclose(f(37.0), -37.0)


class TestSigmoidUtils:
    """Test utility functions used in the rewrites for `sigmoid`/`softplus` expressions."""

    def test_compute_mul(self):
        x, y, z = vectors("x", "y", "z")
        tree = (x * y) * -z
        mul_tree = parse_mul_tree(tree)
        assert parse_mul_tree(compute_mul(mul_tree)) == mul_tree
        assert is_same_graph(compute_mul(parse_mul_tree(tree)), tree)

    def test_parse_mul_tree(self):
        x, y, z = vectors("x", "y", "z")
        assert parse_mul_tree(x * y) == [False, [[False, x], [False, y]]]
        assert parse_mul_tree(-(x * y)) == [True, [[False, x], [False, y]]]
        assert parse_mul_tree(-x * y) == [False, [[True, x], [False, y]]]
        assert parse_mul_tree(-x) == [True, x]
        assert parse_mul_tree((x * y) * -z) == [
            False,
            [[False, [[False, x], [False, y]]], [True, z]],
        ]

    def test_is_1pexp(self):
        x = vector("x")
        exp_op = exp
        assert is_1pexp(1 + exp_op(x), False) == (False, x)
        assert is_1pexp(exp_op(x) + 1, False) == (False, x)
        for neg_, exp_arg in map(
            lambda x: is_1pexp(x, only_process_constants=False),
            [(1 + exp_op(-x)), (exp_op(-x) + 1)],
        ):
            assert not neg_ and is_same_graph(exp_arg, -x)
        assert is_1pexp(1 - exp_op(x), False) is None
        assert is_1pexp(2 + exp_op(x), False) is None
        assert is_1pexp(exp_op(x) + 2, False) is None
        assert is_1pexp(exp_op(x) - 1, False) is None
        assert is_1pexp(-1 + exp_op(x), False) is None
        assert is_1pexp(1 + 2 * exp_op(x), False) is None


def test_local_logit_sigmoid():
    """Test that graphs of the form ``logit(sigmoid(x))`` and ``sigmoid(logit(x))`` get rewritten to ``x``."""

    def logit_fn(x):
        return log(x / (1 - x))

    x = fmatrix()

    out = sigmoid(logit_fn(x))
    fg = rewrite(FunctionGraph([x], [out]))
    assert not list(fg.toposort())
    assert fg.inputs[0] is fg.outputs[0]

    out = logit_fn(sigmoid(x))
    fg = rewrite(FunctionGraph([x], [out]))
    assert not list(fg.toposort())
    assert fg.inputs[0] is fg.outputs[0]


def test_local_useless_conj():
    default_mode = get_default_mode()

    # Test for all zeros
    x = scalar()
    s = _conj(x)
    mode_with_rewrite = default_mode.including("canonicalization", "local_useless_conj")
    f = function([x], s, mode=mode_with_rewrite)
    assert not any(node.op == _conj for node in f.maker.fgraph.apply_nodes)

    x = zscalar()
    s = _conj(x)
    mode_with_rewrite = default_mode.including("canonicalization", "local_useless_conj")
    f = function([x], s, mode=mode_with_rewrite)
    assert any(node.op == _conj for node in f.maker.fgraph.apply_nodes)


def test_deprecations():
    """Make sure we can import from deprecated modules."""
    with pytest.deprecated_call():
        from aesara.tensor.math_opt import AlgebraicCanonizer  # noqa: F401 F811


def test_local_sub_neg_to_add():
    x = scalar("x")
    y = vector("y")

    f = function([x, y], x - (-y), mode=Mode("py"))

    nodes = [
        node.op
        for node in f.maker.fgraph.toposort()
        if not isinstance(node.op, DimShuffle)
    ]
    assert nodes == [at.add]

    x_test = np.full((), 1.0, dtype=config.floatX)
    y_test = np.full(5, 2.0, dtype=config.floatX)
    assert np.allclose(f(x_test, y_test), x_test - (-y_test))


def test_local_sub_neg_to_add_const():
    # This rewrite is achieved by the local_add_canonizer
    x = vector("x")
    const = 5.0

    f = function([x], x - (-const), mode=Mode("py"))

    nodes = [
        node.op
        for node in f.maker.fgraph.toposort()
        if not isinstance(node.op, DimShuffle)
    ]
    assert nodes == [at.add]

    x_test = np.array([3, 4], dtype=config.floatX)
    assert np.allclose(f(x_test), x_test - (-const))


@pytest.mark.parametrize("first_negative", (True, False))
def test_local_add_neg_to_sub(first_negative):
    x = scalar("x")
    y = vector("y")
    out = -x + y if first_negative else x + (-y)

    f = function([x, y], out, mode=Mode("py"))

    nodes = [
        node.op
        for node in f.maker.fgraph.toposort()
        if not isinstance(node.op, DimShuffle)
    ]
    assert nodes == [at.sub]

    x_test = np.full((), 1.0, dtype=config.floatX)
    y_test = np.full(5, 2.0, dtype=config.floatX)
    exp = -x_test + y_test if first_negative else x_test + (-y_test)
    assert np.allclose(f(x_test, y_test), exp)


def test_local_add_neg_to_sub_const():
    x = vector("x")
    const = 5.0

    f = function([x], x + (-const), mode=Mode("py"))

    nodes = [
        node.op
        for node in f.maker.fgraph.toposort()
        if not isinstance(node.op, DimShuffle)
    ]
    assert nodes == [at.sub]

    x_test = np.array([3, 4], dtype=config.floatX)
    assert np.allclose(f(x_test), x_test + (-const))
