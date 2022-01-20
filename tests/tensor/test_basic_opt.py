import copy

import numpy as np
import pytest

import aesara
import aesara.scalar as aes
import aesara.tensor as at
from aesara import shared
from aesara.compile import optdb
from aesara.compile.function import function
from aesara.compile.mode import OPT_NONE, Mode, get_default_mode, get_mode
from aesara.compile.ops import DeepCopyOp, deep_copy_op
from aesara.configdefaults import config
from aesara.graph.basic import Apply, Constant, Variable
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import Op
from aesara.graph.opt import check_stack_trace, local_optimizer, out2in
from aesara.graph.opt_utils import optimize_graph
from aesara.graph.optdb import OptimizationQuery
from aesara.graph.type import Type
from aesara.misc.safe_asarray import _asarray
from aesara.printing import pprint
from aesara.raise_op import Assert, CheckAndRaise
from aesara.tensor.basic import (
    Alloc,
    Join,
    MakeVector,
    Rebroadcast,
    ScalarFromTensor,
    Split,
    TensorFromScalar,
    alloc,
    as_tensor_variable,
    join,
    second,
    tile,
)
from aesara.tensor.basic_opt import (
    ShapeFeature,
    apply_rebroadcast_opt,
    assert_op,
    local_alloc_sink_dimshuffle,
    local_dimshuffle_lift,
    local_merge_alloc,
    local_reshape_to_dimshuffle,
    local_useless_alloc,
    local_useless_dimshuffle_in_reshape,
    local_useless_elemwise,
    local_useless_reshape,
    register_specialize,
)
from aesara.tensor.elemwise import DimShuffle, Elemwise
from aesara.tensor.extra_ops import BroadcastTo, Repeat, Unique, repeat, unique
from aesara.tensor.math import (
    add,
    bitwise_and,
    bitwise_or,
    bitwise_xor,
    cos,
    cosh,
    dot,
    eq,
    exp,
    floor_div,
    ge,
    gt,
    int_div,
    invert,
    iround,
    le,
    log,
    log2,
    log10,
    lt,
    maximum,
    minimum,
    mul,
    neg,
    neq,
)
from aesara.tensor.math import pow as at_pow
from aesara.tensor.math import reciprocal
from aesara.tensor.math import round as at_round
from aesara.tensor.math import sin, sinh, softplus, sqr, sqrt, sub
from aesara.tensor.math import sum as at_sum
from aesara.tensor.math import tan, tanh, true_div, xor
from aesara.tensor.math_opt import local_lift_transpose_through_dot
from aesara.tensor.shape import (
    Reshape,
    Shape_i,
    SpecifyShape,
    reshape,
    shape,
    specify_shape,
)
from aesara.tensor.subtensor import (
    AdvancedIncSubtensor1,
    Subtensor,
    advanced_inc_subtensor,
    advanced_inc_subtensor1,
    inc_subtensor,
    set_subtensor,
)
from aesara.tensor.type import (
    TensorType,
    dmatrices,
    dmatrix,
    dscalar,
    dvector,
    fmatrix,
    fscalar,
    fvector,
    imatrices,
    iscalar,
    ivector,
    lscalar,
    lvector,
    matrices,
    matrix,
    scalar,
    scalars,
    tensor,
    tensor3,
    tensor4,
    values_eq_approx_remove_nan,
    vector,
    vectors,
)
from tests import unittest_tools as utt


mode_opt = config.mode
if mode_opt == "FAST_COMPILE":
    mode_opt = "FAST_RUN"
mode_opt = get_mode(mode_opt)

dimshuffle_lift = out2in(local_dimshuffle_lift)

_optimizer_stabilize = OptimizationQuery(include=["fast_run"])
_optimizer_stabilize.position_cutoff = 1.51
_optimizer_stabilize = optdb.query(_optimizer_stabilize)

_optimizer_specialize = OptimizationQuery(include=["fast_run"])
_optimizer_specialize.position_cutoff = 2.01
_optimizer_specialize = optdb.query(_optimizer_specialize)

_optimizer_fast_run = OptimizationQuery(include=["fast_run"])
_optimizer_fast_run = optdb.query(_optimizer_fast_run)


def ds(x, y):
    return DimShuffle(x.type.broadcastable, y)(x)


def optimize(g, level="fast_run"):
    if level == "fast_run":
        _optimizer_fast_run.optimize(g)
    elif level == "specialize":
        _optimizer_specialize.optimize(g)
    elif level == "stabilize":
        _optimizer_stabilize.optimize(g)
    else:
        raise ValueError(level)
    return g


def inputs(xbc=(0, 0), ybc=(0, 0), zbc=(0, 0)):
    x = TensorType(shape=xbc, dtype="float64")("x")
    y = TensorType(shape=ybc, dtype="float64")("y")
    z = TensorType(shape=zbc, dtype="float64")("z")
    return x, y, z


class TestDimshuffleLift:
    def test_double_transpose(self):
        x, y, z = inputs()
        e = ds(ds(x, (1, 0)), (1, 0))
        g = FunctionGraph([x], [e])
        assert (
            str(g) == "FunctionGraph(InplaceDimShuffle{1,0}(InplaceDimShuffle{1,0}(x)))"
        )
        dimshuffle_lift.optimize(g)
        assert str(g) == "FunctionGraph(x)"
        # no need to check_stack_trace as graph is supposed to be empty

    def test_merge2(self):
        x, y, z = inputs()
        e = ds(ds(x, (1, "x", 0)), (2, 0, "x", 1))
        g = FunctionGraph([x], [e])
        assert (
            str(g)
            == "FunctionGraph(InplaceDimShuffle{2,0,x,1}(InplaceDimShuffle{1,x,0}(x)))"
        ), str(g)
        dimshuffle_lift.optimize(g)
        assert str(g) == "FunctionGraph(InplaceDimShuffle{0,1,x,x}(x))", str(g)
        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(g, ops_to_check="all")

    def test_elim3(self):
        x, y, z = inputs()
        e = ds(ds(ds(x, (0, "x", 1)), (2, 0, "x", 1)), (1, 0))
        g = FunctionGraph([x], [e])
        assert str(g) == (
            "FunctionGraph(InplaceDimShuffle{1,0}(InplaceDimShuffle{2,0,x,1}"
            "(InplaceDimShuffle{0,x,1}(x))))"
        ), str(g)
        dimshuffle_lift.optimize(g)
        assert str(g) == "FunctionGraph(x)", str(g)
        # no need to check_stack_trace as graph is supposed to be empty

    def test_lift(self):
        x, y, z = inputs([False] * 1, [False] * 2, [False] * 3)
        e = x + y + z
        g = FunctionGraph([x, y, z], [e])

        # It does not really matter if the DimShuffles are inplace
        # or not.
        init_str_g_inplace = (
            "FunctionGraph(Elemwise{add,no_inplace}(InplaceDimShuffle{x,0,1}"
            "(Elemwise{add,no_inplace}(InplaceDimShuffle{x,0}(x), y)), z))"
        )
        init_str_g_noinplace = (
            "FunctionGraph(Elemwise{add,no_inplace}(DimShuffle{x,0,1}"
            "(Elemwise{add,no_inplace}(DimShuffle{x,0}(x), y)), z))"
        )
        assert str(g) in (init_str_g_inplace, init_str_g_noinplace), str(g)

        opt_str_g_inplace = (
            "FunctionGraph(Elemwise{add,no_inplace}(Elemwise{add,no_inplace}"
            "(InplaceDimShuffle{x,x,0}(x), InplaceDimShuffle{x,0,1}(y)), z))"
        )
        opt_str_g_noinplace = (
            "FunctionGraph(Elemwise{add,no_inplace}(Elemwise{add,no_inplace}"
            "(DimShuffle{x,x,0}(x), DimShuffle{x,0,1}(y)), z))"
        )
        dimshuffle_lift.optimize(g)
        assert str(g) in (opt_str_g_inplace, opt_str_g_noinplace), str(g)
        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(g, ops_to_check="all")

    def test_recursive_lift(self):
        v = vector(dtype="float64")
        m = matrix(dtype="float64")
        out = ((v + 42) * (m + 84)).T
        g = FunctionGraph([v, m], [out])
        init_str_g = (
            "FunctionGraph(InplaceDimShuffle{1,0}(Elemwise{mul,no_inplace}"
            "(InplaceDimShuffle{x,0}(Elemwise{add,no_inplace}"
            "(<TensorType(float64, (None,))>, "
            "InplaceDimShuffle{x}(TensorConstant{42}))), "
            "Elemwise{add,no_inplace}"
            "(<TensorType(float64, (None, None))>, "
            "InplaceDimShuffle{x,x}(TensorConstant{84})))))"
        )
        assert str(g) == init_str_g
        new_out = local_dimshuffle_lift.transform(g, g.outputs[0].owner)[0]
        new_g = FunctionGraph(g.inputs, [new_out])
        opt_str_g = (
            "FunctionGraph(Elemwise{mul,no_inplace}(Elemwise{add,no_inplace}"
            "(InplaceDimShuffle{0,x}(<TensorType(float64, (None,))>), "
            "InplaceDimShuffle{x,x}(TensorConstant{42})), "
            "Elemwise{add,no_inplace}(InplaceDimShuffle{1,0}"
            "(<TensorType(float64, (None, None))>), "
            "InplaceDimShuffle{x,x}(TensorConstant{84}))))"
        )
        assert str(new_g) == opt_str_g
        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(new_g, ops_to_check="all")

    def test_useless_dimshuffle(self):
        x, _, _ = inputs()
        e = ds(x, (0, 1))
        g = FunctionGraph([x], [e])
        assert str(g) == "FunctionGraph(InplaceDimShuffle{0,1}(x))"
        dimshuffle_lift.optimize(g)
        assert str(g) == "FunctionGraph(x)"
        # Check stacktrace was copied over correctly after opt was applied
        assert hasattr(g.outputs[0].tag, "trace")

    def test_dimshuffle_on_broadcastable(self):
        x, y, z = inputs([False, True], [True, False, True], [False, False, True])
        u = at.constant(1)
        ds_x = ds(x, (0, "x"))  # useless
        ds_y = ds(y, (2, 1, 0))  # useless
        ds_z = ds(z, (2, 1, 0))  # useful
        ds_u = ds(u, ("x"))  # useful
        g = FunctionGraph([x, y, z, u], [ds_x, ds_y, ds_z, ds_u])
        assert (
            str(g)
            == "FunctionGraph(InplaceDimShuffle{0,x}(x), InplaceDimShuffle{2,1,0}(y), InplaceDimShuffle{2,1,0}(z), InplaceDimShuffle{x}(TensorConstant{1}))"
        )
        dimshuffle_lift.optimize(g)
        assert (
            str(g)
            == "FunctionGraph(x, y, InplaceDimShuffle{2,1,0}(z), InplaceDimShuffle{x}(TensorConstant{1}))"
        )
        # Check stacktrace was copied over correctly after opt was applied
        assert hasattr(g.outputs[0].tag, "trace")


def test_local_useless_dimshuffle_in_reshape():
    vec = TensorType(shape=(False,), dtype="float64")("vector")
    mat = TensorType(shape=(False, False), dtype="float64")("mat")
    row = TensorType(shape=(True, False), dtype="float64")("row")
    col = TensorType(shape=(False, True), dtype="float64")("col")

    reshape_dimshuffle_vector = reshape(vec.dimshuffle("x", 0), vec.shape)
    reshape_dimshuffle_mat = reshape(mat.dimshuffle("x", 0, "x", 1), mat.shape)
    reshape_dimshuffle_row = reshape(row.dimshuffle(1, "x"), row.shape)
    reshape_dimshuffle_col = reshape(col.dimshuffle(0), col.shape)

    g = FunctionGraph(
        [vec, mat, row, col],
        [
            reshape_dimshuffle_vector,
            reshape_dimshuffle_mat,
            reshape_dimshuffle_row,
            reshape_dimshuffle_col,
        ],
    )

    assert str(g) == (
        "FunctionGraph(Reshape{1}(InplaceDimShuffle{x,0}(vector), Shape(vector)), "
        "Reshape{2}(InplaceDimShuffle{x,0,x,1}(mat), Shape(mat)), "
        "Reshape{2}(InplaceDimShuffle{1,x}(row), Shape(row)), "
        "Reshape{2}(InplaceDimShuffle{0}(col), Shape(col)))"
    )
    useless_dimshuffle_in_reshape = out2in(local_useless_dimshuffle_in_reshape)
    useless_dimshuffle_in_reshape.optimize(g)
    assert str(g) == (
        "FunctionGraph(Reshape{1}(vector, Shape(vector)), "
        "Reshape{2}(mat, Shape(mat)), "
        "Reshape{2}(row, Shape(row)), "
        "Reshape{2}(col, Shape(col)))"
    )

    # Check stacktrace was copied over correctly after opt was applied
    assert check_stack_trace(g, ops_to_check="all")

    # Check that the optimization does not get applied when the order
    # of dimensions has changed.
    reshape_dimshuffle_mat2 = reshape(mat.dimshuffle("x", 1, "x", 0), mat.shape)
    h = FunctionGraph([mat], [reshape_dimshuffle_mat2])
    str_h = str(h)
    useless_dimshuffle_in_reshape.optimize(h)
    assert str(h) == str_h


class TestFusion:
    opts = OptimizationQuery(
        include=[
            "local_elemwise_fusion",
            "composite_elemwise_fusion",
            "canonicalize",
            "inplace",
        ],
        exclude=["cxx_only", "BlasOpt"],
    )
    mode = Mode(get_default_mode().linker, opts)
    _shared = staticmethod(shared)
    topo_exclude = ()

    def my_init(dtype="float64", num=0):
        return np.zeros((5, 5), dtype=dtype) + num

    fw, fx, fy, fz = [
        tensor(dtype="float32", shape=[False] * 2, name=n) for n in "wxyz"
    ]
    dw, dx, dy, dz = [
        tensor(dtype="float64", shape=[False] * 2, name=n) for n in "wxyz"
    ]
    ix, iy, iz = [tensor(dtype="int32", shape=[False] * 2, name=n) for n in "xyz"]
    fv = fvector("v")
    fs = fscalar("s")
    fwv = my_init("float32", 1)
    fxv = my_init("float32", 2)
    fyv = my_init("float32", 3)
    fzv = my_init("float32", 4)
    fvv = _asarray(np.random.random(5), dtype="float32")
    fsv = np.asarray(np.random.random(), dtype="float32")
    dwv = my_init("float64", 5)
    ixv = _asarray(my_init(num=60), dtype="int32")
    iyv = _asarray(my_init(num=70), dtype="int32")
    izv = _asarray(my_init(num=70), dtype="int32")
    fwx = fw + fx
    ftanx = tan(fx)

    @pytest.mark.parametrize(
        "case",
        [
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
                fx + fy ** fz,
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv + fyv ** fzv,
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
                fx - true_div(fy, 2),
                (fx, fy),
                (fxv, fyv),
                1,
                fxv - (fyv / 2),
                "float32",
            ),
            (
                fx - true_div(fy, fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - (fyv / fzv),
                "float32",
            ),
            (
                fx - int_div(ix * 100, iy * 1000),
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
                fx - fy + sqr(fz),
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
                fv + fy ** fz,
                (fv, fy, fz),
                (fvv, fyv, fzv),
                2,
                fvv + fyv ** fzv,
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
        ],
    )
    def test_elemwise_fusion(self, case, nb_repeat=1, assert_len_topo=True):
        """Verify that `Elemwise` fusion works."""

        g, sym_inputs, val_inputs, nb_elemwise, answer, out_dtype = case

        if isinstance(out_dtype, dict):
            out_dtype = out_dtype[config.cast_policy]

        if self._shared is None:
            f = function(list(sym_inputs), g, mode=self.mode)
            for x in range(nb_repeat):
                out = f(*val_inputs)
        else:
            out = self._shared(np.zeros((5, 5), dtype=out_dtype), "out")
            assert out.dtype == g.dtype
            f = function(sym_inputs, [], updates=[(out, g)], mode=self.mode)
            for x in range(nb_repeat):
                f(*val_inputs)
            out = out.get_value()

        atol = 1e-8
        if out_dtype == "float32":
            atol = 1e-6

        assert np.allclose(out, answer * nb_repeat, atol=atol)

        topo = f.maker.fgraph.toposort()
        topo_ = [n for n in topo if not isinstance(n.op, self.topo_exclude)]
        if assert_len_topo:

            assert len(topo_) == nb_elemwise

            if nb_elemwise == 1:
                # if no variable appears multiple times in the
                # input of g,
                # check that the number of input to the Composite
                # Elemwise is ok
                if len(set(g.owner.inputs)) == len(g.owner.inputs):
                    expected_len_sym_inputs = np.sum(
                        [not isinstance(x, Constant) for x in topo_[0].inputs]
                    )
                    assert expected_len_sym_inputs == len(sym_inputs)

        assert out_dtype == out.dtype

    def test_fusion_35_inputs(self):
        r"""Make sure we don't fuse too many `Op`\s and go past the 31 function arguments limit."""
        inpts = vectors(["i%i" % i for i in range(35)])

        # Make an elemwise graph looking like:
        # sin(i34 + sin(i33 + sin(... i1 + sin(i0) ...)))
        out = sin(inpts[0])
        for idx in range(1, 35):
            out = sin(inpts[idx] + out)

        with config.change_flags(cxx=""):
            f = function(inpts, out, mode=self.mode)

        # Make sure they all weren't fused
        composite_nodes = [
            node
            for node in f.maker.fgraph.toposort()
            if isinstance(getattr(node.op, "scalar_op", None), aes.basic.Composite)
        ]
        assert not any(len(node.inputs) > 31 for node in composite_nodes)

    @pytest.mark.skipif(not config.cxx, reason="No cxx compiler")
    def test_big_fusion(self):
        # In the past, pickle of Composite generated in that case
        # crashed with max recursion limit. So we were not able to
        # generate C code in that case.
        factors = []
        sd = dscalar()
        means = dvector()

        cst_05 = at.constant(0.5)
        cst_m05 = at.constant(-0.5)
        cst_2 = at.constant(2)
        cst_m2 = at.constant(-2)
        ones = at.constant(np.ones(10))
        n = 85
        if config.mode in ["DebugMode", "DEBUG_MODE"]:
            n = 10

        for i in range(n):
            f = cst_m05 * sd ** cst_m2 * (ones - means[i]) ** cst_2 + cst_05 * log(
                cst_05 * (sd ** cst_m2) / np.pi
            )
            factors.append(at_sum(f))

        logp = add(*factors)

        vars = [sd, means]

        # Make sure that C compilation is used
        mode = Mode("cvm", self.opts)
        dlogp = function(vars, [aesara.grad(logp, v) for v in vars], mode=mode)

        # Make sure something was fused
        assert any(
            isinstance(getattr(node.op, "scalar_op", None), aes.basic.Composite)
            for node in dlogp.maker.fgraph.toposort()
        )

    def test_add_mul_fusion_inplace(self):

        opts = OptimizationQuery(
            include=[
                "local_elemwise_fusion",
                "composite_elemwise_fusion",
                "canonicalize",
                "inplace",
            ],
            exclude=["cxx_only", "BlasOpt"],
        )

        mode = Mode(self.mode.linker, opts)

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

    @pytest.mark.skipif(not config.cxx, reason="No cxx compiler")
    def test_no_c_code(self):
        r"""Make sure we avoid fusions for `Op`\s without C code implementations."""

        # This custom `Op` has no `c_code` method
        class NoCCodeOp(aes.basic.UnaryScalarOp):
            def impl(self, x):
                return x * 2

        no_c_code_op = Elemwise(NoCCodeOp(aes.basic.upgrade_to_float))

        mode = Mode(linker="cvm")
        mode._optimizer = mode._optimizer.including(
            "local_elemwise_fusion",
            "composite_elemwise_fusion",
            "canonicalize",
            "inplace",
        )

        x = vector()
        out = x * no_c_code_op(x + 1)
        f = function([x], out, mode=mode)

        assert not any(
            isinstance(getattr(n.op, "scalar_op"), aes.basic.Composite)
            for n in f.maker.fgraph.toposort()
        )


class TimesN(aes.basic.UnaryScalarOp):
    """
    Used in test TestCompositeCodegen

    Must be outside of the class, otherwise, the c cache code can't
    pickle this class and this cause stuff printing during test.
    """

    def __eq__(self, other):
        return super().__eq__(other) and self.n == other.n

    def __hash__(self):
        return super().__hash__() ^ hash(self.n)

    def __init__(self, n, *args, **kwargs):
        self.n = n
        aes.basic.UnaryScalarOp.__init__(self, *args, **kwargs)

    def impl(self, x):
        return x * self.n

    def c_support_code_apply(self, node, nodename):
        n = str(self.n)
        return (
            """
        float %(nodename)s_timesn(float x) { return x * %(n)s; }
        """
            % locals()
        )

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        return f"{z} = {name}_timesn({x});"


class TestCompositeCodegen:
    """
    Test The Composite Ops code generation in a case where there is multiple
    scalar ops with support code.
    """

    def setup_method(self):
        upgrade_to_float = aes.basic.upgrade_to_float

        self.scal_times_2 = TimesN(2, upgrade_to_float, name="times_2")
        self.times_2 = Elemwise(self.scal_times_2, name="times_2")

        self.scal_times_3 = TimesN(3, upgrade_to_float, name="times_3")
        self.times_3 = Elemwise(self.scal_times_3, name="times_3")

        self.x = fvector()

    def test_nested_composite(self):
        y = self.times_2(self.x)
        z = self.times_3(y)
        f = function([self.x], z)
        if config.mode != "FAST_COMPILE":
            assert len(f.maker.fgraph.toposort()) == 1
        fval = f([1, 2, 3])
        assert np.all(fval == [6, 12, 18])

    def test_local_useless_composite(self):
        x = aes.float32()
        c = aes.Composite([x], [x + 1, x - 1])
        X = matrix()
        o = Elemwise(scalar_op=c)(X)
        mode = get_default_mode().including("local_useless_composite")

        f = function([X], o[0], mode=mode)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert len(topo[0].outputs) == 1
        utt.assert_allclose(f([[1.0]]), [[2.0]])

        f = function([X], o[1], mode=mode)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert len(topo[0].outputs) == 1
        utt.assert_allclose(f([[1.0]]), [[0.0]])


def test_local_useless_slice():
    # test a simple matrix
    x = matrix("x")
    mode_unopt = get_default_mode().excluding(
        "local_useless_slice", "local_mul_canonizer"
    )
    mode_opt = (
        get_default_mode()
        .including("local_useless_slice")
        .excluding("local_mul_canonizer")
    )

    # test with and without the useless slice
    o = 2 * x[0, :]
    f_unopt = function([x], o, mode=mode_unopt)
    f_opt = function([x], o, mode=mode_opt)
    rng = np.random.default_rng(utt.fetch_seed())
    test_inp = rng.integers(-10, 10, (4, 4)).astype("float32")
    assert all(
        f_opt(test_inp) == f_unopt(test_inp)
    ), "The optimization caused a mismatch in the result"
    # test to see if the slice is truly gone
    apply_node = f_opt.maker.fgraph.toposort()[0]
    subtens = apply_node.op
    assert not any(
        isinstance(idx, slice) for idx in subtens.idx_list
    ), "Slice should be gone"

    # Now test that the stack trace is copied over properly,
    # before before and after optimization.
    assert check_stack_trace(f_unopt, ops_to_check="all")
    assert check_stack_trace(f_opt, ops_to_check="all")

    # test a 4d tensor
    z = tensor4("z")
    o2 = z[1, :, :, 1]
    o3 = z[0, :, :, :]
    f_opt_check = function([z], o2, mode=mode_opt)
    f_opt_check_apply = function([z], o3, mode=mode_opt)

    # The optimization shouldn't apply here
    apply_node = f_opt_check.maker.fgraph.toposort()[0]
    subtens = apply_node.op
    assert [isinstance(idx, slice) for idx in subtens.idx_list].count(True) == 2
    # But it should here
    apply_node = f_opt_check_apply.maker.fgraph.toposort()[0]
    subtens = apply_node.op
    assert not any(isinstance(idx, slice) for idx in subtens.idx_list)

    # Finally, test that the stack trace is copied over properly,
    # before before and after optimization.
    assert check_stack_trace(f_opt_check, ops_to_check=Subtensor)
    assert check_stack_trace(f_opt_check_apply, ops_to_check=Subtensor)


def test_local_useless_fill():
    x = dvector()
    y = dvector()
    z = lvector()

    x_ = np.random.random((5,))
    y_ = np.random.random((5,))
    z_ = (np.random.random((5,)) * 5).astype("int64")

    # basic case
    f = function([x], at.fill(x, x) * 2, mode=mode_opt)
    assert [node.op for node in f.maker.fgraph.toposort()] == [mul]
    res = f(x_)
    exp_res = np.broadcast_to(x_, x_.shape) * 2
    assert np.array_equal(res, exp_res)

    # basic case
    f = function([x, y], at.second(y, x) * 2, mode=mode_opt)
    assert [node.op for node in f.maker.fgraph.toposort()] == [mul]
    res = f(x_, y_)
    exp_res = np.broadcast_to(x_, y_.shape) * 2
    assert np.array_equal(res, exp_res)

    # basic case
    f = function([x, y], at.fill(x, y) * 2, mode=mode_opt)
    assert [node.op for node in f.maker.fgraph.toposort()] == [mul]
    res = f(x_, y_)
    exp_res = np.broadcast_to(y_, x_.shape) * 2
    assert np.array_equal(res, exp_res)

    # now with different type(cast)
    f = function([x, z], at.fill(z, x) * 2, mode=mode_opt)
    assert [node.op for node in f.maker.fgraph.toposort()] == [mul]
    res = f(x_, z_)
    exp_res = np.broadcast_to(x_, z_.shape) * 2
    assert np.array_equal(res, exp_res)

    # now with different type(cast)
    f = function([x, z], at.fill(x, z) * 2, mode=mode_opt)
    assert [node.op for node in f.maker.fgraph.toposort()] == [mul]
    res = f(x_, z_)
    exp_res = np.broadcast_to(z_, x_.shape) * 2
    assert np.array_equal(res, exp_res)

    # now cutting out the input ??
    f = function([x, y], at.fill(x, y) * 2, mode=mode_opt)
    assert [node.op for node in f.maker.fgraph.toposort()] == [mul]
    res = f(x_, y_)
    exp_res = np.broadcast_to(y_, x_.shape) * 2
    assert np.array_equal(res, exp_res)


def test_local_fill_to_alloc():
    x = dvector()
    m = dmatrix()

    x_ = np.random.random((5,))
    m_ = np.random.random((5, 5))

    y = at.fill(m, x)

    mode = mode_opt.including("stabilize", "local_fill_to_alloc").excluding(
        "useless", "local_useless_fill"
    )

    f = function([m, x], y, mode=mode)
    assert Alloc in [node.op.__class__ for node in f.maker.fgraph.toposort()]

    res = f(m_, x_)
    exp_res = np.broadcast_to(x_, m_.shape)
    assert np.array_equal(res, exp_res)

    y = at.fill(x, m)

    f = function([m, x], y, mode=mode)

    assert Alloc not in [node.op.__class__ for node in f.maker.fgraph.toposort()]

    res = f(m_, x_)
    assert np.array_equal(res, m_)


class TestLocalCanonicalizeAlloc:
    def setup_method(self):
        self.rng = np.random.default_rng(utt.fetch_seed())

    def test_inconsistent_constant(self):
        x = at.as_tensor(self.rng.standard_normal((3, 7)))
        a = at.alloc(x, 6, 7)

        assert a.owner and isinstance(a.owner.op, Alloc)

        # `local_useless_alloc` should replace the `Alloc` with an `Assert`
        with pytest.raises(AssertionError):
            f = function([], a, mode=mode_opt)

        x = at.as_tensor(self.rng.standard_normal((6, 7)))
        a = at.alloc(x, 6, 7)

        f = function([], a, mode=mode_opt)

        # The optimization should then be applied, and remove Alloc
        assert not any(
            isinstance(node.op, (Alloc, Assert)) for node in f.maker.fgraph.toposort()
        )

    def test_inconsistent_shared(self):
        # These shapes don't match!
        x = shared(self.rng.standard_normal((3, 7)))
        a = at.alloc(x, 6, 7)

        assert a.owner and isinstance(a.owner.op, Alloc)

        f = function([], a, mode=mode_opt)

        # The optimization should then be applied, and remove Alloc
        assert not any(isinstance(node.op, Alloc) for node in f.maker.fgraph.toposort())
        assert any(isinstance(node.op, Assert) for node in f.maker.fgraph.toposort())

        with pytest.raises(AssertionError):
            f()

        good_x_val = self.rng.standard_normal((6, 7))
        x.set_value(good_x_val)

        assert np.array_equal(f(), good_x_val)

    def test_basic_fill(self):
        x = matrix("x")
        y = at.fill(x, x)

        # The optimization 'locall_fill_to_alloc' should call at.alloc,
        # which should return x and not alloc(x, ...)
        f = function([x], [y], mode=mode_opt.including("local_fill_to_alloc"))
        assert not any(
            [isinstance(node.op, Alloc) for node in f.maker.fgraph.toposort()]
        )

    def test_basic_tile(self):
        x = matrix("x")
        y = at.tile(x, (1,) * 2)

        mode = mode_opt.including(
            "local_dimshuffle_lift",
            "local_useless_dimshuffle_in_reshape",
            "local_alloc_sink_dimshuffle",
        )
        f = function([x], [y], mode=mode)

        assert not any(
            [isinstance(node.op, Alloc) for node in f.maker.fgraph.toposort()]
        )

    @pytest.mark.parametrize(
        "x, has_alloc",
        [
            (at.alloc(np.ones((2,)), 1, 3, 2), True),
            (at.alloc(np.array(1.0), 1, 1), False),
            (at.alloc(np.ones((1, 1)), 1, 1, 2), True),
            (at.alloc(np.ones((1, 1)), 1, 2), True),
        ],
    )
    def test_useless_alloc_with_shape_one(self, x, has_alloc):
        g = FunctionGraph(outputs=[x])
        assert any(isinstance(node.op, Alloc) for node in g.toposort())

        alloc_lift = out2in(local_alloc_sink_dimshuffle)
        alloc_lift.optimize(g)

        if has_alloc:
            assert any(isinstance(node.op, Alloc) for node in g.toposort())
        else:
            assert not any(isinstance(node.op, Alloc) for node in g.toposort())


class TestLocalUselessIncSubtensorAlloc:
    opt_name = "local_useless_inc_subtensor_alloc"

    def setup_method(self):
        # The optimization requires the shape feature so we need to compile in
        # FAST_RUN mode.
        mode = config.mode
        if mode == "FAST_COMPILE":
            mode = "FAST_RUN"
        self.mode = get_mode(mode)
        self.rng = np.random.default_rng(utt.fetch_seed())

    def test_advanced_inc_subtensor(self):
        x = vector("x")
        y = scalar("y")
        i = matrix("i", dtype="int64")
        z = advanced_inc_subtensor(x, at.alloc(y, *i.shape), i)
        mode1 = self.mode.excluding(self.opt_name)
        mode2 = self.mode.including(self.opt_name)
        f1 = function([x, i, y], z, mode=mode1)
        f2 = function([x, i, y], z, mode=mode2)

        # the alloc op should still be there
        assert (
            len([n for n in f1.maker.fgraph.toposort() if isinstance(n.op, Alloc)]) == 1
        )
        # the alloc op should have been removed
        assert (
            len([n for n in f2.maker.fgraph.toposort() if isinstance(n.op, Alloc)]) == 0
        )

        x_value = np.random.standard_normal((5)).astype(config.floatX)
        y_value = np.random.standard_normal()
        i_value = self.rng.integers(0, 3, size=(2, 3))

        r1 = f1(x_value, i_value, y_value)
        r2 = f2(x_value, i_value, y_value)

        utt.assert_allclose(r1, r2)

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f1, ops_to_check=AdvancedIncSubtensor1)
        assert check_stack_trace(f2, ops_to_check=AdvancedIncSubtensor1)

    def test_advanced_inc_subtensor1(self):
        x = vector("x")
        y = scalar("y")
        i = vector("i", dtype="int64")
        z = advanced_inc_subtensor1(x, at.alloc(y, *i.shape), i)
        mode1 = self.mode.excluding(self.opt_name)
        mode2 = self.mode.including(self.opt_name)
        f1 = function([x, i, y], z, mode=mode1)
        f2 = function([x, i, y], z, mode=mode2)

        # the alloc op should still be there
        assert (
            len([n for n in f1.maker.fgraph.toposort() if isinstance(n.op, Alloc)]) == 1
        )
        # the alloc op should have been removed
        assert (
            len([n for n in f2.maker.fgraph.toposort() if isinstance(n.op, Alloc)]) == 0
        )

        x_value = np.random.standard_normal((5)).astype(config.floatX)
        y_value = np.random.standard_normal()
        i_value = self.rng.integers(0, 3, size=2)

        r1 = f1(x_value, i_value, y_value)
        r2 = f2(x_value, i_value, y_value)

        utt.assert_allclose(r1, r2)

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f1, ops_to_check=AdvancedIncSubtensor1)
        assert check_stack_trace(f2, ops_to_check="all")

    def test_incsubtensor(self):
        x = vector("x")
        y = scalar("y")
        i = scalar("i", dtype="int64")
        z = inc_subtensor(x[:i], at.alloc(y, i))
        mode1 = self.mode.excluding(self.opt_name)
        mode2 = self.mode.including(self.opt_name)
        f1 = function([x, i, y], z, mode=mode1)
        f2 = function([x, i, y], z, mode=mode2)

        # the alloc op should still be there
        assert (
            len([n for n in f1.maker.fgraph.toposort() if isinstance(n.op, Alloc)]) == 1
        )
        # the alloc op should have been removed
        assert (
            len([n for n in f2.maker.fgraph.toposort() if isinstance(n.op, Alloc)]) == 0
        )

        x_value = np.random.standard_normal((5)).astype(config.floatX)
        y_value = np.random.standard_normal()
        i_value = 3

        r1 = f1(x_value, i_value, y_value)
        r2 = f2(x_value, i_value, y_value)

        utt.assert_allclose(r1, r2)

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f1, ops_to_check="last")
        assert check_stack_trace(f2, ops_to_check="last")


class TestShapeOptimizer:
    def test_basic(self):
        mode = config.mode
        if mode == "FAST_COMPILE":
            mode = "FAST_RUN"
        v = vector()
        m = matrix()
        f = function([v, m], (v + m).shape, mode=mode)
        for node in f.maker.fgraph.toposort():
            assert node.op != add

    def test_constant(self):
        mode = config.mode
        if mode == "FAST_COMPILE":
            mode = "FAST_RUN"

        v = vector()
        f = function([v], v.dimshuffle("x", "x", 0).shape[1], mode=mode)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert topo[0].op == deep_copy_op

    @staticmethod
    def max_pool_c01b(c01b, pool_shp, pool_stride, img_shp):
        """
        Like max_pool but with input using axes ('c', 0, 1, 'b')
          (Alex Krizhevsky format)

        pool_shp, pool_stride and img_shp are int that represent
        the same shp in x and y.
        """
        mx = None

        # Compute index in pooled space of last needed pool
        # (needed = each input pixel must appear in at least one pool)
        def last_pool(im_shp, p_shp, p_strd):
            rval = int(np.ceil(float(im_shp - p_shp) / p_strd))
            assert p_strd * rval + p_shp >= im_shp
            assert p_strd * (rval - 1) + p_shp < im_shp
            return rval

        # Compute starting row of the last pool
        last_pool_r = last_pool(img_shp, pool_shp, pool_stride) * pool_stride
        # Compute number of rows needed in img for all indexes to work out
        required_r = last_pool_r + pool_shp

        last_pool_c = last_pool(img_shp, pool_shp, pool_stride) * pool_stride
        required_c = last_pool_c + pool_shp

        wide_infinity = at.alloc(
            -np.inf, c01b.shape[0], required_r, required_c, c01b.shape[3]
        )

        c01b = set_subtensor(wide_infinity[:, 0:img_shp, 0:img_shp, :], c01b)

        for row_within_pool in range(pool_shp):
            row_stop = last_pool_r + row_within_pool + 1
            for col_within_pool in range(pool_shp):
                col_stop = last_pool_c + col_within_pool + 1
                cur = c01b[
                    :,
                    row_within_pool:row_stop:pool_stride,
                    col_within_pool:col_stop:pool_stride,
                    :,
                ]
                if mx is None:
                    mx = cur
                else:
                    mx = maximum(mx, cur)
        return mx

    def test_broadcasted_dims(self):
        # This test a case that caused a crash during optimization
        shp = (1, 1, 1, 1)
        rng = np.random.default_rng(utt.fetch_seed())
        a = shared(rng.random(shp).astype(config.floatX))
        out = self.max_pool_c01b(a, 1, 1, 1)

        # max_pool_c01b use -inf and this will trigger DebugMode error.
        mode = copy.copy(get_default_mode())
        mode.check_isfinite = False
        f = function([], out, mode=mode)
        f()

    def test_constant_merge(self):
        # This test the error in gh-1122 that is a caused by the
        # combination of merge optimizer and ShapeFeature.

        x = at.constant([0, 0])
        y = x[1:]
        x1 = x - at.join(0, y, y)
        x1.eval()

    def test_local_track_shape_i(self):
        class IdentityNoShape(Op):
            """Op that does not infer the output shape from the input one"""

            def make_node(self, x):
                x = as_tensor_variable(x)
                return Apply(self, [x], [x.type()])

            def perform(self, node, inp, out_):
                (x,) = inp
                (out,) = out_
                out[0] = x.copy()

            # def infer_shape(self, fgraph, node, (xshp,)):
            # return [tuple([self.shape_i(i)(r) for i in range(r.ndim)])]

        identity_noshape = IdentityNoShape()

        class IdentityShape(Op):
            """Op that does infer the output shape from the input one"""

            def make_node(self, x):
                x = as_tensor_variable(x)
                return Apply(self, [x], [x.type()])

            def perform(self, node, inp, out_):
                (x,) = inp
                (out,) = out_
                out[0] = x.copy()

            def infer_shape(self, fgraph, node, xshp_):
                # Could also just return.
                (xshp,) = xshp_
                return (xshp,)

        identity_shape = IdentityShape()

        @local_optimizer([IdentityNoShape])
        def local_identity_noshape_to_identity_shape(fgraph, node):
            """Optimization transforming the first Op into the second"""
            if isinstance(node.op, IdentityNoShape):
                return [identity_shape(node.inputs[0])]

        mode = get_default_mode().including("ShapeOpt", "specialize")
        rng = np.random.default_rng(utt.fetch_seed())
        x = tensor3("x")
        ins_x = identity_noshape(x)

        # Without the optimization
        f = function([x], ins_x.shape, mode=mode)
        xval = rng.standard_normal((3, 4, 7)).astype(config.floatX)
        assert np.all(f(xval) == [3, 4, 7])
        f_ops = [node.op for node in f.maker.fgraph.toposort()]
        assert len(f_ops) == 5
        assert identity_noshape in f_ops
        assert identity_shape not in f_ops

        # Register the optimization
        register_specialize(local_identity_noshape_to_identity_shape)

        mode = get_default_mode().including("ShapeOpt", "specialize")
        # With the optimization
        # The identity_shape op should not be needed anymore to compute
        # the shape
        g = function([x], ins_x.shape, mode=mode)
        xval = rng.standard_normal((6, 1, 2)).astype(config.floatX)
        assert np.all(g(xval) == [6, 1, 2])
        g_ops = [node.op for node in g.maker.fgraph.toposort()]
        assert len(g_ops) == 4
        assert identity_noshape not in g_ops
        assert identity_shape not in g_ops

        # test multiple level of op without infer_shape
        ins_x3 = identity_noshape(identity_noshape(identity_noshape(x)))
        h = function([x], ins_x3.shape, mode=mode)
        xval = rng.standard_normal((6, 1, 2)).astype(config.floatX)
        assert np.all(h(xval) == [6, 1, 2])
        h_ops = [node.op for node in h.maker.fgraph.toposort()]
        assert len(h_ops) == 4
        assert identity_noshape not in h_ops
        assert identity_shape not in h_ops

    def test_no_shapeopt(self):
        # Test that a basic example works even when ShapeOpt is excluded
        X = matrix()
        expr = X.shape[0]

        mode = get_default_mode().excluding("ShapeOpt")
        f = function([X], expr, mode=mode)
        # FIXME: This is not a good test.
        f([[1, 2], [2, 3]])


class TestUselessCheckAndRaise:
    def test_basic(self):
        mode = get_default_mode().including(
            "canonicalize", "local_remove_useless_assert"
        )
        x = scalar()
        y = scalar()
        f = function([x, y], assert_op(x, eq(x, y)), mode=mode)
        assert f(1, 1) == 1
        with pytest.raises(AssertionError):
            f(1, 0)

    def test_local_remove_useless_1(self):
        """Remove `CheckAndRaise`s when all the conditions are always true."""
        x = scalar()
        fg = FunctionGraph(outputs=[assert_op(x, 1)], clone=False)
        fg_res = optimize_graph(fg, include=["canonicalize", "specialize"])
        topo = fg_res.toposort()
        assert not any(isinstance(node.op, CheckAndRaise) for node in topo)

    def test_local_remove_useless_2(self):
        """Remove `CheckAndRaise` conditions that are always true."""
        x = scalar()
        y = scalar()
        fg = FunctionGraph(outputs=[assert_op(x, y, 1)], clone=False)
        fg_res = optimize_graph(fg, include=["canonicalize", "specialize"])
        topo = fg_res.toposort()
        (assert_node,) = [node for node in topo if isinstance(node.op, CheckAndRaise)]
        assert assert_node.inputs == [x, y]

    def test_local_remove_useless_3(self):
        """Don't remove `CheckAndRaise` conditions that are always false."""
        x = scalar()
        y = scalar()
        fg = FunctionGraph(outputs=[assert_op(x, y, 0)], clone=False)
        fg_res = optimize_graph(fg, include=["canonicalize", "specialize"])
        topo = fg_res.toposort()
        (assert_node,) = [node for node in topo if isinstance(node.op, CheckAndRaise)]
        assert assert_node.inputs[:2] == [x, y]
        assert assert_node.inputs[-1].data == 0


def test_local_remove_all_assert():
    r"""Remove all `Assert`\s."""
    mode = get_default_mode().including("canonicalize", "local_remove_all_assert")

    x = scalar()
    y = scalar()
    f = function([x, y], assert_op(x, y), mode=mode)
    # Without the optimization, this would fail
    assert f(1, 0) == 1
    topo = f.maker.fgraph.toposort()
    assert not any(isinstance(node.op, CheckAndRaise) for node in topo)

    mode = get_default_mode()
    a = assert_op(x, eq(x, 0).any())
    f = function([x], a, mode=mode.excluding("unsafe"))
    topo = f.maker.fgraph.toposort()
    a_op = [n for n in topo if isinstance(n.op, Assert)]
    assert len(a_op) == 1


class TestTile:
    def test_local_useless_tile(self):
        v = vector()
        m = matrix()
        mode = None
        if config.mode == "FAST_COMPILE":
            mode = "FAST_RUN"
        for var, data in [(v, [1, 2, 3]), (m, [[1, 2], [3, 4]])]:
            # When len(repeat pattern) <= var.ndim, everything is removed
            # for ndim in range(1, var.ndim):
            for ndim in range(var.ndim + 1):
                f = function([var], tile(var, (1,) * ndim), mode=mode)
                topo = f.maker.fgraph.toposort()
                assert len(topo) == 1
                assert isinstance(topo[0].op, DeepCopyOp)
                f(data)
                # In this case the opt only removes nodes,
                # no need to check_stack_trace
            # When len(repeat pattern) > var.ndim, only a dimshuffle should be
            # left, but there can be a DeepCopy as well
            for ndim in range(var.ndim + 1, var.ndim + 3):
                f = function([var], tile(var, (1,) * ndim), mode=mode)
                topo = f.maker.fgraph.toposort()
                assert len(topo) <= 2
                assert isinstance(topo[0].op, DimShuffle)
                assert check_stack_trace(f, ops_to_check=[DimShuffle])
                f(data)


class TestRebroadcast:
    def test_local_useless_rebroadcast(self):
        mode = get_default_mode().including("canonicalize")
        v1 = vector()
        v2 = vector()
        j = at.join(0, v1, v2)
        f = function([v1, v2], j, mode=mode)
        f([1, 2], [3, 4, 5])
        e = f.maker.fgraph.toposort()
        assert len([n for n in e if isinstance(n.op, Rebroadcast)]) == 0

        assert check_stack_trace(f, ops_to_check="all")

    def test_rebroadcast_rebroadcast(self):
        mode = get_default_mode().including("canonicalize")
        m = matrix()
        s = at.addbroadcast(m, 0, 1)
        v = at.unbroadcast(s, 1)
        f = function([m], v, mode=mode)
        f([[76]])
        e = f.maker.fgraph.toposort()
        rebroadcast_nodes = [n for n in e if isinstance(n.op, Rebroadcast)]
        assert len(rebroadcast_nodes) == 1
        assert rebroadcast_nodes[0].op.axis == {0: True}


class TestUselessElemwise:
    def setup_method(self):
        self.mode = get_default_mode().including("canonicalize", "local_fill_to_alloc")

    def test_eq(self):
        x = dmatrix()
        y = dmatrix()
        f = function([x, y], eq(x, y), mode=self.mode)
        vx = np.random.random((5, 4))
        vy = np.random.random((5, 4))
        f(vx, vy)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert isinstance(topo[0].op, Elemwise)
        assert isinstance(topo[0].op.scalar_op, aes.EQ)
        f2 = function([x], eq(x, x), mode=self.mode)
        assert np.all(f2(vx) == np.ones((5, 4)))
        topo2 = f2.maker.fgraph.toposort()
        # Shape_i{1}(<TensorType(float64, (None, None))>),
        # Shape_i{0}(<TensorType(float64, (None, None))>), Alloc([[1]], Shape_i{0}.0,
        # Shape_i{1}.0
        assert len(topo2) == 3
        assert isinstance(topo2[-1].op, Alloc)

    def test_neq(self):
        x = dmatrix()
        y = dmatrix()
        f = function([x, y], neq(x, y), mode=self.mode)
        vx = np.random.random((5, 4))
        vy = np.random.random((5, 4))
        f(vx, vy)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert isinstance(topo[0].op, Elemwise)
        assert isinstance(topo[0].op.scalar_op, aes.NEQ)
        f2 = function([x], neq(x, x), mode=self.mode)
        assert np.all(f2(vx) == np.zeros((5, 4)))
        topo2 = f2.maker.fgraph.toposort()
        assert len(topo2) == 3
        assert isinstance(topo2[-1].op, Alloc)

    def test_mul(self):
        x = dmatrix()
        y = dmatrix()
        f = function([x], mul(x), mode=self.mode)
        vx = np.random.random((5, 4))
        vy = np.random.random((5, 4))
        f(vx)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert topo[0].op == deep_copy_op
        f2 = function([x, y], mul(x, y), mode=self.mode)
        assert np.all(f2(vx, vy) == vx * vy)
        topo2 = f2.maker.fgraph.toposort()
        assert len(topo2) == 1
        assert isinstance(topo2[0].op, Elemwise)
        assert isinstance(topo2[0].op.scalar_op, aes.Mul)

    def test_add(self):
        x = dmatrix()
        y = dmatrix()
        f = function([x], add(x), mode=self.mode)
        vx = np.random.random((5, 4))
        vy = np.random.random((5, 4))
        f(vx)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert topo[0].op == deep_copy_op
        f2 = function([x, y], add(x, y), mode=self.mode)
        assert np.all(f2(vx, vy) == vx + vy)
        topo2 = f2.maker.fgraph.toposort()
        assert len(topo2) == 1
        assert isinstance(topo2[0].op, Elemwise)
        assert isinstance(topo2[0].op.scalar_op, aes.Add)

    def test_identity(self):
        # aes.identity is used in 2 Elemwise functions:
        # tensor_copy, and view
        x = matrix()
        f = function([x], at.tensor_copy(x), mode=self.mode)
        vx = np.random.random((5, 4)).astype(config.floatX)
        f(vx)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert topo[0].op == deep_copy_op


class TestCastCast:
    def setup_method(self):
        mode = get_default_mode()
        self.mode = mode.including("local_cast_cast")

    def test_consecutive(self):
        x = fmatrix()
        o = Elemwise(aes.Cast(aes.Scalar("float64")))(x.astype("float64"))
        f = function([x], o, mode=self.mode)
        dx = np.random.random((5, 4)).astype("float32")
        f(dx)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert isinstance(topo[0].op.scalar_op, aes.basic.Cast)

        x = dmatrix()
        o = Elemwise(aes.Cast(aes.Scalar("float32")))(x.astype("float32"))
        f = function([x], o, mode=self.mode)
        dx = np.random.random((5, 4))
        f(dx)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert isinstance(topo[0].op.scalar_op, aes.basic.Cast)

    def test_upcast(self):
        # Upcast followed by any other cast
        x = fmatrix()
        o = Elemwise(aes.Cast(aes.Scalar("complex128")))(x.astype("complex64"))
        f = function([x], o, mode=self.mode)
        dx = np.random.random((5, 4)).astype("float32")
        f(dx)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert isinstance(topo[0].op.scalar_op, aes.basic.Cast)

        # Upcast followed by a downcast back to the base type
        x = fmatrix()
        o = Elemwise(aes.Cast(aes.Scalar("float32")))(x.astype("float64"))
        f = function([x], o, mode=self.mode)
        dx = np.random.random((5, 4)).astype("float32")
        f(dx)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert isinstance(topo[0].op, DeepCopyOp)

        # Downcast followed by an upcast back to the base type
        # Optimization shouldn't be applied
        x = dmatrix()
        o = Elemwise(aes.Cast(aes.Scalar("float64")))(x.astype("float32"))
        f = function([x], o, mode=self.mode)
        dx = np.random.random((5, 4))
        f(dx)
        topo = f.maker.fgraph.toposort()
        assert (
            len(topo) == 1 and isinstance(topo[0].op.scalar_op, aes.basic.Composite)
        ) or (len(topo) > 1)


def test_constant_folding():
    # Test that constant folding get registered at fast_compile
    # An error removed that registration during the registration.
    x = dvector()
    mode = get_mode("FAST_COMPILE").excluding("fusion")
    f = function([x], [x * 2, x + x], mode=mode)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 2

    # Test that we do not crash when constant folding elemwise scalar
    # as they should not generate c code.

    x = at.constant(3)
    assert x.ndim == 0
    mode = get_mode("FAST_COMPILE").excluding("fusion")
    f = function([], [x * 2, x + x], mode=mode)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 2
    assert all(isinstance(n.op, DeepCopyOp) for n in topo)


@pytest.mark.xfail(
    reason="Aesara optimizes constant before stabilization. "
    "This breaks stabilization optimizations in some "
    "cases. See #504.",
    raises=AssertionError,
)
def test_constant_get_stabilized():
    # Currently Aesara enables the `constant_folding` optimization before stabilization optimization.
    # This caused some stabilization optimizations to not be activated and that
    # caused inf values to appear when they should not.

    # We can't simply move the `constant_folding` optimization to
    # specialize since this will break other optimizations.  We will need to
    # partially duplicate some canonicalize optimizations to fix this issue.

    x2 = scalar()
    y2 = log(1 + exp(x2))
    mode = get_default_mode()
    mode.check_isfinite = False
    f2 = function([x2], y2, mode=mode)

    assert len(f2.maker.fgraph.toposort()) == 1
    assert f2.maker.fgraph.toposort()[0].op == softplus
    assert f2(800) == 800

    x = at.as_tensor_variable(800)
    y = log(1 + exp(x))
    f = function([], y, mode=mode)
    # When this error is fixed, the following line should be ok.
    assert f() == 800, f()


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
        """
        Wrapper around function for this test.

        It disables checking for NaN removed by optimizations in DebugMode
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

        # This case caused a missed optimization in the past.
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
                y = true_div(
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


class TestLocalUselessSwitch:
    def setup_method(self):
        self.mode = mode_opt.excluding("constant_folding")

    @pytest.mark.parametrize(
        "dtype1",
        ["int32", "int64"],
    )
    @pytest.mark.parametrize(
        "dtype2",
        ["int32", "int64"],
    )
    @pytest.mark.parametrize(
        "cond",
        [0, 1, np.array([True])],
    )
    def test_const(self, dtype1, dtype2, cond):
        x = matrix("x", dtype=dtype1)
        y = matrix("y", dtype=dtype2)
        z = at.switch(cond, x, y)
        f = function([x, y], z, mode=self.mode)
        assert not any(
            [
                node.op
                for node in f.maker.fgraph.toposort()
                if (
                    isinstance(node.op, Elemwise)
                    and isinstance(node.op.scalar_op, aes.basic.Switch)
                )
            ]
        )
        vx = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype1)
        vy = np.array([[7, 8, 9], [10, 11, 12]], dtype=dtype2)
        np_res = np.where(cond, vx, vy)
        assert np.array_equal(f(vx, vy), np_res)

    @pytest.mark.parametrize(
        "dtype1",
        ["int32", "int64"],
    )
    def test_left_is_right(self, dtype1):
        x = matrix("x", dtype=dtype1)
        varc = matrix("varc", dtype=dtype1)
        z1 = at.switch(1, x, x)
        z0 = at.switch(0, x, x)
        z2 = at.switch(varc, x, x)
        f1 = function([x], z1, mode=self.mode)
        f0 = function([x], z0, mode=self.mode)
        f2 = function([x, varc], z2, mode=self.mode)

        topo = f1.maker.fgraph.toposort()
        assert len(topo) == 1
        assert topo[0].op == deep_copy_op

        topo = f0.maker.fgraph.toposort()
        assert len(topo) == 1
        assert topo[0].op == deep_copy_op

        topo = f2.maker.fgraph.toposort()
        assert len(topo) == 1
        assert topo[0].op == deep_copy_op

        vx = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype1)
        vc = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype1)
        assert np.array_equal(f1(vx), vx)
        assert np.array_equal(f0(vx), vx)
        assert np.array_equal(f2(vx, vc), vx)

    @pytest.mark.parametrize(
        "dtype1",
        ["float32", "float64"],
    )
    def test_shape_le_0(self, dtype1):
        x = matrix("x", dtype=dtype1)
        z0 = at.switch(le(x.shape[0], 0), 0, x.shape[0])
        f0 = function([x], z0, mode=self.mode)
        assert isinstance(f0.maker.fgraph.toposort()[0].op, Shape_i)

        z1 = at.switch(le(x.shape[1], 0), 0, x.shape[1])
        f1 = function([x], z1, mode=self.mode)
        assert isinstance(f1.maker.fgraph.toposort()[0].op, Shape_i)

        vx = np.random.standard_normal((0, 5)).astype(dtype1)
        assert f0(vx) == 0
        assert f1(vx) == 5

    def test_broadcasting_1(self):
        # test switch(cst, matrix, row)
        x = matrix("x", dtype="int32")
        y = vector("y", dtype="int64")

        z = at.switch(1, x, y)
        f = function([x, y], z, mode=self.mode)

        start_var = f.maker.fgraph.outputs[0].owner.inputs[0]
        assert isinstance(start_var.owner.op, Elemwise)
        assert isinstance(start_var.owner.op.scalar_op, aes.basic.Cast)
        assert not any(node.op == at.switch for node in f.maker.fgraph.toposort())

        vx = np.array([[1, 2, 3], [4, 5, 6]], dtype="int32")
        vy = np.array([10, 11, 12], dtype="int64")
        np_res = np.where(1, vx, vy)
        assert np.array_equal(f(vx, vy), np_res)

        z = at.switch(0, x, y)
        f = function([x, y], z, mode=self.mode)

        assert isinstance(f.maker.fgraph.outputs[0].owner.op, Alloc)
        assert f.maker.fgraph.inputs[1] == f.maker.fgraph.outputs[0].owner.inputs[0]
        assert not any(node.op == at.switch for node in f.maker.fgraph.toposort())

        vx = np.array([[1, 2, 3], [4, 5, 6]], dtype="int32")
        vy = np.array([10, 11, 12], dtype="int64")
        np_res = np.where(0, vx, vy)
        assert np.array_equal(f(vx, vy), np_res)

    def test_broadcasting_2(self):
        # test switch(cst, vector, matrix)

        x = vector("x", dtype="int32")
        y = matrix("y", dtype="int64")

        z = at.switch(1, x, y)
        f = function([x, y], z, mode=self.mode)

        assert isinstance(f.maker.fgraph.outputs[0].owner.op, Alloc)
        assert not any(node.op == at.switch for node in f.maker.fgraph.toposort())

        vx = np.array([4, 5, 6], dtype="int32")
        vy = np.array([[7, 8, 9], [10, 11, 12]], dtype="int64")
        np_res = np.where(1, vx, vy)
        assert np.array_equal(f(vx, vy), np_res)

        z = at.switch(0, x, y)
        f = function([x, y], z, mode=self.mode)

        assert isinstance(f.maker.fgraph.outputs[0].owner.op, DeepCopyOp)
        assert not any(node.op == at.switch for node in f.maker.fgraph.toposort())

        vx = np.array([4, 5, 6], dtype="int32")
        vy = np.array([[7, 8, 9], [10, 11, 12]], dtype="int64")
        np_res = np.where(0, vx, vy)
        assert np.array_equal(f(vx, vy), np_res)

    def test_broadcasting_3(self):
        # test switch(matrix, same_vector, same_vector)

        x = matrix("x", dtype="int32")
        y = vector("y", dtype="int64")
        z = at.switch(x, y, y)
        f = function([x, y], z, mode=self.mode)
        vx = np.array([[0, 1], [1, 0]], dtype="int32")
        vy = np.array([7, 8], dtype="int64")
        utt.assert_allclose(f(vx, vy), np.where(vx, vy, vy))

        assert isinstance(f.maker.fgraph.outputs[0].owner.op, Alloc)
        assert not any(node.op == at.switch for node in f.maker.fgraph.toposort())


class TestLocalMergeSwitchSameCond:
    @pytest.mark.parametrize(
        "op",
        [
            add,
            sub,
            mul,
            true_div,
            int_div,
            floor_div,
            minimum,
            maximum,
            gt,
            lt,
            ge,
            le,
            eq,
            neq,
            at_pow,
        ],
    )
    def test_elemwise_float_ops(self, op):
        # float Ops
        mats = matrices("cabxy")
        c, a, b, x, y = mats
        s1 = at.switch(c, a, b)
        s2 = at.switch(c, x, y)

        g = optimize(FunctionGraph(mats, [op(s1, s2)]))
        assert str(g).count("Switch") == 1

    @pytest.mark.parametrize(
        "op",
        [
            bitwise_and,
            bitwise_or,
            bitwise_xor,
        ],
    )
    def test_elemwise_int_ops(self, op):
        # integer Ops
        mats = imatrices("cabxy")
        c, a, b, x, y = mats
        s1 = at.switch(c, a, b)
        s2 = at.switch(c, x, y)
        g = optimize(FunctionGraph(mats, [op(s1, s2)]))
        assert str(g).count("Switch") == 1

    @pytest.mark.parametrize("op", [add, mul])
    def test_elemwise_multi_inputs(self, op):
        # add/mul with more than two inputs
        mats = imatrices("cabxy")
        c, a, b, x, y = mats
        s1 = at.switch(c, a, b)
        s2 = at.switch(c, x, y)
        u, v = matrices("uv")
        s3 = at.switch(c, u, v)
        g = optimize(FunctionGraph(mats + [u, v], [op(s1, s2, s3)]))
        assert str(g).count("Switch") == 1


class TestLocalOptAlloc:
    """
    TODO FIXME: These tests are incomplete; they need to `assert` something.
    """

    dtype = "float32"

    def test_sum_upcast(self):
        s = lscalar()
        a = at.alloc(np.asarray(5, dtype=self.dtype), s, s)
        with config.change_flags(warn_float64="raise"):
            f = function([s], a.sum())
            f(5)

    def test_prod_upcast(self):
        s = lscalar()
        a = at.alloc(np.asarray(5, dtype=self.dtype), s, s)

        with config.change_flags(warn_float64="raise"):
            f = function([s], a.prod())
            f(5)

    @config.change_flags(on_opt_error="raise")
    def test_sum_bool_upcast(self):
        s = lscalar()
        a = at.alloc(np.asarray(True, dtype="bool"), s, s)
        f = function([s], a.sum())
        f(5)
        # test with user specified dtype
        f = function([s], a.sum(dtype=self.dtype))
        f(5)
        # test only 1 axis summed
        f = function([s], a.sum(axis=0, dtype=self.dtype))
        f(5)


class TestLocalOptAllocF16(TestLocalOptAlloc):
    dtype = "float16"


def test_local_join_1():
    # test for vector
    a = vector("a")
    s = at.stack([a])
    f = function([a], s, mode=mode_opt)
    val = f([1])
    assert np.all(val == [1])
    e = f.maker.fgraph.toposort()
    assert len([n for n in e if isinstance(n.op, Join)]) == 0
    assert f.maker.fgraph.outputs[0].dtype == config.floatX

    # test for matrix join(0,a)
    a = matrix("a")
    s = join(0, a)
    f = function([a], s, mode=mode_opt)
    val = f([[1]])
    assert np.all(val == [[1]])
    e = f.maker.fgraph.toposort()
    assert len([n for n in e if isinstance(n.op, Join)]) == 0
    assert f.maker.fgraph.outputs[0].dtype == config.floatX

    # test for matrix join(1,a)
    s = join(1, a)
    f = function([a], s, mode=mode_opt)
    val = f([[1]])
    assert np.all(val == [[1]])
    e = f.maker.fgraph.toposort()
    assert len([n for n in e if isinstance(n.op, Join)]) == 0
    assert f.maker.fgraph.outputs[0].dtype == config.floatX

    # test we don't apply when their is 2 inputs
    s = join(1, a, a)
    f = function([a], s, mode=mode_opt)
    val = f([[1]])
    assert np.all(val == [[1]])
    e = f.maker.fgraph.toposort()
    assert len([n for n in e if isinstance(n.op, Join)]) == 1
    assert f.maker.fgraph.outputs[0].dtype == config.floatX


def test_local_join_empty():
    # test for vector, vector, empty to vector
    empty_vec = np.asarray([], dtype=config.floatX)
    a = vector("a")
    s = at.join(0, a, a, empty_vec)
    f = function([a], s, mode=mode_opt)
    val = f([1])
    assert np.all(val == [1])
    e = f.maker.fgraph.toposort()
    assert len([n for n in e if isinstance(n.op, Join)]) == 1
    assert all(
        [
            not isinstance(n.op, Join) or len(n.inputs) == 3
            for n in e
            if isinstance(n.op, Join)
        ]
    )
    assert f.maker.fgraph.outputs[0].dtype == config.floatX

    # test for matrix join(1,a)
    empty_mat = np.asarray([[]], dtype=config.floatX)
    m = matrix("m")
    s = join(1, empty_mat, m, m, m)
    f = function([m], s, mode=mode_opt)
    val = f([[1]])
    assert np.all(val == [[1]])
    e = f.maker.fgraph.toposort()
    assert len([n for n in e if isinstance(n.op, Join)]) == 1
    assert all(
        [
            not isinstance(n.op, Join) or len(n.inputs) == 4
            for n in e
            if isinstance(n.op, Join)
        ]
    )
    assert f.maker.fgraph.outputs[0].dtype == config.floatX
    # test for vector, vector, empty to matrix
    # We can't optimize this case.
    s = at.stack([a, a, empty_vec])
    f = function([a], s, mode=mode_opt)
    val = f([])
    assert np.all(val == [1])
    e = f.maker.fgraph.toposort()
    assert len([n for n in e if isinstance(n.op, Join)]) == 1
    assert all(
        [
            not isinstance(n.op, Join) or len(n.inputs) == 4
            for n in e
            if isinstance(n.op, Join)
        ]
    )
    assert f.maker.fgraph.outputs[0].dtype == config.floatX
    # test for matrix join(0,a)
    # We can't optimize this case.
    s = join(0, m, np.asarray([[2.0]], dtype=config.floatX), m)
    f = function([m], s, mode=mode_opt)
    val = f([[1]])
    assert np.all(val == [[1], [2], [1]])
    e = f.maker.fgraph.toposort()
    assert len([n for n in e if isinstance(n.op, Join)]) == 1
    assert all(
        [
            not isinstance(n.op, Join) or len(n.inputs) == 4
            for n in e
            if isinstance(n.op, Join)
        ]
    )
    assert f.maker.fgraph.outputs[0].dtype == config.floatX


def test_local_join_make_vector():
    a, b, c, d, e = scalars("abcde")
    v = vector("v")
    mv = MakeVector(config.floatX)
    s = at.join(0, mv(a), v, mv(b, c), mv(d, e))
    f = function([a, b, c, d, e, v], s, mode=mode_opt)
    val = f(1, 2, 3, 4, 6, [7, 8])
    assert np.all(val == [1, 7, 8, 2, 3, 4, 6])
    e = f.maker.fgraph.toposort()
    assert len([n for n in e if isinstance(n.op, Join)]) == 1
    assert all(
        [
            not isinstance(n.op, Join) or len(n.inputs) == 4
            for n in e
            if isinstance(n.op, Join)
        ]
    )
    assert f.maker.fgraph.outputs[0].dtype == config.floatX

    assert check_stack_trace(f, ops_to_check="all")


@pytest.mark.parametrize(
    "dtype",
    [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
        "complex64",
        "complex128",
    ],
)
def test_local_tensor_scalar_tensor(dtype):
    t_type = TensorType(dtype=dtype, shape=())
    t = t_type()
    s = at.scalar_from_tensor(t)
    t2 = at.tensor_from_scalar(s)

    f = function([t], t2, mode=mode_opt)
    e = f.maker.fgraph.toposort()
    assert not any(
        [n for n in e if isinstance(n.op, (TensorFromScalar, ScalarFromTensor))]
    )


@pytest.mark.parametrize(
    "dtype",
    [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
        "complex64",
        "complex128",
    ],
)
def test_local_scalar_tensor_scalar(dtype):
    s_type = aes.Scalar(dtype=dtype)
    s = s_type()
    t = at.tensor_from_scalar(s)
    s2 = at.scalar_from_tensor(t)

    f = function([s], s2, mode=mode_opt)
    e = f.maker.fgraph.toposort()
    assert not any(
        [n for n in e if isinstance(n.op, (TensorFromScalar, ScalarFromTensor))]
    )


def test_local_useless_split():
    x = matrix("x")
    splits = ivector("splits")
    opt = at.split(x, splits, n_splits=1)
    nonopt = at.split(x, splits, n_splits=3)

    mode = get_default_mode().including("local_useless_split")
    f_opt = function([x, splits], opt, mode=mode)
    f_nonopt = function([x, splits], nonopt, mode=mode)

    f_opt(np.random.random((4, 4)).astype(config.floatX), [4])
    f_nonopt(np.random.random((4, 4)).astype(config.floatX), [1, 2, 1])
    graph_opt = f_opt.maker.fgraph.toposort()
    graph_nonopt = f_nonopt.maker.fgraph.toposort()

    assert isinstance(graph_opt[-1].op, DeepCopyOp)
    assert len(graph_nonopt) == 1
    assert isinstance(graph_nonopt[0].op, Split)

    assert check_stack_trace(f_opt, ops_to_check=[Assert])
    assert check_stack_trace(f_nonopt, ops_to_check="all")


@pytest.mark.parametrize("i", list(range(1, 4)))
def test_local_flatten_lift(i):
    x = tensor4()
    out = at.flatten(exp(x), i)
    assert out.ndim == i
    mode = get_default_mode()
    mode = mode.including("local_reshape_lift")
    f = function([x], out, mode=mode)
    x_np = np.random.random((5, 4, 3, 2)).astype(config.floatX)
    out_np = f(x_np)
    topo = f.maker.fgraph.toposort()
    shape_out_np = tuple(x_np.shape[: i - 1]) + (np.prod(x_np.shape[i - 1 :]),)
    assert shape_out_np == out_np.shape

    reshape_nodes = [n for n in topo if isinstance(n.op, Reshape)]
    assert len(reshape_nodes) == 1 and at.is_flat(reshape_nodes[0].outputs[0], ndim=i)
    assert isinstance(topo[-1].op, Elemwise)


class TestReshape:
    def setup_method(self):
        self.mode = mode_opt
        self.op = Reshape

    def test_local_reshape(self):
        a = fmatrix()
        b = self.op(3)(a, [2, 3, 4])
        c = self.op(1)(b, [24])
        f = function([a], c, mode=self.mode)
        topo = f.maker.fgraph.toposort()
        assert sum(isinstance(node.op, self.op) for node in topo) == 1

        # Check stack trace
        assert check_stack_trace(f, ops_to_check=[self.op])


class TestLocalUselessReshape:
    def setup_method(self):
        self.rng = np.random.default_rng(utt.fetch_seed())

    def test_0(self):
        mode = get_default_mode().including("local_useless_reshape")
        i = iscalar("i")
        m = at.mgrid[
            0:i,
        ]
        f = function([i], m, mode=mode)
        topo = f.maker.fgraph.toposort()
        assert not any(isinstance(n.op, Reshape) for n in topo)

    def test_1(self):
        x = matrix("x")
        r = x.reshape(x.shape)

        m0 = get_default_mode()
        m1 = m0.including("local_useless_reshape")
        f1 = function([x], r, mode=m1)
        topo = f1.maker.fgraph.toposort()
        assert not any(isinstance(n.op, Reshape) for n in topo)

        m2 = m1.excluding("ShapeOpt")
        f2 = function([x], r, mode=m2)
        topo = f2.maker.fgraph.toposort()
        assert not any(isinstance(n.op, Reshape) for n in topo)

        # We do not need tests checking that stack traces are copied over,
        # because local_useless_reshape only removes nodes from the graph

    def test_2(self):
        x = matrix("x")
        r = x.reshape([Shape_i(i)(x) for i in range(x.ndim)])

        m0 = get_default_mode()
        m1 = m0.including("local_useless_reshape")
        f1 = function([x], r, mode=m1)
        topo = f1.maker.fgraph.toposort()
        assert not any(isinstance(n.op, Reshape) for n in topo)

        m2 = m1.excluding("ShapeOpt")
        f2 = function([x], r, mode=m2)
        topo = f2.maker.fgraph.toposort()
        assert not any(isinstance(n.op, Reshape) for n in topo)

    def test_m1(self):
        x = matrix("x")
        r = x.reshape((x.shape[0], -1))

        m0 = get_default_mode()
        m1 = m0.including("local_useless_reshape")
        f1 = function([x], r, mode=m1)
        topo = f1.maker.fgraph.toposort()
        assert not any(isinstance(n.op, Reshape) for n in topo)

        m2 = m1.excluding("ShapeOpt")
        f2 = function([x], r, mode=m2)
        topo = f2.maker.fgraph.toposort()
        assert not any(isinstance(n.op, Reshape) for n in topo)


class TestLocalReshapeToDimshuffle:
    def setup_method(self):
        self.rng = np.random.default_rng(utt.fetch_seed())

    def test_1(self):
        reshape_lift = out2in(local_reshape_to_dimshuffle)
        useless_reshape = out2in(local_useless_reshape)
        x = shared(self.rng.standard_normal((4,)))
        y = shared(self.rng.standard_normal((5, 6)))
        reshape_x = reshape(x, (1, 4))
        reshape_y = reshape(y, (1, 5, 1, 6, 1, 1))

        g = FunctionGraph([x, y], [reshape_x, reshape_y])
        assert str(g) == (
            "FunctionGraph(Reshape{2}"
            "(<TensorType(float64, (None,))>, "
            "TensorConstant{[1 4]}), "
            "Reshape{6}"
            "(<TensorType(float64, (None, None))>, "
            "TensorConstant{[1 5 1 6 1 1]}))"
        )

        reshape_lift.optimize(g)
        useless_reshape.optimize(g)
        assert str(g) == (
            "FunctionGraph(InplaceDimShuffle{x,0}"
            "(<TensorType(float64, (None,))>), "
            "InplaceDimShuffle{x,0,x,1,x,x}"
            "(Reshape{2}(<TensorType(float64, (None, None))>, "
            "TensorConstant{[5 6]})))"
        )

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(g, ops_to_check=(DimShuffle, Reshape))


def test_local_reshape_lift():
    x = tensor4()
    out = exp(x).reshape([x.size])
    assert out.ndim == 1
    mode = get_default_mode()
    mode = mode.including("local_reshape_lift")
    f = function([x], out, mode=mode)
    f(np.random.random((5, 4, 3, 2)).astype(config.floatX))
    topo = f.maker.fgraph.toposort()
    assert isinstance(topo[-2].op, Reshape)
    assert isinstance(topo[-1].op, Elemwise)
    # Check stacktrace was copied over correctly after opt was applied
    assert check_stack_trace(f, ops_to_check="last")


class TestLiftTransposeThroughDot:
    def simple_optimize(self, g):
        out2in(local_useless_elemwise).optimize(g)
        out2in(local_lift_transpose_through_dot).optimize(g)
        out2in(local_useless_elemwise).optimize(g)
        return g

    def test_matrix_matrix(self):
        a, b = matrices("ab")
        g = self.simple_optimize(FunctionGraph([a, b], [dot(a, b).T]))
        sg = "FunctionGraph(dot(InplaceDimShuffle{1,0}(b), InplaceDimShuffle{1,0}(a)))"
        assert str(g) == sg, (str(g), sg)
        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(g, ops_to_check="all")

    def test_row_matrix(self):
        a = vector("a")
        b = matrix("b")
        g = optimize(
            FunctionGraph([a, b], [dot(a.dimshuffle("x", 0), b).T]),
            level="stabilize",
        )
        sg = "FunctionGraph(dot(InplaceDimShuffle{1,0}(b), InplaceDimShuffle{0,x}(a)))"
        assert str(g) == sg, (str(g), sg)
        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(g, ops_to_check="all")

    def test_matrix_col(self):
        a = vector("a")
        b = matrix("b")
        g = optimize(
            FunctionGraph([a, b], [dot(b, a.dimshuffle(0, "x")).T]),
            level="stabilize",
        )
        sg = "FunctionGraph(dot(InplaceDimShuffle{x,0}(a), InplaceDimShuffle{1,0}(b)))"
        assert str(g) == sg, (str(g), sg)
        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(g, ops_to_check="all")


def test_local_upcast_elemwise_constant_inputs():
    s = dvector("s")
    x = at_sum(log(10 ** s))
    f = function([s], [aesara.gradient.grad(x, s)])
    f([-42, -2.1, -1, -0.5, 0, 0.2, 1, 2, 12])

    # This test a corner where the optimization should not be applied.
    with config.change_flags(floatX="float32"):
        v = lvector()
        function([v], true_div(v, 2))


class TestShapeI(utt.InferShapeTester):
    def setup_method(self):
        super().setup_method()

    def test_perform(self):
        rng = np.random.default_rng(utt.fetch_seed())

        advec = vector()
        advec_val = rng.random((3)).astype(config.floatX)
        f = function([advec], Shape_i(0)(advec))
        out = f(advec_val)
        utt.assert_allclose(out, advec_val.shape[0])

        admat = matrix()
        admat_val = rng.random((4, 3)).astype(config.floatX)
        for i in range(2):
            f = function([admat], Shape_i(i)(admat))
            out = f(admat_val)
            utt.assert_allclose(out, admat_val.shape[i])

    def test_infer_shape(self):
        admat = matrix()
        admat_val = np.random.random((3, 4)).astype(config.floatX)
        self._compile_and_check([admat], [Shape_i(0)(admat)], [admat_val], Shape_i)

        self._compile_and_check([admat], [Shape_i(1)(admat)], [admat_val], Shape_i)


class TestShapeFeature:
    def test_scalar(self):
        x = scalar()
        cst = at.constant(1).clone()
        o = x + cst
        fgraph = FunctionGraph([x], [o], clone=False)
        shape_feature = ShapeFeature()
        fgraph.attach_feature(shape_feature)
        assert shape_feature.same_shape(x, o)

    def test_vector(self):
        x = vector()
        cst = at.constant(1).clone()
        o = x + cst
        fgraph = FunctionGraph([x], [o], clone=False)
        shape_feature = ShapeFeature()
        fgraph.attach_feature(shape_feature)
        assert shape_feature.same_shape(x, o)

    def test_vector2(self):
        x = vector()
        y = vector()
        o = x + y
        fgraph = FunctionGraph([x, y], [o], clone=False)
        shape_feature = ShapeFeature()
        fgraph.attach_feature(shape_feature)
        assert shape_feature.same_shape(x, o)
        # The following case isn't implemented
        assert not shape_feature.same_shape(y, o)

    def test_vector_dim(self):
        x = vector()
        y = vector()
        o = x + y
        fgraph = FunctionGraph([x, y], [o], clone=False)
        shape_feature = ShapeFeature()
        fgraph.attach_feature(shape_feature)
        assert shape_feature.same_shape(x, o, 0, 0)
        # The following case isn't implemented
        assert not shape_feature.same_shape(y, o, 0, 0)

    def test_vector_dim_err(self):
        x = vector()
        y = vector()
        o = x + y
        fgraph = FunctionGraph([x, y], [o], clone=False)
        shape_feature = ShapeFeature()
        fgraph.attach_feature(shape_feature)
        with pytest.raises(IndexError):
            shape_feature.same_shape(x, o, 1, 0)
        with pytest.raises(IndexError):
            shape_feature.same_shape(x, o, 0, 1)


@pytest.mark.parametrize(
    "shape",
    [lscalar(), iscalar()],
)
def test_local_Shape_of_SpecifyShape(shape):
    x = vector()
    s = specify_shape(x, shape).shape

    fgraph = FunctionGraph(outputs=[s], clone=False)
    _ = optimize_graph(fgraph, clone=False)

    assert x not in fgraph.variables
    assert shape in fgraph.variables


def test_local_Shape_i_of_broadcastable():
    x = tensor(np.float64, [False, True])
    s = Shape_i(1)(x)

    fgraph = FunctionGraph(outputs=[s], clone=False)
    _ = optimize_graph(fgraph, clone=False)

    assert x not in fgraph.variables
    assert fgraph.outputs[0].data == 1

    # A test for a non-`TensorType`
    class MyType(Type):
        def filter(self, *args, **kwargs):
            raise NotImplementedError()

        def __eq__(self, other):
            return isinstance(other, MyType) and other.thingy == self.thingy

    class MyVariable(Variable):
        ndim = 1

    x = MyVariable(MyType(), None, None)
    s = Shape_i(0)(x)
    fgraph = FunctionGraph(outputs=[s], clone=False)
    _ = optimize_graph(fgraph, clone=False)

    assert fgraph.outputs[0] == s


def test_assert_op_gradient():
    x = vector("x")
    assert_op = Assert()
    cost = at_sum(assert_op(x, x.size < 2))
    grad = aesara.gradient.grad(cost, x)
    func = function([x], grad)

    x_val = np.ones(shape=(1,), dtype=config.floatX)
    assert func(x_val) == 1


def test_local_merge_alloc():
    # Add this opt to the default mode,
    # otherwise, FAST_COMPILE fails.
    default_mode = get_default_mode()
    opt_mode = default_mode.including("local_merge_alloc")

    x = iscalar("x")
    y = iscalar("y")
    y2 = iscalar("y2")
    z = iscalar("z")
    w = iscalar("w")
    m = fscalar("m")
    # case 1
    # Alloc(Alloc(m, x, 1, 1, 1), x, y, z, w) -> Alloc(m, x, y, z, w)
    output = at.alloc(at.alloc(m, 1, y, 1, 1), x, y, z, w)
    f = function([m, x, y, z, w], output, mode=opt_mode)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 1
    assert isinstance(topo[0].op, Alloc)
    o = f(0.0, 1, 2, 3, 4)
    assert o.shape == (1, 2, 3, 4)

    # case 2
    # Alloc(Alloc(m, y, 1, 1), x, y, z, w) -> Alloc(m, x, y, z, w)
    output = at.alloc(at.alloc(m, y, 1, 1), x, y, z, w)
    f = function([m, x, y, z, w], output, mode=opt_mode)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 1
    assert isinstance(topo[0].op, Alloc)
    o = f(0.0, 1, 2, 3, 4)
    assert o.shape == (1, 2, 3, 4)

    # case 3
    # Alloc(Alloc(m, y1, 1, 1), x, y2, z, w) ->
    #   Alloc(m, x, assert(y1, y1==y2), z, w)
    output = at.alloc(at.alloc(m, y, 1, 1), x, y2, z, w)
    f = function([m, x, y, y2, z, w], output, mode=opt_mode)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 3
    assert isinstance(topo[-2].op, Assert)
    assert isinstance(topo[-1].op, Alloc)
    o = f(0.0, 1, 2, 2, 3, 4)
    assert o.shape == (1, 2, 3, 4)
    with pytest.raises((AssertionError, ValueError)):
        f(0.0, 1, 2, 5, 3, 4)


def test_local_useless_alloc():

    useless_alloc = out2in(local_useless_alloc)
    merge_alloc = out2in(local_merge_alloc)

    x = iscalar("x")
    y = iscalar("y")
    y2 = iscalar("y2")
    z = iscalar("z")
    w = iscalar("w")
    m = fscalar("m")

    # case 1
    # Alloc(Alloc(m, x, 1, 1, 1), x, y, z, w) -> Alloc(m, x, y, z, w)
    output = at.alloc(at.alloc(m, 1, y, 1, 1), x, y, z, w)
    g = FunctionGraph([m, x, y, z, w], [output])

    useless_alloc.optimize(g)
    merge_alloc.optimize(g)
    useless_alloc.optimize(g)

    topo = g.toposort()
    assert len(topo) == 1
    assert isinstance(topo[0].op, Alloc)

    # case 2
    # Alloc(Alloc(m, y, 1, 1), x, y, z, w) -> Alloc(m, x, y, z, w)
    output = at.alloc(at.alloc(m, y, 1, 1), x, y, z, w)
    g = FunctionGraph([m, x, y, z, w], [output])

    useless_alloc.optimize(g)
    merge_alloc.optimize(g)
    useless_alloc.optimize(g)

    topo = g.toposort()
    assert len(topo) == 1
    assert isinstance(topo[0].op, Alloc)

    # case 3
    # Alloc(Alloc(m, y1, 1, 1), x, y2, z, w) ->
    #   Alloc(m, x, assert(y1, y1==y2), z, w)
    output = at.alloc(at.alloc(m, y, 1, 1), x, y2, z, w)
    g = FunctionGraph([m, x, y, y2, z, w], [output])

    useless_alloc.optimize(g)
    merge_alloc.optimize(g)
    useless_alloc.optimize(g)

    topo = g.toposort()
    assert len(topo) == 3
    assert isinstance(topo[-2].op, Assert)
    assert isinstance(topo[-1].op, Alloc)


def test_apply_rebroadcast_opt():
    # Test the `Elemwise` case in `local_rebroadcast_lift` with `fgraph=None`.
    # This is called by in `apply_rebroadcast_opt`.
    a = vector(dtype="float32")
    b = tensor("float64", [True])
    x = b.astype(a.dtype)

    broadcastable = (False,)
    axis = [(i, broadcastable[i]) for i in range(len(broadcastable))]
    rval = Rebroadcast(*axis)(x)

    res = apply_rebroadcast_opt(rval)
    assert res is rval


@pytest.mark.parametrize("return_index", [False])
@pytest.mark.parametrize("return_counts", [False])
@pytest.mark.parametrize("return_inverse", [False])
def test_local_Unique_scalar(return_index, return_counts, return_inverse):
    x = dscalar()
    y = unique(
        x,
        return_index=return_index,
        return_counts=return_counts,
        return_inverse=return_inverse,
        axis=None,
    )

    y_fg = FunctionGraph(outputs=[y], copy_inputs=False)
    y_opt_fg = optimize_graph(
        y_fg, clone=False, include=["canonicalize", "local_Unique_scalar"]
    )
    y_opt = y_opt_fg.outputs[0]
    y_opt_start = y_opt

    if isinstance(y_opt.owner.op, Rebroadcast):
        y_opt_start = y_opt.owner.inputs[0]

    assert isinstance(y_opt_start.owner.op, DimShuffle)
    assert y_opt_start.owner.inputs[0] == x

    default_mode = get_default_mode()
    opt_mode = default_mode.excluding("local_Unique_scalar")
    y_fn = function([x], [y, y_opt], mode=opt_mode)

    x_val = np.array(-10.0, dtype=np.float64)
    y_exp_val, y_val = y_fn(x_val)
    assert np.array_equal(y_exp_val, y_val)


@pytest.mark.parametrize(
    "x_val, axis, new_shape",
    [
        (np.array(-10, dtype=np.int64), None, ()),
        (np.array(-10, dtype=np.int64), None, (2, 3)),
        (np.array([[-10, -3], [-10, 2], [-10, 2]], dtype=np.int64), None, (2, 3, 2)),
    ],
)
@pytest.mark.parametrize("return_index", [False])
@pytest.mark.parametrize("return_counts", [False])
@pytest.mark.parametrize("return_inverse", [False])
def test_local_Unique_Alloc_lift(
    x_val, axis, new_shape, return_index, return_counts, return_inverse
):
    x = as_tensor_variable(x_val).type()
    y = unique(
        alloc(x, *new_shape),
        return_index=return_index,
        return_counts=return_counts,
        return_inverse=return_inverse,
        axis=axis,
    )

    if isinstance(y, list):
        y, *_ = y

    # This approach allows us to directly confirm that `x` is in the result.
    y_fg = FunctionGraph(outputs=[y], copy_inputs=False)
    y_opt_fg = optimize_graph(
        y_fg,
        clone=False,
        include=["canonicalize", "local_Unique_Alloc_lift"],
        exclude=["local_Unique_scalar"],
    )
    y_opt = y_opt_fg.outputs[0]
    y_opt_start = y_opt

    # Ignore any initial `Rebroadcast`s (they serve to
    # make the replacement match the original type)
    if isinstance(y_opt.owner.op, Rebroadcast):
        y_opt_start = y_opt.owner.inputs[0]

    assert isinstance(y_opt_start.owner.op, Unique)
    assert y_opt_start.owner.inputs[0] == x
    assert not any(isinstance(node.op, Alloc) for node in y_opt_fg.apply_nodes)

    default_mode = get_default_mode()
    # The optimization has already been applied to `y_opt`, so we can--and
    # should--exclude it from the compilation of both our reference, `y`, and
    # the optimized result, `y_opt`.
    # The remaining exclusions simply allow us to perform the check below that
    # makes sure the original `Alloc` is present in our reference (sub)graph.
    opt_mode = default_mode.excluding(
        "local_useless_alloc", "local_alloc_sink_dimshuffle", "local_Unique_Alloc_lift"
    )
    y_fn = function([x], [y, y_opt], mode=opt_mode)
    # Make sure that the original `Alloc` is used to compute the reference `y`
    # result
    assert any(isinstance(node.op, Alloc) for node in y_fn.maker.fgraph.apply_nodes)

    y_exp_val, y_val = y_fn(x_val)
    assert np.array_equal(y_exp_val, y_val)


@pytest.mark.parametrize(
    "x_val, axis, new_shape",
    [
        (np.array(-10, dtype=np.int64), None, (2, 3)),
        (np.array([[-10, -3], [-10, 2], [-10, 2]], dtype=np.int64), None, (2, 3, 2)),
    ],
)
@pytest.mark.parametrize("return_index", [False])
@pytest.mark.parametrize("return_counts", [False])
@pytest.mark.parametrize("return_inverse", [False])
def test_local_Unique_BroadcastTo(
    x_val, axis, new_shape, return_index, return_counts, return_inverse
):
    x = as_tensor_variable(x_val).type()
    y = unique(
        BroadcastTo()(x, tuple(new_shape)),
        return_index=return_index,
        return_counts=return_counts,
        return_inverse=return_inverse,
        axis=axis,
    )

    if isinstance(y, list):
        y, *_ = y

    # This approach allows us to directly confirm that `x` is in the result.
    y_fg = FunctionGraph(outputs=[y], copy_inputs=False)
    y_opt_fg = optimize_graph(
        y_fg,
        clone=False,
        include=["canonicalize", "local_Unique_BroadcastTo_lift"],
        exclude=["local_Unique_scalar"],
    )
    y_opt = y_opt_fg.outputs[0]
    y_opt_start = y_opt

    # Ignore any initial `Rebroadcast`s (they serve to
    # make the replacement match the original type)
    if isinstance(y_opt.owner.op, Rebroadcast):
        y_opt_start = y_opt.owner.inputs[0]

    assert isinstance(y_opt_start.owner.op, Unique)
    assert y_opt_start.owner.inputs[0] == x
    assert not any(isinstance(node.op, BroadcastTo) for node in y_opt_fg.apply_nodes)

    default_mode = get_default_mode()
    # The optimization has already been applied to `y_opt`, so we can--and
    # should--exclude it from the compilation of both our reference, `y`, and
    # the optimized result, `y_opt`.
    opt_mode = default_mode.excluding("local_Unique_BroadcastTo_lift")
    y_fn = function([x], [y, y_opt], mode=opt_mode)
    # Make sure that the original `BroadcastTo` is used to compute the
    # reference `y` result
    assert any(
        isinstance(node.op, BroadcastTo) for node in y_fn.maker.fgraph.apply_nodes
    )

    y_exp_val, y_val = y_fn(x_val)
    assert np.array_equal(y_exp_val, y_val)


@pytest.mark.parametrize(
    "x_val, unique_axis, repeats, repeat_axis",
    [
        (np.array([[-10, -3], [-10, 2]], dtype=np.int64), None, (1, 2), 0),
    ],
)
@pytest.mark.parametrize("return_index", [False])
@pytest.mark.parametrize("return_counts", [False])
@pytest.mark.parametrize("return_inverse", [False])
def test_local_Unique_Repeat(
    x_val,
    unique_axis,
    repeats,
    repeat_axis,
    return_index,
    return_counts,
    return_inverse,
):
    x = as_tensor_variable(x_val).type()
    y = unique(
        repeat(x, tuple(repeats), axis=repeat_axis),
        return_index=return_index,
        return_counts=return_counts,
        return_inverse=return_inverse,
        axis=unique_axis,
    )

    if isinstance(y, list):
        y, *_ = y

    # This approach allows us to directly confirm that `x` is in the result.
    y_fg = FunctionGraph(outputs=[y], copy_inputs=False)
    y_opt_fg = optimize_graph(
        y_fg,
        clone=False,
        include=["canonicalize", "local_Unique_Repeat_lift"],
        exclude=["local_Unique_scalar"],
    )
    y_opt = y_opt_fg.outputs[0]
    y_opt_start = y_opt

    # Ignore any initial `Rebroadcast`s (they serve to
    # make the replacement match the original type)
    if isinstance(y_opt.owner.op, Rebroadcast):
        y_opt_start = y_opt.owner.inputs[0]

    assert isinstance(y_opt_start.owner.op, Unique)
    assert y_opt_start.owner.inputs[0] == x
    assert not any(isinstance(node.op, Repeat) for node in y_opt_fg.apply_nodes)

    default_mode = get_default_mode()
    # The optimization has already been applied to `y_opt`, so we can--and
    # should--exclude it from the compilation of both our reference, `y`, and
    # the optimized result, `y_opt`.
    opt_mode = default_mode.excluding("local_Unique_Repeat_lift")
    y_fn = function([x], [y, y_opt], mode=opt_mode)
    # Make sure that the original `BroadcastTo` is used to compute the
    # reference `y` result
    assert any(isinstance(node.op, Repeat) for node in y_fn.maker.fgraph.apply_nodes)

    y_exp_val, y_val = y_fn(x_val)
    assert np.array_equal(y_exp_val, y_val)


@pytest.mark.parametrize(
    "x_val, unique_axis, new_shape",
    [
        (np.array(-10, dtype=np.int64), None, ()),
        (np.array(-10, dtype=np.int64), None, (2, 3)),
        (np.array([[-10, -3], [-10, 2], [-10, 2]], dtype=np.int64), None, (2, 3, 2)),
    ],
)
@pytest.mark.parametrize("return_index", [False])
@pytest.mark.parametrize("return_counts", [False])
@pytest.mark.parametrize("return_inverse", [False])
def test_local_Unique_second(
    x_val, unique_axis, new_shape, return_index, return_counts, return_inverse
):
    x = as_tensor_variable(x_val).type()
    a = np.zeros(tuple(new_shape), dtype=x.dtype)
    y = unique(
        second(a, x),
        return_index=return_index,
        return_counts=return_counts,
        return_inverse=return_inverse,
        axis=unique_axis,
    )

    if isinstance(y, list):
        y, *_ = y

    # This approach allows us to directly confirm that `x` is in the result.
    y_fg = FunctionGraph(outputs=[y], copy_inputs=False)
    y_opt_fg = optimize_graph(
        y_fg,
        clone=False,
        include=["canonicalize", "local_Unique_second_lift"],
        exclude=["local_Unique_scalar", "topo_constant_folding"],
    )
    y_opt = y_opt_fg.outputs[0]
    y_opt_start = y_opt

    # Ignore any initial `Rebroadcast`s (they serve to
    # make the replacement match the original type)
    if y_opt.owner and isinstance(y_opt.owner.op, Rebroadcast):
        y_opt_start = y_opt.owner.inputs[0]

    assert isinstance(y_opt_start.owner.op, Unique)

    y_opt_start = y_opt_start.owner.inputs[0]

    if y_opt_start.owner and isinstance(y_opt_start.owner.op, DimShuffle):
        y_opt_start = y_opt_start.owner.inputs[0]

    assert y_opt_start == x
    assert not any(
        isinstance(node.op.scalar_op, aes.Second)
        for node in y_opt_fg.apply_nodes
        if isinstance(node.op, Elemwise)
    )

    # The optimization has already been applied to `y_opt`, so we can--and
    # should--exclude it from the compilation of both our reference, `y`, and
    # the optimized result, `y_opt`.
    y_fn = function([x], [y, y_opt], mode=Mode(optimizer=OPT_NONE))

    # Make sure that the original `BroadcastTo` is used to compute the
    # reference `y` result
    assert any(
        isinstance(node.op.scalar_op, aes.Second)
        for node in y_fn.maker.fgraph.apply_nodes
        if isinstance(node.op, Elemwise)
    )

    y_exp_val, y_val = y_fn(x_val)
    assert np.array_equal(y_exp_val, y_val)


def test_local_useless_SpecifyShape():
    x = matrix()
    s = at.as_tensor([iscalar(), iscalar()])
    y = specify_shape(specify_shape(x, s), s)

    y_fg = FunctionGraph(outputs=[y], copy_inputs=False)
    y_opt_fg = optimize_graph(
        y_fg, clone=False, include=["canonicalize", "local_useless_SpecifyShape"]
    )
    y_opt = y_opt_fg.outputs[0]

    assert isinstance(y_opt.owner.op, SpecifyShape)
    assert y_opt.owner.inputs[0] == x


def test_printing():
    a, b = scalars("ab")
    mv = MakeVector(config.floatX)
    v = mv(a, b)
    assert pprint(v) == "[a, b]"


def test_local_remove_scalar_BroadcastTo():
    x = dscalar()
    y = BroadcastTo()(x, ())

    assert isinstance(y.owner.op, BroadcastTo)

    res = optimize_graph(
        y, clone=False, include=["canonicalize", "local_remove_scalar_BroadcastTo"]
    )

    assert res is x


def test_local_useless_dimshuffle_makevector():
    a = scalar()
    x = MakeVector(config.floatX)(a)
    y = x.dimshuffle(())

    y_fg = FunctionGraph(outputs=[y], copy_inputs=False)

    y_opt_fg = optimize_graph(
        y_fg,
        clone=False,
        include=["canonicalize", "local_useless_dimshuffle_makevector"],
    )

    assert y_opt_fg.outputs[0] == a


def test_Shape_i_canonicalize():
    """Make sure the canonicalizations work together to produce the correct graphs for shapes in a single dimension.

    In other words, ``shape(x)[i]`` should result in a simple ``Shape_i(0)(x)``
    and nothing else.  The rewrites `local_shape_to_shape_i`,
    `local_subtensor_remove_broadcastable_index`, and
    `local_useless_dimshuffle_makevector` need to work together to accomplish
    this, and we confirm that here.
    """
    x = vector()
    y = shape(x)[0]

    y_fg = FunctionGraph(outputs=[y], copy_inputs=False, features=[ShapeFeature()])

    y_opt_fg = optimize_graph(
        y_fg,
        clone=False,
        include=[
            "canonicalize",
        ],
    )

    y_opt = y_opt_fg.outputs[0]

    assert isinstance(y_opt.owner.op, Shape_i)
    assert y_opt.owner.op.i == 0
    assert y_opt.owner.inputs[0] == x
