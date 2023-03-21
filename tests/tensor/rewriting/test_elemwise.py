import contextlib

import numpy as np
import pytest

import aesara
import aesara.scalar as aes
import aesara.tensor as at
from aesara import shared
from aesara.compile.function import function
from aesara.compile.mode import Mode, get_default_mode
from aesara.configdefaults import config
from aesara.graph.basic import Constant
from aesara.graph.fg import FunctionGraph
from aesara.graph.rewriting.basic import check_stack_trace, out2in
from aesara.graph.rewriting.db import RewriteDatabaseQuery
from aesara.graph.rewriting.utils import rewrite_graph
from aesara.misc.safe_asarray import _asarray
from aesara.scalar.basic import Composite
from aesara.tensor.basic import MakeVector
from aesara.tensor.elemwise import DimShuffle, Elemwise
from aesara.tensor.math import (
    add,
    bitwise_and,
    bitwise_or,
    cos,
    cosh,
    dot,
    eq,
    exp,
    floor_divide,
    invert,
    iround,
    log,
    log2,
    log10,
    mul,
    neg,
    neq,
)
from aesara.tensor.math import pow as at_pow
from aesara.tensor.math import reciprocal
from aesara.tensor.math import round as at_round
from aesara.tensor.math import sin, sinh, sqrt, square
from aesara.tensor.math import sum as at_sum
from aesara.tensor.math import tan, tanh, true_divide, xor
from aesara.tensor.rewriting.elemwise import local_dimshuffle_lift
from aesara.tensor.rewriting.shape import local_useless_dimshuffle_in_reshape
from aesara.tensor.shape import reshape
from aesara.tensor.type import (
    TensorType,
    dmatrices,
    dscalar,
    dvector,
    fscalar,
    fvector,
    matrix,
    scalar,
    tensor,
    vector,
    vectors,
)
from tests import unittest_tools as utt


dimshuffle_lift = out2in(local_dimshuffle_lift)


def ds(x, y):
    return DimShuffle(x.type.broadcastable, y)(x)


def inputs(xbc=(0, 0), ybc=(0, 0), zbc=(0, 0)):
    x = TensorType(dtype="float64", shape=xbc)("x")
    y = TensorType(dtype="float64", shape=ybc)("y")
    z = TensorType(dtype="float64", shape=zbc)("z")
    return x, y, z


class TestDimshuffleLift:
    def test_double_transpose(self):
        x, y, z = inputs()
        e = ds(ds(x, (1, 0)), (1, 0))
        g = FunctionGraph([x], [e])
        # TODO FIXME: Construct these graphs and compare them.
        assert (
            str(g) == "FunctionGraph(InplaceDimShuffle{1,0}(InplaceDimShuffle{1,0}(x)))"
        )
        dimshuffle_lift.rewrite(g)
        assert str(g) == "FunctionGraph(x)"
        # no need to check_stack_trace as graph is supposed to be empty

    def test_merge2(self):
        x, y, z = inputs()
        e = ds(ds(x, (1, "x", 0)), (2, 0, "x", 1))
        g = FunctionGraph([x], [e])
        # TODO FIXME: Construct these graphs and compare them.
        assert (
            str(g)
            == "FunctionGraph(InplaceDimShuffle{2,0,x,1}(InplaceDimShuffle{1,x,0}(x)))"
        ), str(g)
        dimshuffle_lift.rewrite(g)
        assert str(g) == "FunctionGraph(InplaceDimShuffle{0,1,x,x}(x))", str(g)
        # Check stacktrace was copied over correctly after rewrite was applied
        assert check_stack_trace(g, ops_to_check="all")

    def test_elim3(self):
        x, y, z = inputs()
        e = ds(ds(ds(x, (0, "x", 1)), (2, 0, "x", 1)), (1, 0))
        g = FunctionGraph([x], [e])
        # TODO FIXME: Construct these graphs and compare them.
        assert str(g) == (
            "FunctionGraph(InplaceDimShuffle{1,0}(InplaceDimShuffle{2,0,x,1}"
            "(InplaceDimShuffle{0,x,1}(x))))"
        ), str(g)
        dimshuffle_lift.rewrite(g)
        assert str(g) == "FunctionGraph(x)", str(g)
        # no need to check_stack_trace as graph is supposed to be empty

    def test_lift(self):
        x, y, z = inputs([False] * 1, [False] * 2, [False] * 3)
        e = x + y + z
        g = FunctionGraph([x, y, z], [e])

        # TODO FIXME: Construct these graphs and compare them.
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

        rewrite_str_g_inplace = (
            "FunctionGraph(Elemwise{add,no_inplace}(Elemwise{add,no_inplace}"
            "(InplaceDimShuffle{x,x,0}(x), InplaceDimShuffle{x,0,1}(y)), z))"
        )
        rewrite_str_g_noinplace = (
            "FunctionGraph(Elemwise{add,no_inplace}(Elemwise{add,no_inplace}"
            "(DimShuffle{x,x,0}(x), DimShuffle{x,0,1}(y)), z))"
        )
        dimshuffle_lift.rewrite(g)
        assert str(g) in (rewrite_str_g_inplace, rewrite_str_g_noinplace), str(g)
        # Check stacktrace was copied over correctly after rewrite was applied
        assert check_stack_trace(g, ops_to_check="all")

    def test_recursive_lift(self):
        v = vector(dtype="float64")
        m = matrix(dtype="float64")
        out = ((v + 42) * (m + 84)).T
        g = FunctionGraph([v, m], [out])
        # TODO FIXME: Construct these graphs and compare them.
        init_str_g = (
            "FunctionGraph(InplaceDimShuffle{1,0}(Elemwise{mul,no_inplace}"
            "(InplaceDimShuffle{x,0}(Elemwise{add,no_inplace}"
            "(<TensorType(float64, (?,))>, "
            "InplaceDimShuffle{x}(TensorConstant{42}))), "
            "Elemwise{add,no_inplace}"
            "(<TensorType(float64, (?, ?))>, "
            "InplaceDimShuffle{x,x}(TensorConstant{84})))))"
        )
        assert str(g) == init_str_g
        new_out = local_dimshuffle_lift.transform(g, g.outputs[0].owner)[0]
        new_g = FunctionGraph(g.inputs, [new_out])
        rewrite_str_g = (
            "FunctionGraph(Elemwise{mul,no_inplace}(Elemwise{add,no_inplace}"
            "(InplaceDimShuffle{0,x}(<TensorType(float64, (?,))>), "
            "InplaceDimShuffle{x,x}(TensorConstant{42})), "
            "Elemwise{add,no_inplace}(InplaceDimShuffle{1,0}"
            "(<TensorType(float64, (?, ?))>), "
            "InplaceDimShuffle{x,x}(TensorConstant{84}))))"
        )
        assert str(new_g) == rewrite_str_g
        # Check stacktrace was copied over correctly after rewrite was applied
        assert check_stack_trace(new_g, ops_to_check="all")

    def test_useless_dimshuffle(self):
        x, _, _ = inputs()
        e = ds(x, (0, 1))
        g = FunctionGraph([x], [e])
        # TODO FIXME: Construct these graphs and compare them.
        assert str(g) == "FunctionGraph(InplaceDimShuffle{0,1}(x))"
        dimshuffle_lift.rewrite(g)
        assert str(g) == "FunctionGraph(x)"
        # Check stacktrace was copied over correctly after rewrite was applied
        assert hasattr(g.outputs[0].tag, "trace")

    def test_dimshuffle_on_broadcastable(self):
        x, y, z = inputs([False, True], [True, False, True], [False, False, True])
        u = at.constant(1)
        ds_x = ds(x, (0, "x"))  # useless
        ds_y = ds(y, (2, 1, 0))  # useless
        ds_z = ds(z, (2, 1, 0))  # useful
        ds_u = ds(u, ("x"))  # useful
        g = FunctionGraph([x, y, z, u], [ds_x, ds_y, ds_z, ds_u])
        # TODO FIXME: Construct these graphs and compare them.
        assert (
            str(g)
            == "FunctionGraph(InplaceDimShuffle{0,x}(x), InplaceDimShuffle{2,1,0}(y), InplaceDimShuffle{2,1,0}(z), InplaceDimShuffle{x}(TensorConstant{1}))"
        )
        dimshuffle_lift.rewrite(g)
        assert (
            str(g)
            == "FunctionGraph(x, y, InplaceDimShuffle{2,1,0}(z), InplaceDimShuffle{x}(TensorConstant{1}))"
        )
        # Check stacktrace was copied over correctly after rewrite was applied
        assert hasattr(g.outputs[0].tag, "trace")


def test_local_useless_dimshuffle_in_reshape():
    vec = TensorType(dtype="float64", shape=(None,))("vector")
    mat = TensorType(dtype="float64", shape=(None, None))("mat")
    row = TensorType(dtype="float64", shape=(1, None))("row")
    col = TensorType(dtype="float64", shape=(None, 1))("col")

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

    # TODO FIXME: Construct these graphs and compare them.
    assert str(g) == (
        "FunctionGraph(Reshape{1}(InplaceDimShuffle{x,0}(vector), Shape(vector)), "
        "Reshape{2}(InplaceDimShuffle{x,0,x,1}(mat), Shape(mat)), "
        "Reshape{2}(InplaceDimShuffle{1,x}(row), Shape(row)), "
        "Reshape{2}(InplaceDimShuffle{0}(col), Shape(col)))"
    )
    useless_dimshuffle_in_reshape = out2in(local_useless_dimshuffle_in_reshape)
    useless_dimshuffle_in_reshape.rewrite(g)
    assert str(g) == (
        "FunctionGraph(Reshape{1}(vector, Shape(vector)), "
        "Reshape{2}(mat, Shape(mat)), "
        "Reshape{2}(row, Shape(row)), "
        "Reshape{2}(col, Shape(col)))"
    )

    # Check stacktrace was copied over correctly after rewrite was applied
    assert check_stack_trace(g, ops_to_check="all")

    # Check that the rewrite does not get applied when the order
    # of dimensions has changed.
    reshape_dimshuffle_mat2 = reshape(mat.dimshuffle("x", 1, "x", 0), mat.shape)
    h = FunctionGraph([mat], [reshape_dimshuffle_mat2])
    str_h = str(h)
    useless_dimshuffle_in_reshape.rewrite(h)
    assert str(h) == str_h


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

    def my_init(dtype="float64", num=0):
        return np.zeros((5, 5), dtype=dtype) + num

    fw, fx, fy, fz = (
        tensor(dtype="float32", shape=(None,) * 2, name=n) for n in "wxyz"
    )
    dw, dx, dy, dz = (
        tensor(dtype="float64", shape=(None,) * 2, name=n) for n in "wxyz"
    )
    ix, iy, iz = (tensor(dtype="int32", shape=(None,) * 2, name=n) for n in "xyz")
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
                    expected_len_sym_inputs = sum(
                        not isinstance(x, Constant) for x in topo_[0].inputs
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
            f = cst_m05 * sd**cst_m2 * (ones - means[i]) ** cst_2 + cst_05 * log(
                cst_05 * (sd**cst_m2) / np.pi
            )
            factors.append(at_sum(f))

        logp = add(*factors)

        vars = [sd, means]

        # Make sure that C compilation is used
        mode = Mode("cvm", self.rewrites)
        dlogp = function(vars, [aesara.grad(logp, v) for v in vars], mode=mode)

        # Make sure something was fused
        assert any(
            isinstance(getattr(node.op, "scalar_op", None), aes.basic.Composite)
            for node in dlogp.maker.fgraph.toposort()
        )

    def test_add_mul_fusion_inplace(self):
        rewrites = RewriteDatabaseQuery(
            include=[
                "local_elemwise_fusion",
                "composite_elemwise_fusion",
                "canonicalize",
                "inplace",
            ],
            exclude=["cxx_only", "BlasOpt"],
        )

        mode = Mode(self.mode.linker, rewrites)

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

    @pytest.mark.parametrize("test_value", [np.c_[[1.0]], np.c_[[]]])
    def test_test_values(self, test_value):
        """Make sure that `local_elemwise_fusion_op` uses test values correctly when they have zero dimensions.

        The test values we're talking about are the ones used when C implementations
        are checked.

        """

        rewrites = RewriteDatabaseQuery(
            include=[
                "local_elemwise_fusion",
                "composite_elemwise_fusion",
                "canonicalize",
            ],
            exclude=["cxx_only", "BlasOpt"],
        )

        mode = Mode(self.mode.linker, rewrites)

        x, y, z = dmatrices("xyz")

        x.tag.test_value = test_value
        y.tag.test_value = test_value
        z.tag.test_value = test_value

        if test_value.size == 0:
            cm = pytest.raises(ValueError)
        else:
            cm = contextlib.suppress()

        with config.change_flags(
            compute_test_value="raise", compute_test_value_opt="raise"
        ):
            out = x * y + z
            with cm:
                f = function([x, y, z], out, mode=mode)

        if test_value.size != 0:
            # Confirm that the fusion happened
            assert isinstance(f.maker.fgraph.outputs[0].owner.op.scalar_op, Composite)
            assert len(f.maker.fgraph.toposort()) == 1

            x_c, y_c, z_c = f.maker.fgraph.outputs[0].owner.inputs
            assert np.array_equal(
                f.maker.fgraph.outputs[0].tag.test_value, np.c_[[2.0]]
            )

    @pytest.mark.parametrize("linker", ["cvm", "py"])
    @pytest.mark.parametrize("axis", [None, 0, 1, (0, 1), (0, 1, 2)])
    def test_CAReduce_single_input(self, linker, axis):
        """Make sure that `CAReduce` and `Elemwise` fusions work with a single input."""

        mode = Mode(linker=linker)
        mode._optimizer = mode._optimizer.including(
            "local_careduce_fusion",
            "canonicalize",
            "inplace",
        )

        x = tensor("floatX", shape=(None, None, None), name="x")
        out = exp(x).sum(axis=axis)

        out_fn = function([x], out, mode=mode)

        if linker != "py":
            (out_node,) = out_fn.maker.fgraph.toposort()
            assert isinstance(getattr(out_node.op, "scalar_op"), aes.basic.Composite)

            rng = np.random.default_rng(2320)
            x_val = rng.random((4, 3, 2), dtype=config.floatX)

            exp_res = np.exp(x_val).sum(axis=axis)

            out_val = out_fn(x_val)
            assert out_val.shape == exp_res.shape
            assert np.allclose(out_val, exp_res)
        else:
            out_nodes = out_fn.maker.fgraph.toposort()
            assert not any(
                isinstance(out_node.op.scalar_op, aes.basic.Composite)
                for out_node in out_nodes
                if hasattr(out_node.op, "scalar_op")
            )

        # `Elemwise`s with more than one client shouldn't be rewritten
        x = tensor("floatX", shape=(None, None, None), name="x")
        exp_x = exp(x)
        out = exp_x.sum(axis=axis) + exp(x)

        out_fn = function([x], out, mode=mode)
        out_nodes = out_fn.maker.fgraph.toposort()
        assert not any(
            isinstance(out_node.op.scalar_op, aes.basic.Composite)
            for out_node in out_nodes
            if hasattr(out_node.op, "scalar_op")
        )

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.parametrize("linker", ["cvm", "py"])
    @pytest.mark.parametrize("axis", [None, 0, 1, (0, 1), (0, 1, 2)])
    def test_CAReduce_multiple_inputs(self, linker, axis):
        """Make sure that `CAReduce` and `Elemwise` fusions work with multiple inputs."""

        mode = Mode(linker=linker)
        mode._optimizer = mode._optimizer.including(
            "local_careduce_fusion",
            "canonicalize",
            "inplace",
        )

        x = tensor("floatX", shape=(None, None, None), name="x")
        y = tensor("floatX", shape=(None, None, None), name="y")
        out = (x + y).sum(axis=axis)

        out_fn = function([x, y], out, mode=mode)
        (out_node,) = out_fn.maker.fgraph.toposort()

        assert isinstance(getattr(out_node.op, "scalar_op"), aes.basic.Composite)

        rng = np.random.default_rng(2320)
        x_val = rng.random((4, 3, 2), dtype=config.floatX)
        y_val = rng.random((4, 3, 2), dtype=config.floatX)
        exp_res = (x_val + y_val).sum(axis=axis)
        out_val = out_fn(x_val, y_val)
        assert out_val.shape == exp_res.shape
        assert np.allclose(out_val, exp_res)


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


def test_local_useless_dimshuffle_makevector():
    a = scalar()
    x = MakeVector(config.floatX)(a)
    y = x.dimshuffle(())

    y_fg = FunctionGraph(outputs=[y], copy_inputs=False)

    y_rewritten_fg = rewrite_graph(
        y_fg,
        clone=False,
        include=["canonicalize", "local_useless_dimshuffle_makevector"],
    )

    assert y_rewritten_fg.outputs[0] == a
