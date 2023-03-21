import numpy as np
import pytest

import aesara
import aesara.scalar as aes
import aesara.tensor as at
from aesara import shared
from aesara.compile.function import function
from aesara.compile.mode import Mode, get_default_mode, get_mode
from aesara.compile.ops import DeepCopyOp
from aesara.configdefaults import config
from aesara.graph.basic import Constant, Variable, ancestors
from aesara.graph.rewriting.basic import check_stack_trace
from aesara.graph.rewriting.db import RewriteDatabaseQuery
from aesara.graph.rewriting.utils import rewrite_graph
from aesara.graph.type import Type
from aesara.raise_op import Assert
from aesara.tensor import inplace
from aesara.tensor.basic import Alloc, MakeVector, _convert_to_int8, make_vector
from aesara.tensor.elemwise import DimShuffle, Elemwise
from aesara.tensor.math import Dot, add, dot, exp, square
from aesara.tensor.rewriting.subtensor import (
    local_replace_AdvancedSubtensor,
    local_subtensor_shape_constant,
)
from aesara.tensor.shape import SpecifyShape, Unbroadcast, _shape, shape, specify_shape
from aesara.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    AdvancedSubtensor,
    AdvancedSubtensor1,
    IncSubtensor,
    Subtensor,
    advanced_inc_subtensor1,
    inc_subtensor,
    set_subtensor,
)
from aesara.tensor.type import (
    bmatrix,
    col,
    dmatrix,
    fmatrix,
    iscalar,
    iscalars,
    ivector,
    lscalar,
    lscalars,
    matrix,
    row,
    scalar,
    tensor,
    tensor3,
    tensor4,
    vector,
)
from aesara.tensor.type_other import make_slice, slicetype
from tests import unittest_tools as utt
from tests.unittest_tools import create_aesara_param


mode_opt = config.mode
if mode_opt == "FAST_COMPILE":
    mode_opt = "FAST_RUN"
mode_opt = get_mode(mode_opt)


y = create_aesara_param(np.random.default_rng().integers(0, 4, size=(2,)))
z = create_aesara_param(np.random.default_rng().integers(0, 4, size=(2, 2)))


@pytest.mark.parametrize(
    ("indices", "is_none"),
    [
        ((slice(None), y, y), True),
        ((y, y, slice(None)), True),
        ((y,), False),
        ((slice(None), y), False),
        ((y, slice(None)), False),
        ((slice(None), y, slice(None)), False),
        ((slice(None), z, slice(None)), False),
        ((slice(None), z), False),
        ((z, slice(None)), False),
        ((slice(None), z, slice(None)), False),
    ],
)
def test_local_replace_AdvancedSubtensor(indices, is_none):
    X_val = np.random.normal(size=(4, 4, 4))
    X = tensor(np.float64, shape=(None, None, None), name="X")
    X.tag.test_value = X_val

    Y = X[indices]

    res_at = local_replace_AdvancedSubtensor.transform(None, Y.owner)

    if is_none:
        assert res_at is None
    else:
        (res_at,) = res_at

        assert not any(
            isinstance(v.owner.op, AdvancedSubtensor)
            for v in ancestors([res_at])
            if v.owner
        )

        inputs = [X] + [i for i in indices if isinstance(i, Variable)]

        res_fn = function(inputs, res_at, mode=Mode("py", None, None))
        exp_res_fn = function(inputs, Y, mode=Mode("py", None, None))

        # Make sure that the expected result graph has an `AdvancedSubtensor`
        assert any(
            isinstance(v.owner.op, AdvancedSubtensor)
            for v in exp_res_fn.maker.fgraph.variables
            if v.owner
        )

        res_val = res_fn(*[i.tag.test_value for i in inputs])
        exp_res_val = exp_res_fn(*[i.tag.test_value for i in inputs])

        assert np.array_equal(res_val, exp_res_val)


@pytest.mark.parametrize("s", [slice(None), slice(None, None, -1)])
def test_local_useless_inc_subtensor(s):
    x = matrix("x")
    y = matrix("y")

    o = set_subtensor(x[:, s], y)

    mode = get_default_mode().including("local_useless_inc_subtensor")

    # Test without shape info (i.e. don't apply the opt)
    f = function([x, y], o, mode=mode)

    topo = f.maker.fgraph.toposort()
    assert len(topo) == 1
    assert isinstance(topo[0].op, IncSubtensor)

    # Test with shape info
    o_shape = set_subtensor(x[:, s], specify_shape(y, x.shape))
    f_shape = function([x, y], o_shape, mode=mode)

    topo = f_shape.maker.fgraph.toposort()
    assert not any(isinstance(n.op, IncSubtensor) for n in topo)

    out = f_shape([[2, 3]], [[3, 4]])
    assert np.array_equal(out, np.asarray([[3, 4]])[::, s])


def test_local_useless_inc_subtensor_increment_zeros():
    r"""Make sure we remove `IncSubtensor`\s that are increments on entire zero arrays."""
    y = matrix("y")

    s = at.zeros((2, 2))[:, :]
    o_shape = inc_subtensor(s, specify_shape(y, s.shape))

    mode = get_default_mode().including("local_useless_inc_subtensor")
    f_shape = function([y], o_shape, mode=mode)

    topo = f_shape.maker.fgraph.toposort()
    assert not any(isinstance(n.op, IncSubtensor) for n in topo)


def test_local_useless_inc_subtensor_no_opt():
    r"""Make sure we don't remove `IncSubtensor`\s that involve slices with steps that skip elements and non-zero increments."""
    x = matrix("x")
    y = matrix("y")

    s = x[:, ::2]
    o_shape = set_subtensor(s, specify_shape(y, s.shape))

    mode = get_default_mode().including("local_useless_inc_subtensor")
    f_shape = function([x, y], o_shape, mode=mode)

    topo = f_shape.maker.fgraph.toposort()
    assert any(isinstance(n.op, IncSubtensor) for n in topo)

    out = f_shape([[2, 3, 6, 7]], [[8, 9]])
    assert np.array_equal(out, np.asarray([[8, 3, 9, 7]]))

    # This is an increment with a non-constant target array
    s = x[:, :]
    o_shape = inc_subtensor(s, specify_shape(y, s.shape))

    f_shape = function([x, y], o_shape, mode=mode)

    topo = f_shape.maker.fgraph.toposort()
    assert any(isinstance(n.op, IncSubtensor) for n in topo)

    # This is an increment with a non-zero target array
    s = at.ones((2, 2))[:, :]
    o_shape = inc_subtensor(s, specify_shape(y, s.shape))

    f_shape = function([y], o_shape, mode=mode)

    topo = f_shape.maker.fgraph.toposort()
    assert any(isinstance(n.op, IncSubtensor) for n in topo)


class TestLocalUselessSubtensor:
    x = matrix("x")
    s = aes.int32("s")
    mode = mode_opt.including(
        "local_useless_subtensor", "local_useless_AdvancedSubtensor1"
    )

    @pytest.mark.parametrize(
        "idx",
        [
            (slice(0, None),),
            (slice(0, None), slice(0, None)),
        ],
    )
    def test_local_useless_subtensor_1(self, idx):
        f = function([self.x], exp(self.x).__getitem__(idx), mode=self.mode)
        prog = f.maker.fgraph.toposort()
        assert prog[0].op == exp
        assert len(prog) == 1

        x_val = np.array([[0, 1, 2], [3, 4, 5]], dtype=aesara.config.floatX)
        idx_val = idx
        exp_res = np.exp(x_val)[idx_val]
        res = f(x_val)
        assert np.allclose(res, exp_res)

    @pytest.mark.parametrize(
        "idx, res",
        [
            ((slice(0, 2),), True),
            ((slice(0, 2), slice(0, None)), True),
            ((slice(0, 2), slice(0, 3)), True),
            ((slice(0, None), slice(0, 3)), True),
            ((slice(0, 3), slice(0, 13)), True),
            ((slice(0, 3), slice(0, 2)), False),
            ((slice(0, 1), slice(0, None)), False),
            ((slice(0, 1), 1), False),
        ],
    )
    def test_local_useless_subtensor_2(self, idx, res):
        x_c = specify_shape(self.x, (2, 3))
        f = function([self.x], exp(x_c).__getitem__(idx), mode=self.mode)
        prog = f.maker.fgraph.toposort()
        if res:
            assert isinstance(prog[0].op, SpecifyShape)
            assert prog[1].op == exp
            assert len(prog) == 2
        else:
            assert any(isinstance(node.op, Subtensor) for node in prog)

        x_val = np.array([[0, 1, 2], [3, 4, 5]], dtype=aesara.config.floatX)
        idx_val = idx
        exp_res = np.exp(x_val)[idx_val]
        res = f(x_val)
        assert np.allclose(res, exp_res)

    @pytest.mark.parametrize(
        "idx_fn, res",
        [
            (lambda x: (slice(0, x.shape[0]),), True),
            (lambda x: (slice(0, x.shape[1]),), False),
            (
                lambda x: (
                    slice(0, x.shape[0]),
                    slice(0, x.shape[1]),
                ),
                True,
            ),
            (
                lambda x: (
                    slice(0, x.shape[0]),
                    slice(0, x.shape[0]),
                ),
                False,
            ),
            (
                lambda x: (
                    slice(0, x.shape[1]),
                    slice(0, x.shape[0]),
                ),
                False,
            ),
            (
                lambda x: (
                    slice(0, x.shape[1]),
                    slice(0, x.shape[1]),
                ),
                False,
            ),
            (lambda x: (slice(0, x.shape[1]), 2), False),
            (
                lambda x: (
                    slice(0, x.shape[1]),
                    slice(x.shape[0] - x.shape[0], x.shape[1]),
                ),
                False,
            ),
            (
                lambda x: (
                    slice(
                        0,
                        at.scalar_from_tensor(x.shape[0])
                        if isinstance(x, Variable)
                        else x.shape[0],
                    ),
                ),
                True,
            ),
        ],
    )
    def test_local_useless_subtensor_3(self, idx_fn, res):
        idx = idx_fn(self.x)
        f = function([self.x], exp(self.x).__getitem__(idx), mode=self.mode)
        prog = f.maker.fgraph.toposort()
        if res:
            assert prog[0].op == exp
            assert len(prog) == 1
        else:
            assert any(isinstance(node.op, Subtensor) for node in prog)

        x_val = np.array([[0, 1, 2], [3, 4, 5]], dtype=aesara.config.floatX)
        idx_val = idx_fn(x_val)
        exp_res = np.exp(x_val)[idx_val]
        res = f(x_val)
        assert np.allclose(res, exp_res)

    @pytest.mark.parametrize(
        "idx_fn, res",
        [
            (lambda x: (slice(0, x.shape[0]), slice(0, 3)), False),
            (lambda x: (slice(0, 3), slice(0, x.shape[1])), False),
        ],
    )
    def test_local_useless_subtensor_4(self, idx_fn, res):
        # Test mix Variable and Constant
        # Currently not supported
        x_c = specify_shape(self.x, (2, 3))
        idx = idx_fn(self.x)
        f = function([self.x], exp(x_c).__getitem__(idx), mode=self.mode)
        prog = f.maker.fgraph.toposort()
        if res:
            assert prog[0].op == exp
            assert len(prog) == 1
        else:
            assert any(isinstance(node.op, Subtensor) for node in prog)

        x_val = np.array([[0, 1, 2], [3, 4, 5]], dtype=aesara.config.floatX)
        idx_val = idx_fn(x_val)
        exp_res = np.exp(x_val)[idx_val]
        res = f(x_val)
        assert np.allclose(res, exp_res)

    @pytest.mark.parametrize(
        "idx_fn, res",
        [
            (lambda s: (slice(0, s),), False),
        ],
    )
    def test_local_useless_subtensor_5(self, idx_fn, res):
        # Test scalar variable
        idx = idx_fn(self.s)
        f = function([self.x, self.s], exp(self.x).__getitem__(idx), mode=mode_opt)

        prog = f.maker.fgraph.toposort()
        if res:
            assert prog[0].op == exp
            assert len(prog) == 1
        else:
            assert any(isinstance(node.op, Subtensor) for node in prog)

        x_val = np.array([[0, 1, 2], [3, 4, 5]], dtype=aesara.config.floatX)
        idx_val = idx_fn(1)
        exp_res = np.exp(x_val)[idx_val]
        res = f(x_val, 1)
        assert np.allclose(res, exp_res)

        idx_val = idx_fn(3)
        exp_res = np.exp(x_val)[idx_val]
        res = f(x_val, 3)
        assert np.allclose(res, exp_res)

    @pytest.mark.parametrize(
        "idx, res",
        [
            ([0, 1], True),
            ([1, 0], False),
            ([0, 0], False),
            ([0, 0, 1], False),
            (at.arange(2), True),
            (at.arange(0, 2), True),
            (at.arange(0, 2, 2), False),
            (at.arange(0, 2, -1), False),
            (at.arange(1, 2), False),
        ],
    )
    def test_local_useless_subtensor_6(self, idx, res):
        # Test AdvancedSubtensor1 case when all rows are selected by a list/vector
        # or ARange op
        x_c = specify_shape(self.x, (2, 3))
        f = function([self.x], exp(x_c).__getitem__(idx), mode=mode_opt)
        prog = f.maker.fgraph.toposort()
        if res:
            assert isinstance(prog[0].op, SpecifyShape)
            assert prog[1].op == exp
            assert len(prog) == 2
        else:
            assert any(isinstance(node.op, AdvancedSubtensor1) for node in prog)

        x_val = np.array([[0, 1, 2], [3, 4, 5]], dtype=aesara.config.floatX)
        idx_val = idx.eval() if isinstance(idx, Variable) else idx
        exp_res = np.exp(x_val)[idx_val]
        res = f(x_val)
        assert np.allclose(res, exp_res)


def test_local_subtensor_remove_broadcastable_index():
    # testing local_subtensor_remove_broadcastable_index optimization
    #
    # tests removing broadcastable dimensions with index 0 or -1,
    # otherwise the optimzation should not be applied

    mode = get_default_mode()
    mode = mode.including("local_subtensor_remove_broadcastable_index")
    x = dmatrix("x")
    y1 = x.dimshuffle(0, "x", 1)
    y2 = x.dimshuffle("x", 1, 0, "x")
    y3 = x.dimshuffle("x", 1, "x", 0, "x")

    # testing for cases that the optimzation should be applied
    z1 = y1[:, 0, :]
    z2 = y1[:, -1, :]
    z3 = y2[0, :, :, -1]
    z4 = y2[0, :, :, 0]
    z5 = y2[-1, :, :, -1]
    z6 = y3[-1, :, 0, :, -1]
    z7 = y3[-1, :, -1, :, -1]
    z8 = y3[0, :, 0, :, 0]
    f = function([x], [z1, z2, z3, z4, z5, z6, z7, z8], mode=mode)
    for elem in f.maker.fgraph.toposort():
        assert not isinstance(
            elem.op,
            (
                Subtensor,
                AdvancedSubtensor,
                AdvancedSubtensor1,
                IncSubtensor,
                AdvancedIncSubtensor,
                AdvancedIncSubtensor1,
            ),
        )
    rng = np.random.default_rng(seed=utt.fetch_seed())
    xn = rng.random((5, 5))
    f(xn)

    # testing for cases that the optimzation should not be applied
    # to verify that other subtensor usage are passed without errors
    w1 = y1[3, 0, :]
    w2 = y1[2:4, -1, :]
    w3 = y2[0, :, 4:, -1]
    w4 = y2[:, :, 0, -1]
    w5 = y2[0, 2:4, :, 0]
    w6 = y2[0, -1, :, -1]
    w7 = y2[-1, 4:, :, -1]
    w8 = y2[-1, :, :3, -1]
    w9 = y2[-1, :, -1, -1]
    w10 = y3[-1, 2, 0, :, -1]
    w11 = y3[-1, 0, -1, :, -1]
    w12 = y3[-1, :, -1, -1, -1]
    w13 = y3[0, 0, 0, :, 0]
    w14 = y3[-1, 2:4, 0, 1:5, -1]
    w15 = y3[-1, 0, -1, 0, -1]
    w16 = y3[0, 2, 0, 4, 0]
    w17 = y3[:, 0, :, 1]
    w18 = y3[0, :, :, 2]
    w19 = y3[:, 2, 0]
    w20 = y3[:, 3]
    f2 = function(
        [x],
        [
            w1,
            w2,
            w3,
            w4,
            w5,
            w6,
            w7,
            w8,
            w9,
            w10,
            w11,
            w12,
            w13,
            w14,
            w15,
            w16,
            w17,
            w18,
            w19,
            w20,
        ],
        mode=mode,
    )
    f2(xn)


class TestSubtensorIncSubtensor:
    @classmethod
    def setup_class(cls):
        cls.rng = np.random.default_rng(utt.fetch_seed())
        cls.mode = get_default_mode().including(
            "local_subtensor_inc_subtensor",
            "local_AdvancedIncSubtensor_to_AdvancedIncSubtensor1",
            "local_replace_AdvancedSubtensor",
        )

    @pytest.mark.parametrize(
        "val, indices, optype",
        [
            (vector(), (iscalar(),), IncSubtensor),
            (vector(), (ivector(),), AdvancedIncSubtensor1),
            (vector(), (ivector(), ivector()), AdvancedIncSubtensor),
        ],
    )
    def test_inplace(self, val, indices, optype):
        x = matrix("x")
        y = set_subtensor((2 * x)[indices], val, inplace=False)
        assert y.owner.op.inplace is False
        f = function(
            [x, val] + list(indices),
            y,
            mode=self.mode.including("inplace"),
        )
        assert isinstance(f.maker.fgraph.outputs[0].owner.op, optype)
        assert f.maker.fgraph.outputs[0].owner.op.inplace is True

    def test_basic(self):
        # basic test
        x = matrix("x")
        i = iscalar("i")
        v = vector("v")
        y = set_subtensor(x[i], v)
        z = y[i]
        f = function([x, i, v], z, mode=self.mode)
        prog = f.maker.fgraph.toposort()
        assert len(prog) == 1
        assert isinstance(prog[0].op, DeepCopyOp)
        # basic test, numerical check
        x_ = np.random.uniform(size=[3, 4]).astype(config.floatX)
        v_ = np.random.uniform(
            size=[
                4,
            ]
        ).astype(config.floatX)
        i_ = 1
        assert np.array_equal(f(x_, i_, v_), v_)

    def test_multiple_idx(self):
        # complicated test
        x = tensor4("x")
        i1 = iscalar("i1")
        i2 = iscalar("i2")
        i3 = iscalar("i3")
        i4 = iscalar("i4")
        v = tensor3("v")
        y = set_subtensor(x[i1, :i2, i3:, ::i4], v)
        z = y[i1, :i2, i3:, ::i4]
        f = function([x, i1, i2, i3, i4, v], z, mode=self.mode)
        prog = f.maker.fgraph.toposort()
        assert len(prog) == 1
        assert isinstance(prog[0].op, DeepCopyOp)
        # complicated test, numerical check
        x_ = np.random.uniform(size=[3, 4, 5, 6]).astype(config.floatX)
        v_ = np.random.uniform(size=[2, 2, 2]).astype(config.floatX)
        i1_, i2_, i3_, i4_ = 1, 2, 3, 4
        assert np.array_equal(f(x_, i1_, i2_, i3_, i4_, v_), v_)

    def test_not_applied(self):
        # case not use this optimization
        x = tensor4("x")
        i1 = iscalar("i1")
        i2 = iscalar("i2")
        i3 = iscalar("i3")
        i4 = iscalar("i4")
        v = tensor3("v")
        y = set_subtensor(x[i1, :i2, i3:, ::i4], v)
        z = y[i1, :i3, i2:, ::i4]
        f = function([x, i1, i2, i3, i4, v], z, mode=self.mode)
        prog = f.maker.fgraph.toposort()
        assert len(prog) != 1
        assert any(isinstance(x.op, IncSubtensor) for x in prog)
        assert any(isinstance(x.op, Subtensor) for x in prog)
        # case not use this optimization, numerical check
        x_ = np.random.uniform(size=[3, 4, 5, 6]).astype(config.floatX)
        v_ = np.random.uniform(size=[2, 2, 2]).astype(config.floatX)
        i1_, i2_, i3_, i4_ = 1, 2, 3, 4
        x_[i1_, :i2_, i3_:, ::i4_] = v_
        assert np.array_equal(f(x_, i1_, i2_, i3_, i4_, v_), x_[i1_, :i3_, i2_:, ::i4_])

    def test_fewer_dims(self):
        # case when v has fewer dimensions
        x = matrix("x")
        i1 = iscalar("i")
        i2 = iscalar("i")
        v = vector("v")
        y = set_subtensor(x[:i1, :i2], v)
        z = y[:i1, :i2]
        f = function([x, i1, i2, v], z, mode=self.mode)
        prog = f.maker.fgraph.toposort()
        assert any(isinstance(x.op, Alloc) for x in prog)
        # case when v is broadcastable, numerical check
        x_ = np.random.uniform(size=[3, 4]).astype(config.floatX)
        v_ = np.random.uniform(
            size=[
                2,
            ]
        ).astype(config.floatX)
        i1_, i2_ = 2, 2
        x_[:i1_, :i2_] = v_
        assert np.array_equal(f(x_, i1_, i2_, v_), x_[:i1_, :i2_])

    def test_broadcasted(self):
        # case when v has the same number of dimensions, some broadcastable
        x = matrix("x")
        i1 = iscalar("i")
        i2 = iscalar("i")
        v = col("v")
        y = set_subtensor(x[:i1, :i2], v)
        z = y[:i1, :i2]
        f = function([x, i1, i2, v], z, mode=self.mode)
        prog = f.maker.fgraph.toposort()
        assert any(isinstance(x.op, Alloc) for x in prog)
        # case when v is broadcastable, numerical check
        x_ = np.random.uniform(size=[3, 4]).astype(config.floatX)
        v_ = np.random.uniform(size=[2, 1]).astype(config.floatX)
        i1_, i2_ = 2, 2
        x_[:i1_, :i2_] = v_
        assert np.array_equal(f(x_, i1_, i2_, v_), x_[:i1_, :i2_])

    def test_different_dtypes(self):
        # Case when the dtype differs
        x = bmatrix("x")
        i = iscalar("i")
        v = vector("v")
        y = set_subtensor(x[i], v)
        z = y[i]
        f = function([x, i, v], z, mode=self.mode)
        prog = f.maker.fgraph.toposort()
        assert len(prog) == 1
        assert prog[0].op == _convert_to_int8
        # basic test, numerical check
        x_ = self.rng.integers(12, size=[3, 4]).astype("int8")
        v_ = np.random.uniform(
            12,
            size=[
                4,
            ],
        ).astype(config.floatX)
        i_ = 1
        assert np.array_equal(f(x_, i_, v_), v_.astype("int8"))


class TestLocalSubtensorMakeVector:
    mode = get_mode("FAST_RUN").including("local_subtensor_make_vector")

    def test_scalar_idx(self):
        x, y, z = lscalars("xyz")
        v = make_vector(x, y, z)
        f = function([x, y, z], v[0], mode=self.mode)

        prog = f.maker.fgraph.toposort()
        assert len(prog) == 1
        assert isinstance(prog[0].op, DeepCopyOp)
        assert f(0, 1, 2) == 0

    def test_idx_symbolic(self):
        x, y, z = iscalars("xyz")
        v = MakeVector("int32")(x, y, z)
        idx = at.as_tensor([0], dtype=np.int64)
        f = function([x, y, z], v[idx], mode=self.mode)

        opt_fgraph = f.maker.fgraph
        assert opt_fgraph.outputs[0].dtype == "int32"
        assert isinstance(opt_fgraph.outputs[0].owner.op, MakeVector)
        assert f(0, 1, 2) == np.array([0], dtype=np.int32)

    def test_slice_idx_start(self):
        x, y, z = iscalars("xyz")
        v = MakeVector("int32")(x, y, z)
        f = function([x, y, z], v[1:], mode=self.mode, on_unused_input="ignore")

        opt_fgraph = f.maker.fgraph
        assert opt_fgraph.outputs[0].dtype == "int32"
        assert isinstance(opt_fgraph.outputs[0].owner.op, MakeVector)
        assert len(opt_fgraph.outputs[0].owner.inputs) == 2
        r = f(0, 1, 2)
        assert r[0] == 1 and r[1] == 2

    def test_slice_idx_stop(self):
        x, y, z = lscalars("xyz")
        v = make_vector(x, y, z)
        f = function([x, y, z], v[:2], mode=self.mode)

        prog = f.maker.fgraph.toposort()
        assert len(prog) == 1
        assert isinstance(prog[0].op, MakeVector)
        assert len(prog[0].inputs) == 2
        r = f(0, 1, 2)
        assert r[0] == 0 and r[1] == 1

    def test_slice_idx_step(self):
        x, y, z = lscalars("xyz")
        v = make_vector(x, y, z)
        f = function([x, y, z], v[::2], mode=self.mode)

        prog = f.maker.fgraph.toposort()
        assert len(prog) == 1
        assert isinstance(prog[0].op, MakeVector)
        assert len(prog[0].inputs) == 2
        r = f(0, 1, 2)
        assert r[0] == 0 and r[1] == 2

    def test_AdvancedSubtensor1_idx(self):
        x, y, z = lscalars("xyz")
        v = make_vector(x, y, z)
        f = function([x, y, z], v[[0, 2]], mode=self.mode)

        prog = f.maker.fgraph.toposort()
        assert len(prog) == 1
        assert isinstance(prog[0].op, MakeVector)
        assert len(prog[0].inputs) == 2
        r = f(0, 1, 2)
        assert r[0] == 0 and r[1] == 2

    def test_MakeVector_idx(self):
        x, y, z, q = lscalars("xyzq")
        v = make_vector(x, y, z)
        q = make_vector(0, 2)
        f = function([x, y, z], v[q], mode=self.mode)

        prog = f.maker.fgraph.toposort()
        assert len(prog) == 1
        assert isinstance(prog[0].op, MakeVector)
        assert len(prog[0].inputs) == 2
        r = f(0, 1, 2)
        assert r[0] == 0 and r[1] == 2

    def test_stack_trace(self):
        x, y, z = lscalars("xyz")
        v = make_vector(x, y, z)

        mode = get_default_mode().including("local_subtensor_make_vector")

        # list of subtensor cases, where local_subtensor_make_vector
        # inserts a new MakeVector node
        v_subtensors = [v[:2], v[::2], v[[0, 2]]]

        for v_subtensor in v_subtensors:
            f = function([x, y, z], v_subtensor, mode=mode)
            assert check_stack_trace(f, ops_to_check="all")


class TestLocalSubtensorLift:
    def test_basic(self):
        # basic test that the Op works
        x = matrix("x")
        f = function([x], exp(x)[0], mode=mode_opt)

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f, ops_to_check="all")

        prog = f.maker.fgraph.toposort()
        assert isinstance(prog[0].op, Subtensor)  # first subtensor
        assert prog[1].op == exp
        assert len(prog) == 2
        f([[0, 1], [2, 3]])  # let debugmode test something

    def test_basic_1(self):
        # as test0, but we reuse the output of the elemwise
        # So we should not lift the subtensor
        x = matrix("x")
        f = function([x], [exp(x)[0], exp(x)], mode=mode_opt)

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f, ops_to_check=[Subtensor, Elemwise])

        prog = f.maker.fgraph.toposort()
        assert prog[0].op == exp
        assert isinstance(prog[1].op, Subtensor)  # first subtensor
        assert isinstance(prog[2].op, DeepCopyOp)
        assert len(prog) == 3
        f([[0, 1], [2, 3]])  # let debugmode test something

    def test_basic_2(self):
        # basic test that the optimization work with scalar broadcasted
        x = matrix("x")
        y = scalar("y")
        z = matrix("z")
        f = function([x, y, z], exp(x + y + z)[0], mode=mode_opt)

        prog = f.maker.fgraph.toposort()
        assert isinstance(prog[0].op, Subtensor)
        assert isinstance(prog[1].op, DimShuffle)
        assert isinstance(prog[2].op, Subtensor)
        assert isinstance(prog[3].op.scalar_op, aes.Composite)  # Composite{add,add}
        assert len(prog) == 4

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f, ops_to_check=[Subtensor])

        # let debugmode test something
        f([[0, 1], [2, 3]], 4, [[4, 5], [6, 7]])

    def test_basic_3(self):
        # as 1, but take a slice
        x = matrix("x")
        y = scalar("y")
        z = matrix("z")
        f = function([x, y, z], exp(x + y + z)[0:2], mode=mode_opt)

        prog = f.maker.fgraph.toposort()
        assert isinstance(prog[0].op, Subtensor)
        assert isinstance(prog[1].op, DimShuffle)
        assert isinstance(prog[2].op, Subtensor)
        assert isinstance(prog[3].op.scalar_op, aes.Composite)  # Composite{add,add}
        assert len(prog) == 4

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f, ops_to_check=[Subtensor])

        # let debugmode test something
        f([[0, 1], [2, 3]], 4, [[4, 5], [6, 7]])

    def test_basic_4(self):
        # basic test that the optimization does work with broadcasting
        # for unary elemwise.
        y = vector("y")
        f = function([y], exp(y.dimshuffle(0, "x"))[0], mode=mode_opt)

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f, ops_to_check="all")

        prog = f.maker.fgraph.toposort()
        assert isinstance(prog[0].op, DimShuffle)
        assert isinstance(prog[1].op, Subtensor)
        assert prog[2].op == exp
        assert len(prog) == 3
        f([4, 5])  # let debugmode test something

    @utt.assertFailure_fast
    def test_basic_5(self):
        # basic test that the optimization doesn't work with broadcasting
        # ... It *could* be extended to,
        # ... but right now it doesn't, so it shouldn't try.
        x = matrix("x")
        y = vector("y")
        f = function([x, y], exp(x + y)[0], mode=mode_opt)

        # Opt doesn't apply, so no need for check_stack_trace
        # assert check_stack_trace(f, ops_to_check='all')

        prog = f.maker.fgraph.toposort()
        assert isinstance(prog[0].op, DimShuffle)
        assert prog[1].op == add
        assert isinstance(prog[2].op, Subtensor)  # first subtensor
        assert prog[3].op == inplace.exp_inplace
        assert len(prog) == 4
        f([[0, 1], [2, 3]], [4, 5])  # let debugmode test something

    def test_basic_6(self):
        # test that we don't lift when we reuse the output of the
        # elemwise for other computation.
        x = matrix("x")
        y = vector("y")
        f = function([x, y], [exp(x + y)[0], exp(x + y) + x], mode=mode_opt)

        # Opt doesn't apply, so no need for check_stack_trace
        # assert check_stack_trace(f, ops_to_check=Subtensor)

        prog = f.maker.fgraph.toposort()
        assert isinstance(prog[0].op, DimShuffle)
        assert isinstance(prog[1].op.scalar_op, aes.Composite)  # Composite{add,exp}
        assert prog[2].op == add or prog[3].op == add
        # first subtensor
        assert isinstance(prog[2].op, Subtensor) or isinstance(prog[3].op, Subtensor)
        assert len(prog) == 4
        f([[0, 1], [2, 3]], [4, 5])  # let debugmode test something

    def test_basic_7(self):
        # basic test that the optimization works with a scalar as input,
        # and a scalar as output (no broadcasting of the scalar needed).
        # The optimization used to fail and display an ERROR message.

        x = vector("x")
        y = scalar("y")
        f = function([x, y], exp(x + y)[0], mode=mode_opt)

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f, ops_to_check=Subtensor)

        prog = f.maker.fgraph.toposort()
        assert isinstance(prog[0].op, Subtensor)
        # Composite{add,exp}
        assert isinstance(prog[1].op.scalar_op, aes.Composite)
        assert len(prog) == 2
        f([1, 2, 3], 4)  # let debugmode test something

    def test_basic_8(self):
        # Test that Subtensor(Unbroadcast(x)) gets optimized into
        # Unbroadcast(Subtensor(x)).

        # test basic case
        x = row("x")
        xval = np.random.random((1, 10)).astype(config.floatX)
        assert x.broadcastable == (True, False)
        newx = Unbroadcast(0)(x)
        assert newx.broadcastable == (False, False)

        f1 = function([x], newx[:2, :5], mode=mode_opt)
        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f1, ops_to_check=[Subtensor, Unbroadcast])
        prog = f1.maker.fgraph.toposort()
        assert isinstance(prog[0].op, Subtensor)
        assert isinstance(prog[1].op, Unbroadcast)
        assert (f1(xval) == xval[:2, :5]).all()

        # corner case 1: Unbroadcast changes dims which are dropped through subtensor
        y = tensor("float64", shape=(1, 10, 1, 3), name="x")
        yval = np.random.random((1, 10, 1, 3)).astype(config.floatX)
        assert y.broadcastable == (True, False, True, False)
        newy = Unbroadcast(0, 2)(y)
        assert newy.broadcastable == (False, False, False, False)

        f2 = function([y], newy[:, 3, 0, :], mode=mode_opt)
        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f2, ops_to_check=[Subtensor, Unbroadcast])
        prog = f2.maker.fgraph.toposort()
        assert isinstance(prog[0].op, Subtensor)
        assert isinstance(prog[1].op, Unbroadcast)
        assert (f2(yval) == yval[:, 3, 0, :]).all()

        # corner case 2: subtensor idx_list is shorter than resulting broadcast pattern
        f3 = function([y], newy[:, 3, 0], mode=mode_opt)
        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f3, ops_to_check=[Subtensor, Unbroadcast])
        prog = f3.maker.fgraph.toposort()
        assert isinstance(prog[0].op, Subtensor)
        assert isinstance(prog[1].op, Unbroadcast)
        assert (f3(yval) == yval[:, 3, 0]).all()

        # corner case 3: subtensor idx_list is shorter than Unbroadcast.axis
        z = tensor("float64", shape=(4, 10, 3, 1), name="x")
        zval = np.random.random((4, 10, 3, 1)).astype(config.floatX)
        assert z.broadcastable == (False, False, False, True)
        newz = Unbroadcast(3)(z)
        assert newz.broadcastable == (False, False, False, False)

        f4 = function([z], newz[:, 3, 0], mode=mode_opt)
        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f4, ops_to_check=[Subtensor, Unbroadcast])
        prog = f4.maker.fgraph.toposort()
        assert isinstance(prog[0].op, Subtensor)
        assert isinstance(prog[1].op, Unbroadcast)
        assert (f4(zval) == zval[:, 3, 0]).all()


class TestLocalSubtensorMerge:
    def setup_method(self):
        self.x_shapes = [(2, 2), (5, 3), (4, 1), (1, 2), (0, 2), (2, 0), (1, 0), (0, 0)]
        self.rng = np.random.default_rng(seed=utt.fetch_seed())

    def test_const(self):
        # var[const::][-1] -> var[-1]
        x = matrix("x")
        for idx in range(-7, 6):
            f = function([x], x[idx::][-1], mode=mode_opt)
            g = function(
                [x], x[idx::][-1], mode=mode_opt.excluding("local_subtensor_merge")
            )

            # Check stacktrace was copied over correctly after opt was applied
            assert check_stack_trace(f, ops_to_check=Subtensor)

            topo = f.maker.fgraph.toposort()
            assert len([t for t in topo if isinstance(t.op, Subtensor)]) == 1
            assert isinstance(topo[-1].op, DeepCopyOp)

            for x_s in self.x_shapes:
                x_val = self.rng.uniform(size=x_s).astype(config.floatX)

                if idx < x_s[0] and x_s[0] > 0:
                    # The first subtensor is non-empty, so it makes sense
                    f(x_val)  # let debugmode test something
                else:
                    # A non-empty subtensor of an empty one should be
                    # an IndexError
                    with pytest.raises(IndexError):
                        f(x_val)
                    with pytest.raises(IndexError):
                        g(x_val)

    def test_scalar(self):
        # var[int::][-1] -> var[-1]
        x = matrix("x")
        y = iscalar("y")
        f = function([x, y], x[y::][-1], mode=mode_opt)
        g = function(
            [x, y], x[y::][-1], mode=mode_opt.excluding("local_subtensor_merge")
        )

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f, ops_to_check=Subtensor)

        topo = f.maker.fgraph.toposort()
        assert len([t for t in topo if isinstance(t.op, Subtensor)]) == 1
        assert isinstance(topo[-1].op, DeepCopyOp)

        for x_s in self.x_shapes:
            x_val = self.rng.uniform(size=x_s).astype(config.floatX)

            for idx in range(-9, 8):
                if (idx < x_s[0]) and (x_s[0] > 0):
                    # The first subtensor is non-empty
                    f(x_val, idx)  # let debugmode test something
                else:
                    with pytest.raises(IndexError):
                        f(x_val, idx)
                    with pytest.raises(IndexError):
                        g(x_val, idx)

    @pytest.mark.slow
    def test_const2(self):
        # var[::-1][const] -> var[-1]
        x = matrix("x")
        for idx in range(-8, 7):
            f = function([x], x[::-1][idx], mode=mode_opt)
            g = function(
                [x], x[::-1][idx], mode=mode_opt.excluding("local_subtensor_merge")
            )

            # Check stacktrace was copied over correctly after opt was applied
            assert check_stack_trace(f, ops_to_check=Subtensor)

            topo = f.maker.fgraph.toposort()
            assert len([t for t in topo if isinstance(t.op, Subtensor)]) == 1
            assert isinstance(topo[-1].op, DeepCopyOp)

            for x_s in self.x_shapes:
                x_val = self.rng.uniform(size=x_s).astype(config.floatX)
                if (idx < x_s[0]) and (idx >= -x_s[0]):
                    # The first subtensor is non-empty, so it makes sense
                    f(x_val)  # let debugmode test something
                else:
                    # A non-empty subtensor of an empty one should be
                    # an IndexError
                    with pytest.raises(IndexError):
                        f(x_val)
                    with pytest.raises(IndexError):
                        g(x_val)

    def test_scalar2(self):
        # var[::-1][int] -> var[-1]
        x = matrix("x")
        y = iscalar("y")
        f = function([x, y], x[::-1][y], mode=mode_opt)
        g = function(
            [x, y], x[::-1][y], mode=mode_opt.excluding("local_subtensor_merge")
        )

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f, ops_to_check=Subtensor)

        topo = f.maker.fgraph.toposort()
        assert len([t for t in topo if isinstance(t.op, Subtensor)]) == 1
        assert isinstance(topo[-1].op, DeepCopyOp)

        for x_s in self.x_shapes:
            x_val = self.rng.uniform(size=x_s).astype(config.floatX)

            for idx in range(-x_s[0], x_s[0]):
                f(x_val, idx)  # let debugmode test something
            for idx in list(range(x_s[0], 9)) + list(range(-9, -x_s[0])):
                with pytest.raises(IndexError):
                    f(x_val, idx)
                with pytest.raises(IndexError):
                    g(x_val, idx)

    def test_const3(self):
        # var[::-1][:const] -> var[-1]
        x = matrix("x")
        for idx in range(-9, 8):
            f = function([x], x[::-1][:idx], mode=mode_opt)

            # Check stacktrace was copied over correctly after opt was applied
            assert check_stack_trace(f, ops_to_check=Subtensor)

            topo = f.maker.fgraph.toposort()
            assert len([t for t in topo if isinstance(t.op, Subtensor)]) == 1
            assert isinstance(topo[-1].op, DeepCopyOp)

            for x_s in self.x_shapes:
                x_val = self.rng.uniform(size=x_s).astype(config.floatX)
                f(x_val)  # let debugmode test something

    def test_scalar3(self):
        # var[::-1][:int] -> var[-1]
        x = matrix("x")
        y = iscalar("y")
        f = function([x, y], x[::-1][:y], mode=mode_opt)

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f, ops_to_check=Subtensor)

        topo = f.maker.fgraph.toposort()
        assert len([t for t in topo if isinstance(t.op, Subtensor)]) == 1
        assert isinstance(topo[-1].op, DeepCopyOp)

        for x_s in self.x_shapes:
            x_val = self.rng.uniform(size=x_s).astype(config.floatX)
            for idx in range(-7, 7):
                f(x_val, idx)  # let debugmode test something

    def test_const4(self):
        # var[const1::][:const2]
        x = matrix("x")
        for idx1 in range(-7, 7):
            for idx2 in range(-7, 7):
                f = function([x], x[idx1:][:idx2], mode=mode_opt)

                # Check stacktrace was copied over correctly after opt was applied
                assert check_stack_trace(f, ops_to_check=Subtensor)

                topo = f.maker.fgraph.toposort()
                assert len([t for t in topo if isinstance(t.op, Subtensor)]) == 1
                assert isinstance(topo[-1].op, DeepCopyOp)

                for x_s in self.x_shapes:
                    x_val = self.rng.uniform(size=x_s).astype(config.floatX)
                    f(x_val)  # let debugmode test something

    def test_scalar4(self):
        # var[int1:][:int2]
        x = matrix("x")
        y = iscalar("y")
        z = iscalar("y")
        f = function([x, y, z], x[y:][:z], mode=mode_opt)

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f, ops_to_check=Subtensor)

        topo = f.maker.fgraph.toposort()
        assert len([t for t in topo if isinstance(t.op, Subtensor)]) == 1
        assert isinstance(topo[-1].op, DeepCopyOp)

        for x_s in self.x_shapes:
            x_val = self.rng.uniform(size=x_s).astype(config.floatX)
            for idx1 in range(-11, 11):
                for idx2 in range(-11, 11):
                    f(x_val, idx1, idx2)  # let debugmode test something

    def test_const_general(self):
        # Some cases of merge: shape, (start, stop, step) of first,
        # (start, stop, step) of second subtensor
        cases = [
            ((2, 3), (None, None, None), (None, None, -1)),
            ((12, 1), (None, None, -4), (None, None, 1)),
            ((5, 3), (1, 4, 2), (None, None, -1)),
        ]
        x = matrix("x")

        for s, sl1, sl2 in cases:
            z = x[slice(*sl1)][slice(*sl2)]
            f = function([x], z, mode=mode_opt)

            # Check stacktrace was copied over correctly after opt was applied
            assert check_stack_trace(f, ops_to_check=Subtensor)

            x_val = self.rng.uniform(size=s).astype(config.floatX)
            f(x_val)

    def test_scalar5(self):
        # General case with two real slices
        # var[b1:e1:s1][b2:e2:s2]
        x = matrix("x")
        b1 = iscalar("b1")
        e1 = iscalar("e1")
        s1 = iscalar("s1")
        b2 = iscalar("b2")
        e2 = iscalar("e2")
        s2 = iscalar("s2")
        f = function([x, b1, e1, s1, b2, e2, s2], x[b1:e1:s1][b2:e2:s2], mode=mode_opt)

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f, ops_to_check=Subtensor)

        topo = f.maker.fgraph.toposort()
        assert len([t for t in topo if isinstance(t.op, Subtensor)]) == 1
        assert isinstance(topo[-1].op, DeepCopyOp)

        b1r = self.rng.permutation(list(range(-8, 8)))[:2]
        e1r = self.rng.permutation(list(range(-8, 8)))[:2]
        b2r = self.rng.permutation(list(range(-8, 8)))[:2]
        e2r = self.rng.permutation(list(range(-8, 8)))[:2]

        s1r = self.rng.permutation([-7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7])[
            :2
        ]
        s2r = self.rng.permutation([-7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7])[
            :2
        ]

        for x_s in self.x_shapes:
            x_val = self.rng.uniform(size=x_s).astype(config.floatX)
            for b1 in b1r:
                for e1 in e1r:
                    for s1 in s1r:
                        for b2 in b2r:
                            for e2 in e2r:
                                for s2 in s2r:
                                    f(x_val, b1, e1, s1, b2, e2, s2)

    def test_const5(self):
        # Bug reported by Razvan
        data = np.asarray(np.arange(8), dtype=config.floatX)
        x = vector("x")
        y = x[7:1:-1]
        t = shared(np.int64(0))

        fun = function([x], y[t])

        val = fun(data)
        assert val == data[7:1:-1][0]

    def test_const6(self):
        # Bug reported by Graham
        data = self.rng.uniform(size=(8, 8, 8)).astype(config.floatX)
        x = tensor3("x")

        # test 1)
        y = x[3:6, 2:6, 1:7][1]
        fun = function([x], y)
        val = fun(data)
        assert np.all(val == data[3:6, 2:6, 1:7][1])
        assert (
            len([n for n in fun.maker.fgraph.toposort() if isinstance(n.op, Subtensor)])
            == 1
        )

        # test 2)
        y = x[2, 3][1]
        fun = function([x], y)
        val = fun(data)
        assert np.all(val == data[2, 3][1])
        assert (
            len([n for n in fun.maker.fgraph.toposort() if isinstance(n.op, Subtensor)])
            == 1
        )

        # test 3)
        y = x[3:6, 2, 1:7][1]
        fun = function([x], y)
        val = fun(data)
        assert np.all(val == data[3:6, 2, 1:7][1])
        assert (
            len([n for n in fun.maker.fgraph.toposort() if isinstance(n.op, Subtensor)])
            == 1
        )

    def test_scalar6(self):
        # General case with one slice and one index
        # var[b:e:s][i]
        x = matrix("x")
        b = iscalar("b")
        e = iscalar("e")
        s = iscalar("s")
        i = iscalar("i")
        f = function([x, b, e, s, i], x[b:e:s][i], mode=mode_opt)

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f, ops_to_check=Subtensor)

        topo = f.maker.fgraph.toposort()
        assert len([t for t in topo if isinstance(t.op, Subtensor)]) == 1
        assert isinstance(topo[-1].op, DeepCopyOp)

        b_r = self.rng.permutation(list(range(-4, 4)))[:3]
        e_r = self.rng.permutation(list(range(-4, 4)))[:3]
        i_r = self.rng.permutation(list(range(-4, 4)))[:3]

        s_r = self.rng.permutation([-3, -2, -1, 1, 2, 3])[:3]

        for x_s in self.x_shapes:
            n_index_err = 0
            n_ok = 0
            x_val = self.rng.uniform(size=x_s).astype(config.floatX)
            for b_v in b_r:
                for e_v in e_r:
                    for s_v in s_r:
                        for i_v in i_r:
                            # The index could be out of bounds
                            # In that case, an Exception should be raised,
                            # otherwise, we let DebugMode check f
                            try:
                                x_val[b_v:e_v:s_v][i_v]
                            except IndexError:
                                n_index_err += 1
                                with pytest.raises(IndexError):
                                    f(x_val, b_v, e_v, s_v, i_v)
                            else:
                                # Executed if the "try" clause did not raise
                                # any exception
                                n_ok += 1
                                f(x_val, b_v, e_v, s_v, i_v)

    @pytest.mark.slow
    def test_none_slice(self):
        # Test case of two slices, var[b1:e1:s1][b2:e2:s2]
        # where any of the b, e, and s can be None
        x = matrix("x")
        b1 = iscalar("b1")
        e1 = iscalar("e1")
        s1 = iscalar("s1")
        b2 = iscalar("b2")
        e2 = iscalar("e2")
        s2 = iscalar("s2")

        # Generate all possible lists of positions for None in those 6 slots
        # A 1 indicates None is present, 0 that there is an Aesara scalar.
        none_positions = np.ndindex(2, 2, 2, 2, 2, 2)

        # Ranges to be used when not None
        b1r = self.rng.permutation(list(range(-4, 4)))[:]
        e1r = self.rng.permutation(list(range(-4, 4)))[:]
        b2r = self.rng.permutation(list(range(-4, 4)))[:]
        e2r = self.rng.permutation(list(range(-4, 4)))[:]
        s1r = self.rng.permutation([-4, -3, -2, -1, 1, 2, 3, 4])[:]
        s2r = self.rng.permutation([-4, -3, -2, -1, 1, 2, 3, 4])[:]

        scalar_vars = [b1, e1, s1, b2, e2, s2]
        scalar_ranges = [b1r, e1r, s1r, b2r, e2r, s2r]

        # For each case, we will build a graph, function, and list of values
        # Then, we test it on each input shape.
        for none_pos in none_positions:
            slice_inputs = []
            input_vars = []
            values = []
            if sum(none_pos) == 0:
                # Those case are already tested in test_scalar4
                continue

            for i, none_i in enumerate(none_pos):
                if none_i:
                    slice_inputs.append(None)
                else:
                    slice_inputs.append(scalar_vars[i])
                    input_vars.append(scalar_vars[i])
                    values.append(scalar_ranges[i])

            slice1 = slice(*slice_inputs[:3])
            slice2 = slice(*slice_inputs[3:])
            sub_x = x[slice1][slice2]
            f = function([x] + input_vars, sub_x, mode=mode_opt)

            # Check stacktrace was copied over correctly after opt was applied
            # for some cases, the optimization may remove all Subtensors,
            # which is why we pass "bug_print='ignore'".
            assert check_stack_trace(f, ops_to_check=Subtensor, bug_print="ignore")

            topo = f.maker.fgraph.toposort()
            assert len([t for t in topo if isinstance(t.op, Subtensor)]) <= 1
            assert isinstance(topo[-1].op, DeepCopyOp)

            for x_s in self.x_shapes:
                x_val = self.rng.uniform(size=x_s).astype(config.floatX)
                for i_val in zip(*values):
                    f(x_val, *i_val)

    def test_none_index(self):
        # Test the general case of indexing into a subvector,
        # like x[b:e:s][i], where any of b, e, and s can be None
        x = matrix("x")
        b = iscalar("b")
        e = iscalar("e")
        s = iscalar("s")
        i = iscalar("i")

        # Generate all possible lists of positions for None in those 6 slots
        # A 1 indicates None is present, 0 that there is an Aesara scalar.
        # The last index (i) is never None
        none_positions = np.ndindex(2, 2, 2, 1)

        # Ranges to be used when not None
        b_r = self.rng.permutation(list(range(-4, 4)))[:]
        e_r = self.rng.permutation(list(range(-4, 4)))[:]
        i_r = self.rng.permutation(list(range(-4, 4)))[:]
        s_r = self.rng.permutation([-4, -3, -2, -1, 1, 2, 3, 4])[:]

        scalar_vars = [b, e, s, i]
        scalar_ranges = [b_r, e_r, s_r, i_r]

        # For each case, we will build a graph, function, and list of values
        # Then, we test it on each input shape.
        for none_pos in none_positions:
            slice_inputs = []
            input_vars = []
            values = []
            if sum(none_pos) == 0:
                # Those case are already tested in test_scalar6
                continue

            for j, none_j in enumerate(none_pos):
                if none_j:
                    slice_inputs.append(None)

                else:
                    slice_inputs.append(scalar_vars[j])
                    input_vars.append(scalar_vars[j])
                    values.append(scalar_ranges[j])

            symbol_slice = slice(*slice_inputs[:3])
            sub_x = x[symbol_slice][i]
            f = function([x] + input_vars, sub_x, mode=mode_opt)

            # Check stacktrace was copied over correctly after opt was applied
            assert check_stack_trace(f, ops_to_check=Subtensor)

            topo = f.maker.fgraph.toposort()
            assert len([t for t in topo if isinstance(t.op, Subtensor)]) <= 1
            assert isinstance(topo[-1].op, DeepCopyOp)

            for x_s in self.x_shapes:
                x_val = self.rng.uniform(size=x_s).astype(config.floatX)
                for i_val in zip(*values):
                    # The index could be out of bounds
                    # In that case, an Exception should be raised,
                    # otherwise, we let DebugMode check f
                    # For that, we need to create a numerical slice.
                    i_val_idx = 0
                    num_slice_inputs = []
                    for none_j in none_pos:
                        if none_j:
                            num_slice_inputs.append(None)
                        else:
                            num_slice_inputs.append(i_val[i_val_idx])
                            i_val_idx += 1
                    num_slice = slice(*num_slice_inputs[:3])
                    num_i = num_slice_inputs[3]

                    try:
                        x_val[num_slice][num_i]
                    except IndexError:
                        with pytest.raises(IndexError):
                            f(x_val, *i_val)
                    else:
                        # Executed if the "try" clause did not raise
                        # any exception
                        f(x_val, *i_val)


class TestLocalAdvSub1AdvIncSub1:
    def setup_method(self):
        mode = get_default_mode()
        self.mode = mode.including(
            "local_replace_AdvancedSubtensor",
            "local_AdvancedIncSubtensor_to_AdvancedIncSubtensor1",
            "local_adv_sub1_adv_inc_sub1",
        ).excluding("fusion")
        self.mode_no_assert = self.mode.including("local_remove_all_assert")

    def test_basic(self):
        for dtype1, dtype2 in [
            ("float32", "float32"),
            ("float32", "float64"),
            ("float64", "float32"),
            ("float64", "float64"),
        ]:
            x = matrix(dtype=dtype1)
            y = matrix(dtype=dtype2)
            idx = ivector()

            dx = np.random.random((4, 5)).astype(dtype1)
            dy = np.random.random((2, 5)).astype(dtype2)
            # Duplicate the last row of dy
            dy = np.vstack([dy, dy[-1:]])
            # Use the same index twice, with the same corresponding value.
            # That makes set_subtensor well-defined, and tests
            # duplication for inc_subtensor.
            didx = np.asarray([1, 3, 3], "int32")

            # set_subtensor
            inc = set_subtensor(x[idx], y)
            o = inc[idx]
            f = function([x, y, idx], o, self.mode_no_assert)

            res = f(dx, dy, didx)
            utt.assert_allclose(dy, res)
            topo = f.maker.fgraph.toposort()
            assert len(topo) == 1
            assert isinstance(topo[0].op, (DeepCopyOp, Elemwise))

            # inc_subtensor(data[idx], y)
            inc = inc_subtensor(x[idx], y)
            o = inc[idx]
            f = function([x, y, idx], o, self.mode_no_assert)

            res = f(dx, dy, didx)
            _dx = dx.copy()
            np.add.at(_dx, didx, dy)
            utt.assert_allclose(_dx[didx], res)
            topo = f.maker.fgraph.toposort()
            len(topo) == 2

            # inc_subtensor(0[idx], y)
            inc = inc_subtensor(x.zeros_like()[idx], y)
            o = inc[idx]
            f = function([x, y, idx], o, self.mode_no_assert)

            res = f(dx, dy, didx)
            utt.assert_allclose(np.vstack([dy[0], 2 * dy[1], 2 * dy[2]]), res)

    def test_assert(self):
        x = matrix("x")
        y = matrix("y")
        idx = ivector()

        dx = np.random.random((4, 5)).astype(config.floatX)
        dy = np.random.random((2, 5)).astype(config.floatX)

        # set_subtensor
        inc = set_subtensor(x[idx], y)
        o = inc[idx]
        f = function([x, y, idx], o, self.mode)
        # test wrong index
        for i in [dx.shape[0], -dx.shape[0] - 1]:
            with pytest.raises((AssertionError, IndexError)):
                f(dx, dy, [i, i])
        # test wrong shape
        with pytest.raises((AssertionError, IndexError)):
            f(dx, dy, [1])

    def test_stack_trace(self):
        x = matrix("x")
        # test cases with y.dtype
        # - equal to x.dtype
        # - different from x.dtype (to trigger the cast in
        #   local_adv_sub1_adv_inc_sub1)
        ys = [matrix("y"), dmatrix("y")]
        idx = ivector()

        # set_subtensor and then subtensor with both ys
        incs = [set_subtensor(x[idx], y) for y in ys]
        outs = [inc[idx] for inc in incs]

        for y, out in zip(ys, outs):
            f = function([x, y, idx], out, self.mode)
            assert check_stack_trace(f, ops_to_check=(Assert, aes.Cast))


class TestSubtensorAllocRewrites:
    def setup_method(self):
        mode = get_default_mode()
        self.mode = mode.including(
            "local_incsubtensor_of_zeros",
            "local_setsubtensor_of_constants",
            "local_0_dot_x",
        )

    def test_setsubtensor_allocs0(self):
        x = matrix()
        y = matrix()
        x0 = at.zeros_like(x)
        y0 = at.zeros_like(y)
        z = set_subtensor(x0[:4], y0)
        f = function([x, y], z, mode=self.mode)
        assert all(
            not isinstance(n.op, IncSubtensor) for n in f.maker.fgraph.toposort()
        )

    def test_setsubtensor_allocs1(self):
        y = matrix()
        x0 = at.constant(np.asarray(np.zeros((4, 4)), dtype=config.floatX))
        y0 = at.zeros_like(y)
        z = set_subtensor(x0[:4], y0)
        f = function([y], z, mode=self.mode)
        assert all(
            not isinstance(n.op, IncSubtensor) for n in f.maker.fgraph.toposort()
        )

    def test_setsubtensor_allocs1t(self):
        y = matrix()
        x0 = at.constant(np.asarray(np.zeros((4, 4)), dtype=config.floatX))
        y0 = at.zeros_like(y)
        z = set_subtensor(x0[:4], y0.T)
        f = function([y], z, mode=mode_opt)
        assert all(
            not isinstance(n.op, IncSubtensor) for n in f.maker.fgraph.toposort()
        )

    def test_setsubtensor_allocs2(self):
        x = matrix()
        y0 = at.constant(np.asarray(np.zeros_like((4, 4)), dtype=config.floatX))
        x0 = at.zeros_like(x)
        z = set_subtensor(x0[:4], y0)
        f = function([x], z, mode=self.mode)
        assert all(
            not isinstance(n.op, IncSubtensor) for n in f.maker.fgraph.toposort()
        )

    def test_incsubtensor_allocs0(self):
        x = matrix()
        y = matrix()
        y0 = at.zeros_like(y)
        z = inc_subtensor(x[:4], y0)
        f = function([x, y], z, mode=self.mode)
        assert all(
            not isinstance(n.op, IncSubtensor) for n in f.maker.fgraph.toposort()
        )

    def test_incsubtensor_allocs0t(self):
        x = matrix()
        y = matrix()
        y0 = at.zeros_like(y)
        z = inc_subtensor(x[:4], y0.T)
        f = function([x, y], z, mode=mode_opt)
        assert all(
            not isinstance(n.op, IncSubtensor) for n in f.maker.fgraph.toposort()
        )

    def test_incsubtensor_allocs1(self):
        x = matrix()
        y0 = at.constant(np.asarray(np.zeros_like((4, 4)), dtype=config.floatX))
        z = inc_subtensor(x[:4], y0)
        f = function([x], z, mode=self.mode)
        assert all(
            not isinstance(n.op, IncSubtensor) for n in f.maker.fgraph.toposort()
        )

    def test_incsubtensor_x_zeros(self):
        x = at.constant(np.asarray(np.zeros((4, 4)), dtype=config.floatX))
        y = matrix()
        z = inc_subtensor(x[:4], y)
        f = function([y], z)
        inc_nodes = [
            n for n in f.maker.fgraph.toposort() if isinstance(n.op, IncSubtensor)
        ]

        assert len(inc_nodes) == 1
        node_is_set_instead_of_inc = inc_nodes[0].op.set_instead_of_inc
        assert node_is_set_instead_of_inc
        test_X = np.random.random((4, 4)).astype(config.floatX)
        utt.assert_allclose(f(test_X), test_X)

        # also check the flag doesn't get set if first input is not zeros:
        not_all_zeros = np.zeros((4, 4))
        not_all_zeros[1, 0] = 0.001
        x = at.constant(np.asarray(not_all_zeros, dtype=config.floatX))
        y = matrix()
        z = inc_subtensor(x[:4], y)
        f = function([y], z)
        inc_nodes = [
            n for n in f.maker.fgraph.toposort() if isinstance(n.op, IncSubtensor)
        ]
        assert len(inc_nodes) == 1
        assert inc_nodes[0].op.set_instead_of_inc is False
        test_X = np.random.random((4, 4)).astype(config.floatX)
        utt.assert_allclose(f(test_X), test_X + not_all_zeros)

    def test_advancedincsubtensor1_allocs0(self):
        x = matrix()
        y = matrix()
        y0 = at.zeros_like(y)
        z = inc_subtensor(x[[0, 1, 2, 3]], y0)
        f = function([x, y], z, mode=self.mode)
        assert all(
            not isinstance(n.op, AdvancedIncSubtensor1)
            for n in f.maker.fgraph.toposort()
        )

    def test_advancedincsubtensor1_allocs0t(self):
        x = matrix()
        y = matrix()
        y0 = at.zeros_like(y)
        z = inc_subtensor(x[[0, 1, 2, 3]], y0.T)
        f = function([x, y], z, mode=mode_opt)
        assert all(
            not isinstance(n.op, AdvancedIncSubtensor1)
            for n in f.maker.fgraph.toposort()
        )

    def test_advancedincsubtensor1_allocs1(self):
        x = matrix()
        y0 = at.constant(np.asarray(np.zeros_like((4, 4)), dtype=config.floatX))
        z = inc_subtensor(x[[0, 1, 2, 3]], y0)
        f = function([x], z, mode=self.mode)
        assert all(
            not isinstance(n.op, AdvancedIncSubtensor1)
            for n in f.maker.fgraph.toposort()
        )

    def test_advancedincsubtensor_allocs0(self):
        x = matrix()
        y = matrix()
        y0 = at.zeros_like(y)
        z = inc_subtensor(x[[[0, 0], [1, 1]], [[0, 1], [0, 1]]], y0)
        f = function([x, y], z, mode=self.mode)
        assert all(
            not isinstance(n.op, AdvancedIncSubtensor)
            for n in f.maker.fgraph.toposort()
        )

    def test_advancedincsubtensor_allocs0t(self):
        x = matrix()
        y = matrix()
        y0 = at.zeros_like(y)
        z = inc_subtensor(x[[[0, 0], [1, 1]], [[0, 1], [0, 1]]], y0.T)
        f = function([x, y], z, mode=mode_opt)
        assert all(
            not isinstance(n.op, AdvancedIncSubtensor)
            for n in f.maker.fgraph.toposort()
        )

    def test_advancedincsubtensor_allocs1(self):
        x = matrix()
        y0 = at.constant(np.asarray(np.zeros_like((2, 2)), dtype=config.floatX))
        z = inc_subtensor(x[[[0, 0], [1, 1]], [[0, 1], [0, 1]]], y0)
        f = function([x], z, mode=self.mode)
        assert all(
            not isinstance(n.op, AdvancedIncSubtensor)
            for n in f.maker.fgraph.toposort()
        )

    def test_dot_allocs_0(self):
        v1 = vector("v1")
        v2 = vector("v2")
        m1 = matrix("m1")
        m2 = matrix("m2")
        vv2 = np.asarray([0, 1], dtype=config.floatX)
        vm2 = np.asarray([[1, 2], [4, 5]], dtype=config.floatX)
        vv3 = np.asarray([0, 1, 2], dtype=config.floatX)
        vm3 = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=config.floatX)
        for _e1 in [(v1, vv2, vv3), (m1, vm2, vm3)]:
            for _e2 in [(v2, vv2, vv3), (m2, vm2, vm3)]:
                for p in [0, 1]:
                    if p == 0:
                        e1 = at.zeros_like(_e1[0])
                        e2 = _e2[0]
                    else:
                        e1 = _e1[0]
                        e2 = at.zeros_like(_e2[0])
                    o = dot(e1, e2)
                    f = function([_e1[0], _e2[0]], o, mode=self.mode)
                    f(_e1[1], _e2[1])
                    f(_e1[2], _e2[2])
                    assert all(
                        not isinstance(n.op, Dot) for n in f.maker.fgraph.toposort()
                    )

                    # test that we don't remove shape errors
                    with pytest.raises((ValueError, AssertionError)):
                        f(_e1[1], _e2[2])
                    with pytest.raises((ValueError, AssertionError)):
                        f(_e1[2], _e2[1])


def test_local_IncSubtensor_serialize():
    d = np.random.normal(0, 0.01, size=(100, 100))
    d = d.astype(config.floatX)

    W = shared(d, name="W")
    i = vector("i", dtype="int64")
    j = vector("j", dtype="int64")
    t = scalar("t")
    y = (W[i] + W[j] + W[1] + W[i, j]).sum()
    cost = square(t - y)
    dW = aesara.grad(cost, W)
    mode = get_default_mode().excluding("fusion")
    mode = mode.including("local_IncSubtensor_serialize")
    f = function([i, j, t], updates=[(W, W - 0.01 * dW)], mode=mode)
    topo = f.maker.fgraph.toposort()
    adds = [
        n
        for n in topo
        if isinstance(n.op, Elemwise) and isinstance(n.op.scalar_op, aes.Add)
    ]
    for a in adds:
        assert not any(
            inp.owner
            and isinstance(
                inp.owner.op,
                (
                    IncSubtensor,
                    AdvancedIncSubtensor,
                    AdvancedIncSubtensor1,
                ),
            )
            for inp in a.inputs
        )

    # Now test that the stack trace is copied over properly,
    # if we return the gradients. We need to use same mode as before.
    f = function([i, j, t], dW, mode=mode)
    assert check_stack_trace(
        f,
        ops_to_check=[
            IncSubtensor,
            AdvancedIncSubtensor,
            AdvancedIncSubtensor1,
        ],
    )


def test_local_set_to_inc_subtensor():
    v = fmatrix()
    s = v[[2, 1]]
    g = s + 3
    r = set_subtensor(s, g)

    mode = get_default_mode().including(
        "local_replace_AdvancedSubtensor",
        "local_AdvancedIncSubtensor_to_AdvancedIncSubtensor1",
    )
    moder = mode.excluding("local_set_to_inc_subtensor")
    modet = mode.including("local_set_to_inc_subtensor")
    f1 = function([v], r, mode=moder)
    f2 = function([v], r, mode=modet)

    advi1 = [
        n for n in f1.maker.fgraph.toposort() if isinstance(n.op, AdvancedIncSubtensor1)
    ]

    advi2 = [
        n for n in f2.maker.fgraph.toposort() if isinstance(n.op, AdvancedIncSubtensor1)
    ]

    # We only have SetSubtensor in f1
    assert all(n.op.set_instead_of_inc for n in advi1)
    # We don't have any SetSubtensor in f2
    assert not any(n.op.set_instead_of_inc for n in advi2)

    val = np.random.standard_normal((3, 2)).astype("float32")

    r1 = f1(val)
    r2 = f2(val)

    utt.assert_allclose(r1, r2)

    # Finally, test that the stack trace is copied over properly,
    # before and after optimization.
    assert check_stack_trace(f1, ops_to_check=AdvancedIncSubtensor1)
    assert check_stack_trace(f2, ops_to_check="all")


def test_local_subtensor_of_alloc():
    # DebugMode should detect if something goes wrong.
    # test shape combination of odd and event shape.
    for s in [(3, 5), (4, 6), (3, 8), (4, 7), (1, 5), (5, 1)]:
        x = tensor(
            dtype=config.floatX,
            shape=(1 if s[0] == 1 else None, 1 if s[1] == 1 else None),
        )

        xval = np.zeros(s, dtype=config.floatX)
        yval = np.arange(s[1], dtype=config.floatX)

        for y in [shared(yval), at.constant([1.0])]:
            # The rows of yx are copies of y
            yx = at.alloc(y, x.shape[0], x.shape[1])

            # Slice of each row
            z_mat = yx[:, 3:]
            assert z_mat.ndim == 2

            # Only one column
            z_vec = yx[:, 3]
            assert z_vec.ndim == 1
            # results are vector
            slicess = []
            if s[0] != 1:
                slicess.append((2, slice(None)))
            if s[1] != 1:
                slicess.append((slice(None), 3))

            # results are matrix
            slicess += [
                (slice(None), slice(3, None)),
                (slice(3, None),),
                (slice(3, None), slice(3, None)),
                (slice(1, 3), slice(None, -1)),
                (slice(None, None, 2)),
                (slice(1, None, 2)),
            ]
            for slices in slicess:
                z = yx.__getitem__(slices)
                f = function([x], z)
                if config.mode != "FAST_COMPILE":
                    # Subtensor can be in the input of Alloc
                    assert not isinstance(f.maker.fgraph.toposort()[-1].op, Subtensor)
                val = f(xval)
                assert xval.__getitem__(slices).shape == val.shape


def test_local_subtensor_shape_constant():
    x = tensor(np.float64, shape=(1, None)).shape[0]
    (res,) = local_subtensor_shape_constant.transform(None, x.owner)
    assert isinstance(res, Constant)
    assert res.data == 1

    # Make sure it's part of the canonicalizations
    res = rewrite_graph(x)
    assert isinstance(res, Constant)
    assert res.data == 1

    x = _shape(tensor(np.float64, shape=(1, None)))[lscalar()]
    assert not local_subtensor_shape_constant.transform(None, x.owner)

    x = _shape(tensor(np.float64, shape=(1, None)))[0:]
    assert not local_subtensor_shape_constant.transform(None, x.owner)

    x = _shape(tensor(np.float64, shape=(1, None)))[lscalar() :]
    assert not local_subtensor_shape_constant.transform(None, x.owner)

    x = _shape(tensor(np.float64, shape=(1, 1)))[1:]
    (res,) = local_subtensor_shape_constant.transform(None, x.owner)
    assert isinstance(res, Constant)
    assert np.array_equal(res.data, [1])

    x = _shape(tensor(np.float64, shape=(None, 1, 1)))[1:]
    (res,) = local_subtensor_shape_constant.transform(None, x.owner)
    assert isinstance(res, Constant)
    assert np.array_equal(res.data, [1, 1])

    # A test for a non-`TensorType`
    class MyType(Type):
        def filter(self, *args, **kwargs):
            raise NotImplementedError()

        def __eq__(self, other):
            return isinstance(other, MyType) and other.thingy == self.thingy

    x = shape(Variable(MyType(), None, None))[0]

    assert not local_subtensor_shape_constant.transform(None, x.owner)


@pytest.mark.parametrize(
    "x, s, idx, x_val, s_val",
    [
        (
            vector(),
            (iscalar(),),
            (1,),
            np.array([1, 2], dtype=config.floatX),
            np.array([2], dtype=np.int64),
        ),
        (
            matrix(),
            (iscalar(), iscalar()),
            (1,),
            np.array([[1, 2], [3, 4]], dtype=config.floatX),
            np.array([2, 2], dtype=np.int64),
        ),
        (
            matrix(),
            (iscalar(), iscalar()),
            (0,),
            np.array([[1, 2, 3], [4, 5, 6]], dtype=config.floatX),
            np.array([2, 3], dtype=np.int64),
        ),
        (
            matrix(),
            (iscalar(), iscalar()),
            (1, 1),
            np.array([[1, 2, 3], [4, 5, 6]], dtype=config.floatX),
            np.array([2, 3], dtype=np.int64),
        ),
        (
            tensor3(),
            (iscalar(), iscalar(), iscalar()),
            (-1,),
            np.arange(2 * 3 * 5, dtype=config.floatX).reshape((2, 3, 5)),
            np.array([2, 3, 5], dtype=np.int64),
        ),
        (
            tensor3(),
            (iscalar(), iscalar(), iscalar()),
            (-1, 0),
            np.arange(2 * 3 * 5, dtype=config.floatX).reshape((2, 3, 5)),
            np.array([2, 3, 5], dtype=np.int64),
        ),
    ],
)
def test_local_subtensor_SpecifyShape_lift(x, s, idx, x_val, s_val):
    y = specify_shape(x, s)[idx]
    assert isinstance(y.owner.inputs[0].owner.op, SpecifyShape)

    rewrites = RewriteDatabaseQuery(include=[None])
    no_rewrites_mode = Mode(optimizer=rewrites)

    y_val_fn = function(
        [x] + list(s), y, on_unused_input="ignore", mode=no_rewrites_mode
    )
    y_val = y_val_fn(*([x_val] + [s_ for s_ in s_val]))

    # This optimization should appear in the canonicalizations
    y_opt = rewrite_graph(y, clone=False)

    if y.ndim == 0:
        # SpecifyShape should be removed altogether
        assert isinstance(y_opt.owner.op, Subtensor)
        assert y_opt.owner.inputs[0] is x
    else:
        assert isinstance(y_opt.owner.op, SpecifyShape)

    y_opt_fn = function([x] + list(s), y_opt, on_unused_input="ignore")
    y_opt_val = y_opt_fn(*([x_val] + [s_ for s_ in s_val]))

    assert np.allclose(y_val, y_opt_val)


@pytest.mark.parametrize(
    "x, s, idx",
    [
        (
            matrix(),
            (iscalar(), iscalar()),
            (slice(1, None),),
        ),
        (
            matrix(),
            (iscalar(), iscalar()),
            (slicetype(),),
        ),
        (
            matrix(),
            (iscalar(), iscalar()),
            (1, 0),
        ),
    ],
)
def test_local_subtensor_SpecifyShape_lift_fail(x, s, idx):
    y = specify_shape(x, s)[idx]

    # This optimization should appear in the canonicalizations
    y_opt = rewrite_graph(y, clone=False)

    assert not isinstance(y_opt.owner.op, SpecifyShape)


@pytest.mark.parametrize(
    "axis, slices_fn, expected_nodes",
    [
        # Below should be merged
        (0, lambda _: ((slice(None, 5, None),), (slice(5, None, None),)), 1),
        (0, lambda _: ((slice(0, 5, 1),), (slice(5, None, 1),)), 1),
        (
            0,
            lambda _: (
                (slice(0, 2, 1),),
                (slice(2, 4, None),),
                (slice(4, None, 1)),
            ),
            1,
        ),
        (
            0,
            lambda _: (
                (slice(None, 5, None), slice(None, -1, None)),
                (slice(5, None, None), slice(None, -1, None)),
            ),
            2,
        ),
        (
            1,
            lambda step: (
                (slice(2, None, step), slice(None, 2, None)),
                (slice(2, None, step), slice(2, 4, None)),
                (slice(2, None, step), slice(4, 6, None)),
            ),
            3,
        ),
        (
            0,
            lambda stop: (
                (slice(1, stop, None),),
                (slice(stop, 5, None),),
                (slice(5, 7, None)),
            ),
            2,
        ),
        (
            0,
            lambda stop: (
                (slice(1, stop + 1, None),),
                (slice(stop + 1, 5, None),),
                (slice(5, 7, None)),
            ),
            2,
        ),
        # Below NotImplemented: These could be merged, but we would need to evaluate the
        # start and stop values
        (0, lambda _: ((slice(None, 6, 3),), (slice(6, None, 3),)), 3),
        (0, lambda step: ((slice(None, 6, step),), (slice(6, None, step),)), 4),
        # Below should not be merged
        (0, lambda _: ((slice(5, None, None),), (slice(None, 5, None),)), 3),
        (0, lambda _: ((slice(None, 5, None),), (slice(4, None, None),)), 3),
        (1, lambda _: ((slice(None, 5, None),), (slice(5, None, None),)), 3),
        (
            0,
            lambda _: (
                (slice(2, None, None), slice(None, 2, None)),
                (slice(2, None, None), slice(2, 4, None)),
                (slice(2, None, None), slice(4, 6, None)),
            ),
            4,
        ),
        (
            0,
            lambda _: (
                (slice(None, 5, 2), slice(None, -1, None)),
                (slice(5, None, 3), slice(None, -1, None)),
            ),
            3,
        ),
        (
            0,
            lambda _: (
                (slice(None, 5, None), slice(None, -1, None)),
                (slice(5, None, None), slice(1, None, None)),
            ),
            3,
        ),
        (0, lambda stop: ((slice(None, stop, None),), (slice(3, None, None),)), 4),
        (0, lambda _: ((slice(None, 5, 2),), (slice(5, None, 2),)), 3),
    ],
)
def test_local_join_subtensors(axis, slices_fn, expected_nodes):
    x = at.dmatrix("x")
    slice_scalar = at.iscalar("slice_scalar")
    slices = slices_fn(slice_scalar)
    y = at.concatenate([x[slice] for slice in slices], axis=axis)
    f = aesara.function(
        [x, slice_scalar],
        y,
        mode=Mode("py").excluding("fusion"),
        on_unused_input="ignore",
    )
    nodes = f.maker.fgraph.toposort()
    assert len(nodes) == expected_nodes, nodes

    x_val = np.arange(100).reshape(10, 10)
    stop_val = 3
    slices_val = slices_fn(stop_val)
    f_val = np.concatenate([x_val[slice_val] for slice_val in slices_val], axis=axis)

    np.testing.assert_array_equal(f(x_val, stop_val), f_val)


def test_deprecations():
    """Make sure we can import from deprecated modules."""
    with pytest.deprecated_call():
        from aesara.tensor.subtensor_opt import get_advsubtensor_axis  # noqa: F401 F811


def test_local_uint_constant_indices():
    mode = get_default_mode().including("specialize", "local_uint_constant_indices")
    rng = np.random.default_rng(20900)

    # Subtensor, don't convert
    x = at.vector("x")
    idx = at.as_tensor_variable(np.array(-1, np.int64))
    z = x[idx]

    z_fn = aesara.function([x], z, mode=mode)

    deepcopy_node = z_fn.maker.fgraph.outputs[0].owner
    subtensor_node = deepcopy_node.inputs[0].owner
    assert isinstance(subtensor_node.op, Subtensor)
    new_index = subtensor_node.inputs[1]
    assert isinstance(new_index, Constant)
    assert new_index.type.dtype == "int64"

    # `Subtensor`, one index, convert
    x = at.vector("x")
    idx = at.as_tensor_variable(np.array(1, np.int64))
    z = x[idx]

    z_fn = aesara.function([x], z, mode=mode)

    deepcopy_node = z_fn.maker.fgraph.outputs[0].owner
    subtensor_node = deepcopy_node.inputs[0].owner
    assert isinstance(subtensor_node.op, Subtensor)
    new_index = subtensor_node.inputs[1]
    assert isinstance(new_index, Constant)
    assert new_index.type.dtype == "uint8"

    # `Subtensor`, two indices, one slice, convert
    x = at.matrix("x")
    indices = (at.as_tensor_variable(np.array(1, np.int64)), slice(None, 10))
    z = x[indices]

    z_fn = aesara.function([x], z, mode=mode)

    deepcopy_node = z_fn.maker.fgraph.outputs[0].owner
    subtensor_node = deepcopy_node.inputs[0].owner
    assert isinstance(subtensor_node.op, Subtensor)
    new_index = subtensor_node.inputs[1]
    assert isinstance(new_index, Constant)
    assert new_index.type.dtype == "uint8"

    # `AdvancedSubtensor`, two indices, one symbolic slice, convert
    x = at.matrix("x")
    indices = (
        at.as_tensor_variable(np.array(1, np.int64)),
        make_slice(slice(None, 10)),
    )
    z = x[indices]

    z_fn = aesara.function([x], z, mode=mode)

    subtensor_node = z_fn.maker.fgraph.outputs[0].owner
    assert isinstance(subtensor_node.op, AdvancedSubtensor)
    new_index = subtensor_node.inputs[1]
    assert isinstance(new_index, Constant)
    assert new_index.type.dtype == "uint8"

    # `AdvancedSubtensor1`, convert
    x = at.vector("x")
    idx = at.as_tensor_variable(rng.integers(0, 10, size=10).astype(np.int64))
    z = x[idx]

    z_fn = aesara.function([x], z, mode=mode)

    subtensor_node = z_fn.maker.fgraph.outputs[0].owner
    assert isinstance(subtensor_node.op, AdvancedSubtensor1)
    new_index = subtensor_node.inputs[1]
    assert isinstance(new_index, Constant)
    assert new_index.type.dtype == "uint8"

    # AdvancedSubtensor, empty, convert
    x = at.matrix("x")
    idx = at.as_tensor_variable(1, dtype=np.int64)
    z = x[idx, []]

    z_fn = aesara.function([x], z, mode=mode)

    subtensor_node = z_fn.maker.fgraph.outputs[0].owner
    assert isinstance(subtensor_node.op, AdvancedSubtensor)
    new_index = subtensor_node.inputs[1]
    assert isinstance(new_index, Constant)
    assert new_index.type.dtype == "uint8"

    # AdvancedSubtensor, bool, don't convert
    x = at.matrix("x")
    idx = at.as_tensor_variable(np.array([True]), dtype=bool)
    z = x[idx, []]

    z_fn = aesara.function([x], z, mode=mode)

    subtensor_node = z_fn.maker.fgraph.outputs[0].owner
    assert isinstance(subtensor_node.op, AdvancedSubtensor)
    new_index = subtensor_node.inputs[1]
    assert isinstance(new_index, Constant)
    assert new_index.type.dtype == "bool"

    # `IncSubtensor`, convert
    x = at.vector("x")
    y = at.scalar("y")
    idx = at.as_tensor_variable(1, dtype=np.int64)
    z = inc_subtensor(x[idx], y)

    z_fn = aesara.function([x, y], z, mode=mode)

    subtensor_node = z_fn.maker.fgraph.outputs[0].owner
    assert isinstance(subtensor_node.op, IncSubtensor)
    new_index = subtensor_node.inputs[2]
    assert isinstance(new_index, Constant)
    assert new_index.type.dtype == "uint8"

    # `AdvancedIncSubtensor1`, convert
    x = at.vector("x")
    y = at.vector("y")
    idx = at.as_tensor_variable(rng.integers(0, 10, size=10).astype(np.int64))
    z = advanced_inc_subtensor1(x, y, idx)

    z_fn = aesara.function([x, y], z, mode=mode)

    subtensor_node = z_fn.maker.fgraph.outputs[0].owner
    assert isinstance(subtensor_node.op, AdvancedIncSubtensor1)
    new_index = subtensor_node.inputs[2]
    assert isinstance(new_index, Constant)
    assert new_index.type.dtype == "uint8"

    # `AdvancedIncSubtensor1`, convert
    x = at.vector("x")
    idx = at.as_tensor_variable(rng.integers(0, 10, size=10).astype(np.int64))
    z = x[idx, None]

    z_fn = aesara.function([x], z, mode=mode)

    subtensor_node = z_fn.maker.fgraph.outputs[0].owner
    assert isinstance(subtensor_node.op, AdvancedSubtensor)
    new_index = subtensor_node.inputs[1]
    assert isinstance(new_index, Constant)
    assert new_index.type.dtype == "uint8"
