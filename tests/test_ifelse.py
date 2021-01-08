from copy import deepcopy
from functools import reduce

import numpy as np
import pytest

import theano
import theano.ifelse
import theano.tensor as tt
from tests import unittest_tools as utt
from theano import function
from theano.compile.mode import Mode, get_mode
from theano.graph.basic import Apply
from theano.graph.op import Op
from theano.graph.type import generic
from theano.ifelse import IfElse, ifelse


__docformat__ = "restructedtext en"
__authors__ = "Razvan Pascanu" "PyMC Development Team"
__copyright__ = "(c) 2010, Universite de Montreal"


class TestIfelse(utt.OptimizationTestMixin):
    mode = None
    dtype = theano.config.floatX
    cast_output = staticmethod(tt.as_tensor_variable)
    shared = staticmethod(theano.shared)

    def get_ifelse(self, n):
        if theano.config.mode == "FAST_COMPILE":
            return IfElse(n)
        else:
            return IfElse(n, as_view=True)

    def test_lazy_if(self):
        # Tests that lazy if works .. even if the two results have different
        # shapes but the same type (i.e. both vectors, or matrices or
        # whatnot of same dtype)
        x = tt.vector("x", dtype=self.dtype)
        y = tt.vector("y", dtype=self.dtype)
        c = tt.iscalar("c")
        f = function([c, x, y], ifelse(c, x, y), mode=self.mode)
        self.assertFunctionContains1(f, self.get_ifelse(1))
        rng = np.random.RandomState(utt.fetch_seed())

        xlen = rng.randint(200)
        ylen = rng.randint(200)

        vx = np.asarray(rng.uniform(size=(xlen,)), self.dtype)
        vy = np.asarray(rng.uniform(size=(ylen,)), self.dtype)

        assert np.allclose(vx, f(1, vx, vy))
        assert np.allclose(vy, f(0, vx, vy))

    def test_not_lazy_if_inplace(self):
        # Tests that if the outputs are scalars and the graph is big,
        # we disable the inplace opt to speed up optimization
        x = tt.vector("x", dtype=self.dtype)
        y = tt.vector("y", dtype=self.dtype)
        c = tt.iscalar("c")
        mode = get_mode(self.mode).excluding(
            # Disable many opt to keep the graph big enough to disable
            # the opt.
            "fusion",
            "local_add_canonizer",
            "inplace",
            "constant_folding",
            "constant_folding",
        )
        y2 = reduce(lambda x, y: x + y, [y] + list(range(200)))
        f = function([c, x, y], ifelse(c, x, y2), mode=mode)
        # For not inplace ifelse
        ifnode = [n for n in f.maker.fgraph.toposort() if isinstance(n.op, IfElse)]
        assert len(ifnode) == 1
        assert not ifnode[0].op.as_view
        rng = np.random.RandomState(utt.fetch_seed())

        xlen = rng.randint(200)
        ylen = rng.randint(200)

        vx = np.asarray(rng.uniform(size=(xlen,)), self.dtype)
        vy = np.asarray(rng.uniform(size=(ylen,)), self.dtype)

        assert np.allclose(vx, f(1, vx, vy))
        assert np.allclose(vy + sum(range(200)), f(0, vx, vy))

    def test_mixed_dtype(self):
        x1 = tt.vector("x1", dtype="int32")
        x2 = tt.vector("x2", dtype=self.dtype)
        y1 = tt.vector("y1", dtype="int32")
        y2 = tt.vector("y2", dtype=self.dtype)
        c = tt.iscalar("c")
        f = function([c, x1, x2, y1, y2], ifelse(c, (x1, x2), (y1, y2)), mode=self.mode)
        self.assertFunctionContains1(f, self.get_ifelse(2))
        rng = np.random.RandomState(utt.fetch_seed())

        xlen = rng.randint(200)
        ylen = rng.randint(200)

        vx1 = np.asarray(rng.uniform(size=(xlen,)) * 3, "int32")
        vx2 = np.asarray(rng.uniform(size=(xlen,)), self.dtype)
        vy1 = np.asarray(rng.uniform(size=(ylen,)) * 3, "int32")
        vy2 = np.asarray(rng.uniform(size=(ylen,)), self.dtype)

        o1, o2 = f(1, vx1, vx2, vy1, vy2)
        assert np.allclose(vx1, o1)
        assert np.allclose(vx2, o2)

        o1, o2 = f(0, vx1, vx2, vy1, vy2)
        assert np.allclose(vy1, o1)
        assert np.allclose(vy2, o2)

    def test_lazy_if_on_generics(self):
        x = generic()
        y = generic()
        c = tt.iscalar("c")
        f = function([c, x, y], ifelse(c, x, y))

        vx = ["testX"]
        vy = ["testY"]
        assert f(1, vx, vy) == vx
        assert f(0, vx, vy) == vy

    def test_grad_lazy_if(self):
        # Tests that we can compute the gradients through lazy if
        x = tt.vector("x", dtype=self.dtype)
        y = tt.vector("y", dtype=self.dtype)
        c = tt.iscalar("c")
        z = ifelse(c, x, y)
        gx, gy = theano.grad(z.sum(), [x, y])

        f = function(
            [c, x, y], [self.cast_output(gx), self.cast_output(gy)], mode=self.mode
        )
        # There is only 2 of the 3 ifelse that are moved on the GPU.
        # The one that stay on the CPU is for the shape.
        self.assertFunctionContains(f, self.get_ifelse(1), min=2, max=3)
        rng = np.random.RandomState(utt.fetch_seed())

        xlen = rng.randint(200)
        ylen = rng.randint(200)

        vx = np.asarray(rng.uniform(size=(xlen,)), self.dtype)
        vy = np.asarray(rng.uniform(size=(ylen,)), self.dtype)
        gx0, gy0 = f(1, vx, vy)
        assert np.allclose(gx0.shape, vx.shape)
        assert np.allclose(gy0.shape, vy.shape)
        assert np.all(np.asarray(gx0) == 1.0)
        assert np.all(np.asarray(gy0) == 0.0)

        gx0, gy0 = f(0, vx, vy)
        assert np.allclose(gx0.shape, vx.shape)
        assert np.allclose(gy0.shape, vy.shape)
        assert np.all(np.asarray(gx0) == 0.0)
        assert np.all(np.asarray(gy0) == 1.0)

    def test_grad_cast_input(self):
        # Tests the gradient when both inputs are on the GPU.
        x = tt.vector("x", dtype=self.dtype)
        y = tt.vector("y", dtype=self.dtype)
        c = tt.iscalar("c")
        z = ifelse(c, self.cast_output(x), self.cast_output(y))
        gx, gy = theano.grad(z.sum(), [x, y])

        function([c, x, y], [gx, gy], mode=self.mode)

    def test_multiple_out(self):
        x1 = tt.vector("x1", dtype=self.dtype)
        x2 = tt.vector("x2", dtype=self.dtype)
        y1 = tt.vector("y1", dtype=self.dtype)
        y2 = tt.vector("y2", dtype=self.dtype)
        c = tt.iscalar("c")
        z = ifelse(c, (x1, x2), (y1, y2))
        f = function([c, x1, x2, y1, y2], z, mode=self.mode)
        self.assertFunctionContains1(f, self.get_ifelse(2))

        ifnode = [x for x in f.maker.fgraph.toposort() if isinstance(x.op, IfElse)][0]
        assert len(ifnode.outputs) == 2

        rng = np.random.RandomState(utt.fetch_seed())

        x1len = rng.randint(200)
        x2len = rng.randint(200)
        y1len = rng.randint(200)
        y2len = rng.randint(200)

        vx1 = np.asarray(rng.uniform(size=(x1len,)), self.dtype)
        vx2 = np.asarray(rng.uniform(size=(x2len,)), self.dtype)
        vy1 = np.asarray(rng.uniform(size=(y1len,)), self.dtype)
        vy2 = np.asarray(rng.uniform(size=(y2len,)), self.dtype)

        ovx1, ovx2 = f(1, vx1, vx2, vy1, vy2)
        ovy1, ovy2 = f(0, vx1, vx2, vy1, vy2)
        assert np.allclose(vx1, ovx1)
        assert np.allclose(vy1, ovy1)
        assert np.allclose(vx2, ovx2)
        assert np.allclose(vy2, ovy2)

    def test_multiple_out_grad(self):
        # Tests that we can compute the gradients through lazy if
        x1 = tt.vector("x1")
        x2 = tt.vector("x2")
        y1 = tt.vector("y1")
        y2 = tt.vector("y2")
        c = tt.iscalar("c")
        z = ifelse(c, (x1, x2), (y1, y2))
        grads = theano.grad(z[0].sum() + z[1].sum(), [x1, x2, y1, y2])

        f = function([c, x1, x2, y1, y2], grads)
        rng = np.random.RandomState(utt.fetch_seed())

        lens = [rng.randint(200) for i in range(4)]
        values = [
            np.asarray(rng.uniform(size=(l,)), theano.config.floatX) for l in lens
        ]
        outs_1 = f(1, *values)
        assert all([x.shape[0] == y for x, y in zip(outs_1, lens)])
        assert np.all(outs_1[0] == 1.0)
        assert np.all(outs_1[1] == 1.0)
        assert np.all(outs_1[2] == 0.0)
        assert np.all(outs_1[3] == 0.0)

        outs_0 = f(0, *values)
        assert all([x.shape[0] == y for x, y in zip(outs_1, lens)])
        assert np.all(outs_0[0] == 0.0)
        assert np.all(outs_0[1] == 0.0)
        assert np.all(outs_0[2] == 1.0)
        assert np.all(outs_0[3] == 1.0)

    def test_multiple_out_crash(self):
        # This test failed up to commit 2faeb62c38
        p0 = self.shared(np.asarray(np.random.random([4, 8]), dtype=self.dtype))
        p1 = self.shared(np.asarray(np.random.random(8), dtype=self.dtype))
        p2 = self.shared(np.asarray(np.random.random([8, 3]), dtype=self.dtype))
        p3 = self.shared(np.asarray(np.random.random(3), dtype=self.dtype))
        p = [p0, p1, p2, p3]

        # in my code these vars are the result of applying scan
        ften0 = tt.tensor3("ft0", dtype=self.dtype)
        fmat1 = tt.matrix("fm1", dtype=self.dtype)
        ften2 = tt.tensor3("ft2", dtype=self.dtype)
        fmat3 = tt.matrix("fm3", dtype=self.dtype)

        # then I keep only the last iteration
        fsub0 = ften0[-1]
        fsub1 = fmat1[-1]
        fsub2 = ften2[-1]
        fsub3 = fmat3[-1]

        fsub = [fsub0, fsub1, fsub2, fsub3]

        acc = tt.constant(1, "int8") >= 0

        new_positions = ifelse(acc, fsub, p)

        new_updates = [(p[0], new_positions[0])]

        f = function(
            [ften0, fmat1, ften2, fmat3], [], updates=new_updates, mode=self.mode
        )
        self.assertFunctionContains1(f, self.get_ifelse(4))

        i1 = np.asarray(np.random.random([19, 4, 8]), dtype=self.dtype)
        i2 = np.asarray(np.random.random([19, 8]), dtype=self.dtype)
        i3 = np.asarray(np.random.random([19, 8, 3]), dtype=self.dtype)
        i4 = np.asarray(np.random.random([19, 3]), dtype=self.dtype)

        f(i1, i2, i3, i4)

    def test_dtype_mismatch(self):
        rng = np.random.RandomState(utt.fetch_seed())
        data = rng.rand(5).astype(self.dtype)
        x = self.shared(data)
        y = tt.cast(x * 10, "int8")
        cond = tt.iscalar("cond")

        with pytest.raises(TypeError):
            ifelse(cond, x, y)
        with pytest.raises(TypeError):
            ifelse(cond, y, x)

    def test_ndim_mismatch(self):
        rng = np.random.RandomState(utt.fetch_seed())
        data = rng.rand(5).astype(self.dtype)
        x = self.shared(data)
        y = tt.col("y", self.dtype)
        cond = tt.iscalar("cond")

        with pytest.raises(TypeError):
            ifelse(cond, x, y)
        with pytest.raises(TypeError):
            ifelse(cond, y, x)

    def test_broadcast_mismatch(self):
        rng = np.random.RandomState(utt.fetch_seed())
        data = rng.rand(5).astype(self.dtype)
        x = self.shared(data)
        # print x.broadcastable
        y = tt.row("y", self.dtype)
        # print y.broadcastable
        cond = tt.iscalar("cond")

        with pytest.raises(TypeError):
            ifelse(cond, x, y)
        with pytest.raises(TypeError):
            ifelse(cond, y, x)

    def test_sparse_tensor_error(self):
        pytest.importorskip("scipy", minversion="0.7.0")

        import theano.sparse

        rng = np.random.RandomState(utt.fetch_seed())
        data = rng.rand(2, 3).astype(self.dtype)
        x = self.shared(data)
        y = theano.sparse.matrix("csc", dtype=self.dtype, name="y")
        z = theano.sparse.matrix("csr", dtype=self.dtype, name="z")
        cond = tt.iscalar("cond")

        with pytest.raises(TypeError):
            ifelse(cond, x, y)
        with pytest.raises(TypeError):
            ifelse(cond, y, x)
        with pytest.raises(TypeError):
            ifelse(cond, x, z)
        with pytest.raises(TypeError):
            ifelse(cond, z, x)
        with pytest.raises(TypeError):
            ifelse(cond, y, z)
        with pytest.raises(TypeError):
            ifelse(cond, z, y)

    @pytest.mark.skip(reason="Optimization temporarily disabled")
    def test_merge(self):
        x = tt.vector("x")
        y = tt.vector("y")
        c = tt.iscalar("c")
        z1 = ifelse(c, x + 1, y + 1)
        z2 = ifelse(c, x + 2, y + 2)
        z = z1 + z2
        f = function([c, x, y], z)
        assert (
            len([n for n in f.maker.fgraph.toposort() if isinstance(n.op, IfElse)]) == 1
        )

    @pytest.mark.skip(reason="Optimization temporarily disabled")
    def test_remove_useless_inputs1(self):
        x = tt.vector("x")
        y = tt.vector("y")
        c = tt.iscalar("c")
        z = ifelse(c, (x, x), (y, y))
        f = function([c, x, y], z)

        ifnode = [n for n in f.maker.fgraph.toposort() if isinstance(n.op, IfElse)][0]
        assert len(ifnode.inputs) == 3

    @pytest.mark.skip(reason="Optimization temporarily disabled")
    def test_remove_useless_inputs2(self):
        x1 = tt.vector("x1")
        x2 = tt.vector("x2")
        y1 = tt.vector("y1")
        y2 = tt.vector("y2")
        c = tt.iscalar("c")
        z = ifelse(c, (x1, x1, x1, x2, x2), (y1, y1, y2, y2, y2))
        f = function([c, x1, x2, y1, y2], z)

        ifnode = [x for x in f.maker.fgraph.toposort() if isinstance(x.op, IfElse)][0]
        assert len(ifnode.outputs) == 3

    @pytest.mark.skip(reason="Optimization temporarily disabled")
    def test_pushout1(self):
        x1 = tt.scalar("x1")
        x2 = tt.scalar("x2")
        y1 = tt.scalar("y1")
        y2 = tt.scalar("y2")
        w1 = tt.scalar("w1")
        w2 = tt.scalar("w2")
        c = tt.iscalar("c")
        x, y = ifelse(c, (x1, y1), (x2, y2), name="f1")
        z = ifelse(c, w1, w2, name="f2")
        out = x * z * y

        f = function([x1, x2, y1, y2, w1, w2, c], out, allow_input_downcast=True)
        assert isinstance(f.maker.fgraph.toposort()[-1].op, IfElse)
        rng = np.random.RandomState(utt.fetch_seed())
        vx1 = rng.uniform()
        vx2 = rng.uniform()
        vy1 = rng.uniform()
        vy2 = rng.uniform()
        vw1 = rng.uniform()
        vw2 = rng.uniform()

        assert np.allclose(f(vx1, vx2, vy1, vy2, vw1, vw2, 1), vx1 * vy1 * vw1)
        assert np.allclose(f(vx1, vx2, vy1, vy2, vw1, vw2, 0), vx2 * vy2 * vw2)

    @pytest.mark.skip(reason="Optimization temporarily disabled")
    def test_pushout3(self):
        x1 = tt.scalar("x1")
        y1 = tt.scalar("x2")
        y2 = tt.scalar("y2")
        c = tt.iscalar("c")
        two = np.asarray(2, dtype=theano.config.floatX)
        x, y = ifelse(c, (x1, y1), (two, y2), name="f1")
        o3 = np.asarray(0.3, dtype=theano.config.floatX)
        o2 = np.asarray(0.2, dtype=theano.config.floatX)
        z = ifelse(c, o3, o2, name="f2")
        out = x * z * y

        f = function([x1, y1, y2, c], out, allow_input_downcast=True)
        assert isinstance(f.maker.fgraph.toposort()[-1].op, IfElse)
        rng = np.random.RandomState(utt.fetch_seed())
        vx1 = rng.uniform()
        vy1 = rng.uniform()
        vy2 = rng.uniform()

        assert np.allclose(f(vx1, vy1, vy2, 1), vx1 * vy1 * 0.3)
        assert np.allclose(f(vx1, vy1, vy2, 0), 2 * vy2 * 0.2)

    @pytest.mark.skip(reason="Optimization temporarily disabled")
    def test_pushout2(self):
        x1 = tt.scalar("x1")
        x2 = tt.scalar("x2")
        y1 = tt.scalar("y1")
        y2 = tt.scalar("y2")
        w1 = tt.scalar("w1")
        w2 = tt.scalar("w2")
        c = tt.iscalar("c")
        x, y = ifelse(c, (x1, y1), (x2, y2), name="f1")
        z = ifelse(x > y, w1, w2, name="f2")
        out = x * z * y

        f = function([x1, x2, y1, y2, w1, w2, c], out, allow_input_downcast=True)
        assert isinstance(f.maker.fgraph.toposort()[-1].op, IfElse)
        rng = np.random.RandomState(utt.fetch_seed())
        vx1 = rng.uniform()
        vx2 = rng.uniform()
        vy1 = rng.uniform()
        vy2 = rng.uniform()
        vw1 = rng.uniform()
        vw2 = rng.uniform()
        if vx1 > vy1:
            vw = vw1
        else:
            vw = vw2
        assert np.allclose(f(vx1, vx2, vy1, vy2, vw1, vw2, 1), vx1 * vy1 * vw)

        if vx2 > vy2:
            vw = vw1
        else:
            vw = vw2
        assert np.allclose(f(vx1, vx2, vy1, vy2, vw1, vw2, 0), vx2 * vy2 * vw)

    @pytest.mark.skip(reason="Optimization temporarily disabled")
    def test_merge_ifs_true_false(self):
        x1 = tt.scalar("x1")
        x2 = tt.scalar("x2")
        y1 = tt.scalar("y1")
        y2 = tt.scalar("y2")
        w1 = tt.scalar("w1")
        w2 = tt.scalar("w2")
        c = tt.iscalar("c")

        out = ifelse(
            c,
            ifelse(c, x1, x2) + ifelse(c, y1, y2) + w1,
            ifelse(c, x1, x2) + ifelse(c, y1, y2) + w2,
        )
        f = function([x1, x2, y1, y2, w1, w2, c], out, allow_input_downcast=True)
        assert (
            len([x for x in f.maker.fgraph.toposort() if isinstance(x.op, IfElse)]) == 1
        )

        rng = np.random.RandomState(utt.fetch_seed())
        vx1 = rng.uniform()
        vx2 = rng.uniform()
        vy1 = rng.uniform()
        vy2 = rng.uniform()
        vw1 = rng.uniform()
        vw2 = rng.uniform()
        assert np.allclose(f(vx1, vx2, vy1, vy2, vw1, vw2, 1), vx1 + vy1 + vw1)
        assert np.allclose(f(vx1, vx2, vy1, vy2, vw1, vw2, 0), vx2 + vy2 + vw2)

    def test_grad_test_values(self):
        # Regression test for test values of `ifelse` gradient.
        with theano.config.change_flags(compute_test_value="raise"):
            x = tt.scalar("x")
            x.tag.test_value = 1
            # Used to crash due to undefined test value.
            theano.grad(ifelse(0, x, x), x)

    def test_grad_int_value(self):
        w = theano.shared(np.random.rand(10))
        b = theano.shared(np.random.rand())
        params = [w, b]

        x = tt.vector()
        y = tt.scalar()

        score = w.dot(x) + b
        correct = score * y > 0

        loss = ifelse(correct, 0, 1)
        [(param, param - 0.5 * theano.grad(cost=loss, wrt=param)) for param in params]


class IfElseIfElseIf(Op):
    def __init__(self, inplace=False):
        # check destroyhandler and others to ensure that a view_map with
        self.inplace = inplace
        # multiple inputs can work
        assert not self.inplace

    def make_node(self, c1, t1, c2, t2, c3, t3, f3):
        assert t1.type == f3.type
        assert t2.type == t3.type
        assert t3.type == f3.type
        return Apply(self, [c1, t1, c2, t2, c3, t3, f3], [t1.type()])

    def make_thunk(self, node, storage_map, compute_map, no_recycling, impl):

        input_computed = [compute_map[v] for v in node.inputs]
        output_computed = [compute_map[v] for v in node.outputs]
        input_registers = [storage_map[v] for v in node.inputs]
        output_registers = [storage_map[v] for v in node.outputs]

        outtype = node.outputs[0].type

        def thunk():
            if not input_computed[0][0]:
                return [0]
            else:
                truthval = input_registers[0][0]
                if truthval:
                    if not input_computed[1][0]:
                        return [1]
                    else:
                        output_computed[0][0] = 1
                        output_registers[0][0] = outtype.filter(
                            deepcopy(input_registers[1][0])
                        )
                        return []
                else:
                    if not input_computed[2][0]:
                        return [2]
                    else:
                        truthval = input_registers[2][0]
                        if truthval:
                            if not input_computed[3][0]:
                                return [3]
                            else:
                                output_computed[0][0] = 1
                                output_registers[0][0] = outtype.filter(
                                    deepcopy(input_registers[3][0])
                                )
                                return []
                        else:
                            if not input_computed[4][0]:
                                return [4]
                            else:
                                truthval = input_registers[4][0]
                                if truthval:
                                    if not input_computed[5][0]:
                                        return [5]
                                    else:
                                        output_computed[0][0] = 1
                                        output_registers[0][0] = outtype.filter(
                                            deepcopy(input_registers[5][0])
                                        )
                                        return []
                                else:
                                    if not input_computed[6][0]:
                                        return [6]
                                    else:
                                        output_computed[0][0] = 1
                                        output_registers[0][0] = outtype.filter(
                                            deepcopy(input_registers[6][0])
                                        )
                                        return []

        thunk.lazy = True
        return thunk

    def perform(self, *args, **kwargs):
        raise NotImplementedError()


class NotImplementedOpException(Exception):
    pass


class NotImplementedOp(Op):
    def make_node(self, x):
        return Apply(self, [x], [x.type()])

    def make_thunk(self, node, storage_map, compute_map, no_recycling, impl):
        def thunk():
            raise NotImplementedOpException()

        thunk.lazy = False
        return thunk

    def perform(self, *args, **kwargs):
        raise NotImplementedError()


def test_ifelse():
    a = tt.scalar()
    b = generic()
    c = generic()

    notimpl = NotImplementedOp()

    lazys = [True]
    # We need lazy to end up being True for this test.
    if theano.config.vm__lazy in [True, None]:
        lazys = [True, None]

    cloops = [True, False]

    if theano.config.cxx == "":
        cloops = [False]

    for cloop in cloops:
        for lazy in lazys:
            linker = theano.link.vm.VMLinker(use_cloop=cloop, lazy=lazy)
            f = function(
                [a, b, c],
                ifelse(a, notimpl(b), c),
                mode=Mode(linker=linker, optimizer="fast_run"),
            )

            with pytest.raises(NotImplementedOpException):
                f(1, "a", "b")

            assert f(0, "a", "b") == "b"


def test_nested():
    notimpl = NotImplementedOp()
    ifelseifelseif = IfElseIfElseIf()

    x1 = tt.scalar("x1")
    x2 = tt.scalar("x2")
    c1 = tt.scalar("c1")
    c2 = tt.scalar("c2")
    t1 = ifelse(c1, x1, notimpl(x2))
    t1.name = "t1"
    t2 = t1 * 10
    t2.name = "t2"
    t3 = ifelse(c2, t2, x1 + t1)
    t3.name = "t3"
    t4 = ifelseifelseif(tt.eq(x1, x2), x1, tt.eq(x1, 5), x2, c2, t3, t3 + 0.5)
    t4.name = "t4"

    linker = theano.link.vm.VMLinker(lazy=False)
    f = function([c1, c2, x1, x2], t4, mode=Mode(linker=linker, optimizer="fast_run"))
    with pytest.raises(NotImplementedOpException):
        f(1, 0, np.array(10, dtype=x1.dtype), 0)

    linker = theano.link.vm.VMLinker(lazy=True)
    f = function([c1, c2, x1, x2], t4, mode=Mode(linker=linker, optimizer="fast_run"))
    assert f(1, 0, np.array(10, dtype=x1.dtype), 0) == 20.5
