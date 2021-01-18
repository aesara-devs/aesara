import numpy as np

import theano
import theano.tensor as tt
from tests import unittest_tools as utt
from theano import function
from theano import scalar as ts
from theano.configdefaults import config
from theano.graph.fg import FunctionGraph
from theano.graph.opt import out2in
from theano.link.basic import PerformLinker
from theano.tensor.basic import MaxAndArgmax
from theano.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from theano.tensor.opt_uncanonicalize import (
    local_alloc_dimshuffle,
    local_dimshuffle_alloc,
    local_dimshuffle_subtensor,
    local_reshape_dimshuffle,
)
from theano.tensor.shape import reshape
from theano.tensor.type import dtensor4, iscalar, matrix, tensor, vector


class TestMaxAndArgmax:
    def test_optimization(self):
        # If we use only the max output, we should replace this op with
        # a faster one.
        mode = theano.compile.mode.get_default_mode().including(
            "canonicalize", "fast_run"
        )

        for axis in [0, 1, -1]:
            n = matrix()

            f = function([n], tt.max_and_argmax(n, axis)[0], mode=mode)
            topo = f.maker.fgraph.toposort()
            assert len(topo) == 1
            assert isinstance(topo[0].op, CAReduce)

            f = function([n], tt.max_and_argmax(n, axis), mode=mode)
            topo = f.maker.fgraph.toposort()
            assert len(topo) == 1
            assert isinstance(topo[0].op, MaxAndArgmax)


class TestMinMax:
    def setup_method(self):
        utt.seed_rng()
        self.mode = theano.compile.mode.get_default_mode().including(
            "canonicalize", "fast_run"
        )

    def test_optimization_max(self):
        data = np.asarray(np.random.rand(2, 3), dtype=config.floatX)
        n = matrix()

        for axis in [0, 1, -1]:
            f = function([n], tt.max(n, axis), mode=self.mode)
            topo = f.maker.fgraph.toposort()
            assert len(topo) == 1
            assert isinstance(topo[0].op, CAReduce)
            f(data)

            f = function([n], tt.max(-n, axis), mode=self.mode)
            topo = f.maker.fgraph.toposort()
            assert len(topo) == 2
            assert isinstance(topo[0].op, Elemwise)
            assert isinstance(topo[0].op.scalar_op, ts.Neg)
            assert isinstance(topo[1].op, CAReduce)
            f(data)

            f = function([n], -tt.max(n, axis), mode=self.mode)
            topo = f.maker.fgraph.toposort()
            assert len(topo) == 2
            assert isinstance(topo[0].op, CAReduce)
            assert isinstance(topo[1].op, Elemwise)
            assert isinstance(topo[1].op.scalar_op, ts.Neg)
            f(data)

            f = function([n], -tt.max(-n, axis), mode=self.mode)
            topo = f.maker.fgraph.toposort()
            assert len(topo) == 1
            assert isinstance(topo[0].op, CAReduce)  # min
            f(data)

    def test_optimization_min(self):
        data = np.asarray(np.random.rand(2, 3), dtype=config.floatX)
        n = matrix()

        for axis in [0, 1, -1]:
            f = function([n], tt.min(n, axis), mode=self.mode)
            topo = f.maker.fgraph.toposort()
            assert len(topo) == 1
            assert isinstance(topo[0].op, CAReduce)
            f(data)

            # test variant with neg to make sure we optimize correctly
            f = function([n], tt.min(-n, axis), mode=self.mode)
            topo = f.maker.fgraph.toposort()
            assert len(topo) == 2
            assert isinstance(topo[0].op, CAReduce)  # max
            assert isinstance(topo[1].op, Elemwise)
            assert isinstance(topo[1].op.scalar_op, ts.Neg)
            f(data)

            f = function([n], -tt.min(n, axis), mode=self.mode)
            topo = f.maker.fgraph.toposort()
            assert len(topo) == 2
            assert isinstance(topo[0].op, Elemwise)
            assert isinstance(topo[0].op.scalar_op, ts.Neg)
            assert isinstance(topo[1].op, CAReduce)  # max
            f(data)

            f = function([n], -tt.min(-n, axis), mode=self.mode)
            topo = f.maker.fgraph.toposort()
            assert len(topo) == 1
            assert isinstance(topo[0].op, CAReduce)  # max
            f(data)


def test_local_alloc_dimshuffle():

    alloc_dimshuffle = out2in(local_alloc_dimshuffle)

    x = vector("x")
    m = iscalar("m")

    y = x.dimshuffle("x", 0)
    out = tt.alloc(y, m, 1, x.shape[0])

    g = FunctionGraph([x, m], [out])
    alloc_dimshuffle(g)

    topo = g.toposort()
    assert any([not isinstance(x, DimShuffle) for x in topo])


def test_local_reshape_dimshuffle():

    reshape_dimshuffle = out2in(local_reshape_dimshuffle)

    x = matrix("x")

    y = x.dimshuffle("x", 0, "x", 1)
    out = reshape(y, (1, x.shape[0] * x.shape[1], 1))

    g = FunctionGraph([x], [out])
    reshape_dimshuffle(g)

    topo = g.toposort()
    assert any([not isinstance(x, DimShuffle) for x in topo])


def test_local_dimshuffle_alloc():

    reshape_dimshuffle = out2in(local_dimshuffle_alloc)

    x = vector("x")

    out = tt.alloc(x, 3, 2).dimshuffle("x", "x", 0, 1)

    g = FunctionGraph([x], [out])
    reshape_dimshuffle(g)

    l = PerformLinker()
    l.accept(g)
    f = l.make_function()

    assert f([3, 4]).ndim == 4

    topo = g.toposort()
    assert any([not isinstance(x, DimShuffle) for x in topo])


def test_local_dimshuffle_subtensor():

    dimshuffle_subtensor = out2in(local_dimshuffle_subtensor)

    x = dtensor4("x")
    x = tt.patternbroadcast(x, (False, True, False, False))
    i = iscalar("i")

    out = x[:, :, 10:30, ::i].dimshuffle(0, 2, 3)

    g = FunctionGraph([x, i], [out])
    dimshuffle_subtensor(g)

    topo = g.toposort()
    assert any([not isinstance(x, DimShuffle) for x in topo])

    # Test dimshuffle remove dimensions the subtensor don't "see".
    x = tensor(broadcastable=(False, True, False), dtype="float64")
    out = x[i].dimshuffle(1)

    g = FunctionGraph([x, i], [out])
    dimshuffle_subtensor(g)

    topo = g.toposort()
    assert any([not isinstance(x, DimShuffle) for x in topo])

    # Test dimshuffle remove dimensions the subtensor don't "see" but
    # have in between dimensions.
    x = tensor(broadcastable=(False, True, False, True), dtype="float64")
    out = x[i].dimshuffle(1)

    f = theano.function([x, i], out)

    topo = f.maker.fgraph.toposort()
    assert any([not isinstance(x, DimShuffle) for x in topo])
    assert f(np.random.rand(5, 1, 4, 1), 2).shape == (4,)

    # Test a corner case that had Theano return a bug.
    x = dtensor4("x")
    x = tt.patternbroadcast(x, (False, True, False, False))

    assert x[:, :, 0:3, ::-1].dimshuffle(0, 2, 3).eval(
        {x: np.ones((5, 1, 6, 7))}
    ).shape == (5, 3, 7)
