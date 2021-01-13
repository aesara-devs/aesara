import os

import numpy as np

import theano
import theano.tensor as tt
from theano.tensor.type import fmatrix


def test_graph_opt_caching():
    opt_db_file = os.path.join(theano.config.compiledir, "optimized_graphs.pkl")
    if os.path.exists(opt_db_file):
        os.remove(opt_db_file)

    floatX = "float32"
    mode = theano.config.mode
    if mode in ["DEBUG_MODE", "DebugMode"]:
        mode = "FAST_RUN"

    with theano.config.change_flags(cache_optimizations=True):
        a = fmatrix("a")
        b = fmatrix("b")
        c = theano.shared(np.ones((10, 10), dtype=floatX))
        d = theano.shared(np.ones((10, 10), dtype=floatX))
        e = tt.sum(tt.sum(tt.sum(a ** 2 + b) + c) + d)
        f1 = theano.function([a, b], e, mode=mode)

        m = fmatrix("x1")
        n = fmatrix("x2")
        p = theano.shared(np.ones((10, 10), dtype=floatX))
        q = theano.shared(np.ones((10, 10), dtype=floatX))
        j = tt.sum(tt.sum(tt.sum(m ** 2 + n) + p) + q)
        f2 = theano.function([m, n], j, mode=mode)

        in1 = np.ones((10, 10), dtype=floatX)
        in2 = np.ones((10, 10), dtype=floatX)
        assert f1(in1, in2) == f2(in1, in2)
