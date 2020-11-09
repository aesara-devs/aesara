import os

import numpy as np

import aesara
import aesara.tensor as tt


floatX = "float32"


def test_graph_opt_caching():
    opt_db_file = os.path.join(aesara.config.compiledir, "optimized_graphs.pkl")
    if os.path.exists(opt_db_file):
        os.remove(opt_db_file)

    mode = aesara.config.mode
    if mode in ["DEBUG_MODE", "DebugMode"]:
        mode = "FAST_RUN"
    default = aesara.config.cache_optimizations
    try:
        aesara.config.cache_optimizations = True
        a = tt.fmatrix("a")
        b = tt.fmatrix("b")
        c = aesara.shared(np.ones((10, 10), dtype=floatX))
        d = aesara.shared(np.ones((10, 10), dtype=floatX))
        e = tt.sum(tt.sum(tt.sum(a ** 2 + b) + c) + d)
        f1 = aesara.function([a, b], e, mode=mode)

        m = tt.fmatrix("x1")
        n = tt.fmatrix("x2")
        p = aesara.shared(np.ones((10, 10), dtype=floatX))
        q = aesara.shared(np.ones((10, 10), dtype=floatX))
        j = tt.sum(tt.sum(tt.sum(m ** 2 + n) + p) + q)
        f2 = aesara.function([m, n], j, mode=mode)

        in1 = np.ones((10, 10), dtype=floatX)
        in2 = np.ones((10, 10), dtype=floatX)
        assert f1(in1, in2) == f2(in1, in2)
    finally:
        aesara.config.cache_optimizations = default


if __name__ == "__main__":
    test_graph_opt_caching()
