"""
This script tests the pickle and unpickle of aesara functions.
When a compiled aesara has shared vars, their values are also being pickled.

Side notes useful for debugging:
The pickling tools aesara uses is here:
aesara.compile.function_module._pickle_Function()
aesara.compile.function_module._pickle_FunctionMaker()
Whether reoptimize the pickled function graph is handled by
FunctionMaker.__init__()
The config option is in configdefaults.py

This note is written by Li Yao.
"""
import pickle
from collections import OrderedDict

import numpy as np

import aesara
import aesara.tensor as tt


floatX = "float32"


def test_pickle_unpickle_with_reoptimization():
    mode = aesara.config.mode
    if mode in ["DEBUG_MODE", "DebugMode"]:
        mode = "FAST_RUN"
    x1 = tt.fmatrix("x1")
    x2 = tt.fmatrix("x2")
    x3 = aesara.shared(np.ones((10, 10), dtype=floatX))
    x4 = aesara.shared(np.ones((10, 10), dtype=floatX))
    y = tt.sum(tt.sum(tt.sum(x1 ** 2 + x2) + x3) + x4)

    updates = OrderedDict()
    updates[x3] = x3 + 1
    updates[x4] = x4 + 1
    f = aesara.function([x1, x2], y, updates=updates, mode=mode)

    # now pickle the compiled aesara fn
    string_pkl = pickle.dumps(f, -1)

    in1 = np.ones((10, 10), dtype=floatX)
    in2 = np.ones((10, 10), dtype=floatX)

    # test unpickle with optimization
    default = aesara.config.reoptimize_unpickled_function
    try:
        # the default is True
        aesara.config.reoptimize_unpickled_function = True
        f_ = pickle.loads(string_pkl)
        assert f(in1, in2) == f_(in1, in2)
    finally:
        aesara.config.reoptimize_unpickled_function = default


def test_pickle_unpickle_without_reoptimization():
    mode = aesara.config.mode
    if mode in ["DEBUG_MODE", "DebugMode"]:
        mode = "FAST_RUN"
    x1 = tt.fmatrix("x1")
    x2 = tt.fmatrix("x2")
    x3 = aesara.shared(np.ones((10, 10), dtype=floatX))
    x4 = aesara.shared(np.ones((10, 10), dtype=floatX))
    y = tt.sum(tt.sum(tt.sum(x1 ** 2 + x2) + x3) + x4)

    updates = OrderedDict()
    updates[x3] = x3 + 1
    updates[x4] = x4 + 1
    f = aesara.function([x1, x2], y, updates=updates, mode=mode)

    # now pickle the compiled aesara fn
    string_pkl = pickle.dumps(f, -1)

    # compute f value
    in1 = np.ones((10, 10), dtype=floatX)
    in2 = np.ones((10, 10), dtype=floatX)

    # test unpickle without optimization
    default = aesara.config.reoptimize_unpickled_function
    try:
        # the default is True
        aesara.config.reoptimize_unpickled_function = False
        f_ = pickle.loads(string_pkl)
        assert f(in1, in2) == f_(in1, in2)
    finally:
        aesara.config.reoptimize_unpickled_function = default


if __name__ == "__main__":
    test_pickle_unpickle_with_reoptimization()
    test_pickle_unpickle_without_reoptimization()
