import timeit

import numpy as np
import pytest

import aesara.tensor as aet
from aesara import config
from aesara.compile.function import function
from aesara.compile.mode import Mode
from aesara.graph.rewriting.db import RewriteDatabaseQuery
from aesara.link.numba.linker import NumbaLinker
from aesara.tensor.math import Max


opts = RewriteDatabaseQuery(include=[None], exclude=["cxx_only", "BlasOpt"])
numba_mode = Mode(NumbaLinker(), opts)
py_mode = Mode("py", opts)


@pytest.mark.parametrize(
    "careduce_fn, numpy_fn, axis, inputs, input_vals",
    [
        pytest.param(
            lambda x, axis=None: Max(axis)(x),
            np.max,
            (0, 1),
            [
                aet.matrix(),
            ],
            [np.arange(3000 * 2000, dtype=config.floatX).reshape((3000, 2000))],
        )
    ],
)
def test_careduce_performance(careduce_fn, numpy_fn, axis, inputs, input_vals):
    g = careduce_fn(*inputs, axis=axis)

    aesara_numba_fn = function(
        inputs,
        g,
        mode=numba_mode,
    )

    # aesara_c_fn = function(
    #     inputs,
    #     g,
    #     mode=Mode("cvm")
    # )

    numpy_res = numpy_fn(*input_vals)
    numba_res = aesara_numba_fn(*input_vals)
    # c_res = aesara_c_fn(*input_vals)

    assert np.array_equal(numba_res, numpy_res)

    # FYI: To test the Numba JITed function directly, use `aesara_numba_fn.vm.jit_fn`

    numpy_timer = timeit.Timer("numpy_fn(*input_vals)", "pass", globals=locals())
    numba_timer = timeit.Timer(
        "aesara_numba_fn.vm.jit_fn(*input_vals)", "pass", globals=locals()
    )
    # c_timer = timeit.Timer("aesara_c_fn(*input_vals)", "pass", globals=locals())

    n_loops, _ = numpy_timer.autorange()

    numpy_times = numpy_timer.repeat(5, n_loops)
    numba_times = numba_timer.repeat(5, n_loops)
    # c_times = c_timer.repeat(5, n_loops)

    mean_numba_time = np.mean(numba_times)
    mean_numpy_time = np.mean(numpy_times)
    # mean_c_time = np.mean(c_times)

    assert mean_numba_time / mean_numpy_time >= 0.75
