import numpy as np

import aesara
import aesara.scalar.basic as aes
import aesara.tensor as aet
from aesara.compile.mode import Mode
from aesara.graph.optdb import Query
from aesara.link.numba.linker import NumbaLinker


# from aesara.graph.fg import FunctionGraph


opts = Query(include=["fusion"], exclude=["cxx_only", "BlasOpt"])
numba_mode = Mode(NumbaLinker(), opts)
py_mode = Mode("py", opts)


def test_Composite():
    y = aet.vector("y")
    x = aet.vector("x")

    z = (x + y) * (x + y) * y

    func = aesara.function([x, y], [z], mode=py_mode)
    numba_fn = aesara.function([x, y], [z], mode=numba_mode)

    # Make sure the graph had a `Composite` `Op` in it
    composite_op = numba_fn.maker.fgraph.outputs[0].owner.op.scalar_op
    assert isinstance(composite_op, aes.Composite)

    x_val = np.random.randn(1000)
    y_val = np.random.randn(1000)

    res = func(x_val, y_val)  # Answer from python mode compilation of FunctionGraph
    numba_res = numba_fn(x_val, y_val)  # Answer from Numba converted FunctionGraph

    assert np.array_equal(res, numba_res)

    y1 = aet.vector("y1")
    x1 = aet.vector("x1")

    z = (x + y) * (x1 + y1) * y

    x1_val = np.random.randn(1000)
    y1_val = np.random.randn(1000)

    func = aesara.function([x, y, x1, y1], [z], mode=py_mode)
    numba_fn = aesara.function([x, y, x1, y1], [z], mode=numba_mode)

    composite_op = numba_fn.maker.fgraph.outputs[0].owner.op.scalar_op
    assert isinstance(composite_op, aes.Composite)

    res = func(
        x_val, y_val, x1_val, y1_val
    )  # Answer from python mode compilation of FunctionGraph
    numba_res = numba_fn(
        x_val, y_val, x1_val, y1_val
    )  # Answer from Numba converted FunctionGraph

    assert np.array_equal(res, numba_res)
