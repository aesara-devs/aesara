import numpy as np

import aesara
import aesara.tensor as aet
from aesara.compile.mode import Mode
from aesara.graph.optdb import Query

# from aesara.graph.fg import FunctionGraph
from aesara.link.numba.linker import compile_graph


opts = Query(include=[None], exclude=["cxx_only", "BlasOpt"])
# numba_mode = Mode(NumbaLinker(), opts)
py_mode = Mode("py", opts)


def test_composite():
    y = aet.vector("y")
    x = aet.vector("x")

    z = (x + y) * (x + y) * y

    opts = Query(include=["fusion"], exclude=["cxx_only", "BlasOpt"])
    mode = Mode("py", opts)

    # z_fg = FunctionGraph([x, y], [z])

    # XXX: Doesn't work
    # numba_fn = numba_linker.compile_graph(z_fg)

    func = aesara.function([x, y], [z], mode=mode)
    numba_fn = compile_graph(func.maker.fgraph, debug=True)

    x = np.random.randn(1000)
    y = np.random.randn(1000)

    res = numba_fn(x, y)

    # TODO: Make a real test
    assert res