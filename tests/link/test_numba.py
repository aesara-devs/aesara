from functools import partial

import numpy as np
import pytest

import aesara
import aesara.scalar.basic as aes
import aesara.tensor as aet
from aesara.compile.function import function
from aesara.compile.mode import Mode
from aesara.compile.sharedvalue import SharedVariable
from aesara.graph.fg import FunctionGraph
from aesara.graph.optdb import Query
from aesara.link.numba.linker import NumbaLinker
from aesara.tensor import subtensor as aet_subtensor


opts = Query(include=[None], exclude=["cxx_only", "BlasOpt"])
numba_mode = Mode(NumbaLinker(), opts)
py_mode = Mode("py", opts)


def compare_numba_and_py(
    fgraph,
    inputs,
    assert_fn=None,
):
    """Function to compare python graph output and Numba compiled output for testing equality

    In the tests below computational graphs are defined in Aesara. These graphs are then passed to
    this function which then compiles the graphs in both Numba and python, runs the calculation
    in both and checks if the results are the same

    Parameters
    ----------
    fgraph: FunctionGraph
        Aesara function Graph object
    inputs: iter
        Inputs for function graph
    assert_fn: func, opt
        Assert function used to check for equality between python and Numba. If not
        provided uses np.testing.assert_allclose

    """
    if assert_fn is None:
        assert_fn = partial(np.testing.assert_allclose, rtol=1e-4)

    fn_inputs = [i for i in fgraph.inputs if not isinstance(i, SharedVariable)]
    aesara_numba_fn = function(fn_inputs, fgraph.outputs, mode=numba_mode)
    numba_res = aesara_numba_fn(*inputs)

    aesara_py_fn = function(fn_inputs, fgraph.outputs, mode=py_mode)
    py_res = aesara_py_fn(*inputs)

    if len(fgraph.outputs) > 1:
        for j, p in zip(numba_res, py_res):
            assert_fn(j, p)
    else:
        assert_fn(numba_res, py_res)

    return numba_res


def test_Composite():
    opts = Query(include=["fusion"], exclude=["cxx_only", "BlasOpt"])
    numba_mode = Mode(NumbaLinker(), opts)
    py_mode = Mode("py", opts)

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


@pytest.mark.parametrize(
    "x, indices",
    [
        (aet.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))), (1,)),
        (
            aet.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            (slice(None)),
        ),
        (aet.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))), (1, 2, 0)),
        (
            aet.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            (slice(1, 2), 1, slice(None)),
        ),
    ],
)
def test_Subtensors(x, indices):
    """Test NumPy's basic indexing."""
    out_aet = x[indices]
    assert isinstance(out_aet.owner.op, aet_subtensor.Subtensor)
    out_fg = FunctionGraph([], [out_aet])
    compare_numba_and_py(out_fg, [])


@pytest.mark.parametrize(
    "x, indices",
    [
        (aet.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))), ([1, 2],)),
        (
            aet.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            ([1, 2], slice(None)),
        ),
    ],
)
def test_AdvancedSubtensor1(x, indices):
    """Test NumPy's advanced indexing in one dimension."""
    out_aet = x[[1, 2]]
    assert isinstance(out_aet.owner.op, aet_subtensor.AdvancedSubtensor1)
    out_fg = FunctionGraph([], [out_aet])
    compare_numba_and_py(out_fg, [])


@pytest.mark.parametrize(
    "x, indices",
    [
        (aet.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))), ([1, 2], [2, 3])),
        # XXX TODO: This will fail because advanced indexing calls into object
        # mode (i.e. Python) and there's no unboxing for Numba's internal/native
        # `slice` objects.
        # (
        #     aet.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
        #     ([1, 2], slice(None), [3, 4]),
        # ),
    ],
)
def test_AdvancedSubtensor(x, indices):
    """Test NumPy's advanced indexing in more than one dimension."""
    out_aet = x[indices]
    assert isinstance(out_aet.owner.op, aet_subtensor.AdvancedSubtensor)
    out_fg = FunctionGraph([], [out_aet])
    compare_numba_and_py(out_fg, [])
