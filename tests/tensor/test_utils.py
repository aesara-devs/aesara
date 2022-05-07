import numpy as np
import pytest

import aesara.tensor as at
from aesara.graph.fg import FunctionGraph
from aesara.tensor.type import matrix
from aesara.tensor.utils import hash_from_ndarray, shape_of_variables


def test_hash_from_ndarray():
    hashes = []
    x = np.random.random((5, 5))

    for data in [
        -2,
        -1,
        0,
        1,
        2,
        np.zeros((1, 5)),
        np.zeros((1, 6)),
        # Data buffer empty but different shapes
        np.zeros((1, 0)),
        np.zeros((2, 0)),
        # Same data buffer and shapes but different strides
        np.arange(25).reshape(5, 5),
        np.arange(25).reshape(5, 5).T,
        # Same data buffer, shapes and strides but different dtypes
        np.zeros((5, 5), dtype="uint32"),
        np.zeros((5, 5), dtype="int32"),
        # Test slice
        x,
        x[1:],
        x[:4],
        x[1:3],
        x[::2],
        x[::-1],
    ]:
        data = np.asarray(data)
        hashes.append(hash_from_ndarray(data))

    assert len(set(hashes)) == len(hashes)

    # test that different type of views and their copy give the same hash
    assert hash_from_ndarray(x[1:]) == hash_from_ndarray(x[1:].copy())
    assert hash_from_ndarray(x[1:3]) == hash_from_ndarray(x[1:3].copy())
    assert hash_from_ndarray(x[:4]) == hash_from_ndarray(x[:4].copy())
    assert hash_from_ndarray(x[::2]) == hash_from_ndarray(x[::2].copy())
    assert hash_from_ndarray(x[::-1]) == hash_from_ndarray(x[::-1].copy())


class TestShapeOfVariables:
    def test_simple(self):
        x = matrix("x")
        y = x + x
        fgraph = FunctionGraph([x], [y], clone=False)
        shapes = shape_of_variables(fgraph, {x: (5, 5)})
        assert shapes == {x: (5, 5), y: (5, 5)}

        x = matrix("x")
        y = at.dot(x, x.T)
        fgraph = FunctionGraph([x], [y], clone=False)
        shapes = shape_of_variables(fgraph, {x: (5, 1)})
        assert shapes[x] == (5, 1)
        assert shapes[y] == (5, 5)

    def test_subtensor(self):
        x = matrix("x")
        subx = x[1:]
        fgraph = FunctionGraph([x], [subx], clone=False)
        shapes = shape_of_variables(fgraph, {x: (10, 10)})
        assert shapes[subx] == (9, 10)

    def test_err(self):
        x = matrix("x")
        subx = x[1:]
        fgraph = FunctionGraph([x], [subx])
        with pytest.raises(ValueError):
            shape_of_variables(fgraph, {x: (10, 10)})
