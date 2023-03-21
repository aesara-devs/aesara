import numba
import numpy as np
import pytest
import scipy as sp

# Make sure the Numba customizations are loaded
import aesara.link.numba.dispatch.sparse  # noqa: F401
from aesara import config
from aesara.sparse import Dot, SparseTensorType
from tests.link.numba.test_basic import compare_numba_and_py


pytestmark = pytest.mark.filterwarnings("error")


def test_sparse_unboxing():
    @numba.njit
    def test_unboxing(x, y):
        return x.shape, y.shape

    x_val = sp.sparse.csr_matrix(np.eye(100))
    y_val = sp.sparse.csc_matrix(np.eye(101))

    res = test_unboxing(x_val, y_val)

    assert res == (x_val.shape, y_val.shape)


def test_sparse_boxing():
    @numba.njit
    def test_boxing(x, y):
        return x, y

    x_val = sp.sparse.csr_matrix(np.eye(100))
    y_val = sp.sparse.csc_matrix(np.eye(101))

    res_x_val, res_y_val = test_boxing(x_val, y_val)

    assert np.array_equal(res_x_val.data, x_val.data)
    assert np.array_equal(res_x_val.indices, x_val.indices)
    assert np.array_equal(res_x_val.indptr, x_val.indptr)
    assert res_x_val.shape == x_val.shape

    assert np.array_equal(res_y_val.data, y_val.data)
    assert np.array_equal(res_y_val.indices, y_val.indices)
    assert np.array_equal(res_y_val.indptr, y_val.indptr)
    assert res_y_val.shape == y_val.shape


def test_sparse_shape():
    @numba.njit
    def test_fn(x):
        return np.shape(x)

    x_val = sp.sparse.csr_matrix(np.eye(100))

    res = test_fn(x_val)

    assert res == (100, 100)


def test_sparse_ndim():
    @numba.njit
    def test_fn(x):
        return x.ndim

    x_val = sp.sparse.csr_matrix(np.eye(100))

    res = test_fn(x_val)

    assert res == 2


def test_sparse_copy():
    @numba.njit
    def test_fn(x):
        y = x.copy()
        return (
            y is not x and np.all(x.data == y.data) and np.all(x.indices == y.indices)
        )

    x_val = sp.sparse.csr_matrix(np.eye(100))

    assert test_fn(x_val)


def test_sparse_objmode():
    x = SparseTensorType("csc", dtype=config.floatX)()
    y = SparseTensorType("csc", dtype=config.floatX)()

    out = Dot()(x, y)

    x_val = sp.sparse.random(2, 2, density=0.25, dtype=config.floatX)
    y_val = sp.sparse.random(2, 2, density=0.25, dtype=config.floatX)

    with pytest.warns(UserWarning):
        compare_numba_and_py(((x, y), (out,)), [x_val, y_val])
