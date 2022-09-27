import pytest
import scipy as sp

from aesara.sparse import matrix as sp_matrix
from aesara.sparse.type import SparseTensorType
from aesara.tensor import dmatrix


def test_SparseTensorType_constructor():
    st = SparseTensorType("csc", "float64")
    assert st.format == "csc"
    assert st.shape == (None, None)

    st = SparseTensorType("bsr", "float64", shape=(None, 1))
    assert st.format == "bsr"
    assert st.shape == (None, 1)

    with pytest.raises(ValueError):
        SparseTensorType("blah", "float64")


def test_SparseTensorType_clone():
    st = SparseTensorType("csr", "float64", shape=(3, None))
    assert st == st.clone()

    st_clone = st.clone(format="csc")
    assert st_clone.format == "csc"
    assert st_clone.dtype == st.dtype
    assert st_clone.shape == st.shape

    st_clone = st.clone(shape=(2, 1))
    assert st_clone.format == st.format
    assert st_clone.dtype == st.dtype
    assert st_clone.shape == (2, 1)


def test_SparseTensorType_convert_variable():
    x = dmatrix(name="x")
    y = sp_matrix("csc", dtype="float64", name="y")
    z = sp_matrix("csr", dtype="float64", name="z")

    assert y.type.convert_variable(z) is None
    assert z.type.convert_variable(y) is None

    res = y.type.convert_variable(x)
    assert res.type == y.type

    res = z.type.convert_variable(x)
    assert res.type == z.type

    # TODO FIXME: This is a questionable result, because `x.type` is associated
    # with a dense `Type`, but, since `TensorType` is a base class of `Sparse`,
    # we would need to added sparse/dense logic to `TensorType`, and we don't
    # want to do that.
    assert x.type.convert_variable(y) is y


def test_SparseTensorType_filter():
    y = sp_matrix("csc", dtype="float64", name="y")
    z = sp_matrix("csr", dtype="float64", name="z")
    w = sp_matrix("csr", dtype="float32", name="z")

    with pytest.raises(TypeError, match="Expected an array-like"):
        y.type.filter(dmatrix())

    x = sp.sparse.csc_matrix(sp.sparse.eye(5, 3))
    x_res = y.type.filter(x)
    assert x is x_res

    x_res = z.type.filter(x)
    assert x_res.format == "csr"

    with pytest.raises(TypeError):
        x_res = z.type.filter(x, strict=True)

    x_res = w.type.filter(x, allow_downcast=True)
    assert x_res.dtype == "float32"

    x_res = z.type.filter(x.astype("float32"), allow_downcast=True)
    assert x_res.dtype == "float64"

    with pytest.raises(TypeError, match=".*dtype but got.*"):
        w.type.filter(x)
