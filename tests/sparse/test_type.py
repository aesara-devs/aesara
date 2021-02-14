import pytest

from aesara.sparse import matrix as sp_matrix
from aesara.sparse.type import SparseType
from aesara.tensor import dmatrix


def test_clone():
    st = SparseType("csr", "float64")
    assert st == st.clone()


def test_Sparse_convert_variable():
    x = dmatrix(name="x")
    y = sp_matrix("csc", dtype="float64", name="y")
    z = sp_matrix("csr", dtype="float64", name="z")

    assert y.type.convert_variable(z) is None

    # TODO FIXME: This is a questionable result, because `x.type` is associated
    # with a dense `Type`, but, since `TensorType` is a base class of `Sparse`,
    # we would need to added sparse/dense logic to `TensorType`, and we don't
    # want to do that.
    assert x.type.convert_variable(y) is y

    # TODO FIXME: We should be able to do this.
    with pytest.raises(NotImplementedError):
        y.type.convert_variable(x)
