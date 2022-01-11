from aesara.sparse.type import SparseType


def test_clone():
    st = SparseType("csr", "float64")
    assert st == st.clone()
