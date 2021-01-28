def test_sparse_type():
    import aesara.sparse

    # They need to be available even if scipy is not available.
    assert hasattr(aesara.sparse, "SparseType")
