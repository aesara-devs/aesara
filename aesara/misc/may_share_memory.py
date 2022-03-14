"""
Function to detect memory sharing for ndarray AND sparse type.
numpy version support only ndarray.
"""


import numpy as np

from aesara.tensor.type import TensorType


try:
    import scipy.sparse

    from aesara.sparse.basic import SparseTensorType

    def _is_sparse(a):
        return scipy.sparse.issparse(a)

except ImportError:

    def _is_sparse(a):
        return False


def may_share_memory(a, b, raise_other_type=True):
    a_ndarray = isinstance(a, np.ndarray)
    b_ndarray = isinstance(b, np.ndarray)
    if a_ndarray and b_ndarray:
        return TensorType.may_share_memory(a, b)

    a_sparse = _is_sparse(a)
    b_sparse = _is_sparse(b)
    if not (a_ndarray or a_sparse) or not (b_ndarray or b_sparse):
        if raise_other_type:
            raise TypeError("may_share_memory support only ndarray" " and scipy.sparse")
        return False

    return SparseTensorType.may_share_memory(a, b)
