import copy

import scipy.sparse

from aesara.compile import shared_constructor
from aesara.sparse.basic import SparseTensorType, _sparse_py_operators
from aesara.tensor.sharedvar import TensorSharedVariable


class SparseTensorSharedVariable(TensorSharedVariable, _sparse_py_operators):
    @property
    def format(self):
        return self.type.format


@shared_constructor.register(scipy.sparse.spmatrix)
def sparse_constructor(
    value, name=None, strict=False, allow_downcast=None, borrow=False, format=None
):
    if format is None:
        format = value.format

    type = SparseTensorType(format=format, dtype=value.dtype)

    if not borrow:
        value = copy.deepcopy(value)

    return SparseTensorSharedVariable(
        type=type, value=value, strict=strict, allow_downcast=allow_downcast, name=name
    )
