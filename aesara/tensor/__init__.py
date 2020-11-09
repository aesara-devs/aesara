"""Define the tensor toplevel"""


__docformat__ = "restructuredtext en"

import warnings

# SpecifyShape is defined in aesara.compile, but should be available in tensor
from aesara.compile import SpecifyShape, specify_shape
from aesara.gradient import (
    Lop,
    Rop,
    consider_constant,
    grad,
    hessian,
    jacobian,
    numeric_grad,
    verify_grad,
)
from aesara.tensor import nnet  # used for softmax, sigmoid, etc.
from aesara.tensor import sharedvar  # adds shared-variable constructors
from aesara.tensor import (
    blas,
    blas_c,
    blas_scipy,
    nlinalg,
    opt,
    opt_uncanonicalize,
    xlogx,
)
from aesara.tensor.basic import *
from aesara.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from aesara.tensor.extra_ops import (
    DiffOp,
    bartlett,
    bincount,
    cumprod,
    cumsum,
    fill_diagonal,
    fill_diagonal_offset,
    ravel_multi_index,
    repeat,
    squeeze,
    unravel_index,
)
from aesara.tensor.io import *

# We import as `_shared` instead of `shared` to avoid confusion between
# `aesara.shared` and `tensor._shared`.
from aesara.tensor.sharedvar import tensor_constructor as _shared
from aesara.tensor.sort import argsort, argtopk, sort, topk, topk_and_argtopk
from aesara.tensor.subtensor import *
from aesara.tensor.type_other import *
from aesara.tensor.var import (
    TensorConstant,
    TensorConstantSignature,
    TensorVariable,
    _tensor_py_operators,
)


# These imports cannot be performed here because the modules depend on tensor.  This is done at the
# end of aesara.__init__.py instead.
# from aesara.tensor import raw_random
# from aesara.tensor import shared_randomstreams
