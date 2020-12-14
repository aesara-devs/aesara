"""Define the tensor toplevel"""


__docformat__ = "restructuredtext en"

import warnings

# SpecifyShape is defined in theano.compile, but should be available in tensor
from theano.compile import SpecifyShape, specify_shape
from theano.gradient import (
    Lop,
    Rop,
    consider_constant,
    grad,
    hessian,
    jacobian,
    numeric_grad,
    verify_grad,
)
from theano.tensor import nnet  # used for softmax, sigmoid, etc.
from theano.tensor import sharedvar  # adds shared-variable constructors
from theano.tensor import (
    blas,
    blas_c,
    blas_scipy,
    nlinalg,
    opt,
    opt_uncanonicalize,
    xlogx,
)
from theano.tensor.basic import *
from theano.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from theano.tensor.extra_ops import (
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
from theano.tensor.io import *

# We import as `_shared` instead of `shared` to avoid confusion between
# `theano.shared` and `tensor._shared`.
from theano.tensor.sharedvar import tensor_constructor as _shared
from theano.tensor.sort import argsort, argtopk, sort, topk, topk_and_argtopk
from theano.tensor.subtensor import *
from theano.tensor.type_other import *
from theano.tensor.var import (
    TensorConstant,
    TensorConstantSignature,
    TensorVariable,
    _tensor_py_operators,
)
