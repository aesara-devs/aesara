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

from theano.tensor.type import (
    cscalar,
    zscalar,
    fscalar,
    dscalar,
    bscalar,
    wscalar,
    iscalar,
    lscalar,
    cvector,
    zvector,
    fvector,
    dvector,
    bvector,
    wvector,
    ivector,
    lvector,
    cmatrix,
    zmatrix,
    fmatrix,
    dmatrix,
    bmatrix,
    wmatrix,
    imatrix,
    lmatrix,
    crow,
    zrow,
    frow,
    drow,
    brow,
    wrow,
    irow,
    lrow,
    int_matrix_types,
    float_matrix_types,
    complex_matrix_types,
    int_types,
    float_types,
    complex_types,
    int_scalar_types,
    float_scalar_types,
    complex_scalar_types,
    int_vector_types,
    float_vector_types,
    complex_vector_types,
    ccol,
    zcol,
    fcol,
    dcol,
    bcol,
    wcol,
    icol,
    lcol,
    ctensor3,
    ztensor3,
    ftensor3,
    dtensor3,
    btensor3,
    wtensor3,
    itensor3,
    ltensor3,
    ctensor4,
    ztensor4,
    ftensor4,
    dtensor4,
    btensor4,
    wtensor4,
    itensor4,
    ltensor4,
    ctensor5,
    ztensor5,
    ftensor5,
    dtensor5,
    btensor5,
    wtensor5,
    itensor5,
    ltensor5,
    ctensor6,
    ztensor6,
    ftensor6,
    dtensor6,
    btensor6,
    wtensor6,
    itensor6,
    ltensor6,
    ctensor7,
    ztensor7,
    ftensor7,
    dtensor7,
    btensor7,
    wtensor7,
    itensor7,
    ltensor7,
)


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


# These imports cannot be performed here because the modules depend on tensor.  This is done at the
# end of theano.__init__.py instead.
# from theano.tensor import raw_random
# from theano.tensor import shared_randomstreams
