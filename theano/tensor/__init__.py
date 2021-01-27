"""Define the tensor toplevel"""


__docformat__ = "restructuredtext en"

import warnings

import theano.tensor.exceptions
from theano.gradient import consider_constant, grad, hessian, jacobian
from theano.tensor import sharedvar  # adds shared-variable constructors
from theano.tensor import (
    basic_opt,
    blas,
    blas_c,
    blas_scipy,
    nlinalg,
    nnet,
    opt_uncanonicalize,
    xlogx,
)
from theano.tensor.basic import *
from theano.tensor.blas import batched_dot, batched_tensordot
from theano.tensor.extra_ops import (
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
from theano.tensor.math import *
from theano.tensor.shape import (
    reshape,
    shape,
    shape_padaxis,
    shape_padleft,
    shape_padright,
    specify_shape,
)

# We import as `_shared` instead of `shared` to avoid confusion between
# `theano.shared` and `tensor._shared`.
from theano.tensor.sort import argsort, argtopk, sort, topk, topk_and_argtopk
from theano.tensor.subtensor import *
from theano.tensor.type import (
    TensorType,
    bcol,
    bmatrix,
    brow,
    bscalar,
    btensor3,
    btensor4,
    btensor5,
    btensor6,
    btensor7,
    bvector,
    ccol,
    cmatrix,
    col,
    cols,
    complex_matrix_types,
    complex_scalar_types,
    complex_types,
    complex_vector_types,
    crow,
    cscalar,
    ctensor3,
    ctensor4,
    ctensor5,
    ctensor6,
    ctensor7,
    cvector,
    dcol,
    dcols,
    dmatrices,
    dmatrix,
    drow,
    drows,
    dscalar,
    dscalars,
    dtensor3,
    dtensor3s,
    dtensor4,
    dtensor4s,
    dtensor5,
    dtensor5s,
    dtensor6,
    dtensor6s,
    dtensor7,
    dtensor7s,
    dvector,
    dvectors,
    fcol,
    fcols,
    float_matrix_types,
    float_scalar_types,
    float_types,
    float_vector_types,
    fmatrices,
    fmatrix,
    frow,
    frows,
    fscalar,
    fscalars,
    ftensor3,
    ftensor3s,
    ftensor4,
    ftensor4s,
    ftensor5,
    ftensor5s,
    ftensor6,
    ftensor6s,
    ftensor7,
    ftensor7s,
    fvector,
    fvectors,
    icol,
    icols,
    imatrices,
    imatrix,
    int_matrix_types,
    int_scalar_types,
    int_types,
    int_vector_types,
    irow,
    irows,
    iscalar,
    iscalars,
    itensor3,
    itensor3s,
    itensor4,
    itensor4s,
    itensor5,
    itensor5s,
    itensor6,
    itensor6s,
    itensor7,
    itensor7s,
    ivector,
    ivectors,
    lcol,
    lcols,
    lmatrices,
    lmatrix,
    lrow,
    lrows,
    lscalar,
    lscalars,
    ltensor3,
    ltensor3s,
    ltensor4,
    ltensor4s,
    ltensor5,
    ltensor5s,
    ltensor6,
    ltensor6s,
    ltensor7,
    ltensor7s,
    lvector,
    lvectors,
    matrices,
    matrix,
    row,
    rows,
    scalar,
    scalars,
    tensor,
    tensor3,
    tensor3s,
    tensor4,
    tensor4s,
    tensor5,
    tensor5s,
    tensor6,
    tensor6s,
    tensor7,
    tensor7s,
    values_eq_approx_always_true,
    vector,
    vectors,
    wcol,
    wrow,
    wscalar,
    wtensor3,
    wtensor4,
    wtensor5,
    wtensor6,
    wtensor7,
    wvector,
    zcol,
    zmatrix,
    zrow,
    zscalar,
    ztensor3,
    ztensor4,
    ztensor5,
    ztensor6,
    ztensor7,
    zvector,
)
from theano.tensor.type_other import *
