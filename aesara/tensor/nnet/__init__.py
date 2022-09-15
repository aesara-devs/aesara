import warnings


warnings.warn(
    "The module `aesara.tensor.nnet` is deprecated and will "
    "be removed from Aesara in version 2.9.0",
    DeprecationWarning,
    stacklevel=2,
)

import aesara.tensor.nnet.rewriting
from aesara.tensor.nnet.abstract_conv import (
    abstract_conv2d,
    conv2d,
    conv2d_grad_wrt_inputs,
    conv2d_transpose,
    conv3d,
    separable_conv2d,
)
from aesara.tensor.nnet.basic import (
    binary_crossentropy,
    categorical_crossentropy,
    confusion_matrix,
    crossentropy_categorical_1hot,
    crossentropy_categorical_1hot_grad,
    crossentropy_softmax_1hot,
    crossentropy_softmax_1hot_with_bias,
    crossentropy_softmax_1hot_with_bias_dx,
    crossentropy_softmax_argmax_1hot_with_bias,
    crossentropy_softmax_max_and_argmax_1hot,
    crossentropy_softmax_max_and_argmax_1hot_with_bias,
    crossentropy_to_crossentropy_with_softmax,
    crossentropy_to_crossentropy_with_softmax_with_bias,
    elu,
    graph_merge_softmax_with_crossentropy_softmax,
    h_softmax,
    logsoftmax,
    prepend_0_to_each_row,
    prepend_1_to_each_row,
    prepend_scalar_to_each_row,
    relu,
    selu,
    sigmoid_binary_crossentropy,
    softmax,
    softmax_grad_legacy,
    softmax_legacy,
    softmax_simplifier,
    softmax_with_bias,
    softsign,
)
from aesara.tensor.nnet.batchnorm import batch_normalization
from aesara.tensor.nnet.sigm import hard_sigmoid, ultra_fast_sigmoid
