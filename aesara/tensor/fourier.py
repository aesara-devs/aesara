import math

import numpy as np

from aesara.graph.basic import Apply
from aesara.graph.op import Op
from aesara.tensor.basic import (
    arange,
    as_tensor_variable,
    concatenate,
    split,
    stack,
    switch,
)
from aesara.tensor.math import exp, lt, outer, tensordot
from aesara.tensor.shape import shape
from aesara.tensor.subtensor import set_subtensor
from aesara.tensor.type import TensorType, integer_dtypes
from aesara.tensor.var import TensorConstant


class Fourier(Op):
    """
    WARNING: for officially supported FFTs, use aesara.tensor.fft, which
    provides real-input FFTs. Gradients are supported, as well as optimization
    transfers to GPU ops.

    An instance of this class returns a finite fourier transform calcutated
    along one dimension of an input array.

    inputs:

    a : Array of at least one dimension.  Can be complex.
    n : Integer, optional. Length of the transformed axis of the output. If n
    is smaller than the length of the input, the input is cropped. If it is
    larger, the input is padded with zeros. If n is not given, the length of
    the input (along the axis specified by axis) is used.
    axis : Integer, optional. Axis over which to compute the FFT. If not
    supplied, the last axis is used.

    output:

    Complex array.  The input array, transformed along the axis
    indicated by 'axis' or along the last axis if 'axis' is not specified. It
    is truncated or zero-padded as required if 'n' is specified.
    (From numpy.fft.fft's documentation:)
    The values in the output follow so-called standard order. If A = fft(a, n),
    then A[0] contains the zero-frequency term (the mean of the signal), which
    is always purely real for real inputs. Then A[1:n/2] contains the
    positive-frequency terms, and A[n/2+1:] contains the negative-frequency
    terms, in order of decreasingly negative frequency. For an even number of
    input points, A[n/2] represents both positive and negative Nyquist
    frequency, and is also purely real for real input. For an odd number of
    input points, A[(n-1)/2] contains the largest positive frequency, while
    A[(n+1)/2] contains the largest negative frequency.
    """

    __props__ = ()

    def make_node(self, a, n, axis):
        a = as_tensor_variable(a)
        if a.ndim < 1:
            raise TypeError(
                f"{self.__class__.__name__}: input must be an array, not a scalar"
            )
        if axis is None:
            axis = a.ndim - 1
            axis = as_tensor_variable(axis)
        else:
            axis = as_tensor_variable(axis)
            if axis.dtype not in integer_dtypes:
                raise TypeError(
                    "%s: index of the transformed axis must be"
                    " of type integer" % self.__class__.__name__
                )
            elif axis.ndim != 0 or (
                isinstance(axis, TensorConstant)
                and (axis.data < 0 or axis.data > a.ndim - 1)
            ):
                raise TypeError(
                    f"{self.__class__.__name__}: index of the transformed axis must be"
                    " a scalar not smaller than 0 and smaller than"
                    " dimension of array"
                )
        if n is None:
            n = a.shape[axis]
            n = as_tensor_variable(n)
        else:
            n = as_tensor_variable(n)
            if n.dtype not in integer_dtypes:
                raise TypeError(
                    "%s: length of the transformed axis must be"
                    " of type integer" % self.__class__.__name__
                )
            elif n.ndim != 0 or (isinstance(n, TensorConstant) and n.data < 1):
                raise TypeError(
                    "%s: length of the transformed axis must be a"
                    " strictly positive scalar" % self.__class__.__name__
                )
        return Apply(
            self,
            [a, n, axis],
            [TensorType("complex128", a.type.broadcastable)()],
        )

    def infer_shape(self, fgraph, node, in_shapes):
        shape_a = in_shapes[0]
        n = node.inputs[1]
        axis = node.inputs[2]
        if len(shape_a) == 1:
            return [(n,)]
        elif isinstance(axis, TensorConstant):
            out_shape = (
                list(shape_a[0 : axis.data.item()])
                + [n]
                + list(shape_a[axis.data + 1 :])
            )
        else:
            l = len(shape_a)
            shape_a = stack(shape_a)
            out_shape = concatenate((shape_a[0:axis], [n], shape_a[axis + 1 :]))
            n_splits = [1] * l
            out_shape = split(out_shape, n_splits, l)
            out_shape = [a[0] for a in out_shape]
        return [out_shape]

    def perform(self, node, inputs, output_storage):
        a = inputs[0]
        n = inputs[1]
        axis = inputs[2]
        output_storage[0][0] = np.fft.fft(a, n=int(n), axis=axis.item())

    def grad(self, inputs, cost_grad):
        """
        In defining the gradient, the Finite Fourier Transform is viewed as
        a complex-differentiable function of a complex variable
        """
        a = inputs[0]
        n = inputs[1]
        axis = inputs[2]
        grad = cost_grad[0]
        if not isinstance(axis, TensorConstant):
            raise NotImplementedError(
                f"{self.__class__.__name__}: gradient is currently implemented"
                " only for axis being an Aesara constant"
            )
        axis = int(axis.data)
        # notice that the number of actual elements in wrto is independent of
        # possible padding or truncation:
        elem = arange(0, shape(a)[axis], 1)
        # accounts for padding:
        freq = arange(0, n, 1)
        outer_res = outer(freq, elem)
        pow_outer = exp(((-2 * math.pi * 1j) * outer_res) / (1.0 * n))
        res = tensordot(grad, pow_outer, (axis, 0))

        # This would be simpler but not implemented by aesara:
        # res = switch(lt(n, shape(a)[axis]),
        # set_subtensor(res[...,n::], 0, False, False), res)

        # Instead we resort to that to account for truncation:
        flip_shape = list(np.arange(0, a.ndim)[::-1])
        res = res.dimshuffle(flip_shape)
        res = switch(
            lt(n, shape(a)[axis]),
            set_subtensor(
                res[
                    n::,
                ],
                0,
                False,
                False,
            ),
            res,
        )
        res = res.dimshuffle(flip_shape)

        # insures that gradient shape conforms to input shape:
        out_shape = (
            list(np.arange(0, axis)) + [a.ndim - 1] + list(np.arange(axis, a.ndim - 1))
        )
        res = res.dimshuffle(*out_shape)
        return [res, None, None]


fft = Fourier()
