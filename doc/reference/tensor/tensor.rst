.. _reference_tensor_objects:

Tensor objects
==============

.. currentmodule:: aesara.tensor


A ``TensorVariable`` represents a multidimensional container of items of the same type and shape. The number of dimensions and items in a tensor is defined by its shape, a tuple of *N* non-negative integers or `None` values that specify the sizes of each dimension (when known). The type of items in the array is determined by its :attr:`dtype <TensorVariable.dtype>` attribute.

.. admonition:: Example

    A 2-dimensional array of size 2x2 composed of integer elements:

    >>> x = at.as_tensor([[2, 3], [3, 4]], "int8")
    >>> x.type.ndim
    2
    >>> x.type.shape
    (2, 2)
    >>> x.type.dtype
    'int8'

    A matrix of unknown dimension size:

    >>> import aesara.tensor as at
    >>> x = at.matrix('x')
    >>> x.type.ndim
    2
    >>> x.type.shape
    (None, None)
    >>> x.type.dtype
    'float32'

    Note that Aesara returns `None` when the shape of the tensor is unknown.

Constructing tensor variables
-----------------------------

New tensors can be constructed using the routines defined in :ref:`reference_tensor_create`, or at a low-level by calling the ``TensorType`` type:

>>> import aesara as at
>>> mytype = at.TensorType("int32", shape=(None, 3))
>>> x = mytype()


.. autosummary::
   :toctree: _autosummary

   TensorType


Tensor attributes
-----------------

.. autosummary::
   :toctree: _autosummary

   TensorVariable.T
   TensorVariable.transpose
   TensorVariable.shape
   TensorVariable.size
   TensorVariable.astype
   TensorVariable.__getitem__
   TensorVariable.take
   TensorVariable.copy
   TensorVariable.__iter__
   TensorVariable.ndim
   TensorVariable.dtype
   TensorVariable.type
   TensorVariable.ndim
   TensorVariable.dtype
   TensorVariable.dimshuffle

Tensor methods
--------------

Tensor comparison
-----------------

Tensor operations
-----------------

.. autosummary::
   :toctree: _autosummary

   TensorVariable.__abs__
   TensorVariable.__neg__
   TensorVariable.__lt__
   TensorVariable.__le__
   TensorVariable.__gt__
   TensorVariable.__ge__
   TensorVariable.__invert__
   TensorVariable.__and__
   TensorVariable.__or__
   TensorVariable.__xor__
   TensorVariable.__rand__
   TensorVariable.__ror__
   TensorVariable.__rxor__

.. autosummary::
   :toctree: _autosummary

   TensorVariable.__add__
   TensorVariable.__sub__
   TensorVariable.__mul__
   TensorVariable.__div__
   TensorVariable.__pow__
   TensorVariable.__mod__
   TensorVariable.__divmod__
   TensorVariable.__truediv__
   TensorVariable.__floordiv__
   TensorVariable.__rtruediv__
   TensorVariable.__rfloordiv__
   TensorVariable.__radd__
   TensorVariable.__rsub__
   TensorVariable.__rmul__
   TensorVariable.__rdiv__
   TensorVariable.__rpow__
   TensorVariable.__rmod__
   TensorVariable.__rdivmod__
   TensorVariable.__ceil__
   TensorVariable.__floor__
   TensorVariable.__trunc__



.. autosummary::
   :toctree: _autosummary

   TensorVariable.any
   TensorVariable.reshape
   TensorVariable.dimshuffle
   TensorVariable.flatten
   TensorVariable.ravel
   TensorVariable.diagonal
   TensorVariable.transfer

.. autosummary::
   :toctree: _autosummary

   TensorVariable.arccos
   TensorVariable.arccosh
   TensorVariable.arcsin
   TensorVariable.arcsinh
   TensorVariable.arctan
   TensorVariable.arctanh
   TensorVariable.ceil
   TensorVariable.cos
   TensorVariable.cosh
   TensorVariable.deg2rad
   TensorVariable.exp
   TensorVariable.exp2
   TensorVariable.expm1
   TensorVariable.floor
   TensorVariable.log
   TensorVariable.log10
   TensorVariable.log1p
   TensorVariable.log2
   TensorVariable.rad2deg
   TensorVariable.sin
   TensorVariable.sinh
   TensorVariable.sqrt
   TensorVariable.tan
   TensorVariable.tanh
   TensorVariable.trunc

.. autosummary::
   :toctree: _autosummary

   TensorVariable.__dot__
   TensorVariable.__rdot__
   TensorVariable.sum
   TensorVariable.prod
   TensorVariable.norm
   TensorVariable.mean
   TensorVariable.var
   TensorVariable.std
   TensorVariable.max
   TensorVariable.argmin
   TensorVariable.argmax
   TensorVariable.nonzero
   TensorVariable.nonzero_values
   TensorVariable.sort
   TensorVariable.argsort
   TensorVariable.clip
   TensorVariable.conj
   TensorVariable.repeat
   TensorVariable.round
   TensorVariable.trace
   TensorVariable.get_scalar_constant_value
   TensorVariable.zeros_like
   TensorVariable.ones_like
   TensorVariable.cumsum
   TensorVariable.cumprod
   TensorVariable.searchsorted
   TensorVariable.ptp
   TensorVariable.swapaxes
   TensorVariable.fill
   TensorVariable.choose
   TensorVariable.squeeze
   TensorVariable.compress
   TensorVariable.real
   TensorVariable.imag


Numerical types
---------------

A string indicating the numerical type of the `ndarray` for which a
`Variable` of this `Type` represents.

.. _dtype_list:

The :attr:`dtype` attribute of a `TensorType` instance can be any of the
following strings.

================= =================== =================
dtype             domain              bits
================= =================== =================
``'int8'``        signed integer      8
``'int16'``       signed integer      16
``'int32'``       signed integer      32
``'int64'``       signed integer      64
``'uint8'``       unsigned integer    8
``'uint16'``      unsigned integer    16
``'uint32'``      unsigned integer    32
``'uint64'``      unsigned integer    64
``'float32'``     floating point      32
``'float64'``     floating point      64
``'complex64'``   complex             64 (two float32)
``'complex128'``  complex             128 (two float64)
================= =================== =================
