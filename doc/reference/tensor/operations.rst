.. _reference_tensor_operations:

Tensor operations
=================

.. currentmodule:: aesara.tensor

The module :mod:`aesara.tensor` allows to create tensors and express symbolic calculations using the NumPy and SciPy API.

Aesara's API tries to mirror NumPy's, so in most cases it is safe to assume that the basic NumPy array functions and methods will be available. If you find an inconsistency, or if a function is missing, please open an `Issue <https://github.com/aesara-devs/aesara>`__.

Shaping and Shuffling
---------------------

To re-order the dimensions of a variable, to insert or remove broadcastable
dimensions, see :meth:`_tensor_py_operators.dimshuffle`.

.. function:: shape(x)

    Returns an `lvector` representing the shape of `x`.

.. function:: reshape(x, newshape, ndim=None)
   :noindex:

    :type x: any `TensorVariable` (or compatible)
    :param x: variable to be reshaped

    :type newshape: `lvector` (or compatible)
    :param newshape: the new shape for `x`

    :param ndim: optional - the length that `newshape`'s value will have.
        If this is ``None``, then `reshape` will infer it from `newshape`.

    :rtype: variable with `x`'s dtype, but `ndim` dimensions

    .. note::

        This function can infer the length of a symbolic `newshape` value in
        some cases, but if it cannot and you do not provide the `ndim`, then
        this function will raise an Exception.


.. function:: shape_padleft(x, n_ones=1)

    Reshape `x` by left padding the shape with `n_ones` 1s. Note that all
    this new dimension will be broadcastable. To make them non-broadcastable
    see the :func:`unbroadcast`.

    :param x: variable to be reshaped
    :type x: any `TensorVariable` (or compatible)

    :type n_ones: int
    :type n_ones: number of dimension to be added to `x`



.. function:: shape_padright(x, n_ones=1)

    Reshape `x` by right padding the shape with `n_ones` ones. Note that all
    this new dimension will be broadcastable. To make them non-broadcastable
    see the :func:`unbroadcast`.

    :param x: variable to be reshaped
    :type x: any TensorVariable (or compatible)

    :type n_ones: int
    :type n_ones: number of dimension to be added to `x`


.. function:: shape_padaxis(t, axis)

    Reshape `t` by inserting ``1`` at the dimension `axis`. Note that this new
    dimension will be broadcastable. To make it non-broadcastable
    see the :func:`unbroadcast`.

    :type x: any `TensorVariable` (or compatible)
    :param x: variable to be reshaped

    :type axis: int
    :param  axis: axis where to add the new dimension to `x`

    Example:

    >>> tensor = aesara.tensor.type.tensor3()
    >>> aesara.tensor.shape_padaxis(tensor, axis=0)
    InplaceDimShuffle{x,0,1,2}.0
    >>> aesara.tensor.shape_padaxis(tensor, axis=1)
    InplaceDimShuffle{0,x,1,2}.0
    >>> aesara.tensor.shape_padaxis(tensor, axis=3)
    InplaceDimShuffle{0,1,2,x}.0
    >>> aesara.tensor.shape_padaxis(tensor, axis=-1)
    InplaceDimShuffle{0,1,2,x}.0

.. function:: flatten(x, ndim=1)

    Similar to :func:`reshape`, but the shape is inferred from the shape of `x`.

    :param x: variable to be flattened
    :type x: any `TensorVariable` (or compatible)

    :type ndim: int
    :param ndim: the number of dimensions in the returned variable

    :rtype: variable with same dtype as `x` and `ndim` dimensions
    :returns: variable with the same shape as `x` in the leading `ndim-1`
        dimensions, but with all remaining dimensions of `x` collapsed into
        the last dimension.

    For example, if we flatten a tensor of shape ``(2, 3, 4, 5)`` with ``flatten(x,
    ndim=2)``, then we'll have the same (i.e. ``2-1=1``) leading dimensions
    ``(2,)``, and the remaining dimensions are collapsed, so the output in this
    example would have shape ``(2, 60)``.


.. function:: tile(x, reps, ndim=None)

    Construct an array by repeating the input `x` according to `reps`
    pattern.

    Tiles its input according to `reps`. The length of `reps` is the
    number of dimension of `x` and contains the number of times to
    tile `x` in each dimension.

    :see: `numpy.tile
        <http://docs.scipy.org/doc/numpy/reference/generated/numpy.tile.html>`_
        documentation for examples.

    :see: :func:`aesara.tensor.extra_ops.repeat
        <aesara.tensor.extra_ops.repeat>`

    :note: Currently, `reps` must be a constant, `x.ndim` and
        ``len(reps)`` must be equal and, if specified, `ndim` must be
        equal to both.

.. autofunction:: roll


Creating Tensors
----------------

.. function:: zeros_like(x, dtype=None)

    :param x: tensor that has the same shape as output
    :param dtype: data-type, optional
                  By default, it will be x.dtype.

    Returns a tensor the shape of `x` filled with zeros of the type of `dtype`.


.. function:: ones_like(x)


    :param x: tensor that has the same shape as output
    :param dtype: data-type, optional
                  By default, it will be `x.dtype`.

    Returns a tensor the shape of `x` filled with ones of the type of `dtype`.


.. function:: zeros(shape, dtype=None)

    :param shape: a tuple/list of scalar with the shape information.
    :param dtype: the dtype of the new tensor. If ``None``, will use ``"floatX"``.

    Returns a tensor filled with zeros of the provided shape.

.. function:: ones(shape, dtype=None)

    :param shape: a tuple/list of scalar with the shape information.
    :param dtype: the dtype of the new tensor. If ``None``, will use ``"floatX"``.

    Returns a tensor filled with ones of the provided shape.

.. function:: fill(a,b)

    :param a: tensor that has same shape as output
    :param b: Aesara scalar or value with which you want to fill the output

    Create a matrix by filling the shape of `a` with `b`.

.. function:: alloc(value, *shape)

    :param value: a value with which to fill the output
    :param shape: the dimensions of the returned array
    :returns: an N-dimensional tensor initialized by `value` and having the specified shape.

.. function:: eye(n, m=None, k=0, dtype=aesara.config.floatX)

    :param n: number of rows in output (value or Aesara scalar)
    :param m: number of columns in output (value or Aesara scalar)
    :param k: Index of the diagonal: ``0`` refers to the main diagonal,
              a positive value refers to an upper diagonal, and a
              negative value to a lower diagonal. It can be an Aesara
              scalar.
    :returns: An array where all elements are equal to zero, except for the `k`-th
              diagonal, whose values are equal to one.

.. function:: identity_like(x, dtype=None)

    :param x: tensor
    :param dtype: The dtype of the returned tensor. If `None`, default to dtype of `x`
    :returns: A tensor of same shape as `x` that is filled with zeros everywhere
              except for the main diagonal, whose values are equal to one. The output
              will have same dtype as `x` unless overridden in `dtype`.

.. function:: stack(tensors, axis=0)

    Stack tensors in sequence on given axis (default is ``0``).

    Take a sequence of tensors and stack them on given axis to make a single
    tensor. The size in dimension `axis` of the result will be equal to the number
    of tensors passed.

    :param tensors: a list or a tuple of one or more tensors of the same rank.
    :param axis: the axis along which the tensors will be stacked. Default value is ``0``.
    :returns: A tensor such that ``rval[0] == tensors[0]``, ``rval[1] == tensors[1]``, etc.

    Examples:

    >>> a = aesara.tensor.type.scalar()
    >>> b = aesara.tensor.type.scalar()
    >>> c = aesara.tensor.type.scalar()
    >>> x = aesara.tensor.stack([a, b, c])
    >>> x.ndim # x is a vector of length 3.
    1
    >>> a = aesara.tensor.type.tensor4()
    >>> b = aesara.tensor.type.tensor4()
    >>> c = aesara.tensor.type.tensor4()
    >>> x = aesara.tensor.stack([a, b, c])
    >>> x.ndim # x is a 5d tensor.
    5
    >>> rval = x.eval(dict((t, np.zeros((2, 2, 2, 2))) for t in [a, b, c]))
    >>> rval.shape # 3 tensors are stacked on axis 0
    (3, 2, 2, 2, 2)

    We can also specify different axis than default value ``0``:

    >>> x = aesara.tensor.stack([a, b, c], axis=3)
    >>> x.ndim
    5
    >>> rval = x.eval(dict((t, np.zeros((2, 2, 2, 2))) for t in [a, b, c]))
    >>> rval.shape # 3 tensors are stacked on axis 3
    (2, 2, 2, 3, 2)
    >>> x = aesara.tensor.stack([a, b, c], axis=-2)
    >>> x.ndim
    5
    >>> rval = x.eval(dict((t, np.zeros((2, 2, 2, 2))) for t in [a, b, c]))
    >>> rval.shape # 3 tensors are stacked on axis -2
    (2, 2, 2, 3, 2)

.. function:: stack(*tensors)
   :noindex:

    .. warning::

        The interface `stack(*tensors)` is deprecated!  Use
        `stack(tensors, axis=0)` instead.

    Stack tensors in sequence vertically (row wise).

    Take a sequence of tensors and stack them vertically to make a single
    tensor.

    :param tensors: one or more tensors of the same rank
    :returns: A tensor such that ``rval[0] == tensors[0]``, ``rval[1] == tensors[1]``, etc.

    >>> x0 = at.scalar()
    >>> x1 = at.scalar()
    >>> x2 = at.scalar()
    >>> x = at.stack(x0, x1, x2)
    >>> x.ndim # x is a vector of length 3.
    1

.. function:: concatenate(tensor_list, axis=0)

    :type tensor_list: a list or tuple of Tensors that all have the same shape in the axes
                        *not* specified by the `axis` argument.
    :param tensor_list: one or more Tensors to be concatenated together into one.
    :type axis: literal or symbolic integer
    :param axis: Tensors will be joined along this axis, so they may have different
        ``shape[axis]``

    >>> x0 = at.fmatrix()
    >>> x1 = at.ftensor3()
    >>> x2 = at.fvector()
    >>> x = at.concatenate([x0, x1[0], at.shape_padright(x2)], axis=1)
    >>> x.ndim
    2

.. function:: stacklists(tensor_list)

    :type tensor_list: an iterable that contains either tensors or other
        iterables of the same type as `tensor_list` (in other words, this
        is a tree whose leaves are tensors).
    :param tensor_list: tensors to be stacked together.

    Recursively stack lists of tensors to maintain similar structure.

    This function can create a tensor from a shaped list of scalars:

    >>> from aesara.tensor import stacklists, scalars, matrices
    >>> from aesara import function
    >>> a, b, c, d = scalars('abcd')
    >>> X = stacklists([[a, b], [c, d]])
    >>> f = function([a, b, c, d], X)
    >>> f(1, 2, 3, 4)
    array([[ 1.,  2.],
           [ 3.,  4.]])

    We can also stack arbitrarily shaped tensors. Here we stack matrices into
    a 2 by 2 grid:

    >>> from numpy import ones
    >>> a, b, c, d = matrices('abcd')
    >>> X = stacklists([[a, b], [c, d]])
    >>> f = function([a, b, c, d], X)
    >>> x = ones((4, 4), 'float32')
    >>> f(x, x, x, x).shape
    (2, 2, 4, 4)

.. autofunction:: aesara.tensor.basic.choose


Reductions
----------


.. function:: max(x, axis=None, keepdims=False)

    :Parameter: *x* -  symbolic Tensor (or compatible)
    :Parameter: *axis* - axis or axes along which to compute the maximum
    :Parameter: *keepdims* - (boolean) If this is set to True, the axes which are reduced are
		left in the result as dimensions with size one. With this option, the result
		will broadcast correctly against the original tensor.
    :Returns: maximum of *x* along *axis*

    axis can be:
     * *None* - in which case the maximum is computed along all axes (like NumPy)
     * an *int* - computed along this axis
     * a *list of ints* - computed along these axes

.. function:: argmax(x, axis=None, keepdims=False)

    :Parameter: *x* - symbolic Tensor (or compatible)
    :Parameter: *axis* - axis along which to compute the index of the maximum
    :Parameter: *keepdims* - (boolean) If this is set to True, the axis which is reduced is
		left in the result as a dimension with size one. With this option, the result
		will broadcast correctly against the original tensor.
    :Returns: the index of the maximum value along a given axis

    if ``axis == None``, `argmax` over the flattened tensor (like NumPy)

.. function:: max_and_argmax(x, axis=None, keepdims=False)

    :Parameter: *x* - symbolic Tensor (or compatible)
    :Parameter: *axis* - axis along which to compute the maximum and its index
    :Parameter: *keepdims* - (boolean) If this is set to True, the axis which is reduced is
		left in the result as a dimension with size one. With this option, the result
		will broadcast correctly against the original tensor.
    :Returns: the maximum value along a given axis and its index.

    if ``axis == None``, `max_and_argmax` over the flattened tensor (like NumPy)

.. function:: min(x, axis=None, keepdims=False)

    :Parameter: *x* -  symbolic Tensor (or compatible)
    :Parameter: *axis* - axis or axes along which to compute the minimum
    :Parameter: *keepdims* - (boolean) If this is set to True, the axes which are reduced are
		left in the result as dimensions with size one. With this option, the result
		will broadcast correctly against the original tensor.
    :Returns: minimum of *x* along *axis*

    `axis` can be:
     * ``None`` - in which case the minimum is computed along all axes (like NumPy)
     * an *int* - computed along this axis
     * a *list of ints* - computed along these axes

.. function:: argmin(x, axis=None, keepdims=False)

    :Parameter: *x* - symbolic Tensor (or compatible)
    :Parameter: *axis* - axis along which to compute the index of the minimum
    :Parameter: *keepdims* - (boolean) If this is set to True, the axes which are reduced are
		left in the result as dimensions with size one. With this option, the result
		will broadcast correctly against the original tensor.
    :Returns: the index of the minimum value along a given axis

    if ``axis == None``, `argmin` over the flattened tensor (like NumPy)

.. function:: sum(x, axis=None, dtype=None, keepdims=False, acc_dtype=None)

    :Parameter: *x* -  symbolic Tensor (or compatible)
    :Parameter: *axis* - axis or axes along which to compute the sum
    :Parameter: *dtype* - The dtype of the returned tensor.
        If None, then we use the default dtype which is the same as
        the input tensor's dtype except when:

        - the input dtype is a signed integer of precision < 64 bit, in
          which case we use int64
        - the input dtype is an unsigned integer of precision < 64 bit, in
          which case we use uint64

        This default dtype does _not_ depend on the value of "acc_dtype".

    :Parameter: *keepdims* - (boolean) If this is set to True, the axes which are reduced are
		left in the result as dimensions with size one. With this option, the result
		will broadcast correctly against the original tensor.

    :Parameter: *acc_dtype* -  The dtype of the internal accumulator.
        If None (default), we use the dtype in the list below,
        or the input dtype if its precision is higher:

        - for int dtypes, we use at least int64;
        - for uint dtypes, we use at least uint64;
        - for float dtypes, we use at least float64;
        - for complex dtypes, we use at least complex128.

    :Returns: sum of *x* along *axis*

    `axis` can be:
     * ``None`` - in which case the sum is computed along all axes (like NumPy)
     * an int - computed along this axis
     * a list of ints - computed along these axes

.. function:: prod(x, axis=None, dtype=None, keepdims=False, acc_dtype=None, no_zeros_in_input=False)

    :Parameter: *x* -  symbolic Tensor (or compatible)
    :Parameter: *axis* - axis or axes along which to compute the product
    :Parameter: *dtype* - The dtype of the returned tensor.
        If None, then we use the default dtype which is the same as
        the input tensor's dtype except when:

        - the input dtype is a signed integer of precision < 64 bit, in
          which case we use int64
        - the input dtype is an unsigned integer of precision < 64 bit, in
          which case we use uint64

        This default dtype does _not_ depend on the value of "acc_dtype".

    :Parameter: *keepdims* - (boolean) If this is set to True, the axes which are reduced are
		left in the result as dimensions with size one. With this option, the result
		will broadcast correctly against the original tensor.

    :Parameter: *acc_dtype* -  The dtype of the internal accumulator.
        If None (default), we use the dtype in the list below,
        or the input dtype if its precision is higher:

        - for int dtypes, we use at least int64;
        - for uint dtypes, we use at least uint64;
        - for float dtypes, we use at least float64;
        - for complex dtypes, we use at least complex128.

    :Parameter: *no_zeros_in_input* - The grad of prod is complicated
         as we need to handle 3 different cases: without zeros in the
         input reduced group, with 1 zero or with more zeros.

	 This could slow you down, but more importantly, we currently
	 don't support the second derivative of the 3 cases. So you
	 cannot take the second derivative of the default prod().

	 To remove the handling of the special cases of 0 and so get
	 some small speed up and allow second derivative set
	 ``no_zeros_in_inputs`` to ``True``. It defaults to ``False``.

	 **It is the user responsibility to make sure there are no zeros
	 in the inputs. If there are, the grad will be wrong.**

    :Returns: product of every term in *x* along *axis*

    `axis` can be:
     * ``None`` - in which case the sum is computed along all axes (like NumPy)
     * an int - computed along this axis
     * a list of ints - computed along these axes

.. function:: mean(x, axis=None, dtype=None, keepdims=False, acc_dtype=None)

    :Parameter: *x* -  symbolic Tensor (or compatible)
    :Parameter: *axis* - axis or axes along which to compute the mean
    :Parameter: *dtype* - The dtype to cast the result of the inner summation into.
        For instance, by default, a sum of a float32 tensor will be
        done in float64 (acc_dtype would be float64 by default),
        but that result will be casted back in float32.
    :Parameter: *keepdims* - (boolean) If this is set to True, the axes which are reduced are
		left in the result as dimensions with size one. With this option, the result
		will broadcast correctly against the original tensor.
    :Parameter: *acc_dtype* -  The dtype of the internal accumulator of the
        inner summation. This will not necessarily be the dtype of the
        output (in particular if it is a discrete (int/uint) dtype, the
        output will be in a float type).  If None, then we use the same
        rules as :func:`sum()`.
    :Returns: mean value of *x* along *axis*

    `axis` can be:
     * ``None`` - in which case the mean is computed along all axes (like NumPy)
     * an int - computed along this axis
     * a list of ints - computed along these axes

.. function:: var(x, axis=None, keepdims=False)

    :Parameter: *x* -  symbolic Tensor (or compatible)
    :Parameter: *axis* - axis or axes along which to compute the variance
    :Parameter: *keepdims* - (boolean) If this is set to True, the axes which are reduced are
		left in the result as dimensions with size one. With this option, the result
		will broadcast correctly against the original tensor.
    :Returns: variance of *x* along *axis*

    `axis` can be:
     * ``None`` - in which case the variance is computed along all axes (like NumPy)
     * an int - computed along this axis
     * a list of ints - computed along these axes

.. function:: std(x, axis=None, keepdims=False)

    :Parameter: *x* -  symbolic Tensor (or compatible)
    :Parameter: *axis* - axis or axes along which to compute the standard deviation
    :Parameter: *keepdims* - (boolean) If this is set to True, the axes which are reduced are
		left in the result as dimensions with size one. With this option, the result
		will broadcast correctly against the original tensor.
    :Returns: variance of *x* along *axis*

    `axis` can be:
     * ``None`` - in which case the standard deviation is computed along all axes (like NumPy)
     * an int - computed along this axis
     * a list of ints - computed along these axes

.. function:: all(x, axis=None, keepdims=False)

    :Parameter: *x* -  symbolic Tensor (or compatible)
    :Parameter: *axis* - axis or axes along which to apply 'bitwise and'
    :Parameter: *keepdims* - (boolean) If this is set to True, the axes which are reduced are
		left in the result as dimensions with size one. With this option, the result
		will broadcast correctly against the original tensor.
    :Returns: bitwise and of *x* along *axis*

    `axis` can be:
     * ``None`` - in which case the 'bitwise and' is computed along all axes (like NumPy)
     * an int - computed along this axis
     * a list of ints - computed along these axes

.. function:: any(x, axis=None, keepdims=False)

    :Parameter: *x* -  symbolic Tensor (or compatible)
    :Parameter: *axis* - axis or axes along which to apply bitwise or
    :Parameter: *keepdims* - (boolean) If this is set to True, the axes which are reduced are
		left in the result as dimensions with size one. With this option, the result
		will broadcast correctly against the original tensor.
    :Returns: bitwise or of *x* along *axis*

    `axis` can be:
     * ``None`` - in which case the 'bitwise or' is computed along all axes (like NumPy)
     * an int - computed along this axis
     * a list of ints - computed along these axes

.. function:: ptp(x, axis = None)

    Range of values (maximum - minimum) along an axis.
    The name of the function comes from the acronym for peak to peak.

    :Parameter: *x* Input tensor.

    :Parameter: *axis* Axis along which to find the peaks. By default,
                flatten the array.

    :Returns: A new array holding the result.

.. _indexing:

Indexing
--------

Like NumPy, Aesara distinguishes between *basic* and *advanced* indexing.
Aesara fully supports basic indexing
(see `NumPy's indexing  <http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html>`_)
and `integer advanced indexing
<http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#integer>`_.

Index-assignment is *not* supported.  If you want to do something like ``a[5]
= b`` or ``a[5]+=b``, see :func:`aesara.tensor.subtensor.set_subtensor` and
:func:`aesara.tensor.subtensor.inc_subtensor` below.

.. autofunction:: aesara.tensor.subtensor.set_subtensor

.. autofunction:: aesara.tensor.subtensor.inc_subtensor

.. _tensor_operator_support:

Operator Support
----------------

Many Python operators are supported.

>>> a, b = at.itensor3(), at.itensor3() # example inputs

Arithmetic
----------

.. doctest::
   :options: +SKIP

   >>> a + 3      # at.add(a, 3) -> itensor3
   >>> 3 - a      # at.sub(3, a)
   >>> a * 3.5    # at.mul(a, 3.5) -> ftensor3 or dtensor3 (depending on casting)
   >>> 2.2 / a    # at.truediv(2.2, a)
   >>> 2.2 // a   # at.intdiv(2.2, a)
   >>> 2.2**a     # at.pow(2.2, a)
   >>> b % a      # at.mod(b, a)

Bitwise
-------

.. doctest::
   :options: +SKIP

   >>> a & b      # at.and_(a,b)    bitwise and (alias at.bitwise_and)
   >>> a ^ 1      # at.xor(a,1)     bitwise xor (alias at.bitwise_xor)
   >>> a | b      # at.or_(a,b)     bitwise or (alias at.bitwise_or)
   >>> ~a         # at.invert(a)    bitwise invert (alias at.bitwise_not)

Inplace
-------

In-place operators are *not* supported.  Aesara's graph rewrites
will determine which intermediate values to use for in-place
computations.  If you would like to update the value of a
:term:`shared variable`, consider using the ``updates`` argument to
:func:`Aesara.function`.

.. _libdoc_tensor_elemwise:

:class:`Elemwise`

Casting
-------

.. function:: cast(x, dtype)

    Cast any tensor `x` to a tensor of the same shape, but with a different
    numerical type `dtype`.

    This is not a reinterpret cast, but a coercion `cast`, similar to
    ``numpy.asarray(x, dtype=dtype)``.

    .. testcode:: cast

        import aesara.tensor as at
        x = at.matrix()
        x_as_int = at.cast(x, 'int32')

    Attempting to casting a complex value to a real value is ambiguous and
    will raise an exception.  Use `real`, `imag`, `abs`, or `angle`.

.. function:: real(x)

    Return the real (not imaginary) components of tensor `x`.
    For non-complex `x` this function returns `x`.

.. function:: imag(x)

    Return the imaginary components of tensor `x`.
    For non-complex `x` this function returns ``zeros_like(x)``.


Comparisons
-----------

The six usual equality and inequality operators share the same interface.
  :Parameter:  *a* - symbolic Tensor (or compatible)
  :Parameter:  *b* - symbolic Tensor (or compatible)
  :Return type: symbolic Tensor
  :Returns: a symbolic tensor representing the application of the logical :class:`Elemwise` operator.

  .. note::

    Aesara has no boolean dtype.  Instead, all boolean tensors are represented
    in ``'int8'``.

  Here is an example with the less-than operator.

  .. testcode:: oper

    import aesara.tensor as at
    x,y = at.dmatrices('x','y')
    z = at.le(x,y)

.. function:: lt(a, b)

    Returns a symbolic ``'int8'`` tensor representing the result of logical less-than (a<b).

    Also available using syntax ``a < b``

.. function:: gt(a, b)

    Returns a symbolic ``'int8'`` tensor representing the result of logical greater-than (a>b).

    Also available using syntax ``a > b``

.. function:: le(a, b)

    Returns a variable representing the result of logical less than or equal (a<=b).

    Also available using syntax ``a <= b``

.. function:: ge(a, b)

    Returns a variable representing the result of logical greater or equal than (a>=b).

    Also available using syntax ``a >= b``

.. function:: eq(a, b)

    Returns a variable representing the result of logical equality (a==b).

.. function:: neq(a, b)

    Returns a variable representing the result of logical inequality (a!=b).

.. function:: isnan(a)

    Returns a variable representing the comparison of ``a`` elements with nan.

    This is equivalent to ``numpy.isnan``.

.. function:: isinf(a)

    Returns a variable representing the comparison of ``a`` elements
    with inf or -inf.

    This is equivalent to ``numpy.isinf``.

.. function:: isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)

    Returns a symbolic ``'int8'`` tensor representing where two tensors are equal
    within a tolerance.

    The tolerance values are positive, typically very small numbers.
    The relative difference `(rtol * abs(b))` and the absolute difference `atol` are
    added together to compare against the absolute difference between `a` and `b`.

    For finite values, isclose uses the following equation to test whether two
    floating point values are equivalent:
    ``|a - b| <= (atol + rtol * |b|)``

    For infinite values, isclose checks if both values are the same signed inf value.

    If equal_nan is True, isclose considers NaN values in the same position to be close.
    Otherwise, NaN values are not considered close.

    This is equivalent to ``numpy.isclose``.

.. function:: allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)

    Returns a symbolic ``'int8'`` value representing if all elements in two tensors are equal
    within a tolerance.

    See notes in `isclose` for determining values equal within a tolerance.

    This is equivalent to ``numpy.allclose``.

Condition
---------

.. function:: switch(cond, ift, iff)

    Returns a variable representing a switch between ift (i.e. "if true") and iff (i.e. "if false")
    based on the condition cond. This is the Aesara equivalent of `numpy.where`.

      :Parameter:  *cond* - symbolic Tensor (or compatible)
      :Parameter:  *ift* - symbolic Tensor (or compatible)
      :Parameter:  *iff* - symbolic Tensor (or compatible)
      :Return type: symbolic Tensor

    .. testcode:: switch

      import aesara.tensor as at
      a,b = at.dmatrices('a','b')
      x,y = at.dmatrices('x','y')
      z = at.switch(at.lt(a,b), x, y)

.. function:: where(cond, ift, iff)

   Alias for `switch`. where is the NumPy name.

.. function:: clip(x, min, max)

    Return a variable representing `x`, but with all elements greater than
    `max` clipped to `max` and all elements less than `min` clipped to `min`.

    Normal broadcasting rules apply to each of `x`, `min`, and `max`.

    Note that there is no warning for inputs that are the wrong way round
    (`min > max`), and that results in this case may differ from `numpy.clip`.

Bit-wise
--------


The bitwise operators possess this interface:
    :Parameter:  *a* - symbolic tensor of integer type.
    :Parameter:  *b* - symbolic tensor of integer type.

    .. note::

        The bitwise operators must have an integer type as input.

        The bit-wise not (invert) takes only one parameter.

    :Return type: symbolic tensor with corresponding dtype.

.. function:: and_(a, b)

    Returns a variable representing the result of the bitwise and.

.. function:: or_(a, b)

    Returns a variable representing the result of the bitwise or.

.. function:: xor(a, b)

    Returns a variable representing the result of the bitwise xor.

.. function:: invert(a)

    Returns a variable representing the result of the bitwise not.

.. function:: bitwise_and(a, b)

   Alias for `and_`. bitwise_and is the NumPy name.

.. function:: bitwise_or(a, b)

   Alias for `or_`. bitwise_or is the NumPy name.

.. function:: bitwise_xor(a, b)

   Alias for `xor_`. bitwise_xor is the NumPy name.

.. function:: bitwise_not(a, b)

   Alias for invert. invert is the NumPy name.

Here is an example using the bit-wise ``and_`` via the ``&`` operator:

.. testcode:: bitwise

    import aesara.tensor as at
    x,y = at.imatrices('x','y')
    z = x & y


Mathematical
------------

.. function:: abs(a)

    Returns a variable representing the absolute of ``a``, i.e. ``|a|``.

    .. note:: Can also be accessed using `builtins.abs`: i.e. ``abs(a)``.

.. function:: angle(a)

    Returns a variable representing angular component of complex-valued Tensor ``a``.

.. function:: exp(a)

    Returns a variable representing the exponential of ``a``.

.. function:: maximum(a, b)

   Returns a variable representing the maximum element by element of a and b

.. function:: minimum(a, b)

   Returns a variable representing the minimum element by element of a and b

.. function:: neg(a)

    Returns a variable representing the negation of ``a`` (also ``-a``).

.. function:: reciprocal(a)

    Returns a variable representing the inverse of a, ie 1.0/a. Also called reciprocal.

.. function:: log(a), log2(a), log10(a)

    Returns a variable representing the base e, 2 or 10 logarithm of a.

.. function:: sgn(a)

    Returns a variable representing the sign of a.

.. function:: ceil(a)

    Returns a variable representing the ceiling of a (for example ceil(2.1) is 3).

.. function:: floor(a)

    Returns a variable representing the floor of a (for example floor(2.9) is 2).

.. function:: round(a, mode="half_away_from_zero")
   :noindex:

    Returns a variable representing the rounding of a in the same dtype as a. Implemented rounding mode are half_away_from_zero and half_to_even.

.. function:: iround(a, mode="half_away_from_zero")

    Short hand for cast(round(a, mode),'int64').

.. function:: sqr(a)

    Returns a variable representing the square of a, ie a^2.

.. function:: sqrt(a)

    Returns a variable representing the of a, ie a^0.5.

.. function:: cos(a), sin(a), tan(a)

    Returns a variable representing the trigonometric functions of a (cosine, sine and tangent).

.. function:: cosh(a), sinh(a), tanh(a)

    Returns a variable representing the hyperbolic trigonometric functions of a (hyperbolic cosine, sine and tangent).

.. function:: erf(a), erfc(a)

    Returns a variable representing the error function or the complementary error function. `wikipedia <http://en.wikipedia.org/wiki/Error_function>`__

.. function:: erfinv(a), erfcinv(a)

    Returns a variable representing the inverse error function or the inverse complementary error function. `wikipedia <http://en.wikipedia.org/wiki/Error_function#Inverse_functions>`__

.. function:: gamma(a)

   Returns a variable representing the gamma function.

.. function:: gammaln(a)

   Returns a variable representing the logarithm of the gamma function.

.. function:: psi(a)

   Returns a variable representing the derivative of the logarithm of
   the gamma function (also called the digamma function).

.. function:: chi2sf(a, df)

   Returns a variable representing the survival function (1-cdf —
   sometimes more accurate).

   C code is provided in the Theano_lgpl repository.
   This makes it faster.

   https://github.com/Theano/Theano_lgpl.git


Linear Algebra
--------------

.. function:: dot(X, Y)

     For 2-D arrays it is equivalent to matrix multiplication, and for
     1-D arrays to inner product of vectors (without complex
     conjugation). For N dimensions it is a sum product over the last
     axis of a and the second-to-last of b:

    :param X: left term
    :param Y: right term
    :type X: symbolic tensor
    :type Y: symbolic tensor
    :rtype: symbolic matrix or vector
    :return: the inner product of `X` and `Y`.

.. function:: outer(X, Y)

    :param X: left term
    :param Y: right term
    :type X: symbolic vector
    :type Y: symbolic vector
    :rtype: symbolic matrix

    :return: vector-vector outer product

.. function:: tensordot(a, b, axes=2)

    Given two tensors a and b,tensordot computes a generalized dot product over
    the provided axes. Aesara's implementation reduces all expressions to
    matrix or vector dot products and is based on code from Tijmen Tieleman's
    `gnumpy` (http://www.cs.toronto.edu/~tijmen/gnumpy.html).

    :param a: the first tensor variable
    :type a: symbolic tensor

    :param b: the second tensor variable
    :type b: symbolic tensor

    :param axes: an integer or array. If an integer, the number of axes
                 to sum over. If an array, it must have two array
                 elements containing the axes to sum over in each tensor.

                 Note that the default value of 2 is not guaranteed to work
                 for all values of a and b, and an error will be raised if
                 that is the case. The reason for keeping the default is to
                 maintain the same signature as NumPy's tensordot function
                 (and np.tensordot raises analogous errors for non-compatible
                 inputs).

                 If an integer i, it is converted to an array containing
                 the last i dimensions of the first tensor and the first
                 i dimensions of the second tensor:

                     axes = [range(a.ndim - i, b.ndim), range(i)]

                 If an array, its two elements must contain compatible axes
                 of the two tensors. For example, [[1, 2], [2, 0]] means sum
                 over the 2nd and 3rd axes of a and the 3rd and 1st axes of b.
                 (Remember axes are zero-indexed!) The 2nd axis of a and the
                 3rd axis of b must have the same shape; the same is true for
                 the 3rd axis of a and the 1st axis of b.
    :type axes: int or array-like of length 2

    :returns: a tensor with shape equal to the concatenation of a's shape
              (less any dimensions that were summed over) and b's shape
              (less any dimensions that were summed over).
    :rtype: symbolic tensor

    It may be helpful to consider an example to see what tensordot does.
    Aesara's implementation is identical to NumPy's. Here a has shape (2, 3, 4)
    and b has shape (5, 6, 4, 3). The axes to sum over are [[1, 2], [3, 2]] --
    note that a.shape[1] == b.shape[3] and a.shape[2] == b.shape[2]; these axes
    are compatible. The resulting tensor will have shape (2, 5, 6) -- the
    dimensions that are not being summed:

    .. testcode:: tensordot

        import numpy as np

        a = np.random.random((2,3,4))
        b = np.random.random((5,6,4,3))

        c = np.tensordot(a, b, [[1,2],[3,2]])

        a0, a1, a2 = a.shape
        b0, b1, _, _ = b.shape
        cloop = np.zeros((a0,b0,b1))

        # Loop over non-summed indices--these exist in the tensor product
        for i in range(a0):
            for j in range(b0):
                for k in range(b1):
                    # Loop over summed indices--these don't exist in the tensor product
                    for l in range(a1):
                        for m in range(a2):
                            cloop[i,j,k] += a[i,l,m] * b[j,k,m,l]

        assert np.allclose(c, cloop)

    This specific implementation avoids a loop by transposing a and b such that
    the summed axes of a are last and the summed axes of b are first. The
    resulting arrays are reshaped to 2 dimensions (or left as vectors, if
    appropriate) and a matrix or vector dot product is taken. The result is
    reshaped back to the required output dimensions.

    In an extreme case, no axes may be specified. The resulting tensor
    will have shape equal to the concatenation of the shapes of a and b:

    .. doctest:: tensordot

        >>> c = np.tensordot(a, b, 0)
        >>> a.shape
        (2, 3, 4)
        >>> b.shape
        (5, 6, 4, 3)
        >>> print(c.shape)
        (2, 3, 4, 5, 6, 4, 3)

    :note: See the documentation of `numpy.tensordot <http://docs.scipy.org/doc/numpy/reference/generated/numpy.tensordot.html>`_ for more examples.

.. function:: batched_dot(X, Y)

    :param x: A Tensor with sizes e.g.: for  3D (dim1, dim3, dim2)
    :param y: A Tensor with sizes e.g.: for 3D (dim1, dim2, dim4)

    This function computes the dot product between the two tensors, by iterating
    over the first dimension using scan.
    Returns a tensor of size e.g. if it is 3D: (dim1, dim3, dim4)
    Example:

    >>> first = at.tensor3('first')
    >>> second = at.tensor3('second')
    >>> result = batched_dot(first, second)

    :note:  This is a subset of `numpy.einsum`, but we do not provide it for now.

    :param X: left term
    :param Y: right term
    :type X: symbolic tensor
    :type Y: symbolic tensor

    :return: tensor of products

.. function:: batched_tensordot(X, Y, axes=2)

    :param x: A Tensor with sizes e.g.: for 3D (dim1, dim3, dim2)
    :param y: A Tensor with sizes e.g.: for 3D (dim1, dim2, dim4)
    :param axes: an integer or array. If an integer, the number of axes
                 to sum over. If an array, it must have two array
                 elements containing the axes to sum over in each tensor.

                 If an integer i, it is converted to an array containing
                 the last i dimensions of the first tensor and the first
                 i dimensions of the second tensor (excluding the first
                 (batch) dimension)::

                     axes = [range(a.ndim - i, b.ndim), range(1,i+1)]

                 If an array, its two elements must contain compatible axes
                 of the two tensors. For example, [[1, 2], [2, 4]] means sum
                 over the 2nd and 3rd axes of a and the 3rd and 5th axes of b.
                 (Remember axes are zero-indexed!) The 2nd axis of a and the
                 3rd axis of b must have the same shape; the same is true for
                 the 3rd axis of a and the 5th axis of b.
    :type axes: int or array-like of length 2

    :returns: a tensor with shape equal to the concatenation of a's shape
              (less any dimensions that were summed over) and b's shape
              (less first dimension and any dimensions that were summed over).
    :rtype: tensor of tensordots

    A hybrid of batch_dot and tensordot, this function computes the
    tensordot product between the two tensors, by iterating over the
    first dimension using scan to perform a sequence of tensordots.

    :note: See :func:`tensordot` and :func:`batched_dot` for
        supplementary documentation.

.. function:: mgrid

    :returns: an instance which returns a dense (or fleshed out) mesh-grid
              when indexed, so that each returned argument has the same shape.
              The dimensions and number of the output arrays are equal to the
              number of indexing dimensions. If the step length is not a complex
	      number, then the stop is not inclusive.

    Example:

    >>> a = at.mgrid[0:5, 0:3]
    >>> a[0].eval()
    array([[0, 0, 0],
           [1, 1, 1],
           [2, 2, 2],
           [3, 3, 3],
           [4, 4, 4]])
    >>> a[1].eval()
    array([[0, 1, 2],
           [0, 1, 2],
           [0, 1, 2],
           [0, 1, 2],
           [0, 1, 2]])

.. function:: ogrid

    :returns: an instance which returns an open (i.e. not fleshed out) mesh-grid
              when indexed, so that only one dimension of each returned array is
              greater than 1. The dimension and number of the output arrays are
              equal to the number of indexing dimensions. If the step length is
              not a complex number, then the stop is not inclusive.

    Example:

    >>> b = at.ogrid[0:5, 0:3]
    >>> b[0].eval()
    array([[0],
           [1],
           [2],
           [3],
           [4]])
    >>> b[1].eval()
    array([[0, 1, 2]])
.. autosummary::
   :toctree: _autosummary

    alloc
    choose
    concatenate
    eye
    fill
    flatten
    identity_like
    ones
    ones_like
    reshape
    roll
    shape
    shape_padleft
    shape_padright
    shape_padaxis
    stack
    stacklists
    tile
    zeros
    zeros_like
    max
    argmax
    max_and_argmax
    min
    argmin
    sum
    prod
    mean
    var
    std
    all
    any
    ptp
    set_subtensor
    inc_subtensor
    cast
    real
    imag
    lt
    gt
    le
    ge
    eq
    neq
    isnan
    isinf
    isclose
    allclose
    switch
    where
    clip
    and_
    or_
    xor
    invert
    bitwise_and
    bitwise_or
    bitwise_xor
    bitwise_not
    abs
    angle
    exp
    maximum
    minimum
    neg
    minimum
    reciprocal
    log
    sgn
    ceil
    floor
    round
    iround
    sqr
    sqrt
    cos
    cosh
    erf
    erfinv
    gamma
    gammaln
    psi
    chi2sf
    dot
    outer
    tensordot
    batched_dot
    batched_tensordot
    mgrid
    ogrid

aesara.tensor.fft
-----------------

.. automodule:: aesara.tensor.fft

.. autosummary::
   :toctree: _autosummary

    irfft
    rfft

aesara.tensor.nlinalg
---------------------

.. automodule:: aesara.tensor.nlinalg

.. autosummary::
   :toctree: _autosummary

   matrix_dot
   matrix_power
   pinv
   qr
   svd
   tensorinv
   tensorsolve
   trace

aesara.tensor.slinalg
---------------------

.. automodule:: aesara.tensor.slinalg

.. autosummary::
   :toctree: _autosummary

    kron
    solve
    solve_lower_triangular
    solve_upper_triangular

aesara.tensor.elemwise
----------------------

.. automodule:: aesara.tensor.elemwise

.. autosummary::
   :toctree: _autosummary

   CAReduce
   DimShuffle
   Elemwise
   scalar_elemwise

aesara.tensor.extra_ops
-----------------------

.. automodule:: aesara.tensor.extra_ops

.. autosummary::
   :toctree: _autosummary

    bartlett
    bincount
    broadcast_arrays
    broadcast_shape
    broadcast_to
    compress
    cumprod
    cumsum
    diff
    fill_diagonal
    fill_diagonal_offset
    ravel_multi_index
    repeat
    searchsorted
    squeeze
    unique
    unravel_index
