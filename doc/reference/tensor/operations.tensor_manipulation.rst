.. _reference_tensor_tensor_manipulation:
.. currentmodule:: aesara.tensor

Tensor manipulation
===================

Basic operation
---------------

.. autosummary::
   :toctree: _autosummary

   shape
   shape.shape_tuple


Casting
-------

.. autosummary::
   :toctree: _autosummary

   cast
   real
   imag


Updating elements
-----------------

.. attention::

   Index assignment is **not** supported in Aesara. If you want to do the equivalent of ``a[5] = b`` or ``a[5]+=b`` you will need to use the ``set_subtensor`` or ``inc_subtensor`` operator respectively.


.. autosummary::
   :toctree: _autosummary

   set_subtensor
   inc_subtensor
   fill_diagonal_offset

Changing tensor shape
---------------------

.. admonition:: Specific to Aesara

   The ``specify_shape`` operator is specific to Aesara.

.. autosummary::
   :toctree: _autosummary

   reshape
   flatten
   specify_shape


Transpose-like operations
-------------------------

.. autosummary::
   :toctree: _autosummary

   moveaxis
   swapaxes
   roll
   TensorVariable.T
   transpose

Changing number of dimensions
-----------------------------

.. autosummary::
   :toctree: _autosummary

   atleast_1d
   atleast_2d
   atleast_3d
   broadcast_to
   broadcast_arrays
   squeeze

Joining tensors
---------------

.. autosummary::
   :toctree: _autosummary

   join
   stack
   stacklists
   horizontal_stack
   vertical_stack
   concatenate

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

Splitting tensors
-----------------

.. autosummary::
   :toctree: _autosummary

   split

Tiling tensors
--------------

.. autosummary::
   :toctree: _autosummary

   tile
   repeat

Adding and removing elements
----------------------------

.. autosummary::
   :toctree: _autosummary

   unique

Rearranging elements
--------------------

.. autosummary::
   :toctree: _autosummary

    reshape
    flatten
    permute_row_elements
    inverse_permutation


Padding tensors
---------------

.. autosummary::
   :toctree: _autosummary

   shape_padleft
   shape_padright
   shape_padaxis
