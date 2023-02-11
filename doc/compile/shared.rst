
.. _libdoc_compile_shared:

===========================================
:mod:`shared` - defines aesara.shared
===========================================

.. module:: aesara.compile.sharedvalue
   :platform: Unix, Windows
   :synopsis: defines aesara.shared and related classes
.. moduleauthor:: LISA

Using Shared Variables
======================

It is possible to make a function with an internal state. For
example, let's say we want to make an accumulator: at the beginning,
the state is initialized to zero, then, on each function call, the state
is incremented by the function's argument.

First let's define the *accumulator* function. It adds its argument to the
internal state and returns the old state value.

.. If you modify this code, also change :
.. tests/test_tutorial.py:T_examples.test_examples_8

>>> from aesara import shared
>>> state = shared(0)
>>> inc = at.iscalar('inc')
>>> accumulator = function([inc], state, updates=[(state, state+inc)])

This code introduces a few new concepts.  The ``shared`` function constructs
so-called :ref:`shared variables<libdoc_compile_shared>`.
These are hybrid symbolic and non-symbolic variables whose value may be shared
between multiple functions.  Shared variables can be used in symbolic expressions just like
the objects returned by `dmatrices` but they also have an internal
value that defines the value taken by this symbolic variable in *all* the
functions that use it.  It is called a *shared* variable because its value is
shared between many functions.  The value can be accessed and modified by the
:meth:`get_value` and :meth:`set_value` methods. We will come back to this soon.

The other new thing in this code is the ``updates`` parameter of :func:`aesara.function`.
``updates`` must be supplied with a list of pairs of the form (shared-variable, new expression).
It can also be a dictionary whose keys are shared-variables and values are
the new expressions.  Either way, it means "whenever this function runs, it
will replace the :attr:`value` of each shared variable with the result of the
corresponding expression".  Above, our accumulator replaces the ``state``'s value with the sum
of the state and the increment amount.

Let's try it out!

.. If you modify this code, also change :
.. tests/test_tutorial.py:T_examples.test_examples_8

>>> print(state.get_value())
0
>>> accumulator(1)
array(0)
>>> print(state.get_value())
1
>>> accumulator(300)
array(1)
>>> print(state.get_value())
301

It is possible to reset the state. Just use the ``.set_value()`` method:

>>> state.set_value(-1)
>>> accumulator(3)
array(-1)
>>> print(state.get_value())
2

As we mentioned above, you can define more than one function to use the same
shared variable.  These functions can all update the value.

.. If you modify this code, also change :
.. tests/test_tutorial.py:T_examples.test_examples_8

>>> decrementor = function([inc], state, updates=[(state, state-inc)])
>>> decrementor(2)
array(2)
>>> print(state.get_value())
0

You might be wondering why the updates mechanism exists.  You can always
achieve a similar result by returning the new expressions, and working with
them in NumPy as usual.  The updates mechanism can be a syntactic convenience,
but it is mainly there for efficiency.  Updates to shared variables can
sometimes be done more quickly using in-place algorithms (e.g. low-rank matrix
updates).

It may happen that you expressed some formula using a shared variable, but
you do *not* want to use its value. In this case, you can use the
``givens`` parameter of :func:`aesara.function` which replaces a particular node in a graph
for the purpose of one particular function.

.. If you modify this code, also change :
.. tests/test_tutorial.py:T_examples.test_examples_8

>>> fn_of_state = state * 2 + inc
>>> # The type of foo must match the shared variable we are replacing
>>> # with the ``givens``
>>> foo = at.scalar(dtype=state.dtype)
>>> skip_shared = function([inc, foo], fn_of_state, givens=[(state, foo)])
>>> skip_shared(1, 3)  # we're using 3 for the state, not state.value
array(7)
>>> print(state.get_value())  # old state still there, but we didn't use it
0

The ``givens`` parameter can be used to replace any symbolic variable, not just a
shared variable. You can replace constants, and expressions, in general.  Be
careful though, not to allow the expressions introduced by a ``givens``
substitution to be co-dependent, the order of substitution is not defined, so
the substitutions have to work in any order.

In practice, a good way of thinking about the ``givens`` is as a mechanism
that allows you to replace any part of your formula with a different
expression that evaluates to a tensor of same shape and dtype.

.. note::

    Aesara shared variable broadcast pattern default to ``False`` for each
    dimensions. Shared variable size can change over time, so we can't
    use the shape to find the broadcastable pattern. If you want a
    different pattern, just pass it as a parameter
    ``aesara.shared(..., broadcastable=(True, False))``

.. note::
    Use the ``shape`` parameter to specify tuples of static shapes instead;
    the old broadcastable values are being phased-out.  Unknown shape values
    for dimensions take the value ``None``; otherwise, integers are used for
    known static shape values.
    For example, ``aesara.shared(..., shape=(1, None))``.

Reference
=========

.. class:: SharedVariable

    Variable with storage that is shared between the compiled functions that it appears in.
    These variables are meant to be created by registered *shared constructors*
    (see :func:`shared_constructor`).

    The user-friendly constructor is :func:`shared`

    .. method:: get_value(self, borrow=False, return_internal_type=False)

       :param borrow: True to permit returning of an object aliased to internal memory.
       :type borrow: bool

       :param return_internal_type: True to permit the returning of an arbitrary type object used
               internally to store the shared variable.
       :type return_internal_type: bool

       By default, return a copy of the data. If ``borrow=True`` (and
       ``return_internal_type=False``), maybe it will return a copy.
       For tensor, it will always return an `ndarray` by default, so if
       the data is on another device, it will return a copy, but if the data
       is on the CPU, it will return the original data.  If you do
       ``borrow=True`` and ``return_internal_type=True``, it will
       always return the original data, not a copy, but this can be a non-`ndarray`
       type of object.

    .. method:: set_value(self, new_value, borrow=False)

       :param new_value: The new value.
       :type new_value: A compatible type for this shared variable.

       :param borrow: True to use the new_value directly, potentially creating problems
           related to aliased memory.
       :type borrow: bool

       The new value will be seen by all functions using this SharedVariable.

    .. method:: __init__(self, name, type, value, strict, container=None)

        :param name: The name for this variable.
        :type name: None or str

        :param type: The :term:`Type` for this Variable.

        :param value: A value to associate with this variable (a new container will be created).

        :param strict: True -> assignments to ``self.value`` will not be casted
          or copied, so they must have the correct type or an exception will be
          raised.

        :param container: The container to use for this variable.   This should
           instead of the `value` parameter.  Using both is an error.

    .. attribute:: container

        A container to use for this SharedVariable when it is an implicit function parameter.


.. autofunction:: shared

.. function:: shared_constructor(ctor)

    Append `ctor` to the list of shared constructors (see :func:`shared`).

    Each registered constructor `ctor` will be called like this:

    .. code-block:: python

        ctor(value, name=name, strict=strict, **kwargs)

    If it do not support given value, it must raise a `TypeError`.
