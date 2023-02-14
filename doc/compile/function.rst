
.. _usingfunction:

===========================================
:mod:`function` - defines aesara.function
===========================================

.. module:: aesara.compile.function
   :platform: Unix, Windows
   :synopsis: defines aesara.function and related classes
.. moduleauthor:: LISA

Guide
=====

This module provides :func:`function`, commonly accessed as `aesara.function`,
the interface for compiling graphs into callable objects.

You've already seen example usage in the basic tutorial... something like this:

>>> import aesara
>>> x = aesara.tensor.dscalar()
>>> f = aesara.function([x], 2*x)
>>> f(4)
array(8.0)

The idea here is that we've compiled the symbolic graph (``2*x``) into a function that can be called on a number and will do some computations.

The behaviour of function can be controlled in several ways, such as
:class:`In`, :class:`Out`, ``mode``, ``updates``, and ``givens``.

Computing More than one Thing at the Same Time
==============================================

Aesara supports functions with multiple outputs. For example, we can
compute the element-wise difference, absolute difference, and
squared difference between two matrices ``a`` and ``b`` at the same time:

.. If you modify this code, also change :
.. tests/test_tutorial.py:T_examples.test_examples_3

>>> a, b = at.dmatrices('a', 'b')
>>> diff = a - b
>>> abs_diff = abs(diff)
>>> diff_squared = diff**2
>>> f = aesara.function([a, b], [diff, abs_diff, diff_squared])

.. note::
   `dmatrices` produces as many outputs as names that you provide.  It is a
   shortcut for allocating symbolic variables that we will often use in the
   tutorials.

When we use the function ``f``, it returns the three variables (the printing
was reformatted for readability):

>>> f([[1, 1], [1, 1]], [[0, 1], [2, 3]])
[array([[ 1.,  0.],
       [-1., -2.]]), array([[ 1.,  0.],
       [ 1.,  2.]]), array([[ 1.,  0.],
       [ 1.,  4.]])]


Setting a Default Value for an Argument
=======================================

Let's say you want to define a function that adds two numbers, except
that if you only provide one number, the other input is assumed to be
one. You can do it like this:

.. If you modify this code, also change :
.. tests/test_tutorial.py:T_examples.test_examples_6

>>> from aesara.compile.io import In
>>> from aesara import function
>>> x, y = at.dscalars('x', 'y')
>>> z = x + y
>>> f = function([x, In(y, value=1)], z)
>>> f(33)
array(34.0)
>>> f(33, 2)
array(35.0)

This makes use of the :ref:`In <function_inputs>` class which allows
you to specify properties of your function's parameters with greater detail. Here we
give a default value of ``1`` for ``y`` by creating a :class:`In` instance with
its ``value`` field set to ``1``.

Inputs with default values must follow inputs without default values (like
Python's functions).  There can be multiple inputs with default values. These
parameters can be set positionally or by name, as in standard Python:


.. If you modify this code, also change :
.. tests/test_tutorial.py:T_examples.test_examples_7

>>> x, y, w = at.dscalars('x', 'y', 'w')
>>> z = (x + y) * w
>>> f = function([x, In(y, value=1), In(w, value=2, name='w_by_name')], z)
>>> f(33)
array(68.0)
>>> f(33, 2)
array(70.0)
>>> f(33, 0, 1)
array(33.0)
>>> f(33, w_by_name=1)
array(34.0)
>>> f(33, w_by_name=1, y=0)
array(33.0)

.. note::
   `In` does not know the name of the local variables ``y`` and ``w``
   that are passed as arguments.  The symbolic variable objects have name
   attributes (set by `dscalars` in the example above) and *these* are the
   names of the keyword parameters in the functions that we build.  This is
   the mechanism at work in ``In(y, value=1)``.  In the case of ``In(w,
   value=2, name='w_by_name')``. We override the symbolic variable's name
   attribute with a name to be used for this function.


You may like to see :ref:`Function<usingfunction>` in the library for more detail.


Copying functions
=================
Aesara functions can be copied, which can be useful for creating similar
functions but with different shared variables or updates. This is done using
the :func:`aesara.compile.function.types.Function.copy` method of :class:`Function` objects.
The optimized graph of the original function is copied, so compilation only
needs to be performed once.

Let's start from the accumulator defined above:

>>> import aesara
>>> import aesara.tensor as at
>>> state = aesara.shared(0)
>>> inc = at.iscalar('inc')
>>> accumulator = aesara.function([inc], state, updates=[(state, state+inc)])

We can use it to increment the state as usual:

>>> accumulator(10)
array(0)
>>> print(state.get_value())
10

We can use :meth:`copy` to create a similar accumulator but with its own internal state
using the ``swap`` parameter, which is a dictionary of shared variables to exchange:

>>> new_state = aesara.shared(0)
>>> new_accumulator = accumulator.copy(swap={state:new_state})
>>> new_accumulator(100)
[array(0)]
>>> print(new_state.get_value())
100

The state of the first function is left untouched:

>>> print(state.get_value())
10

We now create a copy with updates removed using the ``delete_updates``
parameter, which is set to ``False`` by default:

>>> null_accumulator = accumulator.copy(delete_updates=True)

As expected, the shared state is no longer updated:

>>> null_accumulator(9000)
[array(10)]
>>> print(state.get_value())
10
Reference
=========

.. class:: In

    A class for attaching information to function inputs.

    .. attribute:: variable

        A variable in an expression graph to use as a compiled-function parameter

    .. attribute:: name

        A string to identify an argument for this parameter in keyword arguments.

    .. attribute:: value

        The default value to use at call-time (can also be a Container where
        the function will find a value at call-time.)

    .. attribute:: update

        An expression which indicates updates to the Value after each function call.

    .. attribute:: mutable

        ``True`` means the compiled-function is allowed to modify this
        argument. ``False`` means it is not allowed.

    .. attribute:: borrow

        ``True`` indicates that a reference to internal storage may be returned, and that the caller is aware that subsequent function evaluations might overwrite this memory.

    .. attribute:: strict

      If ``False``, a function argument may be copied or cast to match the type
      required by the parameter `variable`.  If ``True``, a function argument
      must exactly match the type required by `variable`.

    .. attribute:: allow_downcast

        ``True`` indicates that the value you pass for this input can be silently downcasted to fit the right type, which may lose precision. (Only applies when `strict` is ``False``.)

    .. attribute:: autoname

        ``True`` means that the `name` is set to variable.name.

    .. attribute:: implicit

        ``True`` means that the input is implicit in the sense that the user is not allowed to provide a value for it. Requires 'value' to be set.
        ``False`` means that the user can provide a value for this input.

    .. method:: __init__(self, variable, name=None, value=None, update=None, mutable=None, strict=False, allow_downcast=None, autoname=True, implicit=None, borrow=None, shared=False)

        Initialize attributes from arguments.

.. class:: Out

    A class for attaching information to function outputs

    .. attribute:: variable

        A variable in an expression graph to use as a compiled-function
        output

    .. attribute:: borrow

        ``True`` indicates that a reference to internal storage may be returned, and that the caller is aware that subsequent function evaluations might overwrite this memory.

    .. method:: __init__(variable, borrow=False)

        Initialize attributes from arguments.


.. function:: function(inputs, outputs, mode=None, updates=None, givens=None, no_default_updates=False, accept_inplace=False, name=None, rebuild_strict=True, allow_input_downcast=None, profile=None, on_unused_input='raise')

    Return a :class:`callable object <aesara.compile.function.types.Function>` that will calculate `outputs` from `inputs`.

    :type params: list of either Variable or In instances, but not shared
        variables.

    :param params: the returned :class:`Function` instance will have
      parameters for these variables.

    :type outputs: list of Variables or Out instances

    :param outputs: expressions to compute.

    :type mode: None, string or :class:`Mode` instance.

    :param mode: compilation mode

    :type updates: iterable over pairs (shared_variable, new_expression).
       List, tuple or dict.

    :param updates: expressions for new :class:`SharedVariable` values

    :type givens: iterable over pairs (Var1, Var2) of Variables.
       List, tuple or dict.  The Var1
       and Var2 in each pair must have the same Type.

    :param givens: specific substitutions to make in the
      computation graph (Var2 replaces Var1).

    :type no_default_updates: either bool or list of Variables
    :param no_default_updates:
        if True, do not perform any automatic update on Variables.
        If False (default), perform them all.
        Else, perform automatic updates on all Variables that are
        neither in ``updates`` nor in ``no_default_updates``.

    :param name: an optional name for this function.
      The profile mode will print the time spent in this function.

    :param rebuild_strict: True (Default) is the safer and better
        tested setting, in which case `givens` must substitute new
        variables with the same Type as the variables they replace.
        False is a you-better-know-what-you-are-doing setting, that
        permits `givens` to replace variables with new variables of
        any Type.  The consequence of changing a Type is that all
        results depending on that variable may have a different Type
        too (the graph is rebuilt from inputs to outputs).  If one of
        the new types does not make sense for one of the Ops in the
        graph, an Exception will be raised.

    :type allow_input_downcast: Boolean or None
    :param allow_input_downcast: True means that the values passed as
        inputs when calling the function can be silently downcasted to
        fit the dtype of the corresponding Variable, which may lose
        precision.  False means that it will only be cast to a more
        general, or precise, type. None (default) is almost like
        False, but allows downcasting of Python float scalars to
        floatX.

    :type profile: None, True, or ProfileStats instance
    :param profile: accumulate profiling information into a given
        ProfileStats instance. If argument is `True` then a new
        ProfileStats instance will be used.  This profiling object
        will be available via self.profile.

    :param on_unused_input: What to do if a variable in the 'inputs'
        list is not used in the graph. Possible values are 'raise',
        'warn', and 'ignore'.

    :rtype: :class:`Function <aesara.compile.function.types.Function>`
            instance

    :returns: a callable object that will compute the outputs (given the inputs)
      and update the implicit function arguments according to the `updates`.


    Inputs can be given as variables or :class:`In` instances.
    :class:`In` instances also have a variable, but they attach some extra
    information about how call-time arguments corresponding to that variable
    should be used.  Similarly, :class:`Out` instances can attach information
    about how output variables should be returned.

    The default is typically 'FAST_RUN' but this can be changed in
    :doc:`aesara.config <../config>`.  The mode
    argument controls the sort of rewrites that will be applied to the
    graph, and the way the rewritten graph will be evaluated.

    After each function evaluation, the `updates` mechanism can replace the
    value of any (implicit) `SharedVariable` inputs with new values computed
    from the expressions in the `updates` list.  An exception will be raised
    if you give two update expressions for the same `SharedVariable` input (that
    doesn't make sense).

    If a `SharedVariable` is not given an update expression, but has a
    :attr:`Variable.default_update` member containing an expression, this expression
    will be used as the update expression for this variable.  Passing
    ``no_default_updates=True`` to ``function`` disables this behavior
    entirely, passing ``no_default_updates=[sharedvar1, sharedvar2]``
    disables it for the mentioned variables.

    Regarding givens: Be careful to make sure that these substitutions are
    independent, because behaviour when ``Var1`` of one pair appears in the graph leading
    to ``Var2`` in another expression is undefined (e.g. with ``{a: x, b: a + 1}``).
    Replacements specified with givens are different from replacements that
    occur during normal rewriting, in that ``Var2`` is not expected to be
    equivalent to ``Var1``.

.. autofunction:: aesara.compile.function.function_dump

.. autoclass:: aesara.compile.function.types.Function
   :members: free, copy, __call__
