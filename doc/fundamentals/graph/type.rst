.. _aesara_type:

===============
:class:`Type`\s
===============

The :class:`Type` class is used to provide "static" information about the types of
:class:`Variable`\s in an Aesara graph.  This information is used for graph rewrites
and compilation to languages with typing that's stricter than Python's.

The types handled by Aesara naturally overlap a lot with NumPy, but
they also differ from it in some very important ways.  In the following, we use
:class:`TensorType` to illustrate some important concepts and functionality
regarding :class:`Type`, because it's the most common and feature rich subclass of
:class:`Type`.  Just be aware that all the same high-level concepts apply
to any other graph objects modeled by a :class:`Type` subclass.


The :class:`TensorType`
-----------------------

Aesara has a :class:`Type` subclass for tensors/arrays called :class:`TensorType`.
It broadly represents a type for tensors, but, more specifically, all of its
computations are performed using instances of the :class:`numpy.ndarray` class, so it
effectively represents the same objects as :class:`numpy.ndarray`.

The expression ``TensorType(dtype, shape)()`` will construct a symbolic
:class:`TensorVariable` instance (a subclass of :class:`Variable`), like
``numpy.ndarray(shape, dtype)`` will construct a :class:`numpy.ndarray`
instance.

Notice the extra parenthesis in the :class:`TensorType` example.  Those are
necessary because ``TensorType(dtype, shape)`` only constructs an instance of a
:class:`TensorType`, then, with that instance, a :class:`Variable` instance can
be constructed using :meth:`TensorType.__call__`.  Just remember that
:class:`Type` objects are *not* Python types/classes; they're instances of the
Python class :class:`Type`.  The purpose is effectively the same, though:
:class:`Type`\s provide high-level typing information and construct instances of
the high-level types they model.  While Python types/classes do this for the Python
VM, Aesara :class:`Type`\s do this for its effective "VM".

In relation to NumPy, the important difference is that Aesara works at the
symbolic level, and, because of that, there are no concrete array instances with
which it can call the :attr:`dtype` and :attr:`shape`
methods and get information about the data type or shape of a symbolic
variable.  Aesara needs static class/type-level information to serve that purpose,
and it can't use the :class:`numpy.ndarray` class itself, because that doesn't have
fixed data types or shapes.

In analogy with NumPy, we could imagine that the expression :class:`TensorType`
is a :class:`numpy.ndarray` class constructor like the following:

.. code::

   def NdarrayType(dtype, shape):
       class fixed_dtype_shape_ndarray(_numpy.ndarray):
           dtype = dtype
           shape = shape

           def __call__(self):
               return super().__call__(dtype, shape)

       return fixed_dtype_shape_ndarray


This hypothetical :class:`NdarrayType` would construct :class:`numpy.ndarray` subclasses that
produces instances with fixed data types and shapes.  Also, the subclasses
created by this class constructor, would provide data type
and shape information about the instances they produce without ever needing to
construct an actual instance (e.g. one can simply inspect the class-level
``shape`` and ``dtype`` members for that information).

:class:`TensorType`\s provide a way to carry around the same array information
at the type level, but they also perform comparisons and conversions
between and to different types.

For instance, :class:`TensorType`\s allow for _partial_ shape information.  In other words, the
shape values for some--or all--dimensions may be unspecified.  The only fixed requirement is that
the _number_ of dimensions be fixed/given (i.e. the length of the shape
``tuple``).  To encode partial shape information, :class:`TensorType` allows its
``shape`` arguments to include ``None``\s.

To illustrate, ``TensorType("float64", (2, None))`` could represent an array of
shape ``(2, 0)``, ``(2, 1)``, etc.  This dynamic opens up some questions
regarding the comparison of :class:`TensorType`\s.

For example, let's say we have two :class:`Variable`\s with the following
:class:`TensorType`\s:

>>> from aesara.tensor.type import TensorType
>>> v1 = TensorType("float64", (2, None))()
>>> v1.type
TensorType(float64, (2, ?))
>>> v2 = TensorType("float64", (2, 1))()
>>> v2.type
TensorType(float64, (2, 1))

If we ever wanted to replace ``v1`` in an Aesara graph with ``v2``, we would first
need to check that they're "compatible".  This could be done by noticing that
their shapes match everywhere except on the second dimension, where ``v1`` has the shape
value ``None`` and ``v2`` has a ``1``.  Since ``None`` indicates "any" shape
value, the two are "compatible" in some sense.

The "compatibility" we're describing here is really that ``v1``'s :class:`Type`
represents a larger set of arrays, and ``v2``'s represents a much more
specific subset, but both belong to the same set of array types.

:class:`Type` provides a generic interface for these kinds of comparisons with its
:meth:`Type.in_same_class` and :meth:`Type.is_super` methods.  These type-comparison methods
are in turn used by the :class:`Variable` conversion methods to "narrow" the
type information received at different stages of graph construction and
rewriting.

For example:

>>> v1.type.in_same_class(v2.type)
False

This result is due to the definition of "type class" used by
:class:`TensorType`.  Its definition is based on the broadcastable dimensions
(i.e. ``1``\s) in the available static shape information.  See the
docstring for :meth:`TensorType.in_same_class` for more information.

>>> v1.type.is_super(v2.type)
True

This result is due to the fact that ``v1.type`` models a superset of the types
that ``v2.type`` models, since ``v2.type`` is a type for arrays with the
specific shape ``(2, 1)`` and ``v1.type`` is a type for _all_ arrays with shape
``(2, N)`` for any ``N``--of which ``v2.type``'s type is only a single instance.

This relation is used to "filter" :class:`Variable`\s through specific :class:`Type`\s
in order to generate a new :class:`Variable` that's compatible with both.  This "filtering"
is an important step in the node replacement process during graph rewriting, for instance.

>>> v1.type.filter_variable(v2)
<TensorType(float64, (2, 1))>

"Filtering" returned a variable of the same :class:`Type` as ``v2``, because ``v2``'s :class:`Type` is
more specific/informative than ``v1``'s--and both are compatible.

>>> v3 = v2.type.filter_variable(v1)
>>> v3
SpecifyShape.0
>>> import aesara
>>> aesara.dprint(v3, print_type=True)
SpecifyShape [id A] <TensorType(float64, (2, 1))>
 |<TensorType(float64, (2, ?))> [id B] <TensorType(float64, (2, ?))>
 |TensorConstant{2} [id C] <TensorType(int8, ())>
 |TensorConstant{1} [id D] <TensorType(int8, ())>


Performing this in the opposite direction returned the output of a
:class:`SpecifyShape`\ :class:`Op`.  This :class:`SpecifyShape` uses ``v1`` static shape as an
input and serves to produce a new :class:`Variable` that has a :class:`Type` compatible with
both ``v1`` and ``v2``.


.. note::
   The :class:`Type` for ``v3`` should really have a static shape of ``(2, 1)``
   (i.e. ``v2``'s shape), but the static shape information feature is still
   under development.

It's important to keep these special type comparisons in mind when developing custom
:class:`Op`\s and graph rewrites in Aesara, because simple naive comparisons
like ``v1.type == v2.type`` may unnecessarily restrict logic and prevent
more refined type information from propagating throughout a graph.  They may not
cause errors, but they could prevent Aesara from performing at its best.


.. _type_contract:

:class:`Type`'s contract
========================

In Aesara's framework, a :class:`Type` is any object which defines the following
methods. To obtain the default methods described below, the :class:`Type` should be an
instance of `Type` or should be an instance of a subclass of `Type`. If you
will write all methods yourself, you need not use an instance of `Type`.

Methods with default arguments must be defined with the same signature,
i.e.  the same default argument names and values. If you wish to add
extra arguments to any of these methods, these extra arguments must have
default values.

.. autoclass:: aesara.graph.type.Type
    :noindex:

    .. automethod:: in_same_class
      :noindex:

    .. automethod:: is_super
      :noindex:

    .. method:: filter_inplace(value, storage, strict=False, allow_downcast=None)
      :noindex:

      If filter_inplace is defined, it will be called instead of
      filter() This is to allow reusing the old allocated memory. This was used
      only when new data was transferred to a shared variable on a GPU.

      ``storage`` will be the old value (e.g. the old `ndarray`).


    .. method:: is_valid_value(value)
      :noindex:

      Returns True iff the value is compatible with the :class:`Type`. If
      ``filter(value, strict = True)`` does not raise an exception, the
      value is compatible with the :class:`Type`.

      *Default:* True iff ``filter(value, strict=True)`` does not raise
      an exception.

    .. method:: values_eq(a, b)
      :noindex:

      Returns True iff ``a`` and ``b`` are equal.

      *Default:* ``a == b``

    .. method:: values_eq_approx(a, b)
      :noindex:

      Returns True iff ``a`` and ``b`` are approximately equal, for a
      definition of "approximately" which varies from :class:`Type` to :class:`Type`.

      *Default:* ``values_eq(a, b)``

    .. method:: make_variable(name=None)
      :noindex:

      Makes a :term:`Variable` of this :class:`Type` with the specified name, if
      ``name`` is not ``None``. If ``name`` is ``None``, then the `Variable` does
      not have a name. The `Variable` will have its ``type`` field set to
      the :class:`Type` object.

      *Default:* there is a generic definition of this in `Type`. The
      `Variable`'s ``type`` will be the object that defines this method (in
      other words, ``self``).

    .. method:: __call__(name=None)
      :noindex:

      Syntactic shortcut to ``make_variable``.

      *Default:* ``make_variable``

    .. method:: __eq__(other)
      :noindex:

      Used to compare :class:`Type` instances themselves

      *Default:* ``object.__eq__``

    .. method:: __hash__()
      :noindex:

      :class:`Type`\s should not be mutable, so it should be OK to define a hash
      function.  Typically this function should hash all of the terms
      involved in ``__eq__``.

      *Default:* ``id(self)``

    .. automethod:: clone
       :noindex:



.. autoclass:: aesara.tensor.type.TensorType
    :noindex:

    .. method:: may_share_memory(a, b)
      :noindex:

        Optional to run, but mandatory for `DebugMode`. Return ``True`` if the Python
        objects `a` and `b` could share memory. Return ``False``
        otherwise. It is used to debug when :class:`Op`\s did not declare memory
        aliasing between variables. Can be a static method.
        It is highly recommended to use and is mandatory for :class:`Type` in Aesara
        as our buildbot runs in `DebugMode`.


    .. method:: get_shape_info(obj)
      :noindex:

      Optional. Only needed to profile the memory of this :class:`Type` of object.

      Return the information needed to compute the memory size of ``obj``.

      The memory size is only the data, so this excludes the container.
      For an ndarray, this is the data, but not the ndarray object and
      other data structures such as shape and strides.

      ``get_shape_info()`` and ``get_size()`` work in tandem for the memory profiler.

      ``get_shape_info()`` is called during the execution of the function.
      So it is better that it is not too slow.

      ``get_size()`` will be called on the output of this function
      when printing the memory profile.

      :param obj: The object that this :class:`Type` represents during execution
      :return: Python object that ``self.get_size()`` understands


    .. method:: get_size(shape_info)
      :noindex:

        Number of bytes taken by the object represented by shape_info.

        Optional. Only needed to profile the memory of this :class:`Type` of object.

        :param shape_info: the output of the call to `get_shape_info`
        :return: the number of bytes taken by the object described by
            ``shape_info``.



Additional definitions
----------------------

For certain mechanisms, you can register functions and other such
things to plus your type into aesara's mechanisms.  These are optional
but will allow people to use you type with familiar interfaces.

`transfer`
~~~~~~~~~~

To plug in additional options for the transfer target, define a
function which takes an Aesara variable and a target argument and
returns eitehr a new transferred variable (which can be the same as
the input if no transfer is necessary) or returns None if the transfer
can't be done.

Then register that function by calling :func:`register_transfer()`
with it as argument.

An example
==========

We are going to base :class:`Type` ``DoubleType`` on Python's ``float``. We
must define ``filter`` and ``values_eq_approx``.


**filter**

.. testcode::

    # note that we shadow python's function ``filter`` with this
    # definition.
    def filter(x, strict=false, allow_downcast=none):
        if strict:
            if isinstance(x, float):
                return x
            else:
                raise typeerror('expected a float!')
        elif allow_downcast:
            return float(x)
        else:   # covers both the false and none cases.
            x_float = float(x)
            if x_float == x:
                return x_float
            else:
                  raise TypeError('The double type cannot accurately represent '
                                  f'value {x} (of type {type(x)}): you must explicitly '
                                  'allow downcasting if you want to do this.')

If ``strict`` is True we need to return ``x``. If ``strict`` is True and ``x`` is not a
``float`` (for example, ``x`` could easily be an ``int``) then it is
incompatible with our :class:`Type` and we must raise an exception.

If ``strict is False`` then we are allowed to cast ``x`` to a ``float``,
so if ``x`` is an ``int`` it we will return an equivalent ``float``.
However if this cast triggers a precision loss (``x != float(x)``) and
``allow_downcast`` is not True, then we also raise an exception.
Note that here we decided that the default behavior of our type
(when ``allow_downcast`` is set to ``None``) would be the same as
when ``allow_downcast`` is False, i.e. no precision loss is allowed.


**values_eq_approx**

.. testcode::

   def values_eq_approx(x, y, tolerance=1e-4):
       return abs(x - y) / (abs(x) + abs(y)) < tolerance

The second method we define is ``values_eq_approx``. This method
allows approximate comparison between two values respecting our :class:`Type`'s
constraints. It might happen that a rewrite changes the computation
graph in such a way that it produces slightly different variables, for
example because of numerical instability like rounding errors at the
end of the mantissa. For instance, ``a + a + a + a + a + a`` might not
actually produce the exact same output as ``6 * a`` (try with a=0.1),
but with ``values_eq_approx`` we do not necessarily mind.

We added an extra ``tolerance`` argument here. Since this argument is
not part of the API, it must have a default value, which we
chose to be 1e-4.

.. note::

    ``values_eq`` is never actually used by Aesara, but it might be used
    internally in the future. Equality testing in
    :ref:`DebugMode <debugmode>` is done using ``values_eq_approx``.

**Putting them together**

What we want is an object that respects the aforementioned
contract. Recall that :class:`Type` defines default implementations for all
required methods of the interface, except ``filter``.

.. code-block:: python

   from aesara.graph.type import Type

   class DoubleType(Type):

       def filter(self, x, strict=False, allow_downcast=None):
           # See code above.
           ...

       def values_eq_approx(self, x, y, tolerance=1e-4):
           # See code above.
           ...

   double = DoubleType()

``double`` is then an instance of :class:`Type`\ :class:`DoubleType`, which in turn is a
subclass of `Type`.

There is a small issue with our :class:`DoubleType`: all
instances of `DoubleType` are technically the same :class:`Type`; however, different
`DoubleType`\ :class:`Type` instances do not compare the same:

>>> double1 = DoubleType()
>>> double2 = DoubleType()
>>> double1 == double2
False

Aesara compares :class:`Type`\s using ``==`` to see if they are the same.
This happens in :class:`DebugMode`.  Also, :class:`Op`\s can (and should) ensure that their inputs
have the expected :class:`Type` by checking something like
``x.type.is_super(lvector)`` or ``x.type.in_same_class(lvector)``.

There are several ways to make sure that equality testing works properly:

 #. Define :meth:`DoubleType.__eq__` so that instances of type :class:`DoubleType`
    are equal. For example:

    .. testcode::

        def __eq__(self, other):
            return type(self) == type(other)

 #. Override :meth:`DoubleType.__new__` to always return the same instance.
 #. Hide the :class:`DoubleType` class and only advertise a single instance of it.

Here we will prefer the final option, because it is the simplest.
:class:`Op`\s in the Aesara code often define the :meth:`__eq__` method though.


Untangling some concepts
========================

Initially, confusion is common on what an instance of :class:`Type` is versus
a subclass of :class:`Type` or an instance of :class:`Variable`. Some of this confusion is
syntactic. A :class:`Type` is any object which has fields corresponding to the
functions defined above. The :class:`Type` class provides sensible defaults for
all of them except :meth:`Type.filter`, so when defining new :class:`Type`\s it is natural
to subclass :class:`Type`. Therefore, we often end up with :class:`Type` subclasses and
it is can be confusing what these represent semantically. Here is an
attempt to clear up the confusion:


* An **instance of :class:`Type`** (or an instance of a subclass)
  is a set of constraints on real data. It is
  akin to a primitive type or class in C. It is a *static*
  annotation.

* An **instance of :class:`Variable`** symbolizes data nodes in a data flow
  graph. If you were to parse the C expression ``int x;``, ``int``
  would be a :class:`Type` instance and ``x`` would be a :class:`Variable` instance of
  that :class:`Type` instance. If you were to parse the C expression ``c = a +
  b;``, ``a``, ``b`` and ``c`` would all be :class:`Variable` instances.

* A **subclass of :class:`Type`** is a way of implementing
  a set of :class:`Type` instances that share
  structural similarities. In the :class:`DoubleType` example that we are doing,
  there is actually only one :class:`Type` in that set, therefore the subclass
  does not represent anything that one of its instances does not. In this
  case it is a singleton, a set with one element. However, the
  :class:`TensorType`
  class in Aesara (which is a subclass of :class:`Type`)
  represents a set of types of tensors
  parametrized by their data type or number of dimensions. We could say
  that subclassing :class:`Type` builds a hierarchy of :class:`Type`\s which is based upon
  structural similarity rather than compatibility.
