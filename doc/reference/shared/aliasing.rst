
.. _aliasing:

=======================================================
Understanding Memory Aliasing for Speed and Correctness
=======================================================

The aggressive reuse of memory is one of the ways through which Aesara makes code fast, and
it is important for the correctness and speed of your program that you understand
how Aesara might alias buffers.

This section describes the principles based on which Aesara handles memory, and explains
when you might want to alter the default behaviour of some functions and
methods for faster performance.


The Memory Model: Two Spaces
============================

There are some simple principles that guide Aesara's handling of memory.  The
main idea is that there is a pool of memory managed by Aesara, and Aesara tracks
changes to values in that pool.

- Aesara manages its own memory space, which typically does not overlap with
  the memory of normal Python variables that non-Aesara code creates.

- Aesara functions only modify buffers that are in Aesara's memory space.

- Aesara's memory space includes the buffers allocated to store ``shared``
  variables and the temporaries used to evaluate functions.

- Physically, Aesara's memory space may be spread across the host, a GPU
  device(s), and in the future may even include objects on a remote machine.

- The memory allocated for a ``shared`` variable buffer is unique: it is never
  aliased to another ``shared`` variable.

- Aesara's managed memory is constant while Aesara functions are not running
  and Aesara's library code is not running.

- The default behaviour of a function is to return user-space values for
  outputs, and to expect user-space values for inputs.

The distinction between Aesara-managed memory and user-managed memory can be
broken down by some Aesara functions (e.g. ``shared``, ``get_value`` and the
constructors for ``In`` and ``Out``) by using a ``borrow=True`` flag.
This can make those methods faster (by avoiding copy operations) at the expense
of risking subtle bugs in the overall program (by aliasing memory).

The rest of this section is aimed at helping you to understand when it is safe
to use the ``borrow=True`` argument and reap the benefits of faster code.

Borrowing when Creating Shared Variables
========================================

A ``borrow`` argument can be provided to the shared-variable constructor.

.. testcode:: borrow

   import numpy, aesara
   np_array = numpy.ones(2, dtype='float32')

   s_default = aesara.shared(np_array)
   s_false   = aesara.shared(np_array, borrow=False)
   s_true    = aesara.shared(np_array, borrow=True)

By default (``s_default``) and when explicitly setting ``borrow=False``, the
shared variable we construct gets a (deep) copy of ``np_array``.  So changes we
subsequently make to ``np_array`` have no effect on our shared variable.

.. testcode:: borrow

   np_array += 1 # now it is an array of 2.0 s

   print(s_default.get_value())
   print(s_false.get_value())
   print(s_true.get_value())

.. testoutput:: borrow

   [ 1.  1.]
   [ 1.  1.]
   [ 2.  2.]


If we are running this with the CPU as the device,
then changes we make to ``np_array`` right away will show up in
``s_true.get_value()``
because NumPy arrays are mutable, and ``s_true`` is using the ``np_array``
object as it's internal buffer.

However, this aliasing of ``np_array`` and ``s_true`` is not guaranteed to occur,
and may occur only temporarily even if it occurs at all.
It is not guaranteed to occur because if Aesara is using a GPU device, then the
``borrow`` flag has no effect. It may occur only temporarily because
if we call an Aesara function that updates the value of ``s_true`` the aliasing
relationship may or may not be broken (the function is allowed to
update the ``shared`` variable by modifying its buffer, which will preserve
the aliasing, or by changing which buffer the variable points to, which
will terminate the aliasing).

Take home message:

It is a safe practice (and a good idea) to use ``borrow=True`` in a ``shared``
variable constructor when the ``shared`` variable stands for a large object (in
terms of memory footprint) and you do not want to create copies of it in
memory.

It is not a reliable technique to use ``borrow=True`` to modify ``shared`` variables
through side-effect, because with some devices (e.g. GPU devices) this technique will
not work.

Borrowing when Accessing Value of Shared Variables
==================================================

Retrieving
----------

A ``borrow`` argument can also be used to control how a ``shared`` variable's value is
retrieved.


.. testcode:: borrow

   s = aesara.shared(np_array)

   v_false = s.get_value(borrow=False) # N.B. borrow default is False
   v_true = s.get_value(borrow=True)


When ``borrow=False`` is passed to ``get_value``, it means that the return value
may not be aliased to any part of Aesara's internal memory.
When ``borrow=True`` is passed to ``get_value``, it means that the return value
might be aliased to some of Aesara's internal memory.
But both of these calls might create copies of the internal memory.

The reason that ``borrow=True`` might still make a copy is that the internal
representation of a ``shared`` variable might not be what you expect.  When you
create a ``shared`` variable by passing a NumPy array for example, then ``get_value()``
must return a NumPy array too.  That's how Aesara can make the GPU use
transparent.  But when you are using a GPU (or in the future perhaps a remote machine),
then the numpy.ndarray is not the internal representation of your data.
If you really want Aesara to return its internal representation and never copy it
then you should use the ``return_internal_type=True`` argument to
``get_value``.  It will never cast the internal object (always return in
constant time), but might return various datatypes depending on contextual
factors (e.g. the compute device, the dtype of the NumPy array).

.. testcode:: borrow

    v_internal = s.get_value(borrow=True, return_internal_type=True)

It is possible to use ``borrow=False`` in conjunction with
``return_internal_type=True``, which will return a deep copy of the internal object.
This is primarily for internal debugging, not for typical use.

For the transparent use rewrites, there is the policy that ``get_value()``
always return by default the same object type it received when the ``shared``
variable was created. So if you created manually data on the gpu and create a
``shared`` variable on the gpu with this data, ``get_value`` will always return
gpu data even when ``return_internal_type=False``.

Take home message:

It is safe (and sometimes much faster) to use ``get_value(borrow=True)`` when
your code does not modify the return value.  Do not use this to modify a ``shared``
variable by side-effect because it will make your code device-dependent.
Modification of GPU variables through this sort of side-effect is impossible.

Assigning
---------

``Shared`` variables also have a ``set_value`` method that can accept an optional
``borrow=True`` argument. The semantics are similar to those of creating a new
``shared`` variable - ``borrow=False`` is the default and ``borrow=True`` means
that Aesara may reuse the buffer you provide as the internal storage for the variable.

A standard pattern for manually updating the value of a ``shared`` variable is as
follows:

.. testsetup:: borrow

   def some_inplace_fn(v):
       return v

.. testcode:: borrow

    s.set_value(
        some_inplace_fn(s.get_value(borrow=True)),
        borrow=True)

This pattern works regardless of the computing device, and when the latter
makes it possible to expose Aesara's internal variables without a copy, then it
proceeds as fast as an in-place update.


..
   When ``shared`` variables are allocated on the GPU, the transfers to and from the GPU device memory can
   be costly.  Here are a few tips to ensure fast and efficient use of GPU memory and bandwidth:

   * Prior to Aesara 0.3.1, ``set_value`` did not work in-place on the GPU. This meant that, sometimes,
     GPU memory for the new value would be allocated before the old memory was released. If you're
     running near the limits of GPU memory, this could cause you to run out of GPU memory
     unnecessarily.

     *Solution*: update to a newer version of Aesara.

   * If you are going to swap several chunks of data in and out of a ``shared`` variable repeatedly,
     you will want to reuse the memory that you allocated the first time if possible - it is both
     faster and more memory efficient.

     *Solution*: upgrade to a recent version of Aesara (>0.3.0) and consider padding your source
     data to make sure that every chunk is the same size.

   * It is also worth mentioning that, current GPU copying routines
     support only contiguous memory.  So Aesara must make the value you
     provide *C-contiguous* prior to copying it.  This can require an
     extra copy of the data on the host.

     *Solution*: make sure that the value
     you assign to a GpuArraySharedVariable is *already*  *C-contiguous*.


.. _borrowfunction:

Borrowing when Constructing Function Objects
============================================

A ``borrow`` argument can also be provided to the ``In`` and ``Out`` objects
that control how ``aesara.function`` handles its argument[s] and return value[s].

.. testcode::

    import aesara
    import aesara.tensor as at
    from aesara.compile.io import In, Out

    x = at.matrix()
    y = 2 * x
    f = aesara.function([In(x, borrow=True)], Out(y, borrow=True))

Borrowing an input means that Aesara will treat the argument you provide as if
it were part of Aesara's pool of temporaries.  Consequently, your input
may be reused as a buffer (and overwritten!) during the computation of other variables in the
course of evaluating that function (e.g. ``f``).


Borrowing an output means that Aesara will not insist on allocating a fresh
output buffer every time you call the function.  It will possibly reuse the same one as
on a previous call, and overwrite the old content.  Consequently, it may overwrite
old return values through side-effect.
Those return values may also be overwritten in
the course of evaluating another compiled function (for example, the output
may be aliased to a ``shared`` variable).  So be careful to use a borrowed return
value right away before calling any more Aesara functions.
The default is of course to not borrow internal results.

It is also possible to pass a ``return_internal_type=True`` flag to the ``Out``
variable which has the same interpretation as the ``return_internal_type`` flag
to the ``shared`` variable's ``get_value`` function.  Unlike ``get_value()``, the
combination of ``return_internal_type=True`` and ``borrow=True`` arguments to
``Out()`` are not guaranteed to avoid copying an output value.  They are just
hints that give more flexibility to the compilation and rewriting of the
graph.

Take home message:

When an input ``x`` to a function is not needed after the function
returns and you would like to make it available to Aesara as
additional workspace, then consider marking it with ``In(x, borrow=True)``.  It
may make the function faster and reduce its memory requirement.  When a return
value ``y`` is large (in terms of memory footprint), and you only need to read
from it once, right away when it's returned, then consider marking it with an
``Out(y, borrow=True)``.
