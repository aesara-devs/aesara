====
Tips
====


..
   Reusing outputs
   ===============

   .. todo:: Write this.


Don't define new :class:`Op`\s unless you have to
=================================================

It is usually not useful to define :class:`Op`\s that can be easily
implemented using other already existing :class:`Op`\s. For example, instead of
writing a "sum_square_difference" :class:`Op`, you should probably just write a
simple function:

.. code::

   from aesara import tensor as at

   def sum_square_difference(a, b):
       return at.sum((a - b)**2)

Even without taking Aesara's rewrites into account, it is likely
to work just as well as a custom implementation. It also supports all
data types, tensors of all dimensions as well as broadcasting, whereas
a custom implementation would probably only bother to support
contiguous vectors/matrices of doubles...


Use Aesara's high order :class:`Op`\s when applicable
=====================================================

Aesara provides some generic :class:`Op` classes which allow you to generate a
lot of :class:`Op`\s at a lesser effort. For instance, :class:`Elemwise` can be used to
make :term:`elemwise` operations easily, whereas :class:`DimShuffle` can be
used to make transpose-like transformations. These higher order :class:`Op`\s
are mostly tensor-related, as this is Aesara's specialty.


..
   .. _opchecklist:

   :class:`Op` Checklist
   =====================

   Use this list to make sure you haven't forgotten anything when
   defining a new :class:`Op`. It might not be exhaustive but it covers a lot of
   common mistakes.

   .. todo:: Write a list.
