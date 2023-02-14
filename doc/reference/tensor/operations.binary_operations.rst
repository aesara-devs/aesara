.. _reference_tensor_binary_operations:
.. currentmodule:: aesara.tensor

Binary operations
==================

.. note::

   The bitwise operators take an integer as an input.

.. autosummary::
   :toctree: _autosummary

   bitwise_and
   bitwise_or
   bitwise_xor
   bitwise_not
   invert

.. doctest::
   :options: +SKIP

   >>> a, b = at.itensor3(), at.itensor3() # example inputs
   >>> a & b      # at.and_(a,b)    bitwise and (alias at.bitwise_and)
   >>> a ^ 1      # at.xor(a,1)     bitwise xor (alias at.bitwise_xor)
   >>> a | b      # at.or_(a,b)     bitwise or (alias at.bitwise_or)
   >>> ~a         # at.invert(a)    bitwise invert (alias at.bitwise_not)
