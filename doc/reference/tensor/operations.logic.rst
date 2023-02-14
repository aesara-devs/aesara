.. _reference_tensor_logic:
.. currentmodule:: aesara.tensor


Logic functions
===============


Truth value testing
-------------------

.. autosummary::
   :toctree: _autosummary

   allclose
   any

Array contents
--------------

.. autosummary::
   :toctree: _autosummary

   isinf
   isnan

Comparisons
-----------

.. note::

   Aesara does not have a boolean dtype. Instead the result of comparison operators are represented in ``int8``.

.. danger::

   The Python operator ``==`` does not work as a comparison operator in the usual sense in Aesara. Use :func:`eq` instead.

.. autosummary::
   :toctree: _autosummary

   lt
   gt
   ge
   eq
   neq
   allclose
   isclose
