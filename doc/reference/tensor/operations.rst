.. _reference_tensor_operations:

Tensor operations
=================

.. currentmodule:: aesara.tensor

The module :mod:`aesara.tensor` allows to create tensors and express symbolic calculations using the NumPy and SciPy API. Docstings are grouped by functionality, and assume that ``aesara.tensor`` is imported as

>>> import aesara.tensor as at

Aesara's API tries to mirror NumPy's, so in most cases it is safe to assume that the basic NumPy array functions and methods will be available. If you find an inconsistency, or if a function is missing, please open an `Issue <https://github.com/aesara-devs/aesara>`__.

.. toctree::
   :maxdepth: 1

   operations.tensor_creation
   operations.tensor_manipulation
   operations.indexing
   operations.binary_operations
   operations.discrete_fourier
   operations.linalg
   operations.logic
   operations.mathematical_functions
   operations.padding
   operations.sorting
   operations.statistics
   operations.window_functions
