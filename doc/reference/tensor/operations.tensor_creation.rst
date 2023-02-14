.. _reference_tensor_tensor_creation:
.. currentmodule:: aesara.tensor

Tensor creation
===============

.. admonition:: See also

   :ref:`reference_tensor_create`

From shape or value
-------------------

.. autosummary::
   :toctree: _autosummary

   empty
   empty_like
   eye
   ones
   ones_like
   zeros
   zeros_like
   full
   full_like
   fill
   identity_like
   alloc
   second


From existing data
-------------------

.. autosummary::
   :toctree: _autosummary

   as_tensor


Numerical ranges
----------------

.. autosummary::
   :toctree: _autosummary

   arange
   linspace
   logspace
   geomspace
   mgrid
   ogrid

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


Building matrices
-----------------

.. autosummary::
   :toctree: _autosummary

   diag
   tri
   tril
   triu
