.. _reference_tensor_indexing:
.. currentmodule:: aesara.tensor


Indexing
--------

Like NumPy, Aesara distinguishes between *basic* and *advanced* indexing.
Aesara fully supports basic indexing
(see `NumPy's indexing  <http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html>`_)
and `integer advanced indexing
<http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#integer>`_.

Index-assignment is *not* supported.  If you want to do something like ``a[5]
= b`` or ``a[5]+=b``, see :func:`aesara.tensor.subtensor.set_subtensor` and
:func:`aesara.tensor.subtensor.inc_subtensor` below.


Generating index tensors
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   where
   ogrid
   ravel_multi_index
   unravel_index
   tril_indices_from
   tril_indices
   triu_indices
   triu_indices_from


Indexing-like operations
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   take
   take_along_axis
   choose
   compress
   diag
   diagonal

Inserting data into tensors
~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. autosummary::
   :toctree: _autosummary

   set_subtensor
   inc_subtensor
