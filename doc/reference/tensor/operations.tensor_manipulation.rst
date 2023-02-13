.. _reference_tensor_tensor_manipulation:
.. currentmodule:: aesara.tensor

Tensor manipulation
===================

Basic operation
---------------

.. autosummary::
   :toctree: _autosummary

   shape


Updating elements
-----------------

.. attention::

   Index assignment is **not** supported in Aesara. If you want to do the equivalent of ``a[5] = b`` or ``a[5]+=b`` you will need to use the ``set_subtensor`` or ``inc_subtensor`` operator respectively.


.. autosummary::
   :toctree: _autosummary

   set_subtensor
   inc_subtensor
   fill_diagonal_offset

Changing tensor shape
---------------------

.. admonition:: Specific to Aesara

   The ``specify_shape`` operator is specific to Aesara.

.. autosummary::
   :toctree: _autosummary

   reshape
   flatten
   specify_shape


Transpose-like operations
-------------------------

.. autosummary::
   :toctree: _autosummary

   moveaxis
   swapaxes
   roll
   TensorVariable.T
   transpose

Changing number of dimensions
-----------------------------

.. autosummary::
   :toctree: _autosummary

   atleast_1d
   atleast_2d
   atleast_3d
   broadcast_to
   broadcast_arrays
   squeeze

Joining tensors
---------------

.. autosummary::
   :toctree: _autosummary

   join
   stack
   stacklists
   horizontal_stack
   vertical_stack

Splitting tensors
-----------------

.. autosummary::
   :toctree: _autosummary

   split

Tiling tensors
--------------

.. autosummary::
   :toctree: _autosummary

   tile
   repeat

Adding and removing elements
----------------------------

.. autosummary::
   :toctree: _autosummary

   unique

Rearranging elements
--------------------

.. autosummary::
   :toctree: _autosummary

    reshape
    permute_row_elements
    inverse_permutation


Padding tensors
---------------

.. autosummary::
   :toctree: _autosummary

   shape_padleft
   shape_padright
   shape_padaxis
