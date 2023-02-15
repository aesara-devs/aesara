.. _reference_tensor:

Tensors
=======


Aesara supports symbolic tensor expressions.  When you type,

>>> import aesara.tensor as at
>>> x = at.fmatrix()

the ``x`` is a :class:`TensorVariable` instance.

The ``at.fmatrix`` object itself is an instance of :class:`TensorType`.
Aesara knows what type of variable ``x`` is because ``x.type``
points back to ``at.fmatrix``.

This section of the documentation is organized as follows:

* :ref:`Tensor objects <reference_tensor_objects>` page explains the various ways in which a tensor variable can be created, the attributes and methods of :class:`TensorVariable` and :class:`TensorType`.
* :ref:`Tensor creation <reference_tensor_create>` describes all the ways one can create a :class:`TensorVariable`.
* :ref:`Tensor operations <reference_tensor_operations>` lists the available operations on :class:`TensorVariable`.

.. toctree::
    :maxdepth: 1
    :hidden:

    tensor
    Creation <create>
    Operations <operations>
    shapes
    sparse/index
    shared/index
    Utils <utils>
