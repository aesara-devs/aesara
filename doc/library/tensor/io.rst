===================================================================
:mod:`tensor.io` --  Tensor IO Ops
===================================================================

.. module:: tensor.io
   :platform: Unix, Windows
   :synopsis: Tensor IO Ops
.. moduleauthor:: LISA

File operation
==============

- Load from disk with the function :func:`load <aesara.tensor.io.load>` and its associated op :class:`LoadFromDisk <aesara.tensor.io.LoadFromDisk>`

MPI operation
=============
- Non-blocking transfer: :func:`isend <aesara.tensor.io.isend>` and :func:`irecv <aesara.tensor.io.irecv>`.
- Blocking transfer: :func:`send <aesara.tensor.io.send>` and :func:`recv <aesara.tensor.io.recv>`

Details
=======

.. automodule:: aesara.tensor.io
    :members:
