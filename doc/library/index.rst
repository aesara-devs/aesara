
.. _libdoc:
.. _Library documentation:

=================
API Documentation
=================

This documentation covers Aesara module-wise.  This is suited to finding the
Types and Ops that you can use to build and compile expression graphs.

.. toctree::
   :maxdepth: 1

   compile/index
   config
   d3viz/index
   graph/index
   gpuarray/index
   gradient
   misc/pkl_utils
   printing
   sandbox/index
   scalar/index
   scan
   sparse/index
   sparse/sandbox
   tensor/index
   typed_list

There are also some top-level imports that you might find more convenient:


.. module:: aesara
   :platform: Unix, Windows
   :synopsis: Aesara top-level import
.. moduleauthor:: LISA

.. function:: function(...)

    Alias for :func:`aesara.compile.function.function`


.. function:: function_dump(...)

    Alias for :func:`aesara.compile.function.function_dump`

.. function:: shared(...)

    Alias for :func:`aesara.compile.sharedvalue.shared`
.. autofunction:: aesara.scan(...)

.. class:: In
   Alias for :func:`aesara.scan.basic.scan`

    Alias for :class:`function.In`

.. autofunction:: aesara.dprint(...)

   Alias for :func:`aesara.printing.debugprint`

.. autofunction:: aesara.clone_replace
