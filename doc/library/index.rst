
.. _libdoc:
.. _Library documentation:

=================
API Documentation
=================

This documentation covers Aesara module-wise.  This is suited to finding the
Types and Ops that you can use to build and compile expression graphs.

Modules
=======

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

.. module:: aesara
   :platform: Unix, Windows
   :synopsis: Aesara top-level import
.. moduleauthor:: LISA

There are also some top-level imports that you might find more convenient:

Graph
=====

.. function:: shared(...)

   Alias for :func:`aesara.compile.sharedvalue.shared`

.. function:: function(...)

   Alias for :func:`aesara.compile.function.function`

.. autofunction:: aesara.clone_replace(...)

   Alias for :func:`aesara.graph.basic.clone_replace`

Control flow
============

.. autofunction:: aesara.scan(...)

   Alias for :func:`aesara.scan.basic.scan`

Convert to Variable
====================

.. autofunction:: aesara.as_symbolic(...)

Debug
=====

.. autofunction:: aesara.dprint(...)

   Alias for :func:`aesara.printing.debugprint`

