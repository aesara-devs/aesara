.. _reference_scan:

Loops
=====

The module :mod:`aesara.scan` provides the basic functionality needed to do loops
in Aesara.

.. automodule:: aesara.scan

`aesara.scan`
-------------

.. autofunction:: aesara.scan
   :noindex:

Other ways to create loops
--------------------------

:func:`aesara.scan` comes with bells and whistles that are not always all necessary, which is why Aesara provides several other functions to create a :class:`Scan` operator:

.. autofunction:: aesara.map
.. autofunction:: aesara.reduce
.. autofunction:: aesara.foldl
.. autofunction:: aesara.foldr

.. toctree::
    :maxdepth: 1

    loops_api
    loops_tutorial
    scan_extend
