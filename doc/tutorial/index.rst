
.. _tutorial:

========
Tutorial
========

Let us start an interactive session (e.g. with ``python`` or ``ipython``) and import Aesara.

>>> from aesara import *

Several of the symbols you will need to use are in the ``tensor`` subpackage
of Aesara. Let us import that subpackage under a handy name like
``at`` (the tutorials will frequently use this convention).

>>> import aesara.tensor as at

If that succeeded you are ready for the tutorial, otherwise check your
installation (see :ref:`install`).

Throughout the tutorial, bear in mind that there is a :ref:`glossary` as well
as *index* and *modules* links in the upper-right corner of each page to help
you out.

Prerequisites
-------------
.. toctree::

    python
    numpy

Basics
------

.. toctree::

    adding
    examples
    gradients
    conditions
    loop
    shape_info
    broadcasting

Advanced
--------

.. toctree::

    sparse
    conv_arithmetic

Advanced configuration and debugging
------------------------------------

.. toctree::

    modes
    printing_drawing
    debug_faq
    nan_tutorial
    profiling

Further reading
---------------

.. toctree::

    loading_and_saving
    aliasing
    multi_cores
    faq_tutorial
