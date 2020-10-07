|Tests Status| |Coverage|

|Project Name| is a Python library that allows you to define, optimize, and
efficiently evaluate mathematical expressions involving multi-dimensional
arrays.  It can use GPUs and perform efficient symbolic differentiation.

This is a fork of the `original Theano library <https://github.com/Theano/Theano>`__ that is being
maintained by the `PyMC team <https://github.com/pymc-devs>`__.

.. warning::
   The name of this repository/project may change in the near future.


Features
========

- A hackable, pure-Python codebase
- Extensible graph framework suitable for rapid development of custom symbolic optimizations
- Implements an extensible graph transpilation framework that currently provides
  compilation to C and JAX JITed Python functions
- Built on top of one of the most widely-used Python tensor libraries: Theano

Getting started
===============

The legacy documentation is located `here <http://deeplearning.net/software/theano/>`__.

.. warning::
    As development progresses, the legacy documentation may become less applicable.


Installation
============

The latest release of |Project Name| can be installed from PyPI using ``pip``:

::

    pip install Theano-PyMC


Or via conda-forge:

::

    conda install -c conda-forge theano-pymc


The current development branch of |Project Name| can be installed from GitHub, also using ``pip``:

::

    pip install git+https://github.com/pymc-devs/Theano-PyMC


For platform-specific installation information see the legacy documentation `here <http://deeplearning.net/software/theano/install.html>`__.


Support
=======

The PyMC group operates under the NumFOCUS umbrella. If you want to support us financially, you can donate `here <https://numfocus.salsalabs.org/donate-to-pymc3/index.html>`__.


.. |Project Name| replace:: Theano-PyMC
.. |Tests Status| image:: https://github.com/pymc-devs/Theano-PyMC/workflows/Tests/badge.svg
  :target: https://github.com/pymc-devs/Theano/actions?query=workflow%3ATests
.. |Coverage| image:: https://coveralls.io/repos/github/pymc-devs/Theano-PyMC/badge.svg?branch=master
  :target: https://coveralls.io/github/pymc-devs/Theano-PyMC?branch=master
