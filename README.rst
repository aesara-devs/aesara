|Tests Status| |Coverage|

|Project Name| is a Python library that allows you to define, optimize, and
efficiently evaluate mathematical expressions involving multi-dimensional
arrays.  It can use GPUs and perform efficient symbolic differentiation.

This is a fork of the `original Aesara library <https://github.com/Aesara/Aesara>`__ that is being
maintained by the `PyMC team <https://github.com/pymc-devs>`__.

.. warning::
   The name of this repository/project may change in the near future.


Features
========

- A hackable, pure-Python codebase
- Extensible graph framework suitable for rapid development of custom symbolic optimizations
- Implements an extensible graph transpilation framework that currently provides
  compilation to C and JAX JITed Python functions
- Built on top of one of the most widely-used Python tensor libraries: Aesara

Getting started
===============

The legacy documentation is located `here <http://deeplearning.net/software/aesara/>`__.

.. warning::
    As development progresses, the legacy documentation may become less applicable.


Installation
============

The latest release of |Project Name| can be installed from PyPI using ``pip``:

::

    pip install Aesara-PyMC


Or via conda-forge:

::

    conda install -c conda-forge aesara-pymc


The current development branch of |Project Name| can be installed from GitHub, also using ``pip``:

::

    pip install git+https://github.com/pymc-devs/Aesara-PyMC


For platform-specific installation information see the legacy documentation `here <http://deeplearning.net/software/aesara/install.html>`__.


Support
=======

The PyMC group operates under the NumFOCUS umbrella. If you want to support us financially, you can donate `here <https://numfocus.salsalabs.org/donate-to-pymc3/index.html>`__.


.. |Project Name| replace:: Aesara-PyMC
.. |Tests Status| image:: https://github.com/pymc-devs/Aesara-PyMC/workflows/Tests/badge.svg
  :target: https://github.com/pymc-devs/Aesara/actions?query=workflow%3ATests
.. |Coverage| image:: https://coveralls.io/repos/github/pymc-devs/Aesara-PyMC/badge.svg?branch=master
  :target: https://coveralls.io/github/pymc-devs/Aesara-PyMC?branch=master
