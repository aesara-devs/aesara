|Tests Status| |Coverage| |Gitter|

.. raw:: html

  <div align="center">
  <img src="./doc/images/aesara_logo_2400.png" alt="logo"></img>
  </div>


|Project Name| is a Python library that allows one to define, optimize, and
efficiently evaluate mathematical expressions involving multi-dimensional
arrays.

Features
========

- A hackable, pure-Python codebase
- Extensible graph framework suitable for rapid development of custom operators and symbolic optimizations
- Implements an extensible graph transpilation framework that currently provides
  compilation via C, `JAX <https://github.com/google/jax>`__, and `Numba <https://github.com/numba/numba>`__
- Based on one of the most widely-used Python tensor libraries: `Theano <https://github.com/Theano/Theano>`__

Getting started
===============

.. code-block:: python

  import aesara
  from aesara import tensor as aet

  # Declare two symbolic floating-point scalars
  a = aet.dscalar("a")
  b = aet.dscalar("b")

  # Create a simple example expression
  c = a + b

  # Convert the expression into a callable object that takes `(a, b)`
  # values as input and computes the value of `c`.
  f_c = aesara.function([a, b], c)

  assert f_c(1.5, 2.5) == 4.0

  # Compute the gradient of the example expression with respect to `a`
  dc = aesara.grad(c, a)

  f_dc = aesara.function([a, b], dc)

  assert f_dc(1.5, 2.5) == 1.0

  # Compiling functions with `aesara.function` also optimizes
  # expression graphs by removing unnecessary operations and
  # replacing computations with more efficient ones.

  v = aet.vector("v")
  M = aet.matrix("M")

  d = a/a + (M + a).dot(v)

  aesara.dprint(d)
  # Elemwise{add,no_inplace} [id A] ''
  #  |InplaceDimShuffle{x} [id B] ''
  #  | |Elemwise{true_div,no_inplace} [id C] ''
  #  |   |a [id D]
  #  |   |a [id D]
  #  |dot [id E] ''
  #    |Elemwise{add,no_inplace} [id F] ''
  #    | |M [id G]
  #    | |InplaceDimShuffle{x,x} [id H] ''
  #    |   |a [id D]
  #    |v [id I]

  f_d = aesara.function([a, v, M], d)

  # `a/a` -> `1` and the dot product is replaced with a BLAS function
  # (i.e. CGemv)
  aesara.dprint(f_d)
  # Elemwise{Add}[(0, 1)] [id A] ''   5
  #  |TensorConstant{(1,) of 1.0} [id B]
  #  |CGemv{inplace} [id C] ''   4
  #    |AllocEmpty{dtype='float64'} [id D] ''   3
  #    | |Shape_i{0} [id E] ''   2
  #    |   |M [id F]
  #    |TensorConstant{1.0} [id G]
  #    |Elemwise{add,no_inplace} [id H] ''   1
  #    | |M [id F]
  #    | |InplaceDimShuffle{x,x} [id I] ''   0
  #    |   |a [id J]
  #    |v [id K]
  #    |TensorConstant{0.0} [id L]

See `the Aesara documentation <https://aesara.readthedocs.io/en/latest/>`__ for in-depth tutorials.


Installation
============

The latest release of |Project Name| can be installed from PyPI using ``pip``:

::

    pip install aesara


Or via conda-forge:

::

    conda install -c conda-forge aesara


The current development branch of |Project Name| can be installed from GitHub, also using ``pip``:

::

    pip install git+https://github.com/aesara-devs/aesara



Support
=======

Many Aesara developers are also PyMC developers, and, since the PyMC developers
operate under the NumFOCUS umbrella, if you want to support them financially,
consider donating `here <https://numfocus.salsalabs.org/donate-to-pymc3/index.html>`__.


Special thanks to `Bram Timmer <http://beside.ca>`__ for the logo.


.. |Project Name| replace:: Aesara
.. |Tests Status| image:: https://github.com/aesara-devs/aesara/workflows/Tests/badge.svg
  :target: https://github.com/aesara-devs/aesara/actions?query=workflow%3ATests
.. |Coverage| image:: https://codecov.io/gh/aesara-devs/aesara/branch/main/graph/badge.svg?token=WVwr8nZYmc
  :target: https://codecov.io/gh/aesara-devs/aesara
.. |Gitter| image:: https://badges.gitter.im/aesara-devs/aesara.svg
  :target: https://gitter.im/aesara-devs/aesara?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
