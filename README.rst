|Tests Status| |Coverage|

|Project Name| is a Python library that allows you to define, optimize, and
efficiently evaluate mathematical expressions involving multi-dimensional
arrays.  It can use GPUs and perform efficient symbolic differentiation.

This is a fork of the `original Theano library <https://github.com/Theano/Theano>`__ that is being
maintained by the `PyMC team <https://github.com/pymc-devs>`__.

Features
========

- A hackable, pure-Python codebase
- Extensible graph framework suitable for rapid development of custom symbolic optimizations
- Implements an extensible graph transpilation framework that currently provides
  compilation to C and JAX JITed Python functions
- Built on top of one of the most widely-used Python tensor libraries: Theano

Getting started
===============

.. code-block:: python

  import aesara
  from aesara import tensor as aet
  from aesara.printing import debugprint

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

  debugprint(d)
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
  debugprint(f_d)
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

The documentation is located `here <https://aesara.readthedocs.io/en/latest/>`__.


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

    pip install git+https://github.com/pymc-devs/aesara


For platform-specific installation information see the legacy documentation `here <http://deeplearning.net/software/theano/install.html>`__.


Support
=======

The PyMC group operates under the NumFOCUS umbrella. If you want to support us financially, you can donate `here <https://numfocus.salsalabs.org/donate-to-pymc3/index.html>`__.


.. |Project Name| replace:: Aesara
.. |Tests Status| image:: https://github.com/pymc-devs/aesara/workflows/Tests/badge.svg
  :target: https://github.com/pymc-devs/aesara/actions?query=workflow%3ATests
.. |Coverage| image:: https://codecov.io/gh/pymc-devs/aesara/branch/master/graph/badge.svg?token=WVwr8nZYmc
  :target: https://codecov.io/gh/pymc-devs/aesara
