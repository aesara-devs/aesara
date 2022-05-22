|Tests Status| |Coverage| |Gitter|

.. raw:: html

  <div align="center">
  <img src="./doc/images/aesara_logo_2400.png" alt="logo"></img>
  </div>


|Project Name| is a Python library that allows one to define, optimize, and
efficiently evaluate mathematical expressions involving multi-dimensional
arrays.

What is |Project Name|
========

Aesara is a fork of Theano, and Theano was commonly referred to as a "deep learning" (DL) library, but Aesara is not a DL library.

Designations like "deep learning library" reflect the priorities/goals of a library; specifically, that the library serves the purposes of DL and its computational needs. Aesara is not explicitly intended to serve the purpose of constructing and evaluating DL models, but that doesn't mean it can't serve that purpose well.

As far as designations or labels are concerned, instead of describing our project's priorities/goals based on an area of study or application (e.g. DL, machine learning, statistical modeling, etc.), we prefer to focus on the functionality that Aesara is expected to provide, and that's primarily symbolic tensor computations.

The designation "tensor library" is more apt, but, unlike most other tensor libraries (e.g. TensorFlow, PyTorch, etc.), Aesara is more focused on what one might call the symbolic functionality.

As a library, Aesara focuses on and advocates the extension of its core offerings, which are as follows:

* a framework for flexible graph-based representations of computations,
      * E.g. the construction of custom ``Type`` s, ``Variable`` s, ``Op`` s, and lower-level graph elements

* implementations of basic tensor objects and operations,
      * E.g. ``Type``, ``Variable``, and ``Op`` implementations that mirror "tensor"-based NumPy and SciPy offerings, and their gradients

* graph analysis and rewriting,
      * E.g. the general manipulation of graphs for the purposes of "optimization", automation, etc.

* and code transpilation.
      * E.g. the conversion of graphs into performant code via other target languages/representations

Most tensor libraries perform these operations to some extent, but many do not expose the underlying operations for use at any level other than internal library development. Furthermore, when they do, many libraries cross a large language barrier that unnecessarily hampers rapid development (e.g. moving from Python to C++ and back).

For most tensor libraries, a NumPy-like interface to compiled tensor computations is the primary/only offering of the library. Aesara takes the opposite approach and views all the aforementioned operations as part of the core offerings of the library, but it also stitches them together so that the library can be used like other tensor libraries.

There are some concrete reasons for taking this approach, and one is the representation and construction of efficient domain-specific symbolic computations. If you follow the history of this project, you can see that it grew out of work on PyMC, and PyMC is a library for domain-specific (i.e. probabilistic modeling) computations. Likewise, the other ``aesara-devs`` projects demonstrate the use of Aesara graphs as an intermediate representation (IR) for a domain-specific language/interface (e.g. `aeppl <https://github.com/aesara-devs/aeppl>`_ provides a graph representation for a PPL) and advanced automations based on IR (e.g. `aemcmc <https://github.com/aesara-devs/aemcmc>`_ as a means of constructing custom samplers from IR, ``aeppl`` as a means of automatically deriving log-probabilities for basic tensor operations represented in IR).

This topic is a little more advanced and doesn't really have parallels in other tensor libraries, but it's one of the things that Aesara uniquely facilitates.

The PyMC/probabilistic programming connection is similar to the DL connection Theano had, but—unlike Theano—we don't want Aesara to be conflated with one of its domains of application—like probabilistic modeling. Those primary domains of application will always have some influence on the development of Aesara, but that's also why we need to avoid labels/designations like "deep learning library" and focus on the functionality, so that we don't unnecessarily compromise Aesara's general applicability, relative simplicity, and/or prevent useful input/collaboration from other domains.

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
  from aesara import tensor as at

  # Declare two symbolic floating-point scalars
  a = at.dscalar("a")
  b = at.dscalar("b")

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

  v = at.vector("v")
  M = at.matrix("M")

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

Special thanks to `Bram Timmer <http://beside.ca>`__ for the logo.


.. |Project Name| replace:: Aesara
.. |Tests Status| image:: https://github.com/aesara-devs/aesara/workflows/Tests/badge.svg
  :target: https://github.com/aesara-devs/aesara/actions?query=workflow%3ATests
.. |Coverage| image:: https://codecov.io/gh/aesara-devs/aesara/branch/main/graph/badge.svg?token=WVwr8nZYmc
  :target: https://codecov.io/gh/aesara-devs/aesara
.. |Gitter| image:: https://badges.gitter.im/aesara-devs/aesara.svg
  :target: https://gitter.im/aesara-devs/aesara?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
