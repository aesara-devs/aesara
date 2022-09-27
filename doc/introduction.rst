
.. _introduction:

==================
Aesara at a Glance
==================

Aesara is a Python library that allows one to define, optimize/rewrite, and
evaluate mathematical expressions, especially ones involving multi-dimensional
arrays (e.g. :class:`numpy.ndarray`\s).  Using Aesara, it is possible to attain
speeds rivaling hand-crafted C implementations for problems involving large
amounts of data.

Aesara combines aspects of a computer algebra system (CAS) with aspects of an
optimizing compiler. It can also generate customized code for multiple compiled
languages and/or their Python-based interfaces, such as C, Numba, and JAX.  This
combination of CAS features with optimizing compilation and transpilation
is particularly useful for tasks in which complicated mathematical expressions
are evaluated repeatedly and evaluation speed is critical.  For situations
where many different expressions are each evaluated once, Aesara can minimize
the amount of compilation and analysis overhead, but still provide symbolic
features such as automatic differentiation.

Aesara's compiler applies many default optimizations of varying
complexity. These optimizations include, but are not limited to:

* constant folding
* merging of similar sub-graphs, to avoid redundant calculations
* arithmetic simplifications (e.g. ``x * y / x -> y``, ``-(-x) -> x``)
* inserting efficient BLAS_ operations (e.g. ``GEMM``) in a variety of
  contexts
* using memory aliasing to avoid unnecessary calculations
* using in-place operations wherever it does not interfere with aliasing
* loop fusion for element-wise sub-expressions
* improvements to numerical stability (e.g.  :math:`\log(1+\exp(x))` and :math:`\log(\sum_i \exp(x[i]))`)

For more information see :ref:`optimizations`.

Theano
------

The library that Aesara is based on, Theano, was written at the LISA lab to support rapid development of efficient machine learning algorithms but while Theano was commonly referred to as a "deep learning" (DL) library, Aesara is not a DL library.

Designations like "deep learning library" reflect the priorities/goals of a library; specifically, that the library serves the purposes of DL and its computational needs. Aesara is not explicitly intended to serve the purpose of constructing and evaluating DL models, but that doesn't mean it can't serve that purpose well.

The designation "tensor library" is more apt, but, unlike most other tensor libraries (e.g. TensorFlow, PyTorch, etc.), Aesara is more focused on what one might call the symbolic functionality.

Most tensor libraries perform similar operations to some extent, but many do not expose the underlying operations for use at any level other than internal library development. Furthermore, when they do, many libraries cross a large language barrier that unnecessarily hampers rapid development (e.g. moving from Python to C++ and back).

If you follow the history of this project, you can see that it grew out of work on PyMC, and PyMC is a library for domain-specific (i.e. probabilistic modeling) computations. Likewise, the other ``aesara-devs`` projects demonstrate the use of Aesara graphs as an intermediate representation (IR) for a domain-specific language/interface (e.g. `aeppl <https://github.com/aesara-devs/aeppl>`_ provides a graph representation for a PPL) and advanced automations based on IR (e.g. `aemcmc <https://github.com/aesara-devs/aemcmc>`_ as a means of constructing custom samplers from IR, ``aeppl`` as a means of automatically deriving log-probabilities for basic tensor operations represented in IR).

This topic is a little more advanced and doesn't really have parallels in other tensor libraries, but it's one of the things that Aesara uniquely facilitates.

The PyMC/probabilistic programming connection is similar to the DL connection Theano had, but—unlike Theano—we don't want Aesara to be conflated with one of its domains of application—like probabilistic modeling. Those primary domains of application will always have some influence on the development of Aesara, but that's also why we need to avoid labels/designations like "deep learning library" and focus on the functionality, so that we don't unnecessarily compromise Aesara's general applicability, relative simplicity, and/or prevent useful input/collaboration from other domains.

Sneak peek
==========

Here is an example of how to use Aesara. It doesn't show off many of
its features, but it illustrates concretely what Aesara is.


.. If you modify this code, also change :
.. tests/test_tutorial.py:T_introduction.test_introduction_1

.. code-block:: python

    import aesara
    from aesara import tensor as at

    # declare two symbolic floating-point scalars
    a = at.dscalar()
    b = at.dscalar()

    # create a simple expression
    c = a + b

    # convert the expression into a callable object that takes `(a, b)`
    # values as input and computes a value for `c`
    f = aesara.function([a, b], c)

    # bind 1.5 to 'a', 2.5 to 'b', and evaluate 'c'
    assert 4.0 == f(1.5, 2.5)


Aesara is not a programming language in the normal sense because you
write a program in Python that builds expressions for Aesara. Still it
is like a programming language in the sense that you have to

- declare variables ``a`` and ``b`` and give their types,
- build expressions graphs using those variables,
- compile the expression graphs into functions that can be used for computation.

It is good to think of :func:`aesara.function` as the interface to a
compiler which builds a callable object from a purely symbolic graph.
One of Aesara's most important features is that :func:`aesara.function`
can optimize a graph and even compile some or all of it into native
machine instructions.


What does it do that NumPy doesn't
==================================

Aesara is a essentially an optimizing compiler for manipulating
and evaluating expressions, especially tensor-valued
ones. Manipulation of tensors is typically done using the NumPy
package, so what does Aesara do that Python and NumPy don't do?

- *execution speed optimizations*: Aesara can use C, Numba, or JAX to compile
  parts your expression graph into CPU or GPU instructions, which run
  much faster than pure Python.

- *symbolic differentiation*: Aesara can automatically build symbolic graphs
  for computing gradients.

- *stability optimizations*: Aesara can recognize some numerically unstable
  expressions and compute them with more stable algorithms.

The closest Python package to Aesara is sympy_.
Aesara focuses more on tensor expressions than Sympy, and has more machinery
for compilation.  Sympy has more sophisticated algebra rules and can
handle a wider variety of mathematical operations (such as series, limits, and integrals).

If numpy_ is to be compared to MATLAB_ and sympy_ to Mathematica_,
Aesara is a sort of hybrid of the two which tries to combine the best of
both worlds.


Getting started
===============

:ref:`install`
  Instructions to download and install Aesara on your system.

:ref:`tutorial`
  Getting started with Aesara's basic features. Go here if you are
  new!

:ref:`libdoc`
  Details of what Aesara provides. It is recommended to go through
  the :ref:`tutorial` first though.


Contact us
==========

Questions and bug reports should be submitted in the form of an issue at
aesara-dev_

We welcome all kinds of contributions. If you have any questions regarding how
to extend Aesara, please feel free to ask.


.. _LISA:  https://mila.umontreal.ca/
.. _Greek mathematician: http://en.wikipedia.org/wiki/Theano_(mathematician)
.. _numpy: http://numpy.scipy.org/
.. _BLAS: http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms

.. _sympy: http://www.sympy.org/
.. _MATLAB: http://www.mathworks.com/products/matlab/
.. _Mathematica: http://www.wolfram.com/mathematica/

.. _aesara-dev: https://github.com/aesara-devs/aesara/issues
