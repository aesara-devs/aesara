
.. _libdoc_tensor_random:

=============================================
:mod:`random` -- Low-level random numbers
=============================================

.. module:: aesara.tensor.random
   :synopsis: symbolic random variables
.. moduleauthor:: pymc-team


The `aesara.tensor.random` module provides random-number drawing functionality
that closely resembles the `numpy.random` module.

Reference
=========

.. class:: RandomStream()

   A helper class that tracks changes in a shared ``numpy.random.RandomState``
   and behaves like ``numpy.random.RandomState`` by managing access
   to `RandomVariable`s.  For example:

   .. testcode:: constructors

      from aesara.tensor.random.utils import RandomStream

      rng = RandomStream()
      sample = rng.normal(0, 1, size=(2, 2))

.. class:: RandomStateType(Type)

    A `Type` for variables that will take ``numpy.random.RandomState``
    values.

.. function:: random_state_type(name=None)

    Return a new Variable whose ``.type`` is ``random_state_type``.

.. class:: RandomVariable(Op)

    `Op` that draws random numbers from a `numpy.random.RandomState` object.
    This `Op` is parameterized to draw numbers from many possible
    distributions.
