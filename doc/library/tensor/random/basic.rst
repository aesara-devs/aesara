
.. _libdoc_tensor_random_basic:

=============================================
:mod:`basic` -- Low-level random numbers
=============================================

.. module:: aesara.tensor.random
   :synopsis: symbolic random variables


The :mod:`aesara.tensor.random` module provides random-number drawing functionality
that closely resembles the :mod:`numpy.random` module.

Reference
=========

.. class:: RandomStream()

   A helper class that tracks changes in a shared :class:`numpy.random.RandomState`
   and behaves like :class:`numpy.random.RandomState` by managing access
   to :class:`RandomVariable`\s.  For example:

   .. testcode:: constructors

      from aesara.tensor.random.utils import RandomStream

      rng = RandomStream()
      sample = rng.normal(0, 1, size=(2, 2))

.. class:: RandomStateType(Type)

    A :class:`Type` for variables that will take :class:`numpy.random.RandomState`
    values.

.. function:: random_state_type(name=None)

    Return a new :class:`Variable` whose :attr:`Variable.type` is an instance of
    :class:`RandomStateType`.

.. class:: RandomVariable(Op)

    :class:`Op` that draws random numbers from a :class:`numpy.random.RandomState` object.
    This :class:`Op` is parameterized to draw numbers from many possible
    distributions.
