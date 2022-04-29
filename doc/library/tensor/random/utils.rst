.. _libdoc_tensor_random_utils:

======================================================
:mod:`utils` -- Friendly random numbers
======================================================

.. module:: aesara.tensor.random.utils
   :platform: Unix, Windows
   :synopsis: symbolic random variables
.. moduleauthor:: LISA

Guide
=====

Aesara assignes NumPy RNG states (e.g. `Generator` or `RandomState` objects) to
each `RandomVariable`.  The combination of an RNG state, a specific
`RandomVariable` type (e.g. `NormalRV`), and a set of distribution parameters
uniquely defines the `RandomVariable` instances in a graph.

This means that a "stream" of distinct RNG states is required in order to
produce distinct random variables of the same kind.  `RandomStream` provides a
means of generating distinct random variables in a fully reproducible way.

`RandomStream` is also designed to produce simpler graphs and work with more
sophisticated `Op`\s like `Scan`, which makes it the de facto random variable
interface in Aesara.

For an example of how to use random numbers, see :ref:`Using Random Numbers <using_random_numbers>`.


Reference
=========

.. class:: RandomStream()

    This is a symbolic stand-in for `numpy.random.Generator`.

    .. method:: updates()

        :returns: a list of all the (state, new_state) update pairs for the
          random variables created by this object

        This can be a convenient shortcut to enumerating all the random
        variables in a large graph in the ``update`` argument to
        `aesara.function`.

    .. method:: seed(meta_seed)

        `meta_seed` will be used to seed a temporary random number generator,
        that will in turn generate seeds for all random variables
        created by this object (via `gen`).

        :returns: None

    .. method:: gen(op, *args, **kwargs)

        Return the random variable from ``op(*args, **kwargs)``.

        This function also adds the returned variable to an internal list so
        that it can be seeded later by a call to `seed`.

    .. method:: uniform, normal, binomial, multinomial, random_integers, ...

        See :class:`basic.RandomVariable`.
