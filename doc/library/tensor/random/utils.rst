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

Since Aesara uses a functional design, producing pseudo-random numbers in a
graph is not quite as straightforward as it is in numpy.

The way to think about putting randomness into Aesara's computations is to
put random variables in your graph.  Aesara will allocate a numpy RandomState
object for each such variable, and draw from it as necessary.  We will call this sort of sequence of
random numbers a *random stream*.

For an example of how to use random numbers, see
:ref:`Using Random Numbers <using_random_numbers>`.


Reference
=========

.. class:: RandomStream()

    This is a symbolic stand-in for ``numpy.random.RandomState``.

    .. method:: updates()

        :returns: a list of all the (state, new_state) update pairs for the
          random variables created by this object

        This can be a convenient shortcut to enumerating all the random
        variables in a large graph in the ``update`` parameter of function.

    .. method:: seed(meta_seed)

        `meta_seed` will be used to seed a temporary random number generator,
        that will in turn generate seeds for all random variables
        created by this object (via `gen`).

        :returns: None

    .. method:: gen(op, *args, **kwargs)

        Return the random variable from `op(*args, **kwargs)`, but
        also install special attributes (``.rng`` and ``update``, see
        :class:`RandomVariable` ) into it.

        This function also adds the returned variable to an internal list so
        that it can be seeded later by a call to `seed`.

    .. method:: uniform, normal, binomial, multinomial, random_integers, ...

        See :class:`basic.RandomVariable`.
