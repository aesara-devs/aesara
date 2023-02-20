.. _reference_tensor_random:

Random variables
================

Using Random Numbers
--------------------

Because in Aesara you first express everything symbolically and
afterwards compile this expression to get functions,
using pseudo-random numbers is not as straightforward as it is in
NumPy, though also not too complicated.

The way to think about putting randomness into Aesara's computations is
to put random variables in your graph. Aesara will allocate a NumPy
`RandomStream` object (a random number generator) for each such
variable, and draw from it as necessary. We will call this sort of
sequence of random numbers a *random stream*. *Random streams* are at
their core shared variables, so the observations on shared variables
hold here as well. Aesara's random objects are defined and implemented in
:class:`RandomStream` and, at a lower level,
in :class:`RandomVariable`.

Brief Example
~~~~~~~~~~~~~

Here's a brief example.  The setup code is:

.. If you modify this code, also change :
.. tests/test_tutorial.py:T_examples.test_examples_9

.. testcode::

    from aesara.tensor.random.utils import RandomStream
    from aesara import function


    srng = RandomStream(seed=234)
    rv_u = srng.uniform(0, 1, size=(2,2))
    rv_n = srng.normal(0, 1, size=(2,2))
    f = function([], rv_u)
    g = function([], rv_n, no_default_updates=True)
    nearly_zeros = function([], rv_u + rv_u - 2 * rv_u)

Here, ``rv_u`` represents a random stream of 2x2 matrices of draws from a uniform
distribution.  Likewise,  ``rv_n`` represents a random stream of 2x2 matrices of
draws from a normal distribution.  The distributions that are implemented are
defined as :class:`RandomVariable`\s. They only work on CPU.


Now let's use these objects.  If we call ``f()``, we get random uniform numbers.
The internal state of the random number generator is automatically updated,
so we get different random numbers every time.

>>> f_val0 = f()
>>> f_val1 = f()  #different numbers from f_val0

When we add the extra argument ``no_default_updates=True`` to
``function`` (as in ``g``), then the random number generator state is
not affected by calling the returned function.  So, for example, calling
``g`` multiple times will return the same numbers.

>>> g_val0 = g()  # different numbers from f_val0 and f_val1
>>> g_val1 = g()  # same numbers as g_val0!

An important remark is that a random variable is drawn at most once during any
single function execution.  So the `nearly_zeros` function is guaranteed to
return approximately 0 (except for rounding error) even though the ``rv_u``
random variable appears three times in the output expression.

>>> nearly_zeros = function([], rv_u + rv_u - 2 * rv_u)

Seeding Streams
~~~~~~~~~~~~~~~

You can seed all of the random variables allocated by a :class:`RandomStream`
object by that object's :meth:`RandomStream.seed` method.  This seed will be used to seed a
temporary random number generator, that will in turn generate seeds for each
of the random variables.

>>> srng.seed(902340)  # seeds rv_u and rv_n with different seeds each

Sharing Streams Between Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As usual for shared variables, the random number generators used for random
variables are common between functions.  So our ``nearly_zeros`` function will
update the state of the generators used in function ``f`` above.

Copying Random State Between Aesara Graphs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In some use cases, a user might want to transfer the "state" of all random
number generators associated with a given Aesara graph (e.g. ``g1``, with compiled
function ``f1`` below) to a second graph (e.g. ``g2``, with function ``f2``). This might
arise for example if you are trying to initialize the state of a model, from
the parameters of a pickled version of a previous model. For
:class:`aesara.tensor.random.utils.RandomStream` and
:class:`aesara.sandbox.rng_mrg.MRG_RandomStream`
this can be achieved by copying elements of the `state_updates` parameter.

Each time a random variable is drawn from a `RandomStream` object, a tuple is
added to its :attr:`state_updates` list. The first element is a shared variable,
which represents the state of the random number generator associated with this
*particular* variable, while the second represents the Aesara graph
corresponding to the random number generation process.


A Real Example: Logistic Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The preceding elements are featured in this more realistic example.
It will be used repeatedly.

.. testcode::

    import numpy as np
    import aesara
    import aesara.tensor as at


    rng = np.random.default_rng(2882)

    N = 400                                   # training sample size
    feats = 784                               # number of input variables

    # generate a dataset: D = (input_values, target_class)
    D = (rng.standard_normal((N, feats)), rng.integers(size=N, low=0, high=2))
    training_steps = 10000

    # Declare Aesara symbolic variables
    x = at.dmatrix("x")
    y = at.dvector("y")

    # initialize the weight vector w randomly
    #
    # this and the following bias variable b
    # are shared so they keep their values
    # between training iterations (updates)
    w = aesara.shared(rng.standard_normal(feats), name="w")

    # initialize the bias term
    b = aesara.shared(0., name="b")

    print("Initial model:")
    print(w.get_value())
    print(b.get_value())

    # Construct Aesara expression graph
    p_1 = 1 / (1 + at.exp(-at.dot(x, w) - b))       # Probability that target = 1
    prediction = p_1 > 0.5                          # The prediction thresholded
    xent = -y * at.log(p_1) - (1-y) * at.log(1-p_1) # Cross-entropy loss function
    cost = xent.mean() + 0.01 * (w ** 2).sum()      # The cost to minimize
    gw, gb = at.grad(cost, [w, b])                  # Compute the gradient of the cost
                                                    # w.r.t weight vector w and
                                                    # bias term b (we shall
                                                    # return to this in a
                                                    # following section of this
                                                    # tutorial)

    # Compile
    train = aesara.function(
              inputs=[x,y],
              outputs=[prediction, xent],
              updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
    predict = aesara.function(inputs=[x], outputs=prediction)

    # Train
    for i in range(training_steps):
        pred, err = train(D[0], D[1])

    print("Final model:")
    print(w.get_value())
    print(b.get_value())
    print("target values for D:")
    print(D[1])
    print("prediction on D:")
    print(predict(D[0]))

The :mod:`aesara.tensor.random` module provides random-number drawing functionality
that closely resembles the :mod:`numpy.random` module.


Guide
-----

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

Quick start
-----------

.. currentmodule:: aesara.tensor.random

Create a new :class:`RandomStream` instance, then call its methods to generate :class:`RandomVariable` with different distributions. The :class:`RandomStream` interface follows that of NumPy's ``Generator``; the implementation details depend on the backend to which the Aesara graph is compiled.

.. testcode:: quickstart_random

   import aesara.tensor as at

   srng = at.random.RandomStream(0)
   x_rv = srng.normal(0, 1)
   y_rv = srng.poisson(1.)


Distributions
-------------

Aesara can produce :class:`RandomVariable`\s that draw samples from many different statistical distributions, using the following :class:`Op`\s. The :class:`RandomVariable`\s behave similarly to NumPy's *Generalized Universal Functions* (or `gunfunc`): it supports "core" random variable :class:`Op`\s that map distinctly shaped inputs to potentially non-scalar outputs. We document this behavior in the following with `gufunc`-like signatures.

.. automodule:: aesara.tensor.random

.. autosummary::
   :toctree: _autosummary

   bernoulli
   beta
   betabinom
   binomial
   categorical
   cauchy
   chisquare
   choice
   dirichlet
   exponential
   gengamma
   geometric
   gamma
   gumbel
   halfcauchy
   halfnormal
   hypergeometric
   laplace
   logistic
   lognormal
   integers
   invgamma
   multinomial
   multivariate_normal
   negative_binomial
   nbinom
   normal
   permutation
   pareto
   poisson
   random
   rayleigh
   standard_normal
   t
   triangular
   truncexpon
   uniform
   vonmises
   wald
   weibull


Reference
---------

.. class:: RandomStream()

   This is a symbolic stand-in for `numpy.random.Generator`.

   A helper class that tracks changes in a shared :class:`numpy.random.RandomState`
   and behaves like :class:`numpy.random.RandomState` by managing access
   to :class:`RandomVariable`\s.  For example:

   .. testcode:: constructors

      from aesara.tensor.random.utils import RandomStream

      rng = RandomStream()
      sample = rng.normal(0, 1, size=(2, 2))

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
