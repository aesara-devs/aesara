
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

Distributions
==============

Aesara can produce :class:`RandomVariable`\s that draw samples from many different statistical distributions, using the following :class:`Op`\s. The :class:`RandomVariable`\s behave similarly to NumPy's *Generalized Universal Functions* (or `gunfunc`): it supports "core" random variable :class:`Op`\s that map distinctly shaped inputs to potentially non-scalar outputs. We document this behavior in the following with `gufunc`-like signatures.

.. autoclass:: aesara.tensor.random.basic.UniformRV
   :members: __call__

.. autoclass:: aesara.tensor.random.basic.RandIntRV
   :members: __call__

.. autoclass:: aesara.tensor.random.basic.IntegersRV
   :members: __call__

.. autoclass:: aesara.tensor.random.basic.ChoiceRV
   :members: __call__

.. autoclass:: aesara.tensor.random.basic.PermutationRV
   :members: __call__

.. autoclass:: aesara.tensor.random.basic.BernoulliRV
   :members: __call__

.. autoclass:: aesara.tensor.random.basic.BetaRV
   :members: __call__

.. autoclass:: aesara.tensor.random.basic.BetaBinomialRV
   :members: __call__

.. autoclass:: aesara.tensor.random.basic.BinomialRV
   :members: __call__

.. autoclass:: aesara.tensor.random.basic.CauchyRV
   :members: __call__

.. autoclass:: aesara.tensor.random.basic.CategoricalRV
   :members: __call__

.. autoclass:: aesara.tensor.random.basic.ChiSquareRV
   :members: __call__

.. autoclass:: aesara.tensor.random.basic.DirichletRV
   :members: __call__

.. autoclass:: aesara.tensor.random.basic.ExponentialRV
   :members: __call__

.. autoclass:: aesara.tensor.random.basic.GammaRV
   :members: __call__

.. autoclass:: aesara.tensor.random.basic.GenGammaRV
   :members: __call__

.. autoclass:: aesara.tensor.random.basic.GeometricRV
   :members: __call__

.. autoclass:: aesara.tensor.random.basic.GumbelRV
   :members: __call__

.. autoclass:: aesara.tensor.random.basic.HalfCauchyRV
   :members: __call__

.. autoclass:: aesara.tensor.random.basic.HalfNormalRV
   :members: __call__

.. autoclass:: aesara.tensor.random.basic.HyperGeometricRV
   :members: __call__

.. autoclass:: aesara.tensor.random.basic.InvGammaRV
   :members: __call__

.. autoclass:: aesara.tensor.random.basic.LaplaceRV
   :members: __call__

.. autoclass:: aesara.tensor.random.basic.LogisticRV
   :members: __call__

.. autoclass:: aesara.tensor.random.basic.LogNormalRV
   :members: __call__

.. autoclass:: aesara.tensor.random.basic.MultinomialRV
   :members: __call__

.. autoclass:: aesara.tensor.random.basic.MvNormalRV
   :members: __call__

.. autoclass:: aesara.tensor.random.basic.NegBinomialRV
   :members: __call__

.. autoclass:: aesara.tensor.random.basic.NormalRV
   :members: __call__

.. autoclass:: aesara.tensor.random.basic.ParetoRV
   :members: __call__

.. autoclass:: aesara.tensor.random.basic.PoissonRV
   :members: __call__

.. autoclass:: aesara.tensor.random.basic.StandardNormalRV
   :members: __call__

.. autoclass:: aesara.tensor.random.basic.StudentTRV
   :members: __call__

.. autoclass:: aesara.tensor.random.basic.TriangularRV
   :members: __call__

.. autoclass:: aesara.tensor.random.basic.TruncExponentialRV
   :members: __call__

.. autoclass:: aesara.tensor.random.basic.VonMisesRV
   :members: __call__

.. autoclass:: aesara.tensor.random.basic.WaldRV
   :members: __call__

.. autoclass:: aesara.tensor.random.basic.WeibullRV
   :members: __call__
