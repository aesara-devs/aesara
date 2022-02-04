import abc
from typing import List, Optional, Union

import numpy as np
import scipy.stats as stats

import aesara
from aesara.tensor.basic import as_tensor_variable
from aesara.tensor.random.op import RandomVariable, default_shape_from_params
from aesara.tensor.random.type import RandomGeneratorType, RandomStateType
from aesara.tensor.random.utils import broadcast_params
from aesara.tensor.random.var import (
    RandomGeneratorSharedVariable,
    RandomStateSharedVariable,
)


try:
    broadcast_shapes = np.broadcast_shapes
except AttributeError:
    from numpy.lib.stride_tricks import _broadcast_shape

    def broadcast_shapes(*shapes):
        return _broadcast_shape(*[np.empty(x, dtype=[]) for x in shapes])


class ScipyRandomVariable(RandomVariable):
    r"""A class for `RandomVariable`\s that use SciPy-based samplers.

    This will only work for `RandomVariable`\s for which the output shape is
    entirely determined by broadcasting the distribution parameters (e.g. basic
    scalar distributions).

    The more sophisticated shape logic performed by `RandomVariable` is avoided
    in order to reduce the amount of unnecessary extra steps taken to correct
    SciPy's shape-removing defect.

    """

    @classmethod
    @abc.abstractmethod
    def rng_fn_scipy(cls, rng, *args, **kwargs):
        r"""

        `RandomVariable`\s implementations that want to use SciPy-based samplers
        need to implement this method instead of the base
        `RandomVariable.rng_fn`; otherwise their broadcast dimensions will be
        dropped by SciPy.

        """

    @classmethod
    def rng_fn(cls, *args, **kwargs):
        size = args[-1]
        res = cls.rng_fn_scipy(*args, **kwargs)
        return np.broadcast_to(
            res,
            size
            if size is not None
            else broadcast_shapes(*[np.shape(a) for a in args[1:-1]]),
        )


class UniformRV(RandomVariable):
    name = "uniform"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "floatX"
    _print_name = ("U", "\\operatorname{U}")

    def __call__(self, low=0.0, high=1.0, size=None, **kwargs):
        return super().__call__(low, high, size=size, **kwargs)


uniform = UniformRV()


class TriangularRV(RandomVariable):
    name = "triangular"
    ndim_supp = 0
    ndims_params = [0, 0, 0]
    dtype = "floatX"
    _print_name = ("Triang", "\\operatorname{Triang}")


triangular = TriangularRV()


class BetaRV(RandomVariable):
    name = "beta"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "floatX"
    _print_name = ("Beta", "\\operatorname{Beta}")


beta = BetaRV()


class NormalRV(RandomVariable):
    name = "normal"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "floatX"
    _print_name = ("N", "\\operatorname{N}")

    def __call__(self, loc=0.0, scale=1.0, size=None, **kwargs):
        return super().__call__(loc, scale, size=size, **kwargs)


normal = NormalRV()


class StandardNormalRV(NormalRV):
    def __call__(self, size=None, **kwargs):
        return super().__call__(loc=0.0, scale=1.0, size=size, **kwargs)


standard_normal = StandardNormalRV()


class HalfNormalRV(ScipyRandomVariable):
    name = "halfnormal"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "floatX"
    _print_name = ("N**+", "\\operatorname{N^{+}}")

    def __call__(self, loc=0.0, scale=1.0, size=None, **kwargs):
        return super().__call__(loc, scale, size=size, **kwargs)

    @classmethod
    def rng_fn_scipy(cls, rng, loc, scale, size):
        return stats.halfnorm.rvs(loc, scale, random_state=rng, size=size)


halfnormal = HalfNormalRV()


class LogNormalRV(RandomVariable):
    name = "lognormal"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "floatX"
    _print_name = ("LogN", "\\operatorname{LogN}")

    def __call__(self, mean=0.0, sigma=1.0, size=None, **kwargs):
        return super().__call__(mean, sigma, size=size, **kwargs)


lognormal = LogNormalRV()


class GammaRV(ScipyRandomVariable):
    name = "gamma"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "floatX"
    _print_name = ("Gamma", "\\operatorname{Gamma}")

    def __call__(self, shape, rate, size=None, **kwargs):
        return super().__call__(shape, 1.0 / rate, size=size, **kwargs)

    @classmethod
    def rng_fn_scipy(cls, rng, shape, scale, size):
        return stats.gamma.rvs(shape, scale=scale, size=size, random_state=rng)


gamma = GammaRV()


class ChiSquareRV(RandomVariable):
    name = "chisquare"
    ndim_supp = 0
    ndims_params = [0]
    dtype = "floatX"
    _print_name = ("ChiSquare", "\\operatorname{ChiSquare}")


chisquare = ChiSquareRV()


class ParetoRV(ScipyRandomVariable):
    name = "pareto"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "floatX"
    _print_name = ("Pareto", "\\operatorname{Pareto}")

    def __call__(self, b, scale=1.0, size=None, **kwargs):
        return super().__call__(b, scale, size=size, **kwargs)

    @classmethod
    def rng_fn_scipy(cls, rng, b, scale, size):
        return stats.pareto.rvs(b, scale=scale, size=size, random_state=rng)


pareto = ParetoRV()


class GumbelRV(ScipyRandomVariable):
    name = "gumbel"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "floatX"
    _print_name = ("Gumbel", "\\operatorname{Gumbel}")

    def __call__(
        self,
        loc: Union[np.ndarray, float],
        scale: Union[np.ndarray, float] = 1.0,
        size: Optional[Union[List[int], int]] = None,
        **kwargs,
    ) -> RandomVariable:
        return super().__call__(loc, scale, size=size, **kwargs)

    @classmethod
    def rng_fn_scipy(
        cls,
        rng: Union[np.random.Generator, np.random.RandomState],
        loc: Union[np.ndarray, float],
        scale: Union[np.ndarray, float],
        size: Optional[Union[List[int], int]],
    ) -> np.ndarray:
        return stats.gumbel_r.rvs(loc=loc, scale=scale, size=size, random_state=rng)


gumbel = GumbelRV()


class ExponentialRV(RandomVariable):
    name = "exponential"
    ndim_supp = 0
    ndims_params = [0]
    dtype = "floatX"
    _print_name = ("Exp", "\\operatorname{Exp}")

    def __call__(self, scale=1.0, size=None, **kwargs):
        return super().__call__(scale, size=size, **kwargs)


exponential = ExponentialRV()


class WeibullRV(RandomVariable):
    name = "weibull"
    ndim_supp = 0
    ndims_params = [0]
    dtype = "floatX"
    _print_name = ("Weibull", "\\operatorname{Weibull}")


weibull = WeibullRV()


class LogisticRV(RandomVariable):
    name = "logistic"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "floatX"
    _print_name = ("Logistic", "\\operatorname{Logistic}")

    def __call__(self, loc=0, scale=1, size=None, **kwargs):
        return super().__call__(loc, scale, size=size, **kwargs)


logistic = LogisticRV()


class VonMisesRV(RandomVariable):
    name = "vonmises"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "floatX"
    _print_name = ("VonMises", "\\operatorname{VonMises}")


vonmises = VonMisesRV()


def safe_multivariate_normal(mean, cov, size=None, rng=None):
    """A shape consistent multivariate normal sampler.

    What we mean by "shape consistent": SciPy will return scalars when the
    arguments are vectors with dimension of size 1.  We require that the output
    be at least 1D, so that it's consistent with the underlying random
    variable.

    """
    res = np.atleast_1d(
        stats.multivariate_normal(mean=mean, cov=cov, allow_singular=True).rvs(
            size=size, random_state=rng
        )
    )

    if size is not None:
        res = res.reshape(list(size) + [-1])

    return res


class MvNormalRV(RandomVariable):
    name = "multivariate_normal"
    ndim_supp = 1
    ndims_params = [1, 2]
    dtype = "floatX"
    _print_name = ("N", "\\operatorname{N}")

    def __call__(self, mean=None, cov=None, size=None, **kwargs):

        dtype = aesara.config.floatX if self.dtype == "floatX" else self.dtype

        if mean is None:
            mean = np.array([0.0], dtype=dtype)
        if cov is None:
            cov = np.array([[1.0]], dtype=dtype)
        return super().__call__(mean, cov, size=size, **kwargs)

    @classmethod
    def rng_fn(cls, rng, mean, cov, size):

        if mean.ndim > 1 or cov.ndim > 2:
            # Neither SciPy nor NumPy implement parameter broadcasting for
            # multivariate normals (or any other multivariate distributions),
            # so we need to implement that here
            mean, cov = broadcast_params([mean, cov], cls.ndims_params)
            size = tuple(size or ())

            if size:
                if (
                    0 < mean.ndim - 1 <= len(size)
                    and size[-mean.ndim + 1 :] != mean.shape[:-1]
                ):
                    raise ValueError(
                        "shape mismatch: objects cannot be broadcast to a single shape"
                    )
                mean = np.broadcast_to(mean, size + mean.shape[-1:])
                cov = np.broadcast_to(cov, size + cov.shape[-2:])

            res = np.empty(mean.shape)
            for idx in np.ndindex(mean.shape[:-1]):
                m = mean[idx]
                c = cov[idx]
                res[idx] = safe_multivariate_normal(m, c, rng=rng)
            return res
        else:
            return safe_multivariate_normal(mean, cov, size=size, rng=rng)


multivariate_normal = MvNormalRV()


class DirichletRV(RandomVariable):
    name = "dirichlet"
    ndim_supp = 1
    ndims_params = [1]
    dtype = "floatX"
    _print_name = ("Dir", "\\operatorname{Dir}")

    @classmethod
    def rng_fn(cls, rng, alphas, size):
        if alphas.ndim > 1:
            if size is None:
                size = ()

            size = tuple(np.atleast_1d(size))

            if size:
                if (
                    0 < alphas.ndim - 1 <= len(size)
                    and size[-alphas.ndim + 1 :] != alphas.shape[:-1]
                ):
                    raise ValueError(
                        "shape mismatch: objects cannot be broadcast to a single shape"
                    )
                samples_shape = size + alphas.shape[-1:]
            else:
                samples_shape = alphas.shape

            samples = np.empty(samples_shape)
            alphas_bcast = np.broadcast_to(alphas, samples_shape)

            for index in np.ndindex(*samples_shape[:-1]):
                samples[index] = rng.dirichlet(alphas_bcast[index])

            return samples
        else:
            return rng.dirichlet(alphas, size=size)


dirichlet = DirichletRV()


class PoissonRV(RandomVariable):
    name = "poisson"
    ndim_supp = 0
    ndims_params = [0]
    dtype = "int64"
    _print_name = ("Pois", "\\operatorname{Pois}")

    def __call__(self, lam=1.0, size=None, **kwargs):
        return super().__call__(lam, size=size, **kwargs)


poisson = PoissonRV()


class GeometricRV(RandomVariable):
    name = "geometric"
    ndim_supp = 0
    ndims_params = [0]
    dtype = "int64"
    _print_name = ("Geom", "\\operatorname{Geom}")


geometric = GeometricRV()


class HyperGeometricRV(RandomVariable):
    name = "hypergeometric"
    ndim_supp = 0
    ndims_params = [0, 0, 0]
    dtype = "int64"
    _print_name = ("HyperGeom", "\\operatorname{HyperGeom}")


hypergeometric = HyperGeometricRV()


class CauchyRV(ScipyRandomVariable):
    name = "cauchy"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "floatX"
    _print_name = ("C", "\\operatorname{C}")

    def __call__(self, loc=0.0, scale=1.0, size=None, **kwargs):
        return super().__call__(loc, scale, size=size, **kwargs)

    @classmethod
    def rng_fn_scipy(cls, rng, loc, scale, size):
        return stats.cauchy.rvs(loc=loc, scale=scale, random_state=rng, size=size)


cauchy = CauchyRV()


class HalfCauchyRV(ScipyRandomVariable):
    name = "halfcauchy"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "floatX"
    _print_name = ("C**+", "\\operatorname{C^{+}}")

    def __call__(self, loc=0.0, scale=1.0, size=None, **kwargs):
        return super().__call__(loc, scale, size=size, **kwargs)

    @classmethod
    def rng_fn_scipy(cls, rng, loc, scale, size):
        return stats.halfcauchy.rvs(loc=loc, scale=scale, random_state=rng, size=size)


halfcauchy = HalfCauchyRV()


class InvGammaRV(ScipyRandomVariable):
    name = "invgamma"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "floatX"
    _print_name = ("InvGamma", "\\operatorname{Gamma^{-1}}")

    @classmethod
    def rng_fn_scipy(cls, rng, shape, rate, size):
        return stats.invgamma.rvs(shape, scale=rate, size=size, random_state=rng)


invgamma = InvGammaRV()


class WaldRV(RandomVariable):
    name = "wald"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "floatX"
    _print_name_ = ("Wald", "\\operatorname{Wald}")

    def __call__(self, mean=1.0, scale=1.0, size=None, **kwargs):
        return super().__call__(mean, scale, size=size, **kwargs)


wald = WaldRV()


class TruncExponentialRV(ScipyRandomVariable):
    name = "truncexpon"
    ndim_supp = 0
    ndims_params = [0, 0, 0]
    dtype = "floatX"
    _print_name = ("TruncExp", "\\operatorname{TruncExp}")

    @classmethod
    def rng_fn_scipy(cls, rng, b, loc, scale, size):
        return stats.truncexpon.rvs(
            b, loc=loc, scale=scale, size=size, random_state=rng
        )


truncexpon = TruncExponentialRV()


class BernoulliRV(ScipyRandomVariable):
    name = "bernoulli"
    ndim_supp = 0
    ndims_params = [0]
    dtype = "int64"
    _print_name = ("Bern", "\\operatorname{Bern}")

    @classmethod
    def rng_fn_scipy(cls, rng, p, size):
        return stats.bernoulli.rvs(p, size=size, random_state=rng)


bernoulli = BernoulliRV()


class LaplaceRV(RandomVariable):
    name = "laplace"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "floatX"
    _print_name = ("Laplace", "\\operatorname{Laplace}")


laplace = LaplaceRV()


class BinomialRV(RandomVariable):
    name = "binomial"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "int64"
    _print_name = ("Binom", "\\operatorname{Binom}")


binomial = BinomialRV()


class NegBinomialRV(ScipyRandomVariable):
    name = "nbinom"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "int64"
    _print_name = ("NB", "\\operatorname{NB}")

    @classmethod
    def rng_fn_scipy(cls, rng, n, p, size):
        return stats.nbinom.rvs(n, p, size=size, random_state=rng)


nbinom = NegBinomialRV()


class BetaBinomialRV(ScipyRandomVariable):
    name = "beta_binomial"
    ndim_supp = 0
    ndims_params = [0, 0, 0]
    dtype = "int64"
    _print_name = ("BetaBinom", "\\operatorname{BetaBinom}")

    @classmethod
    def rng_fn_scipy(cls, rng, n, a, b, size):
        return stats.betabinom.rvs(n, a, b, size=size, random_state=rng)


betabinom = BetaBinomialRV()


class MultinomialRV(RandomVariable):
    """A Multinomial random variable type.

    Notes
    -----
    The length of the support dimension is determined by the last
    dimension in the *second* parameter (i.e.  the probabilities vector).
    """

    name = "multinomial"
    ndim_supp = 1
    ndims_params = [0, 1]
    dtype = "int64"
    _print_name = ("MN", "\\operatorname{MN}")

    def _shape_from_params(self, dist_params, rep_param_idx=1, param_shapes=None):
        return default_shape_from_params(
            self.ndim_supp, dist_params, rep_param_idx, param_shapes
        )

    @classmethod
    def rng_fn(cls, rng, n, p, size):
        if n.ndim > 0 or p.ndim > 1:
            n, p = broadcast_params([n, p], cls.ndims_params)
            size = tuple(size or ())

            if size:
                if 0 < p.ndim - 1 <= len(size) and size[-p.ndim + 1 :] != p.shape[:-1]:
                    raise ValueError(
                        "shape mismatch: objects cannot be broadcast to a single shape"
                    )
                n = np.broadcast_to(n, size)
                p = np.broadcast_to(p, size + p.shape[-1:])

            res = np.empty(p.shape, dtype=cls.dtype)
            for idx in np.ndindex(p.shape[:-1]):
                res[idx] = rng.multinomial(n[idx], p[idx])
            return res
        else:
            return rng.multinomial(n, p, size=size)


multinomial = MultinomialRV()

vsearchsorted = np.vectorize(np.searchsorted, otypes=[int], signature="(n),()->()")


class CategoricalRV(RandomVariable):
    name = "categorical"
    ndim_supp = 0
    ndims_params = [1]
    dtype = "int64"
    _print_name = ("Cat", "\\operatorname{Cat}")

    @classmethod
    def rng_fn(cls, rng, p, size):
        if size is None:
            size = ()

        size = tuple(np.atleast_1d(size))
        ind_shape = p.shape[:-1]

        if len(ind_shape) > 0:
            if len(size) > 0 and size[-len(ind_shape) :] != ind_shape:
                raise ValueError("Parameters shape and size do not match.")

            samples_shape = size[: -len(ind_shape)] + ind_shape
        else:
            samples_shape = size

        unif_samples = rng.uniform(size=samples_shape)
        samples = vsearchsorted(p.cumsum(axis=-1), unif_samples)

        return samples


categorical = CategoricalRV()


class RandIntRV(RandomVariable):
    name = "randint"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "int64"
    _print_name = ("randint", "\\operatorname{randint}")

    def __call__(self, low, high=None, size=None, **kwargs):
        if high is None:
            low, high = 0, low
        return super().__call__(low, high, size=size, **kwargs)

    def make_node(self, rng, *args, **kwargs):
        if not isinstance(
            getattr(rng, "type", None), (RandomStateType, RandomStateSharedVariable)
        ):
            raise TypeError("`randint` is only available for `RandomStateType`s")
        return super().make_node(rng, *args, **kwargs)


randint = RandIntRV()


class IntegersRV(RandomVariable):
    name = "integers"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "int64"
    _print_name = ("integers", "\\operatorname{integers}")

    def __call__(self, low, high=None, size=None, **kwargs):
        if high is None:
            low, high = 0, low
        return super().__call__(low, high, size=size, **kwargs)

    def make_node(self, rng, *args, **kwargs):
        if not isinstance(
            getattr(rng, "type", None),
            (RandomGeneratorType, RandomGeneratorSharedVariable),
        ):
            raise TypeError("`integers` is only available for `RandomGeneratorType`s")
        return super().make_node(rng, *args, **kwargs)


integers = IntegersRV()


class ChoiceRV(RandomVariable):
    name = "choice"
    ndim_supp = 0
    ndims_params = [1, 1, 0]
    dtype = None
    _print_name = ("choice", "\\operatorname{choice}")

    @classmethod
    def rng_fn(cls, rng, a, p, replace, size):
        return rng.choice(a, size, replace, p)

    def _shape_from_params(self, *args, **kwargs):
        raise NotImplementedError()

    def _infer_shape(self, size, dist_params, param_shapes=None):
        return size

    def __call__(self, a, size=None, replace=True, p=None, **kwargs):

        a = as_tensor_variable(a, ndim=1)

        if p is None:
            p = aesara.tensor.type_other.NoneConst.clone()

        if isinstance(replace, bool):
            replace = aesara.tensor.constant(np.array(replace))

        return super().__call__(a, p, replace, size=size, dtype=a.dtype, **kwargs)


choice = ChoiceRV()


class PermutationRV(RandomVariable):
    name = "permutation"
    ndim_supp = 1
    ndims_params = [1]
    dtype = None
    _print_name = ("permutation", "\\operatorname{permutation}")

    @classmethod
    def rng_fn(cls, rng, x, size):
        return rng.permutation(x if x.ndim > 0 else x.item())

    def _infer_shape(self, size, dist_params, param_shapes=None):

        param_shapes = param_shapes or [p.shape for p in dist_params]

        (x,) = dist_params
        (x_shape,) = param_shapes

        if x.ndim == 0:
            return (x,)
        else:
            return x_shape

    def __call__(self, x, **kwargs):
        x = as_tensor_variable(x)
        return super().__call__(x, dtype=x.dtype, **kwargs)


permutation = PermutationRV()


__all__ = [
    "permutation",
    "choice",
    "integers",
    "randint",
    "categorical",
    "multinomial",
    "betabinom",
    "nbinom",
    "binomial",
    "laplace",
    "bernoulli",
    "truncexpon",
    "wald",
    "invgamma",
    "halfcauchy",
    "cauchy",
    "hypergeometric",
    "geometric",
    "poisson",
    "dirichlet",
    "multivariate_normal",
    "vonmises",
    "logistic",
    "weibull",
    "exponential",
    "gumbel",
    "pareto",
    "chisquare",
    "gamma",
    "lognormal",
    "halfnormal",
    "normal",
    "beta",
    "triangular",
    "uniform",
    "standard_normal",
]
