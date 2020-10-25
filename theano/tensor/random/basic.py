from functools import partial

import numpy as np
import scipy.stats as stats

import theano
from theano.tensor.basic import as_tensor_variable
from theano.tensor.random.op import RandomVariable, param_supp_shape_fn


try:
    from pypolyagamma import PyPolyaGamma
except ImportError:  # pragma: no cover

    def PyPolyaGamma(*args, **kwargs):
        raise RuntimeError("pypolygamma not installed!")


class UniformRV(RandomVariable):
    _print_name = ("U", "\\operatorname{U}")

    def __init__(self):
        super().__init__(
            "uniform", 0, [0, 0], "uniform", dtype=theano.config.floatX, inplace=True
        )

    def make_node(self, lower, upper, size=None, rng=None, name=None):
        return super().make_node(lower, upper, size=size, rng=rng, name=name)


uniform = UniformRV()


class BetaRV(RandomVariable):
    _print_name = ("Beta", "\\operatorname{Beta}")

    def __init__(self):
        super().__init__(
            "beta", 0, [0, 0], "beta", dtype=theano.config.floatX, inplace=True
        )

    def make_node(self, alpha, beta, size=None, rng=None, name=None):
        return super().make_node(alpha, beta, size=size, rng=rng, name=name)


beta = BetaRV()


class NormalRV(RandomVariable):
    _print_name = ("N", "\\operatorname{N}")

    def __init__(self):
        super().__init__(
            "normal", 0, [0, 0], "normal", dtype=theano.config.floatX, inplace=True
        )

    def make_node(self, mu, sigma, size=None, rng=None, name=None):
        return super().make_node(mu, sigma, size=size, rng=rng, name=name)


normal = NormalRV()


class HalfNormalRV(RandomVariable):
    _print_name = ("N**+", "\\operatorname{N^{+}}")

    def __init__(self):
        super().__init__(
            "halfnormal",
            0,
            [0, 0],
            lambda rng, *args: stats.halfnorm.rvs(*args, random_state=rng),
            dtype=theano.config.floatX,
            inplace=True,
        )

    def make_node(self, mu=0.0, sigma=1.0, size=None, rng=None, name=None):
        return super().make_node(mu, sigma, size=size, rng=rng, name=name)


halfnormal = HalfNormalRV()


class GammaRV(RandomVariable):
    _print_name = ("Gamma", "\\operatorname{Gamma}")

    def __init__(self):
        super().__init__(
            "gamma",
            0,
            [0, 0],
            lambda rng, shape, rate, size: stats.gamma.rvs(
                shape, scale=1.0 / rate, size=size, random_state=rng
            ),
            dtype=theano.config.floatX,
            inplace=True,
        )

    def make_node(self, shape, rate, size=None, rng=None, name=None):
        return super().make_node(shape, rate, size=size, rng=rng, name=name)


gamma = GammaRV()


class ExponentialRV(RandomVariable):
    _print_name = ("Exp", "\\operatorname{Exp}")

    def __init__(self):
        super().__init__(
            "exponential",
            0,
            [0],
            "exponential",
            dtype=theano.config.floatX,
            inplace=True,
        )

    def make_node(self, scale, size=None, rng=None, name=None):
        return super().make_node(scale, size=size, rng=rng, name=name)


exponential = ExponentialRV()


class MvNormalRV(RandomVariable):
    _print_name = ("N", "\\operatorname{N}")

    def __init__(self):
        super().__init__(
            "multivariate_normal",
            1,
            [1, 2],
            self._smpl_fn,
            dtype=theano.config.floatX,
            inplace=True,
        )

    @classmethod
    def _smpl_fn(cls, rng, mean, cov, size):
        res = np.atleast_1d(
            stats.multivariate_normal(mean=mean, cov=cov, allow_singular=True).rvs(
                size=size, random_state=rng
            )
        )

        if size is not None:
            res = res.reshape(list(size) + [-1])

        return res

    def make_node(self, mean, cov, size=None, rng=None, name=None):
        return super().make_node(mean, cov, size=size, rng=rng, name=name)


mvnormal = MvNormalRV()


def sample_dirichlet(rng, alphas, size=None):
    if size is None:
        size = ()
    samples_shape = tuple(np.atleast_1d(size)) + alphas.shape
    samples = np.empty(samples_shape)
    alphas_bcast = np.broadcast_to(alphas, samples_shape)

    for index in np.ndindex(*samples_shape[:-1]):
        samples[index] = rng.dirichlet(alphas_bcast[index])

    return samples


class DirichletRV(RandomVariable):
    _print_name = ("Dir", "\\operatorname{Dir}")

    def __init__(self):
        super().__init__(
            "dirichlet",
            1,
            [1],
            sample_dirichlet,
            dtype=theano.config.floatX,
            inplace=True,
        )

    def make_node(self, alpha, size=None, rng=None, name=None):
        return super().make_node(alpha, size=size, rng=rng, name=name)


dirichlet = DirichletRV()


class PoissonRV(RandomVariable):
    _print_name = ("Pois", "\\operatorname{Pois}")

    def __init__(self):
        super().__init__("poisson", 0, [0], "poisson", dtype="int64", inplace=True)

    def make_node(self, rate, size=None, rng=None, name=None):
        return super().make_node(rate, size=size, rng=rng, name=name)


poisson = PoissonRV()


class CauchyRV(RandomVariable):
    _print_name = ("C", "\\operatorname{C}")

    def __init__(self):
        super().__init__(
            "cauchy",
            0,
            [0, 0],
            lambda rng, *args: stats.cauchy.rvs(*args, random_state=rng),
            dtype=theano.config.floatX,
            inplace=True,
        )

    def make_node(self, loc, scale, size=None, rng=None, name=None):
        return super().make_node(loc, scale, size=size, rng=rng, name=name)


cauchy = CauchyRV()


class HalfCauchyRV(RandomVariable):
    _print_name = ("C**+", "\\operatorname{C^{+}}")

    def __init__(self):
        super().__init__(
            "halfcauchy",
            0,
            [0, 0],
            lambda rng, *args: stats.halfcauchy.rvs(*args, random_state=rng),
            dtype=theano.config.floatX,
            inplace=True,
        )

    def make_node(self, loc=0.0, scale=1.0, size=None, rng=None, name=None):
        return super().make_node(loc, scale, size=size, rng=rng, name=name)


halfcauchy = HalfCauchyRV()


class InvGammaRV(RandomVariable):
    _print_name = ("InvGamma", "\\operatorname{Gamma^{-1}}")

    def __init__(self):
        super().__init__(
            "invgamma",
            0,
            [0, 0],
            lambda rng, shape, rate, size: stats.invgamma.rvs(
                shape, scale=rate, size=size, random_state=rng
            ),
            dtype=theano.config.floatX,
            inplace=True,
        )

    def make_node(self, shape, rate=1.0, size=None, rng=None, name=None):
        return super().make_node(shape, rate, size=size, rng=rng, name=name)


invgamma = InvGammaRV()


class TruncExponentialRV(RandomVariable):
    _print_name = ("TruncExp", "\\operatorname{Exp}")

    def __init__(self):
        super().__init__(
            "truncexpon",
            0,
            [0, 0, 0],
            lambda rng, *args: stats.truncexpon.rvs(*args, random_state=rng),
            dtype=theano.config.floatX,
            inplace=True,
        )

    def make_node(self, b, loc=0.0, scale=1.0, size=None, rng=None, name=None):
        return super().make_node(b, loc, scale, size=size, rng=rng, name=name)


trunc_exponential = TruncExponentialRV()


class BernoulliRV(RandomVariable):
    _print_name = ("Bern", "\\operatorname{Bern}")

    def __init__(self):
        super().__init__(
            "bernoulli",
            0,
            [0],
            lambda rng, *args: stats.bernoulli.rvs(
                args[0], size=args[1], random_state=rng
            ),
            dtype="int64",
            inplace=True,
        )

    def make_node(self, p, size=None, rng=None, name=None):
        return super().make_node(p, size=size, rng=rng, name=name)


bernoulli = BernoulliRV()


class BinomialRV(RandomVariable):
    _print_name = ("Binom", "\\operatorname{Binom}")

    def __init__(self):
        super().__init__("binomial", 0, [0, 0], "binomial", dtype="int64", inplace=True)

    def make_node(self, n, p, size=None, rng=None, name=None):
        return super().make_node(n, p, size=size, rng=rng, name=name)


binomial = BinomialRV()


class NegBinomialRV(RandomVariable):
    _print_name = ("NB", "\\operatorname{NB}")

    def __init__(self):
        super().__init__(
            "neg-binomial",
            0,
            [0, 0],
            lambda rng, n, p, size: stats.nbinom.rvs(n, p, size=size, random_state=rng),
            dtype="int64",
            inplace=True,
        )

    def make_node(self, n, p, size=None, rng=None, name=None):
        return super().make_node(n, p, size=size, rng=rng, name=name)


negbinomial = NegBinomialRV()


class BetaBinomialRV(RandomVariable):
    _print_name = ("BetaBinom", "\\operatorname{BetaBinom}")

    def __init__(self):
        super().__init__(
            "beta_binomial",
            0,
            [0, 0, 0],
            lambda rng, *args: stats.betabinom.rvs(
                *args[:-1], size=args[-1], random_state=rng
            ),
            dtype="int64",
            inplace=True,
        )

    def make_node(self, n, a, b, size=None, rng=None, name=None):
        return super().make_node(n, a, b, size=size, rng=rng, name=name)


betabinomial = BetaBinomialRV()


class MultinomialRV(RandomVariable):
    """A Multinomial random variable type.

    FYI: Support shape is determined by the first dimension in the *second*
    parameter (i.e.  the probabilities vector).

    """

    _print_name = ("MN", "\\operatorname{MN}")

    def __init__(self):
        super().__init__(
            "multinomial",
            1,
            [0, 1],
            "multinomial",
            supp_shape_fn=partial(param_supp_shape_fn, rep_param_idx=1),
            dtype="int64",
            inplace=True,
        )

    def make_node(self, n, pvals, size=None, rng=None, name=None):
        return super().make_node(n, pvals, size=size, rng=rng, name=name)


multinomial = MultinomialRV()

vsearchsorted = np.vectorize(np.searchsorted, otypes=[np.int], signature="(n),()->()")


def sample_categorical(rng, p, size=None):
    if size is None:
        size = ()

    size = tuple(np.atleast_1d(size))
    ind_shape = p.shape[:-1]

    if len(size) > 0 and size[-len(ind_shape) :] != ind_shape:
        raise ValueError("Parameters shape and size do not match.")

    samples_shape = size[: -len(ind_shape)] + ind_shape
    unif_samples = rng.uniform(size=samples_shape)
    samples = vsearchsorted(p.cumsum(axis=-1), unif_samples)

    return samples


class CategoricalRV(RandomVariable):
    _print_name = ("Cat", "\\operatorname{Cat}")

    def __init__(self):
        super().__init__(
            "categorical",
            0,
            [1],
            sample_categorical,
            dtype="int64",
            inplace=True,
        )

    def make_node(self, pvals, size=None, rng=None, name=None):
        return super().make_node(pvals, size=size, rng=rng, name=name)


categorical = CategoricalRV()


class PolyaGammaRV(RandomVariable):
    """Polya-Gamma random variable.

    XXX: This doesn't really use the given RNG, due to the narrowness of the
    sampler package's implementation.
    """

    _print_name = ("PG", "\\operatorname{PG}")

    def __init__(self):
        super().__init__(
            "polya-gamma",
            0,
            [0, 0],
            self._smpl_fn,
            dtype=theano.config.floatX,
            inplace=True,
        )

    def make_node(self, b, c, size=None, rng=None, name=None):
        return super().make_node(b, c, size=size, rng=rng, name=name)

    @classmethod
    def _smpl_fn(cls, rng, b, c, size):
        pg = PyPolyaGamma(rng.randint(2 ** 16))

        if not size and b.shape == c.shape == ():
            return pg.pgdraw(b, c)
        else:
            b, c = np.broadcast_arrays(b, c)
            size = tuple(size or ())

            if len(size) > 0:
                b = np.broadcast_to(b, size)
                c = np.broadcast_to(c, size)

            smpl_val = np.empty(b.shape, dtype="double")

            pg.pgdrawv(
                np.asarray(b.flat).astype("double", copy=True),
                np.asarray(c.flat).astype("double", copy=True),
                np.asarray(smpl_val.flat),
            )
            return smpl_val


polyagamma = PolyaGammaRV()


class RandIntRV(RandomVariable):
    _print_name = ("randint", "\\operatorname{randint}")

    def __init__(self):
        super().__init__("randint", 0, [0, 0], "randint", dtype="int64", inplace=True)

    def make_node(self, low=0, high=1, size=None, rng=None, name=None):
        return super().make_node(low, high, size=size, rng=rng, name=name)


randint = RandIntRV()


class ChoiceRV(RandomVariable):

    _print_name = ("choice", "\\operatorname{choice}")

    def __init__(self):
        super().__init__(
            "choice",
            0,
            [0, 0, 0],
            self._sample,
            inplace=True,
        )

    def _sample(self, rng, a, p, replace, size):
        return rng.choice(a, size, replace, p)

    def _infer_shape(self, size, dist_params, param_shapes=None):
        return size

    def make_node(self, a, size=None, replace=False, p=None, rng=None, name=None):
        a = as_tensor_variable(a)

        if a.ndim > 1:
            raise ValueError("Parameter `a` must be a vector.")

        if p is None:
            p = theano.tensor.type_other.NoneConst.clone()

        replace = theano.tensor.constant(np.array(replace))

        return super().make_node(
            a, p, replace, size=size, rng=rng, name=name, dtype=a.dtype
        )


choice = ChoiceRV()


class PermutationRV(RandomVariable):

    _print_name = ("permutation", "\\operatorname{permutation}")

    def __init__(self):
        super().__init__(
            "permutation",
            0,
            [0, 0],
            lambda rng, x, size: rng.permutation(x if x.ndim > 0 else x.item()),
            inplace=True,
        )

    def _infer_shape(self, size, dist_params, param_shapes=None):

        param_shapes = param_shapes or [p.shape for p in dist_params]

        (x,) = dist_params
        (x_shape,) = param_shapes

        if x.ndim == 0:
            return (x,)
        else:
            return x_shape

    def make_node(self, x, rng=None, name=None):
        x = as_tensor_variable(x)
        return super().make_node(x, rng=rng, name=name, dtype=x.dtype)


permutation = PermutationRV()
