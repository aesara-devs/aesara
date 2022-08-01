import abc
from typing import List, Optional, Union

import numpy as np
import scipy.stats as stats

import aesara
from aesara.tensor.basic import as_tensor_variable
from aesara.tensor.random.op import RandomVariable, default_supp_shape_from_params
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
    r"""A class for straightforward `RandomVariable`\s that use SciPy-based samplers.

    By "straightforward" we mean `RandomVariable`\s for which the output shape
    is entirely determined by broadcasting the distribution parameters
    (e.g. basic scalar distributions).

    The more sophisticated shape logic performed by `RandomVariable` is avoided
    in order to reduce the amount of unnecessary steps taken to correct SciPy's
    shape-reducing defects.

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

        if np.ndim(res) == 0:
            # The sample is an `np.number`, and is not writeable, or non-NumPy
            # type, so we need to clone/create a usable NumPy result
            res = np.asarray(res)

        if size is None:
            # SciPy will sometimes drop broadcastable dimensions; we need to
            # check and, if necessary, add them back
            exp_shape = broadcast_shapes(*[np.shape(a) for a in args[1:-1]])
            if res.shape != exp_shape:
                return np.broadcast_to(res, exp_shape).copy()

        return res


class UniformRV(RandomVariable):
    r"""A uniform continuous random variable.

    The probability density function for `uniform` within the interval :math:`[l, h)` is:

    .. math::
        \begin{split}
            f(x; l, h) = \begin{cases}
                          \frac{1}{h-l}\quad \text{for $l \leq x \leq h$},\\
                           0\quad \text{otherwise}.
                       \end{cases}
        \end{split}

    """
    name = "uniform"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "floatX"
    _print_name = ("U", "\\operatorname{U}")

    def __call__(self, low=0.0, high=1.0, size=None, **kwargs):
        r"""Draw samples from a uniform distribution.

        The results are undefined when `high < low`.

        Parameters
        ----------
        low
           Lower boundary :math:`l` of the output interval; all values generated
           will be greater than or equal to `low`.
        high
           Upper boundary :math:`h` of the output interval; all values generated
           will be less than or equal to `high`.
        size
            Sample shape. If the given size is, e.g. `(m, n, k)` then `m * n * k`
            independent, identically distributed random variables are
            returned. Default is `None` in which case a single random variable
            is returned.

        """
        return super().__call__(low, high, size=size, **kwargs)


uniform = UniformRV()


class TriangularRV(RandomVariable):
    r"""A triangular continuous random variable.

    The probability density function for `triangular` within the interval :math:`[l, r)`
    and mode :math:`m` (where the peak of the distribution occurs) is:

    .. math::

        \begin{split}
            f(x; l, m, r) = \begin{cases}
                                \frac{2(x-l)}{(r-l)(m-l)}\quad \text{for $l \leq x \leq m$},\\
                                \frac{2(r-x)}{(r-l)(r-m)}\quad \text{for $m \leq x \leq r$},\\
                                0\quad \text{otherwise}.
                            \end{cases}
        \end{split}

    """
    name = "triangular"
    ndim_supp = 0
    ndims_params = [0, 0, 0]
    dtype = "floatX"
    _print_name = ("Triang", "\\operatorname{Triang}")

    def __call__(self, left, mode, right, size=None, **kwargs):
        r"""Draw samples from a triangular distribution.

        Parameters
        ----------
        left
           Lower boundary :math:`l` of the output interval; all values generated
           will be greater than or equal to `left`.
        mode
           Mode :math:`m` of the distribution, where the peak occurs. Must be such
           that `left <= mode <= right`.
        right
           Upper boundary :math:`r` of the output interval; all values generated
           will be less than or equal to `right`. Must be larger than `left`.
        size
            Sample shape. If the given size is, e.g. `(m, n, k)` then `m * n * k`
            independent, identically distributed random variables are
            returned. Default is `None` in which case a single random variable
            is returned.

        """
        return super().__call__(left, mode, right, size=size, **kwargs)


triangular = TriangularRV()


class BetaRV(RandomVariable):
    r"""A beta continuous random variable.

    The probability density function for `beta` in terms of its parameters :math:`\alpha`
    and :math:`\beta` is:

    .. math::

        f(x; \alpha, \beta) = \frac{1}{B(\alpha, \beta)} x^{\alpha-1} (1-x)^{\beta-1}

    for :math:`0 \leq x \leq 1`. :math:`B` is the beta function defined as:

    .. math::

        B(\alpha, \beta) = \int_0^1 t^{\alpha-1} (1-t)^{\beta-1} \mathrm{d}t

    """
    name = "beta"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "floatX"
    _print_name = ("Beta", "\\operatorname{Beta}")

    def __call__(self, alpha, beta, size=None, **kwargs):
        r"""Draw samples from a beta distribution.

        Parameters
        ----------
        alpha
            Alpha parameter :math:`\alpha` of the distribution. Must be positive.
        beta
            Beta parameter :math:`\beta` of the distribution. Must be positive.
        size
            Sample shape. If the given size is, e.g. `(m, n, k)` then `m * n * k`
            independent, identically distributed random variables are
            returned. Default is `None` in which case a single random variable
            is returned.

        """
        return super().__call__(alpha, beta, size=size, **kwargs)


beta = BetaRV()


class NormalRV(RandomVariable):
    r"""A normal continuous random variable.

    The probability density function for `normal` in terms of its location parameter (mean)
    :math:`\mu` and scale parameter (standard deviation) :math:`\sigma` is:

    .. math::

        f(x; \mu, \sigma) = \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}

    for :math:`\sigma > 0`.

    """
    name = "normal"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "floatX"
    _print_name = ("N", "\\operatorname{N}")

    def __call__(self, loc=0.0, scale=1.0, size=None, **kwargs):
        r"""Draw samples from a normal distribution.

        Parameters
        ----------
        loc
            Mean :math:`\mu` of the normal distribution.
        scale
            Standard deviation :math:`\sigma` of the normal distribution. Must be positive.
        size
            Sample shape. If the given size is, e.g. `(m, n, k)` then `m * n * k`
            independent, identically distributed random variables are
            returned. Default is `None` in which case a single random variable
            is returned.

        """
        return super().__call__(loc, scale, size=size, **kwargs)


normal = NormalRV()


class StandardNormalRV(NormalRV):
    r"""A standard normal continuous random variable.

    The probability density function for `standard_normal` is:

    .. math::

        f(x) = \frac{1}{\sqrt{2 \pi}} e^{-\frac{x^2}{2}}

    """

    def __call__(self, size=None, **kwargs):
        """Draw samples from a standard normal distribution.

        Parameters
        ----------
        size
            Sample shape. If the given size is, e.g. `(m, n, k)` then `m * n * k`
            independent, identically distributed random variables are
            returned. Default is `None` in which case a single random variable
            is returned.

        """
        return super().__call__(loc=0.0, scale=1.0, size=size, **kwargs)


standard_normal = StandardNormalRV()


class HalfNormalRV(ScipyRandomVariable):
    r"""A half-normal continuous random variable.

    The probability density function for `halfnormal` in terms of its location parameter
    :math:`\mu` and scale parameter :math:`\sigma` is:

    .. math::

        f(x; \mu, \sigma) = \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}

    for :math:`x \geq 0` and :math:`\sigma > 0`.

    """
    name = "halfnormal"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "floatX"
    _print_name = ("N**+", "\\operatorname{N^{+}}")

    def __call__(self, loc=0.0, scale=1.0, size=None, **kwargs):
        r"""Draw samples from a half-normal distribution.

        Parameters
        ----------
        loc
            Location parameter :math:`\mu` of the distribution.
        scale
            Scale parameter :math:`\sigma` of the distribution.
        size
            Sample shape. If the given size is, e.g. `(m, n, k)` then `m * n * k`
            independent, identically distributed random variables are
            returned. Default is `None` in which case a single random variable
            is returned.

        """
        return super().__call__(loc, scale, size=size, **kwargs)

    @classmethod
    def rng_fn_scipy(cls, rng, loc, scale, size):
        r"""Draw sample from a half-normal distribution using Scipy's generator.

        Parameters
        ----------
        loc
            Location parameter :math:`\mu` of the distribution.
        scale
            Scale parameter :math:`\sigma` of the distribution.
        size
            Sample shape. If the given size is, e.g. `(m, n, k)` then `m * n * k`
            independent, identically distributed random variables are
            returned. Default is `None` in which case a single random variable
            is returned.

        """
        return stats.halfnorm.rvs(loc, scale, random_state=rng, size=size)


halfnormal = HalfNormalRV()


class LogNormalRV(RandomVariable):
    r"""A lognormal continuous random variable.

    The probability density function for `lognormal` in terms of the mean
    parameter :math:`\mu` and sigma parameter :math:`\sigma` is:

    .. math::

        f(x; \mu, \sigma) = \frac{1}{x \sqrt{2 \pi \sigma^2}} e^{-\frac{(\ln(x)-\mu)^2}{2\sigma^2}}

    for :math:`x > 0` and :math:`\sigma > 0`.

    """
    name = "lognormal"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "floatX"
    _print_name = ("LogN", "\\operatorname{LogN}")

    def __call__(self, mean=0.0, sigma=1.0, size=None, **kwargs):
        r"""Draw sample from a lognormal distribution.

        Parameters
        ----------
        mean
            Mean :math:`\mu` of the random variable's natural logarithm.
        sigma
            Standard deviation :math:`\sigma` of the random variable's natural logarithm.
        size
            Sample shape. If the given size is, e.g. `(m, n, k)` then `m * n * k`
            independent, identically distributed random variables are
            returned. Default is `None` in which case a single random variable
            is returned.

        """
        return super().__call__(mean, sigma, size=size, **kwargs)


lognormal = LogNormalRV()


class GammaRV(ScipyRandomVariable):
    r"""A gamma continuous random variable.

    The probability density function for `gamma` in terms of the shape parameter
    :math:`\alpha` and rate parameter :math:`\beta` is:

    .. math::

        f(x; \alpha, \beta) = \frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x}

    for :math:`x \geq 0`, :math:`\alpha > 0` and :math:`\beta > 0`. :math:`\Gamma` is
    the gamma function:

    .. math::

        \Gamma(x) = \int_0^{\infty} t^{x-1} e^{-t} \mathrm{d}t

    """
    name = "gamma"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "floatX"
    _print_name = ("Gamma", "\\operatorname{Gamma}")

    def __call__(self, shape, rate, size=None, **kwargs):
        r"""Draw samples from a gamma distribution.

        Parameters
        ----------
        shape
            The shape :math:`\alpha` of the gamma distribution. Must be positive.
        rate
            The rate :math:`\beta` of the gamma distribution. Must be positive.
        size
            Sample shape. If the given size is, e.g. `(m, n, k)` then `m * n * k`
            independent, identically distributed random variables are
            returned. Default is `None` in which case a single random variable
            is returned.

        """
        return super().__call__(shape, 1.0 / rate, size=size, **kwargs)

    @classmethod
    def rng_fn_scipy(cls, rng, shape, scale, size):
        return stats.gamma.rvs(shape, scale=scale, size=size, random_state=rng)


gamma = GammaRV()


class ChiSquareRV(RandomVariable):
    r"""A chi square continuous random variable.

    The probability density function for `chisquare` in terms of the number of degrees of
    freedom :math:`k` is:

    .. math::

        f(x; k) = \frac{(1/2)^{k/2}}{\Gamma(k/2)} x^{k/2-1} e^{-x/2}

    for :math:`k > 2`. :math:`\Gamma` is the gamma function:

    .. math::

        \Gamma(x) = \int_0^{\infty} t^{x-1} e^{-t} \mathrm{d}t


    This variable is obtained by summing the squares :math:`k` independent, standard normally
    distributed random variables.

    """
    name = "chisquare"
    ndim_supp = 0
    ndims_params = [0]
    dtype = "floatX"
    _print_name = ("ChiSquare", "\\operatorname{ChiSquare}")

    def __call__(self, df, size=None, **kwargs):
        r"""Draw samples from a chisquare distribution.

        Parameters
        ----------
        df
            The number :math:`k` of degrees of freedom. Must be positive.
        size
            Sample shape. If the given size is, e.g. `(m, n, k)` then `m * n * k`
            independent, identically distributed random variables are
            returned. Default is `None` in which case a single random variable
            is returned.

        """
        return super().__call__(df, size=size, **kwargs)


chisquare = ChiSquareRV()


class ParetoRV(ScipyRandomVariable):
    r"""A pareto continuous random variable.

    The probability density function for `pareto` in terms of its shape parameter :math:`b` and
    scale parameter :math:`x_m` is:

    .. math::

        f(x; b, x_m) = \frac{b x_m^b}{x^{b+1}}

    and is defined for :math:`x \geq x_m`.

    """
    name = "pareto"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "floatX"
    _print_name = ("Pareto", "\\operatorname{Pareto}")

    def __call__(self, b, scale=1.0, size=None, **kwargs):
        r"""Draw samples from a pareto distribution.

        Parameters
        ----------
        b
            The shape :math:`b` (or exponent) of the pareto distribution. Must be positive.
        scale
            The scale :math:`x_m` of the pareto distribution. Must be positive.
        size
            Sample shape. If the given size is, e.g. `(m, n, k)` then `m * n * k`
            independent, identically distributed random variables are
            returned. Default is `None` in which case a single random variable
            is returned.

        """
        return super().__call__(b, scale, size=size, **kwargs)

    @classmethod
    def rng_fn_scipy(cls, rng, b, scale, size):
        return stats.pareto.rvs(b, scale=scale, size=size, random_state=rng)


pareto = ParetoRV()


class GumbelRV(ScipyRandomVariable):
    r"""A gumbel continuous random variable.

    The probability density function for `gumbel` in terms of its location parameter :math:`\mu` and
    scale parameter :math:`\beta` is:

    .. math::

        f(x; \mu, \beta) = \frac{\exp(-(x + e^{(x-\mu)/\beta})}{\beta}

    for :math:`\beta > 0`.

    """
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
        r"""Draw samples from a gumbel distribution.

        Parameters
        ----------
        loc
            The location parameter :math:`\mu` of the distribution.
        scale
            The scale :math:`\beta` of the distribution. Must be positive.
        size
            Sample shape. If the given size is, e.g. `(m, n, k)` then `m * n * k`
            independent, identically distributed random variables are
            returned. Default is `None` in which case a single random variable
            is returned.

        """
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
    r"""An exponential continuous random variable.

    The probability density function for `exponential` in terms of its scale parameter :math:`\beta` is:

    .. math::

        f(x; \beta) = \frac{\exp(-x / \beta)}{\beta}

    for :math:`x \geq 0` and :math:`\beta > 0`.

    """
    name = "exponential"
    ndim_supp = 0
    ndims_params = [0]
    dtype = "floatX"
    _print_name = ("Exp", "\\operatorname{Exp}")

    def __call__(self, scale=1.0, size=None, **kwargs):
        r"""Draw samples from an exponential distribution.

        Parameters
        ----------
        scale
            The scale :math:`\beta` of the distribution. Must be positive.
        size
            Sample shape. If the given size is, e.g. `(m, n, k)` then `m * n * k`
            independent, identically distributed random variables are
            returned. Default is `None` in which case a single random variable
            is returned.

        """
        return super().__call__(scale, size=size, **kwargs)


exponential = ExponentialRV()


class WeibullRV(RandomVariable):
    r"""A weibull continuous random variable.

    The probability density function for `weibull` in terms of its shape parameter :math:`k` is :

    .. math::

        f(x; k) = k x^{k-1} e^{-x^k}

    for :math:`x \geq 0` and :math:`k > 0`.

    """
    name = "weibull"
    ndim_supp = 0
    ndims_params = [0]
    dtype = "floatX"
    _print_name = ("Weibull", "\\operatorname{Weibull}")

    def __call__(self, shape, size=None, **kwargs):
        r"""Draw samples from a weibull distribution.

        Parameters
        ----------
        shape
            The shape :math:`k` of the distribution. Must be positive.
        size
            Sample shape. If the given size is, e.g. `(m, n, k)` then `m * n * k`
            independent, identically distributed random variables are
            returned. Default is `None` in which case a single random variable
            is returned.

        """
        return super().__call__(shape, size=size, **kwargs)


weibull = WeibullRV()


class LogisticRV(RandomVariable):
    r"""A logistic continuous random variable.

    The probability density function for `logistic` in terms of its location parameter :math:`\mu` and
    scale parameter :math:`s` is :

    .. math::

        f(x; \mu, s) = \frac{e^{-(x-\mu)/s}}{s(1+e^{-(x-\mu)/s})^2}

    for :math:`s > 0`.

    """
    name = "logistic"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "floatX"
    _print_name = ("Logistic", "\\operatorname{Logistic}")

    def __call__(self, loc=0, scale=1, size=None, **kwargs):
        r"""Draw samples from a logistic distribution.

        Parameters
        ----------
        loc
            The location parameter :math:`\mu` of the distribution.
        scale
            The scale :math:`s` of the distribution. Must be positive.
        size
            Sample shape. If the given size is, e.g. `(m, n, k)` then `m * n * k`
            independent, identically distributed random variables are
            returned. Default is `None` in which case a single random variable
            is returned.

        """
        return super().__call__(loc, scale, size=size, **kwargs)


logistic = LogisticRV()


class VonMisesRV(RandomVariable):
    r"""A von Misses continuous random variable.

    The probability density function for `vonmisses` in terms of its mode :math:`\mu` and
    dispersion parameter :math:`\kappa` is :

    .. math::

        f(x; \mu, \kappa) = \frac{e^{\kappa \cos(x-\mu)}}{2 \pi I_0(\kappa)}

    for :math:`x \in [-\pi, \pi]` and :math:`\kappa > 0`. :math:`I_0` is the modified Bessel
    function of order 0.

    """
    name = "vonmises"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "floatX"
    _print_name = ("VonMises", "\\operatorname{VonMises}")

    def __call__(self, mu, kappa, size=None, **kwargs):
        r"""Draw samples from a von Mises distribution.

        Parameters
        ----------
        mu
            The mode :math:`\mu` of the distribution.
        kappa
            The dispersion parameter :math:`\kappa` of the distribution. Must be positive.
        size
            Sample shape. If the given size is, e.g. `(m, n, k)` then `m * n * k`
            independent, identically distributed random variables are
            returned. Default is `None` in which case a single random variable
            is returned.

        """
        return super().__call__(mu, kappa, size=size, **kwargs)


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

            size = tuple(size or ())
            if size:
                mean = np.broadcast_to(mean, size + mean.shape[-1:])
                cov = np.broadcast_to(cov, size + cov.shape[-2:])
            else:
                mean, cov = broadcast_params([mean, cov], cls.ndims_params)

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
    r"""A Dirichlet continuous random variable.

    The probability density function for `dirichlet` in terms of the vector of
    concentration parameters :math:`\boldsymbol{\alpha}` is:

    .. math::

        f(x; \boldsymbol{\alpha}) = \prod_{i=1}^k x_i^{\alpha_i-1}

    where :math:`x` is a vector, such that :math:`x_i > 0\;\forall i` and
    :math:`\sum_{i=1}^k x_i = 1`.

    """
    name = "dirichlet"
    ndim_supp = 1
    ndims_params = [1]
    dtype = "floatX"
    _print_name = ("Dir", "\\operatorname{Dir}")

    def __call__(self, alphas, size=None, **kwargs):
        r"""Draw samples from a dirichlet distribution.

        Parameters
        ----------
        alphas
            A sequence of concentration parameters :math:`\boldsymbol{\alpha}` of the
            distribution. A sequence of length `k` will produce samples of length `k`.
        size
            Sample shape. If the given size is, e.g. `(m, n, k)` then `m * n * k`
            independent, identically distributed random variables are
            returned. Default is `None` in which case a vector of length `k`
            is returned.

        """
        return super().__call__(alphas, size=size, **kwargs)

    @classmethod
    def rng_fn(cls, rng, alphas, size):
        if alphas.ndim > 1:
            if size is None:
                size = ()

            size = tuple(np.atleast_1d(size))

            if size:
                alphas = np.broadcast_to(alphas, size + alphas.shape[-1:])

            samples_shape = alphas.shape
            samples = np.empty(samples_shape)
            for index in np.ndindex(*samples_shape[:-1]):
                samples[index] = rng.dirichlet(alphas[index])

            return samples
        else:
            return rng.dirichlet(alphas, size=size)


dirichlet = DirichletRV()


class PoissonRV(RandomVariable):
    r"""A poisson discrete random variable.

    The probability mass function for `poisson` in terms of the expected number
    of events :math:`\lambda` is:

    .. math::

        f(k; \lambda) = \frac{\lambda^k e^{-\lambda}}{k!}

    for :math:`\lambda > 0`.

    """
    name = "poisson"
    ndim_supp = 0
    ndims_params = [0]
    dtype = "int64"
    _print_name = ("Pois", "\\operatorname{Pois}")

    def __call__(self, lam=1.0, size=None, **kwargs):
        r"""Draw samples from a poisson distribution.

        Parameters
        ----------
        lam
            Expected number of events :math:`\lambda`. Must be positive.
        size
            Sample shape. If the given size is, e.g. `(m, n, k)` then `m * n * k`
            independent, identically distributed random variables are
            returned. Default is `None` in which case a single random variable
            is returned.

        """
        return super().__call__(lam, size=size, **kwargs)


poisson = PoissonRV()


class GeometricRV(RandomVariable):
    r"""A geometric discrete random variable.

    The probability mass function for `geometric` for the number of successes :math:`k`
    before the first failure in terms of the probability of success :math:`p` of a single
    trial is:

    .. math::

        f(k; p) = p^{k-1}(1-p)

    for :math:`0 \geq p \geq 1`.

    """
    name = "geometric"
    ndim_supp = 0
    ndims_params = [0]
    dtype = "int64"
    _print_name = ("Geom", "\\operatorname{Geom}")

    def __call__(self, p, size=None, **kwargs):
        r"""Draw samples from a geometric distribution.

        Parameters
        ----------
        p
           Probability of success :math:`p` of an individual trial.
        size
           Sample shape. If the given size is, e.g. `(m, n, k)` then `m * n * k`
           independent, identically distributed random variables are
           returned. Default is `None` in which case a single random variable
           is returned if `p` is a scalar. Otherwise `np.array(p).size` samples
           are drawn.

        """
        return super().__call__(p, size=size, **kwargs)


geometric = GeometricRV()


class HyperGeometricRV(RandomVariable):
    r"""A hypergeometric discrete random variable.

    The probability mass function for `hypergeometric` for the number of
    successes :math:`k` in :math:`n` draws without replacement, from a
    finite population of size :math:`N` with :math:`K` desired items is:

    .. math::

        f(k; n, N, K) = \frac{{K \choose k} {N-K \choose n-k}}{{N \choose n}}

    """
    name = "hypergeometric"
    ndim_supp = 0
    ndims_params = [0, 0, 0]
    dtype = "int64"
    _print_name = ("HyperGeom", "\\operatorname{HyperGeom}")

    def __call__(self, ngood, nbad, nsample, size=None, **kwargs):
        r"""Draw samples from a geometric distribution.

        Parameters
        ----------
        ngood
            Number :math:`K` of desirable items in the population.
        nbad
            Number :math:`N-K` of undesirable items in the population.
        nsample
            Number :math:`n` of items sampled. Must be less than :math:`N`,
            i.e. `ngood + nbad`.`
        size
           Sample shape. If the given size is, e.g. `(m, n, k)` then `m * n * k`
           independent, identically distributed random variables are
           returned. Default is `None` in which case a single random variable
           is returned if `ngood`, `nbad` and `nsample` are all scalars.
           Otherwise `np.broadcast(ngood, nbad, nsample).size` samples are drawn.

        """
        return super().__call__(ngood, nbad, nsample, size=size, **kwargs)


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
negative_binomial = NegBinomialRV()


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


class GenGammaRV(ScipyRandomVariable):
    name = "gengamma"
    ndim_supp = 0
    ndims_params = [0, 0, 0]
    dtype = "floatX"
    _print_name = ("GG", "\\operatorname{GG}")

    def __call__(self, alpha=1.0, p=1.0, lambd=1.0, size=None, **kwargs):
        return super().__call__(alpha, p, lambd, size=size, **kwargs)

    @classmethod
    def rng_fn_scipy(cls, rng, alpha, p, lambd, size):
        return stats.gengamma.rvs(
            alpha / p, p, scale=lambd, size=size, random_state=rng
        )


gengamma = GenGammaRV()


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

    def _supp_shape_from_params(self, dist_params, rep_param_idx=1, param_shapes=None):
        return default_supp_shape_from_params(
            self.ndim_supp, dist_params, rep_param_idx, param_shapes
        )

    @classmethod
    def rng_fn(cls, rng, n, p, size):
        if n.ndim > 0 or p.ndim > 1:
            size = tuple(size or ())

            if size:
                n = np.broadcast_to(n, size)
                p = np.broadcast_to(p, size + p.shape[-1:])
            else:
                n, p = broadcast_params([n, p], cls.ndims_params)

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
            size = p.shape[:-1]
        else:
            # Check that `size` does not define a shape that would be broadcasted
            # to `p.shape[:-1]` in the call to `vsearchsorted` below.
            if len(size) < (p.ndim - 1):
                raise ValueError("`size` is incompatible with the shape of `p`")
            for s, ps in zip(reversed(size), reversed(p.shape[:-1])):
                if s == 1 and ps != 1:
                    raise ValueError("`size` is incompatible with the shape of `p`")

        unif_samples = rng.uniform(size=size)
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

    def _supp_shape_from_params(self, *args, **kwargs):
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
    "negative_binomial",
    "gengamma",
]
