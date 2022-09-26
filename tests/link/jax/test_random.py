import re

import numpy as np
import pytest
from packaging.version import parse as version_parse

import aesara
import aesara.tensor as at
from aesara.compile.function import function
from aesara.compile.sharedvalue import shared
from aesara.configdefaults import config
from aesara.graph.fg import FunctionGraph
from aesara.tensor.random.basic import RandomVariable
from aesara.tensor.random.utils import RandomStream
from tests.link.jax.test_basic import compare_jax_and_py, jax_mode


jax = pytest.importorskip("jax")


@pytest.mark.xfail(
    version_parse(jax.__version__) >= version_parse("0.2.26"),
    reason="JAX samplers require concrete/static shape values?",
)
@pytest.mark.parametrize(
    "at_dist, dist_params, rng, size",
    [
        (
            at.random.normal,
            (),
            shared(np.random.RandomState(123)),
            10000,
        ),
        (
            at.random.normal,
            (),
            shared(np.random.default_rng(123)),
            10000,
        ),
    ],
)
def test_random_stats(at_dist, dist_params, rng, size):
    # The RNG states are not 1:1, so the best we can do is check some summary
    # statistics of the samples
    out = at.random.normal(*dist_params, rng=rng, size=size)
    fgraph = FunctionGraph([out.owner.inputs[0]], [out], clone=False)

    def assert_fn(x, y):
        (x,) = x
        (y,) = y
        assert x.dtype.kind == y.dtype.kind

        d = 2 if config.floatX == "float64" else 1
        np.testing.assert_array_almost_equal(np.abs(x.mean()), np.abs(y.mean()), d)

    compare_jax_and_py(fgraph, [], assert_fn=assert_fn)


def test_random_unimplemented():
    class NonExistentRV(RandomVariable):
        name = "non-existent"
        ndim_supp = 0
        ndims_params = []
        dtype = "floatX"

        def __call__(self, size=None, **kwargs):
            return super().__call__(size=size, **kwargs)

        def rng_fn(cls, rng, size):
            return 0

    nonexistentrv = NonExistentRV()
    rng = shared(np.random.RandomState(123))
    out = nonexistentrv(rng=rng)
    fgraph = FunctionGraph([out.owner.inputs[0]], [out], clone=False)

    with pytest.raises(NotImplementedError):
        compare_jax_and_py(fgraph, [])


def test_RandomStream():
    srng = RandomStream(seed=123)
    out = srng.normal() - srng.normal()

    with pytest.warns(
        UserWarning,
        match=r"The RandomType SharedVariables \[.+\] will not be used",
    ):
        fn = function([], out, mode=jax_mode)
    jax_res_1 = fn()
    jax_res_2 = fn()

    assert not np.array_equal(jax_res_1, jax_res_2)


@pytest.mark.parametrize("rng_ctor", (np.random.RandomState, np.random.default_rng))
def test_random_updates(rng_ctor):
    original_value = rng_ctor(seed=98)
    rng = shared(original_value, name="original_rng", borrow=False)
    next_rng, x = at.random.normal(name="x", rng=rng).owner.outputs

    with pytest.warns(
        UserWarning,
        match=re.escape(
            "The RandomType SharedVariables [original_rng] will not be used"
        ),
    ):
        f = aesara.function([], [x], updates={rng: next_rng}, mode=jax_mode)
    assert f() != f()

    # Check that original rng variable content was not overwritten when calling jax_typify
    assert all(
        a == b if not isinstance(a, np.ndarray) else np.array_equal(a, b)
        for a, b in zip(rng.get_value().__getstate__(), original_value.__getstate__())
    )
