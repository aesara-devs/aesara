from sys import float_info

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings

import aesara
import tests.unittest_tools as utt
from aesara.tensor.type import dscalar, zscalar


pytestmark = pytest.mark.filterwarnings("error")

# functions that accept either real or complex inputs
# the second value in each tupple converts an arbitrary float value into a
# value in the function's domain.
COMPLEX_FUNCTIONS = [
    (np.tanh, np.log),
    (np.cosh, np.log),
    (np.sinh, np.log),
    (np.arcsinh, lambda x: x),
    (np.log, lambda x: x),
    (np.log10, lambda x: x),
    (np.log1p, lambda x: x / 2),
    (np.log2, lambda x: x),
    (np.exp, np.log),
    (np.expm1, np.log),
    (np.exp2, lambda x: np.log2(x) - 1),
    (np.sqrt, abs),
]

# functions that only accept real inputs
# the second value in each tupple converts an arbitrary float value into a
# value in the function's domain
REAL_FUNCTIONS = [
    (np.deg2rad, lambda x: x),
    (np.rad2deg, np.deg2rad),
    (np.cos, lambda x: x),
    (np.sin, lambda x: x),
    (np.tan, lambda x: x),
    (np.arctan, lambda x: x),
    (np.arcsin, lambda x: x / float_info.max),
    (np.arccos, lambda x: x / float_info.max),
]


# tests calling a function with a real value
def do_real_test(fct, value: float):
    # set up
    x = dscalar("x")
    y = fct(x)
    f = aesara.function([x], y)

    # exercise and verify
    utt.assert_allclose(fct(value), f(value))


# tests functions that can be invoked with either real or imaginary inputs
# with real inputs
@pytest.mark.parametrize(
    "fct, to_domain",
    REAL_FUNCTIONS + COMPLEX_FUNCTIONS,
)
@given(value=st.floats(float_info.min, float_info.max))
@settings(deadline=None)
def test_real(fct, to_domain, value):
    do_real_test(fct, to_domain(value))


# arccosh has a domain that is awkward to derive from min/max float
@given(value=st.floats(1, float_info.max))
@settings(deadline=None)
def test_arccosh(value):
    do_real_test(np.arccosh, value)


# arctanh has a domain that is awkward to derive from min/max float
@given(value=st.floats(-1 + float_info.epsilon, 1 - float_info.epsilon))
@settings(deadline=None)
def test_arctanh_real(value):
    do_real_test(np.arctanh, value)


# tests functions with complex inputs
# cscalar doesn't work for because it loses precision
@pytest.mark.parametrize("fct, to_domain", COMPLEX_FUNCTIONS)
@given(
    real_value=st.floats(float_info.min, float_info.max),
    imaginary_value=st.floats(float_info.min, float_info.max),
)
@settings(deadline=None)
def test_complex(fct, to_domain, real_value, imaginary_value):
    # set up
    x = zscalar("x")
    y = fct(x)
    f = aesara.function([x], y)
    value = to_domain(real_value) + 1j * to_domain(imaginary_value)

    # exercise and verify
    utt.assert_allclose(fct(value), f(value))
