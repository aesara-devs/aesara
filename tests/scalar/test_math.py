import numpy as np
import scipy.special as sp

import aesara.tensor as at
from aesara import function
from aesara.compile.mode import Mode
from aesara.graph.fg import FunctionGraph
from aesara.link.c.basic import CLinker
from aesara.scalar.math import betainc, betainc_der, gammainc, gammaincc, gammal, gammau
from tests.link.test_link import make_function


def test_gammainc_python():
    x1 = at.dscalar()
    x2 = at.dscalar()
    y = gammainc(x1, x2)
    test_func = function([x1, x2], y, mode=Mode("py"))
    assert np.isclose(test_func(1, 2), sp.gammainc(1, 2))


def test_gammainc_nan_c():
    x1 = at.dscalar()
    x2 = at.dscalar()
    y = gammainc(x1, x2)
    test_func = make_function(CLinker().accept(FunctionGraph([x1, x2], [y])))
    assert np.isnan(test_func(-1, 1))
    assert np.isnan(test_func(1, -1))
    assert np.isnan(test_func(-1, -1))


def test_gammaincc_python():
    x1 = at.dscalar()
    x2 = at.dscalar()
    y = gammaincc(x1, x2)
    test_func = function([x1, x2], y, mode=Mode("py"))
    assert np.isclose(test_func(1, 2), sp.gammaincc(1, 2))


def test_gammaincc_nan_c():
    x1 = at.dscalar()
    x2 = at.dscalar()
    y = gammaincc(x1, x2)
    test_func = make_function(CLinker().accept(FunctionGraph([x1, x2], [y])))
    assert np.isnan(test_func(-1, 1))
    assert np.isnan(test_func(1, -1))
    assert np.isnan(test_func(-1, -1))


def test_gammal_nan_c():
    x1 = at.dscalar()
    x2 = at.dscalar()
    y = gammal(x1, x2)
    test_func = make_function(CLinker().accept(FunctionGraph([x1, x2], [y])))
    assert np.isnan(test_func(-1, 1))
    assert np.isnan(test_func(1, -1))
    assert np.isnan(test_func(-1, -1))


def test_gammau_nan_c():
    x1 = at.dscalar()
    x2 = at.dscalar()
    y = gammau(x1, x2)
    test_func = make_function(CLinker().accept(FunctionGraph([x1, x2], [y])))
    assert np.isnan(test_func(-1, 1))
    assert np.isnan(test_func(1, -1))
    assert np.isnan(test_func(-1, -1))


def test_betainc():
    a, b, x = at.scalars("a", "b", "x")
    res = betainc(a, b, x)
    test_func = function([a, b, x], res, mode=Mode("py"))
    assert np.isclose(test_func(15, 10, 0.7), sp.betainc(15, 10, 0.7))


def test_betainc_derivative_nan():
    a, b, x = at.scalars("a", "b", "x")
    res = betainc_der(a, b, x, True)
    test_func = function([a, b, x], res, mode=Mode("py"))
    assert not np.isnan(test_func(1, 1, 1))
    assert np.isnan(test_func(1, 1, -1))
    assert np.isnan(test_func(1, 1, 2))
    assert np.isnan(test_func(1, -1, 1))
    assert np.isnan(test_func(1, 1, -1))
