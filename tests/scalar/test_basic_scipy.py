import numpy as np

import aesara.tensor as aet
from aesara.graph.fg import FunctionGraph
from aesara.link.c.basic import CLinker
from aesara.scalar.basic_scipy import gammainc, gammaincc, gammal, gammau


def test_gammainc_nan():
    x1 = aet.dscalar()
    x2 = aet.dscalar()
    y = gammainc(x1, x2)
    test_func = CLinker().accept(FunctionGraph([x1, x2], [y])).make_function()
    assert np.isnan(test_func(-1, 1))
    assert np.isnan(test_func(1, -1))
    assert np.isnan(test_func(-1, -1))


def test_gammaincc_nan():
    x1 = aet.dscalar()
    x2 = aet.dscalar()
    y = gammaincc(x1, x2)
    test_func = CLinker().accept(FunctionGraph([x1, x2], [y])).make_function()
    assert np.isnan(test_func(-1, 1))
    assert np.isnan(test_func(1, -1))
    assert np.isnan(test_func(-1, -1))


def test_gammal_nan():
    x1 = aet.dscalar()
    x2 = aet.dscalar()
    y = gammal(x1, x2)
    test_func = CLinker().accept(FunctionGraph([x1, x2], [y])).make_function()
    assert np.isnan(test_func(-1, 1))
    assert np.isnan(test_func(1, -1))
    assert np.isnan(test_func(-1, -1))


def test_gammau_nan():
    x1 = aet.dscalar()
    x2 = aet.dscalar()
    y = gammau(x1, x2)
    test_func = CLinker().accept(FunctionGraph([x1, x2], [y])).make_function()
    assert np.isnan(test_func(-1, 1))
    assert np.isnan(test_func(1, -1))
    assert np.isnan(test_func(-1, -1))
