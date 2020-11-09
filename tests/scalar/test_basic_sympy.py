import pytest

import aesara


sympy = pytest.importorskip("sympy")

from aesara.scalar.basic import floats
from aesara.scalar.basic_sympy import SymPyCCode


xs = sympy.Symbol("x")
ys = sympy.Symbol("y")

xt, yt = floats("xy")


@pytest.mark.skipif(not aesara.config.cxx, reason="Need cxx for this test")
def test_SymPyCCode():
    op = SymPyCCode([xs, ys], xs + ys)
    e = op(xt, yt)
    g = aesara.gof.FunctionGraph([xt, yt], [e])
    fn = aesara.gof.CLinker().accept(g).make_function()
    assert fn(1.0, 2.0) == 3.0


def test_grad():
    op = SymPyCCode([xs], xs ** 2)
    zt = op(xt)
    ztprime = aesara.grad(zt, xt)
    assert ztprime.owner.op.expr == 2 * xs


def test_multivar_grad():
    op = SymPyCCode([xs, ys], xs ** 2 + ys ** 3)
    zt = op(xt, yt)
    dzdx, dzdy = aesara.grad(zt, [xt, yt])
    assert dzdx.owner.op.expr == 2 * xs
    assert dzdy.owner.op.expr == 3 * ys ** 2
