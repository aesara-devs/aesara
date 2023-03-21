import numpy as np
import pytest

from aesara import shared


@pytest.mark.parametrize(
    "rng", [np.random.RandomState(123), np.random.default_rng(123)]
)
def test_GeneratorSharedVariable(rng):
    s_rng_default = shared(rng)
    s_rng_True = shared(rng, borrow=True)
    s_rng_False = shared(rng, borrow=False)

    # test borrow contract: that False means a copy must have been made
    assert s_rng_default.container.storage[0] is not rng
    assert s_rng_False.container.storage[0] is not rng

    # test current implementation: that True means a copy was not made
    assert s_rng_True.container.storage[0] is rng

    # ensure that all the random number generators are in the same state
    if hasattr(rng, "randn"):
        v = rng.randn()
        v0 = s_rng_default.container.storage[0].randn()
        v1 = s_rng_False.container.storage[0].randn()
    else:
        v = rng.standard_normal()
        v0 = s_rng_default.container.storage[0].standard_normal()
        v1 = s_rng_False.container.storage[0].standard_normal()

    assert v == v0 == v1


@pytest.mark.parametrize(
    "rng", [np.random.RandomState(123), np.random.default_rng(123)]
)
def test_get_value_borrow(rng):
    s_rng = shared(rng)

    r_ = s_rng.container.storage[0]
    r_T = s_rng.get_value(borrow=True)
    r_F = s_rng.get_value(borrow=False)

    # the contract requires that borrow=False returns a copy
    assert r_ is not r_F

    # the current implementation allows for True to return the real thing
    assert r_ is r_T

    # either way, the rngs should all be in the same state
    if hasattr(rng, "rand"):
        assert r_.rand() == r_F.rand()
    else:
        assert r_.standard_normal() == r_F.standard_normal()


@pytest.mark.parametrize(
    "rng", [np.random.RandomState(123), np.random.default_rng(123)]
)
def test_get_value_internal_type(rng):
    s_rng = shared(rng)

    # there is no special behaviour required of return_internal_type
    # this test just ensures that the flag doesn't screw anything up
    # by repeating the get_value_borrow test.
    r_ = s_rng.container.storage[0]
    r_T = s_rng.get_value(borrow=True, return_internal_type=True)
    r_F = s_rng.get_value(borrow=False, return_internal_type=True)

    # the contract requires that borrow=False returns a copy
    assert r_ is not r_F

    # the current implementation allows for True to return the real thing
    assert r_ is r_T

    # either way, the rngs should all be in the same state
    if hasattr(rng, "rand"):
        assert r_.rand() == r_F.rand()
    else:
        assert r_.standard_normal() == r_F.standard_normal()


@pytest.mark.parametrize("rng_ctor", [np.random.RandomState, np.random.default_rng])
def test_set_value_borrow(rng_ctor):
    s_rng = shared(rng_ctor(123))

    new_rng = rng_ctor(234234)

    # Test the borrow contract is respected:
    # assigning with borrow=False makes a copy
    s_rng.set_value(new_rng, borrow=False)
    assert new_rng is not s_rng.container.storage[0]
    if hasattr(new_rng, "randn"):
        assert new_rng.randn() == s_rng.container.storage[0].randn()
    else:
        assert new_rng.standard_normal() == s_rng.container.storage[0].standard_normal()

    # Test that the current implementation is actually borrowing when it can.
    rr = rng_ctor(33)
    s_rng.set_value(rr, borrow=True)
    assert rr is s_rng.container.storage[0]


def test_RandomTypeSharedVariable_str():
    s_rng = shared(np.random.RandomState(123))

    assert str(s_rng).startswith("RandomStateSharedVariable(")

    s_rng = shared(np.random.default_rng(123))

    assert str(s_rng).startswith("RandomGeneratorSharedVariable(")
