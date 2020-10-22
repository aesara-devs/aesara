"""
This test is for testing the NanGuardMode.
"""

import logging

import numpy as np
import pytest

import theano
import theano.tensor as tt
from theano.compile.nanguardmode import NanGuardMode


def test_NanGuardMode():
    # Tests if NanGuardMode is working by feeding in numpy.inf and numpy.nans
    # intentionally. A working implementation should be able to capture all
    # the abnormalties.
    x = tt.matrix()
    w = theano.shared(np.random.randn(5, 7).astype(theano.config.floatX))
    y = tt.dot(x, w)

    fun = theano.function(
        [x], y, mode=NanGuardMode(nan_is_error=True, inf_is_error=True)
    )
    a = np.random.randn(3, 5).astype(theano.config.floatX)
    infa = np.tile((np.asarray(100.0) ** 1000000).astype(theano.config.floatX), (3, 5))
    nana = np.tile(np.asarray(np.nan).astype(theano.config.floatX), (3, 5))
    biga = np.tile(np.asarray(1e20).astype(theano.config.floatX), (3, 5))

    fun(a)  # normal values

    # Temporarily silence logger
    _logger = logging.getLogger("theano.compile.nanguardmode")
    try:
        _logger.propagate = False
        with pytest.raises(AssertionError):
            fun(infa)  # INFs
        with pytest.raises(AssertionError):
            fun(nana)  # NANs
        with pytest.raises(AssertionError):
            fun(biga)  # big values
    finally:
        _logger.propagate = True

    # slices
    a = np.random.randn(3, 4, 5).astype(theano.config.floatX)
    infa = np.tile(
        (np.asarray(100.0) ** 1000000).astype(theano.config.floatX), (3, 4, 5)
    )
    nana = np.tile(np.asarray(np.nan).astype(theano.config.floatX), (3, 4, 5))
    biga = np.tile(np.asarray(1e20).astype(theano.config.floatX), (3, 4, 5))

    x = tt.tensor3()
    y = x[:, tt.arange(2), tt.arange(2), None]
    fun = theano.function(
        [x], y, mode=NanGuardMode(nan_is_error=True, inf_is_error=True)
    )
    fun(a)  # normal values
    try:
        _logger.propagate = False
        with pytest.raises(AssertionError):
            fun(infa)  # INFs
        with pytest.raises(AssertionError):
            fun(nana)  # NANs
        with pytest.raises(AssertionError):
            fun(biga)  # big values
    finally:
        _logger.propagate = True
