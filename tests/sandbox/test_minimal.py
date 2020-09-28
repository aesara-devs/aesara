import pytest
import numpy as np

from theano import tensor, function
from theano.sandbox.minimal import minimal

from tests import unittest_tools as utt


@pytest.mark.skip(reason="Unfinished test")
class TestMinimal:
    """
    TODO: test dtype conversion
    TODO: test that invalid types are rejected by make_node
    TODO: test that each valid type for A and b works correctly
    """

    def setup_method(self):
        self.rng = np.random.RandomState(utt.fetch_seed(666))

    def test_minimal(self):
        A = tensor.matrix()
        b = tensor.vector()

        f = function([A, b], minimal(A, A, b, b, A))

        Aval = self.rng.randn(5, 5)
        bval = np.arange(5, dtype=float)
        f(Aval, bval)
