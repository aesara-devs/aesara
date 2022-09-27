import numpy as np
import pytest

from aesara import function
from aesara.configdefaults import config
from aesara.sandbox import multinomial
from aesara.sandbox.rng_mrg import MRG_RandomStream as RandomStream
from aesara.tensor.type import fmatrix, fvector, iscalar


class TestOP:
    @pytest.mark.xfail(
        reason="This test is designed around very specific random draws from the old NumPy API"
    )
    def test_select_distinct(self):
        # Tests that ChoiceFromUniform always selects distinct elements

        p = fmatrix()
        u = fvector()
        n = iscalar()
        m = multinomial.ChoiceFromUniform(odtype="auto")(p, u, n)

        f = function([p, u, n], m, allow_input_downcast=True)

        n_elements = 1000
        all_indices = range(n_elements)

        rng = np.random.default_rng(12345)

        expected = [
            np.asarray([[931, 318, 185, 209, 559]]),
            np.asarray([[477, 887, 2, 717, 333, 665, 159, 559, 348, 136]]),
            np.asarray(
                [
                    [
                        546,
                        28,
                        79,
                        665,
                        295,
                        779,
                        433,
                        531,
                        411,
                        716,
                        244,
                        234,
                        70,
                        88,
                        612,
                        639,
                        383,
                        335,
                        451,
                        100,
                        175,
                        492,
                        848,
                        771,
                        559,
                        214,
                        568,
                        596,
                        370,
                        486,
                        855,
                        925,
                        138,
                        300,
                        528,
                        507,
                        730,
                        199,
                        882,
                        357,
                        58,
                        195,
                        705,
                        900,
                        66,
                        468,
                        513,
                        410,
                        816,
                        672,
                    ]
                ]
            ),
        ]

        for i in [5, 10, 50, 100, 500, n_elements]:
            uni = rng.random(i).astype(config.floatX)
            pvals = rng.integers(1, 100, (1, n_elements)).astype(config.floatX)
            pvals /= pvals.sum(1)
            res = f(pvals, uni, i)
            for ii in range(len(expected)):
                if expected[ii].shape == res.shape:
                    assert (expected[ii] == res).all()
            res = np.squeeze(res)
            assert len(res) == i
            assert np.all(np.in1d(np.unique(res), all_indices)), res

    def test_fail_select_alot(self):
        # Tests that ChoiceFromUniform fails when asked to sample more
        # elements than the actual number of elements

        p = fmatrix()
        u = fvector()
        n = iscalar()
        m = multinomial.ChoiceFromUniform(odtype="auto")(p, u, n)

        f = function([p, u, n], m, allow_input_downcast=True)

        n_elements = 100
        n_selected = 200
        rng = np.random.default_rng(12345)
        uni = rng.random(n_selected).astype(config.floatX)
        pvals = rng.integers(1, 100, (1, n_elements)).astype(config.floatX)
        pvals /= pvals.sum(1)
        with pytest.raises(ValueError):
            f(pvals, uni, n_selected)

    def test_select_proportional_to_weight(self):
        # Tests that ChoiceFromUniform selects elements, on average,
        # proportional to the their probabilities

        p = fmatrix()
        u = fvector()
        n = iscalar()
        m = multinomial.ChoiceFromUniform(odtype="auto")(p, u, n)

        f = function([p, u, n], m, allow_input_downcast=True)

        n_elements = 100
        n_selected = 10
        mean_rtol = 0.0005
        rng = np.random.default_rng(12345)
        pvals = rng.integers(1, 100, (1, n_elements)).astype(config.floatX)
        pvals /= pvals.sum(1)
        avg_pvals = np.zeros((n_elements,), dtype=config.floatX)

        for rep in range(10000):
            uni = rng.random(n_selected).astype(config.floatX)
            res = f(pvals, uni, n_selected)
            res = np.squeeze(res)
            avg_pvals[res] += 1
        avg_pvals /= avg_pvals.sum()
        avg_diff = np.mean(abs(avg_pvals - pvals))
        assert avg_diff < mean_rtol, avg_diff


class TestFunction:
    def test_select_distinct(self):
        # Tests that multinomial_wo_replacement always selects distinct elements

        th_rng = RandomStream(12345)

        p = fmatrix()
        n = iscalar()
        with pytest.deprecated_call():
            m = th_rng.multinomial_wo_replacement(pvals=p, n=n)

        f = function([p, n], m, allow_input_downcast=True)

        n_elements = 1000
        all_indices = range(n_elements)
        rng = np.random.default_rng(12345)
        for i in [5, 10, 50, 100, 500, n_elements]:
            pvals = rng.integers(1, 100, (1, n_elements)).astype(config.floatX)
            pvals /= pvals.sum(1)
            res = f(pvals, i)
            res = np.squeeze(res)
            assert len(res) == i
            assert np.all(np.in1d(np.unique(res), all_indices)), res

    def test_fail_select_alot(self):
        # Tests that multinomial_wo_replacement fails when asked to sample more
        # elements than the actual number of elements

        th_rng = RandomStream(12345)

        p = fmatrix()
        n = iscalar()
        with pytest.deprecated_call():
            m = th_rng.multinomial_wo_replacement(pvals=p, n=n)

        f = function([p, n], m, allow_input_downcast=True)

        n_elements = 100
        n_selected = 200
        rng = np.random.default_rng(12345)
        pvals = rng.integers(1, 100, (1, n_elements)).astype(config.floatX)
        pvals /= pvals.sum(1)
        with pytest.raises(ValueError):
            f(pvals, n_selected)

    def test_select_proportional_to_weight(self):
        # Tests that multinomial_wo_replacement selects elements, on average,
        # proportional to the their probabilities

        th_rng = RandomStream(12345)

        p = fmatrix()
        n = iscalar()
        m = th_rng.choice(size=n, p=p, replace=False)

        f = function([p, n], m, allow_input_downcast=True)

        n_elements = 100
        n_selected = 10
        mean_rtol = 0.0005
        rng = np.random.default_rng(12345)
        pvals = rng.integers(1, 100, (1, n_elements)).astype(config.floatX)
        pvals /= pvals.sum(1)
        avg_pvals = np.zeros((n_elements,), dtype=config.floatX)

        for rep in range(10000):
            res = f(pvals, n_selected)
            res = np.squeeze(res)
            avg_pvals[res] += 1
        avg_pvals /= avg_pvals.sum()
        avg_diff = np.mean(abs(avg_pvals - pvals))
        assert avg_diff < mean_rtol
