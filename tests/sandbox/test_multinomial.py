import numpy as np

import tests.unittest_tools as utt
from theano import config, function, tensor
from theano.sandbox import multinomial


def test_n_samples_1():
    p = tensor.fmatrix()
    u = tensor.fvector()
    n = tensor.iscalar()
    m = multinomial.MultinomialFromUniform("auto")(p, u, n)

    f = function([p, u, n], m, allow_input_downcast=True)

    np.random.seed(12345)
    for i in [1, 5, 10, 100, 1000, 10000]:
        uni = np.random.rand(2 * i).astype(config.floatX)
        res = f([[1.0, 0.0], [0.0, 1.0]], uni, i)
        utt.assert_allclose(res, [[i * 1.0, 0.0], [0.0, i * 1.0]])


def test_n_samples_2():
    p = tensor.fmatrix()
    u = tensor.fvector()
    n = tensor.iscalar()
    m = multinomial.MultinomialFromUniform("auto")(p, u, n)

    f = function([p, u, n], m, allow_input_downcast=True)

    np.random.seed(12345)
    for i in [1, 5, 10, 100, 1000]:
        uni = np.random.rand(i).astype(config.floatX)
        pvals = np.random.randint(1, 1000, (1, 1000)).astype(config.floatX)
        pvals /= pvals.sum(1)
        res = f(pvals, uni, i)
        assert res.sum() == i

    for i in [1, 5, 10, 100, 1000]:
        uni = np.random.rand(i).astype(config.floatX)
        pvals = np.random.randint(1, 1000000, (1, 1000000)).astype(config.floatX)
        pvals /= pvals.sum(1)
        res = f(pvals, uni, i)
        assert res.sum() == i


def test_multinomial_0():
    # This tests the MultinomialFromUniform Op directly, not going through the
    # multinomial() call in GPU random generation.

    p = tensor.fmatrix()
    u = tensor.fvector()

    m = multinomial.MultinomialFromUniform("auto")(p, u)

    # the m*2 allows the multinomial to reuse output
    f = function([p, u], m * 2, allow_input_downcast=True)

    # test that both first and second samples can be drawn
    utt.assert_allclose(f([[1, 0], [0, 1]], [0.1, 0.1]), [[2, 0], [0, 2]])

    # test that both second labels can be drawn
    r = f([[0.2, 0.8], [0.3, 0.7]], [0.31, 0.31])
    utt.assert_allclose(r, [[0, 2], [0, 2]])

    # test that both first labels can be drawn
    r = f([[0.2, 0.8], [0.3, 0.7]], [0.21, 0.21])
    utt.assert_allclose(r, [[0, 2], [2, 0]])

    # change the size to make sure output gets reallocated ok
    # and also make sure that the GPU version doesn't screw up the
    # transposed-ness
    r = f([[0.2, 0.8]], [0.25])
    utt.assert_allclose(r, [[0, 2]])


# TODO: check a bigger example (make sure blocking on GPU is handled correctly)
def test_multinomial_large():
    p = tensor.fmatrix()
    u = tensor.fvector()
    m = multinomial.MultinomialFromUniform("auto")(p, u)
    f = function([p, u], m * 2, allow_input_downcast=True)

    pval = np.arange(10000 * 4, dtype="float32").reshape((10000, 4)) + 0.1
    pval = pval / pval.sum(axis=1)[:, None]
    uval = np.ones_like(pval[:, 0]) * 0.5
    mval = f(pval, uval)

    assert mval.shape == pval.shape
    if config.cast_policy == "custom":
        assert mval.dtype == pval.dtype
    elif config.cast_policy == "numpy+floatX":
        assert mval.dtype == config.floatX
    elif config.cast_policy == "numpy":
        assert mval.dtype == "float64"
    else:
        raise NotImplementedError(config.cast_policy)
    utt.assert_allclose(mval.sum(axis=1), 2)
    asdf = np.asarray([0, 0, 2, 0]) + 0 * pval
    utt.assert_allclose(mval, asdf)  # broadcast over all rows


def test_multinomial_dtypes():
    p = tensor.dmatrix()
    u = tensor.dvector()
    m = multinomial.MultinomialFromUniform("auto")(p, u)
    assert m.dtype == "float64", m.dtype

    p = tensor.fmatrix()
    u = tensor.fvector()
    m = multinomial.MultinomialFromUniform("auto")(p, u)
    assert m.dtype == "float32", m.dtype

    p = tensor.fmatrix()
    u = tensor.fvector()
    m = multinomial.MultinomialFromUniform("float64")(p, u)
    assert m.dtype == "float64", m.dtype
