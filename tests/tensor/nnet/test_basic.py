from contextlib import ExitStack as does_not_raise

import numpy as np
import pytest
import scipy.special as sp

import aesara
import aesara.tensor as at
from aesara.compile.mode import OPT_FAST_RUN, optdb
from aesara.configdefaults import config
from aesara.gradient import grad
from aesara.graph.fg import FunctionGraph
from aesara.graph.rewriting.basic import check_stack_trace
from aesara.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from aesara.tensor.math import (
    Argmax,
    add,
    argmax,
    dot,
    exp,
    log,
    max_and_argmax,
    mean,
    sigmoid,
)
from aesara.tensor.math import sum as at_sum
from aesara.tensor.math import tanh
from aesara.tensor.nnet.basic import (
    CrossentropyCategorical1Hot,
    CrossentropyCategorical1HotGrad,
    CrossentropySoftmax1HotWithBiasDx,
    CrossentropySoftmaxArgmax1HotWithBias,
    Prepend_scalar_constant_to_each_row,
    Prepend_scalar_to_each_row,
    Softmax,
    SoftmaxGrad,
    SoftmaxWithBias,
    binary_crossentropy,
    categorical_crossentropy,
    confusion_matrix,
    crossentropy_categorical_1hot,
    crossentropy_softmax_1hot,
    crossentropy_softmax_1hot_with_bias,
    crossentropy_softmax_1hot_with_bias_dx,
    crossentropy_softmax_argmax_1hot_with_bias,
    elu,
    h_softmax,
    relu,
    selu,
    sigmoid_binary_crossentropy,
    softmax,
    softmax_grad_legacy,
    softmax_legacy,
    softmax_with_bias,
    softsign,
)
from aesara.tensor.shape import shape_padleft
from aesara.tensor.subtensor import AdvancedSubtensor
from aesara.tensor.type import (
    dmatrix,
    dvector,
    fmatrix,
    fvector,
    ivector,
    lvector,
    matrix,
    scalar,
    tensor3,
    tensor4,
    vector,
    vectors,
)
from tests import unittest_tools as utt
from tests.tensor.utils import (
    _good_broadcast_unary_normal_float_no_complex,
    check_floatX,
    makeBroadcastTester,
    upcast_int8_nfunc,
)


def softmax_graph(c):
    return exp(c) / exp(c).sum(axis=-1, keepdims=True)


def valid_axis_tester(Op):
    with pytest.raises(TypeError):
        Op(1.5)

    x = [tensor3()] * Op.nin
    with does_not_raise():
        Op(2)(*x)

    with pytest.raises(ValueError):
        Op(3)(*x)

    with does_not_raise():
        Op(-3)(*x)

    with pytest.raises(ValueError):
        Op(-4)(*x)


class TestSoftmaxWithBias(utt.InferShapeTester):
    def test_basic(self):
        def f(a, b):
            return softmax_with_bias(a, b)[:, 0]

        rng = np.random.default_rng(utt.fetch_seed())

        utt.verify_grad(f, [rng.random((3, 4)), rng.random(4)])

        def f(a, b):
            return softmax_with_bias(a, b)[:, 1]

        utt.verify_grad(f, [rng.random((3, 4)), rng.random(4)])

        def f(a, b):
            return softmax_with_bias(a, b)[:, 2]

        utt.verify_grad(f, [rng.random((3, 4)), rng.random(4)])

        def f(a, b):
            return softmax_with_bias(a, b)[:, 3]

        utt.verify_grad(f, [rng.random((3, 4)), rng.random(4)])

    def test_broadcast(self):
        """
        Test that we don't raise an error during rewriting for no good reason
        as `softmax_with_bias` don't support correctly some/all broadcasted
        inputs pattern.
        """
        initial_W = np.asarray(
            [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]],
            dtype=config.floatX,
        )
        W = aesara.shared(value=initial_W, name="W")
        vbias = aesara.shared(value=0.1, name="vbias")  # 0.01
        hid = vector("hid")
        f = aesara.function([hid], softmax_legacy(dot(hid, W.T) + vbias))
        ops = [node.op for node in f.maker.fgraph.toposort()]
        assert softmax_with_bias not in ops
        assert softmax_legacy in ops

        f([0, 1, 0])
        # print f.maker.fgraph.toposort()

    def test_softmax_with_bias_trace(self):
        rng = np.random.default_rng(utt.fetch_seed())
        a = aesara.shared(rng.standard_normal((3,)).astype(config.floatX))
        b = aesara.shared(np.float32(rng.standard_normal()))
        sm = softmax(a + b)
        f = aesara.function([], sm)
        assert check_stack_trace(f, ops_to_check="last")

    def test_infer_shape(self):
        admat = matrix()
        advec = vector()
        rng = np.random.default_rng(utt.fetch_seed())
        admat_val = rng.random((3, 4)).astype(config.floatX)
        advec_val = rng.random(4).astype(config.floatX)
        self._compile_and_check(
            [admat, advec],
            [SoftmaxWithBias()(admat, advec)],
            [admat_val, advec_val],
            SoftmaxWithBias,
        )


class TestCrossEntropySoftmax1Hot:
    def test_basic(self):
        y_idx = [0, 1, 3]

        def f(a, b):
            return crossentropy_softmax_1hot_with_bias(a, b, y_idx)[0]

        rng = np.random.default_rng(utt.fetch_seed())

        utt.verify_grad(f, [rng.random((3, 4)), rng.random(4)])

        y_idx = [0, 1, 3]

        def f(a):
            return crossentropy_softmax_1hot(a, y_idx)[0]

        utt.verify_grad(f, [rng.random((3, 4))])

    def test_vector(self):
        y_idx = [3]

        def f(a):
            return crossentropy_softmax_1hot(shape_padleft(a), y_idx)[0]

        rng = np.random.default_rng(utt.fetch_seed())
        utt.verify_grad(f, [rng.random((4,))])

    def test_vectors(self):
        y_idx = [3]

        def f(a, b):
            return crossentropy_softmax_1hot(shape_padleft(a) + b, y_idx)[0]

        rng = np.random.default_rng(utt.fetch_seed())
        utt.verify_grad(f, [rng.random((4,)), rng.random(4)])


class TestCrossEntropySoftmax1HotWithBiasDx(utt.InferShapeTester):
    def test_basic(self):
        rng = np.random.default_rng(utt.fetch_seed())

        def ff(class_dtype):
            def f(sm):
                # Class indices
                y = rng.integers(low=0, high=5, size=10).astype(class_dtype)
                return crossentropy_softmax_1hot_with_bias_dx(
                    rng.random(10),
                    sm,
                    y,  # Gradient w.r.t. NLL.  # Softmax output.
                )

            return f

        # Build a random softmax output whose rows sum to 1.
        softmax_output = rng.random((10, 5))
        softmax_output /= softmax_output.sum(axis=1).reshape(10, 1)
        for dtype in ["uint8", "int8", "uint64", "int64"]:
            utt.verify_grad(ff(dtype), [softmax_output])

    def test_basic_2(self):
        rng = np.random.default_rng(utt.fetch_seed())
        softmax_output = rng.random((10, 5))
        softmax_output /= softmax_output.sum(axis=1).reshape(10, 1)

        def f(dy):
            return crossentropy_softmax_1hot_with_bias_dx(
                dy, softmax_output, rng.integers(low=0, high=5, size=10)
            )

        utt.verify_grad(f, [rng.random(10)])

    def test_infer_shape(self):
        admat = matrix()
        advec = vector()
        alvec = lvector()
        rng = np.random.default_rng(utt.fetch_seed())
        admat_val = rng.random((10, 5)).astype(config.floatX)
        admat_val /= admat_val.sum(axis=1).reshape(10, 1)
        advec_val = rng.random(10).astype(config.floatX)
        alvec_val = rng.integers(low=0, high=5, size=10)
        self._compile_and_check(
            [advec, admat, alvec],
            [CrossentropySoftmax1HotWithBiasDx()(advec, admat, alvec)],
            [advec_val, admat_val, alvec_val],
            CrossentropySoftmax1HotWithBiasDx,
        )

    def test_neg_idx(self):
        admat = matrix()
        advec = vector()
        alvec = lvector()
        rng = np.random.default_rng(utt.fetch_seed())
        admat_val = rng.random((10, 5)).astype(config.floatX)
        admat_val /= admat_val.sum(axis=1).reshape(10, 1)
        advec_val = rng.random(10).astype(config.floatX)
        alvec_val = rng.integers(low=0, high=5, size=10)
        alvec_val[1] = -1
        out = CrossentropySoftmax1HotWithBiasDx()(advec, admat, alvec)
        f = aesara.function([advec, admat, alvec], out)
        with pytest.raises(ValueError):
            f(advec_val, admat_val, alvec_val)


class TestCrossEntropySoftmaxArgmax1HotWithBias(utt.InferShapeTester):
    def setup_method(self):
        self.op = crossentropy_softmax_argmax_1hot_with_bias
        super().setup_method()

    def test_grads(self):
        n_classes = 5
        n_samples = 3

        rng = np.random.default_rng(utt.fetch_seed())

        # First test gradient when getting a gradient on the NLL output.
        def grad_on_nll_dtype(dtype):
            def grad_on_nll(x, b):
                y_idx = rng.integers(low=0, high=n_classes, size=n_samples).astype(
                    dtype
                )
                return self.op(x, b, y_idx=y_idx)[0]

            return grad_on_nll

        for dtype in ["uint8", "int8", "uint64", "int64"]:
            utt.verify_grad(
                grad_on_nll_dtype(dtype),
                [
                    rng.random((n_samples, n_classes)),
                    rng.random(n_classes),
                ],
            )

        # Then test gradient when getting a gradient on the softmax output.
        def grad_on_softmax(x, b):
            return self.op(
                x,
                b,
                y_idx=rng.integers(low=0, high=n_classes, size=n_samples),
            )[1]

        utt.verify_grad(
            grad_on_softmax,
            [rng.random((n_samples, n_classes)), rng.random(n_classes)],
        )

    def test_infer_shape(self):
        admat = matrix()
        advec = vector()
        alvec = lvector()
        rng = np.random.default_rng(utt.fetch_seed())
        admat_val = rng.random((3, 5)).astype(config.floatX)
        advec_val = rng.random(5).astype(config.floatX)
        alvec_val = rng.integers(low=0, high=5, size=3)
        self._compile_and_check(
            [admat, advec, alvec],
            CrossentropySoftmaxArgmax1HotWithBias()(admat, advec, alvec),
            [admat_val, advec_val, alvec_val],
            CrossentropySoftmaxArgmax1HotWithBias,
        )

    def test_neg_idx(self):
        admat = matrix()
        advec = vector()
        alvec = lvector()
        rng = np.random.default_rng(utt.fetch_seed())
        admat_val = rng.random((3, 5)).astype(config.floatX)
        advec_val = rng.random(5).astype(config.floatX)
        alvec_val = rng.integers(low=0, high=5, size=3)
        alvec_val[1] = -1
        out = CrossentropySoftmaxArgmax1HotWithBias()(admat, advec, alvec)
        f = aesara.function([admat, advec, alvec], out)
        with pytest.raises(ValueError):
            f(admat_val, advec_val, alvec_val)


class TestPrepend(utt.InferShapeTester):
    def test_prepend_constant(self):
        x = matrix("x")
        y = Prepend_scalar_constant_to_each_row(4.0)(x)
        f = aesara.function([x], y)
        rng = np.random.default_rng(utt.fetch_seed())
        m = rng.random((3, 5)).astype(config.floatX)
        my = f(m)
        assert my.shape == (3, 6)
        assert np.all(my[:, 0] == 4.0)

    def test_prepend_basic(self):
        """Test basic functionality."""
        x = matrix("x")
        y = Prepend_scalar_to_each_row()(5.0, x)
        f = aesara.function([x], y)
        m = np.ones((3, 5), dtype="float32")
        my = f(m)
        assert my.shape == (3, 6)
        assert np.all(my[:, 0] == 5.0)

    def test_infer_shape(self):
        admat = matrix()
        adscal = scalar()
        rng = np.random.default_rng(utt.fetch_seed())
        admat_val = rng.random((3, 5)).astype(config.floatX)
        adscal_val = np.asarray(rng.random(), dtype=config.floatX).item()
        self._compile_and_check(
            [admat],
            [Prepend_scalar_constant_to_each_row(adscal_val)(admat)],
            [admat_val],
            Prepend_scalar_constant_to_each_row,
        )

        self._compile_and_check(
            [adscal, admat],
            [Prepend_scalar_to_each_row()(adscal, admat)],
            [adscal_val, admat_val],
            Prepend_scalar_to_each_row,
        )


class TestCrossEntropyCategorical1HotGrad(utt.InferShapeTester):
    def test_infer_shape(self):
        advec = vector()
        admat = matrix()
        alvec = lvector()
        rng = np.random.default_rng(utt.fetch_seed())
        advec_val = rng.random(3).astype(config.floatX)
        admat_val = rng.random((3, 2)).astype(config.floatX)
        alvec_val = [0, 1, 0]
        self._compile_and_check(
            [advec, admat, alvec],
            [CrossentropyCategorical1HotGrad()(advec, admat, alvec)],
            [advec_val, admat_val, alvec_val],
            CrossentropyCategorical1HotGrad,
        )


class TestCrossEntropyCategorical1Hot(utt.InferShapeTester):
    def test_input_validation(self):
        with pytest.raises(TypeError, match="Matrix.*"):
            crossentropy_categorical_1hot(vector(), lvector())

        with pytest.raises(TypeError, match="Integer.*"):
            crossentropy_categorical_1hot(matrix(), vector())

    def test_grad(self):
        x = matrix("x")
        one_of_n = lvector("one_of_n")
        op = crossentropy_categorical_1hot
        xe = op(x, one_of_n)
        f = aesara.function([x, one_of_n], xe)
        x_val = np.asarray([[0.4, 0.6, 0.0], [0.1, 0.8, 0.1]], dtype=config.floatX)
        xe_val = f(x_val, [0, 1])
        assert np.allclose(xe_val, -np.log([0.4, 0.8]))

        def oplike(x):
            return op(x, [0, 1])

        rng = np.random.default_rng(utt.fetch_seed())
        utt.verify_grad(oplike, [x_val], rng=rng)

    def test_infer_shape(self):
        admat = matrix()
        alvec = lvector()
        rng = np.random.default_rng(utt.fetch_seed())
        admat_val = rng.random((3, 2)).astype(config.floatX)
        alvec_val = [0, 1, 0]
        self._compile_and_check(
            [admat, alvec],
            [CrossentropyCategorical1Hot()(admat, alvec)],
            [admat_val, alvec_val],
            CrossentropyCategorical1Hot,
        )

    def test_softmax_rewrites(self):
        x = matrix("x")
        one_of_n = lvector("one_of_n")
        op = crossentropy_categorical_1hot
        # xe = op(x, one_of_n)

        fgraph = FunctionGraph([x, one_of_n], [op(softmax_legacy(x), one_of_n)])
        assert fgraph.outputs[0].owner.op == op

        optdb.query(OPT_FAST_RUN).rewrite(fgraph)
        assert fgraph.outputs[0].owner.op == crossentropy_softmax_argmax_1hot_with_bias

    def test_softmax_rewrites_w_bias(self):
        x = matrix("x")
        b = vector("b")
        one_of_n = lvector("one_of_n")
        op = crossentropy_categorical_1hot

        fgraph = FunctionGraph([x, b, one_of_n], [op(softmax_legacy(x + b), one_of_n)])
        assert fgraph.outputs[0].owner.op == op

        optdb.query(OPT_FAST_RUN).rewrite(fgraph)

        assert len(fgraph.toposort()) == 1
        assert fgraph.outputs[0].owner.op == crossentropy_softmax_argmax_1hot_with_bias

    def test_softmax_rewrites_w_bias2(self):
        x = matrix("x")
        b = vector("b")
        c = vector("c")
        one_of_n = lvector("one_of_n")
        op = crossentropy_categorical_1hot

        fgraph = FunctionGraph(
            [x, b, c, one_of_n], [op(softmax_legacy(add(x, b, c)), one_of_n)]
        )
        assert fgraph.outputs[0].owner.op == op

        optdb.query(OPT_FAST_RUN).rewrite(fgraph)

        assert len(fgraph.toposort()) == 2
        assert fgraph.outputs[0].owner.op == crossentropy_softmax_argmax_1hot_with_bias

    def test_softmax_grad_rewrites(self):
        x = matrix("x")
        one_of_n = lvector("one_of_n")
        op = crossentropy_categorical_1hot
        xe = op(softmax_legacy(x), one_of_n)
        sum_xe = at_sum(xe)
        g_x = grad(sum_xe, x)
        fgraph = FunctionGraph([x, one_of_n], [g_x])
        assert check_stack_trace(
            fgraph,
            ops_to_check=[crossentropy_softmax_1hot_with_bias_dx, softmax_legacy],
        )

        optdb.query(OPT_FAST_RUN).rewrite(fgraph)

        ops = {node.op for node in fgraph.toposort()}
        assert crossentropy_softmax_argmax_1hot_with_bias not in ops
        assert crossentropy_softmax_1hot_with_bias_dx in ops
        assert softmax_legacy in ops
        assert softmax_grad_legacy not in ops

    def test_get_rid_of_advanced_indexing_version_of_xent(self):
        x = matrix("x")
        b = vector("b")
        y = lvector("y")

        # Basic case
        expressions = [
            at_sum(-log(softmax(x)[at.arange(y.shape[0]), y])),
            -at_sum(log(softmax(x)[at.arange(y.shape[0]), y])),
            -at_sum(log(softmax(x))[at.arange(y.shape[0]), y]),
            at_sum(-log(softmax(x))[at.arange(y.shape[0]), y]),
        ]
        for expr in expressions:
            fgraph = FunctionGraph([x, y], [expr])
            optdb.query(OPT_FAST_RUN).rewrite(fgraph)

            ops = [node.op for node in fgraph.toposort()]
            assert len(ops) == 4
            assert crossentropy_softmax_argmax_1hot_with_bias in ops
            assert not [1 for o in ops if isinstance(o, AdvancedSubtensor)]

            # Also verify the gradient wrt x
            fgraph = FunctionGraph([x, y], [grad(expr, x)])
            optdb.query(OPT_FAST_RUN).rewrite(fgraph)

            ops = [node.op for node in fgraph.toposort()]
            assert len(ops) == 2
            assert crossentropy_softmax_1hot_with_bias_dx in ops
            assert softmax_legacy in ops
            assert softmax_grad_legacy not in ops

        # Test that a biased softmax is rewritten correctly
        bias_expressions = [
            at_sum(-log(softmax(x + b)[at.arange(y.shape[0]), y])),
            -at_sum(log(softmax(b + x)[at.arange(y.shape[0]), y])),
            -at_sum(log(softmax(x + b))[at.arange(y.shape[0]), y]),
            at_sum(-log(softmax(b + x))[at.arange(y.shape[0]), y]),
        ]

        for expr in bias_expressions:
            fgraph = FunctionGraph([x, b, y], [expr, x])
            optdb.query(OPT_FAST_RUN).rewrite(fgraph)

            ops = [node.op for node in fgraph.toposort()]
            assert len(ops) == 2  # [big_op, sum]
            assert crossentropy_softmax_argmax_1hot_with_bias in ops

            fgraph = FunctionGraph([x, b, y], [grad(expr, x)])
            optdb.query(OPT_FAST_RUN).rewrite(fgraph)

            ops = [node.op for node in fgraph.toposort()]
            assert len(ops) == 2
            assert crossentropy_softmax_1hot_with_bias_dx in ops
            assert softmax_with_bias in ops
            assert softmax_grad_legacy not in ops

        # Test that using "mean" instead of sum works, too
        mean_expressions = [
            mean(-log(softmax(x)[at.arange(y.shape[0]), y])),
            -mean(log(softmax(x)[at.arange(y.shape[0]), y])),
            -mean(log(softmax(x))[at.arange(y.shape[0]), y]),
            mean(-log(softmax(x))[at.arange(y.shape[0]), y]),
        ]

        for expr in mean_expressions:
            fgraph = FunctionGraph([x, y], [expr])
            optdb.query(OPT_FAST_RUN).rewrite(fgraph)

            ops = [node.op for node in fgraph.toposort()]
            assert len(ops) == 6
            assert crossentropy_softmax_argmax_1hot_with_bias in ops
            assert not [1 for o in ops if isinstance(o, AdvancedSubtensor)]

            fgraph = FunctionGraph([x, y], [grad(expr, x)])
            optdb.query(OPT_FAST_RUN).rewrite(fgraph)

            ops = [node.op for node in fgraph.toposort()]
            assert len(ops) == 5
            # there's an extra dimshuffle in there
            # but I can't think of a good rule to get rid of it
            assert crossentropy_softmax_1hot_with_bias_dx in ops
            assert softmax_legacy in ops
            assert softmax_grad_legacy not in ops

        mean_bias_expressions = [
            mean(-log(softmax(x + b)[at.arange(y.shape[0]), y])),
            -mean(log(softmax(b + x)[at.arange(y.shape[0]), y])),
            -mean(log(softmax(x + b))[at.arange(y.shape[0]), y]),
            mean(-log(softmax(b + x))[at.arange(y.shape[0]), y]),
        ]

        for expr in mean_bias_expressions:
            fgraph = FunctionGraph([x, b, y], [expr])
            optdb.query(OPT_FAST_RUN).rewrite(fgraph)

            ops = [node.op for node in fgraph.toposort()]
            assert len(ops) == 4
            assert crossentropy_softmax_argmax_1hot_with_bias in ops
            assert not [1 for o in ops if isinstance(o, AdvancedSubtensor)]

            fgraph = FunctionGraph([x, b, y], [grad(expr, x)])
            optdb.query(OPT_FAST_RUN).rewrite(fgraph)

            ops = [node.op for node in fgraph.toposort()]
            assert len(ops) == 5
            assert crossentropy_softmax_1hot_with_bias_dx in ops
            assert softmax_with_bias in ops
            assert softmax_grad_legacy not in ops

    def test_xent_thing_int32(self):
        x = matrix("x")
        y = lvector("y")
        yi = at.cast(y, "int32")
        expressions = [
            at_sum(-log(softmax(x)[at.arange(yi.shape[0]), yi])),
            -at_sum(log(softmax(x)[at.arange(yi.shape[0]), yi])),
            -at_sum(log(softmax(x))[at.arange(yi.shape[0]), yi]),
            at_sum(-log(softmax(x))[at.arange(yi.shape[0]), yi]),
        ]

        for expr in expressions:
            fgraph = FunctionGraph([x, y], [expr])
            optdb.query(OPT_FAST_RUN).rewrite(fgraph)

            ops = [node.op for node in fgraph.toposort()]
            assert len(ops) == 5
            assert crossentropy_softmax_argmax_1hot_with_bias in ops
            assert not [1 for o in ops if isinstance(o, AdvancedSubtensor)]

            # Also verify the gradient wrt x
            fgraph = FunctionGraph([x, y], [grad(expr, x)])
            optdb.query(OPT_FAST_RUN).rewrite(fgraph)

            ops = [node.op for node in fgraph.toposort()]
            assert len(ops) == 3
            assert crossentropy_softmax_1hot_with_bias_dx in ops
            assert softmax_legacy in ops
            assert softmax_grad_legacy not in ops

    def test_crossentropy_softmax_1hot_with_bias_dxcale_cost(self):
        x = matrix("x")
        y = lvector("y")
        a = scalar("a")

        def validate_grad_graph(func):
            # The graph of the gradient should not have softmaxgrad anymore
            has_cx1hotdx = False
            has_softmax = False
            has_softmaxdx = False
            for node in func.maker.fgraph.toposort():
                if node.op == crossentropy_softmax_1hot_with_bias_dx:
                    has_cx1hotdx = True
                if node.op == softmax_legacy:
                    has_softmax = True
                if node.op == softmax_grad_legacy:
                    has_softmaxdx = True

            assert has_cx1hotdx
            assert has_softmax
            assert not has_softmaxdx

        # Cases to test
        expressions = [
            a * at_sum(-log(softmax(x)[at.arange(y.shape[0]), y])),
            -a * at_sum(log(softmax(x)[at.arange(y.shape[0]), y])),
            a * (-at_sum(log(softmax(x)[at.arange(y.shape[0]), y]))),
            a * at_sum(log(softmax(x)[at.arange(y.shape[0]), y])),
            a * at_sum(-log(softmax(x))[at.arange(y.shape[0]), y]),
            -a * at_sum(log(softmax(x))[at.arange(y.shape[0]), y]),
            a * (-at_sum(log(softmax(x))[at.arange(y.shape[0]), y])),
            a * at_sum(log(softmax(x))[at.arange(y.shape[0]), y]),
            a * mean(-log(softmax(x)[at.arange(y.shape[0]), y])),
            -a * mean(log(softmax(x)[at.arange(y.shape[0]), y])),
            a * (-mean(log(softmax(x)[at.arange(y.shape[0]), y]))),
            a * mean(log(softmax(x)[at.arange(y.shape[0]), y])),
            a * mean(-log(softmax(x))[at.arange(y.shape[0]), y]),
            -a * mean(log(softmax(x))[at.arange(y.shape[0]), y]),
            a * (-mean(log(softmax(x))[at.arange(y.shape[0]), y])),
            a * mean(log(softmax(x))[at.arange(y.shape[0]), y]),
        ]

        for expr in expressions:
            fgraph = FunctionGraph([x, y, a], [expr])
            optdb.query(OPT_FAST_RUN).rewrite(fgraph)

            assert 5 <= len(fgraph.toposort()) <= 10

            ops = {node.op for node in fgraph.toposort()}
            assert crossentropy_softmax_argmax_1hot_with_bias in ops
            assert softmax_legacy not in ops

            # Verify the gradient wrt x
            fgraph = FunctionGraph([x, y, a], [grad(expr, x)])
            optdb.query(OPT_FAST_RUN).rewrite(fgraph)

            assert 3 <= len(fgraph.toposort()) <= 6

            ops = {node.op for node in fgraph.toposort()}
            assert crossentropy_softmax_1hot_with_bias_dx in ops
            assert softmax_legacy in ops
            assert softmax_grad_legacy not in ops

            # Verify the gradient when providing output gradient
            fgraph = FunctionGraph(
                [x, y, a], [grad(expr, x, known_grads={expr: a * x.sum()})]
            )
            optdb.query(OPT_FAST_RUN).rewrite(fgraph)

            assert 6 <= len(fgraph.toposort()) <= 8

            ops = {node.op for node in fgraph.toposort()}
            assert crossentropy_softmax_1hot_with_bias_dx in ops
            assert softmax_legacy in ops
            assert softmax_grad_legacy not in ops


def test_argmax_pushdown():
    x = matrix()
    for sm in [softmax_graph, softmax_legacy]:
        # test that the max_and_argmax is pushed down if the max is not used
        out = max_and_argmax(sm(exp(tanh(sigmoid(x)))), axis=-1)[1]
        fgraph = FunctionGraph([x], [out])
        optdb.query(OPT_FAST_RUN).rewrite(fgraph)

        # print 'AFTER'
        # for node in fgraph.toposort():
        # print node.op
        assert len(fgraph.toposort()) == 1
        assert isinstance(fgraph.toposort()[0].op, Argmax)
        assert check_stack_trace(fgraph, ops_to_check=Argmax)
        x = matrix()
        # test that the max_and_argmax is not pushed down if the max is used
        out = max_and_argmax(sm(exp(tanh(sigmoid(x)))), axis=-1)[0]
        fgraph = FunctionGraph([x], [out])

        assert hasattr(fgraph.outputs[0].tag, "trace")

        optdb.query(OPT_FAST_RUN).rewrite(fgraph)

        # print 'AFTER'
        # for node in fgraph.toposort():
        # print node.op
        assert len(fgraph.toposort()) == 3
        assert isinstance(fgraph.toposort()[0].op, Elemwise)
        assert isinstance(fgraph.toposort()[1].op, Softmax)
        assert isinstance(fgraph.toposort()[2].op, CAReduce)
        assert isinstance(
            fgraph.toposort()[2].op.scalar_op, aesara.scalar.ScalarMaximum
        )


def test_argmax_pushdown_bias():
    x = matrix()
    b = vector()

    out = argmax(softmax_with_bias(x, b), axis=-1)
    fgraph = FunctionGraph([x, b], [out])

    optdb.query(OPT_FAST_RUN).rewrite(fgraph)

    types_to_check = (DimShuffle, Elemwise, Argmax)
    assert len(fgraph.toposort()) == 3

    for i, type in enumerate(types_to_check):
        assert isinstance(fgraph.toposort()[i].op, type)
    assert check_stack_trace(fgraph, ops_to_check=types_to_check)

    x = matrix()
    b = vector()
    out = max_and_argmax(softmax_with_bias(x, b), axis=-1)[0]
    fgraph = FunctionGraph([x, b], [out])

    optdb.query(OPT_FAST_RUN).rewrite(fgraph)

    assert len(fgraph.toposort()) == 2
    assert isinstance(fgraph.toposort()[0].op, SoftmaxWithBias)
    assert isinstance(fgraph.toposort()[1].op, CAReduce)
    assert isinstance(fgraph.toposort()[1].op.scalar_op, aesara.scalar.ScalarMaximum)
    assert check_stack_trace(fgraph, ops_to_check=(SoftmaxWithBias, CAReduce))


def test_asymptotic_32():
    """Test that our functions behave sensibly when huge values are present."""

    # TODO: consider adding the rewrite of crossentropy into the current
    # mode for the purpose of running this test

    for dtype in "float32", "float64":
        if dtype == "float32":
            x = fmatrix()
            x2 = fvector()
        else:
            x = dmatrix()
            x2 = dvector()
        y = lvector()

        c = categorical_crossentropy(softmax(x + x2), y)
        f = aesara.function([x, y, x2], [c.sum(), grad(c.sum(), x)], mode="FAST_RUN")

        xval = np.zeros((5, 5), dtype=dtype).astype(dtype)
        x2val = np.zeros(5, dtype=xval.dtype).astype(dtype)
        for i in range(100):
            cval, gxval = f(xval, np.arange(5), x2val)
            xval -= 100.3 * gxval

        assert cval == 0  # no problem going to zero error

        # what about when x gets really big?

        xval = np.zeros((5, 5), dtype=dtype)
        x2val = np.zeros(5, dtype=xval.dtype)
        for i in range(100):
            cval, gxval = f(xval, np.arange(5), x2val)
            xval += 100000.3 * gxval

        assert cval > 61750000
        assert gxval[0, 0] == -1.0
        assert gxval[0, 1] == 0.25


class TestSoftmaxRewrite:
    """
    Test that expressions of softmax in terms of exponentiated things
    divided by row sums are replaced by softmax expressions.

    `Softmax_grad` isn't that interesting as an Op, but it has the signature
    we look for when trying to insert `CrossEntropySoftmax` grad.  So, for
    now, we add `softmax_grad` to graphs. In the future, we may modify the
    `CrossEntropySoftmax` grad to look for the more basic pattern.

    """

    def setup_method(self):
        self.mode = aesara.compile.mode.get_default_mode()
        self.mode = self.mode.including("canonicalize")

    @pytest.mark.parametrize("axis", [None, 0, 1, -1, (0, 1)])
    def test_basic(self, axis):
        c = matrix()
        if axis is None:
            p_y = exp(c) / exp(c).sum(axis=axis).dimshuffle("x", "x")
        elif axis == 0:
            p_y = exp(c) / exp(c).sum(axis=axis).dimshuffle("x", 0)
        elif axis == (0, 1):
            p_y = exp(c) / exp(c).sum(axis=axis).dimshuffle("x", "x")
        else:
            p_y = exp(c) / exp(c).sum(axis=axis).dimshuffle(0, "x")

        # test that function contains softmax and no div.
        f = aesara.function([c], p_y, mode=self.mode)

        assert check_stack_trace(f, ops_to_check=Softmax)

        f_ops = [n.op for n in f.maker.fgraph.toposort()]

        assert len(f_ops) == 1
        assert isinstance(f_ops[0], Softmax)

        rng = np.random.default_rng(utt.fetch_seed())
        c_val = rng.random((3, 4)).astype(config.floatX)
        assert np.allclose(f(c_val), sp.softmax(c_val, axis=axis))

    @pytest.mark.parametrize("axis", [None, 0, 1, 2, -1, -2, -3, (0, 1, 2)])
    def test_basic_keepdims(self, axis):
        c = tensor3()
        p_y = exp(c) / exp(c).sum(axis=axis, keepdims=True)

        # test that function contains softmax and no div.
        f = aesara.function([c], p_y, mode=self.mode)

        assert check_stack_trace(f, ops_to_check=Softmax)

        f_ops = [n.op for n in f.maker.fgraph.toposort()]

        assert len(f_ops) == 1
        assert isinstance(f_ops[0], Softmax)

        rng = np.random.default_rng(utt.fetch_seed())
        c_val = rng.random((3, 4, 5)).astype(config.floatX)
        assert np.allclose(f(c_val), sp.softmax(c_val, axis=axis))

    @pytest.mark.skip(reason="Rewrite not enabled for the moment")
    def test_grad(self):
        c = matrix()
        p_y = exp(c) / exp(c).sum(axis=1).dimshuffle(0, "x")

        # test that function contains softmax and softmaxgrad
        w = matrix()

        g = aesara.function([c, w], grad((p_y * w).sum(), c), mode=self.mode)

        g_ops = [n.op for n in g.maker.fgraph.toposort()]

        assert len(g_ops) == 2, g_ops
        assert isinstance(g_ops[0], Softmax)
        assert isinstance(g_ops[1], SoftmaxGrad)

        rng = np.random.default_rng(utt.fetch_seed())
        g(rng.random((3, 4)), rng.uniform(0.5, 1, (3, 4)))

    def test_transpose_basic(self):
        # this should be a transposed softmax
        c = matrix()
        p_y = exp(c) / exp(c).sum(axis=0)

        # test that function contains softmax and no div.
        f = aesara.function([c], p_y, mode=self.mode)
        f_ops = [n.op for n in f.maker.fgraph.toposort()]
        assert len(f_ops) == 1
        assert isinstance(f_ops[0], Softmax)

    @pytest.mark.skip(reason="Rewrite not enabled for the moment")
    def test_transpose_grad(self):
        # this should be a transposed softmax
        c = matrix()
        p_y = exp(c) / exp(c).sum(axis=0)

        # test that function contains softmax and no div.
        g = aesara.function([c], grad(p_y.sum(), c), mode=self.mode)
        g_ops = [n.op for n in g.maker.fgraph.toposort()]
        assert len(g_ops) == 2
        assert isinstance(g_ops[0], Softmax)
        assert isinstance(g_ops[1], SoftmaxGrad)

    def test_1d_basic(self):
        c = vector()
        p_y = exp(c) / exp(c).sum()

        # test that function contains softmax and no div.
        f = aesara.function([c], p_y, mode=self.mode)
        f_ops = [n.op for n in f.maker.fgraph.toposort()]
        assert len(f_ops) == 1
        assert isinstance(f_ops[0], Softmax)

    @pytest.mark.skip(reason="Rewrite not enabled for the moment")
    def test_1D_grad(self):
        c = vector()
        p_y = exp(c) / exp(c).sum()

        # test that function contains softmax and no div.
        g = aesara.function([c], grad(p_y.sum(), c), mode=self.mode)
        g_ops = [n.op for n in g.maker.fgraph.toposort()]
        assert len(g_ops) == 2
        assert isinstance(g_ops[0], Softmax)
        assert isinstance(g_ops[1], SoftmaxGrad)

    @pytest.mark.parametrize(
        "f",
        [
            lambda c: exp(c) / exp(c).sum(axis=0).dimshuffle(0, 1, "x"),
            lambda c: exp(c) / exp(c).sum(axis=0).dimshuffle("x", 0, 1, "x"),
            lambda c: exp(c) / exp(c).sum(axis=0).dimshuffle("x", 1, 0),
            lambda c: exp(c) / exp(c).sum(axis=(0, 1), keepdims=True),
        ],
    )
    def test_invalid_softmax_expressions(self, f):
        # Test that graphs are not rewritten into a softmax when a dimshuffle
        # swaps or adds extra dimensions, or when more than one but not all axis
        # are summed over (which is not allowed by the Softmax Op but otherwise
        # valid)
        c = tensor3("c")
        out = f(c)
        f = aesara.function([c], out, mode=self.mode)

        f_ops = [n.op for n in f.maker.fgraph.toposort()]
        assert len(f_ops) > 1
        assert not any(isinstance(op, Softmax) for op in f_ops)


def test_softmax_graph():
    rng = np.random.default_rng(utt.fetch_seed())
    x = aesara.shared(rng.normal(size=(3, 4)))

    def f(inputs):
        y = softmax_graph(x)
        return aesara.grad(None, x, known_grads={y: inputs})

    utt.verify_grad(f, [rng.random((3, 4))])


def test_grad_softmax_grad():
    rng = np.random.default_rng(utt.fetch_seed())
    x = aesara.shared(rng.normal(size=(3, 4)))

    def f(inputs):
        y = softmax_legacy(x)
        return aesara.grad(None, x, known_grads={y: inputs})

    utt.verify_grad(f, [rng.random((3, 4))])


def test_relu():
    x = matrix("x")
    rng = np.random.default_rng(utt.fetch_seed())
    X = rng.standard_normal((20, 30)).astype(config.floatX)

    # Test the base case, without custom alpha value
    y = relu(x).eval({x: X})
    assert np.allclose(y, np.maximum(X, 0))

    # Test for different constant alpha values (also outside of [0, 1])
    for alpha in 0, 0.3, 1, 2, -0.3, -1, -2:
        y = relu(x, alpha).eval({x: X})
        assert np.allclose(y, np.where(X > 0, X, alpha * X))

    # Test for variable alpha (scalar, vector and matrix)
    for alpha in scalar(), vector(), matrix():
        # Create value for alpha (correct ndim and broadcastable against X)
        A = np.array(
            rng.standard_normal(X.shape[::-1][: alpha.ndim][::-1]), dtype=config.floatX
        )
        y = relu(x, alpha).eval({x: X, alpha: A})
        assert np.allclose(y, np.where(X > 0, X, A * X), rtol=3e-5)

        # Test that an alpha of type `ndarray` doesn't generate an upcast
        x = matrix("x", dtype="float32")
        X = rng.standard_normal((20, 30)).astype("float32")
        alpha = np.asarray(0.123, dtype="float32")

        y = relu(x, alpha).eval({x: X})
        assert np.allclose(y, np.where(X > 0, X, alpha * X))
        assert y.dtype == "float32"


def test_h_softmax():
    """Tests the output dimensions of the `h_softmax` when a target is provided or not."""

    input_size = 4
    batch_size = 2
    h_softmax_level1_size = 5
    h_softmax_level2_size = 3
    output_size = h_softmax_level1_size * h_softmax_level2_size

    rng = np.random.default_rng(utt.fetch_seed())

    # First level of h_softmax
    W1 = np.asarray(
        rng.normal(size=(input_size, h_softmax_level1_size)), dtype=config.floatX
    )
    W1 = aesara.shared(W1)
    b1 = aesara.shared(
        np.asarray(np.zeros((h_softmax_level1_size,)), dtype=config.floatX)
    )

    # Second level of h_softmax
    W2 = np.asarray(
        rng.normal(size=(h_softmax_level1_size, input_size, h_softmax_level2_size)),
        dtype=config.floatX,
    )
    W2 = aesara.shared(W2)
    b2 = aesara.shared(
        np.asarray(
            np.zeros((h_softmax_level1_size, h_softmax_level2_size)),
            dtype=config.floatX,
        )
    )

    x = matrix("x")
    y = ivector("y")

    # This only computes the output corresponding to the target
    y_hat_tg = h_softmax(
        x,
        batch_size,
        output_size,
        h_softmax_level1_size,
        h_softmax_level2_size,
        W1,
        b1,
        W2,
        b2,
        y,
    )

    # This computes all the outputs
    y_hat_all = h_softmax(
        x,
        batch_size,
        output_size,
        h_softmax_level1_size,
        h_softmax_level2_size,
        W1,
        b1,
        W2,
        b2,
    )

    fun_output_tg = aesara.function([x, y], y_hat_tg)
    fun_output = aesara.function([x], y_hat_all)

    x_mat = rng.normal(size=(batch_size, input_size)).astype(config.floatX)
    y_mat = rng.integers(0, output_size, batch_size).astype("int32")
    tg_output = fun_output_tg(x_mat, y_mat)
    all_outputs = fun_output(x_mat)

    assert tg_output.shape == (batch_size,)
    assert all_outputs.shape == (batch_size, output_size)

    # Verifies that the outputs computed by fun_output_tg are the same as those
    # computed by fun_output.
    utt.assert_allclose(all_outputs[np.arange(0, batch_size), y_mat], tg_output)


def test_elu():
    x = matrix("x")
    rng = np.random.default_rng(utt.fetch_seed())
    X = rng.standard_normal((20, 30)).astype(config.floatX)

    # test the base case, without custom alpha value
    y = elu(x).eval({x: X})
    utt.assert_allclose(y, np.where(X > 0, X, np.exp(X) - 1))

    # test for different constant alpha values
    for alpha in 1.5, 2, -1, -1.5, -2:
        y = elu(x, alpha).eval({x: X})
        utt.assert_allclose(y, np.where(X > 0, X, alpha * (np.exp(X) - 1)))


def test_selu():
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946

    x = matrix("x")
    rng = np.random.default_rng(utt.fetch_seed())
    X = rng.standard_normal((20, 30)).astype(config.floatX)

    y = selu(x).eval({x: X})
    utt.assert_allclose(y, np.where(X > 0, scale * X, scale * alpha * (np.exp(X) - 1)))


def test_binary_crossentropy_reshape():
    # Reported as https://github.com/Theano/Theano/issues/4086
    a = tensor4("a")
    for c in (
        binary_crossentropy(sigmoid(a.reshape((-1, 1))), 1).sum(),
        binary_crossentropy(sigmoid(a).reshape((-1, 1)), 1).sum(),
    ):
        ga = aesara.grad(c, a)
        # This only works when "specialize" options are included
        mode = aesara.compile.get_default_mode().including("fast_run")
        fga = aesara.function([a], ga, mode=mode)
        utt.assert_allclose(
            fga(np.array([[[[30.0]]]], dtype=config.floatX)),
            np.zeros((1, 1, 1, 1), dtype=config.floatX),
        )


TestSoftsign = makeBroadcastTester(
    op=softsign,
    expected=upcast_int8_nfunc(
        lambda inputs: check_floatX(inputs, inputs / (1.0 + np.fabs(inputs)))
    ),
    good=_good_broadcast_unary_normal_float_no_complex,
    name="SoftsignTester",
)


class TestSigmoidBinaryCrossentropy:
    def test_matches_binary_crossentropy(self):
        # Test sigmoid_binary_crossentropy(p, t) ==
        #      binary_crossentropy(sigmoid(p), t).

        pred, target = inputs = vectors("pt")

        reference_val = binary_crossentropy(sigmoid(pred), target)
        f_reference = aesara.function(inputs, reference_val)

        test_val = sigmoid_binary_crossentropy(pred, target)
        f_test = aesara.function(inputs, test_val)

        rng = np.random.default_rng(utt.fetch_seed())
        pred, target = rng.standard_normal((2, 50)).astype(config.floatX)
        test_inputs = [pred, 1 / (1 + np.exp(-target))]

        utt.assert_allclose(f_reference(*test_inputs), f_test(*test_inputs))

    def test_grad(self):
        rng = np.random.default_rng(utt.fetch_seed())
        pred, target = rng.standard_normal((2, 50)).astype(config.floatX)
        test_inputs = [pred, 1 / (1 + np.exp(-target))]

        utt.verify_grad(sigmoid_binary_crossentropy, test_inputs)


def test_confusion_matrix():
    # Defining numpy implementation of confusion matrix
    def numpy_conf_mat(actual, pred):
        order = np.union1d(actual, pred)
        colA = np.matrix(actual).T
        colP = np.matrix(pred).T
        oneHotA = colA.__eq__(order).astype("int64")
        oneHotP = colP.__eq__(order).astype("int64")
        conf_mat = np.dot(oneHotA.T, oneHotP)
        conf_mat = np.asarray(conf_mat)
        return [conf_mat, order]

    x = vector()
    y = vector()
    f = aesara.function([x, y], confusion_matrix(x, y))
    list_inputs = [
        [[0, 1, 2, 1, 0], [0, 0, 2, 1, 2]],
        [[2, 0, 2, 2, 0, 1], [0, 0, 2, 2, 0, 2]],
    ]

    for case in list_inputs:
        a = np.asarray(case[0])
        b = np.asarray(case[1])
        out_exp = numpy_conf_mat(a, b)
        outs = f(case[0], case[1])
        for exp_res, out in zip(out_exp, outs):
            utt.assert_allclose(exp_res, out)
