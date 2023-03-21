from collections import OrderedDict

import numpy as np
import pytest

import aesara
import aesara.tensor as at
from aesara.configdefaults import config
from aesara.tensor.math import sum as at_sum
from aesara.tensor.nnet import batchnorm
from aesara.tensor.shape import specify_broadcastable
from aesara.tensor.type import (
    TensorType,
    matrix,
    scalar,
    tensor3,
    tensor4,
    tensor5,
    vector,
)
from tests import unittest_tools as utt


def test_BNComposite():
    with config.change_flags(compute_test_value="raise"):

        def bn_ref(x, G, B, M, V):
            n = (x - M) / V
            return n * G + B

        rng = np.random.default_rng(1234)
        X = 1 + rng.random([10, 20]).astype("float32")
        B = 1 + rng.random([20]).astype("float32")
        G = 1 + rng.random([20]).astype("float32")
        M = 1 + rng.random([20]).astype("float32")
        V = 1 + rng.random([20]).astype("float32")

        x = matrix("x")
        b = vector("b")
        g = vector("g")
        m = vector("m")
        v = vector("v")

        x.tag.test_value = rng.random((2, 2)).astype(aesara.config.floatX)
        b.tag.test_value = rng.random(2).astype(aesara.config.floatX)
        g.tag.test_value = rng.random(2).astype(aesara.config.floatX)
        m.tag.test_value = rng.random(2).astype(aesara.config.floatX)
        v.tag.test_value = rng.random(2).astype(aesara.config.floatX)

        bn_ref_op = bn_ref(x, g, b, m, v)
        f_ref = aesara.function([x, b, g, m, v], [bn_ref_op])
        res_ref = f_ref(X, G, B, M, V)
        for mode in ["low_mem", "high_mem"]:
            bn_op = batchnorm.batch_normalization(x, g, b, m, v, mode=mode)
            f = aesara.function([x, b, g, m, v], [bn_op])
            res = f(X, G, B, M, V)
            utt.assert_allclose(res_ref, res)


def test_batch_normalization():
    def bn_ref(x, G, B, M, V):
        n = (x - M) / V
        return n * G + B

    rng = np.random.default_rng(1234)
    X = 1 + rng.random([10, 20]).astype("float32")
    B = 1 + rng.random([20]).astype("float32")
    G = 1 + rng.random([20]).astype("float32")
    M = 1 + rng.random([20]).astype("float32")
    V = 1 + rng.random([20]).astype("float32")

    x = matrix("x")
    b = vector("b")
    g = vector("g")
    m = vector("m")
    v = vector("v")

    bn_ref_op = bn_ref(x, g, b, m, v)
    f_ref = aesara.function([x, g, b, m, v], [bn_ref_op])
    res_ref = f_ref(X, G, B, M, V)
    for mode in ["low_mem", "high_mem"]:
        bn_op = batchnorm.batch_normalization(x, g, b, m, v, mode=mode)
        f = aesara.function([x, g, b, m, v], [bn_op])
        res = f(X, G, B, M, V)
        utt.assert_allclose(res_ref, res)

        def bn_f(inputs, gamma, beta, mean, std):
            return batchnorm.batch_normalization(
                inputs, gamma, beta, mean, std, mode=mode
            )

        utt.verify_grad(bn_f, [X, G, B, M, V])

    bn_ref_op = bn_ref(
        x, g, b, x.mean(axis=0, keepdims=True), x.std(axis=0, keepdims=True)
    )
    f_ref = aesara.function([x, b, g], [bn_ref_op])
    res_ref = f_ref(X, G, B)
    for mode in ["low_mem", "high_mem"]:
        bn_op = batchnorm.batch_normalization(
            x,
            g,
            b,
            x.mean(axis=0, keepdims=True),
            x.std(axis=0, keepdims=True),
            mode=mode,
        )
        f = aesara.function([x, b, g], [bn_op])
        res = f(X, G, B)
        utt.assert_allclose(res_ref, res)

        def bn_f(inputs, gamma, beta, mean, std):
            return batchnorm.batch_normalization(
                inputs, gamma, beta, mean, std, mode=mode
            )

        utt.verify_grad(
            bn_f, [X, G, B, X.mean(axis=0)[np.newaxis], X.std(axis=0)[np.newaxis]]
        )


def test_bn_feature_maps():
    def bn_ref(x, G, B, M, V):
        n = (x - M) / V
        return n * G + B

    rng = np.random.default_rng(1234)
    X = 1 + rng.random([2, 3, 4, 4]).astype("float32")
    B = 1 + rng.random([3]).astype("float32")
    G = 1 + rng.random([3]).astype("float32")
    M = 1 + rng.random([3]).astype("float32")
    V = 1 + rng.random([3]).astype("float32")

    x = tensor4("x")
    b = vector("b")
    g = vector("g")
    m = vector("m")
    v = vector("v")

    bn_ref_op = bn_ref(
        x,
        g.dimshuffle("x", 0, "x", "x"),
        b.dimshuffle("x", 0, "x", "x"),
        m.dimshuffle("x", 0, "x", "x"),
        v.dimshuffle("x", 0, "x", "x"),
    )
    f_ref = aesara.function([x, b, g, m, v], [bn_ref_op])
    res_ref = f_ref(X, G, B, M, V)

    for mode in ["low_mem", "high_mem"]:
        bn_op = batchnorm.batch_normalization(
            x,
            g.dimshuffle("x", 0, "x", "x"),
            b.dimshuffle("x", 0, "x", "x"),
            m.dimshuffle("x", 0, "x", "x"),
            v.dimshuffle("x", 0, "x", "x"),
            mode=mode,
        )
        f = aesara.function([x, b, g, m, v], [bn_op])
        res = f(X, G, B, M, V)
        utt.assert_allclose(res_ref, res)

        def conv_bn(inputs, gamma, beta, mean, std):
            return batchnorm.batch_normalization(
                inputs,
                gamma.dimshuffle("x", 0, "x", "x"),
                beta.dimshuffle("x", 0, "x", "x"),
                mean.dimshuffle("x", 0, "x", "x"),
                std.dimshuffle("x", 0, "x", "x"),
                mode=mode,
            )

        utt.verify_grad(conv_bn, [X, G, B, M, V])


@pytest.mark.slow
def test_batch_normalization_train():
    for axes in ("per-activation", "spatial", (1, 2, 3, 4)):
        for vartype in (tensor5, tensor3, vector):
            x, scale, bias, running_mean, running_var = (
                vartype(n)
                for n in ("x", "scale", "bias", "running_mean", "running_var")
            )
            ndim = x.ndim
            eps = 5e-3  # some non-standard value to test if it's used
            running_average_factor = 0.3

            # remove non-existing axes
            if isinstance(axes, tuple):
                axes = tuple(i for i in axes if i < ndim)
            if len(axes) == 0:
                continue

            # forward pass
            (
                out,
                x_mean,
                x_invstd,
                out_running_mean,
                out_running_var,
            ) = batchnorm.batch_normalization_train(
                x,
                scale,
                bias,
                axes,
                eps,
                running_average_factor,
                running_mean,
                running_var,
            )
            # reference forward pass
            if axes == "per-activation":
                axes2 = (0,)
            elif axes == "spatial":
                axes2 = (0,) + tuple(range(2, ndim))
            else:
                axes2 = axes
            x_mean2 = x.mean(axis=axes2, keepdims=True)
            x_var2 = x.var(axis=axes2, keepdims=True)
            x_invstd2 = at.reciprocal(at.sqrt(x_var2 + eps))
            scale2 = specify_broadcastable(scale, *axes2)
            bias2 = specify_broadcastable(bias, *axes2)
            out2 = (x - x_mean2) * (scale2 * x_invstd2) + bias2
            m = at.cast(at.prod(x.shape) / at.prod(scale.shape), aesara.config.floatX)
            out_running_mean2 = (
                running_mean * (1 - running_average_factor)
                + x_mean2 * running_average_factor
            )
            out_running_var2 = (
                running_var * (1 - running_average_factor)
                + (m / (m - 1)) * x_var2 * running_average_factor
            )
            # backward pass
            dy = vartype("dy")
            grads = at.grad(None, wrt=[x, scale, bias], known_grads={out: dy})
            # reference backward pass
            grads2 = at.grad(None, wrt=[x, scale, bias], known_grads={out2: dy})
            # second-order backward pass
            dx = vartype("dinputs")
            dscale = vartype("dscale")
            dbias = vartype("dbias")
            grad_grads = at.grad(
                None,
                wrt=[x, dy, scale],
                known_grads=OrderedDict(
                    {grads[0]: dx, grads[1]: dscale, grads[2]: dbias}
                ),
                consider_constant=[
                    x,
                    dy,
                    scale,
                    bias,
                    x_mean,
                    x_invstd,
                    running_mean,
                    running_var,
                ],
                return_disconnected="zero",
            )
            # reference second-order backward pass
            grad_grads2 = at.grad(
                None,
                wrt=[x, dy, scale],
                known_grads=OrderedDict(
                    {grads2[0]: dx, grads2[1]: dscale, grads2[2]: dbias}
                ),
                consider_constant=[
                    x,
                    dy,
                    scale,
                    bias,
                    x_mean2,
                    x_var2,
                    running_mean,
                    running_var,
                ],
                return_disconnected="zero",
            )
            # compile
            f = aesara.function(
                [x, scale, bias, running_mean, running_var, dy, dx, dscale, dbias],
                [
                    out,
                    x_mean,
                    x_invstd,
                    out_running_mean,
                    out_running_var,
                    out2,
                    x_mean2,
                    x_invstd2,
                    out_running_mean2,
                    out_running_var2,
                ]
                + grads
                + grads2
                + grad_grads
                + grad_grads2,
            )
            # check if the abstract Ops have been replaced
            assert not any(
                isinstance(
                    n.op,
                    (
                        batchnorm.AbstractBatchNormTrain,
                        batchnorm.AbstractBatchNormInference,
                        batchnorm.AbstractBatchNormTrainGrad,
                    ),
                )
                for n in f.maker.fgraph.toposort()
            )
            # run
            for data_shape in ((5, 10, 30, 40, 10), (4, 3, 1, 1, 1), (2, 3, 5, 5, 5)):
                data_shape = data_shape[:ndim]
                param_shape = tuple(
                    1 if d in axes2 else s for d, s in enumerate(data_shape)
                )

                rng = np.random.default_rng(1234)

                X = 4 + 3 * rng.random(data_shape).astype(aesara.config.floatX)
                Dy = -1 + 2 * rng.random(data_shape).astype(aesara.config.floatX)
                Scale = rng.random(param_shape).astype(aesara.config.floatX)
                Bias = rng.random(param_shape).astype(aesara.config.floatX)
                Running_mean = rng.random(param_shape).astype(aesara.config.floatX)
                Running_var = rng.random(param_shape).astype(aesara.config.floatX)
                Dx = 4 + 3 * rng.random(data_shape).astype(aesara.config.floatX)
                Dscale = -1 + 2 * rng.random(param_shape).astype(aesara.config.floatX)
                Dbias = rng.random(param_shape).astype(aesara.config.floatX)

                outputs = f(
                    X, Scale, Bias, Running_mean, Running_var, Dy, Dx, Dscale, Dbias
                )
                # compare outputs
                utt.assert_allclose(outputs[0], outputs[0 + 5])  # out
                utt.assert_allclose(outputs[1], outputs[1 + 5])  # mean
                utt.assert_allclose(outputs[2], outputs[2 + 5])  # invstd
                utt.assert_allclose(outputs[3], outputs[3 + 5])  # running_mean
                utt.assert_allclose(
                    np.nan_to_num(outputs[4]), np.nan_to_num(outputs[4 + 5])
                )  # running_var
                # compare gradients
                utt.assert_allclose(outputs[10], outputs[10 + 3], atol=1e-4)  # dx
                utt.assert_allclose(
                    outputs[11], outputs[11 + 3], rtol=2e-4, atol=1e-4
                )  # dscale
                utt.assert_allclose(outputs[12], outputs[12 + 3])  # dbias
                # compare second-order gradients
                utt.assert_allclose(outputs[16], outputs[16 + 3], atol=1e-4)  # ddx
                utt.assert_allclose(outputs[17], outputs[17 + 3])  # ddy
                utt.assert_allclose(
                    outputs[18], outputs[18 + 3], rtol=3e-4, atol=1e-4
                )  # ddscale


@pytest.mark.slow
def test_batch_normalization_train_grad_grad():
    for axes in ("per-activation", "spatial", (1, 2, 3, 4)):
        for vartype in (tensor5, tensor4, tensor3, matrix, vector):
            # run these experiments with float64 for sufficient numerical stability
            x, dy, scale, x_mean, x_invstd = (
                vartype(n, dtype="float64")
                for n in ("x", "dy", "scale", "x_mean", "x_invstd")
            )
            ndim = x.ndim

            # reference forward pass
            if axes == "per-activation":
                axes = (0,)
            elif axes == "spatial":
                axes = (0,) + tuple(range(2, ndim))
            else:
                # remove non-existing axes
                axes = tuple(i for i in axes if i < ndim)
            if len(axes) == 0:
                continue

            def bn_grad_wrt_inputs_f(x, dy, scale, x_mean, x_invstd):
                g_inputs, g_scale, g_bias = batchnorm.AbstractBatchNormTrainGrad(axes)(
                    x, dy, scale, x_mean, x_invstd
                )
                return g_inputs

            def bn_grad_wrt_scale_f(x, dy, scale, x_mean, x_invstd):
                g_inputs, g_scale, g_bias = batchnorm.AbstractBatchNormTrainGrad(axes)(
                    x, dy, scale, x_mean, x_invstd
                )
                return g_scale

            def bn_grad_wrt_bias_f(x, dy, scale, x_mean, x_invstd):
                g_inputs, g_scale, g_bias = batchnorm.AbstractBatchNormTrainGrad(axes)(
                    x, dy, scale, x_mean, x_invstd
                )
                return g_bias

            # run
            for data_shape in ((4, 3, 3, 3, 3), (4, 3, 1, 1, 1), (2, 3, 5, 3, 2)):
                data_shape = data_shape[:ndim]
                param_shape = tuple(
                    1 if d in axes else s for d, s in enumerate(data_shape)
                )
                rng = np.random.default_rng(1234)
                # force float64 for sufficient numerical stability
                x_val = 4 + 3 * rng.random(data_shape).astype("float64")
                dy_val = -1 + 2 * rng.random(data_shape).astype("float64")
                scale_val = rng.random(param_shape).astype("float64")
                x_mean_val = rng.random(param_shape).astype("float64")
                x_invstd_val = rng.random(param_shape).astype("float64")

                utt.verify_grad(
                    bn_grad_wrt_inputs_f,
                    [x_val, dy_val, scale_val, x_mean_val, x_invstd_val],
                    abs_tol=5e-4,
                    rel_tol=5e-4,
                )
                utt.verify_grad(
                    bn_grad_wrt_scale_f,
                    [x_val, dy_val, scale_val, x_mean_val, x_invstd_val],
                )
                utt.verify_grad(
                    bn_grad_wrt_bias_f,
                    [x_val, dy_val, scale_val, x_mean_val, x_invstd_val],
                )


def test_batch_normalization_train_without_running_averages():
    # compile and run batch_normalization_train without running averages

    x, scale, bias, dy = (
        tensor4("x"),
        tensor4("scale"),
        tensor4("bias"),
        tensor4("dy"),
    )
    data_shape = (5, 10, 30, 25)
    param_shape = (1, 10, 30, 25)

    # forward pass
    out, x_mean, x_invstd = batchnorm.batch_normalization_train(
        x, scale, bias, "per-activation"
    )
    # backward pass
    grads = at.grad(None, wrt=[x, scale, bias], known_grads={out: dy})
    # compile
    f = aesara.function([x, scale, bias, dy], [out, x_mean, x_invstd] + grads)
    # check if the abstract Ops have been replaced
    assert not any(
        isinstance(
            n.op,
            (
                batchnorm.AbstractBatchNormTrain,
                batchnorm.AbstractBatchNormInference,
                batchnorm.AbstractBatchNormTrainGrad,
            ),
        )
        for n in f.maker.fgraph.toposort()
    )
    # run
    rng = np.random.default_rng(1234)
    X = 4 + 3 * rng.random(data_shape).astype(aesara.config.floatX)
    Dy = -1 + 2 * rng.random(data_shape).astype(aesara.config.floatX)
    Scale = rng.random(param_shape).astype(aesara.config.floatX)
    Bias = rng.random(param_shape).astype(aesara.config.floatX)
    f(X, Scale, Bias, Dy)


def test_batch_normalization_train_broadcast():
    for axes in ("per-activation", "spatial", (1, 2, 3, 4)):
        for vartype in (tensor5, tensor4, tensor3, matrix, vector):
            x = vartype("x")
            ndim = x.ndim
            eps = 5e-3  # some non-standard value to test if it's used
            running_average_factor = 0.3

            # remove non-existing axes
            if isinstance(axes, tuple):
                axes = tuple(i for i in axes if i < ndim)
            if len(axes) == 0:
                continue

            # convert axes to explicit list
            if axes == "per-activation":
                axes2 = (0,)
            elif axes == "spatial":
                axes2 = (0,) + tuple(range(2, ndim))
            else:
                axes2 = axes

            # compute axes for parameter tensors
            non_bc_axes = tuple(i for i in range(ndim) if i not in axes2)
            params_dimshuffle = ["x"] * ndim
            for i, axis in enumerate(non_bc_axes):
                params_dimshuffle[axis] = i

            # construct non-broadcasted parameter variables
            param_type = TensorType(x.dtype, shape=(None,) * len(non_bc_axes))
            scale, bias, running_mean, running_var = (
                param_type(n) for n in ("scale", "bias", "running_mean", "running_var")
            )

            # broadcast parameter variables
            scale_bc = scale.dimshuffle(params_dimshuffle)
            bias_bc = bias.dimshuffle(params_dimshuffle)
            running_mean_bc = running_mean.dimshuffle(params_dimshuffle)
            running_var_bc = running_var.dimshuffle(params_dimshuffle)

            # batch_normalization_train with original, non-broadcasted variables
            train_non_bc = batchnorm.batch_normalization_train(
                x,
                scale,
                bias,
                axes,
                eps,
                running_average_factor,
                running_mean,
                running_var,
            )
            # batch_normalization_train with broadcasted variables
            train_bc = batchnorm.batch_normalization_train(
                x,
                scale_bc,
                bias_bc,
                axes,
                eps,
                running_average_factor,
                running_mean_bc,
                running_var_bc,
            )
            train_bc = tuple(
                [train_bc[0]] + [r.dimshuffle(non_bc_axes) for r in train_bc[1:]]  # out
            )

            # batch_normalization_test with original, non-broadcasted variables
            test_non_bc = batchnorm.batch_normalization_test(
                x, scale, bias, running_mean, running_var, axes, eps
            )
            # batch_normalization_test with broadcasted variables
            test_bc = batchnorm.batch_normalization_test(
                x, scale_bc, bias_bc, running_mean_bc, running_var_bc, axes, eps
            )

            # subtract the results of the non-broadcasted and broadcasted calls
            results_non_bc = train_non_bc + (test_non_bc,)
            results_bc = train_bc + (test_bc,)
            results = [abs(r - r_bc) for (r, r_bc) in zip(results_non_bc, results_bc)]

            # compile to compute all differences
            f = aesara.function(
                [x, scale, bias, running_mean, running_var], at_sum(sum(results))
            )

            # the paired ops are exactly the same, so the optimizer should have
            # collapsed the sum of differences to a constant zero
            nodes = f.maker.fgraph.toposort()
            if aesara.config.mode != "FAST_COMPILE":
                assert len(nodes) == 1
                assert isinstance(nodes[0].op, aesara.compile.DeepCopyOp)
            inputs = [
                np.asarray(np.random.random((4,) * n), x.dtype)
                for n in [
                    x.ndim,
                    scale.ndim,
                    bias.ndim,
                    running_mean.ndim,
                    running_var.ndim,
                ]
            ]
            assert 0.0 == f(*inputs)


@pytest.mark.slow
def test_batch_normalization_test():
    for axes in ("per-activation", "spatial", (1, 2, 3, 4)):
        for vartype in (tensor5, tensor3, vector):
            x, scale, bias, mean, var = (
                vartype(n) for n in ("x", "scale", "bias", "mean", "var")
            )
            ndim = x.ndim
            eps = 5e-3  # some non-standard value to test if it's used

            # remove non-existing axes
            if isinstance(axes, tuple):
                axes = tuple(i for i in axes if i < ndim)
            if len(axes) == 0:
                continue

            # forward pass
            out = batchnorm.batch_normalization_test(
                x, scale, bias, mean, var, axes, eps
            )
            # reference forward pass
            if axes == "per-activation":
                axes2 = (0,)
            elif axes == "spatial":
                axes2 = (0,) + tuple(range(2, ndim))
            else:
                axes2 = axes
            scale2, bias2, mean2, var2 = (
                specify_broadcastable(t, *axes2) for t in (scale, bias, mean, var)
            )
            out2 = (x - mean2) * (scale2 / at.sqrt(var2 + eps)) + bias2
            # backward pass
            dy = vartype("dy")
            grads = at.grad(
                None, wrt=[x, scale, bias, mean, var], known_grads={out: dy}
            )
            # reference backward pass
            grads2 = at.grad(
                None, wrt=[x, scale, bias, mean, var], known_grads={out2: dy}
            )
            # compile
            f = aesara.function(
                [x, scale, bias, mean, var, dy], [out, out2] + grads + grads2
            )
            # check if the abstract Ops have been replaced
            assert not any(
                isinstance(
                    n.op,
                    (
                        batchnorm.AbstractBatchNormTrain,
                        batchnorm.AbstractBatchNormInference,
                        batchnorm.AbstractBatchNormTrainGrad,
                    ),
                )
                for n in f.maker.fgraph.toposort()
            )
            # run
            for data_shape in ((10, 20, 30, 40, 10), (4, 3, 1, 1, 1), (1, 1, 5, 5, 5)):
                data_shape = data_shape[:ndim]
                param_shape = tuple(
                    1 if d in axes2 else s for d, s in enumerate(data_shape)
                )
                rng = np.random.default_rng(1234)
                X = 4 + 3 * rng.random(data_shape).astype(aesara.config.floatX)
                Dy = -1 + 2 * rng.random(data_shape).astype(aesara.config.floatX)
                Scale = rng.random(param_shape).astype(aesara.config.floatX)
                Bias = rng.random(param_shape).astype(aesara.config.floatX)
                Mean = rng.random(param_shape).astype(aesara.config.floatX)
                Var = rng.random(param_shape).astype(aesara.config.floatX)
                outputs = f(X, Scale, Bias, Mean, Var, Dy)
                # compare outputs
                utt.assert_allclose(outputs[0], outputs[1])  # out
                # compare gradients
                utt.assert_allclose(outputs[2], outputs[2 + 5], atol=4e-5)  # dx
                utt.assert_allclose(outputs[3], outputs[3 + 5], atol=4e-5)  # dscale
                utt.assert_allclose(outputs[4], outputs[4 + 5])  # dbias
                utt.assert_allclose(outputs[5], outputs[5 + 5])  # dmean
                utt.assert_allclose(
                    outputs[6], outputs[6 + 5], rtol=2e-3, atol=4e-5
                )  # dvar


def test_batch_normalization_broadcastable():
    # check if the broadcastable pattern is preserved by the optimizations
    x, dy, scale, bias, mean, var = (
        scalar(n).dimshuffle(["x"] * 5)
        for n in ("x", "dy", "scale", "bias", "mean", "var")
    )

    # forward pass
    out_train, x_mean, x_invstd = batchnorm.batch_normalization_train(
        x, scale, bias, "spatial"
    )
    out_test = batchnorm.batch_normalization_test(x, scale, bias, mean, var, "spatial")
    # backward pass
    grads_train = at.grad(None, wrt=[x, scale, bias], known_grads={out_train: dy})
    grads_test = at.grad(None, wrt=[x, scale, bias], known_grads={out_test: dy})
    # compile
    f = aesara.function(
        [x, scale, bias, mean, var, dy],
        [out_train, x_mean, x_invstd, out_test] + grads_train + grads_test,
    )
    assert not any(
        isinstance(
            n.op,
            (
                batchnorm.AbstractBatchNormTrain,
                batchnorm.AbstractBatchNormInference,
                batchnorm.AbstractBatchNormTrainGrad,
            ),
        )
        for n in f.maker.fgraph.toposort()
    )
