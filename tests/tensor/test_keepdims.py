import numpy as np
import pytest

import aesara
from aesara import function
from aesara.compile.mode import Mode
from aesara.tensor.elemwise import DimShuffle
from aesara.tensor.math import all as at_all
from aesara.tensor.math import any as at_any
from aesara.tensor.math import argmax, argmin
from aesara.tensor.math import max as at_max
from aesara.tensor.math import max_and_argmax, mean
from aesara.tensor.math import min as at_min
from aesara.tensor.math import prod, std
from aesara.tensor.math import sum as at_sum
from aesara.tensor.math import var
from aesara.tensor.type import dtensor3


class TestKeepDims:
    r"""This tests other `Op`\s to ensure they keep the dimensions of their inputs correctly."""

    def makeKeepDims_local(self, x, y, axis):
        if axis is None:
            newaxis = list(range(x.ndim))
        elif isinstance(axis, int):
            if axis < 0:
                newaxis = [axis + x.type.ndim]
            else:
                newaxis = [axis]
        else:
            newaxis = []
            for a in axis:
                if a < 0:
                    a += x.type.ndim
                newaxis.append(a)
        i = 0
        new_dims = []
        for j, _ in enumerate(x.shape):
            if j in newaxis:
                new_dims.append("x")
            else:
                new_dims.append(i)
                i += 1

        return DimShuffle(y.type.broadcastable, new_dims)(y)

    @pytest.mark.parametrize(
        "axis",
        [
            0,
            1,
            2,
            [0],
            [1],
            [2],
            None,
            [0, 1, 2],
            [-1],
            [-2],
            [-3],
            [-1, -2, -3],
            [0, -1, -2],
            [-2, -3, 2],
        ],
    )
    @pytest.mark.parametrize(
        "op",
        [max_and_argmax],
    )
    def test_max_and_argmax(self, axis, op):
        x = dtensor3()
        a = np.random.random((3, 2, 4))
        # We don't need to test all opt and C code, as this is tested
        # by the ops tests.
        mode = Mode(optimizer="fast_compile", linker="py")

        # 'max_and_argmax' has two outputs and can be specified with either
        # a single or every axis:
        f = function(
            [x],
            [
                op(x, axis=axis, keepdims=True)[0],
                self.makeKeepDims_local(x, op(x, axis=axis, keepdims=False)[0], axis),
            ],
            mode=mode,
        )
        ans1, ans2 = f(a)
        assert np.allclose(ans1, ans2)
        assert ans1.shape == ans2.shape

        f = function(
            [x],
            [
                op(x, axis=axis, keepdims=True)[1],
                self.makeKeepDims_local(x, op(x, axis=axis, keepdims=False)[1], axis),
            ],
            mode=mode,
        )
        ans1, ans2 = f(a)
        assert np.allclose(ans1, ans2)
        assert ans1.shape == ans2.shape

    @pytest.mark.parametrize(
        "axis",
        [
            0,
            1,
            2,
            [0],
            [1],
            [2],
            None,
            [0, 1, 2],
            [-1],
            [-2],
            [-3],
            [-1, -2, -3],
            [0, -2, 2],
        ],
    )
    @pytest.mark.parametrize(
        "op",
        [argmax, argmin],
    )
    def test_single_or_any_axis(self, axis, op):
        # the following ops can be specified with either a single axis or every
        # axis:
        x = dtensor3()
        a = np.random.random((3, 2, 4))
        # We don't need to test all opt and C code, as this is tested
        # by the ops tests.
        mode = Mode(optimizer="fast_compile", linker="py")

        f = function(
            [x],
            [
                op(x, axis=axis, keepdims=True),
                self.makeKeepDims_local(x, op(x, axis=axis, keepdims=False), axis),
            ],
            mode=mode,
        )
        ans1, ans2 = f(a)
        assert np.allclose(ans1, ans2)
        assert ans1.shape == ans2.shape

    @pytest.mark.parametrize(
        "axis",
        [
            0,
            1,
            2,
            [0],
            [1],
            [2],
            None,
            [0, 1],
            [1, 2],
            [0, 1, 2],
            [-1],
            [-2],
            [-3],
            [-1, -2],
            [-1, -2, -3],
            [0, -2, 2],
        ],
    )
    @pytest.mark.parametrize(
        "op",
        [
            at_sum,
            prod,
            mean,
            var,
            std,
            at_all,
            at_any,
            at_max,
            at_min,
        ],
    )
    def test_free_axis(self, axis, op):
        x = dtensor3()
        a = np.random.random((3, 2, 4))
        # We don't need to test all opt and C code, as this is tested
        # by the ops tests.
        mode = Mode(optimizer="fast_compile", linker="py")

        # the following ops can be specified with a freely specified axis
        # parameter
        f = function(
            [x],
            [
                op(x, axis=axis, keepdims=True),
                self.makeKeepDims_local(x, op(x, axis=axis, keepdims=False), axis),
            ],
            mode=mode,
        )

        ans1, ans2 = f(a)
        assert np.allclose(ans1, ans2)
        assert ans1.shape == ans2.shape

    @pytest.mark.parametrize(
        "axis",
        [
            0,
            1,
            2,
            [0],
            [1],
            [2],
            None,
            [0, 1],
            [1, 2],
            [0, 1, 2],
            [-1],
            [-2],
            [-3],
            [-1, -2],
            [-1, -2, -3],
            [0, -2, 2],
        ],
    )
    def test_norm(self, axis):
        x = dtensor3()
        a = np.random.random((3, 2, 4)).astype(aesara.config.floatX)
        mode = Mode(optimizer="fast_compile", linker="py")

        f = function(
            [x],
            [
                x.norm(L=1, axis=axis, keepdims=True),
                self.makeKeepDims_local(
                    x, x.norm(L=1, axis=axis, keepdims=False), axis
                ),
            ],
            mode=mode,
        )

        ans1, ans2 = f(a)
        assert np.allclose(ans1, ans2)
        assert ans1.shape == ans2.shape

        g = function(
            [x],
            [
                x.norm(L=2, axis=axis, keepdims=True),
                self.makeKeepDims_local(
                    x, x.norm(L=2, axis=axis, keepdims=False), axis
                ),
            ],
            mode=mode,
        )

        ans1, ans2 = g(a)
        assert np.allclose(ans1, ans2)
        assert ans1.shape == ans2.shape
