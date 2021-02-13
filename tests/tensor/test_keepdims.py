import numpy as np
import pytest

import aesara
from aesara import function
from aesara.compile.mode import Mode
from aesara.tensor.elemwise import DimShuffle
from aesara.tensor.math import all as aet_all
from aesara.tensor.math import any as aet_any
from aesara.tensor.math import argmax, argmin
from aesara.tensor.math import max as aet_max
from aesara.tensor.math import max_and_argmax, mean
from aesara.tensor.math import min as aet_min
from aesara.tensor.math import prod, std
from aesara.tensor.math import sum as aet_sum
from aesara.tensor.math import var
from aesara.tensor.type import dtensor3


# this tests other ops to ensure they keep the dimensions of their
# inputs correctly
class TestKeepDims:
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

    @pytest.mark.slow
    def test_keepdims(self):

        x = dtensor3()
        a = np.random.rand(3, 2, 4)
        # We don't need to test all opt and C code, as this is tested
        # by the ops tests.
        mode = Mode(optimizer="fast_compile", linker="py")

        # 'max_and_argmax' has two outputs and can be specified with either
        # a single or every axis:
        for axis in [
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
        ]:

            op = max_and_argmax
            f = function(
                [x],
                [
                    op(x, axis=axis, keepdims=True)[0],
                    self.makeKeepDims_local(
                        x, op(x, axis=axis, keepdims=False)[0], axis
                    ),
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
                    self.makeKeepDims_local(
                        x, op(x, axis=axis, keepdims=False)[1], axis
                    ),
                ],
                mode=mode,
            )
            ans1, ans2 = f(a)
            assert np.allclose(ans1, ans2)
            assert ans1.shape == ans2.shape

        # the following ops can be specified with either a single axis or every
        # axis:
        for op in [argmax, argmin]:
            for axis in [
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
            ]:

                f = function(
                    [x],
                    [
                        op(x, axis=axis, keepdims=True),
                        self.makeKeepDims_local(
                            x, op(x, axis=axis, keepdims=False), axis
                        ),
                    ],
                    mode=mode,
                )
                ans1, ans2 = f(a)
                assert np.allclose(ans1, ans2)
                assert ans1.shape == ans2.shape

        # the following ops can be specified with a freely specified axis
        # parameter
        for op in [
            aet_sum,
            prod,
            mean,
            var,
            std,
            aet_all,
            aet_any,
            aet_max,
            aet_min,
        ]:
            for axis in [
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
            ]:

                f = function(
                    [x],
                    [
                        op(x, axis=axis, keepdims=True),
                        self.makeKeepDims_local(
                            x, op(x, axis=axis, keepdims=False), axis
                        ),
                    ],
                    mode=mode,
                )

                ans1, ans2 = f(a)
                assert np.allclose(ans1, ans2)
                assert ans1.shape == ans2.shape

    def test_norm(self):

        x = dtensor3()
        a = np.random.rand(3, 2, 4).astype(aesara.config.floatX)
        mode = Mode(optimizer="fast_compile", linker="py")

        for axis in [
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
        ]:

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
