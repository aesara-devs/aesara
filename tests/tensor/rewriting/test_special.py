import numpy as np
import pytest

import aesara
import aesara.tensor as at
from aesara import shared
from aesara.compile import optdb
from aesara.compile.function import function
from aesara.compile.mode import OPT_FAST_RUN, Mode, get_mode
from aesara.configdefaults import config
from aesara.graph.fg import FunctionGraph
from aesara.graph.rewriting.basic import check_stack_trace
from aesara.graph.rewriting.db import RewriteDatabaseQuery
from aesara.tensor.math import add, exp, log, true_div
from aesara.tensor.special import LogSoftmax, Softmax, SoftmaxGrad, softmax
from aesara.tensor.type import matrix
from tests import unittest_tools as utt


class TestLogSoftmaxRewrites:
    @pytest.mark.parametrize("axis", [None, 0, -1])
    def test_local_logsoftmax_rewrite(self, axis):
        """Test the `Logsoftmax` substitution.

        Check that ``Log(Softmax(x))`` is substituted with ``Logsoftmax(x)``. Note that
        only the forward pass is checked (i.e., doesn't check the gradient)
        """

        x = matrix("x")
        sm = softmax(x, axis=axis)
        logsm = log(sm)
        f = function([x], logsm)
        assert isinstance(f.maker.fgraph.outputs[0].owner.op, LogSoftmax)
        assert check_stack_trace(f, ops_to_check=LogSoftmax)

    @pytest.mark.parametrize("axis", [None, 0, -1])
    def test_local_logsoftmax_grad_rewrite(self, axis):
        """Test the `Logsoftmax`'s grad substitution.

        Check that ``Log(Softmax(x))``'s grad is substituted with ``Logsoftmax(x)``'s
        grad and that the new operation does not explode for big inputs.
        Note that only the grad is checked.
        """

        m = config.mode
        m = get_mode(m)
        m.check_isfinite = False
        # some inputs that are large to make the gradient explode in the non
        # rewritten case
        rng = np.random.default_rng(utt.fetch_seed())
        a = np.exp(10 * rng.random((5, 10)).astype(config.floatX))

        def myfunc(x):
            sm = softmax(x, axis=axis)
            logsm = log(sm)
            return logsm

        # We set step to 0.1 because for big values we need a big epsilon
        utt.verify_grad(myfunc, [a], eps=0.1, mode=m)
        sa = shared(a)
        f = function([], myfunc(sa))
        assert check_stack_trace(f, ops_to_check="all")

    def test_logsoftmax_grad_true_div_elemwise(self):
        """
        Checks that the gradient of an expression similar to a ``log(softmax)`` but
        with a different elemwise operation than true_div is not rewritten.
        """

        x = matrix("x")
        y = log(softmax(x))
        g = aesara.tensor.grad(y.sum(), x)

        softmax_grad_node = g.owner
        assert softmax_grad_node.op == SoftmaxGrad(axis=-1)
        true_div_node = softmax_grad_node.inputs[0].owner
        assert true_div_node.op == true_div

        # We replace the elemwise true_div op by an elemwise add.
        new_g = SoftmaxGrad(axis=-1)(
            add(*true_div_node.inputs), softmax_grad_node.inputs[1]
        )

        fgraph = FunctionGraph([x], [new_g])
        optdb.query(OPT_FAST_RUN).rewrite(fgraph)

        assert SoftmaxGrad(axis=-1) in [n.op for n in fgraph.toposort()]


def test_log1mexp_stabilization():
    mode = Mode("py").including("stabilize")

    x = vector()
    f = function([x], log(1 - exp(x)), mode=mode)

    nodes = [node.op for node in f.maker.fgraph.toposort()]
    assert nodes == [at.log1mexp]

    # Check values that would under or overflow without rewriting
    assert f([-(2.0**-55)]) != -np.inf
    overflow_value = -500.0 if config.floatX == "float64" else -100.0
    assert f([overflow_value]) < 0

    # Check values around the switch point np.log(0.5)
    assert np.allclose(
        f(np.array([-0.8, -0.6], dtype=config.floatX)),
        np.log(1 - np.exp([-0.8, -0.6])),
    )


def test_log_softmax_stabilization():
    mode = aesara.compile.mode.get_default_mode()
    mode = mode.including("local_log_softmax", "specialize")

    x = matrix()
    y = softmax(x)
    z = log(y)

    f = aesara.function([x], z, mode=mode)
    assert check_stack_trace(f, ops_to_check="all")

    # Check that the softmax has been rewritten
    for node in f.maker.fgraph.toposort():
        assert not isinstance(node.op, y.owner.op.__class__)

    # Call the function so debug mode can verify the rewritten version matches
    # the un-rewritten version
    rng = np.random.default_rng(utt.fetch_seed())
    f(np.cast[config.floatX](rng.random((2, 3))))


def test_softmax_graph():
    """Make sure that sotfmax expressions are turned into
    a softmax Op.

    """
    rng = np.random.default_rng(utt.fetch_seed())
    x = aesara.shared(rng.normal(size=(3, 4)))

    def softmax_graph(c):
        return exp(c) / exp(c).sum(axis=-1, keepdims=True)

    def f(inputs):
        y = softmax_graph(x)
        return aesara.grad(None, x, known_grads={y: inputs})

    utt.verify_grad(f, [rng.random((3, 4))])
