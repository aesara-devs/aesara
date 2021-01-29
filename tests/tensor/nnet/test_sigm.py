import numpy as np

import aesara
import aesara.tensor as aet
from aesara.configdefaults import config
from aesara.graph.opt import check_stack_trace
from aesara.graph.toolbox import is_same_graph
from aesara.tensor.inplace import neg_inplace
from aesara.tensor.math import clip, exp, log, mul, neg
from aesara.tensor.nnet.sigm import (
    ScalarSoftplus,
    compute_mul,
    hard_sigmoid,
    is_1pexp,
    parse_mul_tree,
    perform_sigm_times_exp,
    register_local_1msigmoid,
    sigmoid,
    sigmoid_inplace,
    simplify_mul,
    softplus,
    ultra_fast_sigmoid,
)
from aesara.tensor.shape import Reshape
from aesara.tensor.type import fmatrix, matrix, scalar, vector, vectors
from tests import unittest_tools as utt
from tests.tensor.utils import (
    _good_broadcast_unary_normal_no_complex,
    check_floatX,
    copymod,
    makeBroadcastTester,
    upcast_int8_nfunc,
)


class TestSigmoid:
    def setup_method(self):
        utt.seed_rng()

    def test_elemwise(self):
        utt.verify_grad(sigmoid, [np.random.rand(3, 4)])


TestSigmoidBroadcast = makeBroadcastTester(
    op=sigmoid,
    expected=upcast_int8_nfunc(
        lambda inputs: check_floatX(inputs, 1 / (1 + np.exp(-inputs)))
    ),
    good=copymod(
        _good_broadcast_unary_normal_no_complex, without=["uint16"]
    ),  # The reason that 'uint16' is excluted is that
    # aesara works well but numpy overflows resulting
    # in an assertion error.
    # grad=_grad_broadcast_unary_normal,
    name="SigmoidTester",
    eps=1e-8,
)

TestUltraFastSigmoidBroadcast = makeBroadcastTester(
    op=ultra_fast_sigmoid,
    expected=upcast_int8_nfunc(
        lambda inputs: check_floatX(inputs, 1 / (1 + np.exp(-inputs)))
    ),
    good=copymod(
        _good_broadcast_unary_normal_no_complex, without=["uint16"]
    ),  # numpy fucnting overflows with uint16.
    # grad=_grad_broadcast_unary_normal,
    name="UltraFastSigmoidTester",
    # This is an approx of the sigmoid. That is why we raise eps
    eps=5e-2,
)

TestHardSigmoidBroadcast = makeBroadcastTester(
    op=hard_sigmoid,
    expected=upcast_int8_nfunc(
        lambda inputs: check_floatX(inputs, 1 / (1 + np.exp(-inputs)))
    ),
    good=copymod(
        _good_broadcast_unary_normal_no_complex, without=["uint16"]
    ),  # numpy fucnting overflows with uint16.
    # grad=_grad_broadcast_unary_normal,
    name="HardSigmoidTester",
    # This is an approx of the sigmoid. That is why we raise eps
    eps=1e-1,
)


TestSoftplusBroadcast = makeBroadcastTester(
    op=softplus,
    expected=upcast_int8_nfunc(
        lambda inputs: check_floatX(inputs, np.log1p(np.exp(inputs)))
    ),
    good=dict(
        copymod(
            _good_broadcast_unary_normal_no_complex,
            without=["uint8", "uint16", "big_scalar"],
        ),  # numpy function overflows with uint16.
        uint8=[
            np.arange(0, 89, dtype="uint8")
        ],  # the range is different in new added uint8.
        int8=[np.arange(-127, 89, dtype="int8")],
    ),
    # grad=_grad_broadcast_unary_normal,
    name="SoftplusTester",
)


class TestSoftplus:
    def setup_method(self):
        utt.seed_rng()

    def test_elemwise(self):
        utt.verify_grad(softplus, [np.random.rand(3, 4)])

    def test_accuracy(self):
        # Test all aproximations are working (cutoff points are -37, 18, 33.3)
        x_test = np.array([-40.0, -17.5, 17.5, 18.5, 40.0])
        y_th = softplus(x_test).eval()
        y_np = np.log1p(np.exp(x_test))
        np.testing.assert_allclose(y_th, y_np, rtol=10e-10)


class TestSigmoidOpts:
    def get_mode(self, excluding=None):
        """
        Return appropriate mode for the tests.

        :param excluding: List of optimizations to exclude.

        :return: The current default mode unless the `config.mode` option is
        set to 'FAST_COMPILE' (in which case it is replaced by the 'FAST_RUN'
        mode), without the optimizations specified in `excluding`.
        """
        if excluding is None:
            excluding = []
        m = config.mode
        if m == "FAST_COMPILE":
            mode = aesara.compile.mode.get_mode("FAST_RUN")
        else:
            mode = aesara.compile.mode.get_default_mode()
        if excluding:
            return mode.excluding(*excluding)
        else:
            return mode

    def test_exp_over_1_plus_exp(self):
        m = self.get_mode(excluding=["local_elemwise_fusion"])

        x = vector()
        data = np.random.rand(54).astype(config.floatX)

        backup = config.warn__identify_1pexp_bug
        config.warn__identify_1pexp_bug = False
        try:
            # tests exp_over_1_plus_exp
            f = aesara.function([x], exp(x) / (1 + exp(x)), mode=m)
            assert [node.op for node in f.maker.fgraph.toposort()] == [sigmoid]
            f(data)
            f = aesara.function([x], exp(x) / (2 + exp(x)), mode=m)
            assert [node.op for node in f.maker.fgraph.toposort()] != [sigmoid]
            f(data)
            f = aesara.function([x], exp(x) / (1 - exp(x)), mode=m)
            assert [node.op for node in f.maker.fgraph.toposort()] != [sigmoid]
            f(data)
            f = aesara.function([x], exp(x + 1) / (1 + exp(x)), mode=m)
            assert [node.op for node in f.maker.fgraph.toposort()] != [sigmoid]
            f(data)

            # tests inv_1_plus_exp
            f = aesara.function([x], aet.fill(x, 1.0) / (1 + exp(-x)), mode=m)
            # todo: solve issue #4589 first
            # assert check_stack_trace(f, ops_to_check=sigmoid)
            assert [node.op for node in f.maker.fgraph.toposort()] == [sigmoid]
            f(data)
            f = aesara.function([x], aet.fill(x, 1.0) / (2 + exp(-x)), mode=m)
            assert [node.op for node in f.maker.fgraph.toposort()] != [sigmoid]
            f(data)
            f = aesara.function([x], aet.fill(x, 1.0) / (1 - exp(-x)), mode=m)
            assert [node.op for node in f.maker.fgraph.toposort()] != [sigmoid]
            f(data)
            f = aesara.function([x], aet.fill(x, 1.1) / (1 + exp(-x)), mode=m)
            assert [node.op for node in f.maker.fgraph.toposort()] != [sigmoid]
            f(data)

            # tests inv_1_plus_exp with neg
            f = aesara.function([x], aet.fill(x, -1.0) / (1 + exp(-x)), mode=m)
            # todo: solve issue #4589 first
            # assert check_stack_trace(
            #     f, ops_to_check=[sigmoid, neg_inplace])
            assert [node.op for node in f.maker.fgraph.toposort()] == [
                sigmoid,
                neg_inplace,
            ]
            f(data)
            f = aesara.function([x], aet.fill(x, -1.0) / (1 - exp(-x)), mode=m)
            assert [node.op for node in f.maker.fgraph.toposort()] != [
                sigmoid,
                neg_inplace,
            ]
            f(data)
            f = aesara.function([x], aet.fill(x, -1.0) / (2 + exp(-x)), mode=m)
            assert [node.op for node in f.maker.fgraph.toposort()] != [
                sigmoid,
                neg_inplace,
            ]
            f(data)
            f = aesara.function([x], aet.fill(x, -1.1) / (1 + exp(-x)), mode=m)
            assert [node.op for node in f.maker.fgraph.toposort()] != [
                sigmoid,
                neg_inplace,
            ]
            f(data)

            # tests double inv_1_plus_exp with neg
            # (-1)(exp(x)) / (1+exp(x))(1+exp(-x))
            # = (-1)/(1+exp(-x)) * exp(x)/(1+exp(x))
            # = - (sigm(x) * sigm(x))
            f = aesara.function(
                [x],
                (aet.fill(x, -1.0) * exp(x)) / ((1 + exp(x)) * (1 + exp(-x))),
                mode=m,
            )
            # todo: solve issue #4589 first
            # assert check_stack_trace(f, ops_to_check=[sigmoid, mul])
            assert [node.op for node in f.maker.fgraph.toposort()] == [sigmoid, mul]
            f(data)
            f = aesara.function(
                [x],
                (aet.fill(x, -1.1) * exp(x)) / ((1 + exp(x)) * (1 + exp(-x))),
                mode=m,
            )
            assert [node.op for node in f.maker.fgraph.toposort()] != [
                sigmoid,
                mul,
                neg_inplace,
            ]
            f(data)
            f = aesara.function(
                [x],
                (aet.fill(x, -1.0) * exp(x)) / ((2 + exp(x)) * (1 + exp(-x))),
                mode=m,
            )
            assert [node.op for node in f.maker.fgraph.toposort()] != [
                sigmoid,
                mul,
                neg_inplace,
            ]
            f(data)
            f = aesara.function(
                [x],
                (aet.fill(x, -1.0) * exp(x)) / ((1 + exp(x)) * (2 + exp(-x))),
                mode=m,
            )
            assert [node.op for node in f.maker.fgraph.toposort()] != [
                sigmoid,
                mul,
                neg_inplace,
            ]
            f(data)
            f = aesara.function(
                [x],
                (aet.fill(x, -1.0) * exp(x)) / ((1 + exp(x)) * (1 + exp(x))),
                mode=m,
            )
            assert [node.op for node in f.maker.fgraph.toposort()] != [
                sigmoid,
                mul,
                neg_inplace,
            ]
            f(data)
            f = aesara.function(
                [x],
                (aet.fill(x, -1.0) * exp(x)) / ((1 + exp(x)) * (2 + exp(-x))),
                mode=m,
            )
            assert [node.op for node in f.maker.fgraph.toposort()] != [
                sigmoid,
                mul,
                neg_inplace,
            ]
            f(data)

        finally:
            # Restore config option.
            config.warn__identify_1pexp_bug = backup

    def test_1msigmoid(self):
        if not register_local_1msigmoid:
            return

        m = self.get_mode()
        x = fmatrix()

        # tests exp_over_1_plus_exp
        f = aesara.function([x], 1 - exp(x) / (1 + exp(x)), mode=m)
        assert check_stack_trace(f, ops_to_check=[neg, sigmoid_inplace])
        assert [node.op for node in f.maker.fgraph.toposort()] == [
            neg,
            sigmoid_inplace,
        ]

        # tests inv_1_plus_exp
        f = aesara.function([x], 1 - aet.fill(x, 1.0) / (1 + exp(-x)), mode=m)
        assert check_stack_trace(f, ops_to_check=[neg, sigmoid_inplace])
        assert [node.op for node in f.maker.fgraph.toposort()] == [
            neg,
            sigmoid_inplace,
        ]

    def test_local_sigm_times_exp(self):
        # Test the `local_sigm_times_exp` optimization.
        # exp(x) * sigm(-x) -> sigm(x)
        # exp(-x) * sigm(x) -> sigm(-x)

        def match(func, ops):
            # print [node.op.scalar_op for node in func.maker.fgraph.toposort()]
            assert [node.op for node in func.maker.fgraph.toposort()] == ops

        m = self.get_mode(excluding=["local_elemwise_fusion", "inplace"])
        x, y = vectors("x", "y")

        f = aesara.function([x], sigmoid(-x) * exp(x), mode=m)
        match(f, [sigmoid])
        assert check_stack_trace(f, ops_to_check=sigmoid)

        f = aesara.function([x], sigmoid(x) * exp(-x), mode=m)
        match(f, [neg, sigmoid])
        assert check_stack_trace(f, ops_to_check=sigmoid)

        f = aesara.function([x], -(-(-(sigmoid(x)))) * exp(-x), mode=m)
        match(f, [neg, sigmoid, neg])
        # assert check_stack_trace(f, ops_to_check=sigmoid)

        f = aesara.function(
            [x, y],
            (sigmoid(x) * sigmoid(-y) * -exp(-x) * exp(x * y) * exp(y)),
            mode=m,
        )
        topo = f.maker.fgraph.toposort()
        for op, nb in [(sigmoid, 2), (mul, 2), (neg, 1), (exp, 1)]:
            assert sum([n.op == op for n in topo]) == nb
        # assert check_stack_trace(f, ops_to_check=[sigmoid, mul,
        #                                           exp])

    def test_perform_sigm_times_exp(self):
        # Test the core function doing the `sigm_times_exp` optimization.
        #
        # It is easier to test different graph scenarios this way than by
        # compiling an Aesara function.

        x, y, z, t = vectors("x", "y", "z", "t")
        exp_op = exp

        def ok(expr1, expr2):
            trees = [parse_mul_tree(e) for e in (expr1, expr2)]
            perform_sigm_times_exp(trees[0])
            trees[0] = simplify_mul(trees[0])
            good = is_same_graph(compute_mul(trees[0]), compute_mul(trees[1]))
            if not good:
                print(trees[0])
                print(trees[1])
                print("***")
                aesara.printing.debugprint(compute_mul(trees[0]))
                print("***")
                aesara.printing.debugprint(compute_mul(trees[1]))
            assert good

        ok(sigmoid(x) * exp_op(-x), sigmoid(-x))
        ok(
            -x * sigmoid(x) * (y * (-1 * z) * exp_op(-x)),
            -x * sigmoid(-x) * (y * (-1 * z)),
        )
        ok(
            -sigmoid(-x)
            * (
                exp_op(y)
                * (-exp_op(-z) * 3 * -exp_op(x))
                * (y * 2 * (-sigmoid(-y) * (z + t) * exp_op(z)) * sigmoid(z))
            )
            * -sigmoid(x),
            sigmoid(x)
            * (-sigmoid(y) * (-sigmoid(-z) * 3) * (y * 2 * ((z + t) * exp_op(z))))
            * (-sigmoid(x)),
        )
        ok(
            exp_op(-x) * -exp_op(-x) * (-sigmoid(x) * -sigmoid(x)),
            -sigmoid(-x) * sigmoid(-x),
        )
        ok(-exp_op(x) * -sigmoid(-x) * -exp_op(-x), -sigmoid(-x))

    def test_grad_log1msigm(self):
        # At some point, this returned nan, because (1 - sigm(x)) was
        # on both the numerator and the denominator of a fraction,
        # but the two nodes in question had not been merged.
        x = matrix("x")
        lr = scalar("lr")

        s = sigmoid(x)
        l = log(1 - s)
        c = l.mean()
        ux = x - lr * aesara.grad(c, x)

        # Before the optimization, inf and NaN will be produced in the graph,
        # and DebugMode will complain. Everything is fine afterwards.
        mode = self.get_mode()
        if not isinstance(mode, aesara.compile.debugmode.DebugMode):
            f = aesara.function([x, lr], ux, mode=mode)
            ux_v = f([[50]], 0.1)
            assert not np.isnan(ux_v)

    def test_local_ultra_fast_sigmoid(self):
        x = matrix("x")
        s = sigmoid(x)

        mode = self.get_mode("local_ultra_fast_sigmoid")
        f = aesara.function([x], s, mode=mode)
        assert check_stack_trace(f, ops_to_check=sigmoid)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert topo[0].op == sigmoid

        mode = self.get_mode().including("local_ultra_fast_sigmoid")
        f = aesara.function([x], s, mode=mode)
        assert check_stack_trace(f, ops_to_check=ultra_fast_sigmoid)
        topo = f.maker.fgraph.toposort()
        assert topo[0].op == ultra_fast_sigmoid
        assert len(topo) == 1
        f([[-50, -10, -4, -1, 0, 1, 4, 10, 50]])

    def test_local_hard_sigmoid(self):
        x = matrix("x")
        s = sigmoid(x)

        mode = self.get_mode("local_hard_sigmoid")
        f = aesara.function([x], s, mode=mode)
        assert check_stack_trace(f, ops_to_check=sigmoid)
        topo = f.maker.fgraph.toposort()
        assert topo[0].op == sigmoid
        assert len(topo) == 1

        mode = self.get_mode().including("local_hard_sigmoid")
        f = aesara.function([x], s, mode=mode)
        topo = f.maker.fgraph.toposort()
        assert not any([n.op == sigmoid for n in topo])
        f([[-50, -10, -4, -1, 0, 1, 4, 10, 50]])

        mode2 = mode.excluding("fusion").excluding("inplace")
        f2 = aesara.function([x], s, mode=mode2)
        assert check_stack_trace(f2, ops_to_check=clip)


class TestSoftplusOpts:
    def setup_method(self):
        if aesara.config.mode == "FAST_COMPILE":
            m = aesara.compile.mode.get_mode("FAST_RUN").excluding(
                "local_elemwise_fusion"
            )
        else:
            m = aesara.compile.mode.get_default_mode().excluding(
                "local_elemwise_fusion"
            )
        self.m = m
        utt.seed_rng()

    def test_logsigm_to_softplus(self):
        x = vector()

        out = log(sigmoid(x))
        f = aesara.function([x], out, mode=self.m)

        # Fix ticket #4581 first
        # assert check_stack_trace(
        #     f, ops_to_check=(aesara.scalar.Neg,
        #                      ScalarSoftplus))
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 3
        assert isinstance(topo[0].op.scalar_op, aesara.scalar.Neg)
        assert isinstance(topo[1].op.scalar_op, ScalarSoftplus)
        assert isinstance(topo[2].op.scalar_op, aesara.scalar.Neg)
        f(np.random.rand(54).astype(config.floatX))

    def test_log1msigm_to_softplus(self):
        x = matrix()

        out = log(1 - sigmoid(x))
        f = aesara.function([x], out, mode=self.m)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 2
        assert isinstance(topo[0].op.scalar_op, ScalarSoftplus)
        assert isinstance(topo[1].op.scalar_op, aesara.scalar.Neg)
        # assert check_stack_trace(f, ops_to_check='all')
        f(np.random.rand(54, 11).astype(config.floatX))

        # Same test with a flatten
        out = log(1 - aet.flatten(sigmoid(x)))
        f = aesara.function([x], out, mode=self.m)

        # assert check_stack_trace(f, ops_to_check='all')
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 3
        assert aet.is_flat(topo[0].outputs[0])
        assert isinstance(topo[1].op.scalar_op, ScalarSoftplus)
        assert isinstance(topo[2].op.scalar_op, aesara.scalar.Neg)
        f(np.random.rand(54, 11).astype(config.floatX))

        # Same test with a reshape
        out = log(1 - sigmoid(x).reshape([x.size]))
        f = aesara.function([x], out, mode=self.m)
        topo = f.maker.fgraph.toposort()
        # assert len(topo) == 3
        assert any(isinstance(node.op, Reshape) for node in topo)
        assert any(
            isinstance(
                getattr(node.op, "scalar_op", None),
                ScalarSoftplus,
            )
            for node in topo
        )
        f(np.random.rand(54, 11).astype(config.floatX))

    def test_log1pexp_to_softplus(self):
        m = aesara.config.mode
        if m == "FAST_COMPILE":
            m = "FAST_RUN"

        x = vector()

        out = log(1 + exp(x))
        f = aesara.function([x], out, mode=self.m)

        # Fix ticket #4581 first
        # assert check_stack_trace(f, ops_to_check='all')
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert isinstance(topo[0].op.scalar_op, ScalarSoftplus)
        f(np.random.rand(54).astype(config.floatX))


class TestSigmoidUtils:
    """
    Test utility functions found in 'sigm.py'.
    """

    def test_compute_mul(self):
        x, y, z = vectors("x", "y", "z")
        tree = (x * y) * -z
        mul_tree = parse_mul_tree(tree)
        assert parse_mul_tree(compute_mul(mul_tree)) == mul_tree
        assert is_same_graph(compute_mul(parse_mul_tree(tree)), tree)

    def test_parse_mul_tree(self):
        x, y, z = vectors("x", "y", "z")
        assert parse_mul_tree(x * y) == [False, [[False, x], [False, y]]]
        assert parse_mul_tree(-(x * y)) == [True, [[False, x], [False, y]]]
        assert parse_mul_tree(-x * y) == [False, [[True, x], [False, y]]]
        assert parse_mul_tree(-x) == [True, x]
        assert parse_mul_tree((x * y) * -z) == [
            False,
            [[False, [[False, x], [False, y]]], [True, z]],
        ]

    def test_is_1pexp(self):
        backup = config.warn__identify_1pexp_bug
        config.warn__identify_1pexp_bug = False
        try:
            x = vector("x")
            exp_op = exp
            assert is_1pexp(1 + exp_op(x), False) == (False, x)
            assert is_1pexp(exp_op(x) + 1, False) == (False, x)
            for neg_, exp_arg in map(
                lambda x: is_1pexp(x, only_process_constants=False),
                [(1 + exp_op(-x)), (exp_op(-x) + 1)],
            ):
                assert not neg_ and is_same_graph(exp_arg, -x)
            assert is_1pexp(1 - exp_op(x), False) is None
            assert is_1pexp(2 + exp_op(x), False) is None
            assert is_1pexp(exp_op(x) + 2, False) is None
            assert is_1pexp(exp_op(x) - 1, False) is None
            assert is_1pexp(-1 + exp_op(x), False) is None
            assert is_1pexp(1 + 2 * exp_op(x), False) is None
        finally:
            config.warn__identify_1pexp_bug = backup
