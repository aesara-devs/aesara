from collections import OrderedDict

import numpy as np
import pytest

import theano
import theano.tensor as tt
from tests import unittest_tools as utt
from theano.configdefaults import config
from theano.gradient import (
    DisconnectedInputError,
    DisconnectedType,
    GradClip,
    GradScale,
    NullTypeGradError,
    UndefinedGrad,
    consider_constant,
    consider_constant_,
    disconnected_grad,
    disconnected_grad_,
    grad,
    grad_clip,
    grad_not_implemented,
    grad_scale,
    grad_undefined,
    subgraph_grad,
    zero_grad,
    zero_grad_,
)
from theano.graph.basic import Apply, graph_inputs
from theano.graph.null_type import NullType
from theano.graph.op import Op
from theano.sandbox.rng_mrg import MRG_RandomStream
from theano.tensor.type import (
    discrete_dtypes,
    fscalar,
    fvector,
    imatrix,
    iscalar,
    lscalar,
    matrix,
    scalar,
    vector,
)


one = tt.as_tensor_variable(1.0)


def grad_sources_inputs(sources, inputs):
    """
    This implements the old grad_sources_inputs function in terms of
    the new interface so the tests don't need to be rewritten.
    """
    if inputs is None:
        inputs = list(graph_inputs([source[0] for source in sources]))
    return dict(
        zip(
            inputs,
            grad(
                cost=None,
                known_grads=dict(sources),
                wrt=inputs,
                consider_constant=inputs,
            ),
        )
    )


class TestGradSourcesInputs:
    def test_retNone1(self):
        # Test that it is not ok to return None from op.grad()
        class retNone(Op):
            __props__ = ()

            def make_node(self):
                inputs = [vector()]
                outputs = [vector()]
                return Apply(self, inputs, outputs)

            def grad(self, inp, grads):
                (x,) = inp
                (gz,) = grads

            def perform(self, *args, **kwargs):
                raise NotImplementedError()

        a = retNone().make_node()
        with pytest.raises(TypeError):
            grad_sources_inputs([(a.out, one)], None)

    def test_wrong_rval_len1(self):
        # Test that it is not ok to return the wrong number of gradient terms

        class retOne(Op):
            __props__ = ()

            def make_node(self, *inputs):
                outputs = [vector()]
                return Apply(self, inputs, outputs)

            def grad(self, inputs, grads):
                return [inputs[0].zeros_like()]

            def perform(self, *args, **kwargs):
                raise NotImplementedError()

        i = vector()
        j = vector()
        a1 = retOne().make_node(i)
        grad_sources_inputs([(a1.out, one)], None)
        a2 = retOne().make_node(i, j)
        with pytest.raises(ValueError):
            grad_sources_inputs([(a2.out, one)], None)

    def test_1in_1out(self):
        # Test grad is called correctly for a 1-to-1 op
        gval = matrix()

        class TestOp(Op):
            __props__ = ()

            def make_node(self):
                inputs = [matrix()]
                outputs = [matrix()]
                return Apply(self, inputs, outputs)

            def grad(self, inp, grads):
                return (gval,)

            def perform(self, *args, **kwargs):
                raise NotImplementedError()

        a1 = TestOp().make_node()
        g = grad_sources_inputs([(a1.outputs[0], one)], None)
        assert g[a1.inputs[0]] is gval

    def test_1in_Nout(self):
        # Test grad is called correctly for a 1-to-many op
        gval = matrix()

        class TestOp(Op):
            __props__ = ()

            def make_node(self):
                inputs = [matrix()]
                outputs = [scalar(), scalar()]
                return Apply(self, inputs, outputs)

            def grad(self, inp, grads):
                (x,) = inp
                gz1, gz2 = grads
                return (gval,)

            def perform(self, *args, **kwargs):
                raise NotImplementedError()

        a1 = TestOp().make_node()
        g = grad_sources_inputs([(a1.outputs[0], one)], None)
        assert g[a1.inputs[0]] is gval

    def test_Nin_1out(self):
        # Test grad is called correctly for a many-to-1 op
        gval0 = scalar()
        gval1 = scalar()

        class TestOp(Op):
            __props__ = ()

            def make_node(self):
                inputs = [scalar(), scalar()]
                outputs = [matrix()]
                return Apply(self, inputs, outputs)

            def grad(self, inp, grads):
                x0, x1 = inp
                (gz,) = grads
                return (gval0, gval1)

            def perform(self, *args, **kwargs):
                raise NotImplementedError()

        a1 = TestOp().make_node()
        g = grad_sources_inputs([(a1.outputs[0], one)], None)
        assert g[a1.inputs[0]] is gval0
        assert g[a1.inputs[1]] is gval1

    def test_Nin_Nout(self):
        # Test grad is called correctly for a many-to-many op
        gval0 = matrix()
        gval1 = matrix()

        class TestOp(Op):
            __props__ = ()

            def make_node(self):
                inputs = [matrix(), matrix()]
                outputs = [matrix(), matrix()]
                return Apply(self, inputs, outputs)

            def grad(self, inp, grads):
                return gval0, gval1

            def perform(self, *args, **kwargs):
                raise NotImplementedError()

        a1 = TestOp().make_node()
        g = grad_sources_inputs([(a1.outputs[0], one)], None)
        assert g[a1.inputs[0]] is gval0
        assert g[a1.inputs[1]] is gval1


class TestGrad:
    def test_unimplemented_grad_func(self):
        # tests that function compilation catches unimplemented grads
        # in the graph

        a = vector()
        b = grad_not_implemented(tt.add, 0, a)
        with pytest.raises(TypeError):
            theano.function([a], b, on_unused_input="ignore")

    def test_undefined_grad_func(self):
        # tests that function compilation catches undefined grads in the graph
        a = vector()
        b = grad_undefined(tt.add, 0, a)
        with pytest.raises(TypeError):
            theano.function([a], b, on_unused_input="ignore")

    def test_unimplemented_grad_grad(self):
        # tests that unimplemented grads are caught in the grad method

        class DummyOp(Op):
            __props__ = ()

            def make_node(self, x):
                return Apply(self, [x], [x.type()])

            def grad(self, inputs, output_grads):
                return [grad_not_implemented(self, 0, inputs[0])]

            def perform(self, *args, **kwargs):
                raise NotImplementedError()

        a = scalar()
        b = DummyOp()(a)

        with pytest.raises(TypeError):
            grad(b, a)

    def test_undefined_grad_grad(self):
        # tests that undefined grads are caught in the grad method

        class DummyOp(Op):
            __props__ = ()

            def make_node(self, x):
                return Apply(self, [x], [x.type()])

            def grad(self, inputs, output_grads):
                return [grad_undefined(self, 0, inputs[0])]

            def perform(self, *args, **kwargs):
                raise NotImplementedError()

        a = scalar()
        b = DummyOp()(a)

        with pytest.raises(TypeError):
            grad(b, a)

    def test_grad_name(self):
        A = matrix("A")
        x = vector("x")
        f = tt.dot(x, tt.dot(A, x))
        f.name = "f"
        g = grad(f, x)
        assert g.name == "(df/dx)"

    def test_grad_duplicate_input(self):
        # test that the grad works when a variable
        # appears in more than one place in a node's input list

        def output(x):
            return x * x

        rng = np.random.RandomState([2012, 8, 28])

        vx = rng.randn(2)

        utt.verify_grad(output, [vx])

    def test_grad_quadratic(self):
        # test the gradient on a tiny graph

        def cost(x, A):
            return tt.dot(x, tt.dot(A, x))

        rng = np.random.RandomState([2012, 8, 28])

        vx = rng.randn(2)
        vA = rng.randn(2, 2)

        utt.verify_grad(cost, [vx, vA])

    def test_grad_quadratic_vector(self):
        # test the gradient on a small graph

        def output(x, A):
            return tt.dot(x * x, A)

        rng = np.random.RandomState([2012, 8, 28])

        vx = rng.randn(2)
        vA = rng.randn(2, 2)

        utt.verify_grad(output, [vx, vA])

    def test_grad_cubic(self):
        # test the gradient on a bigger graph

        def cost(x, A):
            return tt.dot(x * x, tt.dot(A, x))

        rng = np.random.RandomState([2012, 8, 28])

        vx = rng.randn(2)
        vA = rng.randn(2, 2)

        utt.verify_grad(cost, [vx, vA])

    def test_grad_grad_quadratic(self):
        # test the gradient on a graph constructed using the gradient

        def output(x, A):
            orig_cost = tt.dot(x, tt.dot(A, x))
            return grad(orig_cost, x)

        rng = np.random.RandomState([2012, 8, 28])

        vx = rng.randn(2)
        vA = rng.randn(2, 2)

        utt.verify_grad(output, [vx, vA])

    def test_grad_grad_cubic(self):
        # test the gradient on a bigger graph constructed using the gradient

        def output(x, A):
            orig_cost = tt.dot(x * x, tt.dot(A, x))
            return grad(orig_cost, x)

        rng = np.random.RandomState([2012, 8, 28])

        vx = rng.randn(2)
        vA = rng.randn(2, 2)

        utt.verify_grad(output, [vx, vA])

    def test_grad_int(self):
        # tests that the gradient with respect to an integer
        # is the same as the gradient with respect to a float

        W = matrix()
        b = vector()

        def make_grad_func(X):
            Z = tt.dot(X, W) + b
            H = theano.tensor.nnet.sigmoid(Z)
            cost = H.sum()
            g = grad(cost, X)
            return theano.function([X, W, b], g, on_unused_input="ignore")

        int_func = make_grad_func(imatrix())
        # we have to use float64 as the float type to get the results to match
        # using an integer for the input makes all the later functions use
        # float64
        float_func = make_grad_func(matrix(dtype="float64"))

        m = 5
        d = 3
        n = 4
        rng = np.random.RandomState([2012, 9, 5])

        int_type = imatrix().dtype
        float_type = "float64"

        X = np.cast[int_type](rng.randn(m, d) * 127.0)
        W = np.cast[W.dtype](rng.randn(d, n))
        b = np.cast[b.dtype](rng.randn(n))

        int_result = int_func(X, W, b)
        float_result = float_func(np.cast[float_type](X), W, b)

        assert np.allclose(int_result, float_result), (int_result, float_result)

    def test_grad_disconnected(self):
        # tests corner cases of gradient for shape and alloc

        x = vector(name="x")
        total = x.sum()
        total.name = "total"
        num_elements = x.shape[0]
        num_elements.name = "num_elements"
        silly_vector = tt.alloc(total / num_elements, num_elements)
        silly_vector.name = "silly_vector"
        cost = silly_vector.sum()
        cost.name = "cost"
        # note that cost simplifies to be the same as "total"
        g = grad(cost, x, add_names=False)
        # we still need to pass in x because it determines the shape of
        # the output
        f = theano.function([x], g)
        rng = np.random.RandomState([2012, 9, 5])
        x = np.cast[x.dtype](rng.randn(3))
        g = f(x)
        assert np.allclose(g, np.ones(x.shape, dtype=x.dtype))

    def test_disconnected_nan(self):
        # test that connection_pattern can prevent getting NaN

        # Op1 has two outputs, f and g
        # x is connected to f but not to g
        class Op1(Op):
            __props__ = ()

            def make_node(self, x):
                return Apply(self, inputs=[x], outputs=[x.type(), scalar()])

            def connection_pattern(self, node):
                return [[True, False]]

            def grad(self, inputs, output_grads):
                return [inputs[0].zeros_like()]

            def perform(self, *args, **kwargs):
                raise NotImplementedError()

        # Op2 has two inputs, f and g
        # Its gradient with respect to g is not defined
        class Op2(Op):
            __props__ = ()

            def make_node(self, f, g):
                return Apply(self, inputs=[f, g], outputs=[scalar()])

            def grad(self, inputs, output_grads):
                return [inputs[0].zeros_like(), NullType()()]

            def perform(self, *args, **kwargs):
                raise NotImplementedError()

        x = vector()
        f, g = Op1()(x)
        cost = Op2()(f, g)

        # cost is differentiable wrt x
        # but we can't tell that without using Op1's connection pattern
        # looking at the theano graph alone, g is an ancestor of cost
        # and has x as an ancestor, so we must compute its gradient

        g = grad(cost, x)

        # If we made it to here without an exception, then the
        # connection_pattern functionality worked correctly

    def test_downcast_dtype(self):
        # Test that the gradient of a cost wrt a float32 variable does not
        # get upcasted to float64.
        # x has dtype float32, regardless of the value of floatX
        x = fscalar("x")
        y = x * 2
        z = lscalar("z")

        c = y + z
        dc_dx, dc_dy, dc_dz, dc_dc = grad(c, [x, y, z, c])
        # The dtype of dc_dy and dc_dz can be either float32 or float64,
        # that might depend on floatX, but is not specified.
        assert dc_dc.dtype in ("float32", "float64")
        assert dc_dz.dtype in ("float32", "float64")
        assert dc_dy.dtype in ("float32", "float64")

        # When the output gradient of y is passed to op.grad, it should
        # be downcasted to float32, so dc_dx should also be float32
        assert dc_dx.dtype == "float32"

    def test_grad_constant(self):
        # Test that the gradient handles Constants and consider_constant variables
        # consistently

        x = scalar()
        y = scalar()
        z_x = x + y
        z_one = one + y
        g_x = grad(z_x, x, consider_constant=[x])
        g_one = grad(z_one, one)

        f = theano.function([x, y], [g_x, g_one])

        g_x, g_one = f(1, 0.5)

        if not np.allclose(g_x, g_one):
            raise AssertionError(
                "Gradient using consider constant is "
                + str(g_x)
                + " but gradient with respect to the same Constant is "
                + str(g_one)
            )


def test_known_grads():
    # Tests that the grad method with no known_grads
    # matches what happens if you put its own known_grads
    # in for each variable

    full_range = tt.arange(10)
    x = scalar("x")
    t = iscalar("t")
    ft = full_range[t]
    ft.name = "ft"
    coeffs = vector("c")
    ct = coeffs[t]
    ct.name = "ct"
    p = x ** ft
    p.name = "p"
    y = ct * p
    y.name = "y"
    cost = tt.sqr(y)
    cost.name = "cost"

    layers = [[cost], [y], [ct, p], [ct, x, ft], [coeffs, t, full_range, x]]

    inputs = [coeffs, t, x]

    rng = np.random.RandomState([2012, 11, 15])
    values = [rng.randn(10), rng.randint(10), rng.randn()]
    values = [np.cast[ipt.dtype](value) for ipt, value in zip(inputs, values)]

    true_grads = grad(cost, inputs, disconnected_inputs="ignore")
    true_grads = theano.function(inputs, true_grads)
    true_grads = true_grads(*values)

    for layer in layers:
        first = grad(cost, layer, disconnected_inputs="ignore")
        known = OrderedDict(zip(layer, first))
        full = grad(
            cost=None, known_grads=known, wrt=inputs, disconnected_inputs="ignore"
        )
        full = theano.function(inputs, full)
        full = full(*values)
        assert len(true_grads) == len(full)
        for a, b, var in zip(true_grads, full, inputs):
            assert np.allclose(a, b)


def test_dxdx():
    # Tests that the gradient of a scalar with respect to itself is 1
    # I use an integer in this case because people keep changing this
    # gradient to be 0 on integers but according to our interpretation
    # of the gradient as defined in the Op contract, it should be 1.
    # If you feel the need to change this unit test you are probably
    # modifying the Op contract and should definitely get the approval
    # of multiple people on theano-dev.

    x = iscalar()
    g = grad(x, x)

    g = g.eval({x: 12})

    assert np.allclose(g, 1.0)


def test_known_grads_integers():
    # Tests that known_grads works on integers

    x = iscalar()
    g_expected = scalar()

    g_grad = grad(cost=None, known_grads={x: g_expected}, wrt=x)

    f = theano.function([g_expected], g_grad)

    x = -3
    gv = np.cast[config.floatX](0.6)

    g_actual = f(gv)

    assert np.allclose(g_actual, gv)


def test_undefined_cost_grad():
    # Tests that if we say the cost is not differentiable via the
    # known_grads mechanism, it is treated as such by the rest of the
    # system.
    # This is so that Ops that are built around minigraphs like OpFromGraph
    # and scan can implement Op.grad by passing ograds to known_grads

    x = iscalar()
    y = iscalar()
    cost = x + y
    assert cost.dtype in discrete_dtypes
    with pytest.raises(NullTypeGradError):
        grad(cost, [x, y], known_grads={cost: NullType()()})


def test_disconnected_cost_grad():
    # Tests that if we say the cost is disconnected via the
    # known_grads mechanism, it is treated as such by the rest of the
    # system.
    # This is so that Ops that are built around minigraphs like OpFromGraph
    # and scan can implement Op.grad by passing ograds to known_grads

    x = iscalar()
    y = iscalar()
    cost = x + y
    assert cost.dtype in discrete_dtypes
    try:
        grad(
            cost,
            [x, y],
            known_grads={cost: DisconnectedType()()},
            disconnected_inputs="raise",
        )
    except DisconnectedInputError:
        return
    raise AssertionError("A disconnected gradient has been ignored.")


def test_subgraph_grad():
    # Tests that the grad method with no known_grads
    # matches what happens if you use successive subgraph_grads

    x = fvector("x")
    t = fvector("t")
    w1 = theano.shared(np.random.randn(3, 4))
    w2 = theano.shared(np.random.randn(4, 2))
    a1 = tt.tanh(tt.dot(x, w1))
    a2 = tt.tanh(tt.dot(a1, w2))
    cost2 = tt.sqr(a2 - t).sum()
    cost2 += tt.sqr(w2.sum())
    cost1 = tt.sqr(w1.sum())

    params = [[w2], [w1]]
    costs = [cost2, cost1]
    grad_ends = [[a1], [x]]

    inputs = [t, x]
    rng = np.random.RandomState([2012, 11, 15])
    values = [rng.randn(2), rng.randn(3)]
    values = [np.cast[ipt.dtype](value) for ipt, value in zip(inputs, values)]

    wrt = [w2, w1]
    cost = cost2 + cost1
    true_grads = grad(cost, wrt)
    true_grads = theano.function(inputs, true_grads)
    true_grads = true_grads(*values)
    next_grad = None
    param_grads = []
    for i in range(2):
        param_grad, next_grad = subgraph_grad(
            wrt=params[i], end=grad_ends[i], start=next_grad, cost=costs[i]
        )
        next_grad = OrderedDict(zip(grad_ends[i], next_grad))
        param_grads.extend(param_grad)

    pgrads = theano.function(inputs, param_grads)
    pgrads = pgrads(*values)

    for true_grad, pgrad in zip(true_grads, pgrads):
        assert np.sum(np.abs(true_grad - pgrad)) < 0.00001


class TestConsiderConstant:
    def setup_method(self):
        utt.seed_rng()
        self.rng = np.random.RandomState(seed=utt.fetch_seed())

    def test_op_removed(self):
        x = matrix("x")
        y = x * consider_constant(x)
        f = theano.function([x], y)
        # need to refer to theano.consider_constant_ here,
        # theano.consider_constant is a wrapper function!
        assert consider_constant_ not in [node.op for node in f.maker.fgraph.toposort()]

    def test_grad(self):
        a = np.asarray(self.rng.randn(5, 5), dtype=config.floatX)

        x = matrix("x")

        expressions_gradients = [
            (x * consider_constant(x), x),
            (x * consider_constant(tt.exp(x)), tt.exp(x)),
            (consider_constant(x), tt.constant(0.0)),
            (x ** 2 * consider_constant(x), 2 * x ** 2),
        ]

        for expr, expr_grad in expressions_gradients:
            g = grad(expr.sum(), x)
            # gradient according to theano
            f = theano.function([x], g, on_unused_input="ignore")
            # desired gradient
            f2 = theano.function([x], expr_grad, on_unused_input="ignore")

            assert np.allclose(f(a), f2(a))


class TestZeroGrad:
    def setup_method(self):
        utt.seed_rng()
        self.rng = np.random.RandomState(seed=utt.fetch_seed())

    def test_op_removed(self):
        x = matrix("x")
        y = x * zero_grad(x)
        f = theano.function([x], y)
        # need to refer to theano.zero_grad here,
        # theano.zero_grad is a wrapper function!
        assert zero_grad_ not in [node.op for node in f.maker.fgraph.toposort()]

    def test_grad(self):
        a = np.asarray(self.rng.randn(5, 5), dtype=config.floatX)

        x = matrix("x")

        expressions_gradients = [
            (x * zero_grad(x), x),
            (x * zero_grad(tt.exp(x)), tt.exp(x)),
            (zero_grad(x), tt.constant(0.0)),
            (x ** 2 * zero_grad(x), 2 * x ** 2),
        ]

        for expr, expr_grad in expressions_gradients:
            g = grad(expr.sum(), x)
            # gradient according to theano
            f = theano.function([x], g, on_unused_input="ignore")
            # desired gradient
            f2 = theano.function([x], expr_grad, on_unused_input="ignore")

            assert np.allclose(f(a), f2(a))

    def test_rop(self):
        x = vector()
        v = vector()
        y = zero_grad(x)

        rop = tt.Rop(y, x, v)
        f = theano.function([x, v], rop, on_unused_input="ignore")

        a = np.asarray(self.rng.randn(5), dtype=config.floatX)
        u = np.asarray(self.rng.randn(5), dtype=config.floatX)

        assert np.count_nonzero(f(a, u)) == 0


class TestDisconnectedGrad:
    def setup_method(self):
        utt.seed_rng()
        self.rng = np.random.RandomState(seed=utt.fetch_seed())

    def test_op_removed(self):
        x = matrix("x")
        y = x * disconnected_grad(x)
        f = theano.function([x], y)
        # need to refer to theano.disconnected_grad here,
        # theano.disconnected_grad is a wrapper function!
        assert disconnected_grad_ not in [node.op for node in f.maker.fgraph.toposort()]

    def test_grad(self):
        a = np.asarray(self.rng.randn(5, 5), dtype=config.floatX)

        x = matrix("x")

        expressions_gradients = [
            (x * disconnected_grad(x), x),
            (x * disconnected_grad(tt.exp(x)), tt.exp(x)),
            (x ** 2 * disconnected_grad(x), 2 * x ** 2),
        ]

        for expr, expr_grad in expressions_gradients:
            g = grad(expr.sum(), x)
            # gradient according to theano
            f = theano.function([x], g, on_unused_input="ignore")
            # desired gradient
            f2 = theano.function([x], expr_grad, on_unused_input="ignore")

            assert np.allclose(f(a), f2(a))

    def test_connection_pattern(self):
        x = matrix("x")
        y = disconnected_grad(x)

        connection_pattern = y.owner.op.connection_pattern(y.owner)
        assert connection_pattern == [[False]]

    def test_disconnected_paths(self):
        # Test that taking gradient going through a disconnected
        # path rasises an exception
        a = np.asarray(self.rng.randn(5, 5), dtype=config.floatX)

        x = matrix("x")

        # This MUST raise a DisconnectedInputError error.
        # This also rasies an additional warning from gradients.py.
        with pytest.raises(DisconnectedInputError):
            grad(disconnected_grad(x).sum(), x)

        # This MUST NOT raise a DisconnectedInputError error.
        y = grad((x + disconnected_grad(x)).sum(), x)

        a = matrix("a")
        b = matrix("b")
        y = a + disconnected_grad(b)
        # This MUST raise a DisconnectedInputError error.
        # This also rasies an additional warning from gradients.py.
        with pytest.raises(DisconnectedInputError):
            grad(y.sum(), b)

        # This MUST NOT raise a DisconnectedInputError error.
        grad(y.sum(), a)


def test_grad_clip():
    x = scalar()

    z = grad(grad_clip(x, -1, 1) ** 2, x)
    z2 = grad(x ** 2, x)

    f = theano.function([x], outputs=[z, z2])

    if config.mode != "FAST_COMPILE":
        topo = f.maker.fgraph.toposort()
        assert not any([isinstance(node.op, GradClip) for node in topo])
    out = f(2.0)
    assert np.allclose(out, (1, 4))
    assert not np.allclose(out[0], out[1])


def test_grad_scale():
    x = scalar()

    z = grad(grad_scale(x, 2) ** 2, x)
    z2 = grad(x ** 2, x)

    f = theano.function([x], outputs=[z, z2])

    if config.mode != "FAST_COMPILE":
        topo = f.maker.fgraph.toposort()
        assert not any([isinstance(node.op, GradScale) for node in topo])
    out = f(2.0)

    assert np.allclose(out, (8, 4))


@config.change_flags(compute_test_value="off")
def test_undefined_grad_opt():
    # Make sure that undefined grad get removed in optimized graph.
    random = MRG_RandomStream(np.random.randint(1, 2147462579))

    pvals = theano.shared(np.random.rand(10, 20).astype(config.floatX))
    pvals = pvals / pvals.sum(axis=1)
    pvals = zero_grad(pvals)

    samples = random.multinomial(pvals=pvals, n=1)
    samples = tt.cast(samples, pvals.dtype)
    samples = zero_grad(samples)

    cost = tt.sum(samples + pvals)
    grad_res = grad(cost, samples)

    f = theano.function([], grad_res)
    assert not any(
        [isinstance(node.op, UndefinedGrad) for node in f.maker.fgraph.apply_nodes]
    )
