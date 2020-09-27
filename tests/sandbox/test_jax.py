import pytest

import numpy as np

import theano
import theano.tensor as tt

jax = pytest.importorskip("jax")

from theano.gof.op import get_test_value  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def set_theano_flags():
    with theano.change_flags(cxx="", compute_test_value="warn"):
        yield


def compare_jax_and_py(fgraph, inputs, cmp_fn=np.allclose):
    # jax_mode = theano.compile.Mode(linker="jax")
    jax_mode = "JAX"
    theano_jax_fn = theano.function(fgraph.inputs, fgraph.outputs, mode=jax_mode)
    jax_res = theano_jax_fn(*inputs)

    if isinstance(jax_res, list):
        assert all(isinstance(res, jax.interpreters.xla.DeviceArray) for res in jax_res)
    else:
        assert isinstance(jax_res, jax.interpreters.xla.DeviceArray)

    py_mode = theano.compile.Mode(linker="py")
    theano_py_fn = theano.function(fgraph.inputs, fgraph.outputs, mode=py_mode)
    py_res = theano_py_fn(*inputs)

    assert cmp_fn(jax_res, py_res)

    return jax_res


def test_jax_Alloc():
    x = tt.alloc(0.0, 2, 3)
    x_fg = theano.gof.FunctionGraph([], [x])

    (jax_res,) = compare_jax_and_py(x_fg, [])

    assert jax_res.shape == (2, 3)

    x = tt.alloc(1.1, 2, 3)
    x_fg = theano.gof.FunctionGraph([], [x])

    compare_jax_and_py(x_fg, [])

    x = theano.tensor.basic.AllocEmpty("float32")(2, 3)
    x_fg = theano.gof.FunctionGraph([], [x])

    def compare_shape_dtype(x, y):
        (x,) = x
        (y,) = y
        return x.shape == y.shape and x.dtype == y.dtype

    (jax_res,) = compare_jax_and_py(x_fg, [], cmp_fn=compare_shape_dtype)

    a = tt.scalar("a")
    x = tt.alloc(a, 20)
    x_fg = theano.gof.FunctionGraph([a], [x])

    (jax_res,) = compare_jax_and_py(x_fg, [10.0])

    a = tt.vector("a")
    x = tt.alloc(a, 20, 10)
    x_fg = theano.gof.FunctionGraph([a], [x])

    (jax_res,) = compare_jax_and_py(x_fg, [np.ones(10, dtype=tt.config.floatX)])


def test_jax_compile_ops():
    x = theano.compile.ops.DeepCopyOp()(tt.as_tensor_variable(1.1))
    x_fg = theano.gof.FunctionGraph([], [x])

    (jax_res,) = compare_jax_and_py(x_fg, [])

    x_np = np.zeros((20, 3))
    x = theano.compile.ops.Shape()(tt.as_tensor_variable(x_np))
    x_fg = theano.gof.FunctionGraph([], [x])

    (jax_res,) = compare_jax_and_py(x_fg, [])

    x = theano.compile.ops.Shape_i(1)(tt.as_tensor_variable(x_np))
    x_fg = theano.gof.FunctionGraph([], [x])

    (jax_res,) = compare_jax_and_py(x_fg, [])

    x = theano.compile.ops.SpecifyShape()(tt.as_tensor_variable(x_np), (20, 3))
    x_fg = theano.gof.FunctionGraph([], [x])

    (jax_res,) = compare_jax_and_py(x_fg, [])

    with theano.change_flags(compute_test_value="off"):
        x = theano.compile.ops.SpecifyShape()(tt.as_tensor_variable(x_np), (2, 3))
        x_fg = theano.gof.FunctionGraph([], [x])

        with pytest.raises(AssertionError):
            (jax_res,) = compare_jax_and_py(x_fg, [])

    x_np = np.zeros((20, 1, 1))
    x = theano.compile.ops.Rebroadcast((0, False), (1, True), (2, False))(
        tt.as_tensor_variable(x_np)
    )
    x_fg = theano.gof.FunctionGraph([], [x])

    (jax_res,) = compare_jax_and_py(x_fg, [])

    with theano.change_flags(compute_test_value="off"):
        x = theano.compile.ops.Rebroadcast((0, True), (1, False), (2, False))(
            tt.as_tensor_variable(x_np)
        )
        x_fg = theano.gof.FunctionGraph([], [x])

        with pytest.raises(ValueError):
            (jax_res,) = compare_jax_and_py(x_fg, [])

    x = theano.compile.ops.ViewOp()(tt.as_tensor_variable(x_np))
    x_fg = theano.gof.FunctionGraph([], [x])

    (jax_res,) = compare_jax_and_py(x_fg, [])


def test_jax_basic():
    x = tt.matrix("x")
    y = tt.matrix("y")

    # `ScalarOp`
    z = tt.cosh(x ** 2 + y / 3.0)

    # `[Inc]Subtensor`
    out = tt.set_subtensor(z[0], -10.0)
    out = tt.inc_subtensor(out[0, 1], 2.0)
    out = out[:5, :3]

    out_fg = theano.gof.FunctionGraph([x, y], [out])

    test_input_vals = [
        np.tile(np.arange(10), (10, 1)).astype(tt.config.floatX),
        np.tile(np.arange(10, 20), (10, 1)).astype(tt.config.floatX),
    ]
    (jax_res,) = compare_jax_and_py(out_fg, test_input_vals)

    # Confirm that the `Subtensor` slice operations are correct
    assert jax_res.shape == (5, 3)

    # Confirm that the `IncSubtensor` operations are correct
    assert jax_res[0, 0] == -10.0
    assert jax_res[0, 1] == -8.0

    out = tt.clip(x, y, 5)
    out_fg = theano.gof.FunctionGraph([x, y], [out])
    (jax_res,) = compare_jax_and_py(out_fg, test_input_vals)

    out = tt.diagonal(x, 0)
    out_fg = theano.gof.FunctionGraph([x], [out])
    (jax_res,) = compare_jax_and_py(
        out_fg, [np.arange(10 * 10).reshape((10, 10)).astype(tt.config.floatX)]
    )


@pytest.mark.skip(reason="Not fully implemented, yet.")
def test_jax_scan():

    theano.config.compute_test_value = "raise"

    a_tt = tt.scalar("a")
    a_tt.tag.test_value = 3.0

    def input_step_fn(y_tm1, y_tm2, a):
        y_tm1.name = "y_tm1"
        y_tm2.name = "y_tm2"
        res = (y_tm1 + y_tm2) * a
        res.name = "y_t"
        return res

    y_scan_tt, _ = theano.scan(
        fn=input_step_fn,
        outputs_info=[
            {
                "initial": tt.as_tensor_variable(
                    np.r_[-1.0, 0.0].astype(tt.config.floatX)
                ),
                "taps": [-1, -2],
            },
        ],
        non_sequences=[a_tt],
        n_steps=10,
        name="y_scan",
    )
    y_scan_tt.name = "y"
    y_scan_tt.owner.inputs[0].name = "y_all"

    theano_scan_fn = theano.function([], y_scan_tt, givens={a_tt: 3.0})
    theano_res = theano_scan_fn()

    #
    # The equivalent JAX `scan`:
    #
    import jax
    import jax.numpy as jnp

    def jax_inner_scan(carry, x):
        (y_tm1, y_tm2), a = carry
        res = (y_tm1 + y_tm2) * a
        return [jnp.array([res, y_tm1]), a], res

    init_carry = [np.r_[0.0, -1.0].astype(tt.config.floatX), 3.0]
    tmp, jax_res = jax.lax.scan(jax_inner_scan, init_carry, None, length=10)

    assert np.allclose(jax_res, theano_res)

    out_fg = theano.gof.FunctionGraph([a_tt], [y_scan_tt])

    test_input_vals = [np.array(10.0).astype(tt.config.floatX)]
    (jax_res,) = compare_jax_and_py(out_fg, test_input_vals)

    assert False


def test_jax_Subtensors():
    # Basic indices
    x_tt = tt.arange(3 * 4 * 5).reshape((3, 4, 5))
    out_tt = x_tt[1, 2, 0]

    out_fg = theano.gof.FunctionGraph([], [out_tt])
    (jax_res,) = compare_jax_and_py(out_fg, [])

    out_tt = x_tt[1:2, 1, :]

    out_fg = theano.gof.FunctionGraph([], [out_tt])
    (jax_res,) = compare_jax_and_py(out_fg, [])

    # Boolean indices
    out_tt = x_tt[x_tt < 0]

    out_fg = theano.gof.FunctionGraph([], [out_tt])
    (jax_res,) = compare_jax_and_py(out_fg, [])

    # Advanced indexing
    out_tt = x_tt[[1, 2]]

    out_fg = theano.gof.FunctionGraph([], [out_tt])
    (jax_res,) = compare_jax_and_py(out_fg, [])

    out_tt = x_tt[[1, 2], [2, 3]]

    out_fg = theano.gof.FunctionGraph([], [out_tt])
    (jax_res,) = compare_jax_and_py(out_fg, [])

    # Advanced and basic indexing
    out_tt = x_tt[[1, 2], :]

    out_fg = theano.gof.FunctionGraph([], [out_tt])
    (jax_res,) = compare_jax_and_py(out_fg, [])

    out_tt = x_tt[[1, 2], :, [3, 4]]

    out_fg = theano.gof.FunctionGraph([], [out_tt])
    (jax_res,) = compare_jax_and_py(out_fg, [])


def test_jax_IncSubtensor():
    x_np = np.empty((3, 4, 5), dtype=tt.config.floatX)
    x_tt = tt.arange(3 * 4 * 5).reshape((3, 4, 5)).astype(tt.config.floatX)

    # "Set" basic indices
    st_tt = tt.as_tensor_variable(np.array(-10.0, dtype=tt.config.floatX))
    out_tt = tt.set_subtensor(x_tt[1, 2, 3], st_tt)
    out_fg = theano.gof.FunctionGraph([], [out_tt])
    (jax_res,) = compare_jax_and_py(out_fg, [])

    st_tt = tt.as_tensor_variable(np.r_[-1.0, 0.0].astype(tt.config.floatX))
    out_tt = tt.set_subtensor(x_tt[:2, 0, 0], st_tt)
    out_fg = theano.gof.FunctionGraph([], [out_tt])
    (jax_res,) = compare_jax_and_py(out_fg, [])

    out_tt = tt.set_subtensor(x_tt[0, 1:3, 0], st_tt)
    out_fg = theano.gof.FunctionGraph([], [out_tt])
    (jax_res,) = compare_jax_and_py(out_fg, [])

    # "Set" advanced indices
    st_tt = tt.as_tensor_variable(np.r_[-1.0, 0.0].astype(tt.config.floatX))
    out_tt = tt.set_subtensor(x_tt[[0, 2], 0, 0], st_tt)
    out_fg = theano.gof.FunctionGraph([], [out_tt])
    (jax_res,) = compare_jax_and_py(out_fg, [])

    st_tt = tt.as_tensor_variable(x_np[[0, 2], 0, :3])
    out_tt = tt.set_subtensor(x_tt[[0, 2], 0, :3], st_tt)
    out_fg = theano.gof.FunctionGraph([], [out_tt])
    (jax_res,) = compare_jax_and_py(out_fg, [])

    # "Set" boolean indices
    mask_tt = tt.as_tensor_variable(x_np) > 0
    out_tt = tt.set_subtensor(x_tt[mask_tt], 0.0)
    out_fg = theano.gof.FunctionGraph([], [out_tt])
    (jax_res,) = compare_jax_and_py(out_fg, [])

    # "Increment" basic indices
    st_tt = tt.as_tensor_variable(np.array(-10.0, dtype=tt.config.floatX))
    out_tt = tt.inc_subtensor(x_tt[1, 2, 3], st_tt)
    out_fg = theano.gof.FunctionGraph([], [out_tt])
    (jax_res,) = compare_jax_and_py(out_fg, [])

    st_tt = tt.as_tensor_variable(np.r_[-1.0, 0.0].astype(tt.config.floatX))
    out_tt = tt.inc_subtensor(x_tt[:2, 0, 0], st_tt)
    out_fg = theano.gof.FunctionGraph([], [out_tt])
    (jax_res,) = compare_jax_and_py(out_fg, [])

    out_tt = tt.set_subtensor(x_tt[0, 1:3, 0], st_tt)
    out_fg = theano.gof.FunctionGraph([], [out_tt])
    (jax_res,) = compare_jax_and_py(out_fg, [])

    # "Increment" advanced indices
    st_tt = tt.as_tensor_variable(np.r_[-1.0, 0.0].astype(tt.config.floatX))
    out_tt = tt.inc_subtensor(x_tt[[0, 2], 0, 0], st_tt)
    out_fg = theano.gof.FunctionGraph([], [out_tt])
    (jax_res,) = compare_jax_and_py(out_fg, [])

    st_tt = tt.as_tensor_variable(x_np[[0, 2], 0, :3])
    out_tt = tt.inc_subtensor(x_tt[[0, 2], 0, :3], st_tt)
    out_fg = theano.gof.FunctionGraph([], [out_tt])
    (jax_res,) = compare_jax_and_py(out_fg, [])

    # "Increment" boolean indices
    mask_tt = tt.as_tensor_variable(x_np) > 0
    out_tt = tt.set_subtensor(x_tt[mask_tt], 1.0)
    out_fg = theano.gof.FunctionGraph([], [out_tt])
    (jax_res,) = compare_jax_and_py(out_fg, [])


def test_jax_ifelse():

    true_vals = np.r_[1, 2, 3]
    false_vals = np.r_[-1, -2, -3]

    x = theano.ifelse.ifelse(np.array(True), true_vals, false_vals)
    x_fg = theano.gof.FunctionGraph([], [x])

    (jax_res,) = compare_jax_and_py(x_fg, [])

    x = theano.ifelse.ifelse(np.array(False), true_vals, false_vals)
    x_fg = theano.gof.FunctionGraph([], [x])

    (jax_res,) = compare_jax_and_py(x_fg, [])


def test_jax_CAReduce():
    a_tt = tt.vector("a")
    a_tt.tag.test_value = np.r_[1, 2, 3].astype(tt.config.floatX)

    x = tt.sum(a_tt, axis=None)
    x_fg = theano.gof.FunctionGraph([a_tt], [x])

    _ = compare_jax_and_py(x_fg, [np.r_[1, 2, 3].astype(tt.config.floatX)])

    a_tt = tt.matrix("a")
    a_tt.tag.test_value = np.c_[[1, 2, 3], [1, 2, 3]].astype(tt.config.floatX)

    x = tt.sum(a_tt, axis=0)
    x_fg = theano.gof.FunctionGraph([a_tt], [x])

    _ = compare_jax_and_py(x_fg, [np.c_[[1, 2, 3], [1, 2, 3]].astype(tt.config.floatX)])

    x = tt.sum(a_tt, axis=1)
    x_fg = theano.gof.FunctionGraph([a_tt], [x])

    _ = compare_jax_and_py(x_fg, [np.c_[[1, 2, 3], [1, 2, 3]].astype(tt.config.floatX)])

    a_tt = tt.matrix("a")
    a_tt.tag.test_value = np.c_[[1, 2, 3], [1, 2, 3]].astype(tt.config.floatX)

    x = tt.prod(a_tt, axis=0)
    x_fg = theano.gof.FunctionGraph([a_tt], [x])

    _ = compare_jax_and_py(x_fg, [np.c_[[1, 2, 3], [1, 2, 3]].astype(tt.config.floatX)])

    x = tt.all(a_tt)
    x_fg = theano.gof.FunctionGraph([a_tt], [x])

    _ = compare_jax_and_py(x_fg, [np.c_[[1, 2, 3], [1, 2, 3]].astype(tt.config.floatX)])


def test_jax_MakeVector():
    x = tt.opt.make_vector(1, 2, 3)
    x_fg = theano.gof.FunctionGraph([], [x])

    _ = compare_jax_and_py(x_fg, [])


def test_jax_Reshape():
    a_tt = tt.vector("a")
    x = tt.basic.reshape(a_tt, (2, 2))
    x_fg = theano.gof.FunctionGraph([a_tt], [x])

    _ = compare_jax_and_py(
        x_fg, [np.r_[1.0, 2.0, 3.0, 4.0].astype(theano.config.floatX)]
    )


def test_jax_Reshape_omnistaging():
    # Test breaking "omnistaging" changes in JAX.
    # See https://github.com/tensorflow/probability/commit/782d0c64eb774b9aac54a1c8488e4f1f96fbbc68
    a_tt = tt.vector("a")
    x = tt.basic.reshape(a_tt, (a_tt.shape[0] // 2, a_tt.shape[0] // 3))
    x_fg = theano.gof.FunctionGraph([a_tt], [x])

    _ = compare_jax_and_py(x_fg, [np.empty((6,)).astype(theano.config.floatX)])


def test_jax_Dimshuffle():
    a_tt = tt.matrix("a")

    x = a_tt.T
    x_fg = theano.gof.FunctionGraph([a_tt], [x])
    _ = compare_jax_and_py(
        x_fg, [np.c_[[1.0, 2.0], [3.0, 4.0]].astype(tt.config.floatX)]
    )

    x = a_tt.dimshuffle([0, 1, "x"])
    x_fg = theano.gof.FunctionGraph([a_tt], [x])
    _ = compare_jax_and_py(
        x_fg, [np.c_[[1.0, 2.0], [3.0, 4.0]].astype(tt.config.floatX)]
    )

    a_tt = tt.tensor(dtype=tt.config.floatX, broadcastable=[False, True])
    x = a_tt.dimshuffle((0,))
    x_fg = theano.gof.FunctionGraph([a_tt], [x])
    _ = compare_jax_and_py(x_fg, [np.c_[[1.0, 2.0, 3.0, 4.0]].astype(tt.config.floatX)])

    a_tt = tt.tensor(dtype=tt.config.floatX, broadcastable=[False, True])
    x = tt.elemwise.DimShuffle([False, True], (0,), inplace=True)(a_tt)
    x_fg = theano.gof.FunctionGraph([a_tt], [x])
    _ = compare_jax_and_py(x_fg, [np.c_[[1.0, 2.0, 3.0, 4.0]].astype(tt.config.floatX)])


def test_jax_variadic_Scalar():
    mu = tt.vector("mu", dtype=tt.config.floatX)
    mu.tag.test_value = np.r_[0.1, 1.1].astype(tt.config.floatX)
    tau = tt.vector("tau", dtype=tt.config.floatX)
    tau.tag.test_value = np.r_[1.0, 2.0].astype(tt.config.floatX)

    res = -tau * mu

    fgraph = theano.gof.FunctionGraph([mu, tau], [res])

    _ = compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    res = -tau * (tau - mu) ** 2

    fgraph = theano.gof.FunctionGraph([mu, tau], [res])

    _ = compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])


def test_jax_logp():

    mu = tt.vector("mu")
    mu.tag.test_value = np.r_[0.0, 0.0].astype(tt.config.floatX)
    tau = tt.vector("tau")
    tau.tag.test_value = np.r_[1.0, 1.0].astype(tt.config.floatX)
    sigma = tt.vector("sigma")
    sigma.tag.test_value = (1.0 / get_test_value(tau)).astype(tt.config.floatX)
    value = tt.vector("value")
    value.tag.test_value = np.r_[0.1, -10].astype(tt.config.floatX)

    logp = (-tau * (value - mu) ** 2 + tt.log(tau / np.pi / 2.0)) / 2.0
    conditions = [sigma > 0]
    alltrue = tt.all([tt.all(1 * val) for val in conditions])
    normal_logp = tt.switch(alltrue, logp, -np.inf)

    fgraph = theano.gof.FunctionGraph([mu, tau, sigma, value], [normal_logp])

    _ = compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])


def test_jax_multioutput():
    x = tt.vector("x")
    x.tag.test_value = np.r_[1.0, 2.0].astype(tt.config.floatX)
    y = tt.vector("y")
    y.tag.test_value = np.r_[3.0, 4.0].astype(tt.config.floatX)

    w = tt.cosh(x ** 2 + y / 3.0)
    v = tt.cosh(x / 3.0 + y ** 2)

    fgraph = theano.gof.FunctionGraph([x, y], [w, v])

    _ = compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])


def test_nnet():
    x = tt.vector("x")
    x.tag.test_value = np.r_[1.0, 2.0].astype(tt.config.floatX)

    out = tt.nnet.sigmoid(x)
    fgraph = theano.gof.FunctionGraph([x], [out])
    _ = compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    out = tt.nnet.ultra_fast_sigmoid(x)
    fgraph = theano.gof.FunctionGraph([x], [out])
    _ = compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    out = tt.nnet.softplus(x)
    fgraph = theano.gof.FunctionGraph([x], [out])
    _ = compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])


def test_tensor_basics():
    y = tt.vector("y")
    y.tag.test_value = np.r_[1.0, 2.0].astype(theano.config.floatX)
    x = tt.vector("x")
    x.tag.test_value = np.r_[3.0, 4.0].astype(theano.config.floatX)
    A = tt.matrix("A")
    A.tag.test_value = np.empty((2, 2), dtype=theano.config.floatX)
    alpha = tt.scalar("alpha")
    alpha.tag.test_value = np.array(3.0, dtype=theano.config.floatX)
    beta = tt.scalar("beta")
    beta.tag.test_value = np.array(5.0, dtype=theano.config.floatX)

    # This should be converted into a `Gemv` `Op` when the non-JAX compatible
    # optimizations are turned on; however, when using JAX mode, it should
    # leave the expression alone.
    out = y.dot(alpha * A).dot(x) + beta * y
    fgraph = theano.gof.FunctionGraph([y, x, A, alpha, beta], [out])
    _ = compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    out = tt.maximum(y, x)
    fgraph = theano.gof.FunctionGraph([y, x], [out])
    _ = compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    out = tt.max(y)
    fgraph = theano.gof.FunctionGraph([y], [out])
    _ = compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])


@pytest.mark.xfail(reason="jax.numpy.arange requires concrete inputs")
def test_arange():
    a = tt.scalar("a")
    a.tag.test_value = 10

    out = tt.arange(a)
    fgraph = theano.gof.FunctionGraph([a], [out])
    _ = compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])
