import numpy as np
import pytest

import theano
import theano.tensor as tt


jax = pytest.importorskip("jax")

from functools import partial  # noqa: E402

from theano.gof.op import get_test_value  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def set_theano_flags():
    with theano.change_flags(cxx="", compute_test_value="ignore"):
        yield


def compare_jax_and_py(
    fgraph,
    inputs,
    assert_fn=None,
    must_be_device_array=True,
):
    """Function to compare python graph output and jax compiled output for testing equality

    In the tests below computational graphs are defined in Theano. These graphs are then passed to
    this function which then compiles the graphs in both jax and python, runs the calculation
    in both and checks if the results are the same

    Parameters
    ----------
    fgraph: theano.gof.FunctionGraph
        Theano function Graph object
    inputs: iter
        Inputs for function graph
    assert_fn: func, opt
        Assert function used to check for equality between python and jax. If not
        provided uses np.testing.assert_allclose
    must_be_device_array: Bool
        Checks for instance of jax.interpreters.xla.DeviceArray. For testing purposes
        if this device array is found it indicates if the result was computed by jax

    Returns
    -------
    jax_res

    """
    if assert_fn is None:
        assert_fn = partial(np.testing.assert_allclose, rtol=1e-4)

    opts = theano.gof.Query(include=[None], exclude=["cxx_only", "BlasOpt"])
    jax_mode = theano.compile.mode.Mode(theano.sandbox.jax_linker.JAXLinker(), opts)
    py_mode = theano.compile.Mode("py", opts)

    theano_jax_fn = theano.function(fgraph.inputs, fgraph.outputs, mode=jax_mode)
    jax_res = theano_jax_fn(*inputs)

    if must_be_device_array:
        if isinstance(jax_res, list):
            assert all(
                isinstance(res, jax.interpreters.xla.DeviceArray) for res in jax_res
            )
        else:
            assert isinstance(jax_res, jax.interpreters.xla.DeviceArray)

    theano_py_fn = theano.function(fgraph.inputs, fgraph.outputs, mode=py_mode)
    py_res = theano_py_fn(*inputs)

    if len(fgraph.outputs) > 1:
        for j, p in zip(jax_res, py_res):
            assert_fn(j, p)
    else:
        assert_fn(jax_res, py_res)

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

    compare_jax_and_py(x_fg, [], assert_fn=compare_shape_dtype)

    a = tt.scalar("a")
    x = tt.alloc(a, 20)
    x_fg = theano.gof.FunctionGraph([a], [x])

    compare_jax_and_py(x_fg, [10.0])

    a = tt.vector("a")
    x = tt.alloc(a, 20, 10)
    x_fg = theano.gof.FunctionGraph([a], [x])

    compare_jax_and_py(x_fg, [np.ones(10, dtype=theano.config.floatX)])


def test_jax_compile_ops():
    x = theano.compile.ops.DeepCopyOp()(tt.as_tensor_variable(1.1))
    x_fg = theano.gof.FunctionGraph([], [x])

    compare_jax_and_py(x_fg, [])

    x_np = np.zeros((20, 3))
    x = theano.compile.ops.Shape()(tt.as_tensor_variable(x_np))
    x_fg = theano.gof.FunctionGraph([], [x])

    compare_jax_and_py(x_fg, [], must_be_device_array=False)

    x = theano.compile.ops.Shape_i(1)(tt.as_tensor_variable(x_np))
    x_fg = theano.gof.FunctionGraph([], [x])

    compare_jax_and_py(x_fg, [], must_be_device_array=False)

    x = theano.compile.ops.SpecifyShape()(tt.as_tensor_variable(x_np), (20, 3))
    x_fg = theano.gof.FunctionGraph([], [x])

    compare_jax_and_py(x_fg, [])

    with theano.change_flags(compute_test_value="off"):
        x = theano.compile.ops.SpecifyShape()(tt.as_tensor_variable(x_np), (2, 3))
        x_fg = theano.gof.FunctionGraph([], [x])

        with pytest.raises(AssertionError):
            compare_jax_and_py(x_fg, [])

    x_np = np.zeros((20, 1, 1))
    x = theano.compile.ops.Rebroadcast((0, False), (1, True), (2, False))(
        tt.as_tensor_variable(x_np)
    )
    x_fg = theano.gof.FunctionGraph([], [x])

    compare_jax_and_py(x_fg, [])

    with theano.change_flags(compute_test_value="off"):
        x = theano.compile.ops.Rebroadcast((0, True), (1, False), (2, False))(
            tt.as_tensor_variable(x_np)
        )
        x_fg = theano.gof.FunctionGraph([], [x])

        with pytest.raises(ValueError):
            compare_jax_and_py(x_fg, [])

    x = theano.compile.ops.ViewOp()(tt.as_tensor_variable(x_np))
    x_fg = theano.gof.FunctionGraph([], [x])

    compare_jax_and_py(x_fg, [])


def test_jax_basic():
    x = tt.matrix("x")
    y = tt.matrix("y")
    b = tt.vector("b")

    # `ScalarOp`
    z = tt.cosh(x ** 2 + y / 3.0)

    # `[Inc]Subtensor`
    out = tt.set_subtensor(z[0], -10.0)
    out = tt.inc_subtensor(out[0, 1], 2.0)
    out = out[:5, :3]

    out_fg = theano.gof.FunctionGraph([x, y], [out])

    test_input_vals = [
        np.tile(np.arange(10), (10, 1)).astype(theano.config.floatX),
        np.tile(np.arange(10, 20), (10, 1)).astype(theano.config.floatX),
    ]
    (jax_res,) = compare_jax_and_py(out_fg, test_input_vals)

    # Confirm that the `Subtensor` slice operations are correct
    assert jax_res.shape == (5, 3)

    # Confirm that the `IncSubtensor` operations are correct
    assert jax_res[0, 0] == -10.0
    assert jax_res[0, 1] == -8.0

    out = tt.clip(x, y, 5)
    out_fg = theano.gof.FunctionGraph([x, y], [out])
    compare_jax_and_py(out_fg, test_input_vals)

    out = tt.diagonal(x, 0)
    out_fg = theano.gof.FunctionGraph([x], [out])
    compare_jax_and_py(
        out_fg, [np.arange(10 * 10).reshape((10, 10)).astype(theano.config.floatX)]
    )

    out = tt.slinalg.cholesky(x)
    out_fg = theano.gof.FunctionGraph([x], [out])
    compare_jax_and_py(
        out_fg,
        [(np.eye(10) + np.random.randn(10, 10) * 0.01).astype(theano.config.floatX)],
    )

    # not sure why this isn't working yet with lower=False
    out = tt.slinalg.Cholesky(lower=False)(x)
    out_fg = theano.gof.FunctionGraph([x], [out])
    compare_jax_and_py(
        out_fg,
        [(np.eye(10) + np.random.randn(10, 10) * 0.01).astype(theano.config.floatX)],
    )

    out = tt.slinalg.solve(x, b)
    out_fg = theano.gof.FunctionGraph([x, b], [out])
    compare_jax_and_py(
        out_fg,
        [
            np.eye(10).astype(theano.config.floatX),
            np.arange(10).astype(theano.config.floatX),
        ],
    )

    out = tt.nlinalg.alloc_diag(b)
    out_fg = theano.gof.FunctionGraph([b], [out])
    compare_jax_and_py(out_fg, [np.arange(10).astype(theano.config.floatX)])

    out = tt.nlinalg.det(x)
    out_fg = theano.gof.FunctionGraph([x], [out])
    compare_jax_and_py(
        out_fg, [np.arange(10 * 10).reshape((10, 10)).astype(theano.config.floatX)]
    )

    out = tt.nlinalg.matrix_inverse(x)
    out_fg = theano.gof.FunctionGraph([x], [out])
    compare_jax_and_py(
        out_fg,
        [(np.eye(10) + np.random.randn(10, 10) * 0.01).astype(theano.config.floatX)],
    )


def test_jax_eye():
    """Tests jaxification of the Eye operator"""
    out = tt.eye(3)
    out_fg = theano.gof.FunctionGraph([], [out])

    compare_jax_and_py(out_fg, [])


def test_jax_basic_multiout():

    np.random.seed(213234)
    M = np.random.normal(size=(3, 3))
    X = M.dot(M.T)

    x = tt.matrix("x")

    outs = tt.nlinalg.eig(x)
    out_fg = theano.gof.FunctionGraph([x], outs)

    def assert_fn(x, y):
        np.testing.assert_allclose(x.astype(theano.config.floatX), y, rtol=1e-3)

    compare_jax_and_py(out_fg, [X.astype(theano.config.floatX)], assert_fn=assert_fn)

    outs = tt.nlinalg.eigh(x)
    out_fg = theano.gof.FunctionGraph([x], outs)
    compare_jax_and_py(out_fg, [X.astype(theano.config.floatX)], assert_fn=assert_fn)

    outs = tt.nlinalg.qr(x, mode="full")
    out_fg = theano.gof.FunctionGraph([x], outs)
    compare_jax_and_py(out_fg, [X.astype(theano.config.floatX)], assert_fn=assert_fn)

    outs = tt.nlinalg.qr(x, mode="reduced")
    out_fg = theano.gof.FunctionGraph([x], outs)
    compare_jax_and_py(out_fg, [X.astype(theano.config.floatX)], assert_fn=assert_fn)

    outs = tt.nlinalg.svd(x)
    out_fg = theano.gof.FunctionGraph([x], outs)
    compare_jax_and_py(out_fg, [X.astype(theano.config.floatX)], assert_fn=assert_fn)

    # Test that a single output of a multi-output `Op` can be used as input to
    # another `Op`
    x = tt.dvector()
    mx, amx = theano.tensor.MaxAndArgmax([0])(x)
    out = mx * amx
    out_fg = theano.gof.FunctionGraph([x], [out])
    compare_jax_and_py(out_fg, [np.r_[1, 2]])


def test_jax_scan_multiple_output():
    """Test a scan implementation of a SEIR model.

    SEIR model definition:
    S[t+1] = S[t] - B[t]
    E[t+1] = E[t] +B[t] - C[t]
    I[t+1] = I[t+1] + C[t] - D[t]

    B[t] ~ Binom(S[t], beta)
    C[t] ~ Binom(E[t], gamma)
    D[t] ~ Binom(I[t], delta)
    """

    def binomln(n, k):
        return tt.gammaln(n + 1) - tt.gammaln(k + 1) - tt.gammaln(n - k + 1)

    def binom_log_prob(n, p, value):
        return binomln(n, value) + value * tt.log(p) + (n - value) * tt.log(1 - p)

    # sequences
    tt_C = tt.ivector("C_t")
    tt_D = tt.ivector("D_t")
    # outputs_info (initial conditions)
    st0 = tt.lscalar("s_t0")
    et0 = tt.lscalar("e_t0")
    it0 = tt.lscalar("i_t0")
    logp_c = tt.scalar("logp_c")
    logp_d = tt.scalar("logp_d")
    # non_sequences
    beta = tt.scalar("beta")
    gamma = tt.scalar("gamma")
    delta = tt.scalar("delta")

    # TODO: Use random streams when their JAX conversions are implemented.
    # trng = tt.shared_randomstreams.RandomStreams(1234)

    def seir_one_step(ct0, dt0, st0, et0, it0, logp_c, logp_d, beta, gamma, delta):
        # bt0 = trng.binomial(n=st0, p=beta)
        bt0 = st0 * beta
        bt0 = bt0.astype(st0.dtype)

        logp_c1 = binom_log_prob(et0, gamma, ct0).astype(logp_c.dtype)
        logp_d1 = binom_log_prob(it0, delta, dt0).astype(logp_d.dtype)

        st1 = st0 - bt0
        et1 = et0 + bt0 - ct0
        it1 = it0 + ct0 - dt0
        return st1, et1, it1, logp_c1, logp_d1

    (st, et, it, logp_c_all, logp_d_all), _ = theano.scan(
        fn=seir_one_step,
        sequences=[tt_C, tt_D],
        outputs_info=[st0, et0, it0, logp_c, logp_d],
        non_sequences=[beta, gamma, delta],
    )
    st.name = "S_t"
    et.name = "E_t"
    it.name = "I_t"
    logp_c_all.name = "C_t_logp"
    logp_d_all.name = "D_t_logp"

    out_fg = theano.gof.FunctionGraph(
        [tt_C, tt_D, st0, et0, it0, logp_c, logp_d, beta, gamma, delta],
        [st, et, it, logp_c_all, logp_d_all],
    )

    s0, e0, i0 = 100, 50, 25
    logp_c0 = np.array(0.0, dtype=theano.config.floatX)
    logp_d0 = np.array(0.0, dtype=theano.config.floatX)
    beta_val, gamma_val, delta_val = [
        np.array(val, dtype=theano.config.floatX)
        for val in [0.277792, 0.135330, 0.108753]
    ]
    C = np.array([3, 5, 8, 13, 21, 26, 10, 3], dtype=np.int32)
    D = np.array([1, 2, 3, 7, 9, 11, 5, 1], dtype=np.int32)

    test_input_vals = [
        C,
        D,
        s0,
        e0,
        i0,
        logp_c0,
        logp_d0,
        beta_val,
        gamma_val,
        delta_val,
    ]
    compare_jax_and_py(out_fg, test_input_vals)


def test_jax_scan_tap_output():

    a_tt = tt.scalar("a")

    def input_step_fn(y_tm1, y_tm3, a):
        y_tm1.name = "y_tm1"
        y_tm3.name = "y_tm3"
        res = (y_tm1 + y_tm3) * a
        res.name = "y_t"
        return res

    y_scan_tt, _ = theano.scan(
        fn=input_step_fn,
        outputs_info=[
            {
                "initial": tt.as_tensor_variable(
                    np.r_[-1.0, 1.3, 0.0].astype(theano.config.floatX)
                ),
                "taps": [-1, -3],
            },
        ],
        non_sequences=[a_tt],
        n_steps=10,
        name="y_scan",
    )
    y_scan_tt.name = "y"
    y_scan_tt.owner.inputs[0].name = "y_all"

    out_fg = theano.gof.FunctionGraph([a_tt], [y_scan_tt])

    test_input_vals = [np.array(10.0).astype(theano.config.floatX)]
    compare_jax_and_py(out_fg, test_input_vals)


def test_jax_Subtensors():
    # Basic indices
    x_tt = tt.arange(3 * 4 * 5).reshape((3, 4, 5))
    out_tt = x_tt[1, 2, 0]
    assert isinstance(out_tt.owner.op, tt.subtensor.Subtensor)
    out_fg = theano.gof.FunctionGraph([], [out_tt])
    compare_jax_and_py(out_fg, [])

    out_tt = x_tt[1:2, 1, :]
    assert isinstance(out_tt.owner.op, tt.subtensor.Subtensor)
    out_fg = theano.gof.FunctionGraph([], [out_tt])
    compare_jax_and_py(out_fg, [])

    # Boolean indices
    out_tt = x_tt[x_tt < 0]
    assert isinstance(out_tt.owner.op, tt.subtensor.AdvancedSubtensor)
    out_fg = theano.gof.FunctionGraph([], [out_tt])
    compare_jax_and_py(out_fg, [])

    # Advanced indexing
    out_tt = x_tt[[1, 2]]
    assert isinstance(out_tt.owner.op, tt.subtensor.AdvancedSubtensor1)
    out_fg = theano.gof.FunctionGraph([], [out_tt])
    compare_jax_and_py(out_fg, [])

    out_tt = x_tt[[1, 2], [2, 3]]
    assert isinstance(out_tt.owner.op, tt.subtensor.AdvancedSubtensor)
    out_fg = theano.gof.FunctionGraph([], [out_tt])
    compare_jax_and_py(out_fg, [])

    # Advanced and basic indexing
    out_tt = x_tt[[1, 2], :]
    assert isinstance(out_tt.owner.op, tt.subtensor.AdvancedSubtensor1)
    out_fg = theano.gof.FunctionGraph([], [out_tt])
    compare_jax_and_py(out_fg, [])

    out_tt = x_tt[[1, 2], :, [3, 4]]
    assert isinstance(out_tt.owner.op, tt.subtensor.AdvancedSubtensor)
    out_fg = theano.gof.FunctionGraph([], [out_tt])
    compare_jax_and_py(out_fg, [])


def test_jax_IncSubtensor():
    x_np = np.random.uniform(-1, 1, size=(3, 4, 5)).astype(theano.config.floatX)
    x_tt = tt.arange(3 * 4 * 5).reshape((3, 4, 5)).astype(theano.config.floatX)

    # "Set" basic indices
    st_tt = tt.as_tensor_variable(np.array(-10.0, dtype=theano.config.floatX))
    out_tt = tt.set_subtensor(x_tt[1, 2, 3], st_tt)
    assert isinstance(out_tt.owner.op, tt.subtensor.IncSubtensor)
    out_fg = theano.gof.FunctionGraph([], [out_tt])
    compare_jax_and_py(out_fg, [])

    st_tt = tt.as_tensor_variable(np.r_[-1.0, 0.0].astype(theano.config.floatX))
    out_tt = tt.set_subtensor(x_tt[:2, 0, 0], st_tt)
    assert isinstance(out_tt.owner.op, tt.subtensor.IncSubtensor)
    out_fg = theano.gof.FunctionGraph([], [out_tt])
    compare_jax_and_py(out_fg, [])

    out_tt = tt.set_subtensor(x_tt[0, 1:3, 0], st_tt)
    assert isinstance(out_tt.owner.op, tt.subtensor.IncSubtensor)
    out_fg = theano.gof.FunctionGraph([], [out_tt])
    compare_jax_and_py(out_fg, [])

    # "Set" advanced indices
    st_tt = tt.as_tensor_variable(
        np.random.uniform(-1, 1, size=(2, 4, 5)).astype(theano.config.floatX)
    )
    out_tt = tt.set_subtensor(x_tt[np.r_[0, 2]], st_tt)
    assert isinstance(out_tt.owner.op, tt.subtensor.AdvancedIncSubtensor1)
    out_fg = theano.gof.FunctionGraph([], [out_tt])
    compare_jax_and_py(out_fg, [])

    st_tt = tt.as_tensor_variable(np.r_[-1.0, 0.0].astype(theano.config.floatX))
    out_tt = tt.set_subtensor(x_tt[[0, 2], 0, 0], st_tt)
    assert isinstance(out_tt.owner.op, tt.subtensor.AdvancedIncSubtensor)
    out_fg = theano.gof.FunctionGraph([], [out_tt])
    compare_jax_and_py(out_fg, [])

    st_tt = tt.as_tensor_variable(x_np[[0, 2], 0, :3])
    out_tt = tt.set_subtensor(x_tt[[0, 2], 0, :3], st_tt)
    assert isinstance(out_tt.owner.op, tt.subtensor.AdvancedIncSubtensor)
    out_fg = theano.gof.FunctionGraph([], [out_tt])
    compare_jax_and_py(out_fg, [])

    # "Set" boolean indices
    mask_tt = tt.as_tensor_variable(x_np) > 0
    out_tt = tt.set_subtensor(x_tt[mask_tt], 0.0)
    assert isinstance(out_tt.owner.op, tt.subtensor.AdvancedIncSubtensor)
    out_fg = theano.gof.FunctionGraph([], [out_tt])
    compare_jax_and_py(out_fg, [])

    # "Increment" basic indices
    st_tt = tt.as_tensor_variable(np.array(-10.0, dtype=theano.config.floatX))
    out_tt = tt.inc_subtensor(x_tt[1, 2, 3], st_tt)
    assert isinstance(out_tt.owner.op, tt.subtensor.IncSubtensor)
    out_fg = theano.gof.FunctionGraph([], [out_tt])
    compare_jax_and_py(out_fg, [])

    st_tt = tt.as_tensor_variable(np.r_[-1.0, 0.0].astype(theano.config.floatX))
    out_tt = tt.inc_subtensor(x_tt[:2, 0, 0], st_tt)
    assert isinstance(out_tt.owner.op, tt.subtensor.IncSubtensor)
    out_fg = theano.gof.FunctionGraph([], [out_tt])
    compare_jax_and_py(out_fg, [])

    out_tt = tt.set_subtensor(x_tt[0, 1:3, 0], st_tt)
    assert isinstance(out_tt.owner.op, tt.subtensor.IncSubtensor)
    out_fg = theano.gof.FunctionGraph([], [out_tt])
    compare_jax_and_py(out_fg, [])

    # "Increment" advanced indices
    st_tt = tt.as_tensor_variable(
        np.random.uniform(-1, 1, size=(2, 4, 5)).astype(theano.config.floatX)
    )
    out_tt = tt.inc_subtensor(x_tt[np.r_[0, 2]], st_tt)
    assert isinstance(out_tt.owner.op, tt.subtensor.AdvancedIncSubtensor1)
    out_fg = theano.gof.FunctionGraph([], [out_tt])
    compare_jax_and_py(out_fg, [])

    st_tt = tt.as_tensor_variable(np.r_[-1.0, 0.0].astype(theano.config.floatX))
    out_tt = tt.inc_subtensor(x_tt[[0, 2], 0, 0], st_tt)
    assert isinstance(out_tt.owner.op, tt.subtensor.AdvancedIncSubtensor)
    out_fg = theano.gof.FunctionGraph([], [out_tt])
    compare_jax_and_py(out_fg, [])

    st_tt = tt.as_tensor_variable(x_np[[0, 2], 0, :3])
    out_tt = tt.inc_subtensor(x_tt[[0, 2], 0, :3], st_tt)
    assert isinstance(out_tt.owner.op, tt.subtensor.AdvancedIncSubtensor)
    out_fg = theano.gof.FunctionGraph([], [out_tt])
    compare_jax_and_py(out_fg, [])

    # "Increment" boolean indices
    mask_tt = tt.as_tensor_variable(x_np) > 0
    out_tt = tt.set_subtensor(x_tt[mask_tt], 1.0)
    assert isinstance(out_tt.owner.op, tt.subtensor.AdvancedIncSubtensor)
    out_fg = theano.gof.FunctionGraph([], [out_tt])
    compare_jax_and_py(out_fg, [])


def test_jax_ifelse():

    import theano.ifelse

    true_vals = np.r_[1, 2, 3]
    false_vals = np.r_[-1, -2, -3]

    x = theano.ifelse.ifelse(np.array(True), true_vals, false_vals)
    x_fg = theano.gof.FunctionGraph([], [x])

    compare_jax_and_py(x_fg, [])

    x = theano.ifelse.ifelse(np.array(False), true_vals, false_vals)
    x_fg = theano.gof.FunctionGraph([], [x])

    compare_jax_and_py(x_fg, [])


def test_jax_CAReduce():
    a_tt = tt.vector("a")
    a_tt.tag.test_value = np.r_[1, 2, 3].astype(theano.config.floatX)

    x = tt.sum(a_tt, axis=None)
    x_fg = theano.gof.FunctionGraph([a_tt], [x])

    compare_jax_and_py(x_fg, [np.r_[1, 2, 3].astype(theano.config.floatX)])

    a_tt = tt.matrix("a")
    a_tt.tag.test_value = np.c_[[1, 2, 3], [1, 2, 3]].astype(theano.config.floatX)

    x = tt.sum(a_tt, axis=0)
    x_fg = theano.gof.FunctionGraph([a_tt], [x])

    compare_jax_and_py(x_fg, [np.c_[[1, 2, 3], [1, 2, 3]].astype(theano.config.floatX)])

    x = tt.sum(a_tt, axis=1)
    x_fg = theano.gof.FunctionGraph([a_tt], [x])

    compare_jax_and_py(x_fg, [np.c_[[1, 2, 3], [1, 2, 3]].astype(theano.config.floatX)])

    a_tt = tt.matrix("a")
    a_tt.tag.test_value = np.c_[[1, 2, 3], [1, 2, 3]].astype(theano.config.floatX)

    x = tt.prod(a_tt, axis=0)
    x_fg = theano.gof.FunctionGraph([a_tt], [x])

    compare_jax_and_py(x_fg, [np.c_[[1, 2, 3], [1, 2, 3]].astype(theano.config.floatX)])

    x = tt.all(a_tt)
    x_fg = theano.gof.FunctionGraph([a_tt], [x])

    compare_jax_and_py(x_fg, [np.c_[[1, 2, 3], [1, 2, 3]].astype(theano.config.floatX)])


def test_jax_MakeVector():
    x = tt.opt.make_vector(1, 2, 3)
    x_fg = theano.gof.FunctionGraph([], [x])

    compare_jax_and_py(x_fg, [])


def test_jax_Reshape():
    a = tt.vector("a")
    x = tt.basic.reshape(a, (2, 2))
    x_fg = theano.gof.FunctionGraph([a], [x])
    compare_jax_and_py(x_fg, [np.r_[1.0, 2.0, 3.0, 4.0].astype(theano.config.floatX)])

    # Test breaking "omnistaging" changes in JAX.
    # See https://github.com/tensorflow/probability/commit/782d0c64eb774b9aac54a1c8488e4f1f96fbbc68
    x = tt.basic.reshape(a, (a.shape[0] // 2, a.shape[0] // 2))
    x_fg = theano.gof.FunctionGraph([a], [x])
    compare_jax_and_py(x_fg, [np.r_[1.0, 2.0, 3.0, 4.0].astype(theano.config.floatX)])


@pytest.mark.xfail(reason="jax.numpy.arange requires concrete inputs")
def test_jax_Reshape_nonconcrete():
    a = tt.vector("a")
    b = tt.iscalar("b")
    x = tt.basic.reshape(a, (b, b))
    x_fg = theano.gof.FunctionGraph([a, b], [x])
    compare_jax_and_py(
        x_fg, [np.r_[1.0, 2.0, 3.0, 4.0].astype(theano.config.floatX), 2]
    )


def test_jax_Dimshuffle():
    a_tt = tt.matrix("a")

    x = a_tt.T
    x_fg = theano.gof.FunctionGraph([a_tt], [x])
    compare_jax_and_py(
        x_fg, [np.c_[[1.0, 2.0], [3.0, 4.0]].astype(theano.config.floatX)]
    )

    x = a_tt.dimshuffle([0, 1, "x"])
    x_fg = theano.gof.FunctionGraph([a_tt], [x])
    compare_jax_and_py(
        x_fg, [np.c_[[1.0, 2.0], [3.0, 4.0]].astype(theano.config.floatX)]
    )

    a_tt = tt.tensor(dtype=theano.config.floatX, broadcastable=[False, True])
    x = a_tt.dimshuffle((0,))
    x_fg = theano.gof.FunctionGraph([a_tt], [x])
    compare_jax_and_py(x_fg, [np.c_[[1.0, 2.0, 3.0, 4.0]].astype(theano.config.floatX)])

    a_tt = tt.tensor(dtype=theano.config.floatX, broadcastable=[False, True])
    x = tt.elemwise.DimShuffle([False, True], (0,), inplace=True)(a_tt)
    x_fg = theano.gof.FunctionGraph([a_tt], [x])
    compare_jax_and_py(x_fg, [np.c_[[1.0, 2.0, 3.0, 4.0]].astype(theano.config.floatX)])


def test_jax_variadic_Scalar():
    mu = tt.vector("mu", dtype=theano.config.floatX)
    mu.tag.test_value = np.r_[0.1, 1.1].astype(theano.config.floatX)
    tau = tt.vector("tau", dtype=theano.config.floatX)
    tau.tag.test_value = np.r_[1.0, 2.0].astype(theano.config.floatX)

    res = -tau * mu

    fgraph = theano.gof.FunctionGraph([mu, tau], [res])

    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    res = -tau * (tau - mu) ** 2

    fgraph = theano.gof.FunctionGraph([mu, tau], [res])

    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])


def test_jax_logp():

    mu = tt.vector("mu")
    mu.tag.test_value = np.r_[0.0, 0.0].astype(theano.config.floatX)
    tau = tt.vector("tau")
    tau.tag.test_value = np.r_[1.0, 1.0].astype(theano.config.floatX)
    sigma = tt.vector("sigma")
    sigma.tag.test_value = (1.0 / get_test_value(tau)).astype(theano.config.floatX)
    value = tt.vector("value")
    value.tag.test_value = np.r_[0.1, -10].astype(theano.config.floatX)

    logp = (-tau * (value - mu) ** 2 + tt.log(tau / np.pi / 2.0)) / 2.0
    conditions = [sigma > 0]
    alltrue = tt.all([tt.all(1 * val) for val in conditions])
    normal_logp = tt.switch(alltrue, logp, -np.inf)

    fgraph = theano.gof.FunctionGraph([mu, tau, sigma, value], [normal_logp])

    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])


def test_jax_multioutput():
    x = tt.vector("x")
    x.tag.test_value = np.r_[1.0, 2.0].astype(theano.config.floatX)
    y = tt.vector("y")
    y.tag.test_value = np.r_[3.0, 4.0].astype(theano.config.floatX)

    w = tt.cosh(x ** 2 + y / 3.0)
    v = tt.cosh(x / 3.0 + y ** 2)

    fgraph = theano.gof.FunctionGraph([x, y], [w, v])

    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])


def test_nnet():
    x = tt.vector("x")
    x.tag.test_value = np.r_[1.0, 2.0].astype(theano.config.floatX)

    out = tt.nnet.sigmoid(x)
    fgraph = theano.gof.FunctionGraph([x], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    out = tt.nnet.ultra_fast_sigmoid(x)
    fgraph = theano.gof.FunctionGraph([x], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    out = tt.nnet.softplus(x)
    fgraph = theano.gof.FunctionGraph([x], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])


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
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    out = tt.maximum(y, x)
    fgraph = theano.gof.FunctionGraph([y, x], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    out = tt.max(y)
    fgraph = theano.gof.FunctionGraph([y], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])


@pytest.mark.xfail(reason="jax.numpy.arange requires concrete inputs")
def test_arange_nonconcrete():

    a = tt.scalar("a")
    a.tag.test_value = 10

    out = tt.arange(a)
    fgraph = theano.gof.FunctionGraph([a], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])


@pytest.mark.xfail(reason="jax.numpy.arange requires concrete inputs")
def test_unique_nonconcrete():
    a = tt.matrix("a")
    a.tag.test_value = np.arange(6, dtype=theano.config.floatX).reshape((3, 2))

    out = tt.extra_ops.Unique()(a)
    fgraph = theano.gof.FunctionGraph([a], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])


def test_identity():
    a = tt.scalar("a")
    a.tag.test_value = 10

    out = theano.scalar.basic.identity(a)
    fgraph = theano.gof.FunctionGraph([a], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])


def test_second():
    a0 = tt.scalar("a0")
    b = tt.scalar("b")

    out = theano.scalar.basic.second(a0, b)
    fgraph = theano.gof.FunctionGraph([a0, b], [out])
    compare_jax_and_py(fgraph, [10.0, 5.0])

    a1 = tt.vector("a1")
    out = tt.second(a1, b)
    fgraph = theano.gof.FunctionGraph([a1, b], [out])
    compare_jax_and_py(fgraph, [np.zeros([5], dtype=theano.config.floatX), 5.0])


def test_shared():
    a = theano.shared(np.array([1, 2, 3], dtype=theano.config.floatX))

    theano_jax_fn = theano.function([], a, mode="JAX")
    jax_res = theano_jax_fn()

    assert isinstance(jax_res, jax.interpreters.xla.DeviceArray)
    np.testing.assert_allclose(jax_res, a.get_value())

    theano_jax_fn = theano.function([], a * 2, mode="JAX")
    jax_res = theano_jax_fn()

    assert isinstance(jax_res, jax.interpreters.xla.DeviceArray)
    np.testing.assert_allclose(jax_res, a.get_value() * 2)

    # Changed the shared value and make sure that the JAX-compiled
    # function also changes.
    new_a_value = np.array([3, 4, 5], dtype=theano.config.floatX)
    a.set_value(new_a_value)

    jax_res = theano_jax_fn()
    assert isinstance(jax_res, jax.interpreters.xla.DeviceArray)
    np.testing.assert_allclose(jax_res, new_a_value * 2)


def test_extra_ops():
    a = tt.matrix("a")
    a.tag.test_value = np.arange(6, dtype=theano.config.floatX).reshape((3, 2))

    out = tt.extra_ops.cumsum(a, axis=0)
    fgraph = theano.gof.FunctionGraph([a], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    out = tt.extra_ops.cumprod(a, axis=1)
    fgraph = theano.gof.FunctionGraph([a], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    out = tt.extra_ops.diff(a, n=2, axis=1)
    fgraph = theano.gof.FunctionGraph([a], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    out = tt.extra_ops.repeat(a, (3, 3), axis=1)
    fgraph = theano.gof.FunctionGraph([a], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    # This function also cannot take symbolic input.
    c = tt.as_tensor(5)
    out = tt.extra_ops.bartlett(c)
    fgraph = theano.gof.FunctionGraph([], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    with pytest.raises(NotImplementedError):
        out = tt.extra_ops.fill_diagonal(a, c)
        fgraph = theano.gof.FunctionGraph([a], [out])
        compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    with pytest.raises(NotImplementedError):
        out = tt.extra_ops.fill_diagonal_offset(a, c, c)
        fgraph = theano.gof.FunctionGraph([a], [out])
        compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    with pytest.raises(NotImplementedError):
        out = tt.extra_ops.Unique(axis=1)(a)
        fgraph = theano.gof.FunctionGraph([a], [out])
        compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    indices = np.arange(np.product((3, 4)))
    out = tt.extra_ops.unravel_index(indices, (3, 4), order="C")
    fgraph = theano.gof.FunctionGraph([], out)
    compare_jax_and_py(
        fgraph, [get_test_value(i) for i in fgraph.inputs], must_be_device_array=False
    )

    multi_index = np.unravel_index(np.arange(np.product((3, 4))), (3, 4))
    out = tt.extra_ops.ravel_multi_index(multi_index, (3, 4))
    fgraph = theano.gof.FunctionGraph([], [out])
    compare_jax_and_py(
        fgraph, [get_test_value(i) for i in fgraph.inputs], must_be_device_array=False
    )

    # The inputs are "concrete", yet it still has problems?
    out = tt.extra_ops.Unique()(
        tt.as_tensor(np.arange(6, dtype=theano.config.floatX).reshape((3, 2)))
    )
    fgraph = theano.gof.FunctionGraph([], [out])
    compare_jax_and_py(fgraph, [])
