import numpy as np
import pytest
from packaging.version import parse as version_parse

import aesara.tensor as at
from aesara import function, grad
from aesara.compile.mode import Mode
from aesara.configdefaults import config
from aesara.graph.fg import FunctionGraph
from aesara.graph.rewriting.db import RewriteDatabaseQuery
from aesara.link.jax.linker import JAXLinker
from aesara.scan.basic import scan
from aesara.scan.op import Scan
from aesara.scan.utils import until
from aesara.tensor.math import gammaln, log
from aesara.tensor.random.utils import RandomStream
from aesara.tensor.type import ivector, lscalar, scalar
from tests.link.jax.test_basic import compare_jax_and_py


jax = pytest.importorskip("jax")

# Disable all optimizations
opts = RewriteDatabaseQuery(include=[None], exclude=["cxx_only", "BlasOpt"])
jax_no_opts = Mode(JAXLinker(), opts)
py_no_opts = Mode("py", opts)


def test_while_cannnot_use_all_outputs():
    """The JAX backend cannot use all the outputs of a while loop.

    Indeed, JAX has fundamental limitations that prevent it from returning
    all the intermediate results computed in a `jax.lax.while_loop` loop.
    """
    res, updates = scan(
        fn=lambda a_tm1: (a_tm1 + 1, until(a_tm1 > 2)),
        outputs_info=[{"initial": at.as_tensor(1, dtype=np.int64), "taps": [-1]}],
        n_steps=5,
    )
    with pytest.raises(NotImplementedError):
        function((), res, updates=updates, mode="JAX")


def test_while_only_last_output():
    """Compile a `Scan` used as a while loop when only the last computed value
    is used.

    """
    res, updates = scan(
        fn=lambda a_tm1: (a_tm1 + 1, until(a_tm1 > 2)),
        outputs_info=[{"initial": at.as_tensor(1, dtype=np.int64), "taps": [-1]}],
        n_steps=5,
    )
    res = res[-1]

    jax_fn = function((), res, updates=updates, mode="JAX")
    fn = function((), res, updates=updates)
    assert np.allclose(fn(), jax_fn())


@pytest.mark.xfail(
    reason="Elemwise{add} transforms concrete values into `TracedArray`s"
)
def test_sit_sot():
    a_at = at.scalar("a", dtype="floatX")

    res, updates = scan(
        fn=lambda a_tm1: 2 * a_tm1,
        outputs_info=[{"initial": a_at, "taps": [-1]}],
        n_steps=3,
    )

    fn = function((a_at,), res, updates=updates)
    jax_fn = function((a_at,), res, updates=updates, mode=jax_no_opts)
    assert np.allclose(fn(1.0), jax_fn(1.0))


def test_sit_sot_opt():
    a_at = at.scalar("a", dtype="floatX")

    res, updates = scan(
        fn=lambda a_tm1: 2 * a_tm1,
        outputs_info=[{"initial": a_at, "taps": [-1]}],
        n_steps=3,
    )

    jax_fn = function((a_at,), res, updates=updates, mode="JAX")
    fn = function((a_at,), res, updates=updates)
    assert np.allclose(fn(1.0), jax_fn(1.0))


def test_nit_sot_shared():
    res, updates = scan(
        fn=lambda: RandomStream(seed=1930, rng_ctor=np.random.RandomState).normal(
            0, 1, name="a"
        ),
        n_steps=3,
    )

    jax_fn = function((), res, updates=updates, mode="JAX")
    res_jax = jax_fn()
    fn = function((), res, updates=updates)
    res = fn()

    assert res_jax.shape == res.shape
    assert not np.all(res_jax == res_jax[0])


@pytest.mark.xfail(
    reason="Elemwise{add} transforms concrete values into `TracedArray`s"
)
def test_mit_sot():
    res, updates = scan(
        fn=lambda a_tm1: 2 * a_tm1,
        outputs_info=[
            {"initial": at.as_tensor([0.0, 1.0], dtype="floatX"), "taps": [-2]}
        ],
        n_steps=6,
    )

    jax_fn = function((), res, updates=updates, mode=jax_no_opts)
    fn = function((), res, updates=updates)
    assert np.allclose(fn(), jax_fn())


def test_mit_sot_opt():
    res, updates = scan(
        fn=lambda a_tm1: 2 * a_tm1,
        outputs_info=[
            {"initial": at.as_tensor([0.0, 1.0], dtype="floatX"), "taps": [-2]}
        ],
        n_steps=6,
    )

    jax_fn = function((), res, updates=updates, mode="JAX")
    fn = function((), res, updates=updates)
    assert np.allclose(fn(), jax_fn())


@pytest.mark.xfail(
    reason="Elemwise{add} transforms concrete values into `TracedArrays`"
)
def test_mit_sot_2():
    res, updates = scan(
        fn=lambda a_tm1, b_tm1: (2 * a_tm1, 2 * b_tm1),
        outputs_info=[
            {"initial": at.as_tensor(1.0, dtype="floatX"), "taps": [-1]},
            {"initial": at.as_tensor(0.5, dtype="floatX"), "taps": [-1]},
        ],
        n_steps=10,
    )
    jax_fn = function((), res, updates=updates, mode=jax_no_opts)
    fn = function((), res, updates=updates)
    assert np.allclose(fn(), jax_fn())


def test_mit_sot_2_opt():
    res, updates = scan(
        fn=lambda a_tm1, b_tm1: (2 * a_tm1, 2 * b_tm1),
        outputs_info=[
            {"initial": at.as_tensor(1.0, dtype="floatX"), "taps": [-1]},
            {"initial": at.as_tensor(0.5, dtype="floatX"), "taps": [-1]},
        ],
        n_steps=10,
    )
    jax_fn = function((), res, updates=updates, mode="JAX")
    fn = function((), res, updates=updates)
    assert np.allclose(fn(), jax_fn())


@pytest.mark.xfail(reason="Indexing with non-static values in the optimized graph")
def test_sequence_opt():
    a_at = at.dvector("a")
    res, updates = scan(fn=lambda a_t: 2 * a_t, sequences=a_at)
    jax_fn = function((a_at,), res, updates=updates, mode="JAX")
    fn = function((a_at,), res, updates=updates)
    assert np.allclose(fn(np.arange(10)), jax_fn(np.arange(10)))


@pytest.mark.parametrize("jax_mode", ("JAX", jax_no_opts))
@pytest.mark.parametrize(
    "fn, sequences, outputs_info, non_sequences, n_steps, input_vals, output_vals, op_check",
    [
        # sequences
        # (
        #     lambda a_t: 2 * a_t,
        #     [at.dvector("a")],
        #     [{}],
        #     [],
        #     None,
        #     [np.arange(10)],
        #     None,
        #     lambda op: op.info.n_seqs > 0,
        # ),
        # # nit-sot
        (
            lambda: at.as_tensor(2.0),
            [],
            [{}],
            [],
            3,
            [],
            None,
            lambda op: op.info.n_nit_sot > 0,
        ),
        # (
        #     lambda: at.as_tensor(2.0),
        #     [],
        #     [{}],
        #     [],
        #     3,
        #     [],
        #     None,
        #     lambda op: op.info.n_nit_sot > 0,
        # ),
        # nit-sot, non_seq
        (
            lambda c: at.as_tensor(2.0) * c,
            [],
            [{}],
            [at.dscalar("c")],
            3,
            [1.0],
            None,
            lambda op: op.info.n_nit_sot > 0 and op.info.n_non_seqs > 0,
        ),
        # sit-sot
        # (
        #     lambda a_tm1: 2 * a_tm1,
        #     [],
        #     [{"initial": at.as_tensor(0.0, dtype="floatX"), "taps": [-1]}],
        #     [],
        #     3,
        #     [],
        #     lambda op: op.info.n_sit_sot > 0,
        # ),
        # # sit-sot, while
        # (
        #     lambda a_tm1: (a_tm1 + 1, until(a_tm1 > 2)),
        #     [],
        #     [{"initial": at.as_tensor(1, dtype=np.int64), "taps": [-1]}],
        #     [],
        #     3,
        #     [],
        #     None,
        #     lambda op: op.info.n_sit_sot > 0,
        # ),
        # # nit-sot, shared input/output
        (
            lambda: RandomStream(seed=1930, rng_ctor=np.random.RandomState).normal(
                0, 1, name="a"
            ),
            [],
            [{}],
            [],
            3,
            [],
            [np.array([-0.4587753, -0.89655604, 2.13323775])],
            lambda op: op.info.n_shared_outs > 0,
        ),
        # mit-sot (that's also a type of sit-sot)
        # (
        #     lambda a_tm1: 2 * a_tm1,
        #     [],
        #     [{"initial": at.as_tensor([0.0, 1.0], dtype="floatX"), "taps": [-2]}],
        #     [],
        #     6,
        #     [],
        #     None,
        #     lambda op: op.info.n_mit_sot > 0,
        # ),
        # mit-sot
        # (
        #     lambda a_tm1, b_tm1: (2 * a_tm1, 2 * b_tm1),
        #     [],
        #     [
        #         {"initial": at.as_tensor(1.0, dtype="floatX"), "taps": [-1]},
        #         {"initial": at.as_tensor(0.3, dtype="floatX"), "taps": [-1]},
        #     ],
        #     [],
        #     10,
        #     [],
        #     None,
        #     lambda op: op.info.n_mit_sot > 0,
        # ),
    ],
)
def test_xit_xot_types(
    jax_mode,
    fn,
    sequences,
    outputs_info,
    non_sequences,
    n_steps,
    input_vals,
    output_vals,
    op_check,
):
    """Test basic xit-xot configurations."""
    res, updates = scan(
        fn,
        sequences=sequences,
        outputs_info=outputs_info,
        non_sequences=non_sequences,
        n_steps=n_steps,
        strict=True,
    )

    if not isinstance(res, list):
        res = [res]

    # Get rid of any `Subtensor` indexing on the `Scan` outputs
    res = [r.owner.inputs[0] if not isinstance(r.owner.op, Scan) else r for r in res]

    scan_op = res[0].owner.op
    assert isinstance(scan_op, Scan)

    _ = op_check(scan_op)

    if output_vals is None:
        compare_jax_and_py(
            ((sequences + non_sequences), res),
            input_vals,
            updates=updates,
        )
    else:
        jax_fn = function(
            (sequences + non_sequences), res, updates=updates, mode=jax_mode
        )
        res_vals = jax_fn(*input_vals)
        assert np.allclose(res_vals, output_vals)


@pytest.mark.xfail(
    version_parse(jax.__version__) >= version_parse("0.2.12"),
    reason="Omnistaging cannot be disabled",
)
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
        return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)

    def binom_log_prob(n, p, value):
        return binomln(n, value) + value * log(p) + (n - value) * log(1 - p)

    # sequences
    at_C = ivector("C_t")
    at_D = ivector("D_t")
    # outputs_info (initial conditions)
    st0 = lscalar("s_t0")
    et0 = lscalar("e_t0")
    it0 = lscalar("i_t0")
    logp_c = scalar("logp_c")
    logp_d = scalar("logp_d")
    # non_sequences
    beta = scalar("beta")
    gamma = scalar("gamma")
    delta = scalar("delta")

    # TODO: Use random streams when their JAX conversions are implemented.
    # trng = aesara.tensor.random.RandomStream(1234)

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

    (st, et, it, logp_c_all, logp_d_all), _ = scan(
        fn=seir_one_step,
        sequences=[at_C, at_D],
        outputs_info=[st0, et0, it0, logp_c, logp_d],
        non_sequences=[beta, gamma, delta],
    )
    st.name = "S_t"
    et.name = "E_t"
    it.name = "I_t"
    logp_c_all.name = "C_t_logp"
    logp_d_all.name = "D_t_logp"

    out_fg = FunctionGraph(
        [at_C, at_D, st0, et0, it0, logp_c, logp_d, beta, gamma, delta],
        [st, et, it, logp_c_all, logp_d_all],
    )

    s0, e0, i0 = 100, 50, 25
    logp_c0 = np.array(0.0, dtype=config.floatX)
    logp_d0 = np.array(0.0, dtype=config.floatX)
    beta_val, gamma_val, delta_val = [
        np.array(val, dtype=config.floatX) for val in [0.277792, 0.135330, 0.108753]
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


@pytest.mark.xfail(
    version_parse(jax.__version__) >= version_parse("0.2.12"),
    reason="Omnistaging cannot be disabled",
)
def test_jax_scan_tap_output():

    a_at = scalar("a")

    def input_step_fn(y_tm1, y_tm3, a):
        y_tm1.name = "y_tm1"
        y_tm3.name = "y_tm3"
        res = (y_tm1 + y_tm3) * a
        res.name = "y_t"
        return res

    y_scan_at, _ = scan(
        fn=input_step_fn,
        outputs_info=[
            {
                "initial": at.as_tensor_variable(
                    np.r_[-1.0, 1.3, 0.0].astype(config.floatX)
                ),
                "taps": [-1, -3],
            },
        ],
        non_sequences=[a_at],
        n_steps=10,
        name="y_scan",
    )
    y_scan_at.name = "y"
    y_scan_at.owner.inputs[0].name = "y_all"

    out_fg = FunctionGraph([a_at], [y_scan_at])

    test_input_vals = [np.array(10.0).astype(config.floatX)]
    compare_jax_and_py(out_fg, test_input_vals)


def test_scan_multiple_none_output():
    A = at.dvector("A")

    def power_step(prior_result, x):
        return prior_result * x, prior_result * x * x, prior_result * x * x * x

    result, _ = scan(
        power_step,
        non_sequences=[A],
        outputs_info=[at.ones_like(A), None, None],
        n_steps=3,
    )

    FunctionGraph([A], result)
    test_input_vals = (np.array([1.0, 2.0]),)

    jax_fn = function((A,), result, mode="JAX")
    jax_res = jax_fn(*test_input_vals)

    fn = function((A,), result)
    res = fn(*test_input_vals)

    for output_jax, output in zip(jax_res, res):
        assert np.allclose(jax_res, res)


@pytest.mark.xfail(reason="Fails for reasons unrelated to `Scan`")
def test_mitmots_basic():

    init_x = at.dvector()
    seq = at.dvector()

    def inner_fct(seq, state_old, state_current):
        return state_old * 2 + state_current + seq

    out, _ = scan(
        inner_fct, sequences=seq, outputs_info={"initial": init_x, "taps": [-2, -1]}
    )

    g_outs = grad(out.sum(), [seq, init_x])

    out_fg = FunctionGraph([seq, init_x], g_outs)

    seq_val = np.arange(3)
    init_x_val = np.r_[-2, -1]
    (seq_val, init_x_val)

    fn = function(out_fg.inputs, out_fg.outputs)
    jax_fn = function(out_fg.inputs, out_fg.outputs, mode="JAX")
    print(fn(seq_val, init_x_val))
    print(jax_fn(seq_val, init_x_val))
