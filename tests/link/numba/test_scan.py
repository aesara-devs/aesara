import numpy as np

import aesara.tensor as at
from aesara import config
from aesara.graph.fg import FunctionGraph
from aesara.scan.basic import scan
from aesara.scan.utils import until
from tests.link.numba.test_basic import compare_numba_and_py


rng = np.random.default_rng(42849)


def test_scan_multiple_output():
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
        return at.exp(n + 1) - at.exp(k + 1) - at.exp(n - k + 1)

    def binom_log_prob(n, p, value):
        return binomln(n, value) + value * at.exp(p) + (n - value) * at.exp(1 - p)

    # sequences
    at_C = at.ivector("C_t")
    at_D = at.ivector("D_t")
    # outputs_info (initial conditions)
    st0 = at.lscalar("s_t0")
    et0 = at.lscalar("e_t0")
    it0 = at.lscalar("i_t0")
    logp_c = at.scalar("logp_c")
    logp_d = at.scalar("logp_d")
    # non_sequences
    beta = at.scalar("beta")
    gamma = at.scalar("gamma")
    delta = at.scalar("delta")

    def seir_one_step(ct0, dt0, st0, et0, it0, logp_c, logp_d, beta, gamma, delta):
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
    compare_numba_and_py(out_fg, test_input_vals)


@config.change_flags(compute_test_value="raise")
def test_scan_tap_output():

    a_at = at.scalar("a")
    a_at.tag.test_value = 10.0

    b_at = at.arange(11).astype(config.floatX)
    b_at.name = "b"

    c_at = at.arange(20, 31, dtype=config.floatX)
    c_at.name = "c"

    def input_step_fn(b, b2, c, x_tm1, y_tm1, y_tm3, a):
        x_tm1.name = "x_tm1"
        y_tm1.name = "y_tm1"
        y_tm3.name = "y_tm3"
        y_t = (y_tm1 + y_tm3) * a + b + b2
        z_t = y_t * c
        x_t = x_tm1 + 1
        x_t.name = "x_t"
        y_t.name = "y_t"
        return x_t, y_t, at.fill((10,), z_t)

    scan_res, _ = scan(
        fn=input_step_fn,
        sequences=[
            {
                "input": b_at,
                "taps": [-1, -2],
            },
            {
                "input": c_at,
                "taps": [-2],
            },
        ],
        outputs_info=[
            {
                "initial": at.as_tensor_variable(0.0, dtype=config.floatX),
                "taps": [-1],
            },
            {
                "initial": at.as_tensor_variable(
                    np.r_[-1.0, 1.3, 0.0].astype(config.floatX)
                ),
                "taps": [-1, -3],
            },
            None,
        ],
        non_sequences=[a_at],
        n_steps=5,
        name="yz_scan",
        strict=True,
    )

    out_fg = FunctionGraph([a_at, b_at, c_at], scan_res)

    test_input_vals = [
        np.array(10.0).astype(config.floatX),
        np.arange(11, dtype=config.floatX),
        np.arange(20, 31, dtype=config.floatX),
    ]
    compare_numba_and_py(out_fg, test_input_vals)


def test_scan_while():
    def power_of_2(previous_power, max_value):
        return previous_power * 2, until(previous_power * 2 > max_value)

    max_value = at.scalar()
    values, _ = scan(
        power_of_2,
        outputs_info=at.constant(1.0),
        non_sequences=max_value,
        n_steps=1024,
    )

    out_fg = FunctionGraph([max_value], [values])

    test_input_vals = [
        np.array(45).astype(config.floatX),
    ]
    compare_numba_and_py(out_fg, test_input_vals)


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

    out_fg = FunctionGraph([A], result)
    test_input_vals = (np.array([1.0, 2.0]),)

    compare_numba_and_py(out_fg, test_input_vals)
