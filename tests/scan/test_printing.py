import numpy as np
import pytest

import aesara
import aesara.tensor as at
from aesara.printing import debugprint, pydot_imported, pydotprint
from aesara.tensor.type import dvector, iscalar, scalar, vector


def test_scan_debugprint1():
    k = iscalar("k")
    A = dvector("A")

    # Symbolic description of the result
    result, updates = aesara.scan(
        fn=lambda prior_result, A: prior_result * A,
        outputs_info=at.ones_like(A),
        non_sequences=A,
        n_steps=k,
    )

    final_result = result[-1]
    output_str = debugprint(final_result, file="str")
    lines = output_str.split("\n")

    expected_output = r"""Subtensor{int64} [id A] ''
 |Subtensor{int64::} [id B] ''
 | |for{cpu,scan_fn} [id C] ''
 | | |k [id D]
 | | |IncSubtensor{Set;:int64:} [id E] ''
 | | | |AllocEmpty{dtype='float64'} [id F] ''
 | | | | |Elemwise{add,no_inplace} [id G] ''
 | | | | | |k [id D]
 | | | | | |Subtensor{int64} [id H] ''
 | | | | |   |Shape [id I] ''
 | | | | |   | |Rebroadcast{(0, False)} [id J] ''
 | | | | |   |   |InplaceDimShuffle{x,0} [id K] ''
 | | | | |   |     |Elemwise{second,no_inplace} [id L] ''
 | | | | |   |       |A [id M]
 | | | | |   |       |InplaceDimShuffle{x} [id N] ''
 | | | | |   |         |TensorConstant{1.0} [id O]
 | | | | |   |ScalarConstant{0} [id P]
 | | | | |Subtensor{int64} [id Q] ''
 | | | |   |Shape [id R] ''
 | | | |   | |Rebroadcast{(0, False)} [id J] ''
 | | | |   |ScalarConstant{1} [id S]
 | | | |Rebroadcast{(0, False)} [id J] ''
 | | | |ScalarFromTensor [id T] ''
 | | |   |Subtensor{int64} [id H] ''
 | | |A [id M]
 | | \*0-<TensorType(float64, (None,))> [id U]
 | | \*1-<TensorType(float64, (None,))> [id V]
 | | \Elemwise{mul,no_inplace} [id W] ''
 | |   |*0-<TensorType(float64, (None,))> [id U]
 | |   |*1-<TensorType(float64, (None,))> [id V]
 | |ScalarConstant{1} [id X]
 |ScalarConstant{-1} [id Y]

Inner graphs:

for{cpu,scan_fn} [id C] ''
 >Elemwise{mul,no_inplace} [id W] ''"""

    for truth, out in zip(expected_output.split("\n"), lines):
        assert truth.strip() == out.strip()


def test_scan_debugprint2():
    coefficients = vector("coefficients")
    x = scalar("x")

    max_coefficients_supported = 10000

    # Generate the components of the polynomial
    components, updates = aesara.scan(
        fn=lambda coefficient, power, free_variable: coefficient
        * (free_variable ** power),
        outputs_info=None,
        sequences=[coefficients, at.arange(max_coefficients_supported)],
        non_sequences=x,
    )
    # Sum them up
    polynomial = components.sum()

    output_str = debugprint(polynomial, file="str")
    lines = output_str.split("\n")

    expected_output = r"""Sum{acc_dtype=float64} [id A] ''
 |for{cpu,scan_fn} [id B] ''
   |Elemwise{scalar_minimum,no_inplace} [id C] ''
   | |Subtensor{int64} [id D] ''
   | | |Shape [id E] ''
   | | | |Subtensor{int64::} [id F] 'coefficients[0:]'
   | | |   |coefficients [id G]
   | | |   |ScalarConstant{0} [id H]
   | | |ScalarConstant{0} [id I]
   | |Subtensor{int64} [id J] ''
   |   |Shape [id K] ''
   |   | |Subtensor{int64::} [id L] ''
   |   |   |ARange{dtype='int64'} [id M] ''
   |   |   | |TensorConstant{0} [id N]
   |   |   | |TensorConstant{10000} [id O]
   |   |   | |TensorConstant{1} [id P]
   |   |   |ScalarConstant{0} [id Q]
   |   |ScalarConstant{0} [id R]
   |Subtensor{:int64:} [id S] ''
   | |Subtensor{int64::} [id F] 'coefficients[0:]'
   | |ScalarFromTensor [id T] ''
   |   |Elemwise{scalar_minimum,no_inplace} [id C] ''
   |Subtensor{:int64:} [id U] ''
   | |Subtensor{int64::} [id L] ''
   | |ScalarFromTensor [id V] ''
   |   |Elemwise{scalar_minimum,no_inplace} [id C] ''
   |Elemwise{scalar_minimum,no_inplace} [id C] ''
   |x [id W]
   \*0-<TensorType(float64, ())> [id X]
   \*1-<TensorType(int64, ())> [id Y]
   \*2-<TensorType(float64, ())> [id Z]
   \Elemwise{mul,no_inplace} [id BA] ''
     |*0-<TensorType(float64, ())> [id X]
     |Elemwise{pow,no_inplace} [id BB] ''
       |*2-<TensorType(float64, ())> [id Z]
       |*1-<TensorType(int64, ())> [id Y]

Inner graphs:

for{cpu,scan_fn} [id B] ''
 >Elemwise{mul,no_inplace} [id BA] ''"""

    for truth, out in zip(expected_output.split("\n"), lines):
        assert truth.strip() == out.strip()


def test_scan_debugprint3():
    coefficients = dvector("coefficients")
    max_coefficients_supported = 10

    k = iscalar("k")
    A = dvector("A")

    # compute A**k
    def compute_A_k(A, k):
        # Symbolic description of the result
        result, updates = aesara.scan(
            fn=lambda prior_result, A: prior_result * A,
            outputs_info=at.ones_like(A),
            non_sequences=A,
            n_steps=k,
        )

        A_k = result[-1]

        return A_k

    # Generate the components of the polynomial
    components, updates = aesara.scan(
        fn=lambda coefficient, power, some_A, some_k: coefficient
        * (compute_A_k(some_A, some_k) ** power),
        outputs_info=None,
        sequences=[coefficients, at.arange(max_coefficients_supported)],
        non_sequences=[A, k],
    )
    # Sum them up
    polynomial = components.sum()

    final_result = polynomial

    output_str = debugprint(final_result, file="str")
    lines = output_str.split("\n")

    expected_output = r"""Sum{acc_dtype=float64} [id A] ''
 |for{cpu,scan_fn} [id B] ''
   |Elemwise{scalar_minimum,no_inplace} [id C] ''
   | |Subtensor{int64} [id D] ''
   | | |Shape [id E] ''
   | | | |Subtensor{int64::} [id F] 'coefficients[0:]'
   | | |   |coefficients [id G]
   | | |   |ScalarConstant{0} [id H]
   | | |ScalarConstant{0} [id I]
   | |Subtensor{int64} [id J] ''
   |   |Shape [id K] ''
   |   | |Subtensor{int64::} [id L] ''
   |   |   |ARange{dtype='int64'} [id M] ''
   |   |   | |TensorConstant{0} [id N]
   |   |   | |TensorConstant{10} [id O]
   |   |   | |TensorConstant{1} [id P]
   |   |   |ScalarConstant{0} [id Q]
   |   |ScalarConstant{0} [id R]
   |Subtensor{:int64:} [id S] ''
   | |Subtensor{int64::} [id F] 'coefficients[0:]'
   | |ScalarFromTensor [id T] ''
   |   |Elemwise{scalar_minimum,no_inplace} [id C] ''
   |Subtensor{:int64:} [id U] ''
   | |Subtensor{int64::} [id L] ''
   | |ScalarFromTensor [id V] ''
   |   |Elemwise{scalar_minimum,no_inplace} [id C] ''
   |Elemwise{scalar_minimum,no_inplace} [id C] ''
   |A [id W]
   |k [id X]
   \*0-<TensorType(float64, ())> [id Y]
   \*1-<TensorType(int64, ())> [id Z]
   \*2-<TensorType(float64, (None,))> [id BA]
   \*3-<TensorType(int32, ())> [id BB]
   \Elemwise{mul,no_inplace} [id BC] ''
     |InplaceDimShuffle{x} [id BD] ''
     | |*0-<TensorType(float64, ())> [id Y]
     |Elemwise{pow,no_inplace} [id BE] ''
       |Subtensor{int64} [id BF] ''
       | |Subtensor{int64::} [id BG] ''
       | | |for{cpu,scan_fn} [id BH] ''
       | | | |*3-<TensorType(int32, ())> [id BB]
       | | | |IncSubtensor{Set;:int64:} [id BI] ''
       | | | | |AllocEmpty{dtype='float64'} [id BJ] ''
       | | | | | |Elemwise{add,no_inplace} [id BK] ''
       | | | | | | |*3-<TensorType(int32, ())> [id BB]
       | | | | | | |Subtensor{int64} [id BL] ''
       | | | | | |   |Shape [id BM] ''
       | | | | | |   | |Rebroadcast{(0, False)} [id BN] ''
       | | | | | |   |   |InplaceDimShuffle{x,0} [id BO] ''
       | | | | | |   |     |Elemwise{second,no_inplace} [id BP] ''
       | | | | | |   |       |*2-<TensorType(float64, (None,))> [id BA]
       | | | | | |   |       |InplaceDimShuffle{x} [id BQ] ''
       | | | | | |   |         |TensorConstant{1.0} [id BR]
       | | | | | |   |ScalarConstant{0} [id BS]
       | | | | | |Subtensor{int64} [id BT] ''
       | | | | |   |Shape [id BU] ''
       | | | | |   | |Rebroadcast{(0, False)} [id BN] ''
       | | | | |   |ScalarConstant{1} [id BV]
       | | | | |Rebroadcast{(0, False)} [id BN] ''
       | | | | |ScalarFromTensor [id BW] ''
       | | | |   |Subtensor{int64} [id BL] ''
       | | | |*2-<TensorType(float64, (None,))> [id BA]
       | | | \*0-<TensorType(float64, (None,))> [id BX]
       | | | \*1-<TensorType(float64, (None,))> [id BY]
       | | | \Elemwise{mul,no_inplace} [id BZ] ''
       | | |   |*0-<TensorType(float64, (None,))> [id BX]
       | | |   |*1-<TensorType(float64, (None,))> [id BY]
       | | |ScalarConstant{1} [id CA]
       | |ScalarConstant{-1} [id CB]
       |InplaceDimShuffle{x} [id CC] ''
         |*1-<TensorType(int64, ())> [id Z]

Inner graphs:

for{cpu,scan_fn} [id B] ''
 >Elemwise{mul,no_inplace} [id BC] ''

for{cpu,scan_fn} [id BH] ''
 >Elemwise{mul,no_inplace} [id BZ] ''
"""

    for truth, out in zip(expected_output.split("\n"), lines):
        assert truth.strip() == out.strip()


def test_scan_debugprint4():
    def fn(a_m2, a_m1, b_m2, b_m1):
        return a_m1 + a_m2, b_m1 + b_m2

    a0 = aesara.shared(np.arange(2, dtype="int64"))
    b0 = aesara.shared(np.arange(2, dtype="int64"))

    (a, b), _ = aesara.scan(
        fn,
        outputs_info=[
            {"initial": a0, "taps": [-2, -1]},
            {"initial": b0, "taps": [-2, -1]},
        ],
        n_steps=5,
    )

    final_result = a + b
    output_str = debugprint(final_result, file="str")
    lines = output_str.split("\n")

    expected_output = r"""Elemwise{add,no_inplace} [id A] ''
 |Subtensor{int64::} [id B] ''
 | |for{cpu,scan_fn}.0 [id C] ''
 | | |TensorConstant{5} [id D]
 | | |IncSubtensor{Set;:int64:} [id E] ''
 | | | |AllocEmpty{dtype='int64'} [id F] ''
 | | | | |Elemwise{add,no_inplace} [id G] ''
 | | | |   |TensorConstant{5} [id D]
 | | | |   |Subtensor{int64} [id H] ''
 | | | |     |Shape [id I] ''
 | | | |     | |Subtensor{:int64:} [id J] ''
 | | | |     |   |<TensorType(int64, (None,))> [id K]
 | | | |     |   |ScalarConstant{2} [id L]
 | | | |     |ScalarConstant{0} [id M]
 | | | |Subtensor{:int64:} [id J] ''
 | | | |ScalarFromTensor [id N] ''
 | | |   |Subtensor{int64} [id H] ''
 | | |IncSubtensor{Set;:int64:} [id O] ''
 | | | |AllocEmpty{dtype='int64'} [id P] ''
 | | | | |Elemwise{add,no_inplace} [id Q] ''
 | | | |   |TensorConstant{5} [id D]
 | | | |   |Subtensor{int64} [id R] ''
 | | | |     |Shape [id S] ''
 | | | |     | |Subtensor{:int64:} [id T] ''
 | | | |     |   |<TensorType(int64, (None,))> [id U]
 | | | |     |   |ScalarConstant{2} [id V]
 | | | |     |ScalarConstant{0} [id W]
 | | | |Subtensor{:int64:} [id T] ''
 | | | |ScalarFromTensor [id X] ''
 | | |   |Subtensor{int64} [id R] ''
 | | \*0-<TensorType(int64, ())> [id Y]
 | | \*1-<TensorType(int64, ())> [id Z]
 | | \*2-<TensorType(int64, ())> [id BA]
 | | \*3-<TensorType(int64, ())> [id BB]
 | | \Elemwise{add,no_inplace} [id BC] ''
 | | \ |*1-<TensorType(int64, ())> [id Z]
 | | \ |*0-<TensorType(int64, ())> [id Y]
 | | \Elemwise{add,no_inplace} [id BD] ''
 | |   |*3-<TensorType(int64, ())> [id BB]
 | |   |*2-<TensorType(int64, ())> [id BA]
 | |ScalarConstant{2} [id BE]
 |Subtensor{int64::} [id BF] ''
   |for{cpu,scan_fn}.1 [id C] ''
   |ScalarConstant{2} [id BG]

Inner graphs:

for{cpu,scan_fn}.0 [id C] ''
 >Elemwise{add,no_inplace} [id BC] ''
 >Elemwise{add,no_inplace} [id BD] ''

for{cpu,scan_fn}.1 [id C] ''
 >Elemwise{add,no_inplace} [id BC] ''
 >Elemwise{add,no_inplace} [id BD] ''"""

    for truth, out in zip(expected_output.split("\n"), lines):
        assert truth.strip() == out.strip()


def test_scan_debugprint5():

    k = iscalar("k")
    A = dvector("A")

    # Symbolic description of the result
    result, updates = aesara.scan(
        fn=lambda prior_result, A: prior_result * A,
        outputs_info=at.ones_like(A),
        non_sequences=A,
        n_steps=k,
    )

    final_result = aesara.grad(result[-1].sum(), A)

    output_str = debugprint(final_result, file="str")
    lines = output_str.split("\n")

    expected_output = r"""Subtensor{int64} [id A] ''
 |for{cpu,grad_of_scan_fn}.1 [id B] ''
 | |Elemwise{sub,no_inplace} [id C] ''
 | | |Subtensor{int64} [id D] ''
 | | | |Shape [id E] ''
 | | | | |for{cpu,scan_fn} [id F] ''
 | | | |   |k [id G]
 | | | |   |IncSubtensor{Set;:int64:} [id H] ''
 | | | |   | |AllocEmpty{dtype='float64'} [id I] ''
 | | | |   | | |Elemwise{add,no_inplace} [id J] ''
 | | | |   | | | |k [id G]
 | | | |   | | | |Subtensor{int64} [id K] ''
 | | | |   | | |   |Shape [id L] ''
 | | | |   | | |   | |Rebroadcast{(0, False)} [id M] ''
 | | | |   | | |   |   |InplaceDimShuffle{x,0} [id N] ''
 | | | |   | | |   |     |Elemwise{second,no_inplace} [id O] ''
 | | | |   | | |   |       |A [id P]
 | | | |   | | |   |       |InplaceDimShuffle{x} [id Q] ''
 | | | |   | | |   |         |TensorConstant{1.0} [id R]
 | | | |   | | |   |ScalarConstant{0} [id S]
 | | | |   | | |Subtensor{int64} [id T] ''
 | | | |   | |   |Shape [id U] ''
 | | | |   | |   | |Rebroadcast{(0, False)} [id M] ''
 | | | |   | |   |ScalarConstant{1} [id V]
 | | | |   | |Rebroadcast{(0, False)} [id M] ''
 | | | |   | |ScalarFromTensor [id W] ''
 | | | |   |   |Subtensor{int64} [id K] ''
 | | | |   |A [id P]
 | | | |   \*0-<TensorType(float64, (None,))> [id X]
 | | | |   \*1-<TensorType(float64, (None,))> [id Y]
 | | | |   \Elemwise{mul,no_inplace} [id Z] ''
 | | | |     |*0-<TensorType(float64, (None,))> [id X]
 | | | |     |*1-<TensorType(float64, (None,))> [id Y]
 | | | |ScalarConstant{0} [id BA]
 | | |TensorConstant{1} [id BB]
 | |Subtensor{:int64:} [id BC] ''
 | | |Subtensor{::int64} [id BD] ''
 | | | |Subtensor{:int64:} [id BE] ''
 | | | | |for{cpu,scan_fn} [id F] ''
 | | | | |ScalarConstant{-1} [id BF]
 | | | |ScalarConstant{-1} [id BG]
 | | |ScalarFromTensor [id BH] ''
 | |   |Elemwise{sub,no_inplace} [id C] ''
 | |Subtensor{:int64:} [id BI] ''
 | | |Subtensor{:int64:} [id BJ] ''
 | | | |Subtensor{::int64} [id BK] ''
 | | | | |for{cpu,scan_fn} [id F] ''
 | | | | |ScalarConstant{-1} [id BL]
 | | | |ScalarConstant{-1} [id BM]
 | | |ScalarFromTensor [id BN] ''
 | |   |Elemwise{sub,no_inplace} [id C] ''
 | |Subtensor{::int64} [id BO] ''
 | | |IncSubtensor{Inc;int64::} [id BP] ''
 | | | |Elemwise{second,no_inplace} [id BQ] ''
 | | | | |for{cpu,scan_fn} [id F] ''
 | | | | |InplaceDimShuffle{x,x} [id BR] ''
 | | | |   |TensorConstant{0.0} [id BS]
 | | | |IncSubtensor{Inc;int64} [id BT] ''
 | | | | |Elemwise{second,no_inplace} [id BU] ''
 | | | | | |Subtensor{int64::} [id BV] ''
 | | | | | | |for{cpu,scan_fn} [id F] ''
 | | | | | | |ScalarConstant{1} [id BW]
 | | | | | |InplaceDimShuffle{x,x} [id BX] ''
 | | | | |   |TensorConstant{0.0} [id BY]
 | | | | |Elemwise{second} [id BZ] ''
 | | | | | |Subtensor{int64} [id CA] ''
 | | | | | | |Subtensor{int64::} [id BV] ''
 | | | | | | |ScalarConstant{-1} [id CB]
 | | | | | |InplaceDimShuffle{x} [id CC] ''
 | | | | |   |Elemwise{second,no_inplace} [id CD] ''
 | | | | |     |Sum{acc_dtype=float64} [id CE] ''
 | | | | |     | |Subtensor{int64} [id CA] ''
 | | | | |     |TensorConstant{1.0} [id CF]
 | | | | |ScalarConstant{-1} [id CB]
 | | | |ScalarConstant{1} [id BW]
 | | |ScalarConstant{-1} [id CG]
 | |Alloc [id CH] ''
 | | |TensorConstant{0.0} [id CI]
 | | |Elemwise{add,no_inplace} [id CJ] ''
 | | | |Elemwise{sub,no_inplace} [id C] ''
 | | | |TensorConstant{1} [id CK]
 | | |Subtensor{int64} [id CL] ''
 | |   |Shape [id CM] ''
 | |   | |A [id P]
 | |   |ScalarConstant{0} [id CN]
 | |A [id P]
 | \*0-<TensorType(float64, (None,))> [id X]
 | \*1-<TensorType(float64, (None,))> [id Y]
 | \*2-<TensorType(float64, (None,))> [id CO]
 | \*3-<TensorType(float64, (None,))> [id CP]
 | \*4-<TensorType(float64, (None,))> [id CQ]
 | \*5-<TensorType(float64, (None,))> [id CR]
 | \Elemwise{add,no_inplace} [id CS] ''
 | \ |Elemwise{mul} [id CT] ''
 | \ | |*2-<TensorType(float64, (None,))> [id CO]
 | \ | |*5-<TensorType(float64, (None,))> [id CR]
 | \ |*3-<TensorType(float64, (None,))> [id CP]
 | \Elemwise{add,no_inplace} [id CU] ''
 |   |Elemwise{mul} [id CV] ''
 |   | |*2-<TensorType(float64, (None,))> [id CO]
 |   | |*0-<TensorType(float64, (None,))> [id X]
 |   |*4-<TensorType(float64, (None,))> [id CQ]
 |ScalarConstant{-1} [id CW]

Inner graphs:

for{cpu,grad_of_scan_fn}.1 [id B] ''
 >Elemwise{add,no_inplace} [id CS] ''
 >Elemwise{add,no_inplace} [id CU] ''

for{cpu,scan_fn} [id F] ''
 >Elemwise{mul,no_inplace} [id Z] ''

for{cpu,scan_fn} [id F] ''
 >Elemwise{mul,no_inplace} [id Z] ''

for{cpu,scan_fn} [id F] ''
 >Elemwise{mul,no_inplace} [id Z] ''

for{cpu,scan_fn} [id F] ''
 >Elemwise{mul,no_inplace} [id Z] ''

for{cpu,scan_fn} [id F] ''
 >Elemwise{mul,no_inplace} [id Z] ''"""

    for truth, out in zip(expected_output.split("\n"), lines):
        assert truth.strip() == out.strip()


@pytest.mark.skipif(not pydot_imported, reason="pydot not available")
def test_printing_scan():
    def f_pow2(x_tm1):
        return 2 * x_tm1

    state = scalar("state")
    n_steps = iscalar("nsteps")
    output, updates = aesara.scan(
        f_pow2, [], state, [], n_steps=n_steps, truncate_gradient=-1, go_backwards=False
    )
    f = aesara.function(
        [state, n_steps], output, updates=updates, allow_input_downcast=True
    )
    pydotprint(output, scan_graphs=True)
    pydotprint(f, scan_graphs=True)
