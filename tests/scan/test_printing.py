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

    expected_output = """Subtensor{int64} [id A] ''
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
     | |ScalarConstant{1} [id U]
     |ScalarConstant{-1} [id V]

    Inner graphs:

    for{cpu,scan_fn} [id C] ''
     >Elemwise{mul,no_inplace} [id W] ''
     > |<TensorType(float64, (None,))> [id X] -> [id E]
     > |A_copy [id Y] -> [id M]"""

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

    expected_output = """Sum{acc_dtype=float64} [id A] ''
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

    Inner graphs:

    for{cpu,scan_fn} [id B] ''
     >Elemwise{mul,no_inplace} [id X] ''
     > |coefficients[t] [id Y] -> [id S]
     > |Elemwise{pow,no_inplace} [id Z] ''
     >   |x_copy [id BA] -> [id W]
     >   |<TensorType(int64, ())> [id BB] -> [id U]"""

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

    expected_output = """Sum{acc_dtype=float64} [id A] ''
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

    Inner graphs:

    for{cpu,scan_fn} [id B] ''
     >Elemwise{mul,no_inplace} [id Y] ''
     > |InplaceDimShuffle{x} [id Z] ''
     > | |coefficients[t] [id BA] -> [id S]
     > |Elemwise{pow,no_inplace} [id BB] ''
     >   |Subtensor{int64} [id BC] ''
     >   | |Subtensor{int64::} [id BD] ''
     >   | | |for{cpu,scan_fn} [id BE] ''
     >   | | | |k_copy [id BF] -> [id X]
     >   | | | |IncSubtensor{Set;:int64:} [id BG] ''
     >   | | | | |AllocEmpty{dtype='float64'} [id BH] ''
     >   | | | | | |Elemwise{add,no_inplace} [id BI] ''
     >   | | | | | | |k_copy [id BF] -> [id X]
     >   | | | | | | |Subtensor{int64} [id BJ] ''
     >   | | | | | |   |Shape [id BK] ''
     >   | | | | | |   | |Rebroadcast{(0, False)} [id BL] ''
     >   | | | | | |   |   |InplaceDimShuffle{x,0} [id BM] ''
     >   | | | | | |   |     |Elemwise{second,no_inplace} [id BN] ''
     >   | | | | | |   |       |A_copy [id BO] -> [id W]
     >   | | | | | |   |       |InplaceDimShuffle{x} [id BP] ''
     >   | | | | | |   |         |TensorConstant{1.0} [id BQ]
     >   | | | | | |   |ScalarConstant{0} [id BR]
     >   | | | | | |Subtensor{int64} [id BS] ''
     >   | | | | |   |Shape [id BT] ''
     >   | | | | |   | |Rebroadcast{(0, False)} [id BL] ''
     >   | | | | |   |ScalarConstant{1} [id BU]
     >   | | | | |Rebroadcast{(0, False)} [id BL] ''
     >   | | | | |ScalarFromTensor [id BV] ''
     >   | | | |   |Subtensor{int64} [id BJ] ''
     >   | | | |A_copy [id BO] -> [id W]
     >   | | |ScalarConstant{1} [id BW]
     >   | |ScalarConstant{-1} [id BX]
     >   |InplaceDimShuffle{x} [id BY] ''
     >     |<TensorType(int64, ())> [id BZ] -> [id U]

    for{cpu,scan_fn} [id BE] ''
     >Elemwise{mul,no_inplace} [id CA] ''
     > |<TensorType(float64, (None,))> [id CB] -> [id BG]
     > |A_copy [id CC] -> [id BO]"""

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

    expected_output = """Elemwise{add,no_inplace} [id A] ''
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
     | |   |AllocEmpty{dtype='int64'} [id P] ''
     | |   | |Elemwise{add,no_inplace} [id Q] ''
     | |   |   |TensorConstant{5} [id D]
     | |   |   |Subtensor{int64} [id R] ''
     | |   |     |Shape [id S] ''
     | |   |     | |Subtensor{:int64:} [id T] ''
     | |   |     |   |<TensorType(int64, (None,))> [id U]
     | |   |     |   |ScalarConstant{2} [id V]
     | |   |     |ScalarConstant{0} [id W]
     | |   |Subtensor{:int64:} [id T] ''
     | |   |ScalarFromTensor [id X] ''
     | |     |Subtensor{int64} [id R] ''
     | |ScalarConstant{2} [id Y]
     |Subtensor{int64::} [id Z] ''
       |for{cpu,scan_fn}.1 [id C] ''
       |ScalarConstant{2} [id BA]

    Inner graphs:

    for{cpu,scan_fn}.0 [id C] ''
     >Elemwise{add,no_inplace} [id BB] ''
     > |<TensorType(int64, ())> [id BC] -> [id E]
     > |<TensorType(int64, ())> [id BD] -> [id E]
     >Elemwise{add,no_inplace} [id BE] ''
     > |<TensorType(int64, ())> [id BF] -> [id O]
     > |<TensorType(int64, ())> [id BG] -> [id O]

    for{cpu,scan_fn}.1 [id C] ''
     >Elemwise{add,no_inplace} [id BB] ''
     >Elemwise{add,no_inplace} [id BE] ''"""

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

    expected_output = """Subtensor{int64} [id A] ''
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
    | | | |ScalarConstant{0} [id X]
    | | |TensorConstant{1} [id Y]
    | |Subtensor{:int64:} [id Z] ''
    | | |Subtensor{::int64} [id BA] ''
    | | | |Subtensor{:int64:} [id BB] ''
    | | | | |for{cpu,scan_fn} [id F] ''
    | | | | |ScalarConstant{-1} [id BC]
    | | | |ScalarConstant{-1} [id BD]
    | | |ScalarFromTensor [id BE] ''
    | |   |Elemwise{sub,no_inplace} [id C] ''
    | |Subtensor{:int64:} [id BF] ''
    | | |Subtensor{:int64:} [id BG] ''
    | | | |Subtensor{::int64} [id BH] ''
    | | | | |for{cpu,scan_fn} [id F] ''
    | | | | |ScalarConstant{-1} [id BI]
    | | | |ScalarConstant{-1} [id BJ]
    | | |ScalarFromTensor [id BK] ''
    | |   |Elemwise{sub,no_inplace} [id C] ''
    | |Subtensor{::int64} [id BL] ''
    | | |IncSubtensor{Inc;int64::} [id BM] ''
    | | | |Elemwise{second,no_inplace} [id BN] ''
    | | | | |for{cpu,scan_fn} [id F] ''
    | | | | |InplaceDimShuffle{x,x} [id BO] ''
    | | | |   |TensorConstant{0.0} [id BP]
    | | | |IncSubtensor{Inc;int64} [id BQ] ''
    | | | | |Elemwise{second,no_inplace} [id BR] ''
    | | | | | |Subtensor{int64::} [id BS] ''
    | | | | | | |for{cpu,scan_fn} [id F] ''
    | | | | | | |ScalarConstant{1} [id BT]
    | | | | | |InplaceDimShuffle{x,x} [id BU] ''
    | | | | |   |TensorConstant{0.0} [id BV]
    | | | | |Elemwise{second} [id BW] ''
    | | | | | |Subtensor{int64} [id BX] ''
    | | | | | | |Subtensor{int64::} [id BS] ''
    | | | | | | |ScalarConstant{-1} [id BY]
    | | | | | |InplaceDimShuffle{x} [id BZ] ''
    | | | | |   |Elemwise{second,no_inplace} [id CA] ''
    | | | | |     |Sum{acc_dtype=float64} [id CB] ''
    | | | | |     | |Subtensor{int64} [id BX] ''
    | | | | |     |TensorConstant{1.0} [id CC]
    | | | | |ScalarConstant{-1} [id BY]
    | | | |ScalarConstant{1} [id BT]
    | | |ScalarConstant{-1} [id CD]
    | |Alloc [id CE] ''
    | | |TensorConstant{0.0} [id CF]
    | | |Elemwise{add,no_inplace} [id CG] ''
    | | | |Elemwise{sub,no_inplace} [id C] ''
    | | | |TensorConstant{1} [id CH]
    | | |Subtensor{int64} [id CI] ''
    | |   |Shape [id CJ] ''
    | |   | |A [id P]
    | |   |ScalarConstant{0} [id CK]
    | |A [id P]
    |ScalarConstant{-1} [id CL]

    Inner graphs:

    for{cpu,grad_of_scan_fn}.1 [id B] ''
    >Elemwise{add,no_inplace} [id CM] ''
    > |Elemwise{mul} [id CN] ''
    > | |<TensorType(float64, (None,))> [id CO] -> [id BL]
    > | |A_copy [id CP] -> [id P]
    > |<TensorType(float64, (None,))> [id CQ] -> [id BL]
    >Elemwise{add,no_inplace} [id CR] ''
    > |Elemwise{mul} [id CS] ''
    > | |<TensorType(float64, (None,))> [id CO] -> [id BL]
    > | |<TensorType(float64, (None,))> [id CT] -> [id Z]
    > |<TensorType(float64, (None,))> [id CU] -> [id CE]

    for{cpu,scan_fn} [id F] ''
    >Elemwise{mul,no_inplace} [id CV] ''
    > |<TensorType(float64, (None,))> [id CT] -> [id H]
    > |A_copy [id CP] -> [id P]

    for{cpu,scan_fn} [id F] ''
    >Elemwise{mul,no_inplace} [id CV] ''

    for{cpu,scan_fn} [id F] ''
    >Elemwise{mul,no_inplace} [id CV] ''

    for{cpu,scan_fn} [id F] ''
    >Elemwise{mul,no_inplace} [id CV] ''

    for{cpu,scan_fn} [id F] ''
    >Elemwise{mul,no_inplace} [id CV] ''"""

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
