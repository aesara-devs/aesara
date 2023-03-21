import numpy as np
import pytest

import aesara
import aesara.tensor as at
from aesara.configdefaults import config
from aesara.graph.fg import FunctionGraph
from aesara.printing import debugprint, pydot_imported, pydotprint
from aesara.tensor.type import dvector, iscalar, scalar, vector


@config.change_flags(floatX="float64")
def test_debugprint_sitsot():
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
    output_str = debugprint(final_result, file="str", print_op_info=True)
    lines = output_str.split("\n")

    expected_output = """Subtensor{int64} [id A]
     |Subtensor{int64::} [id B]
     | |for{cpu,scan_fn} [id C] (outer_out_sit_sot-0)
     | | |k [id D] (n_steps)
     | | |IncSubtensor{Set;:int64:} [id E] (outer_in_sit_sot-0)
     | | | |AllocEmpty{dtype='float64'} [id F]
     | | | | |Elemwise{add,no_inplace} [id G]
     | | | | | |k [id D]
     | | | | | |Subtensor{int64} [id H]
     | | | | |   |Shape [id I]
     | | | | |   | |Unbroadcast{0} [id J]
     | | | | |   |   |InplaceDimShuffle{x,0} [id K]
     | | | | |   |     |Elemwise{second,no_inplace} [id L]
     | | | | |   |       |A [id M]
     | | | | |   |       |InplaceDimShuffle{x} [id N]
     | | | | |   |         |TensorConstant{1.0} [id O]
     | | | | |   |ScalarConstant{0} [id P]
     | | | | |Subtensor{int64} [id Q]
     | | | |   |Shape [id R]
     | | | |   | |Unbroadcast{0} [id J]
     | | | |   |ScalarConstant{1} [id S]
     | | | |Unbroadcast{0} [id J]
     | | | |ScalarFromTensor [id T]
     | | |   |Subtensor{int64} [id H]
     | | |A [id M] (outer_in_non_seqs-0)
     | |ScalarConstant{1} [id U]
     |ScalarConstant{-1} [id V]

    Inner graphs:

    for{cpu,scan_fn} [id C] (outer_out_sit_sot-0)
     >Elemwise{mul,no_inplace} [id W] (inner_out_sit_sot-0)
     > |*0-<TensorType(float64, (?,))> [id X] -> [id E] (inner_in_sit_sot-0)
     > |*1-<TensorType(float64, (?,))> [id Y] -> [id M] (inner_in_non_seqs-0)"""

    for truth, out in zip(expected_output.split("\n"), lines):
        assert truth.strip() == out.strip()


def test_debugprint_sitsot_no_extra_info():
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
    output_str = debugprint(final_result, file="str", print_op_info=False)
    lines = output_str.split("\n")

    expected_output = """Subtensor{int64} [id A]
     |Subtensor{int64::} [id B]
     | |for{cpu,scan_fn} [id C]
     | | |k [id D]
     | | |IncSubtensor{Set;:int64:} [id E]
     | | | |AllocEmpty{dtype='float64'} [id F]
     | | | | |Elemwise{add,no_inplace} [id G]
     | | | | | |k [id D]
     | | | | | |Subtensor{int64} [id H]
     | | | | |   |Shape [id I]
     | | | | |   | |Unbroadcast{0} [id J]
     | | | | |   |   |InplaceDimShuffle{x,0} [id K]
     | | | | |   |     |Elemwise{second,no_inplace} [id L]
     | | | | |   |       |A [id M]
     | | | | |   |       |InplaceDimShuffle{x} [id N]
     | | | | |   |         |TensorConstant{1.0} [id O]
     | | | | |   |ScalarConstant{0} [id P]
     | | | | |Subtensor{int64} [id Q]
     | | | |   |Shape [id R]
     | | | |   | |Unbroadcast{0} [id J]
     | | | |   |ScalarConstant{1} [id S]
     | | | |Unbroadcast{0} [id J]
     | | | |ScalarFromTensor [id T]
     | | |   |Subtensor{int64} [id H]
     | | |A [id M]
     | |ScalarConstant{1} [id U]
     |ScalarConstant{-1} [id V]

    Inner graphs:

    for{cpu,scan_fn} [id C]
     >Elemwise{mul,no_inplace} [id W]
     > |*0-<TensorType(float64, (?,))> [id X] -> [id E]
     > |*1-<TensorType(float64, (?,))> [id Y] -> [id M]"""

    for truth, out in zip(expected_output.split("\n"), lines):
        assert truth.strip() == out.strip()


@config.change_flags(floatX="float64")
def test_debugprint_nitsot():
    coefficients = vector("coefficients")
    x = scalar("x")

    max_coefficients_supported = 10000

    # Generate the components of the polynomial
    components, updates = aesara.scan(
        fn=lambda coefficient, power, free_variable: coefficient
        * (free_variable**power),
        outputs_info=None,
        sequences=[coefficients, at.arange(max_coefficients_supported)],
        non_sequences=x,
    )
    # Sum them up
    polynomial = components.sum()

    output_str = debugprint(polynomial, file="str", print_op_info=True)
    lines = output_str.split("\n")

    expected_output = """Sum{acc_dtype=float64} [id A]
     |for{cpu,scan_fn} [id B] (outer_out_nit_sot-0)
       |Elemwise{scalar_minimum,no_inplace} [id C] (outer_in_nit_sot-0)
       | |Subtensor{int64} [id D]
       | | |Shape [id E]
       | | | |Subtensor{int64::} [id F] 'coefficients[0:]'
       | | |   |coefficients [id G]
       | | |   |ScalarConstant{0} [id H]
       | | |ScalarConstant{0} [id I]
       | |Subtensor{int64} [id J]
       |   |Shape [id K]
       |   | |Subtensor{int64::} [id L]
       |   |   |ARange{dtype='int64'} [id M]
       |   |   | |TensorConstant{0} [id N]
       |   |   | |TensorConstant{10000} [id O]
       |   |   | |TensorConstant{1} [id P]
       |   |   |ScalarConstant{0} [id Q]
       |   |ScalarConstant{0} [id R]
       |Subtensor{:int64:} [id S] (outer_in_seqs-0)
       | |Subtensor{int64::} [id F] 'coefficients[0:]'
       | |ScalarFromTensor [id T]
       |   |Elemwise{scalar_minimum,no_inplace} [id C]
       |Subtensor{:int64:} [id U] (outer_in_seqs-1)
       | |Subtensor{int64::} [id L]
       | |ScalarFromTensor [id V]
       |   |Elemwise{scalar_minimum,no_inplace} [id C]
       |Elemwise{scalar_minimum,no_inplace} [id C] (outer_in_nit_sot-0)
       |x [id W] (outer_in_non_seqs-0)

    Inner graphs:

    for{cpu,scan_fn} [id B] (outer_out_nit_sot-0)
     >Elemwise{mul,no_inplace} [id X] (inner_out_nit_sot-0)
     > |*0-<TensorType(float64, ())> [id Y] -> [id S] (inner_in_seqs-0)
     > |Elemwise{pow,no_inplace} [id Z]
     >   |*2-<TensorType(float64, ())> [id BA] -> [id W] (inner_in_non_seqs-0)
     >   |*1-<TensorType(int64, ())> [id BB] -> [id U] (inner_in_seqs-1)"""

    for truth, out in zip(expected_output.split("\n"), lines):
        assert truth.strip() == out.strip()


@config.change_flags(floatX="float64")
def test_debugprint_nested_scans():
    c = dvector("c")
    n = 10

    k = iscalar("k")
    A = dvector("A")

    def compute_A_k(A, k):
        result, updates = aesara.scan(
            fn=lambda prior_result, A: prior_result * A,
            outputs_info=at.ones_like(A),
            non_sequences=A,
            n_steps=k,
        )

        A_k = result[-1]

        return A_k

    components, updates = aesara.scan(
        fn=lambda c, power, some_A, some_k: c * (compute_A_k(some_A, some_k) ** power),
        outputs_info=None,
        sequences=[c, at.arange(n)],
        non_sequences=[A, k],
    )
    final_result = components.sum()

    output_str = debugprint(final_result, file="str", print_op_info=True)
    lines = output_str.split("\n")

    expected_output = """Sum{acc_dtype=float64} [id A]
    |for{cpu,scan_fn} [id B] (outer_out_nit_sot-0)
    |Elemwise{scalar_minimum,no_inplace} [id C] (outer_in_nit_sot-0)
    | |Subtensor{int64} [id D]
    | | |Shape [id E]
    | | | |Subtensor{int64::} [id F] 'c[0:]'
    | | |   |c [id G]
    | | |   |ScalarConstant{0} [id H]
    | | |ScalarConstant{0} [id I]
    | |Subtensor{int64} [id J]
    |   |Shape [id K]
    |   | |Subtensor{int64::} [id L]
    |   |   |ARange{dtype='int64'} [id M]
    |   |   | |TensorConstant{0} [id N]
    |   |   | |TensorConstant{10} [id O]
    |   |   | |TensorConstant{1} [id P]
    |   |   |ScalarConstant{0} [id Q]
    |   |ScalarConstant{0} [id R]
    |Subtensor{:int64:} [id S] (outer_in_seqs-0)
    | |Subtensor{int64::} [id F] 'c[0:]'
    | |ScalarFromTensor [id T]
    |   |Elemwise{scalar_minimum,no_inplace} [id C]
    |Subtensor{:int64:} [id U] (outer_in_seqs-1)
    | |Subtensor{int64::} [id L]
    | |ScalarFromTensor [id V]
    |   |Elemwise{scalar_minimum,no_inplace} [id C]
    |Elemwise{scalar_minimum,no_inplace} [id C] (outer_in_nit_sot-0)
    |A [id W] (outer_in_non_seqs-0)
    |k [id X] (outer_in_non_seqs-1)

    Inner graphs:

    for{cpu,scan_fn} [id B] (outer_out_nit_sot-0)
    >Elemwise{mul,no_inplace} [id Y] (inner_out_nit_sot-0)
    > |InplaceDimShuffle{x} [id Z]
    > | |*0-<TensorType(float64, ())> [id BA] -> [id S] (inner_in_seqs-0)
    > |Elemwise{pow,no_inplace} [id BB]
    >   |Subtensor{int64} [id BC]
    >   | |Subtensor{int64::} [id BD]
    >   | | |for{cpu,scan_fn} [id BE] (outer_out_sit_sot-0)
    >   | | | |*3-<TensorType(int32, ())> [id BF] -> [id X] (inner_in_non_seqs-1) (n_steps)
    >   | | | |IncSubtensor{Set;:int64:} [id BG] (outer_in_sit_sot-0)
    >   | | | | |AllocEmpty{dtype='float64'} [id BH]
    >   | | | | | |Elemwise{add,no_inplace} [id BI]
    >   | | | | | | |*3-<TensorType(int32, ())> [id BF] -> [id X] (inner_in_non_seqs-1)
    >   | | | | | | |Subtensor{int64} [id BJ]
    >   | | | | | |   |Shape [id BK]
    >   | | | | | |   | |Unbroadcast{0} [id BL]
    >   | | | | | |   |   |InplaceDimShuffle{x,0} [id BM]
    >   | | | | | |   |     |Elemwise{second,no_inplace} [id BN]
    >   | | | | | |   |       |*2-<TensorType(float64, (?,))> [id BO] -> [id W] (inner_in_non_seqs-0)
    >   | | | | | |   |       |InplaceDimShuffle{x} [id BP]
    >   | | | | | |   |         |TensorConstant{1.0} [id BQ]
    >   | | | | | |   |ScalarConstant{0} [id BR]
    >   | | | | | |Subtensor{int64} [id BS]
    >   | | | | |   |Shape [id BT]
    >   | | | | |   | |Unbroadcast{0} [id BL]
    >   | | | | |   |ScalarConstant{1} [id BU]
    >   | | | | |Unbroadcast{0} [id BL]
    >   | | | | |ScalarFromTensor [id BV]
    >   | | | |   |Subtensor{int64} [id BJ]
    >   | | | |*2-<TensorType(float64, (?,))> [id BO] -> [id W] (inner_in_non_seqs-0) (outer_in_non_seqs-0)
    >   | | |ScalarConstant{1} [id BW]
    >   | |ScalarConstant{-1} [id BX]
    >   |InplaceDimShuffle{x} [id BY]
    >     |*1-<TensorType(int64, ())> [id BZ] -> [id U] (inner_in_seqs-1)

    for{cpu,scan_fn} [id BE] (outer_out_sit_sot-0)
    >Elemwise{mul,no_inplace} [id CA] (inner_out_sit_sot-0)
    > |*0-<TensorType(float64, (?,))> [id CB] -> [id BG] (inner_in_sit_sot-0)
    > |*1-<TensorType(float64, (?,))> [id CC] -> [id BO] (inner_in_non_seqs-0)"""

    for truth, out in zip(expected_output.split("\n"), lines):
        assert truth.strip() == out.strip()

    fg = FunctionGraph([c, k, A], [final_result])

    output_str = debugprint(
        fg, file="str", print_op_info=True, print_fgraph_inputs=True
    )
    lines = output_str.split("\n")

    expected_output = """-c [id A]
    -k [id B]
    -A [id C]
    Sum{acc_dtype=float64} [id D] 13
    |for{cpu,scan_fn} [id E] 12 (outer_out_nit_sot-0)
    |Elemwise{scalar_minimum,no_inplace} [id F] 7 (outer_in_nit_sot-0)
    | |Subtensor{int64} [id G] 6
    | | |Shape [id H] 5
    | | | |Subtensor{int64::} [id I] 'c[0:]' 4
    | | |   |c [id A]
    | | |   |ScalarConstant{0} [id J]
    | | |ScalarConstant{0} [id K]
    | |Subtensor{int64} [id L] 3
    |   |Shape [id M] 2
    |   | |Subtensor{int64::} [id N] 1
    |   |   |ARange{dtype='int64'} [id O] 0
    |   |   | |TensorConstant{0} [id P]
    |   |   | |TensorConstant{10} [id Q]
    |   |   | |TensorConstant{1} [id R]
    |   |   |ScalarConstant{0} [id S]
    |   |ScalarConstant{0} [id T]
    |Subtensor{:int64:} [id U] 11 (outer_in_seqs-0)
    | |Subtensor{int64::} [id I] 'c[0:]' 4
    | |ScalarFromTensor [id V] 10
    |   |Elemwise{scalar_minimum,no_inplace} [id F] 7
    |Subtensor{:int64:} [id W] 9 (outer_in_seqs-1)
    | |Subtensor{int64::} [id N] 1
    | |ScalarFromTensor [id X] 8
    |   |Elemwise{scalar_minimum,no_inplace} [id F] 7
    |Elemwise{scalar_minimum,no_inplace} [id F] 7 (outer_in_nit_sot-0)
    |A [id C] (outer_in_non_seqs-0)
    |k [id B] (outer_in_non_seqs-1)

    Inner graphs:

    for{cpu,scan_fn} [id E] (outer_out_nit_sot-0)
    -*0-<TensorType(float64, ())> [id Y] -> [id U] (inner_in_seqs-0)
    -*1-<TensorType(int64, ())> [id Z] -> [id W] (inner_in_seqs-1)
    -*2-<TensorType(float64, (?,))> [id BA] -> [id C] (inner_in_non_seqs-0)
    -*3-<TensorType(int32, ())> [id BB] -> [id B] (inner_in_non_seqs-1)
    >Elemwise{mul,no_inplace} [id BC] (inner_out_nit_sot-0)
    > |InplaceDimShuffle{x} [id BD]
    > | |*0-<TensorType(float64, ())> [id Y] (inner_in_seqs-0)
    > |Elemwise{pow,no_inplace} [id BE]
    >   |Subtensor{int64} [id BF]
    >   | |Subtensor{int64::} [id BG]
    >   | | |for{cpu,scan_fn} [id BH] (outer_out_sit_sot-0)
    >   | | | |*3-<TensorType(int32, ())> [id BB] (inner_in_non_seqs-1) (n_steps)
    >   | | | |IncSubtensor{Set;:int64:} [id BI] (outer_in_sit_sot-0)
    >   | | | | |AllocEmpty{dtype='float64'} [id BJ]
    >   | | | | | |Elemwise{add,no_inplace} [id BK]
    >   | | | | | | |*3-<TensorType(int32, ())> [id BB] (inner_in_non_seqs-1)
    >   | | | | | | |Subtensor{int64} [id BL]
    >   | | | | | |   |Shape [id BM]
    >   | | | | | |   | |Unbroadcast{0} [id BN]
    >   | | | | | |   |   |InplaceDimShuffle{x,0} [id BO]
    >   | | | | | |   |     |Elemwise{second,no_inplace} [id BP]
    >   | | | | | |   |       |*2-<TensorType(float64, (?,))> [id BA] (inner_in_non_seqs-0)
    >   | | | | | |   |       |InplaceDimShuffle{x} [id BQ]
    >   | | | | | |   |         |TensorConstant{1.0} [id BR]
    >   | | | | | |   |ScalarConstant{0} [id BS]
    >   | | | | | |Subtensor{int64} [id BT]
    >   | | | | |   |Shape [id BU]
    >   | | | | |   | |Unbroadcast{0} [id BN]
    >   | | | | |   |ScalarConstant{1} [id BV]
    >   | | | | |Unbroadcast{0} [id BN]
    >   | | | | |ScalarFromTensor [id BW]
    >   | | | |   |Subtensor{int64} [id BL]
    >   | | | |*2-<TensorType(float64, (?,))> [id BA] (inner_in_non_seqs-0) (outer_in_non_seqs-0)
    >   | | |ScalarConstant{1} [id BX]
    >   | |ScalarConstant{-1} [id BY]
    >   |InplaceDimShuffle{x} [id BZ]
    >     |*1-<TensorType(int64, ())> [id Z] (inner_in_seqs-1)

    for{cpu,scan_fn} [id BH] (outer_out_sit_sot-0)
    -*0-<TensorType(float64, (?,))> [id CA] -> [id BI] (inner_in_sit_sot-0)
    -*1-<TensorType(float64, (?,))> [id CB] -> [id BA] (inner_in_non_seqs-0)
    >Elemwise{mul,no_inplace} [id CC] (inner_out_sit_sot-0)
    > |*0-<TensorType(float64, (?,))> [id CA] (inner_in_sit_sot-0)
    > |*1-<TensorType(float64, (?,))> [id CB] (inner_in_non_seqs-0)"""

    for truth, out in zip(expected_output.split("\n"), lines):
        assert truth.strip() == out.strip()


@config.change_flags(floatX="float64")
def test_debugprint_mitsot():
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
    output_str = debugprint(final_result, file="str", print_op_info=True)
    lines = output_str.split("\n")

    expected_output = """Elemwise{add,no_inplace} [id A]
    |Subtensor{int64::} [id B]
    | |for{cpu,scan_fn}.0 [id C] (outer_out_mit_sot-0)
    | | |TensorConstant{5} [id D] (n_steps)
    | | |IncSubtensor{Set;:int64:} [id E] (outer_in_mit_sot-0)
    | | | |AllocEmpty{dtype='int64'} [id F]
    | | | | |Elemwise{add,no_inplace} [id G]
    | | | |   |TensorConstant{5} [id D]
    | | | |   |Subtensor{int64} [id H]
    | | | |     |Shape [id I]
    | | | |     | |Subtensor{:int64:} [id J]
    | | | |     |   |<TensorType(int64, (?,))> [id K]
    | | | |     |   |ScalarConstant{2} [id L]
    | | | |     |ScalarConstant{0} [id M]
    | | | |Subtensor{:int64:} [id J]
    | | | |ScalarFromTensor [id N]
    | | |   |Subtensor{int64} [id H]
    | | |IncSubtensor{Set;:int64:} [id O] (outer_in_mit_sot-1)
    | |   |AllocEmpty{dtype='int64'} [id P]
    | |   | |Elemwise{add,no_inplace} [id Q]
    | |   |   |TensorConstant{5} [id D]
    | |   |   |Subtensor{int64} [id R]
    | |   |     |Shape [id S]
    | |   |     | |Subtensor{:int64:} [id T]
    | |   |     |   |<TensorType(int64, (?,))> [id U]
    | |   |     |   |ScalarConstant{2} [id V]
    | |   |     |ScalarConstant{0} [id W]
    | |   |Subtensor{:int64:} [id T]
    | |   |ScalarFromTensor [id X]
    | |     |Subtensor{int64} [id R]
    | |ScalarConstant{2} [id Y]
    |Subtensor{int64::} [id Z]
    |for{cpu,scan_fn}.1 [id C] (outer_out_mit_sot-1)
    |ScalarConstant{2} [id BA]

    Inner graphs:

    for{cpu,scan_fn}.0 [id C] (outer_out_mit_sot-0)
    >Elemwise{add,no_inplace} [id BB] (inner_out_mit_sot-0)
    > |*1-<TensorType(int64, ())> [id BC] -> [id E] (inner_in_mit_sot-0-1)
    > |*0-<TensorType(int64, ())> [id BD] -> [id E] (inner_in_mit_sot-0-0)
    >Elemwise{add,no_inplace} [id BE] (inner_out_mit_sot-1)
    > |*3-<TensorType(int64, ())> [id BF] -> [id O] (inner_in_mit_sot-1-1)
    > |*2-<TensorType(int64, ())> [id BG] -> [id O] (inner_in_mit_sot-1-0)

    for{cpu,scan_fn}.1 [id C] (outer_out_mit_sot-1)
    >Elemwise{add,no_inplace} [id BB] (inner_out_mit_sot-0)
    >Elemwise{add,no_inplace} [id BE] (inner_out_mit_sot-1)"""

    for truth, out in zip(expected_output.split("\n"), lines):
        assert truth.strip() == out.strip()


@config.change_flags(floatX="float64")
def test_debugprint_mitmot():
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

    output_str = debugprint(final_result, file="str", print_op_info=True)
    lines = output_str.split("\n")

    expected_output = """Subtensor{int64} [id A]
    |for{cpu,grad_of_scan_fn}.1 [id B] (outer_out_sit_sot-0)
    | |Elemwise{sub,no_inplace} [id C] (n_steps)
    | | |Subtensor{int64} [id D]
    | | | |Shape [id E]
    | | | | |for{cpu,scan_fn} [id F] (outer_out_sit_sot-0)
    | | | |   |k [id G] (n_steps)
    | | | |   |IncSubtensor{Set;:int64:} [id H] (outer_in_sit_sot-0)
    | | | |   | |AllocEmpty{dtype='float64'} [id I]
    | | | |   | | |Elemwise{add,no_inplace} [id J]
    | | | |   | | | |k [id G]
    | | | |   | | | |Subtensor{int64} [id K]
    | | | |   | | |   |Shape [id L]
    | | | |   | | |   | |Unbroadcast{0} [id M]
    | | | |   | | |   |   |InplaceDimShuffle{x,0} [id N]
    | | | |   | | |   |     |Elemwise{second,no_inplace} [id O]
    | | | |   | | |   |       |A [id P]
    | | | |   | | |   |       |InplaceDimShuffle{x} [id Q]
    | | | |   | | |   |         |TensorConstant{1.0} [id R]
    | | | |   | | |   |ScalarConstant{0} [id S]
    | | | |   | | |Subtensor{int64} [id T]
    | | | |   | |   |Shape [id U]
    | | | |   | |   | |Unbroadcast{0} [id M]
    | | | |   | |   |ScalarConstant{1} [id V]
    | | | |   | |Unbroadcast{0} [id M]
    | | | |   | |ScalarFromTensor [id W]
    | | | |   |   |Subtensor{int64} [id K]
    | | | |   |A [id P] (outer_in_non_seqs-0)
    | | | |ScalarConstant{0} [id X]
    | | |TensorConstant{1} [id Y]
    | |Subtensor{:int64:} [id Z] (outer_in_seqs-0)
    | | |Subtensor{::int64} [id BA]
    | | | |Subtensor{:int64:} [id BB]
    | | | | |for{cpu,scan_fn} [id F] (outer_out_sit_sot-0)
    | | | | |ScalarConstant{-1} [id BC]
    | | | |ScalarConstant{-1} [id BD]
    | | |ScalarFromTensor [id BE]
    | |   |Elemwise{sub,no_inplace} [id C]
    | |Subtensor{:int64:} [id BF] (outer_in_seqs-1)
    | | |Subtensor{:int64:} [id BG]
    | | | |Subtensor{::int64} [id BH]
    | | | | |for{cpu,scan_fn} [id F] (outer_out_sit_sot-0)
    | | | | |ScalarConstant{-1} [id BI]
    | | | |ScalarConstant{-1} [id BJ]
    | | |ScalarFromTensor [id BK]
    | |   |Elemwise{sub,no_inplace} [id C]
    | |Subtensor{::int64} [id BL] (outer_in_mit_mot-0)
    | | |IncSubtensor{Inc;int64::} [id BM]
    | | | |Elemwise{second,no_inplace} [id BN]
    | | | | |for{cpu,scan_fn} [id F] (outer_out_sit_sot-0)
    | | | | |InplaceDimShuffle{x,x} [id BO]
    | | | |   |TensorConstant{0.0} [id BP]
    | | | |IncSubtensor{Inc;int64} [id BQ]
    | | | | |Elemwise{second,no_inplace} [id BR]
    | | | | | |Subtensor{int64::} [id BS]
    | | | | | | |for{cpu,scan_fn} [id F] (outer_out_sit_sot-0)
    | | | | | | |ScalarConstant{1} [id BT]
    | | | | | |InplaceDimShuffle{x,x} [id BU]
    | | | | |   |TensorConstant{0.0} [id BV]
    | | | | |Elemwise{second} [id BW]
    | | | | | |Subtensor{int64} [id BX]
    | | | | | | |Subtensor{int64::} [id BS]
    | | | | | | |ScalarConstant{-1} [id BY]
    | | | | | |InplaceDimShuffle{x} [id BZ]
    | | | | |   |Elemwise{second,no_inplace} [id CA]
    | | | | |     |Sum{acc_dtype=float64} [id CB]
    | | | | |     | |Subtensor{int64} [id BX]
    | | | | |     |TensorConstant{1.0} [id CC]
    | | | | |ScalarConstant{-1} [id BY]
    | | | |ScalarConstant{1} [id BT]
    | | |ScalarConstant{-1} [id CD]
    | |Alloc [id CE] (outer_in_sit_sot-0)
    | | |TensorConstant{0.0} [id CF]
    | | |Elemwise{add,no_inplace} [id CG]
    | | | |Elemwise{sub,no_inplace} [id C]
    | | | |TensorConstant{1} [id CH]
    | | |Subtensor{int64} [id CI]
    | |   |Shape [id CJ]
    | |   | |A [id P]
    | |   |ScalarConstant{0} [id CK]
    | |A [id P] (outer_in_non_seqs-0)
    |ScalarConstant{-1} [id CL]

    Inner graphs:

    for{cpu,grad_of_scan_fn}.1 [id B] (outer_out_sit_sot-0)
    >Elemwise{add,no_inplace} [id CM] (inner_out_mit_mot-0-0)
    > |Elemwise{mul} [id CN]
    > | |*2-<TensorType(float64, (?,))> [id CO] -> [id BL] (inner_in_mit_mot-0-0)
    > | |*5-<TensorType(float64, (?,))> [id CP] -> [id P] (inner_in_non_seqs-0)
    > |*3-<TensorType(float64, (?,))> [id CQ] -> [id BL] (inner_in_mit_mot-0-1)
    >Elemwise{add,no_inplace} [id CR] (inner_out_sit_sot-0)
    > |Elemwise{mul} [id CS]
    > | |*2-<TensorType(float64, (?,))> [id CO] -> [id BL] (inner_in_mit_mot-0-0)
    > | |*0-<TensorType(float64, (?,))> [id CT] -> [id Z] (inner_in_seqs-0)
    > |*4-<TensorType(float64, (?,))> [id CU] -> [id CE] (inner_in_sit_sot-0)

    for{cpu,scan_fn} [id F] (outer_out_sit_sot-0)
    >Elemwise{mul,no_inplace} [id CV] (inner_out_sit_sot-0)
    > |*0-<TensorType(float64, (?,))> [id CT] -> [id H] (inner_in_sit_sot-0)
    > |*1-<TensorType(float64, (?,))> [id CW] -> [id P] (inner_in_non_seqs-0)"""

    for truth, out in zip(expected_output.split("\n"), lines):
        assert truth.strip() == out.strip()


def test_debugprint_compiled_fn():
    M = at.tensor(np.float64, shape=(20000, 2, 2))
    one = at.as_tensor(1, dtype=np.int64)
    zero = at.as_tensor(0, dtype=np.int64)

    def no_shared_fn(n, x_tm1, M):
        p = M[n, x_tm1]
        return at.switch(at.lt(zero, p[0]), one, zero)

    out, updates = aesara.scan(
        no_shared_fn,
        outputs_info=[{"initial": zero, "taps": [-1]}],
        sequences=[at.arange(M.shape[0])],
        non_sequences=[M],
        allow_gc=False,
        mode="FAST_RUN",
    )

    # In this case, `debugprint` should print the compiled inner-graph
    # (i.e. from `Scan._fn`)
    out = aesara.function([M], out, updates=updates, mode="FAST_RUN")

    expected_output = """forall_inplace,cpu,scan_fn} [id A] 2 (outer_out_sit_sot-0)
    |TensorConstant{20000} [id B] (n_steps)
    |TensorConstant{[    0    ..998 19999]} [id C] (outer_in_seqs-0)
    |IncSubtensor{InplaceSet;:int64:} [id D] 1 (outer_in_sit_sot-0)
    | |AllocEmpty{dtype='int64'} [id E] 0
    | | |TensorConstant{20000} [id B]
    | |TensorConstant{(1,) of 0} [id F]
    | |ScalarConstant{1} [id G]
    |<TensorType(float64, (20000, 2, 2))> [id H] (outer_in_non_seqs-0)

    Inner graphs:

    forall_inplace,cpu,scan_fn} [id A] (outer_out_sit_sot-0)
    >Elemwise{Composite{Switch(LT(i0, i1), i2, i0)}} [id I] (inner_out_sit_sot-0)
    > |TensorConstant{0} [id J]
    > |Subtensor{int64, int64, uint8} [id K]
    > | |*2-<TensorType(float64, (20000, 2, 2))> [id L] -> [id H] (inner_in_non_seqs-0)
    > | |ScalarFromTensor [id M]
    > | | |*0-<TensorType(int64, ())> [id N] -> [id C] (inner_in_seqs-0)
    > | |ScalarFromTensor [id O]
    > | | |*1-<TensorType(int64, ())> [id P] -> [id D] (inner_in_sit_sot-0)
    > | |ScalarConstant{0} [id Q]
    > |TensorConstant{1} [id R]
    """

    output_str = debugprint(out, file="str", print_op_info=True)
    lines = output_str.split("\n")

    for truth, out in zip(expected_output.split("\n"), lines):
        assert truth.strip() == out.strip()


@pytest.mark.skipif(not pydot_imported, reason="pydot not available")
def test_pydotprint():
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
