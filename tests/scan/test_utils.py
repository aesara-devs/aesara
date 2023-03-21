from copy import copy

import numpy as np
import pytest

import aesara
from aesara import tensor as at
from aesara.scan.utils import ScanArgs


@pytest.fixture(scope="module", autouse=True)
def set_aesara_flags():
    with aesara.config.change_flags(cxx="", mode="FAST_COMPILE"):
        yield


def create_test_hmm():
    srng = at.random.RandomStream()

    N_tt = at.iscalar("N")
    N_tt.tag.test_value = 10
    M_tt = at.iscalar("M")
    M_tt.tag.test_value = 2

    mus_tt = at.matrix("mus")
    mus_tt.tag.test_value = np.stack(
        [np.arange(0.0, 10), np.arange(0.0, -10, -1)], axis=-1
    ).astype(aesara.config.floatX)

    sigmas_tt = at.ones((N_tt,))
    sigmas_tt.name = "sigmas"

    pi_0_rv = srng.dirichlet(at.ones((M_tt,)), name="pi_0")
    Gamma_rv = srng.dirichlet(at.ones((M_tt, M_tt)), name="Gamma")

    S_0_rv = srng.categorical(pi_0_rv, name="S_0")

    def scan_fn(mus_t, sigma_t, S_tm1, Gamma_t):
        S_t = srng.categorical(Gamma_t[S_tm1], name="S_t")
        Y_t = srng.normal(mus_t[S_t], sigma_t, name="Y_t")
        return S_t, Y_t

    (S_rv, Y_rv), scan_updates = aesara.scan(
        fn=scan_fn,
        sequences=[mus_tt, sigmas_tt],
        non_sequences=[Gamma_rv],
        outputs_info=[{"initial": S_0_rv, "taps": [-1]}, {}],
        strict=True,
        name="scan_rv",
    )
    Y_rv.name = "Y_rv"

    scan_op = Y_rv.owner.op
    scan_args = ScanArgs.from_node(Y_rv.owner)

    Gamma_in = scan_args.inner_in_non_seqs[0]
    Y_t = scan_args.inner_out_nit_sot[0]
    mus_t = scan_args.inner_in_seqs[0]
    sigmas_t = scan_args.inner_in_seqs[1]
    S_t = scan_args.inner_out_sit_sot[0]
    rng_in = scan_args.inner_out_shared[0]

    mus_in = Y_rv.owner.inputs[1]
    mus_in.name = "mus_in"
    sigmas_in = Y_rv.owner.inputs[2]
    sigmas_in.name = "sigmas_in"

    # The output `S_rv` is really `S_rv[1:]`, so we have to extract the actual
    # `Scan` output: `S_rv`.
    S_in = S_rv.owner.inputs[0]
    S_in.name = "S_in"

    return locals()


def test_ScanArgs():
    with pytest.raises(TypeError):
        ScanArgs.from_node(at.ones(2).owner)

    hmm_model_env = create_test_hmm()
    scan_args = hmm_model_env["scan_args"]
    scan_op = hmm_model_env["scan_op"]

    # Make sure we can get alternate variables
    test_v = scan_args.outer_out_sit_sot[0]
    alt_test_v = scan_args.get_alt_field(test_v, "inner_out")
    assert alt_test_v == scan_args.inner_out_sit_sot[0]

    alt_test_v = scan_args.get_alt_field(test_v, "outer_in")
    assert alt_test_v == scan_args.outer_in_sit_sot[0]

    # Check the `__repr__` and `__str__`
    scan_args_repr = repr(scan_args)
    # Just make sure it doesn't err-out
    assert scan_args_repr.startswith("ScanArgs")

    # Check the properties that allow us to use
    # `Scan.get_oinp_iinp_iout_oout_mappings` as-is to implement
    # `ScanArgs.var_mappings`
    assert scan_args.n_nit_sot == scan_op.info.n_nit_sot
    assert scan_args.n_mit_mot == scan_op.info.n_mit_mot
    # The `scan_args` base class always clones the inner-graph;
    # here we make sure it doesn't (and that all the inputs are the same)
    assert scan_args.inputs == scan_op.inner_inputs
    assert scan_args.info == scan_op.info

    # Check that `ScanArgs.find_among_fields` works
    test_v = scan_op.inner_seqs(scan_op.inner_inputs)[1]
    field_info = scan_args.find_among_fields(test_v)
    assert field_info.name == "inner_in_seqs"
    assert field_info.index == 1
    assert field_info.inner_index is None
    assert scan_args.inner_inputs[field_info.agg_index] == test_v

    test_l = scan_op.inner_non_seqs(scan_op.inner_inputs)
    # We didn't index this argument, so it's a `list` (i.e. bad input)
    field_info = scan_args.find_among_fields(test_l)
    assert field_info is None

    test_v = test_l[0]
    field_info = scan_args.find_among_fields(test_v)
    assert field_info.name == "inner_in_non_seqs"
    assert field_info.index == 0
    assert field_info.inner_index is None
    assert scan_args.inner_inputs[field_info.agg_index] == test_v

    scan_args_copy = copy(scan_args)
    assert scan_args_copy is not scan_args
    assert scan_args_copy == scan_args

    assert scan_args_copy != test_v
    scan_args_copy.outer_in_seqs.pop()
    assert scan_args_copy != scan_args


def test_ScanArgs_basics_mit_sot():
    srng = at.random.RandomStream()

    N_tt = at.iscalar("N")
    N_tt.tag.test_value = 10
    M_tt = at.iscalar("M")
    M_tt.tag.test_value = 2

    mus_tt = at.matrix("mus")
    mus_tt.tag.test_value = np.stack(
        [np.arange(0.0, 10), np.arange(0.0, -10, -1)], axis=-1
    ).astype(aesara.config.floatX)

    sigmas_tt = at.ones((N_tt,))
    sigmas_tt.name = "sigmas"

    pi_0_rv = srng.dirichlet(at.ones((M_tt,)), name="pi_0")
    Gamma_rv = srng.dirichlet(at.ones((M_tt, M_tt)), name="Gamma")

    S_0_rv = srng.categorical(pi_0_rv, name="S_0")

    def scan_fn(mus_t, sigma_t, S_tm2, S_tm1, Gamma_t):
        S_t = srng.categorical(Gamma_t[S_tm2], name="S_t")
        Y_t = srng.normal(mus_t[S_tm1], sigma_t, name="Y_t")
        return S_t, Y_t

    (S_rv, Y_rv), scan_updates = aesara.scan(
        fn=scan_fn,
        sequences=[mus_tt, sigmas_tt],
        non_sequences=[Gamma_rv],
        outputs_info=[{"initial": at.stack([S_0_rv, S_0_rv]), "taps": [-2, -1]}, {}],
        strict=True,
        name="scan_rv",
    )
    # Adding names should make output easier to read
    Y_rv.name = "Y_rv"
    # This `S_rv` outer-output is actually a `Subtensor` of the "real" output
    S_rv = S_rv.owner.inputs[0]
    S_rv.name = "S_rv"
    mus_in = Y_rv.owner.inputs[1]
    mus_in.name = "mus_in"
    sigmas_in = Y_rv.owner.inputs[2]
    sigmas_in.name = "sigmas_in"

    scan_args = ScanArgs.from_node(Y_rv.owner)

    test_v = scan_args.inner_in_mit_sot[0][1]
    field_info = scan_args.find_among_fields(test_v)

    assert field_info.name == "inner_in_mit_sot"
    assert field_info.index == 0
    assert field_info.inner_index == 1
    assert field_info.agg_index == 3

    rm_info = scan_args._remove_from_fields(at.ones(2))
    assert rm_info is None

    rm_info = scan_args._remove_from_fields(test_v)

    assert rm_info.name == "inner_in_mit_sot"
    assert rm_info.index == 0
    assert rm_info.inner_index == 1
    assert rm_info.agg_index == 3


def test_ScanArgs_remove_inner_input():
    hmm_model_env = create_test_hmm()
    scan_args = hmm_model_env["scan_args"]
    hmm_model_env["scan_op"]
    Y_t = hmm_model_env["Y_t"]
    Y_rv = hmm_model_env["Y_rv"]
    sigmas_in = hmm_model_env["sigmas_in"]
    sigmas_t = hmm_model_env["sigmas_t"]
    Gamma_rv = hmm_model_env["Gamma_rv"]
    Gamma_in = hmm_model_env["Gamma_in"]
    hmm_model_env["S_rv"]
    S_in = hmm_model_env["S_in"]
    S_t = hmm_model_env["S_t"]
    rng_in = hmm_model_env["rng_in"]
    scan_updates = hmm_model_env["scan_updates"]

    # Check `ScanArgs.remove_from_fields` by removing `sigmas[t]` (i.e. the
    # inner-graph input)
    scan_args_copy = copy(scan_args)
    test_v = sigmas_t

    rm_info = scan_args_copy.remove_from_fields(test_v, rm_dependents=False)
    removed_nodes, _ = zip(*rm_info)

    assert sigmas_t in removed_nodes
    assert sigmas_t not in scan_args_copy.inner_in_seqs
    assert Y_t not in removed_nodes
    assert len(scan_args_copy.inner_out_nit_sot) == 1

    scan_args_copy = copy(scan_args)
    test_v = sigmas_t

    # This removal includes dependents
    rm_info = scan_args_copy.remove_from_fields(test_v, rm_dependents=True)
    removed_nodes, _ = zip(*rm_info)

    # `sigmas[t]` (i.e. inner-graph input) should be gone
    assert sigmas_t in removed_nodes
    assert sigmas_t not in scan_args_copy.inner_in_seqs
    # `Y_t` (i.e. inner-graph output) should be gone
    assert Y_t in removed_nodes
    assert len(scan_args_copy.inner_out_nit_sot) == 0
    # `Y_rv` (i.e. outer-graph output) should be gone
    assert Y_rv in removed_nodes
    assert Y_rv not in scan_args_copy.outer_outputs
    assert len(scan_args_copy.outer_out_nit_sot) == 0
    # `sigmas_in` (i.e. outer-graph input) should be gone
    assert sigmas_in in removed_nodes
    assert test_v not in scan_args_copy.inner_in_seqs

    # These shouldn't have been removed
    assert S_t in scan_args_copy.inner_out_sit_sot
    assert S_in in scan_args_copy.outer_out_sit_sot
    assert Gamma_in in scan_args_copy.inner_in_non_seqs
    assert Gamma_rv in scan_args_copy.outer_in_non_seqs
    assert rng_in in scan_args_copy.inner_out_shared
    assert list(scan_updates.values()) == scan_args.outer_out_shared

    # The other `Y_rv`-related inputs currently aren't removed, even though
    # they're no longer needed.
    # TODO: Would be nice if we did this, too
    # assert len(scan_args_copy.outer_in_seqs) == 0
    # TODO: Would be nice if we did this, too
    # assert len(scan_args_copy.inner_in_seqs) == 0

    # We shouldn't be able to remove the removed node
    with pytest.raises(ValueError):
        rm_info = scan_args_copy.remove_from_fields(test_v, rm_dependents=True)


def test_ScanArgs_remove_outer_input():
    hmm_model_env = create_test_hmm()
    scan_args = hmm_model_env["scan_args"]
    hmm_model_env["scan_op"]
    Y_t = hmm_model_env["Y_t"]
    Y_rv = hmm_model_env["Y_rv"]
    sigmas_in = hmm_model_env["sigmas_in"]
    sigmas_t = hmm_model_env["sigmas_t"]
    Gamma_rv = hmm_model_env["Gamma_rv"]
    Gamma_in = hmm_model_env["Gamma_in"]
    hmm_model_env["S_rv"]
    S_in = hmm_model_env["S_in"]
    S_t = hmm_model_env["S_t"]
    rng_in = hmm_model_env["rng_in"]
    scan_updates = hmm_model_env["scan_updates"]

    # Remove `sigmas` (i.e. the outer-input)
    scan_args_copy = copy(scan_args)
    test_v = sigmas_in
    rm_info = scan_args_copy.remove_from_fields(test_v, rm_dependents=True)
    removed_nodes, _ = zip(*rm_info)

    # `sigmas_in` (i.e. outer-graph input) should be gone
    assert scan_args.outer_in_seqs[-1] in removed_nodes
    assert test_v not in scan_args_copy.inner_in_seqs

    # `sigmas[t]` should be gone
    assert sigmas_t in removed_nodes
    assert sigmas_t not in scan_args_copy.inner_in_seqs

    # `Y_t` (i.e. inner-graph output) should be gone
    assert Y_t in removed_nodes
    assert len(scan_args_copy.inner_out_nit_sot) == 0

    # `Y_rv` (i.e. outer-graph output) should be gone
    assert Y_rv not in scan_args_copy.outer_outputs
    assert len(scan_args_copy.outer_out_nit_sot) == 0

    assert S_t in scan_args_copy.inner_out_sit_sot
    assert S_in in scan_args_copy.outer_out_sit_sot
    assert Gamma_in in scan_args_copy.inner_in_non_seqs
    assert Gamma_rv in scan_args_copy.outer_in_non_seqs
    assert rng_in in scan_args_copy.inner_out_shared
    assert list(scan_updates.values()) == scan_args.outer_out_shared


def test_ScanArgs_remove_inner_output():
    hmm_model_env = create_test_hmm()
    scan_args = hmm_model_env["scan_args"]
    hmm_model_env["scan_op"]
    Y_t = hmm_model_env["Y_t"]
    Y_rv = hmm_model_env["Y_rv"]
    hmm_model_env["sigmas_in"]
    hmm_model_env["sigmas_t"]
    Gamma_rv = hmm_model_env["Gamma_rv"]
    Gamma_in = hmm_model_env["Gamma_in"]
    hmm_model_env["S_rv"]
    S_in = hmm_model_env["S_in"]
    S_t = hmm_model_env["S_t"]
    rng_in = hmm_model_env["rng_in"]
    scan_updates = hmm_model_env["scan_updates"]

    # Remove `Y_t` (i.e. the inner-output)
    scan_args_copy = copy(scan_args)
    test_v = Y_t
    rm_info = scan_args_copy.remove_from_fields(test_v, rm_dependents=True)
    removed_nodes, _ = zip(*rm_info)

    # `Y_t` (i.e. inner-graph output) should be gone
    assert Y_t in removed_nodes
    assert len(scan_args_copy.inner_out_nit_sot) == 0

    # `Y_rv` (i.e. outer-graph output) should be gone
    assert Y_rv not in scan_args_copy.outer_outputs
    assert len(scan_args_copy.outer_out_nit_sot) == 0

    assert S_t in scan_args_copy.inner_out_sit_sot
    assert S_in in scan_args_copy.outer_out_sit_sot
    assert Gamma_in in scan_args_copy.inner_in_non_seqs
    assert Gamma_rv in scan_args_copy.outer_in_non_seqs
    assert rng_in in scan_args_copy.inner_out_shared
    assert list(scan_updates.values()) == scan_args.outer_out_shared


def test_ScanArgs_remove_outer_output():
    hmm_model_env = create_test_hmm()
    scan_args = hmm_model_env["scan_args"]
    hmm_model_env["scan_op"]
    Y_t = hmm_model_env["Y_t"]
    Y_rv = hmm_model_env["Y_rv"]
    hmm_model_env["sigmas_in"]
    hmm_model_env["sigmas_t"]
    Gamma_rv = hmm_model_env["Gamma_rv"]
    Gamma_in = hmm_model_env["Gamma_in"]
    S_in = hmm_model_env["S_in"]
    S_t = hmm_model_env["S_t"]
    rng_in = hmm_model_env["rng_in"]
    scan_updates = hmm_model_env["scan_updates"]

    # Remove `Y_rv` (i.e. a nit-sot outer-output)
    scan_args_copy = copy(scan_args)
    test_v = Y_rv
    rm_info = scan_args_copy.remove_from_fields(test_v, rm_dependents=True)
    removed_nodes, _ = zip(*rm_info)

    # `Y_t` (i.e. inner-graph output) should be gone
    assert Y_t in removed_nodes
    assert len(scan_args_copy.inner_out_nit_sot) == 0

    # `Y_rv` (i.e. outer-graph output) should be gone
    assert Y_rv not in scan_args_copy.outer_outputs
    assert len(scan_args_copy.outer_out_nit_sot) == 0

    assert S_t in scan_args_copy.inner_out_sit_sot
    assert S_in in scan_args_copy.outer_out_sit_sot
    assert Gamma_in in scan_args_copy.inner_in_non_seqs
    assert Gamma_rv in scan_args_copy.outer_in_non_seqs
    assert rng_in in scan_args_copy.inner_out_shared
    assert list(scan_updates.values()) == scan_args.outer_out_shared


def test_ScanArgs_remove_nonseq_outer_input():
    hmm_model_env = create_test_hmm()
    scan_args = hmm_model_env["scan_args"]
    hmm_model_env["scan_op"]
    Y_t = hmm_model_env["Y_t"]
    Y_rv = hmm_model_env["Y_rv"]
    mus_in = hmm_model_env["mus_in"]
    mus_t = hmm_model_env["mus_t"]
    sigmas_in = hmm_model_env["sigmas_in"]
    sigmas_t = hmm_model_env["sigmas_t"]
    Gamma_rv = hmm_model_env["Gamma_rv"]
    Gamma_in = hmm_model_env["Gamma_in"]
    S_in = hmm_model_env["S_in"]
    S_t = hmm_model_env["S_t"]
    rng_in = hmm_model_env["rng_in"]

    # Remove `Gamma` (i.e. a non-sequence outer-input)
    scan_args_copy = copy(scan_args)
    test_v = Gamma_rv
    rm_info = scan_args_copy.remove_from_fields(test_v, rm_dependents=True)
    removed_nodes, _ = zip(*rm_info)

    assert Gamma_rv in removed_nodes
    assert Gamma_in in removed_nodes
    assert S_in in removed_nodes
    assert S_t in removed_nodes
    assert Y_t in removed_nodes
    assert Y_rv in removed_nodes

    assert mus_in in scan_args_copy.outer_in_seqs
    assert sigmas_in in scan_args_copy.outer_in_seqs
    assert mus_t in scan_args_copy.inner_in_seqs
    assert sigmas_t in scan_args_copy.inner_in_seqs
    assert rng_in not in scan_args_copy.inner_out_shared
    assert not scan_args_copy.outer_out_shared


def test_ScanArgs_remove_nonseq_inner_input():
    hmm_model_env = create_test_hmm()
    scan_args = hmm_model_env["scan_args"]
    hmm_model_env["scan_op"]
    hmm_model_env["Y_t"]
    hmm_model_env["Y_rv"]
    mus_in = hmm_model_env["mus_in"]
    mus_t = hmm_model_env["mus_t"]
    sigmas_in = hmm_model_env["sigmas_in"]
    sigmas_t = hmm_model_env["sigmas_t"]
    Gamma_rv = hmm_model_env["Gamma_rv"]
    Gamma_in = hmm_model_env["Gamma_in"]
    S_in = hmm_model_env["S_in"]
    S_t = hmm_model_env["S_t"]
    rng_in = hmm_model_env["rng_in"]
    scan_updates = hmm_model_env["scan_updates"]

    # Remove `Gamma` (i.e. a non-sequence inner-input)
    scan_args_copy = copy(scan_args)
    test_v = Gamma_in
    rm_info = scan_args_copy.remove_from_fields(test_v, rm_dependents=True)
    removed_nodes, _ = zip(*rm_info)

    assert Gamma_in in removed_nodes
    assert Gamma_rv in removed_nodes
    assert S_in in removed_nodes
    assert S_t in removed_nodes

    assert mus_in in scan_args_copy.outer_in_seqs
    assert sigmas_in in scan_args_copy.outer_in_seqs
    assert mus_t in scan_args_copy.inner_in_seqs
    assert sigmas_t in scan_args_copy.inner_in_seqs
    assert rng_in not in scan_args_copy.inner_out_shared
    assert list(scan_updates.values()) == scan_args.outer_out_shared


def test_ScanArgs_remove_shared_inner_output():
    hmm_model_env = create_test_hmm()
    scan_args = hmm_model_env["scan_args"]
    hmm_model_env["scan_op"]
    hmm_model_env["Y_t"]
    Y_rv = hmm_model_env["Y_rv"]
    mus_in = hmm_model_env["mus_in"]
    mus_t = hmm_model_env["mus_t"]
    sigmas_in = hmm_model_env["sigmas_in"]
    sigmas_t = hmm_model_env["sigmas_t"]
    hmm_model_env["Gamma_rv"]
    hmm_model_env["Gamma_in"]
    S_in = hmm_model_env["S_in"]
    hmm_model_env["S_t"]
    rng_in = hmm_model_env["rng_in"]

    # Remove a shared inner-output
    scan_update = scan_args.inner_out_shared[0]
    scan_args_copy = copy(scan_args)
    rm_info = scan_args_copy.remove_from_fields(scan_update, rm_dependents=True)
    removed_nodes, _ = zip(*rm_info)

    assert rng_in in removed_nodes
    assert all(v in removed_nodes for v in scan_args.inner_out_shared)
    assert Y_rv in removed_nodes
    assert S_in in removed_nodes

    assert sigmas_in in scan_args_copy.outer_in_seqs
    assert sigmas_t in scan_args_copy.inner_in_seqs
    assert mus_in in scan_args_copy.outer_in_seqs
    assert mus_t in scan_args_copy.inner_in_seqs
