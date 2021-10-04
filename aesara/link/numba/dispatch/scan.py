import numba
import numpy as np

from aesara.graph.fg import FunctionGraph
from aesara.link.numba.dispatch.basic import create_tuple_string, numba_funcify
from aesara.link.utils import compile_function_src
from aesara.scan.op import Scan


def idx_to_str(idx):
    res = "[i"
    if idx < 0:
        res += str(idx)
    elif idx > 0:
        res += "+" + str(idx)
    return res + "]"


@numba_funcify.register(Scan)
def numba_funcify_Scan(op, node, **kwargs):
    inner_fg = FunctionGraph(op.inputs, op.outputs)
    numba_aet_inner_func = numba.njit(numba_funcify(inner_fg, **kwargs))

    n_seqs = op.info.n_seqs
    n_mit_mot = op.info.n_mit_mot
    n_mit_sot = op.info.n_mit_sot
    n_nit_sot = op.info.n_nit_sot
    n_sit_sot = op.info.n_sit_sot
    tap_array = op.info.tap_array
    n_shared_outs = op.info.n_shared_outs
    mit_mot_in_taps = tuple(tap_array[:n_mit_mot])
    mit_sot_in_taps = tuple(tap_array[n_mit_mot : n_mit_mot + n_mit_sot])

    p_in_mit_mot = n_seqs
    p_in_mit_sot = p_in_mit_mot + n_mit_mot
    p_in_sit_sot = p_in_mit_sot + n_mit_sot
    p_outer_in_shared = p_in_sit_sot + n_sit_sot
    p_outer_in_nit_sot = p_outer_in_shared + n_shared_outs
    p_outer_in_non_seqs = p_outer_in_nit_sot + n_nit_sot

    input_names = [n.auto_name for n in node.inputs[1:]]
    outer_in_seqs_names = input_names[:n_seqs]
    outer_in_mit_mot_names = input_names[p_in_mit_mot : p_in_mit_mot + n_mit_mot]
    outer_in_mit_sot_names = input_names[p_in_mit_sot : p_in_mit_sot + n_mit_sot]
    outer_in_sit_sot_names = input_names[p_in_sit_sot : p_in_sit_sot + n_sit_sot]
    outer_in_shared_names = input_names[
        p_outer_in_shared : p_outer_in_shared + n_shared_outs
    ]
    outer_in_nit_sot_names = input_names[
        p_outer_in_nit_sot : p_outer_in_nit_sot + n_nit_sot
    ]
    outer_in_feedback_names = input_names[n_seqs:p_outer_in_non_seqs]
    outer_in_non_seqs_names = input_names[p_outer_in_non_seqs:]

    inner_in_indexed = []
    inner_out_indexed = []
    allocate_mem_to_nit_sot = ""

    for _name in outer_in_seqs_names:
        # TODO:Index sould be updating according to sequence's taps
        index = "[i]"
        inner_in_indexed.append(_name + index)

    name_to_input_map = dict(zip(input_names, node.inputs[1:]))
    mit_sot_name_to_taps = dict(zip(outer_in_mit_sot_names, mit_sot_in_taps))
    for _name in outer_in_feedback_names:
        if _name in outer_in_mit_sot_names:
            curr_taps = mit_sot_name_to_taps[_name]
            min_tap = min(*curr_taps)

            for _tap in curr_taps:
                index = idx_to_str(_tap - min_tap)
                inner_in_indexed.append(_name + index)

            index = idx_to_str(-min_tap)
            inner_out_indexed.append(_name + index)

        if _name in outer_in_sit_sot_names:
            # TODO: Input according to taps
            index = "[i]"
            inner_in_indexed.append(_name + index)
            index = "[i+1]"
            inner_out_indexed.append(_name + index)

        if _name in outer_in_nit_sot_names:
            # TODO: Allocate this properly
            index = "[i]"
            inner_out_indexed.append(_name + index)
            allocate_mem_to_nit_sot += f"""
    {_name} = np.zeros(n_steps)
"""
    # The non_seqs are passed to inner function as-is
    inner_in_indexed += outer_in_non_seqs_names

    global_env = locals()
    global_env["np"] = np

    scan_op_src = f"""
def scan(n_steps, {", ".join(input_names)}):
    outer_in_seqs = {create_tuple_string(outer_in_seqs_names)}
    outer_in_mit_sot = {create_tuple_string(outer_in_mit_sot_names)}
    outer_in_sit_sot = {create_tuple_string(outer_in_sit_sot_names)}
    outer_in_shared = {create_tuple_string(outer_in_shared_names)}
    outer_in_non_seqs = {create_tuple_string(outer_in_non_seqs_names)}
{allocate_mem_to_nit_sot}
    outer_in_nit_sot = {create_tuple_string(outer_in_nit_sot_names)}

    for i in range(n_steps):
        inner_args = {create_tuple_string(inner_in_indexed)}
        {create_tuple_string(inner_out_indexed)} = numba_aet_inner_func(*inner_args)

    return (
        outer_in_mit_sot +
        outer_in_sit_sot +
        outer_in_nit_sot
        )
    """
    scalar_op_fn = compile_function_src(scan_op_src, "scan", global_env)

    return numba.njit(scalar_op_fn)
