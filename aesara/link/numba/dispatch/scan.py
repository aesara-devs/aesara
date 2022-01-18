import numpy as np
from numba import types
from numba.extending import overload

from aesara.graph.fg import FunctionGraph
from aesara.link.numba.dispatch import basic as numba_basic
from aesara.link.numba.dispatch.basic import (
    create_arg_string,
    create_tuple_string,
    numba_funcify,
)
from aesara.link.utils import compile_function_src
from aesara.scan.op import Scan


def idx_to_str(idx):
    res = "[i"
    if idx < 0:
        res += str(idx)
    elif idx > 0:
        res += "+" + str(idx)
    return res + "]"


@overload(range)
def array0d_range(x):
    if isinstance(x, types.Array) and x.ndim == 0:

        def range_arr(x):
            return range(x.item())

        return range_arr


@numba_funcify.register(Scan)
def numba_funcify_Scan(op, node, **kwargs):
    inner_fg = FunctionGraph(op.inputs, op.outputs)
    numba_at_inner_func = numba_basic.numba_njit(numba_funcify(inner_fg, **kwargs))

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
    allocate_mem_to_nit_sot = ""

    for _name in outer_in_seqs_names:
        # A sequence with multiple taps is provided as multiple modified
        # input sequences to the Scan Op sliced appropriately
        # to keep following the logic of a normal sequence.
        index = "[i]"
        inner_in_indexed.append(_name + index)

    name_to_input_map = dict(zip(input_names, node.inputs[1:]))
    mit_sot_name_to_taps = dict(zip(outer_in_mit_sot_names, mit_sot_in_taps))
    inner_out_name_to_index = {}
    for _name in outer_in_feedback_names:
        if _name in outer_in_mit_sot_names:
            curr_taps = mit_sot_name_to_taps[_name]
            min_tap = min(curr_taps)

            for _tap in curr_taps:
                index = idx_to_str(_tap - min_tap)
                inner_in_indexed.append(_name + index)

            inner_out_name_to_index[_name] = -min_tap

        if _name in outer_in_sit_sot_names:
            # Note that the outputs with single taps which are not
            # -1 are (for instance taps = [-2]) are classified
            # as mit-sot so the code for handling sit-sots remains
            # constant as follows
            index = "[i]"
            inner_in_indexed.append(_name + index)
            inner_out_name_to_index[_name] = 1

        if _name in outer_in_nit_sot_names:
            inner_out_name_to_index[_name] = 0
            # In case of nit-sots we are provided shape of the array
            # instead of actual arrays like other cases, hence we
            # allocate space for the results accordingly.
            curr_nit_sot_position = input_names.index(_name) - n_seqs
            curr_nit_sot = inner_fg.outputs[curr_nit_sot_position]
            mem_shape = ["1"] * curr_nit_sot.ndim
            curr_dtype = curr_nit_sot.type.numpy_dtype.name
            allocate_mem_to_nit_sot += f"""
    {_name} = [np.zeros(({create_arg_string(mem_shape)}), dtype=np.{curr_dtype})]*{_name}.item()
"""
    # The non_seqs are passed to inner function as-is
    inner_in_indexed += outer_in_non_seqs_names
    inner_out_indexed = [
        _name + idx_to_str(idx) for _name, idx in inner_out_name_to_index.items()
    ]

    while_logic = ""
    if op.as_while:
        # The inner function will be returning a boolean as last argument
        inner_out_indexed.append("while_flag")
        while_logic += """
        if while_flag:
        """
        for _name, idx in inner_out_name_to_index.items():
            while_logic += f"""
            {_name} = {_name}[:i+{idx+1}]
            """
        while_logic += """
            break
        """

    global_env = locals()
    global_env["np"] = np

    scan_op_src = f"""
def scan(n_steps, {", ".join(input_names)}):
{allocate_mem_to_nit_sot}
    for i in range(n_steps):
        inner_args = {create_tuple_string(inner_in_indexed)}
        {create_tuple_string(inner_out_indexed)} = numba_at_inner_func(*inner_args)
{while_logic}
    return {create_arg_string(
        outer_in_mit_sot_names +
        outer_in_sit_sot_names +
        outer_in_nit_sot_names
    )}
    """
    scalar_op_fn = compile_function_src(
        scan_op_src, "scan", {**globals(), **global_env}
    )

    return numba_basic.numba_njit(scalar_op_fn)
