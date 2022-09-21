from itertools import groupby
from textwrap import dedent, indent
from typing import Dict, List, Optional, Tuple

import numpy as np
from numba import types
from numba.extending import overload

from aesara.link.numba.dispatch import basic as numba_basic
from aesara.link.numba.dispatch.basic import (
    create_arg_string,
    create_tuple_string,
    numba_funcify,
)
from aesara.link.utils import compile_function_src
from aesara.scan.op import Scan


def idx_to_str(
    array_name: str, offset: int, size: Optional[str] = None, idx_symbol: str = "i"
) -> str:
    if offset < 0:
        indices = f"{idx_symbol} + {array_name}.shape[0] - {offset}"
    elif offset > 0:
        indices = f"{idx_symbol} + {offset}"
    else:
        indices = idx_symbol

    if size:
        # TODO FIXME: The `Scan` `Op` should tell us which outputs are computed
        # in this way.  We shouldn't have to waste run-time efforts in order to
        # compensate for this poor `Op`/rewrite design and implementation.
        indices = f"({indices}) % {size}"

    return f"{array_name}[{indices}]"


@overload(range)
def array0d_range(x):
    if isinstance(x, types.Array) and x.ndim == 0:

        def range_arr(x):
            return range(x.item())

        return range_arr


@numba_funcify.register(Scan)
def numba_funcify_Scan(op, node, **kwargs):
    scan_inner_func = numba_basic.numba_njit(numba_funcify(op.fgraph))

    n_seqs = op.info.n_seqs

    outer_in_names_to_vars = {
        (f"outer_in_{i}" if i > 0 else "n_steps"): v for i, v in enumerate(node.inputs)
    }
    outer_in_names = list(outer_in_names_to_vars.keys())
    outer_in_seqs_names = op.outer_seqs(outer_in_names)
    outer_in_mit_mot_names = op.outer_mitmot(outer_in_names)
    outer_in_mit_sot_names = op.outer_mitsot(outer_in_names)
    outer_in_sit_sot_names = op.outer_sitsot(outer_in_names)
    outer_in_nit_sot_names = op.outer_nitsot(outer_in_names)
    outer_in_outtap_names = (
        outer_in_mit_mot_names
        + outer_in_mit_sot_names
        + outer_in_sit_sot_names
        + outer_in_nit_sot_names
    )
    outer_in_non_seqs_names = op.outer_non_seqs(outer_in_names)

    inner_in_to_index_offset: List[Tuple[str, Optional[int], Optional[int]]] = []
    allocate_taps_storage: List[str] = []

    for outer_in_name in outer_in_seqs_names:
        # A sequence with multiple taps is provided as multiple modified input
        # sequences--all sliced so as to keep following the logic of a normal
        # sequence.
        inner_in_to_index_offset.append((outer_in_name, 0, None))

    inner_in_names_to_input_taps: Dict[str, Tuple[int]] = dict(
        zip(
            outer_in_mit_mot_names + outer_in_mit_sot_names + outer_in_sit_sot_names,
            op.info.mit_mot_in_slices
            + op.info.mit_sot_in_slices
            + op.info.sit_sot_in_slices,
        )
    )
    inner_in_names_to_output_taps: Dict[str, Optional[Tuple[int, ...]]] = dict(
        zip(outer_in_mit_mot_names, op.info.mit_mot_out_slices)
    )

    inner_output_names = [f"inner_out_{i}" for i in range(len(op.inner_outputs))]

    # Maps storage array names to their tap values (i.e. maximum absolute tap
    # value) and storage sizes
    inner_out_name_to_taps_storage: List[Tuple[str, int, Optional[str]]] = []
    outer_in_to_storage_name: Dict[str, str] = {}
    outer_in_sot_names = set(
        outer_in_mit_mot_names + outer_in_mit_sot_names + outer_in_sit_sot_names
    )
    inner_out_post_processing_stmts: List[str] = []
    for outer_in_name in outer_in_outtap_names:
        outer_in_var = outer_in_names_to_vars[outer_in_name]

        if outer_in_name in outer_in_sot_names:
            if outer_in_name in outer_in_mit_mot_names:
                storage_name = f"{outer_in_name}_mitmot_storage"
            elif outer_in_name in outer_in_mit_sot_names:
                storage_name = f"{outer_in_name}_mitsot_storage"
            else:
                # Note that the outputs with single, non-`-1` taps are (e.g. `taps
                # = [-2]`) are classified as mit-sot, so the code for handling
                # sit-sots remains constant as follows
                storage_name = f"{outer_in_name}_sitsot_storage"

            output_idx = len(outer_in_to_storage_name)
            outer_in_to_storage_name[outer_in_name] = storage_name

            input_taps = inner_in_names_to_input_taps[outer_in_name]
            tap_storage_size = -min(input_taps)
            assert tap_storage_size >= 0

            storage_size_name = f"{outer_in_name}_len"

            for in_tap in input_taps:
                tap_offset = in_tap + tap_storage_size
                assert tap_offset >= 0
                # In truncated storage situations (i.e. created by
                # `save_mem_new_scan`), the taps and output storage overlap,
                # instead of the standard situation in which the output storage
                # is large enough to contain both the initial taps values and
                # the output storage.
                inner_in_to_index_offset.append(
                    (outer_in_name, tap_offset, storage_size_name)
                )

            output_taps = inner_in_names_to_output_taps.get(
                outer_in_name, [tap_storage_size]
            )
            for out_tap in output_taps:
                inner_out_name_to_taps_storage.append(
                    (storage_name, out_tap, storage_size_name)
                )

            if output_idx in node.op.destroy_map:
                storage_alloc_stmt = f"{storage_name} = {outer_in_name}"
            else:
                storage_alloc_stmt = f"{storage_name} = np.copy({outer_in_name})"

            storage_alloc_stmt = dedent(
                f"""
                # {outer_in_var.type}
                {storage_size_name} = {outer_in_name}.shape[0]
                {storage_alloc_stmt}
                """
            ).strip()

            allocate_taps_storage.append(storage_alloc_stmt)

        elif outer_in_name in outer_in_nit_sot_names:
            # This is a special case in which there are no outer-inputs used
            # for outer-output storage, so we need to create our own storage
            # from scratch.

            storage_name = f"{outer_in_name}_nitsot_storage"
            outer_in_to_storage_name[outer_in_name] = storage_name

            storage_size_name = f"{outer_in_name}_len"
            inner_out_name_to_taps_storage.append((storage_name, 0, storage_size_name))

            # In case of nit-sots we are provided the length of the array in
            # the iteration dimension instead of actual arrays, hence we
            # allocate space for the results accordingly.
            curr_nit_sot_position = outer_in_names[1:].index(outer_in_name) - n_seqs
            curr_nit_sot = op.inner_outputs[curr_nit_sot_position]
            needs_alloc = curr_nit_sot.ndim > 0

            storage_shape = create_tuple_string(
                [storage_size_name] + ["0"] * curr_nit_sot.ndim
            )
            storage_dtype = curr_nit_sot.type.numpy_dtype.name

            allocate_taps_storage.append(
                dedent(
                    f"""
                # {curr_nit_sot.type}
                {storage_size_name} = to_numba_scalar({outer_in_name})
                {storage_name} = np.empty({storage_shape}, dtype=np.{storage_dtype})
                """
                ).strip()
            )

            if needs_alloc:
                allocate_taps_storage.append(f"{outer_in_name}_ready = False")

                # In this case, we don't know the shape of the output storage
                # array until we get some output from the inner-function.
                # With the following we add delayed output storage initialization:
                inner_out_name = inner_output_names[curr_nit_sot_position]
                inner_out_post_processing_stmts.append(
                    dedent(
                        f"""
                    if not {outer_in_name}_ready:
                        {storage_name} = np.empty(({storage_size_name},) + {inner_out_name}.shape, dtype=np.{storage_dtype})
                        {outer_in_name}_ready = True
                    """
                    ).strip()
                )

    # The non_seqs are passed to the inner function as-is
    for name in outer_in_non_seqs_names:
        inner_in_to_index_offset.append((name, None, None))

    inner_out_storage_indexed = [
        name if taps is None else idx_to_str(name, taps, size=size)
        for (name, taps, size) in inner_out_name_to_taps_storage
    ]

    output_storage_post_processing_stmts: List[str] = []

    for outer_in_name, grp_vals in groupby(
        inner_out_name_to_taps_storage, lambda x: x[0]
    ):

        _, tap_sizes, storage_sizes = zip(*grp_vals)

        tap_size = max(tap_sizes)
        storage_size = storage_sizes[0]

        if op.info.as_while:
            # While loops need to truncate the output storage to a length given
            # by the number of iterations performed.
            output_storage_post_processing_stmts.append(
                dedent(
                    f"""
                    if i + {tap_size} < {storage_size}:
                        {storage_size} = i + {tap_size}
                        {outer_in_name} = {outer_in_name}[:{storage_size}]
                    """
                ).strip()
            )

        # Rotate the storage so that the last computed value is at the end of
        # the storage array.
        # This is needed when the output storage array does not have a length
        # equal to the number of taps plus `n_steps`.
        output_storage_post_processing_stmts.append(
            dedent(
                f"""
                {outer_in_name}_shift = (i + {tap_size}) % ({storage_size})
                if {outer_in_name}_shift > 0:
                    {outer_in_name}_left = {outer_in_name}[:{outer_in_name}_shift]
                    {outer_in_name}_right = {outer_in_name}[{outer_in_name}_shift:]
                    {outer_in_name} = np.concatenate(({outer_in_name}_right, {outer_in_name}_left))
                """
            ).strip()
        )

    if op.info.as_while:
        # The inner function will return a boolean as the last value
        inner_out_storage_indexed.append("cond")

    output_names = [outer_in_to_storage_name[n] for n in outer_in_outtap_names]

    # Construct the inner-input expressions
    inner_inputs: List[str] = []
    for outer_in_name, tap_offset, size in inner_in_to_index_offset:
        storage_name = outer_in_to_storage_name.get(outer_in_name, outer_in_name)
        indexed_inner_in_str = (
            idx_to_str(storage_name, tap_offset, size=size)
            if tap_offset is not None
            else storage_name
        )
        # if outer_in_names_to_vars[outer_in_name].type.ndim - 1 <= 0:
        #     # Convert scalar inner-inputs to Numba scalars
        #     indexed_inner_in_str = f"to_numba_scalar({indexed_inner_in_str})"
        inner_inputs.append(indexed_inner_in_str)

    inner_inputs = create_arg_string(inner_inputs)
    inner_outputs = create_tuple_string(inner_output_names)
    input_storage_block = "\n".join(allocate_taps_storage)
    output_storage_post_processing_block = "\n".join(
        output_storage_post_processing_stmts
    )
    inner_out_post_processing_block = "\n".join(inner_out_post_processing_stmts)

    scan_op_src = f"""
def scan({", ".join(outer_in_names)}):

{indent(input_storage_block, " " * 4)}

    i = 0
    cond = False
    while i < n_steps and not cond:
        {inner_outputs} = scan_inner_func({inner_inputs})
{indent(inner_out_post_processing_block, " " * 8)}
        {create_tuple_string(inner_out_storage_indexed)} = {inner_outputs}
        i += 1

{indent(output_storage_post_processing_block, " " * 4)}

    return {create_arg_string(output_names)}
    """

    global_env = {
        "scan_inner_func": scan_inner_func,
        "to_numba_scalar": numba_basic.to_scalar,
    }
    global_env["np"] = np

    scalar_op_fn = compile_function_src(
        scan_op_src, "scan", {**globals(), **global_env}
    )

    return numba_basic.numba_njit(scalar_op_fn)
