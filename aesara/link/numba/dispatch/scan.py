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
from aesara.tensor.type import TensorType


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

    outer_in_names_to_vars = {
        (f"outer_in_{i}" if i > 0 else "n_steps"): v for i, v in enumerate(node.inputs)
    }
    outer_in_names = list(outer_in_names_to_vars.keys())
    outer_in_seqs_names = op.outer_seqs(outer_in_names)
    outer_in_mit_mot_names = op.outer_mitmot(outer_in_names)
    outer_in_mit_sot_names = op.outer_mitsot(outer_in_names)
    outer_in_sit_sot_names = op.outer_sitsot(outer_in_names)
    outer_in_nit_sot_names = op.outer_nitsot(outer_in_names)
    outer_in_shared_names = op.outer_shared(outer_in_names)
    outer_in_non_seqs_names = op.outer_non_seqs(outer_in_names)

    # These are all the outer-input names that have produce outputs/have output
    # taps (i.e. they have inner-outputs and corresponding outer-outputs).
    # Outer-outputs are ordered as follows:
    # mit-mot-outputs + mit-sot-outputs + sit-sot-outputs + nit-sots + shared-outputs
    outer_in_outtap_names = (
        outer_in_mit_mot_names
        + outer_in_mit_sot_names
        + outer_in_sit_sot_names
        + outer_in_nit_sot_names
        + outer_in_shared_names
    )

    # We create distinct variables for/references to the storage arrays for
    # each output.
    outer_in_to_storage_name: Dict[str, str] = {}
    for outer_in_name in outer_in_mit_mot_names:
        outer_in_to_storage_name[outer_in_name] = f"{outer_in_name}_mitmot_storage"

    for outer_in_name in outer_in_mit_sot_names:
        outer_in_to_storage_name[outer_in_name] = f"{outer_in_name}_mitsot_storage"

    for outer_in_name in outer_in_sit_sot_names:
        outer_in_to_storage_name[outer_in_name] = f"{outer_in_name}_sitsot_storage"

    for outer_in_name in outer_in_nit_sot_names:
        outer_in_to_storage_name[outer_in_name] = f"{outer_in_name}_nitsot_storage"

    for outer_in_name in outer_in_shared_names:
        outer_in_to_storage_name[outer_in_name] = f"{outer_in_name}_shared_storage"

    outer_output_names = list(outer_in_to_storage_name.values())
    assert len(outer_output_names) == len(node.outputs)

    # Construct the inner-input expressions (e.g. indexed storage expressions)
    # Inner-inputs are ordered as follows:
    # sequences + mit-mot-inputs + mit-sot-inputs + sit-sot-inputs +
    # shared-inputs + non-sequences.
    inner_in_exprs: List[str] = []

    def add_inner_in_expr(
        outer_in_name: str, tap_offset: Optional[int], storage_size_var: Optional[str]
    ):
        """Construct an inner-input expression."""
        storage_name = outer_in_to_storage_name.get(outer_in_name, outer_in_name)
        indexed_inner_in_str = (
            storage_name
            if tap_offset is None
            else idx_to_str(storage_name, tap_offset, size=storage_size_var)
        )
        inner_in_exprs.append(indexed_inner_in_str)

    for outer_in_name in outer_in_seqs_names:
        # These outer-inputs are indexed without offsets or storage wrap-around
        add_inner_in_expr(outer_in_name, 0, None)

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

    # Inner-outputs consist of:
    # mit-mot-outputs + mit-sot-outputs + sit-sot-outputs + nit-sots +
    # shared-outputs [+ while-condition]
    inner_output_names = [f"inner_out_{i}" for i in range(len(op.inner_outputs))]

    # inner_out_shared_names = op.inner_shared_outs(inner_output_names)

    # The assignment statements that copy inner-outputs into the outer-outputs
    # storage
    inner_out_to_outer_in_stmts: List[str] = []

    # Special statements that perform storage truncation for `while`-loops and
    # rotation for initially truncated storage.
    output_storage_post_proc_stmts: List[str] = []

    # In truncated storage situations (e.g. created by `save_mem_new_scan`),
    # the taps and output storage overlap, instead of the standard situation in
    # which the output storage is large enough to contain both the initial taps
    # values and the output storage.  In this truncated case, we use the
    # storage array like a circular buffer, and that's why we need to track the
    # storage size along with the taps length/indexing offset.
    def add_output_storage_post_proc_stmt(
        outer_in_name: str, tap_sizes: Tuple[int], storage_size: str
    ):

        tap_size = max(tap_sizes)

        if op.info.as_while:
            # While loops need to truncate the output storage to a length given
            # by the number of iterations performed.
            output_storage_post_proc_stmts.append(
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
        output_storage_post_proc_stmts.append(
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

    # Special in-loop statements that create (nit-sot) storage arrays after a
    # single iteration is performed.  This is necessary because we don't know
    # the exact shapes of the storage arrays that need to be allocated until
    # after an iteration is performed.
    inner_out_post_processing_stmts: List[str] = []

    # Storage allocation statements
    # For output storage allocated/provided by the inputs, these statements
    # will either construct aliases between the input names and the entries in
    # `outer_in_to_storage_name` or assign the latter to expressions that
    # create copies of those storage inputs.
    # In the nit-sot case, empty dummy arrays are assigned to the storage
    # variables and updated later by the statements in
    # `inner_out_post_processing_stmts`.
    storage_alloc_stmts: List[str] = []

    for outer_in_name in outer_in_outtap_names:
        outer_in_var = outer_in_names_to_vars[outer_in_name]

        if outer_in_name not in outer_in_nit_sot_names:

            storage_name = outer_in_to_storage_name[outer_in_name]

            is_tensor_type = isinstance(outer_in_var.type, TensorType)
            if is_tensor_type:
                storage_size_name = f"{outer_in_name}_len"
                storage_size_stmt = f"{storage_size_name} = {outer_in_name}.shape[0]"
                input_taps = inner_in_names_to_input_taps[outer_in_name]
                tap_storage_size = -min(input_taps)
                assert tap_storage_size >= 0

                for in_tap in input_taps:
                    tap_offset = in_tap + tap_storage_size
                    assert tap_offset >= 0
                    add_inner_in_expr(outer_in_name, tap_offset, storage_size_name)

                output_taps = inner_in_names_to_output_taps.get(
                    outer_in_name, [tap_storage_size]
                )
                for out_tap in output_taps:
                    inner_out_to_outer_in_stmts.append(
                        idx_to_str(storage_name, out_tap, size=storage_size_name)
                    )

                add_output_storage_post_proc_stmt(
                    storage_name, output_taps, storage_size_name
                )

            else:
                storage_size_stmt = ""
                add_inner_in_expr(outer_in_name, None, None)
                inner_out_to_outer_in_stmts.append(storage_name)

            output_idx = outer_output_names.index(storage_name)
            if output_idx in node.op.destroy_map or not is_tensor_type:
                storage_alloc_stmt = f"{storage_name} = {outer_in_name}"
            else:
                storage_alloc_stmt = f"{storage_name} = np.copy({outer_in_name})"

            storage_alloc_stmt = dedent(
                f"""
                {storage_size_stmt}
                {storage_alloc_stmt}
                """
            ).strip()

            storage_alloc_stmts.append(storage_alloc_stmt)

        else:
            assert outer_in_name in outer_in_nit_sot_names

            # This is a special case in which there are no outer-inputs used
            # for outer-output storage, so we need to create our own storage
            # from scratch.
            storage_name = outer_in_to_storage_name[outer_in_name]
            storage_size_name = f"{outer_in_name}_len"

            inner_out_to_outer_in_stmts.append(
                idx_to_str(storage_name, 0, size=storage_size_name)
            )
            add_output_storage_post_proc_stmt(storage_name, (0,), storage_size_name)

            # In case of nit-sots we are provided the length of the array in
            # the iteration dimension instead of actual arrays, hence we
            # allocate space for the results accordingly.
            curr_nit_sot_position = outer_in_nit_sot_names.index(outer_in_name)
            curr_nit_sot = op.inner_nitsot_outs(op.inner_outputs)[curr_nit_sot_position]

            storage_shape = create_tuple_string(
                [storage_size_name] + ["0"] * curr_nit_sot.ndim
            )
            storage_dtype = curr_nit_sot.type.numpy_dtype.name

            storage_alloc_stmts.append(
                dedent(
                    f"""
                {storage_size_name} = to_numba_scalar({outer_in_name})
                {storage_name} = np.empty({storage_shape}, dtype=np.{storage_dtype})
                """
                ).strip()
            )

            if curr_nit_sot.type.ndim > 0:
                storage_alloc_stmts.append(f"{outer_in_name}_ready = False")

                # In this case, we don't know the shape of the output storage
                # array until we get some output from the inner-function.
                # With the following we add delayed output storage initialization:
                inner_out_name = op.inner_nitsot_outs(inner_output_names)[
                    curr_nit_sot_position
                ]
                inner_out_post_processing_stmts.append(
                    dedent(
                        f"""
                    if not {outer_in_name}_ready:
                        {storage_name} = np.empty(({storage_size_name},) + np.shape({inner_out_name}), dtype=np.{storage_dtype})
                        {outer_in_name}_ready = True
                    """
                    ).strip()
                )

    for name in outer_in_non_seqs_names:
        add_inner_in_expr(name, None, None)

    if op.info.as_while:
        # The inner function will return a boolean as the last value
        inner_out_to_outer_in_stmts.append("cond")

    assert len(inner_in_exprs) == len(op.fgraph.inputs)

    inner_in_args = create_arg_string(inner_in_exprs)
    inner_outputs = create_tuple_string(inner_output_names)
    input_storage_block = "\n".join(storage_alloc_stmts)
    output_storage_post_processing_block = "\n".join(output_storage_post_proc_stmts)
    inner_out_post_processing_block = "\n".join(inner_out_post_processing_stmts)

    inner_out_to_outer_out_stmts = "\n".join(
        [f"{s} = {d}" for s, d in zip(inner_out_to_outer_in_stmts, inner_output_names)]
    )

    scan_op_src = f"""
def scan({", ".join(outer_in_names)}):

{indent(input_storage_block, " " * 4)}

    i = 0
    cond = False
    while i < n_steps and not cond:
        {inner_outputs} = scan_inner_func({inner_in_args})
{indent(inner_out_post_processing_block, " " * 8)}
{indent(inner_out_to_outer_out_stmts, " " * 8)}
        i += 1

{indent(output_storage_post_processing_block, " " * 4)}

    return {create_arg_string(outer_output_names)}
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
