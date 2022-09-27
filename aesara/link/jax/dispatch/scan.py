import jax
import jax.numpy as jnp

from aesara.graph.fg import FunctionGraph
from aesara.link.jax.dispatch.basic import jax_funcify
from aesara.scan.op import Scan
from aesara.scan.utils import ScanArgs


@jax_funcify.register(Scan)
def jax_funcify_Scan(op, **kwargs):
    inner_fg = FunctionGraph(op.inputs, op.outputs)
    jax_at_inner_func = jax_funcify(inner_fg, **kwargs)

    def scan(*outer_inputs):
        scan_args = ScanArgs(
            list(outer_inputs), [None] * op.info.n_outs, op.inputs, op.outputs, op.info
        )

        # `outer_inputs` is a list with the following composite form:
        # [n_steps]
        # + outer_in_seqs
        # + outer_in_mit_mot
        # + outer_in_mit_sot
        # + outer_in_sit_sot
        # + outer_in_shared
        # + outer_in_nit_sot
        # + outer_in_non_seqs
        n_steps = scan_args.n_steps
        seqs = scan_args.outer_in_seqs

        # TODO: mit_mots
        mit_mot_in_slices = []

        mit_sot_in_slices = []
        for tap, seq in zip(scan_args.mit_sot_in_slices, scan_args.outer_in_mit_sot):
            neg_taps = [abs(t) for t in tap if t < 0]
            pos_taps = [abs(t) for t in tap if t > 0]
            max_neg = max(neg_taps) if neg_taps else 0
            max_pos = max(pos_taps) if pos_taps else 0
            init_slice = seq[: max_neg + max_pos]
            mit_sot_in_slices.append(init_slice)

        sit_sot_in_slices = [seq[0] for seq in scan_args.outer_in_sit_sot]

        init_carry = (
            mit_mot_in_slices,
            mit_sot_in_slices,
            sit_sot_in_slices,
            scan_args.outer_in_shared,
            scan_args.outer_in_non_seqs,
        )

        def jax_args_to_inner_scan(op, carry, x):
            # `carry` contains all inner-output taps, non_seqs, and shared
            # terms
            (
                inner_in_mit_mot,
                inner_in_mit_sot,
                inner_in_sit_sot,
                inner_in_shared,
                inner_in_non_seqs,
            ) = carry

            # `x` contains the in_seqs
            inner_in_seqs = x

            # `inner_scan_inputs` is a list with the following composite form:
            # inner_in_seqs
            # + sum(inner_in_mit_mot, [])
            # + sum(inner_in_mit_sot, [])
            # + inner_in_sit_sot
            # + inner_in_shared
            # + inner_in_non_seqs
            inner_in_mit_sot_flatten = []
            for array, index in zip(inner_in_mit_sot, scan_args.mit_sot_in_slices):
                inner_in_mit_sot_flatten.extend(array[jnp.array(index)])

            inner_scan_inputs = sum(
                [
                    inner_in_seqs,
                    inner_in_mit_mot,
                    inner_in_mit_sot_flatten,
                    inner_in_sit_sot,
                    inner_in_shared,
                    inner_in_non_seqs,
                ],
                [],
            )

            return inner_scan_inputs

        def inner_scan_outs_to_jax_outs(
            op,
            old_carry,
            inner_scan_outs,
        ):
            (
                inner_in_mit_mot,
                inner_in_mit_sot,
                inner_in_sit_sot,
                inner_in_shared,
                inner_in_non_seqs,
            ) = old_carry

            def update_mit_sot(mit_sot, new_val):
                return jnp.concatenate([mit_sot[1:], new_val[None, ...]], axis=0)

            inner_out_mit_sot = [
                update_mit_sot(mit_sot, new_val)
                for mit_sot, new_val in zip(inner_in_mit_sot, inner_scan_outs)
            ]

            # This should contain all inner-output taps, non_seqs, and shared
            # terms
            if not inner_in_sit_sot:
                inner_out_sit_sot = []
            else:
                inner_out_sit_sot = inner_scan_outs
            new_carry = (
                inner_in_mit_mot,
                inner_out_mit_sot,
                inner_out_sit_sot,
                inner_in_shared,
                inner_in_non_seqs,
            )

            return new_carry

        def jax_inner_func(carry, x):
            inner_args = jax_args_to_inner_scan(op, carry, x)
            inner_scan_outs = list(jax_at_inner_func(*inner_args))
            new_carry = inner_scan_outs_to_jax_outs(op, carry, inner_scan_outs)
            return new_carry, inner_scan_outs

        _, scan_out = jax.lax.scan(jax_inner_func, init_carry, seqs, length=n_steps)

        # We need to prepend the initial values so that the JAX output will
        # match the raw `Scan` `Op` output and, thus, work with a downstream
        # `Subtensor` `Op` introduced by the `scan` helper function.
        def append_scan_out(scan_in_part, scan_out_part):
            return jnp.concatenate([scan_in_part[:-n_steps], scan_out_part], axis=0)

        if scan_args.outer_in_mit_sot:
            scan_out_final = [
                append_scan_out(init, out)
                for init, out in zip(scan_args.outer_in_mit_sot, scan_out)
            ]
        elif scan_args.outer_in_sit_sot:
            scan_out_final = [
                append_scan_out(init, out)
                for init, out in zip(scan_args.outer_in_sit_sot, scan_out)
            ]

        if len(scan_out_final) == 1:
            scan_out_final = scan_out_final[0]
        return scan_out_final

    return scan
