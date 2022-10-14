from collections import defaultdict
from typing import Callable, Dict, List

import jax

from aesara.link.jax.dispatch.basic import jax_funcify
from aesara.scan.op import Scan
from aesara.tensor.var import TensorVariable


@jax_funcify.register(Scan)
def jax_funcify_Scan(op, node, **kwargs):
    scan_inner_fn = jax_funcify(op.fgraph)
    input_taps = {
        "mit_sot": op.info.mit_sot_in_slices,
        "sit_sot": op.info.sit_sot_in_slices,
    }

    # Outer-inputs are the inputs to the `Scan` apply node, built from the
    # the variables provided by the caller to the `scan` function at construction
    # time.
    def parse_outer_inputs(outer_inputs):
        outer_in = {
            "n_steps": outer_inputs[0],
            "sequences": list(op.outer_seqs(outer_inputs)),
            "mit_mot": list(op.outer_mitmot(outer_inputs)),
            "mit_sot": list(op.outer_mitsot(outer_inputs)),
            "nit_sot": list(op.outer_nitsot(outer_inputs)),
            "sit_sot": list(op.outer_sitsot(outer_inputs)),
            "shared": list(op.outer_shared(outer_inputs)),
            "non_sequences": list(op.outer_non_seqs(outer_inputs)),
        }
        if len(outer_in["mit_mot"]) > 0:
            raise NotImplementedError("mit-mot not supported")

        return outer_in

    if op.info.as_while:
        raise NotImplementedError("While loops are not supported in the JAX backend.")
    else:
        return make_jax_scan_fn(
            scan_inner_fn,
            parse_outer_inputs,
            input_taps,
        )


def make_jax_scan_fn(
    scan_inner_fn: Callable,
    parse_outer_inputs: Callable[[TensorVariable], Dict[str, List[TensorVariable]]],
    input_taps: Dict,
):
    """Create a `jax.lax.scan` function to perform `Scan` computations.

    `jax.lax.scan` takes an initial `carry` value and a sequence it scans over,
    or a number of iterations. The first output of the loop body function, the
    `carry`, is carried over to the next iteration. The second, the `output`, is
    stacked to the previous outputs. We use this to our advantage to build
    `Scan` outputs without having to post-process the storage arrays.

    The JAX scan function needs to perform the following operations:
    1. Extract the inner-inputs;
    2. Build the initial `carry` and `sequence` values;
    3. Inside the loop:
        1. `carry` + sequence elements -> inner-inputs
        2. inner-outputs -> `carry`
        3. inner-outputs -> `output`
    4. Append the last `shared`  value to the stacked `output`s

    """

    def build_jax_scan_inputs(outer_in: Dict):
        """Build the inputs to `jax.lax.scan` from the outer-inputs."""
        n_steps = outer_in["n_steps"]
        sequences = outer_in["sequences"]
        init_carry = {
            name: outer_in[name]
            for name in ["mit_sot", "sit_sot", "shared", "non_sequences"]
        }
        init_carry["step"] = 0
        return n_steps, sequences, init_carry

    def build_inner_outputs_map(outer_in):
        """Map the inner-output variables to their position in the tuple returned by the inner function.

        Inner-outputs are ordered as follow:
        - mit-mot-outputs
        - mit-sot-outputs
        - sit-sot-outputs
        - nit-sots (no carry)
        - shared-outputs
        [+ while-condition]

        """
        inner_outputs_names = ["mit_sot", "sit_sot", "nit_sot", "shared"]

        offset = 0
        inner_output_idx = defaultdict(list)
        for name in inner_outputs_names:
            num_outputs = len(outer_in[name])
            for i in range(num_outputs):
                inner_output_idx[name].append(offset + i)
            offset += num_outputs

        return inner_output_idx

    def from_carry_storage(carry, step, input_taps):
        """Fetch the inner inputs from the values stored in the carry array.

        `Scan` passes storage arrays as inputs, which are then read from and
        updated in the loop body. At each step we need to read from this array
        the inputs that will be passed to the inner function.

        This mechanism is necessary because we handle multiple-input taps within
        the `scan` instead of letting users manage the memory in the use cases
        where this is necessary.

        """

        def fetch(carry, step, offset):
            return carry[step + offset]

        inner_inputs = []
        for taps, carry_element in zip(input_taps, carry):
            storage_size = -min(taps)
            offsets = [storage_size + tap for tap in taps]
            inner_inputs.append(
                [fetch(carry_element, step, offset) for offset in offsets]
            )

        return sum(inner_inputs, [])

    def to_carry_storage(inner_outputs, carry, step, input_taps):
        """Create the new carry array from the inner output

        `Scan` passes storage arrays as inputs, which are then read from and
        updated in the loop body. At each step we need to update this array
        with the outputs of the inner function

        """
        new_carry_element = []
        for taps, carry_element, output in zip(input_taps, carry, inner_outputs):
            new_carry_element.append(
                [carry_element.at[step - tap].set(output) for tap in taps]
            )

        return sum(new_carry_element, [])

    def scan(*outer_inputs):

        outer_in = parse_outer_inputs(outer_inputs)
        n_steps, sequences, init_carry = build_jax_scan_inputs(outer_in)
        inner_output_idx = build_inner_outputs_map(outer_in)

        def scan_inner_in_args(carry, x):
            """Get inner-inputs from the arguments passed to the `jax.lax.scan` body function.

            Inner-inputs are ordered as follows:
            - sequences
            - mit-mot inputs
            - mit-sot inputs
            - sit-sot inputs
            - shared-inputs
            - non-sequences

            """
            current_step = carry["step"]

            inner_in_seqs = x
            inner_in_mit_sot = from_carry_storage(
                carry["mit_sot"], current_step, input_taps["mit_sot"]
            )
            inner_in_sit_sot = from_carry_storage(
                carry["sit_sot"], current_step, input_taps["sit_sot"]
            )
            inner_in_shared = carry.get("shared", [])
            inner_in_non_sequences = carry.get("non_sequences", [])

            return sum(
                [
                    inner_in_seqs,
                    inner_in_mit_sot,
                    inner_in_sit_sot,
                    inner_in_shared,
                    inner_in_non_sequences,
                ],
                [],
            )

        def scan_new_carry(carry, inner_outputs):
            """Create a new carry value from the values returned by the inner function (inner-outputs)."""
            step = carry["step"]
            new_carry = {
                "mit_sot": [],
                "sit_sot": [],
                "shared": [],
                "step": step + 1,
                "non_sequences": carry["non_sequences"],
            }

            if "shared" in inner_output_idx:
                shared_inner_outputs = [
                    inner_outputs[idx] for idx in inner_output_idx["shared"]
                ]
                new_carry["shared"] = shared_inner_outputs

            if "mit_sot" in inner_output_idx:
                mit_sot_inner_outputs = [
                    inner_outputs[idx] for idx in inner_output_idx["mit_sot"]
                ]
                new_carry["mit_sot"] = to_carry_storage(
                    mit_sot_inner_outputs, carry["mit_sot"], step, input_taps["mit_sot"]
                )

            if "sit_sot" in inner_output_idx:
                sit_sot_inner_outputs = [
                    inner_outputs[idx] for idx in inner_output_idx["sit_sot"]
                ]
                new_carry["sit_sot"] = to_carry_storage(
                    sit_sot_inner_outputs, carry["sit_sot"], step, input_taps["sit_sot"]
                )

            return new_carry

        def scan_new_outputs(inner_outputs):
            """Create a new outer-output value from the outputs of the inner function.

            Outer-outputs are ordered as follows:
            - mit-mot-outputs
            - mit-sot-outputs
            - sit-sot-outputs
            - nit-sots
            - shared-outputs

            The shared output corresponds to the last value found in the last
            carry value returned by `jax.lax.scan`. It is thus not returned in
            the body function.

            """
            outer_outputs = []
            if "mit_sot" in inner_output_idx:
                outer_outputs.append(
                    [inner_outputs[idx] for idx in inner_output_idx["mit_sot"]]
                )
            if "sit_sot" in inner_output_idx:
                outer_outputs.append(
                    [inner_outputs[idx] for idx in inner_output_idx["sit_sot"]]
                )
            if "nit_sot" in inner_output_idx:
                outer_outputs.append(
                    [inner_outputs[idx] for idx in inner_output_idx["nit_sot"]]
                )

            return tuple(sum(outer_outputs, []))

        def body_fn(carry, x):
            inner_in_args = scan_inner_in_args(carry, x)
            inner_outputs = scan_inner_fn(*inner_in_args)
            new_carry = scan_new_carry(carry, inner_outputs)
            outer_outputs = scan_new_outputs(inner_outputs)
            return new_carry, outer_outputs

        last_carry, results = jax.lax.scan(
            body_fn, init_carry, sequences, length=n_steps
        )

        shared_output = tuple(last_carry["shared"])
        results = results + shared_output

        if len(results) == 1:
            return results[0]

        return results

    return scan
