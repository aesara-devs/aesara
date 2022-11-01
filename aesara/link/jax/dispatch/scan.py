from collections import defaultdict
from typing import Callable, Dict, List

import jax
import jax.numpy as jnp

from aesara.link.jax.dispatch.basic import jax_funcify
from aesara.scan.op import Scan
from aesara.tensor.shape import Shape_i
from aesara.tensor.subtensor import Subtensor
from aesara.tensor.var import TensorVariable


def assert_while_returns_last_output(fgraph, node):
    """Check that the clients of the `Scan` outputs are `Subtensor` operators.

    JAX cannot accumulate the intermediate values in a `jax.lax.while` loop, we
    thus cannot in general compile a `Scan` operator used as a while loop.
    However, when only the last output of the `Scan` computation is used in the
    rest of the graph we can transpile `Scan` directly to `jax.lax.while`.

    """
    msg = """JAX cannot accumulate the results inside a `jax.lax.while_loop` loop.

    As a result, Aesara cannot compile the graph you provided to code that can
    be run with JAX. In case you only need the value computed at the last iteration
    of the while loop, extract this value from the output of `aesara.scan` explicitly
    like so:

    >>> res, _ = aesara.scan(...)
    >>> value = res[-1]

    So Aesara can compile the graph to code that can be run with JAX.

    """

    # Count the number of outputs of the outer function. We ignore
    # `shared` variables since they are not not accumulated and not
    # returned to the user.
    op = node.op
    num_outer_outputs = (
        op.info.n_mit_mot + op.info.n_mit_sot + op.info.n_sit_sot + op.info.n_nit_sot
    )
    for out in node.outputs[:num_outer_outputs]:
        for client, _ in fgraph.clients[out]:
            if isinstance(client, str):
                raise NotImplementedError(msg)
            elif isinstance(client.op, Subtensor):
                idx_list = client.op.idx_list
                if isinstance(idx_list[0], slice):
                    raise NotImplementedError(msg)
            elif not isinstance(client.op, Shape_i):
                raise NotImplementedError(msg)


@jax_funcify.register(Scan)
def jax_funcify_Scan(op, node, **kwargs):
    scan_inner_fn = jax_funcify(op.fgraph)
    input_taps = {
        "mit_sot": op.info.mit_sot_in_slices,
        "sit_sot": op.info.sit_sot_in_slices,
        "nit_sot": op.info.sit_sot_in_slices,
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
        global_fgraph = kwargs.get("global_fgraph", None)
        assert_while_returns_last_output(global_fgraph, node)
        return make_jax_while_fn(scan_inner_fn, parse_outer_inputs, input_taps)
    else:
        return make_jax_scan_fn(scan_inner_fn, parse_outer_inputs, input_taps)


def make_jax_while_fn(
    scan_inner_fn: Callable,
    parse_outer_inputs: Callable[[TensorVariable], Dict[str, List[TensorVariable]]],
    input_taps: Dict,
):
    """Create a `jax.lax.while_loop` function to perform `Scan` computations when it
    is used as while loop.

    `jax.lax.while_loop` iterates by passing a value `carry` to a `body_fun` that
    must return a value of the same type (Pytree structure, shape and dtype of
    the leaves). Before calling `body_fn`, it calls `cond_fn` which takes the
    current value and returns a boolean that indicates whether to keep iterating
    or not.

    The JAX `while_loop` needs to perform the following operations:

    1. Extract the inner-inputs;
    2. Build the initial carry value;
    3. Inside the loop:
        1. `carry` -> inner-inputs;
        2. inner-outputs -> `carry`
    4. Post-process the `carry`  storage and return outputs
    """

    def build_while_carry(outer_in):
        """Build the inputs to `jax.lax.scan` from the outer-inputs."""
        init_carry = {
            "mit_sot": [],
            "mit_sot_storage": outer_in["mit_sot"],
            "sit_sot": [],
            "sit_sot_storage": outer_in["sit_sot"],
            "shared": outer_in["shared"],
            "sequences": outer_in["sequences"],
            "non_sequences": outer_in["non_sequences"],
        }
        init_carry["step"] = 0
        init_carry["do_stop"] = False
        return init_carry

    def build_inner_outputs_map(outer_in):
        """Map the inner-output variables to their position in the tuple returned by the inner function.

        TODO: Copied from the scan builder

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

        TODO: Copied from the scan builder

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

        TODO: Copied from the scan builder

        """
        new_carry_element = []
        for taps, carry_element, output in zip(input_taps, carry, inner_outputs):
            new_carry_element.append(
                [carry_element.at[step - tap].set(output) for tap in taps]
            )

        return sum(new_carry_element, [])

    def while_loop(*outer_inputs):

        outer_in = parse_outer_inputs(outer_inputs)
        init_carry = build_while_carry(outer_in)
        inner_output_idx = build_inner_outputs_map(outer_in)

        def inner_inputs_from_carry(carry):
            """Get inner-inputs from the arguments passed to the `jax.lax.while_loop` body function.

            Inner-inputs are ordered as follows:
            - sequences
            - mit-mot inputs
            - mit-sot inputs
            - sit-sot inputs
            - shared-inputs
            - non-sequences

            """
            current_step = carry["step"]

            inner_in_mit_sot = from_carry_storage(
                carry["mit_sot_storage"], current_step, input_taps["mit_sot"]
            )
            inner_in_sit_sot = from_carry_storage(
                carry["sit_sot_storage"], current_step, input_taps["sit_sot"]
            )
            inner_in_shared = carry.get("shared", [])
            inner_in_non_sequences = carry.get("non_sequences", [])

            return sum(
                [
                    inner_in_mit_sot,
                    inner_in_sit_sot,
                    inner_in_shared,
                    inner_in_non_sequences,
                ],
                [],
            )

        def carry_from_inner_outputs(carry, inner_outputs):
            step = carry["step"]
            new_carry = {
                "mit_sot": [],
                "sit_sot": [],
                "sit_sot_storage": [],
                "nit_sot": [],
                "mit_sot_storage": [],
                "shared": [],
                "step": step + 1,
                "sequences": carry["sequences"],
                "non_sequences": carry["non_sequences"],
                "do_stop": inner_outputs[-1],
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
                new_carry["mit_sot"] = mit_sot_inner_outputs
                new_carry["mit_sot_storage"] = to_carry_storage(
                    mit_sot_inner_outputs,
                    carry["mit_sot_storage"],
                    step,
                    input_taps["mit_sot"],
                )

            if "sit_sot" in inner_output_idx:
                sit_sot_inner_outputs = [
                    inner_outputs[idx] for idx in inner_output_idx["sit_sot"]
                ]
                new_carry["sit_sot"] = sit_sot_inner_outputs
                new_carry["sit_sot_storage"] = to_carry_storage(
                    sit_sot_inner_outputs,
                    carry["sit_sot_storage"],
                    step,
                    input_taps["sit_sot"],
                )

            if "nit_sot" in inner_output_idx:
                nit_sot_inner_outputs = [
                    inner_outputs[idx] for idx in inner_output_idx["nit_sot"]
                ]
                new_carry["nit_sot"] = nit_sot_inner_outputs

            return new_carry

        def cond_fn(carry):
            # The inner-function of `Scan` returns a boolean as the last
            # value. This needs to be included in `carry`.
            # TODO: Will it return `False` if the number of steps is exceeded?
            return ~carry["do_stop"]

        def body_fn(carry):
            inner_inputs = inner_inputs_from_carry(carry)
            inner_outputs = scan_inner_fn(*inner_inputs)
            new_carry = carry_from_inner_outputs(carry, inner_outputs)
            return new_carry

        # The `Scan` implementation in the C backend will execute the
        # function once before checking the termination condition, while
        # `jax.lax.while_loop` checks the condition first. We thus need to call
        # `body_fn` once before calling `jax.lax.while_loop`. This allows us,
        # along with `n_steps`, to build the storage array for the `nit-sot`s
        # since there is no way to know their shape and dtype before executing
        # the function.
        inner_inputs = inner_inputs_from_carry(init_carry)
        inner_outputs = scan_inner_fn(*inner_inputs)
        carry = carry_from_inner_outputs(init_carry, inner_outputs)
        carry = jax.lax.while_loop(cond_fn, body_fn, carry)

        # Post-process the storage arrays
        # We make sure that the outputs are not scalars in case an array
        # is expected downstream since `Scan` is supposed to always return arrays
        carry["sit_sot"] = [jnp.atleast_1d(element) for element in carry["sit_sot"]]
        carry["mit_sot"] = [jnp.atleast_1d(element) for element in carry["mit_sot"]]
        carry["nit_not"] = [jnp.atleast_1d(element) for element in carry["nit_sot"]]

        outer_outputs = ["mit_sot", "sit_sot", "nit_sot", "shared"]
        results = sum([carry[output] for output in outer_outputs], [])
        if len(results) == 1:
            return results[0]
        else:
            return results

    return while_loop


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

    The JAX `scan` function needs to perform the following operations:

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
        outer_outputs = results + shared_output

        if len(outer_outputs) == 1:
            return outer_outputs[0]

        return outer_outputs

    return scan
