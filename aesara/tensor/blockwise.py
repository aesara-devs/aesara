from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np

import aesara
from aesara.gradient import DisconnectedType
from aesara.graph.basic import Apply, Variable
from aesara.graph.null_type import NullType
from aesara.graph.op import Op
from aesara.graph.rewriting.utils import rewrite_graph
from aesara.scalar.basic import constant as scalar_constant
from aesara.scalar.basic import int64
from aesara.tensor import get_scalar_constant_value
from aesara.tensor.basic import atleast_Nd
from aesara.tensor.elemwise import DimShuffle
from aesara.tensor.exceptions import NotScalarConstantError
from aesara.tensor.extra_ops import broadcast_shape
from aesara.tensor.math import sum as at_sum
from aesara.tensor.shape import shape_tuple
from aesara.tensor.type import TensorType, lscalar


if TYPE_CHECKING:
    from aesara.graph.fg import FunctionGraph
    from aesara.tensor.var import TensorVariable


def _update_dim_sizes(
    dim_sizes: Dict[str, "TensorVariable"],
    arg: "TensorVariable",
    core_dims: Tuple[str, ...],
):
    """Incrementally check and update core dimension sizes for a single argument.

    From `numpy.lib.function_base`.

    Parameters
    ----------
    dim_sizes
        Sizes of existing core dimensions. Will be updated in-place.
    arg
        Argument to examine.
    core_dims
        Core dimensions for this argument.
    """
    if not core_dims:
        return

    num_core_dims = len(core_dims)
    if arg.type.ndim < num_core_dims:
        raise ValueError(
            f"{arg.type.ndim}-dimensional argument does not have enough "
            f"dimensions for all core dimensions: {core_dims}"
        )

    core_shape = shape_tuple(arg)[-num_core_dims:]
    for dim, size in zip(core_dims, core_shape):
        if dim not in dim_sizes:
            dim_sizes[dim] = cast("TensorVariable", size)
        # else:
        #     # This check can't be done (sufficiently) at compile-time
        #     if size != dim_sizes[dim]:
        #         raise ValueError(
        #             f"Inconsistent size for core dimension {dim}: {size} vs {dim_sizes[dim]}"
        #         )


def _parse_input_dimensions(
    args: Tuple["TensorVariable", ...], input_core_dims: List[Tuple[str, ...]]
) -> Tuple[Tuple[Variable, ...], Dict[str, "TensorVariable"]]:
    """Parse broadcast and core dimensions for vectorize with a signature.

    From `numpy.lib.function_base`.

    Parameters
    ----------
    args
        Tuple of input arguments to examine.
    input_core_dims
        List of core dimensions corresponding to each input.

    Returns
    -------
    broadcast_shape
        Common shape to broadcast all non-core dimensions to.
    dim_sizes
        Common sizes for named core dimensions.
    """
    broadcast_args = []
    dim_sizes: Dict[str, "TensorVariable"] = {}
    for arg, core_dims in zip(args, input_core_dims):
        _update_dim_sizes(dim_sizes, arg, core_dims)
        ndim = arg.ndim - len(core_dims)
        arg_shape = shape_tuple(arg)
        broadcast_args.append(arg_shape[:ndim])
    bcast_shape = broadcast_shape(*broadcast_args, arrays_are_shapes=True)
    return bcast_shape, dim_sizes


def _calculate_shapes(
    broadcast_shape: Tuple[Variable, ...],
    dim_sizes: Dict[str, Variable],
    list_of_core_dims: List[Tuple[str, ...]],
) -> List[Tuple[Variable, ...]]:
    """Helper for calculating broadcast shapes with core dimensions.

    From `numpy.lib.function_base`.

    """

    def get_dim_size(x):
        res = dim_sizes.get(x)

        if res is None:
            try:
                return scalar_constant(int(x))
            except (ValueError, TypeError):
                # Return a symbolic placeholder for new dimension references
                # For example, a signature like `("m", "n") -> ("p",)` means
                # that there will be no `"p"` label to reference in `dim_sizes`
                # (i.e. pre-existing dimension labels that already have values
                # assigned to them).
                return int64(name=x)

        return res

    return [
        broadcast_shape + tuple(get_dim_size(dim) for dim in core_dims)
        for core_dims in list_of_core_dims
    ]


def gufunc_sign_to_str(sign):
    in_sign = [f"({','.join(_sign)})" for _sign in sign[0]]
    out_sign = [f"({','.join(_sign)})" for _sign in sign[1]]
    return f"{','.join(in_sign)}->{','.join(out_sign)}"


def infer_shape_to_gufunc_sig(node: Apply, fgraph: Optional["FunctionGraph"] = None):
    """Derive a gufunc signature from an `Op.infer_shape`.

    Parameters
    ==========
    node
        The `Apply` node with the `Op.infer_shape` we want to use.
    fgraph
        A `FunctionGraph` containing `node`.

    """
    op = node.op
    in_shapes = tuple(
        tuple(lscalar(f"i{s}") for s in range(inp.type.ndim)) for inp in node.inputs
    )
    out_shapes = op.infer_shape(fgraph, node, in_shapes)

    flat_shape = []
    for out_shape in out_shapes:
        for shape in out_shape:
            if isinstance(shape, int):
                shape = scalar_constant(shape)
            flat_shape.append(shape)

    # We need to canonicalize in order to match output shape labels with input
    # shape labels
    flat_out_shapes = rewrite_graph(flat_shape, clone=False)
    assert isinstance(flat_out_shapes, list)

    # Unflatten the canonicalized shape-graph outputs
    shape_sizes = tuple(len(x) for x in out_shapes)
    out_shapes = [()] * len(out_shapes)
    for i, shape_size in enumerate(shape_sizes):
        for j in range(shape_size):
            out_shapes[i] += (flat_out_shapes.pop(0),)

    out_shapes = tuple(out_shapes)

    gufunc_inputs_sig = tuple(tuple(s.name for s in in_shape) for in_shape in in_shapes)

    all_input_names = sum(gufunc_inputs_sig, ())

    gufunc_outputs_sig = tuple(
        # If an output label doesn't match an input label, create a new label
        # for the output
        tuple(
            s.name if s.name in all_input_names else f"o{output_idx}" for s in out_shape
        )
        for output_idx, out_shape in enumerate(out_shapes)
    )
    return (gufunc_inputs_sig, gufunc_outputs_sig)


def safe_const_val(x):
    try:
        return get_scalar_constant_value(x)
    except NotScalarConstantError:
        return None


class Blockwise(Op):
    __props__ = ("op", "signature")

    def __init__(self, op, signature=None):
        self.op = op
        self.signature = signature or self.op.gufunc_sig

    def get_core_inputs_outputs(self, inputs: Sequence["TensorVariable"]):
        """Get the core inputs and outputs for a given set of `inputs`.

        Parameters
        ==========
        inputs
            The normalized, blocked inputs (i.e. "broadcasted" inputs with all
            the necessary dimensions added).  They're needed for their dtype
            and static shape information.

        """

        core_inputs = []
        for _inp, _inp_sig in zip(inputs, self.signature[0]):
            curr_dtype = _inp.type.dtype
            # Extract the static shape values of the core dimensions in the
            # signature.  Doing so will produce a much more precise
            # `TensorType`.
            curr_static_shape = _inp.type.shape[_inp.type.ndim - len(_inp_sig) :]
            core_inputs.append(TensorType(curr_dtype, curr_static_shape)())

        # TODO: This shouldn't be necessary; `Op.make_node` doesn't call
        # `compute_test_value`, only `Op.__call__` does.
        with aesara.config.change_flags(compute_test_value="off"):
            core_outputs: Sequence[Variable] = self.op.make_node(*core_inputs).outputs

        return core_inputs, core_outputs

    def get_output_info(self, *inputs):
        r"""Return the outputs dtype and broadcastable pattern and the `DimShuffle`\d inputs.

        Parameters
        ==========
        inputs
             The blocked inputs (i.e. "broadcasted" inputs).
        """

        # Ensure that all blocked inputs have the same number of core
        # dimensions
        blocked_inputs = []
        for inp, signature in zip(inputs, self.signature[0]):
            core_ndim = len(signature)
            difference = core_ndim - inp.type.ndim

            # Do we need to _add_ core dimensions?
            if difference > 0:
                core_inp = DimShuffle(
                    inp.type.broadcastable,
                    list(range(inp.type.ndim)) + ["x"] * difference,
                )(inp)
            else:
                core_inp = inp

            blocked_inputs.append(core_inp)

        # Remove the core dimension first, then broadcast the rest of the
        # dimensions
        max_loop_dimension = max(
            blocked_inputs[i].type.ndim - len(self.signature[0][i])
            for i in range(len(blocked_inputs))
        )

        # Normalize the inputs by adding missing broadcast dimensions
        broadcasted_inputs = []
        for inp, signature in zip(blocked_inputs, self.signature[0]):
            core_ndim = len(signature)
            loop_dimension = inp.type.ndim - core_ndim
            difference = max_loop_dimension - loop_dimension
            assert difference >= 0

            if difference > 0:
                bcast_inp = DimShuffle(
                    inp.type.broadcastable,
                    ["x"] * difference + list(range(inp.type.ndim)),
                )(inp)
            else:
                bcast_inp = inp

            broadcasted_inputs.append(bcast_inp)

        _, core_outputs = self.get_core_inputs_outputs(broadcasted_inputs)
        out_dtypes = [o.type.dtype for o in core_outputs]

        bcast_shape, dim_sizes = _parse_input_dimensions(
            broadcasted_inputs, self.signature[0]
        )
        output_shapes = _calculate_shapes(bcast_shape, dim_sizes, self.signature[1])

        return out_dtypes, output_shapes, broadcasted_inputs

    def make_node(self, *inputs):
        """
        Parameters
        ==========
        inputs
            The blocked inputs (i.e. "broadcasted" inputs).
        """
        num_expected_inps = len(self.signature[0])
        if len(inputs) != num_expected_inps:
            raise ValueError(
                f"Expected {int(num_expected_inps)} inputs, got {len(inputs)}"
            )

        out_dtypes, output_shapes, inputs = self.get_output_info(*inputs)

        outputs = [
            TensorType(
                out_dtypes[i], shape=tuple(safe_const_val(s) for s in output_shapes[i])
            )()
            for i in range(len(output_shapes))
        ]
        return Apply(self, list(inputs), outputs)

    def __str__(self):
        return f"{type(self).__name__}{{op={self.op}}}"

    def infer_shape(self, fgraph, node, shapes):
        bcast_shape, dim_sizes = _parse_input_dimensions(node.inputs, self.signature[0])
        output_shapes = _calculate_shapes(bcast_shape, dim_sizes, self.signature[1])
        return output_shapes

    def L_op(self, inputs, outs, ograds):
        # Compute grad with respect to broadcasted input
        rval = self._bgrad(inputs, outs, ograds)

        # TODO: This is very broken.  See #1089.
        # Sum out the broadcasted dimensions
        for i, ipt in enumerate(inputs):
            if isinstance(rval[i].type, (NullType, DisconnectedType)):
                continue

            # List of all the dimensions that are broadcastable for input[i] so
            # we can sum over them
            # TODO: only count dimensions that were effectively broadcasted
            to_sum = [
                j
                for j, bcast in enumerate(ipt.type.broadcastable)
                if bcast and not outs[0].broadcastable[j]
            ]

            if to_sum:
                sr = at_sum(rval[i], axis=to_sum, keepdims=True)
                rval[i] = sr

        for inp, grad in zip(inputs, rval):
            assert inp.ndim == grad.ndim

        return rval

    def _bgrad(
        self,
        inputs: Sequence["TensorVariable"],
        outputs: Sequence["TensorVariable"],
        ograds: Sequence["TensorVariable"],
    ):
        with aesara.config.change_flags(compute_test_value="off"):
            core_inputs, core_outputs = self.get_core_inputs_outputs(inputs)

            core_out_grads = []
            for _out_grad, _out_sig in zip(ograds, self.signature[1]):
                curr_dtype = _out_grad.type.dtype
                start_idx = _out_grad.type.ndim - len(_out_sig)
                curr_static_shape = _out_grad.type.shape[start_idx:]
                core_out_grads.append(TensorType(curr_dtype, curr_static_shape)())

            core_inp_grads = self.op.L_op(core_inputs, core_outputs, core_out_grads)

            for igrad in core_inp_grads:
                assert igrad is not None, self.op

        def transform(
            var: "TensorVariable", client_node: Optional[Apply]
        ) -> "TensorVariable":
            """Walk a graph and expand single gradient \"block\"s into their block-wise equivalents."""

            if isinstance(var.type, (NullType, DisconnectedType)):
                return var

            if var in core_inputs:
                idx: int = core_inputs.index(var)
                return inputs[idx]
            elif var in core_outputs:
                idx = core_outputs.index(var)
                return outputs[idx]
            elif var in core_out_grads:
                idx = core_out_grads.index(var)
                return ograds[idx]

            node = var.owner
            if node is None:
                # The gradient contains a constant
                # res = aesara.tensor.basic.constant(
                #     np.asarray(var.data), dtype=var.type.dtype
                # )
                res = var

                # TODO FIXME: Use dimensions of relevant/appropriate inputs.
                # What exactly are those in this case?
                nd = inputs[0].type.ndim

                return atleast_Nd(res, n=nd)

            blocked_inputs = [transform(ipt, node) for ipt in node.inputs]
            # grad_signature = get_gufunc_signature(node.op, blocked_inputs)
            grad_signature = infer_shape_to_gufunc_sig(node)
            new_r = Blockwise(node.op, signature=grad_signature)(*blocked_inputs)

            assert isinstance(new_r, Variable)

            return cast("TensorVariable", new_r)

        ret = []
        for core_inp_grad, ipt in zip(core_inp_grads, inputs):
            ret.append(transform(core_inp_grad, None))

        return ret

    def perform(self, node, inputs, outputs):
        def py_func(*inner_inputs):
            res = [[None] for i in range(len(outputs))]
            # TODO:This can be avoided by making a single dummy node
            # But will that cover all cases?
            inner_node = self.op.make_node(*inner_inputs)
            inner_inputs = [np.asarray(i) for i in inner_inputs]
            if isinstance(self.op, DimShuffle):
                self.op.perform(inner_node, inner_inputs, res, params=None)
            else:
                self.op.perform(inner_node, inner_inputs, res)

            # Numpy always expects outputs to be Numpy arrays
            # And since we have a variable number of outputs
            if len(res) == 1:
                return res[0][0]
            else:
                return tuple(_res[0] for _res in res)

        numpy_vec_func = np.vectorize(
            py_func, signature=gufunc_sign_to_str(self.signature)
        )
        res_variables = numpy_vec_func(*inputs)

        if isinstance(res_variables, tuple):
            for i, out in enumerate(outputs):
                outputs[i][0] = res_variables[i]
        else:
            outputs[0][0] = res_variables
