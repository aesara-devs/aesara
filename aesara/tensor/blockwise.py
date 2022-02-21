from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np

import aesara
from aesara.gradient import DisconnectedType
from aesara.graph.basic import Apply, Variable
from aesara.graph.null_type import NullType
from aesara.graph.op import Op
from aesara.tensor import get_scalar_constant_value
from aesara.tensor.basic import atleast_Nd
from aesara.tensor.exceptions import NotScalarConstantError
from aesara.tensor.extra_ops import broadcast_shape
from aesara.tensor.shape import shape_tuple
from aesara.tensor.type import TensorType


if TYPE_CHECKING:
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
    return [
        broadcast_shape + tuple(dim_sizes[dim] for dim in core_dims)
        for core_dims in list_of_core_dims
    ]


def gufunc_sign_to_str(sign):
    in_sign = [f"({','.join(_sign)})" for _sign in sign[0]]
    out_sign = [f"({','.join(_sign)})" for _sign in sign[1]]
    return f"{','.join(in_sign)}->{','.join(out_sign)}"


class Blockwise(Op):
    __props__ = ("op", "signature")

    def __init__(self, op, signature=None):
        self.op = op
        self.signature = signature or self.op.gufunc_sig

    def make_node(self, *inputs):

        num_expected_inps = len(self.signature[0])
        if len(inputs) != num_expected_inps:
            raise ValueError(
                f"Expected {int(num_expected_inps)} inputs, got {len(inputs)}"
            )

        # TODO: Correct this
        out_dtype = inputs[0].dtype

        bcast_shape, dim_sizes = _parse_input_dimensions(inputs, self.signature[0])
        output_shapes = _calculate_shapes(bcast_shape, dim_sizes, self.signature[1])

        def safe_const_val(x):
            try:
                return get_scalar_constant_value(x)
            except NotScalarConstantError:
                return None

        outputs = [
            TensorType(out_dtype, shape=tuple(safe_const_val(s) for s in shp))()
            for shp in output_shapes
        ]
        return Apply(self, list(inputs), outputs)

    def infer_shape(self, fgraph, node, shapes):
        bcast_shape, dim_sizes = _parse_input_dimensions(node.inputs, self.signature[0])
        output_shapes = _calculate_shapes(bcast_shape, dim_sizes, self.signature[1])
        return output_shapes

    def L_op(
        self,
        inputs: Sequence[Variable],
        outputs: Sequence[Variable],
        ograds: Sequence[Variable],
    ):

        with aesara.config.change_flags(compute_test_value="off"):
            core_inputs = []
            for _inp, _inp_sig in zip(inputs, self.signature[0]):
                curr_dtype = _inp.type.dtype
                curr_static_shape = _inp.type.shape[: len(_inp_sig)]
                core_inputs.append(TensorType(curr_dtype, curr_static_shape)())

            core_out_grads = []
            for _out_grad, _out_sig in zip(ograds, self.signature[1]):
                curr_dtype = _out_grad.type.dtype
                curr_static_shape = _out_grad.type.shape[: len(_out_sig)]
                core_out_grads.append(TensorType(curr_dtype, curr_static_shape)())

            core_outputs: Sequence[Variable] = self.op.make_node(*core_inputs).outputs
            core_inp_grads = self.op.L_op(core_inputs, core_outputs, core_out_grads)

            for igrad in core_inp_grads:
                assert igrad is not None, self.op

        def transform(var: "TensorVariable", client_node: Optional[Apply]) -> Variable:
            """Walk a graph and expand single gradient \"block\"s into their block-wise equivalents."""

            if isinstance(var.type, (NullType, DisconnectedType)):
                return var

            if var in core_inputs:
                return inputs[core_inputs.index(var)]
            if var in core_outputs:
                return outputs[core_outputs.index(var)]
            if var in core_out_grads:
                return ograds[core_out_grads.index(var)]

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

                return atleast_Nd(res, nd)

            blocked_inputs = [transform(ipt, node) for ipt in node.inputs]

            grad_signature = getattr(node.op, "gufunc_sig", None)

            if grad_signature is None:
                # TODO: Can we manually derive gufunc signatures for any `Op`
                # in this situation?
                grad_signature = None

            new_r = Blockwise(node.op, signature=grad_signature)(*blocked_inputs)
            assert isinstance(new_r, Variable)
            return new_r

        ret = []
        for core_inp_grad, ipt in zip(core_inp_grads, inputs):
            ret.append(transform(core_inp_grad, None))

        return ret

    def perform(self, node, inputs, outputs):
        def py_func(*inner_inputs):
            res = [[None]] * len(outputs)
            # TODO:This can be avoided by making a single dummy node
            # But will that cover all cases?
            inner_node = self.op.make_node(*inner_inputs)
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
