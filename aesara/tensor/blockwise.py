from typing import Dict, List, Tuple

from aesara.graph.basic import Apply, Variable
from aesara.graph.op import Op
from aesara.tensor import get_scalar_constant_value
from aesara.tensor.exceptions import NotScalarConstantError
from aesara.tensor.extra_ops import broadcast_shape
from aesara.tensor.shape import shape_tuple
from aesara.tensor.type import TensorType


def _update_dim_sizes(
    dim_sizes: Dict[str, int], arg: Variable, core_dims: Tuple[str, ...]
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
    if arg.ndim < num_core_dims:
        raise ValueError(
            f"{arg.ndim}-dimensional argument does not have enough "
            f"dimensions for all core dimensions: {core_dims}"
        )

    core_shape = shape_tuple(arg)[-num_core_dims:]
    for dim, size in zip(core_dims, core_shape):
        if dim not in dim_sizes:
            dim_sizes[dim] = size
        # else:
        #     # This check can't be done (sufficiently) at compile-time
        #     if size != dim_sizes[dim]:
        #         raise ValueError(
        #             f"Inconsistent size for core dimension {dim}: {size} vs {dim_sizes[dim]}"
        #         )


def _parse_input_dimensions(
    args: Tuple[Variable, ...], input_core_dims: List[Tuple[str, ...]]
) -> Tuple[Tuple[Variable, ...], Dict[str, Variable]]:
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
    dim_sizes = {}
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
        raise NotImplementedError()

    def grad(self, *args):
        raise NotImplementedError()

    def L_op(self, *args):
        raise NotImplementedError()

    def perform(self, node, inputs, outputs):
        # TODO: Use `np.vectorize`, or the same steps within it.
        raise NotImplementedError()
