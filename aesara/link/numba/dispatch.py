import ast
from functools import singledispatch
from tempfile import NamedTemporaryFile

import numba
import numpy as np

from aesara.compile.ops import DeepCopyOp
from aesara.graph.fg import FunctionGraph
from aesara.graph.type import Type
from aesara.link.utils import fgraph_to_python
from aesara.scalar.basic import Composite, ScalarOp
from aesara.tensor.elemwise import Elemwise
from aesara.tensor.subtensor import AdvancedSubtensor, AdvancedSubtensor1, Subtensor
from aesara.tensor.type_other import MakeSlice


@singledispatch
def numba_typify(data, dtype=None, **kwargs):
    return data


@singledispatch
def numba_funcify(op, **kwargs):
    """Create a Numba compatible function from an Aesara `Op`."""
    raise NotImplementedError(f"No Numba conversion for the given `Op`: {op}")


@numba_funcify.register(FunctionGraph)
def numba_funcify_FunctionGraph(
    fgraph,
    order=None,
    input_storage=None,
    output_storage=None,
    storage_map=None,
    **kwargs,
):
    return fgraph_to_python(
        fgraph,
        numba_funcify,
        numba_typify,
        order,
        input_storage,
        output_storage,
        storage_map,
        fgraph_name="numba_funcified_fgraph",
        **kwargs,
    )


@numba_funcify.register(ScalarOp)
def numba_funcify_ScalarOp(op, **kwargs):

    numpy_func = getattr(np, op.nfunc_spec[0])

    @numba.njit
    def scalar_func(*args):
        result = args[0]
        for arg in args[1:]:
            result = numpy_func(arg, result)
        return result

    return scalar_func


@numba_funcify.register(Elemwise)
def numba_funcify_Elemwise(op, **kwargs):
    scalar_op = op.scalar_op
    # TODO: Vectorize this
    return numba_funcify(scalar_op)


@numba_funcify.register(Composite)
def numba_funcify_Composite(op, vectorize=True, **kwargs):
    numba_impl = numba.njit(numba_funcify(op.fgraph))

    @numba.njit
    def composite(*args):
        return numba_impl(*args)[0]

    return composite


def create_index_func(node, idx_list, objmode=False):
    """Create a Python function that assembles and uses an index on an array."""

    def convert_indices(indices, entry):
        if indices and isinstance(entry, Type):
            rval = indices.pop(0)
            return rval.auto_name
        elif isinstance(entry, slice):
            return (
                f"slice({convert_indices(indices, entry.start)}, "
                f"{convert_indices(indices, entry.stop)}, "
                f"{convert_indices(indices, entry.step)})"
            )
        elif isinstance(entry, type(None)):
            return "None"
        else:
            raise ValueError()

    input_names = [v.auto_name for v in node.inputs]
    op_indices = list(node.inputs[1:])

    indices_creation_src = (
        tuple(convert_indices(op_indices, idx) for idx in idx_list)
        if idx_list
        else tuple(input_names[1:])
    )

    if len(indices_creation_src) == 1:
        indices_creation_src = f"indices = ({indices_creation_src[0]},)"
    else:
        indices_creation_src = ", ".join(indices_creation_src)
        indices_creation_src = f"indices = ({indices_creation_src})"

    if objmode:
        output_var = node.outputs[0]
        output_sig = f"{output_var.dtype}[{', '.join([':'] * output_var.ndim)}]"
        index_body = f"""
    with objmode(z="{output_sig}"):
        z = {input_names[0]}[indices]
        """
    else:
        index_body = f"z = {input_names[0]}[indices]"

    subtensor_def_src = f"""
def subtensor({", ".join(input_names)}):
    {indices_creation_src}
    {index_body}
    return z
    """

    return subtensor_def_src


@numba_funcify.register(Subtensor)
@numba_funcify.register(AdvancedSubtensor)
@numba_funcify.register(AdvancedSubtensor1)
def numba_funcify_Subtensor(op, node, **kwargs):

    idx_list = getattr(op, "idx_list", None)
    subtensor_def_src = create_index_func(
        node, idx_list, objmode=isinstance(op, AdvancedSubtensor)
    )

    subtensor_def_ast = ast.parse(subtensor_def_src)

    with NamedTemporaryFile(delete=False) as f:
        filename = f.name
        f.write(subtensor_def_src.encode())

    local_env = {}
    mod_code = compile(subtensor_def_ast, filename, mode="exec")
    exec(mod_code, {"objmode": numba.objmode}, local_env)

    subtensor_def = local_env["subtensor"]

    return numba.njit(subtensor_def)


@numba_funcify.register(DeepCopyOp)
def numba_funcify_DeepCopyOp(op, node, **kwargs):

    # Scalars are apparently returned as actual Python scalar types and not
    # NumPy scalars, so we need two separate Numba functions for each case.
    if node.outputs[0].type.ndim == 0:
        # TODO: Do we really need to compile a pass-through function like this?
        @numba.njit
        def deepcopyop(x):
            return x

    else:

        @numba.njit
        def deepcopyop(x):
            return x.copy()

    return deepcopyop


@numba_funcify.register(MakeSlice)
def numba_funcify_MakeSlice(op, **kwargs):
    # XXX: This won't work when calling into object mode (e.g. for advanced
    # indexing), because there's no Numba unboxing for its native `slice`
    # objects.

    @numba.njit
    def makeslice(*x):
        return slice(*x)

    return makeslice
