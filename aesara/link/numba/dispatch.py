from functools import reduce, singledispatch

import numba
import numpy as np
import scipy
import scipy.special
from llvmlite.llvmpy.core import Type as llvm_Type
from numba import types
from numba.extending import box

from aesara.compile.ops import DeepCopyOp
from aesara.graph.fg import FunctionGraph
from aesara.graph.type import Type
from aesara.link.utils import compile_function_src, fgraph_to_python
from aesara.scalar.basic import Composite, ScalarOp
from aesara.tensor.elemwise import Elemwise
from aesara.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    AdvancedSubtensor,
    AdvancedSubtensor1,
    IncSubtensor,
    Subtensor,
)
from aesara.tensor.type_other import MakeSlice


incsubtensor_ops = (IncSubtensor, AdvancedIncSubtensor1)


def slice_new(self, start, stop, step):
    fnty = llvm_Type.function(self.pyobj, [self.pyobj, self.pyobj, self.pyobj])
    fn = self._get_function(fnty, name="PySlice_New")
    return self.builder.call(fn, [start, stop, step])


@box(types.SliceType)
def box_slice(typ, val, c):
    """Implement boxing for ``slice`` objects in Numba.

    This makes it possible to return an Numba's internal representation of a
    ``slice`` object as a proper ``slice`` to Python.
    """

    start = c.box(types.int64, c.builder.extract_value(val, 0))
    stop = c.box(types.int64, c.builder.extract_value(val, 1))
    if typ.has_step:
        step = c.box(types.int64, c.builder.extract_value(val, 2))
    else:
        step = c.pyapi.get_null_object()

    slice_val = slice_new(c.pyapi, start, stop, step)

    return slice_val


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
def numba_funcify_ScalarOp(op, node, **kwargs):

    scalar_func_name = op.nfunc_spec[0]

    if scalar_func_name.startswith("scipy."):
        func_package = scipy
        scalar_func_name = scalar_func_name.split(".", 1)[-1]
    else:
        func_package = np

    if "." in scalar_func_name:
        scalar_func = reduce(getattr, [scipy] + scalar_func_name.split("."))
    else:
        scalar_func = getattr(func_package, scalar_func_name)

    input_names = ", ".join([v.auto_name for v in node.inputs])

    global_env = {"scalar_func": scalar_func}

    scalar_op_src = f"""
def scalar_op({input_names}):
    return scalar_func({input_names})
    """
    scalar_op_fn = compile_function_src(scalar_op_src, "scalar_op", global_env)

    return numba.njit(scalar_op_fn)


@numba_funcify.register(Elemwise)
def numba_funcify_Elemwise(op, node, **kwargs):
    scalar_op_fn = numba_funcify(op.scalar_op, node, **kwargs)

    input_names = ", ".join([v.auto_name for v in node.inputs])

    global_env = {"scalar_op": scalar_op_fn, "vectorize": numba.vectorize}

    elemwise_src = f"""
@vectorize
def elemwise({input_names}):
    return scalar_op({input_names})
    """
    elemwise_fn = compile_function_src(elemwise_src, "elemwise", global_env)

    return elemwise_fn


@numba_funcify.register(Composite)
def numba_funcify_Composite(op, vectorize=True, **kwargs):
    numba_impl = numba.njit(numba_funcify(op.fgraph, **kwargs))

    @numba.njit
    def composite(*args):
        return numba_impl(*args)[0]

    return composite


def create_index_func(node, objmode=False):
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

    set_or_inc = isinstance(
        node.op, (IncSubtensor, AdvancedIncSubtensor1, AdvancedIncSubtensor)
    )
    index_start_idx = 1 + int(set_or_inc)

    input_names = [v.auto_name for v in node.inputs]
    op_indices = list(node.inputs[index_start_idx:])
    idx_list = getattr(node.op, "idx_list", None)

    indices_creation_src = (
        tuple(convert_indices(op_indices, idx) for idx in idx_list)
        if idx_list
        else tuple(input_names[index_start_idx:])
    )

    if len(indices_creation_src) == 1:
        indices_creation_src = f"indices = ({indices_creation_src[0]},)"
    else:
        indices_creation_src = ", ".join(indices_creation_src)
        indices_creation_src = f"indices = ({indices_creation_src})"

    if set_or_inc:
        fn_name = "incsubtensor"
        if node.op.inplace:
            index_prologue = f"z = {input_names[0]}"
        else:
            index_prologue = f"z = np.copy({input_names[0]})"

        if node.inputs[1].ndim == 0:
            # TODO FIXME: This is a hack to get around a weird Numba typing
            # issue.  See https://github.com/numba/numba/issues/6000
            y_name = f"{input_names[1]}.item()"
        else:
            y_name = input_names[1]

        if node.op.set_instead_of_inc:
            index_body = f"z[indices] = {y_name}"
        else:
            index_body = f"z[indices] += {y_name}"
    else:
        fn_name = "subtensor"
        index_prologue = ""
        index_body = f"z = {input_names[0]}[indices]"

    if objmode:
        output_var = node.outputs[0]

        if not set_or_inc:
            # Since `z` is being "created" while in object mode, it's
            # considered an "outgoing" variable and needs to be manually typed
            output_sig = f"z='{output_var.dtype}[{', '.join([':'] * output_var.ndim)}]'"
        else:
            output_sig = ""

        index_body = f"""
    with objmode({output_sig}):
        {index_body}
        """

    subtensor_def_src = f"""
def {fn_name}({", ".join(input_names)}):
    {index_prologue}
    {indices_creation_src}
    {index_body}
    return z
    """

    return subtensor_def_src


@numba_funcify.register(Subtensor)
@numba_funcify.register(AdvancedSubtensor)
@numba_funcify.register(AdvancedSubtensor1)
def numba_funcify_Subtensor(op, node, **kwargs):

    subtensor_def_src = create_index_func(
        node, objmode=isinstance(op, AdvancedSubtensor)
    )

    global_env = {"np": np, "objmode": numba.objmode}

    subtensor_fn = compile_function_src(subtensor_def_src, "subtensor", global_env)

    return numba.njit(subtensor_fn)


@numba_funcify.register(IncSubtensor)
@numba_funcify.register(AdvancedIncSubtensor)
@numba_funcify.register(AdvancedIncSubtensor1)
def numba_funcify_IncSubtensor(op, node, **kwargs):

    incsubtensor_def_src = create_index_func(
        node, objmode=isinstance(op, AdvancedIncSubtensor)
    )

    global_env = {"np": np, "objmode": numba.objmode}

    incsubtensor_fn = compile_function_src(
        incsubtensor_def_src, "incsubtensor", global_env
    )

    return numba.njit(incsubtensor_fn)


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
    """
    XXX: This requires a ``slice`` boxing implementation to work with Numba's
    object mode.
    """

    @numba.njit
    def makeslice(*x):
        return slice(*x)

    return makeslice
