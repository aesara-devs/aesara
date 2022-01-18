from textwrap import indent

import numba
import numpy as np

from aesara.link.numba.dispatch import basic as numba_basic
from aesara.link.numba.dispatch.basic import create_tuple_string, numba_funcify
from aesara.link.utils import compile_function_src, unique_name_generator
from aesara.tensor.basic import (
    Alloc,
    AllocDiag,
    AllocEmpty,
    ARange,
    ExtractDiag,
    Eye,
    Join,
    MakeVector,
    Rebroadcast,
    ScalarFromTensor,
    TensorFromScalar,
)


@numba_funcify.register(AllocEmpty)
def numba_funcify_AllocEmpty(op, node, **kwargs):

    global_env = {
        "np": np,
        "to_scalar": numba_basic.to_scalar,
        "dtype": np.dtype(op.dtype),
    }

    unique_names = unique_name_generator(
        ["np", "to_scalar", "dtype", "allocempty", "scalar_shape"], suffix_sep="_"
    )
    shape_var_names = [unique_names(v, force_unique=True) for v in node.inputs]
    shape_var_item_names = [f"{name}_item" for name in shape_var_names]
    shapes_to_items_src = indent(
        "\n".join(
            [
                f"{item_name} = to_scalar({shape_name})"
                for item_name, shape_name in zip(shape_var_item_names, shape_var_names)
            ]
        ),
        " " * 4,
    )

    alloc_def_src = f"""
def allocempty({", ".join(shape_var_names)}):
{shapes_to_items_src}
    scalar_shape = {create_tuple_string(shape_var_item_names)}
    return np.empty(scalar_shape, dtype)
    """

    alloc_fn = compile_function_src(
        alloc_def_src, "allocempty", {**globals(), **global_env}
    )

    return numba_basic.numba_njit(alloc_fn)


@numba_funcify.register(Alloc)
def numba_funcify_Alloc(op, node, **kwargs):

    global_env = {"np": np, "to_scalar": numba_basic.to_scalar}

    unique_names = unique_name_generator(
        ["np", "to_scalar", "alloc", "val_np", "val", "scalar_shape", "res"],
        suffix_sep="_",
    )
    shape_var_names = [unique_names(v, force_unique=True) for v in node.inputs[1:]]
    shape_var_item_names = [f"{name}_item" for name in shape_var_names]
    shapes_to_items_src = indent(
        "\n".join(
            [
                f"{item_name} = to_scalar({shape_name})"
                for item_name, shape_name in zip(shape_var_item_names, shape_var_names)
            ]
        ),
        " " * 4,
    )

    alloc_def_src = f"""
def alloc(val, {", ".join(shape_var_names)}):
    val_np = np.asarray(val)
{shapes_to_items_src}
    scalar_shape = {create_tuple_string(shape_var_item_names)}
    res = np.empty(scalar_shape, dtype=val_np.dtype)
    res[...] = val_np
    return res
    """

    alloc_fn = compile_function_src(alloc_def_src, "alloc", {**globals(), **global_env})

    return numba_basic.numba_njit(alloc_fn)


@numba_funcify.register(AllocDiag)
def numba_funcify_AllocDiag(op, **kwargs):
    offset = op.offset

    @numba_basic.numba_njit(inline="always")
    def allocdiag(v):
        return np.diag(v, k=offset)

    return allocdiag


@numba_funcify.register(ARange)
def numba_funcify_ARange(op, **kwargs):
    dtype = np.dtype(op.dtype)

    @numba_basic.numba_njit(inline="always")
    def arange(start, stop, step):
        return np.arange(
            numba_basic.to_scalar(start),
            numba_basic.to_scalar(stop),
            numba_basic.to_scalar(step),
            dtype=dtype,
        )

    return arange


@numba_funcify.register(Join)
def numba_funcify_Join(op, **kwargs):
    view = op.view

    if view != -1:
        # TODO: Where (and why) is this `Join.view` even being used?  From a
        # quick search, the answer appears to be "nowhere", so we should
        # probably just remove it.
        raise NotImplementedError("The `view` parameter to `Join` is not supported")

    @numba_basic.numba_njit
    def join(axis, *tensors):
        return np.concatenate(tensors, numba_basic.to_scalar(axis))

    return join


@numba_funcify.register(ExtractDiag)
def numba_funcify_ExtractDiag(op, **kwargs):
    offset = op.offset
    # axis1 = op.axis1
    # axis2 = op.axis2

    @numba_basic.numba_njit(inline="always")
    def extract_diag(x):
        return np.diag(x, k=offset)

    return extract_diag


@numba_funcify.register(Eye)
def numba_funcify_Eye(op, **kwargs):
    dtype = np.dtype(op.dtype)

    @numba_basic.numba_njit(inline="always")
    def eye(N, M, k):
        return np.eye(
            numba_basic.to_scalar(N),
            numba_basic.to_scalar(M),
            numba_basic.to_scalar(k),
            dtype=dtype,
        )

    return eye


@numba_funcify.register(MakeVector)
def numba_funcify_MakeVector(op, node, **kwargs):
    dtype = np.dtype(op.dtype)

    global_env = {"np": np, "to_scalar": numba_basic.to_scalar}

    unique_names = unique_name_generator(
        ["np", "to_scalar"],
        suffix_sep="_",
    )
    input_names = [unique_names(v, force_unique=True) for v in node.inputs]

    def create_list_string(x):
        args = ", ".join([f"to_scalar({i})" for i in x] + ([""] if len(x) == 1 else []))
        return f"[{args}]"

    makevector_def_src = f"""
def makevector({", ".join(input_names)}):
    return np.array({create_list_string(input_names)}, dtype=np.{dtype})
    """

    makevector_fn = compile_function_src(
        makevector_def_src, "makevector", {**globals(), **global_env}
    )

    return numba_basic.numba_njit(makevector_fn)


@numba_funcify.register(Rebroadcast)
def numba_funcify_Rebroadcast(op, **kwargs):
    op_axis = tuple(op.axis.items())

    @numba_basic.numba_njit
    def rebroadcast(x):
        for axis, value in numba.literal_unroll(op_axis):
            if value and x.shape[axis] != 1:
                raise ValueError(
                    ("Dimension in Rebroadcast's input was supposed to be 1")
                )
        return x

    return rebroadcast


@numba_funcify.register(TensorFromScalar)
def numba_funcify_TensorFromScalar(op, **kwargs):
    @numba_basic.numba_njit(inline="always")
    def tensor_from_scalar(x):
        return np.array(x)

    return tensor_from_scalar


@numba_funcify.register(ScalarFromTensor)
def numba_funcify_ScalarFromTensor(op, **kwargs):
    @numba_basic.numba_njit(inline="always")
    def scalar_from_tensor(x):
        return x.item()

    return scalar_from_tensor
