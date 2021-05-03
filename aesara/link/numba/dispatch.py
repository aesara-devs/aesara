import operator
from functools import reduce, singledispatch
from textwrap import indent

import numba
import numpy as np
import scipy
import scipy.special
from llvmlite.llvmpy.core import Type as llvm_Type
from numba import types
from numba.core.errors import TypingError
from numba.cpython.unsafe.tuple import tuple_setitem
from numba.extending import box

from aesara.compile.ops import DeepCopyOp, ViewOp
from aesara.graph.basic import Apply
from aesara.graph.fg import FunctionGraph
from aesara.graph.type import Type
from aesara.link.utils import (
    compile_function_src,
    fgraph_to_python,
    get_name_for_object,
)
from aesara.scalar.basic import (
    Cast,
    Clip,
    Composite,
    Identity,
    Scalar,
    ScalarOp,
    Second,
)
from aesara.tensor.basic import (
    Alloc,
    AllocDiag,
    AllocEmpty,
    ARange,
    MakeVector,
    Rebroadcast,
    ScalarFromTensor,
    TensorFromScalar,
)
from aesara.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from aesara.tensor.shape import Reshape, Shape, Shape_i, SpecifyShape
from aesara.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    AdvancedSubtensor,
    AdvancedSubtensor1,
    IncSubtensor,
    Subtensor,
)
from aesara.tensor.type import TensorType, tensor
from aesara.tensor.type_other import MakeSlice


def get_numba_type(
    aesara_type: Type, layout: str = "A", force_scalar: bool = False
) -> numba.types.Type:
    """Create a Numba type object for a ``Type``."""

    if isinstance(aesara_type, TensorType):
        dtype = aesara_type.numpy_dtype
        numba_dtype = numba.from_dtype(dtype)
        if force_scalar:
            return numba_dtype
        return numba.types.Array(numba_dtype, aesara_type.ndim, layout)
    elif isinstance(aesara_type, Scalar):
        dtype = np.dtype(aesara_type.dtype)
        numba_dtype = numba.from_dtype(dtype)
        return numba_dtype
    else:
        raise NotImplementedError(f"Numba type not implemented for {aesara_type}")


def create_numba_signature(node: Apply, force_scalar: bool = False) -> numba.types.Type:
    """Create a Numba type for the signature of an ``Apply`` node."""
    input_types = []
    for inp in node.inputs:
        input_types.append(get_numba_type(inp.type, force_scalar=force_scalar))

    output_types = []
    for out in node.outputs:
        output_types.append(get_numba_type(out.type, force_scalar=force_scalar))

    if len(output_types) > 1:
        return numba.types.Tuple(output_types)(*input_types)
    elif len(output_types) == 1:
        return output_types[0](*input_types)
    else:
        return numba.types.void(*input_types)


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


@numba.generated_jit(nopython=True)
def to_scalar(x):
    if isinstance(x, (numba.types.Number, numba.types.Boolean)):
        return lambda x: x
    elif isinstance(x, numba.types.Array):
        return lambda x: x.item()
    else:
        raise TypingError(f"{x} must be a scalar compatible type.")


def create_tuple_creator(f, n):
    """Construct a compile-time ``tuple``-comprehension-like loop.

    See https://github.com/numba/numba/issues/2771#issuecomment-414358902
    """
    assert n > 0

    f = numba.njit(f)

    @numba.njit
    def creator(args):
        return (f(0, *args),)

    for i in range(1, n):

        @numba.njit
        def creator(args, creator=creator, i=i):
            return creator(args) + (f(i, *args),)

    return numba.njit(lambda *args: creator(args))


def create_tuple_string(x):
    args = ", ".join(x + ([""] if len(x) == 1 else []))
    return f"({args})"


@numba.extending.overload(operator.contains)
def in_seq_empty_tuple(x, y):
    if isinstance(x, types.Tuple) and not x.types:
        return lambda x, y: False


@singledispatch
def numba_typify(data, dtype=None, **kwargs):
    return data


@singledispatch
def numba_funcify(op, node=None, storage_map=None, **kwargs):
    """Create a Numba compatible function from an Aesara `Op`."""
    raise NotImplementedError(f"No Numba conversion for the given `Op`: {op}")


@numba_funcify.register(FunctionGraph)
def numba_funcify_FunctionGraph(
    fgraph,
    node=None,
    fgraph_name="numba_funcified_fgraph",
    **kwargs,
):
    return fgraph_to_python(
        fgraph,
        numba_funcify,
        type_conversion_fn=numba_typify,
        fgraph_name=fgraph_name,
        **kwargs,
    )


@numba_funcify.register(ScalarOp)
def numba_funcify_ScalarOp(op, node, **kwargs):
    # TODO: Do we need to cache these functions so that we don't end up
    # compiling the same Numba function over and over again?

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

    scalar_op_fn_name = get_name_for_object(scalar_func)
    scalar_op_src = f"""
def {scalar_op_fn_name}({input_names}):
    return scalar_func({input_names})
    """
    scalar_op_fn = compile_function_src(scalar_op_src, scalar_op_fn_name, global_env)

    return numba.njit(scalar_op_fn)


@numba_funcify.register(Elemwise)
def numba_funcify_Elemwise(op, node, use_signature=False, identity=None, **kwargs):
    scalar_op_fn = numba_funcify(op.scalar_op, node, **kwargs)

    input_names = ", ".join([v.auto_name for v in node.inputs])

    if use_signature:
        signature = [create_numba_signature(node, force_scalar=True)]
    else:
        signature = []

    numba_vectorize = numba.vectorize(signature, identity=identity)
    global_env = {"scalar_op": scalar_op_fn, "numba_vectorize": numba_vectorize}

    elemwise_fn_name = f"elemwise_{get_name_for_object(scalar_op_fn)}"
    elemwise_src = f"""
@numba_vectorize
def {elemwise_fn_name}({input_names}):
    return scalar_op({input_names})
    """
    elemwise_fn = compile_function_src(elemwise_src, elemwise_fn_name, global_env)

    return elemwise_fn


@numba_funcify.register(CAReduce)
def numba_funcify_CAReduce(op, node, **kwargs):
    axes = op.axis
    if axes is None:
        axes = list(range(node.inputs[0].ndim))

    if hasattr(op, "acc_dtype") and op.acc_dtype is not None:
        acc_dtype = op.acc_dtype
    else:
        acc_dtype = node.outputs[0].type.dtype

    np_acc_dtype = np.dtype(acc_dtype)

    scalar_op_identity = np.asarray(op.scalar_op.identity, dtype=np_acc_dtype)

    acc_dtype = numba.np.numpy_support.from_dtype(np_acc_dtype)

    scalar_nfunc_spec = op.scalar_op.nfunc_spec

    # We construct a dummy `Apply` that has the minimum required number of
    # inputs for the scalar `Op`.  Without this, we would get a scalar function
    # with too few arguments.
    dummy_node = Apply(
        op,
        [tensor(acc_dtype, [False]) for i in range(scalar_nfunc_spec[1])],
        [tensor(acc_dtype, [False]) for o in range(scalar_nfunc_spec[2])],
    )
    elemwise_fn = numba_funcify_Elemwise(op, dummy_node, use_signature=True, **kwargs)

    def create_careduce_axis(axis, ndim):
        if ndim > 1:
            res_shape_tuple_ctor = create_tuple_creator(
                lambda i, shape: shape[i] if i < axis else shape[i + 1], ndim - 1
            )

            reaxis_first = (axis,) + tuple(i for i in range(ndim) if i != axis)

            @numba.njit(boundscheck=False)
            def careduce_axis(x):
                res_shape = res_shape_tuple_ctor(x.shape)
                x_axis_first = x.transpose(reaxis_first)

                res = np.full(res_shape, scalar_op_identity.item(), dtype=acc_dtype)
                for m in range(x.shape[axis]):
                    elemwise_fn(res, x_axis_first[m], res)

                return res

        else:

            @numba.njit(boundscheck=False)
            def careduce_axis(x):
                res = scalar_op_identity.item()
                for val in x:
                    res = elemwise_fn(res, val)
                return res

        return careduce_axis

    careduce_fn_name = f"careduce_{get_name_for_object(elemwise_fn)}"
    ndim = node.inputs[0].ndim
    careduce_axes_fns = ()
    to_reduce = reversed(sorted(axes))
    careduce_lines_src = []
    input_name = get_name_for_object(node.inputs[0])
    var_name = input_name
    for i, axis in enumerate(to_reduce):
        careduce_axes_fns += (create_careduce_axis(axis - i, ndim),)
        ndim -= 1
        last_var_name = var_name
        var_name = f"axis_{i}_res"
        careduce_lines_src.append(
            f"{var_name} = careduce_axes_fns[{i}]({last_var_name})"
        )

    careduce_assign_lines = indent("\n".join(careduce_lines_src), " " * 4)
    careduce_def_src = f"""
def {careduce_fn_name}({input_name}):
{careduce_assign_lines}
    return {var_name}
    """

    global_env = {"careduce_axes_fns": careduce_axes_fns}

    careduce_fn = compile_function_src(careduce_def_src, careduce_fn_name, global_env)

    return numba.njit(careduce_fn)


@numba_funcify.register(Composite)
def numba_funcify_Composite(op, node, **kwargs):
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
    @numba.njit
    def makeslice(*x):
        return slice(*x)

    return makeslice


@numba_funcify.register(MakeVector)
def numba_funcify_MakeVector(op, **kwargs):
    dtype = np.dtype(op.dtype)

    @numba.njit
    def makevector(*args):
        return np.array([a.item() for a in args], dtype=dtype)

    return makevector


@numba_funcify.register(Shape)
def numba_funcify_Shape(op, **kwargs):
    @numba.njit
    def shape(x):
        return np.asarray(np.shape(x))

    return shape


@numba_funcify.register(Shape_i)
def numba_funcify_Shape_i(op, **kwargs):
    i = op.i

    @numba.njit
    def shape_i(x):
        return np.shape(x)[i]

    return shape_i


@numba_funcify.register(TensorFromScalar)
def numba_funcify_TensorFromScalar(op, **kwargs):
    @numba.njit
    def tensor_from_scalar(x):
        return np.array(x)

    return tensor_from_scalar


@numba_funcify.register(ScalarFromTensor)
def numba_funcify_ScalarFromTensor(op, **kwargs):
    @numba.njit
    def scalar_from_tensor(x):
        return x.item()

    return scalar_from_tensor


@numba_funcify.register(AllocEmpty)
def numba_funcify_AllocEmpty(op, node, **kwargs):

    global_env = {"np": np, "to_scalar": to_scalar, "dtype": op.dtype}

    shape_var_names = [v.auto_name for v in node.inputs]
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

    alloc_fn = compile_function_src(alloc_def_src, "allocempty", global_env)

    return numba.njit(alloc_fn)


@numba_funcify.register(Alloc)
def numba_funcify_Alloc(op, node, **kwargs):

    global_env = {"np": np, "to_scalar": to_scalar}

    shape_var_names = [v.auto_name for v in node.inputs[1:]]
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

    alloc_fn = compile_function_src(alloc_def_src, "alloc", global_env)

    return numba.njit(alloc_fn)


@numba_funcify.register(AllocDiag)
def numba_funcify_AllocDiag(op, **kwargs):
    offset = op.offset

    @numba.njit
    def allocdiag(v):
        return np.diag(v, k=offset)

    return allocdiag


@numba_funcify.register(Second)
def numba_funcify_Second(op, node, **kwargs):
    @numba.njit
    def second(x, y):
        return y

    return second


@numba_funcify.register(DimShuffle)
def numba_funcify_DimShuffle(op, **kwargs):
    shuffle = tuple(op.shuffle)
    drop = tuple(op.drop)
    augment = tuple(op.augment)
    inplace = op.inplace

    ndim_new_shape = len(shuffle) + len(augment)
    create_zeros_tuple = create_tuple_creator(lambda _: 0, ndim_new_shape)

    if len(shuffle) > 0:

        @numba.njit
        def populate_new_shape(i, j, new_shape, shuffle_shape):
            if i in augment:
                new_shape = tuple_setitem(new_shape, i, 1)
                return j, new_shape
            else:
                new_shape = tuple_setitem(new_shape, i, shuffle_shape[j])
                return j + 1, new_shape

    else:
        # When `len(shuffle) == 0`, the `shuffle_shape[j]` expression above is
        # is typed as `getitem(Tuple(), int)`, which has no implementation
        # (since getting an item from an empty sequence doesn't make sense).
        # To avoid this compile-time error, we omit the expression altogether.
        @numba.njit
        def populate_new_shape(i, j, new_shape, shuffle_shape):
            new_shape = tuple_setitem(new_shape, i, 1)
            return j, new_shape

    @numba.njit
    def dimshuffle_inner(x, shuffle):
        res = np.transpose(x, shuffle + drop)
        shuffle_shape = res.shape[: len(shuffle)]

        new_shape = create_zeros_tuple()

        j = 0
        for i in range(len(new_shape)):
            j, new_shape = populate_new_shape(i, j, new_shape, shuffle_shape)

        # FIXME: Numba's `array.reshape` only accepts C arrays.
        res_reshape = np.reshape(np.ascontiguousarray(res), new_shape)

        if not inplace:
            return res_reshape.copy()
        else:
            return res_reshape

    # Without the following wrapper function we would see this error:
    # E   No implementation of function Function(<built-in function getitem>) found for signature:
    # E
    # E    >>> getitem(UniTuple(int64 x 2), slice<a:b>)
    # E
    # E   There are 22 candidate implementations:
    # E      - Of which 22 did not match due to:
    # E      Overload of function 'getitem': File: <numerous>: Line N/A.
    # E        With argument(s): '(UniTuple(int64 x 2), slice<a:b>)':
    # E       No match.
    # ...(on this line)...
    # E           shuffle_shape = res.shape[: len(shuffle)]
    @numba.njit
    def dimshuffle(x):
        return dimshuffle_inner(x, shuffle)

    return dimshuffle


@numba_funcify.register(Rebroadcast)
def numba_funcify_Rebroadcast(op, **kwargs):
    op_axis = tuple(op.axis.items())

    @numba.njit
    def rebroadcast(x):
        for axis, value in numba.literal_unroll(op_axis):
            if value and x.shape[axis] != 1:
                raise ValueError(
                    ("Dimension in Rebroadcast's input was supposed to be 1")
                )
        return x

    return rebroadcast


@numba.extending.intrinsic
def direct_cast(typingctx, val, typ):
    casted = typ.instance_type
    sig = casted(casted, typ)

    def codegen(context, builder, signature, args):
        val, _ = args
        context.nrt.incref(builder, signature.return_type, val)
        return val

    return sig, codegen


@numba_funcify.register(Cast)
def numba_funcify_Cast(op, **kwargs):

    dtype = np.dtype(op.o_type.dtype)
    dtype = numba.np.numpy_support.from_dtype(dtype)

    @numba.njit
    def cast(x):
        return direct_cast(x, dtype)

    return cast


@numba_funcify.register(Reshape)
def numba_funcify_Reshape(op, **kwargs):

    ndim = op.ndim
    # TODO: It might be possible/better to use
    # `numba.np.unsafe.ndarray.to_fixed_tuple` here instead
    create_zeros_tuple = create_tuple_creator(lambda _: 0, ndim)

    @numba.njit
    def reshape(x, shape):

        new_shape = create_zeros_tuple()

        for i in numba.literal_unroll(range(ndim)):
            new_shape = tuple_setitem(new_shape, i, shape[i])

        return np.reshape(x, new_shape)

    return reshape


@numba_funcify.register(SpecifyShape)
def numba_funcify_SpecifyShape(op, **kwargs):
    @numba.njit
    def specifyshape(x, shape):
        assert np.array_equal(x.shape, shape)
        return x

    return specifyshape


@numba_funcify.register(Identity)
@numba_funcify.register(ViewOp)
def numba_funcify_ViewOp(op, **kwargs):
    @numba.njit
    def viewop(x):
        return x

    return viewop


@numba_funcify.register(Clip)
def numba_funcify_Clip(op, **kwargs):
    @numba.njit
    def clip(_x, _min, _max):
        x = to_scalar(_x)
        min = to_scalar(_min)
        max = to_scalar(_max)
        return np.where(x < min, min, to_scalar(np.where(x > max, max, x)))

    return clip


@numba_funcify.register(ARange)
def numba_funcify_ARange(op, **kwargs):
    dtype = np.dtype(op.dtype)
    dtype = numba.np.numpy_support.from_dtype(dtype)

    @numba.njit
    def arange(start, stop, step):
        return np.arange(
            to_scalar(start), to_scalar(stop), to_scalar(step), dtype=dtype
        )

    return arange
