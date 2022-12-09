import operator
import warnings
from contextlib import contextmanager
from functools import singledispatch
from textwrap import dedent
from typing import TYPE_CHECKING, Callable, Optional, Union, cast

import numba
import numba.np.unsafe.ndarray as numba_ndarray
import numpy as np
import scipy
import scipy.special
from llvmlite.ir import FunctionType
from numba import types
from numba.core.decorators import _jit
from numba.core.errors import TypingError
from numba.cpython.unsafe.tuple import tuple_setitem  # noqa: F401
from numba.extending import box

from aesara import config
from aesara.compile.builders import OpFromGraph
from aesara.compile.ops import DeepCopyOp
from aesara.graph.basic import Apply, NoParams
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import Op
from aesara.graph.type import Type
from aesara.ifelse import IfElse
from aesara.link.utils import (
    compile_function_src,
    fgraph_to_python,
    unique_name_generator,
)
from aesara.scalar.basic import ScalarType
from aesara.scalar.math import Softplus
from aesara.tensor.blas import BatchedDot
from aesara.tensor.math import Dot
from aesara.tensor.shape import Reshape, Shape, Shape_i, SpecifyShape
from aesara.tensor.slinalg import Cholesky, Solve
from aesara.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    AdvancedSubtensor,
    AdvancedSubtensor1,
    IncSubtensor,
    Subtensor,
)
from aesara.tensor.type import TensorType
from aesara.tensor.type_other import MakeSlice, NoneConst


if TYPE_CHECKING:
    from aesara.graph.basic import Variable
    from aesara.graph.op import StorageMapType


def numba_njit(*args, **kwargs):

    if len(args) > 0 and callable(args[0]):
        return numba.njit(*args[1:], cache=config.numba__cache, **kwargs)(args[0])

    return numba.njit(*args, cache=config.numba__cache, **kwargs)


def numba_vectorize(*args, **kwargs):
    if len(args) > 0 and callable(args[0]):
        return numba.vectorize(*args[1:], cache=config.numba__cache, **kwargs)(args[0])

    return numba.vectorize(*args, cache=config.numba__cache, **kwargs)


# This can be removed after Numba 0.57.0 and `_jit` replaced with standard `[n]jit`
generated_jit = _jit(
    sigs=None,
    locals={},
    cache=False,
    target="cpu",
    targetoptions={},
    impl_kind="generated",
)


@singledispatch
def get_numba_type(aesara_type: Type, var: "Variable", **kwargs) -> numba.types.Type:
    r"""Create a Numba type object for a :class:`Type`.

    Parameters
    ----------
    aesara_type
        The :class:`Type` to convert.
    var
        The :class:`Variable` corresponding to `aesara_type`.
    """
    return numba.types.pyobject


@get_numba_type.register(ScalarType)
def get_numba_type_ScalarType(aesara_type, var, **kwargs):
    dtype = np.dtype(aesara_type.dtype)
    numba_dtype = numba.from_dtype(dtype)
    return numba_dtype


@get_numba_type.register(TensorType)
def get_numba_type_TensorType(
    aesara_type,
    var: "Variable",
    layout: str = "A",
    force_scalar: bool = False,
    reduce_to_scalar: bool = False,
):
    r"""
    Parameters
    ----------
    aesara_type
        The :class:`Type` to convert.
    var
        The :class:`Variable` corresponding to `aesara_type`.
    layout
        The :class:`numpy.ndarray` layout to use.
    force_scalar
        Ignore dimension information and return the corresponding Numba scalar types.
    reduce_to_scalar
        Return Numba scalars for zero dimensional :class:`TensorType`\s.
    """
    dtype = aesara_type.numpy_dtype
    numba_dtype = numba.from_dtype(dtype)
    if force_scalar or (reduce_to_scalar and getattr(aesara_type, "ndim", None) == 0):
        return numba_dtype

    readonly = getattr(var.tag, "indestructible", False)

    return numba.types.Array(numba_dtype, aesara_type.ndim, layout, readonly=readonly)


def create_numba_signature(
    node_or_fgraph: Union[FunctionGraph, Apply], **kwargs
) -> numba.types.Type:
    """Create a Numba type for the signature of an `Apply` node or `FunctionGraph`."""
    input_types = []
    for inp in node_or_fgraph.inputs:
        input_types.append(get_numba_type(inp.type, inp, **kwargs))

    output_types = []
    for out in node_or_fgraph.outputs:
        output_types.append(get_numba_type(out.type, inp, **kwargs))

    if isinstance(node_or_fgraph, FunctionGraph):
        return numba.types.Tuple(output_types)(*input_types)

    if len(output_types) > 1:
        return numba.types.Tuple(output_types)(*input_types)
    elif len(output_types) == 1:
        return output_types[0](*input_types)
    else:
        return numba.types.void(*input_types)


def slice_new(self, start, stop, step):
    fnty = FunctionType(self.pyobj, [self.pyobj, self.pyobj, self.pyobj])
    fn = self._get_function(fnty, name="PySlice_New")
    return self.builder.call(fn, [start, stop, step])


def enable_slice_boxing():
    """Enable boxing for Numba's native ``slice``s.

    TODO: this can be removed when https://github.com/numba/numba/pull/6939 is
    merged and a release is made.
    """

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

    @numba.extending.overload(operator.contains)
    def in_seq_empty_tuple(x, y):
        if isinstance(x, types.Tuple) and not x.types:
            return lambda x, y: False


enable_slice_boxing()


@generated_jit
def to_scalar(x):
    if isinstance(x, (numba.types.Number, numba.types.Boolean)):
        return lambda x: x
    elif isinstance(x, numba.types.Array):
        return lambda x: x.item()
    else:
        raise TypingError(f"{x} must be a scalar compatible type.")


def enable_slice_literals():
    """Enable lowering for ``SliceLiteral``s.

    TODO: This can be removed once https://github.com/numba/numba/pull/6996 is merged
    and a release is made.
    """
    from numba.core import types
    from numba.core.datamodel.models import SliceModel
    from numba.core.datamodel.registry import register_default
    from numba.core.imputils import lower_cast, lower_constant
    from numba.core.types.misc import SliceLiteral
    from numba.cpython.slicing import get_defaults

    register_default(numba.types.misc.SliceLiteral)(SliceModel)

    @property
    def key(self):
        return self.name

    SliceLiteral.key = key

    def make_slice_from_constant(context, builder, ty, pyval):
        sli = context.make_helper(builder, ty)
        lty = context.get_value_type(types.intp)

        (
            default_start_pos,
            default_start_neg,
            default_stop_pos,
            default_stop_neg,
            default_step,
        ) = [context.get_constant(types.intp, x) for x in get_defaults(context)]

        step = pyval.step
        if step is None:
            step_is_neg = False
            step = default_step
        else:
            step_is_neg = step < 0
            step = lty(step)

        start = pyval.start
        if start is None:
            if step_is_neg:
                start = default_start_neg
            else:
                start = default_start_pos
        else:
            start = lty(start)

        stop = pyval.stop
        if stop is None:
            if step_is_neg:
                stop = default_stop_neg
            else:
                stop = default_stop_pos
        else:
            stop = lty(stop)

        sli.start = start
        sli.stop = stop
        sli.step = step

        return sli._getvalue()

    @lower_constant(numba.types.SliceType)
    def constant_slice(context, builder, ty, pyval):
        if isinstance(ty, types.Literal):
            typ = ty.literal_type
        else:
            typ = ty

        return make_slice_from_constant(context, builder, typ, pyval)

    @lower_cast(numba.types.misc.SliceLiteral, numba.types.SliceType)
    def cast_from_literal(context, builder, fromty, toty, val):
        return make_slice_from_constant(
            context,
            builder,
            toty,
            fromty.literal_value,
        )


enable_slice_literals()


def create_tuple_creator(f, n):
    """Construct a compile-time ``tuple``-comprehension-like loop.

    See https://github.com/numba/numba/issues/2771#issuecomment-414358902
    """
    assert n > 0

    f = numba_njit(f)

    @numba_njit
    def creator(args):
        return (f(0, *args),)

    for i in range(1, n):

        @numba_njit
        def creator(args, creator=creator, i=i):
            return creator(args) + (f(i, *args),)

    return numba_njit(lambda *args: creator(args))


def create_tuple_string(x):
    args = ", ".join(x + ([""] if len(x) == 1 else []))
    return f"({args})"


def create_arg_string(x):
    args = ", ".join(x)
    return args


@contextmanager
def use_optimized_cheap_pass(*args, **kwargs):
    """Temporarily replace the cheap optimization pass with a better one."""
    from numba.core.registry import cpu_target

    context = cpu_target.target_context._internal_codegen
    old_pm = context._mpm_cheap
    new_pm = context._module_pass_manager(
        loop_vectorize=True, slp_vectorize=True, opt=3, cost="cheap"
    )
    context._mpm_cheap = new_pm
    try:
        yield
    finally:
        context._mpm_cheap = old_pm


@singledispatch
def numba_const_convert(data, dtype=None, **kwargs):
    """Create a Numba compatible constant from an Aesara `Constant`."""
    return data


def numba_funcify(obj, node=None, storage_map=None, **kwargs) -> Callable:
    """Convert `obj` to a Numba-JITable object."""
    return _numba_funcify(obj, node=node, storage_map=storage_map, **kwargs)


@singledispatch
def _numba_funcify(
    obj,
    node: Optional[Apply] = None,
    storage_map: Optional["StorageMapType"] = None,
    **kwargs,
) -> Callable:
    r"""Dispatch on Aesara object types to perform Numba conversions.

    Arguments
    ---------
    obj
        The object used to determine the appropriate conversion function based
        on its type.  This is generally an `Op` instance, but `FunctionGraph`\s
        are also supported.
    node
        When `obj` is an `Op`, this value should be the corresponding `Apply` node.
    storage_map
        A storage map with, for example, the constant and `SharedVariable` values
        of the graph being converted.

    Returns
    -------
    A `Callable` that can be JIT-compiled in Numba using `numba.jit`.

    """
    raise NotImplementedError()


@_numba_funcify.register(Op)
def numba_funcify_perform(op, node, storage_map=None, **kwargs) -> Callable:
    """Create a Numba compatible function from an Aesara `Op.perform`."""

    warnings.warn(
        f"Numba will use object mode to run {op}'s perform method",
        UserWarning,
    )

    n_outputs = len(node.outputs)

    if n_outputs > 1:
        ret_sig = numba.types.Tuple([get_numba_type(o.type, o) for o in node.outputs])
    else:
        ret_sig = get_numba_type(node.outputs[0].type, node.outputs[0])

    output_types = tuple(out.type for out in node.outputs)
    params = node.run_params()

    if params is not NoParams:
        params_val = dict(node.params_type.filter(params))

        def py_perform(inputs):
            outputs = [[None] for i in range(n_outputs)]
            op.perform(node, inputs, outputs, params_val)
            return outputs

    else:

        def py_perform(inputs):
            outputs = [[None] for i in range(n_outputs)]
            op.perform(node, inputs, outputs)
            return outputs

    if n_outputs == 1:

        def py_perform_return(inputs):
            return output_types[0].filter(py_perform(inputs)[0][0])

    else:

        def py_perform_return(inputs):
            return tuple(
                out_type.filter(out[0])
                for out_type, out in zip(output_types, py_perform(inputs))
            )

    @numba_njit
    def perform(*inputs):
        with numba.objmode(ret=ret_sig):
            ret = py_perform_return(inputs)
        return ret

    return cast(Callable, perform)


@_numba_funcify.register(OpFromGraph)
def numba_funcify_OpFromGraph(op, node=None, **kwargs):

    _ = kwargs.pop("storage_map", None)

    fgraph_fn = numba_njit(numba_funcify(op.fgraph, **kwargs))

    if len(op.fgraph.outputs) == 1:

        @numba_njit
        def opfromgraph(*inputs):
            return fgraph_fn(*inputs)[0]

    else:

        @numba_njit
        def opfromgraph(*inputs):
            return fgraph_fn(*inputs)

    return opfromgraph


@_numba_funcify.register(FunctionGraph)
def numba_funcify_FunctionGraph(
    fgraph,
    node=None,
    fgraph_name="numba_funcified_fgraph",
    **kwargs,
):
    return fgraph_to_python(
        fgraph,
        numba_funcify,
        const_conversion_fn=numba_const_convert,
        fgraph_name=fgraph_name,
        **kwargs,
    )


def create_index_func(node, objmode=False):
    """Create a Python function that assembles and uses an index on an array."""

    unique_names = unique_name_generator(
        ["subtensor", "incsubtensor", "z"], suffix_sep="_"
    )

    def convert_indices(indices, entry):
        if indices and isinstance(entry, Type):
            rval = indices.pop(0)
            return unique_names(rval)
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

    input_names = [unique_names(v, force_unique=True) for v in node.inputs]
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


@_numba_funcify.register(Subtensor)
@_numba_funcify.register(AdvancedSubtensor)
@_numba_funcify.register(AdvancedSubtensor1)
def numba_funcify_Subtensor(op, node, **kwargs):

    subtensor_def_src = create_index_func(
        node, objmode=isinstance(op, AdvancedSubtensor)
    )

    global_env = {"np": np, "objmode": numba.objmode}

    subtensor_fn = compile_function_src(
        subtensor_def_src, "subtensor", {**globals(), **global_env}
    )

    return numba_njit(subtensor_fn)


@_numba_funcify.register(IncSubtensor)
@_numba_funcify.register(AdvancedIncSubtensor)
def numba_funcify_IncSubtensor(op, node, **kwargs):

    incsubtensor_def_src = create_index_func(
        node, objmode=isinstance(op, AdvancedIncSubtensor)
    )

    global_env = {"np": np, "objmode": numba.objmode}

    incsubtensor_fn = compile_function_src(
        incsubtensor_def_src, "incsubtensor", {**globals(), **global_env}
    )

    return numba_njit(incsubtensor_fn)


@_numba_funcify.register(AdvancedIncSubtensor1)
def numba_funcify_AdvancedIncSubtensor1(op, node, **kwargs):
    inplace = op.inplace
    set_instead_of_inc = op.set_instead_of_inc

    if set_instead_of_inc:

        @numba_njit
        def advancedincsubtensor1_inplace(x, vals, idxs):
            for idx, val in zip(idxs, vals):
                x[idx] = val
            return x

    else:

        @numba_njit
        def advancedincsubtensor1_inplace(x, vals, idxs):
            for idx, val in zip(idxs, vals):
                x[idx] += val
            return x

    if inplace:
        return advancedincsubtensor1_inplace
    else:

        @numba_njit
        def advancedincsubtensor1(x, vals, idxs):
            x = x.copy()
            return advancedincsubtensor1_inplace(x, vals, idxs)

        return advancedincsubtensor1


@_numba_funcify.register(DeepCopyOp)
def numba_funcify_DeepCopyOp(op, node, **kwargs):

    # Scalars are apparently returned as actual Python scalar types and not
    # NumPy scalars, so we need two separate Numba functions for each case.

    # The type can also be RandomType with no ndims
    if not hasattr(node.outputs[0].type, "ndim") or node.outputs[0].type.ndim == 0:
        # TODO: Do we really need to compile a pass-through function like this?
        @numba_njit(inline="always")
        def deepcopyop(x):
            return x

    else:

        @numba_njit(inline="always")
        def deepcopyop(x):
            return x.copy()

    return deepcopyop


@_numba_funcify.register(MakeSlice)
def numba_funcify_MakeSlice(op, node, **kwargs):
    @numba_njit
    def makeslice(*x):
        return slice(*x)

    return makeslice


@_numba_funcify.register(Shape)
def numba_funcify_Shape(op, node, **kwargs):
    @numba_njit(inline="always")
    def shape(x):
        return np.asarray(np.shape(x))

    return shape


@_numba_funcify.register(Shape_i)
def numba_funcify_Shape_i(op, node, **kwargs):
    i = op.i

    @numba_njit(inline="always")
    def shape_i(x):
        return np.shape(x)[i]

    return shape_i


@numba.extending.intrinsic
def direct_cast(typingctx, val, typ):

    if isinstance(typ, numba.types.TypeRef):
        casted = typ.instance_type
    elif isinstance(typ, numba.types.DTypeSpec):
        casted = typ.dtype
    else:
        casted = typ

    sig = casted(casted, typ)

    def codegen(context, builder, signature, args):
        val, _ = args
        context.nrt.incref(builder, signature.return_type, val)
        return val

    return sig, codegen


@_numba_funcify.register(Reshape)
def numba_funcify_Reshape(op, node, **kwargs):
    ndim = op.ndim

    if ndim == 0:

        @numba_njit(inline="always")
        def reshape(x, shape):
            return x.item()

    else:

        @numba_njit(inline="always")
        def reshape(x, shape):
            # TODO: Use this until https://github.com/numba/numba/issues/7353 is closed.
            return np.reshape(
                np.ascontiguousarray(np.asarray(x)),
                numba_ndarray.to_fixed_tuple(shape, ndim),
            )

    return reshape


@_numba_funcify.register(SpecifyShape)
def numba_funcify_SpecifyShape(op, node, **kwargs):
    shape_inputs = node.inputs[1:]
    shape_input_names = ["shape_" + str(i) for i in range(len(shape_inputs))]

    func_conditions = [
        f"assert x.shape[{i}] == {shape_input_names}"
        for i, (shape_input, shape_input_names) in enumerate(
            zip(shape_inputs, shape_input_names)
        )
        if shape_input is not NoneConst
    ]

    func = dedent(
        f"""
        def specify_shape(x, {create_arg_string(shape_input_names)}):
            {"; ".join(func_conditions)}
            return x
        """
    )

    specify_shape = compile_function_src(func, "specify_shape", globals())
    return numba_njit(specify_shape)


def int_to_float_fn(inputs, out_dtype):
    """Create a Numba function that converts integer and boolean ``ndarray``s to floats."""

    if any(i.type.numpy_dtype.kind in "ib" for i in inputs):

        args_dtype = np.dtype(f"f{out_dtype.itemsize}")

        @numba_njit(inline="always")
        def inputs_cast(x):
            return x.astype(args_dtype)

    else:
        args_dtype_sz = max(_arg.type.numpy_dtype.itemsize for _arg in inputs)
        args_dtype = np.dtype(f"f{args_dtype_sz}")

        @numba_njit(inline="always")
        def inputs_cast(x):
            return x.astype(args_dtype)

    return inputs_cast


@_numba_funcify.register(Dot)
def numba_funcify_Dot(op, node, **kwargs):
    # Numba's `np.dot` does not support integer dtypes, so we need to cast to
    # float.

    out_dtype = node.outputs[0].type.numpy_dtype
    inputs_cast = int_to_float_fn(node.inputs, out_dtype)

    @numba_njit(inline="always")
    def dot(x, y):
        return np.asarray(np.dot(inputs_cast(x), inputs_cast(y))).astype(out_dtype)

    return dot


@_numba_funcify.register(Softplus)
def numba_funcify_Softplus(op, node, **kwargs):

    x_dtype = np.dtype(node.inputs[0].dtype)

    @numba_njit
    def softplus(x):
        if x < -37.0:
            return direct_cast(np.exp(x), x_dtype)
        elif x < 18.0:
            return direct_cast(np.log1p(np.exp(x)), x_dtype)
        elif x < 33.3:
            return direct_cast(x + np.exp(-x), x_dtype)
        else:
            return direct_cast(x, x_dtype)

    return softplus


@_numba_funcify.register(Cholesky)
def numba_funcify_Cholesky(op, node, **kwargs):
    lower = op.lower

    out_dtype = node.outputs[0].type.numpy_dtype

    if lower:

        inputs_cast = int_to_float_fn(node.inputs, out_dtype)

        @numba_njit(inline="always")
        def cholesky(a):
            return np.linalg.cholesky(inputs_cast(a)).astype(out_dtype)

    else:
        # TODO: Use SciPy's BLAS/LAPACK Cython wrappers.

        warnings.warn(
            (
                "Numba will use object mode to allow the "
                "`lower` argument to `scipy.linalg.cholesky`."
            ),
            UserWarning,
        )

        ret_sig = get_numba_type(node.outputs[0].type, node.outputs[0])

        @numba_njit
        def cholesky(a):
            with numba.objmode(ret=ret_sig):
                ret = scipy.linalg.cholesky(a, lower=lower).astype(out_dtype)
            return ret

    return cholesky


@_numba_funcify.register(Solve)
def numba_funcify_Solve(op, node, **kwargs):

    assume_a = op.assume_a
    # check_finite = op.check_finite

    if assume_a != "gen":

        lower = op.lower

        warnings.warn(
            (
                "Numba will use object mode to allow the "
                "`compute_uv` argument to `numpy.linalg.svd`."
            ),
            UserWarning,
        )

        ret_sig = get_numba_type(node.outputs[0].type, node.outputs[0])

        @numba_njit
        def solve(a, b):
            with numba.objmode(ret=ret_sig):
                ret = scipy.linalg.solve_triangular(
                    a,
                    b,
                    lower=lower,
                    # check_finite=check_finite
                )
            return ret

    else:
        out_dtype = node.outputs[0].type.numpy_dtype
        inputs_cast = int_to_float_fn(node.inputs, out_dtype)

        @numba_njit(inline="always")
        def solve(a, b):
            return np.linalg.solve(
                inputs_cast(a),
                inputs_cast(b),
                # assume_a=assume_a,
                # check_finite=check_finite,
            ).astype(out_dtype)

    return solve


@_numba_funcify.register(BatchedDot)
def numba_funcify_BatchedDot(op, node, **kwargs):
    dtype = node.outputs[0].type.numpy_dtype

    @numba_njit
    def batched_dot(x, y):
        shape = x.shape[:-1] + y.shape[2:]
        z0 = np.empty(shape, dtype=dtype)
        for i in range(z0.shape[0]):
            z0[i] = np.dot(x[i], y[i])

        return z0

    return batched_dot


# NOTE: The remaining `aesara.tensor.blas` `Op`s appear unnecessary, because
# they're only used to optimize basic `Dot` nodes, and those GEMV and GEMM
# optimizations are apparently already performed by Numba


@_numba_funcify.register(IfElse)
def numba_funcify_IfElse(op, **kwargs):
    n_outs = op.n_outs

    if n_outs > 1:

        @numba_njit
        def ifelse(cond, *args):
            if cond:
                res = args[:n_outs]
            else:
                res = args[n_outs:]

            return res

    else:

        @numba_njit
        def ifelse(cond, *args):
            if cond:
                res = args[:n_outs]
            else:
                res = args[n_outs:]

            return res[0]

    return ifelse
