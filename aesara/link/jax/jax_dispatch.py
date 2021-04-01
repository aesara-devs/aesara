import warnings
from collections.abc import Sequence
from functools import reduce, singledispatch, update_wrapper
from warnings import warn

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from numpy.random import RandomState

from aesara.compile.ops import DeepCopyOp, ViewOp
from aesara.configdefaults import config
from aesara.graph.fg import FunctionGraph
from aesara.graph.type import CType
from aesara.ifelse import IfElse
from aesara.scalar.basic import Cast, Clip, Composite, Identity, ScalarOp, Second
from aesara.scan.op import Scan
from aesara.scan.utils import scan_args as ScanArgs
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
from aesara.tensor.blas import BatchedDot
from aesara.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from aesara.tensor.extra_ops import (
    Bartlett,
    CumOp,
    DiffOp,
    FillDiagonal,
    FillDiagonalOffset,
    RavelMultiIndex,
    RepeatOp,
    Unique,
    UnravelIndex,
)
from aesara.tensor.math import Dot, MaxAndArgmax
from aesara.tensor.nlinalg import (
    SVD,
    Det,
    Eig,
    Eigh,
    MatrixInverse,
    QRFull,
    QRIncomplete,
)
from aesara.tensor.nnet.basic import LogSoftmax, Softmax
from aesara.tensor.nnet.sigm import ScalarSoftplus
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.shape import Reshape, Shape, Shape_i, SpecifyShape
from aesara.tensor.slinalg import Cholesky, Solve
from aesara.tensor.subtensor import (  # This is essentially `np.take`; Boolean mask indexing and setting
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    AdvancedSubtensor,
    AdvancedSubtensor1,
    IncSubtensor,
    Subtensor,
    get_idx_list,
)
from aesara.tensor.type_other import MakeSlice


# For use with JAX since JAX doesn't support 'str' arguments
numpy_bit_gens = {"MT19937": 0, "PCG64": 1, "Philox": 2, "SFC64": 3}


if config.floatX == "float64":
    jax.config.update("jax_enable_x64", True)
else:
    jax.config.update("jax_enable_x64", False)

# XXX: Enabling this will break some shape-based functionality, and severely
# limit the types of graphs that can be converted.
# See https://github.com/google/jax/blob/4d556837cc9003492f674c012689efc3d68fdf5f/design_notes/omnistaging.md
# Older versions < 0.2.0 do not have this flag so we don't need to set it.
try:
    jax.config.disable_omnistaging()
except AttributeError:
    pass
except Exception as e:
    # The version might be >= 0.2.12, which means that omnistaging can't be
    # disabled
    warnings.warn(f"JAX omnistaging couldn't be disabled: {e}")

subtensor_ops = (Subtensor, AdvancedSubtensor1, AdvancedSubtensor)
incsubtensor_ops = (IncSubtensor, AdvancedIncSubtensor1)


def compose_jax_funcs(out_node, fgraph_inputs, memo=None):
    """Compose JAX implementations of node operations.

    This function walks the graph given by the `Apply` node, `out_node`, and
    creates JAX JIT-able functions for its input and output variables.

    Parameters
    ----------
    out_node: aesara.graph.basic.Apply
        The node for which we want to construct a JAX JIT-able function.
    fgraph_inputs: List[Variable]
        The inputs--in a `FunctionGraph` sense--to `out_node`.
    memo: Mapping (Optional)
        A map from visited nodes to their JAX functions.

    Outputs
    -------
    A `function` object that represents the composed JAX operations and takes
    the same form of inputs as `fgraph_inputs`.

    """
    if memo is None:
        memo = {}

    if out_node in memo:
        return memo[out_node]

    jax_return_func = jax_funcify(out_node.op)

    # We create a list of JAX-able functions that produce the values of each
    # input variable for `out_node`.
    input_funcs = []
    for i in out_node.inputs:
        if i in fgraph_inputs:
            # This input is a top-level input (i.e. an input to the
            # `FunctionGraph` in which this `out_node` resides)
            idx = fgraph_inputs.index(i)

            i_dtype = getattr(i, "dtype", None)

            def jax_inputs_func(*inputs, i_dtype=i_dtype, idx=idx):
                return jax_typify(inputs[idx], i_dtype)

            input_f = jax_inputs_func

        elif i.owner is None:
            # This input is something like an `aesara.graph.basic.Constant`

            i_dtype = getattr(i, "dtype", None)
            i_data = i.data

            def jax_data_func(*inputs, i_dtype=i_dtype, i_data=i_data):
                return jax_typify(i_data, i_dtype)

            input_f = jax_data_func
        else:
            # This input is the output of another node, so we need to
            # generate a JAX-able function for its subgraph
            input_f = compose_jax_funcs(i.owner, fgraph_inputs, memo)

            if i.owner.nout > 1:
                # This input is one of multiple outputs from the `i.owner`
                # node, and we need to determine exactly which one it is and
                # create a JAX-able function that returns only it.
                out_idx = i.owner.outputs.index(i)
                (out_fn,) = input_f

                def jax_multiout_func(*inputs, out_idx=out_idx, out_fn=out_fn):
                    return out_fn(*inputs)[out_idx]

                input_f = jax_multiout_func

        assert callable(input_f)

        input_funcs.append(input_f)

    if not isinstance(jax_return_func, Sequence):
        jax_return_func = [jax_return_func]

    jax_funcs = []
    for return_func in jax_return_func:

        def jax_func(*inputs):
            func_args = [fn(*inputs) for fn in input_funcs]
            return return_func(*func_args)

        jax_funcs.append(update_wrapper(jax_func, return_func))

    if len(out_node.outputs) == 1:
        jax_funcs = jax_funcs[0]

    memo[out_node] = jax_funcs

    return jax_funcs


@singledispatch
def jax_typify(data, dtype):
    """Convert instances of Aesara `Type`s to JAX types."""
    if dtype is None:
        return data
    else:
        return jnp.array(data, dtype=dtype)


@jax_typify.register(np.ndarray)
def jax_typify_ndarray(data, dtype):
    return jnp.array(data, dtype=dtype)


@jax_typify.register(RandomState)
def jax_typify_RandomState(state, dtype):
    state = state.get_state(legacy=False)
    state["bit_generator"] = numpy_bit_gens[state["bit_generator"]]
    return state


@singledispatch
def jax_funcify(op):
    """Create a JAX compatible function from an Aesara `Op`."""
    raise NotImplementedError(f"No JAX conversion for the given `Op`: {op}")


@jax_funcify.register(MakeSlice)
def jax_funcify_MakeSlice(op):
    def makeslice(*x):
        return slice(*x)

    return makeslice


@jax_funcify.register(ScalarOp)
def jax_funcify_ScalarOp(op):
    func_name = op.nfunc_spec[0]

    if "." in func_name:
        jnp_func = reduce(getattr, [jax] + func_name.split("."))
    else:
        jnp_func = getattr(jnp, func_name)

    if hasattr(op, "nfunc_variadic"):
        # These are special cases that handle invalid arities due to the broken
        # Aesara `Op` type contract (e.g. binary `Op`s that also function as
        # their own variadic counterparts--even when those counterparts already
        # exist as independent `Op`s).
        jax_variadic_func = getattr(jnp, op.nfunc_variadic)

        def elemwise(*args):
            if len(args) > op.nfunc_spec[1]:
                return jax_variadic_func(
                    jnp.stack(jnp.broadcast_arrays(*args), axis=0), axis=0
                )
            else:
                return jnp_func(*args)

        return elemwise
    else:
        return jnp_func


@jax_funcify.register(Clip)
def jax_funcify_Clip(op):
    def clip(x, min, max):
        return jnp.where(x < min, min, jnp.where(x > max, max, x))

    return clip


@jax_funcify.register(Identity)
def jax_funcify_Identity(op):
    def identity(x):
        return x

    return identity


@jax_funcify.register(Softmax)
def jax_funcify_Softmax(op):
    def softmax(x):
        return jax.nn.softmax(x)

    return softmax


@jax_funcify.register(LogSoftmax)
def jax_funcify_LogSoftmax(op):
    def log_softmax(x):
        return jax.nn.log_softmax(x)

    return log_softmax


@jax_funcify.register(ScalarSoftplus)
def jax_funcify_ScalarSoftplus(op):
    def scalarsoftplus(x):
        return jnp.where(x < -30.0, 0.0, jnp.where(x > 30.0, x, jnp.log1p(jnp.exp(x))))

    return scalarsoftplus


@jax_funcify.register(Second)
def jax_funcify_Second(op):
    def second(x, y):
        return jnp.broadcast_to(y, x.shape)

    return second


@jax_funcify.register(AllocDiag)
def jax_funcify_AllocDiag(op):
    offset = op.offset

    def allocdiag(v, offset=offset):
        return jnp.diag(v, k=offset)

    return allocdiag


@jax_funcify.register(AllocEmpty)
def jax_funcify_AllocEmpty(op):
    def allocempty(*shape):
        return jnp.empty(shape, dtype=op.dtype)

    return allocempty


@jax_funcify.register(Alloc)
def jax_funcify_Alloc(op):
    def alloc(x, *shape):
        res = jnp.broadcast_to(x, shape)
        return res

    return alloc


@jax_funcify.register(Dot)
def jax_funcify_Dot(op):
    def dot(x, y):
        return jnp.dot(x, y)

    return dot


@jax_funcify.register(ARange)
def jax_funcify_ARange(op):
    # XXX: This currently requires concrete arguments.
    def arange(start, stop, step):
        return jnp.arange(start, stop, step, dtype=op.dtype)

    return arange


def jnp_safe_copy(x):
    try:
        res = jnp.copy(x)
    except NotImplementedError:
        warn("`jnp.copy` is not implemented yet. " "Using the object's `copy` method.")
        if hasattr(x, "copy"):
            res = jnp.array(x.copy())
        else:
            warn(f"Object has no `copy` method: {x}")
            res = x

    return res


@jax_funcify.register(DeepCopyOp)
def jax_funcify_DeepCopyOp(op):
    def deepcopyop(x):
        return jnp_safe_copy(x)

    return deepcopyop


@jax_funcify.register(Shape)
def jax_funcify_Shape(op):
    def shape(x):
        return jnp.shape(x)

    return shape


@jax_funcify.register(Shape_i)
def jax_funcify_Shape_i(op):
    i = op.i

    def shape_i(x):
        return jnp.shape(x)[i]

    return shape_i


@jax_funcify.register(SpecifyShape)
def jax_funcify_SpecifyShape(op):
    def specifyshape(x, shape):
        assert x.ndim == len(shape)
        assert jnp.all(x.shape == tuple(shape)), (
            "got shape",
            x.shape,
            "expected",
            shape,
        )
        return x

    return specifyshape


@jax_funcify.register(Rebroadcast)
def jax_funcify_Rebroadcast(op):
    op_axis = op.axis

    def rebroadcast(x):
        for axis, value in op_axis.items():
            if value and x.shape[axis] != 1:
                raise ValueError(
                    "Dimension %s in Rebroadcast's input was"
                    " supposed to be 1 (got %s instead)" % (axis, x.shape[axis])
                )
        return x

    return rebroadcast


@jax_funcify.register(ViewOp)
def jax_funcify_ViewOp(op):
    def viewop(x):
        return x

    return viewop


@jax_funcify.register(Cast)
def jax_funcify_Cast(op):
    def cast(x):
        return jnp.array(x).astype(op.o_type.dtype)

    return cast


@jax_funcify.register(TensorFromScalar)
def jax_funcify_TensorFromScalar(op):
    def tensor_from_scalar(x):
        return jnp.array(x)

    return tensor_from_scalar


@jax_funcify.register(ScalarFromTensor)
def jax_funcify_ScalarFromTensor(op):
    def scalar_from_tensor(x):
        return jnp.array(x).flatten()[0]

    return scalar_from_tensor


@jax_funcify.register(Elemwise)
def jax_funcify_Elemwise(op):
    scalar_op = op.scalar_op
    return jax_funcify(scalar_op)


@jax_funcify.register(Composite)
def jax_funcify_Composite(op):
    jax_impl = jax_funcify(op.fgraph)
    return jax_impl


@jax_funcify.register(Scan)
def jax_funcify_Scan(op):
    inner_fg = FunctionGraph(op.inputs, op.outputs)
    jax_aet_inner_func = jax_funcify(inner_fg)

    def scan(*outer_inputs):
        scan_args = ScanArgs(
            list(outer_inputs), [None] * op.n_outs, op.inputs, op.outputs, op.info
        )

        # `outer_inputs` is a list with the following composite form:
        # [n_steps]
        # + outer_in_seqs
        # + outer_in_mit_mot
        # + outer_in_mit_sot
        # + outer_in_sit_sot
        # + outer_in_shared
        # + outer_in_nit_sot
        # + outer_in_non_seqs
        n_steps = scan_args.n_steps
        seqs = scan_args.outer_in_seqs

        # TODO: mit_mots
        mit_mot_in_slices = []

        mit_sot_in_slices = []
        for tap, seq in zip(scan_args.mit_sot_in_slices, scan_args.outer_in_mit_sot):
            neg_taps = [abs(t) for t in tap if t < 0]
            pos_taps = [abs(t) for t in tap if t > 0]
            max_neg = max(neg_taps) if neg_taps else 0
            max_pos = max(pos_taps) if pos_taps else 0
            init_slice = seq[: max_neg + max_pos]
            mit_sot_in_slices.append(init_slice)

        sit_sot_in_slices = [seq[0] for seq in scan_args.outer_in_sit_sot]

        init_carry = (
            mit_mot_in_slices,
            mit_sot_in_slices,
            sit_sot_in_slices,
            scan_args.outer_in_shared,
            scan_args.outer_in_non_seqs,
        )

        def jax_args_to_inner_scan(op, carry, x):
            # `carry` contains all inner-output taps, non_seqs, and shared
            # terms
            (
                inner_in_mit_mot,
                inner_in_mit_sot,
                inner_in_sit_sot,
                inner_in_shared,
                inner_in_non_seqs,
            ) = carry

            # `x` contains the in_seqs
            inner_in_seqs = x

            # `inner_scan_inputs` is a list with the following composite form:
            # inner_in_seqs
            # + sum(inner_in_mit_mot, [])
            # + sum(inner_in_mit_sot, [])
            # + inner_in_sit_sot
            # + inner_in_shared
            # + inner_in_non_seqs
            inner_in_mit_sot_flatten = []
            for array, index in zip(inner_in_mit_sot, scan_args.mit_sot_in_slices):
                inner_in_mit_sot_flatten.extend(array[jnp.array(index)])

            inner_scan_inputs = sum(
                [
                    inner_in_seqs,
                    inner_in_mit_mot,
                    inner_in_mit_sot_flatten,
                    inner_in_sit_sot,
                    inner_in_shared,
                    inner_in_non_seqs,
                ],
                [],
            )

            return inner_scan_inputs

        def inner_scan_outs_to_jax_outs(
            op,
            old_carry,
            inner_scan_outs,
        ):
            (
                inner_in_mit_mot,
                inner_in_mit_sot,
                inner_in_sit_sot,
                inner_in_shared,
                inner_in_non_seqs,
            ) = old_carry

            def update_mit_sot(mit_sot, new_val):
                return jnp.concatenate([mit_sot[1:], new_val[None, ...]], axis=0)

            inner_out_mit_sot = [
                update_mit_sot(mit_sot, new_val)
                for mit_sot, new_val in zip(inner_in_mit_sot, inner_scan_outs)
            ]

            # This should contain all inner-output taps, non_seqs, and shared
            # terms
            if not inner_in_sit_sot:
                inner_out_sit_sot = []
            else:
                inner_out_sit_sot = inner_scan_outs
            new_carry = (
                inner_in_mit_mot,
                inner_out_mit_sot,
                inner_out_sit_sot,
                inner_in_shared,
                inner_in_non_seqs,
            )

            return new_carry

        def jax_inner_func(carry, x):
            inner_args = jax_args_to_inner_scan(op, carry, x)
            inner_scan_outs = [fn(*inner_args) for fn in jax_aet_inner_func]
            new_carry = inner_scan_outs_to_jax_outs(op, carry, inner_scan_outs)
            return new_carry, inner_scan_outs

        _, scan_out = jax.lax.scan(jax_inner_func, init_carry, seqs, length=n_steps)

        # We need to prepend the initial values so that the JAX output will
        # match the raw `Scan` `Op` output and, thus, work with a downstream
        # `Subtensor` `Op` introduced by the `scan` helper function.
        def append_scan_out(scan_in_part, scan_out_part):
            return jnp.concatenate([scan_in_part[:-n_steps], scan_out_part], axis=0)

        if scan_args.outer_in_mit_sot:
            scan_out_final = [
                append_scan_out(init, out)
                for init, out in zip(scan_args.outer_in_mit_sot, scan_out)
            ]
        elif scan_args.outer_in_sit_sot:
            scan_out_final = [
                append_scan_out(init, out)
                for init, out in zip(scan_args.outer_in_sit_sot, scan_out)
            ]

        if len(scan_out_final) == 1:
            scan_out_final = scan_out_final[0]
        return scan_out_final

    return scan


@jax_funcify.register(IfElse)
def jax_funcify_IfElse(op):
    n_outs = op.n_outs

    def ifelse(cond, *args, n_outs=n_outs):
        res = jax.lax.cond(
            cond, lambda _: args[:n_outs], lambda _: args[n_outs:], operand=None
        )
        return res if n_outs > 1 else res[0]

    return ifelse


def convert_indices(indices, entry):
    if indices and isinstance(entry, CType):
        rval = indices.pop(0)
        return rval
    elif isinstance(entry, slice):
        return slice(
            convert_indices(indices, entry.start),
            convert_indices(indices, entry.stop),
            convert_indices(indices, entry.step),
        )
    else:
        return entry


@jax_funcify.register(Subtensor)
def jax_funcify_Subtensor(op):

    idx_list = getattr(op, "idx_list", None)

    def subtensor(x, *ilists):

        if idx_list:
            cdata = get_idx_list((x,) + ilists, idx_list)
        else:
            cdata = ilists

        if len(cdata) == 1:
            cdata = cdata[0]

        return x.__getitem__(cdata)

    return subtensor


_ = [jax_funcify.register(op, jax_funcify_Subtensor) for op in subtensor_ops]


def jax_funcify_IncSubtensor(op):

    idx_list = getattr(op, "idx_list", None)

    if getattr(op, "set_instead_of_inc", False):
        jax_fn = jax.ops.index_update
    else:
        jax_fn = jax.ops.index_add

    def incsubtensor(x, y, *ilist, jax_fn=jax_fn, idx_list=idx_list):
        _ilist = list(ilist)
        cdata = (
            tuple(convert_indices(_ilist, idx) for idx in idx_list)
            if idx_list
            else _ilist
        )
        if len(cdata) == 1:
            cdata = cdata[0]

        return jax_fn(x, cdata, y)

    return incsubtensor


_ = [jax_funcify.register(op, jax_funcify_IncSubtensor) for op in incsubtensor_ops]


@jax_funcify.register(AdvancedIncSubtensor)
def jax_funcify_AdvancedIncSubtensor(op):

    if getattr(op, "set_instead_of_inc", False):
        jax_fn = jax.ops.index_update
    else:
        jax_fn = jax.ops.index_add

    def advancedincsubtensor(x, y, *ilist, jax_fn=jax_fn):
        return jax_fn(x, ilist, y)

    return advancedincsubtensor


@jax_funcify.register(FunctionGraph)
def jax_funcify_FunctionGraph(fgraph):

    out_nodes = [r.owner for r in fgraph.outputs if r.owner is not None]
    jax_funcs = [compose_jax_funcs(o, fgraph.inputs) for o in out_nodes]

    return jax_funcs


@jax_funcify.register(CAReduce)
def jax_funcify_CAReduce(op):
    axis = op.axis
    op_nfunc_spec = getattr(op, "nfunc_spec", None)
    scalar_nfunc_spec = getattr(op.scalar_op, "nfunc_spec", None)
    scalar_op_name = getattr(op.scalar_op, "name", None)
    scalar_op_identity = getattr(op.scalar_op, "identity", None)
    acc_dtype = getattr(op, "acc_dtype", None)

    def careduce(x):
        nonlocal axis, op_nfunc_spec, scalar_nfunc_spec, scalar_op_name, scalar_op_identity, acc_dtype

        if axis is None:
            axis = list(range(x.ndim))

        if acc_dtype is None:
            acc_dtype = x.dtype.type

        if op_nfunc_spec:
            jax_op = getattr(jnp, op_nfunc_spec[0])
            return jax_op(x, axis=axis).astype(acc_dtype)

        # The Aesara `Op` didn't tell us which NumPy equivalent to use (or
        # there isn't one), so we use this fallback approach
        if scalar_nfunc_spec:
            scalar_fn_name = scalar_nfunc_spec[0]
        elif scalar_op_name:
            scalar_fn_name = scalar_op_name

        to_reduce = reversed(sorted(axis))

        if to_reduce:
            # In this case, we need to use the `jax.lax` function (if there
            # is one), and not the `jnp` version.
            jax_op = getattr(jax.lax, scalar_fn_name)
            init_value = jnp.array(scalar_op_identity, dtype=acc_dtype)
            return jax.lax.reduce(x, init_value, jax_op, to_reduce).astype(acc_dtype)
        else:
            return x

    return careduce


@jax_funcify.register(MakeVector)
def jax_funcify_MakeVector(op):
    def makevector(*x):
        return jnp.array(x, dtype=op.dtype)

    return makevector


@jax_funcify.register(Reshape)
def jax_funcify_Reshape(op):
    def reshape(x, shape):
        return jnp.reshape(x, shape)

    return reshape


@jax_funcify.register(DimShuffle)
def jax_funcify_DimShuffle(op):
    def dimshuffle(x):

        res = jnp.transpose(x, op.shuffle + op.drop)

        shape = list(res.shape[: len(op.shuffle)])

        for augm in op.augment:
            shape.insert(augm, 1)

        res = jnp.reshape(res, shape)

        if not op.inplace:
            res = jnp_safe_copy(res)

        return res

    return dimshuffle


@jax_funcify.register(Join)
def jax_funcify_Join(op):
    def join(axis, *tensors):
        # tensors could also be tuples, and in this case they don't have a ndim
        tensors = [jnp.asarray(tensor) for tensor in tensors]
        view = op.view
        if (view != -1) and all(
            [
                tensor.shape[axis] == 0
                for tensor in tensors[0:view] + tensors[view + 1 :]
            ]
        ):
            return tensors[view]

        else:
            ndim = tensors[0].ndim
            if axis < -ndim:
                raise IndexError(
                    f"Join axis {int(axis)} out of bounds [0, {int(ndim)})"
                )

            return jnp.concatenate(tensors, axis=axis)

    return join


@jax_funcify.register(MaxAndArgmax)
def jax_funcify_MaxAndArgmax(op):
    axis = op.axis

    def maxandargmax(x, axis=axis):
        if axis is None:
            axes = tuple(range(x.ndim))
        else:
            axes = tuple(int(ax) for ax in axis)

        max_res = jnp.max(x, axis)

        # NumPy does not support multiple axes for argmax; this is a
        # work-around
        keep_axes = jnp.array(
            [i for i in range(x.ndim) if i not in axes], dtype="int64"
        )
        # Not-reduced axes in front
        transposed_x = jnp.transpose(
            x, jnp.concatenate((keep_axes, jnp.array(axes, dtype="int64")))
        )
        kept_shape = transposed_x.shape[: len(keep_axes)]
        reduced_shape = transposed_x.shape[len(keep_axes) :]

        # Numpy.prod returns 1.0 when arg is empty, so we cast it to int64
        # Otherwise reshape would complain citing float arg
        new_shape = kept_shape + (
            jnp.prod(jnp.array(reduced_shape, dtype="int64"), dtype="int64"),
        )
        reshaped_x = transposed_x.reshape(new_shape)

        max_idx_res = jnp.argmax(reshaped_x, axis=-1).astype("int64")

        return max_res, max_idx_res

    return maxandargmax


@jax_funcify.register(ExtractDiag)
def jax_funcify_ExtractDiag(op):
    offset = op.offset
    axis1 = op.axis1
    axis2 = op.axis2

    def extract_diag(x, offset=offset, axis1=axis1, axis2=axis2):
        return jnp.diagonal(x, offset=offset, axis1=axis1, axis2=axis2)

    return extract_diag


@jax_funcify.register(Cholesky)
def jax_funcify_Cholesky(op):
    lower = op.lower

    def cholesky(a, lower=lower):
        return jsp.linalg.cholesky(a, lower=lower).astype(a.dtype)

    return cholesky


@jax_funcify.register(Solve)
def jax_funcify_Solve(op):

    if op.A_structure == "lower_triangular":
        lower = True
    else:
        lower = False

    def solve(a, b, lower=lower):
        return jsp.linalg.solve(a, b, lower=lower)

    return solve


@jax_funcify.register(Det)
def jax_funcify_Det(op):
    def det(x):
        return jnp.linalg.det(x)

    return det


@jax_funcify.register(Eig)
def jax_funcify_Eig(op):
    def eig(x):
        return jnp.linalg.eig(x)

    return eig


@jax_funcify.register(Eigh)
def jax_funcify_Eigh(op):
    uplo = op.UPLO

    def eigh(x, uplo=uplo):
        return jnp.linalg.eigh(x, UPLO=uplo)

    return eigh


@jax_funcify.register(MatrixInverse)
def jax_funcify_MatrixInverse(op):
    def matrix_inverse(x):
        return jnp.linalg.inv(x)

    return matrix_inverse


@jax_funcify.register(QRFull)
def jax_funcify_QRFull(op):
    mode = op.mode

    def qr_full(x, mode=mode):
        return jnp.linalg.qr(x, mode=mode)

    return qr_full


@jax_funcify.register(QRIncomplete)
def jax_funcify_QRIncomplete(op):
    mode = op.mode

    def qr_incomplete(x, mode=mode):
        return jnp.linalg.qr(x, mode=mode)

    return qr_incomplete


@jax_funcify.register(SVD)
def jax_funcify_SVD(op):
    full_matrices = op.full_matrices
    compute_uv = op.compute_uv

    def svd(x, full_matrices=full_matrices, compute_uv=compute_uv):
        return jnp.linalg.svd(x, full_matrices=full_matrices, compute_uv=compute_uv)

    return svd


@jax_funcify.register(CumOp)
def jax_funcify_CumOp(op):
    axis = op.axis
    mode = op.mode

    def cumop(x, axis=axis, mode=mode):
        if mode == "add":
            return jnp.cumsum(x, axis=axis)
        else:
            return jnp.cumprod(x, axis=axis)

    return cumop


@jax_funcify.register(DiffOp)
def jax_funcify_DiffOp(op):
    n = op.n
    axis = op.axis

    def diffop(x, n=n, axis=axis):
        return jnp.diff(x, n=n, axis=axis)

    return diffop


@jax_funcify.register(RepeatOp)
def jax_funcify_RepeatOp(op):
    axis = op.axis

    def repeatop(x, repeats, axis=axis):
        return jnp.repeat(x, repeats, axis=axis)

    return repeatop


@jax_funcify.register(Bartlett)
def jax_funcify_Bartlett(op):
    def bartlett(x):
        return jnp.bartlett(x)

    return bartlett


@jax_funcify.register(FillDiagonal)
def jax_funcify_FillDiagonal(op):

    # def filldiagonal(a, val):
    #     if a.ndim == 2:
    #         step = a.shape[1] + 1
    #         end = a.shape[1] * a.shape[1]
    #         a.flat[:end:step] = val
    #     else:
    #         jnp.fill_diagonal(a, val)
    #
    #     return a
    #
    # return filldiagonal

    raise NotImplementedError("flatiter not implemented in JAX")


@jax_funcify.register(FillDiagonalOffset)
def jax_funcify_FillDiagonalOffset(op):

    # def filldiagonaloffset(a, val, offset):
    #     height, width = a.shape
    #
    #     if offset >= 0:
    #         start = offset
    #         num_of_step = min(min(width, height), width - offset)
    #     else:
    #         start = -offset * a.shape[1]
    #         num_of_step = min(min(width, height), height + offset)
    #
    #     step = a.shape[1] + 1
    #     end = start + step * num_of_step
    #     a.flat[start:end:step] = val
    #
    #     return a
    #
    # return filldiagonaloffset

    raise NotImplementedError("flatiter not implemented in JAX")


@jax_funcify.register(Unique)
def jax_funcify_Unique(op):
    axis = op.axis

    if axis is not None:
        raise NotImplementedError(
            "jax.numpy.unique is not implemented for the axis argument"
        )

    return_index = op.return_index
    return_inverse = op.return_inverse
    return_counts = op.return_counts

    def unique(
        x,
        return_index=return_index,
        return_inverse=return_inverse,
        return_counts=return_counts,
        axis=axis,
    ):
        ret = jnp.lax_numpy._unique1d(x, return_index, return_inverse, return_counts)
        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    return unique


@jax_funcify.register(UnravelIndex)
def jax_funcify_UnravelIndex(op):
    order = op.order

    warn("JAX ignores the `order` parameter in `unravel_index`.")

    def unravelindex(indices, dims, order=order):
        return jnp.unravel_index(indices, dims)

    return unravelindex


@jax_funcify.register(RavelMultiIndex)
def jax_funcify_RavelMultiIndex(op):
    mode = op.mode
    order = op.order

    def ravelmultiindex(*inp, mode=mode, order=order):
        multi_index, dims = inp[:-1], inp[-1]
        return jnp.ravel_multi_index(multi_index, dims, mode=mode, order=order)

    return ravelmultiindex


@jax_funcify.register(Eye)
def jax_funcify_Eye(op):
    dtype = op.dtype

    def eye(N, M, k):
        return jnp.eye(N, M, k, dtype=dtype)

    return eye


@jax_funcify.register(BatchedDot)
def jax_funcify_BatchedDot(op):
    def batched_dot(a, b):
        if a.shape[0] != b.shape[0]:
            raise TypeError("Shapes must match in the 0-th dimension")
        if a.ndim == 2 or b.ndim == 2:
            return jnp.einsum("n...j,nj...->n...", a, b)
        return jnp.einsum("nij,njk->nik", a, b)

    return batched_dot


@jax_funcify.register(RandomVariable)
def jax_funcify_RandomVariable(op):
    name = op.name

    if not hasattr(jax.random, name):
        raise NotImplementedError(
            f"No JAX conversion for the given distribution: {name}"
        )

    def random_variable(rng, size, dtype, *args):
        prng = jax.random.PRNGKey(rng["state"]["key"][0])
        dtype = jnp.dtype(dtype)
        data = getattr(jax.random, name)(key=prng, shape=size)
        smpl_value = jnp.array(data, dtype=dtype)
        prng = jax.random.split(prng, num=1)[0]
        jax.ops.index_update(rng["state"]["key"], 0, prng[0])
        return (rng, smpl_value)

    return random_variable
