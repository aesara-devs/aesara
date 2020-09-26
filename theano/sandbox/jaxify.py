import theano

import jax
import jax.numpy as jnp

from warnings import warn
from functools import partial, update_wrapper, reduce
from collections.abc import Sequence

from functools import singledispatch as dispatch

from theano.gof import FunctionGraph

from theano.ifelse import IfElse
from theano.tensor.subtensor import (
    get_idx_list,
    Subtensor,
    IncSubtensor,
    # This is essentially `np.take`
    AdvancedSubtensor1,
    AdvancedIncSubtensor1,
    # Boolean mask indexing and setting
    BaseAdvancedSubtensor,
    BaseAdvancedIncSubtensor,
)
from theano.scan_module.scan_op import Scan
from theano.scan_module.scan_utils import scan_args as ScanArgs
from theano.tensor.basic import (
    TensorFromScalar,
    ScalarFromTensor,
    AllocEmpty,
    Alloc,
    Reshape,
    Join,
)
from theano.scalar.basic import (
    ScalarOp,
    Composite,
    Cast,
    Clip,
)
from theano.tensor.elemwise import Elemwise, CAReduce, DimShuffle
from theano.compile.ops import (
    DeepCopyOp,
    Shape,
    Shape_i,
    SpecifyShape,
    Rebroadcast,
    ViewOp,
)
from theano.tensor.opt import MakeVector


# XXX: Enabling this will break some shape-based functionality, and severely
# limit the types of graphs that can be converted.
# See https://github.com/google/jax/blob/4d556837cc9003492f674c012689efc3d68fdf5f/design_notes/omnistaging.md
jax.config.disable_omnistaging()
jax.config.update("jax_enable_x64", True)

subtensor_ops = (Subtensor, AdvancedSubtensor1, BaseAdvancedSubtensor)
incsubtensor_ops = (IncSubtensor, AdvancedIncSubtensor1, BaseAdvancedIncSubtensor)


def compose_jax_funcs(out_node, fgraph_inputs, memo=None):
    """Compose JAX implementations of node operations.

    Parameters
    ----------
    out_node: Node
        The output node.
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

    input_funcs = []
    for i in out_node.inputs:
        if i in fgraph_inputs:
            idx = fgraph_inputs.index(i)

            def jax_inputs_func(*inputs, i_dtype=i.dtype, idx=idx):
                return jnp.array(inputs[idx], dtype=jnp.dtype(i_dtype))

            input_f = jax_inputs_func

        elif i.owner is None:

            def jax_data_func(*inputs, i_dtype=i.dtype, i_data=i.data):
                return jnp.array(i_data, dtype=jnp.dtype(i_dtype))

            input_f = jax_data_func
        else:
            input_f = compose_jax_funcs(i.owner, fgraph_inputs, memo)

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


@dispatch
def jax_funcify(op):
    """Create a JAX "perform" function for a Theano `Variable` and its `Op`."""
    raise NotImplementedError("No JAX conversion for the given `Op`: {}".format(op))


@jax_funcify.register(ScalarOp)
def jax_funcify_ScalarOp(op):
    func_name = op.nfunc_spec[0]

    if "." in func_name:
        jnp_func = reduce(getattr, [jax] + func_name.split("."))
    else:
        jnp_func = getattr(jnp, func_name)

    if hasattr(op, "nfunc_variadic"):
        # These are special cases that handle invalid arities due to the broken
        # Theano `Op` type contract (e.g. binary `Op`s that also function as
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
    return partial(op.impl, None)


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


def jnp_safe_copy(x):
    try:
        res = jnp.copy(x)
    except NotImplementedError:
        warn("`jnp.copy` is not implemented yet. " "Using the object's `copy` method.")
        if hasattr(x, "copy"):
            res = jnp.array(x.copy())
        else:
            warn("Object has no `copy` method: {}".format(x))
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
        assert x.ndim == shape.size
        assert jnp.all(x.shape == shape), ("got shape", x.shape, "expected", shape)
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
    jax_tt_inner_func = jax_funcify(inner_fg)

    def scan(*outer_inputs):
        scan_args = ScanArgs(
            outer_inputs, [None] * op.n_outs, op.inputs, op.outputs, op.info
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

        n_non_seqs = len(scan_args.outer_in_non_seqs)

        # TODO: sit_sots
        mit_sot_in_slices = []
        for tap, seq in zip(scan_args.mit_sot_in_slices, scan_args.outer_in_mit_sot):
            neg_taps = [abs(t) for t in tap if t < 0]
            pos_taps = [abs(t) for t in tap if t > 0]
            max_neg = max(neg_taps) if neg_taps else 0
            max_pos = max(pos_taps) if pos_taps else 0
            init_slice = seq[: max_neg + max_pos]
            mit_sot_in_slices.append(init_slice)

        init_carry = [mit_sot_in_slices, scan_args.outer_in_non_seqs]

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
            inner_scan_inputs = [
                inner_in_seqs,
                inner_in_mit_mot,
                inner_in_mit_sot,
                inner_in_sit_sot,
                inner_in_non_seqs,
            ]

            raise NotImplementedError()
            return inner_scan_inputs

        def inner_scan_outs_to_jax_outs(
            op,
            old_carry,
            inner_scan_outs,
        ):
            # `inner_scan_outs` is a list with the following
            # composite form:
            # outer_out_mit_mot
            # + outer_out_mit_sot
            # + outer_out_sit_sot
            # + outer_out_nit_sot
            # + outer_out_shared
            # + cond
            (
                outer_out_mit_mot,
                outer_out_mit_sot,
                outer_out_sit_sot,
                outer_out_nit_sot,
                outer_out_shared,
                cond,
            ) = inner_scan_outs
            outer_out_non_seqs = old_carry[:-n_non_seqs]

            # This should contain all inner-output taps, non_seqs, and shared
            # terms
            carry = [
                outer_out_mit_mot,
                outer_out_mit_sot,
                outer_out_sit_sot,
                outer_out_shared,
                outer_out_non_seqs,
            ]
            # This should contain all inner-outputs that produce
            # outer-outputs
            y = []

            raise NotImplementedError()
            return (carry, y)

        def jax_inner_func(carry, x):
            inner_args = jax_args_to_inner_scan(op, carry, x)
            inner_scan_outs = jax_tt_inner_func(*inner_args)
            new_carry, y = inner_scan_outs_to_jax_outs(op, inner_scan_outs)
            return new_carry, y

        return jax.lax.scan(jax_inner_func, init_carry, seqs, length=n_steps)

    return scan


@jax_funcify.register(IfElse)
def jax_funcify_IfElse(op):
    def ifelse(cond, *args):
        if cond:
            return args[: op.n_outs]
        else:
            return args[op.n_outs :]

    return ifelse


def convert_indices(indices, entry):
    if indices and isinstance(entry, theano.gof.Type):
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

        # breakpoint()

        if len(cdata) == 1:
            cdata = cdata[0]

        return x.__getitem__(cdata)
        # return x.take(ilists, axis=0)

    return subtensor


_ = [jax_funcify.register(op, jax_funcify_Subtensor) for op in subtensor_ops]


def jax_funcify_IncSubtensor(op):

    if getattr(op, "set_instead_of_inc", False):
        jax_fn = jax.ops.index_update
    else:
        jax_fn = jax.ops.index_add

    def incsubtensor(x, y, *ilist, jax_fn=jax_fn):
        _ilist = list(ilist)
        cdata = tuple(convert_indices(_ilist, idx) for idx in op.idx_list)
        if len(cdata) == 1:
            cdata = cdata[0]

        return jax_fn(x, cdata, y)

    return incsubtensor


_ = [jax_funcify.register(op, jax_funcify_IncSubtensor) for op in incsubtensor_ops]


@jax_funcify.register(FunctionGraph)
def jax_funcify_FunctionGraph(fgraph):

    out_nodes = [r.owner for r in fgraph.outputs if r.owner is not None]
    jax_funcs = [compose_jax_funcs(o, fgraph.inputs) for o in out_nodes]

    return jax_funcs


@jax_funcify.register(CAReduce)
def jax_funcify_CAReduce(op):
    def careduce(x):
        axis = op.axis

        if axis is None:
            axis = list(range(x.ndim))

        to_reduce = reversed(sorted(axis))

        if hasattr(op, "acc_dtype") and op.acc_dtype is not None:
            acc_dtype = op.acc_dtype
        else:
            acc_dtype = x.dtype.type

        if to_reduce:
            if getattr(op.scalar_op, "name", None):
                jax_op = getattr(jax.lax, op.scalar_op.name)
            elif getattr(op.scalar_op, "nfunc_spec", None):
                # In this case, we need to use the `jax.lax` function (if there
                # is one), and not the `jnp` version.
                jax_op = getattr(jax.lax, op.scalar_op.nfunc_spec[0])

            init_value = jnp.array(op.scalar_op.identity, dtype=acc_dtype)
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
                raise IndexError("Join axis %d out of bounds [0, %d)" % (axis, ndim))

            return jnp.concatenate(tensors, axis=axis)

    return join
