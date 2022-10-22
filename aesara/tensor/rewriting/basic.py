""" Tensor optimizations addressing the ops in basic.py."""

import logging
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

import aesara.scalar.basic as aes
from aesara import compile
from aesara.compile.ops import ViewOp
from aesara.graph.basic import Constant, Variable
from aesara.graph.rewriting.basic import (
    NodeRewriter,
    RemovalNodeRewriter,
    Rewriter,
    copy_stack_trace,
    in2out,
    node_rewriter,
)
from aesara.graph.rewriting.db import RewriteDatabase
from aesara.raise_op import Assert, CheckAndRaise, assert_op
from aesara.tensor.basic import (
    Alloc,
    AllocEmpty,
    Join,
    MakeVector,
    ScalarFromTensor,
    Split,
    TensorFromScalar,
    alloc,
    as_tensor_variable,
    cast,
    extract_constant,
    fill,
    get_scalar_constant_value,
    join,
    ones_like,
    switch,
    tensor_copy,
    zeros,
    zeros_like,
)
from aesara.tensor.elemwise import DimShuffle, Elemwise
from aesara.tensor.exceptions import NotScalarConstantError
from aesara.tensor.extra_ops import broadcast_shape, broadcast_to
from aesara.tensor.math import all as at_all
from aesara.tensor.math import eq
from aesara.tensor.shape import Shape_i
from aesara.tensor.sort import TopKOp
from aesara.tensor.type import DenseTensorType, TensorType
from aesara.tensor.var import TensorConstant
from aesara.utils import NoDuplicateOptWarningFilter


if TYPE_CHECKING:
    from aesara.tensor.rewriting.shape import ShapeFeature


_logger = logging.getLogger("aesara.tensor.rewriting.basic")
_logger.addFilter(NoDuplicateOptWarningFilter())


def encompasses_broadcastable(b1, b2):
    """

    Parameters
    ----------
    b1
        The broadcastable attribute of a tensor type.
    b2
        The broadcastable attribute of a tensor type.

    Returns
    -------
    bool
        True if the broadcastable patterns b1 and b2 are such that b2 is
        broadcasted to b1's shape and not the opposite.

    """
    if len(b1) < len(b2):
        return False
    b1 = b1[-len(b2) :]
    return not any(v1 and not v2 for v1, v2 in zip(b1, b2))


def merge_broadcastables(broadcastables):
    return [all(bcast) for bcast in zip(*broadcastables)]


def broadcast_like(value, template, fgraph, dtype=None):
    """
    Return a Variable with the same shape and dtype as the template,
    filled by broadcasting value through it. `value` will be cast as
    necessary.

    """
    value = as_tensor_variable(value)
    if value.type.is_super(template.type):
        return value
    if template not in fgraph.variables:
        raise NotImplementedError(
            "broadcast_like currently requires the "
            "template Variable to be in the fgraph already"
        )
    if dtype is None:
        dtype = template.dtype
    value = cast(value, dtype)
    if value.type.is_super(template.type):
        return value
    if hasattr(fgraph, "shape_feature"):
        new_shape = fgraph.shape_feature.shape_of[template]
    else:
        new_shape = template.shape
    rval = alloc(value, *new_shape)
    assert rval.type.dtype == dtype

    return rval


def register_useless(
    node_rewriter: Union[RewriteDatabase, NodeRewriter, str], *tags, **kwargs
):
    if isinstance(node_rewriter, str):

        def register(inner_rewriter: Union[RewriteDatabase, Rewriter]):
            return register_useless(inner_rewriter, node_rewriter, *tags, **kwargs)

        return register
    else:
        name = kwargs.pop("name", None) or node_rewriter.__name__

        compile.mode.local_useless.register(
            name, node_rewriter, "fast_run", *tags, position="last", **kwargs
        )
        return node_rewriter


def register_canonicalize(
    node_rewriter: Union[RewriteDatabase, NodeRewriter, str], *tags: str, **kwargs
):
    if isinstance(node_rewriter, str):

        def register(inner_rewriter: Union[RewriteDatabase, Rewriter]):
            return register_canonicalize(inner_rewriter, node_rewriter, *tags, **kwargs)

        return register
    else:
        name = kwargs.pop("name", None) or node_rewriter.__name__
        compile.optdb["canonicalize"].register(
            name, node_rewriter, "fast_run", "fast_compile", *tags, **kwargs
        )
        return node_rewriter


def register_stabilize(
    node_rewriter: Union[RewriteDatabase, NodeRewriter, str], *tags: str, **kwargs
):
    if isinstance(node_rewriter, str):

        def register(inner_rewriter: Union[RewriteDatabase, Rewriter]):
            return register_stabilize(inner_rewriter, node_rewriter, *tags, **kwargs)

        return register
    else:
        name = kwargs.pop("name", None) or node_rewriter.__name__
        compile.optdb["stabilize"].register(
            name, node_rewriter, "fast_run", *tags, **kwargs
        )
        return node_rewriter


def register_specialize(
    node_rewriter: Union[RewriteDatabase, NodeRewriter, str], *tags: str, **kwargs
):
    if isinstance(node_rewriter, str):

        def register(inner_rewriter: Union[RewriteDatabase, Rewriter]):
            return register_specialize(inner_rewriter, node_rewriter, *tags, **kwargs)

        return register
    else:
        name = kwargs.pop("name", None) or node_rewriter.__name__
        compile.optdb["specialize"].register(
            name, node_rewriter, "fast_run", *tags, **kwargs
        )
        return node_rewriter


def register_uncanonicalize(
    node_rewriter: Union[RewriteDatabase, NodeRewriter, str], *tags: str, **kwargs
):
    if isinstance(node_rewriter, str):

        def register(inner_rewriter: Union[RewriteDatabase, Rewriter]):
            return register_uncanonicalize(
                inner_rewriter, node_rewriter, *tags, **kwargs
            )

        return register
    else:
        name = (kwargs and kwargs.pop("name", None)) or node_rewriter.__name__
        compile.optdb["uncanonicalize"].register(
            name, node_rewriter, "fast_run", *tags, **kwargs
        )
        return node_rewriter


def register_specialize_device(
    node_rewriter: Union[RewriteDatabase, Rewriter, str], *tags: str, **kwargs
):
    if isinstance(node_rewriter, str):

        def register(inner_rewriter: Union[RewriteDatabase, Rewriter]):
            return register_specialize_device(
                inner_rewriter, node_rewriter, *tags, **kwargs
            )

        return register
    else:
        name = (kwargs and kwargs.pop("name", None)) or node_rewriter.__name__
        compile.optdb["specialize_device"].register(
            name, node_rewriter, "fast_run", *tags, **kwargs
        )
        return node_rewriter


@register_canonicalize
@register_specialize
@node_rewriter([TensorFromScalar])
def local_tensor_scalar_tensor(fgraph, node):
    """tensor_from_scalar(scalar_from_tensor(x)) -> x"""
    if isinstance(node.op, TensorFromScalar):
        s = node.inputs[0]
        if s.owner and isinstance(s.owner.op, ScalarFromTensor):
            t = s.owner.inputs[0]

            # We don't need to copy over any stack traces here
            return [t]


@register_canonicalize
@register_specialize
@node_rewriter([ScalarFromTensor])
def local_scalar_tensor_scalar(fgraph, node):
    """scalar_from_tensor(tensor_from_scalar(x)) -> x"""
    if isinstance(node.op, ScalarFromTensor):
        t = node.inputs[0]
        if t.owner and isinstance(t.owner.op, TensorFromScalar):
            s = t.owner.inputs[0]

            # We don't need to copy over any stack traces here
            return [s]


@register_specialize("local_alloc_elemwise")
@node_rewriter([Elemwise])
def local_elemwise_alloc(fgraph, node):
    r"""Remove unnecessary `Alloc`\s that occur as inputs of `Elemwise` `Op`\s.

    `Alloc`\s are effectively a type of `Elemwise` operation
    (e.g. ``Elemwise{second}(y, x)`` is the same as ``Alloc(x, *y.shape)``), so
    this rewrite uses that fact to reduce `Elemwise`\s on `Alloc`\s to
    `Elemwise`\s of the `Alloc`\s first/value input (i.e. the value it
    broadcasts).

    In other words, this rewrite causes `Elemwise` `Op`\s to "absorb" redundant
    `Alloc`\s.

    The rewrite essentially performs the following replacement:
    ``Elemwise{op}(..., Alloc(x, s), ..., y, ...) -> Elemwise{op}(..., x, ..., y, ...)``,
    when ``y.shape`` for some input ``y`` (or the combined shapes of the
    non-`Alloc`\s) is sufficient to maintain the same/correct output shape.

    In it's current form, it also explicitly accounts for `DimShuffle`\s of
    `Alloc`\s.  This is largely due to `local_alloc_sink_dimshuffle`, which
    introduces them as a canonicalization of `Alloc`'s with leading
    broadcastable dimensions.
    """
    # Rewrite is only applicable when there are at least two inputs
    if len(node.inputs) == 1:
        return False

    if len(node.outputs) > 1:
        return False

    def dimshuffled_alloc(i):
        return (
            isinstance(i.owner.op, DimShuffle)
            and i.owner.inputs[0].owner
            and isinstance(i.owner.inputs[0].owner.op, Alloc)
        )

    # At least one input must have an owner that is either a `Alloc` or a
    # `DimShuffle` with an owner that is a `Alloc` -- otherwise there is
    # nothing to optimize.
    alloc_idxs = [
        idx
        for idx, i in enumerate(node.inputs)
        if i.owner and (isinstance(i.owner.op, Alloc) or dimshuffled_alloc(i))
    ]
    if len(alloc_idxs) == 0:
        return False

    # Search for a non `Alloc` or `DimShuffle` of `Alloc` input that we can use as a
    # baseline for the dimensions.
    ref_var_idx = None
    for idx, i in enumerate(node.inputs):
        if i.type.broadcastable == node.outputs[0].type.broadcastable:
            # Prefer an input that is not an `Alloc` nor a `DimShuffle` of an
            # `Alloc`, so that all `Alloc`s can be rewritten.
            if idx not in alloc_idxs:
                ref_var_idx = idx
                break

    # If only `Alloc` and `DimShuffle` of `Alloc` exist, we pick the first suitable one
    if ref_var_idx is None:
        for idx, i in enumerate(node.inputs):
            # XXX: This broadcastable comparison doesn't work
            if (
                i.type.broadcastable == node.outputs[0].type.broadcastable
            ) and idx in alloc_idxs:
                ref_var_idx = idx
                break

    if not hasattr(fgraph, "shape_feature"):
        return False

    input_shapes = [
        tuple(fgraph.shape_feature.get_shape(i, j) for j in range(i.type.ndim))
        for i in node.inputs
    ]
    bcasted_shape = broadcast_shape(
        *input_shapes,
        arrays_are_shapes=True,
    )

    new_inputs = list(node.inputs)
    for idx in alloc_idxs:
        i = node.inputs[idx]

        # Remove `Alloc`
        if isinstance(i.owner.op, Alloc):
            new_alloc = broadcast_to(i.owner.inputs[0], bcasted_shape)

        # TODO FIXME: This shouldn't be handled here.
        # `DimShuffle`s should be lifted through `Alloc`s
        # by other, more general rewrites.
        # Remove `Alloc` in `DimShuffle`
        elif isinstance(i.owner.op, DimShuffle):
            old_alloc = i.owner.inputs[0]
            new_alloc = old_alloc.owner.inputs[0]
            # We need to keep the old `DimShuffle`. It could swap axes or
            # add dimensions anywhere.
            if new_alloc.ndim != old_alloc.ndim:
                # The `Alloc` can add dimensions to the value.
                # We replace those cases with a `DimShuffle` here.
                nb_dim_to_add = old_alloc.ndim - new_alloc.ndim
                new_alloc = new_alloc.dimshuffle(
                    ["x"] * nb_dim_to_add + list(range(new_alloc.ndim))
                )
            new_alloc = broadcast_to(i.owner.op(new_alloc), bcasted_shape)

        copy_stack_trace(i, new_alloc)
        new_inputs[idx] = new_alloc

    # If this assert is triggered, it means we are recreating an equivalent graph
    # which would result in cyclical merge rewrites.
    if all(new is old for new, old in zip(new_inputs, node.inputs)):
        return

    ret = node.op(*new_inputs, return_list=True)
    copy_stack_trace(node.outputs, ret)
    return ret


@register_canonicalize
@node_rewriter([Elemwise])
def local_fill_sink(fgraph, node):
    """
    f(fill(a, b), fill(c, d), e) -> fill(c, fill(a, f(b, d, e)))
    f need to be an elemwise that isn't a fill.
    """
    if not hasattr(node, "op") or not isinstance(node.op, Elemwise) or node.op == fill:
        return False
    models = []
    inputs = []
    for inp in node.inputs:
        if inp.owner and inp.owner.op == fill:
            models.append(inp.owner.inputs[0])
            inputs.append(inp.owner.inputs[1])
        else:
            inputs.append(inp)
    if not models:
        return False
    c = node.op(*inputs)
    for model in models:
        if (
            model.type.dtype != c.type.dtype
            or model.type.broadcastable != c.type.broadcastable
        ):
            c = fill(model, c)

    # The newly created node c doesn't has 'clients',
    # so this iteration is took place with node.outputs[0]
    replacements = {node.outputs[0]: c}
    for client, cl_idx in fgraph.clients[node.outputs[0]]:
        if (
            hasattr(client, "op")
            and isinstance(client.op, Elemwise)
            and client.op != fill
        ):
            client_inputs = client.inputs[:]
            client_inputs[cl_idx] = c
            new_client = client.op(*client_inputs)

            # Add clients to new_client
            fgraph.clients[new_client.owner.outputs[0]] = fgraph.clients[
                client.outputs[0]
            ]
            r = local_fill_sink.transform(fgraph, new_client.owner)
            if not r:
                continue
            replacements.update(r)
    return replacements


@register_specialize
@register_stabilize
@node_rewriter([fill])
def local_fill_to_alloc(fgraph, node):
    r"""Remove `fill`\s or replace them with `Alloc`\s.

    `Alloc`\s are preferable because they replace explicit tensor dependencies
    with their dependencies on those tensors' shapes, and sometimes those
    shapes can be computed without needing to compute the tensors themselves.

    XXX: This rewrite can produce inconsistent results, so do *not* consider
    making it a canonicalization until those inconsistencies are
    resolved/justified.
    """
    shape_ref, values_ref = node.inputs
    out_type = node.outputs[0].type

    if values_ref.type.broadcastable == out_type.broadcastable:
        # The assumption here is that `values_ref` already has the same shape
        # as `shape_ref`, so a `fill`/`Alloc` is unnecessary.

        # XXX FIXME TODO: The only way this can be determined is if one
        # absolutely knows that the shapes of `shape_ref` and `values_ref` are
        # equal.
        # This is an old rewrite, and it's only a
        # "specialization/stabilization", so we're going to leave it be for
        # now.
        return [values_ref]

    if shape_ref.type.broadcastable == out_type.broadcastable:
        # In this case, we assume that some broadcasting is needed (otherwise
        # the condition above would've been true), so we replace the `fill`
        # with an `Alloc`.
        o = broadcast_like(values_ref, shape_ref, fgraph, dtype=values_ref.dtype)
        copy_stack_trace(node.outputs[0], o)
        return [o]

    return


# Register this after stabilize at 1.5 to make sure stabilize don't
# get affected by less canonicalized graph due to alloc.
compile.optdb.register(
    "local_fill_to_alloc", in2out(local_fill_to_alloc), "fast_run", position=1.51
)
# Needed to clean some extra alloc added by local_fill_to_alloc
compile.optdb.register(
    "local_elemwise_alloc", in2out(local_elemwise_alloc), "fast_run", position=1.52
)


@register_canonicalize("fast_compile")
@register_useless
@node_rewriter([fill])
def local_useless_fill(fgraph, node):
    """fill(s,v) -> v

    This rewrite is only needed in FAST_COMPILE mode to make the code
    more readable. Normally, it is done by the `local_fill_to_alloc`
    rewrite.

    """
    r, v = node.inputs
    out_type = node.outputs[0].type

    if (
        v.type.dtype == out_type.dtype
        and v.type.broadcastable == out_type.broadcastable
    ):
        return [v]


@register_specialize
@register_stabilize
@register_canonicalize
@register_useless
@node_rewriter([Alloc])
def local_useless_alloc(fgraph, node):
    """
    If the input type is the same as the output type (dtype and broadcast)
    there is no change in the shape of the input. So this is just a simple copy
    of the input. This is not needed.
    """
    if not isinstance(node.op, Alloc):
        return False

    inp = node.inputs[0]
    output = node.outputs[0]

    if (
        inp.type.dtype == output.type.dtype
        and inp.type.broadcastable == output.type.broadcastable
    ):
        if inp.ndim == 0:
            return [inp]
        else:
            return [
                Assert("Shapes must be equal")(
                    inp, at_all(eq(inp.shape, node.inputs[1:]))
                )
            ]


@register_specialize
@register_stabilize
@register_canonicalize
@node_rewriter([Alloc])
def local_alloc_sink_dimshuffle(fgraph, node):
    r"""Convert broadcastable leading dimensions in an `Alloc` to `DimShuffle`\s."""
    op = node.op
    if not isinstance(op, Alloc):
        return False

    inp = node.inputs[0]
    output = node.outputs[0]

    # Check if alloc adds a broadcastable dimension with shape 1.
    output_shape = node.inputs[1:]
    num_dims_with_size_1_added_to_left = 0
    for i in range(len(output_shape) - inp.ndim):
        if extract_constant(output_shape[i], only_process_constants=True) == 1:
            num_dims_with_size_1_added_to_left += 1
        else:
            break

    new_output_shape = output_shape[num_dims_with_size_1_added_to_left:]
    if num_dims_with_size_1_added_to_left > 0 and len(new_output_shape) >= inp.ndim:
        if (
            output.broadcastable[num_dims_with_size_1_added_to_left:]
            == inp.broadcastable
        ):
            inner = inp
        else:
            inner = op(*([inp] + new_output_shape))
        dimshuffle_new_order = ["x"] * num_dims_with_size_1_added_to_left + list(
            range(len(new_output_shape))
        )
        return [DimShuffle(inner.type.broadcastable, dimshuffle_new_order)(inner)]


@node_rewriter([AllocEmpty])
def local_alloc_empty_to_zeros(fgraph, node):
    """This convert AllocEmpty to Alloc of 0.

    This helps one investigate NaNs in `NanGuardMode`.  Not registered by
    default. To activate it, use the setting
    ``optimizer_including == alloc_empty_to_zeros``.
    """
    if isinstance(node.op, AllocEmpty):
        return [zeros(node.inputs, dtype=node.outputs[0].dtype)]


compile.optdb.register(
    "local_alloc_empty_to_zeros",
    in2out(local_alloc_empty_to_zeros),
    # After move to gpu and merge2, before inplace.
    "alloc_empty_to_zeros",
    position=49.3,
)


@register_useless
@register_canonicalize("fast_compile")
@register_specialize
@node_rewriter([Elemwise])
def local_useless_elemwise(fgraph, node):
    """
        eq(x, x) -> 1
        neq(x, x) -> 0
        mul(x) -> x
        add(x) -> x
        identity(x) -> x
        and(x, 1) -> x  (if x.dtype == 'bool')
        and(x, 0) -> zeros_like(x)
        or(x, 0) -> x
        or(x, 1) -> ones_like(x)  (if x.dtype == 'bool')
        xor(x, x) -> zeros_like(x)

    TODO: This implementation is painfully redundant.

    """
    if isinstance(node.op, Elemwise):
        # We call zeros_like and one_like with opt=True to generate a
        # cleaner graph.
        dtype = node.outputs[0].dtype

        if node.op.scalar_op == aes.eq and len(node.inputs) == 2:
            if node.inputs[0] == node.inputs[1]:
                # it is the same var in the graph. That will always be true
                ret = ones_like(node.inputs[0], dtype=dtype, opt=True)

                # Copy stack trace from input to constant output
                copy_stack_trace(node.outputs[0], ret)
                return [ret]
        elif node.op.scalar_op == aes.neq and len(node.inputs) == 2:
            if node.inputs[0] == node.inputs[1]:
                # it is the same var in the graph. That will always be false
                ret = zeros_like(node.inputs[0], dtype=dtype, opt=True)

                # Copy stack trace from input to constant output
                copy_stack_trace(node.outputs[0], ret)
                return [ret]

        elif node.op.scalar_op == aes.mul and len(node.inputs) == 1:
            # No need to copy over any stack trace
            return [node.inputs[0]]

        elif node.op.scalar_op == aes.add and len(node.inputs) == 1:
            # No need to copy over any stack trace
            return [node.inputs[0]]
        elif node.op.scalar_op == aes.identity and len(node.inputs) == 1:
            return [node.inputs[0]]

        elif isinstance(node.op.scalar_op, aes.AND) and len(node.inputs) == 2:

            if isinstance(node.inputs[0], TensorConstant):
                const_val = extract_constant(
                    node.inputs[0], only_process_constants=True
                )
                if not isinstance(const_val, Variable):
                    if const_val == 0:
                        return [zeros_like(node.inputs[1], dtype=dtype, opt=True)]
                    elif node.outputs[0].dtype == "bool":
                        # If the output is not Boolean, it is the bitwise AND,
                        # and this rewrite would be wrong
                        return [node.inputs[1].astype(node.outputs[0].dtype)]

            if isinstance(node.inputs[1], TensorConstant):
                const_val = extract_constant(
                    node.inputs[1], only_process_constants=True
                )
                if not isinstance(const_val, Variable):
                    if const_val == 0:
                        return [zeros_like(node.inputs[0], dtype=dtype, opt=True)]
                    elif node.outputs[0].dtype == "bool":
                        # If the output is not Boolean, it is the bitwise AND,
                        # and this rewrite would be wrong
                        return [node.inputs[0].astype(node.outputs[0].dtype)]

        elif isinstance(node.op.scalar_op, aes.OR) and len(node.inputs) == 2:

            if isinstance(node.inputs[0], TensorConstant):
                const_val = extract_constant(
                    node.inputs[0], only_process_constants=True
                )
                if not isinstance(const_val, Variable):
                    if const_val == 0:
                        return [node.inputs[1].astype(node.outputs[0].dtype)]
                    elif node.outputs[0].dtype == "bool":
                        # If the output is not Boolean, it is the bitwise OR,
                        # and this rewrite would be wrong
                        return [ones_like(node.inputs[1], dtype=dtype, opt=True)]

            if isinstance(node.inputs[1], TensorConstant):
                const_val = extract_constant(
                    node.inputs[1], only_process_constants=True
                )
                if not isinstance(const_val, Variable):
                    if const_val == 0:
                        return [node.inputs[0].astype(node.outputs[0].dtype)]
                    elif node.outputs[0].dtype == "bool":
                        # If the output is not Boolean, it is the bitwise OR,
                        # and this rewrite would be wrong
                        return [ones_like(node.inputs[0], dtype=dtype, opt=True)]

        elif isinstance(node.op.scalar_op, aes.XOR) and len(node.inputs) == 2:
            if node.inputs[0] is node.inputs[1]:
                return [zeros_like(node.inputs[0], dtype=dtype, opt=True)]


@register_specialize
@node_rewriter([Elemwise])
def local_alloc_unary(fgraph, node):
    """unary(alloc(x, shp)) -> alloc(unary(x), shp)"""
    if isinstance(node.op, Elemwise) and len(node.inputs) == 1:
        a = node.inputs[0]
        if a.owner and isinstance(a.owner.op, Alloc):
            x = a.owner.inputs[0]
            shp = a.owner.inputs[1:]
            v = node.op(x)
            # at.alloc does not preserve the stacktrace of v,
            # so we need to copy it over from x.
            copy_stack_trace(node.outputs[0], v)
            ret = alloc(cast(v, node.outputs[0].dtype), *shp)

            # at.cast does not preserve the stacktrace of x,
            # so we need to copy it over to the output.
            copy_stack_trace([node.outputs[0], a], ret)
            return [ret]


@register_canonicalize
@register_specialize
@node_rewriter([Elemwise])
def local_cast_cast(fgraph, node):
    """cast(cast(x, dtype1), dtype2)

    when those contrain:
    dtype1 == dtype2
    OR the base dtype is the same (int, uint, float, complex)
          and the first cast cause an upcast.

    """
    if not isinstance(node.op, Elemwise) or not isinstance(node.op.scalar_op, aes.Cast):
        return
    x = node.inputs[0]
    if (
        not x.owner
        or not isinstance(x.owner.op, Elemwise)
        or not isinstance(x.owner.op.scalar_op, aes.Cast)
    ):
        return

    type1 = x.owner.op.scalar_op.o_type
    type2 = node.op.scalar_op.o_type
    base = x.owner.inputs[0]

    if type1 == type2:
        # We don't need to copy over any stack traces here
        return [x]

    if is_an_upcast(base.dtype, type1.dtype):
        # Checking for further redundancy. Eg: int8 -> int32 -> int8
        if type2.dtype == base.dtype:
            return x.owner.inputs
        else:
            # Apply the second cast only
            v = node.op(base)
            # Copy stack trace from the output of the original cast
            copy_stack_trace(node.outputs[0], v)
            return [v]


def is_an_upcast(type1, type2):
    """Given two data types (as strings), check if converting to
    type2 from type1 constitutes an upcast.
    Differs from aesara.scalar.upcast

    """
    category = {
        # The first number in the pair is the dtype (bool, uint, int, float,
        # complex). Conversion from higher to lower is never an upcast.
        # The second number roughly indicates the precision. Again, conversion
        # from higher to lower is never an upcast.
        "bool": (0, 0),
        "uint8": (1, 1),
        "uint16": (1, 2),
        "uint32": (1, 3),
        "uint64": (1, 4),
        "int8": (2, 1),
        "int16": (2, 2),
        "int32": (2, 3),
        "int64": (2, 4),
        "float16": (3, 1.5),
        "float32": (3, 2.5),
        "float64": (3, 3.5),
        "complex64": (4, 3),
        "complex128": (4, 4),
    }

    cat1 = category[type1]
    cat2 = category[type2]

    if cat2[0] >= cat1[0] and cat2[1] > cat1[1]:
        return True
    else:
        return False


@register_useless
@register_specialize
@node_rewriter(None)
def local_remove_useless_assert(fgraph, node):
    if not isinstance(node.op, CheckAndRaise):
        return False

    new_conds = []
    n_conds = len(node.inputs[1:])
    for c in node.inputs[1:]:
        try:
            const = get_scalar_constant_value(c)

            if 0 != const.ndim or const == 0:
                # Should we raise an error here? How to be sure it
                # is not caught?
                new_conds.append(c)
        except NotScalarConstantError:
            new_conds.append(c)

    if len(new_conds) == 0:
        return [node.inputs[0]]

    if len(new_conds) < n_conds:
        new_var = node.op(*(node.inputs[:1] + new_conds))
        copy_stack_trace(node.outputs[0], new_var)
        return [new_var]


@node_rewriter([Assert])
def local_remove_all_assert(fgraph, node):
    r"""A rewrite that removes all `Assert`\s from a graph.

    Notes
    -----
    See the :ref:`unsafe` section.

    """
    if not isinstance(node.op, Assert):
        return

    return [node.inputs[0]]


compile.optdb["canonicalize"].register(
    "local_remove_all_assert",
    local_remove_all_assert,
    "unsafe",
    use_db_name_as_tag=False,
)
compile.optdb["stabilize"].register(
    "local_remove_all_assert",
    local_remove_all_assert,
    "unsafe",
    use_db_name_as_tag=False,
)
compile.optdb["specialize"].register(
    "local_remove_all_assert",
    local_remove_all_assert,
    "unsafe",
    use_db_name_as_tag=False,
)
compile.optdb["useless"].register(
    "local_remove_all_assert",
    local_remove_all_assert,
    "unsafe",
    use_db_name_as_tag=False,
)


@register_specialize
@register_canonicalize
@register_useless
@node_rewriter([Join])
def local_join_1(fgraph, node):
    """Join(i, x) => x

    Remove Join() when only one element is joined.

    """
    if not isinstance(node.op, Join):
        return
    tensors = node.inputs[1:]
    if len(tensors) == 1:
        # We don't need to copy over any stacktrace here, because the
        # input variable should already have its own stacktrace.
        return [tensors[0]]


# TODO: merge in local_useless_join
@register_useless
@register_specialize
@register_canonicalize
@node_rewriter([Join])
def local_join_empty(fgraph, node):
    """Join(i, x, y, empty) => Join(i, x, y)

    Remove empty inputs to joins. The empty inputs can be anywhere.

    """
    if not isinstance(node.op, Join):
        return
    new_inputs = []
    try:
        join_idx = get_scalar_constant_value(
            node.inputs[0], only_process_constants=True
        )
    except NotScalarConstantError:
        return
    for idx in range(1, len(node.inputs)):
        inp = node.inputs[idx]
        # We can not use size == 0,, as this can change shape from 3,0
        # to 2,0.  This trigger DebugMode error. This happen with
        # stack(...,[]) as this add a dimshuffle on [], that add a
        # dimensions with shape 1.
        if isinstance(inp, Constant) and inp.data.shape[join_idx] == 0:
            continue
        new_inputs.append(inp)
    if len(new_inputs) < len(node.inputs) - 1:
        if len(new_inputs) == 0:
            # at.join do not work in that case.
            # constant folding will take care of this case.
            return
        ret = join(node.inputs[0], *new_inputs)
        o = node.outputs[0]
        if ret.dtype != o.dtype:
            # Join can upcast some inputs
            return

        # Copy over stacktrace from previous output (after join op)
        # to new output, because an error in the new op must be caused
        # by an error in the old join op.
        copy_stack_trace(node.outputs, ret)

        return [ret]


@register_specialize
@register_canonicalize
@register_useless
@node_rewriter([Join])
def local_join_make_vector(fgraph, node):
    r"""Merge `MakeVector` inputs within a `Join`.

    For example:

        Join(0, make_vector1, make_vector2, ...) => Join(0, make_vector12, ...)

    This, in combination with the `local_join_1` rewrite, can make `Join`\s
    completely disappear.
    """
    if not isinstance(node.op, Join) or node.outputs[0].ndim != 1:
        return
    new_inputs = [node.inputs[1]]
    for idx in range(2, len(node.inputs)):
        inp = node.inputs[idx]
        if (
            inp.owner
            and isinstance(inp.owner.op, MakeVector)
            and new_inputs[-1].owner
            and isinstance(new_inputs[-1].owner.op, MakeVector)
            and
            # MakeVector have a dtype parameter
            inp.owner.op == new_inputs[-1].owner.op
        ):
            inps = new_inputs[-1].owner.inputs + inp.owner.inputs
            new_inputs[-1] = inp.owner.op(*inps)

            # Copy over stacktrace from previous output (after join op)
            # to new intermediate output, because an error in the intermediate
            # op must be caused by an error in the old join op.
            copy_stack_trace(node.outputs, new_inputs[-1])
        else:
            new_inputs.append(inp)
    if len(new_inputs) < len(node.inputs) - 1:
        ret = join(node.inputs[0], *new_inputs)

        # Copy over stacktrace from previous output (after join op)
        # to new output, because an error in the new op must be caused
        # by an error in the old join op.
        copy_stack_trace(node.outputs, ret)
        return [ret]


@register_useless("local_remove_switch_const_cond")
@register_canonicalize("fast_compile", "local_remove_switch_const_cond")
@register_specialize
@node_rewriter([Elemwise])
def local_useless_switch(fgraph, node):
    """
    This rewrite makes the following changes in a graph:

        at.switch(cond, left, right) ->
                if cond is constant and cond == 0: right
                if cond is constant and cond != 0: left
                if left is right -> left

    and

        at.switch(le(shape_i{id}(X), 0), 0, shape_i{id}(X)) -> shape_i{id}(X)

    """
    if not isinstance(node.op.scalar_op, aes.Switch):
        return False

    shape_feature: Optional["ShapeFeature"] = getattr(fgraph, "shape_feature", None)

    if shape_feature is None:
        return False

    left = node.inputs[1]
    right = node.inputs[2]
    cond_var = node.inputs[0]
    cond = extract_constant(cond_var, only_process_constants=True)

    if (isinstance(cond, np.ndarray) and cond.ndim == 0) or isinstance(
        cond, (np.number, np.bool_)
    ):
        if cond == 0:
            correct_out = right
        else:
            correct_out = left

        if correct_out.dtype != node.outputs[0].dtype:
            out = cast(correct_out, node.outputs[0].dtype)
        else:
            out = correct_out

        input_shapes = [
            tuple(shape_feature.get_shape(inp, i) for i in range(inp.type.ndim))
            for inp in node.inputs
        ]

        out_shape = broadcast_shape(*input_shapes, arrays_are_shapes=True)

        out = alloc(out, *out_shape)

        # Copy over stacktrace from selected output to new output
        copy_stack_trace(node.outputs + correct_out, out)
        return [out]

    # if left is right -> left
    if left == right:
        # Note: No need to copy over stacktrace, because the input node
        # already has its own stacktrace
        if cond.type.is_super(left.type):
            return [left]

        ret = fill(cond, left)

        # Copy over stacktrace from switch output and correct branch
        copy_stack_trace(node.outputs + left, ret)
        return [ret]

    # This case happens with scan.
    # Elemwise{switch}(le(shape_i{id}(X), 0), 0, shape_i{id}(X)) -> shape_i{id}(X)
    if (
        cond_var.owner
        and isinstance(cond_var.owner.op, Elemwise)
        and isinstance(cond_var.owner.op.scalar_op, aes.LE)
        and cond_var.owner.inputs[0].owner
        and isinstance(cond_var.owner.inputs[0].owner.op, Shape_i)
        and extract_constant(cond_var.owner.inputs[1], only_process_constants=True) == 0
        and extract_constant(left, only_process_constants=True) == 0
        and right == cond_var.owner.inputs[0]
    ):
        assert node.outputs[0].type.is_super(right.type)
        # No need to copy over stacktrace, because the right input node
        # already has its own stacktrace
        return [right]


@register_canonicalize
@node_rewriter([Elemwise])
def local_merge_switch_same_cond(fgraph, node):
    """
    Merge add/sub/mul/div/minimum/maximum/... of switches sharing the same
    condition, to enable further simplification of their branches
    Example: switch(c, a, b) + switch(c, x, y) -> switch(c, a+x, b+y)
    """
    # node must be binary elemwise or add or mul
    if not isinstance(node.op, Elemwise) or not isinstance(
        node.op.scalar_op, (aes.BinaryScalarOp, aes.Add, aes.Mul)
    ):
        return
    # all inputs must be switch
    if not all(
        s.owner
        and isinstance(s.owner.op, Elemwise)
        and isinstance(s.owner.op.scalar_op, aes.Switch)
        for s in node.inputs
    ):
        return
    # all switch conditions must be the same
    cond = node.inputs[0].owner.inputs[0]
    if not all(s.owner.inputs[0] is cond for s in node.inputs[1:]):
        return
    # pull out switch
    return [
        switch(
            cond,
            node.op(*[s.owner.inputs[1] for s in node.inputs]),
            node.op(*[s.owner.inputs[2] for s in node.inputs]),
        )
    ]


@register_useless
@register_canonicalize
@register_specialize
@node_rewriter([Split])
def local_useless_split(fgraph, node):
    """Split{n_splits=1}(x, y) -> x

    Remove Split with only 1 split.

    """
    if isinstance(node.op, Split):
        if node.op.len_splits == 1:
            x, axis, splits = node.inputs
            out = assert_op(x, eq(splits.shape[0], 1))
            # Copy over stacktrace from previous output node.
            copy_stack_trace(node.outputs, out)
            out2 = assert_op(out, eq(x.shape[axis], splits[0]))
            # Copy over stacktrace from previous output node.
            copy_stack_trace(out, out2)

            return [out2]


@node_rewriter(None)
def constant_folding(fgraph, node):

    if not node.op.do_constant_folding(fgraph, node):
        return False

    if not all(isinstance(inp, Constant) for inp in node.inputs):
        return False

    storage_map = {i: [i.data] for i in node.inputs}
    compute_map = {i: [True] for i in node.inputs}
    for o in node.outputs:
        storage_map[o] = [None]
        compute_map[o] = [False]

    thunk = node.op.make_thunk(node, storage_map, compute_map, no_recycling=[])
    required = thunk()

    # A node whose inputs are all provided should always return successfully
    assert not required

    rval = []
    for output in node.outputs:
        data = storage_map[output][0]
        assert compute_map[output][0], (output, data)

        # TODO: `Type` itself should provide an interface for constructing
        # instances appropriate for a given constant.
        # TODO: Add handling for sparse types.
        if isinstance(output.type, DenseTensorType):
            output_type = TensorType(
                output.type.dtype,
                shape=data.shape,
                name=output.type.name,
            )
        else:
            output_type = output.type

        v = output_type.make_constant(data)

        # We need to "narrow" types when we have additional information,
        # and not "broaden" them.  This is a case in which types are
        # unnecessarily "broadened"
        # assert not hasattr(output.type, "broadcastable") or output.type.broadcastable == tuple(s == 1 for s in data.shape)

        copy_stack_trace(output, v)

        rval.append(v)

    return rval


topo_constant_folding = in2out(
    constant_folding, ignore_newtrees=True, name="topo_constant_folding"
)
register_canonicalize(topo_constant_folding, "fast_compile", final_rewriter=True)
register_uncanonicalize(topo_constant_folding, "fast_compile", final_rewriter=True)
register_stabilize(topo_constant_folding, "fast_compile", final_rewriter=True)
register_specialize(topo_constant_folding, "fast_compile", final_rewriter=True)


@register_canonicalize("fast_compile")
@register_useless("fast_compile")
@node_rewriter(None)
def local_view_op(fgraph, node):
    if isinstance(node.op, ViewOp):
        return node.inputs


@register_useless
@register_canonicalize
@register_stabilize
@register_specialize
@node_rewriter([Alloc])
def local_merge_alloc(fgraph, node):
    """
    This rewriter takes care of the following cases:

        Alloc(Alloc(m, x, 1, 1, 1), x, y, z, w) -> Alloc(m, x, y, z, w)
        Alloc(Alloc(m, y, 1, 1), x, y, z, w) -> Alloc(m, x, y, z, w)
        Alloc(Alloc(m, y1, 1, 1), x, y2, z, w) -> Alloc(m, x, assert(y1, y1==y2), z, w)

    """
    if not isinstance(node.op, Alloc):
        return False
    if not node.inputs[0].owner or not isinstance(node.inputs[0].owner.op, Alloc):
        return False
    inputs_outer = node.inputs
    inputs_inner = node.inputs[0].owner.inputs
    dims_outer = inputs_outer[1:]
    dims_inner = inputs_inner[1:]
    dims_outer_rev = dims_outer[::-1]
    dims_inner_rev = dims_inner[::-1]
    # check if the pattern of broadcasting is matched, in the reversed ordering.
    # The reverse ordering is needed when an Alloc add an implicit new
    # broadcasted dimensions to its inputs[0]. Eg:
    # Alloc(Alloc(m, y, 1, 1), x, y, z, w) -> Alloc(m, x, y, z, w)
    i = 0
    for dim_inner, dim_outer in zip(dims_inner_rev, dims_outer_rev):
        if dim_inner != dim_outer:
            if isinstance(dim_inner, Constant) and dim_inner.data == 1:
                pass
            else:
                dims_outer[-1 - i] = Assert(
                    "You have a shape error in your graph. To see a better"
                    " error message and a stack trace of where in your code"
                    " the error is created, use the Aesara flags"
                    " optimizer=None or optimizer=fast_compile."
                )(dim_outer, eq(dim_outer, dim_inner))
        i += 1
    return [alloc(inputs_inner[0], *dims_outer)]


@register_useless("fast_compile")
@node_rewriter([TopKOp])
def local_useless_topk(fgraph, node):
    """Remove unused `TopKOp` outputs."""
    op = node.op
    if not isinstance(op, TopKOp):
        return
    if not (op.return_values and op.return_indices):
        return False

    x, k = node.inputs
    ret_val = bool(fgraph.clients[node.outputs[0]])
    ret_idx = bool(fgraph.clients[node.outputs[1]])

    if not (ret_val ^ ret_idx):
        # both true -> nothing to remove
        # both false -> let pruner handle
        return False

    old_output = node.outputs[ret_idx]
    new_output = TopKOp(
        axis=op.axis,
        sorted=op.sorted,
        idx_dtype=op.idx_dtype,
        return_values=ret_val,
        return_indices=ret_idx,
    )(x, k)
    copy_stack_trace(node.outputs[0], new_output)
    return {old_output: new_output}


def import_ShapeFeature():
    from aesara.tensor.rewriting.shape import ShapeFeature

    return ShapeFeature


DEPRECATED_NAMES = {
    "ShapeFeature": (
        "`ShapeFeature` is now located in `aesara.tensor.rewriting.shape`.",
        import_ShapeFeature,
    ),
}


def __getattr__(name):
    """Intercept module-level attribute access of deprecated symbols.

    Adapted from https://stackoverflow.com/a/55139609/3006474.

    """
    from warnings import warn

    res = DEPRECATED_NAMES.get(name)
    if res:
        msg, fn = res
        warn(msg, DeprecationWarning, stacklevel=2)
        return fn()

    raise AttributeError(f"module {__name__} has no attribute {name}")


register_canonicalize(RemovalNodeRewriter(tensor_copy), name="remove_tensor_copy")
