from functools import wraps

import numpy as np

from aesara import scalar as aes
from aesara.gpuarray.basic_ops import (
    GpuAllocEmpty,
    GpuFromHost,
    GpuReshape,
    HostFromGpu,
    host_from_gpu,
)
from aesara.gpuarray.elemwise import GpuDimShuffle, GpuElemwise
from aesara.gpuarray.type import GpuArrayType, get_context, move_to_gpu
from aesara.graph.basic import Constant
from aesara.graph.op import Op
from aesara.graph.opt import copy_stack_trace, local_optimizer
from aesara.tensor.basic import as_tensor, cast, get_scalar_constant_value, join
from aesara.tensor.elemwise import DimShuffle
from aesara.tensor.exceptions import NotScalarConstantError
from aesara.tensor.math import prod
from aesara.tensor.shape import shape_padright
from aesara.tensor.type import TensorType


# Define a few operations to use in optimizations,
# in order to avoid introducin new CPU Ops, or useless ones.
def safe_to_gpu(x, ctx_name):
    if isinstance(x.type, TensorType):
        return GpuFromHost(ctx_name)(x)
    else:
        return x


def safe_to_cpu(x):
    if isinstance(x.type, GpuArrayType):
        return x.transfer("cpu")
    else:
        return x


def grab_cpu_scalar(v, nd):
    """
    Get a scalar variable value from the tree at `v`.

    This function will dig through transfers and dimshuffles to get
    the constant value. If no such constant is found, it returns None.

    Parameters
    ----------
    v
        Aesara variable to extract the constant value from.
    nd : int
        Expected number of dimensions for the variable (for
        broadcasted constants).

    """
    if v.owner is not None:
        n = v.owner
        if (
            isinstance(n.op, (GpuDimShuffle, DimShuffle))
            and n.op.new_order == ("x",) * nd
        ):
            return grab_cpu_scalar(n.inputs[0], n.inputs[0].ndim)
        elif isinstance(n.op, (GpuFromHost, HostFromGpu)):
            return grab_cpu_scalar(n.inputs[0], nd)
        else:
            return None
    else:
        if isinstance(v, Constant) and v.broadcastable == (True,) * nd:
            return v.dimshuffle(())


def find_node(fgraph, v, cls, ignore_clients=False):
    """
    Find the node that has an op of of type `cls` in `v`.

    This digs through possibly redundant transfers to for the node
    that has the type `cls`. If `ignore_clients` is False (the
    default) it will only dig through nodes that have a single client
    to avoid duplicating computations.

    Parameters
    ----------
    v
        The variable to dig through
    cls : Op class
        The type of the node we are looking for
    ignore_clients : bool, optional
        Whether to ignore multiple clients or not.

    """
    if v.owner is not None and (ignore_clients or len(fgraph.clients[v]) == 1):
        if isinstance(v.owner.op, cls):
            return v.owner
        elif (
            isinstance(v.owner.op, GpuFromHost)
            and v.owner.inputs[0].owner is not None
            and (ignore_clients or len(fgraph.clients[v.owner.inputs[0]]) == 1)
            and isinstance(v.owner.inputs[0].owner.op, HostFromGpu)
        ):
            return find_node(fgraph, v.owner.inputs[0].owner.inputs[0], cls)
        else:
            return None


def is_equal(var, val):
    """
    Returns True if `var` is always equal to `val`.

    This will only return True if the variable will always be equal to
    the value.  If it might not be true in some cases then it returns False.

    Parameters
    ----------
    var
        Variable to compare
    val
        Python value

    """
    try:
        v = get_scalar_constant_value(var)
        return v == val
    except NotScalarConstantError:
        return False


def alpha_merge(cls, alpha_in, beta_in):
    """
    Decorator to merge multiplication by a scalar on the output.

    This will find a pattern of `aes * <yourop>(some, params, alpha,
    beta)` and update it so that the scalar multiplication happens as
    part of your op.

    The op needs to accept an alpha and a beta scalar which act this way::

       out = Op() * alpha + out_like * beta

    Where out_like is a buffer that has the same size as the output
    and gets added to the "real" output of the operation.  An example
    of an operation that respects this pattern is GEMM from blas.

    The decorated function must have this signature::

        maker(node, *inputs)

    The `node` argument you receive is the original apply node that
    contains your op.  You should use it to grab relevant properties
    for your op so that the new version performs the same computation.
    The `*inputs` parameters contains the new inputs for your op.  You
    MUST use those inputs instead of the ones on `node`.  Note that
    this function can be as simple as::

        def maker(node, *inputs):
            return node.op(*inputs)

    Parameters
    ----------
    cls : op class
        The class of the op you want to merge
    alpha_in : int
        The input index for the alpha scalar for your op (in node.inputs).
    beta_in : int
        The input index for the beta scalar for your op (in node.inputs).

    Returns
    -------
    local optimizer
        an unregistered local optimizer that has the same name as the
        decorated function.

    Notes
    -----
    This was factored out since the code to deal with intervening
    transfers and correctness in the presence of different values of
    alpha and beta scaling factors is not trivial.

    """

    def wrapper(maker):
        @local_optimizer([GpuElemwise])
        @wraps(maker)
        def opt(fgraph, node):
            if (
                isinstance(node.op, GpuElemwise)
                and node.op.scalar_op == aes.mul
                and node.nin == 2
            ):
                targ = find_node(fgraph, node.inputs[0], cls)
                if targ is None:
                    targ = find_node(fgraph, node.inputs[1], cls)
                    if targ is None:
                        return
                    lr = grab_cpu_scalar(node.inputs[0], nd=targ.outputs[0].ndim)
                else:
                    lr = grab_cpu_scalar(node.inputs[1], nd=targ.outputs[0].ndim)
                if lr is None or lr.dtype != targ.outputs[0].dtype:
                    return None
                inputs = list(targ.inputs)
                try:
                    c = get_scalar_constant_value(lr)
                    if c == 0:
                        inputs[alpha_in] = lr
                        inputs[beta_in] = lr
                    elif c == 1:
                        inputs[alpha_in] = targ.inputs[alpha_in]
                        inputs[beta_in] = targ.inputs[beta_in]
                    else:
                        inputs[alpha_in] = lr * targ.inputs[alpha_in]
                        inputs[beta_in] = lr * targ.inputs[beta_in]
                except NotScalarConstantError:
                    inputs[alpha_in] = lr * targ.inputs[alpha_in]
                    inputs[beta_in] = lr * targ.inputs[beta_in]
                new_out = maker(targ, *inputs)
                copy_stack_trace(node.outputs, new_out)
                return new_out

        return opt

    return wrapper


def output_merge(cls, alpha_in, beta_in, out_in):
    """
    Decorator to merge addition by a value on the output.

    This will find a pattern of `val * <yourop>(some, params, alpha,
    beta, out_like)` and update it so that the addtition happens as
    part of your op.

    The op needs to accept an alpha and a beta scalar which act this way::

       out = Op() * alpha + out_like * beta

    Where out_like is a buffer that has the same size as the output
    and gets added to the "real" output of the operation.  An example
    of an operation that respects this pattern is GEMM from blas.

    The decorated function must have this signature::

        maker(node, *inputs)

    The `node` argument you receive is the original apply node that
    contains your op.  You should use it to grab relevant properties
    for your op so that the new version performs the same computation.
    The `*inputs` parameters contains the new inputs for your op.  You
    MUST use those inputs instead of the ones on `node`.  Note that
    this function can be as simple as::

        def maker(node, *inputs):
            return node.op(*inputs)

    Parameters
    ----------
    cls : op class
        The class of the op you want to merge
    alpha_in : int
        The input index for the alpha scalar for your op (in node.inputs).
    beta_in : int
        The input index for the beta scalar for your op (in node.inputs).
    out_in : int
        The input index for the out_like input for your op (in node.inputs).

    Returns
    -------
    local optimizer
        an unregistered local optimizer that has the same name as the
        decorated function.

    Notes
    -----
    This was factored out since the code to deal with intervening
    transfers and correctness in the presence of different values of
    alpha and beta scaling factors is not trivial.

    This also correctly handles the case where the added value is
    broadcasted (by not performing the replacement).

    """

    def wrapper(maker):
        @local_optimizer([GpuElemwise])
        @wraps(maker)
        def opt(fgraph, node):
            if (
                isinstance(node.op, GpuElemwise)
                and node.op.scalar_op == aes.add
                and node.nin == 2
            ):
                targ = find_node(fgraph, node.inputs[0], cls)
                W = node.inputs[1]
                if targ is None:
                    targ = find_node(fgraph, node.inputs[1], cls)
                    W = node.inputs[0]
                if targ is None:
                    return None
                if W.dtype != targ.outputs[0].dtype:
                    return None
                if not is_equal(targ.inputs[beta_in], 0.0):
                    # other cases are too complex for now
                    return None
                if W.broadcastable != targ.inputs[out_in].broadcastable:
                    # Would need to explicitly tile the output to fill
                    # the full shape here.  Disable for now.
                    return None
                inputs = list(targ.inputs)
                inputs[out_in] = W
                dtype = inputs[beta_in].dtype
                one = aes.constant(np.asarray(1.0, dtype=dtype))
                inputs[beta_in] = one
                new_out = maker(targ, *inputs)
                copy_stack_trace(node.outputs, new_out)
                return new_out

        return opt

    return wrapper


def inplace_allocempty(op, idx):
    """
    Wrapper to make an inplace optimization that deals with AllocEmpty

    This will duplicate the alloc input if it has more than one client
    to allow the op to work on it inplace.

    The decorated function must have this signature::

        maker(node, inputs)

    The `node` argument you receive is the original apply node that
    contains your op.  You should use it to grab relevant properties
    for your op so that the new version performs the same computation.
    You should also switch the op to work inplace.  The `*inputs`
    parameters contains the new inputs for your op.  You MUST use
    those inputs instead of the ones on `node`.  Note that this
    function can be as simple as::

        def maker(node, inputs):
            return [node.op.__class__(inplace=True)(*inputs)]

    Parameters
    ----------
    op : op class
        The op class to look for to make inplace
    idx : int
        The index of the (possibly) AllocEmpty input (in node.inputs).

    Returns
    -------
    local optimizer
        an unregistered inplace local optimizer that has the same name
        as the decorated function.

    """

    def wrapper(maker):
        @local_optimizer([op], inplace=True)
        @wraps(maker)
        def opt(fgraph, node):
            if not isinstance(node.op, op) or node.op.inplace:
                return
            inputs = list(node.inputs)
            alloc = inputs[idx]
            if (
                alloc.owner
                and isinstance(alloc.owner.op, GpuAllocEmpty)
                and len(fgraph.clients[alloc]) > 1
            ):
                alloc_op = GpuAllocEmpty(
                    alloc.owner.op.dtype, alloc.owner.op.context_name
                )
                inputs[idx] = alloc_op(*alloc.owner.inputs)
            new_out = maker(node, inputs)
            copy_stack_trace(node.outputs, new_out)
            return new_out

        return opt

    return wrapper


def pad_dims(input, leftdims, rightdims):
    """Reshapes the input to a (leftdims + rightdims) tensor

    This helper function is used to convert pooling inputs with arbitrary
    non-pooling dimensions to the correct number of dimensions for the
    GPU pooling ops.

    This reduces or expands the number of dimensions of the input to
    exactly `leftdims`, by adding extra dimensions on the left or by
    combining some existing dimensions on the left of the input.

    Use `unpad_dims` to reshape back to the original dimensions.

    Examples
    --------
    Given input of shape (3, 5, 7), ``pad_dims(input, 2, 2)``
    adds a singleton dimension and reshapes to (1, 3, 5, 7).
    Given that output from pad_dims, ``unpad_dims(output, input, 2, 2)``
    reshapes back to (3, 5, 7).

    Given input of shape (3, 5, 7, 9), ``pad_dims(input, 2, 2)``
    does not reshape and returns output with shape (3, 5, 7, 9).

    Given input of shape (3, 5, 7, 9, 11), ``pad_dims(input, 2, 2)``
    combines the first two dimensions and reshapes to (15, 7, 9, 11).

    Given input of shape (3, 5, 7, 9), ``pad_dims(input, 2, 3)``
    adds a singleton dimension and reshapes to (1, 3, 5, 7, 9).
    """
    assert input.ndim >= rightdims

    if input.ndim == (leftdims + rightdims):
        return input

    # extract image dimensions
    img_shape = input.shape[-rightdims:]

    non_pool_ndim = input.ndim - rightdims
    if non_pool_ndim < leftdims:
        # too few dimensions, pad on the left
        dummy_dims = as_tensor([1] * (leftdims - non_pool_ndim))
        new_shape = join(0, dummy_dims, input.shape[:non_pool_ndim], img_shape)
    else:
        # too many dimensions, combine the leading dimensions
        batched_ndim = non_pool_ndim - leftdims + 1
        batch_size = prod(input.shape[:batched_ndim])
        # convert to a vector for join
        batch_size = shape_padright(batch_size, 1)
        new_shape = join(
            0, batch_size, input.shape[batched_ndim:non_pool_ndim], img_shape
        )

    # store in the required shape
    new_shape = cast(new_shape, "int64")
    input_ND = GpuReshape(leftdims + rightdims)(input, new_shape)
    return input_ND


def unpad_dims(output, input, leftdims, rightdims):
    """Reshapes the output after pad_dims.

    This reverts the padding by `pad_dims`.
    """
    if output.ndim == input.ndim:
        return output

    # restore the output to the original shape
    outshp = join(0, input.shape[:-rightdims], output.shape[-rightdims:])
    return GpuReshape(input.ndim)(output, outshp)


def op_lifter(OP, cuda_only=False):
    """
    OP(..., host_from_gpu(), ...) -> host_from_gpu(GpuOP(...))

    gpu_from_host(OP(inp0, ...)) -> GpuOP(inp0, ...)

    """

    def f(maker):
        def local_opt(fgraph, node):
            if isinstance(node.op, OP):
                # Either one of our inputs is on the gpu or
                # all of our clients are on the gpu
                replace = False
                # TODO: Maybe set context_name with infer_context_name()?
                context_name = None
                # We replace if any input is a host_from_gpu
                for i in node.inputs:
                    if i.owner and i.owner.op == host_from_gpu and move_to_gpu(i):
                        context_name = i.owner.inputs[0].type.context_name
                        replace = True
                        break

                if not replace:
                    # We replace if *all* clients are on the GPU
                    clients = [c for o in node.outputs for c in fgraph.clients[o]]
                    replace = len(clients) != 0
                    for c, idx in clients:
                        if c == "output" or not isinstance(c.op, GpuFromHost):
                            replace = False
                    # TODO: check that the clients want the same context?
                    if replace:
                        # All clients are GpuFromHost and we have at least one
                        context_name = clients[0][0].op.context_name

                # Check if we should replace
                if (
                    not replace
                    or (cuda_only and get_context(context_name).kind != b"cuda")
                    or any("complex" in getattr(i, "dtype", "") for i in node.inputs)
                ):
                    return False

                # tag the inputs with the context in case
                # the context was derived from the outputs
                for i in node.inputs:
                    i.tag.context_name = context_name

                new_op = maker(node.op, context_name, node.inputs, node.outputs)

                # This is needed as sometimes new_op inherits from OP.
                if new_op and new_op != node.op:
                    if isinstance(new_op, Op):
                        new_outputs = new_op(*node.inputs, return_list=True)
                        to_cpu_fn = safe_to_cpu
                    elif isinstance(new_op, (tuple, list)):
                        new_outputs = new_op
                        to_cpu_fn = safe_to_cpu
                    else:  # suppose it is a variable on the GPU
                        new_outputs = [new_op]

                        def to_cpu_fn(x):
                            return x.transfer("cpu")

                    # copy stack traces onto gpu outputs
                    # also copy the stack traces onto HostFromGpu outputs
                    on_cpu = []
                    for old_output, new_output in zip(node.outputs, new_outputs):
                        copy_stack_trace(old_output, new_output)
                        cpu = to_cpu_fn(new_output)
                        on_cpu.append(cpu)
                        copy_stack_trace(old_output, cpu)
                    return on_cpu
            return False

        local_opt.__name__ = maker.__name__
        return local_optimizer(OP)(local_opt)

    return f
