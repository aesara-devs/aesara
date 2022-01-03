import aesara
from aesara.compile import optdb
from aesara.gpuarray.basic_ops import (
    GpuAllocEmpty,
    GpuArrayType,
    as_gpuarray_variable,
    gpu_contiguous,
    infer_context_name,
)
from aesara.gpuarray.dnn import (
    GpuDnnBatchNorm,
    GpuDnnBatchNormInference,
    GpuDnnConv,
    GpuDnnConvDesc,
    GpuDnnConvGradI,
    GpuDnnConvGradW,
    GpuDnnPoolGrad,
    GpuDnnReduction,
    GpuDnnSoftmax,
    GpuDnnSoftmaxGrad,
    cudnn,
    dnn_available,
    dnn_conv,
    dnn_conv3d,
    dnn_pool,
    get_precision,
    local_abstractconv3d_cudnn_graph,
    local_abstractconv_cudnn_graph,
    version,
)
from aesara.gpuarray.elemwise import GpuCAReduceCuda, GpuElemwise
from aesara.gpuarray.nnet import GpuSoftmax
from aesara.gpuarray.opt_util import (
    alpha_merge,
    inplace_allocempty,
    op_lifter,
    output_merge,
    pad_dims,
    unpad_dims,
)
from aesara.gpuarray.optdb import (
    gpu_seqopt,
    pool_db,
    pool_db2,
    register_inplace,
    register_opt,
    register_opt2,
)
from aesara.gpuarray.reduction import GpuMaxAndArgmax
from aesara.gpuarray.type import list_contexts
from aesara.graph.opt import GlobalOptimizer, inherit_stack_trace, local_optimizer
from aesara.scalar import Log
from aesara.tensor.math import Argmax
from aesara.tensor.nnet.abstract_conv import (
    AbstractConv2d,
    AbstractConv2d_gradInputs,
    AbstractConv2d_gradWeights,
    AbstractConv3d,
    AbstractConv3d_gradInputs,
    AbstractConv3d_gradWeights,
    assert_conv_shape,
    get_conv_output_shape,
)
from aesara.tensor.nnet.basic import LogSoftmax, SoftmaxGrad
from aesara.tensor.shape import shape_i_op
from aesara.tensor.signal.pool import AveragePoolGrad, MaxPoolGrad, Pool


@local_optimizer([AbstractConv2d, AbstractConv3d])
def local_abstractconv_cudnn(fgraph, node):
    ctx = infer_context_name(*node.inputs)
    if not isinstance(node.inputs[0].type, GpuArrayType):
        return
    if node.op.unshared:
        return None
    if isinstance(node.op.border_mode, tuple) and any(
        isinstance(p, tuple) for p in node.op.border_mode
    ):
        # Asymmetric padding not yet supported
        return None
    if isinstance(node.op, AbstractConv2d):
        with inherit_stack_trace(node.outputs):
            return local_abstractconv_cudnn_graph(
                node.op, ctx, node.inputs, node.outputs
            )
    elif isinstance(node.op, AbstractConv3d):
        with inherit_stack_trace(node.outputs):
            return local_abstractconv3d_cudnn_graph(
                node.op, ctx, node.inputs, node.outputs
            )


@local_optimizer(
    [AbstractConv2d, AbstractConv2d_gradWeights, AbstractConv2d_gradInputs]
)
def local_abstractconv_cudnn_alt(fgraph, node):
    if not isinstance(
        node.op, (AbstractConv2d, AbstractConv2d_gradWeights, AbstractConv2d_gradInputs)
    ):
        return

    if version(raises=False) < 6000 and node.op.filter_dilation != (1, 1):
        return None
    if node.op.unshared:
        return None
    if isinstance(node.op.border_mode, tuple) and any(
        isinstance(p, tuple) for p in node.op.border_mode
    ):
        # Asymmetric padding not yet supported
        return None
    inp1 = node.inputs[0]
    inp2 = node.inputs[1]

    if not dnn_available(inp1.type.context_name):
        return

    op = node.op
    border_mode = node.op.border_mode
    subsample = node.op.subsample
    filter_dilation = node.op.filter_dilation
    num_groups = node.op.num_groups
    precision, _ = get_precision(None, [inp1, inp2])

    if node.op.filter_flip:
        conv_mode = "conv"
    else:
        conv_mode = "cross"

    if isinstance(op, AbstractConv2d):
        if border_mode == "half" or subsample != (1, 1) or num_groups != 1:
            return None
        if border_mode == "full":
            direction_hint = "bprop inputs"
        elif border_mode == "valid" and filter_dilation == (1, 1):
            direction_hint = "bprop weights"
        else:
            return None

        rval = dnn_conv(
            inp1,
            inp2,
            border_mode=border_mode,
            subsample=subsample,
            dilation=filter_dilation,
            direction_hint=direction_hint,
            conv_mode=conv_mode,
            num_groups=num_groups,
        )

    elif isinstance(op, AbstractConv2d_gradWeights):
        if (
            border_mode == "valid"
            and subsample == (1, 1)
            and filter_dilation == (1, 1)
            and num_groups == 1
        ):
            img = gpu_contiguous(inp1)
            topgrad = gpu_contiguous(inp2)
            ctx_name = infer_context_name(img, topgrad)
            img = gpu_contiguous(img.dimshuffle(1, 0, 2, 3))
            topgrad = gpu_contiguous(topgrad.dimshuffle(1, 0, 2, 3))
            ishape = [shape_i_op(i)(img) for i in range(img.ndim)]
            tshape = [shape_i_op(i)(topgrad) for i in range(topgrad.ndim)]
            out_shp = get_conv_output_shape(
                ishape,
                tshape,
                border_mode=border_mode,
                subsample=subsample,
                filter_dilation=filter_dilation,
            )

            out_shp = assert_conv_shape(out_shp)
            out = GpuAllocEmpty(dtype=img.dtype, context_name=ctx_name)(*out_shp)
            desc = GpuDnnConvDesc(
                border_mode=border_mode,
                subsample=subsample,
                dilation=filter_dilation,
                conv_mode="cross",
                precision=precision,
            )(out.shape)

            conv = GpuDnnConv(algo=None, num_groups=num_groups)(img, topgrad, out, desc)
            if conv_mode == "conv":
                conv = conv[:, :, ::-1, ::-1]

            rval = as_gpuarray_variable(conv.dimshuffle(1, 0, 2, 3), ctx_name)
        else:
            return None

    elif isinstance(op, AbstractConv2d_gradInputs):
        if border_mode == "valid" and subsample == (1, 1) and num_groups == 1:
            kerns = gpu_contiguous(inp1.dimshuffle(1, 0, 2, 3))
            topgrad = gpu_contiguous(inp2)
            ctx_name = infer_context_name(kerns, topgrad)
            conv_mode = "cross" if conv_mode == "conv" else "conv"
            desc = GpuDnnConvDesc(
                border_mode="full",
                subsample=subsample,
                dilation=filter_dilation,
                conv_mode=conv_mode,
                precision=precision,
            )(kerns.shape)

            tshape = [shape_i_op(i)(topgrad) for i in range(topgrad.ndim)]
            kshape = [shape_i_op(i)(kerns) for i in range(kerns.ndim)]
            shape = get_conv_output_shape(
                tshape,
                kshape,
                border_mode="full",
                subsample=subsample,
                filter_dilation=filter_dilation,
            )

            shape = assert_conv_shape(shape)
            out = GpuAllocEmpty(dtype=topgrad.dtype, context_name=ctx_name)(*shape)
            rval = GpuDnnConv(algo=None, num_groups=num_groups)(
                topgrad, kerns, out, desc
            )
        else:
            return None

    return [rval]


@local_optimizer(
    [AbstractConv3d, AbstractConv3d_gradWeights, AbstractConv3d_gradInputs]
)
def local_abstractconv3d_cudnn_alt(fgraph, node):
    if not isinstance(
        node.op, (AbstractConv3d, AbstractConv3d_gradWeights, AbstractConv3d_gradInputs)
    ):
        return

    if version(raises=False) < 6000 and node.op.filter_dilation != (1, 1, 1):
        return None
    inp1 = node.inputs[0]
    inp2 = node.inputs[1]

    if not dnn_available(inp1.type.context_name):
        return

    op = node.op
    border_mode = node.op.border_mode
    subsample = node.op.subsample
    filter_dilation = node.op.filter_dilation
    num_groups = node.op.num_groups
    precision, _ = get_precision(None, [inp1, inp2])

    if node.op.filter_flip:
        conv_mode = "conv"
    else:
        conv_mode = "cross"

    if isinstance(op, AbstractConv3d):
        if border_mode == "half" or subsample != (1, 1, 1) or num_groups > 1:
            return None
        if border_mode == "full":
            direction_hint = "bprop inputs"
        elif border_mode == "valid" and filter_dilation == (1, 1, 1):
            direction_hint = "bprop weights"
        else:
            return None

        rval = dnn_conv3d(
            fgraph,
            inp1,
            inp2,
            border_mode=border_mode,
            subsample=subsample,
            dilation=filter_dilation,
            direction_hint=direction_hint,
            conv_mode=conv_mode,
        )

    elif isinstance(op, AbstractConv3d_gradWeights):
        if (
            border_mode == "valid"
            and subsample == (1, 1, 1)
            and filter_dilation == (1, 1, 1)
            and num_groups == 1
        ):
            img = gpu_contiguous(inp1)
            topgrad = gpu_contiguous(inp2)
            ctx_name = infer_context_name(img, topgrad)
            img = gpu_contiguous(img.dimshuffle(1, 0, 2, 3, 4))
            topgrad = gpu_contiguous(topgrad.dimshuffle(1, 0, 2, 3, 4))
            ishape = [shape_i_op(i)(img) for i in range(img.ndim)]
            tshape = [shape_i_op(i)(topgrad) for i in range(topgrad.ndim)]
            out_shp = get_conv_output_shape(
                ishape,
                tshape,
                border_mode=border_mode,
                subsample=subsample,
                filter_dilation=filter_dilation,
            )

            out_shp = assert_conv_shape(out_shp)
            out = GpuAllocEmpty(dtype=img.dtype, context_name=ctx_name)(*out_shp)
            desc = GpuDnnConvDesc(
                border_mode=border_mode,
                subsample=subsample,
                dilation=filter_dilation,
                conv_mode="cross",
                num_groups=num_groups,
                precision=precision,
            )(out.shape)

            conv = GpuDnnConv(algo=None, num_groups=num_groups)(img, topgrad, out, desc)
            if conv_mode == "conv":
                conv = conv[:, :, ::-1, ::-1, ::-1]

            rval = as_gpuarray_variable(conv.dimshuffle(1, 0, 2, 3, 4), ctx_name)
        else:
            return None

    elif isinstance(op, AbstractConv3d_gradInputs):
        if border_mode == "valid" and subsample == (1, 1, 1) and num_groups == 1:
            kerns = gpu_contiguous(inp1.dimshuffle(1, 0, 2, 3, 4))
            topgrad = gpu_contiguous(inp2)
            ctx_name = infer_context_name(kerns, topgrad)
            conv_mode = "cross" if conv_mode == "conv" else "conv"
            desc = GpuDnnConvDesc(
                border_mode="full",
                subsample=subsample,
                dilation=filter_dilation,
                conv_mode=conv_mode,
                num_groups=num_groups,
                precision=precision,
            )(kerns.shape)

            tshape = [shape_i_op(i)(topgrad) for i in range(topgrad.ndim)]
            kshape = [shape_i_op(i)(kerns) for i in range(kerns.ndim)]
            shape = get_conv_output_shape(
                tshape,
                kshape,
                border_mode="full",
                subsample=subsample,
                filter_dilation=filter_dilation,
            )

            shape = assert_conv_shape(shape)
            out = GpuAllocEmpty(dtype=topgrad.dtype, context_name=ctx_name)(*shape)
            rval = GpuDnnConv(algo=None, num_groups=num_groups)(
                topgrad, kerns, out, desc
            )
        else:
            return None

    return [rval]


@local_optimizer([AbstractConv2d_gradWeights, AbstractConv3d_gradWeights])
def local_abstractconv_gw_cudnn(fgraph, node):
    ctx = infer_context_name(*node.inputs)
    if not isinstance(node.inputs[0].type, GpuArrayType):
        return
    if node.op.unshared:
        return None
    if isinstance(node.op.border_mode, tuple) and any(
        isinstance(p, tuple) for p in node.op.border_mode
    ):
        # Asymmetric padding not yet supported
        return None
    if isinstance(node.op, AbstractConv2d_gradWeights):
        with inherit_stack_trace(node.outputs):
            return local_abstractconv_cudnn_graph(
                node.op, ctx, node.inputs, node.outputs
            )
    elif isinstance(node.op, AbstractConv3d_gradWeights):
        with inherit_stack_trace(node.outputs):
            return local_abstractconv3d_cudnn_graph(
                node.op, ctx, node.inputs, node.outputs
            )


@local_optimizer([AbstractConv2d_gradInputs, AbstractConv3d_gradInputs])
def local_abstractconv_gi_cudnn(fgraph, node):
    ctx = infer_context_name(*node.inputs)
    if not isinstance(node.inputs[0].type, GpuArrayType):
        return
    if node.op.unshared:
        return None
    if isinstance(node.op.border_mode, tuple) and any(
        isinstance(p, tuple) for p in node.op.border_mode
    ):
        # Asymmetric padding not yet supported
        return None
    if isinstance(node.op, AbstractConv2d_gradInputs):
        with inherit_stack_trace(node.outputs):
            return local_abstractconv_cudnn_graph(
                node.op, ctx, node.inputs, node.outputs
            )
    elif isinstance(node.op, AbstractConv3d_gradInputs):
        with inherit_stack_trace(node.outputs):
            return local_abstractconv3d_cudnn_graph(
                node.op, ctx, node.inputs, node.outputs
            )


@inplace_allocempty(GpuDnnConv, 2)
def local_dnn_conv_inplace(node, inputs):
    return [
        GpuDnnConv(algo=node.op.algo, inplace=True, num_groups=node.op.num_groups)(
            *inputs
        )
    ]


@inplace_allocempty(GpuDnnConvGradW, 2)
def local_dnn_convgw_inplace(node, inputs):
    return [
        GpuDnnConvGradW(algo=node.op.algo, inplace=True, num_groups=node.op.num_groups)(
            *inputs
        )
    ]


@inplace_allocempty(GpuDnnConvGradI, 2)
def local_dnn_convgi_inplace(node, inputs):
    return [
        GpuDnnConvGradI(algo=node.op.algo, inplace=True, num_groups=node.op.num_groups)(
            *inputs
        )
    ]


optdb.register(
    "local_dnna_conv_inplace",
    aesara.graph.opt.in2out(
        local_dnn_conv_inplace,
        local_dnn_convgw_inplace,
        local_dnn_convgi_inplace,
        name="local_dnna_conv_inplace",
    ),
    70.0,
    "fast_run",
    "inplace",
    "gpuarray",
    "cudnn",
)


@register_opt("cudnn")
@alpha_merge(GpuDnnConv, alpha_in=4, beta_in=5)
def local_dnn_conv_alpha_merge(node, *inputs):
    return [GpuDnnConv(algo=node.op.algo, num_groups=node.op.num_groups)(*inputs)]


@register_opt("cudnn")
@alpha_merge(GpuDnnConvGradW, alpha_in=4, beta_in=5)
def local_dnn_convw_alpha_merge(node, *inputs):
    return [GpuDnnConvGradW(algo=node.op.algo, num_groups=node.op.num_groups)(*inputs)]


@register_opt("cudnn")
@alpha_merge(GpuDnnConvGradI, alpha_in=4, beta_in=5)
def local_dnn_convi_alpha_merge(node, *inputs):
    return [GpuDnnConvGradI(algo=node.op.algo, num_groups=node.op.num_groups)(*inputs)]


@register_opt("cudnn")
@output_merge(GpuDnnConv, alpha_in=4, beta_in=5, out_in=2)
def local_dnn_conv_output_merge(node, *inputs):
    inputs = inputs[0:2] + (gpu_contiguous(inputs[2]),) + inputs[3:]
    return [GpuDnnConv(algo=node.op.algo, num_groups=node.op.num_groups)(*inputs)]


@register_opt("cudnn")
@output_merge(GpuDnnConvGradW, alpha_in=4, beta_in=5, out_in=2)
def local_dnn_convw_output_merge(node, *inputs):
    inputs = inputs[0:2] + (gpu_contiguous(inputs[2]),) + inputs[3:]
    return [GpuDnnConvGradW(algo=node.op.algo, num_groups=node.op.num_groups)(*inputs)]


@register_opt("cudnn")
@output_merge(GpuDnnConvGradI, alpha_in=4, beta_in=5, out_in=2)
def local_dnn_convi_output_merge(node, *inputs):
    inputs = inputs[0:2] + (gpu_contiguous(inputs[2]),) + inputs[3:]
    return [GpuDnnConvGradI(algo=node.op.algo, num_groups=node.op.num_groups)(*inputs)]


def local_gpua_pool_dnn_alternative(fgraph, op, ctx_name, inputs, outputs):
    if not dnn_available(ctx_name):
        return
    if not op.ignore_border:
        return
    img, ws, stride, pad = inputs
    nd = op.ndim
    if nd not in (2, 3):
        return
    img = gpu_contiguous(as_gpuarray_variable(img, ctx_name))
    mode = op.mode
    # dnn_pool expects exactly 2 non-pooling dimensions
    if img.ndim == nd + 2:
        return dnn_pool(img, ws, stride=stride, pad=pad, mode=mode)
    else:
        # reshape to 4D or 5D with 2 non-pooling dimensions
        img_padded = pad_dims(img, 2, nd)
        ret_padded = dnn_pool(img_padded, ws, stride=stride, pad=pad, mode=mode)
        return unpad_dims(ret_padded, img, 2, nd)


pool_db.register(
    "local_gpua_pool_dnn_alternative",
    op_lifter([Pool])(local_gpua_pool_dnn_alternative),
    "gpuarray",
    "fast_compile",
    "fast_run",
    "cudnn",
    position=0,
)
pool_db2.register(
    "local_gpua_pool_dnn_alternative",
    local_optimizer([Pool])(local_gpua_pool_dnn_alternative),
    "gpuarray",
    "fast_compile",
    "fast_run",
    "cudnn",
    position=0,
)


def local_gpua_pool_dnn_grad_stride(fgraph, op, ctx_name, inputs, outputs):
    if not dnn_available(ctx_name):
        return
    if not op.ignore_border:
        return
    inp, out, out_grad, ws, stride, pad = inputs
    nd = op.ndim
    if nd not in (2, 3):
        return
    inp = gpu_contiguous(as_gpuarray_variable(inp, ctx_name))
    out = gpu_contiguous(as_gpuarray_variable(out, ctx_name))
    out_grad = gpu_contiguous(as_gpuarray_variable(out_grad, ctx_name))
    mode = op.mode

    # the GPU ops expect exactly 2 non-pooling dimensions
    if inp.ndim == nd + 2:
        return GpuDnnPoolGrad(mode=mode)(inp, out, out_grad, ws, stride, pad)
    else:
        # reshape to 4D or 5D with 2 non-pooling dimensions
        inp_padded = pad_dims(inp, 2, nd)
        out_padded = pad_dims(out, 2, nd)
        out_grad_padded = pad_dims(out_grad, 2, nd)
        ret_padded = GpuDnnPoolGrad(mode=mode)(
            inp_padded, out_padded, out_grad_padded, ws, stride, pad
        )
        return unpad_dims(ret_padded, inp, 2, nd)


pool_db.register(
    "local_gpua_pool_dnn_grad_stride",
    op_lifter([MaxPoolGrad])(local_gpua_pool_dnn_grad_stride),
    "gpuarray",
    "fast_compile",
    "fast_run",
    "cudnn",
    position=0,
)
pool_db2.register(
    "local_gpua_pool_dnn_grad_stride",
    local_optimizer([MaxPoolGrad])(local_gpua_pool_dnn_grad_stride),
    "gpuarray",
    "fast_compile",
    "fast_run",
    "cudnn",
    position=0,
)


def local_gpua_avg_pool_dnn_grad_stride(fgraph, op, ctx_name, inputs, outputs):
    if not dnn_available(ctx_name):
        return
    if not op.ignore_border:
        return
    inp, out_grad, ws, stride, pad = inputs
    nd = op.ndim
    if nd not in (2, 3):
        return
    inp = gpu_contiguous(as_gpuarray_variable(inp, ctx_name))
    out_grad = gpu_contiguous(as_gpuarray_variable(out_grad, ctx_name))
    mode = op.mode

    # the GPU ops expect exactly 2 non-pooling dimensions
    if inp.ndim == nd + 2:
        # We reuse out_grad because cuDNN does not use the value of the `out`
        # argument but still checks its shape for average pooling. This
        # has been observed in v2 and v3 as far as I know.
        return GpuDnnPoolGrad(mode=mode)(inp, out_grad, out_grad, ws, stride, pad)
    else:
        # reshape to 4D or 5D with 2 non-pooling dimensions
        inp_padded = pad_dims(inp, 2, nd)
        out_grad_padded = pad_dims(out_grad, 2, nd)
        ret_padded = GpuDnnPoolGrad(mode=mode)(
            inp_padded, out_grad_padded, out_grad_padded, ws, stride, pad
        )
        return unpad_dims(ret_padded, inp, 2, nd)


pool_db.register(
    "local_gpua_avg_pool_dnn_grad_stride",
    op_lifter([AveragePoolGrad])(local_gpua_avg_pool_dnn_grad_stride),
    "gpuarray",
    "fast_compile",
    "fast_run",
    "cudnn",
    position=0,
)
pool_db2.register(
    "local_gpua_avg_pool_dnn_grad_stride",
    local_optimizer([AveragePoolGrad])(local_gpua_avg_pool_dnn_grad_stride),
    "gpuarray",
    "fast_compile",
    "fast_run",
    "cudnn",
    position=0,
)


@register_opt("cudnn", "fast_compile")
@local_optimizer([GpuSoftmax])
def local_softmax_dnn(fgraph, node):
    if isinstance(node.op, GpuSoftmax):
        if not dnn_available(node.outputs[0].type.context_name):
            return
        ins = node.inputs[0].dimshuffle(0, 1, "x", "x")
        ins = gpu_contiguous(ins)
        out = GpuDnnSoftmax("accurate", "channel")(ins)
        out = as_gpuarray_variable(out.dimshuffle(0, 1), out.type.context_name)
        return [out]


@register_opt("cudnn", "stabilize")
@local_optimizer([GpuElemwise])
def local_log_softmax_dnn(fgraph, node):
    # This looks for GpuDnnSoftmax so we know that we have cudnn.
    if (
        isinstance(node.op, GpuElemwise)
        and isinstance(node.op.scalar_op, Log)
        and node.inputs[0].owner
        and isinstance(node.inputs[0].owner.op, GpuDnnSoftmax)
        and len(fgraph.clients[node.inputs[0]]) == 1
    ):
        softmax_node = node.inputs[0].owner
        new_softmax = GpuDnnSoftmax("log", softmax_node.op.mode)
        return [new_softmax(softmax_node.inputs[0])]


@register_opt("cudnn", "fast_compile")
@op_lifter([LogSoftmax])
@register_opt2([LogSoftmax], "fast_compile", "cudnn")
def local_gpua_logsoftmax_to_dnn(op, ctx_name, inputs, outputs):
    # Transform the input in the format expected by GpuDnnSoftmax
    inp = inputs[0]
    if inp.ndim != 2:
        return
    if not dnn_available(ctx_name):
        return

    inp = inp.dimshuffle(0, 1, "x", "x")
    inp.tag.context_name = ctx_name

    # Apply GpuDnnSoftmax and return the result
    out = GpuDnnSoftmax("log", "channel")(gpu_contiguous(inp))
    return [out.dimshuffle(0, 1)]


@register_opt("cudnn", "fast_compile")
@op_lifter([SoftmaxGrad])
@register_opt2([SoftmaxGrad], "cudnn", "fast_compile")
def local_gpua_softmax_dnn_grad(op, ctx_name, inputs, outputs):
    if not dnn_available(ctx_name):
        return
    ins = []
    for n in inputs:
        n = as_gpuarray_variable(n, ctx_name)
        if n.ndim != 2:
            return
        ins.append(n.dimshuffle(0, "x", 1, "x"))

    out = GpuDnnSoftmaxGrad("accurate", "instance")(
        gpu_contiguous(ins[0]), gpu_contiguous(ins[1])
    )
    return [out.dimshuffle(0, 2)]


@register_opt("cudnn")
@local_optimizer([GpuCAReduceCuda])
def local_dnn_reduction(fgraph, node):
    if not isinstance(node.op, GpuCAReduceCuda):
        return

    if not dnn_available(node.inputs[0].type.context_name):
        return

    if version(raises=False) < 6000:
        return

    if node.inputs[0].ndim > 8:
        return

    acc_dtype = node.op._acc_dtype(node.inputs[0].dtype)

    if node.inputs[0].dtype != node.outputs[0].dtype:
        # We can mix float16 and float32, but not float64.
        if node.inputs[0].dtype == "float64" or node.outputs[0].dtype == "float64":
            return
        if acc_dtype != "float32":
            return

    if node.inputs[0].dtype not in ("float16", "float32", "float64"):
        return

    if node.inputs[0].dtype == "float64" and acc_dtype != "float64":
        return

    if node.inputs[0].dtype == "float32" and acc_dtype != "float32":
        return

    if node.inputs[0].dtype == "float16" and acc_dtype == "float64":
        return

    def _identity(a):
        return a

    def _square(a):
        return GpuElemwise(aesara.scalar.basic.sqr)(a)

    scal = node.op.scalar_op.name
    post = _identity

    if node.op.pre_scalar_op is not None:
        if isinstance(node.op.scalar_op, aesara.scalar.basic.Add):
            if isinstance(node.op.pre_scalar_op, aesara.scalar.basic.Sqr):
                scal = "norm2"
                post = _square
            elif isinstance(node.op.pre_scalar_op, aesara.scalar.basic.Abs):
                scal = "norm1"
            else:
                return
        elif isinstance(
            node.op.scalar_op, aesara.scalar.basic.ScalarMaximum
        ) and isinstance(node.op.pre_scalar_op, aesara.scalar.basic.Abs):
            scal = "absmax"
        else:
            return

    if not cudnn.cudnnReduceTensorOp_t.has_alias(scal):
        return

    with inherit_stack_trace(node.outputs):
        ret = GpuDnnReduction(scal, node.op.axis, acc_dtype, node.op.dtype, False)(
            node.inputs[0]
        )
        return [post(ret)]


@register_opt("cudnn")
@local_optimizer([GpuMaxAndArgmax])
def local_cudnn_maxandargmax(fgraph, node):
    if not isinstance(node.op, GpuMaxAndArgmax):
        return

    if not dnn_available(node.inputs[0].type.context_name):
        return

    if version(raises=False) < 6000:
        return

    if node.inputs[0].ndim > 8:
        return

    if node.inputs[0].dtype != node.outputs[0].dtype:
        return

    if node.inputs[0].dtype not in ("float16", "float32", "float64"):
        return

    # order of the axes influences the output indices
    if node.op.axis is not None and tuple(sorted(node.op.axis)) != node.op.axis:
        return

    max, arg = GpuDnnReduction(
        "maximum", node.op.axis, node.outputs[0].dtype, node.outputs[0].dtype, True
    )(node.inputs[0])

    # cudnn can only return int32 indices
    return (
        max,
        as_gpuarray_variable(arg.astype("int64"), node.outputs[1].type.context_name),
    )


@register_opt("cudnn", "fast_compile")
@op_lifter([Argmax])
@register_opt2([Argmax], "fast_compile", "cudnn")
def local_dnn_argmax(op, ctx_name, inputs, outputs):
    if not dnn_available(ctx_name):
        return

    if version(raises=False) < 6000:
        return

    if inputs[0].ndim > 8:
        return

    if inputs[0].dtype not in ("float16", "float32", "float64"):
        return

    # order of the axes influences the output indices
    if op.axis is not None and tuple(sorted(op.axis)) != op.axis:
        return

    max, arg = GpuDnnReduction(
        "maximum", op.axis, inputs[0].dtype, inputs[0].dtype, True
    )(*inputs)

    return [as_gpuarray_variable(arg.astype("int64"), ctx_name)]


class NoCuDNNRaise(GlobalOptimizer):
    def apply(self, fgraph):
        """
        Raise a error if cudnn can't be used.

        """
        for c in list_contexts():
            if not dnn_available(c):
                # Make an assert error as we want Aesara to fail, not
                # just skip this optimization.
                raise AssertionError(
                    "cuDNN optimization was enabled, but Aesara was not able "
                    "to use it for context "
                    + str(c)
                    + ". We got this error: \n"
                    + dnn_available.msg
                )


gpu_seqopt.register("NoCuDNNRaise", NoCuDNNRaise(), 0, "cudnn")


@register_inplace()
@local_optimizer([GpuDnnBatchNorm], inplace=True)
def local_batch_norm_inplace_output(fgraph, node):
    if isinstance(node.op, GpuDnnBatchNorm) and not node.op.inplace_output:
        return GpuDnnBatchNorm(
            mode=node.op.mode,
            running_averages=node.op.running_averages,
            inplace_running_mean=node.op.inplace_running_mean,
            inplace_running_var=node.op.inplace_running_var,
            inplace_output=True,
        )(*node.inputs)


@register_inplace()
@local_optimizer([GpuDnnBatchNorm], inplace=True)
def local_batch_norm_inplace_running_mean(fgraph, node):
    if (
        isinstance(node.op, GpuDnnBatchNorm)
        and node.op.running_averages
        and not node.op.inplace_running_mean
    ):
        return GpuDnnBatchNorm(
            mode=node.op.mode,
            running_averages=node.op.running_averages,
            inplace_running_mean=True,
            inplace_running_var=node.op.inplace_running_var,
            inplace_output=node.op.inplace_output,
        )(*node.inputs)


@register_inplace()
@local_optimizer([GpuDnnBatchNorm], inplace=True)
def local_batch_norm_inplace_running_var(fgraph, node):
    if (
        isinstance(node.op, GpuDnnBatchNorm)
        and node.op.running_averages
        and not node.op.inplace_running_var
    ):
        return GpuDnnBatchNorm(
            mode=node.op.mode,
            running_averages=node.op.running_averages,
            inplace_running_mean=node.op.inplace_running_mean,
            inplace_running_var=True,
            inplace_output=node.op.inplace_output,
        )(*node.inputs)


@register_inplace()
@local_optimizer([GpuDnnBatchNormInference], inplace=True)
def local_batch_norm_inference_inplace(fgraph, node):
    if isinstance(node.op, GpuDnnBatchNormInference) and not node.op.inplace:
        return [GpuDnnBatchNormInference(mode=node.op.mode, inplace=True)(*node.inputs)]
