"""
Optimizations addressing the ops in nnet root directory
"""

import aesara
from aesara import compile
from aesara.compile import optdb
from aesara.configdefaults import config
from aesara.graph.rewriting.basic import (
    MetaNodeRewriterSkip,
    WalkingGraphRewriter,
    copy_stack_trace,
    in2out,
    node_rewriter,
)
from aesara.tensor.nnet.abstract_conv import (
    AbstractConv2d,
    AbstractConv2d_gradInputs,
    AbstractConv2d_gradWeights,
    AbstractConv3d,
    AbstractConv3d_gradInputs,
    AbstractConv3d_gradWeights,
    get_conv_output_shape,
)
from aesara.tensor.nnet.blocksparse import (
    SparseBlockGemv,
    SparseBlockOuter,
    sparse_block_gemv_inplace,
    sparse_block_outer_inplace,
)

# Cpu implementation
from aesara.tensor.nnet.conv import ConvOp, conv2d
from aesara.tensor.nnet.corr import CorrMM, CorrMM_gradInputs, CorrMM_gradWeights
from aesara.tensor.nnet.corr3d import Corr3dMM, Corr3dMMGradInputs, Corr3dMMGradWeights
from aesara.tensor.rewriting.basic import register_specialize_device
from aesara.tensor.type import TensorType


@node_rewriter([SparseBlockGemv], inplace=True)
def local_inplace_sparse_block_gemv(fgraph, node):
    """
    SparseBlockGemv(inplace=False) -> SparseBlockGemv(inplace=True)
    """
    if isinstance(node.op, SparseBlockGemv) and not node.op.inplace:
        new_node = sparse_block_gemv_inplace(*node.inputs)
        copy_stack_trace(node.outputs[0], new_node)
        return [new_node]
    return False


compile.optdb.register(
    "local_inplace_sparse_block_gemv",
    WalkingGraphRewriter(
        local_inplace_sparse_block_gemv,
        failure_callback=WalkingGraphRewriter.warn_inplace,
    ),
    "fast_run",
    "inplace",
    position=60,
)


@node_rewriter([SparseBlockOuter], inplace=True)
def local_inplace_sparse_block_outer(fgraph, node):
    """
    SparseBlockOuter(inplace=False) -> SparseBlockOuter(inplace=True)
    """
    if isinstance(node.op, SparseBlockOuter) and not node.op.inplace:
        new_node = sparse_block_outer_inplace(*node.inputs)
        copy_stack_trace(node.outputs[0], new_node)
        return [new_node]
    return False


compile.optdb.register(
    "local_inplace_sparse_block_outer",
    WalkingGraphRewriter(
        local_inplace_sparse_block_outer,
        failure_callback=WalkingGraphRewriter.warn_inplace,
    ),
    "fast_run",
    "inplace",
    position=60,
)


# Conv opts
@node_rewriter([AbstractConv2d])
def local_abstractconv_gemm(fgraph, node):
    # If config.blas__ldflags is empty, Aesara will use
    # a NumPy C implementation of [sd]gemm_.
    if config.cxx == "" or node.inputs[0].dtype == "float16":
        return
    if not isinstance(node.op, AbstractConv2d):
        return None
    img, kern = node.inputs
    if not isinstance(img.type, TensorType) or not isinstance(kern.type, TensorType):
        return None

    # need to flip the kernel if necessary
    if node.op.filter_flip:
        flip = (slice(None),) * (kern.ndim - 2) + (slice(None, None, -1),) * 2
        kern = kern[flip]
    rval = CorrMM(
        border_mode=node.op.border_mode,
        subsample=node.op.subsample,
        filter_dilation=node.op.filter_dilation,
        num_groups=node.op.num_groups,
        unshared=node.op.unshared,
    )(img, kern)
    copy_stack_trace(node.outputs[0], rval)

    return [rval]


@node_rewriter([AbstractConv3d])
def local_abstractconv3d_gemm(fgraph, node):
    # If config.blas__ldflags is empty, Aesara will use
    # a NumPy C implementation of [sd]gemm_.
    if config.cxx == "" or node.inputs[0].dtype == "float16":
        return
    if not isinstance(node.op, AbstractConv3d):
        return None
    img, kern = node.inputs
    if not isinstance(img.type, TensorType) or not isinstance(kern.type, TensorType):
        return None

    # need to flip the kernel if necessary
    if node.op.filter_flip:
        kern = kern[:, :, ::-1, ::-1, ::-1]
    rval = Corr3dMM(
        border_mode=node.op.border_mode,
        subsample=node.op.subsample,
        filter_dilation=node.op.filter_dilation,
        num_groups=node.op.num_groups,
    )(img, kern)
    copy_stack_trace(node.outputs[0], rval)

    return [rval]


@node_rewriter([AbstractConv2d_gradWeights])
def local_abstractconv_gradweight_gemm(fgraph, node):
    # If config.blas__ldflags is empty, Aesara will use
    # a NumPy C implementation of [sd]gemm_.
    if config.cxx == "" or node.inputs[0].dtype == "float16":
        return
    if not isinstance(node.op, AbstractConv2d_gradWeights):
        return None
    img, topgrad, shape = node.inputs
    if not isinstance(img.type, TensorType) or not isinstance(topgrad.type, TensorType):
        return None

    rval = CorrMM_gradWeights(
        border_mode=node.op.border_mode,
        subsample=node.op.subsample,
        filter_dilation=node.op.filter_dilation,
        num_groups=node.op.num_groups,
        unshared=node.op.unshared,
    )(img, topgrad, shape)
    copy_stack_trace(node.outputs[0], rval)

    # need to flip the kernel if necessary
    if node.op.filter_flip:
        flip = (slice(None),) * (rval.ndim - 2) + (slice(None, None, -1),) * 2
        rval = rval[flip]
        copy_stack_trace(node.outputs[0], rval)

    return [rval]


@node_rewriter([AbstractConv3d_gradWeights])
def local_abstractconv3d_gradweight_gemm(fgraph, node):
    # If config.blas__ldflags is empty, Aesara will use
    # a NumPy C implementation of [sd]gemm_.
    if config.cxx == "" or node.inputs[0].dtype == "float16":
        return
    if not isinstance(node.op, AbstractConv3d_gradWeights):
        return None
    img, topgrad, shape = node.inputs
    if not isinstance(img.type, TensorType) or not isinstance(topgrad.type, TensorType):
        return None

    rval = Corr3dMMGradWeights(
        border_mode=node.op.border_mode,
        subsample=node.op.subsample,
        filter_dilation=node.op.filter_dilation,
        num_groups=node.op.num_groups,
    )(img, topgrad, shape)
    copy_stack_trace(node.outputs[0], rval)

    # need to flip the kernel if necessary
    if node.op.filter_flip:
        rval = rval[:, :, ::-1, ::-1, ::-1]
        copy_stack_trace(node.outputs[0], rval)

    return [rval]


@node_rewriter([AbstractConv2d_gradInputs])
def local_abstractconv_gradinputs_gemm(fgraph, node):
    # If config.blas__ldflags is empty, Aesara will use
    # a NumPy C implementation of [sd]gemm_.
    if config.cxx == "" or node.inputs[0].dtype == "float16":
        return
    if not isinstance(node.op, AbstractConv2d_gradInputs):
        return None
    kern, topgrad, shape = node.inputs
    if not isinstance(kern.type, TensorType) or not isinstance(
        topgrad.type, TensorType
    ):
        return None

    # need to flip the kernel if necessary
    if node.op.filter_flip:
        flip = (slice(None),) * (kern.ndim - 2) + (slice(None, None, -1),) * 2
        kern = kern[flip]
    rval = CorrMM_gradInputs(
        border_mode=node.op.border_mode,
        subsample=node.op.subsample,
        filter_dilation=node.op.filter_dilation,
        num_groups=node.op.num_groups,
        unshared=node.op.unshared,
    )(kern, topgrad, shape)
    copy_stack_trace(node.outputs[0], rval)

    return [rval]


@node_rewriter([AbstractConv3d_gradInputs])
def local_abstractconv3d_gradinputs_gemm(fgraph, node):
    # If config.blas__ldflags is empty, Aesara will use
    # a NumPy C implementation of [sd]gemm_.
    if config.cxx == "" or node.inputs[0].dtype == "float16":
        return
    if not isinstance(node.op, AbstractConv3d_gradInputs):
        return None
    kern, topgrad, shape = node.inputs
    if not isinstance(kern.type, TensorType) or not isinstance(
        topgrad.type, TensorType
    ):
        return None

    # need to flip the kernel if necessary
    if node.op.filter_flip:
        kern = kern[:, :, ::-1, ::-1, ::-1]
    rval = Corr3dMMGradInputs(
        border_mode=node.op.border_mode,
        subsample=node.op.subsample,
        filter_dilation=node.op.filter_dilation,
        num_groups=node.op.num_groups,
    )(kern, topgrad, shape)
    copy_stack_trace(node.outputs[0], rval)

    return [rval]


@node_rewriter([AbstractConv2d])
def local_conv2d_cpu(fgraph, node):

    if not isinstance(node.op, AbstractConv2d) or node.inputs[0].dtype == "float16":
        return None

    img, kern = node.inputs
    if not isinstance(img.type, TensorType) or not isinstance(kern.type, TensorType):
        return None
    if node.op.border_mode not in ("full", "valid"):
        return None
    if not node.op.filter_flip:
        # Not tested yet
        return None
    if node.op.num_groups > 1 or node.op.unshared:
        return None
    if node.op.filter_dilation != (1, 1):
        return None

    rval = conv2d(
        img,
        kern,
        node.op.imshp,
        node.op.kshp,
        border_mode=node.op.border_mode,
        subsample=node.op.subsample,
    )

    copy_stack_trace(node.outputs[0], rval)
    return [rval]


@node_rewriter([AbstractConv2d_gradWeights])
def local_conv2d_gradweight_cpu(fgraph, node):
    if (
        not isinstance(node.op, AbstractConv2d_gradWeights)
        or node.inputs[0].dtype == "float16"
    ):
        return None

    img, topgrad, shape = node.inputs

    if not isinstance(img.type, TensorType) or not isinstance(topgrad.type, TensorType):
        return None
    if node.op.border_mode not in ("full", "valid"):
        return None
    if not node.op.filter_flip:
        # Not tested yet
        return
    if node.op.num_groups > 1 or node.op.unshared:
        return None

    if node.op.border_mode == "valid" and (node.op.subsample != (1, 1)):
        return None

    dx, dy = node.op.subsample
    if dx not in (1, 2) or dy not in (1, 2):
        # Not implemented in the gradient of ConvOp
        return None

    if node.op.imshp is None:
        op_imshp = (None, None, None, None)
    else:
        op_imshp = node.op.imshp

    if node.op.kshp is None:
        op_kshp = (None, None, None, None)
    else:
        op_kshp = node.op.kshp

    if None in op_imshp or None in op_kshp:
        if (dx, dy) != (1, 1):
            # We cannot infer the shapes
            return None

    # Determine gradient on kernels
    assert len(op_imshp) == 4 and len(op_kshp) == 4

    outshp = get_conv_output_shape(
        op_imshp,
        op_kshp,
        node.op.border_mode,
        node.op.subsample,
        node.op.filter_dilation,
    )[2:]
    fulloutshp = get_conv_output_shape(op_imshp, op_kshp, node.op.border_mode, (1, 1))[
        2:
    ]

    newimg = img.dimshuffle((1, 0, 2, 3))
    newtopgrad = topgrad.dimshuffle((1, 0, 2, 3))

    if node.op.border_mode == "valid":
        (img, filters) = (newimg, newtopgrad)
        kshp_logical = fulloutshp
        kshp_logical_top_aligned = False
        imshp_logical = None
        (bsize, nkern) = (op_imshp[1], op_kshp[0])
        imshp = (op_imshp[0], op_imshp[2], op_imshp[3])
        kshp = outshp
    elif node.op.border_mode == "full":
        (img, filters) = (newtopgrad, newimg)
        kshp_logical = None
        kshp_logical_top_aligned = True
        imshp_logical = (op_imshp[0], fulloutshp[0], fulloutshp[1])
        (bsize, nkern) = (op_kshp[0], op_imshp[1])
        imshp = (op_imshp[0], outshp[0], outshp[1])
        kshp = op_imshp[2:]
    else:
        raise NotImplementedError("Only [full,valid] modes are currently supported.")

    # Flip the kernels
    filters = filters[:, :, ::-1, ::-1]

    dw = ConvOp(
        imshp,
        kshp,
        nkern,
        bsize,
        1,
        1,
        output_mode="valid",
        unroll_batch=None,
        unroll_kern=None,
        unroll_patch=None,
        imshp_logical=imshp_logical,
        kshp_logical=kshp_logical,
        kshp_logical_top_aligned=kshp_logical_top_aligned,
        direction_hint="bprop weights",
    )
    res = dw(img, filters)
    copy_stack_trace(node.outputs[0], res)

    if node.op.border_mode == "valid":
        res = res.dimshuffle((1, 0, 2, 3))
        res = res[:, :, ::-1, ::-1]
        copy_stack_trace(node.outputs[0], res)

    return [res]


@node_rewriter([AbstractConv2d_gradInputs])
def local_conv2d_gradinputs_cpu(fgraph, node):
    if (
        not isinstance(node.op, AbstractConv2d_gradInputs)
        or node.inputs[0].dtype == "float16"
    ):
        return None

    kern, topgrad, shape = node.inputs

    if not isinstance(kern.type, TensorType) or not isinstance(
        topgrad.type, TensorType
    ):
        return None
    if node.op.border_mode not in ("full", "valid"):
        return None
    if not node.op.filter_flip:
        # Not tested yet
        return None
    if node.op.num_groups > 1 or node.op.unshared:
        return None

    # Conv 3d implementation, needed when subsample > 2
    if node.op.border_mode == "valid" and node.op.subsample != (1, 1):
        # The op don't support that anymore.
        return False

    # Conv2d Implementation
    dx, dy = node.op.subsample
    if dx not in (1, 2) or dy not in (1, 2):
        # Not implemented in the gradient of ConvOp
        return None

    if node.op.imshp is None:
        op_imshp = (None, None, None, None)
    else:
        op_imshp = node.op.imshp

    if node.op.kshp is None:
        op_kshp = (None, None, None, None)
    else:
        op_kshp = node.op.kshp

    if None in op_imshp or None in op_kshp:
        if (dx, dy) != (1, 1):
            return None

    mode = "valid"
    if node.op.border_mode != "full":
        mode = "full"
    filters = kern.dimshuffle((1, 0, 2, 3))
    filters = filters[:, :, ::-1, ::-1]

    outshp = get_conv_output_shape(
        op_imshp,
        op_kshp,
        node.op.border_mode,
        node.op.subsample,
        node.op.filter_dilation,
    )[2:]
    fulloutshp = get_conv_output_shape(op_imshp, op_kshp, node.op.border_mode, (1, 1))[
        2:
    ]

    nkern = op_imshp[1]
    imshp = (op_kshp[0], outshp[0], outshp[1])
    imshp_logical = (op_kshp[0], fulloutshp[0], fulloutshp[1])
    din = ConvOp(
        imshp,
        op_kshp[2:],
        nkern,
        op_imshp[0],
        1,
        1,
        output_mode=mode,
        unroll_batch=None,
        unroll_kern=None,
        unroll_patch=None,
        imshp_logical=imshp_logical,
        kshp_logical=None,
        version=-1,
        direction_hint="bprop inputs",
    )
    din = din(topgrad, filters)
    copy_stack_trace(node.outputs[0], din)
    return [din]


# Register Cpu Optimization
conv_groupopt = aesara.graph.rewriting.db.LocalGroupDB()
conv_groupopt.__name__ = "conv_opts"
register_specialize_device(conv_groupopt, "fast_compile", "fast_run")

# GEMM-based convolution
# It can be disabled by excluding 'conv_gemm'.
conv_groupopt.register(
    "local_abstractconv_gemm",
    local_abstractconv_gemm,
    "conv_gemm",
    "fast_compile",
    "fast_run",
    position=30,
)
conv_groupopt.register(
    "local_abstractconv_gradweight_gemm",
    local_abstractconv_gradweight_gemm,
    "conv_gemm",
    "fast_compile",
    "fast_run",
    position=30,
)
conv_groupopt.register(
    "local_abstractconv_gradinputs_gemm",
    local_abstractconv_gradinputs_gemm,
    "conv_gemm",
    "fast_compile",
    "fast_run",
    position=30,
)
conv_groupopt.register(
    "local_abstractconv3d_gemm",
    local_abstractconv3d_gemm,
    "conv_gemm",
    "fast_compile",
    "fast_run",
    position=30,
)
conv_groupopt.register(
    "local_abstractconv3d_gradweight_gemm",
    local_abstractconv3d_gradweight_gemm,
    "conv_gemm",
    "fast_compile",
    "fast_run",
    position=30,
)
conv_groupopt.register(
    "local_abstractconv3d_gradinputs_gemm",
    local_abstractconv3d_gradinputs_gemm,
    "conv_gemm",
    "fast_compile",
    "fast_run",
    position=30,
)

# Legacy convolution
conv_groupopt.register(
    "local_conv2d_cpu", local_conv2d_cpu, "fast_compile", "fast_run", position=40
)
conv_groupopt.register(
    "local_conv2d_gradweight_cpu",
    local_conv2d_gradweight_cpu,
    "fast_compile",
    "fast_run",
    position=40,
)
conv_groupopt.register(
    "local_conv2d_gradinputs_cpu",
    local_conv2d_gradinputs_cpu,
    "fast_compile",
    "fast_run",
    position=40,
)


# Verify that no AbstractConv are present in the graph
@node_rewriter(
    [
        AbstractConv2d,
        AbstractConv2d_gradWeights,
        AbstractConv2d_gradInputs,
        AbstractConv3d,
        AbstractConv3d_gradWeights,
        AbstractConv3d_gradInputs,
    ]
)
def local_abstractconv_check(fgraph, node):
    if isinstance(
        node.op,
        (
            AbstractConv2d,
            AbstractConv2d_gradWeights,
            AbstractConv2d_gradInputs,
            AbstractConv3d,
            AbstractConv3d_gradWeights,
            AbstractConv3d_gradInputs,
        ),
    ):
        raise MetaNodeRewriterSkip(
            f"{node.op.__class__.__name__} Aesara rewriting failed: there is no implementation "
            "available supporting the requested options. If on CPU, "
            "do you have a BLAS library installed Aesara can link against? "
            "On the CPU we do not support float16."
        )


optdb.register(
    "AbstractConvCheck",
    in2out(local_abstractconv_check, name="AbstractConvCheck"),
    "fast_compile",
    "fast_run",
    position=48.7,
)
