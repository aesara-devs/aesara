import numpy as np
import pytest

import aesara.scalar as aes
from aesara.compile.function import function
from aesara.compile.mode import OPT_NONE, Mode, get_default_mode
from aesara.graph.fg import FunctionGraph
from aesara.graph.rewriting.utils import rewrite_graph
from aesara.tensor.basic import Alloc, alloc, as_tensor_variable, second
from aesara.tensor.elemwise import DimShuffle, Elemwise
from aesara.tensor.extra_ops import BroadcastTo, Repeat, Unique, repeat, unique
from aesara.tensor.type import dscalar


@pytest.mark.parametrize("return_index", [False])
@pytest.mark.parametrize("return_counts", [False])
@pytest.mark.parametrize("return_inverse", [False])
def test_local_Unique_scalar(return_index, return_counts, return_inverse):
    x = dscalar()
    y = unique(
        x,
        return_index=return_index,
        return_counts=return_counts,
        return_inverse=return_inverse,
        axis=None,
    )

    y_fg = FunctionGraph(outputs=[y], copy_inputs=False)
    y_rewritten_fg = rewrite_graph(
        y_fg, clone=False, include=["canonicalize", "local_Unique_scalar"]
    )
    y_rewritten = y_rewritten_fg.outputs[0]
    y_rewritten_start = y_rewritten

    assert isinstance(y_rewritten_start.owner.op, DimShuffle)
    assert y_rewritten_start.owner.inputs[0] == x

    default_mode = get_default_mode()
    rewrite_mode = default_mode.excluding("local_Unique_scalar")
    y_fn = function([x], [y, y_rewritten], mode=rewrite_mode)

    x_val = np.array(-10.0, dtype=np.float64)
    y_exp_val, y_val = y_fn(x_val)
    assert np.array_equal(y_exp_val, y_val)


@pytest.mark.parametrize(
    "x_val, axis, new_shape",
    [
        (np.array(-10, dtype=np.int64), None, ()),
        (np.array(-10, dtype=np.int64), None, (2, 3)),
        (np.array([[-10, -3], [-10, 2], [-10, 2]], dtype=np.int64), None, (2, 3, 2)),
    ],
)
@pytest.mark.parametrize("return_index", [False])
@pytest.mark.parametrize("return_counts", [False])
@pytest.mark.parametrize("return_inverse", [False])
def test_local_Unique_Alloc_lift(
    x_val, axis, new_shape, return_index, return_counts, return_inverse
):
    x = as_tensor_variable(x_val).type()
    y = unique(
        alloc(x, *new_shape),
        return_index=return_index,
        return_counts=return_counts,
        return_inverse=return_inverse,
        axis=axis,
    )

    if isinstance(y, list):
        y, *_ = y

    # This approach allows us to directly confirm that `x` is in the result.
    y_fg = FunctionGraph(outputs=[y], copy_inputs=False)
    y_rewritten_fg = rewrite_graph(
        y_fg,
        clone=False,
        include=["canonicalize", "local_Unique_Alloc_lift"],
        exclude=["local_Unique_scalar"],
    )
    y_rewritten = y_rewritten_fg.outputs[0]
    y_rewritten_start = y_rewritten

    assert isinstance(y_rewritten_start.owner.op, Unique)
    assert y_rewritten_start.owner.inputs[0] == x
    assert not any(isinstance(node.op, Alloc) for node in y_rewritten_fg.apply_nodes)

    default_mode = get_default_mode()
    # The rewrite has already been applied to `y_rewritten`, so we can--and
    # should--exclude it from the compilation of both our reference, `y`, and
    # the rewritten result, `y_rewritten`.
    # The remaining exclusions simply allow us to perform the check below that
    # makes sure the original `Alloc` is present in our reference (sub)graph.
    rewrite_mode = default_mode.excluding(
        "local_useless_alloc", "local_alloc_sink_dimshuffle", "local_Unique_Alloc_lift"
    )
    y_fn = function([x], [y, y_rewritten], mode=rewrite_mode)
    # Make sure that the original `Alloc` is used to compute the reference `y`
    # result
    assert any(isinstance(node.op, Alloc) for node in y_fn.maker.fgraph.apply_nodes)

    y_exp_val, y_val = y_fn(x_val)
    assert np.array_equal(y_exp_val, y_val)


@pytest.mark.parametrize(
    "x_val, axis, new_shape",
    [
        (np.array(-10, dtype=np.int64), None, (2, 3)),
        (np.array([[-10, -3], [-10, 2], [-10, 2]], dtype=np.int64), None, (2, 3, 2)),
    ],
)
@pytest.mark.parametrize("return_index", [False])
@pytest.mark.parametrize("return_counts", [False])
@pytest.mark.parametrize("return_inverse", [False])
def test_local_Unique_BroadcastTo(
    x_val, axis, new_shape, return_index, return_counts, return_inverse
):
    x = as_tensor_variable(x_val).type()
    y = unique(
        BroadcastTo()(x, tuple(new_shape)),
        return_index=return_index,
        return_counts=return_counts,
        return_inverse=return_inverse,
        axis=axis,
    )

    if isinstance(y, list):
        y, *_ = y

    # This approach allows us to directly confirm that `x` is in the result.
    y_fg = FunctionGraph(outputs=[y], copy_inputs=False)
    y_rewritten_fg = rewrite_graph(
        y_fg,
        clone=False,
        include=["canonicalize", "local_Unique_BroadcastTo_lift"],
        exclude=["local_Unique_scalar"],
    )
    y_rewritten = y_rewritten_fg.outputs[0]
    y_rewritten_start = y_rewritten

    assert isinstance(y_rewritten_start.owner.op, Unique)
    assert y_rewritten_start.owner.inputs[0] == x
    assert not any(
        isinstance(node.op, BroadcastTo) for node in y_rewritten_fg.apply_nodes
    )

    default_mode = get_default_mode()
    # The rewrite has already been applied to `y_rewritten`, so we can--and
    # should--exclude it from the compilation of both our reference, `y`, and
    # the rewritten result, `y_rewritten`.
    rewrite_mode = default_mode.excluding("local_Unique_BroadcastTo_lift")
    y_fn = function([x], [y, y_rewritten], mode=rewrite_mode)
    # Make sure that the original `BroadcastTo` is used to compute the
    # reference `y` result
    assert any(
        isinstance(node.op, BroadcastTo) for node in y_fn.maker.fgraph.apply_nodes
    )

    y_exp_val, y_val = y_fn(x_val)
    assert np.array_equal(y_exp_val, y_val)


@pytest.mark.parametrize(
    "x_val, unique_axis, repeats, repeat_axis",
    [
        (np.array([[-10, -3], [-10, 2]], dtype=np.int64), None, (1, 2), 0),
    ],
)
@pytest.mark.parametrize("return_index", [False])
@pytest.mark.parametrize("return_counts", [False])
@pytest.mark.parametrize("return_inverse", [False])
def test_local_Unique_Repeat(
    x_val,
    unique_axis,
    repeats,
    repeat_axis,
    return_index,
    return_counts,
    return_inverse,
):
    x = as_tensor_variable(x_val).type()
    y = unique(
        repeat(x, tuple(repeats), axis=repeat_axis),
        return_index=return_index,
        return_counts=return_counts,
        return_inverse=return_inverse,
        axis=unique_axis,
    )

    if isinstance(y, list):
        y, *_ = y

    # This approach allows us to directly confirm that `x` is in the result.
    y_fg = FunctionGraph(outputs=[y], copy_inputs=False)
    y_rewritten_fg = rewrite_graph(
        y_fg,
        clone=False,
        include=["canonicalize", "local_Unique_Repeat_lift"],
        exclude=["local_Unique_scalar"],
    )
    y_rewritten = y_rewritten_fg.outputs[0]
    y_rewritten_start = y_rewritten

    assert isinstance(y_rewritten_start.owner.op, Unique)
    assert y_rewritten_start.owner.inputs[0] == x
    assert not any(isinstance(node.op, Repeat) for node in y_rewritten_fg.apply_nodes)

    default_mode = get_default_mode()
    # The rewrite has already been applied to `y_rewritten`, so we can--and
    # should--exclude it from the compilation of both our reference, `y`, and
    # the rewritten result, `y_rewritten`.
    rewrite_mode = default_mode.excluding("local_Unique_Repeat_lift")
    y_fn = function([x], [y, y_rewritten], mode=rewrite_mode)
    # Make sure that the original `BroadcastTo` is used to compute the
    # reference `y` result
    assert any(isinstance(node.op, Repeat) for node in y_fn.maker.fgraph.apply_nodes)

    y_exp_val, y_val = y_fn(x_val)
    assert np.array_equal(y_exp_val, y_val)


@pytest.mark.parametrize(
    "x_val, unique_axis, new_shape",
    [
        (np.array(-10, dtype=np.int64), None, ()),
        (np.array(-10, dtype=np.int64), None, (2, 3)),
        (np.array([[-10, -3], [-10, 2], [-10, 2]], dtype=np.int64), None, (2, 3, 2)),
    ],
)
@pytest.mark.parametrize("return_index", [False])
@pytest.mark.parametrize("return_counts", [False])
@pytest.mark.parametrize("return_inverse", [False])
def test_local_Unique_second(
    x_val, unique_axis, new_shape, return_index, return_counts, return_inverse
):
    x = as_tensor_variable(x_val).type()
    a = np.zeros(tuple(new_shape), dtype=x.dtype)
    y = unique(
        second(a, x),
        return_index=return_index,
        return_counts=return_counts,
        return_inverse=return_inverse,
        axis=unique_axis,
    )

    if isinstance(y, list):
        y, *_ = y

    # This approach allows us to directly confirm that `x` is in the result.
    y_fg = FunctionGraph(outputs=[y], copy_inputs=False)
    y_rewritten_fg = rewrite_graph(
        y_fg,
        clone=False,
        include=["canonicalize", "local_Unique_second_lift"],
        exclude=["local_Unique_scalar", "topo_constant_folding"],
    )
    y_rewritten = y_rewritten_fg.outputs[0]
    y_rewritten_start = y_rewritten

    assert isinstance(y_rewritten_start.owner.op, Unique)

    y_rewritten_start = y_rewritten_start.owner.inputs[0]

    if y_rewritten_start.owner and isinstance(y_rewritten_start.owner.op, DimShuffle):
        y_rewritten_start = y_rewritten_start.owner.inputs[0]

    assert y_rewritten_start == x
    assert not any(
        isinstance(node.op.scalar_op, aes.Second)
        for node in y_rewritten_fg.apply_nodes
        if isinstance(node.op, Elemwise)
    )

    # The rewrite has already been applied to `y_rewritten`, so we can--and
    # should--exclude it from the compilation of both our reference, `y`, and
    # the rewritten result, `y_rewritten`.
    y_fn = function([x], [y, y_rewritten], mode=Mode(optimizer=OPT_NONE))

    # Make sure that the original `BroadcastTo` is used to compute the
    # reference `y` result
    assert any(
        isinstance(node.op.scalar_op, aes.Second)
        for node in y_fn.maker.fgraph.apply_nodes
        if isinstance(node.op, Elemwise)
    )

    y_exp_val, y_val = y_fn(x_val)
    assert np.array_equal(y_exp_val, y_val)


def test_local_remove_scalar_BroadcastTo():
    x = dscalar()
    y = BroadcastTo()(x, ())

    assert isinstance(y.owner.op, BroadcastTo)

    res = rewrite_graph(
        y, clone=False, include=["canonicalize", "local_remove_scalar_BroadcastTo"]
    )

    assert res is x
