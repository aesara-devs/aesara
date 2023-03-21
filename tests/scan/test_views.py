import numpy as np

import aesara.tensor as at
from aesara import config, function, grad, shared
from aesara.compile.mode import FAST_RUN
from aesara.scan.views import foldl, foldr
from aesara.scan.views import map as at_map
from aesara.scan.views import reduce as at_reduce
from aesara.tensor.type import scalar, vector
from tests import unittest_tools as utt
from tests.scan.test_basic import clone_optimized_graph, grab_scan_node


def test_reduce():
    v = vector("v")
    s = scalar("s")
    result, updates = at_reduce(lambda x, y: x + y, v, s)

    f = function([v, s], result, updates=updates, allow_input_downcast=True)
    rng = np.random.default_rng(utt.fetch_seed())
    v_v = rng.uniform(-5.0, 5.0, size=(5,))
    assert abs(np.sum(v_v) - f(v_v, 0.0)) < 1e-3


def test_map():
    v = vector("v")
    abs_expr, abs_updates = at_map(
        lambda x: abs(x), v, [], truncate_gradient=-1, go_backwards=False
    )

    f = function([v], abs_expr, updates=abs_updates, allow_input_downcast=True)

    rng = np.random.default_rng(utt.fetch_seed())
    vals = rng.uniform(-5.0, 5.0, size=(10,))
    abs_vals = abs(vals)
    aesara_vals = f(vals)
    utt.assert_allclose(abs_vals, aesara_vals)


def test_reduce_memory_consumption():
    x = shared(np.asarray(np.random.uniform(size=(10,)), dtype=config.floatX))
    o, _ = at_reduce(
        lambda v, acc: acc + v,
        x,
        at.constant(np.asarray(0.0, dtype=config.floatX)),
    )
    mode = FAST_RUN
    mode = mode.excluding("inplace")
    f1 = function([], o, mode=mode)
    inputs, outputs = clone_optimized_graph(f1)

    scan_nodes = grab_scan_node(outputs[0])
    assert scan_nodes is not None
    scan_node = scan_nodes[0]
    f1 = function(inputs, scan_node.inputs[2])

    # Originally, the shape would have been 1 due to the SaveMem
    # optimization reducing the size to the number of taps (in this case
    # 1) provided to the inner function. Now, because of the memory-reuse
    # feature in Scan it can be 2 because SaveMem needs to keep a
    # larger buffer to avoid aliasing between the inputs and the outputs.
    if config.scan__allow_output_prealloc:
        assert f1().shape[0] == 2
    else:
        assert f1().shape[0] == 1

    gx = grad(o, x)
    f2 = function([], gx)
    utt.assert_allclose(f2(), np.ones((10,)))


def test_foldl_memory_consumption():
    x = shared(np.asarray(np.random.uniform(size=(10,)), dtype=config.floatX))
    o, _ = foldl(
        lambda v, acc: acc + v,
        x,
        at.constant(np.asarray(0.0, dtype=config.floatX)),
    )

    mode = FAST_RUN
    mode = mode.excluding("inplace")
    f0 = function([], o, mode=mode)
    inputs, outputs = clone_optimized_graph(f0)

    scan_nodes = grab_scan_node(outputs[0])
    assert scan_nodes is not None
    scan_node = scan_nodes[0]
    f1 = function(inputs, scan_node.inputs[2])

    # Originally, the shape would have been 1 due to the SaveMem
    # optimization reducing the size to the number of taps (in this case
    # 1) provided to the inner function. Now, because of the memory-reuse
    # feature in Scan it can be 2 because SaveMem needs to keep a
    # larger buffer to avoid aliasing between the inputs and the outputs.
    if config.scan__allow_output_prealloc:
        assert f1().shape[0] == 2
    else:
        assert f1().shape[0] == 1

    gx = grad(o, x)
    f2 = function([], gx)
    utt.assert_allclose(f2(), np.ones((10,)))


def test_foldr_memory_consumption():
    x = shared(np.asarray(np.random.uniform(size=(10,)), dtype=config.floatX))
    o, _ = foldr(
        lambda v, acc: acc + v,
        x,
        at.constant(np.asarray(0.0, dtype=config.floatX)),
    )

    mode = FAST_RUN
    mode = mode.excluding("inplace")
    f1 = function([], o, mode=mode)
    inputs, outputs = clone_optimized_graph(f1)

    scan_nodes = grab_scan_node(outputs[0])
    assert scan_nodes is not None
    scan_node = scan_nodes[0]
    f1 = function(inputs, scan_node.inputs[2])

    # Originally, the shape would have been 1 due to the SaveMem
    # optimization reducing the size to the number of taps (in this case
    # 1) provided to the inner function. Now, because of the memory-reuse
    # feature in Scan it can be 2 because SaveMem needs to keep a
    # larger buffer to avoid aliasing between the inputs and the outputs.
    if config.scan__allow_output_prealloc:
        assert f1().shape[0] == 2
    else:
        assert f1().shape[0] == 1

    gx = grad(o, x)
    f2 = function([], gx)
    utt.assert_allclose(f2(), np.ones((10,)))
