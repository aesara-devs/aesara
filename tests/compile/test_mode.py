import pytest

import theano
from theano.compile.mode import AddFeatureOptimizer, Mode
from theano.graph.toolbox import NoOutputFromInplace
from theano.tensor.math import dot, tanh
from theano.tensor.type import matrix


@pytest.mark.skipif(
    not theano.config.cxx, reason="G++ not available, so we need to skip this test."
)
def test_no_output_from_implace():
    x = matrix()
    y = matrix()
    a = dot(x, y)
    b = tanh(a)

    # Ensure that the elemwise op that produces the output is inplace when
    # using a mode that does not include the optimization
    fct_no_opt = theano.function([x, y], b, mode="FAST_RUN")
    op = fct_no_opt.maker.fgraph.outputs[0].owner.op
    assert hasattr(op, "destroy_map") and 0 in op.destroy_map

    # Ensure that the elemwise op that produces the output is not inplace when
    # using a mode that includes the optimization
    opt = AddFeatureOptimizer(NoOutputFromInplace())
    mode_opt = Mode(linker="cvm", optimizer="fast_run").register((opt, 49.9))

    fct_opt = theano.function([x, y], b, mode=mode_opt)
    op = fct_opt.maker.fgraph.outputs[0].owner.op
    assert not hasattr(op, "destroy_map") or 0 not in op.destroy_map


def test_including():
    mode = Mode(optimizer="merge")
    mode.including("fast_compile")
