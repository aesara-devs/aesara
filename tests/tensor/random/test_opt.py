import numpy as np

from theano.compile.function import function
from theano.compile.mode import Mode
from theano.gof.optdb import Query
from theano.tensor.random.basic import normal


opts = Query(include=["random_make_inplace"], exclude=[])
inplace_mode = Mode("py", opts)


def test_inplace_optimization():

    out = normal(0, 1)

    assert out.owner.op.inplace is False

    f = function(
        [],
        out,
        mode=inplace_mode,
    )

    (new_out,) = f.maker.fgraph.outputs
    assert new_out.type == out.type
    assert isinstance(new_out.owner.op, type(out.owner.op))
    assert new_out.owner.op.inplace is True
    assert all(
        np.array_equal(a.data, b.data)
        for a, b in zip(new_out.owner.inputs[1:], out.owner.inputs[1:])
    )
