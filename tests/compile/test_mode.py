from aesara.compile.function import function
from aesara.compile.mode import AddFeatureOptimizer, Mode
from aesara.graph.features import NoOutputFromInplace
from aesara.graph.rewriting.db import RewriteDatabaseQuery, SequenceDB
from aesara.tensor.math import dot, tanh
from aesara.tensor.type import matrix


def test_Mode_basic():
    db = SequenceDB()
    mode = Mode(linker="py", optimizer=RewriteDatabaseQuery(include=None), db=db)

    assert mode.optdb is db

    assert str(mode).startswith("Mode(linker=py, optimizer=RewriteDatabaseQuery")


def test_NoOutputFromInplace():
    x = matrix()
    y = matrix()
    a = dot(x, y)
    b = tanh(a)
    c = tanh(dot(2 * x, y))

    # Ensure that the elemwise op that produces the output is inplace when
    # using a mode that does not include the optimization
    fct_no_opt = function([x, y], [b, c], mode="FAST_RUN")
    op = fct_no_opt.maker.fgraph.outputs[0].owner.op
    assert op.destroy_map and 0 in op.destroy_map
    op = fct_no_opt.maker.fgraph.outputs[1].owner.op
    assert op.destroy_map and 0 in op.destroy_map

    # Ensure that the elemwise op that produces the output is not inplace when
    # using a mode that includes the optimization
    opt = AddFeatureOptimizer(NoOutputFromInplace([1]))
    mode_opt = Mode(linker="py", optimizer="fast_run").register((opt, 49.9))

    fct_opt = function([x, y], [b, c], mode=mode_opt)
    op = fct_opt.maker.fgraph.outputs[0].owner.op
    assert op.destroy_map and 0 in op.destroy_map
    op = fct_opt.maker.fgraph.outputs[1].owner.op
    assert not op.destroy_map or 0 not in op.destroy_map


def test_including():
    mode = Mode(optimizer="merge")
    assert set(mode._optimizer.include) == {"merge"}

    new_mode = mode.including("fast_compile")
    assert set(new_mode._optimizer.include) == {"merge", "fast_compile"}
