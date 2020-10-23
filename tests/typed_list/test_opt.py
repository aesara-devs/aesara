import numpy as np

import theano
import theano.tensor as tt
import theano.typed_list
from tests.tensor.utils import rand_ranged
from theano import In
from theano.typed_list.basic import Append, Extend, Insert, Remove, Reverse
from theano.typed_list.type import TypedListType


class TestInplace:
    def test_reverse_inplace(self):
        mySymbolicMatricesList = TypedListType(
            tt.TensorType(theano.config.floatX, (False, False))
        )()

        z = Reverse()(mySymbolicMatricesList)
        m = theano.compile.mode.get_default_mode().including("typed_list_inplace_opt")
        f = theano.function(
            [In(mySymbolicMatricesList, borrow=True, mutable=True)],
            z,
            accept_inplace=True,
            mode=m,
        )
        assert f.maker.fgraph.toposort()[0].op.inplace

        x = rand_ranged(-1000, 1000, [100, 101])

        y = rand_ranged(-1000, 1000, [100, 101])

        assert np.array_equal(f([x, y]), [y, x])

    def test_append_inplace(self):
        mySymbolicMatricesList = TypedListType(
            tt.TensorType(theano.config.floatX, (False, False))
        )()
        mySymbolicMatrix = tt.matrix()
        z = Append()(mySymbolicMatricesList, mySymbolicMatrix)
        m = theano.compile.mode.get_default_mode().including("typed_list_inplace_opt")
        f = theano.function(
            [
                In(mySymbolicMatricesList, borrow=True, mutable=True),
                In(mySymbolicMatrix, borrow=True, mutable=True),
            ],
            z,
            accept_inplace=True,
            mode=m,
        )
        assert f.maker.fgraph.toposort()[0].op.inplace

        x = rand_ranged(-1000, 1000, [100, 101])

        y = rand_ranged(-1000, 1000, [100, 101])

        assert np.array_equal(f([x], y), [x, y])

    def test_extend_inplace(self):
        mySymbolicMatricesList1 = TypedListType(
            tt.TensorType(theano.config.floatX, (False, False))
        )()

        mySymbolicMatricesList2 = TypedListType(
            tt.TensorType(theano.config.floatX, (False, False))
        )()

        z = Extend()(mySymbolicMatricesList1, mySymbolicMatricesList2)
        m = theano.compile.mode.get_default_mode().including("typed_list_inplace_opt")
        f = theano.function(
            [
                In(mySymbolicMatricesList1, borrow=True, mutable=True),
                mySymbolicMatricesList2,
            ],
            z,
            mode=m,
        )
        assert f.maker.fgraph.toposort()[0].op.inplace

        x = rand_ranged(-1000, 1000, [100, 101])

        y = rand_ranged(-1000, 1000, [100, 101])

        assert np.array_equal(f([x], [y]), [x, y])

    def test_insert_inplace(self):
        mySymbolicMatricesList = TypedListType(
            tt.TensorType(theano.config.floatX, (False, False))
        )()
        mySymbolicIndex = tt.scalar(dtype="int64")
        mySymbolicMatrix = tt.matrix()

        z = Insert()(mySymbolicMatricesList, mySymbolicIndex, mySymbolicMatrix)
        m = theano.compile.mode.get_default_mode().including("typed_list_inplace_opt")

        f = theano.function(
            [
                In(mySymbolicMatricesList, borrow=True, mutable=True),
                mySymbolicIndex,
                mySymbolicMatrix,
            ],
            z,
            accept_inplace=True,
            mode=m,
        )
        assert f.maker.fgraph.toposort()[0].op.inplace

        x = rand_ranged(-1000, 1000, [100, 101])

        y = rand_ranged(-1000, 1000, [100, 101])

        assert np.array_equal(f([x], np.asarray(1, dtype="int64"), y), [x, y])

    def test_remove_inplace(self):
        mySymbolicMatricesList = TypedListType(
            tt.TensorType(theano.config.floatX, (False, False))
        )()
        mySymbolicMatrix = tt.matrix()
        z = Remove()(mySymbolicMatricesList, mySymbolicMatrix)
        m = theano.compile.mode.get_default_mode().including("typed_list_inplace_opt")
        f = theano.function(
            [
                In(mySymbolicMatricesList, borrow=True, mutable=True),
                In(mySymbolicMatrix, borrow=True, mutable=True),
            ],
            z,
            accept_inplace=True,
            mode=m,
        )
        assert f.maker.fgraph.toposort()[0].op.inplace

        x = rand_ranged(-1000, 1000, [100, 101])

        y = rand_ranged(-1000, 1000, [100, 101])

        assert np.array_equal(f([x, y], y), [x])


def test_constant_folding():
    m = tt.ones((1,), dtype="int8")
    l = theano.typed_list.make_list([m, m])
    f = theano.function([], l)
    topo = f.maker.fgraph.toposort()
    assert len(topo)
    assert isinstance(topo[0].op, theano.compile.ops.DeepCopyOp)
    assert f() == [1, 1]
