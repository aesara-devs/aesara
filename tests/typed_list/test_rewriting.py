import numpy as np

import aesara
import aesara.tensor as at
import aesara.typed_list
from aesara.compile.io import In
from aesara.tensor.type import TensorType, matrix, scalar
from aesara.typed_list.basic import Append, Extend, Insert, Remove, Reverse
from aesara.typed_list.type import TypedListType
from tests.tensor.utils import random_ranged


class TestInplace:
    def test_reverse_inplace(self):
        mySymbolicMatricesList = TypedListType(
            TensorType(aesara.config.floatX, shape=(None, None))
        )()

        z = Reverse()(mySymbolicMatricesList)
        m = aesara.compile.mode.get_default_mode().including(
            "typed_list_inplace_rewrite"
        )
        f = aesara.function(
            [In(mySymbolicMatricesList, borrow=True, mutable=True)],
            z,
            accept_inplace=True,
            mode=m,
        )
        assert f.maker.fgraph.toposort()[0].op.inplace

        x = random_ranged(-1000, 1000, [100, 101])

        y = random_ranged(-1000, 1000, [100, 101])

        assert np.array_equal(f([x, y]), [y, x])

    def test_append_inplace(self):
        mySymbolicMatricesList = TypedListType(
            TensorType(aesara.config.floatX, shape=(None, None))
        )()
        mySymbolicMatrix = matrix()
        z = Append()(mySymbolicMatricesList, mySymbolicMatrix)
        m = aesara.compile.mode.get_default_mode().including(
            "typed_list_inplace_rewrite"
        )
        f = aesara.function(
            [
                In(mySymbolicMatricesList, borrow=True, mutable=True),
                In(mySymbolicMatrix, borrow=True, mutable=True),
            ],
            z,
            accept_inplace=True,
            mode=m,
        )
        assert f.maker.fgraph.toposort()[0].op.inplace

        x = random_ranged(-1000, 1000, [100, 101])

        y = random_ranged(-1000, 1000, [100, 101])

        assert np.array_equal(f([x], y), [x, y])

    def test_extend_inplace(self):
        mySymbolicMatricesList1 = TypedListType(
            TensorType(aesara.config.floatX, shape=(None, None))
        )()

        mySymbolicMatricesList2 = TypedListType(
            TensorType(aesara.config.floatX, shape=(None, None))
        )()

        z = Extend()(mySymbolicMatricesList1, mySymbolicMatricesList2)
        m = aesara.compile.mode.get_default_mode().including(
            "typed_list_inplace_rewrite"
        )
        f = aesara.function(
            [
                In(mySymbolicMatricesList1, borrow=True, mutable=True),
                mySymbolicMatricesList2,
            ],
            z,
            mode=m,
        )
        assert f.maker.fgraph.toposort()[0].op.inplace

        x = random_ranged(-1000, 1000, [100, 101])

        y = random_ranged(-1000, 1000, [100, 101])

        assert np.array_equal(f([x], [y]), [x, y])

    def test_insert_inplace(self):
        mySymbolicMatricesList = TypedListType(
            TensorType(aesara.config.floatX, shape=(None, None))
        )()
        mySymbolicIndex = scalar(dtype="int64")
        mySymbolicMatrix = matrix()

        z = Insert()(mySymbolicMatricesList, mySymbolicIndex, mySymbolicMatrix)
        m = aesara.compile.mode.get_default_mode().including(
            "typed_list_inplace_rewrite"
        )

        f = aesara.function(
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

        x = random_ranged(-1000, 1000, [100, 101])

        y = random_ranged(-1000, 1000, [100, 101])

        assert np.array_equal(f([x], np.asarray(1, dtype="int64"), y), [x, y])

    def test_remove_inplace(self):
        mySymbolicMatricesList = TypedListType(
            TensorType(aesara.config.floatX, shape=(None, None))
        )()
        mySymbolicMatrix = matrix()
        z = Remove()(mySymbolicMatricesList, mySymbolicMatrix)
        m = aesara.compile.mode.get_default_mode().including(
            "typed_list_inplace_rewrite"
        )
        f = aesara.function(
            [
                In(mySymbolicMatricesList, borrow=True, mutable=True),
                In(mySymbolicMatrix, borrow=True, mutable=True),
            ],
            z,
            accept_inplace=True,
            mode=m,
        )
        assert f.maker.fgraph.toposort()[0].op.inplace

        x = random_ranged(-1000, 1000, [100, 101])

        y = random_ranged(-1000, 1000, [100, 101])

        assert np.array_equal(f([x, y], y), [x])


def test_constant_folding():
    m = at.ones((1,), dtype="int8")
    l = aesara.typed_list.make_list([m, m])
    f = aesara.function([], l)
    topo = f.maker.fgraph.toposort()
    assert len(topo)
    assert isinstance(topo[0].op, aesara.compile.ops.DeepCopyOp)
    assert f() == [1, 1]
