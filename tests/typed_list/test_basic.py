import numpy as np
import pytest

import aesara
import aesara.typed_list
from aesara import sparse
from aesara.tensor.type import (
    TensorType,
    integer_dtypes,
    matrix,
    scalar,
    tensor3,
    vector,
)
from aesara.tensor.type_other import SliceType
from aesara.tensor.var import TensorVariable
from aesara.typed_list.basic import (
    Append,
    Count,
    Extend,
    GetItem,
    Index,
    Insert,
    Length,
    Remove,
    Reverse,
    make_list,
)
from aesara.typed_list.type import TypedListType


def rand_ranged_matrix(minimum, maximum, shape):
    return np.asarray(
        np.random.random(shape) * (maximum - minimum) + minimum,
        dtype=aesara.config.floatX,
    )


def random_lil(shape, dtype, nnz):
    sp = pytest.importorskip("scipy")
    rval = sp.sparse.lil_matrix(shape, dtype=dtype)
    huge = 2**30
    for k in range(nnz):
        # set non-zeros in random locations (row x, col y)
        idx = np.random.default_rng().integers(1, huge + 1, size=2) % shape
        value = np.random.random()
        # if dtype *int*, value will always be zeros!
        if dtype in integer_dtypes:
            value = int(value * 100)
        # The call to tuple is needed as scipy 0.13.1 do not support
        # ndarray with length 2 as idx tuple.
        rval.__setitem__(tuple(idx), value)
    return rval


class TestGetItem:
    def test_sanity_check_slice(self):
        mySymbolicMatricesList = TypedListType(
            TensorType(aesara.config.floatX, shape=(None, None))
        )()

        mySymbolicSlice = SliceType()()

        z = GetItem()(mySymbolicMatricesList, mySymbolicSlice)

        assert not isinstance(z, TensorVariable)

        f = aesara.function([mySymbolicMatricesList, mySymbolicSlice], z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        assert np.array_equal(f([x], slice(0, 1, 1)), [x])

    def test_sanity_check_single(self):
        mySymbolicMatricesList = TypedListType(
            TensorType(aesara.config.floatX, shape=(None, None))
        )()

        mySymbolicScalar = scalar(dtype="int64")

        z = GetItem()(mySymbolicMatricesList, mySymbolicScalar)

        f = aesara.function([mySymbolicMatricesList, mySymbolicScalar], z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        assert np.array_equal(f([x], np.asarray(0, dtype="int64")), x)

    def test_interface(self):
        mySymbolicMatricesList = TypedListType(
            TensorType(aesara.config.floatX, shape=(None, None))
        )()
        mySymbolicScalar = scalar(dtype="int64")

        z = mySymbolicMatricesList[mySymbolicScalar]

        f = aesara.function([mySymbolicMatricesList, mySymbolicScalar], z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        assert np.array_equal(f([x], np.asarray(0, dtype="int64")), x)

        z = mySymbolicMatricesList[0]

        f = aesara.function([mySymbolicMatricesList], z)

        assert np.array_equal(f([x]), x)

    def test_wrong_input(self):
        mySymbolicMatricesList = TypedListType(
            TensorType(aesara.config.floatX, shape=(None, None))
        )()
        mySymbolicMatrix = matrix()

        with pytest.raises(TypeError):
            GetItem()(mySymbolicMatricesList, mySymbolicMatrix)

    def test_constant_input(self):
        mySymbolicMatricesList = TypedListType(
            TensorType(aesara.config.floatX, shape=(None, None))
        )()

        z = GetItem()(mySymbolicMatricesList, 0)

        f = aesara.function([mySymbolicMatricesList], z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        assert np.array_equal(f([x]), x)

        z = GetItem()(mySymbolicMatricesList, slice(0, 1, 1))

        f = aesara.function([mySymbolicMatricesList], z)

        assert np.array_equal(f([x]), [x])


class TestAppend:
    def test_inplace(self):
        mySymbolicMatricesList = TypedListType(
            TensorType(aesara.config.floatX, shape=(None, None))
        )()
        myMatrix = matrix()

        z = Append(True)(mySymbolicMatricesList, myMatrix)

        f = aesara.function([mySymbolicMatricesList, myMatrix], z, accept_inplace=True)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        assert np.array_equal(f([x], y), [x, y])

    def test_sanity_check(self):
        mySymbolicMatricesList = TypedListType(
            TensorType(aesara.config.floatX, shape=(None, None))
        )()
        myMatrix = matrix()

        z = Append()(mySymbolicMatricesList, myMatrix)

        f = aesara.function([mySymbolicMatricesList, myMatrix], z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        assert np.array_equal(f([x], y), [x, y])

    def test_interfaces(self):
        mySymbolicMatricesList = TypedListType(
            TensorType(aesara.config.floatX, shape=(None, None))
        )()
        myMatrix = matrix()

        z = mySymbolicMatricesList.append(myMatrix)

        f = aesara.function([mySymbolicMatricesList, myMatrix], z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        assert np.array_equal(f([x], y), [x, y])


class TestExtend:
    def test_inplace(self):
        mySymbolicMatricesList1 = TypedListType(
            TensorType(aesara.config.floatX, shape=(None, None))
        )()
        mySymbolicMatricesList2 = TypedListType(
            TensorType(aesara.config.floatX, shape=(None, None))
        )()

        z = Extend(True)(mySymbolicMatricesList1, mySymbolicMatricesList2)

        f = aesara.function(
            [mySymbolicMatricesList1, mySymbolicMatricesList2], z, accept_inplace=True
        )

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        assert np.array_equal(f([x], [y]), [x, y])

    def test_sanity_check(self):
        mySymbolicMatricesList1 = TypedListType(
            TensorType(aesara.config.floatX, shape=(None, None))
        )()
        mySymbolicMatricesList2 = TypedListType(
            TensorType(aesara.config.floatX, shape=(None, None))
        )()

        z = Extend()(mySymbolicMatricesList1, mySymbolicMatricesList2)

        f = aesara.function([mySymbolicMatricesList1, mySymbolicMatricesList2], z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        assert np.array_equal(f([x], [y]), [x, y])

    def test_interface(self):
        mySymbolicMatricesList1 = TypedListType(
            TensorType(aesara.config.floatX, shape=(None, None))
        )()
        mySymbolicMatricesList2 = TypedListType(
            TensorType(aesara.config.floatX, shape=(None, None))
        )()

        z = mySymbolicMatricesList1.extend(mySymbolicMatricesList2)

        f = aesara.function([mySymbolicMatricesList1, mySymbolicMatricesList2], z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        assert np.array_equal(f([x], [y]), [x, y])


class TestInsert:
    def test_inplace(self):
        mySymbolicMatricesList = TypedListType(
            TensorType(aesara.config.floatX, shape=(None, None))
        )()
        myMatrix = matrix()
        myScalar = scalar(dtype="int64")

        z = Insert(True)(mySymbolicMatricesList, myScalar, myMatrix)

        f = aesara.function(
            [mySymbolicMatricesList, myScalar, myMatrix], z, accept_inplace=True
        )

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        assert np.array_equal(f([x], np.asarray(1, dtype="int64"), y), [x, y])

    def test_sanity_check(self):
        mySymbolicMatricesList = TypedListType(
            TensorType(aesara.config.floatX, shape=(None, None))
        )()
        myMatrix = matrix()
        myScalar = scalar(dtype="int64")

        z = Insert()(mySymbolicMatricesList, myScalar, myMatrix)

        f = aesara.function([mySymbolicMatricesList, myScalar, myMatrix], z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        assert np.array_equal(f([x], np.asarray(1, dtype="int64"), y), [x, y])

    def test_interface(self):
        mySymbolicMatricesList = TypedListType(
            TensorType(aesara.config.floatX, shape=(None, None))
        )()
        myMatrix = matrix()
        myScalar = scalar(dtype="int64")

        z = mySymbolicMatricesList.insert(myScalar, myMatrix)

        f = aesara.function([mySymbolicMatricesList, myScalar, myMatrix], z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        assert np.array_equal(f([x], np.asarray(1, dtype="int64"), y), [x, y])


class TestRemove:
    def test_inplace(self):
        mySymbolicMatricesList = TypedListType(
            TensorType(aesara.config.floatX, shape=(None, None))
        )()
        myMatrix = matrix()

        z = Remove(True)(mySymbolicMatricesList, myMatrix)

        f = aesara.function([mySymbolicMatricesList, myMatrix], z, accept_inplace=True)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        assert np.array_equal(f([x, y], y), [x])

    def test_sanity_check(self):
        mySymbolicMatricesList = TypedListType(
            TensorType(aesara.config.floatX, shape=(None, None))
        )()
        myMatrix = matrix()

        z = Remove()(mySymbolicMatricesList, myMatrix)

        f = aesara.function([mySymbolicMatricesList, myMatrix], z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        assert np.array_equal(f([x, y], y), [x])

    def test_interface(self):
        mySymbolicMatricesList = TypedListType(
            TensorType(aesara.config.floatX, shape=(None, None))
        )()
        myMatrix = matrix()

        z = mySymbolicMatricesList.remove(myMatrix)

        f = aesara.function([mySymbolicMatricesList, myMatrix], z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        assert np.array_equal(f([x, y], y), [x])


class TestReverse:
    def test_inplace(self):
        mySymbolicMatricesList = TypedListType(
            TensorType(aesara.config.floatX, shape=(None, None))
        )()

        z = Reverse(True)(mySymbolicMatricesList)

        f = aesara.function([mySymbolicMatricesList], z, accept_inplace=True)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        assert np.array_equal(f([x, y]), [y, x])

    def test_sanity_check(self):
        mySymbolicMatricesList = TypedListType(
            TensorType(aesara.config.floatX, shape=(None, None))
        )()

        z = Reverse()(mySymbolicMatricesList)

        f = aesara.function([mySymbolicMatricesList], z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        assert np.array_equal(f([x, y]), [y, x])

    def test_interface(self):
        mySymbolicMatricesList = TypedListType(
            TensorType(aesara.config.floatX, shape=(None, None))
        )()

        z = mySymbolicMatricesList.reverse()

        f = aesara.function([mySymbolicMatricesList], z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        assert np.array_equal(f([x, y]), [y, x])


class TestIndex:
    def test_sanity_check(self):
        mySymbolicMatricesList = TypedListType(
            TensorType(aesara.config.floatX, shape=(None, None))
        )()
        myMatrix = matrix()

        z = Index()(mySymbolicMatricesList, myMatrix)

        f = aesara.function([mySymbolicMatricesList, myMatrix], z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        assert f([x, y], y) == 1

    def test_interface(self):
        mySymbolicMatricesList = TypedListType(
            TensorType(aesara.config.floatX, shape=(None, None))
        )()
        myMatrix = matrix()

        z = mySymbolicMatricesList.ind(myMatrix)

        f = aesara.function([mySymbolicMatricesList, myMatrix], z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        assert f([x, y], y) == 1

    def test_non_tensor_type(self):
        mySymbolicNestedMatricesList = TypedListType(
            TensorType(aesara.config.floatX, shape=(None, None)), 1
        )()
        mySymbolicMatricesList = TypedListType(
            TensorType(aesara.config.floatX, shape=(None, None))
        )()

        z = Index()(mySymbolicNestedMatricesList, mySymbolicMatricesList)

        f = aesara.function([mySymbolicNestedMatricesList, mySymbolicMatricesList], z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        assert f([[x, y], [x, y, y]], [x, y]) == 0

    def test_sparse(self):
        sp = pytest.importorskip("scipy")
        mySymbolicSparseList = TypedListType(
            sparse.SparseTensorType("csr", aesara.config.floatX)
        )()
        mySymbolicSparse = sparse.csr_matrix()

        z = Index()(mySymbolicSparseList, mySymbolicSparse)

        f = aesara.function([mySymbolicSparseList, mySymbolicSparse], z)

        x = sp.sparse.csr_matrix(random_lil((10, 40), aesara.config.floatX, 3))
        y = sp.sparse.csr_matrix(random_lil((10, 40), aesara.config.floatX, 3))

        assert f([x, y], y) == 1


class TestCount:
    def test_sanity_check(self):
        mySymbolicMatricesList = TypedListType(
            TensorType(aesara.config.floatX, shape=(None, None))
        )()
        myMatrix = matrix()

        z = Count()(mySymbolicMatricesList, myMatrix)

        f = aesara.function([mySymbolicMatricesList, myMatrix], z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        assert f([y, y, x, y], y) == 3

    def test_interface(self):
        mySymbolicMatricesList = TypedListType(
            TensorType(aesara.config.floatX, shape=(None, None))
        )()
        myMatrix = matrix()

        z = mySymbolicMatricesList.count(myMatrix)

        f = aesara.function([mySymbolicMatricesList, myMatrix], z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        assert f([x, y], y) == 1

    def test_non_tensor_type(self):
        mySymbolicNestedMatricesList = TypedListType(
            TensorType(aesara.config.floatX, shape=(None, None)), 1
        )()
        mySymbolicMatricesList = TypedListType(
            TensorType(aesara.config.floatX, shape=(None, None))
        )()

        z = Count()(mySymbolicNestedMatricesList, mySymbolicMatricesList)

        f = aesara.function([mySymbolicNestedMatricesList, mySymbolicMatricesList], z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        assert f([[x, y], [x, y, y]], [x, y]) == 1

    def test_sparse(self):
        sp = pytest.importorskip("scipy")
        mySymbolicSparseList = TypedListType(
            sparse.SparseTensorType("csr", aesara.config.floatX)
        )()
        mySymbolicSparse = sparse.csr_matrix()

        z = Count()(mySymbolicSparseList, mySymbolicSparse)

        f = aesara.function([mySymbolicSparseList, mySymbolicSparse], z)

        x = sp.sparse.csr_matrix(random_lil((10, 40), aesara.config.floatX, 3))
        y = sp.sparse.csr_matrix(random_lil((10, 40), aesara.config.floatX, 3))

        assert f([x, y, y], y) == 2


class TestLength:
    def test_sanity_check(self):
        mySymbolicMatricesList = TypedListType(
            TensorType(aesara.config.floatX, shape=(None, None))
        )()

        z = Length()(mySymbolicMatricesList)

        f = aesara.function([mySymbolicMatricesList], z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        assert f([x, x, x, x]) == 4

    def test_interface(self):
        mySymbolicMatricesList = TypedListType(
            TensorType(aesara.config.floatX, shape=(None, None))
        )()
        z = mySymbolicMatricesList.__len__()

        f = aesara.function([mySymbolicMatricesList], z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        assert f([x, x]) == 2


class TestMakeList:
    def test_wrong_shape(self):
        a = vector()
        b = matrix()

        with pytest.raises(TypeError):
            make_list((a, b))

    def test_correct_answer(self):
        a = matrix()
        b = matrix()

        x = tensor3()
        y = tensor3()

        A = np.cast[aesara.config.floatX](np.random.random((5, 3)))
        B = np.cast[aesara.config.floatX](np.random.random((7, 2)))
        X = np.cast[aesara.config.floatX](np.random.random((5, 6, 1)))
        Y = np.cast[aesara.config.floatX](np.random.random((1, 9, 3)))

        make_list((3.0, 4.0))
        c = make_list((a, b))
        z = make_list((x, y))
        fc = aesara.function([a, b], c)
        fz = aesara.function([x, y], z)
        for m, n in zip(fc(A, B), [A, B]):
            assert (m == n).all()
        for m, n in zip(fz(X, Y), [X, Y]):
            assert (m == n).all()
