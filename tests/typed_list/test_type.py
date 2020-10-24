import numpy as np
import pytest

import theano
import theano.tensor as tt
from tests import unittest_tools as utt
from tests.tensor.utils import rand_ranged
from theano.typed_list.basic import TypedListVariable
from theano.typed_list.type import TypedListType


class TestTypedListType:
    def setup_method(self):
        utt.seed_rng()

    def test_wrong_input_on_creation(self):
        # Typed list type should raises an
        # error if the argument passed for
        # type is not a valid theano type

        with pytest.raises(TypeError):
            TypedListType(None)

    def test_wrong_input_on_filter(self):
        # Typed list type should raises an
        # error if the argument given to filter
        # isn't of the same type as the one
        # specified on creation

        # list of matrices
        myType = TypedListType(tt.TensorType(theano.config.floatX, (False, False)))

        with pytest.raises(TypeError):
            myType.filter([4])

    def test_not_a_list_on_filter(self):
        # Typed List Value should raises an error
        # if no iterable variable is given on input

        # list of matrices
        myType = TypedListType(tt.TensorType(theano.config.floatX, (False, False)))

        with pytest.raises(TypeError):
            myType.filter(4)

    def test_type_equality(self):
        # Typed list types should only be equal
        # when they contains the same theano
        # variables

        # list of matrices
        myType1 = TypedListType(tt.TensorType(theano.config.floatX, (False, False)))
        # list of matrices
        myType2 = TypedListType(tt.TensorType(theano.config.floatX, (False, False)))
        # list of scalars
        myType3 = TypedListType(tt.TensorType(theano.config.floatX, ()))

        assert myType2 == myType1
        assert myType3 != myType1

    def test_filter_sanity_check(self):
        # Simple test on typed list type filter

        myType = TypedListType(tt.TensorType(theano.config.floatX, (False, False)))

        x = rand_ranged(-1000, 1000, [100, 100])

        assert np.array_equal(myType.filter([x]), [x])

    def test_intern_filter(self):
        # Test checking if values contained are themselves
        # filtered. If they weren't this code would raise
        # an exception.

        myType = TypedListType(tt.TensorType("float64", (False, False)))

        x = np.asarray([[4, 5], [4, 5]], dtype="float32")

        assert np.array_equal(myType.filter([x]), [x])

    def test_load_alot(self):
        myType = TypedListType(tt.TensorType(theano.config.floatX, (False, False)))

        x = rand_ranged(-1000, 1000, [10, 10])
        testList = []
        for i in range(10000):
            testList.append(x)

        assert np.array_equal(myType.filter(testList), testList)

    def test_basic_nested_list(self):
        # Testing nested list with one level of depth

        myNestedType = TypedListType(
            tt.TensorType(theano.config.floatX, (False, False))
        )

        myType = TypedListType(myNestedType)

        x = rand_ranged(-1000, 1000, [100, 100])

        assert np.array_equal(myType.filter([[x]]), [[x]])

    def test_comparison_different_depth(self):
        # Nested list with different depth aren't the same

        myNestedType = TypedListType(
            tt.TensorType(theano.config.floatX, (False, False))
        )

        myNestedType2 = TypedListType(myNestedType)

        myNestedType3 = TypedListType(myNestedType2)

        assert myNestedType2 != myNestedType3

    def test_nested_list_arg(self):
        # test for the 'depth' optionnal argument

        myNestedType = TypedListType(
            tt.TensorType(theano.config.floatX, (False, False)), 3
        )

        myType = TypedListType(tt.TensorType(theano.config.floatX, (False, False)))

        myManualNestedType = TypedListType(TypedListType(TypedListType(myType)))

        assert myNestedType == myManualNestedType

    def test_get_depth(self):
        # test case for get_depth utilitary function

        myType = TypedListType(tt.TensorType(theano.config.floatX, (False, False)))

        myManualNestedType = TypedListType(TypedListType(TypedListType(myType)))

        assert myManualNestedType.get_depth() == 3

    def test_comparison_uneven_nested(self):
        # test for comparison between uneven nested list

        myType = TypedListType(tt.TensorType(theano.config.floatX, (False, False)))

        myManualNestedType1 = TypedListType(TypedListType(TypedListType(myType)))

        myManualNestedType2 = TypedListType(TypedListType(myType))

        assert myManualNestedType1 != myManualNestedType2
        assert myManualNestedType2 != myManualNestedType1

    def test_variable_is_Typed_List_variable(self):
        mySymbolicVariable = TypedListType(
            tt.TensorType(theano.config.floatX, (False, False))
        )()

        assert isinstance(mySymbolicVariable, TypedListVariable)
