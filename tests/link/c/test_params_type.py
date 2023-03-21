import numpy as np
import pytest

import aesara
from aesara import tensor as at
from aesara.graph.basic import Apply
from aesara.link.c.op import COp, ExternalCOp
from aesara.link.c.params_type import Params, ParamsType
from aesara.link.c.type import EnumList, Generic
from aesara.scalar import ScalarType
from aesara.tensor.type import TensorType, matrix
from tests import unittest_tools as utt


tensor_type_0d = TensorType("float64", shape=tuple())
scalar_type = ScalarType("float64")
generic_type = Generic()


# A test op to compute `y = a*x^2 + bx + c` for any tensor x, with a, b, c as op params.
class QuadraticOpFunc(COp):
    __props__ = ("a", "b", "c")
    params_type = ParamsType(a=tensor_type_0d, b=scalar_type, c=generic_type)

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def make_node(self, x):
        x = at.as_tensor_variable(x)
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage, coefficients):
        x = inputs[0]
        y = output_storage[0]
        y[0] = coefficients.a * (x**2) + coefficients.b * x + coefficients.c

    def c_code_cache_version(self):
        return (1, 5)

    def c_support_code_apply(self, node, name):
        float_type = node.inputs[0].type.dtype_specs()[1]
        return """
        /* Computes: x = a*x*x + b*x + c for x in tensor. */
        int quadratic_{name}(PyArrayObject* tensor, {float_type} a, {float_type} b, {float_type} c) {{
            NpyIter* iterator = NpyIter_New(tensor,
                NPY_ITER_READWRITE | NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK,
                NPY_KEEPORDER, NPY_NO_CASTING, NULL);
            if(iterator == NULL) {{
                PyErr_SetString(PyExc_RuntimeError, "Unable to iterate over a tensor for an elemwise operation.");
                return -1;
            }}
            NpyIter_IterNextFunc* get_next = NpyIter_GetIterNext(iterator, NULL);
            char** data_ptr = NpyIter_GetDataPtrArray(iterator);
            npy_intp* stride_ptr = NpyIter_GetInnerStrideArray(iterator);
            npy_intp* innersize_ptr = NpyIter_GetInnerLoopSizePtr(iterator);
            do {{
                char* data = *data_ptr;
                npy_intp stride = *stride_ptr;
                npy_intp count = *innersize_ptr;
                while(count) {{
                    {float_type} x = *(({float_type}*)data);
                    *(({float_type}*)data) = a*x*x + b*x + c;
                    data += stride;
                    --count;
                }}
            }} while(get_next(iterator));
            NpyIter_Deallocate(iterator);
            return 0;
        }}
        """.format(
            name=name,
            float_type=float_type,
        )

    def c_code(self, node, name, inputs, outputs, sub):
        return """
        %(float_type)s a = (%(float_type)s) (*(npy_float64*) PyArray_GETPTR1(%(coeff)s->a, 0)); // 0-D TensorType.
        %(float_type)s b =                                                   %(coeff)s->b;      // ScalarType.
        %(float_type)s c =                 (%(float_type)s) PyFloat_AsDouble(%(coeff)s->c);     // Generic.
        Py_XDECREF(%(Y)s);
        %(Y)s = (PyArrayObject*)PyArray_EMPTY(PyArray_NDIM(%(X)s), PyArray_DIMS(%(X)s), PyArray_TYPE(%(X)s), PyArray_IS_F_CONTIGUOUS(%(X)s));
        if (PyArray_CopyInto(%(Y)s, %(X)s) != 0) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to copy input into output.");
            %(fail)s
        };
        if (quadratic_%(name)s(%(Y)s, a, b, c) != 0) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to compute quadratic function.");
            %(fail)s
        }
        """ % dict(
            name=name,
            coeff=sub["params"],
            fail=sub["fail"],
            X=inputs[0],
            Y=outputs[0],
            float_type=node.inputs[0].type.c_element_type(),
        )


# Same op as above, but implemented as a ExternalCOp (with C code in an
# external file).
class QuadraticCOpFunc(ExternalCOp):
    __props__ = ("a", "b", "c")
    params_type = ParamsType(a=tensor_type_0d, b=scalar_type, c=generic_type)

    def __init__(self, a, b, c):
        super().__init__(
            "c_code/test_quadratic_function.c", "APPLY_SPECIFIC(compute_quadratic)"
        )
        self.a = a
        self.b = b
        self.c = c

    def make_node(self, x):
        x = at.as_tensor_variable(x)
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage, coefficients):
        x = inputs[0]
        y = output_storage[0]
        y[0] = coefficients.a * (x**2) + coefficients.b * x + coefficients.c


class TestParamsType:
    def test_hash_and_eq_params(self):
        wp1 = ParamsType(
            a=Generic(),
            array=TensorType("int64", shape=(None,)),
            floatting=ScalarType("float64"),
            npy_scalar=TensorType("float64", shape=tuple()),
        )
        wp2 = ParamsType(
            a=Generic(),
            array=TensorType("int64", shape=(None,)),
            floatting=ScalarType("float64"),
            npy_scalar=TensorType("float64", shape=tuple()),
        )
        w1 = Params(
            wp1,
            a=1,
            array=np.asarray([1, 2, 4, 5, 7]),
            floatting=-4.5,
            npy_scalar=np.asarray(12),
        )
        w2 = Params(
            wp2,
            a=1,
            array=np.asarray([1, 2, 4, 5, 7]),
            floatting=-4.5,
            npy_scalar=np.asarray(12),
        )
        assert w1 == w2
        assert not (w1 != w2)
        assert hash(w1) == hash(w2)
        # Changing attributes names only (a -> other_name).
        wp2_other = ParamsType(
            other_name=Generic(),
            array=TensorType("int64", shape=(None,)),
            floatting=ScalarType("float64"),
            npy_scalar=TensorType("float64", shape=tuple()),
        )
        w2 = Params(
            wp2_other,
            other_name=1,
            array=np.asarray([1, 2, 4, 5, 7]),
            floatting=-4.5,
            npy_scalar=np.asarray(12),
        )
        assert w1 != w2
        # Changing attributes values only (now a=2).
        w2 = Params(
            wp2,
            a=2,
            array=np.asarray([1, 2, 4, 5, 7]),
            floatting=-4.5,
            npy_scalar=np.asarray(12),
        )
        assert w1 != w2
        # Changing NumPy array values (5 -> -5).
        w2 = Params(
            wp2,
            a=1,
            array=np.asarray([1, 2, 4, -5, 7]),
            floatting=-4.5,
            npy_scalar=np.asarray(12),
        )
        assert w1 != w2

    def test_hash_and_eq_params_type(self):
        w1 = ParamsType(
            a1=TensorType("int64", shape=(None, None)),
            a2=TensorType("int64", shape=(None, 1, None, None, 1)),
            a3=Generic(),
        )
        w2 = ParamsType(
            a1=TensorType("int64", shape=(None, None)),
            a2=TensorType("int64", shape=(None, 1, None, None, 1)),
            a3=Generic(),
        )
        assert w1 == w2
        assert not (w1 != w2)
        assert hash(w1) == hash(w2)
        assert w1.name == w2.name
        # Changing attributes names only.
        w2 = ParamsType(
            a1=TensorType("int64", shape=(None, None)),
            other_name=TensorType(
                "int64", shape=(None, 1, None, None, 1)
            ),  # a2 -> other_name
            a3=Generic(),
        )
        assert w1 != w2
        # Changing attributes types only.
        w2 = ParamsType(
            a1=TensorType("int64", shape=(None, None)),
            a2=Generic(),  # changing class
            a3=Generic(),
        )
        assert w1 != w2
        # Changing attributes types characteristics only.
        w2 = ParamsType(
            a1=TensorType("int64", shape=(None, 1)),  # changing broadcasting
            a2=TensorType("int64", shape=(None, 1, None, None, 1)),
            a3=Generic(),
        )
        assert w1 != w2

    def test_params_type_filtering(self):
        shape_tensor5 = (1, 2, 2, 3, 2)
        size_tensor5 = (
            shape_tensor5[0]
            * shape_tensor5[1]
            * shape_tensor5[2]
            * shape_tensor5[3]
            * shape_tensor5[4]
        )
        random_tensor = np.random.normal(size=size_tensor5).reshape(shape_tensor5)

        w = ParamsType(
            a1=TensorType("int32", shape=(None, None)),
            a2=TensorType("float64", shape=(None, None, None, None, None)),
            a3=Generic(),
        )

        # With a value that does not match the params type.
        o = Params(
            w,
            a1=np.asarray([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]).astype("int64"),
            a2=random_tensor.astype("float32"),
            a3=2000,
        )
        # should fail (o.a1 is not int32, o.a2 is not float64)
        with pytest.raises(TypeError):
            w.filter(o, True)
        # should fail (o.a1 is not int32, o.a2 is not float64, and downcast is disallowed)
        with pytest.raises(TypeError):
            w.filter(o, False, False)
        # Should pass.
        w.filter(o, strict=False, allow_downcast=True)

        # With a value that matches the params type.
        o1 = Params(
            w,
            a1=np.asarray([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]).astype("int32"),
            a2=random_tensor.astype("float64"),
            a3=2000,
        )
        # All should pass.
        w.filter(o1, strict=True)
        w.filter(o1, strict=False, allow_downcast=False)
        w.filter(o1, strict=False, allow_downcast=True)

        # Check values_eq and values_eq_approx.
        o2 = Params(
            w,
            a1=np.asarray([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]).astype("int32"),
            a2=random_tensor.astype("float64"),
            a3=2000,
        )
        assert w.values_eq(o1, o2)
        assert w.values_eq_approx(o1, o2)

        # Check value_eq_approx.
        # NB: I don't know exactly which kind of differences is rejected by values_eq but accepted by values_eq_approx.
        # So, I just play a little with float values.
        o3 = Params(
            w,
            a1=np.asarray([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]).astype("int32"),
            a2=(random_tensor.astype("float32") * 10 / 2.2 * 2.19999999999 / 10).astype(
                "float64"
            ),
            a3=2000.0 - 0.00000000000000001,
        )
        assert w.values_eq_approx(o1, o3)

    def test_params_type_with_enums(self):
        # Test that we fail if we create a params type with common enum names inside different enum types.
        try:
            ParamsType(enum1=EnumList("A", "B", "C"), enum2=EnumList("A", "B", "F"))
        except AttributeError:
            pass
        else:
            raise Exception(
                "ParamsType should fail with common enum names inside different enum types."
            )

        # Test that we fail if we create a params type with common names in both aliases and constants.
        try:
            ParamsType(
                enum1=EnumList(("A", "a"), ("B", "b")),
                enum2=EnumList(("ONE", "a"), ("TWO", "two")),
            )
        except AttributeError:
            ParamsType(
                enum1=EnumList(("A", "a"), ("B", "b")),
                enum2=EnumList(("ONE", "one"), ("TWO", "two")),
            )
        else:
            raise Exception(
                "ParamsType should fail when there are aliases with same names as some constants."
            )

        # Test that we can access enum values through wrapper directly.
        w = ParamsType(
            enum1=EnumList("A", ("B", "beta"), "C"),
            enum2=EnumList(("D", "delta"), "E", "F"),
        )
        assert w.A == 0 and w.B == 1 and w.C == 2
        assert w.D == 0 and w.E == 1 and w.F == 2
        # Test constants access through aliases.
        assert w.enum_from_alias("beta") == w.B
        assert w.enum_from_alias("delta") == w.D
        assert (
            w.enum_from_alias("C") == w.C
        )  # C is not an alias, so it should return a constant named C.
        # Test that other regular wrapper attributes are still available.
        assert len(w.fields) == len(w.types) == w.length
        assert w.name

    def test_op_params(self):
        a, b, c = 2, 3, -7
        x = matrix(dtype="float64")
        y1 = QuadraticOpFunc(a, b, c)(x)
        y2 = QuadraticCOpFunc(a, b, c)(x)
        f1 = aesara.function([x], y1)
        f2 = aesara.function([x], y2)
        shape = (100, 100)
        vx = (
            np.random.normal(size=shape[0] * shape[1]).astype("float64").reshape(*shape)
        )
        vy1 = f1(vx)
        vy2 = f2(vx)
        ref = a * (vx**2) + b * vx + c
        utt.assert_allclose(vy1, vy2)
        utt.assert_allclose(ref, vy1)
