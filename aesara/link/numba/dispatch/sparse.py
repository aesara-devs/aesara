import scipy as sp
import scipy.sparse
from numba.core import cgutils, types
from numba.extending import (
    NativeValue,
    box,
    make_attribute_wrapper,
    models,
    register_model,
    typeof_impl,
    unbox,
)


class CSMatrixType(types.Type):
    """A Numba `Type` modeled after the base class `scipy.sparse.compressed._cs_matrix`."""

    name: str
    instance_class: type

    def __init__(self, dtype):
        self.dtype = dtype
        self.data = types.Array(dtype, 1, "A")
        self.indices = types.Array(types.int32, 1, "A")
        self.indptr = types.Array(types.int32, 1, "A")
        self.shape = types.UniTuple(types.int64, 2)
        super().__init__(self.name)


make_attribute_wrapper(CSMatrixType, "data", "data")
make_attribute_wrapper(CSMatrixType, "indices", "indices")
make_attribute_wrapper(CSMatrixType, "indptr", "indptr")
make_attribute_wrapper(CSMatrixType, "shape", "shape")


class CSRMatrixType(CSMatrixType):
    name = "csr_matrix"

    @staticmethod
    def instance_class(data, indices, indptr, shape):
        return sp.sparse.csr_matrix((data, indices, indptr), shape, copy=False)


class CSCMatrixType(CSMatrixType):
    name = "csc_matrix"

    @staticmethod
    def instance_class(data, indices, indptr, shape):
        return sp.sparse.csc_matrix((data, indices, indptr), shape, copy=False)


@typeof_impl.register(sp.sparse.csc_matrix)
def typeof_csc_matrix(val, c):
    data = typeof_impl(val.data, c)
    return CSCMatrixType(data.dtype)


@typeof_impl.register(sp.sparse.csr_matrix)
def typeof_csr_matrix(val, c):
    data = typeof_impl(val.data, c)
    return CSRMatrixType(data.dtype)


@register_model(CSRMatrixType)
class CSRMatrixModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("data", fe_type.data),
            ("indices", fe_type.indices),
            ("indptr", fe_type.indptr),
            ("shape", fe_type.shape),
        ]
        super().__init__(dmm, fe_type, members)


@register_model(CSCMatrixType)
class CSCMatrixModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("data", fe_type.data),
            ("indices", fe_type.indices),
            ("indptr", fe_type.indptr),
            ("shape", fe_type.shape),
        ]
        super().__init__(dmm, fe_type, members)


@unbox(CSCMatrixType)
@unbox(CSRMatrixType)
def unbox_matrix(typ, obj, c):

    struct_ptr = cgutils.create_struct_proxy(typ)(c.context, c.builder)

    data = c.pyapi.object_getattr_string(obj, "data")
    indices = c.pyapi.object_getattr_string(obj, "indices")
    indptr = c.pyapi.object_getattr_string(obj, "indptr")
    shape = c.pyapi.object_getattr_string(obj, "shape")

    struct_ptr.data = c.unbox(typ.data, data).value
    struct_ptr.indices = c.unbox(typ.indices, indices).value
    struct_ptr.indptr = c.unbox(typ.indptr, indptr).value
    struct_ptr.shape = c.unbox(typ.shape, shape).value

    c.pyapi.decref(data)
    c.pyapi.decref(indices)
    c.pyapi.decref(indptr)
    c.pyapi.decref(shape)

    is_error_ptr = cgutils.alloca_once_value(c.builder, cgutils.false_bit)
    is_error = c.builder.load(is_error_ptr)

    res = NativeValue(struct_ptr._getvalue(), is_error=is_error)

    return res


@box(CSCMatrixType)
@box(CSRMatrixType)
def box_matrix(typ, val, c):
    struct_ptr = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)

    data_obj = c.box(typ.data, struct_ptr.data)
    indices_obj = c.box(typ.indices, struct_ptr.indices)
    indptr_obj = c.box(typ.indptr, struct_ptr.indptr)
    shape_obj = c.box(typ.shape, struct_ptr.shape)

    c.pyapi.incref(data_obj)
    c.pyapi.incref(indices_obj)
    c.pyapi.incref(indptr_obj)
    c.pyapi.incref(shape_obj)

    cls_obj = c.pyapi.unserialize(c.pyapi.serialize_object(typ.instance_class))
    obj = c.pyapi.call_function_objargs(
        cls_obj, (data_obj, indices_obj, indptr_obj, shape_obj)
    )

    c.pyapi.decref(data_obj)
    c.pyapi.decref(indices_obj)
    c.pyapi.decref(indptr_obj)
    c.pyapi.decref(shape_obj)

    return obj
