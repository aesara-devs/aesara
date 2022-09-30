import warnings
from textwrap import dedent

import numpy as np
import scipy

from aesara.graph.basic import Apply
from aesara.link.c.op import COp
from aesara.tensor.basic import as_tensor_variable
from aesara.tensor.math import neg, sum


class SoftmaxGrad(COp):
    """
    Gradient wrt x of the Softmax Op.

    """

    nin = 2
    nout = 1
    __props__ = ("axis",)

    def __init__(self, axis):
        if axis is not None and not isinstance(axis, int):
            raise TypeError("axis must be an integer or `None`")
        self.axis = axis

    def make_node(self, dy, sm):
        dy = as_tensor_variable(dy)
        sm = as_tensor_variable(sm)

        if self.axis is not None and (self.axis >= sm.ndim or self.axis < -sm.ndim):
            raise ValueError(
                f"SoftmaxGrad axis(={self.axis}) out of bounds for {sm.ndim}D array {sm}"
            )

        return Apply(self, [dy, sm], [sm.type()])

    def perform(self, node, input_storage, output_storage):
        dy, sm = input_storage

        dy_times_sm = dy * sm
        dx = dy_times_sm - np.sum(dy_times_sm, axis=self.axis, keepdims=True) * sm
        output_storage[0][0] = dx

    def grad(self, inp, grads):
        dy, sm = inp
        (g,) = grads

        tmp = g + neg(sum(g * sm, axis=self.axis, keepdims=True))
        g_dy = tmp * sm

        tmp2 = sum(dy * sm, axis=self.axis, keepdims=True)
        g_sm = tmp * dy - g * tmp2

        return g_dy, g_sm

    def infer_shape(self, fgraph, node, shape):
        return [shape[1]]

    def c_code_cache_version(self):
        return (4,)

    def c_code(self, node, name, inp, out, sub):
        dy, sm = inp
        (dx,) = out
        axis = self.axis if self.axis is not None else np.MAXDIMS
        fail = sub["fail"]

        return dedent(
            f"""
            PyArrayObject* op[3];
            npy_uint32 op_flags[3];
            npy_uint32 iter_flags;
            NpyIter* iter;
            NpyIter_IterNextFunc* get_next;
            char** data_ptr;

            int sm_ndim = PyArray_NDIM({sm});
            int axis = {axis};
            int iterate_axis = !(axis == NPY_MAXDIMS || sm_ndim == 1);

            // Validate inputs
            if ((PyArray_TYPE({dy}) != NPY_DOUBLE) &&
                (PyArray_TYPE({dy}) != NPY_FLOAT))
            {{
                PyErr_SetString(PyExc_TypeError, "types should be float or float64");
                {fail};
            }}
            if ((PyArray_TYPE({sm}) != NPY_DOUBLE) &&
                (PyArray_TYPE({sm}) != NPY_FLOAT))
            {{
                PyErr_SetString(PyExc_TypeError, "types should be float or float64");
                {fail};
            }}

            if (axis < 0) axis = sm_ndim + axis;
            if ((axis < 0) || (iterate_axis && (axis > sm_ndim)))
            {{
                PyErr_SetString(PyExc_ValueError, "invalid axis in SoftmaxGrad");
                {fail};
            }}

            if (({dx} == NULL)
                || !(PyArray_CompareLists(PyArray_DIMS({dx}), PyArray_DIMS({sm}), sm_ndim)))
            {{
                Py_XDECREF({dx});
                {dx} = (PyArrayObject*)PyArray_SimpleNew(sm_ndim,
                                                         PyArray_DIMS({sm}),
                                                         PyArray_TYPE({sm}));
                if (!{dx})
                {{
                    PyErr_SetString(PyExc_MemoryError, "failed to alloc SoftMaxGrad dx output");
                    {fail};
                }}
            }}

            // Create numpy iterator
            op[0] = {dy};
            op[1] = {sm};
            op[2] = {dx};
            op_flags[0] = NPY_ITER_READONLY;
            op_flags[1] = NPY_ITER_READONLY;
            op_flags[2] = NPY_ITER_READWRITE;
            iter_flags = (iterate_axis)? NPY_ITER_MULTI_INDEX : 0;
            iter = NpyIter_MultiNew(
                3,
                op,
                iter_flags,
                NPY_KEEPORDER,
                NPY_NO_CASTING,
                op_flags,
                NULL
            );

            if (iter == NULL)
            {{
                PyErr_SetString(PyExc_MemoryError, "failed to create softmax iterator");
                {fail};
            }}

            // SoftmaxGrad is applied across the entire array
            if (!iterate_axis)
            {{
                get_next = NpyIter_GetIterNext(iter, NULL);
                if (get_next == NULL)
                {{
                    NpyIter_Deallocate(iter);
                    PyErr_SetString(PyExc_RuntimeError, "Failed to obtain SoftMaxGrad GetIterNext");
                    {fail};
                }}
                data_ptr = NpyIter_GetDataPtrArray(iter);

                // Compute and accumulate dy * sm
                dtype_{dx} sum_dy_times_sm = 0.0;
                do
                {{
                    dtype_{dy}* dy_ptr = (dtype_{dy}*)data_ptr[0];
                    dtype_{sm}* sm_ptr = (dtype_{sm}*)data_ptr[1];
                    dtype_{dx}* dx_ptr = (dtype_{dx}*)data_ptr[2];

                    *dx_ptr = (dtype_{dx})((*dy_ptr) * (*sm_ptr));
                    sum_dy_times_sm += *dx_ptr;
                }} while(get_next(iter));

                // Reset Iterator
                if (NpyIter_GotoIterIndex(iter, 0) == NPY_FAIL)
                {{
                    PyErr_SetString(PyExc_RuntimeError, "Failed to reset softmax iterator");
                    {fail};
                }}

                // Subtract sum(dy*sm) * sm
                do
                {{
                    dtype_{sm}* sm_ptr = (dtype_{sm}*)data_ptr[1];
                    dtype_{dx}* dx_ptr = (dtype_{dx}*)data_ptr[2];
                    *dx_ptr -= sum_dy_times_sm * ((dtype_{dx})(*sm_ptr));
                }} while(get_next(iter));
            }}

            // SoftmaxGrad is applied across a specific axis
            else {{
                // Collect axis strides and remove it from iteration
                npy_intp axis_size = PyArray_DIM({sm}, axis);
                npy_intp* axis_stride = NpyIter_GetAxisStrideArray(iter, axis);
                if  (axis_stride == NULL)
                {{
                    PyErr_SetString(PyExc_RuntimeError, "Failed to obtain softmax axis strides");
                    {fail};
                }}
                npy_intp dy_axis_stride = axis_stride[0] / sizeof(dtype_{dy});
                npy_intp sm_axis_stride = axis_stride[1] / sizeof(dtype_{sm});
                npy_intp dx_axis_stride = axis_stride[2] / sizeof(dtype_{dx});

                if (NpyIter_RemoveAxis(iter, axis) == NPY_FAIL)
                {{
                    PyErr_SetString(PyExc_RuntimeError, "Failed to remove SoftmaxGrad axis from iterator");
                    {fail};
                }}

                // Iterate over remaining axes
                get_next = NpyIter_GetIterNext(iter, NULL);
                if (get_next == NULL)
                {{
                    NpyIter_Deallocate(iter);
                    PyErr_SetString(PyExc_RuntimeError, "Failed to obtain SoftamGrad GetIterNext");
                    {fail};
                }}

                data_ptr = NpyIter_GetDataPtrArray(iter);
                do
                {{
                    dtype_{dy}* dy_axis = (dtype_{dy}*)data_ptr[0];
                    dtype_{sm}* sm_axis = (dtype_{sm}*)data_ptr[1];
                    dtype_{dx}* dx_axis = (dtype_{dx}*)data_ptr[2];

                    // Compute and accumulate dy * sm
                    dtype_{dx} sum_dy_times_sm = 0.0;
                    for (npy_intp i = 0; i < axis_size; i++)
                    {{
                        dx_axis[i * dx_axis_stride] = (dtype_{dx})(dy_axis[i * dy_axis_stride] * sm_axis[i * sm_axis_stride]);
                        sum_dy_times_sm += dx_axis[i * dx_axis_stride];
                    }}

                    // Subtract sum(dy*sm) * sm
                    for (npy_intp i = 0; i < axis_size; i++)
                    {{
                        dx_axis[i * dx_axis_stride] -= sum_dy_times_sm * (dtype_{dx})(sm_axis[i * sm_axis_stride]);
                    }}

                }} while(get_next(iter));
            }}
            NpyIter_Deallocate(iter);
            """
        )


class Softmax(COp):
    r"""
    Softmax activation function
    :math:`\\varphi(\\mathbf{x})_j =
    \\frac{e^{\mathbf{x}_j}}{\sum_{k=1}^K e^{\mathbf{x}_k}}`
    where :math:`K` is the total number of neurons in the layer. This
    activation function gets applied row-wise.

    """

    nin = 1
    nout = 1
    __props__ = ("axis",)

    def __init__(self, axis):
        if axis is not None and not isinstance(axis, int):
            raise TypeError("axis must be an integer or `None`")
        self.axis = axis

    def make_node(self, x):
        x = as_tensor_variable(x)

        if self.axis is not None and (self.axis >= x.ndim or self.axis < -x.ndim):
            raise ValueError(
                f"Softmax axis(={self.axis}) out of bounds for {x.ndim}D array {x}"
            )

        return Apply(self, [x], [x.type()])

    def perform(self, node, input_storage, output_storage):
        (x,) = input_storage
        (z,) = output_storage
        z[0] = scipy.special.softmax(x, axis=self.axis)

    def L_op(self, inp, outputs, grads):
        (x,) = inp
        (g_sm,) = grads
        return [SoftmaxGrad(axis=self.axis)(g_sm, outputs[0])]

    def R_op(self, inputs, eval_points):
        # I think the Jacobian is symmetric so the R_op
        # is the same as the grad
        if None in eval_points:
            return [None]
        return self.L_op(inputs, [self(*inputs)], eval_points)

    def infer_shape(self, fgraph, node, shape):
        return shape

    def c_headers(self, **kwargs):
        return ["<iostream>", "<cmath>"]

    def c_code(self, node, name, inp, out, sub):
        (x,) = inp
        (sm,) = out
        axis = self.axis if self.axis is not None else np.MAXDIMS
        fail = sub["fail"]
        # dtype = node.inputs[0].type.dtype_specs()[1]
        # TODO: put this into a templated function, in the support code
        # TODO: declare the max of each row as an Op output
        # TODO: use this to accept float32 and int32: node.inputs[0].type.dtype_specs()[1]
        return dedent(
            f"""
            PyArrayObject* op[2];
            npy_uint32 op_flags[2];
            npy_uint32 iter_flags;
            NpyIter* iter;
            NpyIter_IterNextFunc* get_next;
            char** data_ptr;

            int x_ndim = PyArray_NDIM({x});
            int axis = {axis};
            int iterate_axis = !(axis == NPY_MAXDIMS || x_ndim == 1);

            // Validate inputs
            if ((PyArray_TYPE({x}) != NPY_DOUBLE) &&
                (PyArray_TYPE({x}) != NPY_FLOAT))
            {{
                PyErr_SetString(PyExc_TypeError, "not a float");
                {fail}
            }}

            if (axis < 0) axis = x_ndim + axis;
            if ((axis < 0) || (iterate_axis && (axis > x_ndim)))
            {{
                PyErr_SetString(PyExc_ValueError, "invalid axis in Softmax");
                {fail}
            }}

            // Allocate Output Array
            if (({sm}) == NULL || !(PyArray_CompareLists(PyArray_DIMS({sm}), PyArray_DIMS({x}), x_ndim)))
            {{
                Py_XDECREF({sm});
                {sm} = (PyArrayObject*)PyArray_SimpleNew(x_ndim, PyArray_DIMS({x}), PyArray_TYPE({x}));
                if(!{sm}) {{
                    PyErr_SetString(PyExc_MemoryError, "failed to alloc Softmax output");
                    {fail}
                }}
            }}

            // Create numpy iterator
            op[0] = {x};
            op[1] = {sm};
            op_flags[0] = NPY_ITER_READONLY;
            op_flags[1] = NPY_ITER_READWRITE;
            iter_flags = (iterate_axis)? NPY_ITER_MULTI_INDEX : 0;
            iter = NpyIter_MultiNew(
                2,
                op,
                iter_flags,
                NPY_KEEPORDER,
                NPY_NO_CASTING,
                op_flags,
                NULL
            );

            if (iter == NULL)
            {{
                PyErr_SetString(PyExc_MemoryError, "failed to create Softmax iterator");
                {fail}
            }}

            // Softmax is applied across the entire array
            if (!iterate_axis)
            {{
                get_next = NpyIter_GetIterNext(iter, NULL);
                if (get_next == NULL)
                {{
                    NpyIter_Deallocate(iter);
                    PyErr_SetString(PyExc_RuntimeError, "Failed to obtain Softmax GetIterNext");
                    {fail}
                }}
                data_ptr = NpyIter_GetDataPtrArray(iter);

                // Find axis max
                dtype_{x}* x_ptr = (dtype_{x}*)data_ptr[0];
                dtype_{x} max = *x_ptr;
                if (get_next(iter))
                {{
                    do
                    {{
                        dtype_{x}* x_ptr = (dtype_{x}*)data_ptr[0];
                        max = (*x_ptr > max)? *x_ptr : max;
                    }} while(get_next(iter));
                }}

                // Reset Iterator
                if (NpyIter_GotoIterIndex(iter, 0) == NPY_FAIL)
                {{
                    PyErr_SetString(PyExc_RuntimeError, "Failed to reset Softmax iterator");
                    {fail}
                }}

                // Compute and accumulate exp(x-max(x)) exponent
                double sum_exp_dev = 0.0;
                do
                {{
                    dtype_{x}* x_ptr = (dtype_{x}*)data_ptr[0];
                    dtype_{sm}* sm_ptr = (dtype_{sm}*)data_ptr[1];
                    *sm_ptr = (dtype_{sm}) exp(*x_ptr - max);
                    sum_exp_dev += *sm_ptr;
                }} while(get_next(iter));

                // Reset Iterator
                if (NpyIter_GotoIterIndex(iter, 0) == NPY_FAIL)
                {{
                    PyErr_SetString(PyExc_RuntimeError, "Failed to reset Softmax iterator");
                    {fail}
                }}

                // Divide by sum(exp(x-max(x)))
                double inv_sum_exp_dev = 1.0 / sum_exp_dev;
                do
                {{
                    dtype_{sm}* sm_ptr = (dtype_{sm}*)data_ptr[1];
                    *sm_ptr *= inv_sum_exp_dev;
                }} while(get_next(iter));
            }}

            // Softmax is applied across a specific axis
            else {{
                // Collect axis strides and remove it from iteration
                npy_intp axis_size = PyArray_DIM({x}, axis);
                npy_intp* axis_stride = NpyIter_GetAxisStrideArray(iter, axis);
                if  (axis_stride == NULL)
                {{
                    PyErr_SetString(PyExc_RuntimeError, "Failed to obtain Softmax axis strides");
                    {fail}
                }}
                npy_intp x_axis_stride = axis_stride[0] / sizeof(dtype_{x});
                npy_intp sm_axis_stride = axis_stride[1] / sizeof(dtype_{sm});

                if (NpyIter_RemoveAxis(iter, axis) == NPY_FAIL)
                {{
                    PyErr_SetString(PyExc_RuntimeError, "Failed to remove softmax axis from iterator");
                    {fail}
                }}

                // Iterate over remaining axes
                get_next = NpyIter_GetIterNext(iter, NULL);
                if (get_next == NULL)
                {{
                    NpyIter_Deallocate(iter);
                    PyErr_SetString(PyExc_RuntimeError, "Failed to obtain softmax GetIterNext");
                    {fail}
                }}

                data_ptr = NpyIter_GetDataPtrArray(iter);
                do
                {{
                    dtype_{x}* x_axis = (dtype_{x}*)data_ptr[0];
                    dtype_{sm}* sm_axis = (dtype_{sm}*)data_ptr[1];

                    // Find axis max
                    dtype_{x} max = x_axis[0];
                    for (npy_intp i = 1; i < axis_size; i++)
                    {{
                        dtype_{x} x_val = x_axis[i * x_axis_stride];
                        max = (x_val > max)? x_val : max;
                    }}

                    // Compute and accumulate exp(x-max(x)) exponent
                    dtype_{sm} sum_exp_dev = 0.0;
                    for (npy_intp i = 0; i < axis_size; i++)
                    {{
                        sm_axis[i * sm_axis_stride] = (dtype_{sm}) exp(x_axis[i * x_axis_stride] - max);
                        sum_exp_dev += sm_axis[i * sm_axis_stride];
                    }}

                    // Divide by sum(exp(x-max(x)))
                    dtype_{sm} inv_sum_exp_dev = 1.0 / sum_exp_dev;
                    for (npy_intp i = 0; i < axis_size; i++)
                    {{
                        sm_axis[i * sm_axis_stride] *= inv_sum_exp_dev;
                    }}

                }} while(get_next(iter));
            }}
            NpyIter_Deallocate(iter);
            """
        )

    @staticmethod
    def c_code_cache_version():
        return (4,)


UNSET_AXIS = object()


def softmax(c, axis=UNSET_AXIS):
    if axis is UNSET_AXIS:
        warnings.warn(
            "Softmax now accepts an axis argument. For backwards-compatibility it defaults to -1 when not specified, "
            "but in the future the default will be `None`.\nTo suppress this warning specify axis explicitly.",
            FutureWarning,
        )
        axis = -1

    c = as_tensor_variable(c)
    if c.ndim == 1:
        # TODO: Create Specific warning type that can be suppressed?
        warnings.warn(
            "Softmax no longer converts a vector to a row matrix.",
            UserWarning,
        )
    return Softmax(axis=axis)(c)


class LogSoftmax(COp):
    r"""
    LogSoftmax activation function
    :math:`\\varphi(\\mathbf{x})_j =
    \\e^{(\mathbf{x}_j - log{\sum_{k=1}^K e^{\mathbf{x}_k})}}
    where :math:`K` is the total number of neurons in the layer. This
    activation function gets applied row-wise.

    """

    nin = 1
    nout = 1
    __props__ = ("axis",)

    def __init__(self, axis):
        if axis is not None and not isinstance(axis, int):
            raise TypeError("axis must be an integer or `None`")
        self.axis = axis

    def make_node(self, x):
        x = as_tensor_variable(x)

        if self.axis is not None and (self.axis >= x.ndim or self.axis < -x.ndim):
            raise ValueError(
                f"LogSoftmax axis(={self.axis}) out of bounds for {x.ndim}D array {x}"
            )

        return Apply(self, [x], [x.type()])

    def perform(self, node, input_storage, output_storage):
        (x,) = input_storage
        (z,) = output_storage
        z[0] = scipy.special.log_softmax(x, axis=self.axis)

    def grad(self, inp, grads):
        (x,) = inp
        sm = Softmax(axis=self.axis)(x)
        return [grads[0] - sum(grads[0], axis=self.axis, keepdims=True) * sm]

    def R_op(self, inputs, eval_points):
        # I think the Jacobian is symmetric so the R_op
        # is the same as the grad
        if None in eval_points:
            return [None]
        return self.grad(inputs, eval_points)

    def infer_shape(self, fgraph, node, shape):
        return shape

    def c_headers(self, **kwargs):
        return ["<cmath>"]

    def c_code(self, node, name, inp, out, sub):
        (x,) = inp
        (sm,) = out
        axis = self.axis if self.axis is not None else np.MAXDIMS
        fail = sub["fail"]

        return dedent(
            f"""
            PyArrayObject* op[2];
            npy_uint32 op_flags[2];
            npy_uint32 iter_flags;
            NpyIter* iter;
            NpyIter_IterNextFunc* get_next;
            char** data_ptr;

            int x_ndim = PyArray_NDIM({x});
            int axis = {axis};
            int iterate_axis = !(axis == NPY_MAXDIMS || x_ndim == 1);

            // Validate inputs
            if ((PyArray_TYPE({x}) != NPY_DOUBLE) &&
                (PyArray_TYPE({x}) != NPY_FLOAT))
            {{
                PyErr_SetString(PyExc_TypeError, "not a float");
                {fail}
            }}

            if (axis < 0) axis = x_ndim + axis;
            if ((axis < 0) || (iterate_axis && (axis > x_ndim)))
            {{
                PyErr_SetString(PyExc_ValueError, "invalid axis in LogSoftmax");
                {fail}
            }}

            // Allocate Output Array
            if (({sm}) == NULL || !(PyArray_CompareLists(PyArray_DIMS({sm}), PyArray_DIMS({x}), x_ndim)))
            {{
                Py_XDECREF({sm});
                {sm} = (PyArrayObject*)PyArray_SimpleNew(x_ndim, PyArray_DIMS({x}), PyArray_TYPE({x}));
                if(!{sm}) {{
                    PyErr_SetString(PyExc_MemoryError, "failed to alloc LogSoftmax output");
                    {fail}
                }}
            }}

            // Create numpy iterator
            op[0] = {x};
            op[1] = {sm};
            op_flags[0] = NPY_ITER_READONLY;
            op_flags[1] = NPY_ITER_READWRITE;
            iter_flags = (iterate_axis)? NPY_ITER_MULTI_INDEX : 0;
            iter = NpyIter_MultiNew(
                2,
                op,
                iter_flags,
                NPY_KEEPORDER,
                NPY_NO_CASTING,
                op_flags,
                NULL
            );

            if (iter == NULL)
            {{
                PyErr_SetString(PyExc_MemoryError, "failed to create LogSoftmax iterator");
                {fail}
            }}

            // LogSoftmax is applied across the entire array
            if (!iterate_axis)
            {{
                get_next = NpyIter_GetIterNext(iter, NULL);
                if (get_next == NULL)
                {{
                    NpyIter_Deallocate(iter);
                    PyErr_SetString(PyExc_RuntimeError, "Failed to obtain LogSoftmax GetIterNext");
                    {fail}
                }}
                data_ptr = NpyIter_GetDataPtrArray(iter);

                // Find axis max
                dtype_{x}* x_ptr = (dtype_{x}*)data_ptr[0];
                dtype_{x} max = *x_ptr;
                if (get_next(iter))
                {{
                    do
                    {{
                        dtype_{x}* x_ptr = (dtype_{x}*)data_ptr[0];
                        max = (*x_ptr > max)? *x_ptr : max;
                    }} while(get_next(iter));
                }}

                // Reset Iterator
                if (NpyIter_GotoIterIndex(iter, 0) == NPY_FAIL)
                {{
                    PyErr_SetString(PyExc_RuntimeError, "Failed to reset LogSoftmax iterator");
                    {fail}
                }}

                // Compute xdev and sum(exp(xdev))
                dtype_{sm} sum_exp_xdev = 0.0;
                do
                {{
                    dtype_{x}* x_ptr = (dtype_{x}*)data_ptr[0];
                    dtype_{sm}* sm_ptr = (dtype_{sm}*)data_ptr[1];
                    *sm_ptr = (dtype_{sm})((*x_ptr) - max);
                    sum_exp_xdev += exp(*sm_ptr);
                }} while(get_next(iter));

                // Reset Iterator
                if (NpyIter_GotoIterIndex(iter, 0) == NPY_FAIL)
                {{
                    PyErr_SetString(PyExc_RuntimeError, "Failed to reset LogSoftmax iterator");
                    {fail}
                }}

                // Subtract log(sum(exp(xdev)))
                dtype_{sm} log_sum_exp_xdev = log(sum_exp_xdev);
                do
                {{
                    dtype_{sm}* sm_ptr = (dtype_{sm}*)data_ptr[1];
                    *sm_ptr -= log_sum_exp_xdev;
                }} while(get_next(iter));
            }}

            // LogSoftmax is applied across a specific axis
            else {{
                // Collect axis strides and remove it from iteration
                npy_intp axis_size = PyArray_DIM({x}, axis);
                npy_intp* axis_stride = NpyIter_GetAxisStrideArray(iter, axis);
                if  (axis_stride == NULL)
                {{
                    PyErr_SetString(PyExc_RuntimeError, "Failed to obtain LogSoftmax axis strides");
                    {fail}
                }}
                npy_intp x_axis_stride = axis_stride[0] / sizeof(dtype_{x});
                npy_intp sm_axis_stride = axis_stride[1] / sizeof(dtype_{sm});

                if (NpyIter_RemoveAxis(iter, axis) == NPY_FAIL)
                {{
                    PyErr_SetString(PyExc_RuntimeError, "Failed to remove LogSoftmax axis from iterator");
                    {fail}
                }}

                // Iterate over remaining axes
                get_next = NpyIter_GetIterNext(iter, NULL);
                if (get_next == NULL)
                {{
                    NpyIter_Deallocate(iter);
                    PyErr_SetString(PyExc_RuntimeError, "Failed to obtain LogSoftmax GetIterNext");
                    {fail}
                }}

                data_ptr = NpyIter_GetDataPtrArray(iter);
                do
                {{
                    dtype_{x}* x_axis = (dtype_{x}*)data_ptr[0];
                    dtype_{sm}* sm_axis = (dtype_{sm}*)data_ptr[1];

                    // Find axis max
                    dtype_{x} max = x_axis[0];
                    for (npy_intp i = 1; i < axis_size; i++)
                    {{
                        dtype_{x} x_val = x_axis[i * x_axis_stride];
                        max = (x_val > max)? x_val : max;
                    }}

                    // Compute xdev and sum(exp(xdev))
                    dtype_{sm} sum_exp_xdev = 0.0;
                    for (npy_intp i = 0; i < axis_size; i++)
                    {{
                        sm_axis[i * sm_axis_stride] = (dtype_{x})(x_axis[i * x_axis_stride] - max);
                        sum_exp_xdev += exp(sm_axis[i * sm_axis_stride]);
                    }}

                    // Subtract log(sum(exp(xdev))
                    dtype_{sm} log_sum_exp_xdev = log(sum_exp_xdev);
                    for (npy_intp i = 0; i < axis_size; i++)
                    {{
                        sm_axis[i * sm_axis_stride] -= log_sum_exp_xdev;
                    }}

                }} while(get_next(iter));
            }}
            NpyIter_Deallocate(iter);
            """
        )

    @staticmethod
    def c_code_cache_version():
        return (1,)


def log_softmax(c, axis=UNSET_AXIS):
    if axis is UNSET_AXIS:
        warnings.warn(
            "logsoftmax now accepts an axis argument. For backwards-compatibility it defaults to -1 when not specified, "
            "but in the future the default will be `None`.\nTo suppress this warning specify axis explicitly.",
            FutureWarning,
        )
        axis = -1

    c = as_tensor_variable(c)
    if c.ndim == 1:
        # TODO: Create Specific warning type that can be suppressed?
        warnings.warn(
            "Softmax no longer converts a vector to a row matrix.",
            UserWarning,
        )
    return LogSoftmax(axis=axis)(c)


__all__ = [
    "softmax",
    "log_softmax",
]
