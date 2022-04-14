import warnings
from numbers import Number
from textwrap import dedent
from typing import Dict, List, Tuple, Union

import numpy as np

import aesara
from aesara.gradient import DisconnectedType
from aesara.graph.basic import Apply, Variable
from aesara.link.c.op import COp
from aesara.link.c.params_type import ParamsType
from aesara.misc.safe_asarray import _asarray
from aesara.scalar import int32
from aesara.tensor import _get_vector_length
from aesara.tensor import basic as at
from aesara.tensor import get_vector_length
from aesara.tensor.exceptions import NotScalarConstantError
from aesara.tensor.type import DenseTensorType, TensorType, int_dtypes, tensor
from aesara.tensor.type_other import NoneConst
from aesara.tensor.var import TensorConstant, TensorVariable


def register_shape_c_code(type, code, version=()):
    """
    Tell Shape Op how to generate C code for an Aesara Type.

    Parameters
    ----------
    typ : Aesara type
        It must be the Aesara class itself and not an instance of the class.
    code : C code
        Returns a vector representing the shape for the Aesara type 'typ'.
        Use %(iname)s and %(oname)s for the input and output C variable names
        respectively.
    version
        A number indicating the version of the code, for cache.

    """
    Shape.c_code_and_version[type] = (code, version)


class Shape(COp):
    """
    L{Op} to return the shape of a matrix.

    Notes
    -----
    Non-differentiable.

    """

    _f16_ok = True

    # Mapping from Type to C code (and version) to use.
    # In the C code, the name of the input variable is %(iname)s,
    # the output variable is %(oname)s.
    c_code_and_version: Dict = {}

    check_input = False
    __props__ = ()

    def make_node(self, x):
        if not isinstance(x, Variable):
            x = at.as_tensor_variable(x)

        if isinstance(x.type, TensorType):
            out_var = TensorType("int64", (x.type.ndim,))()
        else:
            out_var = aesara.tensor.type.lvector()

        return Apply(self, [x], [out_var])

    def perform(self, node, inp, out_):
        (x,) = inp
        (out,) = out_
        out[0] = _asarray(np.shape(x), dtype="int64")

    def infer_shape(self, fgraph, node, in_shapes):
        return [[len(in_shapes[0])]]

    def connection_pattern(self, node):
        # the grad returns the gradient with respect to the
        # elements of a tensor variable
        # the elements of the tensor variable do not participate
        # in the computation of the shape, so they are not really
        # part of the graph
        return [[False]]

    def grad(self, inp, grads):
        # the grad returns the gradient with respect to the
        # elements of a tensor variable
        # the elements of the tensor variable do not participate
        # in the computation of the shape, so they are not really
        # part of the graph
        return [aesara.gradient.DisconnectedType()()]

    def R_op(self, inputs, eval_points):
        return [None]

    def c_code(self, node, name, inames, onames, sub):
        (iname,) = inames
        (oname,) = onames
        fail = sub["fail"]

        itype = node.inputs[0].type.__class__
        if itype in self.c_code_and_version:
            code, version = self.c_code_and_version[itype]
            return code % locals()

        # Else, no C code
        raise NotImplementedError()

    def c_code_cache_version(self):
        version = []
        # If any of the c code is unversioned, we have to return ()
        # Else, we will return a list of (type name, version) pairs.
        for t, (c, v) in sorted(
            self.c_code_and_version.items(), key=lambda pair: str(pair[0])
        ):
            if not v:
                warnings.warn(
                    f"Type {t} has C code for Shape, but it has no "
                    "version. You should add a 'version' keyword "
                    "arg when calling register_shape_c_code.",
                    stacklevel=2,
                )
                return ()
            version.append((str(t), v))

        if version:
            version.append(1)

        return tuple(version)


_shape = Shape()


def shape(x: Union[np.ndarray, Number, Variable]) -> Variable:
    """Return the shape of `x`."""
    if not isinstance(x, Variable):
        x = at.as_tensor_variable(x)

    x_type = x.type

    if isinstance(x_type, TensorType) and all(s is not None for s in x_type.shape):
        res = at.as_tensor_variable(x_type.shape, ndim=1, dtype=np.int64)
    else:
        res = _shape(x)

    return res


@_get_vector_length.register(Shape)
def _get_vector_length_Shape(op, var):
    return var.owner.inputs[0].type.ndim


def shape_tuple(x: TensorVariable) -> Tuple[Variable, ...]:
    """Get a tuple of symbolic shape values.

    This will return a `ScalarConstant` with the value ``1`` wherever
    broadcastable is ``True``.
    """
    one_at = aesara.scalar.ScalarConstant(aesara.scalar.int64, 1)
    return tuple(
        one_at if getattr(sh, "value", sh) == 1 or bcast else sh
        for sh, bcast in zip(
            shape(x), getattr(x, "broadcastable", (False,) * x.type.ndim)
        )
    )


class Shape_i(COp):
    """
    L{Op} to return the shape of a matrix.

    Notes
    -----
    Non-differentiable.

    """

    _f16_ok = True

    # Mapping from Type to C code (and version) to use.
    # In the C code, the name of the input variable is %(iname)s,
    # the output variable is %(oname)s.
    c_code_and_version: Dict = {}

    check_input = False

    __props__ = ("i",)

    def __init__(self, i):
        # As i will be used in the hash and that ndarray are not hashable,
        # we need to convert it to an int as it is hashable.
        if isinstance(i, np.ndarray):
            assert i.dtype in aesara.tensor.type.integer_dtypes
        assert i == int(i)
        i = int(i)
        self.i = i

    # NB:
    # 1) params_type is defined as a property to avoid
    #    loop in Python import caused by importing aesara.scalar below
    #    when params_type is defined directly in class code.
    # 2) We wrap scalar into ParamsType (instead of directly using scalar as op param)
    #    to avoid Aesara converting scalar param to constant that would be later
    #    hardcoded as literal in C code, making us loose all the advantages of
    #    using params.
    @property
    def params_type(self):
        return ParamsType(i=aesara.scalar.basic.int64)

    def __str__(self):
        return "%s{%i}" % (self.__class__.__name__, self.i)

    def make_node(self, x):
        if not isinstance(x, Variable) or not hasattr(x.type, "ndim"):
            raise TypeError(
                f"{x} must be `Variable` with a `Type` having an ndim attribute"
            )
        if x.type.ndim <= self.i:
            raise TypeError(f"{x} has too few dimensions for Shape_i")
        return Apply(self, [x], [aesara.tensor.type.lscalar()])

    def perform(self, node, inp, out_, params):
        (x,) = inp
        (out,) = out_
        if out[0] is None:
            out[0] = _asarray(np.shape(x)[self.i], dtype="int64")
        else:
            out[0][...] = np.shape(x)[self.i]

    def c_code_cache_version(self):
        version = []
        # If any of the c code is unversioned, we have to return ()
        # Else, we will return a list of (type name, version) pairs.
        for t, (c, ci, v) in sorted(
            self.c_code_and_version.items(), key=lambda pair: str(pair[0])
        ):
            if not v:
                warnings.warn(
                    f"Type {t} has C code for Shape_i, but it has "
                    "no version. You should add a 'version' keyword "
                    "arg when calling register_shape_i_c_code.",
                    stacklevel=2,
                )
                return ()
            version.append((str(t), v))

        if version:
            version.append(2)

        return tuple(version)

    def c_code(self, node, name, inames, onames, sub):
        (iname,) = inames
        (oname,) = onames
        fail = sub["fail"]
        # i is then 'params->i', not just 'params'.
        i = sub["params"] + "->i"

        itype = node.inputs[0].type.__class__
        if itype in self.c_code_and_version:
            code, check_input, version = self.c_code_and_version[itype]
            return (check_input + code) % locals()

        # Else, no C code
        raise NotImplementedError()

    def infer_shape(self, fgraph, node, input_shapes):
        return [()]

    def connection_pattern(self, node):
        # the grad returns the gradient with respect to the
        # elements of a tensor variable
        # the elements of the tensor variable do not participate
        # in the computation of the shape, so they are not really
        # part of the graph
        return [[False]]

    def grad(self, inp, grads):
        return [
            aesara.gradient.grad_not_implemented(
                op=self,
                x_pos=0,
                x=inp[0],
                comment=("No gradient for the shape of a matrix " "is implemented."),
            )
        ]


def shape_i(var, i, fgraph=None):
    """
    Equivalent of var.shape[i], but apply if possible the shape feature
    optimization.

    This is useful in optimization that need to get the shape. This
    remove the need of the following shape_feature optimization that
    convert it. So this speed up optimization and remove Equilibrium
    max iteration problems.

    Parameters
    ----------
    var : Variable
        The variable we want to take the shape of.
    i : int
        The shape dimensions we want
    fgraph : FunctionGraph (optional)

    """
    if fgraph and hasattr(fgraph, "shape_feature"):
        shape_feature = fgraph.shape_feature
        shape_of = shape_feature.shape_of

        def recur(node):
            if not node.outputs[0] in shape_of:
                for inp in node.inputs:
                    if inp.owner:
                        recur(inp.owner)
                # If the output var isn't marked as being in the graph,
                # we need to add it in the ShapeFeature.
                shape_feature.on_import(fgraph, node, "graph.ops.shape_i")

        if var not in shape_of:
            recur(var.owner)
        return shape_of[var][i]

    # If we are not able to use the shape feature, we should not put
    # Shape_i in the graph. Otherwise, the shape feature optimization
    # won't get applied.
    return shape(var)[i]


def shape_i_op(i):
    key = i
    if key not in shape_i_op.cache:
        shape_i_op.cache[key] = Shape_i(i)
    return shape_i_op.cache[key]


shape_i_op.cache = {}


def register_shape_i_c_code(typ, code, check_input, version=()):
    """
    Tell Shape_i how to generate C code for an Aesara Type.

    Parameters
    ----------
    typ : Aesara type
        It must be the Aesara class itself and not an instance of the class.
    code : C code
        Gets the shape of dimensions %(i)s for the Aesara type 'typ'.
        Use %(iname)s and %(oname)s for the input and output C variable names
        respectively.
    version
        A number indicating the version of the code, for cache.

    """
    Shape_i.c_code_and_version[typ] = (code, check_input, version)


class SpecifyShape(COp):
    """
    L{Op} that puts into the graph the user-provided shape.

    In the case where this `Op` stays in the final graph, we assert the shape.
    For this the output of this op must be used in the graph. This is not
    the case most of the time if we only take the shape of the output.
    Maybe there are other optimizations that will mess with this.

    Notes
    -----
    Maybe in the future we will never do the assert!
    """

    view_map = {0: [0]}
    __props__ = ()
    _f16_ok = True

    def make_node(self, x, *shape):
        from aesara.tensor.basic import get_scalar_constant_value

        x = at.as_tensor_variable(x)

        shape = tuple(
            NoneConst
            if (s is None or NoneConst.equals(s))
            else at.as_tensor_variable(s, ndim=0)
            for s in shape
        )

        if any(
            s.dtype not in aesara.tensor.type.integer_dtypes
            for s in shape
            if hasattr(s, "dtype")
        ):
            raise TypeError("Shape values must be integer types")

        if len(shape) != x.type.ndim:
            raise ValueError(
                f"Input `x` is {x.type.ndim}-dimensional and will never match a shape of length {len(shape)}."
            )

        type_shape = [None] * x.ndim
        for i, (xts, s) in enumerate(zip(x.type.shape, shape)):
            if xts is not None:
                type_shape[i] = xts
            else:
                try:
                    type_s = get_scalar_constant_value(s)
                    if type_s is not None:
                        type_shape[i] = int(type_s)
                except NotScalarConstantError:
                    pass

        out_var = x.type.clone(shape=type_shape)()

        return Apply(self, [x, *shape], [out_var])

    def perform(self, node, inp, out_):
        x, *shape = inp
        (out,) = out_
        ndim = len(shape)
        if x.ndim != ndim:
            raise AssertionError(
                f"SpecifyShape: Got {x.ndim} dimensions (shape {x.shape}), expected {ndim} dimensions with shape {tuple(shape)}."
            )
        if not all(xs == s for xs, s in zip(x.shape, shape) if s is not None):
            raise AssertionError(
                f"SpecifyShape: Got shape {x.shape}, expected {tuple(int(s) if s is not None else None for s in shape)}."
            )
        out[0] = x

    def infer_shape(self, fgraph, node, shapes):
        xshape, *_ = shapes
        shape = node.inputs[1:]
        new_shape = []
        for dim in range(node.inputs[0].type.ndim):
            s = shape[dim]
            try:
                s = at.get_scalar_constant_value(s)
                # We assume that `None` shapes are always retrieved by
                # `get_scalar_constant_value`, and only in that case do we default to
                # the shape of the input variable
                if s is None:
                    s = xshape[dim]
            except NotScalarConstantError:
                pass
            new_shape.append(at.as_tensor_variable(s))

        assert len(new_shape) == len(xshape)
        return [new_shape]

    def connection_pattern(self, node):
        return [[True], *[[False]] * len(node.inputs[1:])]

    def grad(self, inp, grads):
        x, *shape = inp
        (gz,) = grads
        # Should I set an SpecifyShape on gz? I think so
        # But I don't do it now as we need to make an optimization
        # to remove that op from the graph to don't block other optimization
        # Should I do an optimizer that will remove the SpecifyShape?
        # I think Yes
        # return [specify_shape(gz, s)] + [aesara.gradient.DisconnectedType()() for _ in range(len(shape))]
        return [gz] + [aesara.gradient.DisconnectedType()() for _ in range(len(shape))]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            # It means that this op sits on top of a non-differentiable path
            return [None]
        return self.make_node(eval_points[0], *inputs[1:]).outputs

    def c_code(self, node, name, i_names, o_names, sub):
        if not isinstance(node.inputs[0].type, DenseTensorType):
            raise NotImplementedError(
                f"Specify_shape c_code not implemented for input type {node.inputs[0].type}"
            )

        x_name, *shape_names = i_names
        (o_name,) = o_names
        fail = sub["fail"]

        code = dedent(
            f"""
            if (PyArray_NDIM({x_name}) != {len(shape_names)}) {{
                PyErr_Format(PyExc_AssertionError,
                    "SpecifyShape: Got %d dimensions, expected %d dimensions.",
                    PyArray_NDIM({x_name}), {len(shape_names)}
                );
                {fail};
            }}
            """
        )

        for i, (shp_name, shp) in enumerate(zip(shape_names, node.inputs[1:])):
            if NoneConst.equals(shp):
                continue
            code += dedent(
                f"""
                if (py_{shp_name} != Py_None){{
                    dtype_{shp_name} shp = ((dtype_{shp_name}*)PyArray_GETPTR1({shp_name}, 0))[0];
                    if (PyArray_DIMS({x_name})[{i}] != shp) {{
                        PyErr_Format(PyExc_AssertionError,
                            "SpecifyShape: dim %d of input has shape %d, expected %d.",
                            {i}, PyArray_DIMS({x_name})[{i}], shp
                        );
                        {fail};
                    }}
                }}
                """
            )

        code += dedent(
            f"""
            Py_XDECREF({o_name});
            {o_name} = {x_name};
            Py_XINCREF({o_name});
            """
        )
        return code

    def c_code_cache_version(self):
        return (2,)


_specify_shape = SpecifyShape()


def specify_shape(
    x: Union[np.ndarray, Number, Variable],
    shape: Union[
        int, List[Union[int, Variable]], Tuple[Union[int, Variable]], Variable
    ],
):
    """Specify a fixed shape for a `Variable`.

    If a dimension's shape value is ``None``, the size of that dimension is not considered fixed/static at runtime.
    """

    if not isinstance(shape, (tuple, list)):
        shape = (shape,)

    # If shape is a symbolic 1d vector of fixed length, we separate the items into a
    # tuple with one entry per shape dimension
    if len(shape) == 1 and shape[0] is not None:
        shape_vector = at.as_tensor_variable(shape[0])
        if shape_vector.ndim == 1:
            try:
                shape = tuple(shape_vector)
            except ValueError:
                raise ValueError("Shape vector must have fixed dimensions")

    return _specify_shape(x, *shape)


@_get_vector_length.register(SpecifyShape)
def _get_vector_length_SpecifyShape(op, var):
    try:
        return at.get_scalar_constant_value(var.owner.inputs[1]).item()
    except NotScalarConstantError:
        raise ValueError(f"Length of {var} cannot be determined")


class Reshape(COp):
    """Perform a reshape operation of the input x to the new shape shp.
    The number of dimensions to which to reshape to (ndim) must be
    known at graph build time.
    """

    view_map = {0: [0]}  # output 0 is potentially aliased to inputs [0]
    _f16_ok = True

    check_input = False
    __props__ = ("ndim",)
    params_type = ParamsType(ndim=int32)
    # name does not participate because it doesn't affect computations

    def __init__(self, ndim, name=None):
        self.ndim = int(ndim)
        if ndim < 0:
            raise ValueError("The output dimensions after reshape must be 0 or greater")
        assert name is None, "name attribute for Reshape has been deprecated"

    def __str__(self):
        return f"{self.__class__.__name__}{{{self.ndim}}}"

    def make_node(self, x, shp):
        x = at.as_tensor_variable(x)
        shp_orig = shp
        shp = at.as_tensor_variable(shp, ndim=1)
        if not (
            shp.dtype in int_dtypes
            or (isinstance(shp, TensorConstant) and shp.data.size == 0)
        ):
            # It raises an error if shp is not of integer type,
            # except when shp is constant and empty
            # (in this case, shp.dtype does not matter anymore).
            raise TypeError(f"Shape must be integers; got {shp.dtype}")
        assert shp.ndim == 1
        if isinstance(shp, TensorConstant):
            bcast = [s == 1 for s in shp.data]
            return Apply(self, [x, shp], [tensor(x.type.dtype, bcast)])
        else:
            bcasts = [False] * self.ndim
            shp_list = shp_orig
            if hasattr(shp_orig, "ndim") and shp_orig.ndim == 0:
                shp_list = [shp_orig]
            for index in range(self.ndim):
                y = shp_list[index]
                y = at.as_tensor_variable(y)
                # Try to see if we can infer that y has a constant value of 1.
                # If so, that dimension should be broadcastable.
                try:
                    bcasts[index] = (
                        hasattr(y, "get_scalar_constant_value")
                        and y.get_scalar_constant_value() == 1
                    )
                except NotScalarConstantError:
                    pass
            return Apply(self, [x, shp], [tensor(x.type.dtype, bcasts)])

    def perform(self, node, inp, out_, params):
        x, shp = inp
        (out,) = out_
        if len(shp) != self.ndim:
            raise ValueError(
                (
                    "Shape argument to Reshape has incorrect"
                    f" length: {len(shp)}, should be {self.ndim}"
                )
            )
        out[0] = np.reshape(x, shp)

    def connection_pattern(self, node):
        return [[True], [False]]

    def grad(self, inp, grads):
        x, shp = inp
        (g_out,) = grads
        return [reshape(g_out, shape(x), ndim=x.ndim), DisconnectedType()()]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return [None]
        return self(eval_points[0], *inputs[1:], return_list=True)

    def infer_shape(self, fgraph, node, ishapes):
        from aesara.tensor.math import eq, maximum, mul

        # inputs[1] can contain at most one value of '-1', meaning the actual
        # shape of the output will be automatically computed by reshape, so
        # that the total number of elements stays the same.
        # TODO: Maybe put that formula here?
        # It's not trivial, because we would have to check if the product of
        # all the non-minus-one shapes is a divisor of the product of the
        # original shapes.
        # The following expression leads to cycles in feature_shape,
        # because it tries to replace the Shape_i node by the switch
        # statement, which depends on Shape_i.
        # return [tuple([switch(eq(node.inputs[1][i], -1),
        #                      Shape_i(i)(node.outputs[0]),
        #                      node.inputs[1][i])
        #                    for i in range(self.ndim)]
        #    )]
        # Here, we only simplify if the shape (node.inputs[1]) is a constant,
        # ideally it would suffice to check that it is always non-negative.
        # If current variable is a scalar and its dimensionality should
        # change to self.ndim, then use size 1 for all new dimensions.
        if len(ishapes[0]) == 0:
            return [(1,) * self.ndim]

        requ = node.inputs[1]
        input_size = mul(*ishapes[0])
        if isinstance(requ, TensorConstant):
            requ = list(requ.data)
            requ_part = [ele for ele in requ if ele != -1]
            crit = len(requ) - len(requ_part)
            if crit == 1 and len(requ_part) > 0:
                # If there are both 0 and -1 in requ_size, it is impossible
                # to determine a right output, but we can at least prevent
                # a division by 0. We do not want to keep a negative
                # size here as it could lead to further weird errors
                # after other optimizations.
                requ_size = mul(*requ_part)
                missing = input_size // (1 if requ_size == 0 else requ_size)
                for i, ele in enumerate(requ):
                    if ele == -1:
                        requ[i] = missing
            elif crit == 1:  # we reshape to -1
                requ = [input_size] if ishapes[0] else [1]
            elif crit > 1:
                raise ValueError(
                    "shape argument to Reshape.perform"
                    " must have at most one entry equal to -1"
                )
            return [requ]
        else:

            requ = [requ[i] for i in range(self.ndim)]
            # since new_dims can have negative value (-1), the
            # multiplication of all values should be negated
            # to give a positive value.
            # To avoid optimization complexity, we avoid checking
            # for the case when there are two or more '-1' values.
            if self.ndim:
                requ_size = -mul(*requ)
                # If there are both 0 and -1 in requ_size, it is impossible
                # to determine a right output, but we can at least prevent
                # a division by 0. We do not want to keep a negative
                # size here as it could lead to further weird errors
                # after other optimizations.
                rest_size = input_size // maximum(requ_size, 1)
            return [
                tuple(
                    [
                        at.switch(eq(requ[i], -1), rest_size, requ[i])
                        for i in range(self.ndim)
                    ]
                )
            ]

    def c_code_cache_version(self):
        return (9,)

    def c_code(self, node, name, inputs, outputs, sub):
        x, shp = inputs
        (z,) = outputs
        fail = sub["fail"]
        params = sub["params"]
        return f"""
        assert (PyArray_NDIM({shp}) == 1);

        PyArray_Dims newshape;

        if (!PyArray_IntpConverter((PyObject *){shp}, &newshape)) {{
            {fail};
        }}

        if ({params}->ndim != newshape.len) {{
            PyErr_SetString(PyExc_ValueError, "Shape argument to Reshape has incorrect length");
            PyDimMem_FREE(newshape.ptr);
            {fail};
        }}

        Py_XDECREF({z});
        {z} = (PyArrayObject *) PyArray_Newshape({x}, &newshape, NPY_CORDER);

        PyDimMem_FREE(newshape.ptr);

        if (!{z}) {{
            //The error message should have been set by PyArray_Newshape
            {fail};
        }}
        """


def reshape(x, newshape, ndim=None):
    if ndim is None:
        newshape = at.as_tensor_variable(newshape)
        if newshape.ndim != 1:
            raise TypeError(
                "New shape in reshape must be a vector or a list/tuple of"
                f" scalar. Got {newshape} after conversion to a vector."
            )
        try:
            ndim = get_vector_length(newshape)
        except ValueError:
            raise ValueError(
                f"The length of the provided shape ({newshape}) cannot "
                "be automatically determined, so Aesara is not able "
                "to know what the number of dimensions of the reshaped "
                "variable will be. You can provide the 'ndim' keyword "
                "argument to 'reshape' to avoid this problem."
            )
    op = Reshape(ndim)
    rval = op(x, newshape)
    return rval


def shape_padleft(t, n_ones=1):
    """Reshape `t` by left-padding the shape with `n_ones` 1s.

    See Also
    --------
    shape_padaxis
    shape_padright
    Dimshuffle

    """
    _t = at.as_tensor_variable(t)

    pattern = ["x"] * n_ones + [i for i in range(_t.type.ndim)]
    return _t.dimshuffle(pattern)


def shape_padright(t, n_ones=1):
    """Reshape `t` by right-padding the shape with `n_ones` 1s.

    See Also
    --------
    shape_padaxis
    shape_padleft
    Dimshuffle

    """
    _t = at.as_tensor_variable(t)

    pattern = [i for i in range(_t.type.ndim)] + ["x"] * n_ones
    return _t.dimshuffle(pattern)


def shape_padaxis(t, axis):
    """Reshape `t` by inserting 1 at the dimension `axis`.

    Examples
    --------
    >>> tensor = aesara.tensor.type.tensor3()
    >>> aesara.tensor.shape_padaxis(tensor, axis=0)
    DimShuffle{x,0,1,2}.0
    >>> aesara.tensor.shape_padaxis(tensor, axis=1)
    DimShuffle{0,x,1,2}.0
    >>> aesara.tensor.shape_padaxis(tensor, axis=3)
    DimShuffle{0,1,2,x}.0
    >>> aesara.tensor.shape_padaxis(tensor, axis=-1)
    DimShuffle{0,1,2,x}.0

    See Also
    --------
    shape_padleft
    shape_padright
    Dimshuffle

    """
    _t = at.as_tensor_variable(t)

    ndim = _t.ndim + 1
    if not -ndim <= axis < ndim:
        msg = "axis {0} is out of bounds [-{1}, {1})".format(axis, ndim)
        raise IndexError(msg)
    if axis < 0:
        axis += ndim

    pattern = [i for i in range(_t.type.ndim)]
    pattern.insert(axis, "x")
    return _t.dimshuffle(pattern)


register_shape_c_code(
    TensorType,
    """
    npy_intp shape[] = {PyArray_NDIM(%(iname)s)};
    if(%(oname)s == NULL || (PyArray_DIMS(%(oname)s)[0] != shape[0]))
    {
        Py_XDECREF(%(oname)s);
        %(oname)s = (PyArrayObject*) PyArray_SimpleNew(1, shape, NPY_INT64);
    }
    for(int i=0;i<shape[0];i++)
    {
        ((npy_int64*)PyArray_GETPTR1(%(oname)s, i))[0] = PyArray_DIMS(%(iname)s)[i];
    }
    """,
    version=1,
)


register_shape_i_c_code(
    TensorType,
    """
    if(!%(oname)s)
        %(oname)s=(PyArrayObject*)PyArray_EMPTY(0, NULL, NPY_INT64, 0);
    ((npy_int64*)PyArray_DATA(%(oname)s))[0]=PyArray_DIMS(%(iname)s)[%(i)s];
    """,
    """
    if (%(i)s>=PyArray_NDIM(%(iname)s)){
        PyErr_SetString(PyExc_TypeError,
            "Number of dimensions lower than expected");
        %(fail)s
    }
    """,
    version=3,
)
