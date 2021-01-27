"""
This file contains auxiliary Ops, used during the compilation phase and Ops
building class (:class:`FromFunctionOp`) and decorator (:func:`as_op`) that
help make new Ops more rapidly.

"""

import copy
import pickle
import warnings
from collections import OrderedDict

import numpy as np

import theano
from theano.graph.basic import Apply
from theano.graph.op import COp, Op
from theano.graph.type import CType


def register_view_op_c_code(type, code, version=()):
    """
    Tell ViewOp how to generate C code for a Theano Type.

    Parameters
    ----------
    type : Theano type
        It must be the Theano class itself and not an instance of the class.
    code : C code
        Returns a view for the Theano type 'type'. Use %(iname)s and %(oname)s
        for the input and output C variable names respectively.
    version
        A number indicating the version of the code, for cache.

    """
    ViewOp.c_code_and_version[type] = (code, version)


class ViewOp(COp):
    """
    Returns an inplace view of the input. Used internally by Theano.

    """

    view_map = {0: [0]}
    # Mapping from Type to C code (and version) to use.
    # In the C code, the name of the input variable is %(iname)s,
    # the output variable is %(oname)s.
    c_code_and_version = {}
    __props__ = ()
    _f16_ok = True

    def make_node(self, x):
        return Apply(self, [x], [x.type()])

    def perform(self, node, inp, out):
        (x,) = inp
        (z,) = out
        z[0] = x

    def __str__(self):
        return f"{self.__class__.__name__}"

    def c_code(self, node, nodename, inp, out, sub):
        (iname,) = inp
        (oname,) = out
        fail = sub["fail"]

        itype = node.inputs[0].type.__class__
        if itype in self.c_code_and_version:
            code, version = self.c_code_and_version[itype]
            return code % locals()

        # Else, no C code
        raise NotImplementedError()

    def c_code_cache_version(self):
        version = []
        # If any of the c code is unversionned, we have to return ()
        # Else, we will return a list of (type name, version) pairs.
        for t, (c, v) in sorted(
            self.c_code_and_version.items(), key=lambda pair: str(pair[0])
        ):
            if not v:
                warnings.warn(
                    f"Type {t} has C code for ViewOp, but it has no "
                    "version. You should add a 'version' keyword "
                    "arg when calling register_view_op_c_code.",
                    stacklevel=2,
                )
                return ()
            version.append((str(t), v))

        return tuple(version)

    def infer_shape(self, fgraph, node, input_shapes):
        return input_shapes

    def grad(self, args, g_outs):
        return g_outs


view_op = ViewOp()


class OutputGuard(ViewOp):
    """
    This op is used only internally by Theano.

    Only the AddDestroyHandler optimizer tries to insert them in the graph.

    This Op is declared as destructive while it is not destroying anything.
    It returns a view. This is used to prevent destruction of the output
    variables of a Theano function.

    There is a mechanism in Theano that should prevent this, but the use
    of OutputGuard adds a safeguard: it may be possible for some optimization
    run before the add_destroy_handler phase to bypass this mechanism, by
    making in-place optimizations.

    TODO: find a current full explanation.

    """

    destroy_map = {0: [0]}

    check_input = False


_output_guard = OutputGuard()


def register_deep_copy_op_c_code(typ, code, version=()):
    """
    Tell DeepCopyOp how to generate C code for a Theano Type.

    Parameters
    ----------
    typ : Theano type
        It must be the Theano class itself and not an instance of the class.
    code: C code
        Deep copies the Theano type 'typ'. Use %(iname)s and %(oname)s for the
        input and output C variable names respectively.
    version
        A number indicating the version of the code, for cache.

    """
    DeepCopyOp.c_code_and_version[typ] = (code, version)


class DeepCopyOp(COp):
    # Mapping from Type to C code (and version) to use.
    # In the C code, the name of the input variable is %(iname)s,
    # the output variable is %(oname)s.
    c_code_and_version = {}

    check_input = False
    __props__ = ()
    _f16_ok = True

    def __init__(self):
        pass

    def make_node(self, x):
        return Apply(self, [x], [x.type()])

    def perform(self, node, args, outs):
        if hasattr(args[0], "copy"):
            # when args[0] is a an ndarray of 0 dimensions,
            # this return a numpy.dtype and not an ndarray
            # So when the args have a copy attribute we use it
            # as this don't have this problem
            outs[0][0] = args[0].copy()
        else:
            outs[0][0] = copy.deepcopy(args[0])

    def c_code_cache_version(self):
        version = []
        # If any of the c code is unversionned, we have to return ()
        # Else, we will return a list of (type name, version) pairs.
        for t, (c, v) in sorted(
            self.c_code_and_version.items(), key=lambda pair: str(pair[0])
        ):
            if not v:
                warnings.warn(
                    f"Type {t} has C code for DeepCopyOp, but it has "
                    "no version. You should add a 'version' keyword"
                    " arg when calling "
                    "register_deep_copy_op_c_code.",
                    stacklevel=2,
                )
                return ()
            version.append((str(t), v))

        if version:
            version.append(1)
        return tuple(version)

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


deep_copy_op = DeepCopyOp()


# List of Theano Types that one can add an extra dimension and for which
# Scan can deal with.
expandable_types = ()


def load_back(mod, name):
    __import__(mod)
    import sys

    module = sys.modules[mod]
    obj = getattr(module, name)
    return obj


class FromFunctionOp(Op):
    """
    Build a basic Theano Op around a function.

    Since the resulting Op is very basic and is missing most of the
    optional functionalities, some optimizations may not apply.  If you
    want to help, you can supply an infer_shape function that computes
    the shapes of the output given the shapes of the inputs.

    Also the gradient is undefined in the resulting op and Theano will
    raise an error if you attempt to get the gradient of a graph
    containing this op.

    """

    def __init__(self, fn, itypes, otypes, infer_shape):
        self.__fn = fn
        self.itypes = itypes
        self.otypes = otypes
        self.__infer_shape = infer_shape
        if self.__infer_shape is not None:
            self.infer_shape = self._infer_shape

    def __eq__(self, other):
        return type(self) == type(other) and self.__fn == other.__fn

    def __hash__(self):
        return hash(type(self)) ^ hash(self.__fn)

    def __str__(self):
        return "FromFunctionOp{%s}" % self.__fn.__name__

    def perform(self, node, inputs, outputs):
        outs = self.__fn(*inputs)
        if not isinstance(outs, (list, tuple)):
            outs = (outs,)
        assert len(outs) == len(outputs)
        for i in range(len(outs)):
            outputs[i][0] = outs[i]

    def __reduce__(self):
        mod = self.__fn.__module__
        name = self.__fn.__name__
        try:
            obj = load_back(mod, name)
        except (ImportError, KeyError, AttributeError):
            raise pickle.PicklingError(
                f"Can't pickle as_op(), not found as {mod}.{name}"
            )
        else:
            if obj is not self:
                raise pickle.PicklingError(
                    f"Can't pickle as_op(), not the object at {mod}.{name}"
                )
        return load_back, (mod, name)

    def _infer_shape(self, fgraph, node, input_shapes):
        return self.__infer_shape(fgraph, node, input_shapes)


def as_op(itypes, otypes, infer_shape=None):
    """
    Decorator that converts a function into a basic Theano op that will call
    the supplied function as its implementation.

    It takes an optional infer_shape parameter that should be a callable with
    this signature:

        def infer_shape(fgraph, node, input_shapes):
            ...
            return output_shapes

    Here `input_shapes` and `output_shapes` are lists of tuples that represent
    the shape of the corresponding inputs/outputs.

    This should not be used when performance is a concern since the very basic
    nature of the resulting Op may interfere with certain graph optimizations.

    Examples
    --------
    @as_op(itypes=[theano.tensor.fmatrix, theano.tensor.fmatrix],
           otypes=[theano.tensor.fmatrix])
    def numpy_dot(a, b):
        return numpy.dot(a, b)

    """
    if not isinstance(itypes, (list, tuple)):
        itypes = [itypes]
    if any(not isinstance(t, CType) for t in itypes):
        raise TypeError("itypes has to be a list of Theano types")
    if not isinstance(otypes, (list, tuple)):
        otypes = [otypes]
    if any(not isinstance(t, CType) for t in otypes):
        raise TypeError("otypes has to be a list of Theano types")

    # make sure they are lists and not tuples
    itypes = list(itypes)
    otypes = list(otypes)

    if infer_shape is not None and not callable(infer_shape):
        raise TypeError("infer_shape needs to be a callable")

    def make_op(fn):
        return FromFunctionOp(fn, itypes, otypes, infer_shape)

    return make_op


def register_rebroadcast_c_code(typ, code, version=()):
    """
    Tell Rebroadcast how to generate C code for a Theano Type.

    typ : Theano type
        It must be the Theano class itself and not an instance of the class.
    code : C code
        That checks if the dimension %(axis)s is of shape 1 for the Theano type
        'typ'. Use %(iname)s and %(oname)s for the input and output C variable
        names respectively, and %(axis)s for the axis that we need to check.
        This code is put in a loop for all axes.
    version
        A number indicating the version of the code, for cache.

    """
    Rebroadcast.c_code_and_version[typ] = (code, version)


class Rebroadcast(COp):
    """
    Change the input's broadcastable fields in some predetermined way.

    See Also
    --------
    unbroadcast <theano.tensor.unbroadcast>
    addbroadcast <theano.tensor.addbroadcast>
    patternbroadcast <theano.tensor.patternbroadcast>

    Notes
    -----
    Works inplace and works for CudaNdarrayType.

    Examples
    --------
    `Rebroadcast((0, True), (1, False))(x)` would make `x` broadcastable in
    axis 0 and not broadcastable in axis 1.

    """

    view_map = {0: [0]}
    _f16_ok = True
    # Mapping from Type to C code (and version) to use.
    # In the C code, the name of the input variable is %(iname)s,
    # the output variable is %(oname)s.
    c_code_and_version = {}

    check_input = False
    __props__ = ("axis",)
    _f16_ok = True

    def __init__(self, *axis):
        # Sort them to make sure we merge all possible case.
        items = sorted(axis)
        self.axis = OrderedDict(items)
        for axis, broad in self.axis.items():
            if not isinstance(axis, (np.integer, int)):
                raise TypeError(f"Rebroadcast needs integer axes. Got {axis}")

            if not isinstance(broad, (np.bool_, bool)):
                raise TypeError(
                    f"Rebroadcast needs bool for new broadcast pattern. Got {broad}"
                )

    def __hash__(self):
        # Need special __hash__ as dict aren't hashable.
        # no ambiguity because each item key is unique
        items = sorted(self.axis.items())
        return hash((type(self), tuple(items)))

    def __str__(self):
        if len(self.axis) == 0:
            broadcast_pattern = []
        else:
            broadcast_pattern = ["?" for i in range(1 + max(self.axis.keys()))]
        for k, v in self.axis.items():
            broadcast_pattern[k] = str(int(v))
        return f"{self.__class__.__name__}{{{','.join(broadcast_pattern)}}}"

    def make_node(self, x):
        if self.axis.keys() and (x.ndim <= max(self.axis.keys())):
            raise ValueError("Trying to rebroadcast non-existent dimension")
        t = x.type.clone(
            broadcastable=[
                self.axis.get(i, b) for i, b in enumerate(x.type.broadcastable)
            ]
        )
        return Apply(self, [x], [t()])

    def perform(self, node, inp, out_):
        (x,) = inp
        (out,) = out_
        for axis, value in self.axis.items():
            if value and x.shape[axis] != 1:
                raise ValueError(
                    f"Dimension {axis} in Rebroadcast's input was"
                    f" supposed to be 1 (got {x.shape[axis]} instead)"
                )
        out[0] = x

    def grad(self, inp, grads):
        (x,) = inp
        (gz,) = grads
        # restore the broadcasting pattern of the input
        return (
            Rebroadcast(
                *[
                    (axis, x.type.broadcastable[axis])
                    for axis, value in self.axis.items()
                ]
            )(gz),
        )

    def infer_shape(self, fgraph, node, ishapes):
        assert len(ishapes) == 1
        l = []
        one = theano.tensor.basic.constant(1)
        for ax in range(len(ishapes[0])):
            if self.axis.get(ax, False):
                l.append(one)
            else:
                l.append(ishapes[0][ax])

        return [tuple(l)]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return [None]
        return self(*eval_points, **dict(return_list=True))

    def c_code(self, node, nodename, inp, out, sub):
        (iname,) = inp
        (oname,) = out
        fail = sub["fail"]

        itype = node.inputs[0].type.__class__
        if itype in self.c_code_and_version:
            code, version = self.c_code_and_version[itype]
            final_code = ""
            for axis, value in self.axis.items():
                if value:
                    final_code += code % locals()
            return (
                final_code
                + f"""
            Py_XDECREF({oname});
            {oname} = {iname};
            Py_XINCREF({oname});
            """
            )
        raise NotImplementedError()

    def c_code_cache_version(self):
        version = []
        # If any of the c code is unversionned, we have to return ()
        # Else, we will return a list of (type name, version) pairs.
        for t, (c, v) in sorted(
            self.c_code_and_version.items(), key=lambda pair: str(pair[0])
        ):
            if not v:
                warnings.warn(
                    f"Type {t} has C code for Rebroadcast, but it "
                    "has no version. You should add a 'version' "
                    "keyword arg when calling "
                    "register_rebroadcast_c_code.",
                    stacklevel=2,
                )
                return ()
            version.append((str(t), v))

        if version:
            version.append(1)
        return tuple(version)
