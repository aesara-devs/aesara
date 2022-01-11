#
# Slice type and Op. None Type and NoneConst.
#

import numpy as np

import aesara
from aesara import _as_symbolic
from aesara.gradient import DisconnectedType
from aesara.graph.basic import Apply, Constant, Variable
from aesara.graph.op import Op
from aesara.graph.type import Generic, Type
from aesara.tensor.type import integer_dtypes


def as_int_none_variable(x):
    if x is None:
        return NoneConst
    elif NoneConst.equals(x):
        return x
    x = aesara.tensor.as_tensor_variable(x, ndim=0)
    if x.type.dtype not in integer_dtypes:
        raise TypeError("index must be integers")
    return x


class MakeSlice(Op):

    __props__ = ()

    def make_node(self, slc, stop=None, step=None):
        # We need to accept and handle in make_node inputs the node
        # inputs to allow redoing a new op elsewhere in the graph by
        # optimization.
        if isinstance(slc, slice):
            assert stop is None
            assert step is None
            inp = [slc.start, slc.stop, slc.step]
        else:
            inp = [slc, stop, step]
        return Apply(self, list(map(as_int_none_variable, inp)), [slicetype()])

    def perform(self, node, inp, out_):
        (out,) = out_
        out[0] = slice(*inp)

    def grad(self, inputs, grads):
        return [DisconnectedType()() for i in inputs]


make_slice = MakeSlice()


class SliceType(Type):
    def clone(self, **kwargs):
        return type(self)()

    def filter(self, x, strict=False, allow_downcast=None):
        if isinstance(x, slice):
            return x
        else:
            raise TypeError("Expected a slice!")

    def __str__(self):
        return "slice"

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    @staticmethod
    def may_share_memory(a, b):
        # Slices never shared memory between object
        return isinstance(a, slice) and a is b


slicetype = SliceType()


class SliceConstant(Constant):
    def __init__(self, type, data, name=None):
        assert isinstance(data, slice)
        # Numpy ndarray aren't hashable, so get rid of them.
        if isinstance(data.start, np.ndarray):
            assert data.start.ndim == 0
            assert str(data.start.dtype) in integer_dtypes
            data = slice(int(data.start), data.stop, data.step)
        elif isinstance(data.stop, np.ndarray):
            assert data.stop.ndim == 0
            assert str(data.stop.dtype) in integer_dtypes
            data = slice(data.start, int(data.stop), data.step)
        elif isinstance(data.step, np.ndarray):
            assert data.step.ndim == 0
            assert str(data.step.dtype) in integer_dtypes
            data = slice(data.start, int(data.stop), data.step)
        Constant.__init__(self, type, data, name)

    def signature(self):
        return (SliceConstant, self.data.start, self.data.stop, self.data.step)

    def __str__(self):
        return "{}{{{}, {}, {}}}".format(
            self.__class__.__name__,
            self.data.start,
            self.data.stop,
            self.data.step,
        )


SliceType.Constant = SliceConstant


@_as_symbolic.register(slice)
def as_symbolic_slice(x, **kwargs):

    if any(isinstance(i, Variable) for i in (x.start, x.stop, x.step)):
        return make_slice(x)

    return SliceConstant(slicetype, x)


class NoneTypeT(Generic):
    """
    Inherit from Generic to have c code working.

    """

    def filter(self, x, strict=False, allow_downcast=None):
        if x is None:
            return x
        else:
            raise TypeError("Expected None!")

    @staticmethod
    def may_share_memory(a, b):
        # None never share memory between object, in the sense of DebugMode.
        # Python None are singleton
        return False


none_type_t = NoneTypeT()

NoneConst = Constant(none_type_t, None, name="NoneConst")


@_as_symbolic.register(type(None))
def as_symbolic_None(x, **kwargs):
    return NoneConst


__all__ = ["make_slice", "slicetype", "none_type_t", "NoneConst"]
