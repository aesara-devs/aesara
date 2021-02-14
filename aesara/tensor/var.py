import copy
import traceback as tb
import warnings
from collections.abc import Iterable
from numbers import Number
from typing import Optional

import numpy as np

from aesara import tensor as at
from aesara.configdefaults import config
from aesara.graph.basic import Constant, Variable
from aesara.graph.utils import MetaType
from aesara.scalar import ComplexError, IntegerDivisionError
from aesara.tensor import _get_vector_length, as_tensor_variable
from aesara.tensor.exceptions import AdvancedIndexingError
from aesara.tensor.type import TensorType
from aesara.tensor.utils import hash_from_ndarray


class _tensor_py_operators:
    def __abs__(self):
        return at.math.abs(self)

    def __neg__(self):
        return at.math.neg(self)

    # These won't work because Python requires an int return value
    # def __int__(self): return convert_to_int32(self)
    # def __float__(self): return convert_to_float64(self)
    # def __complex__(self): return convert_to_complex128(self)

    _is_nonzero = True

    def __lt__(self, other):
        rval = at.math.lt(self, other)
        rval._is_nonzero = False
        return rval

    def __le__(self, other):
        rval = at.math.le(self, other)
        rval._is_nonzero = False
        return rval

    def __gt__(self, other):
        rval = at.math.gt(self, other)
        rval._is_nonzero = False
        return rval

    def __ge__(self, other):
        rval = at.math.ge(self, other)
        rval._is_nonzero = False
        return rval

    def __bool__(self):
        # This is meant to prohibit stuff like a < b < c, which is internally
        # implemented as (a < b) and (b < c). The trouble with this is the
        # side-effect that checking for a non-NULL a by typing "if a: ..."
        # uses the same __nonzero__ method.  We want these both to work, but
        # it seems impossible.  Currently, all vars evaluate to nonzero except
        # the return values of comparison operators, which raise this
        # exception.  If you can think of a better solution, go for it!
        #
        # __bool__ is Python 3.x data model. __nonzero__ is Python 2.x.
        if self._is_nonzero:
            return True
        else:
            raise TypeError("Variables do not support boolean operations.")

    def __invert__(self):
        return at.math.invert(self)

    def __and__(self, other):
        return at.math.and_(self, other)

    def __or__(self, other):
        return at.math.or_(self, other)

    def __xor__(self, other):
        return at.math.xor(self, other)

    def __rand__(self, other):
        return at.math.and_(other, self)

    def __ror__(self, other):
        return at.math.or_(other, self)

    def __rxor__(self, other):
        return at.math.xor(other, self)

    # def __iand__(self, other):
    #    return _and_inplace(self, other)
    #
    # def __ior__(self, other):
    #    return _or_inplace(self, other)
    #
    # def __ixor__(self, other):
    #    return _xor_inplace(self, other)

    def __add__(self, other):
        try:
            return at.math.add(self, other)
        # We should catch the minimum number of exception here.
        # Otherwise this will convert error when Aesara flags
        # compute_test_value is used
        # Evidently, we need to catch NotImplementedError
        # TypeError from as_tensor_variable are caught in Elemwise.make_node
        # Otherwise TensorVariable * SparseVariable won't work!
        except (NotImplementedError, TypeError):
            # We must return NotImplemented and not an
            # NotImplementedError or raise an NotImplementedError.
            # That way python will give a good error message like this
            # `TypeError: unsupported operand type(s) for +:
            # 'TensorVariable' and 'TensorVariable'`
            return NotImplemented

    def __sub__(self, other):
        # See explanation in __add__ for the error caught
        # and the return value in that case
        try:
            return at.math.sub(self, other)
        except (NotImplementedError, TypeError):
            return NotImplemented

    def __mul__(self, other):
        # See explanation in __add__ for the error caught
        # and the return value in that case
        try:
            return at.math.mul(self, other)
        except (NotImplementedError, TypeError):
            return NotImplemented

    def __div__(self, other):
        # See explanation in __add__ for the error caught
        # and the return value in that case
        try:
            return at.math.div_proxy(self, other)
        except IntegerDivisionError:
            # This is to raise the exception that occurs when trying to divide
            # two integer arrays (currently forbidden).
            raise
        except (NotImplementedError, TypeError):
            return NotImplemented

    def __pow__(self, other):
        # See explanation in __add__ for the error caught
        # and the return value in that case
        try:
            return at.math.pow(self, other)
        except (NotImplementedError, TypeError):
            return NotImplemented

    def __mod__(self, other):
        # See explanation in __add__ for the error caught
        # and the return value in that case
        try:
            return at.math.mod_check(self, other)
        except ComplexError:
            # This is to raise the exception that occurs when trying to compute
            # x % y with either x or y a complex number.
            raise
        except (NotImplementedError, TypeError):
            return NotImplemented

    def __divmod__(self, other):
        return at.math.divmod(self, other)

    def __truediv__(self, other):
        return at.math.true_div(self, other)

    def __floordiv__(self, other):
        return at.math.floor_div(self, other)

    def __rtruediv__(self, other):
        return at.math.true_div(other, self)

    def __rfloordiv__(self, other):
        return at.math.floor_div(other, self)

    # Do not use these; in-place `Op`s should be inserted by optimizations
    # only!
    # def __iadd__(self, other):
    #    return _add_inplace(self, other)
    # def __isub__(self, other):
    #    return _sub_inplace(self, other)
    #
    # def __imul__(self, other):
    #    return _mul_inplace(self, other)
    #
    # def __idiv__(self, other):
    #    return _div_inplace(self, other)
    #
    # def __ipow__(self, other):
    #    return _pow_inplace(self, other)

    def __radd__(self, other):
        return at.math.add(other, self)

    def __rsub__(self, other):
        return at.math.sub(other, self)

    def __rmul__(self, other):
        return at.math.mul(other, self)

    def __rdiv__(self, other):
        return at.math.div_proxy(other, self)

    def __rmod__(self, other):
        return at.math.mod(other, self)

    def __rdivmod__(self, other):
        return at.math.divmod(other, self)

    def __rpow__(self, other):
        return at.math.pow(other, self)

    def __ceil__(self):
        return at.math.ceil(self)

    def __floor__(self):
        return at.math.floor(self)

    def __trunc__(self):
        return at.math.trunc(self)

    # NumPy-like transpose property
    @property
    def T(self):
        return at.basic.transpose(self)

    def transpose(self, *axes):
        """Transpose this array.

        Returns
        -------
        object
            `tensor.transpose(self, axes)` or `tensor.transpose(self, axes[0])`.

        If only one `axes` argument is provided and it is iterable, then it is
        assumed to be the entire axes tuple, and passed intact to
        tensor.transpose.

        """
        if len(axes) == 0:
            return at.basic.transpose(self)
        try:
            iter(axes[0])
            iterable = True
        except TypeError:
            iterable = False
        if len(axes) == 1 and iterable:
            return at.basic.transpose(self, axes[0])
        else:
            return at.basic.transpose(self, axes)

    @property
    def shape(self):
        if not any(s is None for s in self.type.shape):
            return as_tensor_variable(self.type.shape, ndim=1, dtype=np.int64)

        return at.shape(self)

    @property
    def size(self):
        if self.ndim == 1:
            return self.shape[0]
        else:
            return at.math.prod(self.shape)

    def any(self, axis=None, keepdims=False):
        return at.math.any(self, axis=axis, keepdims=keepdims)

    def all(self, axis=None, keepdims=False):
        return at.math.all(self, axis=axis, keepdims=keepdims)

    # Old note: "We can't implement this because Python requests that this
    # function returns an integer."
    # TODO: We could use `get_vector_length` and let it raise an exception just like
    # `__iter__` does
    # def __len__(self):
    #     raise Exception("Aesara Variables can't work with len(Aesara "
    #                     "Variable) due to Python restriction. You can use "
    #                     "AesaraVariable.shape[0] instead.")

    def reshape(self, shape, ndim=None):
        """Return a reshaped view/copy of this variable.

        Parameters
        ----------
        shape
            Something that can be converted to a symbolic vector of integers.
        ndim
            The length of the shape. Passing None here means for
            Aesara to try and guess the length of `shape`.


        .. warning:: This has a different signature than numpy's
                     ndarray.reshape!
                     In numpy you do not need to wrap the shape arguments
                     in a tuple, in aesara you do need to.

        """
        if ndim is not None:
            if not isinstance(ndim, int):
                raise ValueError(
                    "Expected ndim to be an integer, is " + str(type(ndim))
                )

        return at.reshape(self, shape, ndim=ndim)

    def dimshuffle(self, *pattern):
        """
        Reorder the dimensions of this variable, optionally inserting
        broadcasted dimensions.

        Parameters
        ----------
        pattern
            List/tuple of int mixed with 'x' for broadcastable dimensions.

        Examples
        --------
        For example, to create a 3D view of a [2D] matrix, call
        ``dimshuffle([0,'x',1])``.  This will create a 3D view such that the
        middle dimension is an implicit broadcasted dimension.  To do the same
        thing on the transpose of that matrix, call ``dimshuffle([1, 'x', 0])``.

        Notes
        -----
        This function supports the pattern passed as a tuple, or as a
        variable-length argument (e.g. ``a.dimshuffle(pattern)`` is equivalent
        to ``a.dimshuffle(*pattern)`` where ``pattern`` is a list/tuple of ints
        mixed with 'x' characters).

        See Also
        --------
        DimShuffle

        """
        if (len(pattern) == 1) and (isinstance(pattern[0], (list, tuple))):
            pattern = pattern[0]
        op = at.elemwise.DimShuffle(list(self.type.broadcastable), pattern)
        return op(self)

    def flatten(self, ndim=1):
        return at.basic.flatten(self, ndim)

    def ravel(self):
        return at.basic.flatten(self)

    def diagonal(self, offset=0, axis1=0, axis2=1):
        return at.basic.diagonal(self, offset, axis1, axis2)

    def transfer(self, target):
        """Transfer this this array's data to another device.

        If `target` is `'cpu'` this will transfer to a TensorType (if
        not already one).  Other types may define additional targets.

        Parameters
        ----------
        target : str
            The desired location of the output variable
        """
        return at.basic.transfer(self, target)

    def arccos(self):
        return at.math.arccos(self)

    def arccosh(self):
        return at.math.arccosh(self)

    def arcsin(self):
        return at.math.arcsin(self)

    def arcsinh(self):
        return at.math.arcsinh(self)

    def arctan(self):
        return at.math.arctan(self)

    def arctanh(self):
        return at.math.arctanh(self)

    def ceil(self):
        return at.math.ceil(self)

    def cos(self):
        return at.math.cos(self)

    def cosh(self):
        return at.math.cosh(self)

    def deg2rad(self):
        return at.math.deg2rad(self)

    def exp(self):
        return at.math.exp(self)

    def exp2(self):
        return at.math.exp2(self)

    def expm1(self):
        return at.math.expm1(self)

    def floor(self):
        return at.math.floor(self)

    def log(self):
        return at.math.log(self)

    def log10(self):
        return at.math.log10(self)

    def log1p(self):
        return at.math.log1p(self)

    def log2(self):
        return at.math.log2(self)

    def rad2deg(self):
        return at.math.rad2deg(self)

    def sin(self):
        return at.math.sin(self)

    def sinh(self):
        return at.math.sinh(self)

    def sqrt(self):
        return at.math.sqrt(self)

    def tan(self):
        return at.math.tan(self)

    def tanh(self):
        return at.math.tanh(self)

    def trunc(self):
        return at.math.trunc(self)

    def astype(self, dtype):
        return at.basic.cast(self, dtype)

    def __getitem__(self, args):
        def includes_bool(args_el):
            if isinstance(args_el, (np.bool_, bool)) or (
                hasattr(args_el, "dtype") and args_el.dtype == "bool"
            ):
                return True
            if not isinstance(args_el, Variable) and isinstance(args_el, Iterable):
                for el in args_el:
                    if includes_bool(el):
                        return True
            return False

        if isinstance(args, list) and any(isinstance(a, slice) for a in args):
            pass
        elif not isinstance(args, tuple):
            args = (args,)

        # Count the dimensions, check for bools and find ellipses.
        ellipses = []
        index_dim_count = 0
        for i, arg in enumerate(args):
            if arg is np.newaxis:
                # no increase in index_dim_count
                pass
            elif arg is Ellipsis:
                # no increase in index_dim_count
                ellipses.append(i)
            elif (
                isinstance(arg, (np.ndarray, Variable))
                and hasattr(arg, "dtype")
                and arg.dtype == "bool"
            ):
                index_dim_count += arg.ndim
            else:
                # Python arrays can contain a mixture of bools and integers,
                # which requires complex rules to handle all special cases.
                # These rules differ slightly between NumPy versions.
                # Since earlier versions of Aesara did not support any boolean
                # indexing, it is safe to throw an error if we encounter
                # any of these difficult cases.
                if includes_bool(arg):
                    raise TypeError(
                        "TensorType does not support Python bools "
                        "for indexing, such as tensor[[True, False]]. "
                        "To use a boolean mask, convert the mask to "
                        "a NumPy array first, e.g., "
                        "tensor[numpy.array([True, False])]."
                    )
                index_dim_count += 1

        # Check if the number of dimensions isn't too large.
        if self.ndim < index_dim_count:
            raise IndexError("too many indices for array")

        # Convert an Ellipsis if provided into an appropriate number of
        # slice(None).
        if len(ellipses) > 1:
            raise IndexError("an index can only have a single Ellipsis (`...`)")
        elif len(ellipses) == 1:
            ellipsis_at = ellipses[0]
            args = list(args)
            args[ellipsis_at : ellipsis_at + 1] = [slice(None)] * (
                self.ndim - index_dim_count
            )

        def is_empty_array(val):
            return (isinstance(val, (tuple, list)) and len(val) == 0) or (
                isinstance(val, np.ndarray) and val.size == 0
            )

        # Force input to be int64 datatype if input is an empty list or tuple
        # Else leave it as is if it is a real number
        # Convert python literals to aesara constants
        args = tuple(
            [
                at.subtensor.as_index_constant(
                    np.array(inp, dtype=np.int64) if is_empty_array(inp) else inp
                )
                for inp in args
            ]
        )

        # Determine if advanced indexing is needed or not.  The logic is
        # already in `index_vars_to_types`: if it succeeds, standard indexing is
        # used; if it fails with `AdvancedIndexingError`, advanced indexing is
        # used
        advanced = False
        for i, arg in enumerate(args):
            if includes_bool(arg):
                advanced = True
                break

            if arg is not np.newaxis:
                try:
                    at.subtensor.index_vars_to_types(arg)
                except AdvancedIndexingError:
                    if advanced:
                        break
                    else:
                        advanced = True

        if advanced:
            return at.subtensor.advanced_subtensor(self, *args)
        else:
            if np.newaxis in args:
                # `np.newaxis` (i.e. `None`) in NumPy indexing mean "add a new
                # broadcastable dimension at this location".  Since Aesara adds
                # new broadcastable dimensions via the `DimShuffle` `Op`, the
                # following code uses said `Op` to add one of the new axes and
                # then uses recursion to apply any other indices and add any
                # remaining new axes.

                counter = 0
                pattern = []
                new_args = []
                for arg in args:
                    if arg == np.newaxis:
                        pattern.append("x")
                        new_args.append(slice(None, None, None))
                    else:
                        pattern.append(counter)
                        counter += 1
                        new_args.append(arg)

                pattern.extend(list(range(counter, self.ndim)))

                view = self.dimshuffle(pattern)
                full_slices = True
                for arg in new_args:
                    # We can't do arg == slice(None, None, None) as in
                    # Python 2.7, this call __lt__ if we have a slice
                    # with some symbolic variable.
                    if not (
                        isinstance(arg, slice)
                        and arg.start is None
                        and arg.stop is None
                        and arg.step is None
                    ):
                        full_slices = False
                if full_slices:
                    return view
                else:
                    return view.__getitem__(tuple(new_args))
            else:
                return at.subtensor.Subtensor(args)(
                    self,
                    *at.subtensor.get_slice_elements(
                        args, lambda entry: isinstance(entry, Variable)
                    ),
                )

    def take(self, indices, axis=None, mode="raise"):
        return at.subtensor.take(self, indices, axis, mode)

    def copy(self, name=None):
        """Return a symbolic copy and optionally assign a name.

        Does not copy the tags.
        """
        copied_variable = at.basic.tensor_copy(self)
        copied_variable.name = name
        return copied_variable

    def __iter__(self):
        try:
            for i in range(at.basic.get_vector_length(self)):
                yield self[i]
        except TypeError:
            # This prevents accidental iteration via sum(self)
            raise TypeError(
                "TensorType does not support iteration. "
                "Maybe you are using builtins.sum instead of "
                "aesara.tensor.math.sum? (Maybe .max?)"
            )

    @property
    def ndim(self):
        """The rank of this tensor."""
        return self.type.ndim

    @property
    def broadcastable(self):
        """
        The broadcastable signature of this tensor.

        See Also
        --------
        broadcasting

        """
        return self.type.broadcastable

    @property
    def dtype(self):
        """The dtype of this tensor."""
        return self.type.dtype

    def __dot__(left, right):
        return at.math.dense_dot(left, right)

    def __rdot__(right, left):
        return at.math.dense_dot(left, right)

    dot = __dot__
    __matmul__ = __dot__
    __rmatmul__ = __rdot__

    def sum(self, axis=None, dtype=None, keepdims=False, acc_dtype=None):
        """See `aesara.tensor.math.sum`."""
        return at.math.sum(
            self, axis=axis, dtype=dtype, keepdims=keepdims, acc_dtype=acc_dtype
        )

    def prod(self, axis=None, dtype=None, keepdims=False, acc_dtype=None):
        """See `aesara.tensor.math.prod`."""
        return at.math.prod(
            self, axis=axis, dtype=dtype, keepdims=keepdims, acc_dtype=acc_dtype
        )

    def norm(self, L, axis=None, keepdims=False):
        if L == 0:
            raise NotImplementedError()
        if np.isinf(L):
            raise NotImplementedError()
        # optimizations will/should catch cases like L=1, L=2
        y = at.math.pow(
            at.math.pow(at.math.abs(self), L).sum(axis=axis),
            1.0 / L,
        )
        if keepdims:
            return at.math.makeKeepDims(self, y, axis)
        else:
            return y

    def mean(self, axis=None, dtype=None, keepdims=False, acc_dtype=None):
        """See `aesara.tensor.math.mean`."""
        return at.math.mean(
            self, axis=axis, dtype=dtype, keepdims=keepdims, acc_dtype=acc_dtype
        )

    def var(self, axis=None, ddof=0, keepdims=False, corrected=False):
        """See `aesara.tensor.math.var`."""
        return at.math.var(
            self, axis=axis, ddof=ddof, keepdims=keepdims, corrected=corrected
        )

    def std(self, axis=None, ddof=0, keepdims=False, corrected=False):
        """See `aesara.tensor.math.std`."""
        return at.math.std(
            self, axis=axis, ddof=ddof, keepdims=keepdims, corrected=corrected
        )

    def min(self, axis=None, keepdims=False):
        """See `aesara.tensor.math.min`."""
        return at.math.min(self, axis, keepdims=keepdims)

    def max(self, axis=None, keepdims=False):
        """See `aesara.tensor.math.max`."""
        return at.math.max(self, axis, keepdims=keepdims)

    def argmin(self, axis=None, keepdims=False):
        """See `aesara.tensor.math.argmin`."""
        return at.math.argmin(self, axis, keepdims=keepdims)

    def argmax(self, axis=None, keepdims=False):
        """See `aesara.tensor.math.argmax`."""
        return at.math.argmax(self, axis, keepdims=keepdims)

    def nonzero(self, return_matrix=False):
        """See `aesara.tensor.basic.nonzero`."""
        return at.nonzero(self, return_matrix=return_matrix)

    def nonzero_values(self):
        """See `aesara.tensor.basic.nonzero_values`."""
        return at.nonzero_values(self)

    def sort(self, axis=-1, kind="quicksort", order=None):
        """See `aesara.tensor.sort.sort`."""
        return at.sort(self, axis, kind, order)

    def argsort(self, axis=-1, kind="quicksort", order=None):
        """See `aesara.tensor.sort.argsort`."""
        from aesara.tensor.sort import argsort

        return argsort(self, axis, kind, order)

    def clip(self, a_min, a_max):
        "See `aesara.tensor.math.clip`."
        return at.math.clip(self, a_min, a_max)

    def conj(self):
        """See `aesara.tensor.math.conj`."""
        return at.math.conj(self)

    conjugate = conj

    def repeat(self, repeats, axis=None):
        """See `aesara.tensor.basic.repeat`."""
        return at.extra_ops.repeat(self, repeats, axis)

    def round(self, mode=None):
        """See `aesara.tensor.math.round`."""
        return at.math.round(self, mode)

    def trace(self):
        return at.linalg.trace(self)

    # This value is set so that Aesara arrays will trump NumPy operators.
    __array_priority__ = 1000

    def get_scalar_constant_value(self):
        return at.basic.get_scalar_constant_value(self)

    def zeros_like(model, dtype=None):
        return at.basic.zeros_like(model, dtype=dtype)

    def ones_like(model, dtype=None):
        return at.basic.ones_like(model, dtype=dtype)

    def cumsum(self, axis=None):
        return at.extra_ops.cumsum(self, axis)

    def cumprod(self, axis=None):
        return at.extra_ops.cumprod(self, axis)

    def searchsorted(self, v, side="left", sorter=None):
        return at.extra_ops.searchsorted(self, v, side, sorter)

    def ptp(self, axis=None):
        """See `aesara.tensor.math.ptp`."""

        return at.math.ptp(self, axis)

    def swapaxes(self, axis1, axis2):
        """See `aesara.tensor.basic.swapaxes`.

        If a matrix is provided with the right axes, its transpose
        will be returned.

        """
        return at.basic.swapaxes(self, axis1, axis2)

    def fill(self, value):
        """Fill inputted tensor with the assigned value."""
        return at.basic.fill(self, value)

    def choose(self, choices, out=None, mode="raise"):
        """
        Construct an array from an index array and a set of arrays to choose
        from.

        """
        return at.basic.choose(self, choices, out=None, mode="raise")

    def squeeze(self):
        """
        Remove broadcastable dimensions from the shape of an array.

        It returns the input array, but with the broadcastable dimensions
        removed. This is always `x` itself or a view into `x`.

        """
        return at.extra_ops.squeeze(self)

    def compress(self, a, axis=None):
        """Return selected slices only."""
        return at.extra_ops.compress(self, a, axis=axis)


class TensorVariable(_tensor_py_operators, Variable):
    """
    Subclass to add the tensor operators to the basic `Variable` class.

    """

    def __init__(self, type, owner=None, index=None, name=None):
        super().__init__(type, owner=owner, index=index, name=name)
        if config.warn_float64 != "ignore" and type.dtype == "float64":
            msg = (
                "You are creating a TensorVariable "
                "with float64 dtype. You requested an action via "
                "the Aesara flag warn_float64={ignore,warn,raise,pdb}."
            )
            if config.warn_float64 == "warn":
                # Get the user stack. We don't want function inside the
                # tensor and graph directory to be shown to the user.
                x = tb.extract_stack()
                nb_rm = 0
                while x:
                    file_path = x[-1][0]
                    rm = False
                    for p in [
                        "aesara/tensor/",
                        "aesara\\tensor\\",
                        "aesara/graph/",
                        "aesara\\tensor\\",
                    ]:
                        if p in file_path:
                            x = x[:-1]
                            nb_rm += 1
                            rm = True
                            break
                    if not rm:
                        break
                warnings.warn(msg, stacklevel=1 + nb_rm)
            elif config.warn_float64 == "raise":
                raise Exception(msg)
            elif config.warn_float64 == "pdb":
                import pdb

                pdb.set_trace()


@_get_vector_length.register(TensorVariable)
def _get_vector_length_TensorVariable(op_or_var, var):
    if var.type.shape[0] is None:
        raise ValueError(f"Length of {var} cannot be determined")
    return var.type.shape[0]


TensorType.variable_type = TensorVariable


class TensorConstantSignature(tuple):
    """
    A Signature object for comparing TensorConstant instances.

    An instance is a pair: (Type instance, ndarray).

    """

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        try:
            (t0, d0), (t1, d1) = self, other
        except Exception:
            return False

        # N.B. compare shape to ensure no broadcasting in ==
        if t0 != t1 or d0.shape != d1.shape:
            return False

        self.no_nan  # Ensure has_nan is computed.
        # Note that in the comparisons below, the elementwise comparisons
        # come last because they are the most expensive checks.
        if self.has_nan:
            other.no_nan  # Ensure has_nan is computed.
            return (
                other.has_nan
                and self.sum == other.sum
                and (self.no_nan.mask == other.no_nan.mask).all()
                and
                # Note that the second test below (==) may crash e.g. for
                # a single scalar NaN value, so we do not run it when all
                # values are missing.
                (self.no_nan.mask.all() or (self.no_nan == other.no_nan).all())
            )
        else:
            # Simple case where we do not need to worry about NaN values.
            # (note that if there are NaN values in d1, this will return
            # False, which is why we do not bother with testing `other.has_nan`
            # here).
            return (self.sum == other.sum) and np.all(d0 == d1)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        t, d = self
        return hash((type(self), t, d.shape, self.sum))

    def aesara_hash(self):
        _, d = self
        return hash_from_ndarray(d)

    def _get_sum(self):
        """Compute sum of non NaN / Inf values in the array."""
        try:
            return self._sum
        except AttributeError:
            self._sum = self.no_nan.sum()
            # The following 2 lines are needede as in Python 3.3 with NumPy
            # 1.7.1, numpy.ndarray and numpy.memmap aren't hashable.
            if isinstance(self._sum, np.memmap):
                self._sum = np.asarray(self._sum).item()
            if self.has_nan and self.no_nan.mask.all():
                # In this case the sum is not properly computed by numpy.
                self._sum = 0
            if np.isinf(self._sum) or np.isnan(self._sum):
                # NaN may happen when there are both -inf and +inf values.
                if self.has_nan:
                    # Filter both NaN and Inf values.
                    mask = self.no_nan.mask + np.isinf(self[1])
                else:
                    # Filter only Inf values.
                    mask = np.isinf(self[1])
                if mask.all():
                    self._sum = 0
                else:
                    self._sum = np.ma.masked_array(self[1], mask).sum()
                # At this point there should be no more NaN.
                assert not np.isnan(self._sum)
        return self._sum

    sum = property(_get_sum)

    def _get_no_nan(self):
        try:
            return self._no_nan
        except AttributeError:
            nan_mask = np.isnan(self[1])
            if nan_mask.any():
                self._no_nan = np.ma.masked_array(self[1], nan_mask)
                self.has_nan = True
            else:
                self._no_nan = self[1]
                self.has_nan = False
        return self._no_nan

    no_nan = property(_get_no_nan)


def get_unique_value(x: TensorVariable) -> Optional[Number]:
    """Return the unique value of a tensor, if there is one"""
    if isinstance(x, Constant):
        data = x.data

        if isinstance(data, np.ndarray) and data.ndim > 0:
            flat_data = data.ravel()
            if flat_data.shape[0]:
                if (flat_data == flat_data[0]).all():
                    return flat_data[0]

    return None


class TensorConstant(TensorVariable, Constant):
    """Subclass to add the tensor operators to the basic `Constant` class."""

    def __init__(self, type, data, name=None):
        data_shape = np.shape(data)

        if len(data_shape) != type.ndim or any(
            ds != ts for ds, ts in zip(np.shape(data), type.shape) if ts is not None
        ):
            raise ValueError(
                f"Shape of data ({data_shape}) does not match shape of type ({type.shape})"
            )

        # We want all the shape information from `data`
        new_type = type.clone(shape=data_shape)

        assert not any(s is None for s in new_type.shape)

        Constant.__init__(self, new_type, data, name)

    def __str__(self):
        unique_val = get_unique_value(self)
        if unique_val is not None:
            val = f"{self.data.shape} of {unique_val}"
        else:
            val = f"{self.data}"
        if len(val) > 20:
            val = val[:10] + ".." + val[-10:]

        if self.name is not None:
            name = self.name
        else:
            name = "TensorConstant"
        return "%s{%s}" % (name, val)

    def signature(self):
        return TensorConstantSignature((self.type, self.data))

    def equals(self, other):
        # Override Constant.equals to allow to compare with
        # numpy.ndarray, and python type.
        if isinstance(other, (np.ndarray, int, float)):
            # Make a TensorConstant to be able to compare
            other = at.basic.constant(other)
        return (
            isinstance(other, TensorConstant) and self.signature() == other.signature()
        )

    def __copy__(self):
        # We need to do this to remove the cached attribute
        return type(self)(self.type, self.data, self.name)

    def __deepcopy__(self, memo):
        # We need to do this to remove the cached attribute
        return type(self)(
            copy.deepcopy(self.type, memo),
            copy.deepcopy(self.data, memo),
            copy.deepcopy(self.name, memo),
        )


TensorType.constant_type = TensorConstant


class DenseVariableMeta(MetaType):
    def __instancecheck__(self, o):
        if type(o) == TensorVariable or isinstance(o, DenseVariableMeta):
            return True
        return False


class DenseTensorVariable(TensorType, metaclass=DenseVariableMeta):
    r"""A `Variable` for dense tensors.

    Instances of this class and `TensorVariable`\s are considered dense
    `Variable`\s.
    """


class DenseConstantMeta(MetaType):
    def __instancecheck__(self, o):
        if type(o) == TensorConstant or isinstance(o, DenseConstantMeta):
            return True
        return False


class DenseTensorConstant(TensorType, metaclass=DenseConstantMeta):
    r"""A `Constant` for dense tensors.

    Instances of this class and `TensorConstant`\s are considered dense
    `Constant`\s.
    """
