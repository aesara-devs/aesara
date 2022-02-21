import logging
import sys
from itertools import chain, groupby
from textwrap import dedent
from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np

import aesara
from aesara import scalar as aes
from aesara.configdefaults import config
from aesara.gradient import DisconnectedType
from aesara.graph.basic import Apply, Constant, Variable
from aesara.graph.op import COp, Op
from aesara.graph.params_type import ParamsType
from aesara.graph.type import Type
from aesara.graph.utils import MethodNotDefined
from aesara.misc.safe_asarray import _asarray
from aesara.printing import Printer, pprint, set_precedence
from aesara.scalar.basic import ScalarConstant
from aesara.tensor import _get_vector_length, as_tensor_variable, get_vector_length
from aesara.tensor.basic import addbroadcast, alloc, get_scalar_constant_value
from aesara.tensor.elemwise import DimShuffle
from aesara.tensor.exceptions import (
    AdvancedIndexingError,
    NotScalarConstantError,
    ShapeError,
)
from aesara.tensor.math import clip
from aesara.tensor.shape import Reshape
from aesara.tensor.type import (
    TensorType,
    bscalar,
    complex_dtypes,
    cscalar,
    discrete_dtypes,
    dscalar,
    fscalar,
    integer_dtypes,
    iscalar,
    lscalar,
    tensor,
    wscalar,
    zscalar,
)
from aesara.tensor.type_other import NoneConst, NoneTypeT, SliceType, make_slice


_logger = logging.getLogger("aesara.tensor.subtensor")

invalid_scal_types = (aes.float64, aes.float32, aes.float16)
scal_types = (aes.int64, aes.int32, aes.int16, aes.int8)
tensor_types = (
    lscalar,
    iscalar,
    wscalar,
    bscalar,
)
invalid_tensor_types = (
    fscalar,
    dscalar,
    cscalar,
    zscalar,
)


def indices_from_subtensor(
    op_indices: Iterable[ScalarConstant],
    idx_list: Optional[List[Union[Type, slice, Variable]]],
) -> Tuple[Union[slice, Variable]]:
    """Recreate the index tuple from which a ``*Subtensor**`` ``Op`` was created.

    Parameters
    ==========
    op_indices
        The flattened indices obtained from ``x.inputs``, when ``x`` is a
        ``*Subtensor*`` node.
    idx_list
        The values describing the types of each dimension's index.  This is
        obtained from ``op.idx_list``, when ``op`` is a ``*Subtensor*``
        ``Op``.

    Example
    =======
        array, *op_indices = subtensor_node.inputs
        idx_list = getattr(subtensor_node.op, "idx_list", None)
        indices = indices_from_subtensor(op_indices, idx_list)

    """

    def convert_indices(indices, entry):
        """Reconstruct ``*Subtensor*`` index input parameter entries."""
        if indices and isinstance(entry, Type):
            rval = indices.pop(0)
            return rval
        elif isinstance(entry, slice):
            return slice(
                convert_indices(indices, entry.start),
                convert_indices(indices, entry.stop),
                convert_indices(indices, entry.step),
            )
        else:
            return entry

    op_indices = list(op_indices)

    return (
        tuple(convert_indices(op_indices, idx) for idx in idx_list)
        if idx_list
        else tuple(op_indices)
    )


def as_index_constant(
    a: Optional[Union[slice, int, np.integer, Variable]]
) -> Optional[Union[Variable, slice]]:
    r"""Convert Python literals to Aesara constants--when possible--in `Subtensor` arguments.

    This will leave `Variable`\s untouched.
    """
    if a is None:
        return a
    elif isinstance(a, slice):
        return slice(
            as_index_constant(a.start),
            as_index_constant(a.stop),
            as_index_constant(a.step),
        )
    elif isinstance(a, (int, np.integer)):
        return aes.ScalarConstant(aes.int64, a)
    elif not isinstance(a, Variable):
        return as_tensor_variable(a)
    else:
        return a


def as_index_literal(
    idx: Union[Variable, slice, type(np.newaxis)]
) -> Union[int, slice, type(np.newaxis)]:
    """Convert a symbolic index element to its Python equivalent.

    This is like the inverse of `as_index_constant`

    Raises
    ------
    NotScalarConstantError
    """
    if idx == np.newaxis or isinstance(getattr(idx, "type", None), NoneTypeT):
        return np.newaxis

    if isinstance(idx, Constant):
        return idx.data.item() if isinstance(idx, np.ndarray) else idx.data

    if isinstance(getattr(idx, "type", None), SliceType):
        idx = slice(*idx.owner.inputs)

    if isinstance(idx, slice):
        return slice(
            as_index_literal(idx.start),
            as_index_literal(idx.stop),
            as_index_literal(idx.step),
        )

    raise NotScalarConstantError()


def get_idx_list(inputs, idx_list):
    return indices_from_subtensor(inputs[1:], idx_list)


def get_canonical_form_slice(
    theslice: Union[slice, Variable], length: Variable
) -> Tuple[Variable, int]:
    """Convert slices to canonical form.

    Given a slice [start:stop:step] transform it into a canonical form
    that respects the conventions imposed by python and numpy.

    In a canonical form a slice is represented by a canonical form slice,
    in which 0 <= start <= stop <= length and step > 0, and a flag which says
    if the resulting set of numbers needs to be reversed or not.

    """
    from aesara.tensor import ge, lt, sgn, switch

    if not isinstance(theslice, slice):
        try:
            value = as_index_literal(theslice)
        except NotScalarConstantError:
            value = theslice

        value = switch(lt(value, 0), (value + length), value)

        return value, 1

    def analyze(x):
        try:
            x_constant = as_index_literal(x)
            is_constant = True
        except NotScalarConstantError:
            x_constant = x
            is_constant = False
        return x_constant, is_constant

    start, is_start_constant = analyze(theslice.start)
    stop, is_stop_constant = analyze(theslice.stop)
    step, is_step_constant = analyze(theslice.step)
    length, is_length_constant = analyze(length)

    if step is None:
        step = 1
        is_step_constant = True

    # First handle the easier and common case where `step` is 1 and
    # either `start` or `stop` is a range boundary. More specializations
    # could be added later. This makes the resulting graph smaller than
    # in the generic case below.
    if step == 1:
        is_start_0 = (
            start is None
            or start == 0
            or (
                is_start_constant
                and is_length_constant
                and start < 0
                and start + length <= 0
            )
        )
        is_stop_length = (
            stop is None
            or stop in [length, sys.maxsize]
            or (is_stop_constant and is_length_constant and stop >= length)
        )
        if is_start_0:
            # 0:stop:1
            if is_stop_length:
                # Full slice.
                return slice(0, length, 1), 1
            if is_stop_constant and stop >= 0:
                return (slice(0, switch(lt(stop, length), stop, length), 1), 1)
            stop_plus_len = stop + length
            stop = switch(
                lt(stop, 0),
                # stop < 0
                switch(
                    lt(stop_plus_len, 0),
                    # stop + len < 0
                    0,
                    # stop + len >= 0
                    stop_plus_len,
                ),
                # stop >= 0: use min(stop, length)
                switch(lt(stop, length), stop, length),
            )
            return slice(0, stop, 1), 1
        elif is_stop_length:
            # start:length:1
            if is_start_constant and start >= 0:
                return slice(switch(lt(start, length), start, length), length, 1), 1
            start_plus_len = start + length
            start = switch(
                lt(start, 0),
                # start < 0
                switch(
                    lt(start_plus_len, 0),
                    # start + len < 0
                    0,
                    # start + len >= 0
                    start_plus_len,
                ),
                # start >= 0: use min(start, length)
                switch(lt(start, length), start, length),
            )
            return slice(start, length, 1), 1

    # This is the generic case.

    if is_step_constant:
        # When we know the sign of `step`, the graph can be made simpler.
        assert step != 0
        if step > 0:

            def switch_neg_step(a, b):
                return b

            abs_step = step
            sgn_step = 1
        else:

            def switch_neg_step(a, b):
                return a

            abs_step = -step
            sgn_step = -1
    else:
        is_step_neg = lt(step, 0)

        def switch_neg_step(a, b):
            return switch(is_step_neg, a, b)

        abs_step = abs(step)
        sgn_step = sgn(step)

    defstart = switch_neg_step(length - 1, 0)
    defstop = switch_neg_step(-1, length)
    if start is None:
        start = defstart
    else:
        start = switch(lt(start, 0), start + length, start)
        start = switch(lt(start, 0), switch_neg_step(-1, 0), start)
        start = switch(ge(start, length), switch_neg_step(length - 1, length), start)
    if stop is None or stop == sys.maxsize:
        # The special "maxsize" case is probably not needed here,
        # as slices containing maxsize are not generated by
        # __getslice__ anymore.
        stop = defstop
    else:
        stop = switch(lt(stop, 0), stop + length, stop)
        stop = switch(lt(stop, 0), -1, stop)
        stop = switch(ge(stop, length), length, stop)

    nw_stop = switch_neg_step(start + 1, stop)
    slice_len = (start - stop - 1) // abs_step + 1
    slice_len = switch(lt(slice_len, 0), 0, slice_len)
    neg_start = nw_stop - (slice_len - 1) * abs_step - 1
    neg_start = switch(lt(neg_start, 0), (nw_stop - 1), neg_start)
    nw_start = switch_neg_step(neg_start, start)
    nw_start = switch(lt(nw_start, 0), 0, nw_start)
    nw_stop = switch(lt(nw_stop, 0), 0, nw_stop)
    # Ensure start <= stop.
    nw_start = switch(lt(nw_start, nw_stop), nw_start, nw_stop)

    nw_step = abs_step
    if step != 1:
        reverse = sgn_step
        return slice(nw_start, nw_stop, nw_step), reverse
    else:
        return slice(nw_start, nw_stop, nw_step), 1


def range_len(slc):
    """Length of a `range` object.

    Adapted from CPython.

    """
    from aesara.tensor import and_, gt, lt, switch

    start, stop, step = tuple(
        as_index_constant(a) for a in [slc.start, slc.stop, slc.step]
    )
    return switch(
        and_(gt(step, 0), lt(start, stop)),
        1 + (stop - 1 - start) // step,
        switch(
            and_(lt(step, 0), gt(start, stop)),
            1 + (start - 1 - stop) // (-step),
            aes.ScalarConstant(aes.int64, 0),
        ),
    )


def slice_len(slc, n):
    """Compute the length of a slice for an array of a given length.

    We're essentially computing `len(range(*slc.indices(n)))`.

    """
    # TODO: Do we need to do this or should we expect `slc` to
    # already be canonicalized?
    canon_slc, _ = get_canonical_form_slice(slc, n)
    return range_len(canon_slc)


def is_basic_idx(idx):
    """Determine if an index is of the NumPy basic type.

    XXX: This only checks a single index, so an integers is *not* considered a
    basic index, because--depending on the other indices its used with--an
    integer can indicate advanced indexing.

    """
    return isinstance(idx, (slice, type(None))) or isinstance(
        getattr(idx, "type", None), (SliceType, NoneTypeT)
    )


def basic_shape(shape, indices):
    r"""Computes the shape resulting from basic NumPy indexing.

    Basic indices are either ``slice``\s or ``None``\s.  ``Ellipsis`` are not
    supported here; convert them to ``slice``\s first.

    Parameters
    ----------
    shape: Tuple[int]
        The shape of the array being indexed
    indices: Sequence[Or[slice, NoneType]]
        A sequence of basic indices used to index an array.

    """
    res_shape = ()
    for idx, n in zip(indices, shape):
        if isinstance(idx, slice):
            res_shape += (slice_len(idx, n),)
        elif isinstance(getattr(idx, "type", None), SliceType):
            if idx.owner:
                idx_inputs = idx.owner.inputs
            else:
                idx_inputs = (None,)
            res_shape += (slice_len(slice(*idx_inputs), n),)
        elif idx is None:
            res_shape += (aes.ScalarConstant(aes.int64, 1),)
        elif isinstance(getattr(idx, "type", None), NoneTypeT):
            res_shape += (aes.ScalarConstant(aes.int64, 1),)
        else:
            raise ValueError(f"Invalid index type: {idx}")
    return res_shape


def group_indices(indices):
    """Group indices sequentially by whether or not they're basic or advanced.

    Returns
    -------
    Tuple[Boolean, List[Tuple[Integer, Any]]]
        The boolean indicates whether or not the group is a set of basic
        indices.  The list contains the contiguous set of indices paired with their
        corresponding dimension number in the array being indexed.
    """
    idx_groups = []
    dim_num = -1
    for basic, grp_indices in groupby(indices, key=is_basic_idx):
        enum_grp_indices = []
        for idx in grp_indices:
            # We "zip" the dimension number to each index, which means we can't
            # count indices that add new axes
            if (idx is not None) and not isinstance(
                getattr(idx, "type", None), NoneTypeT
            ):
                dim_num += 1

            enum_grp_indices.append((dim_num, idx))

        idx_groups.append((basic, enum_grp_indices))

    return idx_groups


def indexed_result_shape(array_shape, indices, indices_are_shapes=False):
    """Compute the symbolic shape resulting from `a[indices]` for `a.shape == array_shape`.

    This function uses NumPy's basic and advanced indexing logic.  It can also
    handle combinations of advanced and basic indices.

    Parameters
    ----------
    array_shape: Tuple[Variable]
        Shape of the array being indexed.
    indices: Sequence[Union[TensorVariable, Tuple[Union[None, slice, Variable]]]]
        Either the indices themselves or the shapes of each index--depending
        on the value of `indices_are_shapes`.
    indices_are_shapes: bool (Optional)
        Indicates whether or not the `indices` contains shape tuples instead of
        the actual index arrays.  If you use this approach, make sure that the
        broadcastable dimensions are (scalar) constants with the value `1`, or `1`
        exactly.
    """
    res_shape = ()

    remaining_dims = range(aesara.tensor.basic.get_vector_length(array_shape))
    idx_groups = group_indices(indices)

    if len(idx_groups) > 2 or len(idx_groups) > 1 and not idx_groups[0][0]:
        # Bring adv. index groups to the front and merge each group
        idx_groups = sorted(idx_groups, key=lambda x: x[0])
        idx_groups = groupby(
            chain.from_iterable(d_idx for _, d_idx in idx_groups),
            key=lambda x: is_basic_idx(x[1]),
        )

    for basic, grp_dim_indices in idx_groups:
        dim_nums, grp_indices = zip(*grp_dim_indices)
        remaining_dims = tuple(dim for dim in remaining_dims if dim not in dim_nums)

        if basic:
            grp_shapes = tuple(array_shape[dim] for dim in dim_nums)
            res_shape += basic_shape(grp_shapes, grp_indices)
        else:
            from aesara.tensor.extra_ops import broadcast_shape

            res_shape += broadcast_shape(
                *grp_indices, arrays_are_shapes=indices_are_shapes
            )

    res_shape += tuple(array_shape[dim] for dim in remaining_dims)

    return res_shape


def get_slice_elements(idxs: List, cond: Callable) -> List:
    """Extract slice elements conditional on a given predicate function.

    Parameters
    ----------
    idxs : a list of indices or slices.
    cond : a callable that returns a bool

    Returns
    -------
    list
        idxs, with the slices flattened out into a list.
        If cond is true for an entry, does not flatten it.

    """
    ret = []

    def helper(entry):
        if cond(entry):
            ret.append(entry)
        elif isinstance(entry, slice):
            helper(entry.start)
            helper(entry.stop)
            helper(entry.step)

    for idx in idxs:
        helper(idx)

    return ret


def index_vars_to_types(entry, slice_ok=True):
    r"""Change references to `Variable`s into references to `Type`s.

    The `Subtensor.idx_list` field is unique to each `Subtensor` instance.  It
    is not unique to each `Apply` node, so it should not refer to specific
    `Variable`s.

    TODO WRITEME: This function also accepts an `entry` already being a `Type`;
    when would that happen?

    """
    if (
        isinstance(entry, (np.ndarray, Variable))
        and hasattr(entry, "dtype")
        and entry.dtype == "bool"
    ):
        raise AdvancedIndexingError("Invalid index type or slice for Subtensor")

    if isinstance(entry, Variable) and (
        entry.type in invalid_scal_types or entry.type in invalid_tensor_types
    ):
        raise TypeError("Expected an integer")

    if isinstance(entry, Variable) and entry.type in scal_types:
        return entry.type
    elif isinstance(entry, Type) and entry in scal_types:
        return entry

    if (
        isinstance(entry, Variable)
        and entry.type in tensor_types
        and np.all(entry.type.broadcastable)
    ):
        return aes.get_scalar_type(entry.type.dtype)
    elif (
        isinstance(entry, Type)
        and entry in tensor_types
        and np.all(entry.broadcastable)
    ):
        return aes.get_scalar_type(entry.dtype)
    elif slice_ok and isinstance(entry, slice):
        a = entry.start
        b = entry.stop
        c = entry.step

        if a is not None:
            slice_a = index_vars_to_types(a, False)
        else:
            slice_a = None

        if b is not None and b != sys.maxsize:
            # The special "maxsize" case is probably not needed here,
            # as slices containing maxsize are not generated by
            # __getslice__ anymore.
            slice_b = index_vars_to_types(b, False)
        else:
            slice_b = None

        if c is not None:
            slice_c = index_vars_to_types(c, False)
        else:
            slice_c = None

        return slice(slice_a, slice_b, slice_c)
    elif isinstance(entry, (int, np.integer)):
        raise TypeError()
    else:
        raise AdvancedIndexingError("Invalid index type or slice for Subtensor")


def get_constant_idx(
    idx_list, inputs, allow_partial=False, only_process_constants=False, elemwise=True
):
    r"""Return an `idx_list` with its constant inputs replaced by their Python scalar equivalents.

    May raise `NotScalarConstantError` if the indices contain non-constant entries.

    If `allow_partial` is ``True``, then entries that are not constant will
    stay as their input variable rather than raising an exception.

    ``None`` entries are always left as-is.

    Parameters
    ----------
    only_process_constants
        If ``True``, we only attempt to obtain the value of an index/slice if
        it's directly constant and don't try to dig through `DimShuffle`\s,
        fills, `Alloc`\s, and other to figure out its value.

    Examples
    --------
    Example usage where `v` and `a` are appropriately typed Aesara variables :
    >>> b = a[v, 1:3]
    >>> b.owner.op.idx_list
    (Scalar(int64), slice(Scalar(int64), Scalar(int64), None))
    >>> get_constant_idx(b.owner.op.idx_list, b.owner.inputs, allow_partial=True)
    [v, slice(1, 3, None)]
    >>> get_constant_idx(b.owner.op.idx_list, b.owner.inputs)
    NotScalarConstantError: v

    """
    real_idx = get_idx_list(inputs, idx_list)

    # TODO: Combine this with `as_index_literal`
    def conv(val):
        if val is None:
            return None
        elif isinstance(val, slice):
            return slice(conv(val.start), conv(val.stop), conv(val.step))
        else:
            try:
                return get_scalar_constant_value(
                    val,
                    only_process_constants=only_process_constants,
                    elemwise=elemwise,
                )
            except NotScalarConstantError:
                if allow_partial:
                    return val
                else:
                    raise

    return list(map(conv, real_idx))


def as_nontensor_scalar(a: Variable) -> aes.ScalarVariable:
    """Convert a value to a `Scalar` variable."""
    # Since aes.as_scalar does not know about tensor types (it would
    # create a circular import) , this method converts either a
    # TensorVariable or a ScalarVariable to a scalar.
    if isinstance(a, Variable) and isinstance(a.type, TensorType):
        return aesara.tensor.scalar_from_tensor(a)
    else:
        return aes.as_scalar(a)


class Subtensor(COp):
    """Basic NumPy indexing operator."""

    check_input = False
    view_map = {0: [0]}
    _f16_ok = True
    __props__ = ("idx_list",)

    def __init__(self, idx_list):
        # TODO: Provide the type of `self.idx_list`
        self.idx_list = tuple(map(index_vars_to_types, idx_list))

    def make_node(self, x, *inputs):
        """
        Parameters
        ----------
        x
            The tensor to take a subtensor of.
        inputs
            A list of aesara Scalars.

        """
        x = as_tensor_variable(x)
        inputs = tuple(as_nontensor_scalar(a) for a in inputs)

        idx_list = list(self.idx_list)
        if len(idx_list) > x.type.ndim:
            raise IndexError("too many indices for array")

        input_types = get_slice_elements(
            idx_list, lambda entry: isinstance(entry, Type)
        )

        assert len(inputs) == len(input_types)

        for input, expected_type in zip(inputs, input_types):
            if not expected_type.is_super(input.type):
                raise TypeError(
                    f"Incompatible types for Subtensor template. Expected {input.type}, got {expected_type}."
                )

        # infer the broadcasting pattern
        padded = get_constant_idx(
            self.idx_list, (None,) + inputs, allow_partial=True
        ) + [slice(None, None, None)] * (x.type.ndim - len(idx_list))
        broadcastable = []
        for i, (p, bc) in enumerate(zip(padded, x.type.broadcastable)):
            if isinstance(p, slice):
                if bc:
                    start = p.start
                    try:
                        start = get_scalar_constant_value(start)
                    except NotScalarConstantError:
                        pass
                    if start is None or start == 0:
                        start = p.start
                        if start is None:
                            start = 0
                        if p.stop is None or (
                            isinstance(p.stop, (int, np.integer, np.ndarray))
                            and p.stop > start
                        ):
                            broadcastable.append(True)
                            continue

                broadcastable.append(False)

        return Apply(
            self,
            (x,) + inputs,
            [tensor(dtype=x.type.dtype, shape=broadcastable)],
        )

    def perform(self, node, inputs, out_):
        (out,) = out_
        x = inputs[0]

        cdata = get_idx_list(inputs, self.idx_list)
        if len(cdata) == 1:
            cdata = cdata[0]

        out[0] = np.asarray(x.__getitem__(cdata))

    def infer_shape(self, fgraph, node, shapes):
        xshp = shapes[0]
        assert len(xshp) == node.inputs[0].ndim
        outshp = []
        actual_idx_list = list(get_idx_list(node.inputs, self.idx_list))
        padded = actual_idx_list + [slice(None, None, None)] * (
            len(xshp) - len(self.idx_list)
        )
        i = 0
        for idx, xl in zip(padded, xshp):
            if isinstance(idx, slice):
                # If it is the default (None, None, None) slice, or a variant,
                # the shape will be xl
                if (
                    (idx.start in [None, 0])
                    and (idx.stop in [None, sys.maxsize])
                    and (idx.step is None or idx.step == 1)
                ):
                    outshp.append(xl)
                else:
                    cnf = get_canonical_form_slice(idx, xl)[0]
                    if cnf.step == 1:
                        length = cnf.stop - cnf.start
                    else:
                        length = (cnf.stop - cnf.start - 1) // cnf.step + 1
                    outshp.append(length)
                i += 1
            else:
                # That dimension is dropped
                pass
        assert i == node.outputs[0].ndim
        assert len(outshp) == node.outputs[0].ndim
        return [outshp]

    def grad(self, inputs, grads):
        (gz,) = grads
        x = inputs[0]
        rest = inputs[1:]
        if x.dtype in discrete_dtypes:
            first = x.zeros_like().astype(config.floatX)
        else:
            # For best optimization, we let this as an inc.
            # This allow the opt local_IncSubtensor_serialize to apply first.
            # We have an optimization that will convert this to a
            # set subtensor here at:
            # aesara/tensor/opt.py:local_incsubtensor_of_zeros_to_setsubtensor()
            first = IncSubtensor(self.idx_list)(x.zeros_like(), gz, *rest)
        return [first] + [DisconnectedType()()] * len(rest)

    def connection_pattern(self, node):

        rval = [[True]]

        for ipt in node.inputs[1:]:
            rval.append([False])

        return rval

    def __hash__(self):
        msg = []
        for entry in self.idx_list:
            if isinstance(entry, slice):
                msg += [(entry.start, entry.stop, entry.step)]
            else:
                msg += [entry]

        idx_list = tuple(msg)
        # backport
        # idx_list = tuple((entry.start, entry.stop, entry.step)
        #                 if isinstance(entry, slice)
        #                 else entry
        #                 for entry in self.idx_list)
        return hash(idx_list)

    @staticmethod
    def str_from_slice(entry):
        msg = []
        for x in [entry.start, entry.stop, entry.step]:
            if x is None:
                msg.append("")
            else:
                msg.append(str(x))
        return ":".join(msg)

    def __str__(self):
        indices = []
        for entry in self.idx_list:
            if isinstance(entry, slice):
                indices.append(self.str_from_slice(entry))
            else:
                indices.append(str(entry))
        return f"{self.__class__.__name__}{{{', '.join(indices)}}}"

    @staticmethod
    def default_helper_c_code_args():
        """
        Returns a dictionary of default arguments to helper_c_code.

        """

        return {"c_prefix": "PyArray", "strides_mul": 1}

    @staticmethod
    def helper_c_code(
        node,
        name,
        inputs,
        outputs,
        sub,
        idx_list,
        view_ndim,
        c_prefix=None,
        strides_mul=None,
    ):
        """
        The parameters c_prefix are there to allow reusing this
        function on PyArray and GpuArray object.

        This fct take as input the x.

        """

        default_args = Subtensor.default_helper_c_code_args()

        if strides_mul is None:
            strides_mul = default_args["strides_mul"]

        if c_prefix is None:
            c_prefix = default_args["c_prefix"]

        #
        # two arrays are created in C code:
        # is_slice: len == ndim, 0 means int, 1 means slice
        # subtensor_spec: len = n_ints + 3 * n_slices
        #
        fail = sub["fail"]
        init_cmds = []  # initialization for subtensor_spec
        is_slice = []
        # TODO: change that, it might lead to unexpected results,
        # see assembla-#767
        NONE_CODE = sys.maxsize - 1

        pos = [0, 1]  # annoying version of global variable for init_entry

        def inc_spec_pos(amt):
            pos[0] += amt

        def inc_input_pos(amt):
            pos[1] += amt

        def spec_pos():
            return pos[0]

        def input_pos():
            return pos[1]

        def init_entry(entry, depth=0):
            if isinstance(entry, (np.integer, int)):
                init_cmds.append("subtensor_spec[%i] = %i;" % (spec_pos(), entry))
                inc_spec_pos(1)
                if depth == 0:
                    is_slice.append(0)
            elif isinstance(entry, Type):
                init_cmds.append(
                    "subtensor_spec[%i] = %s;" % (spec_pos(), inputs[input_pos()])
                )
                inc_spec_pos(1)
                inc_input_pos(1)
                if depth == 0:
                    is_slice.append(0)
            elif entry is None:
                init_cmds.append("subtensor_spec[%i] = %i;" % (spec_pos(), NONE_CODE))
                inc_spec_pos(1)
                if depth == 0:
                    is_slice.append(0)
            elif depth == 0 and isinstance(entry, slice):
                init_entry(entry.start, depth + 1)
                init_entry(entry.stop, depth + 1)
                init_entry(entry.step, depth + 1)
                is_slice.append(1)
            else:
                assert 0, entry

        for entry in idx_list:
            init_entry(entry)
        # make sure we used all inputs
        assert input_pos() == len(inputs), input_pos()
        assert len(is_slice) <= node.inputs[0].ndim, node.inputs[0].ndim

        len_is_slice = len(is_slice)

        len_subtensor_spec = spec_pos()
        subensor_spec = f"npy_intp subtensor_spec[{len_subtensor_spec}];"
        if len_subtensor_spec == 0:
            subensor_spec = "npy_intp * subtensor_spec = NULL;"

        if is_slice:
            is_slice_init = (
                "int is_slice[] = {" + ",".join([str(s) for s in is_slice]) + "};"
            )
        else:
            is_slice_init = "int* is_slice = NULL;"
        subtensor_init = "\n".join(init_cmds)

        (x,) = inputs[:1]
        (z,) = outputs

        if view_ndim:
            rval = f"""
        // Argument of the view
        npy_intp xview_dims[{view_ndim}];
        npy_intp xview_strides[{view_ndim}];

        """
        else:
            rval = """
        // Argument of the view
        npy_intp* xview_dims = NULL;
        npy_intp* xview_strides = NULL;

        """

        rval += (
            """
        // One more argument of the view
        npy_intp xview_offset = 0;

        // The subtensor is created by iterating over the dimensions
        // and updating stride, shape, and data pointers

        %(is_slice_init)s
        %(subensor_spec)s
        %(subtensor_init)s;
        int spec_pos = 0; //position in subtensor_spec
        int inner_ii = 0; // the current dimension of zview
        int outer_ii = 0; // current dimension of z


        for (; outer_ii < %(len_is_slice)s; ++outer_ii)
        {
            if (is_slice[outer_ii])
            {
                npy_intp length = %(c_prefix)s_DIMS(%(x)s)[outer_ii];
                npy_intp slicelength;
                npy_intp start = subtensor_spec[spec_pos+0];
                npy_intp stop  = subtensor_spec[spec_pos+1];
                npy_intp step  = subtensor_spec[spec_pos+2];
                if (step == %(NONE_CODE)s) step = 1;

                npy_intp defstart = step < 0 ? length-1 : 0;
                npy_intp defstop = step < 0 ? -1 : length;

                // logic adapted from
                // PySlice_GetIndicesEx in python source
                if (!step)
                {
                    PyErr_Format(PyExc_ValueError,
                                 "slice step cannot be zero");
                    %(fail)s;
                }

                if (start == %(NONE_CODE)s)
                {
                    start = defstart;
                }
                else
                {
                    if (start < 0) start += length;
                    if (start < 0) start = (step < 0) ? -1 : 0;
                    if (start >= length)
                        start = (step < 0) ? length - 1 : length;
                }

                if (stop == %(NONE_CODE)s)
                {
                    stop = defstop;
                }
                else
                {
                    if (stop < 0) stop += length;
                    if (stop < 0) stop = (step < 0) ? -1 : 0;
                    if (stop >= length)
                        stop = (step < 0) ? length - 1 : length;
                }

                if ((step < 0 && stop >= start)
                    || (step > 0 && start >= stop)) {
                    slicelength = 0;
                }
                else if (step < 0) {
                    slicelength = (stop-start+1)/step+1;
                }
                else {
                    slicelength = (stop-start-1)/step+1;
                }

                if (0){
                    fprintf(stdout, "start %%zi\\n", start);
                    fprintf(stdout, "stop %%zi\\n", stop);
                    fprintf(stdout, "step %%zi\\n", step);
                    fprintf(stdout, "length %%zi\\n", length);
                    fprintf(stdout, "slicelength %%zi\\n", slicelength);
                }

                assert (slicelength <= length);

                xview_offset += (npy_intp)%(c_prefix)s_STRIDES(%(x)s)[outer_ii]
                    * start * %(strides_mul)s;
                xview_dims[inner_ii] = slicelength;
                xview_strides[inner_ii] = (npy_intp)%(c_prefix)s_STRIDES(%(x)s)[outer_ii] * step;

                inner_ii += 1;
                spec_pos += 3;
            }
            else // tuple coord `outer_ii` is an int
            {
                int idx = subtensor_spec[spec_pos];
                if (idx < 0) idx += %(c_prefix)s_DIMS(%(x)s)[outer_ii];
                if (idx >= 0)
                {
                    if (idx < %(c_prefix)s_DIMS(%(x)s)[outer_ii])
                    {
                        xview_offset += (npy_intp)%(c_prefix)s_STRIDES(%(x)s)[outer_ii] * idx *
                               %(strides_mul)s;
                    }
                    else
                    {
                        PyErr_Format(PyExc_IndexError,"index out of bounds");
                        %(fail)s;
                    }
                }
                else
                {
                    PyErr_Format(PyExc_IndexError,"index out of bounds");
                    %(fail)s;
                }

                spec_pos += 1;
            }
        }
        assert (inner_ii <= %(view_ndim)s);
        while (inner_ii < %(view_ndim)s)
        {
            assert (outer_ii < %(c_prefix)s_NDIM(%(x)s));
            xview_dims[inner_ii] = %(c_prefix)s_DIMS(%(x)s)[outer_ii];
            xview_strides[inner_ii] = %(c_prefix)s_STRIDES(%(x)s)[outer_ii];

            inner_ii += 1;
            outer_ii += 1;
        }
        """
            % locals()
        )
        # print rval
        return rval

    @staticmethod
    def helper_c_code_cache_version():
        return (9,)

    def c_code(self, node, name, inputs, outputs, sub):  # DEBUG
        if not isinstance(node.inputs[0].type, TensorType):
            raise NotImplementedError()

        x = inputs[0]
        (z,) = outputs
        ndim = node.inputs[0].ndim
        view_ndim = node.outputs[0].ndim
        fail = sub["fail"]

        decl = "PyArrayObject * xview = NULL;"

        checkNDim = (
            """
        if (PyArray_NDIM(%(x)s) != %(ndim)s){
            PyErr_SetString(PyExc_ValueError,
                                     "Expected %(ndim)s dimensions input"
                                        );
            %(fail)s
        }
        """
            % locals()
        )

        get_xview = self.helper_c_code(
            node, name, inputs, outputs, sub, self.idx_list, view_ndim
        )
        build_view = (
            """
        //TODO: give this Op a second output so that this view can be cached
        //TODO: alternatively, fix the memory leak on failure
        Py_INCREF(PyArray_DESCR(%(x)s));
        xview = (PyArrayObject*)PyArray_NewFromDescr(
                &PyArray_Type,
                PyArray_DESCR(%(x)s),
                %(view_ndim)s,
                xview_dims,
                xview_strides,
                PyArray_BYTES(%(x)s) + xview_offset,
                PyArray_FLAGS(%(x)s),
                NULL);
        assert (PyArray_NDIM(xview) == %(view_ndim)s);
        if (!xview)
        {
            %(fail)s;
        }
        """
            % locals()
        )

        finish_view = f"""
        Py_XDECREF({z});
        Py_INCREF(py_{x});
        PyArray_SetBaseObject(xview, py_{x});
        assert(py_{x} == (PyObject*){x});
        {z} = xview;
        """

        return decl + checkNDim + "{" + get_xview + build_view + finish_view + "}"

    def c_code_cache_version(self):
        hv = self.helper_c_code_cache_version()
        # If `helper_c_code_cache_version` is not versioned we do not want to
        # have a versioned version of this op's C code.
        if len(hv) == 0:
            return ()
        return (4, hv)

    def R_op(self, inputs, eval_points):
        # Subtensor is not differentiable wrt to its indices, therefore we
        # do not even need to consider the eval_points provided for those
        # (they should be defaulted to zeros_like by the global R_op)
        if eval_points[0] is None:
            return [None]
        return self(eval_points[0], *inputs[1:], return_list=True)


class SubtensorPrinter(Printer):
    def process(self, r, pstate):
        return self._process(r.owner.op.idx_list, r.owner.inputs, pstate)

    def _process(self, idxs, op_inputs, pstate):
        inputs = list(op_inputs)
        input = inputs.pop(0)
        sidxs = []
        getattr(pstate, "precedence", None)
        for entry in idxs:
            if isinstance(entry, aes.Scalar):
                with set_precedence(pstate):
                    sidxs.append(pstate.pprinter.process(inputs.pop()))
            elif isinstance(entry, slice):
                if entry.start is None or entry.start == 0:
                    msg1 = ""
                else:
                    msg1 = entry.start

                if entry.stop is None or entry.stop == sys.maxsize:
                    msg2 = ""
                else:
                    msg2 = entry.stop

                if entry.step is None:
                    msg3 = ""
                else:
                    msg3 = f":{entry.step}"

                sidxs.append(f"{msg1}:{msg2}{msg3}")

        with set_precedence(pstate, 1000):
            sub = pstate.pprinter.process(input, pstate)

        return f"{sub}[{', '.join(sidxs)}]"


pprint.assign(Subtensor, SubtensorPrinter())


def set_subtensor(x, y, inplace=False, tolerate_inplace_aliasing=False):
    """
    Return x with the given subtensor overwritten by y.

    Parameters
    ----------
    x
        Symbolic variable for the lvalue of = operation.
    y
        Symbolic variable for the rvalue of = operation.
    tolerate_inplace_aliasing
        See inc_subtensor for documentation.

    Examples
    --------
    To replicate the numpy expression "r[10:] = 5", type

    >>> r = ivector()
    >>> new_r = set_subtensor(r[10:], 5)

    """
    return inc_subtensor(
        x,
        y,
        inplace,
        set_instead_of_inc=True,
        tolerate_inplace_aliasing=tolerate_inplace_aliasing,
    )


def inc_subtensor(
    x,
    y,
    inplace=False,
    set_instead_of_inc=False,
    tolerate_inplace_aliasing=False,
    ignore_duplicates=False,
):
    """Update the value of an indexed array by a given amount.

    This is equivalent to ``x[indices] += y`` or ``np.add.at(x, indices, y)``,
    depending on the value of `ignore_duplicates`.

    Parameters
    ----------
    x
        The symbolic result of a Subtensor operation.
    y
        The amount by which to increment the array.
    inplace
        Don't use. Aesara will do in-place operations itself, when possible.
    set_instead_of_inc
        If True, do a set_subtensor instead.
    tolerate_inplace_aliasing:
        Allow `x` and `y` to be views of a single underlying array even while
        working in-place. For correct results, `x` and `y` must not be overlapping
        views; if they overlap, the result of this `Op` will generally be
        incorrect. This value has no effect if ``inplace=False``.
    ignore_duplicates
        This determines whether or not ``x[indices] += y`` is used or
        ``np.add.at(x, indices, y)``.  When the special duplicates handling of
        ``np.add.at`` isn't required, setting this option to ``True``
        (i.e. using ``x[indices] += y``) can resulting in faster compiled
        graphs.

    Examples
    --------
    To replicate the expression ``r[10:] += 5``:

    ..code-block:: python

        r = ivector()
        new_r = inc_subtensor(r[10:], 5)

    To replicate the expression ``r[[0, 1, 0]] += 5``:

    ..code-block:: python

        r = ivector()
        new_r = inc_subtensor(r[10:], 5, ignore_duplicates=True)

    """
    # First of all, y cannot have a higher dimension than x,
    # nor have non-broadcastable dimensions where x is broadcastable.

    x = as_tensor_variable(x)
    y = as_tensor_variable(y)

    if y.ndim > x.ndim:
        raise TypeError(
            f"Trying to increment a {int(x.ndim)}-dimensional "
            f"subtensor with a {int(y.ndim)}-dimensional value."
        )

    dim_offset = x.ndim - y.ndim
    for dim in range(y.ndim):
        if x.broadcastable[dim + dim_offset] and not y.broadcastable[dim]:
            # It is acceptable to try to increment a subtensor with a
            # broadcastable dim with a tensor that is not broadcastable
            # on that dimension. However, its length must then be 1.
            # We insert a Rebroadcast Op to make sure it is the case.
            y = addbroadcast(y, dim)

    if not x.owner:
        raise TypeError("x must be the result of a subtensor operation")

    # retrieve idx_list from x.owner
    if isinstance(x.owner.op, Subtensor):
        if tolerate_inplace_aliasing:
            destroyhandler_tolerate_aliased = [[0, 1]]
        else:
            destroyhandler_tolerate_aliased = []
        the_op = IncSubtensor(
            x.owner.op.idx_list,
            inplace,
            set_instead_of_inc,
            destroyhandler_tolerate_aliased=destroyhandler_tolerate_aliased,
        )
        real_x = x.owner.inputs[0]
        real_idxargs = x.owner.inputs[1:]
        return the_op(real_x, y, *real_idxargs)
    elif isinstance(x.owner.op, AdvancedSubtensor1):
        real_x = x.owner.inputs[0]
        ilist = x.owner.inputs[1]
        if ignore_duplicates:
            the_op = AdvancedIncSubtensor(
                inplace, set_instead_of_inc=set_instead_of_inc, ignore_duplicates=True
            )
        else:
            the_op = AdvancedIncSubtensor1(
                inplace, set_instead_of_inc=set_instead_of_inc
            )
        return the_op(real_x, y, ilist)
    elif isinstance(x.owner.op, AdvancedSubtensor):
        real_x = x.owner.inputs[0]
        ilist = x.owner.inputs[1:]
        the_op = AdvancedIncSubtensor(
            inplace,
            set_instead_of_inc=set_instead_of_inc,
            ignore_duplicates=ignore_duplicates,
        )
        return the_op(real_x, y, *ilist)
    elif isinstance(x.owner.op, DimShuffle):
        inner_x = x.owner.inputs[0]
        # In the dimshuffle case, there are in fact two dimshuffles:
        # one to make the indexed dimension the last one,
        # and one to put it back where it was. So, in the case where we have
        # inc_subtensor(x[:,i], y), the graph is actually
        # inc_subtensor((x.T)[i].T, y).
        # We could get all the way to x, and then get rid of the dimshuffles
        # completely, but the problem is that advanced_inc_subtensor1 can only
        # work on the first (outer-most, left-most) dimension of x,
        # just like advanced_subtensor1.
        # So we call advanced_inc_subtensor1(x.T, i, y.T) (as we also need to
        # transpose y if it is not a scalar or a vector), but then we need to
        # return something that has the same shape as x, not as x.T (inner_x).
        # So re-apply the outer dimshuffle on the new inc_subtensor,
        # and return advanced_inc_subtensor1(x.T, i, y.T).T.

        # Get the dimshuffle pattern to apply to y.
        x_order = x.owner.op.new_order
        y_order = ["x"] * x.ndim
        for i, v in enumerate(x_order):
            if v != "x" and (v - dim_offset) >= 0:
                y_order[v - dim_offset] = i

        inner_incsubtensor = inc_subtensor(
            inner_x,
            y.dimshuffle(y_order),
            inplace=inplace,
            set_instead_of_inc=set_instead_of_inc,
            tolerate_inplace_aliasing=tolerate_inplace_aliasing,
            ignore_duplicates=ignore_duplicates,
        )
        # The broadcastable pattern of inner_x may not be the same as
        # the one of x, so we have to build a new dimshuffle here,
        # instead of reusing x.owner.op().
        return inner_incsubtensor.dimshuffle(x.owner.op.new_order)

    elif isinstance(x.owner.op, Reshape):
        # This case happens when the indices are not arranged as a vector, but
        # as a higher-dimensional array. This is handled by the subtensor
        # by flattening this list, taking the subtensor, then reshaping the
        # result.
        inner_x = x.owner.inputs[0]
        # Try to apply inc_subtensor on inner_x.
        # If it works, there is no need to reshape, as the inc_subtensor
        # will have the same shape as inner_x, which is what we want.
        # We also explicitly duplicate y to its broadcasted shape
        # before we partially flatten it to inner_x dimension. This is
        # not strictly needed in all cases, but it is easier this way.
        if y.ndim > 0:
            # This if is needed to prevent some useless warning about
            # old code bug.
            expanded_y = alloc(y, *[x.shape[i] for i in range(x.ndim)])
            flattened_y = expanded_y.reshape(inner_x.shape)
        else:
            flattened_y = y

        inner_incsubtensor = inc_subtensor(
            inner_x,
            flattened_y,
            inplace=inplace,
            set_instead_of_inc=set_instead_of_inc,
            tolerate_inplace_aliasing=tolerate_inplace_aliasing,
            ignore_duplicates=ignore_duplicates,
        )
        return inner_incsubtensor
    else:
        raise TypeError("x must be the result of a subtensor operation")


class IncSubtensor(COp):
    """
    Increment a subtensor.

    This is like numpy's

        x[i,j,k] += y

    It is used internally to implement the gradient on SubTensor.

    Parameters
    ----------
    set_instead_of_inc
        If True set the subtensor to the value instead of incrementing it by
        that value.

    """

    check_input = False
    __props__ = ("idx_list", "inplace", "set_instead_of_inc")

    def __init__(
        self,
        idx_list,
        inplace=False,
        set_instead_of_inc=False,
        destroyhandler_tolerate_aliased=None,
    ):
        if destroyhandler_tolerate_aliased is None:
            destroyhandler_tolerate_aliased = []
        self.idx_list = list(map(index_vars_to_types, idx_list))
        self.inplace = inplace
        if inplace:
            self.destroy_map = {0: [0]}
        self.destroyhandler_tolerate_aliased = list(destroyhandler_tolerate_aliased)
        self.set_instead_of_inc = set_instead_of_inc

    def __hash__(self):
        idx_list = tuple(
            (entry.start, entry.stop, entry.step) if isinstance(entry, slice) else entry
            for entry in self.idx_list
        )
        return hash((type(self), idx_list, self.inplace, self.set_instead_of_inc))

    def __str__(self):
        indices = []
        for entry in self.idx_list:
            if isinstance(entry, slice):
                indices.append(Subtensor.str_from_slice(entry))
            else:
                indices.append(str(entry))
        if self.inplace:
            msg = "Inplace"
        else:
            msg = ""
        if not self.set_instead_of_inc:
            msg += "Inc"
        else:
            msg += "Set"
        return f"{self.__class__.__name__}{{{msg};{', '.join(indices)}}}"

    def make_node(self, x, y, *inputs):
        """
        Parameters
        ----------
        x
            The tensor to increment.
        y
            The value to increment by.
        inputs: TODO WRITEME

        """
        x, y = map(as_tensor_variable, [x, y])
        if y.ndim > x.ndim:
            raise ValueError(
                f"Trying to increment a {int(x.ndim)}-dimensional "
                f"subtensor with a {int(y.ndim)}-dimensional value."
            )
        inputs = tuple(map(as_nontensor_scalar, inputs))

        idx_list = list(self.idx_list)
        if len(idx_list) > x.type.ndim:
            raise IndexError("too many indices for array")

        input_types = get_slice_elements(
            idx_list, lambda entry: isinstance(entry, Type)
        )
        if len(inputs) != len(input_types):
            raise IndexError(
                "Not enough inputs to fill in the Subtensor template.", inputs, idx_list
            )
        for input, expected_type in zip(inputs, input_types):
            if not expected_type.is_super(input.type):
                raise TypeError(
                    f"Wrong type for Subtensor template. Expected {input.type}, got {expected_type}."
                )

        return Apply(self, (x, y) + inputs, [x.type()])

    def decl_view(self):
        return "PyArrayObject * zview = NULL;"

    def perform(self, node, inputs, out_):
        (out,) = out_
        x, y = inputs[:2]
        indices = list(reversed(inputs[2:]))

        def _convert(entry):
            if isinstance(entry, Type):
                return indices.pop()
            elif isinstance(entry, slice):
                return slice(
                    _convert(entry.start), _convert(entry.stop), _convert(entry.step)
                )
            else:
                return entry

        cdata = tuple(map(_convert, self.idx_list))
        if len(cdata) == 1:
            cdata = cdata[0]
        if not self.inplace:
            x = x.copy()
        sub_x = x.__getitem__(cdata)
        if sub_x.shape:
            # we've sliced out an N-D tensor with N > 0
            if not self.set_instead_of_inc:
                sub_x += y
            else:
                # sub_x += -sub_x + y
                x.__setitem__(cdata, y)
        else:
            # scalar case
            if not self.set_instead_of_inc:
                x.__setitem__(cdata, sub_x + y)
            else:
                x.__setitem__(cdata, y)
        out[0] = x

    def c_code(self, node, name, inputs, outputs, sub):

        # This method delegates much of the work to helper
        # methods. This method implements the main logic
        # but subclasses may override the helper methods
        # to change the particulars, e.g. GpuIncSubtensor
        # turns the view/copy operations on numpy arrays
        # into the same operations on gpu arrays.

        self.do_type_checking(node)

        if self.inplace:  # convert bool to int
            inplace = 1
        else:
            inplace = 0
        x = inputs[0]
        y = inputs[1]
        (z,) = outputs
        if self.set_instead_of_inc:  # convert bool to int
            op_is_set = 1
        else:
            op_is_set = 0
        fail = sub["fail"]
        view_ndim = node.inputs[0].ndim - sum(
            not isinstance(idx, slice) for idx in self.idx_list
        )

        copy_of_x = self.copy_of_x(x)

        copy_input_if_necessary = (
            """
        if (%(inplace)s)
        {
            if (%(x)s != %(z)s)
            {
                Py_XDECREF(%(z)s);
                Py_INCREF(%(x)s);
                %(z)s = %(x)s;
            }
        }
        else
        {
            Py_XDECREF(%(z)s);
            %(z)s = %(copy_of_x)s;
            if (!%(z)s) {
                // Exception already set
                %(fail)s
            }
        }
        """
            % locals()
        )

        # get info needed to make zview: a view of %(z)s
        helper_args = self.get_helper_c_code_args()

        get_zview = Subtensor.helper_c_code(
            node=node,
            name=name,
            inputs=outputs[:1] + inputs[2:],
            outputs=outputs,
            sub=sub,
            idx_list=self.idx_list,
            view_ndim=view_ndim,
            **helper_args,
        )

        # Make a view on the output, as we will write into it.
        alloc_zview = self.make_view_array(z, view_ndim)

        build_view = (
            """
        //TODO: give this Op a second output so that this view can be cached
        //TODO: alternatively, fix the memory leak on failure
        %(alloc_zview)s;
        if (!zview)
        {
            %(fail)s;
        }
        """
            % locals()
        )

        copy_into = self.copy_into("zview", y)

        add_to_zview = self.add_to_zview(name, y, fail)

        make_modification = (
            """
        if (%(op_is_set)s)
        {
            if (%(copy_into)s) // does broadcasting
            {
                Py_DECREF(zview);
                %(fail)s;
            }
        }
        else
        {
            %(add_to_zview)s
        }
        """
            % locals()
        )
        return (
            self.decl_view()
            + copy_input_if_necessary
            + "{"
            + get_zview
            + build_view
            + make_modification
            + "Py_DECREF(zview);"
            + "}"
        )

    def do_type_checking(self, node):
        """
        Should raise NotImplementedError if c_code does not support
        the types involved in this node.

        """

        if not isinstance(node.inputs[0].type, TensorType):
            raise NotImplementedError()

    def c_code_cache_version(self):
        hv = Subtensor.helper_c_code_cache_version()
        if hv:
            return (3, hv)
        else:
            return ()

    def copy_of_x(self, x):
        """
        Parameters
        ----------
        x
            A string giving the name of a C variable pointing to an array.

        Returns
        -------
        object
            C code expression to make a copy of x.

        Base class uses PyArrayObject *, subclasses may override for
        different types of arrays.

        """
        # Parameters of PyArrary_FromAny are:
        # array
        # dtype: we pass NULL to say any dtype is acceptable, so the existing
        #        dtype will be copied
        # min_depth: we pass 0 to have this parameter ignored
        # max_depth: we pass 0 to have this parameter ignored
        # requirements: here we pass NPY_ARRAY_ENSURECOPY to force a copy
        # context: this is almost always NULL, I'm not sure what it's used for
        return f"""(PyArrayObject*)PyArray_FromAny(py_{x}, NULL, 0, 0,
                NPY_ARRAY_ENSURECOPY, NULL)"""

    def make_view_array(self, x, view_ndim):
        """
        Parameters
        ----------
        x
            A string identifying an array to be viewed.
        view_ndim
            A string specifying the number of dimensions to have in the view.

        This doesn't need to actually set up the view with the right indexing;
        we'll do that manually later.

        """

        return (
            """Py_INCREF(PyArray_DESCR(%(x)s));
        zview = (PyArrayObject*)PyArray_NewFromDescr(
                &PyArray_Type,
                PyArray_DESCR(%(x)s),
                %(view_ndim)s,
                xview_dims, //PyArray_DIMS(%(x)s),
                xview_strides, //PyArray_STRIDES(%(x)s),
                PyArray_BYTES(%(x)s) + xview_offset, //PyArray_DATA(%(x)s),
                PyArray_FLAGS(%(x)s),
                NULL);
        """
            % locals()
        )

    def get_helper_c_code_args(self):
        """
        Return a dictionary of arguments to pass to helper_c_code.

        """
        return Subtensor.default_helper_c_code_args()

    def copy_into(self, view, source):
        """
        Parameters
        ----------
        view : string
            C code expression for an array.
        source : string
            C code expression for an array.

        Returns
        -------
        object
            C code expression to copy source into view, and 0 on success.

        """
        return f"""PyArray_CopyInto({view}, {source})"""

    def add_to_zview(self, name, x, fail):
        """
        Return C code to add x to zview. Should DECREF zview if the
        add fails.

        """

        return (
            """
            PyArrayObject * add_rval = (PyArrayObject*)PyNumber_InPlaceAdd(
                    (PyObject*)zview, py_%(x)s);
            if (add_rval)
            {
                assert (PyArray_Check((PyObject*)add_rval));
                assert (PyArray_DATA(add_rval) == PyArray_DATA(zview));
                Py_DECREF(add_rval);
            }
            else
            {
                Py_DECREF(zview);
                %(fail)s;
            }"""
            % locals()
        )

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[0]]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None or eval_points[1] is None:
            return [None]
        # Again we ignore eval points for indices because incsubtensor is
        # not differentiable wrt to those
        return self(eval_points[0], eval_points[1], *inputs[2:], return_list=True)

    def connection_pattern(self, node):

        rval = [[True], [True]]

        for ipt in node.inputs[2:]:
            rval.append([False])

        return rval

    def grad(self, inputs, grads):
        (g_output,) = grads
        x, y = inputs[:2]
        idx_list = inputs[2:]

        if x.dtype in discrete_dtypes:
            # The output dtype is the same as x
            gx = x.zeros_like(dtype=config.floatX)
            if y.dtype in discrete_dtypes:
                gy = y.zeros_like(dtype=config.floatX)
            else:
                gy = y.zeros_like()
        elif x.dtype in complex_dtypes:
            raise NotImplementedError("No support for complex grad yet")
        else:
            if self.set_instead_of_inc:
                gx = set_subtensor(
                    Subtensor(idx_list=self.idx_list)(g_output, *idx_list),
                    aesara.tensor.zeros_like(y),
                )
            else:
                gx = g_output
            gy = Subtensor(idx_list=self.idx_list)(g_output, *idx_list)
            gy = _sum_grad_over_bcasted_dims(y, gy)

        return [gx, gy] + [DisconnectedType()()] * len(idx_list)


class IncSubtensorPrinter(SubtensorPrinter):
    def process(self, r, pstate):
        x, y, *idx_args = r.owner.inputs

        res = self._process(r.owner.op.idx_list, [x] + idx_args, pstate)

        with set_precedence(pstate, 1000):
            y_str = pstate.pprinter.process(r.owner.inputs[1], pstate)

        if r.owner.op.set_instead_of_inc:
            res = f"set_subtensor({res}, {y_str})"
        else:
            res = f"inc_subtensor({res}, {y_str})"
        return res


pprint.assign(IncSubtensor, IncSubtensorPrinter())


def _sum_grad_over_bcasted_dims(x, gx):
    """
    Sum of gx over dimensions to reproduce x.broadcastable.

    This is useful to sum gradients over certain dimensions when
    x has been broadcasted, and we need to sum the gradient contributions
    over all duplications.

    """
    if gx.broadcastable != x.broadcastable:
        x_dim_added = gx.ndim - x.ndim
        x_broad = (True,) * x_dim_added + x.broadcastable
        assert sum(gx.broadcastable) < sum(x_broad)
        axis_to_sum = []
        for i in range(gx.ndim):
            if gx.broadcastable[i] is False and x_broad[i] is True:
                axis_to_sum.append(i)
            elif gx.broadcastable[i] is True and x_broad[i] is False:
                # This means that Aesara was able to infer that
                # gx.shape[i] is 1, so x.shape[i] is 1, but we
                # didn't know it. It is fine.
                pass
            else:
                assert gx.broadcastable[i] == x_broad[i]
        gx = gx.sum(axis=axis_to_sum, keepdims=True)
        if gx.ndim != x.ndim:
            assert gx.ndim > x.ndim
            for i in range(x_dim_added):
                assert gx.broadcastable[i]
            gx = gx.dimshuffle(*list(range(x_dim_added, gx.ndim)))
        assert gx.broadcastable == x.broadcastable
    return gx


class AdvancedSubtensor1(COp):
    """
    Implement x[ilist] where ilist is a vector of integers.

    """

    # sparse_grad doesn't go in here since it only affects the output
    # of the grad() method.
    __props__ = ()
    _f16_ok = True
    check_input = False

    def __init__(self, sparse_grad=False):
        self.sparse_grad = sparse_grad

    def make_node(self, x, ilist):
        x_ = as_tensor_variable(x)
        ilist_ = as_tensor_variable(ilist)
        if ilist_.type.dtype not in integer_dtypes:
            raise TypeError("index must be integers")
        if ilist_.type.ndim != 1:
            raise TypeError("index must be vector")
        if x_.type.ndim == 0:
            raise TypeError("cannot index into a scalar")
        bcast = (ilist_.broadcastable[0],) + x_.broadcastable[1:]
        return Apply(self, [x_, ilist_], [TensorType(dtype=x.dtype, shape=bcast)()])

    def perform(self, node, inp, out_):
        x, i = inp
        (out,) = out_
        # Copy always implied by numpy advanced indexing semantic.
        if out[0] is not None and out[0].shape == (len(i),) + x.shape[1:]:
            o = out[0]
        else:
            o = None

        # If i.dtype is more precise than numpy.intp (int32 on 32-bit machines,
        # int64 on 64-bit machines), numpy may raise the following error:
        # TypeError: array cannot be safely cast to required type.
        # We need to check if values in i can fit in numpy.intp, because
        # if they don't, that should be an error (no array can have that
        # many elements on a 32-bit arch).
        if i.dtype != np.intp:
            i_ = _asarray(i, dtype=np.intp)
            if not np.can_cast(i.dtype, np.intp):
                # Check if there was actually an incorrect conversion
                if np.any(i != i_):
                    raise IndexError(
                        "index contains values that are bigger "
                        "than the maximum array size on this system.",
                        i,
                    )
            i = i_

        out[0] = x.take(i, axis=0, out=o)

    def connection_pattern(self, node):
        rval = [[True]]

        for ipt in node.inputs[1:]:
            rval.append([False])

        return rval

    def grad(self, inputs, grads):
        x, ilist = inputs
        (gz,) = grads
        assert len(inputs) == 2
        if self.sparse_grad:
            if x.type.ndim != 2:
                raise TypeError(
                    "AdvancedSubtensor1: you can't take the sparse grad"
                    " from a tensor with ndim != 2. ndim is " + str(x.type.ndim)
                )

            rval1 = [aesara.sparse.construct_sparse_from_list(x, gz, ilist)]
        else:
            if x.dtype in discrete_dtypes:
                # The output dtype is the same as x
                gx = x.zeros_like(dtype=config.floatX)
            elif x.dtype in complex_dtypes:
                raise NotImplementedError("No support for complex grad yet")
            else:
                gx = x.zeros_like()
            rval1 = [advanced_inc_subtensor1(gx, gz, ilist)]
        return rval1 + [DisconnectedType()()] * (len(inputs) - 1)

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return [None]
        return self.make_node(eval_points[0], *inputs[1:]).outputs

    def infer_shape(self, fgraph, node, ishapes):
        x, ilist = ishapes
        return [ilist + x[1:]]

    def c_support_code(self, **kwargs):
        # In some versions of numpy, NPY_MIN_INTP is defined as MIN_LONG,
        # which is not defined. It should be NPY_MIN_LONG instead in that case.
        return dedent(
            """\
                #ifndef MIN_LONG
                #define MIN_LONG NPY_MIN_LONG
                #endif"""
        )

    def c_code(self, node, name, input_names, output_names, sub):
        if self.__class__ is not AdvancedSubtensor1:
            raise MethodNotDefined(
                "c_code defined for AdvancedSubtensor1," " not for child class",
                type(self),
            )
        a_name, i_name = input_names[0], input_names[1]
        output_name = output_names[0]
        fail = sub["fail"]
        return (
            """
            PyArrayObject *indices;
            int i_type = PyArray_TYPE(%(i_name)s);
            if (i_type != NPY_INTP) {
                // Cast %(i_name)s to NPY_INTP (expected by PyArray_TakeFrom),
                // if all values fit.
                if (!PyArray_CanCastSafely(i_type, NPY_INTP) &&
                    PyArray_SIZE(%(i_name)s) > 0) {
                    npy_int64 min_val, max_val;
                    PyObject* py_min_val = PyArray_Min(%(i_name)s, NPY_MAXDIMS,
                                                       NULL);
                    if (py_min_val == NULL) {
                        %(fail)s;
                    }
                    min_val = PyLong_AsLongLong(py_min_val);
                    Py_DECREF(py_min_val);
                    if (min_val == -1 && PyErr_Occurred()) {
                        %(fail)s;
                    }
                    PyObject* py_max_val = PyArray_Max(%(i_name)s, NPY_MAXDIMS,
                                                       NULL);
                    if (py_max_val == NULL) {
                        %(fail)s;
                    }
                    max_val = PyLong_AsLongLong(py_max_val);
                    Py_DECREF(py_max_val);
                    if (max_val == -1 && PyErr_Occurred()) {
                        %(fail)s;
                    }
                    if (min_val < NPY_MIN_INTP || max_val > NPY_MAX_INTP) {
                        PyErr_SetString(PyExc_IndexError,
                                     "Index contains values "
                                     "that are bigger than the maximum array "
                                     "size on this system.");
                        %(fail)s;
                    }
                }
                indices = (PyArrayObject*) PyArray_Cast(%(i_name)s, NPY_INTP);
                if (indices == NULL) {
                    %(fail)s;
                }
            }
            else {
                 indices = %(i_name)s;
                 Py_INCREF(indices);
            }
            if (%(output_name)s != NULL) {
                npy_intp nd, i, *shape;
                nd = PyArray_NDIM(%(a_name)s) + PyArray_NDIM(indices) - 1;
                if (PyArray_NDIM(%(output_name)s) != nd) {
                    Py_CLEAR(%(output_name)s);
                }
                else {
                    shape = PyArray_DIMS(%(output_name)s);
                    for (i = 0; i < PyArray_NDIM(indices); i++) {
                        if (shape[i] != PyArray_DIMS(indices)[i]) {
                            Py_CLEAR(%(output_name)s);
                            break;
                        }
                    }
                    if (%(output_name)s != NULL) {
                        for (; i < nd; i++) {
                            if (shape[i] != PyArray_DIMS(%(a_name)s)[
                                                i-PyArray_NDIM(indices)+1]) {
                                Py_CLEAR(%(output_name)s);
                                break;
                            }
                        }
                    }
                }
            }
            %(output_name)s = (PyArrayObject*)PyArray_TakeFrom(
                        %(a_name)s, (PyObject*)indices, 0, %(output_name)s, NPY_RAISE);
            Py_DECREF(indices);
            if (%(output_name)s == NULL) %(fail)s;
        """
            % locals()
        )

    def c_code_cache_version(self):
        return (0, 1, 2)


advanced_subtensor1 = AdvancedSubtensor1()


class AdvancedIncSubtensor1(COp):
    """
    Increments a subtensor using advanced slicing (list of index).

    """

    __props__ = ("inplace", "set_instead_of_inc")
    check_input = False
    params_type = ParamsType(inplace=aes.bool, set_instead_of_inc=aes.bool)

    def __init__(self, inplace=False, set_instead_of_inc=False):
        self.inplace = bool(inplace)
        self.set_instead_of_inc = bool(set_instead_of_inc)
        if inplace:
            self.destroy_map = {0: [0]}

    def clone_inplace(self):
        return self.__class__(inplace=True, set_instead_of_inc=self.set_instead_of_inc)

    def __str__(self):
        if self.inplace:
            msg = "inplace"
        else:
            msg = "no_inplace"
        if self.set_instead_of_inc:
            msg += ",set"
        else:
            msg += ",inc"

        return self.__class__.__name__ + "{%s}" % msg

    def make_node(self, x, y, ilist):
        x_ = as_tensor_variable(x)
        y_ = as_tensor_variable(y)
        ilist_ = as_tensor_variable(ilist)

        if ilist_.type.dtype not in integer_dtypes:
            raise TypeError("index must be integers")
        if ilist_.type.ndim != 1:
            raise TypeError("index must be vector")
        if x_.type.ndim == 0:
            raise TypeError("cannot index into a scalar")
        if y_.type.ndim > x_.type.ndim:
            if self.set_instead_of_inc:
                opname = "set"
            else:
                opname = "increment"
            raise TypeError(
                "cannot %s x subtensor with ndim=%s by y with ndim=%s."
                % (opname, x_.type.ndim, y_.type.ndim)
            )

        return Apply(self, [x_, y_, ilist_], [x_.type()])

    def copy_of_x(self, x):
        """
        Parameters
        ----------
        x : string
            Gives the name of a C variable pointing to an array.

        Returns
        -------
        object
            C code expression to make a copy of x.

        Base class uses PyArrayObject *, subclasses may override for
        different types of arrays.

        """
        # Parameters of PyArrary_FromAny are:
        # array
        # dtype: we pass NULL to say any dtype is acceptable, so the existing
        #        dtype will be copied
        # min_depth: we pass 0 to have this parameter ignored
        # max_depth: we pass 0 to have this parameter ignored
        # requirements: here we pass NPY_ARRAY_ENSURECOPY to force a copy
        # context: this is almost always NULL, I'm not sure what it's used for
        return f"""(PyArrayObject*)PyArray_FromAny(py_{x}, NULL, 0, 0,
                NPY_ARRAY_ENSURECOPY, NULL)"""

    def c_support_code(self, **kwargs):
        types = [
            "npy_" + t
            for t in [
                "int8",
                "int16",
                "int32",
                "int64",
                "uint8",
                "uint16",
                "uint32",
                "uint64",
                "float16",
                "float32",
                "float64",
            ]
        ]

        complex_types = ["npy_" + t for t in ("complex32", "complex64", "complex128")]

        inplace_map_template = """
        #if defined(%(typen)s)
        static void %(type)s_inplace_add(PyArrayMapIterObject *mit,
                                        PyArrayIterObject *it, int inc_or_set)
        {
            int index = mit->size;
            while (index--) {
                %(op)s

                PyArray_MapIterNext(mit);
                PyArray_ITER_NEXT(it);
            }
        }
        #endif
        """

        floatadd = (
            "((%(type)s*)mit->dataptr)[0] = "
            "(inc_or_set ? ((%(type)s*)mit->dataptr)[0] : 0)"
            " + ((%(type)s*)it->dataptr)[0];"
        )
        complexadd = """
        ((%(type)s*)mit->dataptr)[0].real =
            (inc_or_set ? ((%(type)s*)mit->dataptr)[0].real : 0)
            + ((%(type)s*)it->dataptr)[0].real;
        ((%(type)s*)mit->dataptr)[0].imag =
            (inc_or_set ? ((%(type)s*)mit->dataptr)[0].imag : 0)
            + ((%(type)s*)it->dataptr)[0].imag;
        """

        fns = "".join(
            [
                inplace_map_template
                % {"type": t, "typen": t.upper(), "op": floatadd % {"type": t}}
                for t in types
            ]
            + [
                inplace_map_template
                % {"type": t, "typen": t.upper(), "op": complexadd % {"type": t}}
                for t in complex_types
            ]
        )

        def gen_binop(type, typen):
            return f"""
    #if defined({typen})
    {type}_inplace_add,
    #endif
    """

        fn_array = (
            "static inplace_map_binop addition_funcs[] = {"
            + "".join(
                [gen_binop(type=t, typen=t.upper()) for t in types + complex_types]
            )
            + "NULL};\n"
        )

        def gen_num(typen):
            return f"""
    #if defined({typen})
    {typen},
    #endif
    """

        type_number_array = (
            "static int type_numbers[] = {"
            + "".join([gen_num(typen=t.upper()) for t in types + complex_types])
            + "-1000};"
        )

        code = (
            """
            typedef void (*inplace_map_binop)(PyArrayMapIterObject *,
                                            PyArrayIterObject *, int inc_or_set);
            """
            + fns
            + fn_array
            + type_number_array
            + """
    static int
    map_increment(PyArrayMapIterObject *mit, PyArrayObject *op,
                inplace_map_binop add_inplace, int inc_or_set)
    {
        PyArrayObject *arr = NULL;
        PyArrayIterObject *it;
        PyArray_Descr *descr;
        if (mit->ait == NULL) {
            return -1;
        }
        descr = PyArray_DESCR(mit->ait->ao);
        Py_INCREF(descr);
        arr = (PyArrayObject *)PyArray_FromAny((PyObject *)op, descr,
                                    0, 0, NPY_ARRAY_FORCECAST, NULL);
        if (arr == NULL) {
            return -1;
        }
        if ((mit->subspace != NULL) && (mit->consec)) {
            PyArray_MapIterSwapAxes(mit, (PyArrayObject **)&arr, 0);
            if (arr == NULL) {
                return -1;
            }
        }
        it = (PyArrayIterObject*)
                PyArray_BroadcastToShape((PyObject*)arr, mit->dimensions, mit->nd);
        if (it  == NULL) {
            Py_DECREF(arr);
            return -1;
        }

        (*add_inplace)(mit, it, inc_or_set);

        Py_DECREF(arr);
        Py_DECREF(it);
        return 0;
    }


    static int
    inplace_increment(PyArrayObject *a, PyObject *index, PyArrayObject *inc,
                    int inc_or_set)
    {
        inplace_map_binop add_inplace = NULL;
        int type_number = -1;
        int i = 0;
        PyArrayMapIterObject * mit;

        if (PyArray_FailUnlessWriteable(a, "input/output array") < 0) {
            return -1;
        }

        if (PyArray_NDIM(a) == 0) {
            PyErr_SetString(PyExc_IndexError, "0-d arrays can't be indexed.");
            return -1;
        }
        type_number = PyArray_TYPE(a);

        while (type_numbers[i] >= 0 && addition_funcs[i] != NULL){
            if (type_number == type_numbers[i]) {
                add_inplace = addition_funcs[i];
                break;
            }
            i++ ;
        }

        if (add_inplace == NULL) {
            PyErr_SetString(PyExc_TypeError, "unsupported type for a");
            return -1;
        }
        mit = (PyArrayMapIterObject *) PyArray_MapIterArray(a, index);
        if (mit == NULL) {
            goto fail;
        }
        if (map_increment(mit, inc, add_inplace, inc_or_set) != 0) {
            goto fail;
        }

        Py_DECREF(mit);

        Py_INCREF(Py_None);
        return 0;

    fail:
        Py_XDECREF(mit);

        return -1;
    }
    """
        )

        return code

    def c_code(self, node, name, input_names, output_names, sub):
        numpy_ver = [int(n) for n in np.__version__.split(".")[:2]]
        if bool(numpy_ver < [1, 8]):
            raise NotImplementedError
        x, y, idx = input_names
        out = output_names[0]
        copy_of_x = self.copy_of_x(x)

        return """
        PyObject* rval = NULL;
        if (%(params)s->inplace)
        {
            if (%(x)s != %(out)s)
            {
                Py_XDECREF(%(out)s);
                Py_INCREF(%(x)s);
                %(out)s = %(x)s;
            }
        }
        else
        {
            Py_XDECREF(%(out)s);
            %(out)s = %(copy_of_x)s;
            if (!%(out)s) {
                // Exception already set
                %(fail)s
            }
        }
        if (inplace_increment(%(out)s, (PyObject *)%(idx)s, %(y)s, (1 - %(params)s->set_instead_of_inc))) {
            %(fail)s;
        }
        Py_XDECREF(rval);
        """ % dict(
            x=x,
            y=y,
            idx=idx,
            out=out,
            copy_of_x=copy_of_x,
            params=sub["params"],
            fail=sub["fail"],
        )

    def c_code_cache_version(self):
        return (8,)

    def perform(self, node, inp, out_, params):
        x, y, idx = inp
        (out,) = out_
        if not self.inplace:
            x = x.copy()

        if self.set_instead_of_inc:
            x[idx] = y
        else:
            # In Numpy, `x[idx] += y` doesn't work if the same index is present
            # many times: it does it only once.
            np.add.at(x, idx, y)

        out[0] = x

    def infer_shape(self, fgraph, node, ishapes):
        x, y, ilist = ishapes
        return [x]

    def R_op(self, inputs, eval_points):
        if None in eval_points[:2]:
            return [None]
        return self.make_node(eval_points[0], eval_points[1], *inputs[2:]).outputs

    def connection_pattern(self, node):

        rval = [[True], [True], [False]]
        return rval

    def grad(self, inputs, grads):
        (g_output,) = grads
        x, y, idx_list = inputs
        if x.dtype in discrete_dtypes:
            # The output dtype is the same as x
            gx = x.zeros_like(dtype=config.floatX)
            if y.dtype in discrete_dtypes:
                gy = y.zeros_like(dtype=config.floatX)
            else:
                gy = y.zeros_like()
        elif x.dtype in complex_dtypes:
            raise NotImplementedError("No support for complex grad yet")
        else:
            if self.set_instead_of_inc:
                gx = advanced_set_subtensor1(g_output, y.zeros_like(), idx_list)
            else:
                gx = g_output
            gy = advanced_subtensor1(g_output, idx_list)
            gy = _sum_grad_over_bcasted_dims(y, gy)

        return [gx, gy] + [DisconnectedType()()]


advanced_inc_subtensor1 = AdvancedIncSubtensor1()
advanced_set_subtensor1 = AdvancedIncSubtensor1(set_instead_of_inc=True)


def as_index_variable(idx):
    if idx is None:
        return NoneConst.clone()
    if isinstance(idx, slice):
        return make_slice(idx)
    if isinstance(idx, Variable) and isinstance(idx.type, SliceType):
        return idx
    if isinstance(idx, Variable) and isinstance(idx.type, NoneTypeT):
        return idx
    idx = as_tensor_variable(idx)
    if idx.type.dtype not in discrete_dtypes:
        raise TypeError("index must be integers or a boolean mask")
    return idx


def check_advanced_indexing_dimensions(input, idx_list):
    """
    This function checks if the index list in idx_list is correct.
    If there are any boolean masks, we check if the mask has the
    same shape as the input. This is enforced in NumPy 0.13.0 and
    newer, but not by earlier versions. If the size is not the same,
    this method raises an IndexError.
    """
    dim_seen = 0
    for index in idx_list:
        if index is np.newaxis:
            # skip, does not count as an input dimension
            pass
        elif isinstance(index, np.ndarray) and index.dtype == "bool":
            for i in range(index.ndim):
                if index.shape[i] != input.shape[dim_seen + i]:
                    raise IndexError(
                        "boolean index did not match indexed array "
                        f"along dimension {int(dim_seen + i)}; dimension is "
                        f"{int(input.shape[dim_seen + i])} but "
                        f"corresponding boolean dimension is {index.shape[i]}"
                    )
            dim_seen += index.ndim
        else:
            dim_seen += 1


class AdvancedSubtensor(Op):
    """Implements NumPy's advanced indexing."""

    __props__ = ()

    def make_node(self, x, *index):
        x = as_tensor_variable(x)
        index = tuple(map(as_index_variable, index))

        # We only want the broadcast information, and we don't need recursive
        # `Subtensor` calls, so we create a fake symbolic shape tuple and
        # identify the broadcast dimensions from the shape result of this
        # entire subtensor operation.
        with config.change_flags(compute_test_value="off"):
            fake_shape = tuple(
                tensor(dtype="int64", shape=()) if not bcast else 1
                for bcast in x.broadcastable
            )

            bcast_index = tuple(
                chain.from_iterable(
                    aesara.tensor.basic.nonzero(idx)
                    if getattr(idx, "ndim", 0) > 0
                    and getattr(idx, "dtype", None) == "bool"
                    else (idx,)
                    for idx in index
                )
            )

            bcast = [
                getattr(i, "value", i) == 1
                for i in indexed_result_shape(fake_shape, bcast_index)
            ]

        return Apply(
            self,
            (x,) + index,
            [tensor(dtype=x.type.dtype, shape=bcast)],
        )

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return [None]
        return self.make_node(eval_points[0], *inputs[1:]).outputs

    def infer_shape(self, fgraph, node, ishapes):
        indices = node.inputs[1:]
        index_shapes = list(ishapes[1:])
        for i, idx in enumerate(indices):
            if (
                isinstance(idx, (np.bool_, bool))
                or getattr(idx, "dtype", None) == "bool"
            ):
                raise ShapeError(
                    "Shape inference for boolean indices is not implemented"
                )
            # The `ishapes` entries for `SliceType`s will be None, and
            # we need to give `indexed_result_shape` the actual slices.
            if isinstance(getattr(idx, "type", None), SliceType):
                index_shapes[i] = idx

        res_shape = indexed_result_shape(
            ishapes[0], index_shapes, indices_are_shapes=True
        )
        assert node.outputs[0].ndim == len(res_shape)
        return [[s for s in res_shape]]

    def perform(self, node, inputs, out_):
        (out,) = out_
        check_advanced_indexing_dimensions(inputs[0], inputs[1:])
        rval = inputs[0].__getitem__(tuple(inputs[1:]))
        # When there are no arrays, we are not actually doing advanced
        # indexing, so __getitem__ will not return a copy.
        # Since no view_map is set, we need to copy the returned value
        if not any(
            isinstance(v.type, TensorType) and v.ndim > 0 for v in node.inputs[1:]
        ):
            rval = rval.copy()
        out[0] = rval

    def connection_pattern(self, node):
        rval = [[True]]

        for ipt in node.inputs[1:]:
            rval.append([False])

        return rval

    def grad(self, inputs, grads):
        (gz,) = grads
        x = inputs[0]
        if x.dtype in discrete_dtypes:
            # The output dtype is the same as x
            gx = x.zeros_like(dtype=config.floatX)
        elif x.dtype in complex_dtypes:
            raise NotImplementedError("No support for complex grad yet")
        else:
            gx = x.zeros_like()
        rest = inputs[1:]
        return [advanced_inc_subtensor(gx, gz, *rest)] + [DisconnectedType()()] * len(
            rest
        )


advanced_subtensor = AdvancedSubtensor()


class AdvancedIncSubtensor(Op):
    """Increments a subtensor using advanced indexing."""

    __props__ = ("inplace", "set_instead_of_inc", "ignore_duplicates")

    def __init__(
        self, inplace=False, set_instead_of_inc=False, ignore_duplicates=False
    ):
        self.set_instead_of_inc = set_instead_of_inc
        self.inplace = inplace
        if inplace:
            self.destroy_map = {0: [0]}
        self.ignore_duplicates = ignore_duplicates

    def __str__(self):
        return "{}{{{}, {}}}".format(
            self.__class__.__name__,
            "inplace=" + str(self.inplace),
            " set_instead_of_inc=" + str(self.set_instead_of_inc),
        )

    def make_node(self, x, y, *inputs):
        x = as_tensor_variable(x)
        y = as_tensor_variable(y)

        new_inputs = []
        for inp in inputs:
            if isinstance(inp, (list, tuple)):
                inp = as_tensor_variable(inp)
            new_inputs.append(inp)
        return Apply(
            self,
            (x, y) + tuple(new_inputs),
            [tensor(dtype=x.type.dtype, shape=x.type.broadcastable)],
        )

    def perform(self, node, inputs, out_):

        x, y, *indices = inputs

        check_advanced_indexing_dimensions(x, indices)

        (out,) = out_
        if not self.inplace:
            out[0] = x.copy()
        else:
            out[0] = x

        if self.set_instead_of_inc:
            out[0][tuple(indices)] = y
        elif self.ignore_duplicates:
            out[0][tuple(indices)] += y
        else:
            np.add.at(out[0], tuple(indices), y)

    def infer_shape(self, fgraph, node, ishapes):
        return [ishapes[0]]

    def connection_pattern(self, node):

        rval = [[True], [True]]

        for ipt in node.inputs[2:]:
            rval.append([False])

        return rval

    def R_op(self, inputs, eval_points):
        if None in eval_points[:2]:
            return [None]
        return self.make_node(eval_points[0], eval_points[1], *inputs[2:]).outputs

    def grad(self, inpt, output_gradients):
        x, y = inpt[:2]
        idxs = inpt[2:]
        (outgrad,) = output_gradients
        if x.dtype in discrete_dtypes:
            # The output dtype is the same as x
            gx = x.zeros_like(dtype=config.floatX)
            if y.dtype in discrete_dtypes:
                gy = y.zeros_like(dtype=config.floatX)
            else:
                gy = y.zeros_like()
        elif x.dtype in complex_dtypes:
            raise NotImplementedError("No support for complex grad yet")
        else:
            if self.set_instead_of_inc:
                gx = advanced_set_subtensor(outgrad, y.zeros_like(), *idxs)
            else:
                gx = outgrad
            gy = advanced_subtensor(outgrad, *idxs)
            # Make sure to sum gy over the dimensions of y that have been
            # added or broadcasted
            gy = _sum_grad_over_bcasted_dims(y, gy)
        return [gx, gy] + [DisconnectedType()() for _ in idxs]


advanced_inc_subtensor = AdvancedIncSubtensor()
advanced_set_subtensor = AdvancedIncSubtensor(set_instead_of_inc=True)
advanced_inc_subtensor_nodup = AdvancedIncSubtensor(ignore_duplicates=True)
advanced_set_subtensor_nodup = AdvancedIncSubtensor(
    set_instead_of_inc=True, ignore_duplicates=True
)


def take(a, indices, axis=None, mode="raise"):
    """Take elements from an array along an axis.

    When axis is not None, this function does the same thing as "fancy"
    indexing (indexing arrays using arrays); however, it can be easier to use
    if you need elements along a given axis. A call such as
    ``np.take(arr, indices, axis=3)`` is equivalent to
    ``arr[:,:,:,indices,...]``.

    See `np.take`

    Parameters
    ----------
    a : TensorVariable
        The source array.
    indices : TensorVariable, ndarray, list, tuple
        The indices of the values to extract.
    axis : int, optional
        The axis over which to select values. By default, the flattened
        input array is used.

    """
    a = as_tensor_variable(a)
    indices = as_tensor_variable(indices)

    if not isinstance(axis, (int, type(None))):
        raise TypeError("`axis` must be an integer or None")

    if axis is None:
        return advanced_subtensor(a.flatten(), indices)
    elif axis < 0:
        axis += a.ndim

    if mode == "clip":
        indices = clip(indices, 0, a.shape[axis] - 1)
    elif mode == "wrap":
        indices = indices % a.shape[axis]

    full_indices = (slice(None),) * axis + (indices,)

    return a[full_indices]


@_get_vector_length.register(Subtensor)
def _get_vector_length_Subtensor(op, var):
    # If we take a slice, we know how many elements it will result in
    # TODO: We can cover more `*Subtensor` cases.
    try:
        indices = aesara.tensor.subtensor.get_idx_list(
            var.owner.inputs, var.owner.op.idx_list
        )
        start = (
            None
            if indices[0].start is None
            else get_scalar_constant_value(indices[0].start)
        )
        stop = (
            None
            if indices[0].stop is None
            else get_scalar_constant_value(indices[0].stop)
        )
        step = (
            None
            if indices[0].step is None
            else get_scalar_constant_value(indices[0].step)
        )

        if start == stop:
            return 0

        arg_len = get_vector_length(var.owner.inputs[0])
        return len(range(*slice(start, stop, step).indices(arg_len)))
    except (ValueError, NotScalarConstantError):
        raise ValueError(f"Length of {var} cannot be determined")


__all__ = [
    "take",
    "inc_subtensor",
    "set_subtensor",
]
