"""Driver for gradient calculations."""

import time
import warnings
from functools import partial, reduce
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Mapping,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
from typing_extensions import Literal

import aesara
from aesara.compile.ops import ViewOp
from aesara.configdefaults import config
from aesara.graph import utils
from aesara.graph.basic import Apply, NominalVariable, Variable
from aesara.graph.null_type import NullType, null_type
from aesara.graph.op import get_test_values
from aesara.graph.type import Type


if TYPE_CHECKING:
    from aesara.compile.mode import Mode


V = TypeVar("V", bound=Optional[Variable])


# TODO: Refactor this so that it's not a global variable
grad_time: float = 0.0


# TODO: Add `overload` variants
def as_list_or_tuple(
    use_list: bool, use_tuple: bool, outputs: Union[V, Sequence[V]]
) -> Union[V, List[V], Tuple[V, ...]]:
    """Return either a single object or a list/tuple of objects.

    If `use_list` is True, `outputs` is returned as a list (if `outputs`
    is not a list or a tuple then it is converted in a one element list).
    If `use_tuple` is True, `outputs` is returned as a tuple (if `outputs`
    is not a list or a tuple then it is converted into a one element tuple).
    Otherwise (if both flags are false), `outputs` is returned.
    """
    if use_list and use_tuple:
        raise ValueError("Both flags cannot be simultaneously True")

    if use_list or use_tuple:
        if isinstance(outputs, Sequence):
            if use_list:
                return list(outputs)
            else:
                return tuple(outputs)
        else:
            if use_list:
                return [outputs]
            else:
                return (outputs,)
    else:
        if isinstance(outputs, Sequence):
            if len(outputs) != 1:
                raise ValueError("Wrong arguments; expected a one element list")
            return outputs[0]
        else:
            return outputs


def grad_not_implemented(op, x_pos, x, comment=""):
    """Return an un-computable symbolic variable of type `x.type`.

    If any call to `grad` results in an expression containing this
    un-computable variable, an exception (e.g. `NotImplementedError`) will be
    raised indicating that the gradient on the
    `x_pos`'th input of `op` has not been implemented. Likewise if
    any call to aesara.function involves this variable.

    Optionally adds a comment to the exception explaining why this
    gradient is not implemented.
    """

    return (
        NullType(
            (
                "This variable is Null because the grad method for "
                f"input {x_pos} ({x}) of the {op} op is not implemented. {comment}"
            )
        )
    )()


def grad_undefined(op, x_pos, x, comment=""):
    """Return an un-computable symbolic variable of type `x.type`.

    If any call to `grad` results in an expression containing this
    un-computable variable, an exception (e.g. `GradUndefinedError`) will be
    raised indicating that the gradient on the
    `x_pos`'th input of `op` is mathematically undefined. Likewise if
    any call to aesara.function involves this variable.

    Optionally adds a comment to the exception explaining why this
    gradient is not defined.
    """

    return (
        NullType(
            (
                "This variable is Null because the grad method for "
                f"input {x_pos} ({x}) of the {op} op is not implemented. {comment}"
            )
        )
    )()


class DisconnectedType(Type):
    """A type indicating that a variable is the result of taking the gradient of
    ``c`` with respect to ``x`` when ``c`` is not a function of ``x``.

    It serves as a symbolic placeholder for ``0``, but conveys the extra
    information that this gradient is ``0`` because it is disconnected.
    """

    def filter(self, data, strict=False, allow_downcast=None):
        raise AssertionError(
            "If you're assigning to a DisconnectedType you're"
            " doing something wrong. It should only be used as"
            " a symbolic placeholder."
        )

    def fiter_variable(self, other):
        raise AssertionError(
            "If you're assigning to a DisconnectedType you're"
            " doing something wrong. It should only be used as"
            " a symbolic placeholder."
        )

    def may_share_memory(a, b):
        return False

    def value_eq(a, b, force_same_dtype=True):
        raise AssertionError(
            "If you're assigning to a DisconnectedType you're"
            " doing something wrong. It should only be used as"
            " a symbolic placeholder."
        )

    def __str__(self):
        return "DisconnectedType"


disconnected_type = DisconnectedType()


def Rop(
    f: Union[Variable, Sequence[Variable]],
    wrt: Union[Variable, Sequence[Variable]],
    eval_points: Union[Variable, Sequence[Variable]],
    disconnected_outputs: Literal["ignore", "warn", "raise"] = "raise",
    return_disconnected: Literal["none", "zero", "disconnected"] = "zero",
) -> Union[Optional[Variable], Sequence[Optional[Variable]]]:
    """Computes the R-operator applied to `f` with respect to `wrt` at `eval_points`.

    Mathematically this stands for the Jacobian of `f` right multiplied by the
    `eval_points`.

    Parameters
    ----------
    f
        The outputs of the computational graph to which the R-operator is
        applied.
    wrt
        Variables for which the R-operator of `f` is computed.
    eval_points
        Points at which to evaluate each of the variables in `wrt`.
    disconnected_outputs
        Defines the behaviour if some of the variables in `f`
        have no dependency on any of the variable in `wrt` (or if
        all links are non-differentiable). The possible values are:

        - ``'ignore'``: considers that the gradient on these parameters is zero.
        - ``'warn'``: consider the gradient zero, and print a warning.
        - ``'raise'``: raise `DisconnectedInputError`.

    return_disconnected
        - ``'zero'`` : If ``wrt[i]`` is disconnected, return value ``i`` will be
          ``wrt[i].zeros_like()``.
        - ``'none'`` : If ``wrt[i]`` is disconnected, return value ``i`` will be
          ``None``
        - ``'disconnected'`` : returns variables of type `DisconnectedType`

    Returns
    -------
        A symbolic expression such obeying
        ``R_op[i] = sum_j (d f[i] / d wrt[j]) eval_point[j]``,
        where the indices in that expression are magic multidimensional
        indices that specify both the position within a list and all
        coordinates of the tensor elements.
        If `wrt` is a list/tuple, then return a list/tuple with the results.
    """

    if not isinstance(wrt, (list, tuple)):
        _wrt: List[Variable] = [aesara.tensor.as_tensor_variable(wrt)]
    else:
        _wrt = [aesara.tensor.as_tensor_variable(x) for x in wrt]

    if not isinstance(eval_points, (list, tuple)):
        _eval_points: List[Variable] = [aesara.tensor.as_tensor_variable(eval_points)]
    else:
        _eval_points = [aesara.tensor.as_tensor_variable(x) for x in eval_points]

    if not isinstance(f, (list, tuple)):
        _f: List[Variable] = [aesara.tensor.as_tensor_variable(f)]
    else:
        _f = [aesara.tensor.as_tensor_variable(x) for x in f]

    if len(_wrt) != len(_eval_points):
        raise ValueError("`wrt` must be the same length as `eval_points`.")

    # Check that each element of wrt corresponds to an element
    # of eval_points with the same dimensionality.
    for i, (wrt_elem, eval_point) in enumerate(zip(_wrt, _eval_points)):

        try:
            if wrt_elem.type.ndim != eval_point.type.ndim:
                raise ValueError(
                    f"Elements {i} of `wrt` and `eval_point` have mismatched dimensionalities: "
                    f"{wrt_elem.type.ndim} and {eval_point.type.ndim}"
                )
        except AttributeError:
            # wrt_elem and eval_point don't always have ndim like random type
            # Tensor, Sparse have the ndim attribute
            pass

    seen_nodes: Dict[Apply, Sequence[Variable]] = {}

    def _traverse(node):
        """TODO: writeme"""

        if node is None:
            return

        op = node.op
        inputs = node.inputs

        # Compute the evaluation points corresponding to each of the
        # inputs of the node
        local_eval_points = []
        for inp in inputs:
            if inp in _wrt:
                local_eval_points.append(_eval_points[_wrt.index(inp)])
            elif inp.owner is None:
                try:
                    local_eval_points.append(inp.zeros_like())
                except Exception:
                    # None should be used for non-differentiable
                    # arguments, like for example random states
                    local_eval_points.append(None)
            elif inp.owner in seen_nodes:

                local_eval_points.append(
                    seen_nodes[inp.owner][inp.owner.outputs.index(inp)]
                )

            else:
                # We actually need to compute the R_op for this node

                _traverse(inp.owner)
                local_eval_points.append(
                    seen_nodes[inp.owner][inp.owner.outputs.index(inp)]
                )
        same_type_eval_points = []
        for x, y in zip(inputs, local_eval_points):
            if y is not None:
                if not isinstance(x, Variable):
                    x = aesara.tensor.as_tensor_variable(x)
                if not isinstance(y, Variable):
                    y = aesara.tensor.as_tensor_variable(y)
                try:
                    y = x.type.filter_variable(y)
                except TypeError:
                    # This is a hack
                    # Originally both grad and Rop were written
                    # with the assumption that a variable and the
                    # gradient wrt that variable would have the same
                    # dtype. This was a bad assumption because the
                    # gradient wrt an integer can take on non-integer
                    # values.
                    # grad is now fixed, but Rop is not, so when grad
                    # does the right thing and violates this assumption
                    # we have to make it be wrong for Rop to keep working
                    # Rop should eventually be upgraded to handle integers
                    # correctly, the same as grad
                    y = aesara.tensor.cast(y, x.type.dtype)
                    y = x.type.filter_variable(y)
                assert x.type.in_same_class(y.type)
                same_type_eval_points.append(y)
            else:
                same_type_eval_points.append(y)

        seen_nodes[node] = op.R_op(node.inputs, same_type_eval_points)

    # end _traverse

    # Populate the dictionary
    for out in _f:
        _traverse(out.owner)

    rval: List[Optional[Variable]] = []
    for out in _f:
        if out in _wrt:
            rval.append(_eval_points[_wrt.index(out)])
        elif (
            seen_nodes.get(out.owner, None) is None
            or seen_nodes[out.owner][out.owner.outputs.index(out)] is None
        ):
            message = (
                "Rop method was asked to compute the gradient "
                "with respect to a variable that is not part of "
                "the computational graph of variables in wrt, or is "
                f"used only by a non-differentiable operator: {out}"
            )
            if disconnected_outputs == "ignore":
                pass
            elif disconnected_outputs == "warn":
                warnings.warn(message, stacklevel=2)
            elif disconnected_outputs == "raise":
                message = utils.get_variable_trace_string(out)
                raise DisconnectedInputError(message)
            else:
                raise ValueError(
                    "Invalid value for keyword "
                    "'disconnected_inputs', valid values are "
                    "'ignore', 'warn' and 'raise'."
                )
            if return_disconnected.lower() == "zero":
                rval.append(aesara.tensor.zeros_like(out))
            elif return_disconnected.lower() == "none":
                rval.append(None)
            elif return_disconnected.lower() == "disconnected":
                rval.append(disconnected_type())
            else:
                raise ValueError(
                    "Invalid value for keyword "
                    "'return_disconnected', valid values are "
                    "'zero', 'None' and 'Disconnected'."
                )
        else:
            rval.append(seen_nodes[out.owner][out.owner.outputs.index(out)])

    using_list = isinstance(f, list)
    using_tuple = isinstance(f, tuple)
    return as_list_or_tuple(using_list, using_tuple, rval)


def Lop(
    f: Union[Variable, Sequence[Variable]],
    wrt: Union[Variable, Sequence[Variable]],
    eval_points: Union[Variable, Sequence[Variable]],
    consider_constant: Optional[Sequence[Variable]] = None,
    disconnected_inputs: Literal["ignore", "warn", "raise"] = "raise",
) -> Union[Optional[Variable], Sequence[Optional[Variable]]]:
    """Computes the L-operator applied to `f` with respect to `wrt` at `eval_points`.

    Mathematically this stands for the Jacobian of `f` with respect to `wrt`
    left muliplied by the `eval_points`.

    Parameters
    ----------
    f
        The outputs of the computational graph to which the L-operator is
        applied.
    wrt
        Variables for which the L-operator of `f` is computed.
    eval_points
        Points at which to evaluate each of the variables in `wrt`.
    consider_constant
        See `grad`.
    disconnected_inputs
        See `grad`.

    Returns
    -------
        A symbolic expression satisfying
        ``L_op[i] = sum_i (d f[i] / d wrt[j]) eval_point[i]``
        where the indices in that expression are magic multidimensional
        indices that specify both the position within a list and all
        coordinates of the tensor elements.
        If `f` is a list/tuple, then return a list/tuple with the results.
    """
    if not isinstance(eval_points, (list, tuple)):
        _eval_points: List[Variable] = [aesara.tensor.as_tensor_variable(eval_points)]
    else:
        _eval_points = [aesara.tensor.as_tensor_variable(x) for x in eval_points]

    if not isinstance(f, (list, tuple)):
        _f: List[Variable] = [aesara.tensor.as_tensor_variable(f)]
    else:
        _f = [aesara.tensor.as_tensor_variable(x) for x in f]

    grads = list(_eval_points)

    if not isinstance(wrt, (list, tuple)):
        _wrt: List[Variable] = [aesara.tensor.as_tensor_variable(wrt)]
    else:
        _wrt = [aesara.tensor.as_tensor_variable(x) for x in wrt]

    assert len(_f) == len(grads)
    known = dict(zip(_f, grads))

    ret = grad(
        cost=None,
        known_grads=known,
        consider_constant=consider_constant,
        wrt=_wrt,
        disconnected_inputs=disconnected_inputs,
    )

    using_list = isinstance(wrt, list)
    using_tuple = isinstance(wrt, tuple)
    return as_list_or_tuple(using_list, using_tuple, ret)


def grad(
    cost: Optional[Variable],
    wrt: Union[Variable, Sequence[Variable]],
    consider_constant: Optional[Sequence[Variable]] = None,
    disconnected_inputs: Literal["ignore", "warn", "raise"] = "raise",
    add_names: bool = True,
    known_grads: Optional[Mapping[Variable, Variable]] = None,
    return_disconnected: Literal["none", "zero", "disconnected"] = "zero",
    null_gradients: Literal["raise", "return"] = "raise",
) -> Union[Optional[Variable], Sequence[Optional[Variable]]]:
    """
    Return symbolic gradients of one cost with respect to one or more variables.

    For more information about how automatic differentiation works in Aesara,
    see :mod:`gradient`. For information on how to implement the gradient of
    a certain Op, see :func:`grad`.

    Parameters
    ----------
    cost
        Value that we are differentiating (i.e. for which we want the
        gradient).  May be `None` if `known_grads` is provided.
    wrt
        The term(s) with respect to which we want gradients.
    consider_constant
        Expressions not to backpropagate through.
    disconnected_inputs : {'ignore', 'warn', 'raise'}
        Defines the behaviour if some of the variables in `wrt` are
        not part of the computational graph computing `cost` (or if
        all links are non-differentiable). The possible values are:

        - ``'ignore'``: considers that the gradient on these parameters is zero
        - ``'warn'``: consider the gradient zero, and print a warning
        - ``'raise'``: raise `DisconnectedInputError`
    add_names
        If ``True``, variables generated by `grad` will be named
        ``(d<cost.name>/d<wrt.name>)`` provided that both `cost` and `wrt`
        have names.
    known_grads
        An ordered dictionary mapping variables to their gradients. This is
        useful in the case where you know the gradients of some
        variables but do not know the original cost.
    return_disconnected
        - ``'zero'`` : If ``wrt[i]`` is disconnected, return value ``i`` will be
          ``wrt[i].zeros_like()``
        - ``'none'`` : If ``wrt[i]`` is disconnected, return value ``i`` will be
          ``None``
        - ``'disconnected'`` : returns variables of type `DisconnectedType`
    null_gradients
        Defines the behaviour when some of the variables in `wrt` have a
        null gradient. The possibles values are:

        - ``'raise'`` : raise a `NullTypeGradError` exception
        - ``'return'`` : return the null gradients

    Returns
    -------
        A symbolic expression for the gradient of `cost` with respect to each
        of the `wrt` terms.  If an element of `wrt` is not differentiable with
        respect to the output, then a zero variable is returned.

    """
    t0 = time.perf_counter()

    if cost is None:
        if known_grads is None:
            raise ValueError("cost and known_grads can't both be None.")

    if cost is not None and isinstance(cost.type, NullType):
        raise ValueError(
            "Can't differentiate a NaN cost. "
            f"Cost is NaN because {cost.type.why_null}"
        )

    if cost is not None and cost.type.ndim != 0:
        raise TypeError("Cost must be a scalar.")

    if not isinstance(wrt, Sequence):
        _wrt: List[Variable] = [wrt]
    else:
        _wrt = [x for x in wrt]

    outputs = []
    if cost is not None:
        outputs.append(cost)
    if known_grads is not None:
        outputs.extend(list(known_grads.keys()))

    var_to_app_to_idx = _populate_var_to_app_to_idx(outputs, _wrt, consider_constant)

    # build a dict mapping var to the gradient of cost with respect to var
    grad_dict = {}

    if known_grads is None:
        known_grads = {}

    assert isinstance(known_grads, dict)

    # The gradient of the cost is 1 unless specified otherwise by known_grads.
    if cost is not None:
        if cost in known_grads:
            g_cost = known_grads[cost]
        else:
            g_cost = _float_ones_like(cost)
        # g_cost may be Disconnected or NullType. A creative use of the
        # function, sure, but nonetheless one we can and should support.
        # So before we try to cast it make sure it even has a dtype
        if (
            hasattr(g_cost.type, "dtype")
            and cost.type.dtype in aesara.tensor.type.continuous_dtypes
        ):
            # Here we enforce the constraint that floating point variables
            # have the same dtype as their gradient.
            g_cost = g_cost.astype(cost.type.dtype)
        # DO NOT enforce g_cost to be 0 if cost is an integer.
        # This is to be enforced by the Op.grad method for the
        # Op that outputs cost.
        if hasattr(g_cost.type, "dtype"):
            assert g_cost.type.dtype in aesara.tensor.type.continuous_dtypes

        grad_dict[cost] = g_cost

    for var in known_grads:
        g_var = known_grads[var]

        if not hasattr(g_var, "type"):
            raise TypeError(
                "output grads must be aesara variables."
                f"Ambiguous whether {type(g_var)} should be made into tensor"
                " or sparse aesara variable"
            )

        if not isinstance(
            g_var.type, (NullType, DisconnectedType)
        ) and "float" not in str(g_var.type.dtype):
            raise TypeError(
                "Gradients must always be NullType, "
                "DisconnectedType, or continuous, but grad was "
                "given a known_grad of type " + str(g_var.type)
            )

        # DO NOT check that these gradients are equal to 0 if var is int
        # The gradient is allowed to be non-zero on var in that case
        # Ops outputting var should not backpropagate its gradient further
        # but that is enforced elsewhere (grep for only_connected_to_int)

        grad_dict[var] = g_var

    def handle_disconnected(var):
        message = (
            "grad method was asked to compute the gradient "
            "with respect to a variable that is not part of "
            "the computational graph of the cost, or is used "
            f"only by a non-differentiable operator: {var}"
        )
        if disconnected_inputs == "ignore":
            pass
        elif disconnected_inputs == "warn":
            warnings.warn(message, stacklevel=2)
        elif disconnected_inputs == "raise":
            message = utils.get_variable_trace_string(var)
            raise DisconnectedInputError(message)
        else:
            raise ValueError(
                "Invalid value for keyword "
                "'disconnected_inputs', valid values are "
                "'ignore', 'warn' and 'raise'."
            )

    # variables that do not influence the cost have zero gradient.
    # if wrt is such a variable, populate the grad_dict with this info
    # so that wrt not being in var_to_app_to_idx won't cause an error below
    # according to the flag, possibly raise an error if wrt is disconnected
    for elem in _wrt:
        if elem not in var_to_app_to_idx and elem is not cost and elem not in grad_dict:
            handle_disconnected(elem)
            grad_dict[elem] = disconnected_type()

    cost_name = None
    if add_names and cost is not None:
        cost_name = cost.name

    # Make sure we didn't initialize the grad_dict with any ints
    # The gradient may NEVER be an int, even if the variable is an int.
    # Read the Op contract and talk to Ian Goodfellow before changing this!
    for var in grad_dict:
        g = grad_dict[var]
        if hasattr(g.type, "dtype"):
            assert g.type.dtype in aesara.tensor.type.float_dtypes

    _rval: Sequence[Variable] = _populate_grad_dict(
        var_to_app_to_idx, grad_dict, _wrt, cost_name
    )

    rval: MutableSequence[Optional[Variable]] = list(_rval)

    for i in range(len(_rval)):
        if isinstance(_rval[i].type, NullType):
            if null_gradients == "raise":
                raise NullTypeGradError(
                    f"`grad` encountered a NaN. {_rval[i].type.why_null}"
                )
            else:
                assert null_gradients == "return"
        if isinstance(_rval[i].type, DisconnectedType):
            handle_disconnected(_rval[i])
            if return_disconnected == "zero":
                rval[i] = _float_zeros_like(_wrt[i])
            elif return_disconnected.lower() == "none":
                rval[i] = None
            else:
                assert return_disconnected.lower() == "disconnected"

    t1 = time.perf_counter()
    global grad_time
    grad_time += t1 - t0

    if isinstance(wrt, tuple):
        return tuple(rval)
    elif not isinstance(wrt, list):
        return rval[0]

    return rval


def subgraph_grad(wrt, end, start=None, cost=None, details=False):
    """
    With respect to `wrt`, computes gradients of cost and/or from
    existing `start` gradients, up to the `end` variables of a
    symbolic digraph.  In other words, computes gradients for a
    subgraph of the symbolic aesara function. Ignores all disconnected
    inputs.

    This can be useful when one needs to perform the gradient descent
    iteratively (e.g. one layer at a time in an MLP), or when a
    particular operation is not differentiable in aesara
    (e.g. stochastic sampling from a multinomial). In the latter case,
    the gradient of the non-differentiable process could be
    approximated by user-defined formula, which could be calculated
    using the gradients of a cost with respect to samples (0s and
    1s). These gradients are obtained by performing a subgraph_grad
    from the `cost` or previously known gradients (`start`) up to the
    outputs of the stochastic process (`end`).  A dictionary mapping
    gradients obtained from the user-defined differentiation of the
    process, to variables, could then be fed into another
    subgraph_grad as `start` with any other `cost` (e.g. weight
    decay).

    In an MLP, we could use subgraph_grad to iteratively backpropagate:

    .. code-block:: python

        x, t = aesara.tensor.fvector('x'), aesara.tensor.fvector('t')
        w1 = aesara.shared(np.random.standard_normal((3,4)))
        w2 = aesara.shared(np.random.standard_normal((4,2)))
        a1 = aesara.tensor.tanh(aesara.tensor.dot(x,w1))
        a2 = aesara.tensor.tanh(aesara.tensor.dot(a1,w2))
        cost2 = aesara.tensor.sqr(a2 - t).sum()
        cost2 += aesara.tensor.sqr(w2.sum())
        cost1 = aesara.tensor.sqr(w1.sum())

        params = [[w2],[w1]]
        costs = [cost2,cost1]
        grad_ends = [[a1], [x]]

        next_grad = None
        param_grads = []
        for i in range(2):
            param_grad, next_grad = aesara.subgraph_grad(
                wrt=params[i], end=grad_ends[i],
                start=next_grad, cost=costs[i]
            )
            next_grad = dict(zip(grad_ends[i], next_grad))
            param_grads.extend(param_grad)

    Parameters
    ----------

    wrt : list of variables
        Gradients are computed with respect to `wrt`.

    end : list of variables
        Aesara variables at which to end gradient descent (they are
        considered constant in aesara.grad).  For convenience, the
        gradients with respect to these variables are also returned.

    start : dictionary of variables
        If not None, a dictionary mapping variables to their
        gradients. This is useful when the gradient on some variables
        are known. These are used to compute the gradients backwards up
        to the variables in `end` (they are used as known_grad in
        aesara.grad).

    cost : :class:`~aesara.graph.basic.Variable` scalar (0-dimensional) variable
        Additional costs for which to compute the gradients.  For
        example, these could be weight decay, an l1 constraint, MSE,
        NLL, etc. May optionally be None if start is provided.

        .. warning::

            If the gradients of `cost` with respect to any of the `start`
            variables is already part of the `start` dictionary, then it
            may be counted twice with respect to `wrt` and `end`.

    details : bool
        When True, additionally returns the list of gradients from
        `start` and of `cost`, respectively, with respect to `wrt` (not
        `end`).

    Returns
    -------
    Tuple of 2 or 4 Lists of Variables
        Returns lists of gradients with respect to `wrt` and `end`,
        respectively.


    .. versionadded:: 0.7
    """
    if cost is None and start is None:
        raise ValueError("`cost` or `start` must be specified.")

    if not isinstance(end, list):
        raise TypeError("`end` must be a list.")

    if not isinstance(wrt, list):
        raise TypeError("`wrt` must be a list.")

    if start is not None:
        if not isinstance(start, dict):
            raise TypeError("`start` must be a dictionary.")

    params = list(set(wrt + end))

    start_grads = None
    cost_grads = None
    if start is not None:
        start_grads = list(
            aesara.grad(
                cost=None,
                wrt=params,
                known_grads=start,
                consider_constant=end,
                disconnected_inputs="ignore",
            )
        )

    if cost is not None:
        cost_grads = list(
            aesara.grad(
                cost=cost,
                wrt=params,
                consider_constant=end,
                disconnected_inputs="ignore",
            )
        )

    grads = None
    if start is None:
        grads = cost_grads
    else:
        grads = start_grads
        if cost_grads is not None:
            for i in range(len(grads)):
                grads[i] += cost_grads[i]

    pgrads = dict(zip(params, grads))
    # separate wrt from end grads:
    wrt_grads = list(pgrads[k] for k in wrt)
    end_grads = list(pgrads[k] for k in end)

    if details:
        return wrt_grads, end_grads, start_grads, cost_grads

    return wrt_grads, end_grads


def _node_to_pattern(node):
    """given an apply node, obtain its connection pattern
    this is just a wrapper around Op.connection_pattern
    that does type checking and supplies the default value
    if the method is not implemented
    """

    if hasattr(node.op, "connection_pattern"):
        connection_pattern = node.op.connection_pattern(node)

        if not isinstance(connection_pattern, list):
            raise TypeError(
                "Op.connection_pattern should return "
                + f"list of list of bool, but for Op={node.op}"
                + f"got {connection_pattern} with type {type(connection_pattern)}."
            )
        if len(connection_pattern) != len(node.inputs):
            raise ValueError(
                f"{node.op}.connection_pattern should have {len(node.inputs)}"
                + f" rows but has {len(connection_pattern)}."
            )
        for ii, output_pattern in enumerate(connection_pattern):
            if not isinstance(output_pattern, list):
                raise TypeError(
                    f"{node.op}.connection_pattern should return"
                    + f" a list of lists, but element {int(ii)}"
                    + f"is {output_pattern} of type {type(output_pattern)}."
                )
    else:
        connection_pattern = [[True for output in node.outputs] for ipt in node.inputs]
    assert isinstance(connection_pattern, list)
    assert len(connection_pattern) == len(node.inputs)
    for ii in range(len(node.inputs)):
        assert isinstance(connection_pattern[ii], list)
        assert len(connection_pattern[ii]) == len(node.outputs)
    return connection_pattern


def _populate_var_to_app_to_idx(outputs, wrt, consider_constant):
    """
    Helper function for grad function.

    Parameters
    ----------
    outputs
        a list of variables we want to take gradients of

    wrt
        a list of variables we want to take the gradient with
        respect to.

    consider_constant
        a list of variables not to backpropagate through.

    Returns
    -------
    var_to_app_to_idx:
        A dictionary mapping a variable to a second dictionary.
        The second dictionary maps apply nodes acting on this
        variable to the variable's index in the apply node's
        input list.

        This dictionary will only contain variables that
        meet two criteria:

        1) The elements of at least one output are a
           function of the elements of the variable

        2) The elements of the variable are a function of the
           elements of at least one member of wrt.

    This set is exactly the set of variables that connect
    the variables in wrt to the cost being differentiated.

    (A variable in consider_constant is not a function of
    anything)

    """

    # Validate and format consider_constant
    if consider_constant is None:
        consider_constant = []
    else:
        # error checking on consider_constant: verify that it is a collection
        # of aesara variables
        # this is important, if someone accidentally passes a nested data
        # structure with aesara variables at the leaves, only the root will
        # be properly considered constant
        try:
            iter(consider_constant)
        except TypeError:
            raise TypeError(
                "consider_constant must be an iterable collection,"
                " got " + str(type(consider_constant))
            )
        for elem in consider_constant:
            if not isinstance(elem, Variable):
                raise TypeError(
                    "Elements of consider_constant must be "
                    "variables, but got " + str(type(elem))
                )

    # var_to_app_to_idx[var][node] = [i,j] means node has
    # var as input at positions i and j
    var_to_app_to_idx = dict()

    # Set of variables that have been added to their true parents
    # ('true' here means that the elements of the variable are a function
    #  of the elements of the parent, according to the op's
    #  connection_pattern)
    # Note: we need to revisit the apply nodes repeatedly, because
    #       different outputs of the apply node are connected to
    #       different subsets of the inputs.
    accounted_for = set()

    def account_for(var):
        # Don't visit the same variable twice
        if var in accounted_for:
            return
        accounted_for.add(var)

        # Constants are not a function of anything
        if var in consider_constant:
            return

        # Recursively add the variables that this variable is
        # a function of.
        if var.owner is not None:
            app = var.owner

            connection_pattern = _node_to_pattern(app)

            var_idx = app.outputs.index(var)

            for i, ipt in enumerate(app.inputs):

                # don't process ipt if it is not a true
                # parent of var
                if not connection_pattern[i][var_idx]:
                    continue

                if ipt not in var_to_app_to_idx:
                    # This object here *must* be ordered, because
                    # we iterate over its keys when adding up the terms of the
                    # gradient on ipt. If it is a regular dict, the grad method
                    # will return something that is analytically correct, but
                    # whose order of doing additions depends on the memory
                    # location of the apply nodes.
                    var_to_app_to_idx[ipt] = {}
                app_to_idx = var_to_app_to_idx[ipt]
                if app not in app_to_idx:
                    app_to_idx[app] = []
                idx = app_to_idx[app]
                if i not in idx:
                    idx.append(i)
                account_for(ipt)

    # add all variables that are true ancestors of the cost
    for output in outputs:
        account_for(output)

    # determine which variables have elements of wrt as a true
    # ancestor. Do this with an upward pass starting from wrt,
    # following only true connections
    visited = set()

    def visit(var):
        if var in visited:
            return
        if var not in var_to_app_to_idx:
            return
        visited.add(var)
        nodes = var_to_app_to_idx[var]
        for node in nodes:
            connection_pattern = _node_to_pattern(node)
            for idx in nodes[node]:
                for ii, output in enumerate(node.outputs):
                    if connection_pattern[idx][ii]:
                        visit(output)

    for elem in wrt:
        visit(elem)

    # Remove variables that don't have wrt as a true ancestor
    orig_vars = list(var_to_app_to_idx.keys())
    for var in orig_vars:
        if var not in visited:
            del var_to_app_to_idx[var]

    return var_to_app_to_idx


class NullTypeGradError(TypeError):
    """
    Raised when grad encounters a NullType.
    """


class DisconnectedInputError(ValueError):
    """
    Raised when grad is asked to compute the gradient
    with respect to a disconnected input and
    disconnected_inputs='raise'.
    """


def _populate_grad_dict(var_to_app_to_idx, grad_dict, wrt, cost_name=None):
    """Helper function for grad function.

    Parameters
    ----------
    var_to_app_to_idx : dict
        a dictionary mapping a variable to a second dictionary.
        the second dictionary maps apply nodes acting on
        this variable to the variable's index in the apply
        node's input list
    grad_dict : dict
        A dictionary mapping variables to their gradients.
        Should be populated by grad function, which should:

        - Set the gradient with respect to the cost to 1
        - Load all gradients from known_grads, possibly
          overriding the cost
        - Set the gradient for disconnected
          inputs to a variable with type DisconnectedType()

    wrt : list of Variables
        the minimal set of variables that must be included in `grad_dict`
    cost_name: string
        The name of the cost being differentiated, optional.
        Used to name the grad with respect to x as (d<cost_name>/dx)

    Returns
    -------
    list of Variables
        A list of gradients corresponding to `wrt`

    """
    # build a dict mapping node to the terms node contributes to each of
    # its inputs' gradients
    term_dict = {}

    def access_term_cache(node):
        """Populates term_dict[node] and returns it"""

        if node not in term_dict:

            inputs = node.inputs

            output_grads = [access_grad_cache(var) for var in node.outputs]

            # list of bools indicating if each output is connected to the cost
            outputs_connected = [
                not isinstance(g.type, DisconnectedType) for g in output_grads
            ]

            connection_pattern = _node_to_pattern(node)

            # list of bools indicating if each input is connected to the cost
            inputs_connected = [
                (
                    True
                    in [
                        input_to_output and output_to_cost
                        for input_to_output, output_to_cost in zip(
                            input_to_outputs, outputs_connected
                        )
                    ]
                )
                for input_to_outputs in connection_pattern
            ]

            # List of bools indicating if each output is an integer dtype
            output_is_int = [
                hasattr(output.type, "dtype")
                and output.type.dtype in aesara.tensor.type.discrete_dtypes
                for output in node.outputs
            ]

            # List of bools indicating if each output is NullType
            ograd_is_nan = [
                isinstance(output.type, NullType) for output in output_grads
            ]

            # List of bools indicating if each input only has NullType outputs
            only_connected_to_nan = [
                (
                    True
                    not in [
                        in_to_out and out_to_cost and not out_nan
                        for in_to_out, out_to_cost, out_nan in zip(
                            in_to_outs, outputs_connected, ograd_is_nan
                        )
                    ]
                )
                for in_to_outs in connection_pattern
            ]

            if True not in inputs_connected:
                # All outputs of this op are disconnected so we can skip
                # Calling the op's grad method and report that the inputs
                # are disconnected
                # (The op's grad method could do this too, but this saves the
                # implementer the trouble of worrying about this case)
                input_grads = [disconnected_type() for ipt in inputs]
            elif False not in only_connected_to_nan:
                # All inputs are only connected to nan gradients, so we don't
                # need to bother calling the grad method. We know the gradient
                # with respect to all connected inputs is nan.
                input_grads = []
                for connected in inputs_connected:
                    if connected:
                        input_grads.append(null_type())
                    else:
                        input_grads.append(disconnected_type())
            else:
                # At least one input of this op is connected to the cost so and
                # not all output gradients are undefined so we must
                # call the op's grad method

                # Each Op's grad function requires inputs and output_grads
                # If the Op destroys any input, but the grad expression uses
                # it, then chances are the resulting graph will have a
                # dependency cycle. We avoid this cycle by passing (symbolic)
                # copies of each destroyed input.
                try:
                    dinputs = [node.inputs[x[0]] for x in node.op.destroy_map.values()]
                except AttributeError:
                    dinputs = []

                def try_to_copy_if_needed(var):
                    if var in dinputs and hasattr(var, "copy"):
                        return var.copy()
                    return var

                inputs = [try_to_copy_if_needed(ipt) for ipt in inputs]

                # Build a list of output gradients with the same dtype as
                # the corresponding output variable.
                # If an output is of a float dtype, we want to cast the
                # output gradient into the same dtype, to avoid having a
                # gradient graph with double precision (taking more memory,
                # and more computation).
                # If an output is of an integer dtype, then we just leave it
                # alone.
                # DO NOT force integer variables to have zero grad. This causes
                # bugs where we fail to detect disconnected or undefined
                # gradients.
                # DO NOT force integer variables to have integer dtype.
                # This is a violation of the op contract.
                new_output_grads = []
                for o, og in zip(node.outputs, output_grads):
                    o_dt = getattr(o.type, "dtype", None)
                    og_dt = getattr(og.type, "dtype", None)
                    if (
                        o_dt not in aesara.tensor.type.discrete_dtypes
                        and og_dt
                        and o_dt != og_dt
                    ):
                        new_output_grads.append(og.astype(o_dt))
                    else:
                        new_output_grads.append(og)

                # Make sure that, if new_output_grads[i] has a floating point
                # dtype, it is the same dtype as outputs[i]
                for o, ng in zip(node.outputs, new_output_grads):
                    o_dt = getattr(o.type, "dtype", None)
                    ng_dt = getattr(ng.type, "dtype", None)
                    if (
                        ng_dt is not None
                        and o_dt not in aesara.tensor.type.discrete_dtypes
                    ):
                        assert ng_dt == o_dt

                assert all(
                    getattr(ng.type, "dtype", None)
                    not in aesara.tensor.type.discrete_dtypes
                    for ng in new_output_grads
                )

                # If config.compute_test_value is turned on, check that the
                # gradients on the outputs of this node have the right shape.
                # We also check the gradient on the inputs later--both checks
                # are needed, because some gradients are only ever specified
                # by the user, not computed by Op.grad, and some gradients are
                # only computed and returned, but never passed as another
                # node's output grads.
                for idx, packed in enumerate(zip(node.outputs, new_output_grads)):
                    orig_output, new_output_grad = packed
                    if not hasattr(orig_output, "shape"):
                        continue
                    if isinstance(new_output_grad.type, DisconnectedType):
                        continue
                    for orig_output_v, new_output_grad_v in get_test_values(*packed):
                        o_shape = orig_output_v.shape
                        g_shape = new_output_grad_v.shape
                        if o_shape != g_shape:
                            raise ValueError(
                                "Got a gradient of shape "
                                + str(o_shape)
                                + " on an output of shape "
                                + str(g_shape)
                            )

                input_grads = node.op.L_op(inputs, node.outputs, new_output_grads)

                if input_grads is None:
                    raise TypeError(
                        f"{node.op}.grad returned NoneType, expected iterable."
                    )

                if len(input_grads) != len(inputs):
                    raise ValueError(
                        f"{node.op} returned the wrong number of gradient terms."
                    )
            # We can not enforce this, as AdvancedSubtensor1 has an option to
            # return the sparse grad for optimization reason.

            #            for ig, i in zip(input_grads, inputs):
            #                if (not isinstance(ig.type, (DisconnectedType, NullType)) and
            #                    type(ig.type) != type(i.type)):
            #                    raise ValueError(
            #                        "%s returned the wrong type for gradient terms."
            #                        " Sparse inputs must have sparse grads and dense"
            #                        " inputs must have dense grad. Got %s, expected %s" %(
            #                            str(node.op), ig.type, i.type))

            # must convert to list in case the op returns a tuple
            # we won't be able to post-process out the Nones if it does that
            input_grads = list(input_grads)

            # Need to propagate the NullType gradients; if an input grad is
            # not disconnected and the corresponding input is connected
            # to at least one output whose gradient is NullType then the input
            # grad should be NullType.
            for inp_idx in range(len(input_grads)):
                for out_idx in range(len(ograd_is_nan)):
                    if (
                        ograd_is_nan[out_idx]
                        and connection_pattern[inp_idx][out_idx]
                        and not isinstance(input_grads[inp_idx].type, DisconnectedType)
                    ):
                        input_grads[inp_idx] = output_grads[out_idx]

            # Do type checking on the result

            # List of bools indicating if each input only has integer outputs
            only_connected_to_int = [
                (
                    True
                    not in [
                        in_to_out and out_to_cost and not out_int
                        for in_to_out, out_to_cost, out_int in zip(
                            in_to_outs, outputs_connected, output_is_int
                        )
                    ]
                )
                for in_to_outs in connection_pattern
            ]

            for i, term in enumerate(input_grads):

                # Disallow Nones
                if term is None:
                    # We don't know what None means. in the past it has been
                    # used to mean undefined, zero, or disconnected.
                    # We therefore don't allow it because its usage has become
                    # so muddied.
                    raise TypeError(
                        (
                            f"{node.op}.grad returned None for a gradient term, "
                            "this is prohibited. Instead of None,"
                            "return zeros_like(input), disconnected_type(),"
                            " or a NullType variable such as those made with "
                            "the grad_undefined or grad_unimplemented helper "
                            "functions."
                        )
                    )

                # Check that the gradient term for this input
                # has the right shape
                if hasattr(term, "shape"):
                    orig_ipt = inputs[i]
                    if not isinstance(orig_ipt, NominalVariable):
                        for orig_ipt_v, term_v in get_test_values(orig_ipt, term):
                            i_shape = orig_ipt_v.shape
                            t_shape = term_v.shape
                            if i_shape != t_shape:
                                raise ValueError(
                                    f"{node.op}.grad returned object of "
                                    f"shape {t_shape} as gradient term on input {int(i)} "
                                    f"of shape {i_shape}"
                                )

                if not isinstance(term.type, (NullType, DisconnectedType)):
                    if term.type.dtype not in aesara.tensor.type.float_dtypes:
                        raise TypeError(
                            str(node.op) + ".grad illegally "
                            " returned an integer-valued variable."
                            f" (Input index {int(i)}, dtype {term.type.dtype})"
                        )

                    if only_connected_to_nan[i]:
                        assert isinstance(term.type, NullType)

                    if only_connected_to_int[i]:
                        # This term has only integer outputs and we know
                        # it's not undefined or disconnected
                        # The only other valid thing it can be is 0

                        is_zero = _is_zero(term)
                        assert is_zero in ("yes", "no", "maybe")
                        if is_zero == "maybe":
                            msg = (
                                f"{node.op}.grad returned {term} of type {type(term)} for input"
                                f" {i}. This input's only connections to "
                                "the cost through this op are via "
                                "integer-valued outputs so it should be "
                                "NullType, DisconnectedType, or some form "
                                "of zeros. It is not NullType or "
                                "DisconnectedType and aesara can't "
                                "simplify it to a constant, so it's not "
                                "verifiably zeros."
                            )
                        elif is_zero == "no":
                            msg = (
                                f"{node.op}.grad returned {term} of type {type(term)} for input"
                                f" {i}. Since this input is only connected "
                                "to integer-valued outputs, it should "
                                "evaluate to zeros, but it evaluates to"
                                f"{aesara.get_scalar_constant_value(term)}."
                            )
                            raise ValueError(msg)

            # Check that op.connection_pattern matches the connectivity
            # logic driving the op.grad method
            for i, (ipt, ig, connected) in enumerate(
                zip(inputs, input_grads, inputs_connected)
            ):
                actually_connected = not isinstance(ig.type, DisconnectedType)

                if actually_connected and not connected:
                    msg = (
                        f"{node.op}.grad returned {ig} of type {ig.type} for input {i}."
                        " Expected DisconnectedType instance based on "
                        " the output of the op's connection_pattern "
                        "method."
                    )
                    raise TypeError(msg)

                elif connected and not actually_connected:
                    msg = f"{node.op}.grad returned DisconnectedType for input {i}."
                    if hasattr(node.op, "connection_pattern"):
                        msg += " Its connection_pattern method does not" " allow this."
                        raise TypeError(msg)
                    else:
                        msg += (
                            " You may want to implement a "
                            "connection_pattern method for it."
                        )
                        warnings.warn(msg)

            # cache the result
            term_dict[node] = input_grads

        return term_dict[node]

    # populate grad_dict[var] and return it
    def access_grad_cache(var):
        if var not in grad_dict:
            # If var is not in grad_dict already, we must compute it
            if var in var_to_app_to_idx:
                null_terms = []
                terms = []
                node_to_idx = var_to_app_to_idx[var]
                for node in node_to_idx:
                    for idx in node_to_idx[node]:

                        term = access_term_cache(node)[idx]

                        if not isinstance(term, Variable):
                            raise TypeError(
                                f"{node.op}.grad returned {type(term)}, expected"
                                " Variable instance."
                            )

                        if isinstance(term.type, NullType):
                            null_terms.append(term)
                            continue

                        # Don't try to sum up DisconnectedType placeholders
                        if isinstance(term.type, DisconnectedType):
                            continue

                        if hasattr(var, "ndim") and term.ndim != var.ndim:
                            raise ValueError(
                                (
                                    f"{node.op}.grad returned a term with"
                                    f" {int(term.ndim)} dimensions, but {int(var.ndim)} are required."
                                )
                            )

                        terms.append(term)

                # Add up the terms to get the total gradient on this variable
                if len(null_terms) > 0:
                    # At least one term is a NullType : the total gradient
                    # will also be a NullType
                    grad_dict[var] = null_terms[0]
                elif len(terms) > 0:
                    # the next line is like sum(terms) but doesn't add an
                    # extraneous TensorConstant(0)
                    grad_dict[var] = reduce(lambda x, y: x + y, terms)
                else:
                    grad_dict[var] = disconnected_type()

                if cost_name is not None and var.name is not None:
                    grad_dict[var].name = f"(d{cost_name}/d{var.name})"
            else:
                # this variable isn't connected to the cost in the
                # computational graph
                grad_dict[var] = disconnected_type()
        # end if cache miss
        return grad_dict[var]

    rval = [access_grad_cache(elem) for elem in wrt]

    return rval


def _float_zeros_like(x):
    """Like zeros_like, but forces the object to have a
    a floating point dtype"""

    rval = x.zeros_like()

    if rval.type.dtype.find("float") != -1:
        return rval

    return rval.astype(config.floatX)


def _float_ones_like(x):
    """Like ones_like, but forces the object to have a
    floating point dtype"""

    dtype = x.type.dtype
    if dtype not in aesara.tensor.type.float_dtypes:
        dtype = config.floatX

    return x.ones_like(dtype=dtype)


class numeric_grad:
    """
    Compute the numeric derivative of a scalar-valued function at a particular
    point.
    """

    # Note on step sizes and tolerances:
    #
    # There is a relationship between the step size and the function value and
    # the measurement error that is incurred due to rounding.  The finite
    # difference we measure is
    # delta = f(x0) - f(x0+eps)
    #
    # For maximum precision, f should be close to zero.
    # For every power of 2 that f departs from zero, we lose a bit of precision
    # in delta.
    #
    # Even in this case of maximum accuracy, there is a tradeoff between
    # stepsize and measurement error.
    # Taking small steps allows us to measure large derivatives accuractly,
    # but longer steps are required to measure small derivatives accurately.
    # However longer steps introduce bias into our measurement in general
    # for non-linear functions.
    #
    # It would be interesting to have a version of numeric grad that used an
    # adaptive stepsize.
    #
    # For now, we use a heuristic that catches very bad gradients, but is not
    # perfectly accurate.
    type_eps = {
        "float64": 1e-7,
        "float32": 3e-4,
        "float16": 1e-1,
        np.dtype("float64"): 1e-7,
        np.dtype("float32"): 3e-4,
        np.dtype("float16"): 1e-1,
    }

    def __init__(self, f, pt, eps=None, out_type=None):
        """Return the gradient of f at pt.

        This function computes the gradient by a one-sided finite
        differences of a fixed step size (eps).

        Parameters
        ----------
        f : a differentiable function such that f(*pt) is a scalar
            The function to compute the gradient of.
            It is assumed that f(...) will return a scalar.
            It is assumed that all f's inputs are numpy.ndarray objects.
        pt : an ndarray, a list of ndarrays or tuple of ndarrays
            The point where to evaluate the gradient
        out_type: float
            dtype of output, if complex (i.e. 'complex32' or 'complex64')
        eps : float, optional
            The stepsize for the finite differencing.  None means
            input dtype-dependent. See `type_eps`.
        """

        def prod(inputs):
            rval = 1
            for i in inputs:
                rval *= i
            return rval

        packed_pt = False
        if not isinstance(pt, (list, tuple)):
            pt = [pt]
            packed_pt = True

        apt = [np.array(p) for p in pt]

        shapes = [p.shape for p in apt]
        dtypes = [str(p.dtype) for p in apt]

        # TODO: remove this eventually (why was this here in the first place ?)
        # In the case of CSM, the arguments are a mixture of floats and
        # integers...
        # if not dtypes == [dtypes[0]] * len(apt):
        #      raise TypeError('All function arguments must have same dtype')

        total_size = sum(prod(sh) for sh in shapes)

        working_dtype = min((self.type_eps[dt], dt) for dt in dtypes)[1]

        # create un-initialized memory
        x = np.ndarray((total_size,), dtype=working_dtype)
        # (not out_type is None) --> (out_type is not None) ???
        if (out_type is not None) and (out_type.startswith("complex")):
            gx = np.ndarray((total_size,), dtype=out_type)
        else:
            gx = np.ndarray((total_size,), dtype=working_dtype)

        if eps is None:
            eps = max(self.type_eps[dt] for dt in dtypes)

        # set up aliases so that apt[i] is backed by memory in x
        # and self.gf is backed by memory in gx
        cur_pos = 0
        self.gf = []
        for i, p in enumerate(apt):
            p_size = prod(p.shape)
            # set up alias
            apt[i] = x[cur_pos : cur_pos + p_size].reshape(p.shape)
            self.gf.append(gx[cur_pos : cur_pos + p_size].reshape(p.shape))
            # initialize with p's value
            apt[i][...] = p
            cur_pos += p_size

        f_x = f(*[p.copy() for p in apt])

        # now iterate over the elements of x, and call f on apt.
        x_copy = x.copy()
        for i in range(total_size):
            x[:] = x_copy

            x[i] += eps
            f_eps = f(*apt)

            # TODO: remove this when it is clear that the next
            # replacemement does not pose problems of its own.  It was replaced
            # for its inability to handle complex variables.
            # gx[i] = numpy.asarray((f_eps - f_x) / eps)

            gx[i] = (f_eps - f_x) / eps

        if packed_pt:
            self.gf = self.gf[0]

    @staticmethod
    def abs_rel_err(a, b):
        """Return absolute and relative error between a and b.

        The relative error is a small number when a and b are close, relative
        to how big they are.

        Formulas used:
            abs_err = abs(a - b)

            rel_err = abs_err / max(abs(a) + abs(b), 1e-8)

        The denominator is clipped at 1e-8 to avoid dividing by 0 when a and b
        are both close to 0.

        The tuple (abs_err, rel_err) is returned
        """
        abs_err = abs(a - b)
        # 1e-8 is to prevent division by zeros.
        # [] is to make sure that if a and b are float16, 1e-8 don't get
        # dowcasted to float16 as that give 0! This would add back the
        # division by zero
        rel_err = abs_err / np.maximum(abs(a) + abs(b), [1e-8])
        # The numpy.asarray are needed as if a or b is a sparse matrix
        # this would result in a numpy.matrix and not a numpy.ndarray
        # and the behave differently causing problem later.
        # In particular a_npy_matrix.flatten().shape == (1, n_element)
        abs_err = np.asarray(abs_err)
        rel_err = np.asarray(rel_err)
        return (abs_err, rel_err)

    def abs_rel_errors(self, g_pt):
        """Return the abs and rel error of gradient estimate `g_pt`

        `g_pt` must be a list of ndarrays of the same length as self.gf,
        otherwise a ValueError is raised.

        Corresponding ndarrays in `g_pt` and `self.gf` must have the same
        shape or ValueError is raised.

        """
        if len(g_pt) != len(self.gf):
            raise ValueError("argument has wrong number of elements", len(g_pt))
        errs = []
        for i, (a, b) in enumerate(zip(g_pt, self.gf)):
            if a.shape != b.shape:
                raise ValueError(
                    f"argument element {i} has wrong shapes {a.shape}, {b.shape}"
                )
            errs.append(numeric_grad.abs_rel_err(a, b))
        return errs

    def max_err(self, g_pt, abs_tol, rel_tol):
        """Find the biggest error between g_pt and self.gf.

        What is measured is the violation of relative and absolute errors,
        wrt the provided tolerances (abs_tol, rel_tol).
        A value > 1 means both tolerances are exceeded.

        Return the argmax of min(abs_err / abs_tol, rel_err / rel_tol) over
        g_pt, as well as abs_err and rel_err at this point.
        """
        pos = []
        errs = []
        abs_errs = []
        rel_errs = []

        abs_rel_errs = self.abs_rel_errors(g_pt)
        for abs_err, rel_err in abs_rel_errs:
            if not np.all(np.isfinite(abs_err)):
                raise ValueError("abs_err not finite", repr(abs_err))
            if not np.all(np.isfinite(rel_err)):
                raise ValueError("rel_err not finite", repr(rel_err))
            scaled_err = np.minimum(abs_err / abs_tol, rel_err / rel_tol)
            max_i = scaled_err.argmax()

            pos.append(max_i)
            errs.append(scaled_err.flatten()[max_i])
            abs_errs.append(abs_err.flatten()[max_i])
            rel_errs.append(rel_err.flatten()[max_i])

        # max over the arrays in g_pt
        max_arg = np.argmax(errs)
        max_pos = pos[max_arg]
        return (max_arg, max_pos, abs_errs[max_arg], rel_errs[max_arg])


def mode_not_slow(mode):
    from aesara.compile.debugmode import DebugMode
    from aesara.compile.mode import FAST_RUN, get_mode

    if mode == "FAST_COMPILE":
        return FAST_RUN
    mode = get_mode(mode)
    if isinstance(mode, DebugMode):
        opt = mode.optimizer
        return FAST_RUN.clone(optimizer=opt)
    else:
        return mode


def verify_grad(
    fun: Callable,
    pt: List[np.ndarray],
    n_tests: int = 2,
    rng: Optional[Union[np.random.Generator, np.random.RandomState]] = None,
    eps: Optional[float] = None,
    out_type: Optional[str] = None,
    abs_tol: Optional[float] = None,
    rel_tol: Optional[float] = None,
    mode: Optional[Union["Mode", str]] = None,
    cast_to_output_type: bool = False,
    no_debug_ref: bool = True,
):
    """Test a gradient by Finite Difference Method. Raise error on failure.

    Raises an Exception if the difference between the analytic gradient and
    numerical gradient (computed through the Finite Difference Method) of a
    random projection of the fun's output to a scalar exceeds the given
    tolerance.

    Examples
    --------
    >>> verify_grad(aesara.tensor.tanh,
    ...             (np.asarray([[2, 3, 4], [-1, 3.3, 9.9]]),),
    ...             rng=np.random.default_rng(23098))

    Parameters
    ----------
    fun
        `fun` takes Aesara variables as inputs, and returns an Aesara variable.
        For instance, an `Op` instance with  a single output.
    pt
        Input values, points where the gradient is estimated.
        These arrays must be either float16, float32, or float64 arrays.
    n_tests
        Number o to run the test.
    rng
        Random number generator used to sample the output random projection `u`,
        we test gradient of ``sum(u * fun)`` at `pt`.
    eps
        Step size used in the Finite Difference Method (Default
        ``None`` is type-dependent).
        Raising the value of `eps` can raise or lower the absolute
        and relative errors of the verification depending on the
        `Op`. Raising `eps` does not lower the verification quality for
        linear operations. It is better to raise `eps` than raising
        `abs_tol` or `rel_tol`.
    out_type
        Dtype of output, if complex (i.e., ``'complex32'`` or ``'complex64'``)
    abs_tol
        Absolute tolerance used as threshold for gradient comparison
    rel_tol
        Relative tolerance used as threshold for gradient comparison
    cast_to_output_type
        If the output is float32 and `cast_to_output_type` is ``True``, cast
        the random projection to float32; otherwise, it is float64.
        float16 is not handled here.
    no_debug_ref
        Don't use `DebugMode` for the numerical gradient function.

    Notes
    -----
    This function does not support multiple outputs. In `tests.scan.test_basic`
    there is an experimental `verify_grad` that covers that case as well by
    using random projections.

    """
    from aesara.compile.function import function
    from aesara.compile.sharedvalue import shared

    if not isinstance(pt, (list, tuple)):
        raise TypeError("`pt` should be a list or tuple")

    pt = [np.array(p) for p in pt]

    for i, p in enumerate(pt):
        if p.dtype not in ("float16", "float32", "float64"):
            raise TypeError(
                (
                    "verify_grad can work only with floating point "
                    f'inputs, but input {i} has dtype "{p.dtype}".'
                )
            )

    _type_tol = dict(  # relative error tolerances for different types
        float16=5e-2, float32=1e-2, float64=1e-4
    )

    if abs_tol is None:
        abs_tol = max(_type_tol[str(p.dtype)] for p in pt)
    if rel_tol is None:
        rel_tol = max(_type_tol[str(p.dtype)] for p in pt)

    if rng is None:
        raise TypeError(
            "rng should be a valid instance of "
            "numpy.random.RandomState. You may "
            "want to use tests.unittest"
            "_tools.verify_grad instead of "
            "aesara.gradient.verify_grad."
        )

    # We allow input downcast in `function`, because `numeric_grad` works in
    # the most precise dtype used among the inputs, so we may need to cast
    # some.
    fn_maker = partial(
        function,
        accept_inplace=True,
        allow_input_downcast=True,
        on_unused_input="ignore",
        mode=mode,
    )

    tensor_pt = []
    for i, p in enumerate(pt):
        p_t = aesara.tensor.as_tensor_variable(p).type()
        p_t.name = f"input {i}"
        tensor_pt.append(p_t)

    # fun can be either a function or an actual Op instance
    o_output = fun(*tensor_pt)

    if isinstance(o_output, list):
        raise NotImplementedError(
            "Can't (yet) auto-test the gradient of a function with multiple outputs"
        )
        # we could make loop over outputs making random projections R for each,
        # but this doesn't handle the case where not all the outputs are
        # differentiable... so I leave this as TODO for now -JB.

    o_fn = fn_maker(tensor_pt, o_output, name="gradient.py fwd")
    o_fn_out = o_fn(*[p.copy() for p in pt])

    if isinstance(o_fn_out, tuple) or isinstance(o_fn_out, list):
        raise TypeError(
            "It seems like you are trying to use verify_grad "
            "on an Op or a function which outputs a list: there should"
            " be a single (array-like) output instead"
        )

    # random_projection should not have elements too small,
    # otherwise too much precision is lost in numerical gradient
    def random_projection():
        plain = rng.random(o_fn_out.shape) + 0.5
        if cast_to_output_type and o_output.dtype == "float32":
            return np.array(plain, o_output.dtype)
        return plain

    t_r = shared(random_projection(), borrow=True)
    t_r.name = "random_projection"

    # random projection of o onto t_r
    # This sum() is defined above, it's not the builtin sum.
    cost = aesara.tensor.sum(t_r * o_output)

    if no_debug_ref:
        mode_for_cost = mode_not_slow(mode)
    else:
        mode_for_cost = mode

    cost_fn = fn_maker(tensor_pt, cost, name="gradient.py cost", mode=mode_for_cost)

    symbolic_grad = grad(cost, tensor_pt, disconnected_inputs="ignore")

    grad_fn = fn_maker(tensor_pt, symbolic_grad, name="gradient.py symbolic grad")

    for test_num in range(n_tests):
        num_grad = numeric_grad(cost_fn, [p.copy() for p in pt], eps, out_type)

        analytic_grad = grad_fn(*[p.copy() for p in pt])

        # Since `tensor_pt` is a list, `analytic_grad` should be one too.
        assert isinstance(analytic_grad, list)

        max_arg, max_err_pos, max_abs_err, max_rel_err = num_grad.max_err(
            analytic_grad, abs_tol, rel_tol
        )

        if max_abs_err > abs_tol and max_rel_err > rel_tol:

            raise GradientError(
                max_arg,
                max_err_pos,
                analytic_grad[max_arg].shape,
                analytic_grad[max_arg].flatten()[max_err_pos],
                num_grad.gf[max_arg].flatten()[max_err_pos],
                max_abs_err,
                max_rel_err,
                abs_tol,
                rel_tol,
            )

        # get new random projection for next test
        if test_num < n_tests - 1:
            t_r.set_value(random_projection(), borrow=True)


class GradientError(Exception):
    """This error is raised when a gradient is incorrectly calculated."""

    def __init__(
        self, arg, err_pos, shape, val1, val2, abs_err, rel_err, abs_tol, rel_tol
    ):
        super().__init__()
        self.arg = arg
        self.err_pos = err_pos
        self.shape = shape
        self.val1 = val1
        self.val2 = val2
        self.abs_err = abs_err
        self.rel_err = rel_err
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol

    def __str__(self):
        if hasattr(self, "args"):
            # `self.args` may have been inserted by
            # `tests.tensor.utils.makeTester`
            args_msg = ", ".join(str(a) for a in self.args)
        else:
            args_msg = ""

        return f"""\
GradientError: numeric gradient and analytic gradient exceed tolerance:
        At position {self.err_pos} of argument {self.arg} with shape {self.shape},
            val1 = {self.val1:f}      ,  val2 = {self.val2:f}
            abs. error = {self.abs_err:f},  abs. tolerance = {self.abs_tol:f}
            rel. error = {self.rel_err:f},  rel. tolerance = {self.rel_tol:f}
Exception args: {args_msg}"""


def jacobian(expression, wrt, consider_constant=None, disconnected_inputs="raise"):
    """
    Compute the full Jacobian, row by row.

    Parameters
    ----------
    expression : Vector (1-dimensional) :class:`~aesara.graph.basic.Variable`
        Values that we are differentiating (that we want the Jacobian of)
    wrt : :class:`~aesara.graph.basic.Variable` or list of Variables
        Term[s] with respect to which we compute the Jacobian
    consider_constant : list of variables
        Expressions not to backpropagate through

    disconnected_inputs: string
        Defines the behaviour if some of the variables
        in `wrt` are not part of the computational graph computing `cost`
        (or if all links are non-differentiable). The possible values are:

        - 'ignore': considers that the gradient on these parameters is zero.
        - 'warn': consider the gradient zero, and print a warning.
        - 'raise': raise an exception.

    Returns
    -------
    :class:`~aesara.graph.basic.Variable` or list/tuple of Variables (depending upon `wrt`)
        The Jacobian of `expression` with respect to (elements of) `wrt`.
        If an element of `wrt` is not differentiable with respect to the
        output, then a zero variable is returned. The return value is
        of same type as `wrt`: a list/tuple or TensorVariable in all cases.
    """

    if not isinstance(expression, Variable):
        raise TypeError("jacobian expects a Variable as `expression`")

    if expression.ndim > 1:
        raise ValueError(
            "jacobian expects a 1 dimensional variable as `expression`."
            " If not use flatten to make it a vector"
        )

    using_list = isinstance(wrt, list)
    using_tuple = isinstance(wrt, tuple)

    if isinstance(wrt, (list, tuple)):
        wrt = list(wrt)
    else:
        wrt = [wrt]

    if expression.ndim == 0:
        # expression is just a scalar, use grad
        return as_list_or_tuple(
            using_list,
            using_tuple,
            grad(
                expression,
                wrt,
                consider_constant=consider_constant,
                disconnected_inputs=disconnected_inputs,
            ),
        )

    def inner_function(*args):
        idx = args[0]
        expr = args[1]
        rvals = []
        for inp in args[2:]:
            rval = grad(
                expr[idx],
                inp,
                consider_constant=consider_constant,
                disconnected_inputs=disconnected_inputs,
            )
            rvals.append(rval)
        return rvals

    # Computing the gradients does not affect the random seeds on any random
    # generator used n expression (because during computing gradients we are
    # just backtracking over old values. (rp Jan 2012 - if anyone has a
    # counter example please show me)
    jacobs, updates = aesara.scan(
        inner_function,
        sequences=aesara.tensor.arange(expression.shape[0]),
        non_sequences=[expression] + wrt,
    )
    assert not updates, "Scan has returned a list of updates; this should not happen."
    return as_list_or_tuple(using_list, using_tuple, jacobs)


def hessian(cost, wrt, consider_constant=None, disconnected_inputs="raise"):
    """
    Parameters
    ----------
    cost: Scalar (0-dimensional) variable.
    wrt: Vector (1-dimensional tensor) 'Variable' or list of
    vectors (1-dimensional tensors) Variables
    consider_constant:
        a list of expressions not to backpropagate through
    disconnected_inputs: string
        Defines the behaviour if some of the variables
        in ``wrt`` are not part of the computational graph computing ``cost``
        (or if all links are non-differentiable). The possible values are:

        - 'ignore': considers that the gradient on these parameters is zero.
        - 'warn': consider the gradient zero, and print a warning.
        - 'raise': raise an exception.

    Returns
    -------
    :class:`~aesara.graph.basic.Variable` or list/tuple of Variables
        The Hessian of the `cost` with respect to (elements of) `wrt`.
        If an element of `wrt` is not differentiable with respect to the
        output, then a zero variable is returned. The return value is
        of same type as `wrt`: a list/tuple or TensorVariable in all cases.
    """

    # Check inputs have the right format
    if not isinstance(cost, Variable):
        raise TypeError("hessian expects a Variable as `cost`")

    if cost.ndim != 0:
        raise ValueError("hessian expects a 0 dimensional variable as `cost`")

    using_list = isinstance(wrt, list)
    using_tuple = isinstance(wrt, tuple)

    if isinstance(wrt, (list, tuple)):
        wrt = list(wrt)
    else:
        wrt = [wrt]

    hessians = []
    for input in wrt:

        if not isinstance(input, Variable):
            raise TypeError("hessian expects a (list of) Variable as `wrt`")

        if input.ndim != 1:
            raise ValueError(
                "hessian expects a (list of) 1 dimensional variable as `wrt`"
            )

        expr = grad(
            cost,
            input,
            consider_constant=consider_constant,
            disconnected_inputs=disconnected_inputs,
        )

        # It is possible that the inputs are disconnected from expr,
        # even if they are connected to cost.
        # This should not be an error.
        hess, updates = aesara.scan(
            lambda i, y, x: grad(
                y[i],
                x,
                consider_constant=consider_constant,
                disconnected_inputs="ignore",
            ),
            sequences=aesara.tensor.arange(expr.shape[0]),
            non_sequences=[expr, input],
        )
        assert (
            not updates
        ), "Scan has returned a list of updates; this should not happen."
        hessians.append(hess)
    return as_list_or_tuple(using_list, using_tuple, hessians)


def _is_zero(x):
    """
    Returns 'yes', 'no', or 'maybe' indicating whether x
    is always 0.
    'maybe' means that x is an expression that is complicated enough
    that we can't tell that it simplifies to 0.
    """
    if not hasattr(x, "type"):
        return np.all(x == 0.0)
    if isinstance(x.type, NullType):
        return "no"
    if isinstance(x.type, DisconnectedType):
        return "yes"

    no_constant_value = True
    try:
        constant_value = aesara.get_scalar_constant_value(x)
        no_constant_value = False
    except aesara.tensor.exceptions.NotScalarConstantError:
        pass

    if no_constant_value:
        return "maybe"

    if constant_value != 0.0:
        return "no"

    return "yes"


class ConsiderConstant(ViewOp):
    def grad(self, args, g_outs):
        return [g_out.zeros_like(g_out) for g_out in g_outs]


consider_constant_ = ConsiderConstant()


def consider_constant(x):
    """Consider an expression constant when computing gradients.

    DEPRECATED: use `zero_grad` or `disconnected_grad` instead.

    The expression itself is unaffected, but when its gradient is
    computed, or the gradient of another expression that this
    expression is a subexpression of, it will not be backpropagated
    through. In other words, the gradient of the expression is
    truncated to 0.

    :param x: A Aesara expression whose gradient should be truncated.

    :return: The expression is returned unmodified, but its gradient
        is now truncated to 0.

    .. versionadded:: 0.7
    """
    warnings.warn(
        (
            "`ConsiderConstant` is deprecated; use `zero_grad` or "
            "`disconnected_grad` instead."
        ),
        category=DeprecationWarning,
        stacklevel=3,
    )

    return ConsiderConstant()(x)


class ZeroGrad(ViewOp):
    def grad(self, args, g_outs):
        return [g_out.zeros_like(g_out) for g_out in g_outs]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return [None]

        return aesara.tensor.zeros(1)


zero_grad_ = ZeroGrad()


def zero_grad(x):
    """
    Consider an expression constant when computing gradients.

    The expression itself is unaffected, but when its gradient is
    computed, or the gradient of another expression that this
    expression is a subexpression of, it will be backpropagated
    through with a value of zero. In other words, the gradient of
    the expression is truncated to 0.

    Parameters
    ----------
    x: :class:`~aesara.graph.basic.Variable`
        A Aesara expression whose gradient should be truncated.

    Returns
    -------
    :class:`~aesara.graph.basic.Variable`
        An expression equivalent to ``x``, with its gradient
        truncated to 0.
    """
    return zero_grad_(x)


class UndefinedGrad(ViewOp):
    def grad(self, args, g_outs):
        return [grad_undefined(self, i, arg) for i, arg in enumerate(args)]

    def R_op(self, inputs, eval_points):
        return [None]

    def connection_pattern(self, node):
        return [[True]]


undefined_grad_ = UndefinedGrad()


def undefined_grad(x):
    """
    Consider the gradient of this variable undefined.

    This will generate an error message if its gradient is taken.

    The expression itself is unaffected, but when its gradient is
    computed, or the gradient of another expression that this
    expression is a subexpression of, an error message will be generated
    specifying such gradient is not defined.

    Parameters
    ----------
    x: :class:`~aesara.graph.basic.Variable`
        A Aesara expression whose gradient should be undefined.

    Returns
    -------
    :class:`~aesara.graph.basic.Variable`
        An expression equivalent to ``x``, with its gradient undefined.
    """
    return undefined_grad_(x)


class DisconnectedGrad(ViewOp):
    def grad(self, args, g_outs):
        return [disconnected_type() for g_out in g_outs]

    def R_op(self, inputs, eval_points):
        return [None]

    def connection_pattern(self, node):
        return [[False]]


disconnected_grad_ = DisconnectedGrad()


def disconnected_grad(x):
    """
    Consider an expression constant when computing gradients.

    It will effectively not backpropagating through it.

    The expression itself is unaffected, but when its gradient is
    computed, or the gradient of another expression that this
    expression is a subexpression of, it will not be backpropagated
    through. This is effectively equivalent to truncating the gradient
    expression to 0, but is executed faster than zero_grad(), which stilll
    has to go through the underlying computational graph related to the
    expression.

    Parameters
    ----------
    x: :class:`~aesara.graph.basic.Variable`
        A Aesara expression whose gradient should not be
        backpropagated through.

    Returns
    -------
    :class:`~aesara.graph.basic.Variable`
        An expression equivalent to ``x``, with its gradient
        now effectively truncated to 0.
    """
    return disconnected_grad_(x)


class GradClip(ViewOp):
    # See doc in user fct grad_clip
    __props__ = ()

    def __init__(self, clip_lower_bound, clip_upper_bound):
        # We do not put those member in __eq__ or __hash__
        # as they do not influence the perform of this op.
        self.clip_lower_bound = clip_lower_bound
        self.clip_upper_bound = clip_upper_bound

        if not self.clip_upper_bound >= self.clip_lower_bound:
            raise ValueError("`clip_upper_bound` should be >= `clip_lower_bound`")

    def grad(self, args, g_outs):
        return [
            aesara.tensor.clip(g_out, self.clip_lower_bound, self.clip_upper_bound)
            for g_out in g_outs
        ]


def grad_clip(x, lower_bound, upper_bound):
    """
    This op do a view in the forward, but clip the gradient.

    This is an elemwise operation.

    Parameters
    ----------
    x:
        The variable we want its gradient inputs clipped
    lower_bound:
        The lower bound of the gradient value
    upper_bound:
        The upper bound of the gradient value.

    Examples
    --------
    >>> x = aesara.tensor.type.scalar()
    >>> z = aesara.gradient.grad(grad_clip(x, -1, 1)**2, x)
    >>> z2 = aesara.gradient.grad(x**2, x)
    >>> f = aesara.function([x], outputs = [z, z2])
    >>> print(f(2.0))
    [array(1.0), array(4.0)]

    Notes
    -----
    We register an opt in tensor/opt.py that remove the GradClip.
    So it have 0 cost in the forward and only do work in the grad.

    """
    return GradClip(lower_bound, upper_bound)(x)


class GradScale(ViewOp):
    def __init__(self, multiplier):
        self.multiplier = multiplier

    def grad(self, args, g_outs):
        return [self.multiplier * g_out for g_out in g_outs]


def grad_scale(x, multiplier):
    """
    This op scale or inverse the gradient in the backpropagation.

    Parameters
    ----------
    x:
        The variable we want its gradient inputs scale
    multiplier:
        Scale of the gradient

    Examples
    --------
    >>> x = aesara.tensor.fscalar()
    >>> fx = aesara.tensor.sin(x)
    >>> fp = aesara.grad(fx, wrt=x)
    >>> fprime = aesara.function([x], fp)
    >>> print(fprime(2))  # doctest: +ELLIPSIS
    -0.416...
    >>> f_inverse=grad_scale(fx, -1.)
    >>> fpp = aesara.grad(f_inverse, wrt=x)
    >>> fpprime = aesara.function([x], fpp)
    >>> print(fpprime(2))  # doctest: +ELLIPSIS
    0.416...
    """
    return GradScale(multiplier)(x)


DEPRECATED_NAMES = [
    (
        "consider_constant_",
        "`consider_constant_` is deprecated; use `zero_grad` or `disconnected_grad` instead.",
        ConsiderConstant(),
    ),
]


def __getattr__(name):
    """Intercept module-level attribute access of deprecated symbols.

    Adapted from https://stackoverflow.com/a/55139609/3006474.

    """
    from warnings import warn

    for old_name, msg, old_object in DEPRECATED_NAMES:
        if name == old_name:
            warn(msg, DeprecationWarning, stacklevel=2)
            return old_object

    raise AttributeError(f"module {__name__} has no attribute {name}")
