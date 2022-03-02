import copy
import sys
import warnings
from abc import abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Text,
    Tuple,
    Union,
    cast,
)

from typing_extensions import Protocol

import aesara
from aesara.configdefaults import config
from aesara.graph.basic import Apply, NoParams, Variable
from aesara.graph.utils import (
    MetaObject,
    MethodNotDefined,
    TestValueError,
    add_tag_trace,
    get_variable_trace_string,
)
from aesara.link.c.params_type import Params, ParamsType


if TYPE_CHECKING:
    from aesara.compile.function.types import Function
    from aesara.graph.fg import FunctionGraph
    from aesara.graph.type import Type

StorageCellType = List[Optional[Any]]
StorageMapType = Dict[Variable, StorageCellType]
ComputeMapType = Dict[Variable, List[bool]]
InputStorageType = List[StorageCellType]
OutputStorageType = List[StorageCellType]
ParamsInputType = Optional[Tuple[Any]]
PerformMethodType = Callable[
    [Apply, List[Any], OutputStorageType, ParamsInputType], None
]
BasicThunkType = Callable[[], None]
ThunkCallableType = Callable[
    [PerformMethodType, StorageMapType, ComputeMapType, Apply], None
]


class ThunkType(Protocol):
    inputs: List[List[Optional[List[Any]]]]
    outputs: List[List[Optional[List[Any]]]]
    lazy: bool
    __call__: ThunkCallableType
    perform: PerformMethodType


def is_thunk_type(thunk: ThunkCallableType) -> ThunkType:
    res = cast(ThunkType, thunk)
    return res


def compute_test_value(node: Apply):
    r"""Computes the test value of a node.

    Parameters
    ----------
    node : Apply
        The `Apply` node for which the test value is computed.

    Returns
    -------
    None
        The `tag.test_value`\s are updated in each `Variable` in `node.outputs`.

    """
    # Gather the test values for each input of the node
    storage_map = {}
    compute_map = {}
    for i, ins in enumerate(node.inputs):
        try:
            storage_map[ins] = [ins.get_test_value()]
            compute_map[ins] = [True]
        except TestValueError:
            # no test-value was specified, act accordingly
            if config.compute_test_value == "warn":
                warnings.warn(
                    f"Warning, Cannot compute test value: input {i} ({ins}) of Op {node} missing default value",
                    stacklevel=2,
                )
                return
            elif config.compute_test_value == "raise":
                detailed_err_msg = get_variable_trace_string(ins)

                raise ValueError(
                    f"Cannot compute test value: input {i} ({ins}) of Op {node} missing default value. {detailed_err_msg}"
                )
            elif config.compute_test_value == "ignore":
                return
            elif config.compute_test_value == "pdb":
                import pdb

                pdb.post_mortem(sys.exc_info()[2])
            else:
                raise ValueError(
                    f"{config.compute_test_value} is invalid for option config.compute_test_value"
                )

    # All inputs have test-values; perform the `Op`'s computation

    # The original values should not be destroyed, so we copy the values of the
    # inputs in `destroy_map`
    destroyed_inputs_idx = set()
    if node.op.destroy_map:
        for i_pos_list in node.op.destroy_map.values():
            destroyed_inputs_idx.update(i_pos_list)
    for inp_idx in destroyed_inputs_idx:
        inp = node.inputs[inp_idx]
        storage_map[inp] = [copy.copy(storage_map[inp][0])]

    # Prepare `storage_map` and `compute_map` for the outputs
    for o in node.outputs:
        storage_map[o] = [None]
        compute_map[o] = [False]

    # Create a thunk that performs the computation
    thunk = node.op.make_thunk(node, storage_map, compute_map, no_recycling=[])
    thunk.inputs = [storage_map[v] for v in node.inputs]
    thunk.outputs = [storage_map[v] for v in node.outputs]

    required = thunk()
    assert not required  # We provided all inputs

    for output in node.outputs:
        # Check that the output has been computed
        assert compute_map[output][0], (output, storage_map[output][0])

        # Add 'test_value' to output tag, so that downstream `Op`s can use
        # these numerical values as test values
        output.tag.test_value = storage_map[output][0]


class Op(MetaObject):
    """A class that models and constructs operations in a graph.

    A `Op` instance has several responsibilities:

    * construct `Apply` nodes via :meth:`Op.make_node` method,
    * perform the numeric calculation of the modeled operation via the
      :meth:`Op.perform` method,
    * and (optionally) build the gradient-calculating sub-graphs via the
      :meth:`Op.grad` method.

    To see how `Op`, `Type`, `Variable`, and `Apply` fit together see the
    page on :doc:`graph`.

    For more details regarding how these methods should behave: see the `Op
    Contract` in the sphinx docs (advanced tutorial on `Op` making).

    """

    default_output: Optional[int] = None
    """
    An ``int`` that specifies which output :meth:`Op.__call__` should return.  If
    ``None``, then all outputs are returned.

    A subclass should not change this class variable, but instead override it
    with a subclass variable or an instance variable.

    """

    view_map: Dict[int, List[int]] = {}
    """
    A ``dict`` that maps output indices to the input indices of which they are
    a view.

    Examples
    ========

    .. code-block:: python

        view_map = {0: [1]} # first output is a view of second input
        view_map = {1: [0]} # second output is a view of first input

    """

    destroy_map: Dict[int, List[int]] = {}
    """
    A ``dict`` that maps output indices to the input indices upon which they
    operate in-place.

    Examples
    ========

    .. code-block:: python

        destroy_map = {0: [1]} # first output operates in-place on second input
        destroy_map = {1: [0]} # second output operates in-place on first input

    """

    itypes: Optional[Sequence["Type"]] = None
    otypes: Optional[Sequence["Type"]] = None
    params_type: Optional[ParamsType] = None

    def make_node(self, *inputs: Variable) -> Apply:
        """Construct an `Apply` node that represent the application of this operation to the given inputs.

        This must be implemented by sub-classes.

        Returns
        -------
        node: Apply
            The constructed `Apply` node.

        """
        if self.itypes is None:
            raise NotImplementedError(
                "You can either define itypes and otypes,\
             or implement make_node"
            )

        if self.otypes is None:
            raise NotImplementedError(
                "You can either define itypes and otypes,\
             or implement make_node"
            )

        if len(inputs) != len(self.itypes):
            raise ValueError(
                f"We expected {len(self.itypes)} inputs but got {len(inputs)}."
            )
        if not all(it.in_same_class(inp.type) for inp, it in zip(inputs, self.itypes)):
            raise TypeError(
                f"Invalid input types for Op {self}:\n"
                + "\n".join(
                    f"Input {i}/{len(inputs)}: Expected {inp}, got {out}"
                    for i, (inp, out) in enumerate(
                        zip(self.itypes, (inp.type for inp in inputs)),
                        start=1,
                    )
                    if inp != out
                )
            )
        return Apply(self, inputs, [o() for o in self.otypes])

    def __call__(self, *inputs: Any, **kwargs) -> Union[Variable, List[Variable]]:
        r"""Construct an `Apply` node using :meth:`Op.make_node` and return its outputs.

        This method is just a wrapper around :meth:`Op.make_node`.

        It is called by code such as:

        .. code-block:: python

           x = aesara.tensor.matrix()
           y = aesara.tensor.exp(x)


        `aesara.tensor.exp` is an `Op` instance, so ``aesara.tensor.exp(x)`` calls
        :meth:`aesara.tensor.exp.__call__` (i.e. this method) and returns its single output
        `Variable`, ``y``.  The `Apply` node constructed by :meth:`self.make_node`
        behind the scenes is available via ``y.owner``.

        `Op` authors are able to determine which output is returned by this method
        via the :attr:`Op.default_output` property.

        Parameters
        ----------
        inputs : tuple of Variable
            The `Op`'s inputs.
        kwargs
            Additional keyword arguments to be forwarded to
            :meth:`Op.make_node` *except* for optional argument ``return_list`` (which
            defaults to ``False``). If ``return_list`` is ``True``, then the returned
            value is always a ``list``. Otherwise it is either a single `Variable`
            when the output of :meth:`Op.make_node` contains a single element, or this
            output (unchanged) when it contains multiple elements.

        Returns
        -------
        outputs : list of Variable or Variable
            Either a list of output `Variable`\s, or a single `Variable`.
            This is determined by the number of outputs produced by the
            `Op`, the value of the keyword ``return_list``, and the value of
            the :attr:`Op.default_output` property.

        """
        return_list = kwargs.pop("return_list", False)
        node = self.make_node(*inputs, **kwargs)

        if config.compute_test_value != "off":
            compute_test_value(node)

        if self.default_output is not None:
            rval = node.outputs[self.default_output]
            if return_list:
                return [rval]
            return rval
        else:
            if return_list:
                return list(node.outputs)
            elif len(node.outputs) == 1:
                return node.outputs[0]
            else:
                return node.outputs

    def __ne__(self, other: Any) -> bool:
        return not (self == other)

    # Convenience so that subclass implementers don't have to import utils
    # just to self.add_tag_trace
    add_tag_trace = staticmethod(add_tag_trace)

    def grad(
        self, inputs: List[Variable], output_grads: List[Variable]
    ) -> List[Variable]:
        """Construct a graph for the gradient with respect to each input variable.

        Each returned `Variable` represents the gradient with respect to that
        input computed based on the symbolic gradients with respect to each
        output. If the output is not differentiable with respect to an input,
        then this method should return an instance of type ``NullType`` for that
        input.

        Parameters
        ----------
        inputs : list of Variable
            The input variables.
        output_grads : list of Variable
            The gradients of the output variables.

        Returns
        -------
        grads : list of Variable
            The gradients with respect to each `Variable` in `inputs`.

        """
        raise NotImplementedError()

    def L_op(
        self,
        inputs: List[Variable],
        outputs: List[Variable],
        output_grads: List[Variable],
    ) -> List[Variable]:
        r"""Construct a graph for the L-operator.

        This method is primarily used by `Lop` and dispatches to
        :meth:`Op.grad` by default.

        The L-operator computes a *row* vector times the Jacobian. The
        mathematical relationship is
        :math:`v \frac{\partial f(x)}{\partial x}`.
        The L-operator is also supported for generic tensors (not only for
        vectors).

        Parameters
        ----------
        inputs : list of Variable
        outputs : list of Variable
        output_grads : list of Variable

        """
        return self.grad(inputs, output_grads)

    def R_op(
        self, inputs: List[Variable], eval_points: Union[Variable, List[Variable]]
    ) -> List[Variable]:
        r"""Construct a graph for the R-operator.

        This method is primarily used by `Rop`.

        Suppose the `Op` outputs ``[ f_1(inputs), ..., f_n(inputs) ]``.

        Parameters
        ----------
        inputs
            The `Op` inputs.
        eval_points
            A `Variable` or list of `Variable`\s with the same length as inputs.
            Each element of `eval_points` specifies the value of the corresponding
            input at the point where the R-operator is to be evaluated.

        Returns
        -------
        ``rval[i]`` should be ``Rop(f=f_i(inputs), wrt=inputs, eval_points=eval_points)``.

        """
        raise NotImplementedError()

    @abstractmethod
    def perform(
        self,
        node: Apply,
        inputs: List[Variable],
        output_storage: OutputStorageType,
        params: ParamsInputType = None,
    ) -> None:
        """Calculate the function on the inputs and put the variables in the output storage.

        Parameters
        ----------
        node
            The symbolic `Apply` node that represents this computation.
        inputs
            Immutable sequence of non-symbolic/numeric inputs.  These
            are the values of each `Variable` in :attr:`node.inputs`.
        output_storage
            List of mutable single-element lists (do not change the length of
            these lists).  Each sub-list corresponds to value of each
            `Variable` in :attr:`node.outputs`.  The primary purpose of this method
            is to set the values of these sub-lists.
        params
            A tuple containing the values of each entry in :attr:`Op.__props__`.

        Notes
        -----
        The `output_storage` list might contain data. If an element of
        output_storage is not ``None``, it has to be of the right type, for
        instance, for a `TensorVariable`, it has to be a NumPy ``ndarray``
        with the right number of dimensions and the correct dtype.
        Its shape and stride pattern can be arbitrary. It is not
        guaranteed that such pre-set values were produced by a previous call to
        this :meth:`Op.perform`; they could've been allocated by another
        `Op`'s `perform` method.
        An `Op` is free to reuse `output_storage` as it sees fit, or to
        discard it and allocate new memory.

        """

    def do_constant_folding(self, fgraph: "FunctionGraph", node: Apply) -> bool:
        """Determine whether or not constant folding should be performed for the given node.

        This allows each `Op` to determine if it wants to be constant
        folded when all its inputs are constant. This allows it to choose where
        it puts its memory/speed trade-off. Also, it could make things faster
        as constants can't be used for in-place operations (see
        ``*IncSubtensor``).

        Parameters
        ----------
        node : Apply
            The node for which the constant folding determination is made.

        Returns
        -------
        res : bool

        """
        return True

    def get_params(self, node: Apply) -> Params:
        """Try to get parameters for the `Op` when :attr:`Op.params_type` is set to a `ParamsType`."""
        if isinstance(self.params_type, ParamsType):
            wrapper = self.params_type
            if not all(hasattr(self, field) for field in wrapper.fields):
                # Let's print missing attributes for debugging.
                not_found = tuple(
                    field for field in wrapper.fields if not hasattr(self, field)
                )
                raise AttributeError(
                    f"{type(self).__name__}: missing attributes {not_found} for ParamsType."
                )
            # ParamsType.get_params() will apply filtering to attributes.
            return self.params_type.get_params(self)
        raise MethodNotDefined("get_params")

    def prepare_node(
        self,
        node: Apply,
        storage_map: Optional[StorageMapType],
        compute_map: Optional[ComputeMapType],
        impl: Optional[Text],
    ) -> None:
        """Make any special modifications that the `Op` needs before doing :meth:`Op.make_thunk`.

        This can modify the node inplace and should return nothing.

        It can be called multiple time with different `impl` values.

        .. warning::

            It is the `Op`'s responsibility to not re-prepare the node when it
            isn't good to do so.

        """

    def make_py_thunk(
        self,
        node: Apply,
        storage_map: StorageMapType,
        compute_map: ComputeMapType,
        no_recycling: bool,
        debug: bool = False,
    ) -> ThunkType:
        """Make a Python thunk.

        Like :meth:`Op.make_thunk` but only makes Python thunks.

        """
        node_input_storage = [storage_map[r] for r in node.inputs]
        node_output_storage = [storage_map[r] for r in node.outputs]

        if debug:
            p = node.op.debug_perform
        else:
            p = node.op.perform

        params = node.run_params()

        if params is NoParams:
            # default arguments are stored in the closure of `rval`
            @is_thunk_type
            def rval(
                p=p, i=node_input_storage, o=node_output_storage, n=node, params=None
            ):
                r = p(n, [x[0] for x in i], o)
                for o in node.outputs:
                    compute_map[o][0] = True
                return r

        else:
            params_val = node.params_type.filter(params)

            @is_thunk_type
            def rval(
                p=p,
                i=node_input_storage,
                o=node_output_storage,
                n=node,
                params=params_val,
            ):
                r = p(n, [x[0] for x in i], o, params)
                for o in node.outputs:
                    compute_map[o][0] = True
                return r

        rval.inputs = node_input_storage
        rval.outputs = node_output_storage
        setattr(rval, "perform", p)
        rval.lazy = False
        return rval

    def make_thunk(
        self,
        node: Apply,
        storage_map: StorageMapType,
        compute_map: ComputeMapType,
        no_recycling: bool,
        impl: Optional[Text] = None,
    ) -> ThunkType:
        r"""Create a thunk.

        This function must return a thunk, that is a zero-arguments
        function that encapsulates the computation to be performed
        by this op on the arguments of the node.

        Parameters
        ----------
        node
            Something previously returned by :meth:`Op.make_node`.
        storage_map
            A ``dict`` mapping `Variable`\s to single-element lists where a
            computed value for each `Variable` may be found.
        compute_map
            A ``dict`` mapping `Variable`\s to single-element lists where a
            boolean value can be found. The boolean indicates whether the
            `Variable`'s `storage_map` container contains a valid value
            (i.e. ``True``) or whether it has not been computed yet
            (i.e. ``False``).
        no_recycling
            List of `Variable`\s for which it is forbidden to reuse memory
            allocated by a previous call.
        impl : str
            Description for the type of node created (e.g. ``"c"``, ``"py"``,
            etc.)

        Notes
        -----
        If the thunk consults the `storage_map` on every call, it is safe
        for it to ignore the `no_recycling` argument, because elements of the
        `no_recycling` list will have a value of ``None`` in the `storage_map`.
        If the thunk can potentially cache return values (like `CLinker` does),
        then it must not do so for variables in the `no_recycling` list.

        :meth:`Op.prepare_node` is always called. If it tries ``'c'`` and it
        fails, then it tries ``'py'``, and :meth:`Op.prepare_node` will be
        called twice.
        """
        self.prepare_node(
            node, storage_map=storage_map, compute_map=compute_map, impl="py"
        )
        return self.make_py_thunk(node, storage_map, compute_map, no_recycling)

    def __str__(self):
        return getattr(type(self), "__name__", super().__str__())


class _NoPythonOp(Op):
    """A class used to indicate that an `Op` does not provide a Python implementation.

    XXX: Do not use this class; it's only for tracking bad implementations internally.

    """

    def perform(self, node, inputs, output_storage, params=None):
        raise NotImplementedError("No Python implementation is provided by this Op.")


class HasInnerGraph:
    r"""A mixin for an `Op` that contain an inner graph."""

    @property
    @abstractmethod
    def fn(self) -> "Function":
        """The inner function."""

    @property
    @abstractmethod
    def inner_inputs(self) -> List[Variable]:
        """The inner function's inputs."""

    @property
    @abstractmethod
    def inner_outputs(self) -> List[Variable]:
        """The inner function's outputs."""


def get_test_value(v: Any) -> Any:
    """Get the test value for `v`.

    If input `v` is not already a variable, it is turned into one by calling
    ``as_tensor_variable(v)``.

    Raises
    ------
    ``AttributeError`` if no test value is set.

    """
    if not isinstance(v, Variable):
        v = aesara.tensor.as_tensor_variable(v)

    return v.get_test_value()


def missing_test_message(msg: Text) -> None:
    """Display a message saying that some test_value is missing.

    This uses the appropriate form based on ``config.compute_test_value``:

        off:
            The interactive debugger is off, so we do nothing.

        ignore:
            The interactive debugger is set to ignore missing inputs, so do
            nothing.

        warn:
            Display `msg` as a warning.


    Raises
    ------
    AttributeError
        With msg as the exception text.

    """
    action = config.compute_test_value
    if action == "raise":
        raise TestValueError(msg)
    elif action == "warn":
        warnings.warn(msg, stacklevel=2)
    else:
        assert action in ("ignore", "off")


def get_test_values(*args: Variable) -> Union[Any, List[Any]]:
    r"""Get test values for multiple `Variable`\s.

    Intended use:

    .. code-block:: python

        for val_1, ..., val_n in get_debug_values(var_1, ..., var_n):
            if some condition on val_1, ..., val_n is not met:
                missing_test_message("condition was not met")


    Given a list of variables, `get_debug_values` does one of three things:

    1. If the interactive debugger is off, returns an empty list
    2. If the interactive debugger is on, and all variables have
       debug values, returns a list containing a single element.
       This single element is either:

           a) if there is only one variable, the element is its
               value
           b) otherwise, a tuple containing debug values of all
               the variables.

    3. If the interactive debugger is on, and some variable does
       not have a debug value, issue a `missing_test_message` about
       the variable, and, if still in control of execution, return
       an empty list.

    """

    if config.compute_test_value == "off":
        return []

    rval = []

    for i, arg in enumerate(args):
        try:
            rval.append(get_test_value(arg))
        except TestValueError:
            if hasattr(arg, "name") and arg.name is not None:
                missing_test_message(f"Argument {i} ('{arg.name}') has no test value")
            else:
                missing_test_message(f"Argument {i} has no test value")
            return []

    if len(rval) == 1:
        return rval

    return [tuple(rval)]
