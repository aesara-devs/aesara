import warnings
warnings.warn("Please replace 'aesara.tensor.sub' with 'aesara.tensor.subtract'.", DeprecationWarning)
import copy
import sys
import warnings
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

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

C = TypeVar("C", bound=Callable)


class ThunkType(Protocol[C]):
    inputs: List[List[Optional[List[Any]]]]
    outputs: List[List[Optional[List[Any]]]]
    lazy: bool
    __call__: C
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

    thunk()

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
                "You can either define itypes and otypes, or implement make_node"
            )

        if self.otypes is None:
            raise NotImplementedError(
                "You can either define itypes and otypes, or implement make_node"
            )

        if len(inputs) != len(self.itypes):
            raise ValueError(
                f"We expected {len(self.itypes)} inputs but got {len(inputs)}."
            )
        if not all(
            expected_type.is_super(var.type)
            for var, expected_type in zip(inputs, self.itypes)
        ):
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

