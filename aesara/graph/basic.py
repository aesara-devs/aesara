"""Core graph classes."""
import warnings
from collections import deque
from copy import copy
from itertools import count
from typing import (
    TYPE_CHECKING,
    Callable,
    Collection,
    Deque,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Reversible,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np

from aesara.configdefaults import config
from aesara.graph.utils import (
    MetaObject,
    MethodNotDefined,
    Scratchpad,
    TestValueError,
    ValidatingScratchpad,
    add_tag_trace,
    get_variable_trace_string,
)
from aesara.misc.ordered_set import OrderedSet


if TYPE_CHECKING:
    from aesara.graph.type import Type


T = TypeVar("T", bound="Node")
NoParams = object()
NodeAndChildren = Tuple[T, Optional[Iterable[T]]]


class Node(MetaObject):
    r"""A `Node` in an Aesara graph.

    Currently, graphs contain two kinds of `Nodes`: `Variable`\s and `Apply`\s.
    Edges in the graph are not explicitly represented.  Instead each `Node`
    keeps track of its parents via `Variable.owner` / `Apply.inputs`.

    """
    type: "Type"
    name: Optional[str]

    def get_parents(self):
        """
        Return a list of the parents of this node.
        Should return a copy--i.e., modifying the return
        value should not modify the graph structure.

        """
        raise NotImplementedError()


class Apply(Node):
    """A `Node` representing the application of an operation to inputs.

    Basically, an `Apply` instance is an object that represents the
    Python statement ``outputs = op(*inputs)``.

    This class is typically instantiated by a `Op.make_node` method, which
    is called by `Op.__call__`.

    The function `aesara.compile.function.function` uses `Apply.inputs`
    together with `Variable.owner` to search the expression graph and determine
    which inputs are necessary to compute the function's outputs.

    A `Linker` uses the `Apply` instance's `op` field to compute numeric values
    for the output variables.

    Notes
    -----
    The `Variable.owner` field of each `Apply.outputs` element is set to `self`
    in `Apply.make_node`.

    If an output element has an owner that is neither `None` nor `self`, then a
    `ValueError` exception will be raised.

    Attributes
    ----------
    op : Op
        The operation that produces `outputs` given `inputs`.
    inputs : List[Variable]
        The arguments of the expression modeled by the `Apply` node.
    outputs : List[Variable]
        The outputs of the expression modeled by the `Apply` node.

    """

    def __init__(self, op, inputs, outputs):
        """
        Parameters
        ----------
        op : Op
        inputs : List[Variable]
        outputs : List[Variable]

        """
        self.op = op
        self.inputs: List[Variable] = []
        self.tag = Scratchpad()

        if not isinstance(inputs, (list, tuple)):
            raise TypeError("The inputs of an Apply must be a list or tuple")

        if not isinstance(outputs, (list, tuple)):
            raise TypeError("The output of an Apply must be a list or tuple")

        # filter inputs to make sure each element is a Variable
        for input in inputs:
            if isinstance(input, Variable):
                self.inputs.append(input)
            else:
                raise TypeError(
                    f"The 'inputs' argument to Apply must contain Variable instances, not {input}"
                )
        self.outputs: List[Variable] = []
        # filter outputs to make sure each element is a Variable
        for i, output in enumerate(outputs):
            if isinstance(output, Variable):
                if output.owner is None:
                    output.owner = self
                    output.index = i
                elif output.owner is not self or output.index != i:
                    raise ValueError(
                        "All output variables passed to Apply must belong to it."
                    )
                self.outputs.append(output)
            else:
                raise TypeError(
                    f"The 'outputs' argument to Apply must contain Variable instances with no owner, not {output}"
                )

    def run_params(self):
        """
        Returns the params for the node, or NoParams if no params is set.

        """
        try:
            return self.op.get_params(self)
        except MethodNotDefined:
            return NoParams

    def __getstate__(self):
        d = self.__dict__
        # ufunc don't pickle/unpickle well
        if hasattr(self.tag, "ufunc"):
            d = copy(self.__dict__)
            t = d["tag"]
            del t.ufunc
            d["tag"] = t
        return d

    def default_output(self):
        """
        Returns the default output for this node.

        Returns
        -------
        Variable instance
            An element of self.outputs, typically self.outputs[0].

        Notes
        -----
        May raise AttributeError self.op.default_output is out of range, or if
        there are multiple outputs and self.op.default_output does not exist.

        """
        do = getattr(self.op, "default_output", None)
        if do is None:
            if len(self.outputs) == 1:
                return self.outputs[0]
            else:
                raise ValueError(f"{self.op}.default_output should be an output index.")
        elif not isinstance(do, int):
            raise ValueError(f"{self.op}.default_output should be an int or long")
        elif do < 0 or do >= len(self.outputs):
            raise ValueError(f"{self.op}.default_output is out of range.")
        return self.outputs[do]

    def __str__(self):
        return op_as_string(self.inputs, self)

    def __repr__(self):
        return str(self)

    def clone(self):
        """
        Duplicate this Apply instance with inputs = self.inputs.

        Returns
        -------
        object
            A new Apply instance (or subclass instance) with new outputs.

        Notes
        -----
        Tags are copied from self to the returned instance.

        """
        cp = self.__class__(
            self.op, self.inputs, [output.clone() for output in self.outputs]
        )
        cp.tag = copy(self.tag)
        return cp

    def clone_with_new_inputs(self, inputs, strict=True):
        """Duplicate this `Apply` instance in a new graph.

        Parameters
        ----------
        inputs : list of Variables
            List of `Variable` instances to use as inputs.
        strict : bool
            If ``True``, the type fields of all the inputs must be equal
            to the current ones (or compatible, for instance `Tensor` /
            `GpuArray` of the same dtype and broadcastable patterns,
            in which case they will be converted into current `Type`), and
            returned outputs are guaranteed to have the same types as
            ``self.outputs``.  If ``False``, then there's no guarantee that the
            clone's outputs will have the same types as ``self.outputs``,
            and cloning may not even be possible (it depends on the `Op`).

        Returns
        -------
        object
            An `Apply` instance with the same `Op` but different outputs.

        """
        assert isinstance(inputs, (list, tuple))
        remake_node = False
        new_inputs = inputs[:]
        for i, (curr, new) in enumerate(zip(self.inputs, new_inputs)):
            if curr.type != new.type:
                if strict:
                    # If compatible, casts new into curr.type
                    new_inputs[i] = curr.type.filter_variable(new)
                else:
                    remake_node = True
        if remake_node:
            new_node = self.op.make_node(*new_inputs)
            new_node.tag = copy(self.tag).__update__(new_node.tag)
        else:
            new_node = self.clone()
            new_node.inputs = new_inputs
        return new_node

    def get_parents(self):
        return list(self.inputs)

    @property
    def out(self):
        """An alias for `self.default_output`"""
        return self.default_output()

    @property
    def nin(self):
        """The number of inputs."""
        return len(self.inputs)

    @property
    def nout(self):
        """The number of outputs."""
        return len(self.outputs)

    @property
    def params_type(self):
        return self.op.params_type


class Variable(Node):
    r"""
    A :term:`Variable` is a node in an expression graph that represents a
    variable.

    The inputs and outputs of every `Apply` are `Variable`
    instances. The input and output arguments to create a `function` are also
    `Variable` instances. A `Variable` is like a strongly-typed variable in
    some other languages; each `Variable` contains a reference to a `Type`
    instance that defines the kind of value the `Variable` can take in a
    computation.

    A `Variable` is a container for four important attributes:

    - :literal:`type` a `Type` instance defining the kind of value this
      `Variable` can have,

    - :literal:`owner` either ``None`` (for graph roots) or the `Apply` instance
      of which ``self`` is an output,

    - :literal:`index` the integer such that ``owner.outputs[index] is this_variable``
      (ignored if ``owner`` is ``None``),

    - :literal:`name` a string to use in pretty-printing and debugging.

    There are a few kinds of `Variable`\s to be aware of: A `Variable` which is the
    output of a symbolic computation has a reference to the `Apply` instance to
    which it belongs (property: owner) and the position of itself in the owner's
    output list (property: index).

    - `Variable` (this base type) is typically the output of a symbolic
      computation.

    - `Constant`: a subclass which adds a default and un-replaceable
      :literal:`value`, and requires that owner is None.

    - `TensorVariable` subclass of `Variable` that represents a ``numpy.ndarray``
       object.

    - `TensorSharedVariable`: a shared version of `TensorVariable`.

    - `SparseVariable`: a subclass of `Variable` that represents
      a ``scipy.sparse.{csc,csr}_matrix`` object.

    - `GpuArrayVariable`: a subclass of `Variable` that represents our object on
      the GPU that is a subset of ``numpy.ndarray``.

    - `RandomVariable`.

    A `Variable` which is the output of a symbolic computation will have an owner
    not equal to None.

    Using a `Variable`\s' owner field and an `Apply` node's inputs fields,
    one can navigate a graph from an output all the way to the inputs. The
    opposite direction is possible with a ``FunctionGraph`` and its
    ``FunctionGraph.clients`` ``dict``, which maps `Variable`\s to a list of their
    clients.

    Parameters
    ----------
    type : a Type instance
        The type governs the kind of data that can be associated with this
        variable.
    owner : None or Apply instance
        The `Apply` instance which computes the value for this variable.
    index : None or int
        The position of this `Variable` in owner.outputs.
    name : None or str
        A string for pretty-printing and debugging.

    Examples
    --------

    .. code-block:: python

        import aesara
        import aesara.tensor as at

        a = at.constant(1.5)            # declare a symbolic constant
        b = at.fscalar()                # declare a symbolic floating-point scalar

        c = a + b                       # create a simple expression

        f = aesara.function([b], [c])   # this works because a has a value associated with it already

        assert 4.0 == f(2.5)            # bind 2.5 to an internal copy of b and evaluate an internal c

        aesara.function([a], [c])       # compilation error because b (required by c) is undefined

        aesara.function([a,b], [c])     # compilation error because a is constant, it can't be an input


    The python variables ``a, b, c`` all refer to instances of type
    `Variable`. The `Variable` referred to by ``a`` is also an instance of
    `Constant`.

    """

    # __slots__ = ['type', 'owner', 'index', 'name']
    __count__ = count(0)

    _owner: Optional[Apply]

    @property
    def owner(self) -> Optional[Apply]:
        return self._owner

    @owner.setter
    def owner(self, value) -> None:
        self._owner = value

    def __init__(
        self,
        type,
        owner: Optional[Apply] = None,
        index: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super().__init__()

        self.tag = ValidatingScratchpad("test_value", type.filter)

        self.type = type

        if owner is not None and not isinstance(owner, Apply):
            raise TypeError("owner must be an Apply instance", owner)
        self._owner = owner

        if index is not None and not isinstance(index, int):
            raise TypeError("index must be an int", index)
        self.index = index

        if name is not None and not isinstance(name, str):
            raise TypeError("name must be a string", name)
        self.name = name

        self.auto_name = "auto_" + str(next(self.__count__))

    def get_test_value(self):
        """Get the test value.

        Raises
        ------
        TestValueError

        """
        if not hasattr(self.tag, "test_value"):
            detailed_err_msg = get_variable_trace_string(self)
            raise TestValueError(f"{self} has no test value {detailed_err_msg}")

        return self.tag.test_value

    def __str__(self):
        """Return a ``str`` representation of the `Variable`."""
        if self.name is not None:
            return self.name
        if self.owner is not None:
            op = self.owner.op
            if self.index == op.default_output:
                return str(self.owner.op) + ".out"
            else:
                return str(self.owner.op) + "." + str(self.index)
        else:
            return f"<{self.type}>"

    def __repr_test_value__(self):
        """Return a ``repr`` of the test value.

        Return a printable representation of the test value. It can be
        overridden by classes with non printable test_value to provide a
        suitable representation of the test_value.
        """
        return repr(self.get_test_value())

    def __repr__(self, firstPass=True):
        """Return a ``repr`` of the `Variable`.

        Return a printable name or description of the Variable. If
        ``config.print_test_value`` is ``True`` it will also print the test
        value, if any.
        """
        to_print = [str(self)]
        if config.print_test_value and firstPass:
            try:
                to_print.append(self.__repr_test_value__())
            except TestValueError:
                pass
        return "\n".join(to_print)

    def clone(self):
        """Return a new `Variable` like `self`.

        Returns
        -------
        Variable instance
            A new `Variable` instance (or subclass instance) with no owner or
            index.

        Notes
        -----
        Tags are copied to the returned instance.

        Name is copied to the returned instance.

        """
        # return copy(self)
        cp = self.__class__(self.type, None, None, self.name)
        cp.tag = copy(self.tag)
        return cp

    def __lt__(self, other):
        raise NotImplementedError(
            "Subclasses of Variable must provide __lt__", self.__class__.__name__
        )

    def __le__(self, other):
        raise NotImplementedError(
            "Subclasses of Variable must provide __le__", self.__class__.__name__
        )

    def __gt__(self, other):
        raise NotImplementedError(
            "Subclasses of Variable must provide __gt__", self.__class__.__name__
        )

    def __ge__(self, other):
        raise NotImplementedError(
            "Subclasses of Variable must provide __ge__", self.__class__.__name__
        )

    def get_parents(self):
        if self.owner is not None:
            return [self.owner]
        return []

    def eval(self, inputs_to_values=None):
        r"""Evaluate the `Variable`.

        Parameters
        ----------
        inputs_to_values :
            A dictionary mapping Aesara `Variable`\s to values.

        Examples
        --------

        >>> import numpy as np
        >>> import aesara.tensor as at
        >>> x = at.dscalar('x')
        >>> y = at.dscalar('y')
        >>> z = x + y
        >>> np.allclose(z.eval({x : 16.3, y : 12.1}), 28.4)
        True

        We passed :meth:`eval` a dictionary mapping symbolic Aesara
        `Variable`\s to the values to substitute for them, and it returned
        the numerical value of the expression.

        Notes
        -----

        :meth:`eval` will be slow the first time you call it on a variable --
        it needs to call :func:`function` to compile the expression behind
        the scenes. Subsequent calls to :meth:`eval` on that same variable
        will be fast, because the variable caches the compiled function.

        This way of computing has more overhead than a normal Aesara
        function, so don't use it too much in real scripts.
        """
        from aesara.compile.function import function

        if inputs_to_values is None:
            inputs_to_values = {}

        if not hasattr(self, "_fn_cache"):
            self._fn_cache = dict()

        inputs = tuple(sorted(inputs_to_values.keys(), key=id))
        if inputs not in self._fn_cache:
            self._fn_cache[inputs] = function(inputs, self)
        args = [inputs_to_values[param] for param in inputs]

        rval = self._fn_cache[inputs](*args)

        return rval

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop("_fn_cache", None)
        if (not config.pickle_test_value) and (hasattr(self.tag, "test_value")):
            if not type(config).pickle_test_value.is_default:
                warnings.warn(
                    "pickle_test_value is not default value (True).\n"
                    f"Test value of variable {d['auto_name']}({d['name']}) will not be dumped."
                )
            t = copy(d["tag"])
            del t.test_value
            d["tag"] = t
        return d


class Constant(Variable):
    """A `Variable` with a fixed `data` field.

    `Constant` nodes make numerous optimizations possible (e.g. constant
    in-lining in C code, constant folding, etc.)

    Notes
    -----
    The data field is filtered by what is provided in the constructor for the
    `Constant`'s type field.

    """

    # __slots__ = ['data']

    def __init__(self, type, data, name=None):
        super().__init__(type, None, None, name)
        self.data = type.filter(data)
        add_tag_trace(self)

    def get_test_value(self):
        return self.data

    def equals(self, other):
        # this does what __eq__ should do, but Variable and Apply should always be hashable by id
        return isinstance(other, Constant) and self.signature() == other.signature()

    def signature(self):
        return (self.type, self.data)

    def merge_signature(self):
        return self.signature()

    def __str__(self):
        if self.name is not None:
            return self.name
        else:
            name = str(self.data)
            if len(name) > 20:
                name = name[:10] + "..." + name[-10:]
            return f"{type(self).__name__}{{{name}}}"

    def clone(self):
        """Return `self`, because there's no reason to clone a constant."""
        return self

    @property
    def owner(self) -> None:
        return None

    @owner.setter
    def owner(self, value) -> None:
        if value is not None:
            raise ValueError("Constant instances cannot have an owner.")

    value = property(lambda self: self.data, doc="read-only data access method")

    # index is not defined, because the `owner` attribute must necessarily be None


def walk(
    nodes: Iterable[T],
    expand: Callable[[T], Optional[Iterable[T]]],
    bfs: bool = True,
    return_children: bool = False,
    hash_fn: Callable[[T], int] = id,
) -> Generator[Union[T, NodeAndChildren], None, None]:
    r"""Walk through a graph, either breadth- or depth-first.

    Parameters
    ----------
    nodes
        The nodes from which to start walking.
    expand
        A callable that is applied to each node in `nodes`, the results of
        which are either new nodes to visit or ``None``.
    bfs
        If ``True``, breath first search is used; otherwise, depth first
        search.
    return_children
        If ``True``, each output node will be accompanied by the output of
        `expand` (i.e. the corresponding child nodes).
    hash_fn
        The function used to produce hashes of the elements in `nodes`.
        The default is ``id``.

    Notes
    -----
    A node will appear at most once in the return value, even if it
    appears multiple times in the `nodes` parameter.

    """

    nodes = deque(nodes)

    rval_set: Set[int] = set()

    nodes_pop: Callable[[], T]
    if bfs:
        nodes_pop = nodes.popleft
    else:
        nodes_pop = nodes.pop

    while nodes:
        node: T = nodes_pop()

        node_hash: int = hash_fn(node)

        if node_hash not in rval_set:

            rval_set.add(node_hash)

            new_nodes: Optional[Iterable[T]] = expand(node)

            if return_children:
                yield node, new_nodes
            else:
                yield node

            if new_nodes:
                nodes.extend(new_nodes)


def ancestors(
    graphs: Iterable[Variable], blockers: Optional[Collection[Variable]] = None
) -> Generator[Variable, None, None]:
    r"""Return the variables that contribute to those in given graphs (inclusive).

    Parameters
    ----------
    graphs : list of `Variable` instances
        Output `Variable` instances from which to search backward through
        owners.
    blockers : list of `Variable` instances
        A collection of `Variable`\s that, when found, prevent the graph search
        from preceding from that point.

    Yields
    ------
    `Variable`\s
        All input nodes, in the order found by a left-recursive depth-first
        search started at the nodes in `graphs`.

    """

    def expand(r: Variable) -> Optional[Iterator[Variable]]:
        if r.owner and (not blockers or r not in blockers):
            return reversed(r.owner.inputs)

    yield from cast(Generator[Variable, None, None], walk(graphs, expand, False))


def graph_inputs(
    graphs: Iterable[Variable], blockers: Optional[Collection[Variable]] = None
) -> Generator[Variable, None, None]:
    r"""Return the inputs required to compute the given Variables.

    Parameters
    ----------
    graphs : list of `Variable` instances
        Output `Variable` instances from which to search backward through
        owners.
    blockers : list of `Variable` instances
        A collection of `Variable`\s that, when found, prevent the graph search
        from preceding from that point.

    Yields
    ------
        Input nodes with no owner, in the order found by a left-recursive
        depth-first search started at the nodes in `graphs`.

    """
    yield from (r for r in ancestors(graphs, blockers) if r.owner is None)


def vars_between(
    ins: Collection[Variable], outs: Iterable[Variable]
) -> Generator[Variable, None, None]:
    r"""Extract the `Variable`\s within the sub-graph between input and output nodes.

    Parameters
    ----------
    ins
        Input `Variable`\s.
    outs
        Output `Variable`\s.

    Yields
    ------
    The `Variable`\s that are involved in the subgraph that lies
    between `ins` and `outs`. This includes `ins`, `outs`,
    ``orphans_between(ins, outs)`` and all values of all intermediary steps from
    `ins` to `outs`.

    """

    def expand(r: Variable) -> Optional[Iterable[Variable]]:
        if r.owner and r not in ins:
            return reversed(r.owner.inputs + r.owner.outputs)

    yield from cast(Generator[Variable, None, None], walk(outs, expand))


def orphans_between(
    ins: Collection[Variable], outs: Iterable[Variable]
) -> Generator[Variable, None, None]:
    r"""Extract the `Variable`\s not within the sub-graph between input and output nodes.

    Parameters
    ----------
    ins : list
        Input `Variable`\s.
    outs : list
        Output `Variable`\s.

    Yields
    -------
    Variable
        The `Variable`\s upon which one or more `Variable`\s in `outs`
        depend, but are neither in `ins` nor in the sub-graph that lies between
        them.

    Examples
    --------
    >>> orphans_between([x], [(x+y).out])
    [y]

    """
    yield from (r for r in vars_between(ins, outs) if r.owner is None and r not in ins)


def applys_between(
    ins: Collection[Variable], outs: Iterable[Variable]
) -> Generator[Apply, None, None]:
    r"""Extract the `Apply`\s contained within the sub-graph between given input and output variables.

    Parameters
    ----------
    ins : list
        Input `Variable`\s.
    outs : list
        Output `Variable`\s.

    Yields
    ------
    The `Apply`\s that are contained within the sub-graph that lies
    between `ins` and `outs`, including the owners of the `Variable`\s in
    `outs` and intermediary `Apply`\s between `ins` and `outs`, but not the
    owners of the `Variable`\s in `ins`.

    """
    yield from (
        r.owner for r in vars_between(ins, outs) if r not in ins and r.owner is not None
    )


def clone(
    inputs: List[Variable],
    outputs: List[Variable],
    copy_inputs: bool = True,
    copy_orphans: Optional[bool] = None,
) -> Tuple[Collection[Variable], Collection[Variable]]:
    r"""Copies the sub-graph contained between inputs and outputs.

    Parameters
    ----------
    inputs
        Input `Variable`\s.
    outputs
        Output `Variable`\s.
    copy_inputs
        If ``True``, the inputs will be copied (defaults to ``True``).
    copy_orphans
        When ``None``, use the `copy_inputs` value.
        When ``True``, new orphans nodes are created.
        When ``False``, original orphans nodes are reused in the new graph.

    Returns
    -------
    The inputs and outputs of that copy.

    Notes
    -----

    A constant, if in the `inputs` list is not an orphan. So it will be copied
    conditional on the `copy_inputs` parameter; otherwise, it will be copied
    conditional on the `copy_orphans` parameter.

    """
    if copy_orphans is None:
        copy_orphans = copy_inputs
    equiv = clone_get_equiv(inputs, outputs, copy_inputs, copy_orphans)
    return [cast(Variable, equiv[input]) for input in inputs], [
        cast(Variable, equiv[output]) for output in outputs
    ]


def clone_get_equiv(
    inputs: Sequence[Variable],
    outputs: Sequence[Variable],
    copy_inputs: bool = True,
    copy_orphans: bool = True,
    memo: Optional[Dict[Node, Node]] = None,
) -> Dict[Node, Node]:
    """
    Return a dictionary that maps from `Variable` and `Apply` nodes in the
    original graph to a new node (a clone) in a new graph.

    This function works by recursively cloning inputs... rebuilding a directed
    graph from the inputs up to eventually building new outputs.

    Parameters
    ----------
    inputs : a list of Variables
    outputs : a list of Variables
    copy_inputs : bool
        True means to create the cloned graph from new input
        nodes (the bottom of a feed-upward graph).
        False means to clone a graph that is rooted at the original input
        nodes.
    copy_orphans :
        When ``True``, new constant nodes are created. When ``False``, original
        constant nodes are reused in the new graph.
    memo : None or dict
        Optionally start with a partly-filled dictionary for the return value.
        If a dictionary is passed, this function will work in-place on that
        dictionary and return it.

    """
    if memo is None:
        memo = {}

    # clone the inputs if necessary
    for input in inputs:
        if copy_inputs:
            cpy = input.clone()
            cpy.owner = None
            cpy.index = None
            memo.setdefault(input, cpy)
        else:
            memo.setdefault(input, input)

    # go through the inputs -> outputs graph cloning as we go
    for apply in io_toposort(inputs, outputs):
        for input in apply.inputs:
            if input not in memo:
                if copy_orphans:
                    cpy = input.clone()
                    memo[input] = cpy
                else:
                    memo[input] = input

        new_apply = apply.clone_with_new_inputs([memo[i] for i in apply.inputs])
        memo.setdefault(apply, new_apply)
        for output, new_output in zip(apply.outputs, new_apply.outputs):
            memo.setdefault(output, new_output)

    # finish up by cloning any remaining outputs (it can happen)
    for output in outputs:
        if output not in memo:
            memo[output] = output.clone()

    return memo


def clone_replace(
    output: List[Variable],
    replace: Optional[
        Union[Iterable[Tuple[Variable, Variable]], Dict[Variable, Variable]]
    ] = None,
    strict: bool = True,
    share_inputs: bool = True,
) -> List[Variable]:
    """Clone a graph and replace subgraphs within it.

    It returns a copy of the initial subgraph with the corresponding
    substitutions.

    Parameters
    ----------
    output : Aesara Variables (or Aesara expressions)
        Aesara expression that represents the computational graph.
    replace : dict
        Dictionary describing which subgraphs should be replaced by what.
    share_inputs : bool
        If ``True``, use the same inputs (and shared variables) as the original
        graph. If ``False``, clone them. Note that cloned shared variables still
        use the same underlying storage, so they will always have the same
        value.

    """
    from aesara.compile.function.pfunc import rebuild_collect_shared

    items: Union[List[Tuple[Variable, Variable]], Tuple[Tuple[Variable, Variable], ...]]
    if isinstance(replace, dict):
        items = list(replace.items())
    elif isinstance(replace, (list, tuple)):
        items = replace
    elif replace is None:
        items = []
    else:
        raise ValueError(
            (
                "replace is neither a dictionary, list, "
                f"tuple or None ! The value provided is {replace},"
                f"of type {type(replace)}"
            )
        )
    tmp_replace = [(x, x.type()) for x, y in items]
    new_replace = [(x, y) for ((_, x), (_, y)) in zip(tmp_replace, items)]
    _, _outs, _ = rebuild_collect_shared(
        output, [], tmp_replace, [], strict, share_inputs
    )

    # TODO Explain why we call it twice ?!
    _, outs, _ = rebuild_collect_shared(
        _outs, [], new_replace, [], strict, share_inputs
    )

    return cast(List[Variable], outs)


def general_toposort(
    outputs: Iterable[T],
    deps: Callable[[T], Union[OrderedSet, List[T]]],
    compute_deps_cache: Optional[
        Callable[[T], Optional[Union[OrderedSet, List[T]]]]
    ] = None,
    deps_cache: Optional[Dict[T, List[T]]] = None,
    clients: Optional[Dict[T, List[T]]] = None,
) -> List[T]:
    """Perform a topological sort of all nodes starting from a given node.

    Parameters
    ----------
    deps : callable
        A Python function that takes a node as input and returns its dependence.
    compute_deps_cache : optional
        If provided, `deps_cache` should also be provided. This is a function like
        `deps`, but that also caches its results in a ``dict`` passed as `deps_cache`.
    deps_cache : dict
        A ``dict`` mapping nodes to their children.  This is populated by
        `compute_deps_cache`.
    clients : dict
        If a ``dict`` is passed, it will be filled with a mapping of
        nodes-to-clients for each node in the subgraph.

    Notes
    -----

    ``deps(i)`` should behave like a pure function (no funny business with
    internal state).

    ``deps(i)`` will be cached by this function (to be fast).

    The order of the return value list is determined by the order of nodes
    returned by the `deps` function.

    The second option removes a Python function call, and allows for more
    specialized code, so it can be faster.

    """
    if compute_deps_cache is None:

        if deps_cache is None:
            deps_cache = {}

        def _compute_deps_cache(io):
            if io not in deps_cache:
                d = deps(io)

                if d:
                    if not isinstance(d, (list, OrderedSet)):
                        raise TypeError(
                            "Non-deterministic collections found; make"
                            " toposort non-deterministic."
                        )
                    deps_cache[io] = list(d)
                else:
                    deps_cache[io] = None

                return d
            else:
                return deps_cache[io]

    else:
        _compute_deps_cache = compute_deps_cache

    if deps_cache is None:
        raise ValueError("deps_cache cannot be None")

    search_res: List[NodeAndChildren] = cast(
        List[NodeAndChildren],
        list(walk(outputs, _compute_deps_cache, bfs=False, return_children=True)),
    )

    _clients: Dict[T, List[T]] = {}
    sources: Deque[T] = deque()
    search_res_len: int = 0
    for snode, children in search_res:
        search_res_len += 1
        if children:
            for child in children:
                _clients.setdefault(child, []).append(snode)
        if not deps_cache.get(snode):
            sources.append(snode)

    if clients is not None:
        clients.update(_clients)

    rset: Set[T] = set()
    rlist: List[T] = []
    while sources:
        node: T = sources.popleft()
        if node not in rset:
            rlist.append(node)
            rset.add(node)
            for client in _clients.get(node, []):
                d = [a for a in deps_cache[client] if a is not node]
                deps_cache[client] = d
                if not d:
                    sources.append(client)

    if len(rlist) != search_res_len:
        raise ValueError("graph contains cycles")

    return rlist


def io_toposort(
    inputs: Iterable[Variable],
    outputs: Reversible[Variable],
    orderings: Optional[Dict[Apply, List[Apply]]] = None,
    clients: Optional[Dict[Variable, List[Variable]]] = None,
) -> List[Apply]:
    """Perform topological sort from input and output nodes.

    Parameters
    ----------
    inputs : list or tuple of Variable instances
        Graph inputs.
    outputs : list or tuple of Apply instances
        Graph outputs.
    orderings : dict
        Keys are `Apply` instances, values are lists of `Apply` instances.
    clients : dict
        If provided, it will be filled with mappings of nodes-to-clients for
        each node in the subgraph that is sorted.

    """
    if not orderings and clients is None:  # ordering can be None or empty dict
        # Specialized function that is faster when more then ~10 nodes
        # when no ordering.

        # Do a new stack implementation with the vm algo.
        # This will change the order returned.
        computed = set(inputs)
        todo = [o.owner for o in reversed(outputs) if o.owner]
        order = []
        while todo:
            cur = todo.pop()
            # We suppose that all outputs are always computed
            if cur.outputs[0] in computed:
                continue
            if all(i in computed or i.owner is None for i in cur.inputs):
                computed.update(cur.outputs)
                order.append(cur)
            else:
                todo.append(cur)
                todo.extend(i.owner for i in cur.inputs if i.owner)
        return order

    compute_deps = None
    compute_deps_cache = None
    iset = set(inputs)
    deps_cache: Dict = {}

    if not orderings:  # ordering can be None or empty dict
        # Specialized function that is faster when no ordering.
        # Also include the cache in the function itself for speed up.

        def compute_deps_cache(obj):
            if obj in deps_cache:
                return deps_cache[obj]
            rval = []
            if obj not in iset:
                if isinstance(obj, Variable):
                    if obj.owner:
                        rval = [obj.owner]
                elif isinstance(obj, Apply):
                    rval = list(obj.inputs)
                if rval:
                    deps_cache[obj] = list(rval)
                else:
                    deps_cache[obj] = rval
            else:
                deps_cache[obj] = rval
            return rval

    else:

        # the inputs are used only here in the function that decides what
        # 'predecessors' to explore
        def compute_deps(obj):
            rval = []
            if obj not in iset:
                if isinstance(obj, Variable):
                    if obj.owner:
                        rval = [obj.owner]
                elif isinstance(obj, Apply):
                    rval = list(obj.inputs)
                rval.extend(orderings.get(obj, []))
            else:
                assert not orderings.get(obj, None)
            return rval

    topo = general_toposort(
        outputs,
        deps=compute_deps,
        compute_deps_cache=compute_deps_cache,
        deps_cache=deps_cache,
        clients=clients,
    )
    return [o for o in topo if isinstance(o, Apply)]


default_leaf_formatter = str


def default_node_formatter(op, argstrings):
    return f"{op.op}({', '.join(argstrings)})"


def io_connection_pattern(inputs, outputs):
    """Return the connection pattern of a subgraph defined by given inputs and outputs."""
    inner_nodes = io_toposort(inputs, outputs)

    # Initialize 'connect_pattern_by_var' by establishing each input as
    # connected only to itself
    connect_pattern_by_var = {}
    nb_inputs = len(inputs)

    for i in range(nb_inputs):
        input = inputs[i]
        inp_connection_pattern = [i == j for j in range(nb_inputs)]
        connect_pattern_by_var[input] = inp_connection_pattern

    # Iterate through the nodes used to produce the outputs from the
    # inputs and, for every node, infer their connection pattern to
    # every input from the connection patterns of their parents.
    for n in inner_nodes:

        # Get the connection pattern of the inner node's op. If the op
        # does not define a connection_pattern method, assume that
        # every node output is connected to every node input
        try:
            op_connection_pattern = n.op.connection_pattern(n)
        except AttributeError:
            op_connection_pattern = [[True] * len(n.outputs)] * len(n.inputs)

        # For every output of the inner node, figure out which inputs it
        # is connected to by combining the connection pattern of the inner
        # node and the connection patterns of the inner node's inputs.
        for out_idx in range(len(n.outputs)):
            out = n.outputs[out_idx]
            out_connection_pattern = [False] * nb_inputs

            for inp_idx in range(len(n.inputs)):
                inp = n.inputs[inp_idx]

                if inp in connect_pattern_by_var:
                    inp_connection_pattern = connect_pattern_by_var[inp]

                    # If the node output is connected to the node input, it
                    # means it is connected to every inner input that the
                    # node inputs is connected to
                    if op_connection_pattern[inp_idx][out_idx]:
                        out_connection_pattern = [
                            out_connection_pattern[i] or inp_connection_pattern[i]
                            for i in range(nb_inputs)
                        ]

            # Store the connection pattern of the node output
            connect_pattern_by_var[out] = out_connection_pattern

    # Obtain the global connection pattern by combining the
    # connection patterns of the individual outputs
    global_connection_pattern = [[] for o in range(len(inputs))]
    for out in outputs:
        out_connection_pattern = connect_pattern_by_var.get(out)
        if out_connection_pattern is None:
            # the output is completely isolated from inputs
            out_connection_pattern = [False] * len(inputs)
        for i in range(len(inputs)):
            global_connection_pattern[i].append(out_connection_pattern[i])

    return global_connection_pattern


def op_as_string(
    i, op, leaf_formatter=default_leaf_formatter, node_formatter=default_node_formatter
):
    """Return a function that returns a string representation of the subgraph between `i` and :attr:`op.inputs`"""
    strs = as_string(i, op.inputs, leaf_formatter, node_formatter)
    return node_formatter(op, strs)


def as_string(
    inputs: List[Variable],
    outputs: List[Variable],
    leaf_formatter=default_leaf_formatter,
    node_formatter=default_node_formatter,
) -> List[str]:
    r"""Returns a string representation of the subgraph between `inputs` and `outputs`.

    Parameters
    ----------
    inputs : list
        Input `Variable`\s.
    outputs : list
        Output `Variable`\s.
    leaf_formatter : callable
        Takes a `Variable` and returns a string to describe it.
    node_formatter : callable
        Takes an `Op` and the list of strings corresponding to its arguments
        and returns a string to describe it.

    Returns
    -------
    list of str
        Returns a string representation of the subgraph between `inputs` and
        `outputs`. If the same node is used by several other nodes, the first
        occurrence will be marked as :literal:`*n -> description` and all
        subsequent occurrences will be marked as :literal:`*n`, where ``n`` is an id
        number (ids are attributed in an unspecified order and only exist for
        viewing convenience).

    """
    i = set(inputs)

    orph = list(orphans_between(i, outputs))

    multi = set()
    seen = set()
    for output in outputs:
        op = output.owner
        if op in seen:
            multi.add(op)
        else:
            seen.add(op)
    for op in applys_between(i, outputs):
        for input in op.inputs:
            op2 = input.owner
            if input in i or input in orph or op2 is None:
                continue
            if op2 in seen:
                multi.add(op2)
            else:
                seen.add(input.owner)
    multi_list = [x for x in multi]
    done: Set = set()

    def multi_index(x):
        return multi_list.index(x) + 1

    def describe(r):
        if r.owner is not None and r not in i and r not in orph:
            op = r.owner
            idx = op.outputs.index(r)
            if len(op.outputs) == 1:
                idxs = ""
            else:
                idxs = f"::{idx}"
            if op in done:
                return f"*{multi_index(op)}{idxs}"
            else:
                done.add(op)
                s = node_formatter(op, [describe(input) for input in op.inputs])
                if op in multi_list:
                    return f"*{multi_index(op)} -> {s}"
                else:
                    return s
        else:
            return leaf_formatter(r)

    return [describe(output) for output in outputs]


def view_roots(node: Variable) -> List[Variable]:
    """Return the leaves from a search through consecutive view-maps."""
    owner = node.owner
    if owner is not None:
        try:
            view_map = owner.op.view_map
            view_map = {owner.outputs[o]: i for o, i in view_map.items()}
        except AttributeError:
            return [node]
        if node in view_map:
            answer = []
            for i in view_map[node]:
                answer += view_roots(owner.inputs[i])
            return answer
        else:
            return [node]
    else:
        return [node]


def list_of_nodes(
    inputs: Collection[Variable], outputs: Iterable[Variable]
) -> List[Apply]:
    r"""Return the `Apply` nodes of the graph between `inputs` and `outputs`.

    Parameters
    ----------
    inputs : list of Variable
        Input `Variable`\s.
    outputs : list of Variable
        Output `Variable`\s.

    """

    def expand(o: Apply) -> List[Apply]:
        return [
            inp.owner
            for inp in o.inputs
            if inp.owner and not any(i in inp.owner.outputs for i in inputs)
        ]

    return list(
        cast(
            Iterable[Apply],
            walk(
                [o.owner for o in outputs if o.owner],
                expand,
            ),
        )
    )


def is_in_ancestors(l_apply: Apply, f_node: Apply) -> bool:
    """Determine if `f_node` is in the graph given by `l_apply`.

    Parameters
    ----------
    l_apply : Apply
        The node to walk.
    f_apply : Apply
        The node to find in `l_apply`.

    Returns
    -------
    bool

    """
    computed = set()
    todo = [l_apply]
    while todo:
        cur = todo.pop()
        if cur.outputs[0] in computed:
            continue
        if all(i in computed or i.owner is None for i in cur.inputs):
            computed.update(cur.outputs)
            if cur is f_node:
                return True
        else:
            todo.append(cur)
            todo.extend(i.owner for i in cur.inputs if i.owner)
    return False


def equal_computations(
    xs: List[Union[np.ndarray, Variable]],
    ys: List[Union[np.ndarray, Variable]],
    in_xs: Optional[List[Variable]] = None,
    in_ys: Optional[List[Variable]] = None,
) -> bool:
    """Checks if Aesara graphs represent the same computations.

    The two lists `xs`, `ys` should have the same number of entries. The
    function checks if for any corresponding pair ``(x, y)`` from ``zip(xs, ys)``
    ``x`` and ``y`` represent the same computations on the same variables
    (unless equivalences are provided using `in_xs`, `in_ys`).

    If `in_xs` and `in_ys` are provided, then when comparing a node ``x`` with
    a node ``y`` they are automatically considered as equal if there is some
    index ``i`` such that ``x == in_xs[i]`` and ``y == in_ys[i]`` (and they both
    have the same type). Note that ``x`` and ``y`` can be in the list `xs` and
    `ys`, but also represent subgraphs of a computational graph in `xs`
    or `ys`.

    Parameters
    ----------
    xs : list of Variable
    ys : list of Variable

    Returns
    -------
    bool

    """
    if len(xs) != len(ys):
        raise ValueError("The number of graphs/Variables in each argument must match.")

    if in_xs is None:
        in_xs = []
    if in_ys is None:
        in_ys = []

    for x, y in zip(xs, ys):
        if not isinstance(x, Variable) and not isinstance(y, Variable):
            return cast(bool, np.array_equal(x, y))
        if not isinstance(x, Variable):
            if isinstance(y, Constant):
                return cast(bool, np.array_equal(y.data, x))
            return False
        if not isinstance(y, Variable):
            if isinstance(x, Constant):
                return cast(bool, np.array_equal(x.data, y))
            return False
        if x.owner and not y.owner:
            return False
        if y.owner and not x.owner:
            return False
        if x.owner and y.owner:
            if x.owner.outputs.index(x) != y.owner.outputs.index(y):
                return False
        if x not in in_xs and not (y.type.in_same_class(x.type)):
            return False

    if len(in_xs) != len(in_ys):
        return False

    for _x, _y in zip(in_xs, in_ys):
        if not (_y.type.in_same_class(_x.type)):
            return False

    common = set(zip(in_xs, in_ys))
    different: Set[Tuple[Variable, Variable]] = set()
    for dx, dy in zip(xs, ys):
        assert isinstance(dx, Variable)
        # We checked above that both dx and dy have an owner or not
        if not dx.owner:
            if isinstance(dx, Constant) and isinstance(dy, Constant):
                if not dx.equals(dy):
                    return False
                else:
                    pass
            elif (dx, dy) not in common and dx != dy:
                return False

    # Explore the two graphs, in parallel, depth first, comparing the nodes
    # along the way for equality.
    def compare_nodes(nd_x, nd_y, common, different):
        """
        Compare two nodes to determine if they perform equal computation.
        This is done by comparing the ops, the number of inputs, outputs and
        by ensuring that the inputs themselves are the result of equal
        computation.

        NOTE : This function relies on the variable common to cache
        results to be more efficient.

        """
        if nd_x is nd_y:
            return True

        if nd_x.op != nd_y.op:
            return False
        elif len(nd_x.inputs) != len(nd_y.inputs):
            return False
        elif len(nd_x.outputs) != len(nd_y.outputs):
            return False
        else:
            all_in_common = True
            for dx, dy in zip(nd_x.outputs, nd_y.outputs):
                if (dx, dy) in different:
                    return False
                if (dx, dy) not in common:
                    all_in_common = False

            if all_in_common:
                return True

            # Compare the individual inputs for equality
            for dx, dy in zip(nd_x.inputs, nd_y.inputs):
                if (dx, dy) not in common:

                    # Equality between the variables is unknown, compare
                    # their respective owners, if they have some
                    if (
                        dx.owner
                        and dy.owner
                        and dx.owner.outputs.index(dx) == dy.owner.outputs.index(dy)
                    ):

                        nodes_equal = compare_nodes(
                            dx.owner, dy.owner, common, different
                        )
                        if not nodes_equal:
                            different.add((dx, dy))
                            return False

                    # If both variables don't have an owner, then they are
                    # inputs and can be directly compared
                    elif dx.owner is None and dy.owner is None:

                        if dx != dy:
                            if isinstance(dx, Constant) and isinstance(dy, Constant):
                                if not dx.equals(dy):
                                    return False
                            else:
                                return False

                    else:
                        return False

            # If the code reaches this statement then the inputs are pair-wise
            # equivalent so the outputs of the current nodes are also
            # pair-wise equivalents
            for dx, dy in zip(nd_x.outputs, nd_y.outputs):
                common.add((dx, dy))

            return True

    # Validate that each xs[i], ys[i] pair represents the same computation
    for i in range(len(xs)):
        x_i: Variable = cast(Variable, xs[i])
        if x_i.owner:
            y_i: Variable = cast(Variable, ys[i])
            # The case where pairs of x[i]s and y[i]s don't both have an owner
            # have already been addressed.
            is_equal = compare_nodes(x_i.owner, y_i.owner, common, different)
            if not is_equal:
                return False

    return True


def get_var_by_name(
    graphs: Iterable[Variable], target_var_id: str, ids: str = "CHAR"
) -> Tuple[Variable, ...]:
    r"""Get variables in a graph using their names.

    Parameters
    ----------
    graphs:
        The graph, or graphs, to search.
    target_var_id:
        The name to match against either ``Variable.name`` or
        ``Variable.auto_name``.

    Returns
    -------
    A ``tuple`` containing all the `Variable`\s that match `target_var_id`.

    """
    from aesara.graph.op import HasInnerGraph

    def expand(r) -> Optional[List[Variable]]:
        if r.owner:
            res = list(r.owner.inputs)

            if isinstance(r.owner.op, HasInnerGraph):
                res.extend(r.owner.op.inner_outputs)

            return res

    results: Tuple[Variable, ...] = ()
    for var in walk(graphs, expand, False):
        var = cast(Variable, var)
        if target_var_id == var.name or target_var_id == var.auto_name:
            results += (var,)

    return results
