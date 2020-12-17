import typing
from copy import copy, deepcopy

from theano import config, utils
from theano.gof.fg import FunctionGraph
from theano.gof.graph import Apply, Constant
from theano.gof.type import Type
from theano.gof.utils import to_return_values
from theano.link.debugging import raise_with_op


class Container:
    """
    This class joins a variable with its computed value.

    It is used in linkers, especially for the inputs and outputs of a Function.

    Parameters
    ----------
    r : a Variable or a Type
    storage
        A list of length 1, whose element is the value for `r`.
    readonly : bool
        True indicates that this should not be setable by Function[r] = val.
    strict : bool
        If True, we don't allow type casting.
    allow_downcast
        If True (and `strict` is False), allow upcasting of type, but not
        downcasting. If False, prevent it. If None (default), allows only
        downcasting of float to floatX scalar.
    name : str
        A string (for pretty-printing?)

    """

    def __init__(
        self,
        r,
        storage,
        *,
        readonly=False,
        strict=False,
        allow_downcast=None,
        name=None,
    ):
        if not isinstance(storage, list) or not len(storage) >= 1:
            raise TypeError("storage must be a list of length at least one")
        if isinstance(r, Type):
            self.type = r
        else:
            self.type = r.type
        if name is None:
            # Some Type do not have a name field.
            self.name = getattr(r, "name", None)
        else:
            self.name = name

        self.storage = storage
        self.readonly = readonly
        self.strict = strict
        self.allow_downcast = allow_downcast

    def __get__(self):
        return self.storage[0]

    def __set__(self, value):
        if self.readonly:
            raise Exception(f"Cannot set readonly storage: {self.name}")
        try:
            if value is None:
                self.storage[0] = None
                return

            kwargs = {}
            if self.strict:
                kwargs["strict"] = True
            if self.allow_downcast is not None:
                kwargs["allow_downcast"] = self.allow_downcast
            if hasattr(self.type, "filter_inplace"):
                self.storage[0] = self.type.filter_inplace(
                    value, self.storage[0], **kwargs
                )
            else:
                self.storage[0] = self.type.filter(value, **kwargs)

        except Exception as e:
            e.args = e.args + (f'Container name "{self.name}"',)
            raise

    data = property(__get__, __set__)
    value = property(__get__, __set__)

    def __str__(self):
        return "<" + str(self.storage[0]) + ">"

    def __repr__(self):
        return "<" + repr(self.storage[0]) + ">"

    def __deepcopy__(self, memo):
        data_was_in_memo = id(self.storage[0]) in memo
        r = type(self)(
            deepcopy(self.type, memo=memo),
            deepcopy(self.storage, memo=memo),
            readonly=deepcopy(self.readonly, memo=memo),
            strict=deepcopy(self.strict, memo=memo),
            allow_downcast=deepcopy(self.allow_downcast, memo=memo),
            name=deepcopy(self.name, memo=memo),
        )
        # Work around NumPy deepcopy of ndarray with 0 dimension that
        # don't return an ndarray.
        if r.storage[0] is not None and not self.type.is_valid_value(r.storage[0]):
            assert not data_was_in_memo
            assert self.type.is_valid_value(self.storage[0])
            # This should also work for read only container.
            r.storage[0] = self.type.filter(
                r.storage[0], strict=False, allow_downcast=False
            )
            memo[id(self.storage[0])] = r.storage[0]
        return r


class Linker:
    """
    Base type for all linkers.

    A linker takes a FunctionGraph and turns it into a callable.

    Parameters
    ----------
    allow_gc : optional, bool
        Configures if garbage collection is enabled.
    scheduler : callable
        A scheduling function that takes a FunctionGraph and returns a list of Apply nodes.
        Defaults to the .toposort() method of the FunctionGraph.
    """

    def __init__(
        self,
        *,
        allow_gc: typing.Optional[bool] = None,
        scheduler: typing.Callable[[FunctionGraph], typing.List[Apply]] = None,
    ):
        self._allow_gc = allow_gc
        self._scheduler = scheduler
        super().__init__()

    @property
    def allow_gc(self) -> typing.Optional[bool]:
        """Determines if the linker may allow garbage collection.

        None means undefined.
        """
        return self._allow_gc

    def clone(self, allow_gc: typing.Optional[bool] = None):
        new = copy(self)
        if allow_gc is not None:
            new._allow_gc = allow_gc
        return new

    def make_thunk(
        self,
    ) -> typing.Tuple[
        typing.Callable[[], typing.NoReturn],
        typing.List[Container],
        typing.List[Container],
    ]:
        """
        This function must return a triplet (function, input_variables,
        output_variables) where function is a thunk that operates on the
        returned variables. If inplace is True, the input_variables and
        output_variables lists will be the same as the inputs and outputs
        of the graph provided to the L{Linker}. Else, independent
        variables will be returned.

        Examples
        --------
        x, y = Variable(Double), Variable(Double)
        e = x + y
        fgraph = FunctionGraph([x, y], [e])
        fn, (new_x, new_y), (new_e, ) = MyLinker(fgraph).make_thunk(inplace)
        new_x.data = 1.0
        new_y.data = 2.0
        fn()
        print new_e.data # 3.0
        print e.data # 3.0 iff inplace == True (else unknown)

        """
        raise NotImplementedError(
            f"make_thunk method of {type(self)} is not implemented."
        )

    @utils.deprecated("Marked for deletion. Only tests use it.")
    def make_function(self, unpack_single=True, **kwargs):
        """
        Returns a function that takes values corresponding to the inputs of the
        fgraph used by this L{Linker} and returns values corresponding the the
        outputs of that fgraph. If inplace is True, the calculations will
        operate in the same storage the fgraph uses, else independent storage
        will be allocated for the function.

        Examples
        --------
        e = x + y
        fgraph = FunctionGraph([x, y], [e])
        fn = MyLinker(fgraph).make_function(inplace)
        print fn(1.0, 2.0) # 3.0
        print e.data # 3.0 iff inplace == True (else unknown)

        If unpack_single is True (default) and that the function has only one
        output, then that output will be returned. Else, a list or tuple of
        length 1 will be returned.

        """
        thunk, inputs, outputs = self.make_thunk(**kwargs)

        def execute(*args):
            takes = len(inputs)
            got = len(args)
            if got != takes:
                raise TypeError(
                    f"Function call takes exactly {takes} args ({got} given)"
                )
            for arg, variable in zip(args, inputs):
                variable.data = arg
            thunk()
            if unpack_single:
                return to_return_values([variable.data for variable in outputs])
            else:
                return [variable.data for variable in outputs]

        execute.thunk = thunk
        execute.inputs = inputs
        execute.outputs = outputs

        return execute

    def schedule(self, fgraph: FunctionGraph) -> typing.List[Apply]:
        """Runs the scheduler (if set) or the toposort on the FunctionGraph.

        Parameters
        ----------
        fgraph : FunctionGraph
            A graph to compute the schedule for.

        Returns
        -------
        nodes : list of Apply nodes
            The result of the scheduling or toposort operation.
        """
        if callable(self._scheduler):
            return self._scheduler(fgraph)
        return fgraph.toposort()


class LocalLinker(Linker):
    """
    Useful base class for L{Linker}s which keep all nodes in the graph, and run
    a thunk associated with each node.

    """

    def make_thunk(self, input_storage=None, output_storage=None, storage_map=None):
        return self.make_all(
            input_storage=input_storage,
            output_storage=output_storage,
            storage_map=storage_map,
        )[:3]

    def make_all(self, input_storage, output_storage):
        # By convention, subclasses of LocalLinker should implement this function!
        #
        # This function should return a tuple of 5 things
        # 1. function to run the program
        # 2. input storage
        # 3. output storage
        # 4. thunks: list of nodes' functions in the order they will be run by the function in (1)
        # 5. order: list of nodes, in the order they will be run by the function in (1)
        raise NotImplementedError(
            f"make_all method of {type(self)} is not implemented."
        )


def map_storage(
    fgraph: FunctionGraph,
    order: typing.Iterable[Apply],
    input_storage: typing.Optional[typing.List],
    output_storage: typing.Optional[typing.List],
    storage_map: typing.Dict = None,
) -> typing.Tuple[typing.List, typing.List, typing.Dict]:
    """Ensure there is storage (a length-1 list) for inputs, outputs, and interior nodes.

    :param fgraph: The current fgraph.  This function uses the inputs and outputs attributes.
    :param order: an iterable over Apply instances (in program running order)
    :param input_storage: None or existing input storage (see below)
    :param output_storage: None or existing output storage (see below)

    :rtype: 3-tuple
    :returns: (list of storage for inputs, list of storage for outputs, and the `storage_map`)

    Parameters
    ----------
    fgraph
        The current fgraph. This function uses the inputs and outputs
        attributes.
    order
        An iterable over Apply instances (in program running order).
    input_storage
        None or existing input storage (see below).
    output_storage
        None or existing output storage (see below).

    Returns
    -------
    3-tuple
        List of storage for inputs, list of storage for outputs, and
        the `storage_map`.

    Extended summary
    ----------------
    This function iterates over the nodes in `order` and ensures that for every
    input and output `Variable`, there is a unique storage container. This is
    returned as a dictionary Variable -> storage called the `storage_map`.

    This function also returns `input_storage`, which is a list of storages
    corresponding to fgraph.inputs.
    This function also returns `output_storage`, which is a list of storages
    corresponding to fgraph.outputs.

    """
    # each Apply argument's data is stored in a list of length 1 (these lists act like pointers)

    if storage_map is None:
        storage_map = {}

    # input_storage is a list of data-containers for the inputs.
    if input_storage is None:
        input_storage = [[None] for input in fgraph.inputs]
    else:
        assert len(fgraph.inputs) == len(input_storage)

    # add input storage into storage_map
    for r, storage in zip(fgraph.inputs, input_storage):
        if r in storage_map:
            assert storage_map[r] is storage, (
                "Given input_storage conflicts "
                "with storage in given storage_"
                "map. Given input_storage: ",
                storage,
                "Storage in storage_ma" "p: ",
                storage_map[r],
            )
        else:
            storage_map[r] = storage
    #     for orphan in fgraph.orphans:
    #         if not isinstance(orphan, Constant):
    #             raise TypeError("Cannot link a graph with non-constant orphans.", orphan)
    #         storage_map[orphan] = [orphan.data]

    # allocate output storage
    if output_storage is not None:
        assert len(fgraph.outputs) == len(output_storage)
        for r, storage in zip(fgraph.outputs, output_storage):
            if r in storage_map:
                assert storage_map[r] is storage, (
                    "Given output_storage confl"
                    "icts with storage in given"
                    " storage_map. Given output"
                    "_storage: ",
                    storage,
                    "Sto" "rage in storage_map: ",
                    storage_map[r],
                )
            else:
                storage_map[r] = storage

    # allocate storage for intermediate computation
    for node in order:
        for r in node.inputs:
            if r not in storage_map:
                assert isinstance(r, Constant)
                storage_map[r] = [r.data]
        for r in node.outputs:
            storage_map.setdefault(r, [None])
    for r in fgraph.outputs:
        if isinstance(r, Constant):
            storage_map.setdefault(r, [r.data])

    # extract output storage
    if output_storage is None:
        output_storage = [storage_map[r] for r in fgraph.outputs]

    return input_storage, output_storage, storage_map


def add_clear_storage(f, computed, storage_map):
    def clear_storage():
        for c in computed:
            storage_map[c][0] = None

    f.clear_storage = clear_storage


def streamline(
    fgraph: FunctionGraph,
    thunks,
    order,
    post_thunk_old_storage=None,
    no_recycling=None,
    nice_errors=True,
) -> typing.Callable[[], typing.NoReturn]:
    """
    WRITEME

    Parameters
    ----------
    fgraph
    thunks
        The list of program instructions.
    order
        The list of apply instances that gave rise to the thunks
        (same order as thunks).
    post_thunk_old_storage
        A list (corresponding to thunks, order) whose elements are lists of
        storage cells, that should be cleared after running thecorresponding
        thunk. A value of None disables this functionality.
    no_recycling
        Storage elements that cannot be 'recycled' by repeatedly executing the
        program. These storage elements are cleared before re-running.
    nice_errors
        Run in such a way that the double-traceback is printed. This costs a
        bit of performance in the inner python loop.

    """
    if no_recycling is None:
        no_recycling = []

    if len(thunks) != len(order):
        raise ValueError(
            "Length of thunks and order must match", (len(thunks), len(order))
        )

    if post_thunk_old_storage:
        if len(thunks) != len(post_thunk_old_storage):
            raise ValueError(
                "Length of thunks and post_thunk_old_storage must match",
                (len(thunks), len(post_thunk_old_storage)),
            )

        def streamline_default_f():
            for x in no_recycling:
                x[0] = None
            try:
                for thunk, node, old_storage in zip(
                    thunks, order, post_thunk_old_storage
                ):
                    thunk()
                    for old_s in old_storage:
                        old_s[0] = None
            except Exception:
                raise_with_op(fgraph, node, thunk)

        f = streamline_default_f
    elif nice_errors:

        def streamline_nice_errors_f():
            for x in no_recycling:
                x[0] = None
            try:
                for thunk, node in zip(thunks, order):
                    thunk()
            except Exception:
                raise_with_op(fgraph, node, thunk)

        f = streamline_nice_errors_f
    else:
        # don't worry about raise_with_op, just go a little faster.
        # there is a mix of python and c thunks
        def streamline_fast_f():
            for x in no_recycling:
                x[0] = None
            for thunk in thunks:
                thunk()

        f = streamline_fast_f
    return f


def gc_helper(node_list: typing.List[Apply]):
    """
    Return the set of Variable instances which are computed by node_list.
    Parameters
    ----------
    node_list
        List of Apply instances in program execution order.

    Returns
    -------
    2-tuple
        FIRST, the set of Variable instances which are computed by node_list,
        and SECOND a dictionary that maps each Variable instance to a the last
        node to use Variable as an input.

    Extended Summary
    ----------------
    This is used to allow garbage collection within graphs.

    It ignores view_map and destroy_map. This isn't needed as python
    have reference count. In Theano gc, we should not take into
    account view_map and destroy_map as if the thunk decided to create
    a new output, we would delay uselessly its gc by Python.

    """
    # for freeing memory
    last_user = {}
    computed = set()
    for node in node_list:
        for input in node.inputs:
            last_user[input] = node
        for output in node.outputs:
            computed.add(output)
    return computed, last_user


class PerformLinker(LocalLinker):
    """
    Basic L{Linker} subclass that calls the perform method on each L{Op} in
    the L{FunctionGraph} in the order given by L{Linker.schedule}.

    """

    def __init__(self, allow_gc=None, schedule=None):
        if allow_gc is None:
            allow_gc = config.allow_gc
        self.fgraph = None
        super().__init__(allow_gc=allow_gc, scheduler=schedule)

    def accept(self, fgraph, no_recycling=None, profile=None):
        """

        Parameters
        ----------
        fgraph
            A PerformLinker can have accepted one FunctionGraph instance at a time.
        no_recycling
            WRITEME

        Returns
        -------
        object
            self (TODO: WHY? Who calls this function?)

        """
        if no_recycling is None:
            no_recycling = []
        if self.fgraph is not None and self.fgraph is not fgraph:
            return type(self)(allow_gc=self.allow_gc).accept(
                fgraph, no_recycling, profile
            )
            # raise Exception("Cannot accept from a Linker that is already tied to another FunctionGraph.")
        self.fgraph = fgraph
        self.no_recycling = no_recycling
        return self

    def make_all(self, input_storage=None, output_storage=None, storage_map=None):
        """
        Returns Function to run all nodes, list of input containers, list of outputs

        Parameters
        ----------
        input_storage
            list of storages corresponding to fgraph.inputs
        output_storage
            list of storages corresponding to fgraph.outputs

        Returns
        -------
        object
            Function to run all nodes, list of input containers, list of output
            containers, list of thunks (for all programs), list of nodes
            (for all programs).

        """
        fgraph = self.fgraph
        order = self.schedule(fgraph)
        no_recycling = self.no_recycling

        input_storage, output_storage, storage_map = map_storage(
            fgraph, order, input_storage, output_storage, storage_map
        )

        compute_map = {}
        for k in storage_map:
            compute_map[k] = [k.owner is None]

        thunks = []
        for node in order:
            # Maker sure we don't use C version of the code, but rather only
            # the python version
            # Note : ops that implement their own make thunk don't usually
            # have this attribute defiend !!
            thunks += [
                node.op.make_thunk(node, storage_map, compute_map, no_recycling, "py")
            ]
            thunks[-1].inputs = [storage_map[v] for v in node.inputs]
            thunks[-1].outputs = [storage_map[v] for v in node.outputs]

        computed, last_user = gc_helper(order)
        if self.allow_gc:
            post_thunk_old_storage = []
        else:
            post_thunk_old_storage = None

        for node in order:
            if self.allow_gc:
                post_thunk_old_storage.append(
                    [
                        storage_map[input]
                        for input in node.inputs
                        if (input in computed)
                        and (input not in fgraph.outputs)
                        and (node == last_user[input])
                    ]
                )

        if no_recycling is True:
            # True seems like some special code for *everything*?? -JB
            # FunctionMaker always passes a list I think   -JB
            no_recycling = list(storage_map.values())
            no_recycling = utils.difference(no_recycling, input_storage)
        else:
            no_recycling = [
                storage_map[r] for r in no_recycling if r not in fgraph.inputs
            ]

        # The function that actually runs your program is one of the f's in streamline.
        f = streamline(
            fgraph, thunks, order, post_thunk_old_storage, no_recycling=no_recycling
        )

        f.allow_gc = (
            self.allow_gc
        )  # HACK: this is a way of passing an arg to Function.__call__
        add_clear_storage(f, computed, storage_map)
        f.storage_map = storage_map

        return (
            f,
            [
                Container(input, storage)
                for input, storage in zip(fgraph.inputs, input_storage)
            ],
            [
                Container(output, storage, readonly=True)
                for output, storage in zip(fgraph.outputs, output_storage)
            ],
            thunks,
            order,
        )


class WrapLinker(Linker):
    """
    This class makes it easier to run several L{LocalLinker}s in parallel, and
    offers some control over how each thunk is run.

    A wrapper function must be provided, and it can be used to execute the
    thunks, inspect the nodes, print stuff out, etc.

    The constructor initializes a WrapLinker.

    Parameters
    ----------
    linkers : list of L{LocalLinker} subclasses, whose make_all() method returns
        thunks in the same order.
        For each node in the graph, each linker will provide a
        thunk.  This class makes it possible to iterate over each linker's
        program in parallel.
    wrapper : lambda (fgraph, i, i_node, i_thunk1, i_thunk2, ...) : None
        Does some user-defined action for the i'th element of the program.
        i_thunk<n> is the thunk returned by the n'th linker. (If you want
        to run the program, make sure to call the necessary thunks in this
        function.)

    Notes
    -----
    The outputs of the first linker will be returned.

    This linker ensures that each linker has its own storage for inputs and
    outputs and intermediate variables. There is no interference between
    linkers.

    """

    def __init__(self, linkers, wrapper):
        self.fgraph = None
        self.linkers = linkers
        self.wrapper = wrapper

    def __copy__(self):
        """
        Shallow copy of a WrapLinker.

        Returns
        -------
        object
            A copy of self, where each of the linkers in self.linkers
            have been shallow-copied.

        It is useful because in FunctionMaker, copy.copy is called on the
        Mode's linker, so that it is not modified inplace when linker.accept()
        is called. In this case, we want the wrapped linkers to be copied too.

        """
        other = self.__class__(
            linkers=[copy(x) for x in self.linkers], wrapper=self.wrapper
        )
        return other

    def clone(self, allow_gc=None):
        return self.__class__(
            linkers=[x.clone(allow_gc=allow_gc) for x in self.linkers],
            wrapper=self.wrapper,
        )

    def accept(self, fgraph, no_recycling=None, profile=None):
        """

        Parameters
        ----------
        fgraph : gof.FunctionGraph
            The fgraph which we will link.
        no_recycling : a list of Variables that belong to fgraph.
            If a Variable is in no_recycling, L{WrapLinker} will clear
            the output storage associated to it (for each linker in linkers)
            during the computation to avoid reusing it.

        """
        if no_recycling is None:
            no_recycling = []
        if self.fgraph is not None and self.fgraph is not fgraph:
            return type(self)(self.linkers, self.wrapper).accept(fgraph, no_recycling)

        self.fgraph = fgraph
        self.no_recycling = no_recycling
        self.linkers = [linker.accept(fgraph, no_recycling) for linker in self.linkers]
        return self

    def pre(self, f, inputs, order, thunk_groups):
        pass

    def make_thunk(self, **kwargs):
        no_recycling = self.no_recycling

        make_all = [self.linkers[0].make_all(**kwargs)]
        kwargs.pop("input_storage", None)
        make_all += [x.make_all(**kwargs) for x in self.linkers[1:]]

        fns, input_lists, output_lists, thunk_lists, order_lists = zip(*make_all)

        order_list0 = order_lists[0]
        for order_list in order_lists[1:]:
            if not order_list0 == order_list:
                raise Exception(
                    "All linkers to WrapLinker should execute operations in the same order."
                )

        inputs0 = input_lists[0]
        outputs0 = output_lists[0]

        thunk_groups = list(zip(*thunk_lists))
        order = [x[0] for x in zip(*order_lists)]

        to_reset = []
        for thunks, node in zip(thunk_groups, order):
            for j, output in enumerate(node.outputs):
                if output in no_recycling:
                    for thunk in thunks:
                        to_reset.append(thunk.outputs[j])

        wrapper = self.wrapper
        pre = self.pre

        def f():
            for inputs in input_lists[1:]:
                for input1, input2 in zip(inputs0, inputs):
                    input2.storage[0] = copy(input1.storage[0])
            for x in to_reset:
                x[0] = None
            pre(self, [input.data for input in input_lists[0]], order, thunk_groups)
            for i, (thunks, node) in enumerate(zip(thunk_groups, order)):
                try:
                    wrapper(self.fgraph, i, node, *thunks)
                except Exception:
                    raise_with_op(self.fgraph, node, *thunks)

        f.thunk_groups = thunk_groups

        return f, inputs0, outputs0


def WrapLinkerMany(linkers, wrappers):
    """
    Variant on WrapLinker that runs a series of wrapper functions instead of
    just one.

    """

    def wrapper(*args):
        for f in wrappers:
            f(*args)

    return WrapLinker(linkers, wrapper)