import typing
from copy import copy, deepcopy

from theano.gof.type import Type
from theano.utils import deprecated


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
            deepcopy(self.readonly, memo=memo),
            deepcopy(self.strict, memo=memo),
            deepcopy(self.allow_downcast, memo=memo),
            deepcopy(self.name, memo=memo),
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
    """

    def __init__(self, *, allow_gc: typing.Optional[bool] = None):
        self._allow_gc = allow_gc
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

    @deprecated("Marked for deletion. Only tests use it.")
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
        from theano.gof import utils

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
                return utils.to_return_values([variable.data for variable in outputs])
            else:
                return [variable.data for variable in outputs]

        execute.thunk = thunk
        execute.inputs = inputs
        execute.outputs = outputs

        return execute

    def schedule(self, fgraph):
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
