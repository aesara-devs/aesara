import copyreg
from abc import ABCMeta, abstractmethod
from itertools import chain
from typing import (
    Annotated,
    Any,
    Optional,
    Text,
    Tuple,
    TypeVar,
    Union,
    get_args,
    get_type_hints,
)

from typing_extensions import Protocol, TypeAlias, runtime_checkable

from aesara.graph import utils
from aesara.graph.basic import Constant, Variable
from aesara.graph.utils import MetaType


D = TypeVar("D")
PropsV = TypeVar("PropsV")
Props = Annotated[Optional[PropsV], "props"]


class NewTypeMeta(ABCMeta):
    """
    Interface specification for variable type instances.

    A :term:`Type` instance is mainly responsible for two things:

    - creating `Variable` instances (conventionally, `__call__` does this), and

    - filtering a value assigned to a `Variable` so that the value
      conforms to restrictions imposed by the type (also known as
      casting, this is done by `filter`).

    """

    variable_type: TypeAlias = Variable
    """
    The `Type` that will be created by a call to `Type.make_variable`.
    """

    constant_type: TypeAlias = Constant
    """
    The `Type` that will be created by a call to `Type.make_constant`.
    """

    _prop_names: tuple[str, ...] = tuple()
    _subclass_cache = dict()

    _base_type: Optional["NewTypeMeta"] = None
    _type_parameters: dict[str, Any] = dict()

    @staticmethod
    def make_key(params):
        res = []
        for k, v in sorted(params.items()):
            if isinstance(v, dict):
                v = NewTypeMeta.make_key(v)
            res.append((k, v))

        return tuple(res)

    @classmethod
    def __new__(cls, *args, **kwargs):
        res = super().__new__(*args, **kwargs)
        props = tuple(
            k
            for k, v in chain(
                get_type_hints(type(res), include_extras=True).items(),
                get_type_hints(res, include_extras=True).items(),
            )
            if "props" in get_args(v)
        )
        res._prop_names = props
        copyreg.pickle(type(res), _pickle_NewTypeMeta)
        return res

    def subtype(cls, *args, **kwargs):
        # For dynamically created types the attribute base_type exists and points to the base type it was derived from
        base_type = cls.base_type
        kwargs = base_type.type_parameters(*args, **kwargs)

        return base_type.subtype_params(kwargs)

    @property
    def base_type(cls):
        if cls._base_type is None:
            return cls
        else:
            return cls._base_type

    def subtype_params(cls, params):
        if not params:
            return cls

        key = (cls, *NewTypeMeta.make_key(params))
        try:
            return NewTypeMeta._subclass_cache[key]
        except KeyError:
            pass
        cls_name = f"{cls.__name__}{params}"

        res = type(cls)(cls_name, (cls,), params)
        res._base_type = cls
        res._type_parameters = params

        NewTypeMeta._subclass_cache[key] = res
        return res

    def __call__(self, name: Optional[Text] = None) -> Any:
        """Return a new `Variable` instance of Type `self`.

        Parameters
        ----------
        name : None or str
            A pretty string for printing and debugging.

        """
        return utils.add_tag_trace(self.make_variable(name))

    def type_parameters(cls, *args, **kwargs):
        if args:
            kwargs |= zip(cls._prop_names, args)
        return kwargs

    @classmethod
    def create(cls, **kwargs):
        MetaType(f"{cls.__name__}[{kwargs}]", (cls,), kwargs)

    def in_same_class(self, otype: "Type") -> Optional[bool]:
        """Determine if another `Type` represents a subset from the same "class" of types represented by `self`.

        A "class" of types could be something like "float64 tensors with four
        dimensions".  One `Type` could represent a set containing only a type
        for "float64 tensors with shape (1, 2, 3, 4)" and another the set of
        "float64 tensors with shape (1, x, x, x)" for all suitable "x".

        It's up to each subclass of `Type` to determine to which "classes" of types this
        method applies.

        The default implementation assumes that all "classes" have only one
        unique element (i.e. it uses `self.__eq__`).

        """
        if self == otype:
            return True

        return False

    def is_super(self, otype: "Type") -> Optional[bool]:
        """Determine if `self` is a supertype of `otype`.

        This method effectively implements the type relation ``>``.

        In general, ``t1.is_super(t2) == True`` implies that ``t1`` can be
        replaced with ``t2``.

        See `Type.in_same_class`.

        Returns
        -------
        ``None`` if the type relation cannot be applied/determined.

        """
        if self.in_same_class(otype):
            return True

        return None

    @abstractmethod
    def filter(
        self, data: Any, strict: bool = False, allow_downcast: Optional[bool] = None
    ) -> D:
        """Return data or an appropriately wrapped/converted data.

        Subclass implementations should raise a TypeError exception if
        the data is not of an acceptable type.

        Parameters
        ----------
        data: array-like
            The data to be filtered/converted.
        strict: bool (optional)
            If ``True``, the data returned must be the same as the
            data passed as an argument.
        allow_downcast: bool (optional)
            If `strict` is ``False``, and `allow_downcast` is ``True``, the
            data may be cast to an appropriate type. If `allow_downcast` is
            ``False``, it may only be up-cast and not lose precision. If
            `allow_downcast` is ``None`` (default), the behaviour can be
            type-dependent, but for now it means only Python floats can be
            down-casted, and only to floatX scalars.

        """

    def filter_inplace(
        self,
        value: Any,
        storage: Any,
        strict: bool = False,
        allow_downcast: Optional[bool] = None,
    ):
        """Return data or an appropriately wrapped/converted data by converting it in-place.

        This method allows one to reuse old allocated memory.  If this method
        is implemented, it will be called instead of `Type.filter`.

        As of now, this method is not implemented and was previously used for transferring memory to and from GPU.

        Parameters
        ----------
        value: array-like
        storage: array-like
            The old value (e.g. the old NumPy array)
        strict: bool
        allow_downcast: bool (optional)

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError()

    def filter_variable(
        self, other: Union[Variable, D], allow_convert: bool = True
    ) -> Any:
        r"""Convert a `other` into a `Variable` with a `Type` that's compatible with `self`.

        If the involved `Type`\s are not compatible, a `TypeError` will be raised.
        """
        if not isinstance(other, Variable):
            # The value is not a Variable: we cast it into
            # a Constant of the appropriate Type.
            other = self.constant_type(type=self, data=other)

        if other.type != self and allow_convert:
            other2 = self.convert_variable(other)
            if other2 is not None:
                return other2

        if other.type != self:
            raise TypeError(
                f"Cannot convert Type {other.type} "
                f"(of Variable {other}) into Type {self}. "
                f"You can try to manually convert {other} into a {self}."
            )
        return other

    def convert_variable(self, var: Variable) -> Optional[Variable]:
        """Produce a `Variable` that's compatible with both `self` and `var.type`, if possible.

        A compatible `Variable` is a `Variable` with a `Type` that's the
        "narrower" of `self` and `var.type`.

        If a compatible `Type` cannot be found, this method will return
        ``None``.

        """
        var_type = var.type

        if self.is_super(var_type):
            # `var.type` is at least as specific as `self`, so we return it
            # as-is
            return var
        elif var_type.is_super(self):
            # `var.type` is less specific than `self`, so we need to convert it
            # to `self`'s `Type`.
            #
            # Note that, in this case, `var.type != self`, because equality is
            # covered by the branch above.
            raise NotImplementedError()

        return None

    def is_valid_value(self, data: D, strict: bool = True) -> bool:
        """Return ``True`` for any python object that would be a legal value for a `Variable` of this `Type`."""
        try:
            self.filter(data, strict=strict)
            return True
        except (TypeError, ValueError):
            return False

    def make_variable(self, name: Optional[Text] = None) -> variable_type:
        """Return a new `Variable` instance of this `Type`.

        Parameters
        ----------
        name: None or str
            A pretty string for printing and debugging.

        """
        return self.variable_type(self, None, name=name)

    def make_constant(self, value: D, name: Optional[Text] = None) -> constant_type:
        """Return a new `Constant` instance of this `Type`.

        Parameters
        ----------
        value: array-like
            The constant value.
        name: None or str
            A pretty string for printing and debugging.

        """
        return self.constant_type(type=self, data=value, name=name)

    def clone(self, *args, **kwargs) -> "Type":
        """Clone a copy of this type with the given arguments/keyword values, if any."""
        return self.subtype(*args, **kwargs)

    @classmethod
    def values_eq(cls, a: D, b: D) -> bool:
        """Return ``True`` if `a` and `b` can be considered exactly equal.

        `a` and `b` are assumed to be valid values of this `Type`.

        """
        return a == b

    @classmethod
    def values_eq_approx(cls, a: D, b: D) -> bool:
        """Return ``True`` if `a` and `b` can be considered approximately equal.

        This function is used by Aesara debugging tools to decide
        whether two values are equivalent, admitting a certain amount
        of numerical instability. For example, for floating-point
        numbers this function should be an approximate comparison.

        By default, this does an exact comparison.

        Parameters
        ----------
        a: array-like
            A potential value for a `Variable` of this `Type`.
        b: array-like
            A potential value for a `Variable` of this `Type`.

        Returns
        -------
        bool

        """
        return cls.values_eq(a, b)

    def _props(self):
        """
        Tuple of properties of all attributes
        """
        return tuple(getattr(self, a) for a in self._prop_names)

    def _props_dict(self):
        """This return a dict of all ``__props__`` key-> value.

        This is useful in optimization to swap op that should have the
        same props. This help detect error that the new op have at
        least all the original props.

        """
        return {a: getattr(self, a) for a in self._prop_names}

    # def __hash__(self):
    #     return hash((type(self), tuple(getattr(self, a, None) for a in self._prop_names)))

    # def __eq__(self, other):
    #     return type(self) == type(other) and tuple(
    #         getattr(self, a) for a in self._prop_names
    #     ) == tuple(getattr(other, a) for a in self._prop_names)

    def __str__(self):
        if self._prop_names is None or len(self._prop_names) == 0:
            return f"{self.__class__.__name__}()"
        else:
            return "{}{{{}}}".format(
                self.__class__.__name__,
                ", ".join(
                    "{}={!r}".format(p, getattr(self, p)) for p in self._prop_names
                ),
            )


def _pickle_NewTypeMeta(type_: NewTypeMeta):
    base_type = type_.base_type
    if base_type is type_:
        return type_.__name__
    return base_type.subtype_params, (type_._type_parameters,)


copyreg.pickle(NewTypeMeta, _pickle_NewTypeMeta)


class Type(metaclass=NewTypeMeta):
    pass


DataType = str


@runtime_checkable
class HasDataType(Protocol):
    """A protocol matching any class with :attr:`dtype` attribute."""

    dtype: DataType


ShapeType = Tuple[Optional[int], ...]


@runtime_checkable
class HasShape(Protocol):
    """A protocol matching any class that has :attr:`shape` and :attr:`ndim` attributes."""

    ndim: int
    shape: ShapeType
