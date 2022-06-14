from abc import abstractmethod
from typing import Any, Generic, Optional, Text, Tuple, TypeVar, Union

from typing_extensions import TypeAlias

from aesara.graph import utils
from aesara.graph.basic import Constant, Variable
from aesara.graph.utils import MetaObject


D = TypeVar("D")


class Type(MetaObject, Generic[D]):
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
    ) -> variable_type:
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
        return type(self)(*args, **kwargs)

    def __call__(self, name: Optional[Text] = None) -> variable_type:
        """Return a new `Variable` instance of Type `self`.

        Parameters
        ----------
        name : None or str
            A pretty string for printing and debugging.

        """
        return utils.add_tag_trace(self.make_variable(name))

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


class HasDataType:
    """A mixin for a type that has a :attr:`dtype` attribute."""

    dtype: str


class HasShape:
    """A mixin for a type that has :attr:`shape` and :attr:`ndim` attributes."""

    ndim: int
    shape: Tuple[Optional[int], ...]
