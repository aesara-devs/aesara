"""Provide a simple user friendly API to Aesara-managed memory."""

import copy
from contextlib import contextmanager
from functools import singledispatch
from typing import List, Optional

from aesara.graph.basic import Variable
from aesara.graph.utils import add_tag_trace
from aesara.link.basic import Container
from aesara.link.c.type import generic


__SHARED_CONTEXT__: Optional[List[Variable]] = None


@contextmanager
def collect_new_shareds():
    r"""Return all the `SharedVariable`\s created within this context manager."""
    global __SHARED_CONTEXT__
    old_context = __SHARED_CONTEXT__
    context = []
    try:
        __SHARED_CONTEXT__ = context
        yield context
    finally:
        __SHARED_CONTEXT__ = old_context


class SharedVariable(Variable):
    """Variable that is shared between compiled functions."""

    container: Optional[Container] = None
    """
    A container to use for this SharedVariable when it is an implicit
    function parameter.
    """

    def __init__(self, name, type, value, strict, allow_downcast=None, container=None):
        super().__init__(type=type, name=name, owner=None, index=None)

        if container is not None:
            self.container = container
            if (value is not None) or (strict is not None):
                raise TypeError(
                    "value and strict are ignored if you pass " "a container here"
                )
        else:
            self.container = Container(
                self,
                storage=[
                    type.filter(value, strict=strict, allow_downcast=allow_downcast)
                ],
                readonly=False,
                strict=strict,
                allow_downcast=allow_downcast,
            )

        global __SHARED_CONTEXT__

        if isinstance(__SHARED_CONTEXT__, list):
            __SHARED_CONTEXT__.append(self)

        self._default_update: Optional[Variable] = None

    def get_value(self, borrow=False, return_internal_type=False):
        """
        Get the non-symbolic value associated with this SharedVariable.

        Parameters
        ----------
        borrow : bool
            True to permit returning of an object aliased to internal memory.
        return_internal_type : bool
            True to permit the returning of an arbitrary type object used
            internally to store the shared variable.

        Only with borrow=False and return_internal_type=True does this function
        guarantee that you actually get the internal object.
        But in that case, you may get different return types when using
        different compute devices.

        """
        if borrow:
            return self.container.value
        else:
            return copy.deepcopy(self.container.value)

    def set_value(self, new_value, borrow=False):
        """
        Set the non-symbolic value associated with this SharedVariable.

        Parameters
        ----------
        borrow : bool
            True to use the new_value directly, potentially creating problems
            related to aliased memory.

        Changes to this value will be visible to all functions using
        this SharedVariable.
        """
        if borrow:
            self.container.value = new_value
        else:
            self.container.value = copy.deepcopy(new_value)

    def get_test_value(self):
        return self.get_value(borrow=True, return_internal_type=True)

    def zero(self, borrow=False):
        """
        Set the values of a shared variable to 0.

        Parameters
        ----------
        borrow : bbol
            True to modify the value of a shared variable directly by using
            its previous value. Potentially this can cause problems
            regarding to the aliased memory.

        Changes done with this function will be visible to all functions using
        this SharedVariable.

        """
        if borrow:
            self.container.value[...] = 0
        else:
            self.container.value = 0 * self.container.value

    def clone(self, **kwargs):
        name = kwargs.get("name", self.name)
        cp = self.__class__(
            name=name,
            type=self.type,
            value=None,
            strict=None,
            container=self.container,
        )
        cp.tag = copy.copy(self.tag)
        return cp

    @property
    def default_update(self) -> Optional[Variable]:
        """A default update expression for this `Variable`.

        If this value is non-``None``, its value will be used as the `update`
        (see `aesara.function`) for this `Variable` when no updates are
        provided through `aesara.function` and `no_default_updates` isn't
        enabled.
        """
        return self._default_update

    @default_update.setter
    def default_update(self, value):
        if value is not None:
            self._default_update = self.type.filter_variable(value, allow_convert=True)
        else:
            self._default_update = value


def shared(value, name=None, strict=False, allow_downcast=None, **kwargs):
    r"""Create a `SharedVariable` initialized with a copy or reference of `value`.

    This function iterates over constructor functions to find a
    suitable `SharedVariable` subclass.  The suitable one is the first
    constructor that accept the given value.  See the documentation of
    :func:`shared_constructor` for the definition of a constructor
    function.

    This function is meant as a convenient default.  If you want to use a
    specific constructor, consider calling it directly.

    `aesara.shared` is a shortcut to this function.

    Notes
    -----
    By passing kwargs, you effectively limit the set of potential constructors
    to those that can accept those kwargs.

    Some shared variable have `borrow` as a kwarg.

    `SharedVariable`\s of `TensorType` have `broadcastable` as a kwarg. As shared
    variable shapes can change, all dimensions default to not being
    broadcastable, even if `value` has a shape of 1 along some dimension.
    This parameter allows one to create for example a row or column tensor.

    """

    if isinstance(value, Variable):
        raise TypeError("Shared variable values can not be symbolic.")

    try:
        var = shared_constructor(
            value,
            name=name,
            strict=strict,
            allow_downcast=allow_downcast,
            **kwargs,
        )
        add_tag_trace(var)
        return var
    except MemoryError as e:
        e.args = e.args + ("Consider using `aesara.shared(..., borrow=True)`",)
        raise


@singledispatch
def shared_constructor(value, name=None, strict=False, allow_downcast=None, **kwargs):
    return SharedVariable(
        type=generic,
        value=value,
        name=name,
        strict=strict,
        allow_downcast=allow_downcast,
    )
