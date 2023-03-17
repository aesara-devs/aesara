import copy
from typing import TypeVar

import numpy as np

from aesara.compile.sharedvalue import SharedVariable, shared_constructor
from aesara.tensor.random.type import (
    RandomGeneratorType,
    RandomStateType,
    RandomType,
    random_generator_type,
    random_state_type,
)


RNGTypeType = TypeVar("RNGTypeType", bound=RandomType)


class RandomTypeSharedVariable(SharedVariable[RNGTypeType]):
    """A `Variable` type representing shared RNG states."""

    def __str__(self):
        return self.name or f"{self.__class__.__name__}({repr(self.container)})"


class RandomStateSharedVariable(RandomTypeSharedVariable[RandomStateType]):
    pass


class RandomGeneratorSharedVariable(RandomTypeSharedVariable[RandomGeneratorType]):
    pass


@shared_constructor.register(np.random.RandomState)
@shared_constructor.register(np.random.Generator)
def randomgen_constructor(
    value, name=None, strict=False, allow_downcast=None, borrow=False
):
    r"""`SharedVariable` constructor for NumPy's `Generator` and/or `RandomState`."""
    if isinstance(value, np.random.RandomState):
        rng_sv_type = RandomStateSharedVariable
        rng_type = random_state_type
    elif isinstance(value, np.random.Generator):
        rng_sv_type = RandomGeneratorSharedVariable
        rng_type = random_generator_type

    if not borrow:
        value = copy.deepcopy(value)

    return rng_sv_type(
        type=rng_type,
        value=value,
        strict=strict,
        allow_downcast=allow_downcast,
        name=name,
    )
