import copy

import numpy as np

from aesara.compile.sharedvalue import SharedVariable, shared_constructor
from aesara.tensor.random.type import random_state_type


class RandomStateSharedVariable(SharedVariable):
    def __str__(self):
        return "RandomStateSharedVariable({})".format(repr(self.container))


@shared_constructor
def randomstate_constructor(
    value, name=None, strict=False, allow_downcast=None, borrow=False
):
    """
    SharedVariable Constructor for RandomState.

    """
    if not isinstance(value, np.random.RandomState):
        raise TypeError
    if not borrow:
        value = copy.deepcopy(value)
    return RandomStateSharedVariable(
        type=random_state_type,
        value=value,
        name=name,
        strict=strict,
        allow_downcast=allow_downcast,
    )
