import numpy as np
import pytest

from aesara.compile.function import function
from aesara.compile.mode import Mode
from aesara.graph.basic import Variable, ancestors
from aesara.tensor.subtensor import AdvancedSubtensor
from aesara.tensor.subtensor_opt import local_replace_AdvancedSubtensor
from aesara.tensor.type import tensor
from tests.unittest_tools import create_aesara_param


y = create_aesara_param(np.random.randint(0, 4, size=(2,)))
z = create_aesara_param(np.random.randint(0, 4, size=(2, 2)))


@pytest.mark.parametrize(
    ("indices", "is_none"),
    [
        ((slice(None), y, y), True),
        ((y, y, slice(None)), True),
        ((y,), False),
        ((slice(None), y), False),
        ((y, slice(None)), False),
        ((slice(None), y, slice(None)), False),
        ((slice(None), z, slice(None)), False),
        ((slice(None), z), False),
        ((z, slice(None)), False),
        ((slice(None), z, slice(None)), False),
    ],
)
def test_local_replace_AdvancedSubtensor(indices, is_none):

    X_val = np.random.normal(size=(4, 4, 4))
    X = tensor(np.float64, [False, False, False], name="X")
    X.tag.test_value = X_val

    Y = X[indices]

    res_at = local_replace_AdvancedSubtensor.transform(None, Y.owner)

    if is_none:
        assert res_at is None
    else:
        (res_at,) = res_at

        assert not any(
            isinstance(v.owner.op, AdvancedSubtensor)
            for v in ancestors([res_at])
            if v.owner
        )

        inputs = [X] + [i for i in indices if isinstance(i, Variable)]

        res_fn = function(inputs, res_at, mode=Mode("py", None, None))
        exp_res_fn = function(inputs, Y, mode=Mode("py", None, None))

        # Make sure that the expected result graph has an `AdvancedSubtensor`
        assert any(
            isinstance(v.owner.op, AdvancedSubtensor)
            for v in exp_res_fn.maker.fgraph.variables
            if v.owner
        )

        res_val = res_fn(*[i.tag.test_value for i in inputs])
        exp_res_val = exp_res_fn(*[i.tag.test_value for i in inputs])

        assert np.array_equal(res_val, exp_res_val)
