import pytest

from aesara.utils import set_index


@pytest.mark.parametrize(
    "x, idx, value, exp_res",
    [
        ((0,), 0, "a", ("a",)),
        ((0, 1), 0, "a", ("a", 1)),
        ((0, 1), 1, "a", (0, "a")),
        ((0, 1, 2), 1, "a", (0, "a", 2)),
        ((), 0, "a", IndexError),
        ((0,), 1, "a", IndexError),
    ],
)
def test_set_index(x, idx, value, exp_res):
    if not isinstance(exp_res, tuple):
        with pytest.raises(exp_res):
            set_index(x, idx, value) == exp_res
    else:
        assert set_index(x, idx, value) == exp_res
