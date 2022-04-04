import numpy as np
import pytest

import aesara
from aesara.compile import SharedVariable


sp = pytest.importorskip("scipy", minversion="0.7.0")


def test_shared_basic():
    x = aesara.shared(sp.sparse.csr_matrix(np.eye(100)), name="blah", borrow=True)

    assert isinstance(x, SharedVariable)
