import copy

import numpy as np

from aesara.compile.function import function
from aesara.compile.io import Out
from aesara.tensor.math import dot
from aesara.tensor.nnet import crossentropy_softmax_argmax_1hot_with_bias
from aesara.tensor.type import dmatrix, dvector, ivector, matrix


def test_bug_2009_07_17_borrowed_output():
    # Regression test for a bug where output was borrowed by mistake.
    a = dmatrix()
    b = dmatrix()
    # The output should *NOT* be borrowed.
    g = function([a, b], Out(dot(a, b), borrow=False))

    x = np.zeros((1, 2))
    y = np.ones((2, 5))

    z = g(x, y)
    # print(z)  # Should be zero.
    x.fill(1)
    # print(g(x, y))  # Should be non-zero.
    # print(z)  # Should still be zero.
    assert np.linalg.norm(z) == 0

    # The code above was supposed to fail when it was written (or, more
    # accurately, on the next revision, i.e. when it was merged with the
    # rest of the code, i.e. on revision cac9c9e9f08e).
    # However, for some reason, it does not fail anymore when at this revision.
    # Thus, a new test (below) was added that exhibits the same issue. Note
    # that it may better be moved into the test_nnet.py test file if it turns
    # out the bug was caused by 'crossentropy_softmax_argmax_1hot_with_bias',
    # and was not a more general issue.
    test_output_activation_no_bias = dmatrix()
    test_b2 = dvector()
    test_target = ivector()
    nll_softmax_argmax = crossentropy_softmax_argmax_1hot_with_bias(
        test_output_activation_no_bias, test_b2, test_target
    )
    output = nll_softmax_argmax[1]
    g = function(
        [test_output_activation_no_bias, test_b2, test_target],
        Out(output, borrow=False),
    )

    a = np.zeros((1, 5))
    b = np.ones(5)
    c = np.zeros(1, dtype=np.int32)

    z = g(a, b, c)
    z_backup = copy.copy(z)
    id_z = id(z)
    # print(f"Output z after first call: {z}")
    a[0, 0] = 1
    id_other = id(g(a, b, c))
    # print(f"Output z after second call: {z}")
    # Ensure that calling the function again returns a pointer towards a new
    # array.
    assert id_z != id_other
    # Just to be 100% sure, ensure that z was not altered.
    assert (z == z_backup).all()


def test_deepcopied_type_filter():
    a = copy.deepcopy(matrix())

    # The following should run cleanly.
    # As of commit 731e2d2fa68487733320d341d08b454a50c90d12
    # it was failing.
    a.type.filter(np.ones((2, 2), dtype=a.dtype), strict=True)
