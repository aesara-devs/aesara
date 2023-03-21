import numpy as np
import numpy.linalg
import pytest
from numpy import inf
from numpy.testing import assert_array_almost_equal

import aesara
from aesara import function
from aesara.configdefaults import config
from aesara.graph.basic import Constant
from aesara.tensor.math import _allclose
from aesara.tensor.nlinalg import (
    SVD,
    Eig,
    MatrixInverse,
    TensorInv,
    det,
    eig,
    eigh,
    lstsq,
    matrix_dot,
    matrix_inverse,
    matrix_power,
    norm,
    pinv,
    qr,
    svd,
    tensorinv,
    tensorsolve,
    trace,
)
from aesara.tensor.type import (
    lmatrix,
    lscalar,
    matrix,
    scalar,
    tensor3,
    tensor4,
    vector,
)
from tests import unittest_tools as utt


def test_pseudoinverse_correctness():
    rng = np.random.default_rng(utt.fetch_seed())
    d1 = rng.integers(4) + 2
    d2 = rng.integers(4) + 2
    r = rng.standard_normal((d1, d2)).astype(config.floatX)

    x = matrix()
    xi = pinv(x)

    ri = function([x], xi)(r)
    assert ri.shape[0] == r.shape[1]
    assert ri.shape[1] == r.shape[0]
    assert ri.dtype == r.dtype
    # Note that pseudoinverse can be quite imprecise so I prefer to compare
    # the result with what np.linalg returns
    assert _allclose(ri, np.linalg.pinv(r))


def test_pseudoinverse_grad():
    rng = np.random.default_rng(utt.fetch_seed())
    d1 = rng.integers(4) + 2
    d2 = rng.integers(4) + 2
    r = rng.standard_normal((d1, d2)).astype(config.floatX)

    utt.verify_grad(pinv, [r])


class TestMatrixInverse(utt.InferShapeTester):
    def setup_method(self):
        super().setup_method()
        self.op_class = MatrixInverse
        self.op = matrix_inverse
        self.rng = np.random.default_rng(utt.fetch_seed())

    def test_inverse_correctness(self):
        r = self.rng.standard_normal((4, 4)).astype(config.floatX)

        x = matrix()
        xi = self.op(x)

        ri = function([x], xi)(r)
        assert ri.shape == r.shape
        assert ri.dtype == r.dtype

        rir = np.dot(ri, r)
        rri = np.dot(r, ri)

        assert _allclose(np.identity(4), rir), rir
        assert _allclose(np.identity(4), rri), rri

    def test_infer_shape(self):
        r = self.rng.standard_normal((4, 4)).astype(config.floatX)

        x = matrix()
        xi = self.op(x)

        self._compile_and_check([x], [xi], [r], self.op_class, warn=False)


def test_matrix_dot():
    rng = np.random.default_rng(utt.fetch_seed())
    n = rng.integers(4) + 2
    rs = []
    xs = []
    for k in range(n):
        rs += [rng.standard_normal((4, 4)).astype(config.floatX)]
        xs += [matrix()]
    sol = matrix_dot(*xs)

    aesara_sol = function(xs, sol)(*rs)
    numpy_sol = rs[0]
    for r in rs[1:]:
        numpy_sol = np.dot(numpy_sol, r)

    assert _allclose(numpy_sol, aesara_sol)


def test_qr_modes():
    rng = np.random.default_rng(utt.fetch_seed())

    A = matrix("A", dtype=config.floatX)
    a = rng.random((4, 4)).astype(config.floatX)

    f = function([A], qr(A))
    t_qr = f(a)
    n_qr = np.linalg.qr(a)
    assert _allclose(n_qr, t_qr)

    for mode in ["reduced", "r", "raw"]:
        f = function([A], qr(A, mode))
        t_qr = f(a)
        n_qr = np.linalg.qr(a, mode)
        if isinstance(n_qr, (list, tuple)):
            assert _allclose(n_qr[0], t_qr[0])
            assert _allclose(n_qr[1], t_qr[1])
        else:
            assert _allclose(n_qr, t_qr)

    try:
        n_qr = np.linalg.qr(a, "complete")
        f = function([A], qr(A, "complete"))
        t_qr = f(a)
        assert _allclose(n_qr, t_qr)
    except TypeError as e:
        assert "name 'complete' is not defined" in str(e)


class TestSvd(utt.InferShapeTester):
    op_class = SVD
    dtype = "float32"

    def setup_method(self):
        super().setup_method()
        self.rng = np.random.default_rng(utt.fetch_seed())
        self.A = matrix(dtype=self.dtype)
        self.op = svd

    def test_svd(self):
        A = matrix("A", dtype=self.dtype)
        U, S, VT = svd(A)
        fn = function([A], [U, S, VT])
        a = self.rng.random((4, 4)).astype(self.dtype)
        n_u, n_s, n_vt = np.linalg.svd(a)
        t_u, t_s, t_vt = fn(a)

        assert _allclose(n_u, t_u)
        assert _allclose(n_s, t_s)
        assert _allclose(n_vt, t_vt)

        fn = function([A], svd(A, compute_uv=False))
        t_s = fn(a)
        assert _allclose(n_s, t_s)

    def test_svd_infer_shape(self):
        self.validate_shape((4, 4), full_matrices=True, compute_uv=True)
        self.validate_shape((4, 4), full_matrices=False, compute_uv=True)
        self.validate_shape((2, 4), full_matrices=False, compute_uv=True)
        self.validate_shape((4, 2), full_matrices=False, compute_uv=True)
        self.validate_shape((4, 4), compute_uv=False)

    def validate_shape(self, shape, compute_uv=True, full_matrices=True):
        A = self.A
        A_v = self.rng.random(shape).astype(self.dtype)
        outputs = self.op(A, full_matrices=full_matrices, compute_uv=compute_uv)
        if not compute_uv:
            outputs = [outputs]
        self._compile_and_check([A], outputs, [A_v], self.op_class, warn=False)


def test_tensorsolve():
    rng = np.random.default_rng(utt.fetch_seed())

    A = tensor4("A", dtype=config.floatX)
    B = matrix("B", dtype=config.floatX)
    X = tensorsolve(A, B)
    fn = function([A, B], [X])

    # slightly modified example from np.linalg.tensorsolve docstring
    a = np.eye(2 * 3 * 4).astype(config.floatX)
    a.shape = (2 * 3, 4, 2, 3 * 4)
    b = rng.random((2 * 3, 4)).astype(config.floatX)

    n_x = np.linalg.tensorsolve(a, b)
    t_x = fn(a, b)
    assert _allclose(n_x, t_x)

    # check the type upcast now
    C = tensor4("C", dtype="float32")
    D = matrix("D", dtype="float64")
    Y = tensorsolve(C, D)
    fn = function([C, D], [Y])

    c = np.eye(2 * 3 * 4, dtype="float32")
    c.shape = (2 * 3, 4, 2, 3 * 4)
    d = rng.random((2 * 3, 4)).astype("float64")
    n_y = np.linalg.tensorsolve(c, d)
    t_y = fn(c, d)
    assert _allclose(n_y, t_y)
    assert n_y.dtype == Y.dtype

    # check the type upcast now
    E = tensor4("E", dtype="int32")
    F = matrix("F", dtype="float64")
    Z = tensorsolve(E, F)
    fn = function([E, F], [Z])

    e = np.eye(2 * 3 * 4, dtype="int32")
    e.shape = (2 * 3, 4, 2, 3 * 4)
    f = rng.random((2 * 3, 4)).astype("float64")
    n_z = np.linalg.tensorsolve(e, f)
    t_z = fn(e, f)
    assert _allclose(n_z, t_z)
    assert n_z.dtype == Z.dtype


def test_inverse_singular():
    singular = np.array([[1, 0, 0]] + [[0, 1, 0]] * 2, dtype=config.floatX)
    a = matrix()
    f = function([a], matrix_inverse(a))
    with pytest.raises(np.linalg.LinAlgError):
        f(singular)


def test_inverse_grad():
    rng = np.random.default_rng(utt.fetch_seed())
    r = rng.standard_normal((4, 4))
    utt.verify_grad(matrix_inverse, [r], rng=np.random)

    rng = np.random.default_rng(utt.fetch_seed())

    r = rng.standard_normal((4, 4))
    utt.verify_grad(matrix_inverse, [r], rng=np.random)


def test_det():
    rng = np.random.default_rng(utt.fetch_seed())

    r = rng.standard_normal((5, 5)).astype(config.floatX)
    x = matrix()
    f = aesara.function([x], det(x))
    assert np.allclose(np.linalg.det(r), f(r))


def test_det_grad():
    rng = np.random.default_rng(utt.fetch_seed())

    r = rng.standard_normal((5, 5)).astype(config.floatX)
    utt.verify_grad(det, [r], rng=np.random)


def test_det_shape():
    x = matrix()
    det_shape = det(x).shape
    assert isinstance(det_shape, Constant)
    assert tuple(det_shape.data) == ()


def test_trace():
    rng = np.random.default_rng(utt.fetch_seed())
    x = matrix()
    g = trace(x)
    f = aesara.function([x], g)

    for shp in [(2, 3), (3, 2), (3, 3)]:
        m = rng.random(shp).astype(config.floatX)
        v = np.trace(m)
        assert v == f(m)

    xx = vector()
    ok = False
    try:
        trace(xx)
    except TypeError:
        ok = True
    except ValueError:
        ok = True

    assert ok


class TestEig(utt.InferShapeTester):
    op_class = Eig
    op = eig
    dtype = "float64"

    def setup_method(self):
        super().setup_method()
        self.rng = np.random.default_rng(utt.fetch_seed())
        self.A = matrix(dtype=self.dtype)
        self.X = np.asarray(self.rng.random((5, 5)), dtype=self.dtype)
        self.S = self.X.dot(self.X.T)

    def test_infer_shape(self):
        A = self.A
        S = self.S
        self._compile_and_check(
            [A],  # aesara.function inputs
            self.op(A),  # aesara.function outputs
            # S must be square
            [S],
            self.op_class,
            warn=False,
        )

    def test_eval(self):
        A = matrix(dtype=self.dtype)
        assert [e.eval({A: [[1]]}) for e in self.op(A)] == [[1.0], [[1.0]]]
        x = [[0, 1], [1, 0]]
        w, v = (e.eval({A: x}) for e in self.op(A))
        assert_array_almost_equal(np.dot(x, v), w * v)


class TestEigh(TestEig):
    op = staticmethod(eigh)

    def test_uplo(self):
        S = self.S
        a = matrix(dtype=self.dtype)
        wu, vu = (out.eval({a: S}) for out in self.op(a, "U"))
        wl, vl = (out.eval({a: S}) for out in self.op(a, "L"))
        assert_array_almost_equal(wu, wl)
        assert_array_almost_equal(vu * np.sign(vu[0, :]), vl * np.sign(vl[0, :]))

    def test_grad(self):
        X = self.X
        # We need to do the dot inside the graph because Eigh needs a
        # matrix that is hermitian
        utt.verify_grad(lambda x: self.op(x.dot(x.T))[0], [X], rng=self.rng)
        utt.verify_grad(lambda x: self.op(x.dot(x.T))[1], [X], rng=self.rng)
        utt.verify_grad(lambda x: self.op(x.dot(x.T), "U")[0], [X], rng=self.rng)
        utt.verify_grad(lambda x: self.op(x.dot(x.T), "U")[1], [X], rng=self.rng)


class TestEighFloat32(TestEigh):
    dtype = "float32"

    def test_uplo(self):
        super().test_uplo()

    def test_grad(self):
        super().test_grad()


class TestLstsq:
    def test_correct_solution(self):
        x = lmatrix()
        y = lmatrix()
        z = lscalar()
        b = lstsq(x, y, z)
        f = function([x, y, z], b)
        TestMatrix1 = np.asarray([[2, 1], [3, 4]])
        TestMatrix2 = np.asarray([[17, 20], [43, 50]])
        TestScalar = np.asarray(1)
        f = function([x, y, z], b)
        m = f(TestMatrix1, TestMatrix2, TestScalar)
        assert np.allclose(TestMatrix2, np.dot(TestMatrix1, m[0]))

    def test_wrong_coefficient_matrix(self):
        x = vector()
        y = vector()
        z = scalar()
        b = lstsq(x, y, z)
        f = function([x, y, z], b)
        with pytest.raises(np.linalg.linalg.LinAlgError):
            f([2, 1], [2, 1], 1)

    def test_wrong_rcond_dimension(self):
        x = vector()
        y = vector()
        z = vector()
        b = lstsq(x, y, z)
        f = function([x, y, z], b)
        with pytest.raises(np.linalg.LinAlgError):
            f([2, 1], [2, 1], [2, 1])


class TestMatrixPower:
    @config.change_flags(compute_test_value="raise")
    @pytest.mark.parametrize("n", [-1, 0, 1, 2, 3, 4, 5, 11])
    def test_numpy_compare(self, n):
        a = np.array([[0.1231101, 0.72381381], [0.28748201, 0.43036511]]).astype(
            config.floatX
        )
        A = matrix("A", dtype=config.floatX)
        A.tag.test_value = a
        Q = matrix_power(A, n)
        n_p = np.linalg.matrix_power(a, n)
        assert np.allclose(n_p, Q.get_test_value())

    def test_non_square_matrix(self):
        A = matrix("A", dtype=config.floatX)
        Q = matrix_power(A, 3)
        f = function([A], [Q])
        a = np.array(
            [
                [0.47497769, 0.81869379],
                [0.74387558, 0.31780172],
                [0.54381007, 0.28153101],
            ]
        ).astype(config.floatX)
        with pytest.raises(ValueError):
            f(a)


class TestNormTests:
    def test_wrong_type_of_ord_for_vector(self):
        with pytest.raises(ValueError):
            norm([2, 1], "fro")

    def test_wrong_type_of_ord_for_matrix(self):
        with pytest.raises(ValueError):
            norm([[2, 1], [3, 4]], 0)

    def test_non_tensorial_input(self):
        with pytest.raises(ValueError):
            norm(3, None)

    def test_tensor_input(self):
        with pytest.raises(NotImplementedError):
            norm(np.random.random((3, 4, 5)), None)

    def test_numpy_compare(self):
        rng = np.random.default_rng(utt.fetch_seed())

        M = matrix("A", dtype=config.floatX)
        V = vector("V", dtype=config.floatX)

        a = rng.random((4, 4)).astype(config.floatX)
        b = rng.random(4).astype(config.floatX)

        A = (
            [None, "fro", "inf", "-inf", 1, -1, None, "inf", "-inf", 0, 1, -1, 2, -2],
            [M, M, M, M, M, M, V, V, V, V, V, V, V, V],
            [a, a, a, a, a, a, b, b, b, b, b, b, b, b],
            [None, "fro", inf, -inf, 1, -1, None, inf, -inf, 0, 1, -1, 2, -2],
        )

        for i in range(0, 14):
            f = function([A[1][i]], norm(A[1][i], A[0][i]))
            t_n = f(A[2][i])
            n_n = np.linalg.norm(A[2][i], A[3][i])
            assert _allclose(n_n, t_n)


class TestTensorInv(utt.InferShapeTester):
    def setup_method(self):
        super().setup_method()
        self.A = tensor4("A", dtype=config.floatX)
        self.B = tensor3("B", dtype=config.floatX)
        self.a = np.random.random((4, 6, 8, 3)).astype(config.floatX)
        self.b = np.random.random((2, 15, 30)).astype(config.floatX)
        self.b1 = np.random.random((30, 2, 15)).astype(
            config.floatX
        )  # for ind=1 since we need prod(b1.shape[:ind]) == prod(b1.shape[ind:])

    def test_infer_shape(self):
        A = self.A
        Ai = tensorinv(A)
        self._compile_and_check(
            [A],  # aesara.function inputs
            [Ai],  # aesara.function outputs
            [self.a],  # value to substitute
            TensorInv,
        )

    def test_eval(self):
        A = self.A
        Ai = tensorinv(A)
        n_ainv = np.linalg.tensorinv(self.a)
        tf_a = function([A], [Ai])
        t_ainv = tf_a(self.a)
        assert _allclose(n_ainv, t_ainv)

        B = self.B
        Bi = tensorinv(B)
        Bi1 = tensorinv(B, ind=1)
        n_binv = np.linalg.tensorinv(self.b)
        n_binv1 = np.linalg.tensorinv(self.b1, ind=1)
        tf_b = function([B], [Bi])
        tf_b1 = function([B], [Bi1])
        t_binv = tf_b(self.b)
        t_binv1 = tf_b1(self.b1)
        assert _allclose(t_binv, n_binv)
        assert _allclose(t_binv1, n_binv1)
