import numpy as np
from pytest import fixture, importorskip, raises

import theano.tensor as tt
from theano import change_flags
from theano.gof.fg import FunctionGraph
from theano.gof.graph import Variable
from theano.gof.graph import inputs as tt_inputs
from theano.gof.op import get_test_value
from theano.tensor.random.basic import (
    categorical,
    choice,
    dirichlet,
    mvnormal,
    normal,
    permutation,
    polyagamma,
    randint,
    sample_categorical,
    sample_dirichlet,
)
from theano.tensor.random.op import RandomVariable


@fixture(scope="module", autouse=True)
def set_theano_flags():
    with change_flags(cxx="", compute_test_value="raise"):
        yield


def rv_numpy_tester(rv, *params, **kwargs):
    """Test for correspondence between `RandomVariable` and NumPy shape and
    broadcast dimensions.
    """
    numpy_fn = kwargs.pop("numpy_fn", None)

    if numpy_fn is None:
        name = getattr(rv, "name", None)

        if name is None:
            name = rv.__name__

        numpy_fn = getattr(np.random, name)

    theano_res = rv(*params, **kwargs)

    param_vals = [get_test_value(p) if isinstance(p, Variable) else p for p in params]
    kwargs_vals = {
        k: get_test_value(v) if isinstance(v, Variable) else v
        for k, v in kwargs.items()
    }

    if "size" in kwargs:
        kwargs["size"] = get_test_value(kwargs["size"])

    numpy_res = np.asarray(numpy_fn(*param_vals, **kwargs_vals))

    numpy_dtype = numpy_res.dtype
    assert theano_res.dtype == numpy_dtype

    numpy_shape = np.shape(numpy_res)
    numpy_bcast = [s == 1 for s in numpy_shape]
    np.testing.assert_array_equal(theano_res.type.broadcastable, numpy_bcast)

    theano_res_val = theano_res.get_test_value()
    np.testing.assert_array_equal(theano_res_val.shape, numpy_res.shape)


def test_RandomVariable():

    str_res = str(
        RandomVariable(
            "normal",
            0,
            [0, 0],
            "normal",
            inplace=True,
        )
    )

    assert str_res == "normal_rv"

    # `ndims_params` should be a `Sequence` type
    with raises(TypeError):
        RandomVariable(
            "normal",
            0,
            0,
            "normal",
            inplace=True,
        )

    # `size` should be a `Sequence` type
    with raises(TypeError):
        RandomVariable(
            "normal",
            0,
            [0, 0],
            "normal",
            inplace=True,
        )(0, 1, size={1, 2})

    # No dtype
    with raises(TypeError):
        RandomVariable(
            "normal",
            0,
            [0, 0],
            "normal",
            inplace=True,
        )(0, 1)


def test_Normal_infer_shape():
    M_tt = tt.iscalar("M")
    M_tt.tag.test_value = 3
    sd_tt = tt.scalar("sd")
    sd_tt.tag.test_value = 1.0

    test_params = [
        ([tt.as_tensor_variable(1.0), sd_tt], None),
        ([tt.as_tensor_variable(1.0), sd_tt], (M_tt,)),
        ([tt.as_tensor_variable(1.0), sd_tt], (2, M_tt)),
        ([tt.zeros((M_tt,)), sd_tt], None),
        ([tt.zeros((M_tt,)), sd_tt], (M_tt,)),
        ([tt.zeros((M_tt,)), sd_tt], (2, M_tt)),
        ([tt.zeros((M_tt,)), tt.ones((M_tt,))], None),
        ([tt.zeros((M_tt,)), tt.ones((M_tt,))], (2, M_tt)),
    ]
    for args, size in test_params:
        rv = normal(*args, size=size)
        rv_shape = tuple(normal._infer_shape(size or (), args, None))
        assert tuple(get_test_value(rv_shape)) == tuple(get_test_value(rv).shape)


def test_Normal_ShapeFeature():
    M_tt = tt.iscalar("M")
    M_tt.tag.test_value = 3
    sd_tt = tt.scalar("sd")
    sd_tt.tag.test_value = 1.0

    d_rv = normal(tt.ones((M_tt,)), sd_tt, size=(2, M_tt))
    d_rv.tag.test_value

    fg = FunctionGraph(
        [i for i in tt_inputs([d_rv]) if not isinstance(i, tt.Constant)],
        [d_rv],
        clone=False,
        features=[tt.opt.ShapeFeature()],
    )
    s1, s2 = fg.shape_feature.shape_of[d_rv]

    assert get_test_value(s1) == get_test_value(d_rv).shape[0]
    assert get_test_value(s2) == get_test_value(d_rv).shape[1]


def test_normal_samples():
    rv_numpy_tester(normal, 0.0, 1.0)
    rv_numpy_tester(normal, 0.0, 1.0, size=[3])
    # Broadcast sd over independent means...
    rv_numpy_tester(normal, [0.0, 1.0, 2.0], 1.0)
    rv_numpy_tester(normal, [0.0, 1.0, 2.0], 1.0, size=[3, 3])
    rv_numpy_tester(normal, [0], [1], size=[1])
    rv_numpy_tester(normal, tt.as_tensor_variable([0]), [1], size=[1])
    rv_numpy_tester(
        normal, tt.as_tensor_variable([0]), [1], size=tt.as_tensor_variable([1])
    )


def test_mvnormal_samples():
    rv_numpy_tester(mvnormal, [0], np.diag([1]))
    rv_numpy_tester(mvnormal, [0], np.diag([1]), size=[1])
    rv_numpy_tester(mvnormal, [0], np.diag([1]), size=[4])
    rv_numpy_tester(mvnormal, [0], np.diag([1]), size=[4, 1])
    rv_numpy_tester(mvnormal, [0], np.diag([1]), size=[4, 1, 1])
    rv_numpy_tester(mvnormal, [0], np.diag([1]), size=[1, 5, 8])
    rv_numpy_tester(mvnormal, [0, 1, 2], np.diag([1, 1, 1]))
    # Broadcast cov matrix across independent means?
    # Looks like NumPy doesn't support that (and it's probably better off for
    # it).
    # rv_numpy_tester(mvnormal, [[0, 1, 2], [4, 5, 6]], np.diag([1, 1, 1]))


def test_mvnormal_ShapeFeature():
    M_tt = tt.iscalar("M")
    M_tt.tag.test_value = 2

    d_rv = mvnormal(tt.ones((M_tt,)), tt.eye(M_tt), size=2)

    fg = FunctionGraph(
        [i for i in tt_inputs([d_rv]) if not isinstance(i, tt.Constant)],
        [d_rv],
        clone=False,
        features=[tt.opt.ShapeFeature()],
    )

    s1, s2 = fg.shape_feature.shape_of[d_rv]

    assert get_test_value(s1) == 2
    assert M_tt in tt_inputs([s2])


def test_polyagamma_samples():

    _ = importorskip("pypolyagamma")

    # Sampled values should be scalars
    pg_rv = polyagamma(1.1, -10.5)
    assert get_test_value(pg_rv).shape == ()

    pg_rv = polyagamma(1.1, -10.5, size=[1])
    assert get_test_value(pg_rv).shape == (1,)

    pg_rv = polyagamma(1.1, -10.5, size=[2, 3])
    bcast_smpl = get_test_value(pg_rv)
    assert bcast_smpl.shape == (2, 3)
    # Make sure they're not all equal
    assert np.all(np.abs(np.diff(bcast_smpl.flat)) > 0.0)

    pg_rv = polyagamma(np.r_[1.1, 3], -10.5)
    bcast_smpl = get_test_value(pg_rv)
    assert bcast_smpl.shape == (2,)
    assert np.all(np.abs(np.diff(bcast_smpl.flat)) > 0.0)

    pg_rv = polyagamma(np.r_[1.1, 3], -10.5, size=(3, 2))
    bcast_smpl = get_test_value(pg_rv)
    assert bcast_smpl.shape == (3, 2)
    assert np.all(np.abs(np.diff(bcast_smpl.flat)) > 0.0)


def test_dirichlet_samples():

    alphas = np.c_[[100, 1, 1], [1, 100, 1], [1, 1, 100]]

    res = get_test_value(dirichlet(alphas))
    assert np.all(np.diag(res) >= res)

    res = get_test_value(dirichlet(alphas, size=2))
    assert res.shape == (2, 3, 3)
    assert all(np.all(np.diag(r) >= r) for r in res)

    for i in range(alphas.shape[0]):
        res = get_test_value(dirichlet(alphas[i]))
        assert np.all(res[i] > np.delete(res, [i]))

        res = get_test_value(dirichlet(alphas[i], size=2))
        assert res.shape == (2, 3)
        assert all(np.all(r[i] > np.delete(r, [i])) for r in res)

    rng_state = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(1234)))

    alphas = np.array([[1000, 1, 1], [1, 1000, 1], [1, 1, 1000]])

    assert sample_dirichlet(rng_state, alphas).shape == alphas.shape
    assert sample_dirichlet(rng_state, alphas, size=10).shape == (10,) + alphas.shape
    assert (
        sample_dirichlet(rng_state, alphas, size=(10, 2)).shape
        == (10, 2) + alphas.shape
    )


def test_dirichlet_infer_shape():
    M_tt = tt.iscalar("M")
    M_tt.tag.test_value = 3

    test_params = [
        ([tt.ones((M_tt,))], None),
        ([tt.ones((M_tt,))], (M_tt + 1,)),
        ([tt.ones((M_tt,))], (2, M_tt)),
        ([tt.ones((M_tt, M_tt + 1))], None),
        ([tt.ones((M_tt, M_tt + 1))], (M_tt + 2,)),
        ([tt.ones((M_tt, M_tt + 1))], (2, M_tt + 2, M_tt + 3)),
    ]
    for args, size in test_params:
        rv = dirichlet(*args, size=size)
        rv_shape = tuple(dirichlet._infer_shape(size or (), args, None))
        assert tuple(get_test_value(rv_shape)) == tuple(get_test_value(rv).shape)


def test_dirichlet_ShapeFeature():
    """Make sure `RandomVariable.infer_shape` works with `ShapeFeature`."""
    M_tt = tt.iscalar("M")
    M_tt.tag.test_value = 2
    N_tt = tt.iscalar("N")
    N_tt.tag.test_value = 3

    d_rv = dirichlet(tt.ones((M_tt, N_tt)), name="Gamma")

    fg = FunctionGraph(
        [i for i in tt_inputs([d_rv]) if not isinstance(i, tt.Constant)],
        [d_rv],
        clone=False,
        features=[tt.opt.ShapeFeature()],
    )

    s1, s2 = fg.shape_feature.shape_of[d_rv]

    assert M_tt in tt_inputs([s1])
    assert N_tt in tt_inputs([s2])


def test_categorical_samples():

    rng_state = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(1234)))

    p = np.array([[100000, 1, 1], [1, 100000, 1], [1, 1, 100000]])
    p = p / p.sum(axis=-1)

    assert sample_categorical(rng_state, p).shape == p.shape[:-1]

    with raises(ValueError):
        sample_categorical(rng_state, p, size=10)

    assert sample_categorical(rng_state, p, size=(10, 3)).shape == (10, 3)
    assert sample_categorical(rng_state, p, size=(10, 2, 3)).shape == (10, 2, 3)

    res = categorical(p)
    assert np.array_equal(get_test_value(res), np.arange(3))

    res = categorical(p, size=(10, 3))
    exp_res = np.tile(np.arange(3), (10, 1))
    assert np.array_equal(get_test_value(res), exp_res)

    res = categorical(p, size=(10, 2, 3))
    exp_res = np.tile(np.arange(3), (10, 2, 1))
    assert np.array_equal(get_test_value(res), exp_res)


def test_random_integer_samples():
    rv_numpy_tester(randint, 0, 1)
    rv_numpy_tester(randint, 0, 1, size=[3])
    rv_numpy_tester(randint, [0, 1, 2], 5)
    rv_numpy_tester(randint, [0, 1, 2], 5, size=[3, 3])
    rv_numpy_tester(randint, [0], [5], size=[1])
    rv_numpy_tester(randint, tt.as_tensor_variable([-1]), [1], size=[1])
    rv_numpy_tester(
        randint, tt.as_tensor_variable([-1]), [1], size=tt.as_tensor_variable([1])
    )


def test_choice_samples():
    rv_numpy_tester(choice, np.asarray(5))
    rv_numpy_tester(choice, [1.0, 5.0])
    rv_numpy_tester(choice, np.asarray(5), 3)
    with raises(ValueError):
        rv_numpy_tester(choice, np.array([[1, 2], [3, 4]]))
    rv_numpy_tester(choice, [1, 2, 3], 1)
    rv_numpy_tester(choice, [1, 2, 3], 1, p=tt.as_tensor([1 / 3.0, 1 / 3.0, 1 / 3.0]))
    rv_numpy_tester(choice, [1, 2, 3], (10, 2), replace=True)
    rv_numpy_tester(choice, tt.as_tensor_variable([1, 2, 3]), 2, replace=True)


def test_permutation_samples():
    rv_numpy_tester(
        permutation, np.asarray(5), numpy_fn=lambda x: np.random.permutation(x.item())
    )
    rv_numpy_tester(permutation, [1, 2, 3])
    rv_numpy_tester(permutation, [[1, 2], [3, 4]])
    rv_numpy_tester(permutation, [1.0, 2.0, 3.0])
