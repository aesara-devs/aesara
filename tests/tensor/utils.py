import os
from copy import copy
from itertools import combinations
from tempfile import mkstemp

import numpy as np
import pytest

import theano
from tests import unittest_tools as utt
from theano import change_flags, config, function, gof, shared, tensor
from theano.compile.mode import get_default_mode
from theano.tensor.type import TensorType


# Used to exclude random numbers too close to certain values
_eps = 1e-2

if theano.config.floatX == "float32":
    angle_eps = 1e-4
else:
    angle_eps = 1e-10


div_grad_rtol = None
if config.floatX == "float32":
    # We raise the relative tolerance for the grad as there can be errors in
    # float32.
    # This is probably caused by our way of computing the gradient error.
    div_grad_rtol = 0.025

# Use a seeded random number generator so that unittests are deterministic
utt.seed_rng()
test_rng = np.random.RandomState(seed=utt.fetch_seed())
# In order to check random values close to the boundaries when designing
# new tests, you can use utt.MockRandomState, for instance:
# test_rng = MockRandomState(0)
# test_rng = MockRandomState(0.99999982)
# test_rng = MockRandomState(1)

# If you update this, don't forget to modify the two lines after!
ALL_DTYPES = (
    "int8",
    "int16",
    "int32",
    "int64",
    "float32",
    "float64",
    "uint8",
    "uint16",
    "complex64",
    "complex128",
)
REAL_DTYPES = ALL_DTYPES[:6]
COMPLEX_DTYPES = ALL_DTYPES[-2:]

ignore_isfinite_mode = copy(theano.compile.get_default_mode())
ignore_isfinite_mode.check_isfinite = False


def multi_dtype_checks(shape1, shape2, dtypes=ALL_DTYPES, nameprefix=""):
    for dtype1, dtype2 in combinations(dtypes, 2):
        name1 = "%s_%s_%s" % (nameprefix, dtype1, dtype2)
        name2 = "%s_%s_%s" % (nameprefix, dtype2, dtype1)
        obj1 = rand_of_dtype(shape1, dtype1)
        obj2 = rand_of_dtype(shape2, dtype2)
        yield (name1, (obj1, obj2))
        yield (name2, (obj2, obj1))


def multi_dtype_cast_checks(shape, dtypes=ALL_DTYPES, nameprefix=""):
    for dtype1, dtype2 in combinations(dtypes, 2):
        name1 = "%s_%s_%s" % (nameprefix, dtype1, dtype2)
        name2 = "%s_%s_%s" % (nameprefix, dtype2, dtype1)
        obj1 = rand_of_dtype(shape, dtype1)
        obj2 = rand_of_dtype(shape, dtype2)
        yield (name1, (obj1, dtype2))
        yield (name2, (obj2, dtype1))


def inplace_func(
    inputs,
    outputs,
    mode=None,
    allow_input_downcast=False,
    on_unused_input="raise",
    name=None,
):
    if mode is None:
        mode = get_default_mode()
    return function(
        inputs,
        outputs,
        mode=mode,
        allow_input_downcast=allow_input_downcast,
        accept_inplace=True,
        on_unused_input=on_unused_input,
        name=name,
    )


def eval_outputs(outputs, ops=(), mode=None):
    f = inplace_func([], outputs, mode=mode)
    variables = f()
    if ops:
        assert any(isinstance(node.op, ops) for node in f.maker.fgraph.apply_nodes)
    if isinstance(variables, (tuple, list)) and len(variables) == 1:
        return variables[0]
    return variables


def get_numeric_subclasses(cls=np.number, ignore=None):
    # Return subclasses of `cls` in the numpy scalar hierarchy.
    #
    # We only return subclasses that correspond to unique data types.
    # The hierarchy can be seen here:
    #     http://docs.scipy.org/doc/numpy/reference/arrays.scalars.html
    if ignore is None:
        ignore = []
    rval = []
    dtype = np.dtype(cls)
    dtype_num = dtype.num
    if dtype_num not in ignore:
        # Safety check: we should be able to represent 0 with this data type.
        np.array(0, dtype=dtype)
        rval.append(cls)
        ignore.append(dtype_num)
    for sub_ in cls.__subclasses__():
        rval += [c for c in get_numeric_subclasses(sub_, ignore=ignore)]
    return rval


def get_numeric_types(
    with_int=True, with_float=True, with_complex=False, only_theano_types=True
):
    # Return numpy numeric data types.
    #
    # :param with_int: Whether to include integer types.
    #
    # :param with_float: Whether to include floating point types.
    #
    # :param with_complex: Whether to include complex types.
    #
    # :param only_theano_types: If True, then numpy numeric data types that are
    # not supported by Theano are ignored (i.e. those that are not declared in
    # scalar/basic.py).
    #
    # :returns: A list of unique data type objects. Note that multiple data types
    # may share the same string representation, but can be differentiated through
    # their `num` attribute.
    #
    # Note that when `only_theano_types` is True we could simply return the list
    # of types defined in the `scalar` module. However with this function we can
    # test more unique dtype objects, and in the future we may use it to
    # automatically detect new data types introduced in numpy.
    if only_theano_types:
        theano_types = [d.dtype for d in theano.scalar.all_types]
    rval = []

    def is_within(cls1, cls2):
        # Return True if scalars defined from `cls1` are within the hierarchy
        # starting from `cls2`.
        # The third test below is to catch for instance the fact that
        # one can use ``dtype=numpy.number`` and obtain a float64 scalar, even
        # though `numpy.number` is not under `numpy.floating` in the class
        # hierarchy.
        return (
            cls1 is cls2
            or issubclass(cls1, cls2)
            or isinstance(np.array([0], dtype=cls1)[0], cls2)
        )

    for cls in get_numeric_subclasses():
        dtype = np.dtype(cls)
        if (
            (not with_complex and is_within(cls, np.complexfloating))
            or (not with_int and is_within(cls, np.integer))
            or (not with_float and is_within(cls, np.floating))
            or (only_theano_types and dtype not in theano_types)
        ):
            # Ignore this class.
            continue
        rval.append([str(dtype), dtype, dtype.num])
    # We sort it to be deterministic, then remove the string and num elements.
    return [x[1] for x in sorted(rval, key=str)]


def _numpy_checker(x, y):
    # Checks if x.data and y.data have the same contents.
    # Used in DualLinker to compare C version with Python version.
    x, y = x[0], y[0]
    if x.dtype != y.dtype or x.shape != y.shape or np.any(np.abs(x - y) > 1e-10):
        raise Exception("Output mismatch.", {"performlinker": x, "clinker": y})


def safe_make_node(op, *inputs):
    # Emulate the behaviour of make_node when op is a function.
    #
    # Normally op in an instead of the Op class.
    node = op(*inputs)
    if isinstance(node, list):
        return node[0].owner
    else:
        return node.owner


def upcast_float16_ufunc(fn):
    # Decorator that enforces computation is not done in float16 by NumPy.
    #
    # Some ufuncs in NumPy will compute float values on int8 and uint8
    # in half-precision (float16), which is not enough, and not compatible
    # with the C code.
    #
    # :param fn: numpy ufunc
    # :returns: function similar to fn.__call__, computing the same
    #     value with a minimum floating-point precision of float32
    def ret(*args, **kwargs):
        out_dtype = np.find_common_type([a.dtype for a in args], [np.float16])
        if out_dtype == "float16":
            # Force everything to float32
            sig = "f" * fn.nin + "->" + "f" * fn.nout
            kwargs.update(sig=sig)
        return fn(*args, **kwargs)

    return ret


def upcast_int8_nfunc(fn):
    # Decorator that upcasts input of dtype int8 to float32.
    #
    # This is so that floating-point computation is not carried using
    # half-precision (float16), as some NumPy functions do.
    #
    # :param fn: function computing a floating-point value from inputs
    # :returns: function similar to fn, but upcasting its uint8 and int8
    #     inputs before carrying out the computation.
    def ret(*args, **kwargs):
        args = list(args)
        for i, a in enumerate(args):
            if getattr(a, "dtype", None) in ("int8", "uint8"):
                args[i] = a.astype("float32")

        return fn(*args, **kwargs)

    return ret


def rand(*shape):
    r = test_rng.rand(*shape) * 2 - 1
    return np.asarray(r, dtype=config.floatX)


def rand_nonzero(shape, eps=3e-4):
    # Like rand, but the absolute value has to be at least eps
    # covers [0, 1)
    r = np.asarray(test_rng.rand(*shape), dtype=config.floatX)
    # covers [0, (1 - eps) / 2) U [(1 + eps) / 2, 1)
    r = r * (1 - eps) + eps * (r >= 0.5)
    # covers [-1, -eps) U [eps, 1)
    r = r * 2 - 1
    return r


def randint(*shape):
    return test_rng.randint(-5, 6, shape)


def randuint32(*shape):
    return np.array(test_rng.randint(5, size=shape), dtype=np.uint32)


def randuint16(*shape):
    return np.array(test_rng.randint(5, size=shape), dtype=np.uint16)


# XXX: this so-called complex random array as all-zero imaginary parts
def randcomplex(*shape):
    r = np.asarray(test_rng.rand(*shape), dtype=config.floatX)
    return np.complex128(2 * r - 1)


def randcomplex_nonzero(shape, eps=1e-4):
    return np.complex128(rand_nonzero(shape, eps))


def randint_nonzero(*shape):
    r = test_rng.randint(-5, 5, shape)
    return r + (r == 0) * 5


def rand_ranged(min, max, shape):
    return np.asarray(test_rng.rand(*shape) * (max - min) + min, dtype=config.floatX)


def randint_ranged(min, max, shape):
    return test_rng.randint(min, max + 1, shape)


def randc128_ranged(min, max, shape):
    return np.asarray(test_rng.rand(*shape) * (max - min) + min, dtype="complex128")


def rand_of_dtype(shape, dtype):
    if dtype in tensor.discrete_dtypes:
        return randint(*shape).astype(dtype)
    elif dtype in tensor.float_dtypes:
        return rand(*shape).astype(dtype)
    elif dtype in tensor.complex_dtypes:
        return randcomplex(*shape).astype(dtype)
    else:
        raise TypeError()


def check_floatX(inputs, rval):
    # :param inputs: Inputs to a function that returned `rval` with these inputs.
    #
    # :param rval: Value returned by a function with inputs set to `inputs`.
    #
    # :returns: Either `rval` unchanged, or `rval` cast in float32. The idea is
    # that when a numpy function would have returned a float64, Theano may prefer
    # to return a float32 instead when `config.cast_policy` is set to
    # 'numpy+floatX' and config.floatX to 'float32', and there was no float64
    # input.
    if (
        isinstance(rval, np.ndarray)
        and rval.dtype == "float64"
        and config.cast_policy == "numpy+floatX"
        and config.floatX == "float32"
        and all(x.dtype != "float64" for x in inputs)
    ):
        # Then we expect float32 instead of float64.
        return rval.astype("float32")
    else:
        return rval


def _numpy_true_div(x, y):
    # Performs true division, and cast the result in the type we expect.
    #
    # We define that function so we can use it in TrueDivTester.expected,
    # because simply calling np.true_divide could cause a dtype mismatch.
    out = np.true_divide(x, y)
    # Use floatX as the result of int / int
    if x.dtype in tensor.discrete_dtypes and y.dtype in tensor.discrete_dtypes:
        out = theano._asarray(out, dtype=config.floatX)
    return out


def copymod(dct, without=None, **kwargs):
    # Return dct but with the keys named by args removed, and with
    # kwargs added.
    if without is None:
        without = []
    rval = copy(dct)
    for a in without:
        if a in rval:
            del rval[a]
    for kw, val in kwargs.items():
        rval[kw] = val
    return rval


def makeTester(
    name,
    op,
    expected,
    checks=None,
    good=None,
    bad_build=None,
    bad_runtime=None,
    grad=None,
    mode=None,
    grad_rtol=None,
    eps=1e-10,
    skip=False,
    test_memmap=True,
    check_name=False,
    grad_eps=None,
):
    # :param check_name:
    #     Use only for tester that aren't in Theano.
    if checks is None:
        checks = {}
    if good is None:
        good = {}
    if bad_build is None:
        bad_build = {}
    if bad_runtime is None:
        bad_runtime = {}
    if grad is None:
        grad = {}
    if grad is True:
        grad = good

    _op, _expected, _checks, _good = op, expected, checks, good
    _bad_build, _bad_runtime, _grad = bad_build, bad_runtime, grad
    _mode, _grad_rtol, _eps, skip_ = mode, grad_rtol, eps, skip
    _test_memmap = test_memmap
    _check_name = check_name
    _grad_eps = grad_eps

    class Checker:

        op = staticmethod(_op)
        expected = staticmethod(_expected)
        checks = _checks
        check_name = _check_name
        good = _good
        bad_build = _bad_build
        bad_runtime = _bad_runtime
        grad = _grad
        mode = _mode
        skip = skip_
        test_memmap = _test_memmap

        def setup_method(self):
            # Verify that the test's name is correctly set.
            # Some tests reuse it outside this module.
            if self.check_name:
                eval(self.__class__.__module__ + "." + self.__class__.__name__)

            # We keep a list of temporary files created in add_memmap_values,
            # to remove them at the end of the test.
            self.tmp_files = []

        def add_memmap_values(self, val_dict):
            # If test_memmap is True, we create a temporary file
            # containing a copy of the data passed in the "val_dict" dict,
            # then open it as a memmapped array, and we can use the result as a
            # new test value.
            if not self.test_memmap:
                return val_dict

            # Copy dict before modifying them
            val_dict = val_dict.copy()

            # Note that we sort items in the dictionary to ensure tests are
            # deterministic (since the loop below will break on the first valid
            # item that can be memmapped).
            for k, v in sorted(val_dict.items()):
                new_k = "_".join((k, "memmap"))
                if new_k in val_dict:
                    # A corresponding key was already provided
                    break

                new_v = []
                for inp in v:
                    if type(inp) is np.ndarray and inp.size > 0:
                        f, fname = mkstemp()
                        self.tmp_files.append((f, fname))
                        new_inp = np.memmap(
                            fname, dtype=inp.dtype, mode="w+", shape=inp.shape
                        )
                        new_inp[...] = inp[...]
                        new_v.append(new_inp)
                    else:
                        new_v.append(inp)
                val_dict[new_k] = new_v

                # We only need one value, no need to copy all of them
                break
            return val_dict

        def teardown_method(self):
            # This is to avoid a problem with deleting memmap files on windows.
            import gc

            gc.collect()
            for f, fname in self.tmp_files:
                os.close(f)
                os.remove(fname)

        @pytest.mark.skipif(skip, reason="Skipped")
        def test_good(self):
            good = self.add_memmap_values(self.good)

            for testname, inputs in good.items():
                inputs = [copy(input) for input in inputs]
                inputrs = [
                    TensorType(
                        dtype=input.dtype,
                        broadcastable=[shape_elem == 1 for shape_elem in input.shape],
                    )()
                    for input in inputs
                ]
                try:
                    node = safe_make_node(self.op, *inputrs)
                except Exception as exc:
                    err_msg = (
                        "Test %s::%s: Error occurred while"
                        " making a node with inputs %s"
                    ) % (self.op, testname, inputs)
                    exc.args += (err_msg,)
                    raise

                try:
                    f = inplace_func(inputrs, node.outputs, mode=mode, name="test_good")
                except Exception as exc:
                    err_msg = (
                        "Test %s::%s: Error occurred while" " trying to make a Function"
                    ) % (self.op, testname)
                    exc.args += (err_msg,)
                    raise
                if isinstance(self.expected, dict) and testname in self.expected:
                    expecteds = self.expected[testname]
                    # with numpy version, when we print a number and read it
                    # back, we don't get exactly the same result, so we accept
                    # rounding error in that case.
                    eps = 5e-9
                else:
                    expecteds = self.expected(*inputs)
                    eps = 1e-10

                if any(
                    [i.dtype in ("float32", "int8", "uint8", "uint16") for i in inputs]
                ):
                    eps = 1e-6
                eps = np.max([eps, _eps])

                try:
                    variables = f(*inputs)
                except Exception as exc:
                    err_msg = (
                        "Test %s::%s: Error occurred while calling"
                        " the Function on the inputs %s"
                    ) % (self.op, testname, inputs)
                    exc.args += (err_msg,)
                    raise

                if not isinstance(expecteds, (list, tuple)):
                    expecteds = (expecteds,)

                for i, (variable, expected) in enumerate(zip(variables, expecteds)):
                    condition = (
                        variable.dtype != expected.dtype
                        or variable.shape != expected.shape
                        or not np.allclose(variable, expected, atol=eps, rtol=eps)
                    )
                    assert not condition, (
                        "Test %s::%s: Output %s gave the wrong"
                        " value. With inputs %s, expected %s (dtype %s),"
                        " got %s (dtype %s). eps=%f"
                        " np.allclose returns %s %s"
                    ) % (
                        self.op,
                        testname,
                        i,
                        inputs,
                        expected,
                        expected.dtype,
                        variable,
                        variable.dtype,
                        eps,
                        np.allclose(variable, expected, atol=eps, rtol=eps),
                        np.allclose(variable, expected),
                    )

                for description, check in self.checks.items():
                    assert check(inputs, variables), (
                        "Test %s::%s: Failed check: %s (inputs"
                        " were %s, outputs were %s)"
                    ) % (self.op, testname, description, inputs, variables)

        @pytest.mark.skipif(skip, reason="Skipped")
        def test_bad_build(self):
            for testname, inputs in self.bad_build.items():
                inputs = [copy(input) for input in inputs]
                inputrs = [shared(input) for input in inputs]
                with pytest.raises(Exception):
                    safe_make_node(self.op, *inputrs)
                # The old error string was ("Test %s::%s: %s was successfully
                # instantiated on the following bad inputs: %s"
                # % (self.op, testname, node, inputs))

        @change_flags(compute_test_value="off")
        @pytest.mark.skipif(skip, reason="Skipped")
        def test_bad_runtime(self):
            for testname, inputs in self.bad_runtime.items():
                inputrs = [shared(input) for input in inputs]
                try:
                    node = safe_make_node(self.op, *inputrs)
                except Exception as exc:
                    err_msg = (
                        "Test %s::%s: Error occurred while trying"
                        " to make a node with inputs %s"
                    ) % (self.op, testname, inputs)
                    exc.args += (err_msg,)
                    raise

                try:
                    f = inplace_func(
                        [], node.outputs, mode=mode, name="test_bad_runtime"
                    )
                except Exception as exc:
                    err_msg = (
                        "Test %s::%s: Error occurred while trying" " to make a Function"
                    ) % (self.op, testname)
                    exc.args += (err_msg,)
                    raise

                # Add tester return a ValueError. Should we catch only this
                # one?
                # TODO: test that only this one is raised and catch only this
                # one or the subset that get raised.
                with pytest.raises(Exception):
                    f([])

        @pytest.mark.skipif(skip, reason="Skipped")
        def test_grad(self):
            # Disable old warning that may be triggered by this test.
            backup = config.warn.sum_div_dimshuffle_bug
            config.warn.sum_div_dimshuffle_bug = False
            try:
                for testname, inputs in self.grad.items():
                    inputs = [copy(input) for input in inputs]
                    try:
                        utt.verify_grad(
                            self.op,
                            inputs,
                            mode=self.mode,
                            rel_tol=_grad_rtol,
                            eps=_grad_eps,
                        )
                    except Exception as exc:
                        err_msg = (
                            "Test %s::%s: Error occurred while"
                            " computing the gradient on the following"
                            " inputs: %s"
                        ) % (self.op, testname, inputs)
                        exc.args += (err_msg,)
                        raise
            finally:
                config.warn.sum_div_dimshuffle_bug = backup

        @pytest.mark.skipif(skip, reason="Skipped")
        def test_grad_none(self):
            # Check that None is never returned as input gradient
            # when calling self.op.grad
            # We use all values in self.good because this has to be true
            # whether or not the values work for utt.verify_grad.
            if not hasattr(self.op, "grad"):
                # This is not actually an Op
                return

            for testname, inputs in self.good.items():
                inputs = [copy(input) for input in inputs]
                inputrs = [
                    TensorType(
                        dtype=input.dtype,
                        broadcastable=[shape_elem == 1 for shape_elem in input.shape],
                    )()
                    for input in inputs
                ]

                if isinstance(self.expected, dict) and testname in self.expected:
                    expecteds = self.expected[testname]
                    # with numpy version, when we print a number and read it
                    # back, we don't get exactly the same result, so we accept
                    # rounding error in that case.
                else:
                    expecteds = self.expected(*inputs)
                if not isinstance(expecteds, (list, tuple)):
                    expecteds = (expecteds,)

                out_grad_vars = []
                for out in expecteds:
                    if str(out.dtype) in tensor.discrete_dtypes:
                        dtype = config.floatX
                    else:
                        dtype = str(out.dtype)
                    bcast = [shape_elem == 1 for shape_elem in out.shape]
                    var = TensorType(dtype=dtype, broadcastable=bcast)()
                    out_grad_vars.append(var)

                try:
                    in_grad_vars = self.op.grad(inputrs, out_grad_vars)
                except (gof.utils.MethodNotDefined, NotImplementedError):
                    pass
                else:
                    assert None not in in_grad_vars

    Checker.__name__ = name
    if hasattr(Checker, "__qualname__"):
        Checker.__qualname__ = name
    return Checker


def makeBroadcastTester(op, expected, checks=None, name=None, **kwargs):
    if checks is None:
        checks = {}
    if name is None:
        name = str(op)
    # Here we ensure the test name matches the name of the variable defined in
    # this script. This is needed to properly identify the test e.g. with the
    # --with-id option of nosetests, or simply to rerun a specific test that
    # failed.
    capitalize = False
    if name.startswith("Elemwise{") and name.endswith(",no_inplace}"):
        # For instance: Elemwise{add,no_inplace} -> Add
        name = name[9:-12]
        capitalize = True
    elif name.endswith("_inplace"):
        # For instance: sub_inplace -> SubInplace
        capitalize = True
    if capitalize:
        name = "".join([x.capitalize() for x in name.split("_")])
    # Some tests specify a name that already ends with 'Tester', while in other
    # cases we need to add it manually.
    if not name.endswith("Tester"):
        name += "Tester"
    if "inplace" in kwargs:
        if kwargs["inplace"]:
            _expected = expected
            if not isinstance(_expected, dict):

                def expected(*inputs):
                    return np.array(_expected(*inputs), dtype=inputs[0].dtype)

            def inplace_check(inputs, outputs):
                # this used to be inputs[0] is output[0]
                # I changed it so that it was easier to satisfy by the
                # DebugMode
                return np.all(inputs[0] == outputs[0])

            checks = dict(checks, inplace_check=inplace_check)
        del kwargs["inplace"]
    return makeTester(name, op, expected, checks, **kwargs)


# Those are corner case when rounding. Their is many rounding algo.
# c round() fct and numpy round are not the same!
corner_case = np.asarray(
    [-2.5, -2.0, -1.5, -1.0, -0.5, -0.51, -0.49, 0, 0.49, 0.5, 0.9, 1, 1.5, 2, 2.5],
    dtype=config.floatX,
)

# we remove 0 here as the grad is not always computable numerically.
corner_case_grad = np.asarray(
    [-2.5, -2.0, -1.5, -1.0, -0.5, -0.51, -0.49, 0.49, 0.5, 0.9, 1, 1.5, 2, 2.5],
    dtype=config.floatX,
)

_good_broadcast_unary_normal = dict(
    normal=[np.asarray(rand_ranged(-5, 5, (2, 3)), dtype=config.floatX)],
    integers=[randint_ranged(-5, 5, (2, 3))],
    # not using -128 because np.allclose would return False
    int8=[np.arange(-127, 128, dtype="int8")],
    uint8=[np.arange(0, 255, dtype="uint8")],
    uint16=[np.arange(0, 65535, dtype="uint16")],
    corner_case=[corner_case],
    complex=[randcomplex(2, 3)],
    empty=[np.asarray([], dtype=config.floatX)],
)

_grad_broadcast_unary_normal = dict(
    normal=[np.asarray(rand_ranged(-5, 5, (2, 3)), dtype=config.floatX)],
    corner_case=[corner_case_grad],
    # empty = [np.asarray([])] # XXX: should this be included?
)

_good_broadcast_unary_normal_float = dict(
    normal=[rand_ranged(-5, 5, (2, 3))],
    corner_case=[corner_case],
    complex=[randcomplex(2, 3)],
    empty=[np.asarray([], dtype=config.floatX)],
)

_good_broadcast_unary_normal_float_no_complex = copymod(
    _good_broadcast_unary_normal_float, without=["complex"]
)

_good_broadcast_unary_normal_float_no_complex_small_neg_range = dict(
    normal=[rand_ranged(-2, 5, (2, 3))],
    corner_case=[corner_case],
    empty=[np.asarray([], dtype=config.floatX)],
)

_grad_broadcast_unary_normal_small_neg_range = dict(
    normal=[np.asarray(rand_ranged(-2, 5, (2, 3)), dtype=config.floatX)],
    corner_case=[corner_case_grad],
)

_grad_broadcast_unary_abs1_no_complex = dict(
    normal=[np.asarray(rand_ranged(-1 + _eps, 1 - _eps, (2, 3)), dtype=config.floatX)],
)

_grad_broadcast_unary_0_2_no_complex = dict(
    # Don't go too close to 0 or 2 for tests in float32
    normal=[np.asarray(rand_ranged(_eps, 1 - _eps, (2, 3)), dtype=config.floatX)],
)

# chi2sf takes two inputs, a value (x) and a degrees of freedom (k).
# not sure how to deal with that here...

_good_broadcast_unary_chi2sf = dict(
    normal=(rand_ranged(1, 10, (2, 3)), np.asarray(1, dtype=config.floatX)),
    empty=(np.asarray([], dtype=config.floatX), np.asarray(1, dtype=config.floatX)),
    integers=(randint_ranged(1, 10, (2, 3)), np.asarray(1, dtype=config.floatX)),
    uint8=(
        randint_ranged(1, 10, (2, 3)).astype("uint8"),
        np.asarray(1, dtype=config.floatX),
    ),
    uint16=(
        randint_ranged(1, 10, (2, 3)).astype("uint16"),
        np.asarray(1, dtype=config.floatX),
    ),
)

_good_broadcast_unary_normal_no_complex = dict(
    normal=[np.asarray(rand_ranged(-5, 5, (2, 3)), dtype=config.floatX)],
    integers=[randint_ranged(-5, 5, (2, 3))],
    int8=[np.arange(-127, 128, dtype="int8")],
    uint8=[np.arange(0, 89, dtype="uint8")],
    uint16=[np.arange(0, 89, dtype="uint16")],
    corner_case=[corner_case],
    empty=[np.asarray([], dtype=config.floatX)],
    big_scalar=[np.arange(17.0, 29.0, 0.5, dtype=config.floatX)],
)

_bad_build_broadcast_binary_normal = dict()

_bad_runtime_broadcast_binary_normal = dict(
    bad_shapes=(rand(2, 3), rand(3, 2)), bad_row=(rand(2, 3), rand(1, 2))
)

_grad_broadcast_binary_normal = dict(
    same_shapes=(rand(2, 3), rand(2, 3)),
    scalar=(rand(2, 3), rand(1, 1)),
    row=(rand(2, 3), rand(1, 3)),
    column=(rand(2, 3), rand(2, 1)),
    # This don't work as verify grad don't support that
    # empty=(np.asarray([]), np.asarray([1]))
    # complex1=(randcomplex(2,3),randcomplex(2,3)),
    # complex2=(randcomplex(2,3),rand(2,3)),
    # Disabled as we test the case where we reuse the same output as the
    # first inputs.
    # complex3=(rand(2,3),randcomplex(2,3)),
)

_good_inv = dict(
    normal=[5 * rand_nonzero((2, 3))],
    integers=[randint_nonzero(2, 3)],
    int8=[np.array(list(range(-127, 0)) + list(range(1, 127)), dtype="int8")],
    uint8=[np.array(list(range(0, 255)), dtype="uint8")],
    uint16=[np.array(list(range(0, 65535)), dtype="uint16")],
    complex=[randcomplex_nonzero((2, 3))],
    empty=[np.asarray([], dtype=config.floatX)],
)

_good_inv_inplace = copymod(
    _good_inv, without=["integers", "int8", "uint8", "uint16", "complex"]
)
_grad_inv = copymod(
    _good_inv, without=["integers", "int8", "uint8", "uint16", "complex", "empty"]
)

_bad_runtime_inv = dict(
    float=[np.zeros((2, 3))],
    integers=[np.zeros((2, 3), dtype="int64")],
    int8=[np.zeros((2, 3), dtype="int8")],
    complex=[np.zeros((2, 3), dtype="complex128")],
)


_good_broadcast_pow_normal_float = dict(
    same_shapes=(rand_ranged(1, 5, (2, 3)), rand_ranged(-3, 3, (2, 3))),
    scalar=(rand_ranged(1, 5, (2, 3)), rand_ranged(-3, 3, (1, 1))),
    row=(rand_ranged(1, 5, (2, 3)), rand_ranged(-3, 3, (1, 3))),
    column=(rand_ranged(1, 5, (2, 3)), rand_ranged(-3, 3, (2, 1))),
    dtype_mixup=(rand_ranged(-3, 3, (2, 3)), randint_ranged(-3, 3, (2, 3))),
    complex1=(randcomplex(2, 3), randcomplex(2, 3)),
    complex2=(randcomplex(2, 3), rand(2, 3)),
    # complex3 = (rand(2,3),randcomplex(2,3)), # Inplace on the first element.
    empty1=(np.asarray([], dtype=config.floatX), np.asarray([1], dtype=config.floatX)),
    empty2=(np.asarray([0], dtype=config.floatX), np.asarray([], dtype=config.floatX)),
    empty3=(np.asarray([], dtype=config.floatX), np.asarray([], dtype=config.floatX)),
)
_grad_broadcast_pow_normal = dict(
    same_shapes=(rand_ranged(1, 5, (2, 3)), rand_ranged(-3, 3, (2, 3))),
    scalar=(rand_ranged(1, 5, (2, 3)), rand_ranged(-3, 3, (1, 1))),
    row=(rand_ranged(1, 5, (2, 3)), rand_ranged(-3, 3, (1, 3))),
    column=(rand_ranged(1, 5, (2, 3)), rand_ranged(-3, 3, (2, 1))),
    # complex1 = (randcomplex(2,3),randcomplex(2,3)),
    # complex2 = (randcomplex(2,3),rand(2,3)),
    # complex3 = (rand(2,3),randcomplex(2,3)),
    # empty1 = (np.asarray([]), np.asarray([1])),
    # empty2 = (np.asarray([0]), np.asarray([])),
    x_eq_zero=(
        np.asarray([0.0], dtype=config.floatX),
        np.asarray([2.0], dtype=config.floatX),
    ),  # Test for issue 1780
)
# empty2 case is not supported by numpy.
_good_broadcast_pow_normal_float_pow = copy(_good_broadcast_pow_normal_float)
del _good_broadcast_pow_normal_float_pow["empty2"]


_good_broadcast_unary_normal_float_no_empty = copymod(
    _good_broadcast_unary_normal_float, without=["empty"]
)

_good_broadcast_unary_normal_float_no_empty_no_complex = copymod(
    _good_broadcast_unary_normal_float_no_empty, without=["complex"]
)

_grad_broadcast_unary_normal_no_complex = dict(
    normal=[np.asarray(rand_ranged(-5, 5, (2, 3)), dtype=config.floatX)],
    corner_case=[corner_case_grad],
)

# Avoid epsilon around integer values
_grad_broadcast_unary_normal_noint = dict(
    normal=[(rand_ranged(_eps, 1 - _eps, (2, 3)) + randint(2, 3)).astype(config.floatX)]
)

_grad_broadcast_unary_normal_no_complex_no_corner_case = copymod(
    _grad_broadcast_unary_normal_no_complex, without=["corner_case"]
)

_good_broadcast_binary_arctan2 = dict(
    same_shapes=(rand(2, 3), rand(2, 3)),
    not_same_dimensions=(rand(2, 2), rand(2)),
    scalar=(rand(2, 3), rand(1, 1)),
    row=(rand(2, 3), rand(1, 3)),
    column=(rand(2, 3), rand(2, 1)),
    integers=(randint(2, 3), randint(2, 3)),
    int8=[
        np.arange(-127, 128, dtype="int8"),
        np.arange(-127, 128, dtype="int8")[:, np.newaxis],
    ],
    uint8=[
        np.arange(0, 128, dtype="uint8"),
        np.arange(0, 128, dtype="uint8")[:, np.newaxis],
    ],
    uint16=[
        np.arange(0, 128, dtype="uint16"),
        np.arange(0, 128, dtype="uint16")[:, np.newaxis],
    ],
    dtype_mixup_1=(rand(2, 3), randint(2, 3)),
    dtype_mixup_2=(randint(2, 3), rand(2, 3)),
    empty=(np.asarray([], dtype=config.floatX), np.asarray([1], dtype=config.floatX)),
)

_good_broadcast_unary_arccosh = dict(
    normal=(rand_ranged(1, 1000, (2, 3)),),
    integers=(randint_ranged(1, 1000, (2, 3)),),
    uint8=[np.arange(1, 256, dtype="uint8")],
    complex=(randc128_ranged(1, 1000, (2, 3)),),
    empty=(np.asarray([], dtype=config.floatX),),
)

_good_broadcast_unary_arctanh = dict(
    normal=(rand_ranged(-1 + _eps, 1 - _eps, (2, 3)),),
    integers=(randint_ranged(-1 + _eps, 1 - _eps, (2, 3)),),
    int8=[np.arange(0, 1, dtype="int8")],
    uint8=[np.arange(0, 1, dtype="uint8")],
    uint16=[np.arange(0, 1, dtype="uint16")],
    complex=(randc128_ranged(-1 + _eps, 1 - _eps, (2, 3)),),
    empty=(np.asarray([], dtype=config.floatX),),
)

_good_broadcast_unary_normal_abs = copy(_good_broadcast_unary_normal)
# Can't do inplace on Abs as the input/output are not of the same type!
del _good_broadcast_unary_normal_abs["complex"]


_good_broadcast_unary_positive = dict(
    normal=(rand_ranged(0.001, 5, (2, 3)),),
    integers=(randint_ranged(1, 5, (2, 3)),),
    uint8=[np.arange(1, 256, dtype="uint8")],
    complex=(randc128_ranged(1, 5, (2, 3)),),
    empty=(np.asarray([], dtype=config.floatX),),
)

_good_broadcast_unary_positive_float = copymod(
    _good_broadcast_unary_positive, without=["integers", "uint8"]
)

_good_broadcast_unary_tan = dict(
    normal=(rand_ranged(-3.14, 3.14, (2, 3)),),
    shifted=(rand_ranged(3.15, 6.28, (2, 3)),),
    integers=(randint_ranged(-3, 3, (2, 3)),),
    int8=[np.arange(-3, 4, dtype="int8")],
    uint8=[np.arange(0, 4, dtype="uint8")],
    uint16=[np.arange(0, 4, dtype="uint16")],
    complex=(randc128_ranged(-3.14, 3.14, (2, 3)),),
    empty=(np.asarray([], dtype=config.floatX),),
)

_good_broadcast_unary_wide = dict(
    normal=(rand_ranged(-1000, 1000, (2, 3)),),
    integers=(randint_ranged(-1000, 1000, (2, 3)),),
    int8=[np.arange(-127, 128, dtype="int8")],
    uint8=[np.arange(0, 255, dtype="uint8")],
    uint16=[np.arange(0, 65535, dtype="uint16")],
    complex=(randc128_ranged(-1000, 1000, (2, 3)),),
    empty=(np.asarray([], dtype=config.floatX),),
)
_good_broadcast_unary_wide_float = copymod(
    _good_broadcast_unary_wide, without=["integers", "int8", "uint8", "uint16"]
)

_good_broadcast_binary_normal = dict(
    same_shapes=(rand(2, 3), rand(2, 3)),
    not_same_dimensions=(rand(2, 2), rand(2)),
    scalar=(rand(2, 3), rand(1, 1)),
    row=(rand(2, 3), rand(1, 3)),
    column=(rand(2, 3), rand(2, 1)),
    integers=(randint(2, 3), randint(2, 3)),
    uint32=(randuint32(2, 3), randuint32(2, 3)),
    uint16=(randuint16(2, 3), randuint16(2, 3)),
    dtype_mixup_1=(rand(2, 3), randint(2, 3)),
    dtype_mixup_2=(randint(2, 3), rand(2, 3)),
    complex1=(randcomplex(2, 3), randcomplex(2, 3)),
    complex2=(randcomplex(2, 3), rand(2, 3)),
    # Disabled as we test the case where we reuse the same output as the
    # first inputs.
    # complex3=(rand(2,3),randcomplex(2,3)),
    empty=(np.asarray([], dtype=config.floatX), np.asarray([1], dtype=config.floatX)),
)

_good_broadcast_div_mod_normal_float_no_complex = dict(
    same_shapes=(rand(2, 3), rand_nonzero((2, 3))),
    scalar=(rand(2, 3), rand_nonzero((1, 1))),
    row=(rand(2, 3), rand_nonzero((1, 3))),
    column=(rand(2, 3), rand_nonzero((2, 1))),
    dtype_mixup_1=(rand(2, 3), randint_nonzero(2, 3)),
    dtype_mixup_2=(randint_nonzero(2, 3), rand_nonzero((2, 3))),
    integer=(randint(2, 3), randint_nonzero(2, 3)),
    uint8=(randint(2, 3).astype("uint8"), randint_nonzero(2, 3).astype("uint8")),
    uint16=(randint(2, 3).astype("uint16"), randint_nonzero(2, 3).astype("uint16")),
    int8=[
        np.tile(np.arange(-127, 128, dtype="int8"), [254, 1]).T,
        np.tile(
            np.array(list(range(-127, 0)) + list(range(1, 128)), dtype="int8"), [255, 1]
        ),
    ],
    # This empty2 doesn't work for some tests. I don't remember why
    # empty2=(np.asarray([0]), np.asarray([])),
)

_good_broadcast_div_mod_normal_float_inplace = copymod(
    _good_broadcast_div_mod_normal_float_no_complex,
    empty1=(np.asarray([]), np.asarray([1])),
    # No complex floor division in python 3.x
)

_good_broadcast_div_mod_normal_float = copymod(
    _good_broadcast_div_mod_normal_float_inplace,
    empty2=(np.asarray([0], dtype=config.floatX), np.asarray([], dtype=config.floatX)),
)

_good_broadcast_unary_arcsin = dict(
    normal=(rand_ranged(-1, 1, (2, 3)),),
    integers=(randint_ranged(-1, 1, (2, 3)),),
    int8=[np.arange(-1, 2, dtype="int8")],
    uint8=[np.arange(0, 2, dtype="uint8")],
    uint16=[np.arange(0, 2, dtype="uint16")],
    complex=(randc128_ranged(-1, 1, (2, 3)),),
    empty=(np.asarray([], dtype=config.floatX),),
)

_good_broadcast_unary_arcsin_float = copymod(
    _good_broadcast_unary_arcsin, without=["integers", "int8", "uint8", "uint16"]
)
