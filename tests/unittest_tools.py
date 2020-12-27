import logging
import sys
from copy import copy, deepcopy
from functools import wraps

import numpy as np
import pytest

import theano
import theano.tensor as tt
from theano.compile.debugmode import str_diagnostic
from theano.configdefaults import config


_logger = logging.getLogger("tests.unittest_tools")


def fetch_seed(pseed=None):
    """
    Returns the seed to use for running the unit tests.
    If an explicit seed is given, it will be used for seeding numpy's rng.
    If not, it will use config.unittest.rseed (its default value is 666).
    If config.unittest.rseed is set to "random", it will seed the rng with
    None, which is equivalent to seeding with a random seed.

    Useful for seeding RandomState objects.
    >>> rng = np.random.RandomState(unittest_tools.fetch_seed())
    """

    seed = pseed or config.unittests__rseed
    if seed == "random":
        seed = None

    try:
        if seed:
            seed = int(seed)
        else:
            seed = None
    except ValueError:
        print(
            (
                "Error: config.unittests__rseed contains "
                "invalid seed, using None instead"
            ),
            file=sys.stderr,
        )
        seed = None

    return seed


def seed_rng(pseed=None):
    """
    Seeds numpy's random number generator with the value returned by fetch_seed.
    Usage: unittest_tools.seed_rng()
    """

    seed = fetch_seed(pseed)
    if pseed and pseed != seed:
        print(
            "Warning: using seed given by config.unittests__rseed=%i"
            "instead of seed %i given as parameter" % (seed, pseed),
            file=sys.stderr,
        )
    np.random.seed(seed)
    return seed


def verify_grad(op, pt, n_tests=2, rng=None, *args, **kwargs):
    """
    Wrapper for gradient.py:verify_grad
    Takes care of seeding the random number generator if None is given
    """
    if rng is None:
        seed_rng()
        rng = np.random
    tt.verify_grad(op, pt, n_tests, rng, *args, **kwargs)


# A helpful class to check random values close to the boundaries
# when designing new tests
class MockRandomState:
    def __init__(self, val):
        self.val = val

    def rand(self, *shape):
        return np.zeros(shape, dtype="float64") + self.val

    def randint(self, minval, maxval=None, size=1):
        if maxval is None:
            minval, maxval = 0, minval
        out = np.zeros(size, dtype="int64")
        if self.val == 0:
            return out + minval
        else:
            return out + maxval - 1


class OptimizationTestMixin:
    def assertFunctionContains(self, f, op, min=1, max=sys.maxsize):
        toposort = f.maker.fgraph.toposort()
        matches = [node for node in toposort if node.op == op]
        assert min <= len(matches) <= max, (
            toposort,
            matches,
            str(op),
            len(matches),
            min,
            max,
        )

    def assertFunctionContains0(self, f, op):
        return self.assertFunctionContains(f, op, min=0, max=0)

    def assertFunctionContains1(self, f, op):
        return self.assertFunctionContains(f, op, min=1, max=1)

    def assertFunctionContainsN(self, f, op, N):
        return self.assertFunctionContains(f, op, min=N, max=N)

    def assertFunctionContainsClass(self, f, op, min=1, max=sys.maxsize):
        toposort = f.maker.fgraph.toposort()
        matches = [node for node in toposort if isinstance(node.op, op)]
        assert min <= len(matches) <= max, (
            toposort,
            matches,
            str(op),
            len(matches),
            min,
            max,
        )

    def assertFunctionContainsClassN(self, f, op, N):
        return self.assertFunctionContainsClass(f, op, min=N, max=N)


class OpContractTestMixin:
    # self.ops should be a list of instantiations of an Op class to test.
    # self.other_op should be an op which is different from every op
    other_op = tt.add

    def copy(self, x):
        return copy(x)

    def deepcopy(self, x):
        return deepcopy(x)

    def clone(self, op):
        raise NotImplementedError("return new instance like `op`")

    def test_eq(self):
        for i, op_i in enumerate(self.ops):
            assert op_i == op_i
            assert op_i == self.copy(op_i)
            assert op_i == self.deepcopy(op_i)
            assert op_i == self.clone(op_i)
            assert op_i != self.other_op
            for j, op_j in enumerate(self.ops):
                if i == j:
                    continue
                assert op_i != op_j

    def test_hash(self):
        for i, op_i in enumerate(self.ops):
            h_i = hash(op_i)
            assert h_i == hash(op_i)
            assert h_i == hash(self.copy(op_i))
            assert h_i == hash(self.deepcopy(op_i))
            assert h_i == hash(self.clone(op_i))
            assert h_i != hash(self.other_op)
            for j, op_j in enumerate(self.ops):
                if i == j:
                    continue
                assert op_i != hash(op_j)

    def test_name(self):
        for op in self.ops:
            s = str(op)  # show that str works
            assert s  # names should not be empty


class InferShapeTester:
    def setup_method(self):
        seed_rng()
        # Take into account any mode that may be defined in a child class
        # and it can be None
        mode = getattr(self, "mode", None)
        if mode is None:
            mode = theano.compile.get_default_mode()
        # This mode seems to be the minimal one including the shape_i
        # optimizations, if we don't want to enumerate them explicitly.
        self.mode = mode.including("canonicalize")

    def _compile_and_check(
        self,
        inputs,
        outputs,
        numeric_inputs,
        cls,
        excluding=None,
        warn=True,
        check_topo=True,
    ):
        """This tests the infer_shape method only

        When testing with input values with shapes that take the same
        value over different dimensions (for instance, a square
        matrix, or a tensor3 with shape (n, n, n), or (m, n, m)), it
        is not possible to detect if the output shape was computed
        correctly, or if some shapes with the same value have been
        mixed up. For instance, if the infer_shape uses the width of a
        matrix instead of its height, then testing with only square
        matrices will not detect the problem. If warn=True, we emit a
        warning when testing with such values.

        :param check_topo: If True, we check that the Op where removed
            from the graph. False is useful to test not implemented case.

        """
        mode = self.mode
        if excluding:
            mode = mode.excluding(*excluding)
        if warn:
            for var, inp in zip(inputs, numeric_inputs):
                if isinstance(inp, (int, float, list, tuple)):
                    inp = var.type.filter(inp)
                if not hasattr(inp, "shape"):
                    continue
                # remove broadcasted dims as it is sure they can't be
                # changed to prevent the same dim problem.
                if hasattr(var.type, "broadcastable"):
                    shp = [
                        inp.shape[i]
                        for i in range(inp.ndim)
                        if not var.type.broadcastable[i]
                    ]
                else:
                    shp = inp.shape
                if len(set(shp)) != len(shp):
                    _logger.warning(
                        "While testing shape inference for %r, we received an"
                        " input with a shape that has some repeated values: %r"
                        ", like a square matrix. This makes it impossible to"
                        " check if the values for these dimensions have been"
                        " correctly used, or if they have been mixed up.",
                        cls,
                        inp.shape,
                    )
                    break

        outputs_function = theano.function(inputs, outputs, mode=mode)
        shapes_function = theano.function(inputs, [o.shape for o in outputs], mode=mode)
        # theano.printing.debugprint(shapes_function)
        # Check that the Op is removed from the compiled function.
        if check_topo:
            topo_shape = shapes_function.maker.fgraph.toposort()
            assert not any(isinstance(t.op, cls) for t in topo_shape)
        topo_out = outputs_function.maker.fgraph.toposort()
        assert any(isinstance(t.op, cls) for t in topo_out)
        # Check that the shape produced agrees with the actual shape.
        numeric_outputs = outputs_function(*numeric_inputs)
        numeric_shapes = shapes_function(*numeric_inputs)
        for out, shape in zip(numeric_outputs, numeric_shapes):
            assert np.all(out.shape == shape), (out.shape, shape)


class WrongValue(Exception):
    def __init__(self, expected_val, val, rtol, atol):
        self.val1 = expected_val
        self.val2 = val
        self.rtol = rtol
        self.atol = atol

    def __str__(self):
        s = "WrongValue\n"
        return s + str_diagnostic(self.val1, self.val2, self.rtol, self.atol)


def assert_allclose(expected, value, rtol=None, atol=None):
    if not tt.basic._allclose(expected, value, rtol, atol):
        raise WrongValue(expected, value, rtol, atol)


class AttemptManyTimes:
    """Decorator for unit tests that forces a unit test to be attempted
    multiple times. The test needs to pass a certain number of times for it to
    be considered to have succeeded. If it doesn't pass enough times, it is
    considered to have failed.

    Warning : care should be exercised when using this decorator. For some
    tests, the fact that they fail randomly could point to important issues
    such as race conditions, usage of uninitialized memory region, etc. and
    using this decorator could hide these problems.

    Usage:
        @AttemptManyTimes(n_attempts=5, n_req_successes=3)
        def fct(args):
            ...
    """

    def __init__(self, n_attempts, n_req_successes=1):
        assert n_attempts >= n_req_successes
        self.n_attempts = n_attempts
        self.n_req_successes = n_req_successes

    def __call__(self, fct):

        # Wrap fct in a function that will attempt to run it multiple
        # times and return the result if the test passes enough times
        # of propagate the raised exception if it doesn't.
        @wraps(fct)
        def attempt_multiple_times(*args, **kwargs):

            # Keep a copy of the current seed for unittests so that we can use
            # a different seed for every run of the decorated test and restore
            # the original after
            original_seed = config.unittests__rseed
            current_seed = original_seed

            # If the decorator has received only one, unnamed, argument
            # and that argument has an attribute _testMethodName, it means
            # that the unit test on which the decorator is used is in a test
            # class. This means that the setup() method of that class will
            # need to be called before any attempts to execute the test in
            # case it relies on data randomly generated in the class' setup()
            # method.
            if len(args) == 1 and hasattr(args[0], "_testMethodName"):
                test_in_class = True
                class_instance = args[0]
            else:
                test_in_class = False

            n_fail = 0
            n_success = 0

            # Attempt to call the test function multiple times. If it does
            # raise any exception for at least one attempt, it passes. If it
            # raises an exception at every attempt, it fails.
            for i in range(self.n_attempts):
                try:
                    # Attempt to make the test use the current seed
                    config.unittests__rseed = current_seed
                    if test_in_class and hasattr(class_instance, "setUp"):
                        class_instance.setup_method()

                    fct(*args, **kwargs)

                    n_success += 1
                    if n_success == self.n_req_successes:
                        break

                except Exception:
                    n_fail += 1

                    # If there is not enough attempts remaining to achieve the
                    # required number of successes, propagate the original
                    # exception
                    if n_fail + self.n_req_successes > self.n_attempts:
                        raise

                finally:
                    # Clean up after the test
                    config.unittests__rseed = original_seed
                    if test_in_class and hasattr(class_instance, "teardown_method"):
                        class_instance.teardown_method()

                    # Update the current_seed
                    if current_seed not in [None, "random"]:
                        current_seed = str(int(current_seed) + 1)

        return attempt_multiple_times


def assertFailure_fast(f):
    """A Decorator to handle the test cases that are failing when
    THEANO_FLAGS =cycle_detection='fast'.
    """
    if theano.config.cycle_detection == "fast":

        def test_with_assert(*args, **kwargs):
            with pytest.raises(Exception):
                f(*args, **kwargs)

        return test_with_assert
    else:
        return f
