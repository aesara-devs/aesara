"""Symbolic Op for raising an exception."""

from textwrap import indent
from typing import Tuple

import numpy as np

from aesara.gradient import DisconnectedType
from aesara.graph.basic import Apply, Variable
from aesara.graph.op import COp
from aesara.graph.params_type import ParamsType
from aesara.graph.type import Generic


class ExceptionType(Generic):
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))


exception_type = ExceptionType()


class CheckAndRaise(COp):
    """An `Op` that checks conditions and raises an exception if they fail.

    This `Op` returns its "value" argument if its condition arguments are all
    ``True``; otherwise, it raises a user-specified exception.

    """

    _f16_ok = True
    __props__ = ("msg", "exc_type")
    view_map = {0: [0]}

    check_input = False
    params_type = ParamsType(exc_type=exception_type)

    def __init__(self, exc_type, msg=""):

        if not issubclass(exc_type, Exception):
            raise ValueError("`exc_type` must be an Exception subclass")

        self.exc_type = exc_type
        self.msg = msg

    def __str__(self):
        return f"CheckAndRaise{{{self.exc_type}({self.msg})}}"

    def __eq__(self, other):
        if type(self) != type(other):
            return False

        if self.msg == other.msg and self.exc_type == other.exc_type:
            return True

        return False

    def __hash__(self):
        return hash((self.msg, self.exc_type))

    def make_node(self, value: Variable, *conds: Tuple[Variable]):
        """

        Parameters
        ==========
        value
            The value to return if `conds` all evaluate to ``True``; otherwise,
            `self.exc_type` is raised.
        conds
            The conditions to evaluate.
        """
        import aesara.tensor as at

        if not isinstance(value, Variable):
            value = at.as_tensor_variable(value)

        conds = [at.as_tensor_variable(c) for c in conds]

        assert all(c.type.ndim == 0 for c in conds)

        return Apply(
            self,
            [value] + conds,
            [value.type()],
        )

    def perform(self, node, inputs, outputs, params):
        (out,) = outputs
        val, *conds = inputs
        out[0] = val
        if not np.all(conds):
            raise self.exc_type(self.msg)

    def grad(self, input, output_gradients):
        return output_gradients + [DisconnectedType()()] * (len(input) - 1)

    def connection_pattern(self, node):
        return [[1]] + [[0]] * (len(node.inputs) - 1)

    def c_code(self, node, name, inames, onames, props):
        value_name, *cond_names = inames
        out_name = onames[0]
        check = []
        fail_code = props["fail"]
        param_struct_name = props["params"]
        msg = self.msg.replace('"', '\\"').replace("\n", "\\n")
        for idx, cond_name in enumerate(cond_names):
            check.append(
                f"""
        if(PyObject_IsTrue((PyObject *){cond_name}) == 0) {{
            PyObject * exc_type = {param_struct_name}->exc_type;
            Py_INCREF(exc_type);
            PyErr_SetString(exc_type, "{msg}");
            Py_XDECREF(exc_type);
            {indent(fail_code, " " * 4)}
        }}
                """
            )
        check = "\n".join(check)
        res = f"""
        {check}
        Py_XDECREF({out_name});
        {out_name} = {value_name};
        Py_INCREF({value_name});
        """
        return res

    def c_code_cache_version(self):
        return (1, 0)

    def infer_shape(self, fgraph, node, input_shapes):
        return [input_shapes[0]]


class Assert(CheckAndRaise):
    """Implements assertion in a computational graph.

    Returns the first parameter if the condition is ``True``; otherwise,
    triggers `AssertionError`.

    Notes
    -----
    This `Op` is a debugging feature. It can be removed from the graph
    because of optimizations, and can hide some possible optimizations to
    the optimizer. Specifically, removing happens if it can be determined
    that condition will always be true. Also, the output of the Op must be
    used in the function computing the graph, but it doesn't have to be
    returned.

    Examples
    --------
    >>> import aesara
    >>> import aesara.tensor as aet
    >>> from aesara.raise_op import Assert
    >>> x = aet.vector("x")
    >>> assert_op = Assert("This assert failed")
    >>> func = aesara.function([x], assert_op(x, x.size < 2))

    """

    def __init__(self, msg="Aesara Assert failed!"):
        super().__init__(AssertionError, msg)

    def __str__(self):
        return f"Assert{{msg={self.msg}}}"


assert_op = Assert()
