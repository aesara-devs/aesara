import numpy as np

from aesara.gradient import DisconnectedType
from aesara.graph.basic import Apply, Variable
from aesara.graph.op import COp


class Assert(COp):
    """
    Implements assertion in a computational graph.

    Returns the first parameter if the condition is true, otherwise, triggers
    AssertionError.

    Notes
    -----
    This Op is a debugging feature. It can be removed from the graph
    because of optimizations, and can hide some possible optimizations to
    the optimizer. Specifically, removing happens if it can be determined
    that condition will always be true. Also, the output of the Op must be
    used in the function computing the graph, but it doesn't have to be
    returned.

    Examples
    --------
    >>> import aesara
    >>> import aesara.tensor as aet
    >>> from aesara.assert_op import Assert
    >>> x = aet.vector("x")
    >>> assert_op = Assert("This assert failed")
    >>> func = aesara.function([x], assert_op(x, x.size < 2))

    """

    _f16_ok = True
    __props__ = ("msg",)
    view_map = {0: [0]}

    check_input = False

    def __init__(self, msg="Aesara Assert failed!"):
        self.msg = msg

    def __setstate__(self, attrs):
        self.__dict__.update(attrs)
        if not hasattr(self, "msg"):
            self.msg = "Aesara Assert failed!"

    def make_node(self, value, *conds):
        from aesara.tensor import as_tensor_variable

        if not isinstance(value, Variable):
            value = as_tensor_variable(value)
        cond = [as_tensor_variable(c) for c in conds]
        assert np.all([c.type.ndim == 0 for c in cond])
        return Apply(self, [value] + cond, [value.type()])

    def perform(self, node, inputs, out_):
        (out,) = out_
        v = inputs[0]
        out[0] = v
        assert np.all(inputs[1:]), self.msg

    def grad(self, input, output_gradients):
        return output_gradients + [DisconnectedType()()] * (len(input) - 1)

    def connection_pattern(self, node):
        return [[1]] + [[0]] * (len(node.inputs) - 1)

    def c_code(self, node, name, inames, onames, props):
        value = inames[0]
        out = onames[0]
        check = []
        fail = props["fail"]
        msg = self.msg.replace('"', '\\"').replace("\n", "\\n")
        for idx in range(len(inames) - 1):
            i = inames[idx + 1]
            dtype = node.inputs[idx + 1].dtype
            check.append(
                "if(!((npy_%(dtype)s*)PyArray_DATA(%(i)s))[0])"
                '{PyErr_SetString(PyExc_AssertionError,"%(msg)s");'
                "%(fail)s}" % locals()
            )
        check = "\n".join(check)
        return f"""
        {check}
        Py_XDECREF({out});
        {out} = {value};
        Py_INCREF({value});
        """

    def c_code_cache_version(self):
        return (3, 0)

    def infer_shape(self, fgraph, node, input_shapes):
        return [input_shapes[0]]


assert_op = Assert()
