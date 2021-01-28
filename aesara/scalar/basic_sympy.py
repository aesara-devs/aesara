import itertools as it

from aesara.scalar.basic import Apply, ScalarOp, as_scalar, float32, float64, int64


imported_sympy = False
try:
    from sympy.utilities.codegen import codegen, get_default_datatype

    imported_sympy = True
except ImportError:
    pass

names = (f"sympy_func_{int(i)}" for i in it.count(0))


def include_line(line):
    return "#include" in line


def sympy_dtype(expr):
    return get_default_datatype(expr).cname


def aesara_dtype(expr):
    return {"double": float64, "float": float32, "int": int64}[sympy_dtype(expr)]


class SymPyCCode(ScalarOp):
    """
    An Operator that wraps SymPy's C code generation.

    Examples
    --------
    >>> from sympy.abc import x, y  # SymPy Variables
    >>> from aesara.scalar.basic_sympy import SymPyCCode
    >>> op = SymPyCCode([x, y], x + y)

    >>> from aesara.scalar.basic import floats
    >>> xt, yt = floats('xy') # Aesara variables
    >>> zt = op(xt, yt)

    >>> import aesara
    >>> f = aesara.function([xt, yt], zt)
    >>> f(1.0, 2.0)
    3.0

    """

    def __init__(self, inputs, expr, name=None):
        self.name = name or next(names)
        self.inputs = inputs
        self.expr = expr

    def _sympy_c_code(self):
        [(c_name, c_code), (h_name, c_header)] = codegen(
            (self.name, self.expr),
            "C",
            "project_name",
            header=False,
            argument_sequence=self.inputs,
        )
        return c_code

    def c_support_code(self, **kwargs):
        c_code = self._sympy_c_code()
        return "\n".join([x for x in c_code.split("\n") if not include_line(x)])

    def c_headers(self, **kwargs):
        c_code = self._sympy_c_code()
        return [
            line.replace("#include", "").strip()
            for line in c_code.split("\n")
            if include_line(line) and "project_name" not in line
        ]

    def c_code(self, node, name, input_names, output_names, sub):
        (y,) = output_names
        xs = ", ".join(input_names)
        f = self.name
        return f"{y} = {f}({xs});"

    def output_types_preference(self, *inputs):
        return [aesara_dtype(self.expr)]

    def make_node(self, *inputs):
        # TODO: assert input types are correct use get_default_datatype

        if len(inputs) != len(self.inputs):
            raise TypeError(
                "Wrong number of inputs for %s.make_node (got %i(%s), expected %i)"
                % (self, len(inputs), str(inputs), self.nin)
            )

        inputs = [as_scalar(input) for input in inputs]
        outputs = [t() for t in self.output_types([input.type for input in inputs])]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, output_storage):
        raise NotImplementedError()

    def grad(self, inputs, output_grads):
        return [
            SymPyCCode(
                self.inputs, self.expr.diff(inp), name=self.name + f"_grad_{int(i)}"
            )(*inputs)
            for i, inp in enumerate(self.inputs)
        ]

    def _info(self):
        return type(self), self.name, tuple(self.inputs), self.expr

    def __eq__(self, other):
        return type(self) == type(other) and self._info() == other._info()

    def __hash__(self):
        return hash(self._info())
