from functools import singledispatch
import ast

import numba
import numpy as np
import scipy.special

import aesara
from .naming import unique_name


class NumbaThunk:
    def __init__(self, call_ast, global_vars):
        self._call_ast = call_ast
        self._global_vars = global_vars

    @classmethod
    def from_func(cls, func, node, storage_map):
        func_name = unique_name(func.__name__)
        global_vars = {func_name: func}

        node_input_storage = [
            ast.Name(storage_map[r], ctx=ast.Load()) for r in node.inputs
        ]
        node_output_storage = [
            ast.Name(storage_map[r], ctx=ast.Store()) for r in node.outputs
        ]

        out_ast = [ast.Tuple(elts=node_output_storage, ctx=ast.Store())]
        call_ast = [
            ast.Assign(
                targets=out_ast,
                value=ast.Call(
                    func=ast.Name(func_name, ctx=ast.Load()),
                    args=node_input_storage,
                    keywords=[],
                ),
            )
        ]

        return NumbaThunk(global_vars=global_vars, call_ast=call_ast)

    @classmethod
    def from_func_one_output(cls, func, node, storage_map):
        func_name = unique_name(func.__name__)
        global_vars = {func_name: func}

        node_input_storage = [
            ast.Name(storage_map[r], ctx=ast.Load()) for r in node.inputs
        ]
        node_output_storage = [
            ast.Name(storage_map[r], ctx=ast.Store()) for r in node.outputs
        ]

        assert len(node.outputs) == 1

        out_ast = [node_output_storage[0]]
        call_ast = [
            ast.Assign(
                targets=out_ast,
                value=ast.Call(
                    func=ast.Name(func_name, ctx=ast.Load()),
                    args=node_input_storage,
                    keywords=[],
                ),
            )
        ]

        return NumbaThunk(global_vars=global_vars, call_ast=call_ast)

    @classmethod
    def from_func_modify_out(cls, func, node, storage_map):
        func_name = unique_name(func.__name__)
        global_vars = {func_name: func}

        node_input_storage = [
            ast.Name(storage_map[r], ctx=ast.Load()) for r in node.inputs + node.outputs
        ]

        call_ast = [
            ast.Call(
                func=ast.Name(func_name, ctx=ast.Load()),
                args=node_input_storage,
                keywords=[],
            )
        ]

        return NumbaThunk(global_vars=global_vars, call_ast=call_ast)


@singledispatch
def make_numba_thunk(op, node, storage_map):
    raise NotImplementedError(f"No numba implementation of {type(op).__name__}.")


@make_numba_thunk.register(aesara.scalar.basic.Add)
def make_numba_thunk_ScalarAdd(op, node, storage_map):
    @numba.njit
    def add(*args):
        result = 0
        for arg in args:
            result += arg
        return result

    return NumbaThunk.from_func_one_output(add, node, storage_map)


@make_numba_thunk.register(aesara.scalar.basic.Mul)
def make_numba_thunk_ScalarMul(op, node, storage_map):
    @numba.njit
    def mul(*args):
        result = 1
        for arg in args:
            result *= arg
        return result

    return NumbaThunk.from_func_one_output(mul, node, storage_map)


@make_numba_thunk.register(aesara.tensor.elemwise.Elemwise)
def make_numba_thunk_Elemwise(op, node, storage_map):

    if isinstance(node.op.scalar_op, aesara.scalar.basic.Composite):
        from .linker import compile_graph

        scalar_graph = node.op.scalar_op.fgraph
        scalar_func = compile_graph(scalar_graph)
    else:
        raise NotImplementedError()

    # TODO Need impl for arbitrary number of args
    @numba.vectorize
    def func(x, y):
        # TODO Always one output?
        return scalar_func(x, y)[0]

    @numba.njit
    def wrapper(x, y):
        return func(x, y)

    return NumbaThunk.from_func_one_output(wrapper, node, storage_map)


def _register_scalar_ops():
    _scalar_ops = {
        aesara.scalar.Abs: np.abs,
        aesara.scalar.Angle: np.angle,
        aesara.scalar.ArcCos: np.arccos,
        aesara.scalar.ArcCosh: np.arccosh,
        aesara.scalar.ArcSin: np.arcsin,
        aesara.scalar.ArcSinh: np.arcsinh,
        aesara.scalar.ArcTan: np.arctan,
        aesara.scalar.ArcTanh: np.arctan2,
        aesara.scalar.Ceil: np.ceil,
        aesara.scalar.Conj: np.conj,
        aesara.scalar.Cos: np.cos,
        aesara.scalar.Cosh: np.cosh,
        aesara.scalar.Deg2Rad: np.deg2rad,
        aesara.scalar.Erf: scipy.special.erf,
        aesara.scalar.Erfc: scipy.special.erfc,
        aesara.scalar.Erfcinv: scipy.special.erfcinv,
        aesara.scalar.Erfcx: scipy.special.erfcx,
        aesara.scalar.Erfinv: scipy.special.erfinv,
        aesara.scalar.Exp: np.exp,
        aesara.scalar.Exp2: np.exp2,
        aesara.scalar.Expm1: np.expm1,
        aesara.scalar.Floor: np.floor,
        aesara.scalar.Gamma: scipy.special.gamma,
        aesara.scalar.GammaLn: scipy.special.gammaln,
        aesara.scalar.I0: scipy.special.i0,
        aesara.scalar.I1: scipy.special.i1,
        aesara.scalar.Imag: np.imag,
        aesara.scalar.IsInf: np.isinf,
        aesara.scalar.IsNan: np.isnan,
        aesara.scalar.J0: scipy.special.j0,
        aesara.scalar.J1: scipy.special.j1,
        aesara.scalar.Log: np.log,
        aesara.scalar.Log10: np.log10,
        aesara.scalar.Log1p: np.log1p,
        aesara.scalar.Log2: np.log2,
        aesara.scalar.Psi: scipy.special.psi,
        aesara.scalar.Rad2Deg: np.rad2deg,
        aesara.scalar.Sgn: np.sign,
        aesara.scalar.Sin: np.sin,
        aesara.scalar.Sinh: np.sinh,
        aesara.scalar.Sqrt: np.sqrt,
        aesara.scalar.Tan: np.tan,
        aesara.scalar.Tanh: np.tanh,
        aesara.scalar.Trunc: np.trunc,
    }

    for op, np_func in _scalar_ops.items():

        @make_numba_thunk.register(op)
        def make_numba_thunk_ScalarOp(op, node, storage_map):

            # TODO This could just generate call ast directly
            @numba.njit
            def func(x):
                return np_func(x)

            return NumbaThunk.from_func_one_output(func, node, storage_map)


_register_scalar_ops()
