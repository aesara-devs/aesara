from functools import singledispatch
import ast

import numba

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
        
        node_input_storage = [ast.Name(storage_map[r], ctx=ast.Load()) for r in node.inputs]
        node_output_storage = [ast.Name(storage_map[r], ctx=ast.Store()) for r in node.outputs]

        out_ast = [ast.Tuple(elts=node_output_storage, ctx=ast.Store())]
        call_ast = [
            ast.Assign(
                targets=out_ast,
                value=ast.Call(
                    func=ast.Name(func_name, ctx=ast.Load()),
                    args=node_input_storage,
                    keywords=[]
                )
            )
        ]

        return NumbaThunk(global_vars=global_vars, call_ast=call_ast)
    
    @classmethod
    def from_func_one_output(cls, func, node, storage_map):
        func_name = unique_name(func.__name__)
        global_vars = {func_name: func}
        
        node_input_storage = [ast.Name(storage_map[r], ctx=ast.Load()) for r in node.inputs]
        node_output_storage = [ast.Name(storage_map[r], ctx=ast.Store()) for r in node.outputs]
        
        assert len(node.outputs) == 1

        out_ast = [node_output_storage[0]]
        call_ast = [
            ast.Assign(
                targets=out_ast,
                value=ast.Call(
                    func=ast.Name(func_name, ctx=ast.Load()),
                    args=node_input_storage,
                    keywords=[]
                )
            )
        ]

        return NumbaThunk(global_vars=global_vars, call_ast=call_ast)


    @classmethod
    def from_func_modify_out(cls, func, node, storage_map):
        func_name = unique_name(func.__name__)
        global_vars = {func_name: func}
        
        node_input_storage = [ast.Name(storage_map[r], ctx=ast.Load()) for r in node.inputs + node.outputs]

        call_ast = [
            ast.Call(
                func=ast.Name(func_name, ctx=ast.Load()),
                args=node_input_storage,
                keywords=[]
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
        from .numba_linker import compile_graph

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
