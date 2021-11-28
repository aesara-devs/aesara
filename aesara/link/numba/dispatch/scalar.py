from functools import reduce
from typing import List

import numba
import numpy as np
import scipy
import scipy.special

from aesara.compile.ops import ViewOp
from aesara.graph.basic import Variable
from aesara.link.numba.dispatch import basic as numba_basic
from aesara.link.numba.dispatch.basic import create_numba_signature, numba_funcify
from aesara.link.utils import (
    compile_function_src,
    get_name_for_object,
    unique_name_generator,
)
from aesara.scalar.basic import (
    Add,
    Cast,
    Clip,
    Composite,
    Identity,
    Inv,
    Mul,
    ScalarOp,
    Second,
    Switch,
)


@numba_funcify.register(ScalarOp)
def numba_funcify_ScalarOp(op, node, **kwargs):
    # TODO: Do we need to cache these functions so that we don't end up
    # compiling the same Numba function over and over again?

    scalar_func_name = op.nfunc_spec[0]

    if scalar_func_name.startswith("scipy."):
        func_package = scipy
        scalar_func_name = scalar_func_name.split(".", 1)[-1]
    else:
        func_package = np

    if "." in scalar_func_name:
        scalar_func = reduce(getattr, [scipy] + scalar_func_name.split("."))
    else:
        scalar_func = getattr(func_package, scalar_func_name)

    scalar_op_fn_name = get_name_for_object(scalar_func)
    unique_names = unique_name_generator(
        [scalar_op_fn_name, "scalar_func"], suffix_sep="_"
    )

    input_names = ", ".join([unique_names(v, force_unique=True) for v in node.inputs])

    global_env = {"scalar_func": scalar_func}

    scalar_op_src = f"""
def {scalar_op_fn_name}({input_names}):
    return scalar_func({input_names})
    """
    scalar_op_fn = compile_function_src(scalar_op_src, scalar_op_fn_name, global_env)

    signature = create_numba_signature(node, force_scalar=True)

    return numba.njit(signature, inline="always")(scalar_op_fn)


@numba_funcify.register(Switch)
def numba_funcify_Switch(op, node, **kwargs):
    @numba.njit(inline="always")
    def switch(condition, x, y):
        if condition:
            return x
        else:
            return y

    return switch


def binary_to_nary_func(inputs: List[Variable], binary_op_name: str, binary_op: str):
    """Create a Numba-compatible N-ary function from a binary function."""
    unique_names = unique_name_generator(["binary_op_name"], suffix_sep="_")
    input_names = [unique_names(v, force_unique=True) for v in inputs]
    input_signature = ", ".join(input_names)
    output_expr = binary_op.join(input_names)

    nary_src = f"""
def {binary_op_name}({input_signature}):
    return {output_expr}
    """
    nary_fn = compile_function_src(nary_src, binary_op_name)

    return nary_fn


@numba_funcify.register(Add)
def numba_funcify_Add(op, node, **kwargs):

    signature = create_numba_signature(node, force_scalar=True)

    nary_add_fn = binary_to_nary_func(node.inputs, "add", "+")

    return numba.njit(signature, inline="always")(nary_add_fn)


@numba_funcify.register(Mul)
def numba_funcify_Mul(op, node, **kwargs):

    signature = create_numba_signature(node, force_scalar=True)

    nary_mul_fn = binary_to_nary_func(node.inputs, "mul", "*")

    return numba.njit(signature, inline="always")(nary_mul_fn)


@numba_funcify.register(Cast)
def numba_funcify_Cast(op, node, **kwargs):

    dtype = np.dtype(op.o_type.dtype)

    @numba.njit(inline="always")
    def cast(x):
        return numba_basic.direct_cast(x, dtype)

    return cast


@numba_funcify.register(Identity)
@numba_funcify.register(ViewOp)
def numba_funcify_ViewOp(op, **kwargs):
    @numba.njit(inline="always")
    def viewop(x):
        return x

    return viewop


@numba_funcify.register(Clip)
def numba_funcify_Clip(op, **kwargs):
    @numba.njit
    def clip(_x, _min, _max):
        x = numba_basic.to_scalar(_x)
        _min_scalar = numba_basic.to_scalar(_min)
        _max_scalar = numba_basic.to_scalar(_max)

        if x < _min_scalar:
            return _min_scalar
        elif x > _max_scalar:
            return _max_scalar
        else:
            return x

    return clip


@numba_funcify.register(Composite)
def numba_funcify_Composite(op, node, **kwargs):
    signature = create_numba_signature(node, force_scalar=True)
    composite_fn = numba.njit(signature)(
        numba_funcify(op.fgraph, squeeze_output=True, **kwargs)
    )
    return composite_fn


@numba_funcify.register(Second)
def numba_funcify_Second(op, node, **kwargs):
    @numba.njit(inline="always")
    def second(x, y):
        return y

    return second


@numba_funcify.register(Inv)
def numba_funcify_Inv(op, node, **kwargs):
    @numba.njit(inline="always")
    def inv(x):
        return 1 / x

    return inv
