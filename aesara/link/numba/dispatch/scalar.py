from functools import reduce
from typing import List

import numpy as np
import scipy
import scipy.special

from aesara import config
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

    global_env = {"scalar_func": scalar_func}

    input_tmp_dtypes = None
    if func_package == scipy and hasattr(scalar_func, "types"):
        # The `numba-scipy` bindings don't provide implementations for all
        # inputs types, so we need to convert the inputs to floats and back.
        inp_dtype_kinds = tuple(np.dtype(inp.type.dtype).kind for inp in node.inputs)
        accepted_inp_kinds = tuple(
            sig_type.split("->")[0] for sig_type in scalar_func.types
        )
        if not any(
            all(dk == ik for dk, ik in zip(inp_dtype_kinds, ok_kinds))
            for ok_kinds in accepted_inp_kinds
        ):
            # They're usually ordered from lower-to-higher precision, so
            # we pick the last acceptable input types
            #
            # XXX: We should pick the first acceptable float/int types in
            # reverse, excluding all the incompatible ones (e.g. `"0"`).
            # The assumption is that this is only used by `numba-scipy`-exposed
            # functions, although it's possible for this to be triggered by
            # something else from the `scipy` package
            input_tmp_dtypes = tuple(np.dtype(k) for k in accepted_inp_kinds[-1])

    if input_tmp_dtypes is None:
        unique_names = unique_name_generator(
            [scalar_op_fn_name, "scalar_func"], suffix_sep="_"
        )
        input_names = ", ".join(
            [unique_names(v, force_unique=True) for v in node.inputs]
        )
        scalar_op_src = f"""
def {scalar_op_fn_name}({input_names}):
    return scalar_func({input_names})
        """
    else:
        global_env["direct_cast"] = numba_basic.direct_cast
        global_env["output_dtype"] = np.dtype(node.outputs[0].type.dtype)
        input_tmp_dtype_names = {
            f"inp_tmp_dtype_{i}": i_dtype for i, i_dtype in enumerate(input_tmp_dtypes)
        }
        global_env.update(input_tmp_dtype_names)

        unique_names = unique_name_generator(
            [scalar_op_fn_name, "scalar_func"] + list(global_env.keys()), suffix_sep="_"
        )

        input_names = [unique_names(v, force_unique=True) for v in node.inputs]
        converted_call_args = ", ".join(
            [
                f"direct_cast({i_name}, {i_tmp_dtype_name})"
                for i_name, i_tmp_dtype_name in zip(
                    input_names, input_tmp_dtype_names.keys()
                )
            ]
        )
        scalar_op_src = f"""
def {scalar_op_fn_name}({', '.join(input_names)}):
    return direct_cast(scalar_func({converted_call_args}), output_dtype)
        """

    scalar_op_fn = compile_function_src(
        scalar_op_src, scalar_op_fn_name, {**globals(), **global_env}
    )

    signature = create_numba_signature(node, force_scalar=True)

    return numba_basic.numba_njit(
        signature, inline="always", fastmath=config.numba__fastmath
    )(scalar_op_fn)


@numba_funcify.register(Switch)
def numba_funcify_Switch(op, node, **kwargs):
    @numba_basic.numba_njit(inline="always")
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
    nary_fn = compile_function_src(nary_src, binary_op_name, globals())

    return nary_fn


@numba_funcify.register(Add)
def numba_funcify_Add(op, node, **kwargs):

    signature = create_numba_signature(node, force_scalar=True)

    nary_add_fn = binary_to_nary_func(node.inputs, "add", "+")

    return numba_basic.numba_njit(
        signature, inline="always", fastmath=config.numba__fastmath
    )(nary_add_fn)


@numba_funcify.register(Mul)
def numba_funcify_Mul(op, node, **kwargs):

    signature = create_numba_signature(node, force_scalar=True)

    nary_mul_fn = binary_to_nary_func(node.inputs, "mul", "*")

    return numba_basic.numba_njit(
        signature, inline="always", fastmath=config.numba__fastmath
    )(nary_mul_fn)


@numba_funcify.register(Cast)
def numba_funcify_Cast(op, node, **kwargs):

    dtype = np.dtype(op.o_type.dtype)

    @numba_basic.numba_njit(inline="always")
    def cast(x):
        return numba_basic.direct_cast(x, dtype)

    return cast


@numba_funcify.register(Identity)
@numba_funcify.register(ViewOp)
def numba_funcify_ViewOp(op, **kwargs):
    @numba_basic.numba_njit(inline="always")
    def viewop(x):
        return x

    return viewop


@numba_funcify.register(Clip)
def numba_funcify_Clip(op, **kwargs):
    @numba_basic.numba_njit
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
    composite_fn = numba_basic.numba_njit(signature, fastmath=config.numba__fastmath)(
        numba_funcify(op.fgraph, squeeze_output=True, **kwargs)
    )
    return composite_fn


@numba_funcify.register(Second)
def numba_funcify_Second(op, node, **kwargs):
    @numba_basic.numba_njit(inline="always")
    def second(x, y):
        return y

    return second


@numba_funcify.register(Inv)
def numba_funcify_Inv(op, node, **kwargs):
    @numba_basic.numba_njit(inline="always")
    def inv(x):
        return 1 / x

    return inv
