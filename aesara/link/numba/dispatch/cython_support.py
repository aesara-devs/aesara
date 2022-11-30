import ctypes
import importlib
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, cast

import numba
import numpy as np
from numpy.typing import DTypeLike
from scipy import LowLevelCallable


_C_TO_NUMPY: Dict[str, DTypeLike] = {
    "bool": np.bool_,
    "signed char": np.byte,
    "unsigned char": np.ubyte,
    "short": np.short,
    "unsigned short": np.ushort,
    "int": np.intc,
    "unsigned int": np.uintc,
    "long": np.int_,
    "unsigned long": np.uint,
    "long long": np.longlong,
    "float": np.single,
    "double": np.double,
    "long double": np.longdouble,
    "float complex": np.csingle,
    "double complex": np.cdouble,
}


@dataclass
class Signature:
    res_dtype: DTypeLike
    res_c_type: str
    arg_dtypes: List[DTypeLike]
    arg_c_types: List[str]
    arg_names: List[Optional[str]]

    @property
    def arg_numba_types(self) -> List[DTypeLike]:
        return [numba.from_dtype(dtype) for dtype in self.arg_dtypes]

    def can_cast_args(self, args: List[DTypeLike]) -> bool:
        ok = True
        count = 0
        for name, dtype in zip(self.arg_names, self.arg_dtypes):
            if name == "__pyx_skip_dispatch":
                continue
            if len(args) <= count:
                raise ValueError("Incorrect number of arguments")
            ok &= np.can_cast(args[count], dtype)
            count += 1
        if count != len(args):
            return False
        return ok

    def provides(self, restype: DTypeLike, arg_dtypes: List[DTypeLike]) -> bool:
        args_ok = self.can_cast_args(arg_dtypes)
        if np.issubdtype(restype, np.inexact):
            result_ok = np.can_cast(self.res_dtype, restype, casting="same_kind")
            # We do not want to provide less accuracy than advertised
            result_ok &= np.dtype(self.res_dtype).itemsize >= np.dtype(restype).itemsize
        else:
            result_ok = np.can_cast(self.res_dtype, restype)
        return args_ok and result_ok

    @staticmethod
    def from_c_types(signature: bytes) -> "Signature":
        # Match strings like "double(int, double)"
        # and extract the return type and the joined arguments
        expr = re.compile(rb"\s*(?P<restype>[\w ]*\w+)\s*\((?P<args>[\w\s,]*)\)")
        re_match = re.fullmatch(expr, signature)

        if re_match is None:
            raise ValueError(f"Invalid signature: {signature.decode()}")

        groups = re_match.groupdict()
        res_c_type = groups["restype"].decode()
        res_dtype: DTypeLike = _C_TO_NUMPY[res_c_type]

        raw_args = groups["args"]

        decl_expr = re.compile(
            rb"\s*(?P<type>((long )|(unsigned )|(signed )|(double )|)"
            rb"((double)|(float)|(int)|(short)|(char)|(long)|(bool)|(complex)))"
            rb"(\s(?P<name>[\w_]*))?\s*"
        )

        arg_dtypes = []
        arg_names: List[Optional[str]] = []
        arg_c_types = []
        for raw_arg in raw_args.split(b","):
            re_match = re.fullmatch(decl_expr, raw_arg)
            if re_match is None:
                raise ValueError(f"Invalid signature: {signature.decode()}")
            groups = re_match.groupdict()
            arg_c_type = groups["type"].decode()
            try:
                arg_dtype = _C_TO_NUMPY[arg_c_type]
            except KeyError:
                raise ValueError(f"Unknown C type: {arg_c_type}")

            arg_c_types.append(arg_c_type)
            arg_dtypes.append(arg_dtype)
            name = groups["name"]
            if not name:
                arg_names.append(None)
            else:
                arg_names.append(name.decode())

        return Signature(res_dtype, res_c_type, arg_dtypes, arg_c_types, arg_names)


def _available_impls(func: Callable) -> List[Tuple[Signature, Any]]:
    """Find all available implementations for a fused cython function."""
    impls = []
    mod = importlib.import_module(func.__module__)

    signatures = getattr(func, "__signatures__", None)
    if signatures is not None:
        # Cython function with __signatures__ should be fused and thus
        # indexable
        func_map = cast(Mapping, func)
        candidates = [func_map[key] for key in signatures]
    else:
        candidates = [func]
    for candidate in candidates:
        name = candidate.__name__
        capsule = mod.__pyx_capi__[name]
        llc = LowLevelCallable(capsule)
        try:
            signature = Signature.from_c_types(llc.signature.encode())
        except KeyError:
            continue
        impls.append((signature, capsule))
    return impls


class _CythonWrapper(numba.types.WrapperAddressProtocol):
    def __init__(self, pyfunc, signature, capsule):
        self._keep_alive = capsule
        get_name = ctypes.pythonapi.PyCapsule_GetName
        get_name.restype = ctypes.c_char_p
        get_name.argtypes = (ctypes.py_object,)

        raw_signature = get_name(capsule)

        get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
        get_pointer.restype = ctypes.c_void_p
        get_pointer.argtypes = (ctypes.py_object, ctypes.c_char_p)
        self._func_ptr = get_pointer(capsule, raw_signature)

        self._signature = signature
        self._pyfunc = pyfunc

    def signature(self):
        return numba.from_dtype(self._signature.res_dtype)(
            *self._signature.arg_numba_types
        )

    def __wrapper_address__(self):
        return self._func_ptr

    def __call__(self, *args, **kwargs):
        args = [dtype(arg) for arg, dtype in zip(args, self._signature.arg_dtypes)]
        if self.has_pyx_skip_dispatch():
            output = self._pyfunc(*args[:-1], **kwargs)
        else:
            output = self._pyfunc(*args, **kwargs)
        return self._signature.res_dtype(output)

    def has_pyx_skip_dispatch(self):
        if not self._signature.arg_names:
            return False
        if any(
            name == "__pyx_skip_dispatch" for name in self._signature.arg_names[:-1]
        ):
            raise ValueError("skip_dispatch parameter must be last")
        return self._signature.arg_names[-1] == "__pyx_skip_dispatch"

    def numpy_arg_dtypes(self):
        return self._signature.arg_dtypes

    def numpy_output_dtype(self):
        return self._signature.res_dtype


def wrap_cython_function(func, restype, arg_types):
    impls = _available_impls(func)
    compatible = []
    for sig, capsule in impls:
        if sig.provides(restype, arg_types):
            compatible.append((sig, capsule))

    def sort_key(args):
        sig, _ = args

        # Prefer functions with less inputs bytes
        argsize = sum(np.dtype(dtype).itemsize for dtype in sig.arg_dtypes)

        # Prefer functions with more exact (integer) arguments
        num_inexact = sum(np.issubdtype(dtype, np.inexact) for dtype in sig.arg_dtypes)
        return (num_inexact, argsize)

    compatible.sort(key=sort_key)

    if not compatible:
        raise NotImplementedError(f"Could not find a compatible impl of {func}")
    sig, capsule = compatible[0]
    return _CythonWrapper(func, sig, capsule)
