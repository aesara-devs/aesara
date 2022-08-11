import inspect
import os
import re
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Collection,
    Dict,
    List,
    Optional,
    Pattern,
    Set,
    Tuple,
    Union,
    cast,
)

import numpy as np

from aesara.configdefaults import config
from aesara.graph.basic import Apply, Variable
from aesara.graph.op import ComputeMapType, Op, StorageMapType, ThunkType
from aesara.graph.type import HasDataType
from aesara.graph.utils import MethodNotDefined
from aesara.link.c.interface import CLinkerOp
from aesara.link.c.params_type import ParamsType
from aesara.utils import hash_from_code


if TYPE_CHECKING:
    from aesara.link.c.basic import _CThunk


class CThunkWrapperType(ThunkType):
    thunk: "_CThunk"
    cthunk: ThunkType


def is_cthunk_wrapper_type(thunk: Callable[[], None]) -> CThunkWrapperType:
    res = cast(CThunkWrapperType, thunk)
    return res


class COp(Op, CLinkerOp):
    """An `Op` with a C implementation."""

    def make_c_thunk(
        self,
        node: Apply,
        storage_map: StorageMapType,
        compute_map: ComputeMapType,
        no_recycling: Collection[Variable],
    ) -> CThunkWrapperType:
        """Create a thunk for a C implementation.

        Like :meth:`Op.make_thunk`, but will only try to make a C thunk.

        """
        # FIXME: Putting the following import on the module level causes an import cycle.
        #        The conclusion should be that the antire "make_c_thunk" method should be defined
        #        in aesara.link.c and dispatched onto the Op!
        import aesara.link.c.basic
        from aesara.graph.fg import FunctionGraph

        node_input_storage = [storage_map[r] for r in node.inputs]
        node_output_storage = [storage_map[r] for r in node.outputs]

        e = FunctionGraph(node.inputs, node.outputs)
        e_no_recycling = [
            new_o
            for (new_o, old_o) in zip(e.outputs, node.outputs)
            if old_o in no_recycling
        ]
        cl = aesara.link.c.basic.CLinker().accept(e, no_recycling=e_no_recycling)
        # float16 gets special treatment since running
        # unprepared C code will get bad results.
        if not getattr(self, "_f16_ok", False):

            def is_f16(t):
                return getattr(t, "dtype", "") == "float16"

            if any(is_f16(i.type) for i in node.inputs) or any(
                is_f16(o.type) for o in node.outputs
            ):
                # get_dynamic_module is a subset of make_thunk that is reused.
                # This just try to build the c code
                # It will raise an error for ops
                # that don't implement c code. In those cases, we
                # don't want to print a warning.
                cl.get_dynamic_module()
                print(f"Disabling C code for {self} due to unsupported float16")
                raise NotImplementedError("float16")
        outputs = cl.make_thunk(
            input_storage=node_input_storage, output_storage=node_output_storage
        )
        thunk, node_input_filters, node_output_filters = outputs

        @is_cthunk_wrapper_type
        def rval():
            thunk()
            for o in node.outputs:
                compute_map[o][0] = True

        rval.thunk = thunk
        rval.cthunk = thunk.cthunk
        rval.inputs = node_input_storage
        rval.outputs = node_output_storage
        rval.lazy = False
        return rval

    def make_thunk(self, node, storage_map, compute_map, no_recycling, impl=None):
        """Create a thunk.

        See :meth:`Op.make_thunk`.

        Parameters
        ----------
        impl :
            Currently, ``None``, ``'c'`` or ``'py'``. If ``'c'`` or ``'py'`` we
            will only try that version of the code.

        """
        if (impl is None and config.cxx) or impl == "c":
            self.prepare_node(
                node, storage_map=storage_map, compute_map=compute_map, impl="c"
            )
            try:
                return self.make_c_thunk(node, storage_map, compute_map, no_recycling)
            except (NotImplementedError, MethodNotDefined):
                # We requested the c code, so don't catch the error.
                if impl == "c":
                    raise

        return super().make_thunk(
            node, storage_map, compute_map, no_recycling, impl=impl
        )


class OpenMPOp(COp):
    r"""Base class for `Op`\s using OpenMP.

    This `Op` will check that the compiler support correctly OpenMP code.
    If not, it will print a warning and disable OpenMP for this `Op`, then it
    will generate the not OpenMP code.

    This is needed, as EPD on the Windows version of ``g++`` says it supports
    OpenMP, but does not include the OpenMP files.

    We also add the correct compiler flags in ``c_compile_args``.

    """

    gxx_support_openmp: Optional[bool] = None
    """
    ``True``/``False`` after we tested this.

    """

    def __init__(self, openmp: Optional[bool] = None):
        if openmp is None:
            openmp = config.openmp
        self.openmp = openmp

    def __setstate__(self, d: Dict):
        self.__dict__.update(d)
        # If we unpickle old op
        if not hasattr(self, "openmp"):
            self.openmp = False

    def c_compile_args(self, **kwargs):
        """Return the compilation argument ``"-fopenmp"`` if OpenMP is supported."""
        self.update_self_openmp()
        if self.openmp:
            return ["-fopenmp"]
        return []

    def c_headers(self, **kwargs):
        """Return the header file name ``"omp.h"`` if OpenMP is supported."""
        self.update_self_openmp()
        if self.openmp:
            return ["omp.h"]
        return []

    @staticmethod
    def test_gxx_support():
        """Check if OpenMP is supported."""
        from aesara.link.c.cmodule import GCC_compiler

        code = """
        #include <omp.h>
int main( int argc, const char* argv[] )
{
        int res[10];

        for(int i=0; i < 10; i++){
            res[i] = i;
        }
}
        """
        default_openmp = GCC_compiler.try_compile_tmp(
            src_code=code, tmp_prefix="test_omp_", flags=["-fopenmp"], try_run=False
        )
        return default_openmp

    def update_self_openmp(self) -> None:
        """Make sure ``self.openmp`` is not ``True`` if there is no OpenMP support in ``gxx``."""
        if self.openmp:
            if OpenMPOp.gxx_support_openmp is None:
                OpenMPOp.gxx_support_openmp = OpenMPOp.test_gxx_support()
                if not OpenMPOp.gxx_support_openmp:
                    # We want to warn only once.
                    warnings.warn(
                        "Your g++ compiler fails to compile OpenMP code. We"
                        " know this happen with some version of the EPD mingw"
                        " compiler and LLVM compiler on Mac OS X."
                        " We disable openmp everywhere in Aesara."
                        " To remove this warning set the aesara flags `openmp`"
                        " to False.",
                        stacklevel=3,
                    )
            if OpenMPOp.gxx_support_openmp is False:
                self.openmp = False
                config.openmp = False

    def prepare_node(self, node, storage_map, compute_map, impl):
        if impl == "c":
            self.update_self_openmp()


def lquote_macro(txt: str) -> str:
    """Turn the last line of text into a ``\\``-commented line."""
    res = []
    spl = txt.split("\n")
    for l in spl[:-1]:
        res.append(l + " \\")
    res.append(spl[-1])
    return "\n".join(res)


def get_sub_macros(sub: Dict[str, str]) -> Union[Tuple[str], Tuple[str, str]]:
    define_macros = []
    undef_macros = []
    define_macros.append(f"#define FAIL {lquote_macro(sub['fail'])}")
    undef_macros.append("#undef FAIL")
    if "params" in sub:
        define_macros.append(f"#define PARAMS {sub['params']}")
        undef_macros.append("#undef PARAMS")

    return "\n".join(define_macros), "\n".join(undef_macros)


def get_io_macros(
    inputs: List[str], outputs: List[str]
) -> Union[Tuple[List[str]], Tuple[str, str]]:
    define_macros = []
    undef_macros = []

    for i, inp in enumerate(inputs):
        define_macros.append(f"#define INPUT_{int(i)} {inp}")
        undef_macros.append(f"#undef INPUT_{int(i)}")

    for i, out in enumerate(outputs):
        define_macros.append(f"#define OUTPUT_{int(i)} {out}")
        undef_macros.append(f"#undef OUTPUT_{int(i)}")

    return "\n".join(define_macros), "\n".join(undef_macros)


class ExternalCOp(COp):
    """Class for an `Op` with an external C implementation.

    One can inherit from this class, provide its constructor with a path to
    an external C source file and the name of a function within it, and define
    an `Op` for said function.

    """

    section_re: ClassVar[Pattern] = re.compile(
        r"^#section ([a-zA-Z0-9_]+)$", re.MULTILINE
    )
    backward_re: ClassVar[Pattern] = re.compile(
        r"^AESARA_(APPLY|SUPPORT)_CODE_SECTION$", re.MULTILINE
    )
    # This is the set of allowed markers
    SECTIONS: ClassVar[Set[str]] = {
        "init_code",
        "init_code_apply",
        "init_code_struct",
        "support_code",
        "support_code_apply",
        "support_code_struct",
        "cleanup_code_struct",
        "code",
        "code_cleanup",
    }
    _cop_num_inputs: Optional[int] = None
    _cop_num_outputs: Optional[int] = None

    @classmethod
    def get_path(cls, f: str) -> str:
        """Convert a path relative to the location of the class file into an absolute path.

        Paths that are already absolute are passed through unchanged.

        """
        if not os.path.isabs(f):
            class_file = inspect.getfile(cls)
            class_dir = os.path.dirname(class_file)
            f = os.path.realpath(os.path.join(class_dir, f))
        return f

    def __init__(
        self, func_files: Union[str, List[str]], func_name: Optional[str] = None
    ):
        """
        Sections are loaded from files in order with sections in later
        files overriding sections in previous files.

        """
        if not isinstance(func_files, list):
            self.func_files = [func_files]
        else:
            self.func_files = func_files

        self.func_codes: List[str] = []
        # Keep the original name. If we reload old pickle, we want to
        # find the new path and new version of the file in Aesara.
        self.func_name = func_name
        self.code_sections: Dict[str, str] = dict()

        self.load_c_code(self.func_files)

        if len(self.code_sections) == 0:
            raise ValueError("No sections where defined in the C files")

        if self.func_name is not None:
            if "op_code" in self.code_sections:
                # maybe a warning instead (and clearing the key)
                raise ValueError(
                    "Cannot have an `op_code` section and specify `func_name`"
                )
            if "op_code_cleanup" in self.code_sections:
                # maybe a warning instead (and clearing the key)
                raise ValueError(
                    "Cannot have an `op_code_cleanup` section and specify `func_name`"
                )

    def load_c_code(self, func_files: List[str]) -> None:
        """Loads the C code to perform the `Op`."""
        func_files = [self.get_path(f) for f in func_files]
        for func_file in func_files:
            with open(func_file) as f:
                self.func_codes.append(f.read())

        # If both the old section markers and the new section markers are
        # present, raise an error because we don't know which ones to follow.
        old_markers_present = False
        new_markers_present = False
        for code in self.func_codes:
            if self.backward_re.search(code):
                old_markers_present = True
            if self.section_re.search(code):
                new_markers_present = True

        if old_markers_present and new_markers_present:
            raise ValueError(
                "Both the new and the old syntax for "
                "identifying code sections are present in the "
                "provided C code. These two syntaxes should not "
                "be used at the same time."
            )

        for i, code in enumerate(self.func_codes):
            if self.backward_re.search(code):
                # This is backward compat code that will go away in a while

                # Separate the code into the proper sections
                split = self.backward_re.split(code)
                n = 1
                while n < len(split):
                    if split[n] == "APPLY":
                        self.code_sections["support_code_apply"] = split[n + 1]
                    elif split[n] == "SUPPORT":
                        self.code_sections["support_code"] = split[n + 1]
                    n += 2
                continue

            elif self.section_re.search(code):

                # Check for code outside of the supported sections
                split = self.section_re.split(code)
                if split[0].strip() != "":
                    raise ValueError(
                        "Stray code before first #section "
                        f"statement (in file {func_files[i]}): {split[0]}"
                    )

                # Separate the code into the proper sections
                n = 1
                while n < len(split):
                    if split[n] not in self.SECTIONS:
                        raise ValueError(
                            f"Unknown section type (in file {func_files[i]}): {split[n]}"
                        )
                    if split[n] not in self.code_sections:
                        self.code_sections[split[n]] = ""
                    self.code_sections[split[n]] += split[n + 1]
                    n += 2

            else:
                raise ValueError(
                    f"No valid section marker was found in file {func_files[i]}"
                )

    def __get_op_params(self) -> List[Tuple[str, Any]]:
        """Construct name, value pairs that will be turned into macros for use within the `Op`'s code.

        The names must be strings that are not a C keyword and the
        values must be strings of literal C representations.

        If op uses a :class:`aesara.graph.params_type.ParamsType` as ``params_type``,
        it returns:
         - a default macro ``PARAMS_TYPE`` which defines the class name of the
           corresponding C struct.
         - a macro ``DTYPE_PARAM_key`` for every ``key`` in the :class:`ParamsType` for which associated
           type implements the method :func:`aesara.graph.type.CLinkerType.c_element_type`.
           ``DTYPE_PARAM_key`` defines the primitive C type name of an item in a variable
           associated to ``key``.

        """
        params: List[Tuple[str, Any]] = []
        if isinstance(self.params_type, ParamsType):
            wrapper = self.params_type
            params.append(("PARAMS_TYPE", wrapper.name))
            for i in range(wrapper.length):
                c_type = wrapper.types[i].c_element_type()
                if c_type:
                    # NB (reminder): These macros are currently used only in ParamsType example test
                    # (`aesara/graph/tests/test_quadratic_function.c`), to demonstrate how we can
                    # access params dtypes when dtypes may change (e.g. if based on config.floatX).
                    # But in practice, params types generally have fixed types per op.
                    params.append(
                        (
                            "DTYPE_PARAM_" + wrapper.fields[i],
                            c_type,
                        )
                    )
        return params

    def c_code_cache_version(self):
        version = (hash_from_code("\n".join(self.func_codes)),)
        if self.params_type is not None:
            version += (self.params_type.c_code_cache_version(),)
        return version

    def c_init_code(self, **kwargs):
        if "init_code" in self.code_sections:
            return [self.code_sections["init_code"]]
        else:
            return super().c_init_code(**kwargs)

    def c_support_code(self, **kwargs):
        if "support_code" in self.code_sections:
            return self.code_sections["support_code"]
        else:
            return super().c_support_code(**kwargs)

    def c_init_code_apply(self, node, name):
        if "init_code_apply" in self.code_sections:
            code = self.code_sections["init_code_apply"]

            define_macros, undef_macros = self.get_c_macros(node, name)
            return "\n".join(["", define_macros, code, undef_macros])
        else:
            return super().c_init_code_apply(node, name)

    def c_support_code_apply(self, node, name):
        if "support_code_apply" in self.code_sections:
            code = self.code_sections["support_code_apply"]

            define_macros, undef_macros = self.get_c_macros(node, name)
            return "\n".join(["", define_macros, code, undef_macros])
        else:
            return super().c_support_code_apply(node, name)

    def c_support_code_struct(self, node, name):
        if "support_code_struct" in self.code_sections:
            code = self.code_sections["support_code_struct"]

            define_macros, undef_macros = self.get_c_macros(node, name)
            return "\n".join(["", define_macros, code, undef_macros])
        else:
            return super().c_support_code_struct(node, name)

    def c_cleanup_code_struct(self, node, name):
        if "cleanup_code_struct" in self.code_sections:
            code = self.code_sections["cleanup_code_struct"]

            define_macros, undef_macros = self.get_c_macros(node, name)
            return "\n".join(["", define_macros, code, undef_macros])
        else:
            return super().c_cleanup_code_struct(node, name)

    def format_c_function_args(self, inp: List[str], out: List[str]) -> str:
        """Generate a string containing the arguments sent to the external C function.

        The result will have the format: ``"input0, input1, input2, &output0, &output1"``.

        """
        inp = list(inp)
        if self._cop_num_inputs is not None:
            numi = self._cop_num_inputs
        else:
            numi = len(inp)

        while len(inp) < numi:
            inp.append("NULL")

        out = [f"&{o}" for o in out]

        if self._cop_num_outputs is not None:
            numo = self._cop_num_outputs
        else:
            numo = len(out)

        while len(out) < numo:
            out.append("NULL")

        return ", ".join(inp + out)

    def get_c_macros(
        self, node: Apply, name: str, check_input: Optional[bool] = None
    ) -> Union[Tuple[str], Tuple[str, str]]:
        "Construct a pair of C ``#define`` and ``#undef`` code strings."
        define_template = "#define %s %s"
        undef_template = "#undef %s"
        define_macros = []
        undef_macros = []

        if check_input is None:
            check_input = getattr(self, "check_input", True)

        if check_input:
            # Extract the various properties of the input and output variables
            variables = node.inputs + node.outputs
            variable_names = [f"INPUT_{i}" for i in range(len(node.inputs))] + [
                f"OUTPUT_{i}" for i in range(len(node.outputs))
            ]

            # Generate dtype macros
            for i, v in enumerate(variables):
                if not isinstance(v.type, HasDataType):
                    continue

                vname = variable_names[i]

                macro_items = (f"DTYPE_{vname}", f"npy_{v.type.dtype}")
                define_macros.append(define_template % macro_items)
                undef_macros.append(undef_template % macro_items[0])

                d = np.dtype(v.type.dtype)

                macro_items_2 = (f"TYPENUM_{vname}", d.num)
                define_macros.append(define_template % macro_items_2)
                undef_macros.append(undef_template % macro_items_2[0])

                macro_items_3 = (f"ITEMSIZE_{vname}", d.itemsize)
                define_macros.append(define_template % macro_items_3)
                undef_macros.append(undef_template % macro_items_3[0])

        # Generate a macro to mark code as being apply-specific
        define_macros.append(define_template % ("APPLY_SPECIFIC(str)", f"str##_{name}"))
        undef_macros.append(undef_template % "APPLY_SPECIFIC")

        for n, v in self.__get_op_params():
            define_macros.append(define_template % (n, v))
            undef_macros.append(undef_template % (n,))

        return "\n".join(define_macros), "\n".join(undef_macros)

    def c_init_code_struct(self, node, name, sub):
        r"""Stitches all the macros and ``init_code_*``\s together."""
        if "init_code_struct" in self.code_sections:
            op_code = self.code_sections["init_code_struct"]

            def_macros, undef_macros = self.get_c_macros(node, name)
            def_sub, undef_sub = get_sub_macros(sub)

            return "\n".join(
                ["", def_macros, def_sub, op_code, undef_sub, undef_macros]
            )
        else:
            return super().c_init_code_struct(node, name, sub)

    def c_code(self, node, name, inp, out, sub):
        if self.func_name is not None:
            assert "code" not in self.code_sections

            define_macros, undef_macros = self.get_c_macros(
                node, name, check_input=False
            )

            params = ""
            if "params" in sub:
                params = f", {sub['params']}"

            # Generate the C code
            return """
                %(define_macros)s
                {
                  if (%(func_name)s(%(func_args)s%(params)s) != 0) {
                    %(fail)s
                  }
                }
                %(undef_macros)s
                """ % dict(
                func_name=self.func_name,
                fail=sub["fail"],
                params=params,
                func_args=self.format_c_function_args(inp, out),
                define_macros=define_macros,
                undef_macros=undef_macros,
            )
        else:
            if "code" in self.code_sections:
                op_code = self.code_sections["code"]

                def_macros, undef_macros = self.get_c_macros(node, name)
                def_sub, undef_sub = get_sub_macros(sub)
                def_io, undef_io = get_io_macros(inp, out)

                return "\n".join(
                    [
                        def_macros,
                        def_sub,
                        def_io,
                        op_code,
                        undef_io,
                        undef_sub,
                        undef_macros,
                    ]
                )
            else:
                raise NotImplementedError()

    def c_code_cleanup(self, node, name, inputs, outputs, sub):
        r"""Stitches all the macros and ``code_cleanup``\s together."""
        if "code_cleanup" in self.code_sections:
            op_code = self.code_sections["code_cleanup"]

            def_macros, undef_macros = self.get_c_macros(node, name)
            def_sub, undef_sub = get_sub_macros(sub)
            def_io, undef_io = get_io_macros(inputs, outputs)

            return "\n".join(
                [
                    def_macros,
                    def_sub,
                    def_io,
                    op_code,
                    undef_io,
                    undef_sub,
                    undef_macros,
                ]
            )
        else:
            return super().c_code_cleanup(node, name, inputs, outputs, sub)


class _NoPythonCOp(COp):
    """A class used to indicate that a `COp` does not provide a Python implementation.

    XXX: Do not use this class; it's only for tracking bad implementations internally.

    """

    def perform(self, node, inputs, output_storage, params=None):
        raise NotImplementedError("No Python implementation is provided by this COp.")


class _NoPythonExternalCOp(ExternalCOp):
    """A class used to indicate that an `ExternalCOp` does not provide a Python implementation.

    XXX: Do not use this class; it's only for tracking bad implementations internally.

    """

    def perform(self, node, inputs, output_storage, params=None):
        raise NotImplementedError(
            "No Python implementation is provided by this ExternalCOp."
        )
