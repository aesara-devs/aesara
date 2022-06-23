"""
Generate and compile C modules for Python.

"""
import distutils.sysconfig
import importlib
import logging
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import sysconfig
import tempfile
import textwrap
import threading
import time
import warnings
from io import BytesIO, StringIO
from typing import Callable, List, Optional, Tuple, cast

import numpy.distutils
from typing_extensions import Protocol

# we will abuse the lockfile mechanism when reading and writing the registry
from aesara.configdefaults import config, gcc_version_str
from aesara.configparser import BoolParam, StrParam
from aesara.link.c.exceptions import CompileError, MissingGXX
from aesara.utils import (
    LOCAL_BITWIDTH,
    hash_from_code,
    maybe_add_to_os_environ_pathlist,
    output_subprocess_Popen,
    subprocess_Popen,
)


class StdLibDirsAndLibsType(Protocol):
    data: Optional[Tuple[List[str], ...]]
    __call__: Callable[[], Optional[Tuple[List[str], ...]]]


def is_StdLibDirsAndLibsType(
    fn: Callable[[], Optional[Tuple[List[str], ...]]]
) -> StdLibDirsAndLibsType:
    return cast(StdLibDirsAndLibsType, fn)


class GCCLLVMType(Protocol):
    is_llvm: Optional[bool]
    __call__: Callable[[], Optional[bool]]


def is_GCCLLVMType(fn: Callable[[], Optional[bool]]) -> GCCLLVMType:
    return cast(GCCLLVMType, fn)


_logger = logging.getLogger("aesara.link.c.cmodule")

METH_VARARGS = "METH_VARARGS"
METH_NOARGS = "METH_NOARGS"
# global variable that represent the total time spent in importing module.
import_time = 0


def debug_counter(name, every=1):
    """
    Debug counter to know how often we go through some piece of code.

    This is a utility function one may use when debugging.

    Examples
    --------
    debug_counter('I want to know how often I run this line')

    """
    setattr(debug_counter, name, getattr(debug_counter, name, 0) + 1)
    n = getattr(debug_counter, name)
    if n % every == 0:
        print(f"debug_counter [{name}]: {n}", file=sys.stderr)


class ExtFunction:
    """
    A C function to put into a DynamicModule.

    """

    name = ""
    """
    str - function's name.

    """
    code_block = ""
    """
    str - the entire code for the function.

    Has the form ``static PyObject* <name>([...]){ ... }

    See Python's C API Reference for how to write c functions for python
    modules.

    """
    method = ""
    """
    str - calling method for this function (i.e. 'METH_VARARGS', 'METH_NOARGS').

    """
    doc = ""
    """
    str - documentation string for this function.

    """

    def __init__(self, name, code_block, method, doc="undocumented"):
        self.name = name
        self.code_block = code_block
        self.method = method
        self.doc = doc

    def method_decl(self):
        """
        Returns the signature for this function.

        It goes into the DynamicModule's method table.

        """
        return f'\t{{"{self.name}", {self.name}, {self.method}, "{self.doc}"}}'


class DynamicModule:
    def __init__(self, name=None):
        assert name is None, (
            "The 'name' parameter of DynamicModule"
            " cannot be specified anymore. Instead, 'code_hash'"
            " will be automatically computed and can be used as"
            " the module's name."
        )
        # While the module is not finalized, we can call add_...
        # when it is finalized, a hash is computed and used instead of
        # the placeholder, and as module name.
        self.finalized = False
        self.code_hash = None
        self.hash_placeholder = "<<<<HASH_PLACEHOLDER>>>>"

        self.support_code = []
        self.functions = []
        self.includes = ["<Python.h>", "<iostream>", '"aesara_mod_helper.h"']
        self.init_blocks = []

    def print_methoddef(self, stream):
        print("static PyMethodDef MyMethods[] = {", file=stream)
        for f in self.functions:
            print(f.method_decl(), ",", file=stream)
        print("\t{NULL, NULL, 0, NULL}", file=stream)
        print("};", file=stream)

    def print_init(self, stream):
        print(
            f"""static struct PyModuleDef moduledef = {{
  PyModuleDef_HEAD_INIT,
  "{self.hash_placeholder}",
  NULL,
  -1,
  MyMethods,
}};
""",
            file=stream,
        )
        print(
            f"PyMODINIT_FUNC PyInit_{self.hash_placeholder}(void) {{",
            file=stream,
        )
        for block in self.init_blocks:
            print("  ", block, file=stream)
        print("    PyObject *m = PyModule_Create(&moduledef);", file=stream)
        print("    return m;", file=stream)
        print("}", file=stream)

    def add_include(self, str):
        assert not self.finalized
        self.includes.append(str)

    def add_init_code(self, code):
        assert not self.finalized
        self.init_blocks.append(code)

    def add_support_code(self, code):
        assert not self.finalized
        if code and code not in self.support_code:  # TODO: KLUDGE
            self.support_code.append(code)

    def add_function(self, fn):
        assert not self.finalized
        self.functions.append(fn)

    def code(self):
        sio = StringIO()
        for inc in self.includes:
            if not inc:
                continue
            if inc[0] == "<" or inc[0] == '"':
                print("#include", inc, file=sio)
            else:
                print(f'#include "{inc}"', file=sio)

        print("//////////////////////", file=sio)
        print("////  Support Code", file=sio)
        print("//////////////////////", file=sio)
        for sc in self.support_code:
            print(sc, file=sio)

        print("//////////////////////", file=sio)
        print("////  Functions", file=sio)
        print("//////////////////////", file=sio)
        for f in self.functions:
            print(f.code_block, file=sio)

        print("//////////////////////", file=sio)
        print("////  Module init", file=sio)
        print("//////////////////////", file=sio)
        self.print_methoddef(sio)
        self.print_init(sio)

        rval = sio.getvalue()
        # Make sure the hash of the code hasn't changed
        h = hash_from_code(rval)
        assert self.code_hash is None or self.code_hash == h
        self.code_hash = h
        rval = re.sub(self.hash_placeholder, self.code_hash, rval)
        # Finalize the Module, so no support code or function
        # can be added
        self.finalized = True

        return rval

    def list_code(self, ofile=sys.stdout):
        """
        Print out the code with line numbers to `ofile`.

        """
        for i, line in enumerate(self.code().split("\n")):
            print(f"{i + 1}", line, file=ofile)
        ofile.flush()

    # TODO: add_type


def _get_ext_suffix():
    """Get the suffix for compiled extensions"""
    dist_suffix = distutils.sysconfig.get_config_var("EXT_SUFFIX")
    if dist_suffix is None:
        dist_suffix = distutils.sysconfig.get_config_var("SO")
    return dist_suffix


def dlimport(fullpath, suffix=None):
    """
    Dynamically load a .so, .pyd, .dll, or .py file.

    Parameters
    ----------
    fullpath : str
        A fully-qualified path do a compiled python module.
    suffix : str
        A suffix to strip from the end of fullpath to get the
        import name.

    Returns
    -------
    object
        The dynamically loaded module (from __import__).

    """
    if not os.path.isabs(fullpath):
        raise ValueError("`fullpath` must be an absolute path", fullpath)
    if suffix is None:
        suffix = ""

        dist_suffix = _get_ext_suffix()
        if dist_suffix is not None and dist_suffix != "":
            if fullpath.endswith(dist_suffix):
                suffix = dist_suffix

        if suffix == "":
            if fullpath.endswith(".so"):
                suffix = ".so"
            elif fullpath.endswith(".pyd"):
                suffix = ".pyd"
            elif fullpath.endswith(".dll"):
                suffix = ".dll"
            elif fullpath.endswith(".py"):
                suffix = ".py"

    rval = None
    if fullpath.endswith(suffix):
        module_name = ".".join(fullpath.split(os.path.sep)[-2:])[: -len(suffix)]
    else:
        raise ValueError("path has wrong suffix", (fullpath, suffix))
    workdir = fullpath[: -len(module_name) - 1 - len(suffix)]

    _logger.debug(f"WORKDIR {workdir}")
    _logger.debug(f"module_name {module_name}")

    sys.path[0:0] = [workdir]  # insert workdir at beginning (temporarily)
    # Explicitly add gcc dll directory on Python 3.8+ on Windows
    if (sys.platform == "win32") & (hasattr(os, "add_dll_directory")):
        gcc_path = shutil.which("gcc")
        if gcc_path is not None:
            os.add_dll_directory(os.path.dirname(gcc_path))
    global import_time
    try:
        importlib.invalidate_caches()
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="numpy.ndarray size changed")
            rval = __import__(module_name, {}, {}, [module_name])
        t1 = time.time()
        import_time += t1 - t0
        if not rval:
            raise Exception("__import__ failed", fullpath)
    finally:
        del sys.path[0]

    assert fullpath.startswith(rval.__file__)
    return rval


def last_access_time(path):
    """
    Return the number of seconds since the epoch of the last access of a
    given file.

    """
    return os.stat(path)[stat.ST_ATIME]


def module_name_from_dir(dirname, err=True, files=None):
    """
    Scan the contents of a cache directory and return full path of the
    dynamic lib in it.

    """
    if files is None:
        try:
            files = os.listdir(dirname)
        except OSError as e:
            if e.errno == 2 and not err:  # No such file or directory
                return None
    names = [file for file in files if file.endswith(".so") or file.endswith(".pyd")]
    if len(names) == 0 and not err:
        return None
    elif len(names) == 1:
        return os.path.join(dirname, names[0])
    else:
        raise ValueError("More than 1 compiled module in this directory:" + dirname)


def is_same_entry(entry_1, entry_2):
    """
    Return True iff both paths can be considered to point to the same module.

    This is the case if and only if at least one of these conditions holds:
        - They are equal.
        - Their real paths are equal.
        - They share the same temporary work directory and module file name.

    """
    if entry_1 == entry_2:
        return True
    if os.path.realpath(entry_1) == os.path.realpath(entry_2):
        return True
    if (
        os.path.basename(entry_1) == os.path.basename(entry_2)
        and (
            os.path.basename(os.path.dirname(entry_1))
            == os.path.basename(os.path.dirname(entry_2))
        )
        and os.path.basename(os.path.dirname(entry_1)).startswith("tmp")
    ):
        return True
    return False


def get_module_hash(src_code, key):
    """
    Return a SHA256 hash that uniquely identifies a module.

    This hash takes into account:
        1. The C source code of the module (`src_code`).
        2. The version part of the key.
        3. The compiler options defined in `key` (command line parameters and
           libraries to link against).
        4. The NumPy ABI version.

    """
    # `to_hash` will contain any element such that we know for sure that if
    # it changes, then the module hash should be different.
    # We start with the source code itself (stripping blanks might avoid
    # recompiling after a basic indentation fix for instance).
    to_hash = [l.strip() for l in src_code.split("\n")]
    # Get the version part of the key (ignore if unversioned).
    if key[0]:
        to_hash += list(map(str, key[0]))
    c_link_key = key[1]
    # Currently, in order to catch potential bugs early, we are very
    # convervative about the structure of the key and raise an exception
    # if it does not match exactly what we expect. In the future we may
    # modify this behavior to be less strict and be able to accommodate
    # changes to the key in an automatic way.
    # Note that if the key structure changes, the `get_safe_part` function
    # below may also need to be modified.
    error_msg = (
        "This should not happen unless someone modified the code "
        "that defines the CLinker key, in which case you should "
        "ensure this piece of code is still valid (and this "
        "AssertionError may be removed or modified to accommodate "
        "this change)"
    )
    assert c_link_key[0] == "CLinker.cmodule_key", error_msg
    for key_element in c_link_key[1:]:
        if isinstance(key_element, tuple):
            # This should be the C++ compilation command line parameters or the
            # libraries to link against.
            to_hash += list(key_element)
        elif isinstance(key_element, str):
            if key_element.startswith("md5:") or key_element.startswith("hash:"):
                # This is actually a sha256 hash of the config options.
                # Currently, we still keep md5 to don't break old Aesara.
                # We add 'hash:' so that when we change it in
                # the futur, it won't break this version of Aesara.
                break
            elif key_element.startswith("NPY_ABI_VERSION=0x") or key_element.startswith(
                "c_compiler_str="
            ):
                to_hash.append(key_element)
            else:
                raise AssertionError(error_msg)
        else:
            raise AssertionError(error_msg)
    return hash_from_code("\n".join(to_hash))


def get_safe_part(key):
    """
    Return a tuple containing a subset of `key`, to be used to find equal keys.

    This tuple should only contain objects whose __eq__ and __hash__ methods
    can be trusted (currently: the version part of the key, as well as the
    SHA256 hash of the config options).
    It is used to reduce the amount of key comparisons one has to go through
    in order to find broken keys (i.e. keys with bad implementations of __eq__
    or __hash__).


    """
    version = key[0]
    # This function should only be called on versioned keys.
    assert version

    # Find the hash part. This is actually a sha256 hash of the config
    # options.  Currently, we still keep md5 to don't break old
    # Aesara.  We add 'hash:' so that when we change it
    # in the futur, it won't break this version of Aesara.
    c_link_key = key[1]
    # In case in the future, we don't have an md5 part and we have
    # such stuff in the cache.  In that case, we can set None, and the
    # rest of the cache mechanism will just skip that key.
    hash = None
    for key_element in c_link_key[1:]:
        if isinstance(key_element, str):
            if key_element.startswith("md5:"):
                hash = key_element[4:]
                break
            elif key_element.startswith("hash:"):
                hash = key_element[5:]
                break

    return key[0] + (hash,)


def get_lib_extension():
    """
    Return the platform-dependent extension for compiled modules.

    """
    if sys.platform == "win32":
        return "pyd"
    elif sys.platform == "cygwin":
        return "dll"
    else:
        return "so"


def get_gcc_shared_library_arg():
    """
    Return the platform-dependent GCC argument for shared libraries.

    """
    if sys.platform == "darwin":
        return "-dynamiclib"
    else:
        return "-shared"


def std_include_dirs():
    numpy_inc_dirs = numpy.distutils.misc_util.get_numpy_include_dirs()
    py_inc = distutils.sysconfig.get_python_inc()
    py_plat_spec_inc = distutils.sysconfig.get_python_inc(plat_specific=True)
    python_inc_dirs = (
        [py_inc] if py_inc == py_plat_spec_inc else [py_inc, py_plat_spec_inc]
    )
    gof_inc_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "c_code")
    return numpy_inc_dirs + python_inc_dirs + [gof_inc_dir]


@is_StdLibDirsAndLibsType
def std_lib_dirs_and_libs() -> Optional[Tuple[List[str], ...]]:
    # We cache the results as on Windows, this trigger file access and
    # this method is called many times.
    if std_lib_dirs_and_libs.data is not None:
        return std_lib_dirs_and_libs.data
    python_inc = distutils.sysconfig.get_python_inc()
    if sys.platform == "win32":
        # Obtain the library name from the Python version instead of the
        # installation directory, in case the user defined a custom
        # installation directory.
        python_version = sysconfig.get_python_version()
        libname = "python" + python_version.replace(".", "")
        # Also add directory containing the Python library to the library
        # directories.
        python_lib_dirs = [os.path.join(os.path.dirname(python_inc), "libs")]
        if "Canopy" in python_lib_dirs[0]:
            # Canopy stores libpython27.a and libmsccr90.a in this directory.
            # For some reason, these files are needed when compiling Python
            # modules, even when libpython27.lib and python27.dll are
            # available, and the *.a files have to be found earlier than
            # the other ones.

            # When Canopy is installed for the user:
            # sys.prefix:C:\Users\username\AppData\Local\Enthought\Canopy\User
            # sys.base_prefix:C:\Users\username\AppData\Local\Enthought\Canopy\App\appdata\canopy-1.1.0.1371.win-x86_64
            # When Canopy is installed for all users:
            # sys.base_prefix: C:\Program Files\Enthought\Canopy\App\appdata\canopy-1.1.0.1371.win-x86_64
            # sys.prefix: C:\Users\username\AppData\Local\Enthought\Canopy\User
            # So we need to use sys.prefix as it support both cases.
            # sys.base_prefix support only one case
            libdir = os.path.join(sys.prefix, "libs")

            for f, lib in [("libpython27.a", "libpython 1.2")]:
                if not os.path.exists(os.path.join(libdir, f)):
                    print(
                        "Your Python version is from Canopy. "
                        + "You need to install the package '"
                        + lib
                        + "' from Canopy package manager."
                    )
            libdirs = [
                # Used in older Canopy
                os.path.join(sys.prefix, "libs"),
                # Used in newer Canopy
                os.path.join(sys.prefix, r"EGG-INFO\mingw\usr\x86_64-w64-mingw32\lib"),
            ]
            for f, lib in [
                ("libmsvcr90.a", "mingw 4.5.2 or 4.8.1-2 (newer could work)")
            ]:
                if not any(
                    os.path.exists(os.path.join(tmp_libdir, f))
                    for tmp_libdir in libdirs
                ):
                    print(
                        "Your Python version is from Canopy. "
                        + "You need to install the package '"
                        + lib
                        + "' from Canopy package manager."
                    )
            python_lib_dirs.insert(0, libdir)
        std_lib_dirs_and_libs.data = [libname], python_lib_dirs

    # Suppress -lpython2.x on OS X since the `-undefined dynamic_lookup`
    # makes it unnecessary.
    elif sys.platform == "darwin":
        std_lib_dirs_and_libs.data = [], []
    else:
        if platform.python_implementation() == "PyPy":
            # Assume Linux (note: Ubuntu doesn't ship this .so)
            libname = "pypy3-c"
            # Unfortunately the only convention of this .so is that it appears
            # next to the location of the interpreter binary.
            libdir = os.path.dirname(os.path.realpath(sys.executable))
        else:
            # Assume Linux
            # Typical include directory: /usr/include/python2.6

            # get the name of the python library (shared object)

            libname = str(distutils.sysconfig.get_config_var("LDLIBRARY"))

            if libname.startswith("lib"):
                libname = libname[3:]

            # remove extension if present
            if libname.endswith(".so"):
                libname = libname[:-3]
            elif libname.endswith(".a"):
                libname = libname[:-2]

            libdir = str(distutils.sysconfig.get_config_var("LIBDIR"))

        std_lib_dirs_and_libs.data = [libname], [libdir]

    # sometimes, the linker cannot find -lpython so we need to tell it
    # explicitly where it is located this returns
    # somepath/lib/python2.x

    python_lib = str(
        distutils.sysconfig.get_python_lib(plat_specific=True, standard_lib=True)
    )
    python_lib = os.path.dirname(python_lib)
    if python_lib not in std_lib_dirs_and_libs.data[1]:
        std_lib_dirs_and_libs.data[1].append(python_lib)
    return std_lib_dirs_and_libs.data


std_lib_dirs_and_libs.data = None


def std_libs():
    return std_lib_dirs_and_libs()[0]


def std_lib_dirs():
    return std_lib_dirs_and_libs()[1]


def gcc_version():
    return gcc_version_str


@is_GCCLLVMType
def gcc_llvm() -> Optional[bool]:
    """
    Detect if the g++ version used is the llvm one or not.

    It don't support all g++ parameters even if it support many of them.

    """
    if gcc_llvm.is_llvm is None:
        try:
            p_out = output_subprocess_Popen([config.cxx, "--version"])
            output = p_out[0] + p_out[1]
        except OSError:
            # Typically means g++ cannot be found.
            # So it is not an llvm compiler.

            # Normally this should not happen as we should not try to
            # compile when g++ is not available. If this happen, it
            # will crash later so supposing it is not llvm is "safe".
            output = b""
        gcc_llvm.is_llvm = b"llvm" in output
    return gcc_llvm.is_llvm


gcc_llvm.is_llvm = None


class CompilerBase(threading.local):
    """
    Meta compiler that offer some generic function.

    """

    def _try_compile_tmp(
        self,
        src_code,
        tmp_prefix="",
        flags=(),
        try_run=False,
        output=False,
        compiler=None,
        comp_args=True,
    ):
        """
        Try to compile (and run) a test program.

        This is useful in various occasions, to check if libraries
        or compilers are behaving as expected.

        If try_run is True, the src_code is assumed to be executable,
        and will be run.

        If try_run is False, returns the compilation status.
        If try_run is True, returns a (compile_status, run_status) pair.
        If output is there, we append the stdout and stderr to the output.

        Compile arguments from the Compiler's compile_args() method are added
        if comp_args=True.
        """
        if not compiler:
            return False
        flags = list(flags)
        # Get compile arguments from compiler method if required
        if comp_args:
            args = self.compile_args()
        else:
            args = []
        compilation_ok = True
        run_ok = False
        out, err = None, None
        try:
            fd, path = tempfile.mkstemp(suffix=".c", prefix=tmp_prefix)
            exe_path = path[:-2]
            if os.name == "nt":
                path = '"' + path + '"'
                exe_path = '"' + exe_path + '"'
            try:
                try:
                    src_code = src_code.encode()
                except AttributeError:  # src_code was already bytes
                    pass
                os.write(fd, src_code)
                os.close(fd)
                fd = None
                out, err, p_ret = output_subprocess_Popen(
                    [compiler] + args + [path, "-o", exe_path] + flags
                )
                if p_ret != 0:
                    compilation_ok = False
                elif try_run:
                    out, err, p_ret = output_subprocess_Popen([exe_path])
                    run_ok = p_ret == 0
            finally:
                try:
                    if fd is not None:
                        os.close(fd)
                finally:
                    if os.path.exists(path):
                        os.remove(path)
                    if os.path.exists(exe_path):
                        os.remove(exe_path)
                    if os.path.exists(exe_path + ".exe"):
                        os.remove(exe_path + ".exe")
        except OSError as e:
            if err is None:
                err = str(e)
            else:
                err = str(err) + "\n" + str(e)
            compilation_ok = False

        if not try_run and not output:
            return compilation_ok
        elif not try_run and output:
            return (compilation_ok, out, err)
        elif not output:
            return (compilation_ok, run_ok)
        else:
            return (compilation_ok, run_ok, out, err)

    def _try_flags(
        self,
        flag_list,
        preamble="",
        body="",
        try_run=False,
        output=False,
        compiler=None,
        comp_args=True,
    ):
        """
        Try to compile a dummy file with these flags.

        Returns True if compilation was successful, False if there
        were errors.

        Compile arguments from the Compiler's compile_args() method are added
        if comp_args=True.

        """
        if not compiler:
            return False

        code = (
            """
        %(preamble)s
        int main(int argc, char** argv)
        {
            %(body)s
            return 0;
        }
        """
            % locals()
        ).encode()
        return self._try_compile_tmp(
            code,
            tmp_prefix="try_flags_",
            flags=flag_list,
            try_run=try_run,
            output=output,
            compiler=compiler,
            comp_args=comp_args,
        )


def try_blas_flag(flags):
    test_code = textwrap.dedent(
        """\
        extern "C" double ddot_(int*, double*, int*, double*, int*);
        int main(int argc, char** argv)
        {
            int Nx = 5;
            int Sx = 1;
            double x[5] = {0, 1, 2, 3, 4};
            double r = ddot_(&Nx, x, &Sx, x, &Sx);

            if ((r - 30.) > 1e-6 || (r - 30.) < -1e-6)
            {
                return -1;
            }
            return 0;
        }
        """
    )
    cflags = list(flags)
    # to support path that includes spaces, we need to wrap it with double quotes on Windows
    path_wrapper = '"' if os.name == "nt" else ""
    cflags.extend([f"-L{path_wrapper}{d}{path_wrapper}" for d in std_lib_dirs()])

    res = GCC_Compiler.try_compile_tmp(
        test_code, tmp_prefix="try_blas_", flags=cflags, try_run=True
    )
    # res[0]: shows successful compilation
    # res[1]: shows successful execution
    if res and res[0] and res[1]:
        return " ".join(flags)
    else:
        return ""


def try_march_flag(flags):
    """
    Try to compile and run a simple C snippet using current flags.
    Return: compilation success (True/False), execution success (True/False)
    """
    test_code = textwrap.dedent(
        """\
            #include <cmath>
            using namespace std;
            int main(int argc, char** argv)
            {
                float Nx = -1.3787706641;
                float Sx = 25.0;
                double r = Nx + sqrt(Sx);
                if (abs(r - 3.621229) > 0.01)
                {
                    return -1;
                }
                return 0;
            }
            """
    )

    cflags = flags + ["-L" + d for d in std_lib_dirs()]
    compilation_result, execution_result = GCC_Compiler.try_compile_tmp(
        test_code, tmp_prefix="try_march_", flags=cflags, try_run=True
    )
    return compilation_result, execution_result


class GCC_CompilerBase(CompilerBase):
    # The equivalent flags of --march=native used by g++.
    def __init__(self) -> None:
        super().__init__()
        self.march_flags = None
        self.supports_amdlibm = True

    @staticmethod
    def version_str():
        return config.cxx + " " + gcc_version_str

    def compile_args(self, march_flags=True):
        cxxflags = [flag for flag in config.gcc__cxxflags.split(" ") if flag]
        if "-fopenmp" in cxxflags:
            raise ValueError(
                "Do not use -fopenmp in Aesara flag gcc__cxxflags."
                " To enable OpenMP, use the Aesara flag openmp=True"
            )
        # Add the equivalent of -march=native flag.  We can't use
        # -march=native as when the compiledir is shared by multiple
        # computers (for example, if the home directory is on NFS), this
        # won't be optimum or cause crash depending if the file is compiled
        # on an older or more recent computer.
        # Those URL discuss how to find witch flags are used by -march=native.
        # http://en.gentoo-wiki.com/wiki/Safe_Cflags#-march.3Dnative
        # http://en.gentoo-wiki.com/wiki/Hardware_CFLAGS
        detect_march = self.march_flags is None and march_flags
        if detect_march:
            for f in cxxflags:
                # If the user give an -march=X parameter, don't add one ourself
                if f.startswith("--march=") or f.startswith("-march="):
                    detect_march = False
                    self.march_flags = []
                    break

        if (
            "g++" not in config.cxx
            and "clang++" not in config.cxx
            and "clang-omp++" not in config.cxx
            and "icpc" not in config.cxx
        ):
            _logger.warning(
                "Your Aesara flag `cxx` seems not to be"
                " the g++ compiler. So we disable the compiler optimization"
                " specific to g++ that tell to compile for a specific CPU."
                " At worst, this could cause slow down.\n"
                "         You can add those parameters to the compiler yourself"
                " via the Aesara flag `gcc__cxxflags`."
            )
            detect_march = False

        if detect_march:
            self.march_flags = []

            def get_lines(cmd, parse=True):
                p = subprocess_Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.PIPE,
                    shell=True,
                )
                # For mingw64 with GCC >= 4.7, passing os.devnull
                # as stdin (which is the default) results in the process
                # waiting forever without returning. For that reason,
                # we use a pipe, and use the empty string as input.
                (stdout, stderr) = p.communicate(input=b"")
                if p.returncode != 0:
                    return None

                lines = BytesIO(stdout + stderr).readlines()
                lines = (l.decode() for l in lines)
                if parse:
                    selected_lines = []
                    for line in lines:
                        if (
                            "COLLECT_GCC_OPTIONS=" in line
                            or "CFLAGS=" in line
                            or "CXXFLAGS=" in line
                            or "-march=native" in line
                        ):
                            continue
                        for reg in ("-march=", "-mtune=", "-target-cpu", "-mabi="):
                            if reg in line:
                                selected_lines.append(line.strip())
                    lines = list(set(selected_lines))  # to remove duplicate

                return lines

            # The '-' at the end is needed. Otherwise, g++ do not output
            # enough information.
            native_lines = get_lines(f"{config.cxx} -march=native -E -v -")
            if native_lines is None:
                _logger.info(
                    "Call to 'g++ -march=native' failed," "not setting -march flag"
                )
                detect_march = False
            else:
                _logger.info(f"g++ -march=native selected lines: {native_lines}")

        if detect_march:
            if len(native_lines) != 1:
                if len(native_lines) == 0:
                    # That means we did not select the right lines, so
                    # we have to report all the lines instead
                    reported_lines = get_lines(
                        f"{config.cxx} -march=native -E -v -", parse=False
                    )
                else:
                    reported_lines = native_lines
                _logger.warning(
                    "Aesara was not able to find the"
                    " g++ parameters that tune the compilation to your "
                    " specific CPU. This can slow down the execution of Aesara"
                    " functions. Please submit the following lines to"
                    " Aesara's mailing list so that we can fix this"
                    f" problem:\n {reported_lines}"
                )
            else:
                default_lines = get_lines(f"{config.cxx} -E -v -")
                _logger.info(f"g++ default lines: {default_lines}")
                if len(default_lines) < 1:
                    _logger.warning(
                        "Aesara was not able to find the"
                        " default g++ parameters. This is needed to tune"
                        " the compilation to your specific"
                        " CPU. This can slow down the execution of Aesara"
                        " functions. Please submit the following lines to"
                        " Aesara's mailing list so that we can fix this"
                        " problem:\n %s",
                        get_lines(f"{config.cxx} -E -v -", parse=False),
                    )
                else:
                    # Some options are actually given as "-option value",
                    # we want to treat them as only one token when comparing
                    # different command lines.
                    # Heuristic: tokens not starting with a dash should be
                    # joined with the previous one.
                    def join_options(init_part):
                        new_part = []
                        for i in range(len(init_part)):
                            p = init_part[i]
                            if p.startswith("-"):
                                p_list = [p]
                                while (i + 1 < len(init_part)) and not init_part[
                                    i + 1
                                ].startswith("-"):
                                    # append that next part to p_list
                                    p_list.append(init_part[i + 1])
                                    i += 1
                                new_part.append(" ".join(p_list))
                            elif i == 0:
                                # The first argument does not usually start
                                # with "-", still add it
                                new_part.append(p)
                            # Else, skip it, as it was already included
                            # with the previous part.
                        return new_part

                    part = join_options(native_lines[0].split())

                    for line in default_lines:
                        if line.startswith(part[0]):
                            part2 = [
                                p
                                for p in join_options(line.split())
                                if (
                                    "march" not in p
                                    and "mtune" not in p
                                    and "target-cpu" not in p
                                )
                            ]
                            if sys.platform == "darwin":
                                # We only use translated target-cpu on
                                # mac since the other flags are not
                                # supported as compiler flags for the
                                # driver.
                                new_flags = [p for p in part if "target-cpu" in p]
                            else:
                                new_flags = [p for p in part if p not in part2]
                            # Replace '-target-cpu value', which is an option
                            # of clang, with '-march=value'.
                            for i, p in enumerate(new_flags):
                                if "target-cpu" in p:
                                    opt = p.split()
                                    if len(opt) == 2:
                                        opt_name, opt_val = opt
                                        new_flags[i] = f"-march={opt_val}"

                            # Some versions of GCC report the native arch
                            # as "corei7-avx", but it generates illegal
                            # instructions, and should be "corei7" instead.
                            # Affected versions are:
                            # - 4.6 before 4.6.4
                            # - 4.7 before 4.7.3
                            # - 4.8 before 4.8.1
                            # Earlier versions did not have arch "corei7-avx"
                            for i, p in enumerate(new_flags):
                                if "march" not in p:
                                    continue
                                opt = p.split("=")
                                if len(opt) != 2:
                                    # Inexpected, but do not crash
                                    continue
                                opt_val = opt[1]
                                if not opt_val.endswith("-avx"):
                                    # OK
                                    continue
                                # Check the version of GCC
                                version = gcc_version_str.split(".")
                                if len(version) != 3:
                                    # Unexpected, but should not be a problem
                                    continue
                                mj, mn, patch = [int(vp) for vp in version]
                                if (
                                    ((mj, mn) == (4, 6) and patch < 4)
                                    or ((mj, mn) == (4, 7) and patch <= 3)
                                    or ((mj, mn) == (4, 8) and patch < 1)
                                ):
                                    new_flags[i] = p.rstrip("-avx")

                            # Go back to split arguments, like
                            # ["-option", "value"],
                            # as this is the way g++ expects them split.
                            split_flags = []
                            for p in new_flags:
                                split_flags.extend(p.split())

                            self.march_flags = split_flags
                            break
                    _logger.info(
                        f"g++ -march=native equivalent flags: {self.march_flags}"
                    )

            # Find working march flag:
            #   -- if current GCC_compiler.march_flags works, we're done.
            #   -- else replace -march and -mtune with ['core-i7-avx', 'core-i7', 'core2']
            #      and retry with all other flags and arguments intact.
            #   -- else remove all other flags and only try with -march = default + flags_to_try.
            #   -- if none of that worked, set GCC_compiler.march_flags = [] (for x86).

            default_compilation_result, default_execution_result = try_march_flag(
                self.march_flags
            )
            if not default_compilation_result or not default_execution_result:
                march_success = False
                march_ind = None
                mtune_ind = None
                default_detected_flag = []
                march_flags_to_try = ["corei7-avx", "corei7", "core2"]

                for m_ in range(len(self.march_flags)):
                    march_flag = self.march_flags[m_]
                    if "march" in march_flag:
                        march_ind = m_
                        default_detected_flag = [march_flag]
                    elif "mtune" in march_flag:
                        mtune_ind = m_

                for march_flag in march_flags_to_try:
                    if march_ind is not None:
                        self.march_flags[march_ind] = "-march=" + march_flag
                    if mtune_ind is not None:
                        self.march_flags[mtune_ind] = "-mtune=" + march_flag

                    compilation_result, execution_result = try_march_flag(
                        self.march_flags
                    )

                    if compilation_result and execution_result:
                        march_success = True
                        break

                if not march_success:
                    # perhaps one of the other flags was problematic; try default flag in isolation again:
                    march_flags_to_try = default_detected_flag + march_flags_to_try
                    for march_flag in march_flags_to_try:
                        compilation_result, execution_result = try_march_flag(
                            ["-march=" + march_flag]
                        )
                        if compilation_result and execution_result:
                            march_success = True
                            self.march_flags = ["-march=" + march_flag]
                            break

                if not march_success:
                    self.march_flags = []

        # Add the detected -march=native equivalent flags
        if march_flags and self.march_flags:
            cxxflags.extend(self.march_flags)

        # NumPy 1.7 Deprecate the old API.
        # The following macro asserts that we don't bring new code
        # that use the old API.
        cxxflags.append("-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION")

        # Platform-specific flags.
        # We put them here, rather than in compile_str(), so they en up
        # in the key of the compiled module, avoiding potential conflicts.

        # Figure out whether the current Python executable is 32
        # or 64 bit and compile accordingly. This step is ignored for
        # ARM (32-bit and 64-bit) and RISC-V architectures in order to make
        # Aesara compatible with the Raspberry Pi, Raspberry Pi 2, or
        # other systems with ARM or RISC-V processors.
        if (not any("arm" in flag or "riscv" in flag for flag in cxxflags)) and (
            not any(arch in platform.machine() for arch in ("arm", "aarch", "riscv"))
        ):
            n_bits = LOCAL_BITWIDTH
            cxxflags.append(f"-m{int(n_bits)}")
            _logger.debug(f"Compiling for {n_bits} bit architecture")

        if sys.platform != "win32":
            # Under Windows it looks like fPIC is useless. Compiler warning:
            # '-fPIC ignored for target (all code is position independent)'
            cxxflags.append("-fPIC")

        if sys.platform == "win32" and LOCAL_BITWIDTH == 64:
            # Under 64-bit Windows installation, sys.platform is 'win32'.
            # We need to define MS_WIN64 for the preprocessor to be able to
            # link with libpython.
            cxxflags.append("-DMS_WIN64")

        if sys.platform == "darwin":
            # Use the already-loaded python symbols.
            cxxflags.extend(["-undefined", "dynamic_lookup"])

        if sys.platform == "win32":
            # Workaround for https://github.com/Theano/Theano/issues/4926.
            # https://github.com/python/cpython/pull/11283/ removed the "hypot"
            # redefinition for recent CPython versions (>=2.7.16 and >=3.7.3).
            # The following nullifies that redefinition, if it is found.
            python_version = sys.version_info[:3]
            if (3,) <= python_version < (3, 7, 3):
                config_h_filename = distutils.sysconfig.get_config_h_filename()
                try:
                    with open(config_h_filename) as config_h:
                        if any(
                            line.startswith("#define hypot _hypot") for line in config_h
                        ):
                            cxxflags.append("-D_hypot=hypot")
                except OSError:
                    pass

        return cxxflags

    def try_compile_tmp(
        self,
        src_code,
        tmp_prefix="",
        flags=(),
        try_run=False,
        output=False,
        comp_args=True,
    ):
        return self._try_compile_tmp(
            src_code,
            tmp_prefix,
            self.patch_ldflags(flags),
            try_run,
            output,
            config.cxx,
            comp_args,
        )

    def try_flags(
        self,
        flag_list,
        preamble="",
        body="",
        try_run=False,
        output=False,
        comp_args=True,
    ):
        return self._try_flags(
            self.patch_ldflags(flag_list),
            preamble,
            body,
            try_run,
            output,
            config.cxx,
            comp_args,
        )

    def patch_ldflags(self, flag_list: List[str]) -> List[str]:
        lib_dirs = [flag[2:].lstrip() for flag in flag_list if flag.startswith("-L")]
        flag_idxs: List[int] = []
        libs: List[str] = []
        for i, flag in enumerate(flag_list):
            if flag.startswith("-l"):
                flag_idxs.append(i)
                libs.append(flag[2:].lstrip())
        if not libs:
            return flag_list
        libs = self.linking_patch(lib_dirs, libs)
        for flag_idx, lib in zip(flag_idxs, libs):
            flag_list[flag_idx] = lib
        return flag_list

    @staticmethod
    def linking_patch(lib_dirs: List[str], libs: List[str]) -> List[str]:
        if sys.platform != "win32":
            return [f"-l{l}" for l in libs]

        def sort_key(lib):  # type: ignore
            name, *numbers, extension = lib.split(".")
            return (extension == "dll", tuple(map(int, numbers)))

        patched_lib_ldflags = []
        for lib in libs:
            ldflag = f"-l{lib}"
            for lib_dir in lib_dirs:
                lib_dir = lib_dir.strip('"')
                windows_styled_libs = [
                    fname
                    for fname in os.listdir(lib_dir)
                    if not (os.path.isdir(os.path.join(lib_dir, fname)))
                    and fname.split(".")[0] == lib
                    and fname.split(".")[-1] in ["dll", "lib"]
                ]
                if windows_styled_libs:
                    selected_lib = sorted(windows_styled_libs, key=sort_key)[-1]
                    ldflag = f'"{os.path.join(lib_dir, selected_lib)}"'
            patched_lib_ldflags.append(ldflag)
        return patched_lib_ldflags

    def compile_str(
        self,
        module_name,
        src_code,
        location=None,
        include_dirs=None,
        lib_dirs=None,
        libs=None,
        preargs=None,
        py_module=True,
        hide_symbols=True,
        exist_ok=False,
    ):
        """
        Parameters
        ----------
        module_name : str
            This has been embedded in the src_code.
        src_code
            A complete c or c++ source listing for the module.
        location
            A pre-existing filesystem directory where the cpp file and .so will
            be written.
        include_dirs
            A list of include directory names (each gets prefixed with -I).
        lib_dirs
            A list of library search path directory names (each gets prefixed
            with -L).
        libs
            A list of libraries to link with (each gets prefixed with -l).
        preargs
            A list of extra compiler arguments.
        py_module
            If False, compile to a shared library, but do not import it as a
            Python module.
        hide_symbols
            If True (the default) all symbols will be hidden from the library
            symbol table (which means that other objects can't use them).
        exist_ok
            If True (False by default) skips the compilation assuming the result is already compiled

        Returns
        -------
        object
            Dynamically-imported python module of the compiled code (unless
            py_module is False, in that case returns None).

        """
        # TODO: Do not do the dlimport in this function

        if not config.cxx:
            raise MissingGXX("g++ not available! We can't compile c code.")

        if platform.python_implementation() == "PyPy":
            suffix = "." + get_lib_extension()

            dist_suffix = distutils.sysconfig.get_config_var("SO")
            if dist_suffix is not None and dist_suffix != "":
                suffix = dist_suffix

            filepath = f"{module_name}{suffix}"
        else:
            filepath = f"{module_name}.{get_lib_extension()}"

        lib_filename = os.path.join(location, filepath)
        if exist_ok and os.path.isfile(lib_filename):
            if py_module:
                open(os.path.join(location, "__init__.py"), "w").close()
                return dlimport(lib_filename)

        if include_dirs is None:
            include_dirs = []
        if lib_dirs is None:
            lib_dirs = []
        if libs is None:
            libs = []
        if preargs is None:
            preargs = []

        # Remove empty string directory
        include_dirs = [d for d in include_dirs if d]
        lib_dirs = [d for d in lib_dirs if d]

        include_dirs = include_dirs + std_include_dirs()
        libs = libs + std_libs()
        lib_dirs = lib_dirs + std_lib_dirs()

        cppfilename = os.path.join(location, "mod.cpp")
        with open(cppfilename, "w") as cppfile:

            _logger.debug(f"Writing module C++ code to {cppfilename}")

            cppfile.write(src_code)
            # Avoid gcc warning "no newline at end of file".
            if not src_code.endswith("\n"):
                cppfile.write("\n")

        _logger.debug(f"Generating shared lib {lib_filename}")
        cmd = [config.cxx, get_gcc_shared_library_arg(), "-g"]

        if config.cmodule__remove_gxx_opt:
            cmd.extend(p for p in preargs if not p.startswith("-O"))
        else:
            cmd.extend(preargs)
        # to support path that includes spaces, we need to wrap it with double quotes on Windows
        path_wrapper = '"' if os.name == "nt" else ""
        cmd.extend([f"-I{path_wrapper}{idir}{path_wrapper}" for idir in include_dirs])
        cmd.extend([f"-L{path_wrapper}{ldir}{path_wrapper}" for ldir in lib_dirs])
        if hide_symbols and sys.platform != "win32":
            # This has been available since gcc 4.0 so we suppose it
            # is always available. We pass it here since it
            # significantly reduces the size of the symbol table for
            # the objects we want to share. This in turns leads to
            # improved loading times on most platforms (win32 is
            # different, as usual).
            cmd.append("-fvisibility=hidden")
        cmd.extend(["-o", f"{path_wrapper}{lib_filename}{path_wrapper}"])
        cmd.append(f"{path_wrapper}{cppfilename}{path_wrapper}")
        cmd.extend(self.linking_patch(lib_dirs, libs))
        # print >> sys.stderr, 'COMPILING W CMD', cmd
        _logger.debug(f"Running cmd: {' '.join(cmd)}")

        def print_command_line_error():
            # Print command line when a problem occurred.
            print(
                ("Problem occurred during compilation with the " "command line below:"),
                file=sys.stderr,
            )
            print(" ".join(cmd), file=sys.stderr)

        try:
            p_out = launchCompilerProcess(cmd)
            compile_stderr = p_out[1].decode()
        except Exception:
            # An exception can occur e.g. if `g++` is not found.
            print_command_line_error()
            raise

        status = p_out[2]

        if status:
            tf = tempfile.NamedTemporaryFile(
                mode="w", prefix="aesara_compilation_error_", delete=False
            )
            # gcc put its messages to stderr, so we add ours now
            tf.write("===============================\n")
            for i, l in enumerate(src_code.split("\n")):
                tf.write(f"{i + 1}\t{l}\n")
            tf.write("===============================\n")
            tf.write(
                "Problem occurred during compilation with the " "command line below:\n"
            )
            tf.write(" ".join(cmd))
            # Print errors just below the command line.
            tf.write(compile_stderr)
            tf.close()
            print("\nYou can find the C code in this temporary file: " + tf.name)
            not_found_libraries = re.findall('-l["."-_a-zA-Z0-9]*', compile_stderr)
            for nf_lib in not_found_libraries:
                print("library " + nf_lib[2:] + " is not found.")
                if re.search('-lPYTHON["."0-9]*', nf_lib, re.IGNORECASE):
                    py_string = re.search(
                        '-lpython["."0-9]*', nf_lib, re.IGNORECASE
                    ).group()[8:]
                    if py_string != "":
                        print(
                            "Check if package python-dev "
                            + py_string
                            + " or python-devel "
                            + py_string
                            + " is installed."
                        )
                    else:
                        print(
                            "Check if package python-dev or python-devel is installed."
                        )

            # We replace '\n' by '. ' in the error message because when Python
            # prints the exception, having '\n' in the text makes it more
            # difficult to read.
            # compile_stderr = compile_stderr.replace("\n", ". ")
            raise CompileError(
                f"Compilation failed (return status={status}):\n{' '.join(cmd)}\n{compile_stderr}"
            )
        elif config.cmodule__compilation_warning and compile_stderr:
            # Print errors just below the command line.
            print(compile_stderr)

        if py_module:
            # touch the __init__ file
            open(os.path.join(location, "__init__.py"), "w").close()
            assert os.path.isfile(lib_filename)
            return dlimport(lib_filename)


def launchCompilerProcess(cmd):  # a proxy function to monitor compiler calls in tests
    return output_subprocess_Popen(cmd)


def icc_module_compile_str(*args):
    raise NotImplementedError()


def check_mkl_openmp():
    if not config.blas__check_openmp:
        return
    if sys.platform == "darwin":
        return
    if (
        "MKL_THREADING_LAYER" in os.environ
        and os.environ["MKL_THREADING_LAYER"] == "GNU"
    ):
        return
    try:
        import numpy._mklinit  # noqa

        return
    except ImportError:
        pass
    try:
        import mkl

        if "2018" in mkl.get_version_string():
            raise RuntimeError(
                """
To use MKL 2018 with Aesara either update the numpy conda packages to
their latest build or set "MKL_THREADING_LAYER=GNU" in your
environment.
"""
            )
    except ImportError:
        raise RuntimeError(
            """
Could not import 'mkl'.  If you are using conda, update the numpy
packages to the latest build otherwise, set MKL_THREADING_LAYER=GNU in
your environment for MKL 2018.

If you have MKL 2017 install and are not in a conda environment you
can set the Aesara flag blas__check_openmp to False.  Be warned that if
you set this flag and don't set the appropriate environment or make
sure you have the right version you *will* get wrong results.
"""
        )


def default_blas_ldflags():
    """Read local NumPy and MKL build settings and construct `ld` flags from them.

    Returns
    -------
    str

    """
    import numpy.distutils  # noqa

    warn_record = []
    try:
        # We do this import only here, as in some setup, if we
        # just import aesara and exit, with the import at global
        # scope, we get this error at exit: "Exception TypeError:
        # "'NoneType' object is not callable" in <bound method
        # Popen.__del__ of <subprocess.Popen object at 0x21359d0>>
        # ignored"

        # This happen with Python 2.7.3 |EPD 7.3-1 and numpy 1.8.1
        # isort: off
        import numpy.distutils.system_info  # noqa

        # We need to catch warnings as in some cases NumPy print
        # stuff that we don't want the user to see.
        # I'm not able to remove all printed stuff
        with warnings.catch_warnings(record=True):
            numpy.distutils.system_info.system_info.verbosity = 0
            blas_info = numpy.distutils.system_info.get_info("blas_opt")

        # If we are in a EPD installation, mkl is available
        if "EPD" in sys.version:
            use_unix_epd = True
            if sys.platform == "win32":
                return " ".join(
                    ['-L"%s"' % os.path.join(sys.prefix, "Scripts")]
                    +
                    # Why on Windows, the library used are not the
                    # same as what is in
                    # blas_info['libraries']?
                    [f"-l{l}" for l in ("mk2_core", "mk2_intel_thread", "mk2_rt")]
                )
            elif sys.platform == "darwin":
                # The env variable is needed to link with mkl
                new_path = os.path.join(sys.prefix, "lib")
                v = os.getenv("DYLD_FALLBACK_LIBRARY_PATH", None)
                if v is not None:
                    # Explicit version could be replaced by a symbolic
                    # link called 'Current' created by EPD installer
                    # This will resolve symbolic links
                    v = os.path.realpath(v)

                # The python __import__ don't seam to take into account
                # the new env variable "DYLD_FALLBACK_LIBRARY_PATH"
                # when we set with os.environ['...'] = X or os.putenv()
                # So we warn the user and tell him what todo.
                if v is None or new_path not in v.split(":"):
                    _logger.warning(
                        "The environment variable "
                        "'DYLD_FALLBACK_LIBRARY_PATH' does not contain "
                        "the '{new_path}' path in its value. This will make "
                        "Aesara use a slow version of BLAS. Update "
                        "'DYLD_FALLBACK_LIBRARY_PATH' to contain the "
                        "said value, this will disable this warning."
                    )

                    use_unix_epd = False
            if use_unix_epd:
                return " ".join(
                    ["-L%s" % os.path.join(sys.prefix, "lib")]
                    + ["-l%s" % l for l in blas_info["libraries"]]
                )

                # Canopy
        if "Canopy" in sys.prefix:
            subsub = "lib"
            if sys.platform == "win32":
                subsub = "Scripts"
            lib_path = os.path.join(sys.base_prefix, subsub)
            if not os.path.exists(lib_path):
                # Old logic to find the path. I don't think we still
                # need it, but I don't have the time to test all
                # installation configuration. So I keep this as a fall
                # back in case the current expectation don't work.

                # This old logic don't work when multiple version of
                # Canopy is installed.
                p = os.path.join(sys.base_prefix, "..", "..", "appdata")
                assert os.path.exists(p), "Canopy changed the location of MKL"
                lib_paths = os.listdir(p)
                # Try to remove subdir that can't contain MKL
                for sub in lib_paths:
                    if not os.path.exists(os.path.join(p, sub, subsub)):
                        lib_paths.remove(sub)
                assert len(lib_paths) == 1, (
                    "Unexpected case when looking for Canopy MKL libraries",
                    p,
                    lib_paths,
                    [os.listdir(os.path.join(p, sub)) for sub in lib_paths],
                )
                lib_path = os.path.join(p, lib_paths[0], subsub)
                assert os.path.exists(lib_path), "Canopy changed the location of MKL"

            if sys.platform == "linux2" or sys.platform == "darwin":
                return " ".join(
                    ["-L%s" % lib_path] + ["-l%s" % l for l in blas_info["libraries"]]
                )
            elif sys.platform == "win32":
                return " ".join(
                    ['-L"%s"' % lib_path]
                    +
                    # Why on Windows, the library used are not the
                    # same as what is in blas_info['libraries']?
                    [f"-l{l}" for l in ("mk2_core", "mk2_intel_thread", "mk2_rt")]
                )

        # MKL
        # If mkl can be imported then use it. On conda:
        # "conda install mkl-service" installs the Python wrapper and
        # the low-level C libraries as well as optimised version of
        # numpy and scipy.
        try:
            import mkl  # noqa
        except ImportError:
            pass
        else:
            # This branch is executed if no exception was raised
            if sys.platform == "win32":
                lib_path = os.path.join(sys.prefix, "Library", "bin")
                flags = [f'-L"{lib_path}"']
            else:
                lib_path = blas_info.get("library_dirs", [])
                flags = []
                if lib_path:
                    flags = [f"-L{lib_path[0]}"]
            if "2018" in mkl.get_version_string():
                thr = "mkl_gnu_thread"
            else:
                thr = "mkl_intel_thread"
            base_flags = list(flags)
            flags += [f"-l{l}" for l in ("mkl_core", thr, "mkl_rt")]
            res = try_blas_flag(flags)

            if not res and sys.platform == "win32" and thr == "mkl_gnu_thread":
                # Check if it would work for intel OpenMP on windows
                flags = base_flags + [
                    f"-l{l}" for l in ("mkl_core", "mkl_intel_thread", "mkl_rt")
                ]
                res = try_blas_flag(flags)

            if res:
                check_mkl_openmp()
                return res

            flags.extend(["-Wl,-rpath," + l for l in blas_info.get("library_dirs", [])])
            res = try_blas_flag(flags)
            if res:
                check_mkl_openmp()
                maybe_add_to_os_environ_pathlist("PATH", lib_path[0])
                return res

        # to support path that includes spaces, we need to wrap it with double quotes on Windows
        path_wrapper = '"' if os.name == "nt" else ""
        ret = (
            # TODO: the Gemm op below should separate the
            # -L and -l arguments into the two callbacks
            # that CLinker uses for that stuff.  for now,
            # we just pass the whole ldflags as the -l
            # options part.
            [
                f"-L{path_wrapper}{l}{path_wrapper}"
                for l in blas_info.get("library_dirs", [])
            ]
            + [f"-l{l}" for l in blas_info.get("libraries", [])]
            + blas_info.get("extra_link_args", [])
        )
        # For some very strange reason, we need to specify -lm twice
        # to get mkl to link correctly.  I have no idea why.
        if any("mkl" in fl for fl in ret):
            ret.extend(["-lm", "-lm"])
        res = try_blas_flag(ret)
        if res:
            if "mkl" in res:
                check_mkl_openmp()
            return res

        # If we are using conda and can't reuse numpy blas, then doing
        # the fallback and test -lblas could give slow computation, so
        # warn about this.
        for warn in warn_record:
            _logger.warning(warn)
        del warn_record

        # Some environment don't have the lib dir in LD_LIBRARY_PATH.
        # So add it.
        ret.extend(["-Wl,-rpath," + l for l in blas_info.get("library_dirs", [])])
        res = try_blas_flag(ret)
        if res:
            if "mkl" in res:
                check_mkl_openmp()
            return res

        # Add sys.prefix/lib to the runtime search path. On
        # non-system installations of Python that use the
        # system linker, this is generally necessary.
        if sys.platform in ("linux", "darwin"):
            lib_path = os.path.join(sys.prefix, "lib")
            ret.append("-Wl,-rpath," + lib_path)
            res = try_blas_flag(ret)
            if res:
                if "mkl" in res:
                    check_mkl_openmp()
                return res

    except KeyError:
        pass

    # Even if we could not detect what was used for numpy, or if these
    # libraries are not found, most Linux systems have a libblas.so
    # readily available. We try to see if that's the case, rather
    # than disable blas. To test it correctly, we must load a program.
    # Otherwise, there could be problem in the LD_LIBRARY_PATH.
    return try_blas_flag(["-lblas"])


def add_blas_configvars():
    config.add(
        "blas__ldflags",
        "lib[s] to include for [Fortran] level-3 blas implementation",
        StrParam(default_blas_ldflags),
        # Added elsewhere in the c key only when needed.
        in_c_key=False,
    )

    config.add(
        "blas__check_openmp",
        "Check for openmp library conflict.\nWARNING: Setting this to False leaves you open to wrong results in blas-related operations.",
        BoolParam(True),
        in_c_key=False,
    )


# Register config parameters that are specific to this module:
add_blas_configvars()

# using thread local instances for compilation API
GCC_Compiler = GCC_CompilerBase()
Compiler = CompilerBase()
