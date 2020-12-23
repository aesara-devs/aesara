from theano.gof.utils import MethodNotDefined


class CLinkerObject:
    """
    Standard elements of an Op or Type used with the CLinker.

    """

    def c_headers(self):
        """
        Optional: Return a list of header files required by code returned by
        this class.

        Examples
        --------
        return ['<iostream>', '<math.h>', '/full/path/to/header.h']

        These strings will be prefixed with "#include " and inserted at the
        beginning of the c source code.

        Strings in this list that start neither with '<' nor '"' will be
        enclosed in double-quotes.

        Raises
        ------
        MethodNotDefined
            Subclass does not implement this method.

        """
        raise MethodNotDefined("c_headers", type(self), self.__class__.__name__)

    def c_header_dirs(self):
        """
        Optional: Return a list of header search paths required by code
        returned by this class.

        Examples
        --------
        return ['/usr/local/include', '/opt/weirdpath/src/include']

        Provides search paths for headers, in addition to those in any relevant
        environment variables.

        Hint: for unix compilers, these are the things that get '-I' prefixed
        in the compiler cmdline.

        Raises
        ------
        MethodNotDefined
            Subclass does not implement this method.

        """
        raise MethodNotDefined("c_header_dirs", type(self), self.__class__.__name__)

    def c_libraries(self):
        """
        Optional: Return a list of libraries required by code returned by
        this class.

        Examples
        --------
        return ['gsl', 'gslcblas', 'm', 'fftw3', 'g2c'].

        The compiler will search the directories specified by the environment
        variable LD_LIBRARY_PATH in addition to any returned by `c_lib_dirs`.

        Hint: for unix compilers, these are the things that get '-l' prefixed
        in the compiler cmdline.

        Raises
        ------
        MethodNotDefined
            Subclass does not implement this method.

        """
        raise MethodNotDefined("c_libraries", type(self), self.__class__.__name__)

    def c_lib_dirs(self):
        """
        Optional: Return a list of library search paths required by code
        returned by this class.

        Examples
        --------
        return ['/usr/local/lib', '/opt/weirdpath/build/libs'].

        Provides search paths for libraries, in addition to those in any
        relevant environment variables (e.g. LD_LIBRARY_PATH).

        Hint: for unix compilers, these are the things that get '-L' prefixed
        in the compiler cmdline.

        Raises
        ------
        MethodNotDefined
            Subclass does not implement this method.

        """
        raise MethodNotDefined("c_lib_dirs", type(self), self.__class__.__name__)

    def c_support_code(self):
        """
        Optional: Return utility code (a string, or a list of strings) for use by a `Variable` or `Op` to be
        included at global scope prior to the rest of the code for this class.

        QUESTION: How many times will this support code be emitted for a graph
        with many instances of the same type?

        Raises
        ------
        MethodNotDefined
            Subclass does not implement this method.

        """
        raise MethodNotDefined("c_support_code", type(self), self.__class__.__name__)

    def c_code_cache_version(self):
        """
        Return a tuple of integers indicating the version of this Op.

        An empty tuple indicates an 'unversioned' Op that will not be cached
        between processes.

        The cache mechanism may erase cached modules that have been superceded
        by newer versions. See `ModuleCache` for details.

        See Also
        --------
        c_code_cache_version_apply()

        """
        return ()

    def c_compile_args(self):
        """
        Optional: Return a list of compile args recommended to compile the
        code returned by other methods in this class.

        Examples
        --------
        return ['-ffast-math']

        Compiler arguments related to headers, libraries and search paths should
        be provided via the functions `c_headers`, `c_libraries`,
        `c_header_dirs`, and `c_lib_dirs`.

        Raises
        ------
        MethodNotDefined
            Subclass does not implement this method.

        """
        raise MethodNotDefined("c_compile_args", type(self), self.__class__.__name__)

    def c_no_compile_args(self):
        """
        Optional: return a list of incompatible gcc compiler arguments.

        We will remove those arguments from the command line of gcc. So if
        another Op adds a compile arg in the graph that is incompatible
        with this Op, the incompatible arg will not be used.
        Useful for instance to remove -ffast-math.

        EXAMPLE

        WRITEME

        Raises
        ------
        MethodNotDefined
            The subclass does not override this method.

        """
        raise MethodNotDefined("c_no_compile_args", type(self), self.__class__.__name__)

    def c_init_code(self):
        """
        Optional: return a list of code snippets to be inserted in module
        initialization.

        Raises
        ------
        MethodNotDefined
            The subclass does not override this method.

        """
        raise MethodNotDefined("c_init_code", type(self), self.__class__.__name__)


class CLinkerOp(CLinkerObject):
    """
    Interface definition for `Op` subclasses compiled by `CLinker`.

    A subclass should implement WRITEME.

    WRITEME: structure of automatically generated C code.
    Put this in doc/code_structure.txt

    """

    def c_code(self, node, name, inputs, outputs, sub):
        """
        Required: return the C implementation of an Op.

        Returns C code that does the computation associated to this `Op`,
        given names for the inputs and outputs.

        Parameters
        ----------
        node : Apply instance
            The node for which we are compiling the current c_code.
           The same Op may be used in more than one node.
        name : str
            A name that is automatically assigned and guaranteed to be
            unique.
        inputs : list of strings
            There is a string for each input of the function, and the
            string is the name of a C variable pointing to that input.
            The type of the variable depends on the declared type of
            the input.  There is a corresponding python variable that
            can be accessed by prepending "py_" to the name in the
            list.
        outputs : list of strings
            Each string is the name of a C variable where the Op should
            store its output.  The type depends on the declared type of
            the output.  There is a corresponding python variable that
            can be accessed by prepending "py_" to the name in the
            list.  In some cases the outputs will be preallocated and
            the value of the variable may be pre-filled.  The value for
            an unallocated output is type-dependent.
        sub : dict of strings
            Extra symbols defined in `CLinker` sub symbols (such as 'fail').
            WRITEME

        Raises
        ------
        MethodNotDefined
            The subclass does not override this method.

        """
        raise MethodNotDefined(f"{self.__class__.__name__}.c_code")

    def c_code_cache_version_apply(self, node):
        """
        Return a tuple of integers indicating the version of this Op.

        An empty tuple indicates an 'unversioned' Op that will not be
        cached between processes.

        The cache mechanism may erase cached modules that have been
        superceded by newer versions.  See `ModuleCache` for details.

        See Also
        --------
        c_code_cache_version()

        Notes
        -----
            This function overrides `c_code_cache_version` unless it explicitly
            calls `c_code_cache_version`. The default implementation simply
            calls `c_code_cache_version` and ignores the `node` argument.

        """
        return self.c_code_cache_version()

    def c_code_cleanup(self, node, name, inputs, outputs, sub):
        """
        Optional: return C code to run after c_code, whether it failed or not.

        This is a convenient place to clean up things allocated by c_code().

        Parameters
        ----------
        node : Apply instance
            WRITEME
        name : str
            A name that is automatically assigned and guaranteed to be
            unique.
        inputs : list of strings
            There is a string for each input of the function, and the
            string is the name of a C variable pointing to that input.
            The type of the variable depends on the declared type of
            the input. There is a corresponding python variable that
            can be accessed by prepending "py_" to the name in the
            list.
        outputs : list of strings
            Each string is the name of a C variable correspoinding to
            one of the outputs of the Op. The type depends on the
            declared type of the output. There is a corresponding
            python variable that can be accessed by prepending "py_" to
            the name in the list.
        sub : dict of strings
            extra symbols defined in `CLinker` sub symbols (such as 'fail').
            WRITEME

        Raises
        ------
        MethodNotDefined
            The subclass does not override this method.

        """
        raise MethodNotDefined(f"{self.__class__.__name__}.c_code_cleanup")

    def c_support_code_apply(self, node, name):
        """
        Optional: return utility code for use by an `Op` that will be
        inserted at global scope, that can be specialized for the
        support of a particular `Apply` node.

        Parameters
        ----------
        node: an Apply instance in the graph being compiled
        name: str
            A string or number that serves to uniquely identify this node.
            Symbol names defined by this support code should include the name,
            so that they can be called from the c_code, and so that they do not
            cause name collisions.

        Notes
        -----
        This function is called in addition to c_support_code and will
        supplement whatever is returned from there.

        Raises
        ------
        MethodNotDefined
            Subclass does not implement this method.

        """
        raise MethodNotDefined(
            "c_support_code_apply", type(self), self.__class__.__name__
        )

    def c_init_code_apply(self, node, name):
        """
        Optional: return a code string specific to the apply
        to be inserted in the module initialization code.

        Parameters
        ----------
        node : an Apply instance in the graph being compiled
        name : str
            A string or number that serves to uniquely identify this node.
            Symbol names defined by this support code should include the name,
            so that they can be called from the c_code, and so that they do not
            cause name collisions.

        Notes
        -----
        This function is called in addition to c_init_code and will supplement
        whatever is returned from there.

        Raises
        ------
        MethodNotDefined
            The subclass does not override this method.

        """
        raise MethodNotDefined("c_init_code_apply", type(self), self.__class__.__name__)

    def c_init_code_struct(self, node, name, sub):
        """
        Optional: return a code string specific to the apply
        to be inserted in the struct initialization code.

        Parameters
        ----------
        node : an Apply instance in the graph being compiled
        name : str
            A unique name to distinguish variables from those of other nodes.
        sub
            A dictionary of values to substitute in the code.
            Most notably it contains a 'fail' entry that you should place in
            your code after setting a python exception to indicate an error.

        Raises
        ------
        MethodNotDefined
            The subclass does not override this method.

        """
        raise MethodNotDefined(
            "c_init_code_struct", type(self), self.__class__.__name__
        )

    def c_support_code_struct(self, node, name):
        """
        Optional: return utility code for use by an `Op` that will be
        inserted at struct scope, that can be specialized for the
        support of a particular `Apply` node.

        Parameters
        ----------
        node : an Apply instance in the graph being compiled
        name : str
            A unique name to distinguish you variables from those of other
            nodes.

        Raises
        ------
        MethodNotDefined
            Subclass does not implement this method.

        """
        raise MethodNotDefined(
            "c_support_code_struct", type(self), self.__class__.__name__
        )

    def c_cleanup_code_struct(self, node, name):
        """
        Optional: return a code string specific to the apply to be
        inserted in the struct cleanup code.

        Parameters
        ----------
        node : an Apply instance in the graph being compiled
        name : str
            A unique name to distinguish variables from those of other nodes.

        Raises
        ------
        MethodNotDefined
            The subclass does not override this method.

        """
        raise MethodNotDefined(
            "c_cleanup_code_struct", type(self), self.__class__.__name__
        )


class CLinkerType(CLinkerObject):
    """
    Interface specification for Types that can be arguments to a `CLinkerOp`.

    A CLinkerType instance is mainly responsible  for providing the C code that
    interfaces python objects with a C `CLinkerOp` implementation.

    See WRITEME for a general overview of code generation by `CLinker`.

    """

    def c_element_type(self):
        """
        Optional: Return the name of the primitive C type of items into variables
        handled by this type.

        e.g:

         - For ``TensorType(dtype='int64', ...)``: should return ``"npy_int64"``.
         - For ``GpuArrayType(dtype='int32', ...)``: should return ``"ga_int"``.

        """
        raise MethodNotDefined("c_element_type", type(self), self.__class__.__name__)

    def c_is_simple(self):
        """
        Optional: Return True for small or builtin C types.

        A hint to tell the compiler that this type is a builtin C type or a
        small struct and that its memory footprint is negligible. Simple
        objects may be passed on the stack.

        """
        return False

    def c_literal(self, data):
        """
        Optional: WRITEME

        Parameters
        ----------
        data : WRITEME
            WRITEME

        Raises
        ------
        MethodNotDefined
            Subclass does not implement this method.

        """
        raise MethodNotDefined("c_literal", type(self), self.__class__.__name__)

    def c_declare(self, name, sub, check_input=True):
        """
        Required: Return c code to declare variables that will be
        instantiated by `c_extract`.

        Parameters
        ----------
        name: str
            The name of the ``PyObject *`` pointer that will
            the value for this Type
        sub: dict string -> string
            a dictionary of special codes.  Most importantly
            sub['fail']. See CLinker for more info on `sub` and ``fail``.

        Notes
        -----
        It is important to include the `name` inside of variables which
        are declared here, so that name collisions do not occur in the
        source file that is generated.

        The variable called ``name`` is not necessarily defined yet
        where this code is inserted. This code might be inserted to
        create class variables for example, whereas the variable ``name``
        might only exist inside certain functions in that class.

        TODO: Why should variable declaration fail?  Is it even allowed to?

        Raises
        ------
        MethodNotDefined
            Subclass does not implement this method.

        Examples
        --------
        .. code-block: python

            return "PyObject ** addr_of_%(name)s;"

        """
        raise MethodNotDefined()

    def c_init(self, name, sub):
        """
        Required: Return c code to initialize the variables that were declared
        by self.c_declare().

        Notes
        -----
        The variable called ``name`` is not necessarily defined yet
        where this code is inserted. This code might be inserted in a
        class constructor for example, whereas the variable ``name``
        might only exist inside certain functions in that class.

        TODO: Why should variable initialization fail?  Is it even allowed to?

        Examples
        --------
        .. code-block: python

            return "addr_of_%(name)s = NULL;"

        """
        raise MethodNotDefined("c_init", type(self), self.__class__.__name__)

    def c_extract(self, name, sub, check_input=True):
        """
        Required: Return c code to extract a PyObject * instance.

        The code returned from this function must be templated using
        ``%(name)s``, representing the name that the caller wants to
        call this `Variable`. The Python object self.data is in a
        variable called "py_%(name)s" and this code must set the
        variables declared by c_declare to something representative
        of py_%(name)s. If the data is improper, set an appropriate
        exception and insert "%(fail)s".

        TODO: Point out that template filling (via sub) is now performed
              by this function. --jpt

        Parameters
        ----------
        name : str
            The name of the ``PyObject *`` pointer that will
            store the value for this Type.
        sub : dict string -> string
            A dictionary of special codes. Most importantly
            sub['fail']. See CLinker for more info on `sub` and ``fail``.

        Raises
        ------
        MethodNotDefined
            Subclass does not implement this method.

        Examples
        --------
        .. code-block: python

            return "if (py_%(name)s == Py_None)" + \\\
                        addr_of_%(name)s = &py_%(name)s;" + \\\
                   "else" + \\\
                   { PyErr_SetString(PyExc_ValueError, \\\
                        'was expecting None'); %(fail)s;}"

        """
        raise MethodNotDefined("c_extract", type(self), self.__class__.__name__)

    def c_extract_out(self, name, sub, check_input=True):
        """
        Optional: C code to extract a PyObject * instance.

        Unlike c_extract, c_extract_out has to accept Py_None,
        meaning that the variable should be left uninitialized.

        """
        return """
        if (py_%(name)s == Py_None)
        {
            %(c_init_code)s
        }
        else
        {
            %(c_extract_code)s
        }
        """ % dict(
            name=name,
            c_init_code=self.c_init(name, sub),
            c_extract_code=self.c_extract(name, sub, check_input),
        )

    def c_cleanup(self, name, sub):
        """
        Return C code to clean up after `c_extract`.

        This returns C code that should deallocate whatever `c_extract`
        allocated or decrease the reference counts. Do not decrease
        py_%(name)s's reference count.

        WRITEME

        Parameters
        ----------
        name : WRITEME
            WRITEME
        sub : WRITEME
            WRITEME

        Raises
        ------
         MethodNotDefined
            Subclass does not implement this method.

        """
        raise MethodNotDefined()

    def c_sync(self, name, sub):
        """
        Required: Return C code to pack C types back into a PyObject.

        The code returned from this function must be templated using
        "%(name)s", representing the name that the caller wants to
        call this Variable. The returned code may set "py_%(name)s"
        to a PyObject* and that PyObject* will be accessible from
        Python via variable.data. Do not forget to adjust reference
        counts if "py_%(name)s" is changed from its original value.

        Parameters
        ----------
        name : WRITEME
            WRITEME
        sub : WRITEME
            WRITEME

        Raises
        ------
        MethodNotDefined
            Subclass does not implement this method.

        """
        raise MethodNotDefined("c_sync", type(self), self.__class__.__name__)

    def c_code_cache_version(self):
        """
        Return a tuple of integers indicating the version of this Type.

        An empty tuple indicates an 'unversioned' Type that will not
        be cached between processes.

        The cache mechanism may erase cached modules that have been
        superceded by newer versions. See `ModuleCache` for details.

        """
        return ()


class HideC(CLinkerOp):
    def __hide(*args):
        raise MethodNotDefined()

    c_code = __hide
    c_code_cleanup = __hide

    c_headers = __hide
    c_header_dirs = __hide
    c_libraries = __hide
    c_lib_dirs = __hide

    c_support_code = __hide
    c_support_code_apply = __hide

    c_compile_args = __hide
    c_no_compile_args = __hide
    c_init_code = __hide
    c_init_code_apply = __hide

    c_init_code_struct = __hide
    c_support_code_struct = __hide
    c_cleanup_code_struct = __hide

    def c_code_cache_version(self):
        return ()

    def c_code_cache_version_apply(self, node):
        return self.c_code_cache_version()
