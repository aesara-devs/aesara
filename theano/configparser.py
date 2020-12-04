import configparser as ConfigParser
import hashlib
import logging
import os
import shlex
import sys
import typing
import warnings
from functools import wraps
from io import StringIO


_logger = logging.getLogger("theano.configparser")


class TheanoConfigWarning(Warning):
    def warn(cls, message, stacklevel=0):
        warnings.warn(message, cls, stacklevel=stacklevel + 3)

    warn = classmethod(warn)


def parse_config_string(config_string, issue_warnings=True):
    """
    Parses a config string (comma-separated key=value components) into a dict.
    """
    config_dict = {}
    my_splitter = shlex.shlex(config_string, posix=True)
    my_splitter.whitespace = ","
    my_splitter.whitespace_split = True
    for kv_pair in my_splitter:
        kv_pair = kv_pair.strip()
        if not kv_pair:
            continue
        kv_tuple = kv_pair.split("=", 1)
        if len(kv_tuple) == 1:
            if issue_warnings:
                TheanoConfigWarning.warn(
                    f"Config key '{kv_tuple[0]}' has no value, ignoring it",
                    stacklevel=1,
                )
        else:
            k, v = kv_tuple
            # subsequent values for k will override earlier ones
            config_dict[k] = v
    return config_dict


# THEANORC can contain a colon-delimited list of config files, like
# THEANORC=~lisa/.theanorc:~/.theanorc
# In that case, definitions in files on the right (here, ~/.theanorc) have
# precedence over those in files on the left.
def config_files_from_theanorc():
    rval = [
        os.path.expanduser(s)
        for s in os.getenv("THEANORC", "~/.theanorc").split(os.pathsep)
    ]
    if os.getenv("THEANORC") is None and sys.platform == "win32":
        # to don't need to change the filename and make it open easily
        rval.append(os.path.expanduser("~/.theanorc.txt"))
    return rval


class change_flags:
    """
    Use this as a decorator or context manager to change the value of
    Theano config variables.

    Useful during tests.
    """

    def __init__(self, args=(), **kwargs):
        confs = dict()
        args = dict(args)
        args.update(kwargs)
        for k in args:
            l = [v for v in _config_var_list if v.fullname == k]
            assert len(l) == 1, l
            confs[k] = l[0]
        self.confs = confs
        self.new_vals = args

    def __call__(self, f):
        @wraps(f)
        def res(*args, **kwargs):
            with self:
                return f(*args, **kwargs)

        return res

    def __enter__(self):
        self.old_vals = {}
        for k, v in self.confs.items():
            self.old_vals[k] = v.__get__(True, None)
        try:
            for k, v in self.confs.items():
                v.__set__(None, self.new_vals[k])
        except Exception:
            _logger.error(f"Failed to change flags for {self.confs}.")
            self.__exit__()
            raise

    def __exit__(self, *args):
        for k, v in self.confs.items():
            v.__set__(None, self.old_vals[k])


def fetch_val_for_key(key, delete_key=False):
    """Return the overriding config value for a key.
    A successful search returns a string value.
    An unsuccessful search raises a KeyError

    The (decreasing) priority order is:
    - THEANO_FLAGS
    - ~./theanorc

    """

    # first try to find it in the FLAGS
    if key in THEANO_FLAGS_DICT:
        if delete_key:
            return THEANO_FLAGS_DICT.pop(key)
        return THEANO_FLAGS_DICT[key]

    # next try to find it in the config file

    # config file keys can be of form option, or section__option
    key_tokens = key.rsplit("__", 1)
    if len(key_tokens) > 2:
        raise KeyError(key)

    if len(key_tokens) == 2:
        section, option = key_tokens
    else:
        section, option = "global", key
    try:
        try:
            return theano_cfg.get(section, option)
        except ConfigParser.InterpolationError:
            return theano_raw_cfg.get(section, option)
    except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
        raise KeyError(key)


def _config_print(thing, buf, print_doc=True):
    for cv in _config_var_list:
        print(cv, file=buf)
        if print_doc:
            print("    Doc: ", cv.doc, file=buf)
        print("    Value: ", cv.__get__(True, None), file=buf)
        print("", file=buf)


def _hash_from_code(msg):
    """This function was copied from theano.gof.utils to get rid of that import."""
    # hashlib.sha256() requires an object that supports buffer interface,
    # but Python 3 (unicode) strings don't.
    if isinstance(msg, str):
        msg = msg.encode()
    # Python 3 does not like module names that start with
    # a digit.
    return "m" + hashlib.sha256(msg).hexdigest()


def get_config_hash():
    """
    Return a string sha256 of the current config options. In the past,
    it was md5.

    The string should be such that we can safely assume that two different
    config setups will lead to two different strings.

    We only take into account config options for which `in_c_key` is True.
    """
    all_opts = sorted(
        [c for c in _config_var_list if c.in_c_key], key=lambda cv: cv.fullname
    )
    return _hash_from_code(
        "\n".join(
            ["{} = {}".format(cv.fullname, cv.__get__(True, None)) for cv in all_opts]
        )
    )


class TheanoConfigParser:
    # properties are installed by AddConfigVar
    _i_am_a_config_class = True

    def __str__(self, print_doc=True):
        sio = StringIO()
        _config_print(self.__class__, sio, print_doc=print_doc)
        return sio.getvalue()


class ConfigParam:
    """Base class of all kinds of configuration parameters.

    A ConfigParam has not only default values and configurable mutability, but
    also documentation text, as well as filtering and validation routines
    that can be context-dependent.
    """

    def __init__(
        self,
        default: typing.Union[object, typing.Callable[[object], object]],
        *,
        apply: typing.Optional[typing.Callable[[object], object]] = None,
        validate: typing.Optional[typing.Callable[[object], bool]] = None,
        mutable: bool = True,
    ):
        """
        Represents a configuration parameter and its associated casting and validation logic.

        Parameters
        ----------
        default : object or callable
            A default value, or function that returns a default value for this parameter.
        apply : callable, optional
            Callable that applies a modification to an input value during assignment.
            Typical use cases: type casting or expansion of '~' to user home directory.
        validate : callable, optional
            A callable that validates the parameter value during assignment.
            It may raise an (informative!) exception itself, or simply return True/False.
            For example to check the availability of a path, device or to restrict a float into a range.
        mutable : bool
            If mutable is False, the value of this config settings can not be changed at runtime.
        """
        self._default = default
        self._apply = apply
        self._validate = validate
        self._mutable = mutable
        self.is_default = True
        # set by AddConfigVar:
        self.fullname = None
        self.doc = None

        # Note that we do not call `self.filter` on the default value: this
        # will be done automatically in AddConfigVar, potentially with a
        # more appropriate user-provided default value.
        # Calling `filter` here may actually be harmful if the default value is
        # invalid and causes a crash or has unwanted side effects.
        super().__init__()

    @property
    def default(self):
        return self._default

    @property
    def mutable(self) -> bool:
        return self._mutable

    def apply(self, value):
        """Applies modifications to a parameter value during assignment.

        Typical use cases are casting or the subsitution of '~' with the user home directory.
        """
        if callable(self._apply):
            return self._apply(value)
        return value

    def validate(self, value) -> None:
        """Validates that a parameter values falls into a supported set or range.

        Raises
        ------
        ValueError
            when the validation turns out negative
        """
        if not callable(self._validate):
            return True
        if self._validate(value) is False:
            raise ValueError(
                f"Invalid value ({value}) for configuration variable '{self.fullname}'."
            )
        return True

    def __get__(self, cls, type_, delete_key=False):
        if cls is None:
            return self
        if not hasattr(self, "val"):
            try:
                val_str = fetch_val_for_key(self.fullname, delete_key=delete_key)
                self.is_default = False
            except KeyError:
                if callable(self.default):
                    val_str = self.default()
                else:
                    val_str = self.default
            self.__set__(cls, val_str)
        return self.val

    def __set__(self, cls, val):
        if not self.mutable and hasattr(self, "val"):
            raise Exception(
                "Can't change the value of {self.fullname} config parameter after initialization!"
            )
        applied = self.apply(val)
        self.validate(applied)
        self.val = applied


class EnumStr(ConfigParam):
    def __init__(
        self, default: str, options: typing.Sequence[str], validate=None, mutable=True
    ):
        """Creates a str-based parameter that takes a predefined set of options.

        Parameters
        ----------
        default : str
            The default setting.
        options : sequence
            Further str values that the parameter may take.
            May, but does not need to include the default.
        validate : callable
            See `ConfigParam`.
        mutable : callable
            See `ConfigParam`.
        """
        self.all = {default, *options}

        # All options should be strings
        for val in self.all:
            if not isinstance(val, str):
                raise ValueError(f"Non-str value '{val}' for an EnumStr parameter.")
        super().__init__(default, apply=self._apply, validate=validate, mutable=mutable)

    def _apply(self, val):
        if val in self.all:
            return val
        else:
            raise ValueError(
                f"Invalid value ('{val}') for configuration variable '{self.fullname}'. "
                f"Valid options are {self.all}"
            )

    def __str__(self):
        return f"{self.fullname} ({self.all}) "


class TypedParam(ConfigParam):
    def __str__(self):
        # The "_apply" callable is the type itself.
        return f"{self.fullname} ({self._apply}) "


class StrParam(TypedParam):
    def __init__(self, default, validate=None, mutable=True):
        super().__init__(default, apply=str, validate=validate, mutable=mutable)


class IntParam(TypedParam):
    def __init__(self, default, validate=None, mutable=True):
        super().__init__(default, apply=int, validate=validate, mutable=mutable)


class FloatParam(TypedParam):
    def __init__(self, default, validate=None, mutable=True):
        super().__init__(default, apply=float, validate=validate, mutable=mutable)


class BoolParam(TypedParam):
    """A boolean parameter that may be initialized from any of the following:
    False, 0, "false", "False", "0"
    True, 1, "true", "True", "1"
    """

    def __init__(self, default, validate=None, mutable=True):
        super().__init__(default, apply=self._apply, validate=validate, mutable=mutable)

    def _apply(self, value):
        if value in {False, 0, "false", "False", "0"}:
            return False
        elif value in {True, 1, "true", "True", "1"}:
            return True
        raise ValueError(
            f"Invalid value ({value}) for configuration variable '{self.fullname}'."
        )


class DeviceParam(ConfigParam):
    def __init__(self, default, *options, **kwargs):
        super().__init__(
            default, apply=self._apply, mutable=kwargs.get("mutable", True)
        )

    def _apply(self, val):
        if val == self.default or val.startswith("opencl") or val.startswith("cuda"):
            return val
        elif val.startswith("gpu"):
            raise ValueError(
                "You are tring to use the old GPU back-end. "
                "It was removed from Theano. Use device=cuda* now. "
                "See https://github.com/Theano/Theano/wiki/Converting-to-the-new-gpu-back-end%28gpuarray%29 "
                "for more information."
            )
        else:
            raise ValueError(
                'Invalid value ("{val}") for configuration '
                'variable "{self.fullname}". Valid options start with '
                'one of "cpu", "opencl" or "cuda".'
            )

    def __str__(self):
        return f"{self.fullname} ({self.default}, opencl*, cuda*) "


class ContextsParam(ConfigParam):
    def __init__(self):
        super().__init__("", apply=self._apply, mutable=False)

    def _apply(self, val):
        if val == "":
            return val
        for v in val.split(";"):
            s = v.split("->")
            if len(s) != 2:
                raise ValueError(f"Malformed context map: {v}")
            if s[0] == "cpu" or s[0].startswith("cuda") or s[0].startswith("opencl"):
                raise ValueError(f"Cannot use {s[0]} as context name")
        return val


# TODO: Not all of the following variables need to exist. Most should be private.
THEANO_FLAGS = os.getenv("THEANO_FLAGS", "")
# The THEANO_FLAGS environment variable should be a list of comma-separated
# [section__]option=value entries. If the section part is omitted, there should
# be only one section that contains the given option.
THEANO_FLAGS_DICT = parse_config_string(THEANO_FLAGS, issue_warnings=True)
_config_var_list = []

config_files = config_files_from_theanorc()
theano_cfg = ConfigParser.ConfigParser(
    {
        "USER": os.getenv("USER", os.path.split(os.path.expanduser("~"))[-1]),
        "LSCRATCH": os.getenv("LSCRATCH", ""),
        "TMPDIR": os.getenv("TMPDIR", ""),
        "TEMP": os.getenv("TEMP", ""),
        "TMP": os.getenv("TMP", ""),
        "PID": str(os.getpid()),
    }
)
theano_cfg.read(config_files)
# Having a raw version of the config around as well enables us to pass
# through config values that contain format strings.
# The time required to parse the config twice is negligible.
theano_raw_cfg = ConfigParser.RawConfigParser()
theano_raw_cfg.read(config_files)

# N.B. all instances of TheanoConfigParser give access to the same properties.
config = TheanoConfigParser()


def AddConfigVar(name, doc, configparam, root=config, in_c_key=True):
    """Add a new variable to `theano.config`.

    The data structure at work here is a tree of classes with class
    attributes/properties that are either a) instantiated dynamically-generated
    classes, or b) `ConfigParam` instances.  The root of this tree is the
    `TheanoConfigParser` class, and the internal nodes are the ``SubObj``
    classes created inside of `AddConfigVar`.

    Why this design?
    - The config object is a true singleton.  Every instance of
    `TheanoConfigParser` is an empty instance that looks up
    attributes/properties in the [single] ``TheanoConfigParser.__dict__``
    - The subtrees provide the same interface as the root
    - `ConfigParser` subclasses control get/set of config properties to guard
    against craziness.

    This method also performs some of the work of initializing `ConfigParam`
    instances

    Parameters
    ----------
    name: string
        The full name for this configuration variable. Takes the form
        ``"[section0__[section1__[etc]]]_option"``.
    doc: string
        A string that provides documentation for the config variable.
    configparam: ConfigParam
        An object for getting and setting this configuration parameter
    root: object
        Used for recursive calls.  Do not provide a value for this parameter.
    in_c_key: boolean
        If ``True``, then whenever this config option changes, the key
        associated to compiled C modules also changes, i.e. it may trigger a
        compilation of these modules (this compilation will only be partial if it
        turns out that the generated C code is unchanged). Set this option to False
        only if you are confident this option should not affect C code compilation.

    """

    if root is config:
        # Only set the name in the first call, not the recursive ones
        configparam.fullname = name
    if "." in name:
        raise ValueError(
            f"Dot-based sections were removed. Use double underscores! ({name})"
        )
    if hasattr(root, name):
        raise AttributeError(f"The name {configparam.fullname} is already taken")
    configparam.doc = doc
    configparam.in_c_key = in_c_key
    # Trigger a read of the value from config files and env vars
    # This allow to filter wrong value from the user.
    if not callable(configparam.default):
        configparam.__get__(root, type(root), delete_key=True)
    else:
        # We do not want to evaluate now the default value
        # when it is a callable.
        try:
            fetch_val_for_key(configparam.fullname)
            # The user provided a value, filter it now.
            configparam.__get__(root, type(root), delete_key=True)
        except KeyError:
            _logger.error(
                f"Suppressed KeyError in AddConfigVar for parameter '{name}' with fullname '{configparam.fullname}'!"
            )
    setattr(root.__class__, name, configparam)
    # TODO: After assigning the configvar to the "root" object, there should be
    # no reason to keep the _config_var_list around!
    _config_var_list.append(configparam)
