from aesara.configdefaults import config
from aesara.link.c.exceptions import MissingGXX
from aesara.link.vm import VM


try:
    # If cxx is explicitly set to an empty string, we do not want to import
    # either lazy-linker C code or lazy-linker compiled C code from the cache.
    if not config.cxx:
        raise MissingGXX(
            "lazylinker will not be imported if aesara.config.cxx is not set."
        )
    from aesara.link.c import lazylinker_c

    class CVM(lazylinker_c.CLazyLinker, VM):
        def __init__(self, fgraph, *args, **kwargs):
            self.fgraph = fgraph
            lazylinker_c.CLazyLinker.__init__(self, *args, **kwargs)
            # skip VM.__init__


except ImportError:
    pass
except (OSError, MissingGXX):
    # OSError happens when g++ is not installed.  In that case, we
    # already changed the default linker to something else then CVM.
    # Currently this is the py linker.
    # Here we assert that the default linker is not cvm.
    if config._config_var_dict["linker"].default.startswith("cvm"):
        raise
