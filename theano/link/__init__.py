import sys

from theano.link.basic import (
    Container,
    Linker,
    LocalLinker,
    PerformLinker,
    WrapLinker,
    WrapLinkerMany,
    gc_helper,
    map_storage,
    streamline,
)
from theano.link.debugging import raise_with_op, set_excepthook


set_excepthook(handler=sys.stdout)
