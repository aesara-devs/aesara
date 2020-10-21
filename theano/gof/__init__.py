"""Graph optimization framework"""

import theano
from theano.gof.cc import CLinker, DualLinker, HideC, OpWiseCLinker
from theano.gof.destroyhandler import DestroyHandler
from theano.gof.fg import FunctionGraph, InconsistencyError, MissingInputError
from theano.gof.graph import Apply, Constant, Variable, view_roots
from theano.gof.link import (
    Container,
    Linker,
    LocalLinker,
    PerformLinker,
    WrapLinker,
    WrapLinkerMany,
)
from theano.gof.op import (
    COp,
    Op,
    OpenMPOp,
    PureOp,
    get_test_value,
    ops_with_inner_function,
)
from theano.gof.opt import (
    CheckStackTraceOptimization,
    EquilibriumOptimizer,
    LocalOptGroup,
    LocalOptimizer,
    MergeOptimizer,
    NavigatorOptimizer,
    OpKeyOptimizer,
    OpRemove,
    OpSub,
    Optimizer,
    PatternSub,
    SeqOptimizer,
    TopoOptimizer,
    inplace_optimizer,
    local_optimizer,
    optimizer,
)
from theano.gof.optdb import DB, EquilibriumDB, LocalGroupDB, ProxyDB, Query, SequenceDB
from theano.gof.params_type import Params, ParamsType
from theano.gof.toolbox import (
    Bookkeeper,
    Feature,
    History,
    NodeFinder,
    NoOutputFromInplace,
    PrintListener,
    ReplacementDidntRemovedError,
    ReplaceValidate,
    Validator,
)
from theano.gof.type import CEnumType, EnumList, EnumType, Generic, Type, generic
from theano.gof.utils import MethodNotDefined, hashtype, object2


if theano.config.cmodule.preload_cache:
    from theano.gof.cc import get_module_cache

    get_module_cache()
