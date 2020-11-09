"""Graph optimization framework"""

import aesara
from aesara.gof.cc import CLinker, DualLinker, HideC, OpWiseCLinker
from aesara.gof.destroyhandler import DestroyHandler
from aesara.gof.fg import FunctionGraph, InconsistencyError, MissingInputError
from aesara.gof.graph import Apply, Constant, Variable, view_roots
from aesara.gof.link import (
    Container,
    Linker,
    LocalLinker,
    PerformLinker,
    WrapLinker,
    WrapLinkerMany,
)
from aesara.gof.op import (
    COp,
    Op,
    OpenMPOp,
    PureOp,
    get_test_value,
    ops_with_inner_function,
)
from aesara.gof.opt import (
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
from aesara.gof.optdb import DB, EquilibriumDB, LocalGroupDB, ProxyDB, Query, SequenceDB
from aesara.gof.params_type import Params, ParamsType
from aesara.gof.toolbox import (
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
from aesara.gof.type import CEnumType, EnumList, EnumType, Generic, Type, generic
from aesara.gof.utils import MethodNotDefined, hashtype, object2


if aesara.config.cmodule.preload_cache:
    from aesara.gof.cc import get_module_cache

    get_module_cache()
