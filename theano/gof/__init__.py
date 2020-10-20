"""Graph optimization framework"""


from theano.gof.cc import CLinker, OpWiseCLinker, DualLinker, HideC

from theano.gof.fg import (
    InconsistencyError,
    MissingInputError,
    FunctionGraph,
)

from theano.gof.destroyhandler import DestroyHandler

from theano.gof.graph import Apply, Variable, Constant, view_roots

from theano.gof.link import (
    Container,
    Linker,
    LocalLinker,
    PerformLinker,
    WrapLinker,
    WrapLinkerMany,
)

from theano.gof.op import (
    Op,
    OpenMPOp,
    PureOp,
    COp,
    ops_with_inner_function,
    get_test_value,
)

from theano.gof.type import EnumType, EnumList, CEnumType

from theano.gof.opt import (
    Optimizer,
    optimizer,
    inplace_optimizer,
    SeqOptimizer,
    MergeOptimizer,
    LocalOptimizer,
    local_optimizer,
    LocalOptGroup,
    OpSub,
    OpRemove,
    PatternSub,
    NavigatorOptimizer,
    TopoOptimizer,
    EquilibriumOptimizer,
    OpKeyOptimizer,
    CheckStackTraceOptimization,
)

from theano.gof.optdb import DB, LocalGroupDB, Query, EquilibriumDB, SequenceDB, ProxyDB

from theano.gof.toolbox import (
    Feature,
    Bookkeeper,
    History,
    Validator,
    ReplaceValidate,
    NodeFinder,
    PrintListener,
    ReplacementDidntRemovedError,
    NoOutputFromInplace,
)

from theano.gof.type import Type, Generic, generic

from theano.gof.utils import hashtype, object2, MethodNotDefined

from theano.gof.params_type import ParamsType, Params

import theano

if theano.config.cmodule.preload_cache:
    from theano.gof.cc import get_module_cache

    get_module_cache()
