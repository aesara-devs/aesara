"""Graph optimization framework"""

import theano
from theano.gof.destroyhandler import DestroyHandler
from theano.gof.fg import FunctionGraph, InconsistencyError, MissingInputError
from theano.gof.graph import Apply, Constant, Variable, view_roots
from theano.gof.op import (
    ExternalCOp,
    Op,
    OpenMPOp,
    get_test_value,
    ops_with_inner_function,
)
from theano.gof.opt import (
    CheckStackTraceOptimization,
    EquilibriumOptimizer,
    GlobalOptimizer,
    LocalOptGroup,
    LocalOptimizer,
    MergeOptimizer,
    NavigatorOptimizer,
    OpKeyOptimizer,
    OpRemove,
    OpSub,
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
    ReplacementDidNotRemoveError,
    ReplaceValidate,
    Validator,
)
from theano.gof.type import CEnumType, EnumList, EnumType, Generic, Type, generic
from theano.gof.utils import MetaObject, MethodNotDefined
