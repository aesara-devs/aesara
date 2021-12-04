import copy
import math
import sys
from functools import cmp_to_key
from io import StringIO
from typing import Dict, Optional, Sequence, Union

from aesara.configdefaults import config
from aesara.graph import opt
from aesara.misc.ordered_set import OrderedSet
from aesara.utils import DefaultOrderedDict


OptimizersType = Union[opt.GlobalOptimizer, opt.LocalOptimizer]


class OptimizationDatabase:
    """A class that represents a collection/database of optimizations.

    These databases are used to logically organize collections of optimizers
    (i.e. ``GlobalOptimizer``s and ``LocalOptimizer``).
    """

    def __init__(self):
        self.__db__ = DefaultOrderedDict(OrderedSet)
        self._names = set()
        # This will be reset by `self.register` (via `obj.name` by the thing
        # doing the registering)
        self.name = None

    def register(
        self,
        name: str,
        optimizer: Union["OptimizationDatabase", OptimizersType],
        *tags: str,
        use_db_name_as_tag=True,
    ):
        """Register a new optimizer to the database.

        Parameters
        ----------
        name:
            Name of the optimizer.
        opt:
            The optimizer to register.
        tags:
            Tag name that allow to select the optimizer.
        use_db_name_as_tag:
            Add the database's name as a tag, so that its name can be used in a
            query.
            By default, all optimizations registered in ``EquilibriumDB`` are
            selected when the ``"EquilibriumDB"`` name is used as a tag. We do
            not want this behavior for some optimizers like
            ``local_remove_all_assert``. Setting `use_db_name_as_tag` to
            ``False`` removes that behavior. This mean only the optimizer name
            and the tags specified will enable that optimization.

        """
        if not isinstance(
            optimizer, (OptimizationDatabase, opt.GlobalOptimizer, opt.LocalOptimizer)
        ):
            raise TypeError(f"{optimizer} is not a valid optimizer type.")

        if name in self.__db__:
            raise ValueError(f"The tag '{name}' is already present in the database.")

        if use_db_name_as_tag:
            if self.name is not None:
                tags = tags + (self.name,)

        optimizer.name = name
        # This restriction is there because in many place we suppose that
        # something in the OptimizationDatabase is there only once.
        if optimizer.name in self.__db__:
            raise ValueError(
                f"Tried to register {optimizer.name} again under the new name {name}. "
                "The same optimization cannot be registered multiple times in"
                " an ``OptimizationDatabase``; use ProxyDB instead."
            )
        self.__db__[name] = OrderedSet([optimizer])
        self._names.add(name)
        self.__db__[optimizer.__class__.__name__].add(optimizer)
        self.add_tags(name, *tags)

    def add_tags(self, name, *tags):
        obj = self.__db__[name]
        assert len(obj) == 1
        obj = obj.copy().pop()
        for tag in tags:
            if tag in self._names:
                raise ValueError(
                    f"The tag '{tag}' for the {obj} collides with an existing name."
                )
            self.__db__[tag].add(obj)

    def remove_tags(self, name, *tags):
        obj = self.__db__[name]
        assert len(obj) == 1
        obj = obj.copy().pop()
        for tag in tags:
            if tag in self._names:
                raise ValueError(
                    f"The tag '{tag}' for the {obj} collides with an existing name."
                )
            self.__db__[tag].remove(obj)

    def __query__(self, q):
        # The ordered set is needed for deterministic optimization.
        variables = OrderedSet()
        for tag in q.include:
            variables.update(self.__db__[tag])
        for tag in q.require:
            variables.intersection_update(self.__db__[tag])
        for tag in q.exclude:
            variables.difference_update(self.__db__[tag])
        remove = OrderedSet()
        add = OrderedSet()
        for obj in variables:
            if isinstance(obj, OptimizationDatabase):
                def_sub_query = q
                if q.extra_optimizations:
                    def_sub_query = copy.copy(q)
                    def_sub_query.extra_optimizations = []
                sq = q.subquery.get(obj.name, def_sub_query)

                replacement = obj.query(sq)
                replacement.name = obj.name
                remove.add(obj)
                add.add(replacement)
        variables.difference_update(remove)
        variables.update(add)
        return variables

    def query(self, *tags, **kwtags):
        if len(tags) >= 1 and isinstance(tags[0], OptimizationQuery):
            if len(tags) > 1 or kwtags:
                raise TypeError(
                    "If the first argument to query is an `OptimizationQuery`,"
                    " there should be no other arguments."
                )
            return self.__query__(tags[0])
        include = [tag[1:] for tag in tags if tag.startswith("+")]
        require = [tag[1:] for tag in tags if tag.startswith("&")]
        exclude = [tag[1:] for tag in tags if tag.startswith("-")]
        if len(include) + len(require) + len(exclude) < len(tags):
            raise ValueError(
                "All tags must start with one of the following"
                " characters: '+', '&' or '-'"
            )
        return self.__query__(
            OptimizationQuery(
                include=include, require=require, exclude=exclude, subquery=kwtags
            )
        )

    def __getitem__(self, name):
        variables = self.__db__[name]
        if not variables:
            raise KeyError(f"Nothing registered for '{name}'")
        elif len(variables) > 1:
            raise ValueError(f"More than one match for {name} (please use query)")
        for variable in variables:
            return variable

    def __contains__(self, name):
        return name in self.__db__

    def print_summary(self, stream=sys.stdout):
        print(f"{self.__class__.__name__} (id {id(self)})", file=stream)
        print("  names", self._names, file=stream)
        print("  db", self.__db__, file=stream)


# This is deprecated and will be removed.
DB = OptimizationDatabase


class OptimizationQuery:
    """An object that specifies a set of optimizations by tag/name."""

    def __init__(
        self,
        include: Sequence[str],
        require: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
        subquery: Optional[Dict[str, "OptimizationQuery"]] = None,
        position_cutoff: float = math.inf,
        extra_optimizations: Optional[Sequence[OptimizersType]] = None,
    ):
        """

        Parameters
        ==========
        include:
            A set of tags such that every optimization obtained through this
            ``OptimizationQuery`` must have **one** of the tags listed. This
            field is required and basically acts as a starting point for the
            search.
        require:
            A set of tags such that every optimization obtained through this
            ``OptimizationQuery`` must have **all** of these tags.
        exclude:
            A set of tags such that every optimization obtained through this
            ``OptimizationQuery`` must have **none** of these tags.
        subquery:
            A dictionary mapping the name of a sub-database to a special
            ``OptimizationQuery``.  If no subquery is given for a sub-database,
            the original ``OptimizationQuery`` will be used again.
        position_cutoff:
            Only optimizations with position less than the cutoff are returned.
        extra_optimizations:
            Extra optimizations to be added.

        """
        self.include = OrderedSet(include)
        self.require = require or OrderedSet()
        self.exclude = exclude or OrderedSet()
        self.subquery = subquery or {}
        self.position_cutoff = position_cutoff
        self.name: Optional[str] = None
        if extra_optimizations is None:
            extra_optimizations = []
        self.extra_optimizations = extra_optimizations
        if isinstance(self.require, (list, tuple)):
            self.require = OrderedSet(self.require)
        if isinstance(self.exclude, (list, tuple)):
            self.exclude = OrderedSet(self.exclude)

    def __str__(self):
        return (
            "OptimizationQuery("
            + f"inc={self.include},ex={self.exclude},"
            + f"require={self.require},subquery={self.subquery},"
            + f"position_cutoff={self.position_cutoff},"
            + f"extra_opts={self.extra_optimizations})"
        )

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, "extra_optimizations"):
            self.extra_optimizations = []

    # add all opt with this tag
    def including(self, *tags):
        return OptimizationQuery(
            self.include.union(tags),
            self.require,
            self.exclude,
            self.subquery,
            self.position_cutoff,
            self.extra_optimizations,
        )

    # remove all opt with this tag
    def excluding(self, *tags):
        return OptimizationQuery(
            self.include,
            self.require,
            self.exclude.union(tags),
            self.subquery,
            self.position_cutoff,
            self.extra_optimizations,
        )

    # keep only opt with this tag.
    def requiring(self, *tags):
        return OptimizationQuery(
            self.include,
            self.require.union(tags),
            self.exclude,
            self.subquery,
            self.position_cutoff,
            self.extra_optimizations,
        )

    def register(self, *optimizations):
        return OptimizationQuery(
            self.include,
            self.require,
            self.exclude,
            self.subquery,
            self.position_cutoff,
            self.extra_optimizations + list(optimizations),
        )


# This is deprecated and will be removed.
Query = OptimizationQuery


class EquilibriumDB(OptimizationDatabase):
    """
    A set of potential optimizations which should be applied in an arbitrary
    order until equilibrium is reached.

    Canonicalize, Stabilize, and Specialize are all equilibrium optimizations.

    Parameters
    ----------
    ignore_newtrees
        If False, we will apply local opt on new node introduced during local
        optimization application. This could result in less fgraph iterations,
        but this doesn't mean it will be faster globally.

    tracks_on_change_inputs
        If True, we will re-apply local opt on nodes whose inputs
        changed during local optimization application. This could
        result in less fgraph iterations, but this doesn't mean it
        will be faster globally.

    Notes
    -----
    We can use `LocalOptimizer` and `GlobalOptimizer` since `EquilibriumOptimizer`
    supports both.

    It is probably not a good idea to have ignore_newtrees=False and
    tracks_on_change_inputs=True

    """

    def __init__(self, ignore_newtrees=True, tracks_on_change_inputs=False):
        """
        Parameters
        ==========
        ignore_newtrees:
            If False, we will apply local opt on new node introduced during local
            optimization application. This could result in less fgraph iterations,
            but this doesn't mean it will be faster globally.

        tracks_on_change_inputs:
            If True, we will re-apply local opt on nodes whose inputs
            changed during local optimization application. This could
            result in less fgraph iterations, but this doesn't mean it
            will be faster globally.
        """
        super().__init__()
        self.ignore_newtrees = ignore_newtrees
        self.tracks_on_change_inputs = tracks_on_change_inputs
        self.__final__ = {}
        self.__cleanup__ = {}

    def register(self, name, obj, *tags, final_opt=False, cleanup=False, **kwtags):
        if final_opt and cleanup:
            raise ValueError("`final_opt` and `cleanup` cannot both be true.")
        super().register(name, obj, *tags, **kwtags)
        self.__final__[name] = final_opt
        self.__cleanup__[name] = cleanup

    def query(self, *tags, **kwtags):
        _opts = super().query(*tags, **kwtags)
        final_opts = [o for o in _opts if self.__final__.get(o.name, False)]
        cleanup_opts = [o for o in _opts if self.__cleanup__.get(o.name, False)]
        opts = [o for o in _opts if o not in final_opts and o not in cleanup_opts]
        if len(final_opts) == 0:
            final_opts = None
        if len(cleanup_opts) == 0:
            cleanup_opts = None
        return opt.EquilibriumOptimizer(
            opts,
            max_use_ratio=config.optdb__max_use_ratio,
            ignore_newtrees=self.ignore_newtrees,
            tracks_on_change_inputs=self.tracks_on_change_inputs,
            failure_callback=opt.NavigatorOptimizer.warn_inplace,
            final_optimizers=final_opts,
            cleanup_optimizers=cleanup_opts,
        )


class SequenceDB(OptimizationDatabase):
    """A sequence of potential optimizations.

    Retrieve a sequence of optimizations (a SeqOptimizer) by calling query().

    Each potential optimization is registered with a floating-point position.
    No matter which optimizations are selected by a query, they are carried
    out in order of increasing position.

    The optdb itself (`aesara.compile.mode.optdb`), from which (among many
    other tags) fast_run and fast_compile optimizers are drawn is a SequenceDB.

    """

    seq_opt = opt.SeqOptimizer

    def __init__(self, failure_callback=opt.SeqOptimizer.warn):
        super().__init__()
        self.__position__ = {}
        self.failure_callback = failure_callback

    def register(self, name, obj, position: Union[str, int, float], *tags, **kwargs):
        super().register(name, obj, *tags, **kwargs)
        if position == "last":
            if len(self.__position__) == 0:
                self.__position__[name] = 0
            else:
                self.__position__[name] = max(self.__position__.values()) + 1
        elif isinstance(position, (int, float)):
            self.__position__[name] = position
        else:
            raise TypeError(f"`position` must be numeric; got {position}")

    def query(
        self, *tags, position_cutoff: Optional[Union[int, float]] = None, **kwtags
    ):
        """

        Parameters
        ----------
        position_cutoff : float or int
            Only optimizations with position less than the cutoff are returned.

        """
        opts = super().query(*tags, **kwtags)

        if position_cutoff is None:
            position_cutoff = config.optdb__position_cutoff

        position_dict = self.__position__

        if len(tags) >= 1 and isinstance(tags[0], OptimizationQuery):
            # the call to super should have raise an error with a good message
            assert len(tags) == 1
            if getattr(tags[0], "position_cutoff", None):
                position_cutoff = tags[0].position_cutoff

            # The OptimizationQuery instance might contain extra optimizations which need
            # to be added the the sequence of optimizations (don't alter the
            # original dictionary)
            if len(tags[0].extra_optimizations) > 0:
                position_dict = position_dict.copy()
                for extra_opt in tags[0].extra_optimizations:
                    # Give a name to the extra optimization (include both the
                    # class name for descriptiveness and id to avoid name
                    # collisions)
                    opt, position = extra_opt
                    opt.name = f"{opt.__class__}_{id(opt)}"

                    # Add the extra optimization to the optimization sequence
                    if position < position_cutoff:
                        opts.add(opt)
                        position_dict[opt.name] = position

        opts = [o for o in opts if position_dict[o.name] < position_cutoff]
        opts.sort(key=lambda obj: (position_dict[obj.name], obj.name))

        if self.failure_callback:
            ret = self.seq_opt(opts, failure_callback=self.failure_callback)
        else:
            ret = self.seq_opt(opts)

        if hasattr(tags[0], "name"):
            ret.name = tags[0].name
        return ret

    def print_summary(self, stream=sys.stdout):
        print(f"{self.__class__.__name__ } (id {id(self)})", file=stream)
        positions = list(self.__position__.items())

        def c(a, b):
            return (a[1] > b[1]) - (a[1] < b[1])

        positions.sort(key=cmp_to_key(c))

        print("\tposition", positions, file=stream)
        print("\tnames", self._names, file=stream)
        print("\tdb", self.__db__, file=stream)

    def __str__(self):
        sio = StringIO()
        self.print_summary(sio)
        return sio.getvalue()


class LocalGroupDB(SequenceDB):
    """
    Generate a local optimizer of type LocalOptGroup instead
    of a global optimizer.

    It supports the tracks, to only get applied to some Op.

    """

    def __init__(
        self,
        apply_all_opts: bool = False,
        profile: bool = False,
        local_opt=opt.LocalOptGroup,
    ):
        super().__init__(failure_callback=None)
        self.apply_all_opts = apply_all_opts
        self.profile = profile
        self.local_opt = local_opt
        self.__name__: str = ""

    def register(self, name, obj, *tags, position="last", **kwargs):
        super().register(name, obj, position, *tags, **kwargs)

    def query(self, *tags, **kwtags):
        opts = list(super().query(*tags, **kwtags))
        ret = self.local_opt(
            *opts, apply_all_opts=self.apply_all_opts, profile=self.profile
        )
        return ret


class TopoDB(OptimizationDatabase):
    """Generate a `GlobalOptimizer` of type TopoOptimizer."""

    def __init__(
        self, db, order="in_to_out", ignore_newtrees=False, failure_callback=None
    ):
        super().__init__()
        self.db = db
        self.order = order
        self.ignore_newtrees = ignore_newtrees
        self.failure_callback = failure_callback

    def query(self, *tags, **kwtags):
        return opt.TopoOptimizer(
            self.db.query(*tags, **kwtags),
            self.order,
            self.ignore_newtrees,
            self.failure_callback,
        )


class ProxyDB(OptimizationDatabase):
    """A object that wraps an existing ``OptimizationDatabase``.

    This is needed because we can't register the same ``OptimizationDatabase``
    multiple times in different positions in a ``SequentialDB``.

    """

    def __init__(self, db):
        if not isinstance(db, OptimizationDatabase):
            raise TypeError("`db` must be an `OptimizationDatabase`.")

        self.db = db

    def query(self, *tags, **kwtags):
        return self.db.query(*tags, **kwtags)
