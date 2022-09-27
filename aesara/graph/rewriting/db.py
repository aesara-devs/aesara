import copy
import math
import sys
from functools import cmp_to_key
from io import StringIO
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

from aesara.configdefaults import config
from aesara.graph.rewriting import basic as aesara_rewriting
from aesara.misc.ordered_set import OrderedSet
from aesara.utils import DefaultOrderedDict


RewritesType = Union[aesara_rewriting.GraphRewriter, aesara_rewriting.NodeRewriter]


class RewriteDatabase:
    r"""A class that represents a collection/database of rewrites.

    These databases are used to logically organize collections of rewrites
    (i.e. `GraphRewriter`\s and `NodeRewriter`).
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
        rewriter: Union["RewriteDatabase", RewritesType],
        *tags: str,
        use_db_name_as_tag=True,
    ):
        """Register a new rewriter to the database.

        Parameters
        ----------
        name:
            Name of the rewriter.
        rewriter:
            The rewriter to register.
        tags:
            Tag name that allows one to select the rewrite using a
            `RewriteDatabaseQuery`.
        use_db_name_as_tag:
            Add the database's name as a tag, so that its name can be used in a
            query.
            By default, all rewrites registered to an `EquilibriumDB` are
            selected when the ``"EquilibriumDB"`` name is used as a tag. We do
            not want this behavior for some rewrites like
            ``local_remove_all_assert``. Setting `use_db_name_as_tag` to
            ``False`` removes that behavior. This means that only the rewrite's name
            and/or its tags will enable it.

        """
        if not isinstance(
            rewriter,
            (
                RewriteDatabase,
                aesara_rewriting.GraphRewriter,
                aesara_rewriting.NodeRewriter,
            ),
        ):
            raise TypeError(f"{rewriter} is not a valid rewrite type.")

        if name in self.__db__:
            raise ValueError(f"The tag '{name}' is already present in the database.")

        if use_db_name_as_tag:
            if self.name is not None:
                tags = tags + (self.name,)

        rewriter.name = name
        # This restriction is there because in many place we suppose that
        # something in the RewriteDatabase is there only once.
        if rewriter.name in self.__db__:
            raise ValueError(
                f"Tried to register {rewriter.name} again under the new name {name}. "
                "The same rewrite cannot be registered multiple times in"
                " an `RewriteDatabase`; use `ProxyDB` instead."
            )
        self.__db__[name] = OrderedSet([rewriter])
        self._names.add(name)
        self.__db__[rewriter.__class__.__name__].add(rewriter)
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
        # The ordered set is needed for deterministic rewriting.
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
            if isinstance(obj, RewriteDatabase):
                def_sub_query = q
                if q.extra_rewrites:
                    def_sub_query = copy.copy(q)
                    def_sub_query.extra_rewrites = []
                sq = q.subquery.get(obj.name, def_sub_query)

                replacement = obj.query(sq)
                replacement.name = obj.name
                remove.add(obj)
                add.add(replacement)
        variables.difference_update(remove)
        variables.update(add)
        return variables

    def query(self, *tags, **kwtags):
        if len(tags) >= 1 and isinstance(tags[0], RewriteDatabaseQuery):
            if len(tags) > 1 or kwtags:
                raise TypeError(
                    "If the first argument to query is an `RewriteDatabaseQuery`,"
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
            RewriteDatabaseQuery(
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


class RewriteDatabaseQuery:
    """An object that specifies a set of rewrites by tag/name."""

    def __init__(
        self,
        include: Iterable[Union[str, None]],
        require: Optional[Union[OrderedSet, Sequence[str]]] = None,
        exclude: Optional[Union[OrderedSet, Sequence[str]]] = None,
        subquery: Optional[Dict[str, "RewriteDatabaseQuery"]] = None,
        position_cutoff: float = math.inf,
        extra_rewrites: Optional[
            Sequence[
                Tuple[Union["RewriteDatabaseQuery", RewritesType], Union[int, float]]
            ]
        ] = None,
    ):
        """

        Parameters
        ==========
        include:
            A set of tags such that every rewirte obtained through this
            `RewriteDatabaseQuery` must have **one** of the tags listed. This
            field is required and basically acts as a starting point for the
            search.
        require:
            A set of tags such that every rewrite obtained through this
            `RewriteDatabaseQuery` must have **all** of these tags.
        exclude:
            A set of tags such that every rewrite obtained through this
            ``RewriteDatabaseQuery` must have **none** of these tags.
        subquery:
            A dictionary mapping the name of a sub-database to a special
            `RewriteDatabaseQuery`.  If no subquery is given for a sub-database,
            the original `RewriteDatabaseQuery` will be used again.
        position_cutoff:
            Only rewrites with position less than the cutoff are returned.
        extra_rewrites:
            Extra rewrites to be added.

        """
        self.include = OrderedSet(include)
        self.require = OrderedSet(require) if require else OrderedSet()
        self.exclude = OrderedSet(exclude) if exclude else OrderedSet()
        self.subquery = subquery or {}
        self.position_cutoff = position_cutoff
        self.name: Optional[str] = None
        if extra_rewrites is None:
            extra_rewrites = []
        self.extra_rewrites = list(extra_rewrites)

    def __str__(self):
        return (
            "RewriteDatabaseQuery("
            + f"inc={self.include},ex={self.exclude},"
            + f"require={self.require},subquery={self.subquery},"
            + f"position_cutoff={self.position_cutoff},"
            + f"extra_rewrites={self.extra_rewrites})"
        )

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, "extra_rewrites"):
            self.extra_rewrites = []

    def including(self, *tags: str) -> "RewriteDatabaseQuery":
        """Add rewrites with the given tags."""
        return RewriteDatabaseQuery(
            self.include.union(tags),
            self.require,
            self.exclude,
            self.subquery,
            self.position_cutoff,
            self.extra_rewrites,
        )

    def excluding(self, *tags: str) -> "RewriteDatabaseQuery":
        """Remove rewrites with the given tags."""
        return RewriteDatabaseQuery(
            self.include,
            self.require,
            self.exclude.union(tags),
            self.subquery,
            self.position_cutoff,
            self.extra_rewrites,
        )

    def requiring(self, *tags: str) -> "RewriteDatabaseQuery":
        """Filter for rewrites with the given tags."""
        return RewriteDatabaseQuery(
            self.include,
            self.require.union(tags),
            self.exclude,
            self.subquery,
            self.position_cutoff,
            self.extra_rewrites,
        )

    def register(
        self, *rewrites: Tuple["RewriteDatabaseQuery", Union[int, float]]
    ) -> "RewriteDatabaseQuery":
        """Include the given rewrites."""
        return RewriteDatabaseQuery(
            self.include,
            self.require,
            self.exclude,
            self.subquery,
            self.position_cutoff,
            self.extra_rewrites + list(rewrites),
        )


class EquilibriumDB(RewriteDatabase):
    """A database of rewrites that should be applied until equilibrium is reached.

    Canonicalize, Stabilize, and Specialize are all equilibrium rewriters.

    Notes
    -----
    We can use `NodeRewriter` and `GraphRewriter` since `EquilibriumGraphRewriter`
    supports both.

    It is probably not a good idea to have both ``ignore_newtrees == False``
    and ``tracks_on_change_inputs == True``.

    """

    def __init__(
        self, ignore_newtrees: bool = True, tracks_on_change_inputs: bool = False
    ):
        """

        Parameters
        ----------
        ignore_newtrees
            If ``False``, apply rewrites to new nodes introduced during
            rewriting.

        tracks_on_change_inputs
            If ``True``, re-apply rewrites on nodes with changed inputs.

        """
        super().__init__()
        self.ignore_newtrees = ignore_newtrees
        self.tracks_on_change_inputs = tracks_on_change_inputs
        self.__final__: Dict[str, bool] = {}
        self.__cleanup__: Dict[str, bool] = {}

    def register(
        self,
        name: str,
        rewriter: Union["RewriteDatabase", RewritesType],
        *tags: str,
        final_rewriter: bool = False,
        cleanup: bool = False,
        **kwargs,
    ):
        if final_rewriter and cleanup:
            raise ValueError("`final_rewriter` and `cleanup` cannot both be true.")
        super().register(name, rewriter, *tags, **kwargs)
        self.__final__[name] = final_rewriter
        self.__cleanup__[name] = cleanup

    def query(self, *tags, **kwtags):
        _rewriters = super().query(*tags, **kwtags)
        final_rewriters = [o for o in _rewriters if self.__final__.get(o.name, False)]
        cleanup_rewriters = [
            o for o in _rewriters if self.__cleanup__.get(o.name, False)
        ]
        rewriters = [
            o
            for o in _rewriters
            if o not in final_rewriters and o not in cleanup_rewriters
        ]
        if len(final_rewriters) == 0:
            final_rewriters = None
        if len(cleanup_rewriters) == 0:
            cleanup_rewriters = None
        return aesara_rewriting.EquilibriumGraphRewriter(
            rewriters,
            max_use_ratio=config.optdb__max_use_ratio,
            ignore_newtrees=self.ignore_newtrees,
            tracks_on_change_inputs=self.tracks_on_change_inputs,
            failure_callback=aesara_rewriting.NodeProcessingGraphRewriter.warn_inplace,
            final_rewriters=final_rewriters,
            cleanup_rewriters=cleanup_rewriters,
        )


class SequenceDB(RewriteDatabase):
    """A sequence of potential rewrites.

    Retrieve a sequence of rewrites as a `SequentialGraphRewriter` by calling
    `SequenceDB.query`.

    Each potential rewrite is registered with a floating-point position.
    No matter which rewrites are selected by a query, they are carried
    out in order of increasing position.

    """

    seq_rewriter_type = aesara_rewriting.SequentialGraphRewriter

    def __init__(self, failure_callback=aesara_rewriting.SequentialGraphRewriter.warn):
        super().__init__()
        self.__position__ = {}
        self.failure_callback = failure_callback

    def register(self, name, obj, *tags, **kwargs):
        position = kwargs.pop("position", "last")

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
            Only rewrites with position less than the cutoff are returned.

        """
        rewrites = super().query(*tags, **kwtags)

        if position_cutoff is None:
            position_cutoff = config.optdb__position_cutoff

        position_dict = self.__position__

        if len(tags) >= 1 and isinstance(tags[0], RewriteDatabaseQuery):
            # the call to super should have raise an error with a good message
            assert len(tags) == 1
            if getattr(tags[0], "position_cutoff", None):
                position_cutoff = tags[0].position_cutoff

            # The RewriteDatabaseQuery instance might contain extra rewrites which need
            # to be added the the sequence of rewrites (don't alter the
            # original dictionary)
            if len(tags[0].extra_rewrites) > 0:
                position_dict = position_dict.copy()
                for extra_rewrite in tags[0].extra_rewrites:
                    # Give a name to the extra rewrites (include both the
                    # class name for descriptiveness and id to avoid name
                    # collisions)
                    rewrite, position = extra_rewrite
                    rewrite.name = f"{rewrite.__class__}_{id(rewrite)}"

                    if position < position_cutoff:
                        rewrites.add(rewrite)
                        position_dict[rewrite.name] = position

        rewrites = [o for o in rewrites if position_dict[o.name] < position_cutoff]
        rewrites.sort(key=lambda obj: (position_dict[obj.name], obj.name))

        if self.failure_callback:
            ret = self.seq_rewriter_type(
                rewrites, failure_callback=self.failure_callback
            )
        else:
            ret = self.seq_rewriter_type(rewrites)

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
    r"""A database that generates `NodeRewriter`\s of type `SequentialNodeRewriter`."""

    def __init__(
        self,
        apply_all_rewrites: bool = False,
        profile: bool = False,
        node_rewriter=aesara_rewriting.SequentialNodeRewriter,
    ):
        super().__init__(failure_callback=None)
        self.apply_all_rewrites = apply_all_rewrites
        self.profile = profile
        self.node_rewriter = node_rewriter
        self.__name__: str = ""

    def register(self, name, obj, *tags, position="last", **kwargs):
        super().register(name, obj, *tags, position=position, **kwargs)

    def query(self, *tags, **kwtags):
        rewrites = list(super().query(*tags, **kwtags))
        ret = self.node_rewriter(
            *rewrites, apply_all_rewrites=self.apply_all_rewrites, profile=self.profile
        )
        return ret


class TopoDB(RewriteDatabase):
    """Generate a `GraphRewriter` of type `WalkingGraphRewriter`."""

    def __init__(
        self, db, order="in_to_out", ignore_newtrees=False, failure_callback=None
    ):
        super().__init__()
        self.db = db
        self.order = order
        self.ignore_newtrees = ignore_newtrees
        self.failure_callback = failure_callback

    def query(self, *tags, **kwtags):
        return aesara_rewriting.WalkingGraphRewriter(
            self.db.query(*tags, **kwtags),
            self.order,
            self.ignore_newtrees,
            self.failure_callback,
        )


class ProxyDB(RewriteDatabase):
    """A object that wraps an existing ``RewriteDatabase``.

    This is needed because we can't register the same ``RewriteDatabase``
    multiple times in different positions in a ``SequentialDB``.

    """

    def __init__(self, db):
        if not isinstance(db, RewriteDatabase):
            raise TypeError("`db` must be an `RewriteDatabase`.")

        self.db = db

    def query(self, *tags, **kwtags):
        return self.db.query(*tags, **kwtags)


DEPRECATED_NAMES = [
    (
        "DB",
        "`DB` is deprecated; use `RewriteDatabase` instead.",
        RewriteDatabase,
    ),
    (
        "Query",
        "`Query` is deprecated; use `RewriteDatabaseQuery` instead.",
        RewriteDatabaseQuery,
    ),
    (
        "OptimizationDatabase",
        "`OptimizationDatabase` is deprecated; use `RewriteDatabase` instead.",
        RewriteDatabase,
    ),
    (
        "OptimizationQuery",
        "`OptimizationQuery` is deprecated; use `RewriteDatabaseQuery` instead.",
        RewriteDatabaseQuery,
    ),
]


def __getattr__(name):
    """Intercept module-level attribute access of deprecated symbols.

    Adapted from https://stackoverflow.com/a/55139609/3006474.

    """
    from warnings import warn

    for old_name, msg, old_object in DEPRECATED_NAMES:
        if name == old_name:
            warn(msg, DeprecationWarning, stacklevel=2)
            return old_object

    raise AttributeError(f"module {__name__} has no attribute {name}")
