import pytest

from aesara.graph import opt
from aesara.graph.optdb import (
    EquilibriumDB,
    LocalGroupDB,
    OptimizationDatabase,
    ProxyDB,
    SequenceDB,
)


class TestOpt(opt.GraphRewriter):
    name = "blah"

    def apply(self, fgraph):
        pass


class TestDB:
    def test_register(self):
        db = OptimizationDatabase()
        db.register("a", TestOpt())

        db.register("b", TestOpt())

        db.register("c", TestOpt(), "z", "asdf")

        assert "a" in db
        assert "b" in db
        assert "c" in db

        with pytest.raises(ValueError, match=r"The tag.*"):
            db.register("c", TestOpt())  # name taken

        with pytest.raises(ValueError, match=r"The tag.*"):
            db.register("z", TestOpt())  # name collides with tag

        with pytest.raises(ValueError, match=r"The tag.*"):
            db.register("u", TestOpt(), "b")  # name new but tag collides with name

        with pytest.raises(TypeError, match=r".* is not a valid.*"):
            db.register("d", 1)

    def test_EquilibriumDB(self):
        eq_db = EquilibriumDB()

        with pytest.raises(ValueError, match=r"`final_opt` and.*"):
            eq_db.register("d", TestOpt(), final_opt=True, cleanup=True)

    def test_SequenceDB(self):
        seq_db = SequenceDB(failure_callback=None)

        res = seq_db.query("+a")

        assert isinstance(res, opt.SeqOptimizer)
        assert res.data == []

        seq_db.register("b", TestOpt(), position=1)

        from io import StringIO

        out_file = StringIO()
        seq_db.print_summary(stream=out_file)

        res = out_file.getvalue()

        assert str(id(seq_db)) in res
        assert "names {'b'}" in res

        with pytest.raises(TypeError, match=r"`position` must be.*"):
            seq_db.register("c", TestOpt(), position=object())

    def test_LocalGroupDB(self):
        lg_db = LocalGroupDB()

        lg_db.register("a", TestOpt(), 1)

        assert "a" in lg_db.__position__

        with pytest.raises(TypeError, match=r"`position` must be.*"):
            lg_db.register("b", TestOpt(), position=object())

    def test_ProxyDB(self):
        with pytest.raises(TypeError, match=r"`db` must be.*"):
            ProxyDB(object())
