import pytest

from theano.gof.optdb import DB, opt


class TestDB:
    def test_name_clashes(self):
        class Opt(opt.Optimizer):  # inheritance buys __hash__
            name = "blah"

        db = DB()
        db.register("a", Opt())

        db.register("b", Opt())

        db.register("c", Opt(), "z", "asdf")

        assert "a" in db
        assert "b" in db
        assert "c" in db

        with pytest.raises(ValueError, match=r"The name.*"):
            db.register("c", Opt())  # name taken

        with pytest.raises(ValueError, match=r"The name.*"):
            db.register("z", Opt())  # name collides with tag

        with pytest.raises(ValueError, match=r"The tag.*"):
            db.register("u", Opt(), "b")  # name new but tag collides with name
