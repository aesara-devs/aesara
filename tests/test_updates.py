import pytest

import aesara
from aesara.tensor.type import vector
from aesara.updates import OrderedUpdates


class TestUpdates:
    def test_updates_init(self):
        with pytest.raises(TypeError):
            OrderedUpdates(dict(d=3))

        sv = aesara.shared("asdf")
        # TODO FIXME: Not a real test.
        OrderedUpdates({sv: 3})

    def test_updates_setitem(self):
        up = OrderedUpdates()

        # keys have to be SharedVariables
        with pytest.raises(TypeError):
            up.__setitem__(5, 7)
        with pytest.raises(TypeError):
            up.__setitem__(vector(), 7)

        # TODO FIXME: Not a real test.
        up[aesara.shared(88)] = 7

    def test_updates_add(self):
        up1 = OrderedUpdates()
        up2 = OrderedUpdates()

        a = aesara.shared("a")
        b = aesara.shared("b")

        assert not up1 + up2

        up1[a] = 5

        # test that addition works
        assert up1
        assert up1 + up2
        assert not up2

        assert len(up1 + up2) == 1
        assert (up1 + up2)[a] == 5

        up2[b] = 7
        assert up1
        assert up1 + up2
        assert up2

        assert len(up1 + up2) == 2
        assert (up1 + up2)[a] == 5
        assert (up1 + up2)[b] == 7

        assert a in (up1 + up2)
        assert b in (up1 + up2)

        # this works even though there is a collision
        # because values all match
        assert len(up1 + up1 + up1) == 1

        up2[a] = 8  # a gets different value in up1 and up2
        with pytest.raises(KeyError):
            up1 + up2

        # TODO FIXME: Not a real test.
        # reassigning to a key works fine right?
        up2[a] = 10
