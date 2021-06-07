"""Symbolic Op for raising an exception."""

from aesara.graph.basic import Apply
from aesara.graph.op import Op


__authors__ = "James Bergstra " "PyMC Dev Team " "Aesara Developers"
__copyright__ = "(c) 2011, Universite de Montreal"
__license__ = "3-clause BSD License"

__docformat__ = "restructuredtext en"


class Raise(Op):
    """Op whose perform() raises an exception."""

    __props__ = ("msg", "exc")

    def __init__(self, msg="", exc=NotImplementedError):
        """
        msg - the argument to the exception
        exc - an exception class to raise in self.perform
        """
        self.msg = msg
        self.exc = exc

    def __str__(self):
        return f"Raise{{{self.exc}({self.msg})}}"

    def make_node(self, x):
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, out_storage):
        raise self.exc(self.msg)
