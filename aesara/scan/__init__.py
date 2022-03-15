"""
This module provides the Scan Op.

Scanning is a general form of recurrence, which can be used for looping.
The idea is that you *scan* a function along some input sequence, producing
an output at each time-step that can be seen (but not modified) by the
function at the next time-step. (Technically, the function can see the
previous K  time-steps of your outputs and L time steps (from past and
future) of your inputs.

So for example, ``sum()`` could be computed by scanning the ``z+x_i``
function over a list, given an initial state of ``z=0``.

Special cases:

* A *reduce* operation can be performed by using only the last
  output of a ``scan``.
* A *map* operation can be performed by applying a function that
  ignores previous steps of the outputs.

Often a for-loop or while-loop can be expressed as a ``scan()`` operation,
and ``scan`` is the closest that aesara comes to looping. The advantages
of using ``scan`` over `for` loops in python (among others) are:

* it allows the number of iterations to be part of the symbolic graph
* it allows computing gradients through the for loop
* there exist a bunch of optimizations that help re-write your loop
  such that less memory is used and that it runs faster

The Scan Op should typically be used by calling any of the following
functions: ``scan()``, ``map()``, ``reduce()``, ``foldl()``,
``foldr()``.

"""


__docformat__ = "restructedtext en"
__authors__ = (
    "Razvan Pascanu "
    "Frederic Bastien "
    "James Bergstra "
    "Pascal Lamblin "
    "Arnaud Bergeron "
    "PyMC Developers "
    "Aesara Developers "
)
__copyright__ = "(c) 2010, Universite de Montreal"

from aesara import configdefaults


configdefaults.add_scan_configvars()

from aesara.scan import opt
from aesara.scan.basic import scan
from aesara.scan.checkpoints import scan_checkpoints
from aesara.scan.utils import until
from aesara.scan.views import foldl, foldr, map, reduce
