import numpy as np

from aesara.graph.basic import Apply, Constant, Variable
from aesara.graph.op import Op
from aesara.graph.sched import key_to_cmp
from aesara.graph.type import Generic
from aesara.tensor.type import tensor


class LoadFromDisk(Op):
    """
    An operation to load an array from disk.

    See Also
    --------
    load

    Notes
    -----
    Non-differentiable.

    """

    __props__ = ("dtype", "broadcastable", "mmap_mode")

    def __init__(self, dtype, broadcastable, mmap_mode=None):
        self.dtype = np.dtype(dtype)  # turn "float64" into np.float64
        self.broadcastable = broadcastable
        if mmap_mode not in (None, "c"):
            raise ValueError(
                "The only supported values for mmap_mode "
                "are None and 'c', got %s" % mmap_mode
            )
        self.mmap_mode = mmap_mode

    def make_node(self, path):
        if isinstance(path, str):
            path = Constant(Generic(), path)
        return Apply(self, [path], [tensor(self.dtype, shape=self.broadcastable)])

    def perform(self, node, inp, out):
        path = inp[0]
        if path.split(".")[-1] == "npz":
            raise ValueError(f"Expected a .npy file, got {path} instead")
        result = np.load(path, mmap_mode=self.mmap_mode)
        if result.dtype != self.dtype:
            raise TypeError(
                f"Expected an array of type {self.dtype}, got {result.dtype} instead"
            )
        out[0][0] = result

    def __str__(self):
        return "Load{{dtype: {}, broadcastable: {}, mmep: {}}}".format(
            self.dtype,
            self.broadcastable,
            self.mmap_mode,
        )


def load(path, dtype, broadcastable, mmap_mode=None):
    """
    Load an array from an .npy file.

    Parameters
    ----------
    path
        A Generic symbolic variable, that will contain a string
    dtype : data-type
        The data type of the array to be read.
    broadcastable
        The broadcastable pattern of the loaded array, for instance,
        (False,) for a vector, (False, True) for a column,
        (False, False) for a matrix.
    mmap_mode
        How the file will be loaded. None means that the
        data will be copied into an array in memory, 'c' means that the file
        will be mapped into virtual memory, so only the parts that are
        needed will be actually read from disk and put into memory.
        Other modes supported by numpy.load ('r', 'r+', 'w+') cannot
        be supported by Aesara.

    Examples
    --------
    >>> from aesara import *
    >>> path = Variable(Generic())
    >>> x = tensor.load(path, 'int64', (False,))
    >>> y = x*2
    >>> fn = function([path], y)
    >>> fn("stored-array.npy")  # doctest: +SKIP
    array([0, 2, 4, 6, 8], dtype=int64)

    """

    return LoadFromDisk(dtype, broadcastable, mmap_mode)(path)


##########################
# MPI
##########################

try:
    from mpi4py import MPI
except ImportError:
    mpi_enabled = False
else:
    comm = MPI.COMM_WORLD
    mpi_enabled = True


class MPIRecv(Op):
    """
    An operation to asynchronously receive an array to a remote host using MPI.

    See Also
    --------
    MPIRecv
    MPIWait

    Notes
    -----
    Non-differentiable.

    """

    __props__ = ("source", "tag", "shape", "dtype")

    def __init__(self, source, tag, shape, dtype):
        self.source = source
        self.tag = tag
        self.shape = shape
        self.dtype = np.dtype(dtype)  # turn "float64" into numpy.float64
        self.broadcastable = (False,) * len(shape)

    def make_node(self):
        return Apply(
            self,
            [],
            [
                Variable(Generic()),
                tensor(self.dtype, shape=self.broadcastable),
            ],
        )

    def perform(self, node, inp, out):

        data = np.zeros(self.shape, dtype=self.dtype)
        request = comm.Irecv(data, self.source, self.tag)

        out[0][0] = request
        out[1][0] = data

    def __str__(self):
        return f"MPIRecv{{source: {int(self.source)}, tag: {int(self.tag)}, shape: {self.shape}, dtype: {self.dtype}}}"

    def infer_shape(self, fgraph, node, shapes):
        return [None, self.shape]

    def do_constant_folding(self, fgraph, node):
        return False


class MPIRecvWait(Op):
    """
    An operation to wait on a previously received array using MPI.

    See Also
    --------
    MPIRecv

    Notes
    -----
    Non-differentiable.

    """

    __props__ = ("tag",)

    def __init__(self, tag):
        self.tag = tag

    def make_node(self, request, data):
        return Apply(
            self,
            [request, data],
            [tensor(data.dtype, shape=data.broadcastable)],
        )

    def perform(self, node, inp, out):

        request = inp[0]
        data = inp[1]

        request.wait()

        out[0][0] = data

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[1]]

    view_map = {0: [1]}


class MPISend(Op):
    """
    An operation to asynchronously Send an array to a remote host using MPI.

    See Also
    --------
    MPIRecv
    MPISendWait

    Notes
    -----
    Non-differentiable.

    """

    __props__ = ("dest", "tag")

    def __init__(self, dest, tag):
        self.dest = dest
        self.tag = tag

    def make_node(self, data):
        return Apply(self, [data], [Variable(Generic()), data.type()])

    view_map = {1: [0]}

    def perform(self, node, inp, out):

        data = inp[0]

        request = comm.Isend(data, self.dest, self.tag)

        out[0][0] = request
        out[1][0] = data

    def __str__(self):
        return f"MPISend{{dest: {int(self.dest)}, tag: {int(self.tag)}}}"


class MPISendWait(Op):
    """
    An operation to wait on a previously sent array using MPI.

    See Also
    --------
    MPISend

    Notes
    -----
    Non-differentiable.

    """

    __props__ = ("tag",)

    def __init__(self, tag):
        self.tag = tag

    def make_node(self, request, data):
        return Apply(self, [request, data], [Variable(Generic())])

    def perform(self, node, inp, out):
        request = inp[0]
        request.wait()
        out[0][0] = True


def isend(var, dest, tag):
    """
    Non blocking send.
    """
    return MPISend(dest, tag)(var)


def send(var, dest, tag):
    """
    Blocking send.
    """
    return MPISendWait(tag)(*isend(var, dest, tag))


def irecv(shape, dtype, source, tag):
    """
    Non-blocking receive.
    """
    return MPIRecv(source, tag, shape, dtype)()


def recv(shape, dtype, source, tag):
    """
    Blocking receive.
    """
    return MPIRecvWait(tag)(*irecv(shape, dtype, source, tag))


# Ordering keys for scheduling
def mpi_send_wait_key(a):
    """Wait as long as possible on Waits, Start Send/Recvs early."""
    if isinstance(a.op, (MPIRecvWait, MPISendWait)):
        return 1
    if isinstance(a.op, (MPIRecv, MPISend)):
        return -1
    return 0


def mpi_tag_key(a):
    """Break MPI ties by using the variable tag - prefer lower tags first."""
    if isinstance(a.op, (MPISend, MPIRecv, MPIRecvWait, MPISendWait)):
        return a.op.tag
    else:
        return 0


mpi_send_wait_cmp = key_to_cmp(mpi_send_wait_key)
mpi_tag_cmp = key_to_cmp(mpi_tag_key)

mpi_keys = (mpi_send_wait_key, mpi_tag_key)
mpi_cmps = (mpi_send_wait_cmp, mpi_tag_cmp)

__all__ = ["load"]
