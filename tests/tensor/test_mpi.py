import os
import subprocess

import pytest

import aesara
from aesara import change_flags
from aesara.gof.sched import sort_schedule_fn
from aesara.tensor.io import (
    MPISend,
    MPISendWait,
    mpi_cmps,
    mpi_enabled,
    mpi_send_wait_cmp,
    recv,
    send,
)


mpi_scheduler = sort_schedule_fn(*mpi_cmps)
mpi_linker = aesara.OpWiseCLinker(schedule=mpi_scheduler)
mpi_mode = aesara.Mode(linker=mpi_linker)


@change_flags(compute_test_value="off")
def test_recv():
    x = recv((10, 10), "float64", 0, 11)
    assert x.dtype == "float64"
    assert x.broadcastable == (False, False)

    recvnode = x.owner.inputs[0].owner
    assert recvnode.op.source == 0
    assert recvnode.op.tag == 11


def test_send():
    x = aesara.tensor.matrix("x")
    y = send(x, 1, 11)
    sendnode = y.owner.inputs[0].owner
    assert sendnode.op.dest == 1
    assert sendnode.op.tag == 11


@change_flags(compute_test_value="off")
def test_can_make_function():
    x = recv((5, 5), "float32", 0, 11)
    y = x + 1
    assert aesara.function([], [y])


@pytest.mark.skipif(not mpi_enabled, reason="MPI not enabled")
def test_mpi_roundtrip():
    aesara_root = aesara.__file__.split("__init__")[0]
    env = os.environ.copy()
    flags = env.get("THEANO_FLAGS", "")
    keep_flags = ",".join(
        f for f in flags.split(",") if not f.startswith("init_gpu_device")
    )
    env["THEANO_FLAGS"] = keep_flags
    p = subprocess.Popen(
        "mpiexec -np 2 python " + aesara_root + "tensor/tests/_test_mpi_roundtrip.py",
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        close_fds=True,
        env=env,
    )
    (stdout, stderr) = p.communicate()

    result = aesara.utils.decode(stdout)
    assert "True" in result, aesara.utils.decode(stderr)


def test_mpi_send_wait_cmp():
    x = aesara.tensor.matrix("x")
    y = send(x, 1, 11)
    z = x + x
    waitnode = y.owner
    sendnode = y.owner.inputs[0].owner
    addnode = z.owner
    assert mpi_send_wait_cmp(sendnode, addnode) < 0  # send happens first
    assert mpi_send_wait_cmp(waitnode, addnode) > 0  # wait happens last


@change_flags(compute_test_value="off")
def test_mpi_tag_ordering():
    x = recv((2, 2), "float32", 1, 12)
    y = recv((2, 2), "float32", 1, 11)
    z = recv((2, 2), "float32", 1, 13)
    f = aesara.function([], [x, y, z], mode=mpi_mode)
    nodes = f.maker.linker.make_all()[-1]

    assert all(node.op.tag == tag for node, tag in zip(nodes, (11, 12, 13, 11, 12, 13)))


def test_mpi_schedule():
    x = aesara.tensor.matrix("x")
    y = send(x, 1, 11)
    z = x + x

    f = aesara.function([x], [y, z], mode=mpi_mode)
    nodes = f.maker.linker.make_all()[-1]
    optypes = [MPISend, aesara.tensor.Elemwise, MPISendWait]
    assert all(isinstance(node.op, optype) for node, optype in zip(nodes, optypes))
