# Run using
# mpiexec -np 2 python _test_mpi_roundtrip.py

from sys import exit, stderr, stdout

import numpy as np
from mpi4py import MPI

import theano
from theano.configparser import change_flags
from theano.gof.sched import sort_schedule_fn
from theano.tensor.io import mpi_cmps, recv, send


comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()

if size != 2:
    stderr.write(
        "mpiexec failed to create a world with two nodes.\n"
        "Closing with success message."
    )
    stdout.write("True")
    exit(0)

shape = (2, 2)
dtype = "float32"

scheduler = sort_schedule_fn(*mpi_cmps)
mode = theano.Mode(optimizer=None, linker=theano.OpWiseCLinker(schedule=scheduler))

with change_flags(compute_test_value="off"):
    if rank == 0:
        x = theano.tensor.matrix("x", dtype=dtype)
        y = x + 1
        send_request = send(y, 1, 11)

        z = recv(shape, dtype, 1, 12)

        f = theano.function([x], [send_request, z], mode=mode)

        xx = np.random.rand(*shape).astype(dtype)
        expected = (xx + 1) * 2

        _, zz = f(xx)

        same = np.linalg.norm(zz - expected) < 0.001
        # The parent test will look for "True" in the output
        stdout.write(str(same))

    if rank == 1:

        y = recv(shape, dtype, 0, 11)
        z = y * 2
        send_request = send(z, 0, 12)

        f = theano.function([], send_request, mode=mode)

        f()
