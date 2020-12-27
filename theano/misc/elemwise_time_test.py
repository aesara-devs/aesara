import sys
import time
from optparse import OptionParser

import numpy as np

import theano
import theano.tensor as tt
from theano.configdefaults import config


parser = OptionParser(
    usage="%prog <options>\n Compute time for" " fast and slow elemwise operations"
)
parser.add_option(
    "-N",
    "--N",
    action="store",
    dest="N",
    default=config.openmp_elemwise_minsize,
    type="int",
    help="Number of vector elements",
)
parser.add_option(
    "--script",
    action="store_true",
    dest="script",
    default=False,
    help="Run program as script and print results on stdoutput",
)


def evalTime(f, v, script=False, loops=1000):
    min = 1e10
    for i in range(0, loops):
        t0 = time.time()
        f(v)
        dt = time.time() - t0
        min = dt if dt < min else min
    if not script:
        print(f" run time in {int(loops)} loops was {min:2.9f} sec")
    return min


def ElemwiseOpTime(N, script=False, loops=1000):
    x = tt.vector("x")
    np.random.seed(1235)
    v = np.random.random(N).astype(config.floatX)
    f = theano.function([x], 2 * x + x * x)
    f1 = theano.function([x], tt.tanh(x))
    if not script:
        if config.openmp:
            print("With openmp:")
        print("Fast op ", end=" ")
    ceapTime = evalTime(f, v, script=script, loops=loops)
    if not script:
        print("Slow op ", end=" ")
    costlyTime = evalTime(f1, v, script=script, loops=loops)
    return (ceapTime, costlyTime)


if __name__ == "__main__":
    options, arguments = parser.parse_args(sys.argv)
    if hasattr(options, "help"):
        print(options.help)
        sys.exit(0)

    (cheapTime, costlyTime) = ElemwiseOpTime(N=options.N, script=options.script)

    if options.script:
        sys.stdout.write(f"{cheapTime:2.9f} {costlyTime:2.9f}\n")
        sys.stdout.flush()
