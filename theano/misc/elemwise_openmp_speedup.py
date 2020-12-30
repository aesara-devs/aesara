import os
import subprocess
import sys
from locale import getpreferredencoding
from optparse import OptionParser

from theano.configdefaults import config


console_encoding = getpreferredencoding()

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


def runScript(N):
    script = "elemwise_time_test.py"
    path = os.path.dirname(os.path.abspath(__file__))
    proc = subprocess.Popen(
        ["python", script, "--script", "-N", str(N)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=path,
    )
    (out, err) = proc.communicate()
    if err:
        print(err)
        sys.exit()
    return list(map(float, out.decode(console_encoding).split(" ")))


if __name__ == "__main__":
    options, arguments = parser.parse_args(sys.argv)
    if hasattr(options, "help"):
        print(options.help)
        sys.exit(0)
    orig_flags = os.environ.get("THEANO_FLAGS", "")
    os.environ["THEANO_FLAGS"] = orig_flags + ",openmp=false"
    (cheapTime, costlyTime) = runScript(N=options.N)
    os.environ["THEANO_FLAGS"] = orig_flags + ",openmp=true"
    (cheapTimeOpenmp, costlyTimeOpenmp) = runScript(N=options.N)

    if cheapTime > cheapTimeOpenmp:
        cheapSpeed = cheapTime / cheapTimeOpenmp
        cheapSpeedstring = "speedup"
    else:
        cheapSpeed = cheapTimeOpenmp / cheapTime
        cheapSpeedstring = "slowdown"

    if costlyTime > costlyTimeOpenmp:
        costlySpeed = costlyTime / costlyTimeOpenmp
        costlySpeedstring = "speedup"
    else:
        costlySpeed = costlyTimeOpenmp / costlyTime
        costlySpeedstring = "slowdown"
    print(f"Timed with vector of {int(options.N)} elements")
    print(
        f"Fast op time without openmp {cheapTime}s with openmp {cheapTimeOpenmp}s {cheapSpeedstring} {cheapSpeed:2.2f}"
    )

    print(
        f"Slow op time without openmp {costlyTime}s with openmp {costlyTimeOpenmp}s {costlySpeedstring} {costlySpeed:2.2f}"
    )
