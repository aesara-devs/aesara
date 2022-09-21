import os

import pytest


def pytest_sessionstart(session):
    os.environ["AESARA_FLAGS"] = ",".join(
        [
            os.environ.setdefault("AESARA_FLAGS", ""),
            "warn__ignore_bug_before=all,on_opt_error=raise,on_shape_error=raise,cmodule__warn_no_version=True",
        ]
    )
    os.environ["NUMBA_BOUNDSCHECK"] = "1"


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
