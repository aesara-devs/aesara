#!/usr/bin/env python
import os
from datetime import datetime, timezone

from setuptools import setup
from setuptools.dist import Distribution


dist = Distribution()
dist.parse_config_files()

# Handle builds of nightly release
is_nightly = "BUILD_AESARA_NIGHTLY" in os.environ
version = "TODO"

if is_nightly:
    suffix = datetime.now(timezone.utc).strftime(r".dev%Y%m%d")
    version = version.split("+")[0] + suffix

if __name__ == "__main__":
    setup(version=version)
