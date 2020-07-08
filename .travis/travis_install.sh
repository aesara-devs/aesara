#!/usr/bin/env bash

set -e

conda create --yes -q -n pyenv python=$TRAVIS_PYTHON_VERSION
conda activate pyenv
conda install --yes -q mkl numpy scipy pip flake8 six pep8 pyflakes sphinx mkl-service graphviz pytest  # libgfortran
python -m pip install -q pydot-ng sphinx_rtd_theme
python -m pip install --no-deps --upgrade -e .
