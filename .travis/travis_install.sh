#!/usr/bin/env bash

conda create --yes -q -n pyenv python=$TRAVIS_PYTHON_VERSION
conda activate pyenv
conda install --yes -q mkl numpy scipy nose pip flake8 six pep8 pyflakes sphinx mkl-service libgfortran graphviz
python -m pip install -q pydot-ng flake8-future-import parameterized sphinx_rtd_theme nose-exclude nose-timer
python -m pip install --no-deps --upgrade -e .
conda deactivate
