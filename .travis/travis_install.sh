#!/usr/bin/env bash

set -e

if [[ $DOC == "1" ]]; then
conda create --yes -q -n pyenv python=3.6 numpy=1.13.1
conda activate pyenv
conda install --yes -q mkl numpy=1.13.1 scipy=0.19.1 nose=1.3.7 pip flake8=3.5 six=1.11.0 pep8=1.7.1 pyflakes=1.6.0 mkl-service graphviz
python -m pip install pydot-ng flake8-future-import parameterized nose-exclude nose-timer sphinx=1.5.1
else
conda create --yes -q -n pyenv python=$TRAVIS_PYTHON_VERSION
conda activate pyenv
conda install --yes -q mkl numpy scipy nose pip flake8 six pep8 pyflakes sphinx mkl-service graphviz  # libgfortran
python -m pip install -q pydot-ng flake8-future-import parameterized sphinx_rtd_theme nose-exclude nose-timer
fi

python -m pip install --no-deps --upgrade -e .
