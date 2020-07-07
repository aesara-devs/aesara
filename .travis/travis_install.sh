#!/usr/bin/env bash

set -e

if [[ $DOC == "1" ]]; then
# this is a hack to deal with the fact that the docs and flake8 config are all set up
# for old versions
conda create --yes -q -n pyenv python=3.6 numpy=1.13.1
conda activate pyenv
conda install --yes -q mkl numpy=1.13.1 scipy=0.19.1 pip flake8=3.5 six=1.11.0 pep8=1.7.1 pyflakes=1.6.0 mkl-service graphviz pytest
python -m pip install pydot-ng
else
conda create --yes -q -n pyenv python=$TRAVIS_PYTHON_VERSION
conda activate pyenv
conda install --yes -q mkl numpy scipy pip flake8 six pep8 pyflakes sphinx mkl-service graphviz pytest  # libgfortran
python -m pip install -q pydot-ng sphinx_rtd_theme
fi

python -m pip install --no-deps --upgrade -e .
