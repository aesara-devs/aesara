#!/usr/bin/env bash
# In Python 3.4, we test the min version of NumPy and SciPy. In Python 2.7, we test more recent version.
if test -e $HOME/miniconda/envs/pyenv; then
    echo "pyenv already exists."
else
    echo "Creating pyenv."
    conda create --yes -q -n pyenv python=$TRAVIS_PYTHON_VERSION
fi

source activate pyenv
conda install --yes -q mkl numpy scipy nose pip flake8 six pep8 pyflakes sphinx mkl-service libgfortran graphviz
python -m pip install -q pydot-ng flake8-future-import parameterized sphinx_rtd_theme nose-exclude nose-timer
python -m pip install --no-deps --upgrade -e .
source deactivate
