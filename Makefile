.PHONY: help venv conda docker docstyle format style black test lint check coverage pypi
.DEFAULT_GOAL = help

PROJECT_NAME = theano
PROJECT_DIR = theano/
PYTHON = python
PIP = pip
CONDA = conda
SHELL = bash

help:
	@printf "Usage:\n"
	@grep -E '^[a-zA-Z_-]+:.*?# .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?# "}; {printf "\033[1;34mmake %-10s\033[0m%s\n", $$1, $$2}'

conda:  # Set up a conda environment for development.
	@printf "Creating conda environment...\n"
	${CONDA} create --yes --name ${PROJECT_NAME}-env python=3.6
	( \
	${CONDA} activate ${PROJECT_NAME}-env; \
	${PIP} install -U pip; \
	${PIP} install -r requirements.txt; \
	${CONDA} deactivate; \
	)
	@printf "\n\nConda environment created! \033[1;34mRun \`conda activate ${PROJECT_NAME}-env\` to activate it.\033[0m\n\n\n"

venv:  # Set up a Python virtual environment for development.
	@printf "Creating Python virtual environment...\n"
	rm -rf ${PROJECT_NAME}-venv
	${PYTHON} -m venv ${PROJECT_NAME}-venv
	( \
	source ${PROJECT_NAME}-venv/bin/activate; \
	${PIP} install -U pip; \
	${PIP} install -r requirements.txt; \
	deactivate; \
	)
	@printf "\n\nVirtual environment created! \033[1;34mRun \`source ${PROJECT_NAME}-venv/bin/activate\` to activate it.\033[0m\n\n\n"

docstyle:
	@printf "Checking documentation with pydocstyle...\n"
	pydocstyle ${PROJECT_DIR}
	@printf "\033[1;34mPydocstyle passes!\033[0m\n\n"

format:
	@printf "Checking code format with black...\n"
	black -t py36 --check ${PROJECT_DIR} tests/ setup.py conftest.py
	@printf "\033[1;34mBlack passes!\033[0m\n\n"

style:
	@printf "Checking code style with pylint...\n"
	flake8
	@printf "\033[1;34mPylint passes!\033[0m\n\n"

black:  # Format code in-place using black.
	black ${PROJECT_DIR} tests/ setup.py conftest.py

test:  # Test code using pytest.
	pytest -v tests/ ${PROJECT_DIR} --cov=${PROJECT_DIR} --cov-report=xml --html=testing-report.html --self-contained-html

coverage: test
	diff-cover coverage.xml --compare-branch=master --fail-under=100

pypi:
	${PYTHON} setup.py clean --all; \
	${PYTHON} setup.py rotate --match=.tar.gz,.whl,.egg,.zip --keep=0; \
	${PYTHON} setup.py sdist bdist_wheel; \
  twine upload --skip-existing dist/*;

lint: docstyle format style  # Lint code using pydocstyle, black and pylint.

check: lint test coverage  # Both lint and test code. Runs `make lint` followed by `make test`.
