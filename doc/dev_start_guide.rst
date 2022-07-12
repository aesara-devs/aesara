.. _dev_start_guide:

=====================
Developer Start Guide
=====================

Contributing
============

Looking for an idea for a first contribution? Check the `GitHub issues
<https://github.com/aesara-devs/aesara/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22>`_.

We recommend creating an issue to discuss proposed changes before making them.
This is a good way to make sure that proposed changes will be accepted.

Resources
=========

See :ref:`aesara-community` for a list of Aesara resources.

The Theano Google group is also relevant to (early) Aesara versions:
`theano-dev`_.

.. _theano-dev: https://groups.google.com/group/theano-dev


.. _quality_contributions:

Requirements for Quality Contributions
======================================

The following are requirements for a quality pull request (PR)/contribution:

* All code should be accompanied by quality unit tests  that provide complete
  coverage of the features added.
* There is an informative high-level description of the changes, or a reference to an issue describing the changes.
* The description and/or commit messages reference any relevant GitHub issues.
* `pre-commit <https://pre-commit.com/#installation>`_ is installed and `set up <https://pre-commit.com/#3-install-the-git-hook-scripts>`_.
* The commit messages follow `these guidelines <https://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html>`_.
* The commits correspond to `relevant logical changes <https://wiki.openstack.org/wiki/GitCommitMessages#Structural_split_of_changes>`_, and there are **no commits that fix changes introduced by other commits in the same branch/BR**.
* There are tests, implemented within the pytest_ framework, covering the changes introduced by the PR.
* `Type hints <https://www.python.org/dev/peps/pep-0484/>`_ are added where appropriate.

Don't worry, your PR doesn't need to be in perfect order to submit it.  As development progresses and/or reviewers request changes, you can always `rewrite the history <https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History#_rewriting_history>`_ of your feature/PR branches.

If your PR is an ongoing effort and you would like to involve us in the process,
simply make it a `draft PR <https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/about-pull-requests#draft-pull-requests>`_.

When you submit a PR, your changes will automatically be tested via our
continuous integration (CI). Just because the tests run automatically does not
mean you shouldn't run them yourself to make sure everything is all right.  You
can run only the portion you are modifying to go faster and have CI make sure
there are no broader problems.

To run the test suite with the default options, see :ref:`test_aesara`.

.. _Sphinx: http://sphinx.pocoo.org/
.. _reStructuredText: http://docutils.sourceforge.net/rst.html
.. _pytest: http://docs.pytest.org/en/latest/


Documentation and docstrings
----------------------------

* The documentation and the API documentation are generated using `Sphinx`_.

* The documentation should be written in `reStructuredText`_ and the
  docstrings of all the classes and functions should respect the
  `PEP257 <https://www.python.org/dev/peps/pep-0257/>`_ rules and follow the
  `Numpy docstring standard
  <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_.

* To cross-reference other objects (e.g. reference other classes or methods) in
  the docstrings, use the
  `cross-referencing objects <http://www.sphinx-doc.org/en/stable/domains.html#cross-referencing-python-objects>`_
  syntax. ``:py`` can be omitted, see e.g. this
  `stackoverflow answer <http://stackoverflow.com/a/7754189>`_.

* See :ref:`metadocumentation`, for some information on how to generate the
  documentation.


A Docstring Example
~~~~~~~~~~~~~~~~~~~
Here is an example on how to add a docstring to a class.

.. testcode:: python

    from aesara.graph.basic import Variable
    from aesara.graph.op import Op


    class DoubleOp(Op):
        """Double each element of a tensor.

        Notes
        -----
        this is a test note

        See Also
        --------
        `Elemwise`: This functionality is already available; just execute
        ``x * 2`` with ``x`` being an Aesara variable.
        """

        def make_node(self, x: Variable):
            """Construct an `Apply` node for this `Op`.

            Parameters
            ----------
            x
                Input tensor

            """
            ...


Installation and configuration
==============================

To submit PRs, create an account on `GitHub <http://www.github.com/>`_ and fork
`Aesara <http://www.github.com/aesara-devs/aesara>`_.

This will create your own clone of the Aesara project on GitHub's servers. It is customary
to assign this Git remote the name "origin", and the official Aesara repository
the name "upstream".


Create a local copy
-------------------

Clone your fork locally with

.. code-block:: bash

    git clone git@github.com:YOUR_GITHUB_LOGIN/Aesara.git

For this URL to work, you must set your public SSH keys inside your
`GitHub account setting <https://github.com/settings/ssh>`_.

From your local repository, your fork on GitHub will be called "origin" by default.

Next, create a remote entry for the original (i.e. upstream) Aesara repository
with the following:

.. code-block:: bash

    git remote add upstream git://github.com/aesara-devs/aesara.git

.. note::

    You can choose a name other than "upstream" to reference the official Aesara
    repository.

Setting up the your local development environment
-------------------------------------------------

You will need to create a virtual environment and install the project requirements within it.

The recommended approach is to install `conda <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_ and
create a virtual environment in the project directory:

.. code-block:: bash

    conda env create -n aesara-dev -f environment.yml
    conda activate aesara-dev

Afterward, you can install the development dependencies:

.. code-block:: bash

    pip install -r requirements.txt

Next, ``pre-commit`` needs to be configured so that the linting and code quality
checks are performed before each commit:

.. code-block:: bash

    pre-commit install


The virtual environment will need to be activated in any environment
(e.g. shells, IDEs, etc.) that plans to run the Aesara tests or add commits to the
project repository.

You can now test your environment/code by running ``pytest`` in the project's root
directory.  See :ref:`test_aesara` for more information about testing.


For a general guide on how to provide open source contributions see `here
<https://opensource.guide/how-to-contribute/#how-to-submit-a-contribution>`_.
For a good overview of the development workflow (e.g. relevant ``git`` commands)
see the `NumPy development guide <https://numpy.org/doc/stable/dev/>`_.

Other tools that might help
===========================

 * `cProfile <http://docs.python.org/library/profile.html>`_: Time profiler that work at function level
 * `line_profiler <http://pypi.python.org/pypi/line_profiler/>`_: Line-by-line profiler
 * `memory_profiler <http://fseoane.net/blog/2012/line-by-line-report-of-memory-usage/>`_: A memory profiler
 * `runsnake <http://www.vrplumber.com/programming/runsnakerun/>`_: GUI for cProfile (time profiler) and Meliae (memory profiler)
 * `Guppy <https://pypi.python.org/pypi/guppy/>`_: Supports object and heap memory sizing, profiling, and debugging
 * `hub <https://github.com/defunkt/hub>`_: A tool that adds GitHub commands to the git command line
 * `git pull-requests <http://www.splitbrain.org/blog/2011-06/19-automate_github_pull_requests>`_: Another command line tool for ``git``/GitHub
