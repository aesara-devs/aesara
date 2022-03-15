Requirements
============

.. note::

    We only support the installation of the requirements through conda.

.. _BLAS: http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms
.. _Python: http://www.python.org/
.. _LaTeX: http://www.latex-project.org/
.. _dvipng: http://savannah.nongnu.org/projects/dvipng/
.. _warp-ctc: https://github.com/baidu-research/warp-ctc

    Python_ == >= 3.7
        |PythonDistRecommended|. Python 2.4 was supported up to and including the
        release 0.6. Python 2.6 was supported up to and including the
        release 0.8.2. Python 3.3 was supported up to and including release 0.9.
        Python 3.6 was supported up to and including release 2.3.4.

    `NumPy <http://numpy.scipy.org/>`_ >= 1.9.1
        Earlier versions could work, but we don’t test it.

    `SciPy <http://scipy.org>`_ >= 0.14
        Only currently required for sparse matrix and special functions support, but highly recommended. SciPy >=0.8 could work, but earlier versions have known bugs with sparse matrices.

    `BLAS`_ installation (with Level 3 functionality)
        * **Recommended**: MKL, which is free through Conda with ``mkl-service`` package.
        * Alternatively, we suggest to install OpenBLAS, with the development headers (``-dev``, ``-devel``, depending on your Linux distribution).

**Optional requirements**

    |PlatformCompiler|
        **Highly recommended.** Aesara can fall back on a NumPy-based Python execution model, but a C compiler allows for vastly faster execution.

    `Sphinx <http://sphinx.pocoo.org/>`_ >= 0.5.1, `pygments <http://pygments.org/>`_
        For building the documentation. LaTeX_ and dvipng_ are also necessary for math to show up as images.

    `pydot-ng <https://github.com/pydot/pydot-ng>`_
        To handle large picture for gif/images.

    `warp-ctc`_
        Required for :ref:`Aesara CTC implementation
        <libdoc_tensor_nnet_ctc>`. It is faster then using an
        equivalent graph of Aesara ops.


Requirements installation through Conda (recommended)
-----------------------------------------------------

Install Miniconda
^^^^^^^^^^^^^^^^^

Follow this `link <http://conda.pydata.org/miniconda.html>`__ to install Miniconda.

.. note::

    If you want fast compiled code (recommended), make sure you have |CompilerName| installed.

.. install_requirements_and_optional_packages

Install requirements and optional packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    conda install numpy scipy mkl pytest <sphinx> <pydot-ng>

* Arguments between <...> are optional.
