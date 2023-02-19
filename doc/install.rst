.. _install:

Installing Aesara
=================

In order to guarantee a complete installation including the necessary libraries, it is recommended to install Aesara via conda-forge into a clean environment. This can be done using either `Mamba`_ or `Conda`_:

.. tab-set::

    .. tab-item:: Mamba

        .. code:: bash

            mamba create --name=aesara-env --channel=conda-forge aesara
            conda activate aesara-env


    .. tab-item:: Conda

        .. code:: bash

            conda create --name=aesara-env --channel=conda-forge aesara
            conda activate aesara-env


Alternatively, Aesara can be installed directly from PyPI using `pip`:

.. code-block:: bash

    pip install aesara


The current development branch of Aesara can be installed from PyPI or GitHub using `pip`:


.. tab-set::

    .. tab-item:: PyPI

        .. code:: bash

            pip install aesara-nightly

    .. tab-item:: GitHub

        .. code:: bash

            pip install git+https://github.com/aesara-devs/aesara


.. attention::

    To use the Numba or JAX backend you will need to install the corresponding library in addition to Aesara. Please refer to `Numba's installation instructions <https://numba.readthedocs.io/en/stable/user/installing.html>`__ and `JAX's installation instructions  <https://github.com/google/jax#installation>`__ respectively.


.. _Mamba: https://mamba.readthedocs.io/en/latest/installation.html
.. _Conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
