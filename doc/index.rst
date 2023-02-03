Aesara
=======

*Aesara is a fast, hackable, meta-tensor library in Python*


Aesara is a Python library that allows you to define, optimize/rewrite, and
evaluate mathematical expressions involving multi-dimensional arrays
efficiently. It is composed of different parts:

- **Symbolic representation** of mathematical operations on arrays
- **Speed and stability optimization**
- **Efficient symbolic differentiation**
- **Powerful rewrite system** to programmatically modify your models
- **Extendable backends.** Aesara currently compiles to C, Jax and Numba.

.. image:: _static/aesara_overview_diagram.png
   :width: 100%
   :alt: Overview of Aesara

Aesara adheres to the following design principles:

- **Familiar**: Aesara follows the NumPy API and can act as a drop-in replacement
- **Modular**: Aesara's graph, rewrites, backends can be easily be extended independently
- **Hackable**: Easily add rewrites, mathematical operators and backends *in pure python*
- **Composable**: Aesara's compiled functions are compatible with the Numba & JAX ecosystems.

We also make a strong commitment to **code quality** and **scalability**.

Aesara is based on `Theano`_, which has been powering large-scale computationally
intensive scientific investigations since 2007.

Install Aesara
===============

Aesara installation can happen in a few different ways. You can install Aesara with `conda` or with `pip`. To get the bleeding edge version you can install `aesara-nightly.`

.. tab-set::

     .. tab-item:: PyPi

         .. code:: bash

             pip install aesara


     .. tab-item:: Conda

         .. code:: bash

            conda install -c conda-forge aesara


     .. tab-item:: Nightly

         .. code:: bash

             pip install aesara-nightly


.. attention::

    To use the Numba and JAX backend you will need to install these libraries in addition to Aesara. Please refer to `Numba's installation instructions <https://numba.readthedocs.io/en/stable/user/installing.html>`__ and `JAX's installation instructions  <https://github.com/google/jax#installation>`__ respectively.


Featured applications
=====================

The following projects illustrate Aesara's unique capabilities:

.. _cards-clickable::

.. card:: AePPL
    :link: https://github.com/aesara-devs/aeppl

    A(e) PPL written with Aesara.

.. card:: AeMCMC
    :link: https://github.com/aesara-devs/aemcmc

    Sampling probabilistic models with Aesara

.. card:: AeHMC
    :link: https://github.com/aesara-devs/aehmc

    Implementations of the HMC and NUTS sampler in Aesara


While these projects are related to probabilistics modelling, Aesara is much more general and can be used to improve any machine learning project.

Community
=========

* Visit `aesara-users`_ to discuss the general use of Aesara with developers and other users
* We use `GitHub issues <http://github.com/aesara-devs/aesara/issues>`__ to
  keep track of issues and `GitHub Discussions <https://github.com/aesara-devs/aesara/discussions>`__ to discuss feature
  additions and design changes


.. toctree::
    :maxdepth: 1
    :hidden:

    introduction
    install
    tutorial/index
    extending/index
    dev_start_guide
    optimizations
    library/index
    troubleshooting
    glossary
    links
    internal/index
    acknowledgement


.. _Theano: https://github.com/Theano/Theano
.. _aesara-users: https://gitter.im/aesara-devs/aesara
