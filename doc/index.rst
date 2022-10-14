
Welcome
=======

Aesara is a Python library that allows you to define, optimize/rewrite, and
evaluate mathematical expressions involving multi-dimensional arrays
efficiently.

Some of Aesara's features are:

* **Tight integration with NumPy**
  - Use `numpy.ndarray` in Aesara-compiled functions
* **Efficient symbolic differentiation**
  - Aesara efficiently computes your derivatives for functions with one or many inputs
* **Speed and stability optimizations**
  - Get the right answer for ``log(1 + x)`` even when ``x`` is near zero
* **Dynamic C/JAX/Numba code generation**
  - Evaluate expressions faster

Aesara is based on `Theano`_, which has been powering large-scale computationally
intensive scientific investigations since 2007.


.. warning::

   Much of the documentation hasn't been updated and is simply the old Theano documentation.

Download
========

Aesara is `available on PyPI`_, and can be installed via ``pip install Aesara``.

Those interested in bleeding-edge features should obtain the latest development
version, available via::

    git clone git://github.com/aesara-devs/aesara.git

You can then place the checkout directory on your ``$PYTHONPATH`` or use
``python setup.py develop`` to install a ``.pth`` into your ``site-packages``
directory, so that when you pull updates via Git, they will be
automatically reflected the "installed" version. For more information about
installation and configuration, see :ref:`installing Aesara <install>`.

.. _available on PyPI: http://pypi.python.org/pypi/aesara
.. _Related Projects: https://github.com/aesara-devs/aesara/wiki/Related-projects

Documentation
=============

Roughly in order of what you'll want to check out:

* :ref:`install` -- How to install Aesara.
* :ref:`introduction` -- What is Aesara?
* :ref:`tutorial` -- Learn the basics.
* :ref:`troubleshooting` -- Tips and tricks for common debugging.
* :ref:`libdoc` -- Aesara's functionality, module by module.
* :ref:`faq` -- A set of commonly asked questions.
* :ref:`optimizations` -- Guide to Aesara's graph optimizations.
* :ref:`extending` -- Learn to add a Type, Op, or graph optimization.
* :ref:`dev_start_guide` -- How to contribute code to Aesara.
* :ref:`internal` -- How to maintain Aesara and more...
* :ref:`acknowledgement` -- What we took from other projects.
* `Related Projects`_ -- link to other projects that implement new functionalities on top of Aesara


.. _aesara-community:

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
