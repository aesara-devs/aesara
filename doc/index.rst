
Welcome
=======

Aesara is a Python library that allows you to define, optimize, and
evaluate mathematical expressions involving multi-dimensional
arrays efficiently. Aesara features:

* **tight integration with NumPy** -- Use `numpy.ndarray` in Aesara-compiled functions.
* **transparent use of a GPU** -- Perform data-intensive computations much faster than on a CPU.
* **efficient symbolic differentiation** -- Aesara does your derivatives for functions with one or many inputs.
* **speed and stability optimizations** -- Get the right answer for ``log(1+x)`` even when ``x`` is really tiny.
* **dynamic C code generation** -- Evaluate expressions faster.
* **extensive unit-testing and self-verification** -- Detect and diagnose many types of errors.

Aesara is based on `Theano`_, which has been powering large-scale computationally
intensive scientific investigations since 2007.

Download
========

Aesara is `available on PyPI`_, and can be installed via ``pip install Aesara``.

Those interested in bleeding-edge features should obtain the latest development
version, available via::

    git clone git://github.com/pymc-devs/aesara.git

You can then place the checkout directory on your ``$PYTHONPATH`` or use
``python setup.py develop`` to install a ``.pth`` into your ``site-packages``
directory, so that when you pull updates via Git, they will be
automatically reflected the "installed" version. For more information about
installation and configuration, see :ref:`installing Aesara <install>`.

.. _available on PyPI: http://pypi.python.org/pypi/aesara
.. _Related Projects: https://github.com/pymc-devs/aesara/wiki/Related-projects

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
* :ref:`release` -- How our release should work.
* :ref:`acknowledgement` -- What we took from other projects.
* `Related Projects`_ -- link to other projects that implement new functionalities on top of Aesara


.. _aesara_community:

* Visit `theano-users`_ if you want to talk to all Theano users.

* Visit `theano-dev`_ if you want to talk to the developers.

* Ask/view questions/answers at `StackOverflow`_

* We use `Github tickets <http://github.com/pymc-devs/aesara/issues>`__ to keep track of issues.

.. toctree::
   :maxdepth: 1
   :hidden:

   NEWS
   introduction
   requirements
   install
   updating
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
   LICENSE


.. _Theano: https://github.com/Theano/Theano
.. _theano-dev: http://groups.google.com/group/theano-dev
.. _theano-users: http://groups.google.com/group/theano-users
.. _StackOverflow: http://stackoverflow.com/questions/tagged/theano
