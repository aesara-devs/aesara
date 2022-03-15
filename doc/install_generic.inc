.. include:: css.inc

.. _install_generic:


Installation
============

Stable Installation
-------------------

With ``conda``
^^^^^^^^^^^^^^

If you use conda, you can directly install aesara.
.. code-block:: bash

    conda install aesara

With ``pip``
^^^^^^^^^^^^

aesara
::::::

Install the latest stable version of Aesara with:

.. raw:: html

    <div class="highlight"><pre><span class="red">&lt;sudo&gt;</span> pip install <span class="blue">&lt;--user&gt;</span> Aesara[test, doc]</pre></div>

- Any argument between <...> is optional.

- Use :red:`sudo` for a root installation.

- Use :blue:`user` for a user installation without admin rights. It will install Aesara in your local site-packages.

- Use `pip install -r requirements.txt` to install the requirements for testing.

- Use `pip install -r requirements-rtd.txt` install the requirements for generating the documentation.

If you encountered any trouble, head to the :ref:`troubleshooting` page.


Bleeding-Edge Installation (recommended)
----------------------------------------

Install the latest, bleeding-edge, development version of Aesara with:

.. raw:: html

    <div class='highlight'><pre><span class="red">&lt;sudo&gt;</span> pip install <span class="blue">&lt;--user&gt;</span> <span class="pink">&lt;--no-deps&gt;</span> git+https://github.com/aesara-devs/aesara.git#egg=aesara</pre></div>

- Any argument between <...> is optional.

- Use :red:`sudo` for a root installation.

- Use :blue:`user` for a user installation without admin rights. It will install Aesara in your local site-packages.

- Use :pink:`no-deps` when you don't want the dependencies of Aesara to be installed through pip. This is important when they have already been installed as system packages.

If you encountered any trouble, head to the :ref:`troubleshooting` page.

Developer Installation
----------------------

Install the developer version of Aesara with:

.. raw:: html

    <div class="highlight"><pre>git clone git://github.com/aesara-devs/aesara.git
    cd aesara
    <span class="red">&lt;sudo&gt;</span> pip install <span class="blue">&lt;--user&gt;</span> <span class="pink">&lt;--no-deps&gt;</span> <span class="green">-e .</span></pre></div>

- Any argument between <...> is optional.

- Use :red:`sudo` for a root installation.

- Use :blue:`user` for a user installation without admin rights. It will install Aesara in your local site-packages.

- Use :pink:`no-deps` when you don't want the dependencies of Aesara to be installed through pip. This is important when they have already been installed as system packages.

- :green:`-e` makes your installation *editable*, i.e., it links it to your
  source directory.

If you encountered any trouble, head to the :ref:`troubleshooting` page.
