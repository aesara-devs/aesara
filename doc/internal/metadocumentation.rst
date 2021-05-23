.. _metadocumentation:

==================================================
Documentation Documentation AKA Meta-Documentation
==================================================


How to build documentation
--------------------------

Let's say you are writing documentation, and want to see the `sphinx
<http://sphinx.pocoo.org/>`__ output before you push it.
The documentation will be generated in the ``html`` directory.

.. code-block:: bash

    cd Aesara/
    python ./doc/scripts/docgen.py

If you don't want to generate the pdf, do the following:

.. code-block:: bash

    cd Aesara/
    python ./doc/scripts/docgen.py --nopdf


For more details:

.. code-block:: bash

   $ python doc/scripts/docgen.py --help
   Usage: doc/scripts/docgen.py [OPTIONS]
     -o <dir>: output the html files in the specified dir
     --rst: only compile the doc (requires sphinx)
     --nopdf: do not produce a PDF file from the doc, only HTML
     --help: this help

Use ReST for documentation
--------------------------

 * `ReST <http://docutils.sourceforge.net/rst.html>`__ is standardized.
   trac wiki-markup is not.
   This means that ReST can be cut-and-pasted between code, other
   docs, and TRAC.  This is a huge win!
 * ReST is extensible: we can write our own roles and directives to automatically link to WIKI, for example.
 * ReST has figure and table directives, and can be converted (using a standard tool) to latex documents.
 * No text documentation has good support for math rendering, but ReST is closest: it has three renderer-specific solutions (render latex, use latex to build images for html, use itex2mml to generate MathML)


How to link to class/function documentations
--------------------------------------------

Link to the generated doc of a function this way::

    :func:`perform`

For example::

    of the :func:`perform` function.

Link to the generated doc of a class this way::

    :class:`RopLop_checker`

For example::

    The class :class:`RopLop_checker`, give the functions


However, if the link target is ambiguous, Sphinx will generate warning or errors.


How to add TODO comments in Sphinx documentation
-------------------------------------------------

To include a TODO comment in Sphinx documentation, use an indented block as
follows::

    .. TODO: This is a comment.
    .. You have to put .. at the beginning of every line :(
    .. These lines should all be indented.

It will not appear in the output generated.

    .. TODO: Check it out, this won't appear.
    .. Nor will this.
