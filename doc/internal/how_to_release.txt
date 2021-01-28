.. _how_to_release:

==================================================
How to make a release
==================================================

Update the version number
=========================

``Aesara/doc/conf.py`` should be updated in the following ways:

 * Change the upper copyright year to the current year if necessary.

Update the year in the ``Aesara/LICENSE.txt`` file too, if necessary.

Update the code and the documentation for the aesara flags
``warn__ignore_bug_before`` to accept the new version. You must modify the
file ``aesara/configdefaults.py`` and ``doc/library/config.txt``.

Tag the release
===============

You will need to commit the previous changes, tag the resulting version, and
push that into the upstream/official repository.  After that, create a new release
via GitHub Releases on the repository's page.  The release tag must start with
``rel-`` in order to be recognized by the CI release process.

This will trigger and build and upload of the PyPI and Conda packages.

The documentation will be automatically regenerated as well.
