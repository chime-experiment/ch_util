=========================
Developing CHIME Software
=========================

This document lists the basic policies, procedures and standards in use for
developing CHIME code.

Basics
======

Most CHIME software is written in python (and C?) and is under version control
using git.

Git Branching Model
===================

We use the `GitHub Flow`_ branching model in most repositories. Importantly
this means no one ever pushes to the production branch 'master'. 'master' is
only ever updated through `pull requests`_ from other 'feature' branches.

.. _`GitHub Flow`: http://scottchacon.com/2011/08/31/github-flow.html
.. _`pull requests`: https://help.github.com/articles/using-pull-requests

Code Style Standards
====================

We follow the python style guide laid out in PEP8_ as well as the docstring
conventions laid out in PEP257_.

.. _PEP8: http://www.python.org/dev/peps/pep-0008/
.. _PEP257: http://www.python.org/dev/peps/pep-0257/

Documentation
=============

Writing good docstrings and comments---which are kept up-to-date---is
considered the bare minimum in terms of documentation. Docstring
self-documentation can generally be propagated into the nightly built sphinx_
documentation with minimal effort. In addition developers are encouraged to
follow the `numpy docstring standard`_.

.. _sphinx: http://sphinx-doc.org/
.. _`numpy docstring standard`: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
