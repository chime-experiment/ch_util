=========================
CHIME Software Quickstart
=========================

This guide is intended to be complete instructions for setting up, testing
and doing running CHIME data analysis software.


Scinet
------

Most of the heavy lifting for CHIME data analysis is done at Scinet. If you are
setting up to run on Scinet, please read :doc:`this page <scinet>` first
before proceeding.


Install External Dependencies
-----------------------------

This section outlines several options for obtaining the external third
party software packages listed in the :ref:`external_dependencies` section of
the :doc:`Packages <packages>` page. On hosts that are commonly used by CHIME
collaborators (such as 'niedermayer') you can safely skip this step.

With Administrative Privileges
''''''''''''''''''''''''''''''

If you are working on a machine for which you have administrative privileges,
third party software is often best installed at the system level from your
operating system's package manager. Any packages that are not shipped with the
operating system can be obtained from the Python Package Index (A.K.A. PyPI)
through ``pip`` or ``easy-install``. On most systems this is quite straight
forward. As an example, I've outlined the procedure for Ubuntu 12.04 and Apple
OSX.

Ubuntu:

1. Use apt-get to install most packages (Python 2.7 is the default in 12.04)::

    $ sudo apt-get install python python-numpy python-scipy cython \
        python-h5py python-networkx python-mysqldb ipython python-matplotlib \
        libhdf5 libhdf5-dev

2. ``healpy`` and ``pyephem`` are not included in Ubuntu, so get it from PyPI using ``pip``::

    $ sudo pip install healpy pyephem


Apple OSX:

1. If you don't already have it, install MacPorts_ which is a package
   manager for OSX.
2. Use MacPorts to install required packages.  In a terminal::

    $ sudo port install python27 py27-numpy py27-scipy py27-matplotlib \
        py27-ipython py27-h5py py27-cython py27-healpy py27-networkx \
        py27-mysql py27-ephem hdf5

3. Use MacPorts to activate the Python 2.7 versions of some of the
   packages, making Python 2.7 the default for your system::

    $ sudo port select --set python python27
    $ sudo port select --set ipython ipython27

.. _MacPorts: http://www.macports.org/

.. _without_administrative_privileges:

Without Administrative Privileges
'''''''''''''''''''''''''''''''''

On shared systems, common third party software such as Python 2.7, ``numpy`` and
``scipy`` will probably already be present, either by default or in a loadable
module (use the commonly available shell command ``module avail`` to see
available modules). Figure out which packages are missing (see :ref:`verifying`
below) and install them locally in your ``home`` directory.

There are at least two good ways to install packages locally in your user
space.  The first and simplest is to use the python package manager ``pip``
(which will probably already be present on your system) with the
``--user`` option::
    
    $ pip install --user Cython h5py  # etc.

If ``pip`` is not installed, you can also try using ``easy_install``::

    $ easy_install install --user Cython h5py  # etc.

The second option is to use virtualenv_ which gives you complete control
over your python environment with no risk of messing things up.  See the
virtualenv documentation for details on how to set this up.

.. _virtualenv: http://www.virtualenv.org/en/latest/

.. _verifying:

Verifying
'''''''''

To check that you have Python 2.7 and are using it by default, type the
following into a shell::

    $ python --version   # Should print "Python 2.7.X"

To check that you have all the required python packages, start up a python
interpreter and try to import them::
    
    >>> import numpy
    >>> print numpy.__version__
    >>> import scipy  # and so on.


Get CHIME Code
--------------

This section details the proper way to get the CHIME software packages
listed on the :doc:`Packages <packages>`, such that you are able to clone
the code and push your changes back upstream.

Setting up GitHub and Bitbucket
'''''''''''''''''''''''''''''''

First sign up for accounts at both https://github.com and
https://bitbucket.org.  Now email the administrators of the packages you
need to access and ask them for push/pull access to the code repositories.
Be sure to include your new user names.  No need to wait for a reply to
continue.

Next ``cd`` to ``~/.ssh`` (create the directory if it doesn't exist) and
generate a new ``ssh`` RSA keypair using the ``ssh-keygen`` command.  When
prompted for an output filename use ``id_rsa_git``.  Encrypt the key with a
pass phrase.

Now, add the public key you just generated to your GitHub and Bitbucket
accounts. Print the key by typing ``cat ~/.ssh/id_rsa_git.pub`` into a
terminal, then copy and paste to GitHub/Bitbucket accounts (under "account
settings" -> "SSH Keys").

Finally, edit ``~/.ssh/config`` (create the file if it doesn't exist) to
include the following lines::

    host github.com
      user git
      identityfile ~/.ssh/id_rsa_git

    host bitbucket.org
      user git
      identityfile ~/.ssh/id_rsa_git

To make sure you you've set up your ssh keys correctly, try to ssh to
GitHub and Bitbucket::
    
    $ ssh github.com
    $ ssh bitbucket.org

You should get a messages like the following (as opposed to a simple "access
denied" message):
    
    PTY allocation request failed on channel 0

    Hi kiyo-masui! You've successfully authenticated, but GitHub does not
    provide shell access.
    
    Connection to github.com closed.

Getting the Code
''''''''''''''''

You can clone a local copy of any of the packages into the current
directory by typing something like the following into a shell::
    
    $ git clone git@github.com:kiyo-masui/bitshuffle.git
    $ git clone git@github.com:kiyo-masui/caput.git
    $ git clone git@bitbucket.org:CHIME/ch_util.git
    $ # etc.

The exact URL of the packages git repository (here
``git@github.com:kiyo-masui/caput.git`` and
``git@bitbucket.org:CHIME/ch_util.git``) can always be found on the package's
GitHub or BitBucket home page (listed on the :doc:`Packages <packages>` page).
Be sure to select the SSH URL, not the HTTPS URL.


Install CHIME Software
----------------------

Once you have the code you can install it on your system. All of the CHIME
packages are considered development software and it is therefore not
recommended that they be installed system-wide.  Instead, install them either
within a virtualenv or using the ``--user`` such that they are installed to
your user space (as described in the :ref:`without_administrative_privileges`
section).

In addition, it is recommended that these packages are installed in
`develop` mode.  Normally the installation scripts copy the code from the
repository into the install directory, such that changes to the repository
do not affect the modules imported by python.  In `develop` mode, a link is
created from the install directory to the repository, such that the
software executed by python always reflects the current state of the
repository.

The CHIME Software packages all include a `setuptools` ``setup.py`` script
for installing the package.  To install a package in your user space, in
`develop` mode::
    
    $ cd path/to/bitshuffle/
    $ python setup.py develop --user
    $ cd path/to/ch_util/
    $ python setup.py develop --user
    $ cd path/to/caput/
    $ python setup.py develop --user

Verifying
'''''''''

To verify that you have a package properly setup first make sure that you
can import the package in a python interpreter::

    >>> import caput

If this succeeds, the next step to to run any unit tests that might have
shipped with the package::
    
    $ python path/to/caput/caput/tests/test_memh5.py
    $ python path/to/ch_util/python/ch_util/tests/test_andata.py
    $ # etc.


Running the Analysis Pipeline
-----------------------------

This section gives instructions for running some basic simulation and data
analysis pipelines, with the aim of getting you started.  The full analysis
pipeline paradigm is explained in the documentation for ``ch_analysis``.

Example pipelines are found in "ch_analysis/input/examples/". Good
pipelines to get started with are "small_cylinder_sim.yaml" or
"read_convert_acq.yaml".  To run one of these examples:

1. Start by creating your own directory,
   "ch_analysis/input/[your-initials]/", where you can keep input files
   under version control without worrying about someone else modifying
   them.

2. Copy one of the example pipeline YAML files to your input directory.
3. You will likely need to make some small edits to the pipeline files to
   sort your input and output paths. Required edits should be clearly
   marked in comments within the file.
4. Now run the pipeline using the ``run_pipeline.py`` script::

    $ python path/to/ch_analysis/scripts/run_pipeline.py \
        path/to/ch_analysis/input/[your-initials]/small_cylinder_sim.yaml


Developing CHIME Software
-------------------------

Before writing any code, please read our
:doc:`Developer Guide <developer_guide>` and the links therein.

Seriously, read it.


Errors in this Document
-----------------------

Please report any errors or omissions in these instructions to Kiyo.

