============
Using SciNet
============

Most of the heavy lifting for CHIME data analysis is done at SciNet.
This page is a general reference for using SciNet for CHIME data analysis.

The primary resource for SciNet users is always the `Scinet Wiki`_.

.. _`SciNet Wiki`: https://support.scinet.utoronto.ca/wiki/index.php

Getting an Account
------------------

To get a SciNet Account and link it to the CHIME resource allocation:

#. Sign up for a `Compute Canada account`_.
#. Log in and link your account to the CHIME allocation on SciNet.  The PI for the
   allocation is Kris Sigurdson and his CCRI is ``kkf-325-01``.
#. Wait for Kris to approve your account.

.. _`Compute Canada account`: https://computecanada.ca/index.php/en/apply-for-an-account


Disk Quotas
-----------

The CHIME collaboration has and 50TB (?) quota on ``/scratch`` and over 200TB
(?) on the High Performance Storage System (HPSS) which is used for long term
storage of data not requiring frequent access.  Instructions on using HPSS can
be found here_.  Note that each user may only use up to 10GB (?) in their home
directory, so this should not be used to store data of any kind.

.. _here: https://support.scinet.utoronto.ca/wiki/index.php/HPSS

Accessing the Data
------------------

We will need scripts that painlessly move data from HPSS to ``/scratch/`` and
put it in a standard location.


Pre-installed CHIME Python Environment
--------------------------------------

There is a standard python distribution installed onto SciNet for use with the
CHIME analysis. It has all the standard CHIME analysis tools, plus a set of
useful packages. It is built against optimised numerical libraries for efficiency,

In theory this should be a *stable* version of all the tools, and will be
upgraded every now and again to a new stable version. There will probably be
unstable versions in the future for people who want to stay on the bleeding
edge.

To load it, you should add::

	source /home/k/krs/jrs65/chime/stable.sh

to your ``.bashrc`` file. This will load the required system wide modules, put
the new python distribution in your path, and load a virtualenv with all the
CHIME analysis tools.

You should be careful when loading other system modules. Check your
``.bashrc`` file to see if there are any other ``module`` commands, and be
careful to ensure they do not conflict.


Installed Packages
''''''''''''''''''

Packages installed in the base distribution:

* ``Python 2.7.6``
* ``ipython 1.1`` - plus all the extras to make the notebooks work
* ``matplotlib 1.3``
* ``numpy 1.8`` - compiled against Intel MKL for fast linear algebra
* ``scipy 0.13`` - compiled against Intel MKL for fast linear algebra
* ``cython 0.19``
* ``h5py 2.2`` - not compiled against MPI capable ``hdf5``
* ``mpi4py 1.3`` - built against ``OpenMPI 1.4``
* ``healpy 1.6``
* ``pyephem 3.7.5.1``
* ``pyfits 3.1``

CHIME packages installed in the virtualenv:

* ``cora``
* ``driftscan``
* ``ch_util``
* ``ch_gpib``

For a more comprehensive list of what's installed, run::

	$ pip list


Developing a CHIME Package on top of Pre-installed Version
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

.. note:: Coming Soon


Running IPython Notebook on SciNet
----------------------------------

The `IPython Notebook`_ is an extremely useful way of running python
computations in an interactive document. As it runs through your web browser
it is ideally suited to remote computation.

As SciNet is firewalled a little trickery is needed to get it running. There
are two parts: setting up an ssh tunnel through the firewall; and setting the
IPython notebook going using a simple helper script.

Setting up the Tunnel
'''''''''''''''''''''

First we will set up the ``ssh`` tunnel, for which we need to find the port we
will use for the tunnel. On **SciNet** run::

	$ ipnsession info

and note down the port number it gives back. We can set up the tunnel in two
ways, for a one off we can we can run a single ``ssh`` command on your
computer::

	$ ssh -L 6500:gpc01:<port> <user>@login.scinet.utoronto.ca

which will initialise the tunnel. Replace ``<port>`` and ``<user>`` with the
value noted down, and your SciNet username respectively.

For a more permanent solution you can modify you ``.ssh/config`` file to save
these values. Simply add the following snippet (filling in the ``<...>``
blanks):

.. code-block:: ini

	Host scinet
	    Hostname login.scinet.utoronto.ca
	    User jrs65
	    LocalForward 6500  gpc01:<port>

Now when you login to SciNet simply use::

	$ ssh scinet

and the tunnel will automatically get started.

Launching IPython
'''''''''''''''''

The second part of this process is to launch an IPython session. For this
there is now a handy helper script ``ipnsession`` which is automatically
available if you are using the CHIME python environoment. This will start an
IPython session either on the current machine, or on a cluster node (which it
will queue up for you), and then it will connect to the end of the tunnel we
started earlier.

To run on the current node (should be a development node: ``gpc01``,
``gpc02``, ``gpc03`` or ``gpc04``) just run::

	$ ipnsession dev

Now it's running you should be able to access IPython through the tunnel by
firing up a browser on your machine and going to ``http://localhost:6500``.

Development nodes have a limit on the amount of CPU time they can use, and the
session will be killed if this is exceeded. If you are running an intensive
session you will need to run on a compute node, which unfortunately needs to
be allocated through the queue. However, this process can be done simply with
the command::

	$ ipnsession cluster <TIME IN MINS>

where you must fill out the time in minutes you want the job to run for. This
will place a job in the queue, which when it starts will fire up IPython and
finish setting up the tunnel. You can check to see if the job has started with
``qstat``, when it has it will be accessible at ``http://localhost:6500``.

.. _`IPython Notebook`: http://ipython.org/