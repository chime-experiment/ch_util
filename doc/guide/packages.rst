=======================
CHIME Software Packages
=======================

This document contains a list software packages relevant to the CHIME
project.

Access
------

All repositories are hosted at either https://github.com or
https://bitbucket.org. You will in general need accounts at both these
websites. You will need to be granted access to any private repositories as
well as write privileges for the public ones.  You can request access by
emailing Adam, Kiyo, Richard or Kevin, being sure to include your GitHub or
Bitbucket user name in the email as appropriate.

.. _external_dependencies:

External Dependencies
---------------------

Python 2.7 is required.

In addition to the those listed here, there are several external third-party
python packages that are required. These are ``numpy``, ``scipy``, ``Cython``,
``h5py``, ``healpy``, ``networkx``, ``MySQL-python`` and ``peewee``. ``ipython`` and
``matplotlib`` are also highly recommended. These should all be available from
your operating system's package manager, as well as the PyPI, the Python
Package Index (for installation using ``pip`` or ``easy-install``).

Finally, the hdf5 library is required along with C header files. In Ubuntu these
are found in the "libhdf5" and "libhdf5-dev" packages. In MacPorts these are in
the "hdf5" port.

Packages
--------

bitshuffle
''''''''''

Compression/decompression library

:homepage: https://github.com/kiyo-masui/bitshuffle
:dependencies: h5py, numpy
:access: public
:documentation: https://github.com/kiyo-masui/bitshuffle


ch_util
'''''''

CHIME specific utilities.

:homepage: https://bitbucket.org/chime/ch_util
:dependencies: h5py, numpy, networkx, MySQL-python, pyephem, peewee, bitshuffle, PyYAML
:access: private
:documentation: http://e-mode.phas.ubc.ca/chime/codedoc/ch_util/


ch_acq
''''''

CHIME data acquisition software. Not required for data analysis.

:homepage: https://bitbucket.org/chime/ch_acq
:access: private


caput
'''''

Cluster Astronomical Python Utilities. Non-CHIME-specific utilities for
computer clusters and large datasets.

:homepage: https://github.com/kiyo-masui/caput
:dependencies: numpy, h5py
:access: public

ch_analysis
'''''''''''

CHIME data analysis.

:homepage: https://github.com/CITA/ch_analysis
:dependencies: caput, driftscan, ch_util
:access: private
:documentation: http://e-mode.phas.ubc.ca/chime/codedoc/ch_analysis/

cora
''''

Cosmology in the Radio Band. Generates sky models and sky simulations.

:homepage: https://github.com/jrs65/cora
:dependencies: numpy, scipy, Cython, healpy, h5py
:access: public


driftscan
'''''''''

Simulation and data analysis for transit telescopes

:homepage: https://github.com/CITA/driftscan
:dependencies: numpy, scipy, h5py, cora
:access: private

