# === Start Python 2/3 compatibility
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614
# === End Python 2/3 compatibility

from future.utils import bytes_to_native_str

from setuptools import setup, find_packages

from ch_util import __version__

# TODO: Python 3 - work around for setuptools issues in Py 2.7
ch_util_data = {
    bytes_to_native_str(b'ch_util.tests'): ['data/*/*'],
    bytes_to_native_str(b'ch_util') : ['catalogs/*.json']
}

setup(
    name='ch_util',
    version=__version__,

    packages=find_packages(),

    install_requires=[
        'chimedb @ git+ssh://git@github.com/chime-experiment/chimedb.git',
        'chimedb.config @ git+ssh://git@github.com/chime-experiment/chimedb_config.git',
        'chimedb.data_index @ git+ssh://git@github.com/chime-experiment/chimedb_di.git',
        'caput @ git+https://github.com/radiocosmology/caput.git',
        'bitshuffle @ git+https://github.com/kiyo-masui/bitshuffle.git',
        'numpy >= 1.15.1', 'scipy', 'networkx >= 2.0', 'h5py',
        'peewee >= 3.10', 'skyfield >= 1.10', 'future'
    ],

    package_data=ch_util_data,
    # metadata for upload to PyPI
    author="CHIME collaboration",
    author_email="richard@phas.ubc.ca",
    description="Utilities for CHIME.",
    license="GPL v3.0",
    url="https://bitbucket.org/chime/ch_util"
)
