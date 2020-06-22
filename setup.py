# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

from future.utils import bytes_to_native_str

from setuptools import setup, find_packages

import versioneer

# TODO: Python 3 - work around for setuptools issues in Py 2.7
ch_util_data = {
    bytes_to_native_str(b"ch_util.tests"): ["data/*/*"],
    bytes_to_native_str(b"ch_util"): ["catalogs/*.json"],
}

# Load the PEP508 formatted requirements from the requirements.txt file. Needs
# pip version > 19.0
with open("requirements.txt", "r") as fh:
    requires = fh.readlines()

setup(
    name="ch_util",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    install_requires=requires,
    extras_require={
        "chimedb_config": [
            "chimedb.config @ git+https://github.com/chime-experiment/chimedb_config.git"
        ],
    },
    package_data=ch_util_data,
    # metadata for upload to PyPI
    author="CHIME collaboration",
    author_email="richard@phas.ubc.ca",
    description="Utilities for CHIME.",
    license="GPL v3.0",
    url="https://bitbucket.org/chime/ch_util",
)
