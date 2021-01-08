from setuptools import setup, find_packages

import versioneer

ch_util_data = {
    "ch_util": ["catalogs/*.json"],
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
    python_requires=">=3.7",
    install_requires=requires,
    extras_require={
        "chimedb_config": [
            "chimedb.config @ git+https://github.com/chime-experiment/chimedb_config.git"
        ]
    },
    package_data=ch_util_data,
    # metadata for upload to PyPI
    author="CHIME collaboration",
    author_email="richard@phas.ubc.ca",
    description="Utilities for CHIME.",
    license="GPL v3.0",
    url="https://bitbucket.org/chime/ch_util",
)
