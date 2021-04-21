"""
General CHIME utilities

Submodules
==========

.. autosummary::
    :toctree: _autosummary

    _db_tables
    andata
    cal_utils
    chan_monitor
    connectdb
    data_index
    data_quality
    ephemeris
    finder
    fluxcat
    holography
    layout
    ni_utils
    plot
    rfi
    timing
    tools
"""
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
