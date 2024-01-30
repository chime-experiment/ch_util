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

from . import _version

__version__ = _version.get_versions()["version"]
