"""
General CHIME utilities

Submodules
==========

.. autosummary::
    :toctree: _autosummary

    andata
    cal_utils
    chan_monitor
    data_quality
    ephemeris
    finder
    fluxcat
    hfbcat
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
