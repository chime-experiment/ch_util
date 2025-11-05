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
    finder
    hfbcat
    holography
    layout
    ni_utils
    plot
    rfi
    timing
    tools
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ch_util")
except PackageNotFoundError:
    # package is not installed
    pass
del version, PackageNotFoundError
