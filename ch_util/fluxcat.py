"""
This module is deprecated.

Use `radiocosmology.fluxcat`.
"""

import warnings

from fluxcat import (
    FitSpectrum as FitSpectrum,
    FluxCatalog as FluxCatalog,
    CurvedPowerLaw as CurvedPowerLaw,
    MetaFluxCatalog as MetaFluxCatalog,
    get_epoch as get_epoch,
    varname as varname,
    format_source_name as format_source_name,
    NumpyEncoder as NumpyEncoder,
    json_numpy_obj_hook as json_numpy_obj_hook,
    FREQ_NOMINAL as FREQ_NOMINAL,
    DEFAULT_COLLECTIONS as DEFAULT_COLLECTIONS,
)

warnings.warn(
    "The `ch_util.fluxcat` module is deprecated. "
    "Use `fluxcat` from `radiocosmology`.",
    DeprecationWarning,
)
