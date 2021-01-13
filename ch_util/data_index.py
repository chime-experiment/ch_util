"""
This module is deprecated.

* for the Data Index tables,           use :mod:`~chimedb.data_index`.
* for the Finder and data flag tables, use :mod:`~ch_util.finder`.
* for holography tables,               use :mod:`~ch_util.holography`.
* for layout tables,                   use :mod:`~ch_util.layout`.
"""

import warnings

warnings.warn("The ch_util.data_index module is deprecated.")

# Restore all the public symbols

from . import layout, ephemeris

from chimedb.core.orm import JSONDictField, EnumField, base_model, name_table

from chimedb.core import ValidationError as Validation
from chimedb.core import AlreadyExistsError as AlreadyExists
from chimedb.core import InconsistencyError as DataBaseError

from chimedb.data_index import (
    AcqType,
    ArchiveAcq,
    ArchiveFile,
    ArchiveFileCopy,
    ArchiveFileCopyRequest,
    ArchiveInst,
    CalibrationGainFileInfo,
    CorrAcqInfo,
    CorrFileInfo,
    DigitalGainFileInfo,
    FileType,
    FlagInputFileInfo,
    HKAcqInfo,
    HKFileInfo,
    HKPFileInfo,
    RawadcAcqInfo,
    RawadcFileInfo,
    StorageGroup,
    StorageNode,
    WeatherFileInfo,
)

from chimedb.dataflag import DataFlagType, DataFlag

from chimedb.data_index.util import (
    fname_atmel,
    md5sum_file,
    parse_acq_name,
    parse_corrfile_name,
    parse_weatherfile_name,
    parse_hkfile_name,
    detect_file_type,
)

from ._db_tables import connect_peewee_tables as connect_database

from .holography import (
    QUALITY_GOOD,
    QUALITY_OFFSOURCE,
    ONSOURCE_DIST_TO_FLAG,
    HolographySource,
    HolographyObservation,
)

_property = property  # Do this since there is a class "property" in _db_tables.
from ._db_tables import (
    EVENT_AT,
    EVENT_BEFORE,
    EVENT_AFTER,
    EVENT_ALL,
    ORDER_ASC,
    ORDER_DESC,
    NoSubgraph,
    BadSubgraph,
    DoesNotExist,
    UnknownUser,
    NoPermission,
    LayoutIntegrity,
    PropertyType,
    PropertyUnchanged,
    ClosestDraw,
    event_table,
    set_user,
    graph_obj,
    global_flag_category,
    global_flag,
    component_type,
    component_type_rev,
    external_repo,
    component,
    component_history,
    component_doc,
    connexion,
    property_type,
    property_component,
    property,
    event_type,
    timestamp,
    event,
    predef_subgraph_spec,
    predef_subgraph_spec_param,
    user_permission_type,
    user_permission,
    compare_connexion,
    add_component,
    remove_component,
    set_property,
    make_connexion,
    sever_connexion,
    connect_peewee_tables,
)

from .finder import (
    GF_REJECT,
    GF_RAISE,
    GF_WARN,
    GF_ACCEPT,
    Finder,
    DataIntervalList,
    BaseDataInterval,
    CorrDataInterval,
    DataInterval,
    HKDataInterval,
    WeatherDataInterval,
    FlagInputDataInterval,
    CalibrationGainDataInterval,
    DigitalGainDataInterval,
    files_in_range,
    DataFlagged,
)
