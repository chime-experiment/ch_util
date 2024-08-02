"""
This module is deprecated.

* for the Data Index tables,           use :mod:`~chimedb.data_index`.
* for the Finder and data flag tables, use :mod:`~ch_util.finder`.
* for holography tables,               use :mod:`~ch_util.holography`.
* for layout tables,                   use :mod:`~ch_util.layout`.
"""

import warnings

# Restore all the public symbols

from chimedb.core.orm import (
    JSONDictField as JSONDictField,
    EnumField as EnumField,
    base_model as base_model,
    name_table as name_table,
)

from chimedb.core import ValidationError as Validation  # noqa F401
from chimedb.core import AlreadyExistsError as AlreadyExists  # noqa F401
from chimedb.core import InconsistencyError as DataBaseError  # noqa F401

from chimedb.data_index import (
    AcqType as AcqType,
    ArchiveAcq as ArchiveAcq,
    ArchiveFile as ArchiveFile,
    ArchiveFileCopy as ArchiveFileCopy,
    ArchiveFileCopyRequest as ArchiveFileCopyRequest,
    ArchiveInst as ArchiveInst,
    CalibrationGainFileInfo as CalibrationGainFileInfo,
    CorrAcqInfo as CorrAcqInfo,
    CorrFileInfo as CorrFileInfo,
    DigitalGainFileInfo as DigitalGainFileInfo,
    FileType as FileType,
    FlagInputFileInfo as FlagInputFileInfo,
    HKAcqInfo as HKAcqInfo,
    HKFileInfo as HKFileInfo,
    HKPFileInfo as HKPFileInfo,
    RawadcAcqInfo as RawadcAcqInfo,
    RawadcFileInfo as RawadcFileInfo,
    StorageGroup as StorageGroup,
    StorageNode as StorageNode,
    WeatherFileInfo as WeatherFileInfo,
)

from chimedb.dataflag import (
    DataFlagType as DataFlagType,
    DataFlag as DataFlag,
)

from chimedb.data_index.util import (
    fname_atmel as fname_atmel,
    md5sum_file as md5sum_file,
    parse_acq_name as parse_acq_name,
    parse_corrfile_name as parse_corrfile_name,
    parse_weatherfile_name as parse_weatherfile_name,
    parse_hkfile_name as parse_hkfile_name,
    detect_file_type as detect_file_type,
)

from ._db_tables import connect_peewee_tables as connect_database  # noqa F401

from .holography import (
    QUALITY_GOOD as QUALITY_GOOD,
    QUALITY_OFFSOURCE as QUALITY_OFFSOURCE,
    ONSOURCE_DIST_TO_FLAG as ONSOURCE_DIST_TO_FLAG,
    HolographySource as HolographySource,
    HolographyObservation as HolographyObservation,
)

from ._db_tables import (
    EVENT_AT as EVENT_AT,
    EVENT_BEFORE as EVENT_BEFORE,
    EVENT_AFTER as EVENT_AFTER,
    EVENT_ALL as EVENT_ALL,
    ORDER_ASC as ORDER_ASC,
    ORDER_DESC as ORDER_DESC,
    NoSubgraph as NoSubgraph,
    BadSubgraph as BadSubgraph,
    DoesNotExist as DoesNotExist,
    UnknownUser as UnknownUser,
    NoPermission as NoPermission,
    LayoutIntegrity as LayoutIntegrity,
    PropertyType as PropertyType,
    PropertyUnchanged as PropertyUnchanged,
    ClosestDraw as ClosestDraw,
    event_table as event_table,
    set_user as set_user,
    graph_obj as graph_obj,
    global_flag_category as global_flag_category,
    global_flag as global_flag,
    component_type as component_type,
    component_type_rev as component_type_rev,
    external_repo as external_repo,
    component as component,
    component_history as component_history,
    component_doc as component_doc,
    connexion as connexion,
    property_type as property_type,
    property_component as property_component,
    property as property,
    event_type as event_type,
    timestamp as timestamp,
    event as event,
    predef_subgraph_spec as predef_subgraph_spec,
    predef_subgraph_spec_param as predef_subgraph_spec_param,
    user_permission_type as user_permission_type,
    user_permission as user_permission,
    compare_connexion as compare_connexion,
    add_component as add_component,
    remove_component as remove_component,
    set_property as set_property,
    make_connexion as make_connexion,
    sever_connexion as sever_connexion,
    connect_peewee_tables as connect_peewee_tables,
)

from .finder import (
    GF_REJECT as GF_REJECT,
    GF_RAISE as GF_RAISE,
    GF_WARN as GF_WARN,
    GF_ACCEPT as GF_ACCEPT,
    Finder as Finder,
    DataIntervalList as DataIntervalList,
    BaseDataInterval as BaseDataInterval,
    CorrDataInterval as CorrDataInterval,
    DataInterval as DataInterval,
    HKDataInterval as HKDataInterval,
    WeatherDataInterval as WeatherDataInterval,
    FlagInputDataInterval as FlagInputDataInterval,
    CalibrationGainDataInterval as CalibrationGainDataInterval,
    DigitalGainDataInterval as DigitalGainDataInterval,
    files_in_range as files_in_range,
    DataFlagged as DataFlagged,
)

warnings.warn("The ch_util.data_index module is deprecated.")
