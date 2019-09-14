"""Deprecated.  Use the chimedb.core package"""

import warnings

warnings.warn(
    "The ch_util.connectdb module is deprecated.  Use the chimedb package instead"
)

from chimedb.core.connectdb import (
    NoRouteToDatabase,
    ConnectionError,
    ALL_RANKS,
    current_connector,
    connect_this_rank,
    MySQLDatabaseReconnect,
    BaseConnector,
    MySQLConnector,
    SqliteConnector,
    tunnel_active,
    create_tunnel,
    connected_mysql,
    close_mysql,
    connect,
)
