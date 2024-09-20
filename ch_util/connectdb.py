"""Deprecated.  Use the chimedb.core package"""

import warnings

from chimedb.core.connectdb import (
    NoRouteToDatabase as NoRouteToDatabase,
    ConnectionError as ConnectionError,
    ALL_RANKS as ALL_RANKS,
    current_connector as current_connector,
    connect_this_rank as connect_this_rank,
    MySQLDatabaseReconnect as MySQLDatabaseReconnect,
    BaseConnector as BaseConnector,
    MySQLConnector as MySQLConnector,
    SqliteConnector as SqliteConnector,
    tunnel_active as tunnel_active,
    connected_mysql as connected_mysql,
    close_mysql as close_mysql,
    connect as connect,
)

warnings.warn(
    "The ch_util.connectdb module is deprecated.  Use the chimedb package instead"
)
