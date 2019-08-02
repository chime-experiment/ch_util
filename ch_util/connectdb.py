"""Deprecated.  Use the chimedb.core package"""

import warnings

warnings.warn("The ch_util.connectdb module is deprecated.  Use the chimedb package instead")

from chimedb.core.connectdb import (
        NoRouteToDatabase,
        ConnectionError,

        DATABASE,
        HOST,
        PORT,
        USER,
        PASSWD,
        RW_USER,
        RW_PASSWD,
        TUNNEL_HOST,
        TUNNEL_USER,
        TUNNEL_IDENTITY,
        LOCALHOST,
        RC_DIR,
        RC_FILE,

        ALL_RANKS,
        DRAO_HOST,
        DRAO_BACKUP_HOST,

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

        DEFAULT_CONNECTORS,
        DEFAULT_CONNECTORS_RW,

        connect,
        )
