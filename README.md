# ch_util

General CHIME Utilities.

## Obtaining `ch_util`

To get `ch_util`, use git to clone the repository:

	$ git clone git@github.com:chime-experiment/ch_util.git


## Python Installation
`ch_util` is availaible as python package in the following flavors with increasing dependency requirements.

|                          Availaible Modules                         |         Local Installation         | PIP                                                                     |
|:-------------------------------------------------------------------:|:----------------------------------:|-------------------------------------------------------------------------|
| cal_utils, ephemeris, fluxcat, tools, rfi                           | poetry install                     | pip install git@github.com:CHIMEFRB/ch_util.git#ch_util                 |
| andata, timing, data_quality, ni_utils, plot                        | poetry install -E data             | pip install git@github.com:CHIMEFRB/ch_util.git#ch_util[data]           |
| _db_tables, chan_monitor, connectdb, data_index, finder, holography | poetry install -E data -E database | pip install git@github.com:CHIMEFRB/ch_util.git#ch_util[data, database] |

### Additional Requirements

```bash
# Linux Packages Required
sudo apt-get install libmysqlclient-dev libopenmpi-dev libhdf5-dev -y
```

```bash
# Environment Variables
HDF5_DIR="/usr/lib/x86_64-linux-gnu/hdf5/serial/"
```



Alternatively installation can be done by `pip` directly from the GitHub repository:

    $ pip install git+https://github.com/chime-experiment/ch_util.git

### CHIME Database Configuration

To install the `ch_util` python package with configuration to connect to the CHIME database:

    $ pip install .[chimedb_config]

With this configuration, the connection to the CHIME database from the standard locations should
be automatic. Standard locations include:

* DRAO
* `cedar`
* `niagara`
* UBC PHAS network

By default, connections from everywhere else use a restricted account called `chunnel` on the
database server to tunnel the database connection to your local computer.  To use this tunnel, send
your SSH public key (the contents of `~/.ssh/id_rsa.pub`) to the `bao` sysadmins
and ask to have your public key added to the list of chunnel authorized keys.


## Documentation

https://chime-experiment.github.io/ch_util