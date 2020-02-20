# ch_util

General CHIME utilities.

## Obtaining `ch_util`

To get `ch_util`, use git to clone the repository:

	$ git clone git@bitbucket.org:chime/ch_util.git


## Python Installation

To install the `ch_util` python package:

	$ python setup.py install [--user]

Or work in develop mode:

	$ python setup.py develop [--user]

Alternatively installation can be done by `pip` directly from the bitbucket repository:

	$ pip install git+ssh://git@bitbucket.org/chime/ch_util.git


## Connecting to the Database

Connection to the CHIME database from the standard locations should be automatic.  Standard
locations include:

* DRAO
* `cedar`
* `niagara`
* UBC PHAS network

By default, connections from everywhere else use a restricted account called `chunnel` on the
database server to tunnel the database connection to your local computer.  To use this tunnel, send
your SSH public key (the contents of `~/.ssh/id_rsa.pub`) to the `bao` sysadmins (an email to
`***REMOVED***` should suffice) and ask to have your public key added to the list of
chunnel authorized keys.
