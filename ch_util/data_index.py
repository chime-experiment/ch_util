"""This module is deprecated.

* for the Data Index tables,           use :mod:`~chimedb.data_index`.
* for the Finder and data flag tables, use :mod:`~ch_util.finder`.
* for holography tables,               use :mod:`~ch_util.holography`.
* for layout tables,                   use :mod:`~ch_util.layout`.
"""

import warnings

warnings.warn("The ch_util.data_index module is deprecated.")

# Restore all the public symbols


_property = property  # Do this since there is a class "property" in _db_tables.
