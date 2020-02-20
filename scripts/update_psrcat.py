"""Query the database for pulsar holography sources and compile them into a
FluxCatalog-style JSON file based on the ATNF pulsar catalog.
"""

import psrqpy
from ch_util import holography as holo
from chimedb.core import connect
from ch_util.fluxcat import FluxCatalog, NumpyEncoder
import numpy as np
import json

# Get list of pulsars in database
connect()
pulsars = [
    p.name
    for p in holo.HolographySource.select().where(
        holo.HolographySource.name.regexp("^[BJ][0-9]{4}\+[0-9]*$")
    )
]
print("Found {:d} pulsars in database.".format(len(pulsars)))

# Query ATNF catalog
psr_par = ["JNAME", "BNAME", "RAJD", "DECJD", "S400", "S600", "S700", "S900", "S800"]
specs = psrqpy.QueryATNF(params=psr_par, psrs=pulsars)

# Construct FluxCat formatted dict
cat = {}
for spec in specs.table:
    # Get possible pulsar names
    alt_names = [spec[n] for n in ["JNAME", "BNAME"] if not np.ma.is_masked(spec[n])]

    # find the name used in the database
    name = None
    for psr in pulsars:
        if psr in alt_names:
            name = psr
            break
    if name is None:
        print(
            "Failed to match ATNF entry {} to queried database pulsars.".format(
                alt_names
            )
        )
        continue

    # Create a new catalog entry
    if name in FluxCatalog:
        print("{} already in catalog. Skipping.".format(name))
        print("Alt names: {}".format(alt_names))
        print("Found: {}".format(FluxCatalog[name].name))
        continue

    # Add flux measurements
    entry = FluxCatalog(
        name, ra=spec["RAJD"], dec=spec["DECJD"], alternate_names=alt_names
    )
    for m in ["S400", "S600", "S700", "S900", "S800"]:
        # Add flux measurements if available
        if not np.ma.is_masked(spec[m]):
            err = None if np.ma.is_masked(spec[m + "_ERR"]) else 1e-3 * spec[m + "_ERR"]
            entry.add_measurement(float(m[1:]), 1e-3 * spec[m], err, True, "ATNF")

    cat[name] = entry.to_dict()

# Write to file
with open("./atnf_psrcat.json", "w") as fh:
    json.dump(cat, fh, cls=NumpyEncoder, indent=4)
