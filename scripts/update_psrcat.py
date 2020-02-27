"""Query the database for pulsar holography sources and compile them into a
FluxCatalog-style JSON file based on the ATNF pulsar catalog.
"""
import os

import numpy as np
import psrqpy

from chimedb.core import connect
from ch_util import holography as holo
from ch_util.fluxcat import FluxCatalog

CATALOG_NAME = os.path.abspath("./atnf_psrcat.json")
DEFAULT_FRAC_ERR = 0.20
DEFAULT_NPARAM = 3

# Delete any catalogs of radio bright sources that are loaded by default
for cat, sources in FluxCatalog.loaded_collections():
    FluxCatalog.delete_loaded_collection(cat)

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
flux_fields = [
    "S40",
    "S50",
    "S60",
    "S80",
    "S100",
    "S150",
    "S200",
    "S300",
    "S400",
    "S600",
    "S700",
    "S800",
    "S900",
    "S1400",
    "S1600",
    "S2000",
    "S3000",
    "S4000",
    "S5000",
    "S6000",
    "S8000",
]

psr_par = ["JNAME", "BNAME", "RAJD", "DECJD"] + flux_fields
specs = psrqpy.QueryATNF(params=psr_par, psrs=pulsars)

# Construct FluxCat formatted dict
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
    nmeas = sum([not np.ma.is_masked(spec[m]) for m in flux_fields])
    entry = FluxCatalog(
        name,
        ra=spec["RAJD"],
        dec=spec["DECJD"],
        alternate_names=alt_names,
        model="CurvedPowerLaw",
        model_kwargs={"nparam": min(DEFAULT_NPARAM, nmeas - 1)},
    )
    for m in flux_fields:
        # Add flux measurements if available
        if not np.ma.is_masked(spec[m]):
            err = (
                DEFAULT_FRAC_ERR * 1e-3 * spec[m]
                if np.ma.is_masked(spec[m + "_ERR"])
                else 1e-3 * spec[m + "_ERR"]
            )
            entry.add_measurement(float(m[1:]), 1e-3 * spec[m], err, True, u"ATNF")

    entry.fit_model()

# Dump to file
print("Saving catalog to: %s" % CATALOG_NAME)
FluxCatalog.dump(CATALOG_NAME)
