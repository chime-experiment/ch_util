"""Collect power usage data for the GPU nodes for a single day in January 2018.

Note: Uses archived metric for the outlet sensor 'active_power', as opposed
to 'apparent_power'.

The day is given as an argument on the comand line. Change the value of
'prestring' to operate on another month.

Author: Nolan Denman
"""

import numpy as np
import pandas as pd
import h5py
from ch_util.andata import HKPData
import sys, pickle

mets = ["pdu_outlet_sensor_value"]

onlen = 1440

day = int(sys.argv[1])
prestring = "201801"

acq_file = HKPData.from_acq_h5(
    "/mnt/gong/archive/{0}01T000000Z_chime_hkp/hkp_prom_{0}{1:02}.h5".format(
        prestring, day
    ),
    metrics=mets,
)
pdu_out_data = acq_file.select(mets[0])

rack_list = [
    "cn0cx",
    "cn1cx",
    "cn2cx",
    "cn3cx",
    "cn4cx",
    "cn5cx",
    "cn6cx",
    "cn8cx",
    "cn9cx",
    "cnAcx",
    "cnBcx",
    "cnCcx",
    "cnDcx",
    "cs0cx",
    "cs1cx",
    "cs2cx",
    "cs3cx",
    "cs4cx",
    "cs5cx",
    "cs6cx",
    "cs8cx",
    "csAcx",
    "cs9cx",
    "csBcx",
    "csCcx",
    "csDcx",
]

outv = []

for rack in rack_list:
    rack = rack[:3]
    rack_power_out = pdu_out_data.query(
        "sensor=='active_power' and instance=='{}pd'".format(rack)
    )

    max_nodes = 10
    if rack[2] == "D":
        max_nodes = 8

    for node_id in range(max_nodes):
        dat_vec = np.zeros(onlen)
        node_power = np.asarray(
            rack_power_out.query("device=='{}g{}'".format(rack, node_id)).value
        )
        plen = len(node_power)
        if plen == onlen:
            dat_vec += node_power
        elif plen > onlen:
            dat_vec += node_power[:onlen]
        elif (plen < onlen) and (plen > 0):
            dat_vec[:plen] += node_power
        outv.append(dat_vec)
        del node_power

outv = np.asarray(outv)
print(prestring, day, outv.shape)

outf = open("/home/denman/hk-uptime-{0}{1:02}".format(prestring, day), "wb")
pickle.dump(outv, outf)
outf.close()
