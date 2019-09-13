"""Generates archive format 2.X test data from real archive data.

Data is written to the current directory.

"""
# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility


from os import path

import numpy as np
import h5py

from ch_util import andata

ARCHIVE_ROOT = "/mnt/gong/archive/"

ARCHIVE_VERSION = "2.2"

if ARCHIVE_VERSION == "2.0":
    ACQ = "20140916T173334Z_blanchard_corr"
    FILENAMES = ["00000000_0000.h5", "00010350_0000.h5"]
    # Data selection: what subset of data goes into each test data file.
    FREQ_SEL = np.s_[64:96]
    # Start and stop time indeces for each file.
    STARTS = [-31, 0]
    STOPS = [None, 17]
elif ARCHIVE_VERSION == "2.2":
    ACQ = "20160114T200138Z_pathfinder_corr"
    FILENAMES = ["00000001_0000.h5", "00022181_0000.h5"]
    # Data selection: what subset of data goes into each test data file.
    FREQ_SEL = np.s_[64:66]
    # Start and stop time indeces for each file.
    STARTS = [-3, 0]
    STOPS = [None, 5]

OUT_FILENAMES = ["00000000_0000.h5", "00000010_0000.h5"]

paths = [path.join(ARCHIVE_ROOT, ACQ, f) for f in FILENAMES]


def main():
    # Open data files and cast as andata objects.
    data_list = [andata.CorrData(h5py.File(p, "r")) for p in paths]

    # Define a dataset filter that takes the first 64 frequenies.
    def dset_filter(dataset):
        # Must have this attribute.
        if "freq" in dataset.attrs["axis"]:
            # Must be first axis.
            if dataset.attrs["axis"][0] != "freq":
                raise RuntimeError("Expected 'freq' to be zeroth axis.")
            dataset = dataset[FREQ_SEL]
        return dataset

    for ii, d in enumerate(data_list):
        out_f = h5py.File(OUT_FILENAMES[ii], "w")
        tdata = andata.concatenate(
            [d],
            start=STARTS[ii],
            stop=STOPS[ii],
            out_group=out_f,
            dataset_filter=dset_filter,
        )
        # Adjust the frequency index map.
        freq = out_f["index_map/freq"][FREQ_SEL]
        del out_f["index_map/freq"]
        out_f.create_dataset("index_map/freq", data=freq)

        # Adjust the attributes. XXX others?
        out_f.attrs["n_freq"] = [len(freq)]

        out_f.close()


if __name__ == "__main__":
    main()
