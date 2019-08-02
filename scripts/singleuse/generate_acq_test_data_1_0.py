"""Genrates small acq format data files from large acq data file.

Extracts two small subsets of the frames in an acq format hdf5 file and writes
them separate test files.  Duplicates the meta-data exactly.

"""
# === Start Python 2/3 compatibility
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614
# === End Python 2/3 compatibility

import sys

import h5py

from ch_util.caput.memh5 import copyattrs, deep_group_copy, is_group

OUT_FILES = ['test_acq.h5.0001', 'test_acq.h5.0002']
START_FRAMES = [15, 28]
END_FRAMES = [28, 46]

def generate(filename):
    f_in = h5py.File(filename, 'r')
    n_out = len(OUT_FILES)
    for ii in range(n_out):
        f_out = h5py.File(OUT_FILES[ii], 'w')
        copyattrs(f_in.attrs, f_out.attrs)
        for key, entry in f_in.items():
            if is_group(f_in[key]):
                # Copy all groups in entirety.
                f_out.create_group(key)
                deep_group_copy(entry, f_out[key])
            else:
                # Copy only the specified range of frames for the datasets.
                f_out.create_dataset(key, dtype=entry.dtype,
                        shape=(END_FRAMES[ii] - START_FRAMES[ii],))
                f_out[key][:] = entry[START_FRAMES[ii]:END_FRAMES[ii]]
                copyattrs(entry.attrs, f_out[key].attrs)
        f_out.close()

if len(sys.argv) != 2:
    print("Usage: python scripts/generate_acq_test_data.py acq_filename.h5")
else:
    filename = sys.argv[1]
    generate(filename)
