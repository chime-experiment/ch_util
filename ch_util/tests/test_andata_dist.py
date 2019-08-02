"""Tests of the distributed features of AnData.

"""
# === Start Python 2/3 compatibility
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614
# === End Python 2/3 compatibility


import unittest

import numpy as np

import h5py
from mpi4py import MPI
from ch_util import andata
from ch_util.tests import data_paths

comm = MPI.COMM_WORLD

#fnames = glob.glob('/scratch/k/krs/jrs65/chime_archive/20140916T173334Z_blanchard_corr/000[0-1]*_0000.h5')

# All the test data file names.
# Test data files have 32 frequencies, 136 products, and 31, 17 times.
fnames = data_paths.paths2_0

# Get shape of visibility dataset, and broadcast to all ranks
nfreq = 0
nprod = 0
ninput = 0
ntime = 0

fmap = None
tmap = []

if comm.rank == 0:

    shape_fp = ()
    shape_t = 0
    for fname in fnames:

        with h5py.File(fname, 'r') as f:
            sh = f['vis'].shape
            nfreq = sh[0]
            nprod = sh[1]
            ntime += sh[2]
            n_input = f.attrs['n_antenna']

            fmap = f['index_map/freq'][:]
            tmap.append(f['index_map/time'][:])

    tmap = np.concatenate(tmap)

# Broadcast over relevant properties
nfreq = comm.bcast(nfreq, root=0)
ntime = comm.bcast(ntime, root=0)
nprod = comm.bcast(nprod, root=0)
ninput = comm.bcast(ninput, root=0)

fmap = comm.bcast(fmap, root=0)
tmap = comm.bcast(tmap, root=0)

    
class TestLoadDist(unittest.TestCase):
    """Tests for loading andata files in distributed mode."""

    def test_load_allfreq(self):

        ad = andata.CorrData.from_acq_h5(fnames, distributed=True, comm=comm)

        f_shape = (nfreq // comm.size,) + (nprod, ntime)

        # Test that shapes seem correct
        self.assertEqual(ad.vis.shape, (nfreq, nprod, ntime))
        self.assertEqual(ad.vis.local_shape, f_shape)

        f_offset = (nfreq // comm.size * comm.rank,) + (0, 0)

        # Test that offset is correct
        self.assertEqual(ad.vis.local_offset, f_offset)


        # Redistribute across times and redo the same tests
        ad.redistribute('time')

        t_shape = (nfreq, nprod) + (ntime // comm.size,)

        # Test that shapes seem correct
        self.assertEqual(ad.vis.shape, (nfreq, nprod, ntime))
        self.assertEqual(ad.vis.local_shape, t_shape)

        t_offset = (0, 0) + (ntime // comm.size * comm.rank,)

        # Test that offset is correct
        self.assertEqual(ad.vis.local_offset, t_offset)

        # Test that index maps are correct
        self.assertTrue((ad.index_map['freq'] == fmap).all())
        self.assertTrue((ad.index_map['time']['ctime'] == tmap['ctime']).all())


    def test_load_freq_sel(self):

        fsel = [4, 18, 23, 27]

        ad = andata.CorrData.from_acq_h5(fnames, freq_sel=fsel, distributed=True, comm=comm)
        ad_all = andata.CorrData.from_acq_h5(fnames, freq_sel=fsel)

        f_shape = (len(fsel) // comm.size,) + (nprod, ntime)

        # Test that shapes seem correct
        self.assertEqual(ad.vis.shape, (len(fsel), nprod, ntime))
        self.assertEqual(ad.vis.local_shape, f_shape)

        # Test that contents are correct
        self.assertTrue((ad.vis[comm.rank] == ad_all.vis[comm.rank]).all())
        self.assertTrue((ad.gain[comm.rank] == ad_all.gain[comm.rank]).all())

        f_offset = (len(fsel) // comm.size * comm.rank,) + (0, 0)

        # Test that offset is correct
        self.assertEqual(ad.vis.local_offset, f_offset)


        # Redistribute across times and redo the same tests
        ad.redistribute('time')

        t_shape = (len(fsel), nprod) + (ntime // comm.size,)

        # Test that shapes seem correct
        self.assertEqual(ad.vis.shape, (len(fsel), nprod, ntime))
        self.assertEqual(ad.vis.local_shape, t_shape)

        t_offset = (0, 0) + (ntime // comm.size * comm.rank,)

        # Test that offset is correct
        self.assertEqual(ad.vis.local_offset, t_offset)

        # Test that contents are correct
        self.assertTrue((ad.vis[:] == ad_all.vis[..., (ntime // comm.size * comm.rank):(ntime // comm.size * (comm.rank + 1))]).all())
        self.assertTrue((ad.gain[:] == ad_all.gain[..., (ntime // comm.size * comm.rank):(ntime // comm.size * (comm.rank + 1))]).all())
        # Test that index maps are correct
        self.assertTrue((ad.index_map['freq'] == fmap[fsel]).all())
        self.assertTrue((ad.index_map['time']['ctime'] == tmap['ctime']).all())


if __name__ == '__main__':
    unittest.main()
