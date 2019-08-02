"""Unit tests for analysis data format."""
# === Start Python 2/3 compatibility
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614
# === End Python 2/3 compatibility


import unittest

import numpy as np
import h5py

from ch_util import andata
from caput.memh5 import MemGroup
from ch_util.tests import data_paths

#archive_acq_dir = ("/scratch/k/krs/jrs65/chime_archive/20140913T055455Z_blanchard_corr/")
#archive_acq_fname_list = []
#fmt_corr = re.compile("([0-9]{8})_([0-9]{4}).h5")
#for f in os.listdir(archive_acq_dir):
#  if re.match(fmt_corr, f):
#    archive_acq_fname_list.append(f)
#  if len(archive_acq_fname_list) >= 3:
#    break
#if len(archive_acq_fname_list) < 3:
#  print "Acquisition %s does not have enough data files for this test " \
#        "(3 needed)." % (archive_acq_dir)
#  exit()
#archive_acq_fname_list = [ archive_acq_dir + "/" + f \
#                           for f in archive_acq_fname_list ]

# All the test data file names.
# Two test data files that have 32 frequencies, 136 products, and 31, 17 times.
archive_acq_fname_list_2_0 = data_paths.paths2_0

# Test data for 2.2 data.
# Two test data files that have 2 frequecies, 32896 products, and 3, 5 times.
archive_acq_fname_list_2_2 = data_paths.paths2_2


class TestLoadACQ(unittest.TestCase):
    """Tests for loading early acquisition data."""

    def test_load_datasets_flags(self):
        hf = h5py.File(archive_acq_fname_list_2_2[0], 'r')
        dset_names = {n for n in hf.keys() if isinstance(hf[n], h5py.Dataset)}
        dset_names = dset_names | {'gain'}
        hff = hf['flags']
        flag_names = {n for n in hff.keys() if isinstance(hff[n], h5py.Dataset)}
        d = andata.AnData.from_acq_h5(archive_acq_fname_list_2_2)
        self.assertEqual(dset_names, set(d.datasets.keys()))
        self.assertEqual(flag_names, set(d.flags.keys()))

    def test_load_fname(self):
        data = andata.AnData.from_acq_h5(archive_acq_fname_list_2_0[0], start=3,
                stop=10, prod_sel=[1, 3, 7], freq_sel=slice(8, 18))
        self.assertTrue(isinstance(data, andata.CorrData))
        self.assertEqual(data.vis.dtype, np.complex64)
        self.assertEqual(data.vis.shape, (10, 3, 7))

    def test_list_selections(self):
        f_sel = np.zeros(32, dtype=bool)
        f_sel[[3, 6, 7, 15, 16, 19, 31]] = True
        data = andata.AnData.from_acq_h5(archive_acq_fname_list_2_0[0], start=3,
                stop=20, prod_sel=[1, 3, 7], freq_sel=f_sel)
        self.assertEqual(data.vis.shape, (7, 3, 17))

    def test_multi_file(self):
        fsel = slice(9, 15)
        psel = [0, 16, 31]
        data = andata.AnData.from_acq_h5(archive_acq_fname_list_2_0, start=3,
                   stop=43, prod_sel=psel, freq_sel=fsel)
        self.assertEqual(data.vis.shape, (6, 3, 40))
        # Selected only autocorrelations.
        self.assertTrue(np.all(data.vis[:].imag == 0))
        self.assertEqual(data.nprod, 3)
        self.assertEqual(data.nfreq, 6)

    def test_no_datasets(self):
        datasets = ()
        data = andata.AnData.from_acq_h5(archive_acq_fname_list_2_0, start=3,
                   stop=43, datasets=datasets)
        self.assertEqual(data.ntime, 40)
        self.assertEqual(len(list(data.datasets.keys())), 0)


class TestRemapInputs(unittest.TestCase):
    """From archive version 2.0 inputs need to be relabled."""

    def setUp(self):
        # Load one of the test files directly (with no processing) and then run
        # it through the channel remapping.
        archive_file = andata.CorrData.from_file(archive_acq_fname_list_2_0[0])
        self.prod = archive_file.index_map['prod'][:]
        self.archive_input = archive_file.index_map['input'][:].copy()
        remapped_file = andata._remap_blanchard(archive_file)
        self.remapped_input = remapped_file.index_map['input'][:].copy()

    def test_load_all(self):
        data = andata.CorrData.from_acq_h5(archive_acq_fname_list_2_0)
        self.assertTrue(np.all(data.index_map['prod'][:] == self.prod))
        self.assertTrue(np.all(data.index_map['input'][:] == self.remapped_input))

    def test_load_prod_sel(self):
        data = andata.CorrData.from_acq_h5(archive_acq_fname_list_2_0,
                                           prod_sel=[0,1])
        self.assertTrue(np.all(data.index_map['prod'] == self.prod[[0,1]]))
        self.assertTrue(np.all(data.index_map['input']
                               == self.remapped_input[[0,1]]))

    def test_load_prod_sel2(self):
        data = andata.CorrData.from_acq_h5(archive_acq_fname_list_2_0,
                                           prod_sel=[4])
        self.assertTrue(np.all(data.index_map['prod'] == self.prod[1]))
        self.assertTrue(np.all(data.index_map['input']
                               == self.remapped_input[[0,4]]))

    def test_load_input_sel(self):
        data = andata.CorrData.from_acq_h5(archive_acq_fname_list_2_0,
                                           input_sel=0)
        self.assertTrue(np.all(data.index_map['prod'] == self.prod[[0]]))
        self.assertTrue(np.all(data.index_map['input']
                               == self.remapped_input[[0]]))

    def test_load_input_sel2(self):
        data = andata.CorrData.from_acq_h5(archive_acq_fname_list_2_0,
                                           input_sel=[4,5])
        self.assertTrue(np.all(data.index_map['prod'] == self.prod[[0,1,16]]))
        self.assertTrue(np.all(data.index_map['input']
                               == self.remapped_input[[4,5]]))

    def test_load_input_sel3(self):
        data = andata.CorrData.from_acq_h5(archive_acq_fname_list_2_0,
                                           input_sel=np.s_[4:10])
        self.assertEqual(len(data.index_map['prod']), 6 * 7 // 2)
        self.assertTrue(np.all(data.index_map['input']
                               == self.remapped_input[np.s_[4:10]]))

    def test_reader_input_sel(self):
        r = andata.CorrReader(archive_acq_fname_list_2_0)
        i_sel = np.s_[5:9]
        inputs_ = r.input[i_sel]
        r.input_sel = i_sel
        data = r.read()
        self.assertTrue(np.all(data.index_map['input'] == inputs_))
        self.assertEqual(data.nprod, 4 * 5 // 2)



class TestApplyGains2_0(unittest.TestCase):

    def setUp(self):
        self.all_data = andata.CorrData.from_acq_h5(archive_acq_fname_list_2_0)

    def test_input_sel(self):
        data = andata.CorrData.from_acq_h5(archive_acq_fname_list_2_0,
                input_sel=np.s_[4:8], datasets=('vis',))
        self.assertEqual(data.datasets['gain'].shape[1], 4)
        # Make sure the gains were properly calculated.
        self.assertTrue(np.all(data.datasets['gain'][:]
                           == self.all_data.datasets['gain'][:,np.s_[4:8]]))
        # Make sure they where properly applied.
        prod_sel, prod_map, input_sel, input_map = \
                _resolve_prod_input_sel(
                    None, self.all_data.index_map['prod'],
                    np.s_[4:8], self.all_data.index_map['input'])
        self.assertTrue(np.all(data.vis[:] == self.all_data.vis[:,prod_sel,:]))

    def test_prod_sel(self):
        data = andata.CorrData.from_acq_h5(archive_acq_fname_list_2_0,
                prod_sel=np.s_[5:14], datasets=('vis',))
        self.assertEqual(data.datasets['gain'].shape[1], 10)
        prod_sel, prod_map, input_sel, input_map = \
                _resolve_prod_input_sel(
                        np.s_[5:14], self.all_data.index_map['prod'],
                        None, self.all_data.index_map['input'])
        # Make sure the gains were properly calculated.
        self.assertTrue(np.all(data.datasets['gain'][:]
                           == self.all_data.datasets['gain'][:,input_sel]))
        # Make sure they where properly applied.
        self.assertTrue(np.all(data.vis[:] == self.all_data.vis[:,prod_sel,:]))






class TestApplyGains2_2(unittest.TestCase):
    
    def setUp(self):
        self.all_data = andata.CorrData.from_acq_h5(archive_acq_fname_list_2_2)

    def test_input_sel(self):
        data = andata.CorrData.from_acq_h5(archive_acq_fname_list_2_2,
                input_sel=np.s_[4:8], datasets=('vis',))
        self.assertEqual(data.datasets['gain'].shape[1], 4)
        # Make sure the gains were properly calculated.
        self.assertTrue(np.all(data.datasets['gain'][:]
                           == self.all_data.datasets['gain'][:,np.s_[4:8]]))
        # Make sure they where properly applied.
        prod_sel, prod_map, input_sel, input_map = \
                _resolve_prod_input_sel(
                    None, self.all_data.index_map['prod'],
                    np.s_[4:8], self.all_data.index_map['input'])
        self.assertTrue(np.all(data.vis[:] == self.all_data.vis[:,prod_sel,:]))

    def test_prod_sel(self):
        data = andata.CorrData.from_acq_h5(archive_acq_fname_list_2_2,
                prod_sel=np.s_[5:14])
        self.assertEqual(data.datasets['gain'].shape[1], 10)
        prod_sel, prod_map, input_sel, input_map = \
                _resolve_prod_input_sel(
                        np.s_[5:14], self.all_data.index_map['prod'],
                        None, self.all_data.index_map['input'])
        # Make sure the gains were properly calculated.
        self.assertTrue(np.all(data.datasets['gain'][:]
                           == self.all_data.datasets['gain'][:,input_sel]))
        # Make sure they where properly applied.
        self.assertTrue(np.all(data.vis[:] == self.all_data.vis[:,prod_sel,:]))

    def test_prod_sel_int_auto(self):
        data = andata.CorrData.from_acq_h5(archive_acq_fname_list_2_2,
                prod_sel=0)
        self.assertEqual(data.datasets['gain'].shape[1], 1)
        prod_sel, prod_map, input_sel, input_map = \
                _resolve_prod_input_sel(
                        0, self.all_data.index_map['prod'],
                        None, self.all_data.index_map['input'])
        # Make sure the gains were properly calculated.
        self.assertTrue(np.all(data.datasets['gain'][:]
                           == self.all_data.datasets['gain'][:,input_sel]))
        # Make sure they where properly applied.
        self.assertTrue(np.all(data.vis[:] == self.all_data.vis[:,prod_sel,:]))


    def test_prod_sel_fancy(self):
        data = andata.CorrData.from_acq_h5(archive_acq_fname_list_2_2,
                prod_sel=np.arange(8000), datasets=('vis',), freq_sel=[0,1])
        self.assertEqual(data.datasets['gain'].shape[1], 256)
        prod_sel, prod_map, input_sel, input_map = \
                _resolve_prod_input_sel(
                        np.arange(8000), self.all_data.index_map['prod'],
                        None, self.all_data.index_map['input'])
        # Make sure the gains were properly calculated.
        self.assertTrue(np.all(data.datasets['gain'][:]
                           == self.all_data.datasets['gain'][:,input_sel]))
        # Make sure they where properly applied.
        self.assertTrue(np.all(data.vis[:] == self.all_data.vis[:,prod_sel,:]))


def _resolve_prod_input_sel(prod_sel, prod_map, input_sel, input_map):
    """Legacy code pasted here for regression testing."""
    if (not prod_sel is None) and (not input_sel is None):
        # This should never happen due to previouse checks.
        raise ValueError("*input_sel* and *prod_sel* both specified.")

    if prod_sel is None and input_sel is None:
        prod_sel = andata._ensure_1D_selection(prod_sel)
        input_sel = andata._ensure_1D_selection(input_sel)
    else:
        if input_sel is None:
            prod_sel = andata._ensure_1D_selection(prod_sel)
            # Choose inputs involved in selected products.
            prod_map = prod_map[prod_sel]
            input_sel = []
            for p0, p1 in prod_map:
                input_sel.append(p0)
                input_sel.append(p1)
            # ensure_1D here deals with h5py issue #425.
            input_sel = andata._ensure_1D_selection(sorted(list(set(input_sel))))
        else:
            input_sel = andata._ensure_1D_selection(input_sel)
            inputs = list(np.arange(len(input_map), dtype=int)[input_sel])
            prod_sel = []
            for ii, p in enumerate(prod_map):
                if p[0] in inputs and p[1] in inputs:
                    prod_sel.append(ii)
            # ensure_1D here deals with h5py issue #425.
            prod_sel = andata._ensure_1D_selection(prod_sel)
            prod_map = prod_map[prod_sel]
        # Now we need to rejig the index maps for the subsets of the inputs.
        inputs = list(np.arange(len(input_map), dtype=int)[input_sel])
        input_map = input_map[input_sel]
        for ii, p in enumerate(prod_map):
            p0 = inputs.index(p[0])
            p1 = inputs.index(p[1])
            prod_map[ii] = (p0, p1)
    return prod_sel, prod_map, input_sel, input_map


if __name__ == '__main__':
    unittest.main()

