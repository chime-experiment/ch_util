"""Unit tests for analysis data format."""

import unittest
import tempfile
import tarfile
import shutil
import glob

import numpy as np
import h5py

from ch_util import andata
from caput.memh5 import MemGroup
import data_paths

tempdir = tempfile.mkdtemp()
tarfile.open(data_paths.archive3_1).extractall(tempdir)
archive_acq_fname_list_3_1 = sorted(glob.glob(tempdir + "/*.h5"))
shutil.rmtree(tempdir)


class TestStack(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        tarfile.open(data_paths.archive3_1).extractall(self.tempdir)
        self.file_list = sorted(glob.glob(self.tempdir + "/*.h5"))

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_load_data(self):
        """Test that loaded data has the right shape and datatypes."""
        ad = andata.CorrData.from_acq_h5(self.file_list)

        # Check shapes
        self.assertEqual(ad.vis.shape, (2, 16356, 12))
        self.assertEqual(ad.weight.shape, (2, 16356, 12))
        self.assertEqual(ad.gain.shape, (2, 2048, 12))

        # Check datatypes
        self.assertEqual(ad.vis.dtype, np.complex64)
        self.assertEqual(ad.weight.dtype, np.float32)
        self.assertEqual(ad.gain.dtype, np.complex64)

    def test_stack_sel(self):
        """Test that loaded data has the right shape and datatypes."""
        ad = andata.CorrData.from_acq_h5(self.file_list, stack_sel=[0, 15])

        # Check the stack map properties
        self.assertTrue((ad.index_map["stack"]["prod"] == np.array([0, 1])).all())
        self.assertFalse(ad.index_map["stack"]["conjugate"].any())

        # Check the reverse_map
        rmap = ad.reverse_map["stack"]["stack"]
        self.assertTrue(((rmap == 0) | (rmap == 1)).all())
        self.assertEqual((rmap == 0).sum(), 256)
        self.assertEqual((rmap == 1).sum(), 241)

        # Check the selection of products and inputs
        self.assertEqual(ad.index_map["prod"].size, 497)  # Sum of above numbers
        self.assertEqual(
            ad.index_map["input"].size, 256
        )  # All single pol feeds in Cyl A

        # Check shapes
        self.assertEqual(ad.vis.shape, (2, 2, 12))
        self.assertEqual(ad.weight.shape, (2, 2, 12))
        self.assertEqual(ad.gain.shape, (2, 256, 12))

    def test_no_prod_input_sel(self):
        """Test that you can't use input/prod sel on stacked data."""
        with self.assertRaises(ValueError):
            ad = andata.CorrData.from_acq_h5(self.file_list, input_sel=[0, 15])

        with self.assertRaises(ValueError):
            ad = andata.CorrData.from_acq_h5(self.file_list, prod_sel=[0, 15])


if __name__ == "__main__":
    unittest.main()
