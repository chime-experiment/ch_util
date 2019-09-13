"""Unit tests for analysis data format."""
# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility


import unittest
import os

import numpy as np
import h5py

from ch_util import andata

# from ch_util.andata import MemGroup
from caput.memh5 import MemGroup
from ch_util.tests import data_paths

# All the test data file names.
acq_fname_list = data_paths.paths1_0
acq_fname = acq_fname_list[0]
acq_fname_root = data_paths.dir1_0

# Old testdata is known to create warnings due to missing gain information.
# Unfortunately, this kill warnings for all modules, not just this one.
import warnings

# warnings.filterwarnings("ignore")
def ignore_warnings(test_func):
    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            test_func(self, *args, **kwargs)

    return do_test


# Inspect test data to figure out some some of the expectations.
with h5py.File(acq_fname, "r") as f:
    NTIME1 = len(f["vis"])
    if "amtel_adc_therm" in f["cal"]:
        ATEL = "amtel_adc_therm"
    elif "atmel_adc_therm" in f["cal"]:
        ATEL = "atmel_adc_therm"
    else:
        raise RuntimeError("Funky test data.")

NTIME = 0
for fname in acq_fname_list:
    with h5py.File(fname, "r") as f:
        NTIME += len(f["vis"])


class TestReader(unittest.TestCase):
    """Tests for high level data reader."""

    def setUp(self):
        self.reader = andata.Reader(acq_fname_list)

    @ignore_warnings
    def test_select_read(self):
        reader = self.reader
        reader.time_sel = (2, 8)
        reader.prod_sel = 9
        sl = slice(100, 900, 40)
        reader.freq_sel = sl
        reader.dataset_sel = ("vis", "timestamp")
        data = reader.read()
        self.assertEqual(data.vis.shape, (20, 1, 6))
        all_data = andata.AnData.from_acq_h5(acq_fname_list)
        self.assertTrue(np.allclose(data.vis, all_data.vis[sl, [9], 2:8]))
        self.assertEqual(
            set(data.datasets.keys()),
            {
                "vis",
                "timestamp_cpu_s",
                "timestamp_cpu_us",
                "timestamp_fpga_count",
                "gain",
            },
        )

    @ignore_warnings
    def test_select_time_range(self):
        dt = 10.0  # seconds
        time = self.reader.time
        start_time = time[0] + 1.6 * dt
        stop_time = time[0] + 6.9 * dt
        self.reader.select_time_range(start_time, stop_time)
        data = self.reader.read()
        self.assertEqual(data.ntime, 5)
        self.assertTrue(np.all(data.timestamp >= start_time))
        self.assertTrue(np.all(data.timestamp < stop_time))
        # Test near edge behaviour.
        start_time = time[0] + 0.9999999 * dt
        stop_time = time[0] + 5.9999999 * dt
        self.reader.select_time_range(start_time, stop_time)
        data = self.reader.read()
        self.assertEqual(data.ntime, 5)
        self.assertTrue(np.all(data.timestamp >= start_time))
        self.assertTrue(np.all(data.timestamp < stop_time))

    @ignore_warnings
    def test_select_freq_range(self):
        freq = self.reader.freq["centre"]
        low_freq = 452.0
        high_freq = 727.0
        self.reader.select_freq_range(low_freq, high_freq)
        data = self.reader.read()
        self.assertTrue(np.all(data.index_map["freq"]["centre"] >= low_freq))
        self.assertTrue(np.all(data.index_map["freq"]["centre"] < high_freq))
        step_freq = 50.0
        self.reader.select_freq_range(low_freq, high_freq, step_freq)
        data = self.reader.read()
        expected_centres = np.arange(low_freq, high_freq, step_freq)[::-1]
        diff_centres = data.index_map["freq"]["centre"] - expected_centres
        self.assertTrue(np.all(abs(diff_centres) <= 400.0 / 1024))

    @ignore_warnings
    def test_select_frequencies(self):
        freqs = [784.0, 465.0, 431.0]
        self.reader.select_freq_physical(freqs)
        data = self.reader.read()
        diff_centres = data.index_map["freq"]["centre"] - np.array(freqs)
        self.assertTrue(np.all(abs(diff_centres) <= 400.0 / 1024 / 2))
        self.assertRaises(ValueError, self.reader.select_freq_physical, [324.0])


class TestAttrsDB(unittest.TestCase):
    """Tests for interacting with layout database."""

    @ignore_warnings
    def setUp(self):
        self.data = andata.AnData.from_acq_h5(acq_fname)

    # def test_trivial(self):
    #    andata.get_attrs_db(self.data, 'a', 'b')
    #    self.assertTrue(self.data.index_map.has_key('prod_map'))
    #    self.assertEqual(self.data.index_map['prod_map'].shape, (8, 8))


class TestLoadACQ(unittest.TestCase):
    """Tests for loading early acquisition data."""

    @ignore_warnings
    def test_load_fname(self):
        data = andata.AnData.from_acq_h5(acq_fname)
        check_result(self, data, NTIME1)

    @ignore_warnings
    def test_load_file_obj(self):
        F = h5py.File(acq_fname, "r")
        data = andata.AnData.from_acq_h5(F)
        F.close()
        check_result(self, data, NTIME1)

    @ignore_warnings
    def test_subset_vis(self):
        data = andata.AnData.from_acq_h5(acq_fname, prod_sel=[1, 3, 9], freq_sel=6)
        self.assertEqual(data.nprod, 3)
        self.assertEqual(data.nfreq, 1)
        self.assertEqual(data.datasets["vis_flag_rfi"].shape, data.vis.shape)
        self.assertAlmostEqual(data.index_map["freq"]["centre"], 797.65625)
        self.assertAlmostEqual(data.index_map["freq"]["width"], 0.390625)
        all_data = andata.AnData.from_acq_h5(acq_fname)
        self.assertTrue(np.allclose(data.vis, all_data.vis[6, [1, 3, 9], :]))
        prod = data.index_map["prod"]
        self.assertEqual(list(prod[0]), [0, 1])
        self.assertEqual(list(prod[2]), [1, 2])

    @ignore_warnings
    def test_subset_datasets(self):
        data = andata.AnData.from_acq_h5(acq_fname, datasets=("fpga_hk",))
        self.assertEqual(list(data.datasets.keys()), ["fpga_hk"])


class TestAnData(unittest.TestCase):
    """Tests for AnData class."""

    def setUp(self):
        self.data = andata.CorrData()

    def test_properties(self):
        # Check that the "properties" are read only.
        self.assertRaises(AttributeError, self.data.__setattr__, "datasets", {})
        self.assertRaises(AttributeError, self.data.__setattr__, "cal", {})
        self.assertRaises(AttributeError, self.data.__setattr__, "attrs", {})

    def test_vis_shortcuts(self):

        # More sophisticated base calculation
        def getbase(a):
            b = a.base
            if b is None:
                return a
            else:
                return getbase(b)

        vis = np.arange(60)
        vis.shape = (3, 2, 10)
        self.data.create_dataset("vis", data=vis)
        self.data["vis"].attrs["cal"] = "stuff"
        self.assertTrue(getbase(self.data.vis[:]) is vis)
        self.assertEqual(self.data.vis.attrs, {"cal": "stuff"})
        self.assertTrue(np.allclose(self.data.vis[0:2:, 0, 1:3:9], vis[0:2:, 0, 1:3:9]))


class TestLoadSave(unittest.TestCase):
    """Tests for loading and saving to/from file."""

    test_fname = "tmp_test_AnData.hdf5"

    @ignore_warnings
    def setUp(self, *args, **kwargs):
        """Makes sure there is test file to work with."""

        data = andata.AnData.from_acq_h5(acq_fname)
        data.save(self.test_fname, mode="w")

    def test_load_fname(self):
        data = andata.AnData.from_file(self.test_fname)
        check_result(self, data, NTIME1)

    def test_load_FO(self):
        F = h5py.File(self.test_fname, mode="r")
        data = andata.AnData.from_file(F)
        check_result(self, data, NTIME1)
        # Make sure I can write to data, since it should be in memory.
        data.vis[0, 0, 0] = 10
        data.datasets["timestamp_cpu_s"][10] = 12
        self.assertEqual(data.datasets["timestamp_cpu_s"][10], 12)
        # Make sure the file is still open.
        self.assertFalse(F["timestamp_cpu_s"][10] == 12)
        F.close()

    def test_on_file_fname_to_memory(self):
        data = andata.AnData.from_file(self.test_fname, ondisk=True)
        check_result(self, data, NTIME1)
        # Now convert to being in memory.
        data = data.to_memory()
        self.assertFalse(data.ondisk)
        check_result(self, data, NTIME1)

    def test_on_file_FO(self):
        F = h5py.File(self.test_fname, mode="r")
        data = andata.AnData.from_file(F, ondisk=True)
        check_result(self, data, NTIME1)
        F.close()
        self.assertRaises(Exception, data.__getattribute__, "vis")

    def tearDown(self):
        """Remove test data."""

        if os.path.isfile(self.test_fname):
            os.remove(self.test_fname)


class TestMultiLoadACQ(unittest.TestCase):
    """Tests for loading multible acq files."""

    @ignore_warnings
    def test_load_list(self):
        data = andata.AnData.from_acq_h5(acq_fname_list)
        check_result(self, data, NTIME)

    @ignore_warnings
    def test_load_glob(self):
        data = andata.AnData.from_acq_h5(acq_fname_root + "/*")
        check_result(self, data, NTIME)

    @ignore_warnings
    def test_load_subsets(self):
        data = andata.AnData.from_acq_h5(acq_fname_list, start=6, stop=-5)
        check_result(self, data, NTIME - 6 - 5)
        data = andata.AnData.from_acq_h5(acq_fname_list, start=20, stop=-1)
        check_result(self, data, NTIME - 20 - 1)
        data = andata.AnData.from_acq_h5(acq_fname_list, start=2, stop=8)
        check_result(self, data, 6)


class TestProvideGroup(unittest.TestCase):
    """Tests for converting from acq format to analysis format on disk."""

    @ignore_warnings
    def test_stores(self):
        group = MemGroup()
        data = andata.AnData.from_acq_h5(acq_fname_list, out_group=group)
        self.assertTrue(data._data is group)
        self.assertTrue("vis" in group)
        self.assertEqual(group["vis"].name, data.vis.name)


class TestRaisesError(unittest.TestCase):
    """Basic acq format loading error checking."""

    def setUp(self):
        self.acq_list = [MemGroup.from_hdf5(f) for f in acq_fname_list]

    @ignore_warnings
    def test_extra_dataset(self):
        nt = self.acq_list[1]["vis"].shape[0]
        self.acq_list[1].create_dataset("stuff", shape=(nt,), dtype=float)
        self.assertRaises(ValueError, andata.AnData.from_acq_h5, self.acq_list)

    @ignore_warnings
    def test_missing_dataset(self):
        nt = self.acq_list[0]["vis"].shape[0]
        self.acq_list[0].create_dataset("stuff", shape=(nt,), dtype=float)
        self.assertRaises(ValueError, andata.AnData.from_acq_h5, self.acq_list)


class TestDataPropertiesACQ(unittest.TestCase):
    """Test getting timestamps from acq files."""

    @ignore_warnings
    def setUp(self):
        self.data = andata.AnData.from_acq_h5(acq_fname)

    def test_timestamp(self):
        """Just makes sure timestamps are calculated for acq data without any
        validation."""
        timestamp = self.data.timestamp
        self.assertEqual(len(timestamp), NTIME1)


class TestConcatenate(unittest.TestCase):
    @ignore_warnings
    def setUp(self):
        data_list = []
        data_list = [andata.AnData.from_acq_h5(fname) for fname in acq_fname_list]
        self.data_list = data_list

    @ignore_warnings
    def test_works(self):
        self.right_answer = andata.AnData.from_acq_h5(acq_fname_list)
        merged_data = andata.concatenate(self.data_list)
        self.assertTrue(np.allclose(merged_data.vis, self.right_answer.vis))
        self.assertEqual(
            len(merged_data.index_map["time"]), len(self.right_answer.index_map["time"])
        )

    @ignore_warnings
    def test_start_end_inds(self):
        self.right_answer = andata.AnData.from_acq_h5(acq_fname_list, start=3, stop=26)
        merged_data = andata.concatenate(self.data_list, start=3, stop=26)
        self.assertTrue(np.allclose(merged_data.vis, self.right_answer.vis))
        self.assertEqual(
            len(merged_data.index_map["time"]), len(self.right_answer.index_map["time"])
        )


def check_result(self, data, ntime):
    """Checks that the data more or less looks like it should if converted from
    the test data."""

    # Make sure the data is there and that it has the expected shape.
    self.assertEqual(data.vis.shape, (1024, 36, ntime))
    self.assertEqual(data.vis.dtype, np.complex64)
    # Check that 'serial_adc was properly split into 8 datasets.
    count_serial_adc = 0
    for dset_name in data.datasets.keys():
        if dset_name[:10] == "serial_adc":
            count_serial_adc += 1
            self.assertTrue("cal" in data.datasets[dset_name].attrs)
            self.assertEqual(data.datasets[dset_name].attrs["cal"], ATEL)
    self.assertEqual(count_serial_adc, 8)
    # Make sure the cal data is acctually there.
    self.assertTrue(ATEL in data.cal)
    # Check the fpga_hk dataset, as it is the only one that didn't need to
    # be split.
    self.assertEqual(data.datasets["fpga_hk"].shape, (1, ntime))
    self.assertEqual(data.datasets["fpga_hk"].dtype, np.float32)
    # Check a few of the attributes for known values.
    self.assertTrue("n_antenna" in data.history["acq"])
    self.assertEqual(data.history["acq"]["n_antenna"], 8)
    # Check a few of the cal entries.
    self.assertTrue("b" in data.cal[ATEL])
    self.assertEqual(data.cal[ATEL]["b"], "3900")
    # This is the only actual check of the content of the datasets.  Make sure
    # the timestamp is increasing in exactly 10s increments.
    self.assertTrue(np.all(np.diff(data.datasets["timestamp_cpu_s"]) == 10))
    # Check that the data attributes are correctly calculated.
    freq_width = 400.0 / 1024
    self.assertTrue(np.allclose(data.index_map["freq"]["width"], freq_width))
    freq_centre = np.linspace(800.0, 400.0, 1024, endpoint=False)
    self.assertTrue(np.allclose(data.index_map["freq"]["centre"], freq_centre))


if __name__ == "__main__":
    unittest.main()
