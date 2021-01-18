"""Unit tests for analysis data format."""


import pytest
import os

import numpy as np
import h5py

from ch_util import andata

from caput.memh5 import MemGroup
import data_paths

# All the test data file names.
acq_fname_list = data_paths.paths1_0
acq_fname = acq_fname_list[0]
acq_fname_root = data_paths.dir1_0

# Old testdata is known to create warnings due to missing gain information.
# Unfortunately, this kill warnings for all modules, not just this one.
import warnings


from functools import wraps


# warnings.filterwarnings("ignore")
def ignore_warnings(test_func):
    @wraps(test_func)
    def do_test(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return test_func(*args, **kwargs)

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

# Tests for high level data reader.
@pytest.fixture
def reader():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        yield andata.Reader(acq_fname_list)


@ignore_warnings
def test_select_read(reader):
    reader.time_sel = (2, 8)
    reader.prod_sel = 9
    sl = slice(100, 900, 40)
    reader.freq_sel = sl
    reader.dataset_sel = ("vis", "timestamp")
    data = reader.read()
    assert data.vis.shape == (20, 1, 6)
    all_data = andata.AnData.from_acq_h5(acq_fname_list)
    assert np.allclose(data.vis, all_data.vis[sl, [9], 2:8])
    assert set(data.datasets.keys()) == {
        "vis",
        "timestamp_cpu_s",
        "timestamp_cpu_us",
        "timestamp_fpga_count",
        "gain",
    }


@ignore_warnings
def test_select_time_range(reader):
    dt = 10.0  # seconds
    time = reader.time
    start_time = time[0] + 1.6 * dt
    stop_time = time[0] + 6.9 * dt
    reader.select_time_range(start_time, stop_time)
    data = reader.read()
    assert data.ntime == 5
    assert np.all(data.timestamp >= start_time)
    assert np.all(data.timestamp < stop_time)
    # Test near edge behaviour.
    start_time = time[0] + 0.9999999 * dt
    stop_time = time[0] + 5.9999999 * dt
    reader.select_time_range(start_time, stop_time)
    data = reader.read()
    assert data.ntime == 5
    assert np.all(data.timestamp >= start_time)
    assert np.all(data.timestamp < stop_time)


@ignore_warnings
def test_select_freq_range(reader):
    low_freq = 452.0
    high_freq = 727.0
    reader.select_freq_range(low_freq, high_freq)
    data = reader.read()
    assert np.all(data.index_map["freq"]["centre"] >= low_freq)
    assert np.all(data.index_map["freq"]["centre"] < high_freq)
    step_freq = 50.0
    reader.select_freq_range(low_freq, high_freq, step_freq)
    data = reader.read()
    expected_centres = np.arange(low_freq, high_freq, step_freq)[::-1]
    diff_centres = data.index_map["freq"]["centre"] - expected_centres
    assert np.all(abs(diff_centres) <= 400.0 / 1024)


@ignore_warnings
def test_select_frequencies(reader):
    freqs = [784.0, 465.0, 431.0]
    reader.select_freq_physical(freqs)
    data = reader.read()
    diff_centres = data.index_map["freq"]["centre"] - np.array(freqs)
    assert np.all(abs(diff_centres) <= 400.0 / 1024 / 2)
    with pytest.raises(ValueError):
        reader.select_freq_physical([324.0])


# Tests for loading early acquisition data.
@ignore_warnings
def test_load_fname():
    data = andata.AnData.from_acq_h5(acq_fname)
    check_result(data, NTIME1)


@ignore_warnings
def test_load_file_obj():
    F = h5py.File(acq_fname, "r")
    data = andata.AnData.from_acq_h5(F)
    F.close()
    check_result(data, NTIME1)


@ignore_warnings
def test_subset_vis():
    data = andata.AnData.from_acq_h5(acq_fname, prod_sel=[1, 3, 9], freq_sel=6)
    assert data.nprod == 3
    assert data.nfreq == 1
    assert data.datasets["vis_flag_rfi"].shape == data.vis.shape
    assert data.index_map["freq"]["centre"] == pytest.approx(797.65625)
    assert data.index_map["freq"]["width"] == pytest.approx(0.390625)
    all_data = andata.AnData.from_acq_h5(acq_fname)
    assert np.allclose(data.vis, all_data.vis[6, [1, 3, 9], :])
    prod = data.index_map["prod"]
    assert list(prod[0]) == [0, 1]
    assert list(prod[2]) == [1, 2]


@ignore_warnings
def test_subset_datasets():
    data = andata.AnData.from_acq_h5(acq_fname, datasets=("fpga_hk",))
    assert list(data.datasets.keys()) == ["fpga_hk"]


# Tests for AnData class.
@pytest.fixture
def corr_data():
    yield andata.CorrData()


def test_properties(corr_data):
    # Check that the "properties" are read only.
    with pytest.raises(AttributeError):
        corr_data.__setattr__("datasets", {})
    with pytest.raises(AttributeError):
        corr_data.__setattr__("cal", {})
    with pytest.raises(AttributeError):
        corr_data.__setattr__("attrs", {})


def test_vis_shortcuts(corr_data):

    # More sophisticated base calculation
    def getbase(a):
        b = a.base
        if b is None:
            return a
        else:
            return getbase(b)

    vis = np.arange(60)
    vis.shape = (3, 2, 10)
    corr_data.create_dataset("vis", data=vis)
    corr_data["vis"].attrs["cal"] = "stuff"
    assert getbase(corr_data.vis[:]) is vis
    assert corr_data.vis.attrs == {"cal": "stuff"}
    assert np.allclose(corr_data.vis[0:2:, 0, 1:3:9], vis[0:2:, 0, 1:3:9])


# Tests for loading and saving to/from file.
@pytest.fixture
def test_fname():
    return "tmp_test_AnData.hdf5"


@pytest.fixture
def write_data(test_fname):
    """Makes sure there is test file to work with."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        data = andata.AnData.from_acq_h5(acq_fname)
    data.save(test_fname, mode="w")
    yield

    # Remove test data.
    if os.path.isfile(test_fname):
        os.remove(test_fname)


def test_load_fname2(test_fname, write_data):
    data = andata.AnData.from_file(test_fname)
    check_result(data, NTIME1)


def test_load_FO(test_fname, write_data):
    F = h5py.File(test_fname, mode="r")
    data = andata.AnData.from_file(F)
    check_result(data, NTIME1)
    # Make sure I can write to data, since it should be in memory.
    data.vis[0, 0, 0] = 10
    data.datasets["timestamp_cpu_s"][10] = 12
    assert data.datasets["timestamp_cpu_s"][10] == 12
    # Make sure the file is still open.
    assert F["timestamp_cpu_s"][10] != 12
    F.close()


# Tests for loading multible acq files."""
@ignore_warnings
def test_load_list():
    data = andata.AnData.from_acq_h5(acq_fname_list)
    check_result(data, NTIME)


@ignore_warnings
def test_load_glob():
    data = andata.AnData.from_acq_h5(acq_fname_root + "/*")
    check_result(data, NTIME)


@ignore_warnings
def test_load_subsets():
    data = andata.AnData.from_acq_h5(acq_fname_list, start=6, stop=-5)
    check_result(data, NTIME - 6 - 5)
    data = andata.AnData.from_acq_h5(acq_fname_list, start=20, stop=-1)
    check_result(data, NTIME - 20 - 1)
    data = andata.AnData.from_acq_h5(acq_fname_list, start=2, stop=8)
    check_result(data, 6)


# Tests for converting from acq format to analysis format on disk."""
@ignore_warnings
def test_stores():
    group = MemGroup()
    data = andata.AnData.from_acq_h5(acq_fname_list, out_group=group)
    assert data._data is group
    assert "vis" in group
    assert group["vis"].name == data.vis.name


# Basic acq format loading error checking."""
@pytest.fixture
def acq_list():
    yield [MemGroup.from_hdf5(f) for f in acq_fname_list]


@ignore_warnings
def test_extra_dataset(acq_list):
    nt = acq_list[1]["vis"].shape[0]
    acq_list[1].create_dataset("stuff", shape=(nt,), dtype=float)
    with pytest.raises(ValueError):
        andata.AnData.from_acq_h5(acq_list)


@ignore_warnings
def test_missing_dataset(acq_list):
    nt = acq_list[0]["vis"].shape[0]
    acq_list[0].create_dataset("stuff", shape=(nt,), dtype=float)
    with pytest.raises(ValueError):
        andata.AnData.from_acq_h5(acq_list)


# Test getting timestamps from acq files."""
@pytest.fixture
def data():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        yield andata.AnData.from_acq_h5(acq_fname)


def test_timestamp(data):
    """Just makes sure timestamps are calculated for acq data without any
    validation."""
    timestamp = data.timestamp
    assert len(timestamp) == NTIME1


@pytest.fixture
def data_list():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        yield [andata.AnData.from_acq_h5(fname) for fname in acq_fname_list]


@ignore_warnings
def test_works(data_list):
    right_answer = andata.AnData.from_acq_h5(acq_fname_list)
    merged_data = andata.concatenate(data_list)
    assert np.allclose(merged_data.vis, right_answer.vis)
    assert len(merged_data.index_map["time"]) == len(right_answer.index_map["time"])


@ignore_warnings
def test_start_end_inds(data_list):
    right_answer = andata.AnData.from_acq_h5(acq_fname_list, start=3, stop=26)
    merged_data = andata.concatenate(data_list, start=3, stop=26)
    assert np.allclose(merged_data.vis, right_answer.vis)
    assert len(merged_data.index_map["time"]) == len(right_answer.index_map["time"])


def check_result(data, ntime):
    """Checks that the data more or less looks like it should if converted from
    the test data."""

    # Make sure the data is there and that it has the expected shape.
    assert data.vis.shape == (1024, 36, ntime)
    assert data.vis.dtype == np.complex64
    # Check that 'serial_adc was properly split into 8 datasets.
    count_serial_adc = 0
    for dset_name in data.datasets.keys():
        if dset_name[:10] == "serial_adc":
            count_serial_adc += 1
            assert "cal" in data.datasets[dset_name].attrs
            assert data.datasets[dset_name].attrs["cal"] == ATEL
    assert count_serial_adc == 8
    # Make sure the cal data is acctually there.
    assert ATEL in data.cal
    # Check the fpga_hk dataset, as it is the only one that didn't need to be split.
    assert data.datasets["fpga_hk"].shape == (1, ntime)
    assert data.datasets["fpga_hk"].dtype == np.float32
    # Check a few of the attributes for known values.
    assert "n_antenna" in data.history["acq"]
    assert data.history["acq"]["n_antenna"] == 8
    # Check a few of the cal entries.
    assert "b" in data.cal[ATEL]
    assert data.cal[ATEL]["b"] == "3900"
    # This is the only actual check of the content of the datasets.  Make sure
    # the timestamp is increasing in exactly 10s increments.
    assert np.all(np.diff(data.datasets["timestamp_cpu_s"]) == 10)
    # Check that the data attributes are correctly calculated.
    freq_width = 400.0 / 1024
    assert np.allclose(data.index_map["freq"]["width"], freq_width)
    freq_centre = np.linspace(800.0, 400.0, 1024, endpoint=False)
    assert np.allclose(data.index_map["freq"]["centre"], freq_centre)
