"""Unit tests for ephemeris module."""


import os
from datetime import datetime

import pytest
from pytest import approx
import numpy as np

import caput.time as ctime
from ch_util import ephemeris


# Download the required Skyfield files from a mirror on a CHIME server.
#
# The upstream servers for the timescale and ephemeris data can be
# flaky. Use this to ensure a copy will be downloaded at the risk of it
# being potentially out of date. This is useful for things like CI
# servers, but otherwise letting Skyfield do it's downloading is a
# better idea.

mirror_url = "https://bao.chimenet.ca/skyfield/"

files = ["Leap_Second.dat", "finals2000A.all", "de421.bsp"]

loader = ctime.skyfield_wrapper.load
for file in files:
    if not os.path.exists(loader.path_to(file)):
        loader.download(mirror_url + file)


def test_transit_delta_t():
    t_start = 21351.34
    t1 = ephemeris.transit_times(123.0, t_start)[0]
    t2 = ephemeris.transit_times(125.0, t_start)[0]
    delta = (t2 - t1) % 86400

    assert delta == approx(8.0 * 60.0 * ephemeris.STELLAR_S, abs=0.1)


def test_transit_sources():

    # Check at an early time (this is close to the UNIX epoch)
    t_start = 12315.123
    t1 = ephemeris.transit_times(350.8664, t_start)[0]  # This is CasA's RA
    t2 = ephemeris.transit_times(ephemeris.CasA, t_start)[0]
    # Due to precession of the polar axis this only matches within ~30s
    assert t1 == approx(t2, abs=30)

    # Check at an early time (this is close to the UNIX epoch)
    t_start = datetime(2001, 1, 1)
    t1 = ephemeris.transit_times(350.8664, t_start)[0]  # This is CasA's RA
    t2 = ephemeris.transit_times(ephemeris.CasA, t_start)[0]
    # This is the J2000 epoch and so the precession should be ~0.
    assert t1 == approx(t2, abs=0.1)


def test_transit_against_transit_ra():
    t_start = 65422.2
    ra = 234.54234
    t = ephemeris.transit_times(ra, t_start)[0]
    ra_back_calculated = ephemeris.lsa(t)

    assert ra == approx(ra_back_calculated, abs=1e-2)


def test_csd():
    """Test CHIME sidereal day definition."""
    # csd_zero = 1384489290.908534
    # csd_zero = 1384489290.224582
    csd_zero = 1384489291.0995445
    et1 = ephemeris.datetime_to_unix(datetime(2013, 11, 14))

    # Check the zero of CSD (1e-7 accuracy ~ 10 milliarcsec)
    assert ephemeris.csd(csd_zero) == approx(0.0, abs=1e-6)

    # Check that the fractional part is equal to the transit RA
    assert (360.0 * (ephemeris.csd(et1) % 1.0)) == approx(
        ephemeris.chime.unix_to_lsa(et1), abs=1e-7
    )

    # Check a specific precalculated CSD
    csd1 = -1.1848347442894998
    assert ephemeris.csd(et1) == approx(csd1, abs=1e-7)

    # Check vectorization
    test_args = np.array([csd_zero, et1])
    test_ans = np.array([0.0, csd1])
    assert ephemeris.csd(test_args) == approx(test_ans, abs=1e-7)


def test_transit_RA():

    # transit RA is deprecated and should just throw an exception
    with pytest.raises(NotImplementedError):
        ephemeris.transit_RA(0.0)
