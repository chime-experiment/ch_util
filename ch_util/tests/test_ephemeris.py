"""Unit tests for ephemeris module."""
# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility


import unittest
import random
import time
import math
from datetime import datetime, timedelta

import numpy as np

from ch_util import ephemeris


class TestUT2RATransit(unittest.TestCase):
    def test_epoch(self):
        # At the J2000 epoch, sidereal time and transit RA should be the same.
        epoch = datetime(2000, 1, 1, 11, 58, 56)
        chime = ephemeris._get_chime()
        chime.date = epoch
        ST = math.degrees(chime.sidereal_time())
        unix_epoch = ephemeris.datetime_to_unix(epoch)
        TRA = ephemeris.transit_RA(unix_epoch)
        # Tolerance limited by stellar aberation.
        self.assertTrue(np.allclose(ST, TRA, atol=0.01, rtol=1e-10))

    def test_array(self):
        # At the J2000 epoch, sidereal time and transit RA should be the same.
        epoch = datetime(2000, 1, 1, 11, 58, 56)
        chime = ephemeris._get_chime()
        chime.date = epoch
        ST = math.degrees(chime.sidereal_time())
        # Drift rate should be very close to 1 degree/4minutes.
        delta_deg = np.arange(20)
        delta_deg.shape = (5, 4)
        ST = ST + delta_deg
        unix_epoch = ephemeris.datetime_to_unix(epoch)
        unix_times = unix_epoch + (delta_deg * 60 * 4 * ephemeris.SIDEREAL_S)
        TRA = ephemeris.transit_RA(unix_times)
        self.assertTrue(np.allclose(ST, TRA, atol=0.02, rtol=1e-10))

    def test_delta(self):
        delta = np.arange(0, 200000, 1000)  # Seconds.
        # time.time() when I wrote this.  No leap seconds for the next few
        # days.
        start = 1383679008.816173
        times = start + delta
        start_ra = ephemeris.transit_RA(start)
        ra = ephemeris.transit_RA(times)
        delta_ra = ra - start_ra
        expected = delta / 3600.0 * 15.0 / ephemeris.SIDEREAL_S
        error = ((expected - delta_ra + 180.0) % 360) - 180
        # Tolerance limited by stellar aberation (40" peak to peak).
        self.assertTrue(np.allclose(error, 0, atol=0.02))


class TestTransits(unittest.TestCase):
    def test_delta_t(self):
        t_start = 21351.34
        t1 = ephemeris.transit_times(123.0, t_start)[0]
        t2 = ephemeris.transit_times(125.0, t_start)[0]
        delta = (t2 - t1) % 86400
        self.assertAlmostEqual(delta, 8.0 * 60.0 * ephemeris.SIDEREAL_S, 1)

    def test_sources(self):
        t_start = 12315.123
        t1 = ephemeris.transit_times(350.86, t_start)[0]
        t2 = ephemeris.transit_times(ephemeris.CasA, t_start)[0]
        self.assertAlmostEqual(t1 / 60.0, t2 / 60.0, 0)

    def test_against_transit_ra(self):
        t_start = 65422.2
        ra = 234.54234
        t = ephemeris.transit_times(ra, t_start)[0]
        ra_back_calulated = ephemeris.transit_RA(t)
        self.assertAlmostEqual(
            ra, ra_back_calulated, 2
        )  # Relax constraint as we're using PyEphem and the more accurate Skyfield


class TestTime(unittest.TestCase):
    def test_datetime_to_string(self):
        dt = datetime(2014, 4, 21, 16, 33, 12, 12356)
        fdt = ephemeris.datetime_to_timestr(dt)
        self.assertEqual(fdt, "20140421T163312Z")

    def test_string_to_datetime(self):
        dt = ephemeris.timestr_to_datetime("20140421T163312Z_stone")
        ans = datetime(2014, 4, 21, 16, 33, 12)
        self.assertEqual(dt, ans)

    def test_datetime_to_unix(self):

        unix_time = time.time()
        dt = datetime.utcfromtimestamp(unix_time)
        new_unix_time = ephemeris.datetime_to_unix(dt)
        self.assertAlmostEqual(new_unix_time, unix_time, 5)

    def quarry_leap_second(self):
        # 'test_' removed from name to deactivate the test untill this can be
        # implemented.
        l_second_date = datetime(2009, 1, 1, 0, 0, 0)
        l_second_date = ephemeris.datetime_to_unix(l_second_date)
        before = l_second_date - 10000
        after = l_second_date + 10000
        after_after = l_second_date + 200
        self.assertTrue(ephemeris.leap_second_between(before, after))
        self.assertFalse(ephemeris.leap_second_between(after, after_after))

    def test_csd(self):
        """Test CHIME sidereal day definition."""
        # csd_zero = 1384489290.908534
        csd_zero = 1384489290.224582
        et1 = ephemeris.datetime_to_unix(datetime(2013, 11, 14))

        # Check the zero of CSD (1e-7 accuracy ~ 10 milliarcsec)
        self.assertAlmostEqual(ephemeris.csd(csd_zero), 0.0, places=6)

        # Check that the fractional part if equal to the transit RA
        self.assertAlmostEqual(
            360.0 * (ephemeris.csd(et1) % 1.0),
            ephemeris.chime_observer().unix_to_lsa(et1),
            places=7,
        )

        # Check a specific precalculated CSD
        csd1 = -1.1848347442894998
        self.assertAlmostEqual(ephemeris.csd(et1), csd1, places=7)

        # Check vectorization
        test_args = np.array([csd_zero, et1])
        test_ans = np.array([0.0, csd1])
        self.assertTrue(np.allclose(ephemeris.csd(test_args), test_ans, atol=1e-7))


if __name__ == "__main__":
    unittest.main()
