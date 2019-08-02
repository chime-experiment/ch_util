#!/usr/bin/env python
# -*- coding: utf-8 -*-
# === Start Python 2/3 compatibility
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614
# === End Python 2/3 compatibility

from ch_util.andata import HKPData
from datetime import date, timedelta
from datetime import datetime as dt
import os.path
import sys


def total_gpu_power(day):
    filename = "/mnt/gong/archive/{}01T000000Z_chime_hkp/hkp_prom_{}.h5". \
               format(day.strftime('%Y%m'), day.strftime('%Y%m%d'))
    if not os.path.isfile(filename):
        return None
    f = HKPData.from_acq_h5(filename, metrics=["pdu_inlet_sensor_value"])
    if 'pdu_inlet_sensor_value' not in f:
        return None
    d = f.select('pdu_inlet_sensor_value')
    m = d.query("sensor=='apparent_power'")
    pdus = m.groupby('instance')
    return pdus.resample('1h').mean().sum()


if __name__ == '__main__':
    if len(sys.argv) > 3:
        print('Usage: %s START_DATE END_DATE' % os.path.basename(__file__))
        print('''

        Prints daily total power consumed in the GPU cans, for each day from
        START to END_DATE.

        The power use is approximated as the sum of hourly averages of each
        PDUs' pdu_inlet_sensor_value metric. The default START and END_DATE are
        yesterday, which should be the latest data available in the archive.

        If given, dates must be specified in the YYYY-MM-DD format.''')
        exit(1)

    if len(sys.argv) > 1:
        start = dt.strptime(sys.argv[1], '%Y-%m-%d').date()
    else:
        start = date.today() - timedelta(1)
    if len(sys.argv) > 2:
        start = dt.strptime(sys.argv[2], '%Y-%m-%d').date()
    else:
        end = date.today() - timedelta(1)

    if start > end:
        start, end = end, start

    for i in range((end - start).days + 1):
        day = start + timedelta(i)
        print("%s: " % day, end='')

        power = total_gpu_power(day)
        if power is not None:
            print('%0.2f kWh' % (power/1000,))
        else:
            print('None')

