"""
This module is deprecated.

For CHIME-specific stuff, use `ch_ephem`.
For instrument-independent stuff, use `caput`.
"""

import warnings

warnings.warn("The ch_util.ephemeris module is deprecated.", DeprecationWarning)

from caput.interferometry import sphdist
from caput.time import (
    unix_to_datetime,
    datetime_to_unix,
    datetime_to_timestr,
    timestr_to_datetime,
    leap_seconds_between,
    time_of_day,
    Observer,
    unix_to_skyfield_time,
    skyfield_time_to_unix,
    skyfield_star_from_ra_dec,
    skyfield_wrapper,
    ensure_unix,
    SIDEREAL_S,
    STELLAR_S,
)

from ch_ephem.coord import star_cirs as Star_cirs
from ch_ephem.coord import peak_ra as peak_RA
from ch_ephem.coord import (
    cirs_radec,
    object_coords,
    hadec_to_bmxy,
    bmxy_to_hadec,
    get_range_rate,
)
from ch_ephem.pointing import galt_pointing_model_ha, galt_pointing_model_dec
from ch_ephem.sources import get_source_dictionary, CasA, CygA, TauA, VirA
from ch_ephem.time import parse_date, utc_lst_to_mjd, chime_local_datetime
from ch_ephem.observers import chime, tone, kko, gbo, hco

from .hfbcat import get_doppler_shifted_freq

CHIMELATITUDE = chime.latitude
CHIMELONGITUDE = chime.longitude
CHIMEALTITUDE = chime.altitude

TONELATITUDE = tone.latitude
TONELONGITUDE = tone.longitude
TONEALTITUDE = tone.altitude

KKOLATITUDE = kko.latitude
KKOLONGITUDE = kko.longitude
KKOALTITUDE = kko.altitude

PCOLATITUDE = KKOLATITUDE
PCOLONGITUDE = KKOLONGITUDE
PCOALTITUDE = KKOALTITUDE

GBOLATITUDE = gbo.latitude
GBOLONGITUDE = gbo.longitude
GBOALTITUDE = gbo.altitude

HCOLATITUDE = hco.latitude
HCOLONGITUDE = hco.longitude
HCOALTITUDE = hco.altitude


lsa_to_unix = chime.lsa_to_unix
unix_to_lsa = chime.unix_to_lsa
unix_to_csd = chime.unix_to_lsd
csd_to_unix = chime.lsd_to_unix
csd = unix_to_csd
lsa = unix_to_lsa
transit_times = chime.transit_times
setting_times = chime.set_times
rising_times = chime.rise_times
CSD_ZERO = chime.lsd_start_day


# These are now caput.time.Observer methods
def solar_setting(start_time, end_time=None, obs=chime):
    return obs.solar_setting(start_time, end_time)


def lunar_setting(start_time, end_time=None, obs=chime):
    return obs.lunar_setting(start_time, end_time)


def solar_rising(start_time, end_time=None, obs=chime):
    return obs.solar_rising(start_time, end_time)


def lunar_rising(start_time, end_time=None, obs=chime):
    return obs.lunar_rising(start_time, end_time)


def solar_transit(start_time, end_time=None, obs=chime):
    return obs.solar_transit(start_time, end_time)


def lunar_transit(start_time, end_time=None, obs=chime):
    return obs.lunar_transit(start_time, end_time)
