"""
This module is deprecated.

For CHIME-specific stuff, use `ch_ephem`.

For instrument-independent stuff, use `caput`.
"""

import warnings

from caput.interferometry import sphdist as sphdist
from caput.time import (
    unix_to_datetime as unix_to_datetime,
    datetime_to_unix as datetime_to_unix,
    datetime_to_timestr as datetime_to_timestr,
    timestr_to_datetime as timestr_to_datetime,
    leap_seconds_between as leap_seconds_between,
    time_of_day as time_of_day,
    Observer as Observer,
    unix_to_skyfield_time as unix_to_skyfield_time,
    skyfield_time_to_unix as skyfield_time_to_unix,
    skyfield_star_from_ra_dec as skyfield_star_from_ra_dec,
    skyfield_wrapper as skyfield_wrapper,
    ensure_unix as ensure_unix,
    SIDEREAL_S as SIDEREAL_S,
    STELLAR_S as STELLAR_S,
)

from ch_ephem.coord import peak_ra as peak_RA  # noqa F401
from ch_ephem.coord import (
    hadec_to_bmxy as hadec_to_bmxy,
    bmxy_to_hadec as bmxy_to_hadec,
    get_range_rate as get_range_rate,
)
from ch_ephem.pointing import (
    galt_pointing_model_ha as galt_pointing_model_ha,
    galt_pointing_model_dec as galt_pointing_model_dec,
)
from ch_ephem.sources import (
    get_source_dictionary as get_source_dictionary,
    source_dictionary as source_dictionary,
    CasA as CasA,
    CygA as CygA,
    TauA as TauA,
    VirA as VirA,
)
from ch_ephem.time import (
    parse_date as parse_date,
    utc_lst_to_mjd as utc_lst_to_mjd,
    chime_local_datetime as chime_local_datetime,
)
from ch_ephem.observers import chime, tone, kko, gbo, hco

from .hfbcat import get_doppler_shifted_freq as get_doppler_shifted_freq

warnings.warn(
    "The ch_util.ephemeris module is deprecated. Use `ch_ephem` instead.",
    DeprecationWarning,
)

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


def cirs_radec(body, obs=chime):
    return obs.cirs_radec(body)


def Star_cirs(ra, dec, epoch, obs=chime):
    return obs.star_cirs(ra, dec, epoch)


def object_coords(body, date=None, deg=False, obs=chime):
    return obs.object_coords(body, date, deg)
