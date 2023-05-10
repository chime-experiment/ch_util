"""
Ephemeris routines

The precession of the Earth's axis gives noticeable shifts in object
positions over the life time of CHIME. To minimise the effects of this we
need to be careful and consistent with our ephemeris calculations.
Historically Right Ascension has been given with respect to the Vernal
Equinox which has a significant (and unnecessary) precession in the origin of
the RA axis. To avoid this we use the new Celestial Intermediate Reference
System which does not suffer from this issue.

Practically this means that when calculating RA, DEC coordinates for a source
position at a *given time* you must be careful to obtain CIRS coordinates
(and not equinox based ones). Internally using `ephemeris.object_coords` does
exactly that for you, so for any lookup of coordinates you should use that on
your requested body.

Note that the actual coordinate positions of sources must be specified using
RA, DEC coordinates in ICRS (which is roughly equivalent to J2000). The
purpose of object_coords is to transform into new RA, DEC coordinates taking
into account the precession and nutation of the Earth's polar axis since
then.

These kind of coordinate issues are tricky, confusing and hard to debug years
later, so if you're unsure you are recommended to seek some advice.

Constants
=========

:const:`CHIMELATITUDE`
    CHIME's latitude [degrees].
:const:`CHIMELONGITUDE`
    CHIME's longitude [degrees].
:const:`CHIMEALTITUDE`
    CHIME's altitude [metres].
:const:`SIDEREAL_S`
    Number of SI seconds in a sidereal second [s/sidereal s]. You probably want
    STELLAR_S instead.
:const:`STELLAR_S`
    Number of SI seconds in a stellar second [s/stellar s].
:const:`CasA`
    :class:`skyfield.starlib.Star` representing Cassiopeia A.
:const:`CygA`
    :class:`skyfield.starlib.Star` representing Cygnus A.
:const:`TauA`
    :class:`skyfield.starlib.Star` representing Taurus A.
:const:`VirA`
    :class:`skyfield.starlib.Star` representing Virgo A.


Telescope Instances
===================

- :const:`chime`


Ephemeris Functions
===================

- :py:meth:`skyfield_star_from_ra_dec`
- :py:meth:`transit_times`
- :py:meth:`solar_transit`
- :py:meth:`lunar_transit`
- :py:meth:`setting_times`
- :py:meth:`solar_setting`
- :py:meth:`lunar_setting`
- :py:meth:`rising_times`
- :py:meth:`solar_rising`
- :py:meth:`lunar_rising`
- :py:meth:`_is_skyfield_obj`
- :py:meth:`peak_RA`
- :py:meth:`get_source_dictionary`
- :py:meth:`lsa`


Time Utilities
==============

- :py:meth:`ensure_unix`
- :py:meth:`chime_local_datetime`
- :py:meth:`unix_to_datetime`
- :py:meth:`datetime_to_unix`
- :py:meth:`datetime_to_timestr`
- :py:meth:`timestr_to_datetime`
- :py:meth:`unix_to_skyfield_time`
- :py:meth:`skyfield_time_to_unix`
- :py:meth:`time_of_day`
- :py:meth:`csd`
- :py:meth:`csd_to_unix`
- :py:meth:`unix_to_csd`
- :py:meth:`parse_date`


Miscellaneous Utilities
=======================

- :py:meth:`galt_pointing_model_ha`
- :py:meth:`galt_pointing_model_dec`
- :py:meth:`sphdist`
"""

from datetime import datetime
from numpy.core.multiarray import unravel_index

# NOTE: Load Skyfield API but be sure to use skyfield_wrapper for loading data
import skyfield.api

import numpy as np

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

# Calvin derived the horizontal position of the center of the focal lines...
# ...and the elevation of the focal line from survey coordinates:
# All altitudes given in meters above sea level
CHIMELATITUDE = 49.3207092194
CHIMELONGITUDE = -119.6236774310
CHIMEALTITUDE = 555.372

# Calvin also positioned the GBO/TONE Outrigger similarly.
# GBO/TONE Outrigger
TONELATITUDE = 38.4292962636
TONELONGITUDE = -79.8451625395
TONEALTITUDE = 810.000

# Rough position for outriggers.
# These will be updated as positioning gets refined.
# https://bao.chimenet.ca/doc/documents/1727
KKOLATITUDE = 49.41905
KKOLONGITUDE = -120.5253
KKOALTITUDE = 835

# Aliases for backwards compatibility
PCOLATITUDE = KKOLATITUDE
PCOLONGITUDE = KKOLONGITUDE
PCOALTITUDE = KKOALTITUDE

GBOLATITUDE = 38.436122
GBOLONGITUDE = -79.827922
GBOALTITUDE = 2710 / 3.28084

HCOLATITUDE = 40.8171082
HCOLONGITUDE = -121.4689584
HCOALTITUDE = 3346 / 3.28084

# Create the Observer instances for CHIME and outriggers
chime = Observer(
    lon=CHIMELONGITUDE,
    lat=CHIMELATITUDE,
    alt=CHIMEALTITUDE,
    lsd_start=datetime(2013, 11, 15),
)

tone = Observer(
    lon=TONELONGITUDE,
    lat=TONELATITUDE,
    alt=TONEALTITUDE,
    lsd_start=datetime(2013, 11, 15),
)

kko = Observer(
    lon=KKOLONGITUDE,
    lat=KKOLATITUDE,
    alt=KKOALTITUDE,
    lsd_start=datetime(2013, 11, 15),
)

gbo = Observer(
    lon=GBOLONGITUDE,
    lat=GBOLATITUDE,
    alt=GBOALTITUDE,
    lsd_start=datetime(2013, 11, 15),
)

hco = Observer(
    lon=HCOLONGITUDE,
    lat=HCOLATITUDE,
    alt=HCOALTITUDE,
    lsd_start=datetime(2013, 11, 15),
)


def _get_chime():
    import warnings

    warnings.warn("Use `ephemeris.chime` instead.", DeprecationWarning)
    return chime


def galt_pointing_model_ha(
    ha_in, dec_in, a=[-5.872, -0.5292, 5.458, -0.076, -0.707, 0.0, 0.0]
):
    """Calculate pointing correction in hour angle for the Galt Telescope
    See description of the pointing model by Lewis Knee CHIME document library
    754 https://bao.chimenet.ca/doc/documents/754

    Parameters
    ----------
    ha, dec : Skyfield Angle objects
        Target hour angle and declination

    a : list of floats
        List of coefficients (in arcmin) for the pointing model
        (NOTE: it is very unlikely that a user will want to change these
        from the defaults, which are taken from the pointing model as of
        2019-2-15)

    Returns
    -------
    Skyfield Angle object
        Angular offset in hour angle
    """

    from skyfield.positionlib import Angle

    ha = ha_in.radians
    dec = dec_in.radians

    # hour angle pointing correction in arcmin
    delta_ha_cos_dec = (
        a[0]
        + a[1] * np.sin(dec)
        + a[2] * np.cos(dec)
        + a[3] * np.sin(ha) * np.sin(dec)
        + a[4] * np.cos(ha) * np.sin(dec)
        + a[5] * np.sin(ha) * np.cos(dec)
        + a[6] * np.cos(ha) * np.cos(dec)
    )

    return Angle(degrees=(delta_ha_cos_dec / np.cos(dec)) / 60.0)


def galt_pointing_model_dec(
    ha_in, dec_in, b=[1.081, 0.707, -0.076, 0.0, 0.0, 0.0, 0.0]
):
    """Calculate pointing correction in declination for the Galt Telescope
    See description of the pointing model by Lewis Knee CHIME document library
    754 https://bao.chimenet.ca/doc/documents/754

    Parameters
    ----------
    ha, dec : Skyfield Angle objects
        Target hour angle and declination

    b : list of floats
        List of coefficients (in arcmin) for the pointing model
        (NOTE: it is very unlikely that a user will want to change these
        from the defaults, which are taken from the pointing model as of
        2019-2-15)

    Returns
    -------
    Skyfield Angle object
        Angular offset in hour angle
    """

    from skyfield.positionlib import Angle

    ha = ha_in.radians
    dec = dec_in.radians

    # declination pointing correction in arcmin
    delta_dec = (
        b[0]
        + b[1] * np.sin(ha)
        + b[2] * np.cos(ha)
        + b[3] * np.sin(dec)
        + b[4] * np.cos(dec)
        + b[5] * np.sin(dec) * np.cos(ha)
        + b[6] * np.sin(dec) * np.sin(ha)
    )

    return Angle(degrees=delta_dec / 60.0)


def parse_date(datestring):
    """Convert date string to a datetime object.

    Parameters
    ----------
    datestring : string
        Date as YYYYMMDD-AAA, where AAA is one of [UTC, PST, PDT]

    Returns
    -------
    date : datetime
        A python datetime object in UTC.
    """
    from datetime import datetime, timedelta
    import re

    rm = re.match("([0-9]{8})-([A-Z]{3})", datestring)
    if rm is None:
        msg = (
            "Wrong format for datestring: {0}.".format(datestring)
            + "\nShould be YYYYMMDD-AAA, "
            + "where AAA is one of [UTC,PST,PDT]"
        )
        raise ValueError(msg)

    datestring = rm.group(1)
    tzoffset = 0.0
    tz = rm.group(2)

    tzs = {"PDT": -7.0, "PST": -8.0, "EDT": -4.0, "EST": -5.0, "UTC": 0.0}

    if tz is not None:
        try:
            tzoffset = tzs[tz.upper()]
        except KeyError:
            print("Time zone {} not known. Known time zones:".format(tz))
            for key, value in tzs.items():
                print(key, value)
            print("Using UTC{:+.1f}.".format(tzoffset))

    return datetime.strptime(datestring, "%Y%m%d") - timedelta(hours=tzoffset)


def utc_lst_to_mjd(datestring, lst, obs=chime):
    """Convert datetime string and LST to corresponding modified Julian Day

    Parameters
    ----------
    datestring : string
        Date as YYYYMMDD-AAA, where AAA is one of [UTC, PST, PDT]
    lst : float
        Local sidereal time at DRAO (CHIME) in decimal hours
    obs : caput.Observer object

    Returns
    -------
    mjd : float
        Modified Julian Date corresponding to the given time.
    """
    return (
        unix_to_skyfield_time(
            obs.lsa_to_unix(lst * 360 / 24, datetime_to_unix(parse_date(datestring)))
        ).tt
        - 2400000.5
    )


def sphdist(long1, lat1, long2, lat2):
    """
    Return the angular distance between two coordinates.

    Parameters
    ----------

    long1, lat1 : Skyfield Angle objects
        longitude and latitude of the first coordinate. Each should be the
        same length; can be one or longer.

    long2, lat2 : Skyfield Angle objects
        longitude and latitude of the second coordinate. Each should be the
        same length. If long1, lat1 have length longer than 1, long2 and
        lat2 should either have the same length as coordinate 1 or length 1.

    Returns
    -------
    dist : Skyfield Angle object
        angle between the two coordinates
    """
    from skyfield.positionlib import Angle

    dsinb = np.sin((lat1.radians - lat2.radians) / 2.0) ** 2

    dsinl = (
        np.cos(lat1.radians)
        * np.cos(lat2.radians)
        * (np.sin((long1.radians - long2.radians) / 2.0)) ** 2
    )

    dist = np.arcsin(np.sqrt(dsinl + dsinb))

    return Angle(radians=2 * dist)


def solar_transit(start_time, end_time=None, obs=chime):
    """Find the Solar transits between two times for CHIME.

    Parameters
    ----------
    start_time : float (UNIX time) or datetime
        Start time to find transits.
    end_time : float (UNIX time) or datetime, optional
        End time for finding transits. If `None` default, search for 24 hours
        after start time.

    Returns
    -------
    transit_times : array_like
        Array of transit times (in UNIX time).

    """

    planets = skyfield_wrapper.ephemeris
    sun = planets["sun"]
    return obs.transit_times(sun, start_time, end_time)


def lunar_transit(start_time, end_time=None, obs=chime):
    """Find the Lunar transits between two times for CHIME.

    Parameters
    ----------
    start_time : float (UNIX time) or datetime
        Start time to find transits.
    end_time : float (UNIX time) or datetime, optional
        End time for finding transits. If `None` default, search for 24 hours
        after start time.

    Returns
    -------
    transit_times : array_like
        Array of transit times (in UNIX time).

    """

    planets = skyfield_wrapper.ephemeris
    moon = planets["moon"]
    return obs.transit_times(moon, start_time, end_time)


# Create CHIME specific versions of various calls.
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


def transit_RA(time):
    """No longer supported. Use `lsa` instead."""
    raise NotImplementedError(
        "No longer supported. Use the better defined `lsa` instead."
    )


def chime_local_datetime(*args):
    """Create a :class:`datetime.datetime` object in Canada/Pacific timezone.

    Parameters
    ----------
    *args
        Any valid arguments to the constructor of :class:`datetime.datetime`
        except *tzinfo*. Local date and time at CHIME.

    Returns
    -------
    dt : :class:`datetime.datetime`
        Timezone naive date and time but converted to UTC.

    """

    from pytz import timezone

    tz = timezone("Canada/Pacific")
    dt_naive = datetime(*args)
    if dt_naive.tzinfo:
        msg = "Time zone should not be supplied."
        raise ValueError(msg)
    dt_aware = tz.localize(dt_naive)
    return dt_aware.replace(tzinfo=None) - dt_aware.utcoffset()


def solar_setting(start_time, end_time=None, obs=chime):
    """Find the Solar settings between two times for CHIME.

    Parameters
    ----------
    start_time : float (UNIX time) or datetime
        Start time to find settings.
    end_time : float (UNIX time) or datetime, optional
        End time for finding settings. If `None` default, search for 24 hours
        after start time.

    Returns
    -------
    setting_times : array_like
        Array of setting times (in UNIX time).

    """

    planets = skyfield_wrapper.ephemeris
    sun = planets["sun"]
    # Use 0.6 degrees for the angular diameter of the Sun to be conservative:
    return obs.set_times(sun, start_time, end_time, diameter=0.6)


def lunar_setting(start_time, end_time=None, obs=chime):
    """Find the Lunar settings between two times for CHIME.

    Parameters
    ----------
    start_time : float (UNIX time) or datetime
        Start time to find settings.
    end_time : float (UNIX time) or datetime, optional
        End time for finding settings. If `None` default, search for 24 hours
        after start time.

    Returns
    -------
    setting_times : array_like
        Array of setting times (in UNIX time).

    """

    planets = skyfield_wrapper.ephemeris
    moon = planets["moon"]
    # Use 0.6 degrees for the angular diameter of the Moon to be conservative:
    return obs.set_times(moon, start_time, end_time, diameter=0.6)


def solar_rising(start_time, end_time=None, obs=chime):
    """Find the Solar risings between two times for CHIME.

    Parameters
    ----------
    start_time : float (UNIX time) or datetime
        Start time to find risings.
    end_time : float (UNIX time) or datetime, optional
        End time for finding risings. If `None` default, search for 24 hours
        after start time.

    Returns
    -------
    rising_times : array_like
        Array of rising times (in UNIX time).

    """

    planets = skyfield_wrapper.ephemeris
    sun = planets["sun"]
    # Use 0.6 degrees for the angular diameter of the Sun to be conservative:
    return obs.rise_times(sun, start_time, end_time, diameter=0.6)


def lunar_rising(start_time, end_time=None, obs=chime):
    """Find the Lunar risings between two times for CHIME.

    Parameters
    ----------
    start_time : float (UNIX time) or datetime
        Start time to find risings.
    end_time : float (UNIX time) or datetime, optional
        End time for finding risings. If `None` default, search for 24 hours after
        start time.

    Returns
    -------
    rising_times : array_like
        Array of rising times (in UNIX time).

    """

    planets = skyfield_wrapper.ephemeris
    moon = planets["moon"]
    # Use 0.6 degrees for the angular diameter of the Moon to be conservative:
    return obs.rise_times(moon, start_time, end_time, diameter=0.6)


def _is_skyfield_obj(body):
    return (
        isinstance(body, skyfield.starlib.Star)
        or isinstance(body, skyfield.vectorlib.VectorSum)
        or isinstance(body, skyfield.jpllib.ChebyshevPosition)
    )


def Star_cirs(ra, dec, epoch):
    """Wrapper for skyfield.api.star that creates a position given CIRS
    coordinates observed from CHIME

    Parameters
    ----------
    ra, dec : skyfield.api.Angle
        RA and dec of the source in CIRS coordinates
    epoch : skyfield.api.Time
        Time of the observation

    Returns
    -------
    body : skyfield.api.Star
        Star object in ICRS coordinates
    """

    from skyfield.api import Star

    return cirs_radec(Star(ra=ra, dec=dec, epoch=epoch))


def cirs_radec(body, date=None, deg=False, obs=chime):
    """Converts a Skyfield body in CIRS coordinates at a given epoch to
    ICRS coordinates observed from CHIME

    Parameters
    ----------
    body : skyfield.api.Star
        Skyfield Star object with positions in CIRS coordinates.

    Returns
    -------
    new_body : skyfield.api.Star
        Skyfield Star object with positions in ICRS coordinates
    """

    from skyfield.positionlib import Angle
    from skyfield.api import Star

    ts = skyfield_wrapper.timescale

    epoch = ts.tt_jd(np.median(body.epoch))

    pos = obs.skyfield_obs().at(epoch).observe(body)

    # Matrix CT transforms from CIRS to ICRF (https://rhodesmill.org/skyfield/time.html)
    r_au, dec, ra = skyfield.functions.to_polar(
        np.einsum("ij...,j...->i...", epoch.CT, pos.position.au)
    )

    return Star(
        ra=Angle(radians=ra, preference="hours"), dec=Angle(radians=dec), epoch=epoch
    )


def object_coords(body, date=None, deg=False, obs=chime):
    """Calculates the RA and DEC of the source.

    Gives the ICRS coordinates if no date is given (=J2000), or if a date is
    specified gives the CIRS coordinates at that epoch.

    This also returns the *apparent* position, including abberation and
    deflection by gravitational lensing. This shifts the positions by up to
    20 arcseconds.

    Parameters
    ----------
    body : skyfield source
        skyfield.starlib.Star or skyfield.vectorlib.VectorSum or
        skyfield.jpllib.ChebyshevPosition body representing the source.
    date : float
        Unix time at which to determine ra of source If None, use Jan 01
        2000.
    deg : bool
        Return RA ascension in degrees if True, radians if false (default).
    obs : `caput.time.Observer`
        An observer instance to use. If not supplied use `chime`. For many
        calculations changing from this default will make little difference.

    Returns
    -------
    ra, dec: float
        Position of the source.
    """

    if date is None:  # No date, get ICRS coords
        if isinstance(body, skyfield.starlib.Star):
            ra, dec = body.ra.radians, body.dec.radians
        else:
            raise ValueError(
                "Body is not fixed, cannot calculate coordinates without a date."
            )

    else:  # Calculate CIRS position with all corrections
        date = unix_to_skyfield_time(date)
        radec = obs.skyfield_obs().at(date).observe(body).apparent().cirs_radec(date)

        ra, dec = radec[0].radians, radec[1].radians

    # If requested, convert to degrees
    if deg:
        ra = np.degrees(ra)
        dec = np.degrees(dec)

    # Return
    return ra, dec


def peak_RA(body, date=None, deg=False):
    """Calculates the RA where a source is expected to peak in the beam.
    Note that this is not the same as the RA where the source is at
    transit, since the pathfinder is rotated with respect to north.

    Parameters
    ----------
    body : ephem.FixedBody
        skyfield.starlib.Star or skyfield.vectorlib.VectorSum or
        skyfield.jpllib.ChebyshevPosition or Ephemeris body
        representing the source.
    date : float
        Unix time at which to determine ra of source
        If None, use Jan 01 2000.
        Ignored if body is not a skyfield object
    deg : bool
        Return RA ascension in degrees if True,
        radians if false (default).

    Returns
    -------
    peak_ra : float
        RA when the transiting source peaks.
    """

    _PF_ROT = np.radians(1.986)  # Pathfinder rotation from north.
    _PF_LAT = np.radians(CHIMELATITUDE)  # Latitude of pathfinder

    # Extract RA and dec of object
    ra, dec = object_coords(body, date=date)

    # Estimate the RA at which the transiting source peaks
    ra = ra + np.tan(_PF_ROT) * (dec - _PF_LAT) / np.cos(_PF_LAT)

    # If requested, convert to degrees
    if deg:
        ra = np.degrees(ra)

    # Return
    return ra


def get_source_dictionary(*args):
    """Returns a dictionary containing :class:`skyfield.starlib.Star`
    objects for common radio point sources.  This is useful for
    obtaining the skyfield representation of a source from a string
    containing its name.

    Parameters
    ----------
    catalog_name : str
        Name of the catalog.  This must be the basename of the json file
        in the `ch_util/catalogs` directory.  Can take multiple catalogs,
        with the first catalog favoured for any overlapping sources.

    Returns
    -------
    src_dict : dictionary
        Format is {'SOURCE_NAME': :class:`skyfield.starlib.Star`, ...}

    """

    import os
    import json

    src_dict = {}
    for catalog_name in reversed(args):
        path_to_catalog = os.path.join(
            os.path.dirname(__file__),
            "catalogs",
            os.path.splitext(catalog_name)[0] + ".json",
        )

        with open(path_to_catalog, "r") as handler:
            catalog = json.load(handler)

        for name, info in catalog.items():
            src_dict[name] = skyfield_star_from_ra_dec(info["ra"], info["dec"], name)

    return src_dict


# Common radio point sources
source_dictionary = get_source_dictionary(
    "primary_calibrators_perley2016",
    "specfind_v2_5Jy_vollmer2009",
    "atnf_psrcat",
    "hfb_target_list",
)

#: :class:`skyfield.starlib.Star` representing Cassiopeia A.
CasA = source_dictionary["CAS_A"]

#: :class:`skyfield.starlib.Star` representing Cygnus A.
CygA = source_dictionary["CYG_A"]

#: :class:`skyfield.starlib.Star` representing Taurus A.
TauA = source_dictionary["TAU_A"]

#: :class:`skyfield.starlib.Star` representing Virgo A.
VirA = source_dictionary["VIR_A"]


if __name__ == "__main__":
    import doctest

    doctest.testmod()
