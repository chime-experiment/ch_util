"""
==============================================
Ephemeris routines (:mod:`~ch_util.ephemeris`)
==============================================

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
:const:`EPOCH`
    Epoch under use.
:const:`SIDEREAL_S`
    Number of SI seconds in a sidereal second [s/sidereal s].
:const:`CasA`
    :class:`skyfield.starlib.Star` representing Cassiopeia A.
:const:`CygA`
    :class:`skyfield.starlib.Star` representing Cygnus A.
:const:`TauA`
    :class:`skyfield.starlib.Star` representing Taurus A.
:const:`VirA`
    :class:`skyfield.starlib.Star` representing Virgo A.


Wrapper Class
=============

.. autosummary::
    :toctree: generated/

    SkyfieldObserverWrapper



Ephemeris Functions
===================

.. autosummary::
    :toctree: generated/

    skyfield_star_from_ra_dec
    _get_chime
    chime_observer
    transit_times
    solar_transit
    lunar_transit
    setting_times
    solar_setting
    lunar_setting
    rising_times
    solar_rising
    lunar_rising
    _is_skyfield_obj
    peak_RA
    get_source_dictionary
    transit_RA


Time Utilities
==============

.. autosummary::
    :toctree: generated/

    ensure_unix
    chime_local_datetime
    unix_to_datetime
    datetime_to_unix
    datetime_to_timestr
    timestr_to_datetime
    leap_second_between
    unix_to_skyfield_time
    time_of_day
    csd



"""
# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import math
from datetime import datetime
import logging

# NOTE: Load Skyfield API but be sure to use skyfield_wrapper for loading data
import skyfield.api

# import ephem # TODO: delete
import numpy as np

import caput.time as ctime
from caput.time import (
    unix_to_datetime,
    datetime_to_unix,
    datetime_to_timestr,
    timestr_to_datetime,
    leap_seconds_between,
    time_of_day,
    Observer,
    unix_to_skyfield_time,
    skyfield_wrapper,
    ensure_unix,
)


# Kiyo looked these up on Google Earth. Should replace with 'official' numbers.
CHIMELATITUDE = 49.32  # degrees
CHIMELONGITUDE = -119.62  # degrees
# Mateus looked this up on Wikipedia. Should replace with 'official' number.
CHIMEALTITUDE = 545.0  # metres

EPOCH = "J2000"

# Number of seconds in a sidereal second.
# SIDEREAL_S = 1. - 1. / 365.2422
# Above only includes first term in taylor series.  Below is more accurate.
# Copied from wikipeadia.
# SIDEREAL_S = 0.99726958
# Even more accurate.
SIDEREAL_S = 1.0 / (1.0 + 1.0 / 365.259636)


def _ephem_body_from_ra_dec(ra, dec, bd_name):
    """Legacy. Here for backwards compatibility"""
    import ephem

    msg = (
        "CHIME code is transitioning from pyephem to skyfield.\n"
        "Use skyfield_star_from_ra_dec() instead of"
        "_ephem_body_from_ra_dec() in the future."
    )
    logging.warning(msg)

    body = ephem.FixedBody()
    body._ra = math.radians(ra)
    body._dec = math.radians(dec)
    body.name = bd_name
    return body


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


def skyfield_star_from_ra_dec(ra, dec, bd_name=""):
    """ra and dec in degrees"""
    body = skyfield.api.Star(
        ra=skyfield.units.Angle(degrees=ra, preference="hours"),
        dec=skyfield.units.Angle(degrees=dec),
        names=bd_name,
    )
    return body


class SkyfieldObserverWrapper(Observer):
    """Class to emulate the behaviour of pyephem Observers
    but using only skyfield code.
    """

    def __init__(self, *args, **kwargs):
        """ """
        super(SkyfieldObserverWrapper, self).__init__(*args, **kwargs)
        self.skfd_obs = self.skyfield_obs()
        self._date = None

    # Properties
    ############

    @property
    def date(self):
        """ """
        if self._date is None:
            msg = "You need to assing a date to the observer before retrieving it."
            raise RuntimeError(msg)
        return self._date

    @date.setter
    def date(self, dt):
        """ """
        # Ensure 'dt' is of type skyfield.timelib.Time:
        if not isinstance(dt, skyfield.timelib.Time):
            dt = unix_to_skyfield_time(ensure_unix(dt))
        self._date = dt

    # Methods
    #########

    def radec(self, src):
        """ src is a skyfield star """
        apparent = self.skfd_obs.at(self.date).observe(src).apparent()
        # 'epoch' takes into account change in Earth's axis:
        ra, dec, _ = apparent.radec(epoch=self.date)
        return ra, dec

    def cirs_radec(self, src):
        """ src is a skyfield star """
        apparent = self.skfd_obs.at(self.date).observe(src).apparent()
        ra, dec, _ = apparent.cirs_radec(self.date)
        return ra, dec

    def altaz(self, src):
        """ src is a skyfield star """
        apparent = self.skfd_obs.at(self.date).observe(src).apparent()
        # 'epoch' takes into account change in Earth's axis:
        alt, az, _ = apparent.altaz()
        return alt, az

    def sidereal_time(self):
        """ In radians"""
        return np.radians(self.date.gast * 15.0 + self.longitude) % (2.0 * np.pi)

    def lha(self, src, symmetric=False):
        """Returns the local hour angle for src.
        Result is in radians

        src is a skyfield star
        """
        # Local sidereal time:
        lst = np.degrees(self.sidereal_time())
        # RA in earth coordinates:
        ra, _ = self.radec(src)
        # Local hour angle:
        lha = (lst - ra._degrees) % 360
        if symmetric:
            lha = lha - 360.0 * (lha // 180)
        return np.radians(lha)

    def next_transit(self, src):
        """Returns the next source transit time

        src : skyfield.starlib.Star or skyfield.vectorlib.VectorSum or skyfield.jpllib.ChebyshevPosition or float
            The body to find settings for. If a float, this is the RA of a
            fixed body in degrees.

        Have to add functionality in case src is an RA in degrees


        """
        from scipy.optimize import newton

        SD = 24.0 * 3600.0 * SIDEREAL_S  # Sidereal day
        time = self.date
        time = ensure_unix(time)  # Ensure 'time' is utime

        def src_lha(tm):
            """ 'tm' is referenced to 'time' """
            self.date = tm + time  # Update self.date
            return np.degrees(self.lha(src, symmetric=True))

        lha = src_lha(0.0)
        if lha > 0.0:
            t0 = (1.0 - lha / 360.0) * SD
        else:
            t0 = -lha / 360.0 * SD

        # Solve with Newton's method:
        transit_time = newton(src_lha, t0, tol=1e-4) + time

        # Return date to original value:
        self.date = time

        return transit_time

    def next_setting(self, src, ang_diam=0.0):
        """Returns the next source set time

        src : skyfield.starlib.Star or skyfield.vectorlib.VectorSum or skyfield.jpllib.ChebyshevPosition
            The body to find settings for.
        ang_diam : float
            Angular diameter of object in degrees. Setting is defined when uper edge of object crosses horizon.


        Have to add functionality in case src is RA,DEC in degrees

        """
        from scipy.optimize import newton

        SD = 24.0 * 3600.0 * SIDEREAL_S  # Sidereal day
        time = self.date
        time = ensure_unix(time)  # Ensure 'time' is utime

        def src_alt(tm):
            """ 'tm' is referenced to 'time' """
            self.date = tm + time  # Update self.date
            alt, az = self.altaz(src)
            return alt.radians + math.radians(ang_diam * 0.5)

        def src_lha(tm):
            """ 'tm' is referenced to 'time' """
            self.date = tm + time  # Update self.date
            return np.degrees(self.lha(src, symmetric=False))

        # Aproximate sky rotation angle between transit and setting in degrees:
        ra, dec = self.radec(src)
        trans_to_set = 180.0 - math.degrees(
            np.arccos(np.tan(math.radians(CHIMELATITUDE)) * np.tan(dec.radians))
        )
        # Approximate sky rotation angle past setting
        angle_past = src_lha(0.0) - trans_to_set - ang_diam * 0.5
        if angle_past > 0.0:
            t0 = (1.0 - angle_past / 360.0) * SD
        else:
            t0 = -angle_past / 360.0 * SD

        # Solve with Newton's method:
        set_time = newton(src_alt, t0, tol=1e-4) + time

        # Return date to original value:
        self.date = time

        return set_time

    def next_rising(self, src, ang_diam=0.0):
        """Returns the next source set time

        src : skyfield.starlib.Star or skyfield.vectorlib.VectorSum or skyfield.jpllib.ChebyshevPosition
            The body to find risings for.
        ang_diam : float
            Angular diameter of object in degrees. Setting is defined when uper edge of object crosses horizon.


        Have to add functionality in case src is RA,dec in degrees

        """
        from scipy.optimize import newton

        SD = 24.0 * 3600.0 * SIDEREAL_S  # Sidereal day
        time = self.date
        time = ensure_unix(time)  # Ensure 'time' is utime

        def src_alt(tm):
            """ 'tm' is referenced to 'time' """
            self.date = tm + time  # Update self.date
            alt, az = self.altaz(src)
            return alt.radians + math.radians(ang_diam * 0.5)

        def src_lha(tm):
            """ 'tm' is referenced to 'time' """
            self.date = tm + time  # Update self.date
            return np.degrees(self.lha(src, symmetric=True))

        # Aproximate sky rotation angle between transit and rising in degrees:
        ra, dec = self.radec(src)
        rise_to_trans = 180.0 - math.degrees(
            np.arccos(np.tan(math.radians(CHIMELATITUDE)) * np.tan(dec.radians))
        )
        # Approximate sky rotation angle past rising
        angle_past = src_lha(0.0) + rise_to_trans + ang_diam * 0.5
        if angle_past > 0.0:
            t0 = (1.0 - angle_past / 360.0) * SD
        else:
            t0 = -angle_past / 360.0 * SD

        # Solve with Newton's method:
        rise_time = newton(src_alt, t0, tol=1e-4) + time

        # Return date to original value:
        self.date = time

        return rise_time


def _get_chime():
    """Create a SkyfieldObserverWrapper object for CHIME.
    """
    chime = SkyfieldObserverWrapper(
        lon=CHIMELONGITUDE, lat=CHIMELATITUDE, alt=CHIMEALTITUDE
    )
    # No support for altitude yet.
    return chime


def transit_times(body, start, end=None):
    """Find the times a body transits in a given interval.

    Call signature changed on Feb 20, 2014. Now no loger requires an observer.
    CHIME is always used.

    Parameters
    ----------
    body : skyfield.starlib.Star or skyfield.vectorlib.VectorSum or skyfield.jpllib.ChebyshevPosition or float
        The body to find the transits for. If a float, this is the RA of a
        fixed body in degrees.
    start : float (UNIX time) or datetime
        Start time to find transits.
    end : float (UNIX time) or datetime, optional
        End time for finding transits. If `None` (default) search for 24 hours
        after start time.

    Returns
    -------
    transit_times : array_like
        Array of transit times (in UNIX time).

    Examples
    --------

    >>> transit_times(CasA, 123456.7)
    array([ 174698.371151])

    """

    if _is_skyfield_obj(body):
        pass
    else:
        ra = float(body)
        body = skyfield_star_from_ra_dec(ra, dec=0.0)

    obs = _get_chime()
    start = ensure_unix(start)
    end = ensure_unix(end) if end is not None else start + (24.0 * 3600.0 * SIDEREAL_S)

    obs.date = start

    transits = []

    while True:
        # Find the next transit
        ttime = obs.next_transit(body)

        # Increment the observer time to just after the transit
        obs.date = ttime + 1.0

        # Check whether it is within the bounds (add to list), or outside them
        # (stop search).
        if ttime > end:
            break
        else:
            transits.append(ttime)

    return np.array(transits)


def solar_transit(start_time, end_time=None):
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
    return transit_times(sun, start_time, end_time)


def lunar_transit(start_time, end_time=None):
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
    return transit_times(moon, start_time, end_time)


# Set up an Observer instance for CHIME
def chime_observer():
    """Get a CHIME observer instance.

    Set to the observatory location, and uses the CSD start epoch (i.e. LSA=0.0
    on 15/11/2013).
    """
    obs = Observer(
        lon=CHIMELONGITUDE,
        lat=CHIMELATITUDE,
        alt=CHIMEALTITUDE,
        lsd_start=datetime(2013, 11, 15),
    )

    return obs


# Create CHIME specific versions of various calls.
lsa_to_unix = chime_observer().lsa_to_unix
unix_to_lsa = chime_observer().unix_to_lsa
lsa = unix_to_lsa
unix_to_csd = chime_observer().unix_to_lsd
csd_to_unix = chime_observer().lsd_to_unix
csd = unix_to_csd

CSD_ZERO = chime_observer().lsd_start_day


def transit_RA(time):
    """transit_RA is now an alias for `lsa` which you should use instead.

    The original version of transit_RA returned the J2000 coordinate of the
    currently transiting point on the sky. While this accounted for *most* of
    the spurious precession, there are better ways to achieve this. The best
    option is to use the CIRS RA that is currently transiting, that is
    equivalent to the Local Stellar Angle (`lsa`) which this function now
    returns.
    """

    import warnings

    warnings.warn(
        "transit_RA is now an alias for `ephemeris.lsa`, please "
        "use that directly instead."
    )

    return lsa(time)


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
        Timezone nieve date and time but converted to UTC.

    """

    from pytz import timezone

    tz = timezone("Canada/Pacific")
    dt_nieve = datetime(*args)
    if dt_nieve.tzinfo:
        msg = "Time zone should not be supplied."
        raise ValueError(msg)
    dt_aware = tz.localize(dt_nieve)
    return dt_aware.replace(tzinfo=None) - dt_aware.utcoffset()


def setting_times(body, start, end=None, ang_diam=0.0, max_n=int(1e4)):
    """Find the times a body sets in a given interval.

    Parameters
    ----------
    body : skyfield.starlib.Star or skyfield.vectorlib.VectorSum or skyfield.jpllib.ChebyshevPosition or tuple of floats
        The body to find settings for. If a tuple, this is the RA and DEC of a
        fixed body in degrees.
    start : float (UNIX time) or datetime
        Start time to find sttings.
    end : float (UNIX time) or datetime, optional
        End time for finding settingss. If `None` (default) search for 24 hours
        after start time.
    max_n : int
        Maximum number of events found


    Returns
    -------
    settings : array_like
        Array of setting times (in UNIX time).

    Examples
    --------

    >>> setting_times(TauA, 123456.7)
    array([ 139216.223948])

    """

    if _is_skyfield_obj(body):
        pass
    else:
        ra = float(body[0])
        dec = float(body[1])
        body = skyfield_star_from_ra_dec(ra, dec)

    obs = _get_chime()
    start = ensure_unix(start)
    end = ensure_unix(end) if end is not None else start + (24.0 * 3600.0 * SIDEREAL_S)

    obs.date = start

    settings = []

    for cnt in range(max_n):

        # Find the next setting
        stime = obs.next_setting(body, ang_diam=ang_diam)

        # Increment the observer time to just after the setting
        obs.date = stime + 1.0

        # Check whether it is within the bounds (add to list), or outside them
        # (stop search).
        if stime > end:
            break
        else:
            settings.append(stime)

    else:
        # TODO: implement this with logging:
        msg = "Warning: Reached maximum number of settings to find"
        print(msg)

    return np.array(settings)


def solar_setting(start_time, end_time=None):
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
    return setting_times(sun, start_time, end_time, ang_diam=0.6)


def lunar_setting(start_time, end_time=None):
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
    return setting_times(moon, start_time, end_time, ang_diam=0.6)


def rising_times(body, start, end=None, ang_diam=0.0, max_n=int(1e4)):
    """Find the times a body rises in a given interval.

    Parameters
    ----------
    body : skyfield.starlib.Star or skyfield.vectorlib.VectorSum or skyfield.jpllib.ChebyshevPosition or tuple of floats
        The body to find risings for. If a tuple, this is the RA and DEC of a
        fixed body in degrees.
    start : float (UNIX time) or datetime
        Start time to find sttings.
    end : float (UNIX time) or datetime, optional
        End time for finding risings. If `None` (default) search for 24 hours
        after start time.
    max_n : int
        Maximum number of events found

    Returns
    -------
    risings : array_like
        Array of rising times (in UNIX time).

    Examples
    --------

    >>> rising_times(TauA, 123456.7)
    array([ 139216.223948])

    """

    if _is_skyfield_obj(body):
        pass
    else:
        ra = float(body[0])
        dec = float(body[1])
        body = skyfield_star_from_ra_dec(ra, dec)

    obs = _get_chime()
    start = ensure_unix(start)
    end = ensure_unix(end) if end is not None else start + (24.0 * 3600.0 * SIDEREAL_S)

    obs.date = start

    risings = []

    for cnt in range(max_n):

        # Find the next rising
        rtime = obs.next_rising(body, ang_diam=ang_diam)

        # Increment the observer time to 10 minutes after the rising
        obs.date = rtime + 10.0 * 60.0

        # Check whether it is within the bounds (add to list), or outside them
        # (stop search).
        if rtime > end:
            break
        else:
            risings.append(rtime)

    else:
        # TODO: implement this with logging:
        msg = "Warning: Reached maximum number of risings to find"
        print(msg)

    return np.array(risings)


def solar_rising(start_time, end_time=None):
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
    return rising_times(sun, start_time, end_time, ang_diam=0.6)


def lunar_rising(start_time, end_time=None):
    """Find the Lunar risings between two times for CHIME.

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
    moon = planets["moon"]
    # Use 0.6 degrees for the angular diameter of the Moon to be conservative:
    return rising_times(moon, start_time, end_time, ang_diam=0.6)


def _is_skyfield_obj(body):
    return (
        isinstance(body, skyfield.starlib.Star)
        or isinstance(body, skyfield.vectorlib.VectorSum)
        or isinstance(body, skyfield.jpllib.ChebyshevPosition)
    )


def _ensure_skyfield_body(body):
    """Ensure body is a Skyfield object, converting if needed.
    """

    if not _is_skyfield_obj(body):
        # Try and get out RA, DEC
        if hasattr(body, "_ra"):  # ephem.FixedBody
            ra, dec = body._ra, body._dec
        elif hasattr(body, "ra"):
            ra, dec = body.ra, body.dec
        else:
            raise ValueError("Cannot convert to skyfield body.")

        body = skyfield_star_from_ra_dec(ra, dec)

    return body


def Star_cirs(ra, dec, epoch):
    """Wrapper for skyfield.api.star that creates a position given CIRS
    coordinates observed from CHIME

    Parameters
    ----------
    ra, dec : Skyfield Angle objects
        RA and dec of the source in CIRS coordinates
    epoch : Skyfield time object
        Time of the observation

    Returns
    -------
    Skyfield Star object
        Star object in ICRS coordinates
    """

    from skyfield.api import Star

    return cirs_radec(Star(ra=ra, dec=dec, epoch=epoch))


def cirs_radec(body, date=None, deg=False):
    """Converts a Skyfield body in CIRS coordinates at a given epoch to
    ICRS coordinates observed from CHIME

    Parameters
    ----------
    body : Skyfield Star object with positions in CIRS coordinates

    Returns
    -------
    Skyfield Star object with positions in ICRS coordinates
    """

    from skyfield.positionlib import Angle
    from skyfield.api import Star

    ts = skyfield_wrapper.timescale

    epoch = ts.tt_jd(np.median(body.epoch))

    pos = _get_chime().skyfield_obs().at(epoch).observe(body)

    # Matrix CT transforms from CIRS to ICRF (https://rhodesmill.org/skyfield/time.html)
    r_au, dec, ra = skyfield.functions.to_polar(
        np.einsum("ij...,j...->i...", epoch.CT, pos.position.au)
    )

    return Star(
        ra=Angle(radians=ra, preference="hours"), dec=Angle(radians=dec), epoch=epoch
    )


def object_coords(body, date=None, deg=False):
    """ Calculates the RA and DEC of the source.

    Gives the ICRS coordinates if no date is given (=J2000), or if a date is
    specified gives the CIRS coordinates at that epoch.

    This also returns the *apparent* position, including abberation and
    deflection by gravitational lensing. This shifts the positions by up to
    20 arcseconds.

    Parameters
    ----------
    body : skyfield or pyephem body
        skyfield.starlib.Star or skyfield.vectorlib.VectorSum or
        skyfield.jpllib.ChebyshevPosition or Ephemeris body representing the
        source.
    date : float
        Unix time at which to determine ra of source If None, use Jan 01
        2000.
    deg : bool
        Return RA ascension in degrees if True, radians if false (default).

    Returns
    -------
    ra, dec: float
        Position of the source.
    """

    # Convert from pyephem body if needed
    body = _ensure_skyfield_body(body)

    if date is None:  # No date, get ICRS coords
        if isinstance(body, skyfield.starlib.Star):
            ra, dec = body.ra.radians, body.dec.radians
        else:
            raise ValueError(
                "Body is not fixed, cannot calculate coordinates without a date."
            )

    else:  # Calculate CIRS position with all corrections

        observer = _get_chime()
        observer.date = date

        radec = observer.cirs_radec(body)
        ra, dec = radec[0].radians, radec[1].radians

    # If requested, convert to degrees
    if deg:
        ra = np.degrees(ra)
        dec = np.degrees(dec)

    # Return
    return ra, dec


def peak_RA(body, date=None, deg=False):
    """ Calculates the RA where a source is expected to peak in the beam.
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
    """ Returns a dictionary containing :class:`skyfield.starlib.Star`
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
source_dictionary = get_source_dictionary("primary_calibrators_perley2016")

CasA = source_dictionary["CAS_A"]
CygA = source_dictionary["CYG_A"]
TauA = source_dictionary["TAU_A"]
VirA = source_dictionary["VIR_A"]


if __name__ == "__main__":
    import doctest

    doctest.testmod()
