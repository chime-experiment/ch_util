"""
===============================================================
Tools for point source calibration (:mod:`cal_utils`)
===============================================================

.. currentmodule:: cal_utils

This module contains tools for performing point-source calibration.

Functions
========

.. autosummary::
    :toctree: generated/

    fit_point_source_transit
    func_point_source_transit
    model_point_source_transit
    fit_point_source_map
    func_2d_gauss
    func_2d_sinc_gauss
    func_dirty_gauss
    func_real_dirty_gauss
    guess_fwhm

"""
# === Start Python 2/3 compatibility
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614
# === End Python 2/3 compatibility

import inspect
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

from ch_util import ephemeris

def fit_point_source_transit(ra, response, response_error, flag=None,
                                 fwhm=None, verbose=False):
    """ Fits the complex point source response to a model that
        consists of a gaussian in amplitude and a line in phase.

    .. math::
        g(ra) = peak_amplitude * \exp{-4 \ln{2} [(ra - centroid)/fwhm]^2} *
                \exp{j [phase_intercept + phase_slope * (ra - centroid)]}

    Parameters
    ----------
    ra : np.ndarray[nra, ]
        Transit right ascension.
    response : np.ndarray[nfreq, ninput, nra]
        Complex array that contains point source response.
    response_error : np.ndarray[nfreq, ninput, nra]
        Real array that contains 1-sigma error on
        point source response.
    flag : np.ndarray[nfreq, ninput, nra]
        Boolean array that indicates which data points to fit.

    Returns
    -------
    param : np.ndarray[nfreq, ninput, nparam]
        Best-fit parameters for each frequency and input:
        [peak_amplitude, centroid, fwhm, phase_intercept, phase_slope].
    param_cov: np.ndarray[nfreq, ninput, nparam, nparam]
        Parameter covariance for each frequency and input.
    """

    # Check if boolean flag was input
    if flag is None:
        flag = np.ones(response.shape, dtype=np.bool)
    elif flag.dtype != np.bool:
        flag = flag.astype(np.bool)

    # Create arrays to hold best-fit parameters and
    # parameter covariance.  Initialize to NaN.
    nfreq = response.shape[0]
    ninput = response.shape[1]
    nparam = 9

    param = np.full([nfreq, ninput, nparam], np.nan, dtype=np.float64)
    param_cov = np.full([nfreq, ninput, nparam, nparam], np.nan, dtype=np.float64)

    # Create initial guess at FWHM if one was not input
    if fwhm is None:
        fwhm = np.full([nfreq, ninput], 2.0, dtype=np.float64)

    # Iterate over frequency/inputs and fit point source transit
    for ff in range(nfreq):
        for ii in range(ninput):

            this_flag = flag[ff, ii]

            # Only perform fit if there is enough data.
            # Otherwise, leave parameter values as NaN.
            if np.sum(this_flag) < 6:
                continue

            # We will fit the complex data.  Break n-element complex array g(ra)
            # into 2n-element real array [Re{g(ra)}, Im{g(ra)}] for fit.
            x = np.tile(ra[this_flag], 2)

            y_complex = response[ff, ii, this_flag]
            y = np.concatenate((y_complex.real, y_complex.imag))

            y_error = np.tile(response_error[ff, ii, this_flag], 2)

            # Initial estimate of parameter values:
            # [peak_amplitude, centroid, fwhm,
            #  phase_intercept, phase_slope,
            #  phase_quad, phase_cube,
            #  phase_quart, phase_quint]
            p0 = np.array([np.max(np.nan_to_num(np.abs(y_complex))), np.median(x), fwhm[ff, ii],
                           np.median(np.nan_to_num(np.angle(y_complex, deg=True))),
                           0.0, 0.0, 0.0, 0.0, 0.0])

            # Perform the fit.  If there is an error,
            # then we leave parameter values as NaN.
            try:
                popt, pcov = curve_fit(func_point_source_transit, x, y,
                                        p0=p0, sigma=y_error, absolute_sigma=True)
            except Exception as error:
                if verbose:
                    print("Feed %d, Freq %d: %s" % (ii, ff, error))
                continue

            # Save the results
            param[ff, ii] = popt
            param_cov[ff, ii] = pcov

    # Return the best-fit parameters and parameter covariance
    return param, param_cov


def func_point_source_transit(x, peak_amplitude, centroid, fwhm,
                                 phase_intercept, phase_slope,
                                 phase_quad, phase_cube,
                                 phase_quart, phase_quint):
    """ Computes parameteric model for the point source transit.
    Model consists of a gaussian in amplitude and a line in phase.
    To be used within curve fitting routine.

    .. math::
        g(ra) = peak_amplitude * \exp{-4 \ln{2} [(ra - centroid)/fwhm]^2} *
                \exp{j [phase_intercept + phase_slope * (ra - centroid)]}

    Parameters
    ----------
    x : np.ndarray[2*nra, ]
        Right ascension in degrees, replicated twice to accomodate
        the real and imaginary components of the response, i.e.,
        x = [ra, ra].
    peak_amplitude : float
        Model parameter.  Normalization of the gaussian.
    centroid : float
        Model parameter.  Centroid of the gaussian in degrees RA.
    fwhm : float
        Model parameter.  Full width at half maximum of the gaussian
        in degrees RA.
    phase_intercept : float
        Model parameter.  Phase at the centroid in units of degrees.
    phase_slope : float
        Model parameter.  Fringe rate in degrees phase per degrees RA.

    Returns
    -------
    model : np.ndarray[2*nra, ]
        Model prediction for the complex point source response,
        packaged as [real{g(ra)}, imag{g(ra)}].
    """

    model = np.empty_like(x)
    nreal = len(x)//2

    dx = x[:nreal] - centroid
    dx = dx - (dx > 180.0)*360.0 + (dx < -180.0)*360.0

    model_amp = peak_amplitude*np.exp(-4.0*np.log(2.0)*(dx/fwhm)**2)
    model_phase = np.deg2rad(phase_intercept + phase_slope * dx +
                             phase_quad * dx**2 + phase_cube * dx**3 +
                             phase_quart * dx**4 + phase_quint * dx**5)
    model[:nreal] = model_amp*np.cos(model_phase)
    model[nreal:] = model_amp*np.sin(model_phase)

    return model


def model_point_source_transit(x, peak_amplitude, centroid, fwhm,
                                 phase_intercept, phase_slope,
                                 phase_quad, phase_cube,
                                 phase_quart, phase_quint):
    """ Computes parameteric model for the point source transit.
    Model consists of a gaussian in amplitude and a line in phase.

    .. math::
        g(ra) = peak_amplitude * \exp{-4 \ln{2} [(ra - centroid)/fwhm]^2} *
                \exp{j [phase_intercept + phase_slope * (ra - centroid)]}

    Parameters
    ----------
    x : np.ndarray[nra, ]
        Right ascension in degrees.
    peak_amplitude : float
        Model parameter.  Normalization of the gaussian.
    centroid : float
        Model parameter.  Centroid of the gaussian in degrees RA.
    fwhm : float
        Model parameter.  Full width at half maximum of the gaussian
        in degrees RA.
    phase_intercept : float
        Model parameter.  Phase at the centroid in units of degrees.
    phase_slope : float
        Model parameter.  Fringe rate in degrees phase per degrees RA.

    Returns
    -------
    model : np.ndarray[nra, ]
        Model prediction for the complex point source response,
        packaged as complex numbers.
    """

    dx = x - centroid
    dx = dx - (dx > 180.0)*360.0 + (dx < -180.0)*360.0

    model_amp = peak_amplitude*np.exp(-4.0*np.log(2.0)*(dx/fwhm)**2)
    model_phase = np.deg2rad(phase_intercept + phase_slope * dx +
                             phase_quad * dx**2 + phase_cube * dx**3 +
                             phase_quart * dx**4 + phase_quint * dx**5)
    model = model_amp*np.exp(1.0j*model_phase)

    return model


def fit_point_source_map(ra, dec, submap, rms=None, dirty_beam=None,
                                          real_map=False, freq=600.0,
                                          ra0=None, dec0=None):
    """ Fits a map of a point source to a model.

    Parameter
    ---------
    ra : np.ndarray[nra, ]
        Transit right ascension.
    dec : np.ndarray[ndec, ]
        Transit declination.
    submap : np.ndarray[..., nra, ndec]
        Region of the ringmap around the point source.
    rms : np.ndarray[..., nra]
        RMS error on the map.
    flag : np.ndarray[..., nra, ndec]
        Boolean array that indicates which pixels to fit.
    dirty_beam : np.ndarray[..., nra, ndec] or [ra, dec, dirty_beam]
        Fourier transform of the weighting function used to create
        the map.  If input, then the interpolated dirty beam will be used
        as the model for the point source response in the declination direction.
        Can either be an array that is the same size as submap, or a list/tuple
        of length 3 that contains [ra, dec, dirty_beam] since the shape of the
        dirty beam is likely to be larger than the shape of the subregion of the
        map, at least in the declination direction.

    Returns
    -------
    param_name : np.ndarray[nparam, ]
        Names of the parameters.
    param : np.ndarray[..., nparam]
        Best-fit parameters for each item.
    param_cov: np.ndarray[..., nparam, nparam]
        Parameter covariance for each item.
    """

    el = _dec_to_el(dec)

    # Check if dirty beam was input
    do_dirty = (dirty_beam is not None) and ((len(dirty_beam) == 3) or
                                            (dirty_beam.shape == submap.shape))
    if do_dirty:

        if real_map:
            model = func_real_dirty_gauss
        else:
            model = func_dirty_gauss

        # Get parameter names through inspection
        param_name = inspect.getargspec(model(None)).args[1:]

        # Define dimensions of the dirty beam
        if (len(dirty_beam) != 3):
            db_ra, db_dec, db = submap.ra, submap.dec, dirty_beam
        else:
            db_ra, db_dec, db = dirty_beam

        db_el = _dec_to_el(db_dec)

        # Define dimensions of the submap
        coord = [ra, el]

    else:
        model = func_2d_gauss
        param_name = inspect.getargspec(model).args[1:]

        # Create 1d vectors that span the (ra, dec) grid
        coord = [ra, dec]

    # Extract parameter names from function
    nparam = len(param_name)

    # Examine dimensions of input data
    dims = submap.shape
    ndims = len(dims)

    # If we are performing a single fit, then we need to recast shape to allow iteration
    if ndims == 2:
        submap = submap[np.newaxis, ...]
        if do_dirty:
            db = db[np.newaxis, ...]
        if rms is not None:
            rms = rms[np.newaxis, ...]

        dims = submap.shape

    dims = dims[0:-2]

    # Create arrays to hold best-fit parameters and
    # parameter covariance.  Initialize to NaN.
    param = np.full(dims + (nparam, ), np.nan, dtype=np.float64)
    param_cov = np.full(dims + (nparam, nparam), np.nan, dtype=np.float64)
    resid_rms = np.full(dims, np.nan, dtype=np.float64)

    # Iterate over dimensions
    for index in np.ndindex(*dims):

        # Extract the RMS for this index.  In the process,
        # check for data flagged as bad (rms == 0.0).
        if rms is not None:
            good_ra = rms[index] > 0.0
            this_rms = np.tile(rms[index][good_ra, np.newaxis], [1, submap.shape[-1]]).ravel()
        else:
            good_ra = np.ones(submap.shape[-2], dtype=np.bool)
            this_rms = None

        if np.sum(good_ra) <= nparam:
            continue

        # Extract map
        this_submap = submap[index][good_ra, :].ravel()
        this_coord = [coord[0][good_ra], coord[1]]

        # Specify initial estimates of parameter and parameter boundaries
        if ra0 is None:
            ra0 = np.median(ra)
        if dec0 is None:
            dec0 = _el_to_dec(np.median(el))
        offset0 = np.median(np.nan_to_num(this_submap))
        peak0 = np.max(np.nan_to_num(this_submap))

        p0_dict = {'peak_amplitude' : peak0,
                   'centroid_x' : ra0,
                   'centroid_y' : dec0,
                   'fwhm_x' : 2.0,
                   'fwhm_y' : 2.0,
                   'offset' : offset0,
                   'fringe_rate' : 22.0 * freq * 1e6 / 3e8}

        lb_dict = {'peak_amplitude' : 0.0,
                   'centroid_x' : ra0 - 1.5,
                   'centroid_y' : dec0 - 0.75,
                   'fwhm_x' : 0.5,
                   'fwhm_y' : 0.5,
                   'offset' : offset0 - 2.0*np.abs(offset0),
                   'fringe_rate' : -200.0}

        ub_dict = {'peak_amplitude' : 1.5*peak0,
                   'centroid_x' : ra0 + 1.5,
                   'centroid_y' : dec0 + 0.75,
                   'fwhm_x' : 6.0,
                   'fwhm_y' : 6.0,
                   'offset' : offset0 + 2.0*np.abs(offset0),
                   'fringe_rate' : 200.0}

        p0 = np.array([p0_dict[key] for key in param_name])

        bounds = (np.array([lb_dict[key] for key in param_name]),
                  np.array([ub_dict[key] for key in param_name]))

        # Define model
        if do_dirty:
            fdirty = interp1d(db_el, db[index][good_ra, :], axis=-1, copy=False, kind='cubic',
                                                            bounds_error=False, fill_value=0.0)
            this_model = model(fdirty)
        else:
            this_model = model

        # Perform the fit.  If there is an error,
        # then we leave parameter values as NaN.
        try:
            popt, pcov = curve_fit(this_model, this_coord, this_submap,
                                   p0=p0, sigma=this_rms, absolute_sigma=True) #, bounds=bounds)
        except Exception as error:
            print("index %s: %s" % ('(' + ', '.join(["%d" % ii for ii in index]) + ')', error))
            continue

        # Save the results
        param[index] = popt
        param_cov[index] = pcov

        # Calculate RMS of the residuals
        resid = this_submap - this_model(this_coord, *popt)
        resid_rms[index] = 1.4826*np.median(np.abs(resid - np.median(resid)))

    # If this is a single fit, then remove singleton dimension
    if ndims == 2:
        param = param[0]
        param_cov = param_cov[0]
        resid_rms = resid_rms[0]
        submap = submap[0]
        if do_dirty:
            db = db[0]

    # Return the best-fit parameters and parameter covariance
    return param_name, param, param_cov, resid_rms


def func_2d_gauss(coord, peak_amplitude, centroid_x, centroid_y,
                         fwhm_x, fwhm_y, offset):
    """ Returns a parameteric model for the map of a point source,
    consisting of a 2-dimensional gaussian.

    Parameters
    ----------
    coord : (ra, dec)
        Tuple containing the right ascension and declination.  These should be
        coordinate vectors of length nra and ndec, respectively.
    peak_amplitude : float
        Model parameter.  Normalization of the gaussian.
    centroid_x : float
        Model parameter.  Centroid of the gaussian in degrees in the
        right ascension direction.
    centroid_y : float
        Model parameter.  Centroid of the gaussian in degrees in the
        declination direction.
    fwhm_x : float
        Model parameter.  Full width at half maximum of the gaussian
        in degrees in the right ascension direction.
    fwhm_y : float
        Model parameter.  Full width at half maximum of the gaussian
        in degrees in the declination direction.
    offset : float
        Model parameter.  Constant background value of the map.

    Returns
    -------
    model : np.ndarray[nra*ndec]
        Model prediction for the map of the point source.
    """
    x, y = coord

    model = (peak_amplitude*np.exp(-4.0*np.log(2.0)*((x[:, np.newaxis] - centroid_x)/fwhm_x)**2)*
                            np.exp(-4.0*np.log(2.0)*((y[np.newaxis, :] - centroid_y)/fwhm_y)**2)) + offset

    return model.ravel()


def func_2d_sinc_gauss(coord, peak_amplitude, centroid_x, centroid_y,
                              fwhm_x, fwhm_y, offset):
    """ Returns a parameteric model for the map of a point source,
        consisting of a sinc function along the declination direction
        and gaussian along the right ascension direction.

    Parameters
    ----------
    coord : (ra, dec)
        Tuple containing the right ascension and declination.  These should be
        coordinate vectors of length nra and ndec, respectively.
    peak_amplitude : float
        Model parameter.  Normalization of the gaussian.
    centroid_x : float
        Model parameter.  Centroid of the gaussian in degrees in the
        right ascension direction.
    centroid_y : float
        Model parameter.  Centroid of the sinc function in degrees in the
        declination direction.
    fwhm_x : float
        Model parameter.  Full width at half maximum of the gaussian
        in degrees in the right ascension direction.
    fwhm_y : float
        Model parameter.  Full width at half maximum of the sinc function
        in degrees in the declination direction.
    offset : float
        Model parameter.  Constant background value of the map.

    Returns
    -------
    model : np.ndarray[nra*ndec]
        Model prediction for the map of the point source.
    """
    x, y = coord

    model = (peak_amplitude*np.exp(-4.0*np.log(2.0)*((x[:, np.newaxis] - centroid_x)/fwhm_x)**2)*
                            np.sinc(1.2075*(y[np.newaxis, :] - centroid_y)/fwhm_y)) + offset

    return model.ravel()


def func_dirty_gauss(dirty_beam):
    """ Returns a parameteric model for the map of a point source,
    consisting of the interpolated dirty beam along the y-axis
    and a gaussian along the x-axis.

    This function is a wrapper that defines the interpolated
    dirty beam.

    Parameter
    ---------
    dirty_beam : scipy.interpolate.interp1d
        Interpolation function that takes as an argument el = sin(za)
        and outputs an np.ndarray[nel, nra] that represents the dirty
        beam evaluated at the same right ascension as the map.

    Returns
    -------
    dirty_gauss : np.ndarray[nra*ndec]
        Model prediction for the map of the point source.
    """

    def dirty_gauss(coord, peak_amplitude, centroid_x, centroid_y, fwhm_x, offset):
        """ Returns a parameteric model for the map of a point source,
        consisting of the interpolated dirty beam along the y-axis
        and a gaussian along the x-axis.

        Parameter
        ---------
        coord : [ra, dec]
            Tuple containing the right ascension and declination.  These should be
            coordinate vectors of length nra and ndec, respectively.
        peak_amplitude : float
            Model parameter.  Normalization of the gaussian
            in the right ascension direction.
        centroid_x : float
            Model parameter.  Centroid of the gaussian in degrees in the
            right ascension direction.
        centroid_y : float
            Model parameter.  Centroid of the dirty beam in degrees in the
            declination direction.
        fwhm_x : float
            Model parameter.  Full width at half maximum of the gaussian
            in degrees in the right ascension direction.
        offset : float
            Model parameter.  Constant background value of the map.

        Returns
        -------
        model : np.ndarray[nra*ndec]
            Model prediction for the map of the point source.
        """

        x, y = coord

        model = (peak_amplitude*np.exp(-4.0*np.log(2.0)*((x[:, np.newaxis] - centroid_x)/fwhm_x)**2)*
                                dirty_beam(y - _dec_to_el(centroid_y))) + offset

        return model.ravel()

    return dirty_gauss


def func_real_dirty_gauss(dirty_beam):
    """ Returns a parameteric model for the map of a point source,
    consisting of the interpolated dirty beam along the y-axis
    and a sinusoid with gaussian envelope along the x-axis.

    This function is a wrapper that defines the interpolated
    dirty beam.

    Parameter
    ---------
    dirty_beam : scipy.interpolate.interp1d
        Interpolation function that takes as an argument el = sin(za)
        and outputs an np.ndarray[nel, nra] that represents the dirty
        beam evaluated at the same right ascension as the map.

    Returns
    -------
    real_dirty_gauss : np.ndarray[nra*ndec]
        Model prediction for the map of the point source.
    """

    def real_dirty_gauss(coord, peak_amplitude, centroid_x, centroid_y, fwhm_x, offset, fringe_rate):
        """ Returns a parameteric model for the map of a point source,
        consisting of the interpolated dirty beam along the y-axis
        and a sinusoid with gaussian envelope along the x-axis.

        Parameter
        ---------
        coord : [ra, dec]
            Tuple containing the right ascension and declination, each
            of which is coordinate vectors of length nra and ndec, respectively.
        peak_amplitude : float
            Model parameter.  Normalization of the gaussian
            in the right ascension direction.
        centroid_x : float
            Model parameter.  Centroid of the gaussian in degrees in the
            right ascension direction.
        centroid_y : float
            Model parameter.  Centroid of the dirty beam in degrees in the
            declination direction.
        fwhm_x : float
            Model parameter.  Full width at half maximum of the gaussian
            in degrees in the right ascension direction.
        offset : float
            Model parameter.  Constant background value of the map.
        fringe_rate : float
            Model parameter.  Frequency of the sinusoid.

        Returns
        -------
        model : np.ndarray[nra*ndec]
            Model prediction for the map of the point source.
        """

        x, y = coord

        model = (peak_amplitude*np.exp(-4.0*np.log(2.0)*((x[:, np.newaxis] - centroid_x)/fwhm_x)**2)*
                                dirty_beam(y - _dec_to_el(centroid_y))) + offset

        phase = np.exp(2.0J * np.pi * np.cos(np.radians(centroid_y)) *
                       np.sin(-np.radians(x - centroid_x)) *
                       fringe_rate)

        return (model*phase[:, np.newaxis]).real.ravel()

    return real_dirty_gauss


def guess_fwhm(freq, pol='X', dec=None, sigma=False):
    """ This function provides a rough estimate of the FWHM
    of the primary antenna beam of a CHIME feed for a
    given frequency and polarization.

    It uses a linear fit to the median FWHM(nu) over all
    feeds of a given polarization for CygA transits.
    CasA and TauA transits also showed good agreement
    with this relationship.

    Parameters
    ----------
    freq : float, list of floats
        Frequency in MHz.
    pol : string or bool
        Polarization, can be 'X'/'E' or 'Y'/'S'
    dec : float
        Declination of the source in radians.  If this quantity
        is input, then the FWHM is divided by cos(dec) to account
        for the increased rate at which a source rotates across
        the sky.  Default is do not correct for this effect.
    sigma : bool
        Return the standard deviation instead of the FWHM.
        Default is to return the FWHM.

    Returns
    -------
    fwhm : float, list of floats
        Rough estimate of the FWHM (or standard deviation if sigma=True).
    """

    # Define linear coefficients based on polarization
    if (pol == 'Y') or (pol == 'S'):
        slope = -0.0025954
        offset = 3.0311712
    else:
        slope = -0.0039784
        offset = 4.3951982

    # Estimate standard deviation
    sig = offset + slope*freq

    # Divide by declination to convert to degrees hour angle
    if dec is not None:
        sig /= np.cos(dec)

    # If requested return standard deviation, otherwise return fwhm
    if sigma:
        return sig
    else:
        return 2.35482*sig


def _el_to_dec(el):
    """ Convert from el = sin(zenith angle) to declination in degrees.
    """

    return np.degrees(np.arcsin(el)) + ephemeris.CHIMELATITUDE

def _dec_to_el(dec):
    """ Convert from declination in degrees to el = sin(zenith angle).
    """

    return np.sin(np.radians(dec - ephemeris.CHIMELATITUDE))