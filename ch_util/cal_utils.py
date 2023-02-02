"""
Tools for point source calibration

This module contains tools for performing point-source calibration.
"""

from abc import ABCMeta, abstractmethod
from datetime import datetime
import inspect
import logging
from typing import Dict, Optional, Union

import numpy as np
import scipy.stats
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.linalg import lstsq, inv

from caput import memh5, time as ctime
from chimedb import dataset as ds
from chimedb.dataset.utils import state_id_of_type, unique_unmasked_entry
from ch_util import ephemeris, tools

# Set up logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class FitTransit(object, metaclass=ABCMeta):
    """Base class for fitting models to point source transits.

    The `fit` method should be used to populate the `param`, `param_cov`, `chisq`,
    and `ndof` attributes.  The `predict` and `uncertainty` methods can then be used
    to obtain the model prediction for the response and uncertainty on this quantity
    at a given hour angle.

    Attributes
    ----------
    param : np.ndarray[..., nparam]
        Best-fit parameters.
    param_cov : np.ndarray[..., nparam, nparam]
        Covariance of the fit parameters.
    chisq : np.ndarray[...]
        Chi-squared of the fit.
    ndof : np.ndarray[...]
        Number of degrees of freedom.

    Abstract Methods
    ----------------
    Any subclass of FitTransit must define these methods:
        peak
        _fit
        _model
        _jacobian
    """

    _tval = {}
    component = np.array(["complex"], dtype=np.string_)

    def __init__(self, *args, **kwargs):
        """Instantiates a FitTransit object.

        Parameters
        ----------
        param : np.ndarray[..., nparam]
            Best-fit parameters.
        param_cov : np.ndarray[..., nparam, nparam]
            Covariance of the fit parameters.
        chisq : np.ndarray[..., ncomponent]
            Chi-squared.
        ndof : np.ndarray[..., ncomponent]
            Number of degrees of freedom.
        """
        # Save keyword arguments as attributes
        self.param = kwargs.pop("param", None)
        self.param_cov = kwargs.pop("param_cov", None)
        self.chisq = kwargs.pop("chisq", None)
        self.ndof = kwargs.pop("ndof", None)
        self.model_kwargs = kwargs

    def predict(self, ha, elementwise=False):
        """Predict the point source response.

        Parameters
        ----------
        ha : np.ndarray[nha,] or float
            The hour angle in degrees.
        elementwise : bool
            If False, then the model will be evaluated at the
            requested hour angles for every set of parameters.
            If True, then the model will be evaluated at a
            separate hour angle for each set of parameters
            (requires `ha.shape == self.N`).

        Returns
        -------
        model : np.ndarray[..., nha] or float
            Model for the point source response at the requested
            hour angles.  Complex valued.
        """
        with np.errstate(all="ignore"):
            mdl = self._model(ha, elementwise=elementwise)
        return np.where(np.isfinite(mdl), mdl, 0.0 + 0.0j)

    def uncertainty(self, ha, alpha=0.32, elementwise=False):
        """Predict the uncertainty on the point source response.

        Parameters
        ----------
        ha : np.ndarray[nha,] or float
            The hour angle in degrees.
        alpha : float
            Confidence level given by 1 - alpha.
        elementwise : bool
            If False, then the uncertainty will be evaluated at
            the requested hour angles for every set of parameters.
            If True, then the uncertainty will be evaluated at a
            separate hour angle for each set of parameters
            (requires `ha.shape == self.N`).

        Returns
        -------
        err : np.ndarray[..., nha]
            Uncertainty on the point source response at the
            requested hour angles.
        """
        x = np.atleast_1d(ha)
        with np.errstate(all="ignore"):
            err = _propagate_uncertainty(
                self._jacobian(x, elementwise=elementwise),
                self.param_cov,
                self.tval(alpha, self.ndof),
            )
        return np.squeeze(np.where(np.isfinite(err), err, 0.0))

    def fit(self, ha, resp, resp_err, width=5, absolute_sigma=False, **kwargs):
        """Apply subclass defined `_fit` method to multiple transits.

        This function can be used to fit the transit for multiple inputs
        and frequencies.  Populates the `param`, `param_cov`, `chisq`, and `ndof`
        attributes.

        Parameters
        ----------
        ha : np.ndarray[nha,]
            Hour angle in degrees.
        resp : np.ndarray[..., nha]
            Measured response to the point source.  Complex valued.
        resp_err : np.ndarray[..., nha]
            Error on the measured response.
        width : np.ndarray[...]
            Initial guess at the width (sigma) of the transit in degrees.
        absolute_sigma : bool
            Set to True if the errors provided are absolute.  Set to False if
            the errors provided are relative, in which case the parameter covariance
            will be scaled by the chi-squared per degree-of-freedom.
        """
        shp = resp.shape[:-1]
        dtype = ha.dtype

        if not np.isscalar(width) and (width.shape != shp):
            ValueError("Keyword with must be scalar or have shape %s." % str(shp))

        self.param = np.full(shp + (self.nparam,), np.nan, dtype=dtype)
        self.param_cov = np.full(shp + (self.nparam, self.nparam), np.nan, dtype=dtype)
        self.chisq = np.full(shp + (self.ncomponent,), np.nan, dtype=dtype)
        self.ndof = np.full(shp + (self.ncomponent,), 0, dtype=int)

        with np.errstate(all="ignore"):
            for ind in np.ndindex(*shp):
                wi = width if np.isscalar(width) else width[ind[: width.ndim]]

                err = resp_err[ind]
                good = np.flatnonzero(err > 0.0)

                if (good.size // 2) <= self.nparam:
                    continue

                try:
                    param, param_cov, chisq, ndof = self._fit(
                        ha[good],
                        resp[ind][good],
                        err[good],
                        width=wi,
                        absolute_sigma=absolute_sigma,
                        **kwargs,
                    )
                except Exception as error:
                    logger.debug("Index %s failed with error: %s" % (str(ind), error))
                    continue

                self.param[ind] = param
                self.param_cov[ind] = param_cov
                self.chisq[ind] = chisq
                self.ndof[ind] = ndof

    @property
    def parameter_names(self):
        """
        Array of strings containing the name of the fit parameters.

        Returns
        -------
        parameter_names : np.ndarray[nparam,]
            Names of the parameters.
        """
        return np.array(["param%d" % p for p in range(self.nparam)], dtype=np.string_)

    @property
    def param_corr(self):
        """
        Parameter correlation matrix.

        Returns
        -------
        param_corr : np.ndarray[..., nparam, nparam]
            Correlation of the fit parameters.
        """
        idiag = tools.invert_no_zero(
            np.sqrt(np.diagonal(self.param_cov, axis1=-2, axis2=-1))
        )
        return self.param_cov * idiag[..., np.newaxis, :] * idiag[..., np.newaxis]

    @property
    def N(self):
        """
        Number of independent transit fits contained in this object.

        Returns
        -------
        N : tuple
            Numpy-style shape indicating the number of
            fits that the object contains.  Is None
            if the object contains a single fit.
        """
        if self.param is not None:
            return self.param.shape[:-1] or None

    @property
    def nparam(self):
        """
        Number of parameters.

        Returns
        -------
        nparam :  int
            Number of fit parameters.
        """
        return self.param.shape[-1]

    @property
    def ncomponent(self):
        """
        Number of components.

        Returns
        -------
        ncomponent : int
            Number of components (i.e, real and imag, amp and phase, complex) that have been fit.
        """
        return self.component.size

    def __getitem__(self, val):
        """Instantiates a new TransitFit object containing some subset of the fits."""

        if self.N is None:
            raise KeyError(
                "Attempting to slice TransitFit object containing single fit."
            )

        return self.__class__(
            param=self.param[val],
            param_cov=self.param_cov[val],
            ndof=self.ndof[val],
            chisq=self.chisq[val],
            **self.model_kwargs,
        )

    @abstractmethod
    def peak(self):
        """Calculate the peak of the transit.

        Any subclass of FitTransit must define this method.
        """
        return

    @abstractmethod
    def _fit(self, ha, resp, resp_err, width=None, absolute_sigma=False):
        """Fit data to the model.

        Any subclass of FitTransit must define this method.

        Parameters
        ----------
        ha : np.ndarray[nha,]
            Hour angle in degrees.
        resp : np.ndarray[nha,]
            Measured response to the point source.  Complex valued.
        resp_err : np.ndarray[nha,]
            Error on the measured response.
        width : np.ndarray
            Initial guess at the width (sigma) of the transit in degrees.
        absolute_sigma : bool
            Set to True if the errors provided are absolute.  Set to False if
            the errors provided are relative, in which case the parameter covariance
            will be scaled by the chi-squared per degree-of-freedom.

        Returns
        -------
        param : np.ndarray[nparam,]
            Best-fit model parameters.
        param_cov : np.ndarray[nparam, nparam]
            Covariance of the best-fit model parameters.
        chisq : float
            Chi-squared of the fit.
        ndof : int
            Number of degrees of freedom of the fit.
        """
        return

    @abstractmethod
    def _model(self, ha):
        """Calculate the model for the point source response.

        Any subclass of FitTransit must define this method.

        Parameters
        ----------
        ha : np.ndarray
            Hour angle in degrees.
        """
        return

    @abstractmethod
    def _jacobian(self, ha):
        """Calculate the jacobian of the model for the point source response.

        Any subclass of FitTransit must define this method.

        Parameters
        ----------
        ha : np.ndarray
            Hour angle in degrees.

        Returns
        -------
        jac : np.ndarray[..., nparam, nha]
            The jacobian defined as
            jac[..., i, j] = d(model(ha)) / d(param[i]) evaluated at ha[j]
        """
        return

    @classmethod
    def tval(cls, alpha, ndof):
        """Quantile of a standardized Student's t random variable.

        This quantity is slow to compute.  Past values will be cached
        in a dictionary shared by all instances of the class.

        Parameters
        ----------
        alpha : float
            Calculate the quantile corresponding to the lower tail probability
            1 - alpha / 2.
        ndof : np.ndarray or int
            Number of degrees of freedom of the Student's t variable.

        Returns
        -------
        tval : np.ndarray or float
            Quantile of a standardized Student's t random variable.
        """
        prob = 1.0 - 0.5 * alpha

        arr_ndof = np.atleast_1d(ndof)
        tval = np.zeros(arr_ndof.shape, dtype=np.float32)

        for ind, nd in np.ndenumerate(arr_ndof):
            key = (int(100.0 * prob), nd)
            if key not in cls._tval:
                cls._tval[key] = scipy.stats.t.ppf(prob, nd)
            tval[ind] = cls._tval[key]

        if np.isscalar(ndof):
            tval = np.squeeze(tval)

        return tval


class FitPoly(FitTransit):
    """Base class for fitting polynomials to point source transits.

    Maps methods of np.polynomial to methods of the class for the
    requested polynomial type.
    """

    def __init__(self, poly_type="standard", *args, **kwargs):
        """Instantiates a FitPoly object.

        Parameters
        ----------
        poly_type : str
            Type of polynomial.  Can be 'standard', 'hermite', or 'chebyshev'.
        """
        super(FitPoly, self).__init__(poly_type=poly_type, *args, **kwargs)

        self._set_polynomial_model(poly_type)

    def _set_polynomial_model(self, poly_type):
        """Map methods of np.polynomial to methods of the class."""
        if poly_type == "standard":
            self._vander = np.polynomial.polynomial.polyvander
            self._eval = np.polynomial.polynomial.polyval
            self._deriv = np.polynomial.polynomial.polyder
            self._root = np.polynomial.polynomial.polyroots
        elif poly_type == "hermite":
            self._vander = np.polynomial.hermite.hermvander
            self._eval = np.polynomial.hermite.hermval
            self._deriv = np.polynomial.hermite.hermder
            self._root = np.polynomial.hermite.hermroots
        elif poly_type == "chebyshev":
            self._vander = np.polynomial.chebyshev.chebvander
            self._eval = np.polynomial.chebyshev.chebval
            self._deriv = np.polynomial.chebyshev.chebder
            self._root = np.polynomial.chebyshev.chebroots
        else:
            raise ValueError(
                "Do not recognize polynomial type %s."
                "Options are 'standard', 'hermite', or 'chebyshev'." % poly_type
            )

        self.poly_type = poly_type

    def _fast_eval(self, ha, param=None, elementwise=False):
        """Evaluate the polynomial at the requested hour angle."""
        if param is None:
            param = self.param

        vander = self._vander(ha, param.shape[-1] - 1)

        if elementwise:
            out = np.sum(vander * param, axis=-1)
        elif param.ndim == 1:
            out = np.dot(vander, param)
        else:
            out = np.matmul(param, np.rollaxis(vander, -1))

        return np.squeeze(out, axis=-1) if np.isscalar(ha) else out


class FitRealImag(FitTransit):
    """Base class for fitting models to the real and imag component.

    Assumes an independent fit to real and imaginary, and provides
    methods for predicting the uncertainty on each.
    """

    component = np.array(["real", "imag"], dtype=np.string_)

    def uncertainty_real(self, ha, alpha=0.32, elementwise=False):
        """Predicts the uncertainty on real component at given hour angle(s).

        Parameters
        ----------
        ha : np.ndarray[nha,] or float
            Hour angle in degrees.
        alpha : float
            Confidence level given by 1 - alpha.

        Returns
        -------
        err : np.ndarray[..., nha] or float
            Uncertainty on the real component.
        """
        x = np.atleast_1d(ha)
        err = _propagate_uncertainty(
            self._jacobian_real(x, elementwise=elementwise),
            self.param_cov[..., : self.nparr, : self.nparr],
            self.tval(alpha, self.ndofr),
        )
        return np.squeeze(err, axis=-1) if np.isscalar(ha) else err

    def uncertainty_imag(self, ha, alpha=0.32, elementwise=False):
        """Predicts the uncertainty on imag component at given hour angle(s).

        Parameters
        ----------
        ha : np.ndarray[nha,] or float
            Hour angle in degrees.
        alpha : float
            Confidence level given by 1 - alpha.

        Returns
        -------
        err : np.ndarray[..., nha] or float
            Uncertainty on the imag component.
        """
        x = np.atleast_1d(ha)
        err = _propagate_uncertainty(
            self._jacobian_imag(x, elementwise=elementwise),
            self.param_cov[..., self.nparr :, self.nparr :],
            self.tval(alpha, self.ndofi),
        )
        return np.squeeze(err, axis=-1) if np.isscalar(ha) else err

    def uncertainty(self, ha, alpha=0.32, elementwise=False):
        """Predicts the uncertainty on the response at given hour angle(s).

        Returns the quadrature sum of the real and imag uncertainty.

        Parameters
        ----------
        ha : np.ndarray[nha,] or float
            Hour angle in degrees.
        alpha : float
            Confidence level given by 1 - alpha.

        Returns
        -------
        err : np.ndarray[..., nha] or float
            Uncertainty on the response.
        """
        with np.errstate(all="ignore"):
            err = np.sqrt(
                self.uncertainty_real(ha, alpha=alpha, elementwise=elementwise) ** 2
                + self.uncertainty_imag(ha, alpha=alpha, elementwise=elementwise) ** 2
            )
        return err

    def _jacobian(self, ha):
        raise NotImplementedError(
            "Fits to real and imaginary are independent.  "
            "Use _jacobian_real and _jacobian_imag instead."
        )

    @abstractmethod
    def _jacobian_real(self, ha):
        """Calculate the jacobian of the model for the real component."""
        return

    @abstractmethod
    def _jacobian_imag(self, ha):
        """Calculate the jacobian of the model for the imag component."""
        return

    @property
    def nparam(self):
        return self.nparr + self.npari


class FitPolyRealPolyImag(FitPoly, FitRealImag):
    """Class that enables separate fits of a polynomial to real and imag components.

    Used to fit cross-polar response that is not well-described by the
    FitPolyLogAmpPolyPhase used for co-polar response.
    """

    def __init__(self, poly_deg=5, even=False, odd=False, *args, **kwargs):
        """Instantiates a FitPolyRealPolyImag object.

        Parameters
        ----------
        poly_deg : int
            Degree of the polynomial to fit to real and imaginary component.
        """
        if even and odd:
            raise RuntimeError("Cannot request both even AND odd.")

        super(FitPolyRealPolyImag, self).__init__(
            poly_deg=poly_deg, even=even, odd=odd, *args, **kwargs
        )

        self.poly_deg = poly_deg
        self.even = even
        self.odd = odd

        ind = np.arange(self.poly_deg + 1)
        if self.even:
            self.coeff_index = np.flatnonzero((ind == 0) | ~(ind % 2))

        elif self.odd:
            self.coeff_index = np.flatnonzero((ind == 0) | (ind % 2))

        else:
            self.coeff_index = ind

        self.nparr = self.coeff_index.size
        self.npari = self.nparr

    def vander(self, ha, *args):
        """Create the Vandermonde matrix."""
        A = self._vander(ha, self.poly_deg)
        return A[:, self.coeff_index]

    def deriv(self, ha, param=None):
        """Calculate the derivative of the transit."""
        if param is None:
            param = self.param

        is_scalar = np.isscalar(ha)
        ha = np.atleast_1d(ha)

        shp = param.shape[:-1]

        param_expanded_real = np.zeros(shp + (self.poly_deg + 1,), dtype=param.dtype)
        param_expanded_real[..., self.coeff_index] = param[..., : self.nparr]
        der1_real = self._deriv(param_expanded_real, m=1, axis=-1)

        param_expanded_imag = np.zeros(shp + (self.poly_deg + 1,), dtype=param.dtype)
        param_expanded_imag[..., self.coeff_index] = param[..., self.nparr :]
        der1_imag = self._deriv(param_expanded_imag, m=1, axis=-1)

        deriv = np.full(shp + (ha.size,), np.nan, dtype=np.complex64)
        for ind in np.ndindex(*shp):
            ider1_real = der1_real[ind]
            ider1_imag = der1_imag[ind]

            if np.any(~np.isfinite(ider1_real)) or np.any(~np.isfinite(ider1_imag)):
                continue

            deriv[ind] = self._eval(ha, ider1_real) + 1.0j * self._eval(ha, ider1_imag)

        return np.squeeze(deriv, axis=-1) if is_scalar else deriv

    def _fit(self, ha, resp, resp_err, absolute_sigma=False):
        """Fit polynomial to real and imaginary component.

        Use weighted least squares.

        Parameters
        ----------
        ha : np.ndarray[nha,]
            Hour angle in degrees.
        resp : np.ndarray[nha,]
            Measured response to the point source.  Complex valued.
        resp_err : np.ndarray[nha,]
            Error on the measured response.
        absolute_sigma : bool
            Set to True if the errors provided are absolute.  Set to False if
            the errors provided are relative, in which case the parameter covariance
            will be scaled by the chi-squared per degree-of-freedom.

        Returns
        -------
        param : np.ndarray[nparam,]
            Best-fit model parameters.
        param_cov : np.ndarray[nparam, nparam]
            Covariance of the best-fit model parameters.
        chisq : np.ndarray[2,]
            Chi-squared of the fit to amplitude and phase.
        ndof : np.ndarray[2,]
            Number of degrees of freedom of the fit to amplitude and phase.
        """
        min_nfit = min(self.nparr, self.npari) + 1

        # Prepare amplitude data
        amp = np.abs(resp)
        w0 = tools.invert_no_zero(resp_err) ** 2

        # Only perform fit if there is enough data.
        this_flag = (amp > 0.0) & (w0 > 0.0)
        ndata = int(np.sum(this_flag))
        if ndata < min_nfit:
            raise RuntimeError("Number of data points less than number of parameters.")

        wf = w0 * this_flag.astype(np.float32)

        # Compute real and imaginary component of complex response
        yr = np.real(resp)
        yi = np.imag(resp)

        # Calculate vandermonde matrix
        A = self.vander(ha)

        # Compute parameter covariance
        cov = inv(np.dot(A.T, wf[:, np.newaxis] * A))

        # Compute best-fit coefficients
        coeffr = np.dot(cov, np.dot(A.T, wf * yr))
        coeffi = np.dot(cov, np.dot(A.T, wf * yi))

        # Compute model estimate
        mr = np.dot(A, coeffr)
        mi = np.dot(A, coeffi)

        # Compute chisq per degree of freedom
        ndofr = ndata - self.nparr
        ndofi = ndata - self.npari

        ndof = np.array([ndofr, ndofi])
        chisq = np.array([np.sum(wf * (yr - mr) ** 2), np.sum(wf * (yi - mi) ** 2)])

        # Scale the parameter covariance by chisq per degree of freedom.
        # Equivalent to using RMS of the residuals to set the absolute error
        # on the measurements.
        if not absolute_sigma:
            scale_factor = chisq * tools.invert_no_zero(ndof.astype(np.float32))
            covr = cov * scale_factor[0]
            covi = cov * scale_factor[1]
        else:
            covr = cov
            covi = cov

        param = np.concatenate((coeffr, coeffi))

        param_cov = np.zeros((self.nparam, self.nparam), dtype=np.float32)
        param_cov[: self.nparr, : self.nparr] = covr
        param_cov[self.nparr :, self.nparr :] = covi

        return param, param_cov, chisq, ndof

    def _model(self, ha, elementwise=False):
        real = self._fast_eval(
            ha, self.param[..., : self.nparr], elementwise=elementwise
        )
        imag = self._fast_eval(
            ha, self.param[..., self.nparr :], elementwise=elementwise
        )

        return real + 1.0j * imag

    def _jacobian_real(self, ha, elementwise=False):
        jac = np.rollaxis(self.vander(ha), -1)
        if not elementwise and self.N is not None:
            slc = (None,) * len(self.N)
            jac = jac[slc]

        return jac

    def _jacobian_imag(self, ha, elementwise=False):
        jac = np.rollaxis(self.vander(ha), -1)
        if not elementwise and self.N is not None:
            slc = (None,) * len(self.N)
            jac = jac[slc]

        return jac

    @property
    def ndofr(self):
        """Number of degrees of freedom for the real fit."""
        return self.ndof[..., 0]

    @property
    def ndofi(self):
        """Number of degrees of freedom for the imag fit."""
        return self.ndof[..., 1]

    @property
    def parameter_names(self):
        """Array of strings containing the name of the fit parameters."""
        return np.array(
            ["%s_poly_real_coeff%d" % (self.poly_type, p) for p in range(self.nparr)]
            + ["%s_poly_imag_coeff%d" % (self.poly_type, p) for p in range(self.npari)],
            dtype=np.string_,
        )

    def peak(self):
        """Calculate the peak of the transit."""
        logger.warning("The peak is not defined for this model.")
        return


class FitAmpPhase(FitTransit):
    """Base class for fitting models to the amplitude and phase.

    Assumes an independent fit to amplitude and phase, and provides
    methods for predicting the uncertainty on each.
    """

    component = np.array(["amplitude", "phase"], dtype=np.string_)

    def uncertainty_amp(self, ha, alpha=0.32, elementwise=False):
        """Predicts the uncertainty on amplitude at given hour angle(s).

        Parameters
        ----------
        ha : np.ndarray[nha,] or float
            Hour angle in degrees.
        alpha : float
            Confidence level given by 1 - alpha.

        Returns
        -------
        err : np.ndarray[..., nha] or float
            Uncertainty on the amplitude in fractional units.
        """
        x = np.atleast_1d(ha)
        err = _propagate_uncertainty(
            self._jacobian_amp(x, elementwise=elementwise),
            self.param_cov[..., : self.npara, : self.npara],
            self.tval(alpha, self.ndofa),
        )
        return np.squeeze(err, axis=-1) if np.isscalar(ha) else err

    def uncertainty_phi(self, ha, alpha=0.32, elementwise=False):
        """Predicts the uncertainty on phase at given hour angle(s).

        Parameters
        ----------
        ha : np.ndarray[nha,] or float
            Hour angle in degrees.
        alpha : float
            Confidence level given by 1 - alpha.

        Returns
        -------
        err : np.ndarray[..., nha] or float
            Uncertainty on the phase in radians.
        """
        x = np.atleast_1d(ha)
        err = _propagate_uncertainty(
            self._jacobian_phi(x, elementwise=elementwise),
            self.param_cov[..., self.npara :, self.npara :],
            self.tval(alpha, self.ndofp),
        )
        return np.squeeze(err, axis=-1) if np.isscalar(ha) else err

    def uncertainty(self, ha, alpha=0.32, elementwise=False):
        """Predicts the uncertainty on the response at given hour angle(s).

        Returns the quadrature sum of the amplitude and phase uncertainty.

        Parameters
        ----------
        ha : np.ndarray[nha,] or float
            Hour angle in degrees.
        alpha : float
            Confidence level given by 1 - alpha.

        Returns
        -------
        err : np.ndarray[..., nha] or float
            Uncertainty on the response.
        """
        with np.errstate(all="ignore"):
            err = np.abs(self._model(ha, elementwise=elementwise)) * np.sqrt(
                self.uncertainty_amp(ha, alpha=alpha, elementwise=elementwise) ** 2
                + self.uncertainty_phi(ha, alpha=alpha, elementwise=elementwise) ** 2
            )
        return err

    def _jacobian(self, ha):
        raise NotImplementedError(
            "Fits to amplitude and phase are independent.  "
            "Use _jacobian_amp and _jacobian_phi instead."
        )

    @abstractmethod
    def _jacobian_amp(self, ha):
        """Calculate the jacobian of the model for the amplitude."""
        return

    @abstractmethod
    def _jacobian_phi(self, ha):
        """Calculate the jacobian of the model for the phase."""
        return

    @property
    def nparam(self):
        return self.npara + self.nparp


class FitPolyLogAmpPolyPhase(FitPoly, FitAmpPhase):
    """Class that enables separate fits of a polynomial to log amplitude and phase."""

    def __init__(self, poly_deg_amp=5, poly_deg_phi=5, *args, **kwargs):
        """Instantiates a FitPolyLogAmpPolyPhase object.

        Parameters
        ----------
        poly_deg_amp : int
            Degree of the polynomial to fit to log amplitude.
        poly_deg_phi : int
            Degree of the polynomial to fit to phase.
        """
        super(FitPolyLogAmpPolyPhase, self).__init__(
            poly_deg_amp=poly_deg_amp, poly_deg_phi=poly_deg_phi, *args, **kwargs
        )

        self.poly_deg_amp = poly_deg_amp
        self.poly_deg_phi = poly_deg_phi

        self.npara = poly_deg_amp + 1
        self.nparp = poly_deg_phi + 1

    def _fit(
        self,
        ha,
        resp,
        resp_err,
        width=None,
        absolute_sigma=False,
        moving_window=0.3,
        niter=5,
    ):
        """Fit polynomial to log amplitude and polynomial to phase.

        Use weighted least squares.  The initial errors on log amplitude
        are set to `resp_err / abs(resp)`.  If the niter parameter is greater than 1,
        then those errors will be updated with `resp_err / model_amp`, where `model_amp`
        is the best-fit model for the amplitude from the previous iteration.  The errors
        on the phase are set to `resp_err / model_amp` where `model_amp` is the best-fit
        model for the amplitude from the log amplitude fit.

        Parameters
        ----------
        ha : np.ndarray[nha,]
            Hour angle in degrees.
        resp : np.ndarray[nha,]
            Measured response to the point source.  Complex valued.
        resp_err : np.ndarray[nha,]
            Error on the measured response.
        width : float
             Initial guess at the width (sigma) of the transit in degrees.
        absolute_sigma : bool
            Set to True if the errors provided are absolute.  Set to False if
            the errors provided are relative, in which case the parameter covariance
            will be scaled by the chi-squared per degree-of-freedom.
        niter : int
            Number of iterations for the log amplitude fit.
        moving_window : float
            Only fit hour angles within +/- window * width from the peak.
            Note that the peak location is updated with each iteration.
            Set to None to fit all hour angles where resp_err > 0.0.

        Returns
        -------
        param : np.ndarray[nparam,]
            Best-fit model parameters.
        param_cov : np.ndarray[nparam, nparam]
            Covariance of the best-fit model parameters.
        chisq : np.ndarray[2,]
            Chi-squared of the fit to amplitude and phase.
        ndof : np.ndarray[2,]
            Number of degrees of freedom of the fit to amplitude and phase.
        """
        min_nfit = min(self.npara, self.nparp) + 1

        window = width * moving_window if (width and moving_window) else None

        # Prepare amplitude data
        model_amp = np.abs(resp)
        w0 = tools.invert_no_zero(resp_err) ** 2

        # Only perform fit if there is enough data.
        this_flag = (model_amp > 0.0) & (w0 > 0.0)
        ndata = int(np.sum(this_flag))
        if ndata < min_nfit:
            raise RuntimeError("Number of data points less than number of parameters.")

        # Prepare amplitude data
        ya = np.log(model_amp)

        # Prepare phase data.
        phi = np.angle(resp)
        phi0 = phi[np.argmin(np.abs(ha))]

        yp = phi - phi0
        yp += (yp < -np.pi) * 2 * np.pi - (yp > np.pi) * 2 * np.pi
        yp += phi0

        # Calculate vandermonde matrix
        A = self._vander(ha, self.poly_deg_amp)
        center = 0.0

        # Iterate to obtain model estimate for amplitude
        for kk in range(niter):
            wk = w0 * model_amp**2

            if window is not None:
                if kk > 0:
                    center = self.peak(param=coeff)

                if np.isnan(center):
                    raise RuntimeError("No peak found.")

                wk *= (np.abs(ha - center) <= window).astype(np.float64)

                ndata = int(np.sum(wk > 0.0))
                if ndata < min_nfit:
                    raise RuntimeError(
                        "Number of data points less than number of parameters."
                    )

            C = np.dot(A.T, wk[:, np.newaxis] * A)
            coeff = lstsq(C, np.dot(A.T, wk * ya))[0]

            model_amp = np.exp(np.dot(A, coeff))

        # Compute final value for amplitude
        center = self.peak(param=coeff)

        if np.isnan(center):
            raise RuntimeError("No peak found.")

        wf = w0 * model_amp**2
        if window is not None:
            wf *= (np.abs(ha - center) <= window).astype(np.float64)

            ndata = int(np.sum(wf > 0.0))
            if ndata < min_nfit:
                raise RuntimeError(
                    "Number of data points less than number of parameters."
                )

        cova = inv(np.dot(A.T, wf[:, np.newaxis] * A))
        coeffa = np.dot(cova, np.dot(A.T, wf * ya))

        mamp = np.dot(A, coeffa)

        # Compute final value for phase
        A = self._vander(ha, self.poly_deg_phi)

        covp = inv(np.dot(A.T, wf[:, np.newaxis] * A))
        coeffp = np.dot(covp, np.dot(A.T, wf * yp))

        mphi = np.dot(A, coeffp)

        # Compute chisq per degree of freedom
        ndofa = ndata - self.npara
        ndofp = ndata - self.nparp

        ndof = np.array([ndofa, ndofp])
        chisq = np.array([np.sum(wf * (ya - mamp) ** 2), np.sum(wf * (yp - mphi) ** 2)])

        # Scale the parameter covariance by chisq per degree of freedom.
        # Equivalent to using RMS of the residuals to set the absolute error
        # on the measurements.
        if not absolute_sigma:
            scale_factor = chisq * tools.invert_no_zero(ndof.astype(np.float32))
            cova *= scale_factor[0]
            covp *= scale_factor[1]

        param = np.concatenate((coeffa, coeffp))

        param_cov = np.zeros((self.nparam, self.nparam), dtype=np.float32)
        param_cov[: self.npara, : self.npara] = cova
        param_cov[self.npara :, self.npara :] = covp

        return param, param_cov, chisq, ndof

    def peak(self, param=None):
        """Find the peak of the transit.

        Parameters
        ----------
        param : np.ndarray[..., nparam]
            Coefficients of the polynomial model for log amplitude.
            Defaults to `self.param`.

        Returns
        -------
        peak : np.ndarray[...]
            Location of the maximum amplitude in degrees hour angle.
            If the polynomial does not have a maximum, then NaN is returned.
        """
        if param is None:
            param = self.param

        der1 = self._deriv(param[..., : self.npara], m=1, axis=-1)
        der2 = self._deriv(param[..., : self.npara], m=2, axis=-1)

        shp = der1.shape[:-1]
        peak = np.full(shp, np.nan, dtype=der1.dtype)

        for ind in np.ndindex(*shp):
            ider1 = der1[ind]

            if np.any(~np.isfinite(ider1)):
                continue

            root = self._root(ider1)
            xmax = np.real(
                [
                    rr
                    for rr in root
                    if (rr.imag == 0) and (self._eval(rr, der2[ind]) < 0.0)
                ]
            )

            peak[ind] = xmax[np.argmin(np.abs(xmax))] if xmax.size > 0 else np.nan

        return peak

    def _model(self, ha, elementwise=False):
        amp = self._fast_eval(
            ha, self.param[..., : self.npara], elementwise=elementwise
        )
        phi = self._fast_eval(
            ha, self.param[..., self.npara :], elementwise=elementwise
        )

        return np.exp(amp) * (np.cos(phi) + 1.0j * np.sin(phi))

    def _jacobian_amp(self, ha, elementwise=False):
        jac = self._vander(ha, self.poly_deg_amp)
        if not elementwise:
            jac = np.rollaxis(jac, -1)
            if self.N is not None:
                slc = (None,) * len(self.N)
                jac = jac[slc]

        return jac

    def _jacobian_phi(self, ha, elementwise=False):
        jac = self._vander(ha, self.poly_deg_phi)
        if not elementwise:
            jac = np.rollaxis(jac, -1)
            if self.N is not None:
                slc = (None,) * len(self.N)
                jac = jac[slc]

        return jac

    @property
    def ndofa(self):
        """
        Number of degrees of freedom for the amplitude fit.

        Returns
        -------
        ndofa : np.ndarray[...]
            Number of degrees of freedom of the amplitude fit.
        """
        return self.ndof[..., 0]

    @property
    def ndofp(self):
        """
        Number of degrees of freedom for the phase fit.

        Returns
        -------
        ndofp : np.ndarray[...]
            Number of degrees of freedom of the phase fit.
        """
        return self.ndof[..., 1]

    @property
    def parameter_names(self):
        """Array of strings containing the name of the fit parameters."""
        return np.array(
            ["%s_poly_amp_coeff%d" % (self.poly_type, p) for p in range(self.npara)]
            + ["%s_poly_phi_coeff%d" % (self.poly_type, p) for p in range(self.nparp)],
            dtype=np.string_,
        )


class FitGaussAmpPolyPhase(FitPoly, FitAmpPhase):
    """Class that enables fits of a gaussian to amplitude and a polynomial to phase."""

    component = np.array(["complex"], dtype=np.string_)
    npara = 3

    def __init__(self, poly_deg_phi=5, *args, **kwargs):
        """Instantiates a FitGaussAmpPolyPhase object.

        Parameters
        ----------
        poly_deg_phi : int
            Degree of the polynomial to fit to phase.
        """
        super(FitGaussAmpPolyPhase, self).__init__(
            poly_deg_phi=poly_deg_phi, *args, **kwargs
        )

        self.poly_deg_phi = poly_deg_phi
        self.nparp = poly_deg_phi + 1

    def _fit(self, ha, resp, resp_err, width=5, absolute_sigma=False, param0=None):
        """Fit gaussian to amplitude and polynomial to phase.

        Uses non-linear least squares (`scipy.optimize.curve_fit`) to
        fit the model to the complex valued data.

        Parameters
        ----------
        ha : np.ndarray[nha,]
            Hour angle in degrees.
        resp : np.ndarray[nha,]
            Measured response to the point source.  Complex valued.
        resp_err : np.ndarray[nha,]
            Error on the measured response.
        width : float
             Initial guess at the width (sigma) of the transit in degrees.
        absolute_sigma : bool
            Set to True if the errors provided are absolute.  Set to False if
            the errors provided are relative, in which case the parameter covariance
            will be scaled by the chi-squared per degree-of-freedom.
        param0 : np.ndarray[nparam,]
            Initial guess at the parameters for the Levenberg-Marquardt algorithm.
            If these are not provided, then this function will make reasonable guesses.

        Returns
        -------
        param : np.ndarray[nparam,]
            Best-fit model parameters.
        param_cov : np.ndarray[nparam, nparam]
            Covariance of the best-fit model parameters.
        chisq : float
            Chi-squared of the fit.
        ndof : int
            Number of degrees of freedom of the fit.
        """
        if ha.size < (min(self.npara, self.nparp) + 1):
            raise RuntimeError("Number of data points less than number of parameters.")

        # We will fit the complex data.  Break n-element complex array y(x)
        # into 2n-element real array [Re{y(x)}, Im{y(x)}] for fit.
        x = np.tile(ha, 2)
        y = np.concatenate((resp.real, resp.imag))
        err = np.tile(resp_err, 2)

        # Initial estimate of parameter values:
        # [peak_amplitude, centroid, fwhm, phi_0, phi_1, phi_2, ...]
        if param0 is None:
            param0 = [np.max(np.nan_to_num(np.abs(resp))), 0.0, 2.355 * width]
            param0.append(np.median(np.nan_to_num(np.angle(resp, deg=True))))
            param0 += [0.0] * (self.nparp - 1)
            param0 = np.array(param0)

        # Perform the fit.
        param, param_cov = curve_fit(
            self._get_fit_func(),
            x,
            y,
            sigma=err,
            p0=param0,
            absolute_sigma=absolute_sigma,
            jac=self._get_fit_jac(),
        )

        chisq = np.sum(
            (
                np.abs(resp - self._model(ha, param=param))
                * tools.invert_no_zero(resp_err)
            )
            ** 2
        )
        ndof = y.size - self.nparam

        return param, param_cov, chisq, ndof

    def peak(self):
        """Return the peak of the transit.

        Returns
        -------
        peak : float
            Centroid of the gaussian fit to amplitude.
        """
        return self.param[..., 1]

    def _get_fit_func(self):
        """Generates a function that can be used by `curve_fit` to compute the model."""

        def fit_func(x, *param):
            """Function used by `curve_fit` to compute the model.

            Parameters
            ----------
            x : np.ndarray[2 * nha,]
                Hour angle in degrees replicated twice for the real
                and imaginary components, i.e., `x = np.concatenate((ha, ha))`.
            *param : floats
                Parameters of the model.

            Returns
            -------
            model : np.ndarray[2 * nha,]
                Model for the complex valued point source response,
                packaged as `np.concatenate((model.real, model.imag))`.
            """
            peak_amplitude, centroid, fwhm = param[:3]
            poly_coeff = param[3:]

            nreal = len(x) // 2
            xr = x[:nreal]

            dxr = _correct_phase_wrap(xr - centroid)

            model_amp = peak_amplitude * np.exp(-4.0 * np.log(2.0) * (dxr / fwhm) ** 2)
            model_phase = self._eval(xr, poly_coeff)

            model = np.concatenate(
                (model_amp * np.cos(model_phase), model_amp * np.sin(model_phase))
            )

            return model

        return fit_func

    def _get_fit_jac(self):
        """Generates a function that can be used by `curve_fit` to compute jacobian of the model."""

        def fit_jac(x, *param):
            """Function used by `curve_fit` to compute the jacobian.

            Parameters
            ----------
            x : np.ndarray[2 * nha,]
                Hour angle in degrees.  Replicated twice for the real
                and imaginary components, i.e., `x = np.concatenate((ha, ha))`.
            *param : float
                Parameters of the model.

            Returns
            -------
            jac : np.ndarray[2 * nha, nparam]
                The jacobian defined as
                jac[i, j] = d(model(ha)) / d(param[j]) evaluated at ha[i]
            """

            peak_amplitude, centroid, fwhm = param[:3]
            poly_coeff = param[3:]

            nparam = len(param)
            nx = len(x)
            nreal = nx // 2

            jac = np.empty((nx, nparam), dtype=x.dtype)

            dx = _correct_phase_wrap(x - centroid)

            dxr = dx[:nreal]
            xr = x[:nreal]

            model_amp = peak_amplitude * np.exp(-4.0 * np.log(2.0) * (dxr / fwhm) ** 2)
            model_phase = self._eval(xr, poly_coeff)
            model = np.concatenate(
                (model_amp * np.cos(model_phase), model_amp * np.sin(model_phase))
            )

            dmodel_dphase = np.concatenate((-model[nreal:], model[:nreal]))

            jac[:, 0] = tools.invert_no_zero(peak_amplitude) * model
            jac[:, 1] = 8.0 * np.log(2.0) * dx * tools.invert_no_zero(fwhm) ** 2 * model
            jac[:, 2] = (
                8.0 * np.log(2.0) * dx**2 * tools.invert_no_zero(fwhm) ** 3 * model
            )
            jac[:, 3:] = (
                self._vander(x, self.poly_deg_phi) * dmodel_dphase[:, np.newaxis]
            )

            return jac

        return fit_jac

    def _model(self, ha, param=None, elementwise=False):
        if param is None:
            param = self.param

        # Evaluate phase
        model_phase = self._fast_eval(
            ha, param[..., self.npara :], elementwise=elementwise
        )

        # Evaluate amplitude
        amp_param = param[..., : self.npara]
        ndim1 = amp_param.ndim
        if not elementwise and (ndim1 > 1) and not np.isscalar(ha):
            ndim2 = ha.ndim
            amp_param = amp_param[(slice(None),) * ndim1 + (None,) * ndim2]
            ha = ha[(None,) * (ndim1 - 1) + (slice(None),) * ndim2]

        slc = (slice(None),) * (ndim1 - 1)
        peak_amplitude = amp_param[slc + (0,)]
        centroid = amp_param[slc + (1,)]
        fwhm = amp_param[slc + (2,)]

        dha = _correct_phase_wrap(ha - centroid)

        model_amp = peak_amplitude * np.exp(-4.0 * np.log(2.0) * (dha / fwhm) ** 2)

        # Return complex valued quantity
        return model_amp * (np.cos(model_phase) + 1.0j * np.sin(model_phase))

    def _jacobian_amp(self, ha, elementwise=False):
        amp_param = self.param[..., : self.npara]

        shp = amp_param.shape
        ndim1 = amp_param.ndim

        if not elementwise:
            shp = shp + ha.shape

            if ndim1 > 1:
                ndim2 = ha.ndim
                amp_param = amp_param[(slice(None),) * ndim1 + (None,) * ndim2]
                ha = ha[(None,) * (ndim1 - 1) + (slice(None),) * ndim2]

        slc = (slice(None),) * (ndim1 - 1)
        peak_amplitude = amp_param[slc + (0,)]
        centroid = amp_param[slc + (1,)]
        fwhm = amp_param[slc + (2,)]

        dha = _correct_phase_wrap(ha - centroid)

        jac = np.zeros(shp, dtype=ha.dtype)
        jac[slc + (0,)] = tools.invert_no_zero(peak_amplitude)
        jac[slc + (1,)] = 8.0 * np.log(2.0) * dha * tools.invert_no_zero(fwhm) ** 2
        jac[slc + (2,)] = 8.0 * np.log(2.0) * dha**2 * tools.invert_no_zero(fwhm) ** 3

        return jac

    def _jacobian_phi(self, ha, elementwise=False):
        jac = self._vander(ha, self.poly_deg_phi)
        if not elementwise:
            jac = np.rollaxis(jac, -1)
            if self.N is not None:
                slc = (None,) * len(self.N)
                jac = jac[slc]

        return jac

    @property
    def parameter_names(self):
        """Array of strings containing the name of the fit parameters."""
        return np.array(
            ["peak_amplitude", "centroid", "fwhm"]
            + ["%s_poly_phi_coeff%d" % (self.poly_type, p) for p in range(self.nparp)],
            dtype=np.string_,
        )

    @property
    def ndofa(self):
        """
        Number of degrees of freedom for the amplitude fit.

        Returns
        -------
        ndofa : np.ndarray[...]
            Number of degrees of freedom of the amplitude fit.
        """
        return self.ndof[..., 0]

    @property
    def ndofp(self):
        """
        Number of degrees of freedom for the phase fit.

        Returns
        -------
        ndofp : np.ndarray[...]
            Number of degrees of freedom of the phase fit.
        """
        return self.ndof[..., 0]


def _propagate_uncertainty(jac, cov, tval):
    """Propagate uncertainty on parameters to uncertainty on model prediction.

    Parameters
    ----------
    jac : np.ndarray[..., nparam] (elementwise) or np.ndarray[..., nparam, nha]
        The jacobian defined as
        jac[..., i, j] = d(model(ha)) / d(param[i]) evaluated at ha[j]
    cov : [..., nparam, nparam]
        Covariance of model parameters.
    tval : np.ndarray[...]
        Quantile of a standardized Student's t random variable.
        The 1-sigma uncertainties will be scaled by this value.

    Returns
    -------
    err : np.ndarray[...] (elementwise) or np.ndarray[..., nha]
        Uncertainty on the model.
    """
    if jac.ndim == cov.ndim:
        # Corresponds to non-elementwise analysis
        df2 = np.sum(jac * np.matmul(cov, jac), axis=-2)
    else:
        # Corresponds to elementwise analysis
        df2 = np.sum(jac * np.sum(cov * jac[..., np.newaxis], axis=-1), axis=-1)

    # Expand the tval array so that it can be broadcast against
    # the sum squared error df2
    add_dim = df2.ndim - tval.ndim
    if add_dim > 0:
        tval = tval[(np.s_[...],) + (None,) * add_dim]

    return tval * np.sqrt(df2)


def _correct_phase_wrap(ha):
    """Ensure hour angle is between -180 and 180 degrees.

    Parameters
    ----------
    ha : np.ndarray or float
        Hour angle in degrees.

    Returns
    -------
    out : same as ha
        Hour angle between -180 and 180 degrees.
    """
    return ((ha + 180.0) % 360.0) - 180.0


def fit_point_source_map(
    ra,
    dec,
    submap,
    rms=None,
    dirty_beam=None,
    real_map=False,
    freq=600.0,
    ra0=None,
    dec0=None,
):
    """Fits a map of a point source to a model.

    Parameters
    ----------
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
    do_dirty = (dirty_beam is not None) and (
        (len(dirty_beam) == 3) or (dirty_beam.shape == submap.shape)
    )
    if do_dirty:
        if real_map:
            model = func_real_dirty_gauss
        else:
            model = func_dirty_gauss

        # Get parameter names through inspection
        param_name = inspect.getargspec(model(None)).args[1:]

        # Define dimensions of the dirty beam
        if len(dirty_beam) != 3:
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
    param = np.full(dims + (nparam,), np.nan, dtype=np.float64)
    param_cov = np.full(dims + (nparam, nparam), np.nan, dtype=np.float64)
    resid_rms = np.full(dims, np.nan, dtype=np.float64)

    # Iterate over dimensions
    for index in np.ndindex(*dims):
        # Extract the RMS for this index.  In the process,
        # check for data flagged as bad (rms == 0.0).
        if rms is not None:
            good_ra = rms[index] > 0.0
            this_rms = np.tile(
                rms[index][good_ra, np.newaxis], [1, submap.shape[-1]]
            ).ravel()
        else:
            good_ra = np.ones(submap.shape[-2], dtype=bool)
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

        p0_dict = {
            "peak_amplitude": peak0,
            "centroid_x": ra0,
            "centroid_y": dec0,
            "fwhm_x": 2.0,
            "fwhm_y": 2.0,
            "offset": offset0,
            "fringe_rate": 22.0 * freq * 1e6 / 3e8,
        }

        lb_dict = {
            "peak_amplitude": 0.0,
            "centroid_x": ra0 - 1.5,
            "centroid_y": dec0 - 0.75,
            "fwhm_x": 0.5,
            "fwhm_y": 0.5,
            "offset": offset0 - 2.0 * np.abs(offset0),
            "fringe_rate": -200.0,
        }

        ub_dict = {
            "peak_amplitude": 1.5 * peak0,
            "centroid_x": ra0 + 1.5,
            "centroid_y": dec0 + 0.75,
            "fwhm_x": 6.0,
            "fwhm_y": 6.0,
            "offset": offset0 + 2.0 * np.abs(offset0),
            "fringe_rate": 200.0,
        }

        p0 = np.array([p0_dict[key] for key in param_name])

        bounds = (
            np.array([lb_dict[key] for key in param_name]),
            np.array([ub_dict[key] for key in param_name]),
        )

        # Define model
        if do_dirty:
            fdirty = interp1d(
                db_el,
                db[index][good_ra, :],
                axis=-1,
                copy=False,
                kind="cubic",
                bounds_error=False,
                fill_value=0.0,
            )
            this_model = model(fdirty)
        else:
            this_model = model

        # Perform the fit.  If there is an error,
        # then we leave parameter values as NaN.
        try:
            popt, pcov = curve_fit(
                this_model,
                this_coord,
                this_submap,
                p0=p0,
                sigma=this_rms,
                absolute_sigma=True,
            )  # , bounds=bounds)
        except Exception as error:
            print(
                "index %s: %s"
                % ("(" + ", ".join(["%d" % ii for ii in index]) + ")", error)
            )
            continue

        # Save the results
        param[index] = popt
        param_cov[index] = pcov

        # Calculate RMS of the residuals
        resid = this_submap - this_model(this_coord, *popt)
        resid_rms[index] = 1.4826 * np.median(np.abs(resid - np.median(resid)))

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


def func_2d_gauss(
    coord, peak_amplitude, centroid_x, centroid_y, fwhm_x, fwhm_y, offset
):
    """Returns a parameteric model for the map of a point source,
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

    model = (
        peak_amplitude
        * np.exp(-4.0 * np.log(2.0) * ((x[:, np.newaxis] - centroid_x) / fwhm_x) ** 2)
        * np.exp(-4.0 * np.log(2.0) * ((y[np.newaxis, :] - centroid_y) / fwhm_y) ** 2)
    ) + offset

    return model.ravel()


def func_2d_sinc_gauss(
    coord, peak_amplitude, centroid_x, centroid_y, fwhm_x, fwhm_y, offset
):
    """Returns a parameteric model for the map of a point source,
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

    model = (
        peak_amplitude
        * np.exp(-4.0 * np.log(2.0) * ((x[:, np.newaxis] - centroid_x) / fwhm_x) ** 2)
        * np.sinc(1.2075 * (y[np.newaxis, :] - centroid_y) / fwhm_y)
    ) + offset

    return model.ravel()


def func_dirty_gauss(dirty_beam):
    """Returns a parameteric model for the map of a point source,
    consisting of the interpolated dirty beam along the y-axis
    and a gaussian along the x-axis.

    This function is a wrapper that defines the interpolated
    dirty beam.

    Parameters
    ----------
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
        """Returns a parameteric model for the map of a point source,
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

        model = (
            peak_amplitude
            * np.exp(
                -4.0 * np.log(2.0) * ((x[:, np.newaxis] - centroid_x) / fwhm_x) ** 2
            )
            * dirty_beam(y - _dec_to_el(centroid_y))
        ) + offset

        return model.ravel()

    return dirty_gauss


def func_real_dirty_gauss(dirty_beam):
    """Returns a parameteric model for the map of a point source,
    consisting of the interpolated dirty beam along the y-axis
    and a sinusoid with gaussian envelope along the x-axis.

    This function is a wrapper that defines the interpolated
    dirty beam.

    Parameters
    ----------
    dirty_beam : scipy.interpolate.interp1d
        Interpolation function that takes as an argument el = sin(za)
        and outputs an np.ndarray[nel, nra] that represents the dirty
        beam evaluated at the same right ascension as the map.

    Returns
    -------
    real_dirty_gauss : np.ndarray[nra*ndec]
        Model prediction for the map of the point source.
    """

    def real_dirty_gauss(
        coord, peak_amplitude, centroid_x, centroid_y, fwhm_x, offset, fringe_rate
    ):
        """Returns a parameteric model for the map of a point source,
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

        model = (
            peak_amplitude
            * np.exp(
                -4.0 * np.log(2.0) * ((x[:, np.newaxis] - centroid_x) / fwhm_x) ** 2
            )
            * dirty_beam(y - _dec_to_el(centroid_y))
        ) + offset

        phase = np.exp(
            2.0j
            * np.pi
            * np.cos(np.radians(centroid_y))
            * np.sin(-np.radians(x - centroid_x))
            * fringe_rate
        )

        return (model * phase[:, np.newaxis]).real.ravel()

    return real_dirty_gauss


def guess_fwhm(freq, pol="X", dec=None, sigma=False, voltage=False, seconds=False):
    """Provide rough estimate of the FWHM of the CHIME primary beam pattern.

    It uses a linear fit to the median FWHM(nu) over all feeds of a given
    polarization for CygA transits.  CasA and TauA transits also showed
    good agreement with this relationship.

    Parameters
    ----------
    freq : float or np.ndarray
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
    voltage : bool
        Return the value for a voltage beam, otherwise returns
        value for a power beam.
    seconds : bool
        Convert to elapsed time in units of seconds.
        Otherwise returns in units of degrees on the sky.

    Returns
    -------
    fwhm : float or np.ndarray
        Rough estimate of the FWHM (or standard deviation if sigma=True).
    """
    # Define linear coefficients based on polarization
    if (pol == "Y") or (pol == "S"):
        coeff = [1.226e-06, -0.004097, 3.790]
    else:
        coeff = [7.896e-07, -0.003226, 3.717]

    # Estimate standard deviation
    sig = np.polyval(coeff, freq)

    # Divide by declination to convert to degrees hour angle
    if dec is not None:
        sig /= np.cos(dec)

    # If requested, convert to seconds
    if seconds:
        earth_rotation_rate = 360.0 / (24.0 * 3600.0)
        sig /= earth_rotation_rate

    # If requested, convert to width of voltage beam
    if voltage:
        sig *= np.sqrt(2)

    # If sigma not explicitely requested, then convert to FWHM
    if not sigma:
        sig *= 2.35482

    return sig


def estimate_directional_scale(z, c=2.1):
    """Calculate robust, direction dependent estimate of scale.

    Parameters
    ----------
    z: np.ndarray
        1D array containing the data.
    c: float
        Cutoff in number of MAD.  Data points whose absolute value is
        larger than c * MAD from the median are saturated at the
        maximum value in the estimator.

    Returns
    -------
    zmed : float
        The median value of z.
    sa : float
        Estimate of scale for z <= zmed.
    sb : float
        Estimate of scale for z > zmed.
    """
    zmed = np.median(z)

    x = z - zmed

    xa = x[x <= 0.0]
    xb = x[x >= 0.0]

    def huber_rho(dx, c=2.1):
        num = float(dx.size)

        s0 = 1.4826 * np.median(np.abs(dx))

        dx_sig0 = dx * tools.invert_no_zero(s0)

        rho = (dx_sig0 / c) ** 2
        rho[rho > 1.0] = 1.0

        return 1.54 * s0 * np.sqrt(2.0 * np.sum(rho) / num)

    sa = huber_rho(xa, c=c)
    sb = huber_rho(xb, c=c)

    return zmed, sa, sb


def fit_histogram(
    arr,
    bins="auto",
    rng=None,
    no_weight=False,
    test_normal=False,
    return_histogram=False,
):
    """
    Fit a gaussian to a histogram of the data.

    Parameters
    ----------
    arr : np.ndarray
        1D array containing the data.  Arrays with more than one dimension are flattened.
    bins : int or sequence of scalars or str
        - If `bins` is an int, it defines the number of equal-width bins in `rng`.
        - If `bins` is a sequence, it defines a monotonically increasing array of bin edges,
          including the rightmost edge, allowing for non-uniform bin widths.
        - If `bins` is a string, it defines a method for computing the bins.
    rng : (float, float)
        The lower and upper range of the bins.  If not provided, then the range spans
        the minimum to maximum value of `arr`.
    no_weight : bool
        Give equal weighting to each histogram bin.  Otherwise use proper weights based
        on number of counts observed in each bin.
    test_normal : bool
        Apply the Shapiro-Wilk and Anderson-Darling tests for normality to the data.
    return_histogram : bool
        Return the histogram.  Otherwise return only the best fit parameters and test statistics.

    Returns
    -------
    results: dict
        Dictionary containing the following fields:
    indmin : int
        Only bins whose index is greater than indmin were included in the fit.
    indmax : int
        Only bins whose index is less than indmax were included in the fit.
    xmin : float
        The data value corresponding to the centre of the `indmin` bin.
    xmax : float
        The data value corresponding to the centre of the `indmax` bin.
    par: [float, float, float]
        The parameters of the fit, ordered as [peak, mu, sigma].
    chisq: float
        The chi-squared of the fit.
    ndof : int
        The number of degrees of freedom of the fit.
    pte : float
        The probability to observe the chi-squared of the fit.

    If `return_histogram` is True, then `results` will also contain the following fields:

        bin_centre : np.ndarray
            The bin centre of the histogram.
        bin_count : np.ndarray
            The bin counts of the histogram.

    If `test_normal` is True, then `results` will also contain the following fields:

        shapiro : dict
            stat : float
                The Shapiro-Wilk test statistic.
            pte : float
                The probability to observe `stat` if the data were drawn from a gaussian.
        anderson : dict
            stat : float
                The Anderson-Darling test statistic.
            critical : list of float
                The critical values of the test statistic.
            alpha : list of float
                The significance levels corresponding to each critical value.
            past : list of bool
                Boolean indicating if the data passes the test for each critical value.
    """
    # Make sure the data is 1D
    data = np.ravel(arr)

    # Histogram the data
    count, xbin = np.histogram(data, bins=bins, range=rng)
    cbin = 0.5 * (xbin[0:-1] + xbin[1:])

    cbin = cbin.astype(np.float64)
    count = count.astype(np.float64)

    # Form initial guess at parameter values using median and MAD
    nparams = 3
    par0 = np.zeros(nparams, dtype=np.float64)
    par0[0] = np.max(count)
    par0[1] = np.median(data)
    par0[2] = 1.48625 * np.median(np.abs(data - par0[1]))

    # Find the first zero points on either side of the median
    cont = True
    indmin = np.argmin(np.abs(cbin - par0[1]))
    while cont:
        indmin -= 1
        cont = (count[indmin] > 0.0) and (indmin > 0)
    indmin += count[indmin] == 0.0

    cont = True
    indmax = np.argmin(np.abs(cbin - par0[1]))
    while cont:
        indmax += 1
        cont = (count[indmax] > 0.0) and (indmax < (len(count) - 1))
    indmax -= count[indmax] == 0.0

    # Restrict range of fit to between zero points
    x = cbin[indmin : indmax + 1]
    y = count[indmin : indmax + 1]
    yerr = np.sqrt(y * (1.0 - y / np.sum(y)))

    sigma = None if no_weight else yerr

    # Require positive values of amp and sigma
    bnd = (np.array([0.0, -np.inf, 0.0]), np.array([np.inf, np.inf, np.inf]))

    # Define the fitting function
    def gauss(x, peak, mu, sigma):
        return peak * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))

    # Perform the fit
    par, var_par = curve_fit(
        gauss,
        cbin[indmin : indmax + 1],
        count[indmin : indmax + 1],
        p0=par0,
        sigma=sigma,
        absolute_sigma=(not no_weight),
        bounds=bnd,
        method="trf",
    )

    # Calculate quality of fit
    chisq = np.sum(((y - gauss(x, *par)) / yerr) ** 2)
    ndof = np.size(y) - nparams
    pte = 1.0 - scipy.stats.chi2.cdf(chisq, ndof)

    # Store results in dictionary
    results_dict = {}
    results_dict["indmin"] = indmin
    results_dict["indmax"] = indmax
    results_dict["xmin"] = cbin[indmin]
    results_dict["xmax"] = cbin[indmax]
    results_dict["par"] = par
    results_dict["chisq"] = chisq
    results_dict["ndof"] = ndof
    results_dict["pte"] = pte

    if return_histogram:
        results_dict["bin_centre"] = cbin
        results_dict["bin_count"] = count

    # If requested, test normality of the main distribution
    if test_normal:
        flag = (data > cbin[indmin]) & (data < cbin[indmax])
        shap_stat, shap_pte = scipy.stats.shapiro(data[flag])

        results_dict["shapiro"] = {}
        results_dict["shapiro"]["stat"] = shap_stat
        results_dict["shapiro"]["pte"] = shap_pte

        ander_stat, ander_crit, ander_signif = scipy.stats.anderson(
            data[flag], dist="norm"
        )

        results_dict["anderson"] = {}
        results_dict["anderson"]["stat"] = ander_stat
        results_dict["anderson"]["critical"] = ander_crit
        results_dict["anderson"]["alpha"] = ander_signif
        results_dict["anderson"]["pass"] = ander_stat < ander_crit

    # Return dictionary
    return results_dict


def _sliding_window(arr, window):
    # Advanced numpy tricks
    shape = arr.shape[:-1] + (arr.shape[-1] - window + 1, window)
    strides = arr.strides + (arr.strides[-1],)
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)


def flag_outliers(raw, flag, window=25, nsigma=5.0):
    """Flag outliers with respect to rolling median.

    Parameters
    ----------
    raw : np.ndarray[nsample,]
        Raw data sampled at fixed rate.  Use the `flag` parameter to indicate missing
        or invalid data.
    flag : np.ndarray[nsample,]
        Boolean array where True indicates valid data and False indicates invalid data.
    window : int
        Window size (in number of samples) used to determine local median.
    nsigma : float
        Data is considered an outlier if it is greater than this number of median absolute
        deviations away from the local median.
    Returns
    -------
    not_outlier : np.ndarray[nsample,]
        Boolean array where True indicates valid data and False indicates data that is
        either an outlier or had flag = True.
    """
    # Make sure we have an even window size
    if window % 2:
        window += 1

    hwidth = window // 2 - 1

    nraw = raw.size
    dtype = raw.dtype

    # Replace flagged samples with nan
    good = np.flatnonzero(flag)

    data = np.full((nraw,), np.nan, dtype=dtype)
    data[good] = raw[good]

    # Expand the edges
    expanded_data = np.concatenate(
        (
            np.full((hwidth,), np.nan, dtype=dtype),
            data,
            np.full((hwidth + 1,), np.nan, dtype=dtype),
        )
    )

    # Apply median filter
    smooth = np.nanmedian(_sliding_window(expanded_data, window), axis=-1)

    # Calculate RMS of residual
    resid = np.abs(data - smooth)

    rwidth = 9 * window
    hrwidth = rwidth // 2 - 1

    expanded_resid = np.concatenate(
        (
            np.full((hrwidth,), np.nan, dtype=dtype),
            resid,
            np.full((hrwidth + 1,), np.nan, dtype=dtype),
        )
    )

    sig = 1.4826 * np.nanmedian(_sliding_window(expanded_resid, rwidth), axis=-1)

    not_outlier = resid < (nsigma * sig)

    return not_outlier


def interpolate_gain(freq, gain, weight, flag=None, length_scale=30.0):
    """Replace gain at flagged frequencies with interpolated values.

    Uses a gaussian process regression to perform the interpolation
    with a Matern function describing the covariance between frequencies.

    Parameters
    ----------
    freq : np.ndarray[nfreq,]
        Frequencies in MHz.
    gain : np.ndarray[nfreq, ninput]
        Complex gain for each input and frequency.
    weight : np.ndarray[nfreq, ninput]
        Uncertainty on the complex gain, expressed as inverse variance.
    flag : np.ndarray[nfreq, ninput]
        Boolean array indicating the good (True) and bad (False) gains.
        If not provided, then it will be determined by evaluating `weight > 0.0`.
    length_scale : float
        Correlation length in frequency in MHz.

    Returns
    -------
    interp_gain : np.ndarray[nfreq, ninput]
        For frequencies with `flag = True`, this will be equal to gain.  For frequencies with
        `flag = False`, this will be an interpolation of the gains with `flag = True`.
    interp_weight : np.ndarray[nfreq, ninput]
        For frequencies with `flag = True`, this will be equal to weight.  For frequencies with
        `flag = False`, this will be the expected uncertainty on the interpolation.
    """
    from sklearn import gaussian_process
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel

    if flag is None:
        flag = weight > 0.0

    nfreq, ninput = gain.shape

    iscomplex = np.any(np.iscomplex(gain))

    interp_gain = gain.copy()
    interp_weight = weight.copy()

    alpha = tools.invert_no_zero(weight)

    x = freq.reshape(-1, 1)

    for ii in range(ninput):
        train = np.flatnonzero(flag[:, ii])
        test = np.flatnonzero(~flag[:, ii])

        if train.size > 0:
            xtest = x[test, :]

            xtrain = x[train, :]
            if iscomplex:
                ytrain = np.hstack(
                    (gain[train, ii, np.newaxis].real, gain[train, ii, np.newaxis].imag)
                )
            else:
                ytrain = gain[train, ii, np.newaxis].real

            # Mean subtract
            ytrain_mu = np.mean(ytrain, axis=0, keepdims=True)
            ytrain = ytrain - ytrain_mu

            # Get initial estimate of variance
            var = 0.5 * np.sum(
                (
                    1.4826
                    * np.median(
                        np.abs(ytrain - np.median(ytrain, axis=0, keepdims=True)),
                        axis=0,
                    )
                )
                ** 2
            )

            # Define kernel
            kernel = ConstantKernel(
                constant_value=var, constant_value_bounds=(0.01 * var, 100.0 * var)
            ) * Matern(length_scale=length_scale, length_scale_bounds="fixed", nu=1.5)

            # Regress against non-flagged data
            gp = gaussian_process.GaussianProcessRegressor(
                kernel=kernel, alpha=alpha[train, ii]
            )

            gp.fit(xtrain, ytrain)

            # Predict error
            ypred, err_ypred = gp.predict(xtest, return_std=True)

            # When the gains are not complex, ypred will have a single dimension for
            # sklearn version 1.1.2, but will have a second dimension of length 1 for
            # earlier versions.  The line below ensures consistent behavior.
            if ypred.ndim == 1:
                ypred = ypred[:, np.newaxis]

            interp_gain[test, ii] = ypred[:, 0] + ytrain_mu[:, 0]
            if iscomplex:
                interp_gain[test, ii] += 1.0j * (ypred[:, 1] + ytrain_mu[:, 1])

            # When the gains are complex, err_ypred will have a second dimension
            # of length 2 for sklearn version 1.1.2, but will have a single dimension
            # for earlier versions.  The line below ensures consistent behavior.
            if err_ypred.ndim > 1:
                err_ypred = np.sqrt(
                    np.sum(err_ypred**2, axis=-1) / err_ypred.shape[-1]
                )

            interp_weight[test, ii] = tools.invert_no_zero(err_ypred**2)

        else:
            # No valid data
            interp_gain[:, ii] = 0.0 + 0.0j
            interp_weight[:, ii] = 0.0

    return interp_gain, interp_weight


def interpolate_gain_quiet(*args, **kwargs):
    """Call `interpolate_gain` with `ConvergenceWarnings` silenced.

    Accepts and passes all arguments and keyword arguments for `interpolate_gain`.
    """
    import warnings
    from sklearn.exceptions import ConvergenceWarning

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
        results = interpolate_gain(*args, **kwargs)

    return results


def thermal_amplitude(delta_T, freq):
    """Computes the amplitude gain correction given a (set of) temperature
    difference and a (set of) frequency based on the thermal model.

    Parameters
    ----------
    delta_T : float or array of foats
        Temperature difference (T - T_0) for which to find a gain correction.
    freq : float or array of foats
        Frequencies in MHz

    Returns
    -------
    g : float or array of floats
        Gain amplitude corrections. Multiply by data
        to correct it.
    """
    m_params = [-4.28268629e-09, 8.39576400e-06, -2.00612389e-03]
    m = np.polyval(m_params, freq)

    return 1.0 + m * delta_T


def _el_to_dec(el):
    """Convert from el = sin(zenith angle) to declination in degrees."""

    return np.degrees(np.arcsin(el)) + ephemeris.CHIMELATITUDE


def _dec_to_el(dec):
    """Convert from declination in degrees to el = sin(zenith angle)."""

    return np.sin(np.radians(dec - ephemeris.CHIMELATITUDE))


def get_reference_times_file(
    times: np.ndarray,
    cal_file: memh5.MemGroup,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, np.ndarray]:
    """For a given set of times determine when and how they were calibrated.

    This uses the pre-calculated calibration time reference files.

    Parameters
    ----------
    times
        Unix times of data points to be calibrated as floats.
    cal_file
        memh5 container which containes the reference times for calibration source
        transits.
    logger
        A logging object to use for messages. If not provided, use a module level
        logger.

    Returns
    -------
    reftime_result : dict
        A dictionary containing four entries:

        - reftime: Unix time of same length as `times`. Reference times of transit of the
          source used to calibrate the data at each time in `times`. Returns `NaN` for
          times without a reference.
        - reftime_prev: The Unix time of the previous gain update. Only set for time
          samples that need to be interpolated, otherwise `NaN`.
        - interp_start: The Unix time of the start of the interpolation period. Only
          set for time samples that need to be interpolated, otherwise `NaN`.
        - interp_stop: The Unix time of the end of the interpolation period. Only
          set for time samples that need to be interpolated, otherwise `NaN`.
    """

    if logger is None:
        logger = logging.getLogger(__name__)

    # Data from calibration file.
    is_restart = cal_file["is_restart"][:]
    tref = cal_file["tref"][:]
    tstart = cal_file["tstart"][:]
    tend = cal_file["tend"][:]
    # Length of calibration file and of data points
    n_cal_file = len(tstart)
    ntimes = len(times)

    # Len of times, indices in cal_file.
    last_start_index = np.searchsorted(tstart, times, side="right") - 1
    # Len of times, indices in cal_file.
    last_end_index = np.searchsorted(tend, times, side="right") - 1
    # Check for times before first update or after last update.
    too_early = last_start_index < 0
    n_too_early = np.sum(too_early)
    if n_too_early > 0:
        msg = (
            "{0} out of {1} time entries have no reference update."
            + "Cannot correct gains for those entries."
        )
        logger.warning(msg.format(n_too_early, ntimes))
    # Fot times after the last update, I cannot be sure the calibration is valid
    # (could be that the cal file is incomplete. To be conservative, raise warning.)
    too_late = (last_start_index >= (n_cal_file - 1)) & (
        last_end_index >= (n_cal_file - 1)
    )
    n_too_late = np.sum(too_late)
    if n_too_late > 0:
        msg = (
            "{0} out of {1} time entries are beyond calibration file time values."
            + "Cannot correct gains for those entries."
        )
        logger.warning(msg.format(n_too_late, ntimes))

    # Array to contain reference times for each entry.
    # NaN for entries with no reference time.
    reftime = np.full(ntimes, np.nan, dtype=np.float64)
    # Array to hold reftimes of previous updates
    # (for entries that need interpolation).
    reftime_prev = np.full(ntimes, np.nan, dtype=np.float64)
    # Arrays to hold start and stop times of gain transition
    # (for entries that need interpolation).
    interp_start = np.full(ntimes, np.nan, dtype=np.float64)
    interp_stop = np.full(ntimes, np.nan, dtype=np.float64)

    # Acquisition restart. We load an old gain.
    acqrestart = is_restart[last_start_index] == 1
    reftime[acqrestart] = tref[last_start_index][acqrestart]

    # FPGA restart. Data not calibrated.
    # There shouldn't be any time points here. Raise a warning if there are.
    fpga_restart = is_restart[last_start_index] == 2
    n_fpga_restart = np.sum(fpga_restart)
    if n_fpga_restart > 0:
        msg = (
            "{0} out of {1} time entries are after an FPGA restart but before the "
            + "next kotekan restart. Cannot correct gains for those entries."
        )
        logger.warning(msg.format(n_fpga_restart, ntimes))

    # This is a gain update
    gainupdate = is_restart[last_start_index] == 0

    # This is the simplest case. Last update was a gain update and
    # it is finished. No need to interpolate.
    calrange = (last_start_index == last_end_index) & gainupdate
    reftime[calrange] = tref[last_start_index][calrange]

    # The next cases might need interpolation. Last update was a gain
    # update and it is *NOT* finished. Update is in transition.
    gaintrans = last_start_index == (last_end_index + 1)

    # This update is in gain transition and previous update was an
    # FPGA restart. Just use new gain, no interpolation.
    prev_is_fpga = is_restart[last_start_index - 1] == 2
    prev_is_fpga = prev_is_fpga & gaintrans & gainupdate
    reftime[prev_is_fpga] = tref[last_start_index][prev_is_fpga]

    # The next two cases need interpolation of gain corrections.
    # It's not possible to correct interpolated gains because the
    # products have been stacked. Just interpolate the gain
    # corrections to avoide a sharp transition.

    # This update is in gain transition and previous update was a
    # Kotekan restart. Need to interpolate gain corrections.
    prev_is_kotekan = is_restart[last_start_index - 1] == 1
    to_interpolate = prev_is_kotekan & gaintrans & gainupdate

    # This update is in gain transition and previous update was a
    # gain update. Need to interpolate.
    prev_is_gain = is_restart[last_start_index - 1] == 0
    to_interpolate = to_interpolate | (prev_is_gain & gaintrans & gainupdate)

    # Reference time of this update
    reftime[to_interpolate] = tref[last_start_index][to_interpolate]
    # Reference time of previous update
    reftime_prev[to_interpolate] = tref[last_start_index - 1][to_interpolate]
    # Start and stop times of gain transition.
    interp_start[to_interpolate] = tstart[last_start_index][to_interpolate]
    interp_stop[to_interpolate] = tend[last_start_index][to_interpolate]

    # For times too early or too late, don't correct gain.
    # This might mean we don't correct gains right after the last update
    # that could in principle be corrected. But there is no way to know
    # If the calibration file is up-to-date and the last update applies
    # to all entries that come after it.
    reftime[too_early | too_late] = np.nan

    # Test for un-identified NaNs
    known_bad_times = (too_early) | (too_late) | (fpga_restart)
    n_bad_times = np.sum(~np.isfinite(reftime[~known_bad_times]))
    if n_bad_times > 0:
        msg = (
            "{0} out of {1} time entries don't have a reference calibration time "
            + "without an identifiable cause. Cannot correct gains for those entries."
        )
        logger.warning(msg.format(n_bad_times, ntimes))

    # Bundle result in dictionary
    result = {
        "reftime": reftime,
        "reftime_prev": reftime_prev,
        "interp_start": interp_start,
        "interp_stop": interp_stop,
    }

    return result


def get_reference_times_dataset_id(
    times: np.ndarray,
    dataset_ids: np.ndarray,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Union[np.ndarray, Dict]]:
    """Calculate the relevant calibration reference times from the dataset IDs.

    .. warning::
        Dataset IDs before 2020/10/10 are corrupt so this routine won't work.

    Parameters
    ----------
    times
        Unix times of data points to be calibrated as floats.
    dataset_ids
        The dataset IDs as an array of strings.
    logger
        A logging object to use for messages. If not provided, use a module level
        logger.

    Returns
    -------
    reftime_result
        A dictionary containing the results. See `get_reference_times_file` for a
        description of the contents.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Dataset IDs before this date are untrustworthy
    ds_start = ephemeris.datetime_to_unix(datetime(2020, 11, 1))
    if (times < ds_start).any():
        raise ValueError(
            "Dataset IDs before 2020/11/01 are corrupt, so this method won't work. "
            f"You passed in a time as early as {ctime.unix_to_datetime(times.min())}."
        )

    # The CHIME calibration sources
    _source_dict = {
        "cyga": ephemeris.CygA,
        "casa": ephemeris.CasA,
        "taua": ephemeris.TauA,
        "vira": ephemeris.VirA,
    }

    # Get the set of gain IDs for each time stamp
    gain_ids = state_id_of_type(dataset_ids, "gains")
    collapsed_ids = unique_unmasked_entry(gain_ids, axis=0)
    unique_gains_ids = np.unique(collapsed_ids.compressed())

    gain_info_dict = {}

    # For each gain update extract all the relevant information
    for state_id in unique_gains_ids:
        d = {}
        gain_info_dict[state_id] = d

        # Extract the update ID
        update_id = ds.DatasetState.from_id(state_id).data["data"]["update_id"]

        # Parse the ID for the required information
        split_id = update_id.split("_")
        # After restart we sometimes have only a timing update without a source
        # reference. These aren't valid for our purposes here, and can be distinguished
        # at the update_id doesn't contain source information, and is thus shorter
        d["valid"] = any([src in split_id for src in _source_dict.keys()])
        d["interpolated"] = "transition" in split_id
        # If it's not a valid update we shouldn't try to extract everything else
        if not d["valid"]:
            continue

        d["gen_time"] = ctime.datetime_to_unix(ctime.timestr_to_datetime(split_id[1]))
        d["source_name"] = split_id[2].lower()

        # Calculate the source transit time, and sanity check it
        source = _source_dict[d["source_name"]]
        d["source_transit"] = ephemeris.transit_times(
            source, d["gen_time"] - 24 * 3600.0
        )
        cal_diff_hours = (d["gen_time"] - d["source_transit"]) / 3600
        if cal_diff_hours > 3:
            logger.warn(
                f"Transit time ({ctime.unix_to_datetime(d['source_transit'])}) "
                f"for source {d['source_name']} was a surprisingly long time "
                f"before the gain update time ({cal_diff_hours} hours)."
            )

    # Array to store the extracted times in
    reftime = np.zeros(len(collapsed_ids), dtype=np.float64)
    reftime_prev = np.zeros(len(collapsed_ids), dtype=np.float64)
    interp_start = np.zeros(len(collapsed_ids), dtype=np.float64)
    interp_stop = np.zeros(len(collapsed_ids), dtype=np.float64)

    # Iterate forward through the updates, setting transit times, and keeping track of
    # the last valid update. This is used to set the previous source transit and the
    # interpolation start time for all blended updates
    last_valid_non_interpolated = None
    last_non_interpolated = None
    for ii, state_id in enumerate(collapsed_ids):
        valid_id = not np.ma.is_masked(state_id)
        update = gain_info_dict[state_id] if valid_id else {}
        valid = valid_id and update["valid"]

        if valid:
            reftime[ii] = update["source_transit"]
        elif last_valid_non_interpolated is not None:
            reftime[ii] = reftime[last_valid_non_interpolated]
        else:
            reftime[ii] = np.nan

        if valid and update["interpolated"] and last_valid_non_interpolated is not None:
            reftime_prev[ii] = reftime[last_valid_non_interpolated]
            interp_start[ii] = times[last_non_interpolated]
        else:
            reftime_prev[ii] = np.nan
            interp_start[ii] = np.nan

        if valid and not update["interpolated"]:
            last_valid_non_interpolated = ii
        if valid_id and not update["interpolated"]:
            last_non_interpolated = ii
    # To identify the end of the interpolation periods we need to iterate
    # backwards in time. As before we need to keep track of the last valid update
    # we see, and then we set the interpolation end in the same manner.
    last_non_interpolated = None
    for ii, state_id in list(enumerate(collapsed_ids))[::-1]:
        valid_id = not np.ma.is_masked(state_id)
        update = gain_info_dict[state_id] if valid_id else {}
        valid = valid_id and update.get("valid", False)

        if valid and update["interpolated"] and last_non_interpolated is not None:
            interp_stop[ii] = times[last_non_interpolated]
        else:
            interp_stop[ii] = np.nan

        if valid_id and not update["interpolated"]:
            last_non_interpolated = ii

    return {
        "reftime": reftime,
        "reftime_prev": reftime_prev,
        "interp_start": interp_start,
        "interp_stop": interp_stop,
        "update_info": gain_info_dict,
    }
