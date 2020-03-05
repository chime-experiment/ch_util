"""Tools for timing jitter and delay corrections.

.. currentmodule:: timing

This module contains tools for using noise sources to correct
timing jitter and timing delay.

Functions
=========

.. autosummary::
    :toctree: generated/

    load_timing_correction
    construct_delay_template
    map_input_to_noise_source
    eigen_decomposition
    fit_poly_to_phase
    func_poly_phase
    model_poly_phase

Classes
=======

.. autosummary::
    :toctree: generated/

    TimingCorrection
    TimingData

Example
=======

The function :meth:`construct_delay_template` generates a delay template from
measurements of the visibility between noise source inputs, which can
be used to remove the timing jitter in other data.

The user seldom needs to work with :meth:`construct_delay_template`
directly and can instead use several high-level functions and containers
that load the timing data, derive the timing correction using
:meth:`construct_delay_template`, and then enable easy application of
the timing correction to other data.

For example, to load the timing data and derive the timing correction from
a list of timing acquisition files (i.e., `YYYYMMSSTHHMMSSZ_chimetiming_corr`),
use the following:

    ```tdata = TimingData.from_acq_h5(timing_acq_filenames)```

This results in a :class:`andata.CorrData` object that has additional
methods avaiable for applying the timing correction to other data.
For example, to obtain the complex gain for some freq, input, and time
that upon multiplication will remove the timing jitter, use the following:

    ```tgain = tdata.get_gain(freq, input, time)```

To apply the timing correction to the visibilities in an :class:`andata.CorrData`
object called `data`, use the following:

    ```tdata.apply_timing_correction(data)```

The timing acquisitions must cover the span of time that you wish to correct.
If you have a list of data acquisition files and would like to obtain
the appropriate timing correction by searching the archive for the
corresponding timing acquisitons files, then use:

    ```tdata = load_timing_correction(data_acq_filenames_full_path)```

To print a summary of the timing correction, use:

    ```print(tdata)```

"""
# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614
from future.utils import native_str

# === End Python 2/3 compatibility

import os
import glob
import numpy as np
import inspect
import logging
import gc

import scipy.signal
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.linalg import eigh

from . import tools, andata, ephemeris
from caput import memh5, mpiarray, tod

FREQ_TO_OMEGA = 2.0 * np.pi * 1e-6
FREQ_PIVOT = 600.0

# Set up logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class TimingCorrection(andata.BaseData):
    """Container that holds a timing correction.

    Provides methods for applying that correction to other datasets.

    Attributes
    ----------
    freq
    input
    noise_source
    time
    tau
    static_phi
    weight_static_phi
    static_phi_fit
    alpha
    static_amp
    weight_static_amp
    reference_input
    amp_to_delay

    Methods
    -------
    from_dict
    set_global_reference_time
    set_reference_time
    get_tau
    get_alpha
    get_stacked_tau
    get_stacked_alpha
    get_timing_correction
    get_gain
    apply_timing_correction
    """

    @classmethod
    def from_dict(self, **kwargs):
        """Instantiate a TimingCorrection object.

        Parameters
        ----------
        freq: np.ndarray[nfreq, ] of dtype=('centre', 'width')
            Frequencies in MHz that were used to construct the timing correction.
        input: np.ndarray[ninput, ] of dtype=('chan_id', 'correlator_input')
            Correlator inputs that were used to construct the timing correction.
        time: np.ndarray[ntime, ]
            Unix time.
        param: np.ndarray[nparam, ]
            Parameters of the model fit to the static phase versus frequency.
        tau:  np.ndarray[ninput, ntime]
            The actual timing correction, which is the relative delay of each of the
            noise source inputs with respect to a reference input versus time.
            The reference input should be index 0 of the noise_source axis.
        static_phi: np.ndarray[nfreq, ninput]
            The phase that was subtracted from each frequency and input prior to fitting
            for the timing correction.  This is necessary to remove the approximately static
            ripple pattern caused by reflections.
        weight_static_phi: np.ndarray[nfreq, ninput]
            Inverse variance on static_phi.
        static_phi_fit: np.ndarray[nparam, ninput]
            Best-fit parameters of a fit to the static phase versus frequency
            for each of the noise source inputs.
        alpha: np.ndarray[ninput, ntime]
            The coefficient of the spectral model of the amplitude variations of
            each of the noise source inputs versus time.
        static_amp: np.ndarray[nfreq, ninput]
            The amplitude that was subtracted from each frequency and input prior to fitting
            for the amplitude variations.  This is necessary to remove the approximately
            static ripple pattern caused by reflections.
        weight_static_amp: np.ndarray[nfreq, ninput]
            Inverse variance on static_amp.
        """
        freq = kwargs.pop("freq")
        inputs = kwargs.pop("input")
        timestamp = kwargs.pop("time")
        param = kwargs.pop("param")

        tau = kwargs.pop("tau")
        static_phi = kwargs.pop("static_phi")
        weight_static_phi = kwargs.pop("weight_static_phi")
        static_phi_fit = kwargs.pop("static_phi_fit")

        alpha = kwargs.pop("alpha")
        static_amp = kwargs.pop("static_amp")
        weight_static_amp = kwargs.pop("weight_static_amp")

        # Run base initialiser
        data = TimingCorrection(**kwargs)

        # Create index maps
        data.create_index_map("freq", freq)
        data.create_index_map("input", inputs)
        data.create_index_map("time", timestamp)
        data.create_index_map("param", param)

        # Create datasets
        dset = data.create_dataset("tau", data=tau)
        dset.attrs["axis"] = np.array(["input", "time"], dtype=np.string_)

        dset = data.create_dataset("static_phi", data=static_phi)
        dset.attrs["axis"] = np.array(["freq", "input"], dtype=np.string_)

        dset = data.create_flag("weight_static_phi", data=weight_static_phi)
        dset.attrs["axis"] = np.array(["freq", "input"], dtype=np.string_)

        dset = data.create_dataset("static_phi_fit", data=static_phi_fit)
        dset.attrs["axis"] = np.array(["param", "input"], dtype=np.string_)

        dset = data.create_dataset("alpha", data=alpha)
        dset.attrs["axis"] = np.array(["input", "time"], dtype=np.string_)

        dset = data.create_dataset("static_amp", data=static_amp)
        dset.attrs["axis"] = np.array(["freq", "input"], dtype=np.string_)

        dset = data.create_flag("weight_static_amp", data=weight_static_amp)
        dset.attrs["axis"] = np.array(["freq", "input"], dtype=np.string_)

        return data

    @classmethod
    def _interpret_and_read(cls, acq_files, start, stop, datasets, out_group):

        # Instantiate an object for each file
        objs = [cls.from_file(d, ondisk=False) for d in acq_files]

        # Reference all dynamic datasets to the static quantities defined in the first file
        iref = 0

        freq = objs[iref].freq

        # Determine the overall delay offset relative to the reference file
        phi = np.stack([obj.static_phi[:] for obj in objs], axis=-1)
        weight = np.stack([obj.weight_static_phi[:] for obj in objs], axis=-1)

        phi_ref = phi[..., iref, np.newaxis]
        weight_ref = weight[..., iref, np.newaxis]

        flag = (weight > 0.0) & (weight_ref > 0.0)
        err = np.sqrt(tools.invert_no_zero(weight) + tools.invert_no_zero(weight_ref))
        err *= flag.astype(err.dtype)

        dphi = phi - phi_ref

        for tt, obj in enumerate(objs):
            for nn in range(dphi.shape[1]):
                if np.any(flag[:, nn, tt]):
                    # Fit the difference in the static phase between this file and the
                    # reference file to a linear relationship with frequency.  Uses
                    # nonlinear-least-squares that is insensitive to phase wrapping
                    param = fit_poly_to_phase(
                        freq, np.exp(1.0j * dphi[:, nn, tt]), err[:, nn, tt], nparam=2
                    )[0]

                    # Add the best-fit slope to the delay template for this file
                    obj.tau[nn, :] += param[1]

        # Determine the overall amplitude offset relative to the reference file
        amp = np.stack([obj.static_amp[:] for obj in objs], axis=-1)
        weight = np.stack([obj.weight_static_amp[:] for obj in objs], axis=-1)

        amp_ref = amp[..., iref, np.newaxis]
        weight_ref = weight[..., iref, np.newaxis]

        flag = (weight > 0.0) & (weight_ref > 0.0)
        weight = tools.invert_no_zero(
            tools.invert_no_zero(weight) + tools.invert_no_zero(weight_ref)
        )
        weight *= flag.astype(weight.dtype)

        damp = amp * tools.invert_no_zero(amp_ref) - 1.0

        asc = _amplitude_scaling(freq[:, np.newaxis, np.newaxis])

        alpha = np.sum(weight * asc * damp, axis=0) * tools.invert_no_zero(
            np.sum(weight * asc ** 2, axis=0)
        )

        for tt, obj in enumerate(objs):
            # Add the offset to the amplitude template for this file
            obj.alpha[:] += alpha[:, tt, np.newaxis]

        # Now concatenate the files.  Dynamic datasets will be concatenated.
        # Static datasets will be extracted from the first file.
        data = tod.concatenate(
            objs, out_group=out_group, start=start, stop=stop, datasets=datasets
        )

        return data

    @property
    def freq(self):
        """Provide convenience access to the frequency bin centres."""
        return self.index_map["freq"]["centre"]

    @property
    def input(self):
        """Provide convenience access to the input index map."""
        return self.index_map["input"]

    @property
    def noise_source(self):
        """Alias for `input`."""
        return self.index_map["input"]

    @property
    def tau(self):
        """Provide convenience access to the tau array."""
        return self.datasets["tau"]

    @property
    def static_phi(self):
        """Provide convenience access to the static_phi array."""
        return self.datasets["static_phi"]

    @property
    def weight_static_phi(self):
        """Provide convenience access to the weight_static_phi array."""
        return self.flags["weight_static_phi"]

    @property
    def static_phi_fit(self):
        """Provide convenience access to the static_phi_fit array."""
        return self.datasets["static_phi_fit"]

    @property
    def alpha(self):
        """Provide convenience access to the alpha array."""
        return self.datasets["alpha"]

    @property
    def static_amp(self):
        """Provide convenience access to the static_amp array."""
        return self.datasets["static_amp"]

    @property
    def weight_static_amp(self):
        """Provide convenience access to the weight_static_amp array."""
        return self.flags["weight_static_amp"]

    @property
    def reference_input(self):
        """Return the index of the reference input."""
        zero_tau = np.flatnonzero(np.all(np.abs(self.tau[:]) < 1e-5, axis=-1))
        iref = zero_tau[0] if zero_tau.size > 0 else None
        return iref

    @property
    def has_amplitude(self):
        """Determine if this timing correction contains amplitude data."""
        return "alpha" in self.datasets

    @property
    def amp_to_delay(self):
        """Return conversion from noise source amplitude variations to delay variations."""
        return self.attrs.get("amp_to_delay", None)

    @amp_to_delay.setter
    def amp_to_delay(self, val):
        """Sets the conversion from noise source amplitude variations to delay variations.

        Note that setting this quantity will result in the following modification to the
        timing correction:  tau --> tau - amp_to_delay * alpha.  This can be used to remove
        variations introduced by the noise source distribution system from the timing correction
        using the amplitude variations as a proxy for temperature.
        """
        if val is not None:
            self.attrs["amp_to_delay"] = val

    def set_global_reference_time(self, tref, window=0.0, interpolate=False, **kwargs):
        """Normalize the delay and alpha template to the value at a single time.

        Useful for referencing the template to the value at the time that
        you plan to calibrate.

        Parameters
        ----------
        tref : unix time
            Reference the templates to the values at this time.
        window: float
            Reference the templates to the median value over a window (in seconds)
            around tref.  If nonzero, this will override the interpolate keyword.
        interpolate : bool
            Interpolate the delay template to time tref.  Otherwise take the measured time
            nearest to tref.  The get_tau method is use to perform the interpolation, and
            kwargs for that method will be passed along.
        """
        tref = ephemeris.ensure_unix(tref)
        tref_string = ephemeris.unix_to_datetime(tref).strftime("%Y-%m-%d %H:%M:%S %Z")
        if (tref < self.time[0]) or (tref > self.time[-1]):
            InputError("Timing correction not available for time %s." % tref_string)
        else:
            logger.info(
                "Referencing timing correction with respect to %s." % tref_string
            )

        if window > 0.0:
            iref = np.flatnonzero(
                (self.time >= (tref - window)) & (self.time <= (tref + window))
            )
            logger.info("Using median of %d samples around reference time." % iref.size)
            tau_ref = np.median(self.tau[:, iref], axis=-1, keepdims=True)
            alpha_ref = np.median(self.alpha[:, iref], axis=-1, keepdims=True)
        elif interpolate:
            tau_ref = self.get_tau(np.atleast_1d(tref), ignore_amp=True, **kwargs)
            alpha_ref = self.get_alpha(np.atleast_1d(tref), **kwargs)
        else:
            iref = np.argmin(np.abs(self.time - tref))
            tau_ref = self.tau[:, iref, np.newaxis]
            alpha_ref = self.alpha[:, iref, np.newaxis]

        self.tau[:] = self.tau[:] - tau_ref
        self.alpha[:] = self.alpha[:] - alpha_ref

    def set_reference_time(
        self,
        tref,
        tstart,
        tend=None,
        tinit=None,
        tau_init=None,
        alpha_init=None,
        interpolate=False,
        **kwargs
    ):
        """Normalize the delay and alpha template to specific times.

        Required if applying the timing correction to data that has
        already been calibrated.

        Parameters
        ----------
        tref : np.ndarray[nref]
            Reference the delays to the values at this unix time.
        tstart : np.ndarray[nref]
            Begin transition to the reference delay at this unix time.
        tend : np.ndarray[nref]
            Complete transition to the reference delay at this unix time.
        tinit : float
            Use the delay at this time for the period before the first tstart.
            Takes prescendent over tau_init.
        tau_init : np.ndarray[nsource]
            Use this delay for times before the first tstart.  Must provide a value
            for each noise source input.  If None, then will reference with respect
            to the average delay over the full time series.
        alpha_init : np.ndarray[nsource]
            Use this alpha for times before the first tstart.  Must provide a value
            for each noise source input.  If None, then will reference with respect
            to the average alpha over the full time series.
        interpolate : bool
            Interpolate the delay template to times tref.  Otherwise take the measured times
            nearest to tref.  The get_tau method is use to perform the interpolation, and
            kwargs for that method will be passed along.
        """
        tref = np.atleast_1d(ephemeris.ensure_unix(tref))

        if interpolate:
            tau_ref = self.get_tau(tref, ignore_amp=True, **kwargs)
            alpha_ref = self.get_alpha(tref, **kwargs)
        else:
            iref = np.array([np.argmin(np.abs(self.time - tt)) for tt in tref])
            tau_ref = self.tau[:, iref]
            alpha_ref = self.alpha[:, iref]

        if tinit is not None:
            if interpolate:
                tau_init = self.get_tau(tinit, ignore_amp=True, **kwargs)
                alpha_init = self.get_alpha(tinit, **kwargs)
            else:
                iinit = np.argmin(np.abs(self.time - tinit))
                tau_init = self.tau[:, iinit]
                alpha_init = self.alpha[:, iinit]

        if tau_init is None:
            tau_init = np.zeros((tau_ref.shape[0], 1), dtype=tau_ref.dtype)
        else:
            if tau_init.size == tau_ref.shape[0]:
                tau_init = tau_init[:, np.newaxis]
            else:
                InputError(
                    "Initial tau has size %d, but there are %d noise sources."
                    % (tau_init.size, tau_ref.shape[0])
                )

        if alpha_init is None:
            alpha_init = np.zeros((alpha_ref.shape[0], 1), dtype=alpha_ref.dtype)
        else:
            if alpha_init.size == alpha_ref.shape[0]:
                alpha_init = alpha_init[:, np.newaxis]
            else:
                InputError(
                    "Initial alpha has size %d, but there are %d noise sources."
                    % (alpha_init.size, alpha_ref.shape[0])
                )

        tau_ref = np.concatenate((tau_init, tau_ref), axis=-1)
        alpha_ref = np.concatenate((alpha_init, alpha_ref), axis=-1)

        tstart = np.atleast_1d(ephemeris.ensure_unix(tstart))
        istart = np.digitize(self.time, tstart)

        if tend is not None:
            tend = np.atleast_1d(ephemeris.ensure_unix(tend))
            iend = np.digitize(self.time, tend)
        else:
            tend = tstart
            iend = istart

        coeff = np.full(self.time.size, 0.5, dtype=np.float32)
        for ts, te in zip(tstart, tend):
            if te > ts:
                fill = np.flatnonzero((self.time >= ts) & (self.time <= te))
                coeff[fill] = np.hanning(2 * fill.size - 1)[0 : fill.size]

        coeff = coeff[np.newaxis, :]
        tau_ref_full = coeff * tau_ref[:, istart] + (1.0 - coeff) * tau_ref[:, iend]
        alpha_ref_full = (
            coeff * alpha_ref[:, istart] + (1.0 - coeff) * alpha_ref[:, iend]
        )

        self.tau[:] = self.tau[:] - tau_ref_full
        self.alpha[:] = self.alpha[:] - alpha_ref_full

    def get_tau(self, timestamp, ignore_amp=False, interp="linear", extrap_limit=None):
        """Return the delay for each noise source at the requested time.

        Uses the scipy.interpolate.inpter1d function to interpolate over time.

        Parameters
        ----------
        timestamp:  np.ndarray[ntime, ]
            Unix timestamp.
        ignore_amp: bool
            Do not apply a noise source based amplitude correction, even if one exists.
        interp: string
            Method to interpolate over time.  Passed to scipy.interpolate.interp1d.
            Default is linear.
        extrap_limit: float
            Do not extrapolate the underlying data beyond its boundaries by this
            amount in seconds.  Default is 2 integrations.

        Returns
        -------
        tau: np.ndarray[nsource, ntime]
            Delay as a function of time for each of the noise sources.
        """
        if ignore_amp or (self.amp_to_delay is None) or not self.has_amplitude:

            tau = self._interp_time(
                self.tau[:], timestamp, interp=interp, extrap_limit=extrap_limit
            )

        else:

            logger.info(
                "Correcting delay template using amplitude template "
                "with coefficient %0.1f." % self.amp_to_delay
            )

            # Determine which input the delay template is referenced to
            iref = self.reference_input
            if iref is None:
                raise RuntimeError(
                    "Could not determine which input the delay template "
                    "is referenced with respect to."
                )

            # Subtract the referenced, scaled alpha template from the delay template
            tau_corrected = self.tau[:] - self.amp_to_delay * (
                self.alpha[:] - self.alpha[iref, np.newaxis, :]
            )

            # Interpolate to the requested times
            tau = self._interp_time(
                tau_corrected, timestamp, interp=interp, extrap_limit=extrap_limit
            )

        return tau

    def get_alpha(self, timestamp, interp="linear", extrap_limit=None):
        """Return the amplitude variation for each noise source at the requested time.

        Uses the scipy.interpolate.inpter1d function to interpolate over time.

        Parameters
        ----------
        timestamp:  np.ndarray[ntime, ]
            Unix timestamp.
        interp: string
            Method to interpolate over time.  Passed to scipy.interpolate.interp1d.
            Options include 'linear', 'nearest', 'zero', 'slinear', 'quadratic',
            'cubic', 'previous', 'next'.  Default is 'linear'.
        extrap_limit: float
            Do not extrapolate the underlying data beyond its boundaries by this
            amount in seconds.  Default is 2 integrations.

        Returns
        -------
        alpha: np.ndarray[nsource, ntime]
            Amplitude coefficient as a function of time for each of the noise sources.
        """
        alpha = self._interp_time(
            self.alpha[:], timestamp, interp=interp, extrap_limit=extrap_limit
        )
        return alpha

    def _interp_time(self, data, timestamp, interp="linear", extrap_limit=None):
        # Make sure we are not extrapolating too much
        dx_beg = self.time[0] - timestamp[0]
        dx_end = timestamp[-1] - self.time[-1]

        if extrap_limit is None:
            extrap_limit = 2.0 * np.median(np.diff(self.time))

        if (dx_beg > extrap_limit) or (dx_end > extrap_limit):
            raise ValueError("Requested times beyond span of TimingData.")

        interpolator = interp1d(
            self.time, data, axis=-1, kind=interp, fill_value="extrapolate"
        )
        data_interp = interpolator(timestamp)

        return data_interp

    def get_stacked_tau(
        self, timestamp, inputs, prod, stack_index, input_flags=None, **kwargs
    ):
        """Return the appropriate delay for each stacked visibility at the requested time.

        Averages the delays from the noise source inputs that map to the set of redundant
        baseline included in each stacked visibility.  This yields the appropriate
        common-mode delay correction.  If input_flags is provided, then the bad inputs
        that were excluded from the stack are also excluded from the delay template averaging
        (takes twice as long).

        Parameters
        ----------
        timestamp:  np.ndarray[ntime, ]
            Unix timestamp.
        inputs: np.ndarray[ninput, ]
            Must contain 'correlator_input' field.
        prod: np.ndarray[nprod]
            The products that were included in the stack.
            Typically found in the `index_map['prod']` attribute of the
            `andata.CorrData` object.
        stack_index: np.ndarray[nprod]
            The index of the stack axis that each product went into.
            Typically found in `reverse_map['stack']['stack']` attribute
            of the `andata.CorrData`.
        input_flags : np.ndarray [ninput, ntime]
            Array indicating which inputs were good at each time.
            Non-zero value indicates that an input was good.

        Returns
        -------
        tau: np.ndarray[nstack, ntime]
            Delay as a function of time for each stacked visibility.
        """
        return self._stack(
            self.get_tau,
            timestamp,
            inputs,
            prod,
            stack_index,
            input_flags=input_flags,
            **kwargs
        )

    def get_stacked_alpha(
        self, timestamp, inputs, prod, stack_index, input_flags=None, **kwargs
    ):
        """Return the equivalent of `get_stacked_tau` for the noise source amplitude variations.

        Averages the alphas from the noise source inputs that map to the set of redundant
        baseline included in each stacked visibility.  If input_flags is provided, then the
        bad inputs that were excluded from the stack are also excluded from the alpha
        template averaging (takes twice as long).  This method can be used to generate a stacked
        alpha template that can be used to correct a stacked tau template for variations in the
        noise source distribution system.  However, it is recommended that the tau template be
        corrected before stacking. This is accomplished by setting the `amp_to_delay` property
        prior to calling `get_stacked_tau`.

        Parameters
        ----------
        timestamp:  np.ndarray[ntime, ]
            Unix timestamp.
        inputs: np.ndarray[ninput, ]
            Must contain 'correlator_input' field.
        prod: np.ndarray[nprod]
            The products that were included in the stack.
            Typically found in the `index_map['prod']` attribute of the
            `andata.CorrData` object.
        stack_index: np.ndarray[nprod]
            The index of the stack axis that each product went into.
            Typically found in `reverse_map['stack']['stack']` attribute
            of the `andata.CorrData`.
        input_flags : np.ndarray [ninput, ntime]
            Array indicating which inputs were good at each time.
            Non-zero value indicates that an input was good.

        Returns
        -------
        alpha: np.ndarray[nstack, ntime]
            Noise source amplitude variation as a function of time for each stacked visibility.
        """
        return self._stack(
            self.get_alpha,
            timestamp,
            inputs,
            prod,
            stack_index,
            input_flags=input_flags,
            **kwargs
        )

    def _stack(
        self, func, timestamp, inputs, prod, stack_index, input_flags=None, **kwargs
    ):

        nstack = np.max(stack_index) + 1
        nprod = prod.size
        ntime = timestamp.size

        # Use the func provided to get the data for the noise source inputs
        data = func(timestamp, **kwargs)

        # Determine which noise source to use for each input
        index = np.array(map_input_to_noise_source(inputs, self.noise_source))

        # Sort the products based on the index of the stack axis they went into.
        isort = np.argsort(stack_index)
        sorted_stack_index = stack_index[isort]
        sorted_prod = prod[isort]

        # Create a new product axis that indicates the pair of noise sources
        # relevant for each sorted product.
        dt = [
            (native_str("stack"), "<u4"),
            (native_str("input_a"), "<u2"),
            (native_str("input_b"), "<u2"),
        ]
        new_prod = np.zeros(sorted_prod.size, dtype=dt)

        new_prod["stack"] = sorted_stack_index
        new_prod["input_a"] = index[sorted_prod["input_a"]]
        new_prod["input_b"] = index[sorted_prod["input_b"]]

        stacked_data = np.zeros((nstack, ntime), dtype=np.float64)

        # If input_flags was not provided, or if it is all True or all False, then we
        # assume all inputs are good and carry out a faster calculation.
        if (input_flags is None) or not np.any(input_flags) or np.all(input_flags):

            unique_prod, weight = np.unique(new_prod, return_counts=True)
            unique_stack, weight_norm = np.unique(new_prod["stack"], return_counts=True)

            weight = weight.astype(np.float32) * tools.invert_no_zero(
                weight_norm[unique_prod["stack"]].astype(np.float32)
            )

            for ii, up in enumerate(unique_prod):
                stacked_data[up["stack"], :] += weight[ii] * (
                    data[up["input_a"], :] - data[up["input_b"], :]
                )
        else:

            # Find boundaries into the sorted products that separate stacks.
            boundary = np.concatenate(
                (
                    np.atleast_1d(0),
                    np.flatnonzero(np.diff(sorted_stack_index) > 0) + 1,
                    np.atleast_1d(nprod),
                )
            )

            weight_norm = np.zeros((nstack, ntime), dtype=np.float64)

            for ss in range(nstack):

                prodo = sorted_prod[boundary[ss] : boundary[ss + 1]]
                prodn = new_prod[boundary[ss] : boundary[ss + 1]]

                unique_prod, rmap = np.unique(prodn, return_inverse=True)

                for ii, up in enumerate(unique_prod):
                    this_group = np.flatnonzero(rmap == ii)

                    ww = np.sum(
                        input_flags[prodo["input_a"][this_group], :]
                        * input_flags[prodo["input_b"][this_group], :],
                        axis=0,
                    )

                    weight_norm[up["stack"], :] += ww
                    stacked_data[up["stack"], :] += ww * (
                        data[up["input_a"], :] - data[up["input_b"], :]
                    )

            stacked_data = stacked_data * tools.invert_no_zero(weight_norm)

        return stacked_data

    def get_timing_correction(self, freq, timestamp, **kwargs):
        """Return the phase correction from each noise source at the requested frequency and time.

        Assumes the phase correction scales with frequency nu as phi = 2 pi nu tau and uses the
        get_tau method to interpolate over time. It acccepts and passes along keyword arguments
        for that method.

        Parameters
        ----------
        freq: np.ndarray[nfreq, ]
            Frequency in MHz.
        timestamp:  np.ndarray[ntime, ]
            Unix timestamp.

        Returns
        -------
        gain: np.ndarray[nfreq, nsource, ntime]
            Complex gain containing a pure phase correction for each of the noise sources.
        """
        tau = self.get_tau(timestamp, **kwargs)

        return np.exp(
            -1.0j
            * FREQ_TO_OMEGA
            * freq[:, np.newaxis, np.newaxis]
            * tau[np.newaxis, :, :]
        )

    def get_gain(self, freq, inputs, timestamp, **kwargs):
        """Return the complex gain for the requested frequencies, inputs, and times.

        Multiplying the visibilities by the outer product of these gains will remove
        the fluctuations in phase due to timing jitter.  This method uses the
        get_timing_correction method.  It acccepts and passes along keyword arguments
        for that method.

        Parameters
        ----------
        freq: np.ndarray[nfreq, ]
            Frequency in MHz.
        inputs: np.ndarray[ninput, ]
            Must contain 'correlator_input' field.
        timestamp: np.ndarray[ntime, ]
            Unix timestamps.

        Returns
        -------
        gain : np.ndarray[nfreq, ninput, ntime]
            Complex gain.  Multiplying the visibilities by the
            outer product of this vector at a given time and
            frequency will correct for the timing jitter.
        """
        # Determine which noise source to use for each input
        index = map_input_to_noise_source(inputs, self.noise_source)

        # Get the gain corrections for the requested times and frequencies
        gain = self.get_timing_correction(freq, timestamp, **kwargs)

        # Associate a set of gains to each input
        gain = gain[:, index, :]

        # Return gains
        return gain

    def apply_timing_correction(self, timestream, copy=False, **kwargs):
        """Apply the timing correction to another visibility dataset.

        This method uses the get_timing_correction method.  It acccepts and passes
        along keyword arguments for that method.

        Parameters
        ----------
        timestream : andata.CorrData / equivalent or np.ndarray[nfreq, nprod, ntime]
            If timestream is an np.ndarray containing the visiblities, then you
            must also pass the corresponding freq, prod, input, and time axis as kwargs.
            Otherwise these quantities are obtained from the attributes of CorrData.
            If the visibilities have been stacked, then you must additionally pass the
            stack and reverse_stack axis as kwargs, and (optionally) the input flags.
        copy : bool
            Create a copy of the input visibilities.  Apply the timing correction to
            the copy and return it, leaving the original untouched.  Default is False.
        freq : np.ndarray[nfreq, ]
            Frequency in MHz.
            Must be passed as keyword argument if timestream is an np.ndarray.
        prod: np.ndarray[nprod,  ]
            Product map.
            Must be passed as keyword argument if timestream is an np.ndarray.
        time: np.ndarray[ntime, ]
            Unix time.
            Must be passed as keyword argument if timestream is an np.ndarray.
        input: np.ndarray[ninput, ] of dtype=('chan_id', 'correlator_input')
            Input axis.
            Must be passed as keyword argument if timestream is an np.ndarray.
        stack : np.ndarray[nstack, ]
            Stack axis.
            Must be passed as keyword argument if timestream is an np.ndarray
            and the visibilities have been stacked.
        reverse_stack : np.ndarray[nprod, ]
            The index of the stack axis that each product went into.
            Typically found in `reverse_map['stack']['stack']` attribute.
            Must be passed as keyword argument if timestream is an np.ndarray
            and the visibilities have been stacked.
        input_flags : np.ndarray [ninput, ntime]
            Array indicating which inputs were good at each time.  Non-zero value
            indicates that an input was good.  Optional.  Only used for stacked visibilities.

        Returns
        -------
        If copy == True:
            vis : np.ndarray[nfreq, nprod(nstack), ntime]
                New set of visibilities with timing correction applied.
        else:
            None
                Correction is applied to the input visibility data.  Also,
                if timestream is an andata.CorrData instance and the gain dataset exists,
                then it will be updated with the complex gains that have been applied.
        """
        if isinstance(timestream, np.ndarray):
            is_obj = False

            vis = timestream if not copy else timestream.copy()

            freq = kwargs.pop("freq")
            prod = kwargs.pop("prod")
            inputs = kwargs.pop("input")
            timestamp = kwargs.pop("time")
            stack = kwargs.pop("stack", None)
            reverse_stack = kwargs.pop("reverse_stack", None)

        else:
            is_obj = True

            vis = timestream.vis[:] if not copy else timestream.vis[:].copy()

            freq = kwargs.pop("freq") if "freq" in kwargs else timestream.freq[:]
            prod = (
                kwargs.pop("prod")
                if "prod" in kwargs
                else timestream.index_map["prod"][:]
            )
            inputs = kwargs.pop("input") if "input" in kwargs else timestream.input[:]
            timestamp = kwargs.pop("time") if "time" in kwargs else timestream.time[:]
            stack = (
                kwargs.pop("stack")
                if "stack" in kwargs
                else timestream.index_map["stack"][:]
            )
            reverse_stack = (
                kwargs.pop("reverse_stack")
                if "reverse_stack" in kwargs
                else timestream.reverse_map["stack"]["stack"][:]
            )

        input_flags = kwargs.pop("input_flags", None)

        # Determine if the visibilities have been stacked
        is_stack = (
            (stack is not None)
            and (reverse_stack is not None)
            and (stack.size < prod.size)
        )

        if is_stack:
            logger.info("Applying timing correction to stacked data.")
            # Visibilities have been stacked.  Stack the timing correction
            # before applying it.  Application is done for each frequency in this case.
            tau = self.get_stacked_tau(
                timestamp,
                inputs,
                prod,
                reverse_stack,
                input_flags=input_flags,
                **kwargs
            )

            # Loop over local frequencies and apply the timing correction
            for ff in range(freq.size):
                vis[ff] *= np.exp(-1.0j * FREQ_TO_OMEGA * freq[ff] * tau)

        else:
            logger.info("Applying timing correction to unstacked data.")
            # Visibilities have not been stacked yet.  Use the timing correction as is.
            # Determine which noise source to use for each input
            index = map_input_to_noise_source(inputs, self.noise_source)

            # Get the gain corrections for the times and frequencies in timestream
            gain = self.get_timing_correction(freq, timestamp, **kwargs)

            # Loop over products and apply the timing correction
            for ii, pp in enumerate(prod):

                aa, bb = index[pp[0]], index[pp[1]]

                if aa != bb:
                    vis[:, ii, :] *= gain[:, aa, :] * gain[:, bb, :].conj()

            # If andata object was input then update the gain
            # dataset so that we have record of what was done
            if is_obj and not copy and hasattr(timestream, "gain"):
                timestream.gain[:] *= gain[:, index, :]

        # If a copy was requested, then return the
        # new vis with phase correction applied
        if copy:
            return vis

    def summary(self):
        """Provide a summary of the timing correction.

        Returns
        -------
        summary : list of strings
            Contains useful information about the timing correction.
            Specifically contains for each noise source input the
            time averaged  phase offset and delay.  Also contains
            estimates of the variance in the timing for both the
            shortest and longest timescale probed by the underlying
            dataset.  Meant to be joined with new lines and printed.
        """
        span = (self.time[-1] - self.time[0]) / 3600.0
        sig_tau = np.std(self.tau[:], axis=-1)

        step = np.median(np.diff(self.time))
        sig2_tau = np.sqrt(
            np.sum(np.diff(self.tau[:], axis=-1) ** 2, axis=-1)
            / (2.0 * (self.tau.shape[-1] - 1.0))
        )

        fmt = "%-10s %10s %10s %15s %15s"
        hdr = fmt % ("", "PHI0", "TAU0", "SIGMA(TAU)", "SIGMA(TAU)")
        per = fmt % ("", "", "", "@ %0.2f sec" % step, "@ %0.2f hr" % span)
        unt = fmt % ("INPUT", "[rad]", "[nsec]", "[psec]", "[psec]")
        line = "".join(["-"] * 65)
        summary = [line, hdr, per, unt, line]

        fmt = "%-10s %10.2f %10.2f %15.2f %15.2f"
        for ii, inp in enumerate(self.noise_source):
            summary.append(
                fmt
                % (
                    inp["correlator_input"],
                    self.static_phi_fit[0, ii],
                    self.static_phi_fit[1, ii] * 1e-3,
                    sig2_tau[ii],
                    sig_tau[ii],
                )
            )

        return summary

    def __repr__(self):
        """Return a summary of the timing correction nicely formatted for printing.

        Calls the method summary and joins the list of strings with new lines.
        """
        summary = self.summary()
        summary.insert(0, self.__class__.__name__)

        return "\n".join(summary)


class TimingData(andata.CorrData, TimingCorrection):
    """Subclass of :class:`andata.CorrData` for timing data.

    Automatically computes the timing correction when data is loaded and
    inherits the methods of :class:`TimingCorrection` that enable the application
    of that correction to other datasets.

    Methods
    -------
    from_acq_h5
    """

    @classmethod
    def from_acq_h5(cls, acq_files, only_correction=False, **kwargs):
        """Load a list of acquisition files and computes the timing correction.

        Accepts and passes on all keyword arguments for andata.CorrData.from_acq_h5
        and the construct_delay_template function.

        Parameters
        ----------
        acq_files: str or list of str
            Path to file(s) containing the timing data.
        only_correction: bool
            Only return the timing correction.  Do not return the underlying
            data from which that correction was derived.

        Returns
        -------
        data: TimingData or TimingCorrection
        """
        # Separate the kwargs for construct_delay_template.  This is necessary
        # because andata will not accept extraneous kwargs.
        insp = inspect.getargspec(construct_delay_template)
        cdt_kwargs_list = set(insp[0][-len(insp[-1]) :]) & set(kwargs)

        cdt_kwargs = {}
        for name in cdt_kwargs_list:
            cdt_kwargs[name] = kwargs.pop(name)

        # Change some of the default parameters for CorrData.from_acq_h5 to reflect
        # the fact that this data will be used to compute a timing correction.
        apply_gain = kwargs.pop("apply_gain", False)
        datasets = kwargs.pop("datasets", ["vis", "flags/vis_weight"])

        # Load the data into an andata.CorrData object
        corr_data = super(TimingData, cls).from_acq_h5(
            acq_files, apply_gain=apply_gain, datasets=datasets, **kwargs
        )

        # Instantiate a TimingCorrection or TimingData object
        dist_kwargs = {"distributed": corr_data.distributed, "comm": corr_data.comm}
        data = (
            TimingCorrection(**dist_kwargs)
            if only_correction
            else TimingData(**dist_kwargs)
        )

        # Redefine input axis to contain only noise sources
        isource = np.unique(corr_data.prod.tolist())
        inputs = corr_data.input[isource]
        data.create_index_map("input", inputs)

        # Copy over relevant data to the newly instantiated object
        if only_correction:
            # We are only returning a correction, so we only need to
            # copy over a subset of index_map.
            for name in ["time", "freq"]:
                data.create_index_map(name, corr_data.index_map[name][:])

        else:
            # We are returning the data in addition to the correction.
            # Redefine prod axis to contain only noise sources.
            prod = np.zeros(corr_data.prod.size, dtype=corr_data.prod.dtype)
            prod["input_a"] = andata._search_array(isource, corr_data.prod["input_a"])
            prod["input_b"] = andata._search_array(isource, corr_data.prod["input_b"])
            data.create_index_map("prod", prod)

            # Copy over remaining index maps
            for name, index_map in corr_data.index_map.items():
                if name not in data.index_map:
                    data.create_index_map(name, index_map[:])

            # Copy over the attributes
            memh5.copyattrs(corr_data.attrs, data.attrs)

            # Iterate over the datasets and copy them over
            for name, old_dset in corr_data.datasets.items():
                new_dset = data.create_dataset(
                    name, data=old_dset[:], distributed=old_dset.distributed
                )
                memh5.copyattrs(old_dset.attrs, new_dset.attrs)

            # Iterate over the flags and copy them over
            for name, old_dset in corr_data.flags.items():
                new_dset = data.create_flag(
                    name, data=old_dset[:], distributed=old_dset.distributed
                )
                memh5.copyattrs(old_dset.attrs, new_dset.attrs)

        # Construct delay template
        res = construct_delay_template(corr_data, **cdt_kwargs)
        (
            tau,
            static_phi,
            w_static_phi,
            static_phi_fit,
            alpha,
            static_amp,
            w_static_amp,
        ) = res

        # Create datasets containing the timing correction
        param = ["intercept", "slope", "quad", "cube", "quart", "quint"]
        param = param[0 : static_phi_fit.shape[0]]
        data.create_index_map("param", np.array(param, dtype=np.string_))

        dset = data.create_dataset("tau", data=tau)
        dset.attrs["axis"] = np.array(["input", "time"], dtype=np.string_)

        dset = data.create_dataset("static_phi", data=static_phi)
        dset.attrs["axis"] = np.array(["freq", "input"], dtype=np.string_)

        dset = data.create_flag("weight_static_phi", data=w_static_phi)
        dset.attrs["axis"] = np.array(["freq", "input"], dtype=np.string_)

        dset = data.create_dataset("static_phi_fit", data=static_phi_fit)
        dset.attrs["axis"] = np.array(["param", "input"], dtype=np.string_)

        dset = data.create_dataset("alpha", data=alpha)
        dset.attrs["axis"] = np.array(["input", "time"], dtype=np.string_)

        dset = data.create_dataset("static_amp", data=static_amp)
        dset.attrs["axis"] = np.array(["freq", "input"], dtype=np.string_)

        dset = data.create_flag("weight_static_amp", data=w_static_amp)
        dset.attrs["axis"] = np.array(["freq", "input"], dtype=np.string_)

        # Delete the temporary corr_data object
        del corr_data
        gc.collect()

        # Return timing data object
        return data

    def summary(self):
        """Provide a summary of the timing data and correction.

        Returns
        -------
        summary : list of strings
            Contains useful information about the timing correction
            and data.  Includes the reduction in the standard deviation
            of the phase after applying the timing correction.  This is
            presented as quantiles over frequency for each of the
            noise source products.
        """
        summary = super(TimingData, self).summary()

        vis = self.apply_timing_correction(
            self.vis[:],
            copy=True,
            freq=self.freq,
            time=self.time,
            prod=self.prod,
            input=self.input,
        )

        phi_before = np.angle(self.vis[:])
        phi_after = np.angle(vis)

        phi_before = _correct_phase_wrap(
            phi_before - np.median(phi_before, axis=-1)[..., np.newaxis]
        )
        phi_after = _correct_phase_wrap(
            phi_after - np.median(phi_after, axis=-1)[..., np.newaxis]
        )

        sig_before = np.median(
            np.abs(phi_before - np.median(phi_before, axis=-1)[..., np.newaxis]),
            axis=-1,
        )
        sig_after = np.median(
            np.abs(phi_after - np.median(phi_after, axis=-1)[..., np.newaxis]), axis=-1
        )

        ratio = sig_before * tools.invert_no_zero(sig_after)

        stats = np.percentile(ratio, [0, 25, 50, 75, 100], axis=0)

        fmt = "%-23s %5s %5s %8s %5s %5s"
        hdr1 = "Factor Reduction in RMS Phase Noise (Quantiles Over Frequency)"
        hdr2 = fmt % ("PRODUCT", "MIN", "25%", "MEDIAN", "75%", "MAX")
        line = "".join(["-"] * 65)
        summary += ["", line, hdr1, hdr2, line]

        fmt = "%-10s x %-10s %5d %5d %8d %5d %5d"
        for ii, pp in enumerate(self.prod):
            if pp[0] != pp[1]:
                summary.append(
                    fmt
                    % (
                        (
                            self.input[pp[0]]["correlator_input"],
                            self.input[pp[1]]["correlator_input"],
                        )
                        + tuple(stats[:, ii])
                    )
                )

        return summary


def load_timing_correction(
    files, start=None, stop=None, window=43200.0, instrument="chime", **kwargs
):
    """Find and load the appropriate timing correction for a list of corr acquisition files.

    For example, if the instrument keyword is set to 'chime',
    then this function will accept all types of chime corr acquisition files,
    such as 'chimetiming', 'chimepb', 'chimeN2', 'chimecal', and then find
    the relevant set of 'chimetiming' files to load.

    Accepts and passes on all keyword arguments for the functions
    andata.CorrData.from_acq_h5 and construct_delay_template.

    Should consider modifying this method to use Finder at some point in future.

    Parameters
    ----------
    files : string or list of strings
        Absolute path to corr acquisition file(s).
    start : integer, optional
        What frame to start at in the full set of files.
    stop : integer, optional
        What frame to stop at in the full set of files.
    window : float
        Use the timing data -window from start and +window from stop.
        Default is 12 hours.
    instrument : string
        Name of the instrument.  Default is 'chime'.

    Returns
    -------
    data: TimingData
    """
    files = np.atleast_1d(files)

    # Check that a single acquisition was requested
    input_dirs = [os.path.dirname(ff) for ff in files]
    if len(set(input_dirs)) > 1:
        raise RuntimeError("Input files span multiple acquisitions!")

    # Extract relevant information from the filename
    node = os.path.dirname(input_dirs[0])
    acq = os.path.basename(input_dirs[0])

    acq_date, acq_inst, acq_type = acq.split("_")
    if not acq_inst.startswith(instrument) or (acq_type != "corr"):
        raise RuntimeError(
            "This function is only able to parse corr type files "
            "from the specified instrument (currently %s)." % instrument
        )

    # Search for all timing acquisitions on this node
    tdirs = sorted(
        glob.glob(os.path.join(node, "_".join(["*", instrument + "timing", acq_type])))
    )
    if not tdirs:
        raise RuntimeError("No timing acquisitions found on node %s." % node)

    # Determine the start time of the requested acquistion and the available timing acquisitions
    acq_start = ephemeris.datetime_to_unix(ephemeris.timestr_to_datetime(acq_date))

    tacq_start = np.array(
        [ephemeris.timestr_to_datetime(os.path.basename(tt)) for tt in tdirs]
    )
    tacq_start = ephemeris.datetime_to_unix(tacq_start)

    # Find the closest timing acquisition to the requested acquisition
    iclose = np.argmin(np.abs(acq_start - tacq_start))
    if np.abs(acq_start - tacq_start[iclose]) > 60.0:
        raise RuntimeError("Cannot find appropriate timing acquisition for %s." % acq)

    # Grab all timing files from this acquisition
    tfiles = sorted(glob.glob(os.path.join(tdirs[iclose], "*.h5")))

    tdata = andata.CorrData.from_acq_h5(tfiles, datasets=())

    # Find relevant span of time
    data = andata.CorrData.from_acq_h5(files, start=start, stop=stop, datasets=())

    time_start = data.time[0] - window
    time_stop = data.time[-1] + window

    tstart = int(np.argmin(np.abs(time_start - tdata.time)))
    tstop = int(np.argmin(np.abs(time_stop - tdata.time)))

    # Load into TimingData object
    data = TimingData.from_acq_h5(tfiles, start=tstart, stop=tstop, **kwargs)

    return data


# ancillary functions
# -------------------


def construct_delay_template(
    data,
    check_amp=True,
    check_sig=True,
    nsigma=5.0,
    threshold=0.50,
    nparam=2,
    min_freq=420.0,
    max_freq=780.0,
    max_iter_weight=2,
    static_phi=None,
    weight_static_phi=None,
    static_phi_fit=None,
    static_amp=None,
    weight_static_amp=None,
):
    """Construct a relative time delay template.

    Fits the phase of the cross-correlation between noise source inputs
    to a model that increases linearly with frequency.

    Parameters
    ----------
    data: andata.CorrData
        Correlation data.  Must contain the following attributes:
            freq: np.ndarray[nfreq, ]
                Frequency in MHz.
            vis: np.ndarray[nfreq, nprod, ntime]
                Upper-triangle, product packed visibility matrix
                containing ONLY the noise source inputs.
            weight: np.ndarray[nfreq, nprod, ntime]
                Flag indicating the data points to fit.
    check_amp: bool
        Do not include frequencies and times where the
        square root of the autocorrelations is an outlier.
        Default is True.
    check_sig: bool
        Do not include frequencies and times where the
        square root of the inverse weight is an outlier.
        Default is True.
    nsigma: float
        Number of median absolute deviations to consider
        a data point an outlier in the checked specified above.
        Default is 5.0.
    threshold: float
        A (frequency, input) must pass the checks specified above
        more than this fraction of the time,  otherwise it will be
        flaged as bad for all times.  Default is 0.50.
    nparam: int
        Number of parameters for polynomial fit to the
        time averaged phase versus frequency.  Default is 2.
    min_freq: float
        Minimum frequency in MHz to include in the fit.
        Default is 420.
    max_freq: float
        Maximum frequency in MHz to include in the fit.
        Default is 780.
    max_iter_weight: int
        The weight for each frequency is estimated from the variance of the
        residuals of the template fit from the previous iteration.  This
        is the total number of times to iterate.  Setting to 0 corresponds
        to linear least squares.  Default is 2.
    static_phi: np.ndarray[nfreq, ninput]
        Subtract this quantity from the noise source phase prior to fitting
        for the timing correction.  If None, then this will be estimated from the median
        of the noise source phase over time.
    weight_static_phi: np.ndarray[nfreq, ninput]
        Inverse variance of the time averaged phased.  Set to zero for frequencies and inputs
        that are missing or should be ignored.  If None, then this will be estimated from the
        residuals of the fit.
    static_phi_fit: np.ndarray[nparam, ninput]
        Polynomial fit to static_phi versus frequency.
    static_amp: np.ndarray[nfreq, ninput]
        Subtract this quantity from the noise source amplitude prior to fitting
        for the amplitude variations.  If None, then this will be estimated from the median
        of the noise source amplitude over time.
    weight_static_amp: np.ndarray[nfreq, ninput]
        Inverse variance of the time averaged amplitude.  Set to zero for frequencies and inputs
        that are missing or should be ignored.  If None, then this will be estimated from the
        residuals of the fit.

    Returns
    -------
    tau: np.ndarray[ninput, ntime]
        Delay template for each noise source input.
    static_phi: np.ndarray[nfreq, ninput]
        Time averaged phase versus frequency.
    weight_static_phi: np.ndarray[nfreq, ninput]
       Inverse variance of the time averaged phase.
    static_phi_fit: np.ndarray[nparam, ninput]
        Best-fit parameters of the polynomial fit to the
        time averaged phase versus frequency.
    alpha: np.ndarray[ninput, ntime]
        Amplitude coefficient for each noise source input.
    static_amp: np.ndarray[nfreq, ninput]
        Time averaged amplitude versus frequency.
    weight_static_amp: np.ndarray[nfreq, ninput]
        Inverse variance of the time averaged amplitude.
    """
    # Check if we are distributed.  If so make sure we are distributed over time.
    parallel = isinstance(data.vis, memh5.MemDatasetDistributed)
    if parallel:
        data.redistribute("time")
        comm = data.vis.comm

    # Extract relevant datasets
    freq = data.freq[:]
    vis = data.vis[:].view(np.ndarray)
    weight = data.weight[:].view(np.ndarray)

    # Check dimensions
    nfreq, nprod, ntime = vis.shape
    nsource = int((np.sqrt(8 * nprod + 1) - 1) // 2)
    ilocal = range(0, nsource)

    assert nfreq == freq.size
    assert nsource >= 2
    assert nparam >= 2

    if static_phi is not None:
        static_phi, sphi_shp, sphi_ind = _resolve_distributed(static_phi, axis=1)
        assert sphi_shp == (nfreq, nsource)

    if weight_static_phi is not None:
        weight_static_phi, wsphi_shp, wsphi_ind = _resolve_distributed(
            weight_static_phi, axis=1
        )
        assert wsphi_shp == (nfreq, nsource)

    if static_phi_fit is not None:
        static_phi_fit, sphifit_shp, sphifit_ind = _resolve_distributed(
            static_phi_fit, axis=1
        )
        assert sphifit_shp == (nparam, nsource)

    if static_amp is not None:
        static_amp, samp_shp, samp_ind = _resolve_distributed(static_amp, axis=1)
        assert samp_shp == (nfreq, nsource)

    if weight_static_amp is not None:
        weight_static_amp, wsamp_shp, wsamp_ind = _resolve_distributed(
            weight_static_amp, axis=1
        )
        assert wsamp_shp == (nfreq, nsource)

    # Compute amplitude of noise source signal from autocorrelation
    iauto = np.array([int(k * (2 * nsource - k + 1) // 2) for k in range(nsource)])

    amp = np.sqrt(vis[:, iauto, :].real)

    # Determine which data points to fit
    flg = amp > 0.0
    if weight is not None:
        flg &= weight[:, iauto, :] > 0.0

    # If requested discard frequencies and times that are outliers in the
    # amplitude or variance of the autocorrelation.
    if check_amp or check_sig:

        if check_amp:

            mu = np.median(amp[flg])
            sigma = 1.48625 * np.median(np.abs(amp[flg] - mu))

            good_amp = np.abs(amp - mu) < (nsigma * sigma)
            good_amp *= (
                (np.sum(good_amp, axis=-1) / float(good_amp.shape[-1])) > threshold
            )[:, :, np.newaxis]

            flg &= good_amp

        if check_sig and (weight is not None):
            wsig = tools.invert_no_zero(np.sqrt(weight[:, iauto, :]))

            mu = np.median(wsig[flg])
            sigma = 1.48625 * np.median(np.abs(wsig[flg] - mu))

            good_wsig = np.abs(wsig - mu) < (nsigma * sigma)
            good_wsig *= (
                (np.sum(good_wsig, axis=-1) / float(good_wsig.shape[-1])) > threshold
            )[:, :, np.newaxis]

            flg &= good_wsig

    # Restrict the range of frequencies that are fit to avoid bandpass edges
    flg &= ((freq > min_freq) & (freq < max_freq))[:, np.newaxis, np.newaxis]

    flg = flg.astype(np.float32)

    # If we only have two noise source inputs, then we use the cross-correlation
    # between them to characterize their relative response to the noise source signal.
    # If we have more than two noise source inputs, then we perform an eigenvalue
    # decomposition of the cross-correlation matrix to obtain an improved estimate
    # of the response of each input to the noise source signal.
    if nsource > 2:

        response = eigen_decomposition(vis, flg)

        phi = np.angle(response)
        amp = np.abs(response)

        wwp = flg

    else:

        phi = np.zeros((nfreq, nsource, ntime), dtype=np.float32)
        phi[:, 1, :] = np.angle(vis[:, 1, :].conj())

        amp = np.sqrt(vis[:, iauto, :].real)

        wwp = np.repeat(flg[:, 0, np.newaxis, :] * flg[:, 1, np.newaxis, :], 2, axis=1)

    # If parallelized we need to redistribute over inputs for the
    # operations below, which require full frequency and time coverage.
    if parallel:
        amp = mpiarray.MPIArray.wrap(amp, axis=2, comm=comm)
        phi = mpiarray.MPIArray.wrap(phi, axis=2, comm=comm)
        wwp = mpiarray.MPIArray.wrap(flg, axis=2, comm=comm)

        amp = amp.redistribute(1)
        phi = phi.redistribute(1)
        wwp = wwp.redistribute(1)

        nsource = amp.local_shape[1]
        ilocal = range(amp.local_offset[1], amp.local_offset[1] + nsource)

        amp = amp[:].view(np.ndarray)
        phi = phi[:].view(np.ndarray)
        wwp = wwp[:].view(np.ndarray)

    wwa = wwp.copy()

    # Subtract the median value over time for each frequency
    if static_phi is None:
        static_phi = _flagged_median(phi, wwp, axis=-1)
    else:
        sphi_ind = np.array([sphi_ind.index(ilcl) for ilcl in ilocal])
        static_phi = static_phi[:, sphi_ind]

    if weight_static_phi is not None:
        wsphi_ind = np.array([wsphi_ind.index(ilcl) for ilcl in ilocal])
        weight_static_phi = weight_static_phi[:, wsphi_ind]
        wwp *= (weight_static_phi[:, :, np.newaxis] > 0.0).astype(np.float32)

    if static_amp is None:
        static_amp = _flagged_median(amp, wwa, axis=-1)
    else:
        samp_ind = np.array([samp_ind.index(ilcl) for ilcl in ilocal])
        static_amp = static_amp[:, samp_ind]

    if weight_static_amp is not None:
        wsamp_ind = np.array([wsamp_ind.index(ilcl) for ilcl in ilocal])
        weight_static_amp = weight_static_amp[:, wsamp_ind]
        wwa *= (weight_static_amp[:, :, np.newaxis] > 0.0).astype(np.float32)

    phi = _correct_phase_wrap(phi - static_phi[:, :, np.newaxis])
    amp = amp * tools.invert_no_zero(static_amp[:, :, np.newaxis]) - 1.0

    # Fit frequency dependence of phase
    # ---------------------------------
    # Construct initial delay template
    omega = FREQ_TO_OMEGA * freq[:, np.newaxis, np.newaxis]

    tau = np.sum(wwp * omega * phi, axis=0) * tools.invert_no_zero(
        np.sum(wwp * omega ** 2, axis=0)
    )

    # Estimate variance of each frequency from residuals
    inv_var_weight = wwp
    for iter_weight in range(max_iter_weight):

        residual_var = np.sum(
            wwp * (phi - omega * tau[np.newaxis, :, :]) ** 2, axis=-1
        ) * tools.invert_no_zero(np.sum(wwp, axis=-1))

        inv_var_weight = wwp * tools.invert_no_zero(residual_var[:, :, np.newaxis])

        tau = np.sum(inv_var_weight * omega * phi, axis=0) * tools.invert_no_zero(
            np.sum(inv_var_weight * omega ** 2, axis=0)
        )

    # Determine the uncertainty on the static_phi
    if weight_static_phi is None:
        weight_static_phi = np.sum(inv_var_weight, axis=-1)

    # Calculate the average delay over this period using non-linear
    # least squares that is insensitive to phase wrapping
    if static_phi_fit is None:

        err_static_phi = np.sqrt(tools.invert_no_zero(weight_static_phi))

        static_phi_fit = np.zeros((nparam, nsource), dtype=np.float64)
        for nn in range(nsource):
            if np.any(err_static_phi[:, nn] > 0.0):
                static_phi_fit[:, nn] = fit_poly_to_phase(
                    freq,
                    np.exp(1.0j * static_phi[:, nn]),
                    err_static_phi[:, nn],
                    nparam=nparam,
                )[0]
    else:
        sphifit_ind = np.array([sphifit_ind.index(ilcl) for ilcl in ilocal])
        static_phi_fit = static_phi_fit[:, sphifit_ind]

    # Fit frequency dependence of amplitude
    # -------------------------------------
    # Construct initial amplitude coefficient
    asc = _amplitude_scaling(freq[:, np.newaxis, np.newaxis])

    alpha = np.sum(wwa * asc * amp, axis=0) * tools.invert_no_zero(
        np.sum(wwa * asc ** 2, axis=0)
    )

    # Estimate variance of each frequency from residuals
    inv_var_weight = wwa
    for iter_weight in range(max_iter_weight):

        residual_var = np.sum(
            wwa * (amp - asc * alpha[np.newaxis, :, :]) ** 2, axis=-1
        ) * tools.invert_no_zero(np.sum(wwa, axis=-1))

        inv_var_weight = wwa * tools.invert_no_zero(residual_var[:, :, np.newaxis])

        alpha = np.sum(inv_var_weight * asc * amp, axis=0) * tools.invert_no_zero(
            np.sum(inv_var_weight * asc ** 2, axis=0)
        )

    # Determine the uncertainty on the static_amp
    if weight_static_amp is None:
        weight_static_amp = np.sum(inv_var_weight, axis=-1)

    # Convert the outputs to MPIArrays distributed over input
    if parallel:
        tau = mpiarray.MPIArray.wrap(tau, axis=0, comm=comm)
        alpha = mpiarray.MPIArray.wrap(alpha, axis=0, comm=comm)

        static_phi = mpiarray.MPIArray.wrap(static_phi, axis=1, comm=comm)
        weight_static_phi = mpiarray.MPIArray.wrap(weight_static_phi, axis=1, comm=comm)
        static_phi_fit = mpiarray.MPIArray.wrap(static_phi_fit, axis=1, comm=comm)

        static_amp = mpiarray.MPIArray.wrap(static_amp, axis=1, comm=comm)
        weight_static_amp = mpiarray.MPIArray.wrap(weight_static_amp, axis=1, comm=comm)

        data.redistribute("freq")

    # Return results
    return (
        tau,
        static_phi,
        weight_static_phi,
        static_phi_fit,
        alpha,
        static_amp,
        weight_static_amp,
    )


def map_input_to_noise_source(inputs, noise_sources):
    """Find the appropriate noise source to use to correct the phase of each input.

    Searches for a noise source connected to the same slot,
    then crate, then hut, then correlator.

    Parameters
    ----------
    inputs:  np.ndarray[ninput, ] of dtype=('chan_id', 'correlator_input')
        The input axis from a data acquisition file.
    noise_sources: np.ndarray[nsource, ] of dtype=('chan_id', 'correlator_input')
        The noise sources.
    """
    # Define functions
    def parse_serial(input_serial):
        # Have to distinguish between CHIME WRH and ERH
        # Otherwise serial numbers already have the
        # desired hierarchical structure.

        # Serial from file is often bytes, ensure it is unicode
        if not isinstance(input_serial, str):
            input_serial = input_serial.decode("utf-8")

        if input_serial.startswith("FCC"):
            if int(input_serial[3:5]) < 4:
                name = "FCCW" + input_serial[3:]
            else:
                name = "FCCE" + input_serial[3:]
        else:
            name = input_serial

        return name

    def count_startswith(x, y):

        cnt = 0
        for ii in range(min(len(x), len(y))):
            if x[ii] == y[ii]:
                cnt += 1
            else:
                break

        return cnt

    # Create hierarchical identifier from serial number for the
    # noise sources and requested inputs
    input_names = list(map(parse_serial, inputs["correlator_input"]))
    source_names = list(map(parse_serial, noise_sources["correlator_input"]))

    # Map each input to a noise source
    imap = [
        np.argmax([count_startswith(inp, src) for src in source_names])
        for inp in input_names
    ]

    return imap


def eigen_decomposition(vis, flag=None):
    """Eigenvalue decomposition of the visibility matrix.

    Parameters
    ----------
    vis: np.ndarray[nfreq, nprod, ntime]
        Upper-triangle, product packed visibility matrix.
    flag: np.ndarray[nfreq, ninput, ntime] (optional)
        Array of 1 or 0 indicating the inputs that should be included
        in the eigenvalue decomposition for each frequency and time.

    Returns
    -------
    resp: np.ndarray[nfreq, ninput, ntime]
        Eigenvector corresponding to the largest eigenvalue for
        each frequency and time.
    """
    nfreq, nprod, ntime = vis.shape
    nsource = int((np.sqrt(8 * nprod + 1) - 1) // 2)

    if flag is None:
        flag = np.ones((nfreq, nsource, ntime), dtype=np.float32)
    else:
        flag = flag.astype(np.float32)

    M = tools.unpack_product_array(vis, axis=1)

    resp = np.zeros((nfreq, nsource, ntime), dtype=vis.dtype)

    for tt in range(ntime):

        for ff in range(nfreq):

            wx = flag[ff, :, tt]

            if np.any(wx):

                Q = np.outer(wx, wx) * M[ff, :, :, tt]

                # Solve for eigenvectors and eigenvalues
                evals, evecs = eigh(Q)

                # Construct response
                sign0 = 1.0 - 2.0 * (evecs[0, -1].real < 0.0)

                resp[ff, :, tt] = wx * sign0 * evecs[:, -1] * evals[-1] ** 0.5

    return resp


def fit_poly_to_phase(freq, resp, resp_error, nparam=2):
    """Fit complex data versus frequency to a model consisting of a polynomial in phase.

    Nonlinear least squares algorithm is applied to the complex data to avoid problems
    caused by phase wrapping.

    Parameters
    ----------
    freq: np.ndarray[nfreq, ]
        Frequency in MHz.
    resp: np.ndarray[nfreq, ]
        Complex data with magnitude equal to 1.0.
    resp_error: np.ndarray[nfreq, ]
        Uncertainty on the complex data.
    nparam: int
        Number of parameters in the polynomial.
        Default is 2 (i.e, linear).

    Returns
    -------
    popt: np.ndarray[nparam, ]
        Best-fit parameters.
    pcov: np.ndarray[nparam, nparam]
        Covariance of the best-fit parameters.
        Assumes that it obtained a good fit
        and returns the errors
        necessary to achieve that.
    """
    flg = np.flatnonzero(resp_error > 0.0)

    if flg.size < (nparam + 1):
        msg = (
            "Number of data points must be greater than number of parameters (%d)."
            % nparam
        )
        raise RuntimeError(msg)

    # We will fit the complex data.  Break n-element complex array g(ra)
    # into 2n-element real array [Re{g(ra)}, Im{g(ra)}] for fit.
    y_complex = resp[flg]
    y = np.concatenate((y_complex.real, y_complex.imag))

    x = np.tile(freq[flg], 2)

    err = np.tile(resp_error[flg], 2)

    # Initial guess for parameters
    p0 = np.zeros(nparam, dtype=np.float32)
    p0[1] = np.median(
        np.diff(np.angle(y_complex)) / (FREQ_TO_OMEGA * np.diff(freq[flg]))
    )

    # Try nonlinear least squares fit
    try:
        popt, pcov = curve_fit(
            _func_poly_phase, x, y, p0=p0, sigma=err, absolute_sigma=False
        )

    except Exception as excep:
        raise RuntimeError("Nonlinear phase fit failed with error:  %s" % excep)

    return popt, pcov


def model_poly_phase(freq, *param):
    """Evaluate a polynomial model for the phase.

    To be used with the parameters output from fit_poly_to_phase.

    Parameters
    ----------
    freq: np.ndarray[nfreq, ]
        Frequency in MHz.
    *param: float
        Coefficients of the polynomial.

    Returns
    -------
    phi: np.ndarray[nfreq, ]
        Phase in radians between -pi and +pi.
    """
    x = FREQ_TO_OMEGA * freq

    model_phase = np.zeros_like(freq)
    for pp, par in enumerate(param):
        model_phase += par * x ** pp

    model_phase = model_phase % (2.0 * np.pi)
    model_phase -= 2.0 * np.pi * (model_phase > np.pi)

    return model_phase


# private functions
# -----------------
def _amplitude_scaling(freq):
    return np.sqrt(freq / FREQ_PIVOT)


def _flagged_median(data, flag, axis=0, keepdims=False):
    bflag = flag.astype(np.bool)
    nandata = np.full(data.shape, np.nan, dtype=data.dtype)
    nandata[bflag] = data[bflag]

    sortdata = np.sort(nandata, axis=axis)
    ieval = np.sum(np.isfinite(sortdata), axis=axis, dtype=np.int, keepdims=True) // 2

    med = np.zeros(ieval.shape, dtype=sortdata.dtype)
    for aind, sind in np.ndenumerate(ieval):

        find = list(aind)
        find[axis] = sind

        sdata = sortdata[tuple(find)]
        if np.isfinite(sdata):
            med[aind] = sdata

    if not keepdims:
        med = np.squeeze(med, axis=axis)

    return med


def _func_poly_phase(freq, *param):

    nreal = len(freq) // 2

    x = FREQ_TO_OMEGA * freq[:nreal]

    model_phase = np.zeros(x.size, dtype=x.dtype)
    for pp, par in enumerate(param):
        model_phase += par * x ** pp

    return np.concatenate((np.cos(model_phase), np.sin(model_phase)))


def _correct_phase_wrap(phi):
    return ((phi + np.pi) % (2.0 * np.pi)) - np.pi


def _resolve_distributed(arr, axis=1):
    if isinstance(arr, mpiarray.MPIArray):
        arr.redistribute(axis)
        global_shape = arr.global_shape
        ilocal = list(
            range(
                arr.local_offset[axis], arr.local_offset[axis] + arr.local_shape[axis]
            )
        )
    else:
        global_shape = arr.shape
        ilocal = list(range(0, global_shape[axis]))

    return arr[:].view(np.ndarray), global_shape, ilocal
