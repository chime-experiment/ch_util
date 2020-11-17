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
    TimingInterpolator

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

    ```tgain, tweight = tdata.get_gain(freq, input, time)```

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

import scipy.interpolate
import scipy.optimize

from . import tools, andata, ephemeris, rfi
from caput import memh5, mpiarray, tod

FREQ_TO_OMEGA = 2.0 * np.pi * 1e-6
FREQ_PIVOT = 600.0

AXES = ["freq", "noise_source", "input", "time", "param"]

DSET_SPEC = {
    "tau": {"axis": ["noise_source", "time"], "flag": False},
    "alpha": {"axis": ["noise_source", "time"], "flag": False},
    "weight_tau": {"axis": ["noise_source", "time"], "flag": True},
    "weight_alpha": {"axis": ["noise_source", "time"], "flag": True},
    "static_phi": {"axis": ["freq", "noise_source"], "flag": False},
    "static_amp": {"axis": ["freq", "noise_source"], "flag": False},
    "weight_static_phi": {"axis": ["freq", "noise_source"], "flag": True},
    "weight_static_amp": {"axis": ["freq", "noise_source"], "flag": True},
    "static_phi_fit": {"axis": ["param", "noise_source"], "flag": False},
    "num_freq": {"axis": ["noise_source", "time"], "flag": True},
    "phi": {"axis": ["freq", "noise_source", "time"], "flag": False},
    "amp": {"axis": ["freq", "noise_source", "time"], "flag": False},
    "weight_phi": {"axis": ["freq", "noise_source", "time"], "flag": True},
    "weight_amp": {"axis": ["freq", "noise_source", "time"], "flag": True},
    "coeff_tau": {"axis": ["input", "noise_source"], "flag": False},
    "coeff_alpha": {"axis": ["input", "noise_source"], "flag": False},
    "reference_noise_source": {"axis": ["input"], "flag": False},
}

# Set up logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class TimingCorrection(andata.BaseData):
    """Container that holds a timing correction.

    Provides methods for applying that correction to other datasets.

    Attributes
    ----------
    freq
    noise_source
    input
    time
    tau
    weight_tau
    static_phi
    weight_static_phi
    static_phi_fit
    alpha
    weight_alpha
    static_amp
    weight_static_amp
    num_freq
    has_num_freq
    coeff_tau
    has_coeff_tau
    coeff_alpha
    has_coeff_alpha
    amp_to_delay
    has_amplitude
    reference_noise_source
    zero_delay_noise_source

    Methods
    -------
    from_dict
    set_coeff
    search_input
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
        noise_source: np.ndarray[nsource,] of dtype=('chan_id', 'correlator_input')
            Correlator inputs that were used to construct the timing correction.
        input: np.ndarray[ninput, ] of dtype=('chan_id', 'correlator_input')
            Correlator inputs to which the timing correction will be applied.
        time: np.ndarray[ntime, ]
            Unix time.
        param: np.ndarray[nparam, ]
            Parameters of the model fit to the static phase versus frequency.
        tau:  np.ndarray[nsource, ntime]
            The actual timing correction, which is the relative delay of each of the
            noise source inputs with respect to a reference input versus time.
        weight_tau:  np.ndarray[nsource, ntime]
            Estimate of the uncertainty (inverse variance) on the timing correction.
        static_phi: np.ndarray[nfreq, nsource]
            The phase that was subtracted from each frequency and input prior to
            fitting for the timing correction.  This is necessary to remove the
             approximately static ripple pattern caused by reflections.
        weight_static_phi: np.ndarray[nfreq, nsource]
            Inverse variance on static_phi.
        static_phi_fit: np.ndarray[nparam, nsource]
            Best-fit parameters of a fit to the static phase versus frequency
            for each of the noise source inputs.
        alpha: np.ndarray[nsource, ntime]
            The coefficient of the spectral model of the amplitude variations of
            each of the noise source inputs versus time.
        weight_alpha: np.ndarray[nsource, ntime]
            Estimate of the uncertainty (inverse variance) on the amplitude coefficients.
        static_amp: np.ndarray[nfreq, nsource]
            The amplitude that was subtracted from each frequency and input prior to
            fitting for the amplitude variations.  This is necessary to remove the
            approximately static ripple pattern caused by reflections.
        weight_static_amp: np.ndarray[nfreq, nsource]
            Inverse variance on static_amp.
        num_freq: np.ndarray[nsource, ntime]
            The number of frequencies used to determine the delay and alpha quantities.
            If num_freq is 0, then that time is ignored when deriving the timing correction.
        coeff_tau: np.ndarray[ninput, nsource]
            If coeff is provided, then the timing correction applied to a particular
            input will be the linear combination of the tau correction from the
            noise source inputs, with the coefficients set by this array.
        coeff_alpha: np.ndarray[ninput, nsource]
            If coeff is provided, then the timing correction applied to a particular
            input will be adjusted by the linear combination of the alpha correction
            from the noise source inputs, with the coefficients set by this array.
        reference_noise_source: np.ndarray[ninput]
            The noise source input that was used as reference when fitting coeff_tau.
        """
        index_map = {key: kwargs.pop(key) for key in AXES if key in kwargs}
        datasets = {key: kwargs.pop(key) for key in DSET_SPEC.keys() if key in kwargs}

        # Run base initialiser
        tcorr = TimingCorrection(**kwargs)

        # Create index maps
        for name, data in index_map.items():
            tcorr.create_index_map(name, data)

        # Create datasets
        for name, data in datasets.items():
            if data is None:
                continue
            spec = DSET_SPEC[name]
            if spec["flag"]:
                dset = tcorr.create_flag(name, data=data)
            else:
                dset = tcorr.create_dataset(name, data=data)

            dset.attrs["axis"] = np.array(spec["axis"], dtype=np.string_)

        return tcorr

    @classmethod
    def _interpret_and_read(cls, acq_files, start, stop, datasets, out_group):

        # Instantiate an object for each file
        objs = [cls.from_file(d, ondisk=False) for d in acq_files]

        # Reference all dynamic datasets to the static quantities
        # defined in the first file
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
                if np.sum(flag[:, nn, tt], dtype=np.int) > 2:
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

        damp = amp - amp_ref

        asc = amp_ref * _amplitude_scaling(freq[:, np.newaxis, np.newaxis])

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
    def noise_source(self):
        """Provide convenience access to the noise source inputs.

        Note that in older versions of the timing correction, the
        noise_source axis does not exist.  Instead, the equivalent
        quantity is labeled as input.  Since the addition of the
        coeff dataset it has become necessary to distinguish between the
        noise source inputs from which the timing correction is derived
        and the correlator inputs to which the timing correction is applied.
        """
        key = "noise_source" if "noise_source" in self.index_map else "input"
        return self.index_map[key]

    @property
    def nsource(self):
        """Provide convenience access to the number of noise source inputs."""
        return self.noise_source.size

    @property
    def input(self):
        """Provide convenience access to the correlator inputs."""
        return self.index_map["input"]

    @property
    def tau(self):
        """Provide convenience access to the tau array."""
        return self.datasets["tau"]

    @property
    def weight_tau(self):
        """Provide convenience access to the weight_tau array."""
        if "weight_tau" not in self.flags:
            # weight_tau does not exist.  This is the case for timing
            # corrections generated with older versions of the code.
            # Create a default weight_tau dataset and return that.
            if self.has_num_freq:
                weight_tau = (self.num_freq[:] > 0).astype(np.float32)
            else:
                weight_tau = np.ones_like(self.tau[:])

            dset = self.create_flag("weight_tau", data=weight_tau)
            dset.attrs["axis"] = np.array(
                DSET_SPEC["weight_tau"]["axis"], dtype=np.string_
            )

        return self.flags["weight_tau"]

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
    def weight_alpha(self):
        """Provide convenience access to the weight_alpha array."""
        if "weight_alpha" not in self.flags:
            # weight_alpha does not exist.  This is the case for timing
            # corrections generated with older versions of the code.
            # Create a default weight_alpha dataset and return that.
            if self.has_num_freq:
                weight_alpha = (self.num_freq[:] > 0).astype(np.float32)
            else:
                weight_alpha = np.ones_like(self.alpha[:])

            scale = (self.amp_to_delay or 1.0) ** 2
            weight_alpha *= scale

            dset = self.create_flag("weight_alpha", data=weight_alpha)
            dset.attrs["axis"] = np.array(
                DSET_SPEC["weight_alpha"]["axis"], dtype=np.string_
            )

        return self.flags["weight_alpha"]

    @property
    def static_amp(self):
        """Provide convenience access to the static_amp array."""
        return self.datasets["static_amp"]

    @property
    def weight_static_amp(self):
        """Provide convenience access to the weight_static_amp array."""
        return self.flags["weight_static_amp"]

    @property
    def num_freq(self):
        """Provide convenience access to the num_freq array."""
        return self.flags["num_freq"]

    @property
    def has_num_freq(self):
        """Inidicates if there is a num_freq flag that identifies missing data."""
        return "num_freq" in self.flags

    @property
    def coeff_tau(self):
        """Provide convenience access to the coeff_tau array."""
        return self.datasets["coeff_tau"]

    @property
    def has_coeff_tau(self):
        """Indicates if there are valid coeff that map noise source tau to inputs."""
        return (
            "coeff_tau" in self.datasets
            and "noise_source" in self.index_map
            and "input" in self.index_map
        )

    @property
    def coeff_alpha(self):
        """Provide convenience access to the coeff_alpha array."""
        return self.datasets["coeff_alpha"]

    @property
    def has_coeff_alpha(self):
        """Indicates if there are valid coeff that map noise source alpha to inputs."""
        return (
            "coeff_alpha" in self.datasets
            and "noise_source" in self.index_map
            and "input" in self.index_map
        )

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
        if self.has_coeff_alpha:
            raise AttributeError(
                "The amplitude variations are already being used to "
                "correct the delay variations through the coeff_alpha dataset."
            )
        elif val is not None:
            self.attrs["amp_to_delay"] = val

    @amp_to_delay.deleter
    def amp_to_delay(self):
        """Remove any conversion from noise source amplitude variations to delay variations."""
        if "amp_to_delay" in self.attrs:
            del self.attrs["amp_to_delay"]

    @property
    def has_amplitude(self):
        """Determine if this timing correction contains amplitude data."""
        return "alpha" in self.datasets

    @property
    def reference_noise_source(self):
        """Return the index of the reference noise source."""
        if "reference_noise_source" in self.datasets:
            iref = self.datasets["reference_noise_source"][:]
            return iref if np.unique(iref).size > 1 else iref[0]
        else:
            return self.zero_delay_noise_source

    @property
    def zero_delay_noise_source(self):
        """Return the index of the noise source with zero delay."""
        zero_tau = np.flatnonzero(np.all(np.abs(self.tau[:]) < 1e-5, axis=-1))
        if zero_tau.size == 0:
            raise AttributeError(
                "Could not determine which input the delay template "
                "is referenced with respect to."
            )
        else:
            return zero_tau[0]

    def set_coeff(
        self,
        coeff_tau,
        inputs,
        noise_source,
        coeff_alpha=None,
        reference_noise_source=None,
    ):
        """Use coefficients to construct timing correction.

        Setting the coefficients changes how the timing corretion for a particular
        correlator input is derived.  Without coefficients, each input is matched
        to the timing correction from a single noise source input through the
        map_input_to_noise_source method.  With coefficients, each input is a
        linear combination of the timing correction from all noise source inputs.

        Parameters
        ----------
        coeff_tau: np.ndarray[ninput, nsource]
            The timing correction applied to a particular input will be the
            linear combination of the tau correction from the noise source inputs,
            with the coefficients set by this array.
        inputs: np.ndarray[ninput, ] of dtype=('chan_id', 'correlator_input')
            Correlator inputs to which the timing correction will be applied.
        noise_source: np.ndarray[nsource,] of dtype=('chan_id', 'correlator_input')
            Correlator inputs that were used to construct the timing correction.
        coeff_alpha: np.ndarray[ninput, nsource]
            The timing correction applied to a particular input will be adjusted by
            the linear combination of the alpha correction from the noise source inputs,
            with the coefficients set by this array.
        reference_noise_source: np.ndarray[ninput,]
            For each input, the index into noise_source that was used as
            reference in the fit for coeff_tau.
        """
        sn_lookup = {
            sn: ii for ii, sn in enumerate(noise_source["correlator_input"][:])
        }

        reod = np.array(
            [sn_lookup[sn] for sn in self.noise_source["correlator_input"][:]]
        )

        datasets = {"coeff_tau": coeff_tau}
        if coeff_alpha is not None:
            if self.amp_to_delay is None:
                datasets["coeff_alpha"] = coeff_alpha
            else:
                raise AttributeError(
                    "The amplitude variations are already "
                    "being used to correct the delay variations "
                    "through the amp_to_delay parameter."
                )

        for name, coeff in datasets.items():

            spec = DSET_SPEC[name]
            if spec["flag"]:
                dset = self.create_flag(name, data=coeff[:, reod])
            else:
                dset = self.create_dataset(name, data=coeff[:, reod])
            dset.attrs["axis"] = np.array(spec["axis"], dtype=np.string_)

        if reference_noise_source is not None:

            ref_sn_lookup = {
                sn: ii for ii, sn in enumerate(self.noise_source["correlator_input"][:])
            }

            reference_reodered = np.array(
                [
                    ref_sn_lookup[sn]
                    for sn in noise_source["correlator_input"][reference_noise_source]
                ]
            )

            name = "reference_noise_source"
            spec = DSET_SPEC[name]
            if spec["flag"]:
                dset = self.create_flag(name, data=reference_reodered)
            else:
                dset = self.create_dataset(name, data=reference_reodered)
            dset.attrs["axis"] = np.array(spec["axis"], dtype=np.string_)

        self.create_index_map("input", inputs)

    def delete_coeff(self):
        """Stop using coefficients to construct timing correction.

        Calling this method will delete the `coeff_tau`, `coeff_alpha`,
        and `reference_noise_source` datasets if they exist.
        """
        for name in ["coeff_tau", "coeff_alpha", "reference_noise_source"]:
            spec = DSET_SPEC[name]
            group = self["flag"] if spec["flag"] else self
            if name in group:
                del group[name]

    def search_input(self, inputs):
        """Find inputs in the input axis.

        Parameters
        ----------
        inputs: np.ndarray[ninput,] of dtype=('chan_id', 'correlator_input')

        Returns
        -------
        index: np.ndarray[ninput,] of np.int
            Indices of the input axis that yield the requested inputs.
        """
        if not hasattr(self, "_input_lookup"):
            self._input_lookup = {
                sn: ind for ind, sn in enumerate(self.input["correlator_input"][:])
            }

        return np.array(
            [self._input_lookup[sn] for sn in inputs["correlator_input"][:]]
        )

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
        logger.info("Referencing timing correction with respect to %s." % tref_string)
        if window > 0.0:
            iref = np.flatnonzero(
                (self.time >= (tref - window)) & (self.time <= (tref + window))
            )
            if iref.size > 0:
                logger.info(
                    "Using median of %d samples around reference time." % iref.size
                )
                if self.has_num_freq:
                    tau_ref = np.zeros((self.nsource, 1), dtype=self.tau.dtype)
                    alpha_ref = np.zeros((self.nsource, 1), dtype=self.alpha.dtype)

                    for ss in range(self.nsource):
                        good = np.flatnonzero(self.num_freq[ss, iref] > 0)
                        if good.size > 0:
                            tau_ref[ss] = np.median(self.tau[ss, iref[good]])
                            alpha_ref[ss] = np.median(self.alpha[ss, iref[good]])

                else:
                    tau_ref = np.median(self.tau[:, iref], axis=-1, keepdims=True)
                    alpha_ref = np.median(self.alpha[:, iref], axis=-1, keepdims=True)

            else:
                raise ValueError(
                    "Timing correction not available for time %s." % tref_string
                )

        elif (tref < self.time[0]) or (tref > self.time[-1]):
            raise ValueError(
                "Timing correction not available for time %s." % tref_string
            )

        else:
            if not interpolate:
                kwargs["interp"] = "nearest"

            tau_ref, _ = self.get_tau(np.atleast_1d(tref), ignore_amp=True, **kwargs)
            alpha_ref, _ = self.get_alpha(np.atleast_1d(tref), **kwargs)

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

        if not interpolate:
            kwargs["interp"] = "nearest"

        tau_ref, _ = self.get_tau(tref, ignore_amp=True, **kwargs)
        alpha_ref, _ = self.get_alpha(tref, **kwargs)

        if tinit is not None:
            tinit = ephemeris.ensure_unix(tinit)
            tau_init, _ = self.get_tau(tinit, ignore_amp=True, **kwargs)
            alpha_init, _ = self.get_alpha(tinit, **kwargs)

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
        """Return the delay for each noise source at the requested times.

        Uses the TimingInterpolator to interpolate to the requested times.

        Parameters
        ----------
        timestamp:  np.ndarray[ntime,]
            Unix timestamp.
        ignore_amp: bool
            Do not apply a noise source based amplitude correction, even if one exists.
        interp: string
            Method to interpolate over time.  Options include 'linear', 'nearest',
            'zero', 'slinear', 'quadratic', 'cubic', 'previous', and 'next'.
        extrap_limit: float
            Do not extrapolate the underlying data beyond its boundaries by this
            amount in seconds.  Default is 2 integrations.

        Returns
        -------
        tau: np.ndarray[nsource, ntime]
            Delay as a function of time for each of the noise sources.
        weight : np.ndarray[nsource, ntime]
            The uncertainty on the delay, expressed as an inverse variance.
        """
        flag = self.num_freq[:] > 0 if self.has_num_freq else None

        if ignore_amp or (self.amp_to_delay is None) or not self.has_amplitude:

            tau_interpolator = TimingInterpolator(
                self.time[:],
                self.tau[:],
                weight=self.weight_tau[:],
                flag=flag,
                kind=interp,
                extrap_limit=extrap_limit,
            )

            tau, weight = tau_interpolator(timestamp)

        else:

            logger.info(
                "Correcting delay template using amplitude template "
                "with coefficient %0.1f." % self.amp_to_delay
            )

            # Determine which input the delay template is referenced to
            iref = self.zero_delay_noise_source

            # Subtract the referenced, scaled alpha template from the delay template
            tau_corrected = self.tau[:] - self.amp_to_delay * (
                self.alpha[:] - self.alpha[iref, np.newaxis, :]
            )

            # Extract the weights
            weight_corrected = _weight_propagation_addition(
                self.weight_tau[:],
                self.weight_alpha[:] / self.amp_to_delay ** 2,
                self.weight_alpha[iref, np.newaxis, :] / self.amp_to_delay ** 2,
            )

            # Interpolate to the requested times
            tau_interpolator = TimingInterpolator(
                self.time[:],
                tau_corrected,
                weight=weight_corrected,
                flag=flag,
                kind=interp,
                extrap_limit=extrap_limit,
            )

            tau, weight = tau_interpolator(timestamp)

        return tau, weight

    def get_alpha(self, timestamp, interp="linear", extrap_limit=None):
        """Return the amplitude variation for each noise source at the requested times.

        Uses the TimingInterpolator to interpolate to the requested times.

        Parameters
        ----------
        timestamp:  np.ndarray[ntime,]
            Unix timestamp.
        interp: string
            Method to interpolate over time.  Options include 'linear', 'nearest',
            'zero', 'slinear', 'quadratic', 'cubic', 'previous', and 'next'.
        extrap_limit: float
            Do not extrapolate the underlying data beyond its boundaries by this
            amount in seconds.  Default is 2 integrations.

        Returns
        -------
        alpha: np.ndarray[nsource, ntime]
            Amplitude coefficient as a function of time for each of the noise sources.
        weight : np.ndarray[nsource, ntime]
            The uncertainty on the amplitude coefficient, expressed as an inverse variance.
        """
        flag = self.num_freq[:] > 0 if self.has_num_freq else None

        alpha_interpolator = TimingInterpolator(
            self.time[:],
            self.alpha[:],
            weight=self.weight_alpha[:],
            flag=flag,
            kind=interp,
            extrap_limit=extrap_limit,
        )

        alpha, weight = alpha_interpolator(timestamp)

        return alpha, weight

    def get_stacked_tau(
        self, timestamp, inputs, prod, reverse_stack, input_flags=None, **kwargs
    ):
        """Return the appropriate delay for each stacked visibility at the requested time.

        Averages the delays from the noise source inputs that map to the set of redundant
        baseline included in each stacked visibility.  This yields the appropriate
        common-mode delay correction.  If input_flags is provided, then the bad inputs
        that were excluded from the stack are also excluded from the delay template averaging.

        Parameters
        ----------
        timestamp:  np.ndarray[ntime,]
            Unix timestamp.
        inputs: np.ndarray[ninput,]
            Must contain 'correlator_input' field.
        prod: np.ndarray[nprod,]
            The products that were included in the stack.
            Typically found in the `index_map['prod']` attribute of the
            `andata.CorrData` object.
        reverse_stack: np.ndarray[nprod,] of dtype=('stack', 'conjugate')
            The index of the stack axis that each product went into.
            Typically found in `reverse_map['stack']` attribute
            of the `andata.CorrData`.
        input_flags : np.ndarray [ninput, ntime]
            Array indicating which inputs were good at each time.
            Non-zero value indicates that an input was good.

        Returns
        -------
        tau: np.ndarray[nstack, ntime]
            Delay as a function of time for each stacked visibility.
        """
        # Use the get_tau method to get the data for the noise source inputs
        # at the requested times.
        data, _ = self.get_tau(timestamp, **kwargs)

        if self.has_coeff_tau:
            # This tau correction has a coefficient array.
            # Find the coefficients for the requested inputs.
            reod = andata._convert_to_slice(self.search_input(inputs))
            coeff = self.coeff_tau[reod, :]

            # Determine how the noise source delays were referenced
            # when fitting for the coefficients
            iref = self.reference_noise_source
            if np.isscalar(iref):
                if iref != self.zero_delay_noise_source:
                    data = data - data[iref, np.newaxis, :]
                iref = None
            else:
                iref = iref[reod]
        else:
            coeff = None
            iref = None

        # Stack the data from the noise source inputs
        return self._stack(
            data,
            inputs,
            prod,
            reverse_stack,
            coeff=coeff,
            input_flags=input_flags,
            reference_noise_source=iref,
        )

    def get_stacked_alpha(
        self, timestamp, inputs, prod, reverse_stack, input_flags=None, **kwargs
    ):
        """Return the equivalent of `get_stacked_tau` for the noise source amplitude variations.

        Averages the alphas from the noise source inputs that map to the set of redundant
        baseline included in each stacked visibility.  If input_flags is provided, then the
        bad inputs that were excluded from the stack are also excluded from the alpha
        template averaging.  This method can be used to generate a stacked alpha template
        that can be used to correct a stacked tau template for variations in the noise source
        distribution system.  However, it is recommended that the tau template be corrected
        before stacking. This is accomplished by setting the `amp_to_delay` property
        prior to calling `get_stacked_tau`.

        Parameters
        ----------
        timestamp:  np.ndarray[ntime,]
            Unix timestamp.
        inputs: np.ndarray[ninput,]
            Must contain 'correlator_input' field.
        prod: np.ndarray[nprod,]
            The products that were included in the stack.
            Typically found in the `index_map['prod']` attribute of the
            `andata.CorrData` object.
        reverse_stack: np.ndarray[nprod,] of dtype=('stack', 'conjugate')
            The index of the stack axis that each product went into.
            Typically found in `reverse_map['stack']` attribute
            of the `andata.CorrData`.
        input_flags : np.ndarray [ninput, ntime]
            Array indicating which inputs were good at each time.
            Non-zero value indicates that an input was good.

        Returns
        -------
        alpha: np.ndarray[nstack, ntime]
            Noise source amplitude variation as a function of time for each stacked visibility.
        """
        if not self.has_amplitude:
            raise AttributeError(
                "This timing correction does not include "
                "an adjustment based on the noise soure amplitude."
            )

        if self.has_coeff_alpha:
            # This alpha correction has a coefficient array.
            # Find the coefficients for the requested inputs.
            reod = andata._convert_to_slice(self.search_input(inputs))
            coeff = self.coeff_alpha[reod, :]
        else:
            coeff = None

        # Use the get_alpha method to get the data for the noise source inputs.
        data, _ = self.get_alpha(timestamp, **kwargs)

        # Stack the data from the noise source inputs
        return self._stack(
            data, inputs, prod, reverse_stack, coeff=coeff, input_flags=input_flags
        )

    def _stack(
        self,
        data,
        inputs,
        prod,
        reverse_stack,
        coeff=None,
        input_flags=None,
        reference_noise_source=None,
    ):

        stack_index = reverse_stack["stack"][:]
        stack_conj = reverse_stack["conjugate"][:].astype(np.bool)

        nstack = np.max(stack_index) + 1
        nprod = prod.size
        ninput = inputs.size

        # Sort the products based on the index of the stack axis they went into.
        isort = np.argsort(stack_index)
        sorted_stack_index = stack_index[isort]
        sorted_stack_conj = stack_conj[isort]
        sorted_prod = prod[isort]

        temp = sorted_prod.copy()
        sorted_prod["input_a"] = np.where(
            sorted_stack_conj, temp["input_b"], temp["input_a"]
        )
        sorted_prod["input_b"] = np.where(
            sorted_stack_conj, temp["input_a"], temp["input_b"]
        )

        # Find boundaries into the sorted products that separate stacks.
        boundary = np.concatenate(
            (
                np.atleast_1d(0),
                np.flatnonzero(np.diff(sorted_stack_index) > 0) + 1,
                np.atleast_1d(nprod),
            )
        )

        # Check for coefficient array that encodes the contribution of
        # each noise source to each input.
        if coeff is None:
            # This timing correction does not have a coefficient array.
            # Construct from the output of the map_input_to_noise_source method.
            index = np.array(map_input_to_noise_source(inputs, self.noise_source))
            coeff = np.zeros((ninput, self.nsource), dtype=np.float64)
            coeff[np.arange(ninput), index] = 1.0

        # Expand the coefficient array to have single element time axis
        nsource = coeff.shape[-1]
        coeff = coeff[:, :, np.newaxis]

        # Construct separate coefficient array that handles the reference noise source
        with_ref = reference_noise_source is not None
        if with_ref:
            cref = np.zeros((ninput, nsource, nsource, 1), dtype=np.float64)
            cref[np.arange(ninput), reference_noise_source] = coeff

        # If input_flags was not provided, or if it is all True or all False, then we
        # assume all inputs are good and carry out a faster calculation.
        no_input_flags = (
            (input_flags is None) or not np.any(input_flags) or np.all(input_flags)
        )

        if no_input_flags:
            # No input flags provided.  All inputs considered good.
            uniq_input_flags = np.ones((ninput, 1), dtype=np.float64)
            index_time = slice(None)
        else:
            # Find the unique sets of input flags.
            uniq_input_flags, index_time = np.unique(
                input_flags, return_inverse=True, axis=1
            )

        ntime_uniq = uniq_input_flags.shape[-1]

        # Initialize arrays to hold the stacked coefficients
        stack_coeff = np.zeros((nstack, nsource, ntime_uniq), dtype=np.float64)
        weight_norm = np.zeros((nstack, ntime_uniq), dtype=np.float64)

        # Loop over stacked products
        for ss, ssi in enumerate(np.unique(sorted_stack_index)):

            # Get the input pairs that went into this stack
            prodo = sorted_prod[boundary[ss] : boundary[ss + 1]]
            aa = prodo["input_a"]
            bb = prodo["input_b"]

            # Sum the difference in coefficients over pairs of inputs,
            # weighted by the product of the input flags for those inputs.
            ww = uniq_input_flags[aa] * uniq_input_flags[bb]
            weight_norm[ssi] = np.sum(ww, axis=0)
            stack_coeff[ssi] = np.sum(
                ww[:, np.newaxis, :] * (coeff[aa] - coeff[bb]), axis=0
            )

            if with_ref:
                stack_coeff[ssi] -= np.sum(
                    ww[:, np.newaxis, :] * np.sum(cref[aa] - cref[bb], axis=2), axis=0
                )

        # The delay for each stacked product is a linear combination of the
        # delay from the noise source inputs.
        stacked_data = np.sum(
            stack_coeff[:, :, index_time] * data[np.newaxis, :, :], axis=1
        )
        stacked_data *= tools.invert_no_zero(weight_norm[:, index_time])

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
        weight: np.ndarray[nfreq, nsource, ntime]
            Uncerainty on the gain for each of the noise sources, expressed as an inverse variance.
        """
        tau, wtau = self.get_tau(timestamp, **kwargs)

        gain = np.exp(
            -1.0j
            * FREQ_TO_OMEGA
            * freq[:, np.newaxis, np.newaxis]
            * tau[np.newaxis, :, :]
        )

        weight = (
            wtau[np.newaxis, :, :]
            * tools.invert_no_zero(FREQ_TO_OMEGA * freq[:, np.newaxis, np.newaxis]) ** 2
        )

        return gain, weight

    def get_gain(self, freq, inputs, timestamp, **kwargs):
        """Return the complex gain for the requested frequencies, inputs, and times.

        Multiplying the visibilities by the outer product of these gains will remove
        the fluctuations in phase due to timing jitter.  This method uses the
        get_tau method.  It acccepts and passes along keyword arguments for that method.

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
        weight: np.ndarray[nfreq, ninput, ntime]
            Uncerainty on the gain expressed as an inverse variance.
        """
        if self.has_coeff_tau:

            # Get the delay template for the noise source inputs
            # at the requested times
            tau, wtau = self.get_tau(timestamp, **kwargs)

            vartau = tools.invert_no_zero(wtau)

            # Find the coefficients for the requested inputs
            reod = andata._convert_to_slice(self.search_input(inputs))

            C = self.coeff_tau[reod, :]

            # Different calculation dependening on whether or not the
            # reference noise source changes with input
            iref = self.reference_noise_source
            if np.isscalar(iref):
                # There is a single reference for all inputs.
                # Check if it is different than the current reference.
                if iref != self.zero_delay_noise_source:
                    tau = tau - tau[iref, np.newaxis, :]
                    vartau = vartau + vartau[iref, np.newaxis, :]

                # The delay for each input is a linear combination of the
                # delay from the noise source inputs
                tau = np.matmul(C, tau)
                vartau = np.matmul(C ** 2, vartau)

            else:
                # Find the reference for the requested inputs
                iref = iref[reod]

                # The delay for each input is a linear combination of the
                # delay from the noise source inputs
                sumC = np.sum(C, axis=-1, keepdims=True)

                tau = np.matmul(C, tau) - sumC * tau[iref, :]

                vartau = np.matmul(C ** 2, vartau) + sumC ** 2 * vartau[iref, :]

            # Check if we need to correct the delay using the noise source amplitude
            if self.has_amplitude and self.has_coeff_alpha:

                # Get the alpha template for the noise source inputs
                # at the requested times
                alpha, walpha = self.get_alpha(timestamp, **kwargs)

                varalpha = tools.invert_no_zero(walpha)

                Calpha = self.coeff_alpha[reod, :]

                # Adjust the delay for each input by the linear combination of the
                # amplitude from the noise source inputs
                tau += np.matmul(Calpha, alpha)

                vartau += np.matmul(Calpha ** 2, varalpha)

            # Scale by 2 pi nu to convert to gain
            gain = np.exp(
                -1.0j
                * FREQ_TO_OMEGA
                * freq[:, np.newaxis, np.newaxis]
                * tau[np.newaxis, :, :]
            )

            weight = tools.invert_no_zero(
                vartau[np.newaxis, :, :]
                * (FREQ_TO_OMEGA * freq[:, np.newaxis, np.newaxis]) ** 2
            )

        else:
            # Get the timing correction for the noise source inputs at the
            # requested times and frequencies
            gain, weight = self.get_timing_correction(freq, timestamp, **kwargs)

            # Determine which noise source to use for each input
            index = map_input_to_noise_source(inputs, self.noise_source)

            gain = gain[:, index, :]
            weight = weight[:, index, :]

        # Return gains
        return gain, weight

    def apply_timing_correction(self, timestream, copy=False, **kwargs):
        """Apply the timing correction to another visibility dataset.

        This method uses the get_gain or get_stacked_tau method, depending
        on whether or not the visibilities have been stacked.  It acccepts
        and passes along keyword arguments for those method.

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
        reverse_stack : np.ndarray[nprod, ] of dtype=('stack', 'conjugate')
            The index of the stack axis that each product went into.
            Typically found in `reverse_map['stack']` attribute.
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
                else timestream.reverse_map["stack"][:]
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
            # Visibilities have been stacked.
            # Stack the timing correction before applying it.
            tau = self.get_stacked_tau(
                timestamp,
                inputs,
                prod,
                reverse_stack,
                input_flags=input_flags,
                **kwargs
            )

            if self.has_amplitude and self.has_coeff_alpha:
                tau += self.get_stacked_alpha(
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
            # Get the gain corrections for the times and frequencies in timestream.
            gain, _ = self.get_gain(freq, inputs, timestamp, **kwargs)

            # Loop over products and apply the timing correction
            for ii, (aa, bb) in enumerate(prod):
                vis[:, ii, :] *= gain[:, aa, :] * gain[:, bb, :].conj()

            # If andata object was input then update the gain
            # dataset so that we have record of what was done
            if is_obj and not copy and "gain" in timestream:
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
        datasets = kwargs.pop(
            "datasets", ["vis", "flags/vis_weight", "flags/frac_lost"]
        )

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
        noise_source = corr_data.input[isource]
        data.create_index_map("noise_source", noise_source)

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

        # If we are only returning the timing correction, then remove
        # the amplitude and phase of the noise source
        if only_correction:
            for name in ["amp", "weight_amp", "phi", "weight_phi"]:
                res.pop(name)

        # Create index map containing names of parameters
        param = ["intercept", "slope", "quad", "cube", "quart", "quint"]
        param = param[0 : res["static_phi_fit"].shape[0]]
        data.create_index_map("param", np.array(param, dtype=np.string_))

        # Create datasets containing the timing correction
        for name, arr in res.items():
            spec = DSET_SPEC[name]
            if spec["flag"]:
                dset = data.create_flag(name, data=arr)
            else:
                dset = data.create_dataset(name, data=arr)

            dset.attrs["axis"] = np.array(spec["axis"], dtype=np.string_)

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
            input=self.noise_source,
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
                            self.noise_source[pp[0]]["correlator_input"],
                            self.noise_source[pp[1]]["correlator_input"],
                        )
                        + tuple(stats[:, ii])
                    )
                )

        return summary


class TimingInterpolator(object):
    """Interpolation that is aware of flagged data and weights.

    Flagged data is ignored during the interpolation.  The weights from
    the data are propagated to obtain weights for the interpolated points.
    """

    def __init__(self, x, y, weight=None, flag=None, kind="linear", extrap_limit=None):
        """Instantiate a callable TimingInterpolator object.

        Parameters
        ----------
        x : np.ndarray[nsample,]
            The points where the data was sampled.
            Must be monotonically increasing.
        y : np.ndarray[..., nsample]
            The data to interpolate.
        weight : np.ndarray[..., nsample]
            The uncertainty on the data, expressed as an
            inverse variance.
        flag : np.ndarray[..., nsample]
            Boolean indicating if the data is to be
            included in the interpolation.
        kind : str
            String that specifies the kind of interpolation.
            The value `nearest`, `previous`, `next`, and `linear` will use
            custom methods that propagate uncertainty to obtain the interpolated
            weights.  The value 'zero', 'slinear', 'quadratic', and `cubic'
            will use spline interpolation from scipy.interpolation.interp1d
            and use the weight from the nearest point.

        Returns
        -------
        interpolator : TimingInterpolator
            Callable that will interpolate the data that was provided
            to a new set of x values.
        """
        self.x = x
        self.y = y

        self._shape = y.shape[:-1]

        if weight is None:
            self.var = np.ones(y.shape, dtype=np.float32)
        else:
            self.var = tools.invert_no_zero(weight)

        if flag is None:
            self.flag = np.ones(y.shape, dtype=np.bool)
        else:
            self.flag = flag

        if extrap_limit is None:
            self._extrap_limit = 2.0 * np.median(np.diff(self.x))
        else:
            self._extrap_limit = extrap_limit

        self._interp = INTERPOLATION_LOOKUP.get(kind, _interpolation_scipy(kind))

    def __call__(self, xeval):
        """Interpolate the data.

        Parameters
        ----------
        xeval : np.ndarray[neval,]
            Evaluate the interpolant at these points.

        Returns
        -------
        yeval : np.ndarray[neval,]
            Interpolated values.
        weval : np.ndarray[neval,]
            Uncertainty on the interpolated values,
            expressed as an inverse variance.
        """
        # Make sure we are not extrapolating too much
        dx_beg = self.x[0] - np.min(xeval)
        dx_end = np.max(xeval) - self.x[-1]

        if (dx_beg > self._extrap_limit) or (dx_end > self._extrap_limit):
            raise ValueError("Extrapolating beyond span of data.")

        # Create arrays to hold interpolation
        shape = self._shape if np.isscalar(xeval) else self._shape + (xeval.size,)

        yeval = np.zeros(shape, dtype=self.y.dtype)
        weval = np.zeros(shape, dtype=np.float32)

        # Loop over other axes and interpolate along last axis
        for ind in np.ndindex(*self._shape):

            to_interp = np.flatnonzero(self.flag[ind])
            if to_interp.size > 0:

                yeval[ind], weval[ind] = self._interp(
                    self.x[to_interp],
                    self.y[ind][to_interp],
                    self.var[ind][to_interp],
                    xeval,
                )

        return yeval, weval


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
    min_frac_kept=0.0,
    threshold=0.50,
    min_freq=420.0,
    max_freq=780.0,
    mask_rfi=False,
    max_iter_weight=None,
    check_amp=False,
    nsigma_amp=None,
    check_phi=True,
    nsigma_phi=None,
    nparam=2,
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
            flags/frac_lost: np.ndarray[nfreq, ntime]
                Flag indicating the fraction of data lost.
                If provided, then data will be weighted by the
                fraction of data that remains when solving
                for the delay template.
    min_frac_kept: float
        Do not include frequencies and times where the fraction
        of data that remains is less than this threshold.
        Default is 0.0.
    threshold: float
        A (frequency, input) must pass the checks specified above
        more than this fraction of the time,  otherwise it will be
        flaged as bad for all times.  Default is 0.50.
    min_freq: float
        Minimum frequency in MHz to include in the fit.
        Default is 420.
    max_freq: float
        Maximum frequency in MHz to include in the fit.
        Default is 780.
    mask_rfi: bool
        Mask frequencies that occur within known RFI bands.  Note that the
        noise source data does not contain RFI, however the real-time pipeline
        does not distinguish between noise source inputs and sky inputs, and as
        a result will discard large amounts of data in these bands.
    max_iter_weight: int
        The weight for each frequency is estimated from the variance of the
        residuals of the template fit from the previous iteration.  Outliers
        are also flagged at each iteration with an increasingly aggresive threshold.
        This is the total number of times to iterate.  Setting to 1 corresponds
        to linear least squares.  Default is 1, unless check_amp or check_phi is True,
        in which case this defaults to the maximum number of thresholds provided.
    check_amp: bool
        Do not fit frequencies and times where the residual amplitude is an outlier.
        Default is False.
    nsigma_amp: list of float
        If check_amp is True, then residuals greater than this number of sigma
        will be considered an outlier.  Provide a list containing the value to be used
        at each iteration.  If the length of the list is less than max_iter_weight,
        then the last value in the list will be repeated for the remaining iterations.
        Default is [1000, 500, 200, 100, 50, 20, 10, 5].
    check_phi: bool
        Do not fit frequencies and times where the residual phase is an outlier.
        Default is True.
    nsigma_phi: list of float
        If check_phi is True, then residuals greater than this number of sigma
        will be considered an outlier.  Provide a list containing the value to be used
        at each iteration.  If the length of the list is less than max_iter_weight,
        then the last value in the list will be repeated for the remaining iterations.
        Default is [1000, 500, 200, 100, 50, 20, 10, 5].
    nparam: int
        Number of parameters for polynomial fit to the
        time averaged phase versus frequency.  Default is 2.
    static_phi: np.ndarray[nfreq, nsource]
        Subtract this quantity from the noise source phase prior to fitting
        for the timing correction.  If None, then this will be estimated from the median
        of the noise source phase over time.
    weight_static_phi: np.ndarray[nfreq, nsource]
        Inverse variance of the time averaged phased.  Set to zero for frequencies and inputs
        that are missing or should be ignored.  If None, then this will be estimated from the
        residuals of the fit.
    static_phi_fit: np.ndarray[nparam, nsource]
        Polynomial fit to static_phi versus frequency.
    static_amp: np.ndarray[nfreq, nsource]
        Subtract this quantity from the noise source amplitude prior to fitting
        for the amplitude variations.  If None, then this will be estimated from the median
        of the noise source amplitude over time.
    weight_static_amp: np.ndarray[nfreq, nsource]
        Inverse variance of the time averaged amplitude.  Set to zero for frequencies and inputs
        that are missing or should be ignored.  If None, then this will be estimated from the
        residuals of the fit.

    Returns
    -------
    phi: np.ndarray[nfreq, nsource, ntime]
        Phase of the signal from the noise source.
    weight_phi: np.ndarray[nfreq, nsource, ntime]
        Inverse variance of the phase of the signal from the noise source.
    tau: np.ndarray[nsource, ntime]
        Delay template for each noise source input.
    weight_tau: np.ndarray[nfreq, nsource]
        Estimate of the uncertainty on the delay template (inverse variance).
    static_phi: np.ndarray[nfreq, nsource]
        Time averaged phase versus frequency.
    weight_static_phi: np.ndarray[nfreq, nsource]
       Inverse variance of the time averaged phase.
    static_phi_fit: np.ndarray[nparam, nsource]
        Best-fit parameters of the polynomial fit to the
        time averaged phase versus frequency.
    amp: np.ndarray[nfreq, nsource, ntime]
        Amplitude of the signal from the noise source.
    weight_amp: np.ndarray[nfreq, nsource, ntime]
        Inverse variance of the amplitude of the signal from the noise source.
    alpha: np.ndarray[nsource, ntime]
        Amplitude coefficient for each noise source input.
    weight_alpha: np.ndarray[nfreq, nsource]
        Estimate of the uncertainty on the amplitude coefficient (inverse variance).
    static_amp: np.ndarray[nfreq, nsource]
        Time averaged amplitude versus frequency.
    weight_static_amp: np.ndarray[nfreq, nsource]
        Inverse variance of the time averaged amplitude.
    num_freq: np.ndarray[nsource, ntime]
        Number of frequencies used to construct the delay and amplitude templates.
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

    # Set default nsigma for flagging outliers
    if nsigma_amp is None:
        nsigma_amp = [1000.0, 500.0, 200.0, 100.0, 50.0, 20.0, 10.0, 5.0]
    elif np.isscalar(nsigma_amp):
        nsigma_amp = [nsigma_amp]

    if nsigma_phi is None:
        nsigma_phi = [1000.0, 500.0, 200.0, 100.0, 50.0, 20.0, 10.0, 5.0]
    elif np.isscalar(nsigma_phi):
        nsigma_phi = [nsigma_phi]

    if max_iter_weight is None:
        max_iter_weight = max(
            len(nsigma_amp) + 1 if check_amp else 1,
            len(nsigma_phi) + 1 if check_phi else 1,
        )
    else:
        max_iter_weight = max(max_iter_weight, 1)

    nsigma_amp = [
        nsigma_amp[min(ii, len(nsigma_amp) - 1)] for ii in range(max_iter_weight)
    ]
    nsigma_phi = [
        nsigma_phi[min(ii, len(nsigma_phi) - 1)] for ii in range(max_iter_weight)
    ]

    # Compute amplitude of noise source signal from autocorrelation
    iauto = np.array([int(k * (2 * nsource - k + 1) // 2) for k in range(nsource)])

    amp = np.sqrt(vis[:, iauto, :].real)

    # Determine which data points to fit
    flg = amp > 0.0
    if weight is not None:
        flg &= weight[:, iauto, :] > 0.0

    # If requested discard frequencies and times that have high frac_lost
    if hasattr(data, "flags") and ("frac_lost" in data.flags):

        logger.info("Fraction of data kept must be greater than %0.2f." % min_frac_kept)

        frac_kept = 1.0 - data.flags["frac_lost"][:].view(np.ndarray)
        flg &= frac_kept[:, np.newaxis, :] >= min_frac_kept

    else:
        frac_kept = np.ones((nfreq, ntime), dtype=np.float32)

    # Restrict the range of frequencies that are fit to avoid bandpass edges
    limit_freq = (freq > min_freq) & (freq < max_freq)
    if mask_rfi:
        logger.info("Masking RFI bands.")
        limit_freq &= ~rfi.frequency_mask(
            freq, freq_width=data.index_map["freq"]["width"][:]
        )

    flg = (flg & limit_freq[:, np.newaxis, np.newaxis]).astype(np.float32)

    # If we only have two noise source inputs, then we use the cross-correlation
    # between them to characterize their relative response to the noise source signal.
    # If we have more than two noise source inputs, then we perform an eigenvalue
    # decomposition of the cross-correlation matrix to obtain an improved estimate
    # of the response of each input to the noise source signal.
    if nsource > 2:

        response = eigen_decomposition(vis, flg)

        phi = np.angle(response)
        amp = np.abs(response)

        ww = flg

    else:

        phi = np.zeros((nfreq, nsource, ntime), dtype=np.float32)
        phi[:, 1, :] = np.angle(vis[:, 1, :].conj())

        amp = np.sqrt(vis[:, iauto, :].real)

        ww = np.repeat(flg[:, 0, np.newaxis, :] * flg[:, 1, np.newaxis, :], 2, axis=1)

    # Scale the flag by the fraction of data that was kept
    ww *= frac_kept[:, np.newaxis, :]

    # If parallelized we need to redistribute over inputs for the
    # operations below, which require full frequency and time coverage.
    if parallel:
        amp = mpiarray.MPIArray.wrap(amp, axis=2, comm=comm)
        phi = mpiarray.MPIArray.wrap(phi, axis=2, comm=comm)
        ww = mpiarray.MPIArray.wrap(ww, axis=2, comm=comm)

        amp = amp.redistribute(1)
        phi = phi.redistribute(1)
        ww = ww.redistribute(1)

        nsource = amp.local_shape[1]
        ilocal = range(amp.local_offset[1], amp.local_offset[1] + nsource)

        logger.info("I am processing %d noise source inputs." % nsource)

        amp = amp[:].view(np.ndarray)
        phi = phi[:].view(np.ndarray)
        ww = ww[:].view(np.ndarray)

    # If a frequency is flagged more than `threshold` fraction of the time, then flag it entirely
    ww *= (
        (
            np.sum(ww > 0.0, axis=-1, dtype=np.float32, keepdims=True)
            / float(ww.shape[-1])
        )
        > threshold
    ).astype(np.float32)

    logger.info(
        "%0.1f percent of frequencies will be used to construct timing correction."
        % (
            100.0
            * np.sum(np.any(ww > 0.0, axis=(1, 2)), dtype=np.float32)
            / float(ww.shape[0]),
        )
    )

    # If the starting values for the mean and variance were not provided,
    # then estimate them from the data.
    if static_phi is None:
        static_phi = _flagged_median(phi, ww, axis=-1)
    else:
        sphi_ind = np.array([sphi_ind.index(ilcl) for ilcl in ilocal])
        static_phi = static_phi[:, sphi_ind]

    if weight_static_phi is None:
        weight_static_phi = np.ones(ww.shape[0:2], dtype=np.float32)
    else:
        wsphi_ind = np.array([wsphi_ind.index(ilcl) for ilcl in ilocal])
        weight_static_phi = weight_static_phi[:, wsphi_ind]

    if static_amp is None:
        static_amp = _flagged_median(amp, ww, axis=-1)
    else:
        samp_ind = np.array([samp_ind.index(ilcl) for ilcl in ilocal])
        static_amp = static_amp[:, samp_ind]

    if weight_static_amp is None:
        weight_static_amp = np.ones(ww.shape[0:2], dtype=np.float32)
    else:
        wsamp_ind = np.array([wsamp_ind.index(ilcl) for ilcl in ilocal])
        weight_static_amp = weight_static_amp[:, wsamp_ind]

    # Fit frequency dependence of amplitude and phase
    # damp = asc * dalpha    and    dphi = omega * dtau
    asc = (
        _amplitude_scaling(freq[:, np.newaxis, np.newaxis])
        * static_amp[:, :, np.newaxis]
    )

    omega = FREQ_TO_OMEGA * freq[:, np.newaxis, np.newaxis]

    # Estimate variance of each frequency from residuals
    for iter_weight in range(max_iter_weight):

        msg = ["Iteration %d of %d" % (iter_weight + 1, max_iter_weight)]

        dphi = _correct_phase_wrap(phi - static_phi[:, :, np.newaxis])
        damp = amp - static_amp[:, :, np.newaxis]

        weight_amp = ww * weight_static_amp[:, :, np.newaxis]
        weight_phi = ww * weight_static_phi[:, :, np.newaxis]

        # Construct alpha template
        alpha = np.sum(weight_amp * asc * damp, axis=0) * tools.invert_no_zero(
            np.sum(weight_amp * asc ** 2, axis=0)
        )

        # Construct delay template
        tau = np.sum(weight_phi * omega * dphi, axis=0) * tools.invert_no_zero(
            np.sum(weight_phi * omega ** 2, axis=0)
        )

        # Calculate amplitude residuals
        ramp = damp - asc * alpha[np.newaxis, :, :]

        # Calculate phase residuals
        rphi = dphi - omega * tau[np.newaxis, :, :]

        # Calculate the mean and variance of the amplitude residuals
        inv_num = tools.invert_no_zero(np.sum(ww, axis=-1))
        mu_ramp = np.sum(ww * ramp, axis=-1) * inv_num
        var_ramp = (
            np.sum(ww * (ramp - mu_ramp[:, :, np.newaxis]) ** 2, axis=-1) * inv_num
        )

        # Calculate the mean and variance of the phase residuals
        mu_rphi = np.sum(ww * rphi, axis=-1) * inv_num
        var_rphi = (
            np.sum(ww * (rphi - mu_rphi[:, :, np.newaxis]) ** 2, axis=-1) * inv_num
        )

        # Update the static quantities
        static_amp = static_amp + mu_ramp
        static_phi = static_phi + mu_rphi

        weight_static_amp = tools.invert_no_zero(var_ramp)
        weight_static_phi = tools.invert_no_zero(var_rphi)

        # Flag outliers
        not_outlier = np.ones_like(ww)
        if check_amp:
            nsigma = np.abs(ramp) * np.sqrt(weight_static_amp[:, :, np.newaxis])
            not_outlier *= (nsigma < nsigma_amp[iter_weight]).astype(np.float32)
            msg.append("nsigma_amp = %0.1f" % nsigma_amp[iter_weight])

        if check_phi:
            nsigma = np.abs(rphi) * np.sqrt(weight_static_phi[:, :, np.newaxis])
            not_outlier *= (nsigma < nsigma_phi[iter_weight]).astype(np.float32)
            msg.append("nsigma_phi = %0.1f" % nsigma_phi[iter_weight])

        if check_amp or check_phi:
            ww *= not_outlier

        logger.info(" | ".join(msg))

    # Calculate the number of frequencies used in the fit
    num_freq = np.sum(weight_amp > 0.0, axis=0, dtype=np.int)

    # Calculate the uncertainties on the fit parameters
    weight_tau = np.sum(weight_phi * omega ** 2, axis=0)
    weight_alpha = np.sum(weight_amp * asc ** 2, axis=0)

    # Calculate the average delay over this period using non-linear
    # least squares that is insensitive to phase wrapping
    if static_phi_fit is None:

        err_static_phi = np.sqrt(tools.invert_no_zero(weight_static_phi))

        static_phi_fit = np.zeros((nparam, nsource), dtype=np.float64)
        for nn in range(nsource):
            if np.sum(err_static_phi[:, nn] > 0.0, dtype=np.int) > nparam:
                static_phi_fit[:, nn] = fit_poly_to_phase(
                    freq,
                    np.exp(1.0j * static_phi[:, nn]),
                    err_static_phi[:, nn],
                    nparam=nparam,
                )[0]
    else:
        sphifit_ind = np.array([sphifit_ind.index(ilcl) for ilcl in ilocal])
        static_phi_fit = static_phi_fit[:, sphifit_ind]

    # Convert the outputs to MPIArrays distributed over input
    if parallel:
        tau = mpiarray.MPIArray.wrap(tau, axis=0, comm=comm)
        alpha = mpiarray.MPIArray.wrap(alpha, axis=0, comm=comm)

        weight_tau = mpiarray.MPIArray.wrap(weight_tau, axis=0, comm=comm)
        weight_alpha = mpiarray.MPIArray.wrap(weight_alpha, axis=0, comm=comm)

        static_phi = mpiarray.MPIArray.wrap(static_phi, axis=1, comm=comm)
        static_amp = mpiarray.MPIArray.wrap(static_amp, axis=1, comm=comm)

        weight_static_phi = mpiarray.MPIArray.wrap(weight_static_phi, axis=1, comm=comm)
        weight_static_amp = mpiarray.MPIArray.wrap(weight_static_amp, axis=1, comm=comm)

        static_phi_fit = mpiarray.MPIArray.wrap(static_phi_fit, axis=1, comm=comm)

        num_freq = mpiarray.MPIArray.wrap(num_freq, axis=0, comm=comm)

        phi = mpiarray.MPIArray.wrap(phi, axis=1, comm=comm)
        amp = mpiarray.MPIArray.wrap(amp, axis=1, comm=comm)

        weight_phi = mpiarray.MPIArray.wrap(weight_phi, axis=1, comm=comm)
        weight_amp = mpiarray.MPIArray.wrap(weight_amp, axis=1, comm=comm)

        data.redistribute("freq")

    # Return results
    return dict(
        tau=tau,
        alpha=alpha,
        weight_tau=weight_tau,
        weight_alpha=weight_alpha,
        static_phi=static_phi,
        static_amp=static_amp,
        weight_static_phi=weight_static_phi,
        weight_static_amp=weight_static_amp,
        static_phi_fit=static_phi_fit,
        num_freq=num_freq,
        phi=phi,
        amp=amp,
        weight_phi=weight_phi,
        weight_amp=weight_amp,
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


def eigen_decomposition(vis, flag):
    """Eigenvalue decomposition of the visibility matrix.

    Parameters
    ----------
    vis: np.ndarray[nfreq, nprod, ntime]
        Upper-triangle, product packed visibility matrix.
    flag: np.ndarray[nfreq, nsource, ntime] (optional)
        Array of 1 or 0 indicating the inputs that should be included
        in the eigenvalue decomposition for each frequency and time.

    Returns
    -------
    resp: np.ndarray[nfreq, nsource, ntime]
        Eigenvector corresponding to the largest eigenvalue for
        each frequency and time.
    """
    nfreq, nprod, ntime = vis.shape
    nsource = int((np.sqrt(8 * nprod + 1) - 1) // 2)

    # Do not bother performing the eigen-decomposition for
    # times and frequencies that are entirely flagged
    ind = np.where(np.any(flag, axis=1))
    ind = (ind[0], slice(None), ind[1])

    # Indexing the flag and vis datasets with ind flattens
    # the frequency and time dimension.  This results in
    # flg having shape (nfreq x ntime, nsource) and
    # Q having shape (nfreq x ntime, nsource, nsource).
    flg = flag[ind].astype(np.float32)

    Q = (
        flg[:, :, np.newaxis]
        * flg[:, np.newaxis, :]
        * tools.unpack_product_array(vis[ind], axis=1)
    )

    # Solve for eigenvectors and eigenvalues
    evals, evecs = np.linalg.eigh(Q)

    # Set phase convention
    sign0 = 1.0 - 2.0 * (evecs[:, np.newaxis, 0, -1].real < 0.0)

    # Determine response of each source
    resp = np.zeros((nfreq, nsource, ntime), dtype=vis.dtype)
    resp[ind] = flg * sign0 * evecs[:, :, -1] * evals[:, np.newaxis, -1] ** 0.5

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
    y = np.concatenate((y_complex.real, y_complex.imag)).astype(np.float64)

    x = np.tile(freq[flg], 2).astype(np.float64)

    err = np.tile(resp_error[flg], 2).astype(np.float64)

    # Initial guess for parameters
    p0 = np.zeros(nparam, dtype=np.float64)
    p0[1] = np.median(
        np.diff(np.angle(y_complex)) / (FREQ_TO_OMEGA * np.diff(freq[flg]))
    )
    p0[0] = np.median(
        _correct_phase_wrap(np.angle(y_complex) - p0[1] * FREQ_TO_OMEGA * freq[flg])
    )

    # Try nonlinear least squares fit
    try:
        popt, pcov = scipy.optimize.curve_fit(
            _func_poly_phase, x, y, p0=p0.copy(), sigma=err, absolute_sigma=False
        )

    except Exception as excep:
        logger.warning("Nonlinear phase fit failed with error:  %s" % excep)
        # Fit failed, return the initial parameter estimates
        popt = p0
        pcov = np.zeros((nparam, nparam), dtype=np.float64)

    finally:
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


def _weight_propagation_addition(*args):

    sum_variance = np.zeros_like(args[0])
    for weight in args:
        sum_variance += tools.invert_no_zero(weight)

    return tools.invert_no_zero(sum_variance)


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


def _search_nearest(x, xeval):

    index_next = np.searchsorted(x, xeval, side="left")

    index_previous = np.maximum(0, index_next - 1)
    index_next = np.minimum(x.size - 1, index_next)

    index = np.where(
        np.abs(xeval - x[index_previous]) < np.abs(xeval - x[index_next]),
        index_previous,
        index_next,
    )

    return index


def _interpolation_nearest(x, y, var, xeval):

    index = _search_nearest(x, xeval)

    yeval = y[index]
    weval = tools.invert_no_zero(var[index])

    return yeval, weval


def _interpolation_previous(x, y, var, xeval):

    index = np.maximum(np.searchsorted(x, xeval, side="right") - 1, 0)
    return y[index], tools.invert_no_zero(var[index])


def _interpolation_next(x, y, var, xeval):

    index = np.minimum(np.searchsorted(x, xeval, side="left"), x.size - 1)
    return y[index], tools.invert_no_zero(var[index])


def _interpolation_linear(x, y, var, xeval):

    index = np.searchsorted(x, xeval, side="left")

    ind1 = index - 1
    ind2 = index

    below = np.flatnonzero(ind1 == -1)
    if below.size > 0:
        ind1[below] = 0
        ind2[below] = 1

    above = np.flatnonzero(ind2 == x.size)
    if above.size > 0:
        ind1[above] = x.size - 2
        ind2[above] = x.size - 1

    adx1 = xeval - x[ind1]
    adx2 = x[ind2] - xeval

    norm = tools.invert_no_zero(adx1 + adx2)
    a1 = adx2 * norm
    a2 = adx1 * norm

    yeval = a1 * y[ind1] + a2 * y[ind2]
    weval = tools.invert_no_zero(a1 ** 2 * var[ind1] + a2 ** 2 * var[ind2])

    return yeval, weval


def _interpolation_scipy(kind):
    def _interp1d(x, y, var, xeval):

        interpolator = scipy.interpolate.interp1d(
            x, y, kind=kind, fill_value="extrapolate"
        )
        yeval = interpolator(xeval)

        # For the scipy interpolation, we do not attempt to propagate the errors.
        # Instead we just use the weight from the nearest point.
        index = _search_nearest(x, xeval)
        weval = tools.invert_no_zero(var[index])

        return yeval, weval

    return _interp1d


INTERPOLATION_LOOKUP = {
    "nearest": _interpolation_nearest,
    "previous": _interpolation_previous,
    "next": _interpolation_next,
    "linear": _interpolation_linear,
}
