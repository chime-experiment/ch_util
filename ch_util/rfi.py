"""Tools for RFI flagging

This module contains tools for finding and removing Radio Frequency Interference
(RFI).

Note that this generates masks where the elements containing RFI are marked as
:obj:`True`, and the remaining elements are marked :obj:`False`. This is in
contrast to the routines in :mod:`ch_pipeline.rfi` which generates a inverse
noise weighting, where RFI containing elements are effectively :obj:`False`, and
the remainder are :obj:`True`.

There are general purpose routines for flagging RFI in `andata` like datasets:

- :py:meth:`flag_dataset`
- :py:meth:`number_deviations`

For more control there are specific routines that can be called:

- :py:meth:`mad_cut_2d`
- :py:meth:`mad_cut_1d`
- :py:meth:`mad_cut_rolling`
- :py:meth:`spectral_cut`
- :py:meth:`frequency_mask`
- :py:meth:`sir1d`
- :py:meth:`sir`
"""

import warnings
import logging

import numpy as np
import scipy.signal as sig

from ch_ephem.observers import chime

from . import tools

# Set up logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# Ranges of bad frequencies given by their start time (in unix time) and
# corresponding start and end frequencies (in MHz)
# If the start time is not specified, t = [], the flag is applied to all CSDs
BAD_FREQUENCIES = {
    "chime": [
        ### Bad bands at first light
        [[None, None], [449.41, 450.98]],
        [[None, None], [454.88, 456.05]],
        [[None, None], [457.62, 459.18]],
        [[None, None], [483.01, 485.35]],
        [[None, None], [487.70, 494.34]],
        [[None, None], [497.85, 506.05]],
        [[None, None], [529.10, 536.52]],
        [[None, None], [541.60, 548.00]],
        ### Additional bad bands
        # Narrow, high power bands visible in sensitivities and
        # some longer baselines. There is some sporadic rfi in between
        # the two bands
        [[None, None], [460.15, 460.55]],
        [[None, None], [464.00, 470.32]],
        # 6 MHz band (reported by Simon)
        [[None, None], [505.85, 511.71]],
        # Bright band which has been present since early on
        [[None, None], [517.97, 525.00]],
        # UHF TV Channel 27 ending CSD 3212 inclusive (2022/08/24)
        # This is extended until CSD 3446 (2023/04/13) to account for gain errors
        [[None, 1681410777], [548.00, 554.49]],
        [[None, None], [564.65, 578.00]],
        # UHF TV Channel 32 ending CSD 3213 inclusive (2022/08/25)
        # This is extended until CSD 3446 (2023/04/13) to account for gain errors
        [[None, 1681410777], [578.00, 585.35]],
        # from CSD 2893 (2021/10/09 - ) UHF TV Channel 33 (reported by Seth)
        [[1633758888, None], [584.00, 590.00]],
        # UHF TV Channel 35
        [[1633758888, None], [596.00, 602.00]],
        # Low power band visible in long baselines
        [[None, None], [602.00, 607.82]],
        # from CSD 2243 (2019/12/31 - ) Rogers’ new 600 MHz band
        [[1577755022, None], [617.00, 627.00]],
        [[None, None], [693.16, 693.55]],
        [[None, None], [694.34, 696.68]],
        # from CSD 2080 (2019/07/21 - ) Blobs, Channels 55 and 56
        [[1564051033, None], [716.00, 728.00]],
        [[None, None], [729.88, 745.12]],
        [[None, None], [746.29, 756.45]],
    ],
    "kko": [
        # Bad bands from statistical analysis of Jan 20, 2023 N2 data
        [[None, None], [433.59, 433.98]],
        [[None, None], [439.84, 440.62]],
        [[None, None], [483.20, 484.38]],
        [[None, None], [616.80, 626.95]],
        [[None, None], [799.61, 800.00]],
        # Notch filter stoppband + leakage
        [[None, None], [710.55, 757.81]],
    ],
    "gbo": [],
    "hco": [],
}


def flag_dataset(
    data, freq_width=10.0, time_width=420.0, threshold=5.0, flag1d=False, rolling=False
):
    """RFI flag the dataset.  This function wraps `number_deviations`,
    and remains largely for backwards compatability.  The pipeline code
    now calls `number_deviations` directly.

    Parameters
    ----------
    data : `andata.CorrData`
        Must contain vis and weight attribute that are both
        `np.ndarray[nfreq, nprod, ntime]`.  Note that this
        function does not work with CorrData that has
        been stacked over redundant baselines.
    freq_width : float
        Frequency interval in *MHz* to compare across.
    time_width : float
        Time interval in *seconds* to compare.
    threshold : float
        Threshold in MAD over which to cut out RFI.
    rolling : bool
        Use a rolling window instead of distinct blocks.
    flag1d : bool, optional
        Only apply the MAD cut in the time direction. This is useful if the
        frequency coverage is sparse.

    Returns
    -------
    mask : np.ndarray
        RFI mask, output shape is the same as input visibilities.
    """

    auto_ii, auto_vis, auto_ndev = number_deviations(
        data,
        freq_width=freq_width,
        time_width=time_width,
        flag1d=flag1d,
        rolling=rolling,
        stack=False,
    )

    auto_mask = np.abs(auto_ndev) > threshold

    # Apply the frequency cut to the data (add here because we are distributed
    # over products and its easy)
    if "time" in data.index_map:
        timestamp = data.time[0]
    elif "ra" in data.index_map:
        timestamp = chime.lsd_to_unix(data.attrs["lsd"])

    freq_mask = frequency_mask(data.freq[:], timestamp=timestamp)
    auto_ii, auto_mask = np.logical_or(auto_mask, freq_mask[:, np.newaxis, np.newaxis])

    # Create an empty mask for the full dataset
    mask = np.zeros(data.vis[:].shape, dtype=bool)

    # Loop over all products and flag if either inputs auto correlation was flagged
    for pi in range(data.nprod):
        ii, ij = data.index_map["prod"][pi]

        if ii in auto_ii:
            ai = auto_ii.index(ii)
            mask[:, pi] = np.logical_or(mask[:, pi], auto_mask[:, ai])

        if ij in auto_ii:
            aj = auto_ii.index(ij)
            mask[:, pi] = np.logical_or(mask[:, pi], auto_mask[:, aj])

    return mask


def number_deviations(
    data,
    freq_width=10.0,
    time_width=420.0,
    flag1d=False,
    apply_static_mask=False,
    rolling=False,
    stack=False,
    normalize=False,
    fill_value=None,
):
    """Calculate the number of median absolute deviations (MAD)
    of the autocorrelations from the local median.

    Parameters
    ----------
    data : `andata.CorrData`
        Must contain vis and weight attributes that are both
        `np.ndarray[nfreq, nprod, ntime]`.
    freq_width : float
        Frequency interval in *MHz* to compare across.
    time_width : float
        Time interval in *seconds* to compare across.
    flag1d : bool
        Only apply the MAD cut in the time direction. This is useful if the
        frequency coverage is sparse.
    apply_static_mask : bool
        Apply static mask obtained from `frequency_mask` before computing
        the median absolute deviation.
    rolling : bool
        Use a rolling window instead of distinct blocks.
    stack: bool
        Average over all autocorrelations.
    normalize : bool
        Normalize by the median value over time prior to averaging over
        autocorrelations.  Only relevant if `stack` is True.
    fill_value: float
        Data that was already flagged as bad will be set to this value in
        the output array.  Should be a large positive value that is greater
        than the threshold that will be placed.  Default is float('Inf').

    Returns
    -------
    auto_ii: np.ndarray[ninput,]
        Index of the inputs that have been processed.
        If stack is True, then [0] will be returned.

    auto_vis: np.ndarray[nfreq, ninput, ntime]
        The autocorrelations that were used to calculate
        the number of deviations.

    ndev : np.ndarray[nfreq, ninput, ntime]
        Number of median absolute deviations of the autocorrelations
        from the local median.
    """
    from caput import memh5, mpiarray

    if fill_value is None:
        fill_value = float("Inf")

    # Check if dataset is parallel
    parallel = isinstance(data.vis, memh5.MemDatasetDistributed)

    data.redistribute("freq")

    # Extract the auto correlations
    auto_ii, auto_vis, auto_flag = get_autocorrelations(data, stack, normalize)

    # Calculate time interval in samples. If the data has an ra axis instead,
    # use an estimation of the time per sample
    if "time" in data.index_map:
        twidth = int(time_width / np.median(np.abs(np.diff(data.time)))) + 1
        timestamp = data.time[0]
    elif "ra" in data.index_map:
        twidth = int(time_width * len(data.ra[:]) / 86164.0) + 1
        timestamp = chime.lsd_to_unix(data.attrs["lsd"])
    else:
        raise TypeError(
            f"Expected data type with a `time` or `ra` axis. Got {type(data)}."
        )

    # Create static flag of frequencies that are known to be bad
    static_flag = (
        ~frequency_mask(data.freq[:], timestamp=timestamp)
        if apply_static_mask
        else np.ones(len(data.freq[:]), dtype=bool)
    )[:, np.newaxis]

    if parallel:
        # Ensure these are distributed across frequency
        auto_vis = auto_vis.redistribute(0)
        auto_flag = auto_flag.redistribute(0)
        static_flag = mpiarray.MPIArray.wrap(static_flag[auto_vis.local_bounds], axis=0)

    # Calculate frequency interval in bins
    fwidth = (
        int(freq_width / np.median(np.abs(np.diff(data.freq)))) + 1 if not flag1d else 1
    )

    # Create an empty array for number of median absolute deviations
    ndev = np.zeros_like(auto_vis, dtype=np.float32)

    auto_flag_view = auto_flag.allgather() if parallel else auto_flag
    static_flag_view = static_flag.allgather() if parallel else static_flag
    ndev_view = ndev.local_array if parallel else ndev

    # Loop over extracted autos and create a mask for each
    for ind in range(auto_vis.shape[1]):
        flg = static_flag_view & auto_flag_view[:, ind]
        # Gather enire array onto each rank
        arr = auto_vis[:, ind].allgather() if parallel else auto_vis[:, ind]
        # Use NaNs to ignore previously flagged data when computing the MAD
        arr = np.where(flg, arr.real, np.nan)
        local_bounds = auto_vis.local_bounds if parallel else slice(None)
        # Apply RFI flagger
        if rolling:
            # Limit bounds to the local portion of the array
            ndev_i = mad_cut_rolling(
                arr, twidth=twidth, fwidth=fwidth, mask=False, limit_range=local_bounds
            )
        elif flag1d:
            ndev_i = mad_cut_1d(arr[local_bounds, :], twidth=twidth, mask=False)
        else:
            ndev_i = mad_cut_2d(
                arr[local_bounds, :], twidth=twidth, fwidth=fwidth, mask=False
            )

        ndev_view[:, ind, :] = ndev_i

    # Fill any values equal to NaN with the user specified fill value
    ndev_view[~np.isfinite(ndev_view)] = fill_value

    return auto_ii, auto_vis, ndev


def get_autocorrelations(
    data, stack: bool = False, normalize: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract autocorrelations from a data stack.

    Parameters
    ----------
    data : `andata.CorrData`
        Must contain vis and weight attributes that are both
        `np.ndarray[nfreq, nprod, ntime]`.
    stack: bool, optional
        Average over all autocorrelations.
    normalize : bool, optional
        Normalize by the median value over time prior to averaging over
        autocorrelations.  Only relevant if `stack` is True.

    Returns
    -------
    auto_ii: np.ndarray[ninput,]
        Index of the inputs that have been processed.
        If stack is True, then [0] will be returned.

    auto_vis: np.ndarray[nfreq, ninput, ntime]
        The autocorrelations that were used to calculate
        the number of deviations.

    auto_flag: np.ndarray[nfreq, ninput, ntime]
        Indices where data weights are positive
    """
    # Extract the auto correlations
    prod = data.index_map["prod"][data.index_map["stack"]["prod"]]
    auto_ii, auto_pi = np.array(
        list(zip(*[(pp[0], ind) for ind, pp in enumerate(prod) if pp[0] == pp[1]]))
    )

    auto_vis = data.vis[:, auto_pi, :].real.copy()

    # If requested, average over all inputs to construct the stacked autocorrelations
    # for the instrument (also known as the incoherent beam)
    if stack:
        weight = (data.weight[:, auto_pi, :] > 0.0).astype(np.float32)

        # Do not include bad inputs in the average
        partial_stack = data.index_map["stack"].size < data.index_map["prod"].size

        if not partial_stack and hasattr(data, "input_flags"):
            input_flags = data.input_flags[:]
            good_inputs = np.mean(np.sum(input_flags, axis=0), axis=-1)
            logger.info(f"There are on average {good_inputs} good inputs.")

            if np.any(input_flags) and not np.all(input_flags):
                logger.info("Applying input_flags to weight.")
                weight *= input_flags[np.newaxis, auto_ii, :].astype(weight.dtype)

        if normalize:
            logger.info("Normalizing autocorrelations prior to stacking.")
            med_auto = nanmedian(
                np.where(weight, auto_vis, np.nan), axis=-1, keepdims=True
            )
            med_auto = np.where(np.isfinite(med_auto), med_auto, 0.0)
            auto_vis *= tools.invert_no_zero(med_auto)

        norm = np.sum(weight, axis=1, keepdims=True)

        auto_vis = np.sum(
            weight * auto_vis, axis=1, keepdims=True
        ) * tools.invert_no_zero(norm)

        auto_flag = norm > 0.0
        auto_ii = np.zeros(1, dtype=int)

    else:
        auto_flag = data.weight[:, auto_pi, :] > 0.0

    return auto_ii, auto_vis, auto_flag


def spectral_cut(data, fil_window=15, only_autos=False):
    """Flag out the TV bands, or other constant spectral RFI.

    Parameters
    ----------
    data : `andata.obj`
        If `only_autos` shape is (freq, n_feeds, time), else (freq, n_prod,
        time).
    fil_window : integer
        Window of median filter for baseline of chime spectrum. Default is 15.
    only_autos : boolean
        Whether data contains only autos or not.

    Returns
    -------
    mask: np.ndarray[freq,time]
          RFI mask (no product axis).
    """

    if only_autos:
        data_vis = data.vis[:].real
    else:
        nfeed = int((2 * data.vis.shape[1]) ** 0.5)
        auto_ind = [tools.cmap(i, i, nfeed) for i in range(nfeed)]
        data_vis = data.vis[:, auto_ind].real

    stack_autos = np.mean(data_vis, axis=1)
    stack_autos_time_ave = np.mean(stack_autos, axis=-1)

    # Locations of the generally decent frequency bands
    if "time" in data.index_map:
        timestamp = data.time[0]
    elif "ra" in data.index_map:
        timestamp = chime.lsd_to_unix(data.attrs["lsd"])

    drawn_bool_mask = frequency_mask(data.freq[:], timestamp=timestamp)
    good_data = np.logical_not(drawn_bool_mask)

    # Calculate standard deivation of the average channel
    std_arr = np.std(stack_autos, axis=-1)
    sigma = np.median(std_arr) / np.sqrt(
        stack_autos.shape[1]
    )  # standard deviation of the mean

    # Smooth with a median filter, and then interpolate to estimate the
    # baseline of the spectrum
    fa = np.arange(data_vis.shape[0])
    medfilt = sig.medfilt(stack_autos_time_ave[good_data], fil_window)
    interpolat_arr_baseline = np.interp(fa, fa[good_data], medfilt)
    rel_pow = stack_autos_time_ave - interpolat_arr_baseline

    # Mask out frequencies with too much power
    mask_1d = rel_pow > 10 * sigma

    # Generate mask
    mask = np.zeros((data_vis.shape[0], data_vis.shape[2]), dtype=bool)
    mask[:] = mask_1d[:, None]

    return mask


def frequency_mask(
    freq_centre: np.ndarray,
    freq_width: np.ndarray | float | None = None,
    timestamp: np.ndarray | float | None = None,
    instrument: str | None = "chime",
) -> np.ndarray:
    """Flag known bad frequencies.

    Time dependent static RFI flags that affect the recent observations are added.

    Parameters
    ----------
    freq_centre
        Centre of each frequency channel
    freq_width
        Width of each frequency channel. If `None` (default), calculate the width from
        the frequency centre separation. If supplied as an array it must be
        broadcastable
        against `freq_centre`.
    timestamp
        UNIX observing time. If `None` (default) mask all specified bands regardless of
        their start/end times, otherwise mask only timestamps within the band start and
        end times. If supplied as an array it must be broadcastable against
        `freq_centre`.
    instrument
        Telescope name. [kko, gbo, hco, chime (default)]

    Returns
    -------
    mask
        An array marking the bad frequency channels. The final shape is the result of
        broadcasting `freq_centre` and `timestamp` together.
    """
    if freq_width is None:
        freq_width = np.abs(np.median(np.diff(freq_centre)))

    freq_start = freq_centre - freq_width / 2
    freq_end = freq_centre + freq_width / 2

    # Broadcast to get the output mask
    mask = np.zeros(np.broadcast(freq_centre, timestamp).shape, dtype=bool)

    try:
        bad_freq = BAD_FREQUENCIES[instrument]
    except KeyError as e:
        raise ValueError(f"No RFI flags defined for {instrument}") from e

    for (start_time, end_time), (fs, fe) in bad_freq:
        fmask = (freq_end > fs) & (freq_start < fe)

        # If we don't have a timestamp then just mask all bands
        if timestamp is None:
            tmask = True
        else:
            # Otherwise calculate the mask based on the start and end times
            tmask = np.ones_like(timestamp, dtype=bool)

            if start_time is not None:
                tmask &= timestamp >= start_time

            if end_time is not None:
                tmask &= timestamp <= end_time

        # Mask frequencies and times specified in this band
        mask |= tmask & fmask

    return mask


def mad_cut_2d(data, fwidth=64, twidth=42, threshold=5.0, freq_flat=True, mask=True):
    """Mask out RFI using a median absolute deviation cut in time-frequency blocks.

    Parameters
    ----------
    data : np.ndarray[freq, time]
        Array of data to mask.
    fwidth : integer, optional
        Number of frequency samples to average median over.
    twidth : integer, optional
        Number of time samples to average median over.
    threshold : scalar, optional
        Number of median deviations above which we cut the data.
    freq_flat : boolean, optional
        Flatten in the frequency direction by dividing through by the median.
    mask : boolean, optional
        If True return the mask, if False return the number of
        median absolute deviations.

    Returns
    -------
    mask : np.ndarray[freq, time]
        Mask or number of median absolute deviations for each sample.
    """

    median = nanmedian if np.any(~np.isfinite(data)) else np.median

    flen = int(np.ceil(data.shape[0] * 1.0 / fwidth))
    tlen = int(np.ceil(data.shape[1] * 1.0 / twidth))

    if mask:
        madmask = np.ones(data.shape, dtype="bool")
    else:
        madmask = np.ones(data.shape, dtype=np.float64)

    if freq_flat:
        # Flatten
        mfd = tools.invert_no_zero(median(data, axis=1))
        data *= mfd[:, np.newaxis]

    ## Iterate over all frequency and time blocks
    #
    # This can be done more quickly by reshaping the arrays into blocks, but
    # only works when there are an integer number of blocks. Probably best to
    # rewrite in cython.
    for fi in range(flen):
        fs = fi * fwidth
        fe = min((fi + 1) * fwidth, data.shape[0])

        for ti in range(tlen):
            ts = ti * twidth
            te = min((ti + 1) * twidth, data.shape[1])

            dsec = data[fs:fe, ts:te]
            msec = madmask[fs:fe, ts:te]

            mval = median(dsec.flatten())
            dev = dsec - mval
            med_abs_dev = median(np.abs(dev).flatten())

            med_inv = tools.invert_no_zero(med_abs_dev)

            if mask:
                msec[:] = (np.abs(dev) * med_inv) > threshold
            else:
                msec[:] = dev * med_inv

    return madmask


def mad_cut_1d(data, twidth=42, threshold=5.0, mask=True):
    """Mask out RFI using a median absolute deviation cut in the time direction.

    This is useful for datasets with sparse frequency coverage. Functionally
    this routine is equivalent to :func:`mad_cut_2d` with `fwidth = 1`, but will
    be much faster.

    Parameters
    ----------
    data : np.ndarray[freq, time]
        Array of data to mask.
    twidth : integer, optional
        Number of time samples to average median over.
    threshold : scalar, optional
        Number of median deviations above which we cut the data.
    mask : boolean, optional
        If True return the mask, if False return the number of
        median absolute deviations.

    Returns
    -------
    mask : np.ndarray[freq, time]
        Mask or number of median absolute deviations for each sample.
    """

    median = nanmedian if np.any(~np.isfinite(data)) else np.median

    tlen = int(np.ceil(data.shape[1] * 1.0 / twidth))

    if mask:
        madmask = np.ones(data.shape, dtype="bool")
    else:
        madmask = np.ones(data.shape, dtype=np.float64)

    ## Iterate over all time chunks
    for ti in range(tlen):
        ts = ti * twidth
        te = min((ti + 1) * twidth, data.shape[1])

        dsec = data[:, ts:te]
        msec = madmask[:, ts:te]

        mval = median(dsec, axis=1)
        dev = dsec - mval[:, np.newaxis]
        med_abs_dev = median(np.abs(dev), axis=1)

        med_inv = tools.invert_no_zero(med_abs_dev[:, np.newaxis])

        if mask:
            msec[:] = (np.abs(dev) * med_inv) > threshold
        else:
            msec[:] = dev * med_inv

    return madmask


# Define several functions for creating 2D rolling window
def _rolling_window_lastaxis(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def _rolling_window(a, window):
    if not hasattr(window, "__iter__"):
        return _rolling_window_lastaxis(a, window)
    for i, win in enumerate(window):
        a = a.swapaxes(i, -1)
        a = _rolling_window_lastaxis(a, win)
        a = a.swapaxes(-2, i)
    return a


def mad_cut_rolling(
    data,
    fwidth=64,
    twidth=42,
    threshold=5.0,
    freq_flat=True,
    mask=True,
    limit_range: slice = slice(None),
):
    """Mask out RFI by placing a cut on the absolute deviation.
    Compared to `mad_cut_2d`, this function calculates
    the median and median absolute deviation using a rolling
    2D median filter, i.e., for every (freq, time) sample a
    separate estimates of these statistics is obtained for a
    window that is centered on that sample.

    For sparsely sampled frequency axis, set fwidth = 1.

    Parameters
    ----------
    data : np.ndarray[freq, time]
        Array of data to mask.
    fwidth : integer, optional
        Number of frequency samples to calculate median over.
    twidth : integer, optional
        Number of time samples to calculate median over.
    threshold : scalar, optional
        Number of median absolute deviations above which we cut the data.
    freq_flat : boolean, optional
        Flatten in the frequency direction by dividing each frequency
        by the median over time.
    mask : boolean, optional
        If True return the mask, if False return the number of
        median absolute deviations.
    limit_range : slice, optional
        Data is limited to this range in the freqeuncy axis. Defaults to slice(None).

    Returns
    -------
    mask : np.ndarray[freq, time]
        Mask or number of median absolute deviations for each sample.
    """
    # Make sure we have an odd number of samples
    fwidth += int(not (fwidth % 2))
    twidth += int(not (twidth % 2))

    foff = fwidth // 2
    toff = twidth // 2

    nfreq, ntime = data.shape

    # If requested, flatten over the frequency direction.
    if freq_flat:
        mfd = tools.invert_no_zero(nanmedian(data, axis=1))
        data *= mfd[:, np.newaxis]

    # Add NaNs around the edges of the array so that we don't have to
    # treat them separately
    eshp = [nfreq + fwidth - 1, ntime + twidth - 1]
    exp_data = np.full(eshp, np.nan, dtype=data.dtype)
    exp_data[foff : foff + nfreq, toff : toff + ntime] = data

    if limit_range != slice(None):
        # Get only desired slice
        expsl = slice(
            max(limit_range.start, 0),
            min(limit_range.stop + 2 * foff, exp_data.shape[0]),
        )
        dsl = slice(max(limit_range.start, 0), min(limit_range.stop, data.shape[0]))
        exp_data = exp_data[expsl, :]
        data = data[dsl, :]

    # Use numpy slices to construct the rolling windowed data
    win_data = _rolling_window(exp_data, (fwidth, twidth))

    # Compute the local median and median absolute deviation
    med = nanmedian(win_data, axis=(-2, -1))
    med_abs_dev = nanmedian(
        np.abs(win_data - med[..., np.newaxis, np.newaxis]), axis=(-2, -1)
    )

    inv_med_abs_dev = tools.invert_no_zero(med_abs_dev)

    # Calculate and return the mask or the number of median absolute deviations
    if mask:
        madmask = (np.abs(data - med) * inv_med_abs_dev) > threshold
    else:
        madmask = (data - med) * inv_med_abs_dev

    return madmask


def nanmedian(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
        return np.nanmedian(*args, **kwargs)


# Iterative HPF masking for identifying narrow-band features in gains or spectra
def highpass_delay_filter(freq, tau_cut, flag, epsilon=1e-10):
    """Construct a high-pass delay filter.

    The stop band will range from [-tau_cut, tau_cut].
    DAYENU is used to construct the filter in the presence
    of masked frequencies.  See Ewall-Wice et al. 2021
    (arXiv:2004.11397) for a description.

    Parameters
    ----------
    freq : np.ndarray[nfreq,]
        Frequency in MHz.
    tau_cut : float
        The half width of the stop band in micro-seconds.
    flag : np.ndarray[nfreq,]
        Boolean flag that indicates what frequencies are valid.
    epsilon : float
        The stop-band rejection of the filter.

    Returns
    -------
    pinv : np.ndarray[nfreq, nfreq]
        High pass delay filter.
    """

    nfreq = freq.size
    assert (flag.ndim == 1) and (flag.size == nfreq)

    mflag = (flag[:, np.newaxis] & flag[np.newaxis, :]).astype(np.float64)

    cov = np.eye(nfreq, dtype=np.float64)
    cov += (
        np.sinc(2.0 * tau_cut * (freq[:, np.newaxis] - freq[np.newaxis, :])) / epsilon
    )

    return np.linalg.pinv(cov * mflag, hermitian=True) * mflag


def iterative_hpf_masking(
    freq,
    y,
    flag=None,
    tau_cut=0.60,
    epsilon=1e-10,
    window=65,
    threshold=6.0,
    nperiter=1,
    niter=40,
    timestamp=None,
):
    """Mask features in a spectrum that have significant power at high delays.

    Uses the following iterative procedure to generate the mask:

        - Apply a high-pass filter to the spectrum.
        - For each frequency channel, calculate the median absolute
          deviation of nearby frequency channels to get an estimate
          of the noise.  Divide the high-pass filtered spectrum by
          the noise estimate.
        - Mask excursions with the largest signal to noise.
        - Regenerate the high-pass filter using the new mask.
        - Repeat.

    The procedure stops when the maximum number of iterations is reached
    or there are no excursions beyond some threshold.

    Parameters
    ----------
    freq: np.ndarray[nfreq,]
        Frequency in MHz.
    y: np.ndarray[nfreq,]
        Spectrum to search for narrowband features.
    flag: np.ndarray[nfreq,]
        Boolean flag where True indicates valid data.
    tau_cut: float
        Cutoff of the high-pass filter in microseconds.
    epsilon: float
        Stop-band rejection of the filter.
    threshold: float
        Number of median absolute deviations beyond which
        a frequency channel is considered an outlier.
    window: int
        Width of the window used to estimate the noise
        (by calculating a local median absolute deviation).
    nperiter: int
        Maximum number of frequency channels to flag
        on any iteration.
    niter: int
        Maximum number of iterations.
    timestamp : float
        Start observing time (in unix time)

    Returns
    -------
    yhpf: np.ndarray[nfreq,]
        The high-pass filtered spectrum generated using
        the mask from the last iteration.
    flag: np.ndarray[nfreq,]
        Boolean flag where True indicates valid data.
        This is the logical complement to the mask
        from the last iteration.
    rsigma: np.ndarray[nfreq,]
        The local median absolute deviation from the last
        iteration.
    """

    from caput import weighted_median

    assert y.ndim == 1

    # Make sure the frequencies are float64, otherwise
    # can have problems with construction of filter
    freq = freq.astype(np.float64)

    # Make sure the size of the window is odd
    window = window + int(not (window % 2))

    # If an initial flag was not provided, then use the static rfi mask.
    if flag is None:
        flag = ~frequency_mask(freq, timestamp=timestamp)

    # We will be updating the flags on each iteration.  Make a copy of
    # the input so that we do not overwrite.
    new_flag = flag.copy()

    # Iterate
    itt = 0
    while itt < niter:
        # Construct the filter using the current mask
        NF = highpass_delay_filter(freq, tau_cut, new_flag, epsilon=epsilon)

        # Apply the filter
        yhpf = np.matmul(NF, y)

        # Calculate the local median absolute deviation
        ry = np.ascontiguousarray(yhpf.astype(np.float64))
        w = np.ascontiguousarray(new_flag.astype(np.float64))

        rsigma = 1.48625 * weighted_median.moving_weighted_median(
            np.abs(ry), w, window, method="split"
        )

        # Calculate the signal to noise
        rs2n = np.abs(yhpf * tools.invert_no_zero(rsigma))

        # Identify frequency channels that are above the signal to noise threshold
        above_threshold = np.flatnonzero(rs2n > threshold)

        if above_threshold.size == 0:
            break

        # Find the largest nperiter frequency channels that are above the threshold
        ibad = above_threshold[np.argsort(-np.abs(yhpf[above_threshold]))[0:nperiter]]

        # Flag those frequency channels, increment the counter
        new_flag[ibad] = False

        itt += 1

    # Construct and apply the filter using the final flag
    NF = highpass_delay_filter(freq, tau_cut, new_flag, epsilon=epsilon)

    yhpf = np.matmul(NF, y)

    return yhpf, new_flag, rsigma


# Scale-invariant rank (SIR) functions
def sir1d(basemask, eta=0.2):
    """Numpy implementation of the scale-invariant rank (SIR) operator.

    For more information, see arXiv:1201.3364v2.

    Parameters
    ----------
    basemask : numpy 1D array of boolean type
        Array with the threshold mask previously generated.
        1 (True) for flagged points, 0 (False) otherwise.
    eta : float
        Aggressiveness of the method: with eta=0, no additional samples are
        flagged and the function returns basemask. With eta=1, all samples
        will be flagged. The authors in arXiv:1201.3364v2 seem to be convinced
        that 0.2 is a mostly universally optimal value, but no optimization
        has been done on CHIME data.

    Returns
    -------
    mask : numpy 1D array of boolean type
        The mask after the application of the (SIR) operator. Same shape and
        type as basemask.
    """
    n = basemask.size
    psi = basemask.astype(np.float64) - 1.0 + eta

    M = np.zeros(n + 1, dtype=np.float64)
    M[1:] = np.cumsum(psi)

    MP = np.minimum.accumulate(M)[:-1]
    MQ = np.concatenate((np.maximum.accumulate(M[-2::-1])[-2::-1], M[-1, np.newaxis]))

    return (MQ - MP) >= 0.0


def sir(basemask, eta=0.2, only_freq=False, only_time=False):
    """Apply the SIR operator over the frequency and time axes for each product.

    This is a wrapper for `sir1d`.  It loops over times, applying `sir1d`
    across the frequency axis.  It then loops over frequencies, applying `sir1d`
    across the time axis.  It returns the logical OR of these two masks.

    Parameters
    ----------
    basemask : np.ndarray[nfreq, nprod, ntime] of boolean type
        The previously generated threshold mask.
        1 (True) for masked points, 0 (False) otherwise.
    eta : float
        Aggressiveness of the method: with eta=0, no additional samples are
        flagged and the function returns basemask. With eta=1, all samples
        will be flagged.
    only_freq : bool
        Only apply the SIR operator across the frequency axis.
    only_time : bool
        Only apply the SIR operator across the time axis.

    Returns
    -------
    mask : np.ndarray[nfreq, nprod, ntime] of boolean type
        The mask after the application of the SIR operator.
    """
    if only_freq and only_time:
        raise ValueError("Only one of only_freq and only_time can be True.")

    nfreq, nprod, ntime = basemask.shape

    newmask = basemask.astype(bool).copy()

    for pp in range(nprod):
        if not only_time:
            for tt in range(ntime):
                newmask[:, pp, tt] |= sir1d(basemask[:, pp, tt], eta=eta)

        if not only_freq:
            for ff in range(nfreq):
                newmask[ff, pp, :] |= sir1d(basemask[ff, pp, :], eta=eta)

    return newmask
