"""
Data quality routines


Data quality functions
======================
- :py:meth:`good_channels`


Issues
======

Auxiliary functions are still lacking documentation.

"""

import numpy as np

import caput.time as ctime

import ch_util.ephemeris as ch_eph
from ch_util import andata
from ch_util import tools
from ch_util import ni_utils


def good_channels(
    data,
    gain_tol=10.0,
    noise_tol=2.0,
    fit_tol=0.02,
    test_freq=0,
    noise_synced=None,
    inputs=None,
    res_plot=False,
    verbose=True,
):
    """Test data for misbehaving channels.

    Three tests are performed:

    1. Excessively high digital gains,
    2. Compliance of noise to the radiometer equation and
    3. Goodness of fit to a template Tsky.

    See `Doclib:235
    <https://bao.phas.ubc.ca/doc/cgi-bin/general/documents/display?Id=235>`_
    file 'data_quality.pdf' for details on how the filters and  tolerances work.

    Parameters
    ----------
    data : ch_util.andata.CorrData object
        Data to run test on.
        If andata object contains cross-correlations,
        test is performed on auto-correlations only.
    gain_tol : float
        Tolerance for digital gains filter. Flag channels whose
        digital gain fractional absolute deviation
        is above 'gain_tol' (default is 10.)
    noise_tol : float
        Tolerance for radiometer noise filter. Flag channels whose
        noise rms is higher then 'noise_tol' times the expected
        from the radiometer equation. (default = 2.)
    fit_tol : float
        Tolerance for the fit-to-Tsky filter. Flag channels whose
        fractional rms for the 'gain' fit parameter is above
        'fit_tol' (default = 0.02)
    test_freq : integer
        Index of frequency to test. Default is 0.
    noise_synced : boolean
        Use this to force the code to call (or not call)
        ni_utils.process_synced_data(). If not given,
        the code will determine if syncronized noise injection was on.
        For acquisitions newer then 20150626T200540Z_pathfinder_corr,
        noise injection info is written in the attributes. For older
        acquisitions the function _check_ni() is called to determine
        if noise injection is On.
    inputs : list of CorrInputs, optional
        List of CorrInput objects describing the channels in this
        dataset. This is optional, if not set (default), then it will
        look the data up in the database. This option just allows
        control of the database accesses.
    res_plot : boolean, optional
        If True, a plot with all the tested channels and the
        Tsky fits is generated. File naming is
        `plot_fit_{timestamp}.pdf`
    verbose : boolean, optional
        Print out useful output as the tests are run.

    Returns
    -------
    good_gains : list of int
        1. for channels that pass the gains filter, 0. otherwise.
    good_noise : list of int
        1. for channels that pass the noise filter, 0. otherwise.
    good_fit : list of int
        1. for channels that pass the fit-to-Tsky filter,
        0. otherwise.
    test_chans : list of int
        A list of the channels tested in the same order as they
        appear in all the other lists returned

    Examples
    --------

    Run test on frequency index 3. data is an andata object:

    >>> good_gains, good_noise, good_fit, test_chans = good_channels(data,test_freq=3)

    And to create a plot of the results:

    >>> good_gains, good_noise, good_fit, test_chans = good_channels(data,test_freq=3,res_plot=True)

    """

    if verbose:
        print("Running data quality test.")

    # Determine if data has a gated visibility:
    is_gated_format = "gated_vis0" in data

    # Get number of samples during an integration period:
    if "gpu.gpu_intergration_period" in data.attrs:
        # From attributes:
        n_samp = data.attrs["gpu.gpu_intergration_period"][0]
    else:
        # Or from integration time and bandwidth:
        t_step = (
            data.index_map["time"]["ctime"][1] - data.index_map["time"]["ctime"][0]
        )  # Integration time
        bwdth = data.index_map["freq"][0][1] * 1e6  # Bandwidth comes in MHz
        n_samp = t_step * bwdth

    # Processing noise synced data, if noise_synced != False:
    if noise_synced == False:
        pass
    elif noise_synced == True:
        if is_gated_format:
            # If data is gated, ignore noise_synced argument:
            msg = (
                "Warning: noise_synced=True argument given "
                + "but data seems to be gated.\n"
                + "Ignoring noise_synced argument"
            )
            print(msg)
        else:
            # Process noise synced data:
            data = ni_utils.process_synced_data(data)
    elif noise_synced == None:
        # If noise_synced is not given, try to read ni_enable from data:
        try:
            # Newer data have a noise-injection flag
            ni_enable = data.attrs["fpga.ni_enable"][0].astype(bool)
        except:
            # If no info is found, run function to determine ni_enable:
            ni_enable = _check_ni(data, test_freq)
        # If noise injection is enabled and data is not gated:
        if ni_enable and not is_gated_format:
            # Process noise synced data:
            data = ni_utils.process_synced_data(data)

    # Read full product array in data:
    prod_array_full = data.index_map["prod"]
    # Get indices for auto-corrs:
    autos_index, autos_chan = _get_autos_index(prod_array_full)
    # Select auto-corrs and test_freq only:
    visi = np.array([data.vis[test_freq, jj, :] for jj in autos_index])
    chan_array = np.array([chan for chan in autos_chan])
    tmstp = data.index_map["time"]["ctime"]

    # Remove non-chime channels (Noise source, RFI, 26m...):
    visi, test_chans = _cut_non_chime(data, visi, chan_array, inputs)

    # Digital gains test:
    if "gain" in data:
        if verbose:
            print("Testing quality of digital gains")

        good_gains = _gains_test(data, test_freq, test_chans, tol=gain_tol)
    else:
        if verbose:
            msg = (
                "Could not obtain digital gains information from data. "
                + "Ignoring gains test."
            )
            print(msg)

        good_gains = None

    # Radiometer noise test:
    if verbose:
        print("Testing noise levels")

    good_noise, rnt = _noise_test(visi, tmstp, n_samp, tol=noise_tol)

    # Standard channels to fit for Tsky:
    if (good_gains is not None) and (good_noise is not None):
        # Use channels that pass both tests
        stand_chans = good_gains * good_noise
    elif (good_gains is None) and (good_noise is None):
        # If gains and noise tests are missing, run fit on all channels:
        stand_chans = [1.0] * len(test_chans)
    else:
        if good_gains is not None:
            # If only gains test was run
            stand_chans = good_gains
        if good_noise is not None:
            # If only noise test was run
            stand_chans = good_noise

    # Median filter visibilities for fit test:
    cut_vis = _median_filter(visi)

    # Cut sun transit from visibilities:
    if verbose:
        print("Cutting Sun transist from visibilities")
    cut_vis, cut_tmstp = _cut_sun_transit(cut_vis, tmstp)

    # Only run fit test if there are enough good channels
    # and enogh time around Sun transits:
    if np.sum(stand_chans) > 50 and len(cut_tmstp) > 100:
        # Getting template visibility (most typical visibility):
        if verbose:
            print("Getting template visibility")
        gn, Ts = _get_template(cut_vis, stand_chans)

        # Fit template to visibilities:
        if verbose:
            print("Fitting template to visibilities")
        good_fit, popt, perr, sky = _fit_template(Ts, cut_vis, tol=fit_tol)

        # Create plot with results, if res_plot is True:
        if res_plot:
            print("Generating plots")
            _create_plot(
                visi,
                tmstp,
                cut_tmstp,
                sky,
                popt,
                test_chans,
                good_gains,
                good_noise,
                good_fit,
            )
    else:
        if verbose:
            if not np.sum(stand_chans) > 50:
                print("Not enough channels for fit test.")
            if not len(cut_tmstp) > 100:
                print("Not enough time around Sun transit.")
            print("Skipping template fit test.")
        good_fit = None

    if verbose:
        # Computing some statistics to the filter:
        Nact, Nnoisy, Ngains, Nfit, Nbad = _stats_print(
            good_noise, good_gains, good_fit, test_chans
        )

        print("Finished running data quality test.")

    return good_gains, good_noise, good_fit, test_chans


def _check_ni(data, test_freq=0):
    """This is a quick and dirt function to determine if
        noise injection was ON or OFF for acquisitions
        older then ctime = 1435349183, when noise injection
        info started to be written to the h5py files

    Parameters
    ----------
    data : andata.CorrData
        Data to check for noise injection.
    test_freq : int
        frequency bin within data, to be run the test on

    Returns
    -------
    ni_on : boolean
        True if noise injection is On, False otherwise.
    """

    visi = data.vis[test_freq].real
    # Divide visibility in even and odd time bins
    if visi.shape[1] % 2 == 0:
        v_even = visi[:, 0::2]
    else:
        v_even = visi[:, 0:-1:2]  # v_even and v_odd have same length
    v_odd = visi[:, 1::2]

    # Average difference ON-OFF. Should be the same as Off-Off
    # if noise injection is Off.
    diff_on_off = np.mean(abs(v_even - v_odd))

    # Divide odd visibility again in odd and even
    if v_odd.shape[1] % 2 == 0:
        v_1 = v_odd[:, 0::2]
    else:
        v_1 = v_odd[:, 0:-1:2]
    v_2 = v_odd[:, 1::2]

    # Average difference OFF-OFF.
    diff_off_off = np.mean(abs(v_1 - v_2))

    # Ratio of differences. Sould be close to 1
    # if noise injection is off.
    ratio = diff_on_off / diff_off_off

    if ratio > 3.0:
        ni_on = True
    else:
        ni_on = False

    return ni_on


def _get_autos_index(prod_array):
    """Obtain auto-correlation indices from the 'prod' index map
    returned by andata.
    """
    autos_index, autos_chan = [], []
    for ii in range(len(prod_array)):
        if prod_array[ii][0] == prod_array[ii][1]:
            autos_index.append(ii)
            autos_chan.append(prod_array[ii][0])

    return autos_index, autos_chan


def _get_prod_array(path):
    """Function to get visibility product array from file path

    Useful when desired file is known but not the time span, so that
    finder and as_reader are not useful. Or when file is not known
    to alpenhorn

    Parameters:
    ***********
    path : string, path to file

    Returns:
    ********
    prod_array : array-like, the visibility products.
    """

    # If given list of files, use first one:
    if isinstance(path, list):
        path = path[0]

    # Get file with single time, single frequency:
    data_aux = andata.AnData.from_acq_h5(path, start=0, stop=1, freq_sel=0)

    return data_aux.index_map["prod"]


def _cut_non_chime(data, visi, chan_array, inputs=None):
    """
    Remove non CHIME channels (noise injection, RFI antenna,
    26m, etc...) from visibility. Also remove channels marked
    as powered-off in layout DB.
    """

    # Map of channels to corr. inputs:
    input_map = data.input
    tmstp = data.index_map["time"]["ctime"]  # time stamp
    # Datetime halfway through data:
    half_time = ctime.unix_to_datetime(tmstp[int(len(tmstp) // 2)])
    # Get information on correlator inputs, if not already supplied
    if inputs is None:
        inputs = tools.get_correlator_inputs(half_time)
    # Reorder inputs to have sema order as input map (and data)
    inputs = tools.reorder_correlator_inputs(input_map, inputs)
    # Get noise source channel index:

    # Test if inputs are attached to CHIME antenna and powered on:
    pwds = tools.is_chime_on(inputs)

    for ii in range(len(inputs)):
        #        if ( (not tools.is_chime(inputs[ii]))
        if (not pwds[ii]) and (ii in chan_array):
            # Remove non-CHIME-on channels from visibility matrix...
            idx = np.where(chan_array == ii)[0][0]  # index of channel
            visi = np.delete(visi, idx, axis=0)
            # ...and from product array:
            chan_array = np.delete(chan_array, idx, axis=0)

    return visi, chan_array


def _noise_test(visi, tmstp, n_samp, tol):
    """Calls radiom_noise to obtain radiometer statistics
    and aplies the noise tolerance to get a list of
    channels that pass the radiometer noise test
    """
    Nchans = visi.shape[0]
    # Array to hold radiom noise fractions
    rnt = np.full((Nchans), np.nan)

    # Cut daytime from visibility:
    visi_night, tmstp_night = _cut_daytime(visi, tmstp)

    run_noise_test = True
    if tmstp_night is None:
        # All data is in day-time
        run_noise_test = False
    elif (not isinstance(tmstp_night, list)) and (len(tmstp_night) < 20):
        # To little night-time:
        run_noise_test = False

    if not run_noise_test:
        msg = "Not enough night-time for noise test. Ignoring noise test."
        print(msg)
        good_noise = None
        rnt = None
    else:
        # Run noise test
        for ii in range(Nchans):
            # If multiple nights are present, result is a list:
            if isinstance(tmstp_night, list):
                rnt_aux = []  # rnt parameter for each night
                for jj in range(len(visi_night)):
                    rnt_array, rnt_med, rnt_max, rnt_min = _radiom_noise(
                        visi_night[jj][ii, :].real, n_samp
                    )
                    rnt_aux.append(rnt_med)
                # Use median of rnt's as parameter:
                rnt[ii] = np.median(rnt_aux)
            else:
                rnt_array, rnt_med, rnt_max, rnt_min = _radiom_noise(
                    visi_night[ii, :].real, n_samp
                )
                # Use median of rnt's as parameter:
                rnt[ii] = rnt_med

        # List of good noise channels (Initialized with all True):
        good_noise = np.ones((Nchans))
        # Test noise against tolerance and isnan, isinf:
        for ii in range(Nchans):
            is_nan_inf = np.isnan(rnt[ii]) or np.isinf(rnt[ii])
            if is_nan_inf or rnt[ii] > tol:
                good_noise[ii] = 0.0
    return good_noise, rnt


def _radiom_noise(trace, n_samp, wind=100):
    """Generates radiometer noise test statistics"""

    # If window is < the length, use length of trace:
    wind = min(len(trace), wind)

    # Window has to be even in length:
    if wind % 2 == 1:
        wind = wind - 1

    # Separate trace in windows:
    t_w = [trace[ii * wind : (ii + 1) * wind] for ii in range(int(len(trace) // wind))]

    # Estimate total Temp by median of each window:
    T = [np.median(entry) for entry in t_w]

    # Subtract even - odd bins to get rid of general trends in data:
    t_s = [
        [t_w[ii][jj] - t_w[ii][jj + 1] for jj in range(0, int(wind), 2)]
        for ii in range(len(t_w))
    ]

    # RMS of each window:
    # Use MAD to estimate RMS. More robust against RFI/correlator spikes.
    # sqrt(2) factor is due to my subtracting even - odd time bins.
    # 1.4826 factor is to go from MAD to RMS of a normal distribution:
    # rms = [ np.std(entry)/np.sqrt(2) for entry in t_s ] # Using MAD to estimate rms for now
    rms = [
        np.median([np.abs(entry[ii] - np.median(entry)) for ii in range(len(entry))])
        * 1.4826
        / np.sqrt(2)
        for entry in t_s
    ]

    # Radiometer equation proporcionality factor:
    r_fact = (0.5 * n_samp) ** 0.5
    # Radiometer noise factor (should be ~1):
    rnt = [rms[ii] * r_fact / (T[ii]) for ii in range(len(rms))]

    rnt_med = np.median(rnt)
    rnt_max = np.max(rnt)
    rnt_min = np.min(rnt)

    return rnt, rnt_med, rnt_max, rnt_min


def _cut_daytime(visi, tmstp):
    """Returns visibilities with night time only.
    Returns an array if a single night is present.
    Returns a list of arrays if multiple nights are present.
    """

    tstp = tmstp[1] - tmstp[0]  # Get time step

    risings = ch_eph.solar_rising(tmstp[0], tmstp[-1])
    settings = ch_eph.solar_setting(tmstp[0], tmstp[-1])

    if len(risings) == 0 and len(settings) == 0:
        next_rising = ch_eph.solar_rising(tmstp[-1])
        next_setting = ch_eph.solar_setting(tmstp[-1])

        if next_setting < next_rising:
            # All data is in daylight time
            cut_vis = None
            cut_tmstp = None
        else:
            # All data is in night time
            cut_vis = np.copy(visi)
            cut_tmstp = tmstp

    elif len(settings) == 0:  # Only one rising:
        sr = risings[0]
        # Find time bin index closest to solar rising:
        idx = np.argmin(np.abs(tmstp - sr))

        # Determine time limits to cut:
        # (20 min after setting and before rising, if within range)
        cut_low = max(0, idx - int(20.0 * 60.0 / tstp))  # lower limit of time cut

        # Cut daylight times:
        cut_vis = np.copy(visi[:, :cut_low])
        cut_tmstp = tmstp[:cut_low]

    elif len(risings) == 0:  # Only one setting:
        ss = settings[0]
        # Find time bin index closest to solar setting:
        idx = np.argmin(np.abs(tmstp - ss))

        # Determine time limits to cut:
        # (20 min after setting and before rising, if within range)
        cut_up = min(
            len(tmstp), idx + int(20.0 * 60.0 / tstp)
        )  # upper limit of time to cut

        # Cut daylight times:
        cut_vis = np.copy(visi[:, cut_up:])
        cut_tmstp = tmstp[cut_up:]

    else:
        cut_pairs = []
        if risings[0] > settings[0]:
            cut_pairs.append([tmstp[0], settings[0]])
            for ii in range(1, len(settings)):
                cut_pairs.append([risings[ii - 1], settings[ii]])
            if len(risings) == len(settings):
                cut_pairs.append([risings[-1], tmstp[-1]])
        else:
            for ii in range(len(settings)):
                cut_pairs.append([risings[ii], settings[ii]])
            if len(risings) > len(settings):
                cut_pairs.append([risings[-1], tmstp[-1]])

        cut_tmstp = []
        cut_vis = []
        tmstp_remain = tmstp
        vis_remain = np.copy(visi)

        for cp in cut_pairs:
            # Find time bin index closest to cuts:
            idx_low = np.argmin(np.abs(tmstp_remain - cp[0]))
            idx_up = np.argmin(np.abs(tmstp_remain - cp[1]))

            # Determine time limits to cut:
            # (20 min after setting and before rising, if within range)
            cut_low = max(
                0, idx_low - int(20.0 * 60.0 / tstp)
            )  # lower limit of time cut
            cut_up = min(
                len(tmstp_remain), idx_up + int(20.0 * 60.0 / tstp)
            )  # upper limit of time to cut

            if len(tmstp_remain[:cut_low]) > 0:
                cut_vis.append(vis_remain[:, :cut_low])
                cut_tmstp.append(
                    tmstp_remain[:cut_low]
                )  # Append times before rising to cut_tmstp
            vis_remain = vis_remain[:, cut_up:]
            tmstp_remain = tmstp_remain[
                cut_up:
            ]  # Use times after setting for further cuts
        if len(tmstp_remain) > 0:
            # If there is a bit of night data in the end, append it:
            cut_tmstp.append(tmstp_remain)
            cut_vis.append(vis_remain)

    return cut_vis, cut_tmstp


def _gains_test(data, test_freq, test_chans, tol):
    """Test channels for excessive digital gains."""

    input_map = [entry[0] for entry in data.input]

    # Get gains:
    # (only gains of channels being tested)
    gains = abs(
        np.array(
            [data.gain[test_freq, input_map.index(chan), 0] for chan in test_chans]
        )
    )

    g_med = np.median(gains)  # median
    g_devs = [abs(entry - g_med) for entry in gains]  # deviations from median
    g_mad = np.median(g_devs)  # MAD is insensitive to outlier deviations
    g_frac_devs = [dev / g_mad for dev in g_devs]  # Fractional deviations

    Nchans = len(gains)  # Number of channels

    good_gains = np.ones(Nchans)  # Good gains initialized to ones

    for ii in range(Nchans):
        if g_frac_devs[ii] > tol:  # Tolerance for gain deviations
            good_gains[ii] = 0.0

    return good_gains


def _stats_print(good_noise, good_gains, good_fit, test_chans):
    """Generate a simple set of statistics for the test
    and print them to screen.
    """
    print("\nFilter statistics:")

    good_chans = [1] * len(test_chans)

    Nact = len(test_chans)  # Number of active channels
    if good_noise is not None:
        Nnoisy = Nact - int(np.sum(good_noise))
        print(
            "Noisy channels: {0} out of {1} active channels ({2:2.1f}%)".format(
                Nnoisy, Nact, Nnoisy * 100 / Nact
            )
        )
        good_chans = good_chans * good_noise
    else:
        Nnoisy = None
    if good_gains is not None:
        Ngains = Nact - int(np.sum(good_gains))
        print(
            "High digital gains: {0} out of {1} active channels ({2:2.1f}%)".format(
                Ngains, Nact, Ngains * 100 / Nact
            )
        )
        good_chans = good_chans * good_gains
    else:
        Ngains = None
    if good_fit is not None:
        Nfit = Nact - int(np.sum(good_fit))
        print(
            "Bad fit to T_sky: {0} out of {1} active channels ({2:2.1f}%)".format(
                Nfit, Nact, Nfit * 100 / Nact
            )
        )
        good_chans = good_chans * good_fit
    else:
        Nfit = None

    # Obtain total number of bad channels:

    if not ((good_noise is None) and (good_gains is None) and (good_fit is None)):
        Nbad = Nact - int(np.sum(good_chans))
        print(
            "Overall bad: {0} out of {1} active channels ({2:2.1f}%)\n".format(
                Nbad, Nact, Nbad * 100 / Nact
            )
        )
    else:
        Nbad = None

    return Nact, Nnoisy, Ngains, Nfit, Nbad


def _cut_sun_transit(cut_vis, tmstp, tcut=120.0):
    """Cut sun transit times from visibilities.

    Parameters
    ----------
    cut_vis : numpy 2D array
        visibilities to cut (prod,time).
    tmstp : numpy 1D array
        time stamps (u-time)
    tcut : float
        time (in minutes) to cut on both sides of Sun transit.

    """

    # Start looking for transits tcut minutes before start time:
    st_time = tmstp[0] - tcut * 60.0
    # Stop looking for transits tcut minutes after end time:
    end_time = tmstp[-1] + tcut * 60.0

    # Find Sun transits between start time and end time:
    sun_trans = ch_eph.solar_transit(st_time, end_time)

    cut_tmstp = tmstp  # Time stamps to be cut
    tstp = tmstp[1] - tmstp[0]  # Get time step
    for st in sun_trans:
        # Find time bin index closest to solar transit:
        idx = np.argmin(np.abs(cut_tmstp - st))

        # Determine time limits to cut:
        # (tcut min on both sides of solar transit, if within range)
        # lower limit of time cut:
        cut_low = max(0, idx - int(tcut * 60.0 / tstp))
        # upper limit of time cut:
        cut_up = min(len(cut_tmstp), idx + int(tcut * 60.0 / tstp))

        # Cut times of solar transit:
        cut_vis = np.concatenate((cut_vis[:, :cut_low], cut_vis[:, cut_up:]), axis=1)
        cut_tmstp = np.concatenate((cut_tmstp[:cut_low], cut_tmstp[cut_up:]))

    return cut_vis, cut_tmstp


def _median_filter(visi, ks=3):
    """Median filter visibilities for fit test."""
    from scipy.signal import medfilt

    # Median filter visibilities:
    cut_vis = np.array(
        [medfilt(visi[jj, :].real, kernel_size=ks) for jj in range(visi.shape[0])]
    )
    return cut_vis


def _get_template(cut_vis_full, stand_chans):
    """Obtain template visibility through an SVD.
    This template will be compared to the actual
    visibilities in _fit_template.
    """

    # Full copy of visibilities without sun:
    cut_vis = np.copy(cut_vis_full)

    # Cut out noisy and bad-gain channels:
    cut_vis = np.array(
        [cut_vis[jj, :] for jj in range(cut_vis.shape[0]) if stand_chans[jj]]
    )

    Nchans = cut_vis.shape[0]  # Number of channels after cut

    # Perform a first cut of the most outlying visibilities:
    # Remove the offset of the visibilities (aprox T_receiver):
    vis_test = np.array(
        [cut_vis[jj, :] - np.min(cut_vis[jj, :]) for jj in range(Nchans)]
    )
    # Normalize visibilities:
    vis_test = np.array(
        [
            vis_test[jj, :] / (np.max(vis_test[jj, :]) - np.min(vis_test[jj, :]))
            for jj in range(vis_test.shape[0])
        ]
    )
    medn = np.median(vis_test, axis=0)  # Median visibility across channels
    devs = [np.sum(abs(entry - medn)) for entry in vis_test]  # Deviations sumed in time
    # Number of channels that can be ignored due
    # to excessive deviations:
    Ncut = 10
    # Find the channels with largest deviations:
    indices = list(range(len(devs)))
    del_ind = []
    for nn in range(Ncut):
        max_ind = np.argmax(devs)
        del_ind.append(indices[max_ind])
        devs = np.delete(devs, max_ind)
        indices = np.delete(indices, max_ind)
    # Cut-out channels with largest deviations:
    cut_vis = np.array(
        [cut_vis[jj, :] for jj in range(len(cut_vis)) if not (jj in del_ind)]
    )

    Nchans = cut_vis.shape[0]  # Number of channels after cut

    # Find most typical channel within each frequency:
    # Model: visi = gn*(Ts + Tr) = gn*Ts + gTr
    # Where:
    # gn : gains
    # Ts : Sky temperature
    # Tr : reciver temperature
    # gTr = gn * Tr

    # Determine first guess for receiver temperature * gain: Tr * gn = gTr
    # (lower value of low-pass filtered visibility)
    gTr = np.array([np.min(cut_vis[jj, :]) for jj in range(Nchans)])
    # Matrix Vs is the visibilities minus the guessed receiver temperature * gn
    # For the exact gTr it should be: Vs = visi - gTr = gn * Ts
    Vs = np.array([cut_vis[jj, :] - gTr[jj] for jj in range(Nchans)])

    ss = 1.0
    ss_diff = 1.0
    tries = 0
    while ss_diff > 0.01 or ss < 3.0:
        # SVD on Vs:
        U, s, V = np.linalg.svd(Vs, full_matrices=False)

        ss_diff = abs(ss - s[0] / s[1])
        ss = s[0] / s[1]
        # print 'Ratio of first to second singular value: ', ss

        # Updated gains, sky temp, sky visibility and
        # gTr approximations:
        gn = U[:, 0] * s[0]  # Only first singular value
        Ts = V[0, :]  # Only first singular value
        Vs = np.outer(gn, Ts)  # Outer product
        # New gTr is visi minus spprox sky vis, averaged over time:
        gTr = np.mean((cut_vis - Vs), axis=1)
        Vs = np.array([cut_vis[jj] - gTr[jj] for jj in range(Nchans)])

        tries += 1
        if tries == 100:
            msg = (
                "SVD search for Tsky at freq {0} did NOT converge.\n"
                + "Bad channels list might not be accurate."
            )
            print(msg)
            break

    # Solution could have negative gains and negative T sky:
    if np.sum(gn) < 0.0:  # Most gains are < 0 means sign is wrong
        # if np.all(Ts < 0) and np.all(gn < 0):
        Ts = Ts * (-1.0)
        gn = gn * (-1.0)

    return gn, Ts


def _fit_template(Ts, cut_vis, tol):
    """Fits template visibility to actual ones
    to identify bad channels.
    """

    from scipy.optimize import curve_fit

    class Template(object):
        def __init__(self, tmplt):
            self.tmplt = tmplt

        def fit_func(self, t, gn, Tr):
            return gn * self.tmplt[t] + Tr  # Template*gains + receiver temper.

    sky = Template(Ts)

    # Amplitude of template used in first guesses:
    amp_t = np.max(Ts) - np.min(Ts)
    min_t = np.min(Ts)

    popt, perr = [], []
    for chan in range(cut_vis.shape[0]):
        # First guesses:
        amp = np.max(cut_vis[chan]) - np.min(cut_vis[chan])
        gn0 = amp / amp_t  # gain as fraction of amplitudes
        Tr0 = np.min(cut_vis[chan]) - min_t * gn0
        p0 = [gn0, Tr0]  # First guesses at parameters

        # Make the fit:
        xdata = list(range(len(cut_vis[chan])))
        popt_aux, pcov = curve_fit(sky.fit_func, xdata, cut_vis[chan], p0)
        perr_aux = np.sqrt(np.diag(pcov))  # Standard deviation of parameters

        popt.append(popt_aux)
        perr.append(perr_aux)

    Nchans = cut_vis.shape[0]
    good_fit = np.ones(Nchans)
    for ii in range(Nchans):
        neg_gain = popt[ii][0] < 0.0  # Negative gains fail the test
        if neg_gain or (abs(perr[ii][0] / popt[ii][0]) > tol):
            good_fit[ii] = 0.0

    return good_fit, popt, perr, sky


def _create_plot(
    visi, tmstp, cut_tmstp, sky, popt, test_chans, good_gains, good_noise, good_fit
):
    """Creates plot of the visibilities and the fits
    with labels for those that fail the tests
    """
    import matplotlib

    matplotlib.use("PDF")
    import matplotlib.pyplot as plt
    import time

    # Visibilities to plot:
    visi1 = visi  # Raw data
    tmstp1 = tmstp  # Raw data
    visi2 = np.array(
        [
            [sky.fit_func(tt, popt[ii][0], popt[ii][1]) for tt in range(len(cut_tmstp))]
            for ii in range(len(popt))
        ]
    )
    tmstp2 = cut_tmstp

    # For title, use start time stamp:
    title = "Good channels result for {0}".format(
        ctime.unix_to_datetime(tmstp1[0]).date()
    )

    # I need to know the slot for each channel:
    def get_slot(channel):
        slot_array = [4, 2, 16, 14, 3, 1, 15, 13, 8, 6, 12, 10, 7, 5, 11, 9]
        return slot_array[int(channel) // 16]

    fig = plt.figure(figsize=(8, 64))
    fig.suptitle(title, fontsize=16)

    if (tmstp1[-1] - tmstp1[0]) / (24.0 * 3600.0) > 3.0:
        # Days since starting time
        # Notice: same starting time for both
        time_pl1 = (tmstp1 - tmstp1[0]) / (3600 * 24)
        time_pl2 = (tmstp2 - tmstp1[0]) / (3600 * 24)
        time_unit = "days"
    else:
        # Hours since starting time
        time_pl1 = (tmstp1 - tmstp1[0]) / (3600)
        time_pl2 = (tmstp2 - tmstp1[0]) / (3600)
        time_unit = "hours"

    for ii in range(len(visi1)):
        chan = test_chans[ii]

        # Determine position in subplot:
        if chan < 64:
            pos = chan * 4 + 1
        elif chan < 128:
            pos = (chan - 64) * 4 + 2
        elif chan < 192:
            pos = (chan - 128) * 4 + 3
        elif chan < 256:
            pos = (chan - 192) * 4 + 4

        # Create subplot:
        plt.subplot(64, 4, pos)

        lab = ""
        # Or print standard label:
        if good_gains is not None:
            if not good_gains[ii]:
                lab = lab + "bad gains | "
        if good_noise is not None:
            if not good_noise[ii]:
                lab = lab + "noisy | "
        if not good_fit[ii]:
            lab = lab + "bad fit"

        if lab != "":
            plt.plot([], [], "1.0", label=lab)
            plt.legend(loc="best", prop={"size": 6})

        trace_pl1 = visi1[ii, :].real
        plt.plot(time_pl1, trace_pl1, "b-")

        trace_pl2 = visi2[ii, :].real
        plt.plot(time_pl2, trace_pl2, "r-")

        tm_brd = (time_pl1[-1] - time_pl1[0]) / 10.0
        plt.xlim(time_pl1[0] - tm_brd, time_pl1[-1] + tm_brd)

        # Determine limits of plots:
        med = np.median(trace_pl1)
        mad = np.median([abs(entry - med) for entry in trace_pl1])
        plt.ylim(med - 7.0 * mad, med + 7.0 * mad)

        # labels:
        plt.ylabel("Ch{0} (Sl.{1})".format(chan, get_slot(chan)), fontsize=8)

        # Hide numbering:
        frame = plt.gca()
        frame.axes.get_yaxis().set_ticks([])
        if (chan != 63) and (chan != 127) and (chan != 191) and (chan != 255):
            # Remove x-axis, except on bottom plots:
            frame.axes.get_xaxis().set_ticks([])
        else:
            # Change size of numbers in x axis:
            frame.tick_params(axis="both", which="major", labelsize=10)
            frame.tick_params(axis="both", which="minor", labelsize=8)
            if chan == 127:
                # Put x-labels on bottom plots:
                if time_unit == "days":
                    plt.xlabel(
                        "Time (days since {0} UTC)".format(
                            ctime.unix_to_datetime(tmstp1[0])
                        ),
                        fontsize=10,
                    )
                else:
                    plt.xlabel(
                        "Time (hours since {0} UTC)".format(
                            ctime.unix_to_datetime(tmstp1[0])
                        ),
                        fontsize=10,
                    )

        if chan == 0:
            plt.title("West cyl. P1(N-S)", fontsize=12)
        elif chan == 64:
            plt.title("West cyl. P2(E-W)", fontsize=12)
        elif chan == 128:
            plt.title("East cyl. P1(N-S)", fontsize=12)
        elif chan == 192:
            plt.title("East cyl. P2(E-W)", fontsize=12)

    filename = "plot_fit_{0}.pdf".format(int(time.time()))
    plt.savefig(filename)
    plt.close()
    print("Finished creating plot. File name: {0}".format(filename))
