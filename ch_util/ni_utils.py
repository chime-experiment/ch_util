"""Tools for noise injection data"""

import numpy as np
import os
import datetime
from numpy import linalg as LA
from scipy import linalg as sciLA
import warnings
import copy
from caput import memh5
from caput import mpiarray

from . import andata


def process_synced_data(data, ni_params=None, only_off=False):
    """Turn a synced noise source observation into gated form.

    This will decimate the visibility to only the noise source off bins, and
    will add 1 or more gated on-off dataset according to the specification in
    doclib:5.

    Parameters
    ----------
    data : andata.CorrData
        Correlator data with noise source switched synchronously with the
        integration.
    ni_params : dict
        Dictionary with the noise injection parameters. Optional
        for data after ctime=1435349183. ni_params has the following keys
        - ni_period: Noise injection period in GPU integrations.
        It is assummed to be the same for all the enabled noise sources
        - ni_on_bins: A list of lists, one per enabled noise source,
        with the corresponding ON gates (within a period). For each
        noise source, the list contains the indices of the time frames
        for which the source is ON.
        Example: For 3 GPU integration period (3 gates: 0, 1, 2), two enabled
        noise sources, one ON during gate 0, the other ON during gate 1,
        and both OFF during gate 2, then
        ```
        ni_params = {'ni_period':3, 'ni_on_bins':[[0], [1]]}
        ```
    only_off : boolean
        Only return the off dataset.  Do not return gated datasets.

    Returns
    -------
    newdata : andata.CorrData
        Correlator data folded on the noise source.

    Comments
    --------
    - The function assumes that the fpga frame counter, which is used to
    determine the noise injection gating parameters, is unwrapped.
    - For noise injection data before ctime=1435349183 (i.e. for noise
    injection data before 20150626T200540Z_pathfinder_corr) the noise
    injection information is not in the headers so this function cannot be
    used to determine the noise injection parameters. A different method is
    required. Although it is recommended to check the data directly in this
    case, the previous version of this function assumed that
    ni_params = {'ni_period':2, 'ni_on_bins':[[0],]}
    for noise injection data before ctime=1435349183. Although this is not
    always true, it is true for big old datasets like pass1g.
    Use the value of ni_params recommended above to reproduce the
    results of the old function with the main old datasets.
    - Data (visibility, gain and weight datasets) are averaged for all the
    off gates within the noise source period, and also for all the on
    gates of each noise source.
    - For the time index map, only one timestamp per noise period is kept
    (no averaging)
    """

    if ni_params is None:
        # ctime before which the noise injection information is not in the
        # headers so this function cannot be used to determine the noise
        # injection parameters.
        ctime_no_noise_inj_data = 1435349183
        if data.index_map["time"]["ctime"][0] > ctime_no_noise_inj_data:
            # All the data required to figure out the noise inj gating is in
            # the data header
            try:
                ni_params = _find_ni_params(data)
            except ValueError:
                warn_str = (
                    "There are no enabled noise sources for these data. "
                    "Returning input"
                )
                warnings.warn(warn_str)
                return data
        else:
            # This is data before ctime = 1435349183. Noise injection
            # parameters are not in the data header. Raise error
            t = datetime.datetime.utcfromtimestamp(ctime_no_noise_inj_data)
            t_str = t.strftime("%Y %b %d %H:%M:%S UTC")
            err_str = (
                "ni_params parameter is required for data before "
                "%s (ctime=%i)." % (t_str, ctime_no_noise_inj_data)
            )
            raise Exception(err_str)

    if len([s for s in data.datasets.keys() if "gated_vis" in s]):
        # If there are datasets with gated_vis in their names then assume
        # this is fast gating data, where the vis dataset has on+off and
        # the vis_gatedxx has onxx-off. Process separatedly since in
        # this case the noise injection parameters are not in gpu
        # integration frames but in fpga frames and the gates are already
        # separated
        newdata = process_gated_data(data, only_off=only_off)
    else:
        # time bins with noise ON for each source (within a noise period)
        # This is a list of lists, each list corresponding to the ON time bins
        # for each noise source.
        ni_on_bins = ni_params["ni_on_bins"]

        # Number of enabled noise sources
        N_ni_sources = len(ni_on_bins)

        # Noise injection period (assume all sources have same period)
        ni_period = ni_params["ni_period"]

        # time bins with all noise sources off (within a noise period)
        ni_off_bins = np.delete(list(range(ni_period)), np.concatenate(ni_on_bins))

        # Find largest number of exact noise injection periods
        nt = ni_period * (data.ntime // ni_period)

        # Make sure we're distributed over something other than time
        data.redistribute("freq")

        # Get distribution parameters
        dist = isinstance(data.vis, memh5.MemDatasetDistributed)
        comm = data.vis.comm

        # Construct new CorrData object for gated dataset
        newdata = andata.CorrData.__new__(andata.CorrData)
        if dist:
            memh5.BasicCont.__init__(newdata, distributed=dist, comm=comm)
        else:
            memh5.BasicCont.__init__(newdata, distributed=dist)
        memh5.copyattrs(data.attrs, newdata.attrs)

        # Add index maps to newdata
        newdata.create_index_map("freq", data.index_map["freq"])
        newdata.create_index_map("prod", data.index_map["prod"])
        newdata.create_index_map("input", data.input)
        # Extract timestamps for OFF bins. Only one timestamp per noise period is
        # kept. These will be the timestamps for both the noise on ON and OFF data
        time = data.index_map["time"][ni_off_bins[0] : nt : ni_period]
        folding_period = time["ctime"][1] - time["ctime"][0]
        folding_start = time["ctime"][0]
        # Add index map for noise OFF timestamps.
        newdata.create_index_map("time", time)

        # Add datasets (for noise OFF) to newdata
        # Extract the noise source off data
        if len(ni_off_bins) > 1:
            # Average all time bins with noise OFF within a period
            vis_sky = [data.vis[..., gate:nt:ni_period] for gate in ni_off_bins]
            vis_sky = np.mean(vis_sky, axis=0)
        else:
            vis_sky = data.vis[..., ni_off_bins[0] : nt : ni_period]

        # Turn vis_sky into MPIArray if we are distributed
        if dist:
            vis_sky = mpiarray.MPIArray.wrap(vis_sky, axis=0, comm=comm)

        # Add new visibility dataset
        vis_dset = newdata.create_dataset("vis", data=vis_sky, distributed=dist)
        memh5.copyattrs(data.vis.attrs, vis_dset.attrs)

        # Add gain dataset (if exists) for noise OFF data.
        # Gain dataset also averaged (within a period)
        # These will be the gains for both the noise on ON and OFF data
        if "gain" in data:
            if len(ni_off_bins) > 1:
                gain = [data.gain[..., gate:nt:ni_period] for gate in ni_off_bins]
                gain = np.mean(gain, axis=0)
            else:
                gain = data.gain[..., ni_off_bins[0] : nt : ni_period]

            # Turn gain into MPIArray if we are distributed
            if dist:
                gain = mpiarray.MPIArray.wrap(gain, axis=0, comm=comm)

            # Add new gain dataset
            gain_dset = newdata.create_dataset("gain", data=gain, distributed=dist)
            memh5.copyattrs(data.gain.attrs, gain_dset.attrs)

        # Pull out weight dataset if it exists.
        # vis_weight dataset also averaged (within a period)
        # These will be the weights for both the noise on ON and OFF data
        if "vis_weight" in data.flags:
            if len(ni_off_bins) > 1:
                vis_weight = [
                    data.weight[..., gate:nt:ni_period] for gate in ni_off_bins
                ]
                vis_weight = np.mean(vis_weight, axis=0)
            else:
                vis_weight = data.weight[..., ni_off_bins[0] : nt : ni_period]

            # Turn vis_weight into MPIArray if we are distributed
            if dist:
                vis_weight = mpiarray.MPIArray.wrap(vis_weight, axis=0, comm=comm)

            # Add new vis_weight dataset
            vis_weight_dset = newdata.create_flag(
                "vis_weight", data=vis_weight, distributed=dist
            )
            memh5.copyattrs(data.weight.attrs, vis_weight_dset.attrs)

        # Add gated datasets for each noise source:
        if not only_off:
            for i in range(N_ni_sources):
                # Construct the noise source only data
                vis_noise = [data.vis[..., gate:nt:ni_period] for gate in ni_on_bins[i]]
                vis_noise = np.mean(vis_noise, axis=0)  # Averaging
                vis_noise -= vis_sky  # Subtracting sky contribution

                # Turn vis_noise into MPIArray if we are distributed
                if dist:
                    vis_noise = mpiarray.MPIArray.wrap(vis_noise, axis=0, comm=comm)

                # Add noise source dataset
                gate_dset = newdata.create_dataset(
                    "gated_vis{0}".format(i + 1), data=vis_noise, distributed=dist
                )
                gate_dset.attrs["axis"] = np.array(
                    ["freq", "prod", "gated_time{0}".format(i + 1)]
                )
                gate_dset.attrs["folding_period"] = folding_period
                gate_dset.attrs["folding_start"] = folding_start

                # Construct array of gate weights (sum = 0)
                gw = np.zeros(ni_period, dtype=np.float64)
                gw[ni_off_bins] = -1.0 / len(ni_off_bins)
                gw[ni_on_bins[i]] = 1.0 / len(ni_on_bins[i])
                gate_dset.attrs["gate_weight"] = gw

    return newdata


def _find_ni_params(data, verbose=0):
    """
    Finds the noise injection gating parameters.

    Parameters
    ----------
    data : andata.CorrData
        Correlator data with noise source switched synchronously with the
        integration.
    verbose: bool
        If True, print messages.

    Returns
    -------
    ni_params : dict
        Dictionary with the noise injection parameters. ni_params has the
        following keys
            ni_period: Noise injection period in GPU integrations. It is
                assummed to be the same for all the enabled noise sources
            ni_on_bins: A list of lists, one per enabled noise source,
                with the corresponding ON gates (within a period). For each
                noise source, the list contains the indices of the time frames
                for which the source is ON.

        Example: For 3 GPU integration period (3 gates: 0, 1, 2), two enabled
            noise sources, one ON during gate 0, the other ON during gate 1,
            and both OFF during gate 2, then
            ni_params = {'ni_period':3, 'ni_on_bins':[[0], [1]]}

    Comments
    --------
    - The function assumes that the fpga frame counter, which is used to
    determine the noise injection gating parameters, is unwrapped.
    - For noise injection data before ctime=1435349183 (i.e. for noise
    injection data before 20150626T200540Z_pathfinder_corr) the noise
    injection information is not in the headers so this function cannot be
    used to determine the noise injection parameters. A different method is
    required (e.g. check the data directly). The previous version of this
    function assumed that
    ni_params = {'ni_period':2, 'ni_on_bins':[[0],]}
    for noise injection data before ctime=1435349183. Although this is not
    always true, it is true for big old datasets like pass1g.
    Use the value of ni_params recommended above to reproduce the
    results of the old function with the main old datasets.
    """

    # ctime before which the noise injection information is not in the headers
    # so this function cannot be used to determine the noise injection
    # parameters.
    ctime_no_noise_inj_data = 1435349183

    # ctime of first data frame
    ctime0 = data.index_map["time"]["ctime"][0]
    if ctime0 < ctime_no_noise_inj_data:
        # This is data before ctime = 1435349183. Noise injection parameters
        # are not in the data header. Raise error
        err_str = (
            "Noise injection parameters are not in the header for "
            "these data. See help for details."
        )
        raise Exception(err_str)

    ni_period = []  # Noise source period in GPU integrations
    ni_high_time = []  # Noise source high time in GPU integrations
    ni_offset = []  # Noise source offset in GPU integrations
    ni_board = []  # Noise source PWM board

    # Noise inj information is in the headers. Assume the fpga frame
    # counter is unwrapped
    if verbose:
        print("Reading noise injection data from header")

    # Read noise injection parameters from header. Currently the system
    # Can handle up to two noise sources. Only the enabled sources are
    # analyzed
    if ("fpga.ni_enable" in data.attrs) and (data.attrs["fpga.ni_enable"][0]):
        # It seems some old data.attrs may have 'fpga.ni_enable' but not
        # 'fpga.ni_high_time' (this has to be checked!!)
        if "fpga.ni_period" in data.attrs:
            ni_period.append(data.attrs["fpga.ni_period"][0])
        else:
            ni_period.append(2)
            if verbose:
                debug_str = (
                    '"fpga.ni_period" not in data header. '
                    "Assuming noise source period = 2"
                )
                print(debug_str)

        if "fpga.ni_high_time" in data.attrs:
            ni_high_time.append(data.attrs["fpga.ni_high_time"][0])
        else:
            ni_high_time.append(1)
            if verbose:
                debug_str = (
                    '"fpga.ni_high_time" not in data header. '
                    "Assuming noise source high time = 1"
                )
                print(debug_str)

        if "fpga.ni_offset" in data.attrs:
            ni_offset.append(data.attrs["fpga.ni_offset"][0])
        else:
            ni_offset.append(0)
            if verbose:
                debug_str = (
                    '"fpga.ni_offset" not in data header. '
                    "Assuming noise source offset = 0"
                )
                print(debug_str)

        if "fpga.ni_board" in data.attrs:
            ni_board.append(data.attrs["fpga.ni_board"])
        else:
            ni_board.append("")
            if verbose:
                debug_str = '"fpga.ni_board" not in data header.'
                print(debug_str)

    if ("fpga.ni_enable_26m" in data.attrs) and (data.attrs["fpga.ni_enable_26m"][0]):
        # It seems some old data.attrs may have 'fpga.ni_enable_26m' but
        # not 'fpga.ni_high_time_26m' (this has to be checked!!)
        if "fpga.ni_period_26m" in data.attrs:
            ni_period.append(data.attrs["fpga.ni_period_26m"][0])
        else:
            ni_period.append(2)
            if verbose:
                debug_str = (
                    '"fpga.ni_period_26m" not in data header. '
                    "Assuming noise source period = 2"
                )
                print(debug_str)

        if "fpga.ni_high_time_26m" in data.attrs:
            ni_high_time.append(data.attrs["fpga.ni_high_time_26m"][0])
        else:
            ni_high_time.append(1)
            if verbose:
                debug_str = (
                    '"fpga.ni_high_time_26m" not in data header.'
                    " Assuming noise source high time = 1"
                )
                print(debug_str)

        if "fpga.ni_offset_26m" in data.attrs:
            ni_offset.append(data.attrs["fpga.ni_offset_26m"][0])
        else:
            ni_offset.append(0)
            if verbose:
                debug_str = (
                    '"fpga.ni_offset_26m" not in data header. '
                    "Assuming noise source offset = 0"
                )
                print(debug_str)

        if "fpga.ni_board_26m" in data.attrs:
            ni_board.append(data.attrs["fpga.ni_board_26m"])
        else:
            ni_board.append("")
            if verbose:
                debug_str = '"fpga.ni_board_26m" not in data header.'
                print(debug_str)

    # Number of enabled noise sources
    N_ni_sources = len(ni_period)
    if N_ni_sources == 0:
        # There are not enabled noise sources. Raise error
        raise ValueError("There are no enabled noise sources for these data")

    if np.any(np.array(ni_period - ni_period[0])):
        # Enabled sources do not have same period. Raise error
        raise Exception("Enabled sources do not have same period")

    # Period of first noise source (assume all have same period)
    ni_period = ni_period[0]

    if verbose:
        for i in range(N_ni_sources):
            print("\nPWM signal from board %s is enabled" % ni_board[i])
            print("Period: %i GPU integrations" % ni_period)
            print("High time: %i GPU integrations" % ni_high_time[i])
            print("FPGA offset: %i GPU integrations\n" % ni_offset[i])

    # Number of fpga frames within a GPU integration
    int_period = data.attrs["gpu.gpu_intergration_period"][0]

    # fpga counts for first period
    fpga_counts = data.index_map["time"]["fpga_count"][:ni_period]

    # Start of high time for each noise source (within a noise period)
    ni_on_start_bin = [
        np.argmin(np.remainder((fpga_counts // int_period - ni_offset[i]), ni_period))
        for i in range(N_ni_sources)
    ]

    # time bins with noise ON for each source (within a noise period)
    ni_on_bins = [
        np.arange(ni_on_start_bin[i], ni_on_start_bin[i] + ni_high_time[i])
        for i in range(N_ni_sources)
    ]

    ni_params = {"ni_period": ni_period, "ni_on_bins": ni_on_bins}

    return ni_params


def process_gated_data(data, only_off=False):
    """
    Processes fast gating data and turns it into gated form.

    Parameters
    ----------
    data : andata.CorrData
        Correlator data with noise source switched synchronously with the
        integration.
    only_off : boolean
        Only return the off dataset.  Do not return gated datasets.

    Returns
    -------
    newdata : andata.CorrData
        Correlator data folded on the noise source.

    Comments
    --------
    For now the correlator only supports fast gating with one gate
    (gated_vis1) and 50% duty cycle. The vis dataset contains on+off
    and the gated_vis1 contains on-off. This function returns a new
    andata object with vis containing the off data only and gated_vis1
    as in the original andata object. The attribute
    'gpu.gpu_intergration_period' is divided by 2 since during an
    integration half of the frames have on data.
    """
    # Make sure we're distributed over something other than time
    data.redistribute("freq")

    # Get distribution parameters
    dist = isinstance(data.vis, memh5.MemDatasetDistributed)
    comm = data.vis.comm

    # Construct new CorrData object for gated dataset
    newdata = andata.CorrData.__new__(andata.CorrData)
    if dist:
        memh5.BasicCont.__init__(newdata, distributed=dist, comm=comm)
    else:
        memh5.BasicCont.__init__(newdata, distributed=dist)
    memh5.copyattrs(data.attrs, newdata.attrs)

    # Add index maps to newdata
    newdata.create_index_map("freq", data.index_map["freq"])
    newdata.create_index_map("prod", data.index_map["prod"])
    newdata.create_index_map("input", data.input)
    newdata.create_index_map("time", data.index_map["time"])

    # Add datasets (for noise OFF) to newdata
    # Extract the noise source off data
    vis_off = 0.5 * (
        data.vis[:].view(np.ndarray) - data["gated_vis1"][:].view(np.ndarray)
    )

    # Turn vis_off into MPIArray if we are distributed
    if dist:
        vis_off = mpiarray.MPIArray.wrap(vis_off, axis=0, comm=comm)

    # Add new visibility dataset
    vis_dset = newdata.create_dataset("vis", data=vis_off, distributed=dist)
    memh5.copyattrs(data.vis.attrs, vis_dset.attrs)

    # Add gain dataset (if exists) for vis_off.
    # These will be the gains for both the noise on ON and OFF data
    if "gain" in data:
        gain = data.gain[:].view(np.ndarray)
        # Turn gain into MPIArray if we are distributed
        if dist:
            gain = mpiarray.MPIArray.wrap(gain, axis=0, comm=comm)

        gain_dset = newdata.create_dataset("gain", data=gain, distributed=dist)
        memh5.copyattrs(data.gain.attrs, gain_dset.attrs)

    # Pull out weight dataset if it exists.
    # These will be the weights for both the noise on ON and OFF data
    if "vis_weight" in data.flags:
        vis_weight = data.weight[:].view(np.ndarray)
        # Turn vis_weight into MPIArray if we are distributed
        if dist:
            vis_weight = mpiarray.MPIArray.wrap(vis_weight, axis=0, comm=comm)

        vis_weight_dset = newdata.create_flag(
            "vis_weight", data=vis_weight, distributed=dist
        )
        memh5.copyattrs(data.weight.attrs, vis_weight_dset.attrs)

    # Add gated dataset (only gated_vis1 currently supported by correlator
    # with 50% duty cycle)
    if not only_off:
        gated_vis1 = data["gated_vis1"][:].view(np.ndarray)
        # Turn gated_vis1 into MPIArray if we are distributed
        if dist:
            gated_vis1 = mpiarray.MPIArray.wrap(gated_vis1, axis=0, comm=comm)

        gate_dset = newdata.create_dataset(
            "gated_vis1", data=gated_vis1, distributed=dist
        )
        memh5.copyattrs(data["gated_vis1"].attrs, gate_dset.attrs)

    # The CHIME pipeline uses gpu.gpu_intergration_period to estimate the integration period
    # for both the on and off gates. That number has to be changed (divided by 2) since
    # with fast gating one integration period has 1/2 of data for the on gate and 1/2
    # for the off gate
    newdata.attrs["gpu.gpu_intergration_period"] = (
        data.attrs["gpu.gpu_intergration_period"] // 2
    )

    return newdata


class ni_data(object):
    """Provides analysis utilities for CHIME noise injection data.

    This is just a wrapper for all the utilities created in this module.

    Parameters
    -----------
    Reader_read_obj : andata.Reader.read() like object
        Contains noise injection data. Must have 'vis' and 'timestamp' property.
        Assumed to contain all the Nadc_channels*(Nadc_channels+1)/2 correlation
        products, in chime's canonical vector, for an
        Nadc_channels x Nadc_channels correlation matrix
    Nadc_channels : int
        Number of channels read in Reader_read_obj
    adc_ch_ref : int in the range 0 <= adc_ch_ref <= Nadc_channels-1
        Reference channel (used to find on/off points).
    fbin_ref : int in the range
        0 <= fbin_ref <= np.size(Reader_read_obj.vis, 0)-1
        Reference frequency bin (used to find on/off points).

    Methods
    -------
    subtract_sky_noise : Removes sky and system noise contributions from noise
        injection visibility data.
    get_ni_gains : Solve for gains from decimated sky-and-noise-subtracted
        visibilities
    get_als_gains : Compute gains, sky and system noise covariance matrices from
        a combination of noise injection gains and point source gains
    """

    def __init__(self, Reader_read_obj, Nadc_channels, adc_ch_ref=None, fbin_ref=None):
        """Processes raw noise injection data so it is ready to compute gains."""

        self.adc_channels = np.arange(Nadc_channels)
        self.Nadc_channels = Nadc_channels
        self.raw_vis = Reader_read_obj.vis
        self.Nfreqs = np.size(self.raw_vis, 0)  # Number of frequencies
        if adc_ch_ref != None:
            self.adc_ch_ref = adc_ch_ref
        else:
            self.adc_ch_ref = self.adc_channels[0]  # Default reference channel

        if fbin_ref != None:
            self.fbin_ref = fbin_ref
        else:  # Default reference frequency bin (rather arbitrary)
            self.fbin_ref = self.Nfreqs // 3

        self.timestamp = Reader_read_obj.timestamp
        try:
            self.f_MHz = Reader_read_obj.freq
        except AttributeError:
            pass  # May happen if TimeStream type does not have this property

        self.subtract_sky_noise()

    def subtract_sky_noise(self):
        """Removes sky and system noise contributions from noise injection
        visibility data.

        See also
        --------
        subtract_sky_noise function
        """

        ni_dict = subtract_sky_noise(
            self.raw_vis,
            self.Nadc_channels,
            self.timestamp,
            self.adc_ch_ref,
            self.fbin_ref,
        )
        self.time_index_on = ni_dict["time_index_on"]
        self.time_index_off = ni_dict["time_index_off"]
        self.vis_on_dec = ni_dict["vis_on_dec"]
        self.vis_off_dec = ni_dict["vis_off_dec"]
        self.vis_dec_sub = ni_dict["vis_dec_sub"]
        self.timestamp_on_dec = ni_dict["timestamp_on_dec"]
        self.timestamp_off_dec = ni_dict["timestamp_off_dec"]
        self.timestamp_dec = ni_dict["timestamp_dec"]
        self.cor_prod_ref = ni_dict["cor_prod_ref"]

    def get_ni_gains(self, normalize_vis=False, masked_channels=None):
        """Computes gains and evalues from noise injection visibility data.

        See also
        --------
        ni_gains_evalues_tf

        Additional parameters
        ---------------------
        masked_channels : list of integers
            channels which are not considered in the calculation of the gains.
        """

        self.channels = np.arange(self.Nadc_channels)
        if masked_channels != None:
            self.channels = np.delete(self.channels, masked_channels)

        self.Nchannels = len(self.channels)
        # Correlation product indices for selected channels
        cor_prod = gen_prod_sel(self.channels, total_N_channels=self.Nadc_channels)
        self.ni_gains, self.ni_evals = ni_gains_evalues_tf(
            self.vis_dec_sub[:, cor_prod, :], self.Nchannels, normalize_vis
        )

    def get_als_gains(self):
        """Compute gains, sky and system noise covariance matrices from a
        combination of noise injection gains and point source gains
        """

        pass

    def save(self):
        """Save gain solutions"""

        pass


def gen_prod_sel(channels_to_select, total_N_channels):
    """Generates correlation product indices for selected channels.

    For a correlation matrix with total_N_channels total number of channels,
    generates indices for correlation products corresponding to channels in
    the list channels_to_select.

    Parameters
    ----------
    channels_to_select : list of integers
        Indices of channels to select
    total_N_channels : int
        Total number of channels

    Returns
    -------
    prod_sel : array
        indices of correlation products for channels in channels_to_select

    """

    prod_sel = []
    k = 0
    for i in range(total_N_channels):
        for j in range(i, total_N_channels):
            if (i in channels_to_select) and (j in channels_to_select):
                prod_sel.append(k)

            k = k + 1

    return np.array(prod_sel)


def mat2utvec(A):
    """Vectorizes its upper triangle of the (hermitian) matrix A.

    Parameters
    ----------
    A : 2d array
        Hermitian matrix

    Returns
    -------
    1d array with vectorized form of upper triangle of A

    Example
    -------
    if A is a 3x3 matrix then the output vector is
    outvector = [A00, A01, A02, A11, A12, A22]

    See also
    --------
    utvec2mat
    """

    iu = np.triu_indices(np.size(A, 0))  # Indices for upper triangle of A

    return A[iu]


def utvec2mat(n, utvec):
    """
    Recovers a hermitian matrix a from its upper triangle vectorized version.

    Parameters
    ----------
    n : int
        order of the output hermitian matrix
    utvec : 1d array
        vectorized form of upper triangle of output matrix

    Returns
    -------
    A : 2d array
        hermitian matrix
    """

    iu = np.triu_indices(n)
    A = np.zeros((n, n), dtype=np.complex128)
    A[iu] = utvec  # Filling uppper triangle of A
    A = A + np.triu(A, 1).conj().T  # Filling lower triangle of A
    return A


def ktrprod(A, B):
    """Khatri-Rao or column-wise Kronecker product of two matrices.

    A and B have the same number of columns

    Parameters
    ----------
    A : 2d array
    B : 2d array

    Returns
    -------
    C : 2d array
        Khatri-Rao product of A and B
    """
    nrowsA = np.size(A, 0)
    nrowsB = np.size(B, 0)
    ncols = np.size(A, 1)
    C = np.zeros((nrowsA * nrowsB, ncols), dtype=np.complex128)
    for i in range(ncols):
        C[:, i] = np.kron(A[:, i], B[:, i])

    return C


def ni_als(R, g0, Gamma, Upsilon, maxsteps, abs_tol, rel_tol, weighted_als=True):
    """Implementation of the Alternating Least Squares algorithm for noise
    injection.

    Implements the Alternating Least Squares algorithm to recover the system
    gains, sky covariance matrix and system output noise covariance matrix
    from the data covariance matrix R. All the variables and definitions are as
    in http://bao.phas.ubc.ca/doc/library/doc_0103/rev_01/chime_calibration.pdf

    Parameters
    ----------
    R : 2d array
        Data covariance matrix
    g0 : 1d array
        First estimate of system gains
    Gamma : 2d array
        Matrix that characterizes parametrization of sky covariance matrix
    Upsilon : 2d array
        Matrix characterizing parametrization of system noise covariance matrix
    maxsteps : int
        Maximum number of iterations
    abs_tol : float
        Absolute tolerance on error function
    rel_tol : float
        Relative tolerance on error function
    weighted_als : bool
        If True, perform weighted ALS

    Returns
    -------
    g : 1d array
        System gains
    C : 2d array
        Sky covariance matrix
    N : 2d array
        System output noise covariance matrix
    err : 1d array
        Error function for every step

    See also
    --------
    http://bao.phas.ubc.ca/doc/library/doc_0103/rev_01/chime_calibration.pdf
    """

    g = g0.copy()
    G = np.diag(g)
    Nchannels = np.size(R, 0)  # Number of receiver channels
    rank_Gamma = np.size(Gamma, 1)  # Number of sky covariance matrix parameters
    # Calculate initial weight matrix
    if weighted_als:
        inv_W = sciLA.sqrtm(R)
        W = LA.inv(inv_W)
    else:
        W = np.eye(Nchannels)
        inv_W = W.copy()

    W_kron_W = np.kron(W.conj(), W)
    G_kron_G = np.kron(G.conj(), G)
    Psi = np.hstack((np.dot(G_kron_G, Gamma), Upsilon))
    psi = np.dot(np.dot(np.linalg.pinv(np.dot(W_kron_W, Psi)), W_kron_W), R)
    gamma = psi[:rank_Gamma]
    upsilon = psi[rank_Gamma:]
    # Estimate of sky covariance matrix
    C = np.dot(Gamma, gamma).reshape((Nchannels, Nchannels), order="F")
    # Estimate of output noise covariance matrix
    N = np.dot(Upsilon, upsilon).reshape((Nchannels, Nchannels), order="F")
    # Make sure C and N are positive (semi-)definite
    evals, V = LA.eigh(C, "U")  # Get eigens of C
    D = np.diag(np.maximum(evals, 0.0))  # Replace negative eigenvalues by zeros
    C = np.dot(V, np.dot(D, V.conj().T))  # Positive (semi-)definite version of C
    evals, V = LA.eigh(N, "U")
    D = np.diag(np.maximum(evals, 0))
    N = np.dot(V, np.dot(D, V.conj().T))
    # Calculate error
    err = [
        LA.norm(np.dot(W, np.dot(R - np.dot(G, np.dot(C, G.conj())) - N, W)), ord="fro")
    ]

    for i in range(1, maxsteps):
        if (err[-1] >= abs_tol) or (
            (i > 1) and (abs(err[-2] - err[-1]) <= rel_tol * err[-2])
        ):
            break

        if weighted_als:
            inv_W = sciLA.sqrtm(R + np.dot(G, np.dot(C, G.conj())) + N)
            W = LA.inv(inv_W)
        else:
            W = np.eye(Nchannels)
            inv_W = W.copy()

        W_pow2 = np.dot(W, W)
        W_pow2GC = np.dot(W_pow2, np.dot(G, C))
        g = np.dot(
            LA.pinv(np.dot(C, np.dot(G.conj().T, W_pow2GC)).conj() * W_pow2),
            np.dot(
                ktrprod(W_pow2GC, W_pow2).conj().T,
                (R - N).reshape(Nchannels**2, order="F"),
            ),
        )

        G = np.diag(g)
        G_kron_G = np.kron(G.conj(), G)
        Psi = np.hstack((np.dot(G_kron_G, Gamma), Upsilon))
        psi = np.dot(np.dot(np.linalg.pinv(np.dot(W_kron_W, Psi)), W_kron_W), R)
        gamma = psi[:rank_Gamma]
        upsilon = psi[rank_Gamma:]
        C = np.dot(Gamma, gamma).reshape((Nchannels, Nchannels), order="F")
        N = np.dot(Upsilon, upsilon).reshape((Nchannels, Nchannels), order="F")
        evals, V = LA.eigh(C, "U")
        D = np.diag(np.maximum(evals, 0.0))
        C = np.dot(V, np.dot(D, V.conj().T))
        evals, V = LA.eigh(N, "U")
        D = np.diag(np.maximum(evals, 0))
        N = np.dot(V, np.dot(D, V.conj().T))
        err.append(
            LA.norm(
                np.dot(W, np.dot(R - np.dot(G, np.dot(C, G.conj())) - N, W)), ord="fro"
            )
        )

    return g, C, N, np.array(err)


def sort_evalues_mag(evalues):
    """Sorts eigenvalue array by magnitude for all frequencies and time frames

    Parameters
    ----------
    evalues : 3d array
        Array of evalues. Its shape is [Nfreqs, Nevalues, Ntimeframes]

    Returns
    -------
    ev : 3d array
        Array of same shape as evalues
    """

    ev = np.zeros(evalues.shape, dtype=float)
    for f in range(np.size(ev, 0)):
        for t in range(np.size(ev, 2)):
            ev[f, :, t] = evalues[f, np.argsort(abs(evalues[f, :, t])), t]

    return ev


def ni_gains_evalues(C, normalize_vis=False):
    """Basic algorithm to compute gains and evalues from noise injection data.

    C is a correlation matrix from which the gains are calculated.
    If normalize_vis = True, the visibility matrix is weighted by the diagonal
    matrix that turns it into a crosscorrelation coefficient matrix before the
    gain calculation. The eigenvalues are not sorted. The returned gain solution
    vector is normalized (LA.norm(g) = 1.)

    Parameters
    ----------
    C : 2d array
        Data covariance matrix from which the gains are calculated. It is
        assumed that both the sky and system noise contributions have already
        been subtracted using noise injection
    normalize_vis : bool
        If True, the visibility matrix is weighted by the diagonal matrix that
        turns it into a crosscorrelation coefficient matrix before the
        gain calculation.

    Returns
    -------
    g : 1d array
        Noise injection gains
    ev : 1d array
        Noise injection eigenvalues

    See also
    --------
    ni_gains_evalues_tf, subtract_sky_noise
    """

    Nchannels = np.size(C, 0)  # Number of receiver channels
    if normalize_vis:  # Convert to correlation coefficient matrix
        W = np.diag(1 / np.sqrt(np.diag(C).real))
        Winv = np.diag(np.sqrt(np.diag(C).real))
    else:
        W = np.identity(Nchannels)
        Winv = np.identity(Nchannels)

    ev, V = LA.eigh(np.dot(np.dot(W, C), W), "U")
    g = np.sqrt(ev.max()) * np.dot(Winv, V[:, ev.argmax()])

    return g, ev


def ni_gains_evalues_tf(
    vis_gated, Nchannels, normalize_vis=False, vis_on=None, vis_off=None, niter=0
):
    """Computes gains and evalues from noise injection visibility data.

    Gains and eigenvalues are calculated for all frames and
    frequencies in vis_gated.  The returned gain solution
    vector is normalized (LA.norm(gains[f, :, t]) = 1.)

    Parameters
    ----------
    vis_gated : 3d array
        Visibility array in chime's canonical format. vis_gated has dimensions
        [frequency, corr. number, time]. It is assumed that both the sky and
        system noise contributions have already been subtracted using noise
        injection.
    Nchannels : int
        Order of the visibility matrix (number of channels)
    normalize_vis : bool
        If True, then the visibility matrix is weighted by the diagonal matrix that
        turns it into a crosscorrelation coefficient matrix before the
        gain calculation.
    vis_on : 3d array
        If input and normalize_vis is True, then vis_gated is weighted
        by the diagonal elements of the matrix vis_on.
        vis_on must be the same shape as vis_gated.
    vis_off : 3d array
        If input and normalize_vis is True, then vis_gated is weighted
        by the diagonal elements of the matrix: vis_on = vis_gated + vis_off.
        vis_off must be the same shape as vis_gated.  Keyword vis_on
        supersedes keyword vis_off.
    niter : 0
        Number of iterations to perform.  At each iteration, the diagonal
        elements of vis_gated are replaced with their rank 1 approximation.
        If niter == 0 (default), then no iterations are peformed and the
        autocorrelations are used instead.

    Returns
    -------
    gains : 3d array
            Noise injection gains
    evals : 3d array
            Noise injection eigenvalues

    Dependencies
    ------------
    tools.normalise_correlations, tools.eigh_no_diagonal

    See also
    --------
    ni_gains_evalues, subtract_sky_noise
    """

    from .tools import normalise_correlations
    from .tools import eigh_no_diagonal

    # Determine the number of frequencies and time frames
    Nfreqs = np.size(vis_gated, 0)
    Ntimeframes = np.size(vis_gated, 2)

    # Create NaN matrices to hold the gains and eigenvalues
    gains = np.zeros((Nfreqs, Nchannels, Ntimeframes), dtype=np.complex) * (
        np.nan + 1j * np.nan
    )
    evals = np.zeros((Nfreqs, Nchannels, Ntimeframes), dtype=np.float64) * np.nan

    # Determine if we will weight by the square root of the autos
    # of the matrix vis_on = vis_gated + vis_off
    vis_on_is_input = (vis_on is not None) and (vis_on.shape == vis_gated.shape)
    vis_off_is_input = (vis_off is not None) and (vis_off.shape == vis_gated.shape)
    weight_by_autos_on = normalize_vis and (vis_on_is_input or vis_off_is_input)

    sqrt_autos = np.ones(Nchannels)

    # Loop through the input frequencies and time frames
    for f in range(Nfreqs):
        for t in range(Ntimeframes):

            # Create Nchannel x Nchannel matrix of noise-injection-on visibilities
            if weight_by_autos_on:
                if vis_on_is_input:
                    mat_slice_vis_on = utvec2mat(Nchannels, vis_on[f, :, t])
                else:
                    mat_slice_vis_on = utvec2mat(
                        Nchannels, np.add(vis_gated[f, :, t], vis_off[f, :, t])
                    )
            else:
                mat_slice_vis_on = None

            # Create Nchannel x Nchannel matrix of gated visibilities
            mat_slice_vis_gated = utvec2mat(Nchannels, vis_gated[f, :, t])

            # If requested, then normalize the gated visibilities
            # by the square root of the autocorrelations
            if normalize_vis:
                mat_slice_vis_gated, sqrt_autos = normalise_correlations(
                    mat_slice_vis_gated, norm=mat_slice_vis_on
                )

            # Solve for eigenvalues and eigenvectors.
            # The gain solutions for the zero'th feed
            # are forced to be real and positive.
            # This means that the phases of the gain
            # solutions are relative phases with respect
            # to the zero'th feed.
            try:
                eigenvals, eigenvecs = eigh_no_diagonal(
                    mat_slice_vis_gated, niter=niter
                )

                if eigenvecs[0, eigenvals.argmax()] < 0:
                    sign0 = -1
                else:
                    sign0 = 1

                gains[f, :, t] = (
                    sign0
                    * sqrt_autos
                    * eigenvecs[:, eigenvals.argmax()]
                    * np.sqrt(np.abs(eigenvals.max()))
                )
                evals[f, :, t] = eigenvals

            except LA.LinAlgError:
                pass

    return gains, evals


def subtract_sky_noise(vis, Nchannels, timestamp, adc_ch_ref, fbin_ref):
    """Removes sky and system noise contributions from noise injection visibility
    data.

    By looking at the autocorrelation of the reference channel adc_ch_ref
    for frequency bin fbin_ref, finds timestamps indices for which the signal is
    on and off. For every noise signal period, the subcycles with the noise
    signal on and off are averaged separatedly and then subtracted.

    It is assumed that there are at least 5 noise signal cycles in the data.
    The first and last noise on subcycles are discarded since those cycles may
    be truncated.

    Parameters
    ----------
    vis: 3d array
        Noise injection visibility array in chime's canonical format. vis has
        dimensions [frequency, corr. number, time].
    Nchannels : int
        Order of the visibility matrix (number of channels)
    timestamp : 1d array
        Timestamps for the visibility array vis
    adc_ch_ref : int in the range 0 <= adc_ch_ref <= N_channels-1
        Reference channel (typically, but not necessaritly the channel
        corresponding to the directly injected noise signal) used to find
        timestamps indices for which the signal is on and off.
        on and off.
    fbin_ref : int in the range 0 <= fbin_ref <= np.size(vis, 0)-1
        frequency bin used to find timestamps indices for which the signal is
        on and off

    Returns
    -------
    A dictionary with keys
    time_index_on : 1d array
        timestamp indices for noise signal on.
    time_index_off : 1d array
        timestamp indices for noise signal off.
    timestamp_on_dec : 1d array
        timestamps for noise signal on after averaging.
    timestamp_off_dec : 1d array
        timestamps for noise signal off after averaging.
    timestamp_dec : 1d array
        timestamps for visibility data after averaging and subtracting on and
        off subcycles. These timestaps represent the time for every noise cycle
        and thus, these are the timestaps for the gain solutions.
    vis_on_dec : 3d array
        visibilities for noise signal on after averaging.
    vis_off_dec : 3d array
        visibilities for noise signal off after averaging.
    vis_dec_sub : 3d array
        visibilities data after averaging and subtracting on and
        off subcycles.
    cor_prod_ref : int
        correlation index corresponding to the autocorrelation of the reference
        channel
    """

    # Find correlation product of autocorrelation of ref channel in read data
    # Indices of autocorrelations for selected channels
    cor_prod_auto = [k * Nchannels - (k * (k - 1)) // 2 for k in range(Nchannels)]
    cor_prod_ref = cor_prod_auto[adc_ch_ref]
    auto_ref = np.real(vis[fbin_ref, cor_prod_ref, :])

    # Find timestamp indices for noise signal on and off
    # auto_ref points above/below auto_ref_mean are considered to be on/off
    auto_ref_mean = np.mean(auto_ref)
    time_index_on = np.where(auto_ref >= auto_ref_mean)[0]
    time_index_off = np.where(auto_ref < auto_ref_mean)[0]
    diff_index_on = np.diff(time_index_on)
    # Indices indicating ends of noise-on subsets
    index_end_on_cycle = time_index_on[np.where(diff_index_on > 1)[0]]
    # Indices indicating starts of noise-on subsets
    index_start_on_cycle = time_index_on[np.where(diff_index_on > 1)[0] + 1]
    vis_on_dec = []  # Decimated visibility on points
    vis_off_dec = []
    timestamp_on_dec = []  # Timestamps of visibility on points
    timestamp_off_dec = []
    timestamp_dec = []  # Timestamp of decimated visibility (on minus off)

    for i in range(len(index_end_on_cycle) - 1):
        # Visibilities with noise on for cycle i
        vis_on_cycle_i = vis[
            :, :, index_start_on_cycle[i] : index_end_on_cycle[i + 1] + 1
        ]
        # Visibilities with noise off for cycle i
        vis_off_cycle_i = vis[:, :, index_end_on_cycle[i] + 1 : index_start_on_cycle[i]]

        # New lines to find indices of maximum and minimum point of each cycle based on the reference channel
        index_max_i = auto_ref[
            index_start_on_cycle[i] : index_end_on_cycle[i + 1] + 1
        ].argmax()
        index_min_i = auto_ref[
            index_end_on_cycle[i] + 1 : index_start_on_cycle[i]
        ].argmin()
        vis_on_dec.append(vis_on_cycle_i[:, :, index_max_i])
        vis_off_dec.append(vis_off_cycle_i[:, :, index_min_i])

        # Instead of averaging all the data with noise on of a cycle, we take the median
        # vis_on_dec.append(np.median(vis_on_cycle_i.real, axis=2)+1j*np.median(vis_on_cycle_i.imag, axis=2))
        # vis_off_dec.append(np.median(vis_off_cycle_i.real, axis=2)+1j*np.median(vis_off_cycle_i.imag, axis=2))
        timestamp_on_dec.append(
            np.mean(timestamp[index_start_on_cycle[i] : index_end_on_cycle[i + 1] + 1])
        )
        timestamp_off_dec.append(
            np.mean(timestamp[index_end_on_cycle[i] + 1 : index_start_on_cycle[i]])
        )
        timestamp_dec.append(
            np.mean(
                timestamp[index_end_on_cycle[i] + 1 : index_end_on_cycle[i + 1] + 1]
            )
        )

    vis_on_dec = np.dstack(vis_on_dec)
    vis_off_dec = np.dstack(vis_off_dec)
    vis_dec_sub = vis_on_dec - vis_off_dec
    timestamp_on_dec = np.array(timestamp_on_dec)
    timestamp_off_dec = np.array(timestamp_off_dec)
    timestamp_dec = np.array(timestamp_dec)

    return {
        "time_index_on": time_index_on,
        "time_index_off": time_index_off,
        "vis_on_dec": vis_on_dec,
        "vis_off_dec": vis_off_dec,
        "vis_dec_sub": vis_dec_sub,
        "timestamp_on_dec": timestamp_on_dec,
        "timestamp_off_dec": timestamp_off_dec,
        "timestamp_dec": timestamp_dec,
        "cor_prod_ref": cor_prod_ref,
    }


def gains2utvec_tf(gains):
    """Converts gain array to CHIME visibility format for all frequencies and
    time frames.

    For every frequency and time frame, converts a gain vector into an outer
    product matrix and then vectorizes its upper triangle to obtain a vector in
    the same format as the CHIME visibility matrix.

    Converting the gain arrays to CHIME visibility format makes easier to
    apply the gain corrections to the visibility data. See example below.

    Parameters
    ----------
    gains : 3d array
        Input array with the gains for all frequencies, channels and time frames
        in the fromat of ni_gains_evalues_tf. g has dimensions
        [frequency, channels, time].

    Returns
    -------
    G_ut : 3d array
        Output array with dimmensions [frequency, corr. number, time]. For
        every frequency and time frame, contains the vectorized form of upper
        triangle for the outer product of the respective gain vector.

    Example
    -------
    To compute the gains from a set of noise injection pass0 data and apply the
    gains to the visibilities run:

    >>> from ch_util import andata
    >>> from ch_util import import ni_utils as ni
    >>> data = andata.Reader('/scratch/k/krs/jrs65/chime_archive/20140916T173334Z_blanchard_corr/000[0-3]*.h5')
    >>> readdata = data.read()
    >>> nidata = ni.ni_data(readdata, 16)
    >>> nidata.get_ni_gains()
    >>> G_ut = ni.gains2utvec(nidata.ni_gains)
    >>> corrected_vis = nidata.vis_off_dec/G_ut

    See also
    --------
    gains2utvec, ni_gains_evalues_tf
    """

    Nfreqs = np.size(gains, 0)  # Number of frequencies
    Ntimeframes = np.size(gains, 2)  # Number of time frames
    Nchannels = np.size(gains, 1)
    Ncorrprods = Nchannels * (Nchannels + 1) // 2  # Number of correlation products
    G_ut = np.zeros((Nfreqs, Ncorrprods, Ntimeframes), dtype=np.complex)

    for f in range(Nfreqs):
        for t in range(Ntimeframes):
            G_ut[f, :, t] = gains2utvec(gains[f, :, t])

    return G_ut


def gains2utvec(g):
    """Converts a vector into an outer product matrix and vectorizes its upper
    triangle to obtain a vector in same format as the CHIME visibility matrix.

    Parameters
    ----------
    g : 1d array
        gain vector

    Returns
    -------
    1d array with vectorized form of upper triangle for the outer product of g
    """

    n = len(g)
    G = np.dot(g.reshape(n, 1), g.conj().reshape(1, n))
    return mat2utvec(G)
