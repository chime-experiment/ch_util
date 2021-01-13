"""Plotting routines for CHIME data"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import warnings
import datetime
import scipy.signal as sig
from . import andata
from . import ephemeris


def waterfall(
    data, freq_sel=None, prod_sel=None, time_sel=None, part_sel=None, **kwargs
):
    """Two dimensional plot of a visibility vs time and frequency.

    Parameters
    ----------
    data : numpy array or :class:`~ch_util.andata.AnData` object
        Data to plot. If a numpy array, must be 2D or 3D.
    freq_sel : valid numpy index
        Selects data to include along the frequency axis.
    prod_sel : valid numpy index
        Selects data to include along the correlation product axis.  If *data*
        is a 2D array, this argument is ignored.
    time_sel : valid numpy index
        Selects data to include along the time axis.
    part_sel : string, one of: 'real', 'imag', 'mag', 'phase' or 'complex'
        Selects what part of data to plot. If 'None', plot real part.

    Examples
    --------

    >>> data = np.ones((100, 100))
    >>> waterfall(data)

    >>> data = andata.AnData.from_acq("...")
    >>> waterfall(data, prod_sel=5, out_file='filename.png')

    To make a plot normalized by a baseline of the median-filtered
    power spectrum averaged over 200 time bins starting at bin 0 with
    a median filter window of 40 bins:
    >>> data = andata.AnData.from_acq("...")
    >>> med_filt_arg = ['new',200,0,40]
    >>> waterfall(data, prod_sel=21, med_filt=med_filt_arg)

    You can also make it save the calculated baseline to a file,
    by providing the filename:
    >>> data = andata.AnData.from_acq("...")
    >>> med_filt_arg = ['new',200,0,40,'base_filename.dat']
    >>> waterfall(data, prod_sel=21, med_filt=med_filt_arg)

    ...or to use a previously obtained baseline to normalize data:
    (where bsln is either a numpy array or a list with length equal
    to the frequency axis of the data)
    >>> data = andata.AnData.from_acq("...")
    >>> med_filt_arg = ['old',bsln]
    >>> waterfall(data, prod_sel=21, med_filt=med_filt_arg)

    To make a full day plot of 01/14/2014,
    rebinned to 4000 time bins:
    >>> data = andata.AnData.from_acq("...")
    >>> full_day_arg = [[2014,01,14],4000,'time']
    >>> waterfall(data, prod_sel=21, full_day=full_day_arg)

    """
    ########## Section for retrieving keyword arguments.##############
    # Please remove from the kwargs dictionary any arguments for
    # which you provided functionality in waterfall(). The resulting
    # dictionary is going to be passed on to imshow()

    aspect = kwargs.pop("aspect", None)  # float. Aspect ratio of image
    show_plot = kwargs.pop("show_plot", None)  # True or False. Interactive plot
    out_file = kwargs.pop("out_file", None)  # str. File name to save to
    res = kwargs.pop("res", None)  # int. Resolution of saved image in dpi
    title = kwargs.pop("title", None)  # str. Graph title.
    x_label = kwargs.pop("x_label", None)  # str.
    y_label = kwargs.pop("y_label", None)  # str.
    med_filt = kwargs.pop("med_filt", None)  # List of length 2 or 4. See examples.
    full_day = kwargs.pop("full_day", None)  # List of length 3. See examples.
    cbar_label = kwargs.pop("cbar_label", None)  # str. Colorbar label.

    ##################################################################

    # waterfall() does not accept 'complex'
    if part_sel == "complex":
        msg = 'waterfall() does not take "complex" for "part_sel"' " argument."
        raise ValueError(msg)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Set title, if given:
    if title != None:
        ax.set_title(title)

    # Setting labels, if given:
    if x_label != None:
        ax.set_xlabel(x_label)
    if y_label != None:
        ax.set_ylabel(y_label)

    # Preparing data shape for plotting:
    plt_data = _coerce_data_shape(
        data, freq_sel, prod_sel, time_sel, part_sel, axes=(1,)
    )
    if isinstance(data, andata.AnData):
        tmstp = _select_time(data, time_sel)

    # Apply median filter, if 'med_filt' is given:
    if med_filt != None:
        msg = "Warning: Wrong value for 'med_filt'. Ignoring argument."
        if med_filt[0] == "new":
            # Apply median filter:
            plt_data, baseline = _med_filter(
                plt_data, med_filt[1], med_filt[2], med_filt[3]
            )
            if len(med_filt) == 5:
                # Save baseline to file, if given:
                fileBaseOut = open(med_filt[4], "w")
                for ii in range(len(baseline)):
                    fileBaseOut.write("{0}\n".format(baseline[ii, 0]))
                fileBaseOut.close()
        elif med_filt[0] == "old":
            # Reshape baseline to ensure type and shape:
            baseline = np.array(med_filt[1]).reshape(len(med_filt[1]), 1)
            # Normalize data:
            plt_data = plt_data / baseline
        else:
            print(msg)

    # Shape data to full day, if 'full_day' is given:
    if full_day != None:
        plt_data = _full_day_shape(
            plt_data,
            tmstp,
            date=full_day[0],
            n_bins=full_day[1],
            axis=full_day[2],
            ax=ax,
        )

    # Call imshow reversing frequency order
    wtfl = ax.imshow(plt_data[::-1, :], **kwargs)

    # Ajust aspect ratio of image if aspect is provided:
    if aspect != None:
        _force_aspect(ax, aspect)

        # Ajust colorbar size:
        if aspect >= 1.0:
            shrink = 1 / float(aspect)
        else:
            shrink = 1.0
        cbar = fig.colorbar(wtfl, shrink=shrink)
    else:
        cbar = fig.colorbar(wtfl)

    # Set label to colorbar, if given:
    if cbar_label != None:
        cbar.set_label(cbar_label)

    # Output depends on keyword arguments:
    if show_plot == True:
        plt.show()
    elif (show_plot != None) and (show_plot != False):
        msg = (
            'Optional keyword argument "show_plot" should receive either'
            ' "True" or "False". Received "{0}". Ignoring argument.'.format(show_plot)
        )
        warnings.warn(msg, SyntaxWarning)

    # Save to file if filename is provided:
    if out_file != None:
        if res != None:
            fig.savefig(out_file, dpi=res)
        else:
            fig.savefig(out_file)

    plt.close(fig)


def spectra(data, freq_sel=None, prod_sel=None, time_sel=None, part_sel=None, **kwargs):
    """Plots spectra at different times and for different correlation products."""

    plt_data = _coerce_data_shape(data, freq_sel, prod_sel, time_sel, axes=())
    ntime = plt_data.shape[2]
    nprod = plt_data.shape[1]
    for ii in range(ntime):
        for jj in range(nprod):
            plt.plot(plt_data[:, ii, jj])


def time_ordered(
    data, freq_sel=None, prod_sel=None, time_sel=None, part_sel=None, **kwargs
):
    """Plots data vs time for different frequencies and corr-pords."""

    pass


def _coerce_data_shape(
    data, freq_sel=None, prod_sel=None, time_sel=None, part_sel=None, axes=()
):
    """Gets well shaped data array for plotting.

    Parameters
    ----------
    data : numpy array or :class:`~ch_util.andata.AnData` object
        Data to coerse.
    freq_sel : valid numpy index
        Selects data to include along the frequency axis.  Default slices the
        full axis.
    prod_sel : valid numpy index
        Selects data to include along the correlation product axis.  If *data*
        is a 2D array, this argument is ignored. Default slices the
        full axis.
    time_sel : valid numpy index
        Selects data to include along the time axis. Default slices the
        full axis.
    part_sel : string, one of: 'real', 'imag', 'mag', 'phase' or 'complex'
        Selects what part of data to plot. If 'None', plot real part.
    axes : tuple or axis numbers
        Axes to eliminate

    Returns
    -------
    plt_data : numpy array
        The dimentionality of the array is guaranteed to be
        ``plt_data == 3 - len(axes)``.

    Raises
    ------
    ValueError
        If data provided could not be coersed.

    Examples
    --------

    Lets start with simple sliceing of numpy array data.

    >>> data = np.ones((5, 7, 3))
    >>> _coerse_data_shape(data, [2, 3, 4], 3, None).shape
    (3, 1, 3)

    Notice that the out put is 3D even though normal numpy indexing would have
    eliminated the correation-product axis.  This is because the *axes*
    parameter is set to it's default meaning 3D output is required.  If we
    instead tell it to eliminate the product axis:

    >>> _coerse_data_shape(data, [2, 3, 4], 3, None, axes=(1,)).shape
    (3, 3)

    If an axis to be eliminated is not length 1, a :exc:`ValueError` is raised.

    >>>  _coerse_data_shape(data, [2, 3, 4], [3, 2], None, axes=(1,))
    Traceback (most recent call last)
        ...
    ValueError: Need to eliminate axis 1 but it is not length 1.

    The input data may be less than 3D. In this case *axes* indicates which axes
    are missing.

    >>> data = np.ones((2, 3))
    >>> _coerse_data_shape(data, 1, None, None, axes=(1,)).shape
    (1, 3)

    Example of selecting part to plot:

    >>> data = np.ones((2,3,4))*(5+6j)
    >>> _coerce_data_shape(data,None,1,None,part_sel='imag',axes=(1,))
    array([[ 6.,  6.,  6.,  6.],
           [ 6.,  6.,  6.,  6.]])

    All this works with :class:`~chutil.andata.AnData` input data, where the
    visibilities are treated as a 3D array.

    """

    axes = sorted(axes)
    if isinstance(data, andata.AnData):
        data = data.vis
    if data.ndim != 3:
        if data.ndim != 3 - len(axes):
            msg = (
                "Could no interpret input data axes. Got %dD data and need"
                " coerse to %dD data"
            ) % (data.ndim, 3 - len(axes))
            raise ValueError(msg)
        # Temporarily make the data 3D (for slicing), will reshape in the end.
        shape = data.shape
        for axis in axes:
            shape = shape[:axis] + (1,) + shape[axis:]
        data = np.reshape(data, shape)
    # Select data.
    if isinstance(freq_sel, int):
        freq_sel = [freq_sel]
    elif freq_sel is None:
        freq_sel = slice(None)
    data = data[freq_sel]
    if isinstance(prod_sel, int):
        prod_sel = [prod_sel]
    elif prod_sel is None:
        prod_sel = slice(None)
    data = data[:, prod_sel]
    if isinstance(time_sel, int):
        time_sel = [time_sel]
    elif time_sel is None:
        time_sel = slice(None)
    data = data[:, :, time_sel]
    if data.ndim != 3:
        raise RuntimeError("Shouldn't have happend")
    # Now reshape to the correct dimensionality.
    shape = data.shape
    axes.reverse()
    for axis in axes:
        if not shape[axis] == 1:
            msg = "Need to eliminate axis %d but it is not length 1." % axis
            raise ValueError(msg)
        shape = shape[:axis] + shape[axis + 1 :]
    data.shape = shape

    # Selects what part to plot:
    # Defaults to plotting real part of data.
    if part_sel == "real" or part_sel == None:
        data = data.real
    elif part_sel == "imag":
        data = data.imag
    elif part_sel == "mag":
        data = (data.real ** 2 + data.imag ** 2) ** (0.5)
    elif part_sel == "phase":
        data = np.arctan(data.imag / data.real)
    elif part_sel == "complex":
        pass
    else:
        msg = (
            'Optional keyword argument "part_sel" has to receive'
            ' one of "real", "imag", "mag", "phase" or "complex".'
            ' Received "{0}"'.format(part_sel)
        )
        raise ValueError(msg)

    return data


def _select_time(data, time_sel):
    """Reshape time stamp vector acording to 'time_sel'

    Parameters
    ----------
    data : class:`~ch_util.andata.AnData` object
        Data to take time stamp from.
    time_sel : valid numpy index
        Selects data to include along the time axis. Default slices the
        full axis.

    Returns
    -------
    tmstp : numpy array
        time stamp with selected times

    """
    if isinstance(data, andata.AnData):
        tmstp = data.timestamp
        if isinstance(time_sel, int):
            time_sel = [time_sel]
        elif time_sel is None:
            time_sel = slice(None)
        tmstp = tmstp[time_sel]

    return tmstp


def _full_day_shape(data, tmstp, date, n_bins=8640, axis="solar", ax=None):
    """Rebin data in linear time or solar azimuth.

    Parameters
    ----------
    data : numpy array
        Data to plot. Must be 2D.
    tmstp : numpy array
        Time stamp of data to plot.
    ax : matplotlib.axes.Axes instance
        Axes to receive plot. Time/azimuth ticks
        and labels will be set acordingly
    date : python list of length 3
        Date of day to plot in the format:
        [yyyy,mm,dd], al entries 'int'.
    n_bins : int
        Number of time/azimuth bins in new matrix.
    axis : str
        If 'solar': rebin by solar azimuth
        If 'time': rebin by time

    Returns
    -------
    Z : numpy ndarray
        New rebinned matrix

    Example
    -------
    For example of usage, see plot.waterfall() documentation

    """
    n_bins = int(n_bins)
    start_time = datetime.datetime(date[0], date[1], date[2], 8, 0, 0)  # UTC-8
    end_time = start_time + datetime.timedelta(days=1)
    unix_start = ephemeris.datetime_to_unix(start_time)
    unix_end = ephemeris.datetime_to_unix(end_time)
    print("Re-binning full day data to plot")

    if axis == "solar":
        bin_width = float(2 * np.pi) / float(n_bins)
        bin_ranges = []
        for ii in range(n_bins):
            az1 = ii * bin_width
            az2 = az1 + bin_width
            bin_ranges.append([az1, az2])

        values_to_sum = []
        for ii in range(n_bins):
            values_to_sum.append([])

        start_range = [unix_start - 1.5 * 3600, unix_start + 0.5 * 3600]
        end_range = [unix_end - 1.5 * 3600, unix_end + 0.5 * 3600]

        n_added = 0

        for ii in range(len(tmstp)):
            in_range = (tmstp[ii] > start_range[0]) and (tmstp[ii] < end_range[1])
            if in_range:
                sf_time = ephemeris.unix_to_skyfield_time(tmstp[ii])
                sun = ephemeris.skyfield_wrapper.ephemeris["sun"]
                obs = ephemeris.chime.skyfield_obs().at(sf_time)
                azim = obs.observe(sun).apparent().altaz()[1].radians

                in_start_range = (tmstp[ii] > start_range[0]) and (
                    tmstp[ii] < start_range[1]
                )
                in_end_range = (tmstp[ii] > end_range[0]) and (tmstp[ii] < end_range[1])

                if in_start_range:
                    for jj in range(int(n_bins // 2)):
                        if (azim > bin_ranges[jj][0]) and (azim <= bin_ranges[jj][1]):
                            values_to_sum[jj].append(ii)
                            n_added = n_added + 1
                            break
                elif in_end_range:
                    for jj in range(int(n_bins // 2)):
                        kk = n_bins - jj - 1
                        if (azim > bin_ranges[kk][0]) and (azim <= bin_ranges[kk][1]):
                            values_to_sum[kk].append(ii)
                            n_added = n_added + 1
                            break
                else:
                    for jj in range(n_bins):
                        if (azim > bin_ranges[jj][0]) and (azim <= bin_ranges[jj][1]):
                            values_to_sum[jj].append(ii)
                            n_added = n_added + 1
                            break

        # Set azimuth ticks, if given:
        if ax != None:
            tck_stp = n_bins / 6.0
            ticks = np.array(
                [
                    int(tck_stp),
                    int(2 * tck_stp),
                    int(3 * tck_stp),
                    int(4 * tck_stp),
                    int(5 * tck_stp),
                ]
            )
            ax.set_xticks(ticks)
            labels = ["60", "120", "180", "240", "300"]
            ax.set_xticklabels(labels)
            # Set label:
            ax.set_xlabel("Solar azimuth (degrees)")

    elif axis == "time":
        bin_width = float(86400) / float(n_bins)
        bin_ranges = []
        for ii in range(n_bins):
            t1 = unix_start + ii * bin_width
            t2 = t1 + bin_width
            bin_ranges.append([t1, t2])

        values_to_sum = []
        for ii in range(n_bins):
            values_to_sum.append([])

        n_added = 0

        for ii in range(len(tmstp)):
            in_range = (tmstp[ii] >= unix_start) and (tmstp[ii] <= unix_end)
            if in_range:
                time = tmstp[ii]
                for jj in range(n_bins):
                    if (time > bin_ranges[jj][0]) and (time <= bin_ranges[jj][1]):
                        values_to_sum[jj].append(ii)
                        n_added = n_added + 1
                        break

        # Set time ticks, if given:
        if ax != None:
            tck_stp = n_bins / 6.0
            ticks = np.array(
                [
                    int(tck_stp),
                    int(2 * tck_stp),
                    int(3 * tck_stp),
                    int(4 * tck_stp),
                    int(5 * tck_stp),
                ]
            )
            ax.set_xticks(ticks)
            labels = ["04:00", "08:00", "12:00", "16:00", "20:00"]
            ax.set_xticklabels(labels)
            # Set label:
            ax.set_xlabel("Time (UTC-8 hours)")

    print("Number of 10-second bins added to full day data: {0}".format(n_added))

    # Set new array to NaN for subsequent masking:
    Z = np.ones((1024, n_bins))
    for ii in range(1024):
        for jj in range(n_bins):
            Z[ii, jj] = float("NaN")

    for ii in range(n_bins):
        n_col = len(values_to_sum[ii])
        if n_col > 0:
            col = np.zeros((1024))
            for jj in range(n_col):
                col = col + data[:, values_to_sum[ii][jj]]
            Z[:, ii] = col / float(n_col)

    return Z


def _force_aspect(ax, aspect=1.0):
    """Force desired aspect ratio into image axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes instance
        Axes that will be set to the desired aspect ratio
    aspect : float or int
        Desired aspect ratio in horizontal/vertical order

    Motivation
    ----------
    Apparently, the 'aspect' keyword argument in Imshow() is
    not working properlly in this version of matplotlib (1.1.1rc)

    Example
    -------

    data = np.ones((100,200))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(data)
    _force_aspect(ax,aspect=1.)
    plt.show()

    Will produce a square solid image.

    """

    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / float(extent[3] - extent[2])) / aspect)


def _med_filter(data, n_bins=200, i_bin=0, filt_window=37):
    """Normalize a 2D array by its power spectrum averaged over 'n_bins' starting at 'i_bin'.

    Parameters
    ----------
    data : numpy.ndarray
    Data to be normalized

    n_bins : integer
    Number of bins over which to average the power spectrum

    i_bin : integer
    First bin of the range over which to average the power spectrum

    filt_window : integer
    Width of the window for the median filter. The filter is applied
    once with this window and a second time with 1/3 of this window width.

    Returns
    -------
    rel_power : 2d array normalized by average power spectrum (baseline)
    medfilt_baseline : Average power spectrum

    Issues
    ------
    Assumes frequency in first index and time in second index
    Assumes data has the standard 1024 frequency bins

    Comments
    --------
    If entry is 0 in data and in baseline, entry is set to 1. in
    normalized matrix

    """
    # If n_bins biger than array, average over entire array:
    if data.shape[1] > n_bins:
        sliced2darray = data[:, 0 : (n_bins - 1)]
    else:
        sliced2darray = data

    # Mean of range selected:
    mean_arr = np.mean(sliced2darray, axis=-1)
    # Standard deviation:
    std_arr = np.std(sliced2darray, axis=-1)
    # Standard deviation of the mean:
    sigma = np.median(std_arr) / (sliced2darray.shape[1]) ** (0.5)
    print("Taking median filter")
    medfilt_arr = sig.medfilt(mean_arr, filt_window)
    # Extract RFI:
    non_rfi_mask = (mean_arr - medfilt_arr) < 5 * sigma
    print("Number of good data points for baseline: ", np.sum(non_rfi_mask))
    print("out of 1024 points - ", np.sum(non_rfi_mask / float(1024)) * 100, "%")
    # Interpolate result:
    freq = np.linspace(400, 800, 1024)
    interpolat_arr_baseline = np.interp(
        freq, freq[non_rfi_mask], mean_arr[non_rfi_mask]
    )
    # Median filter a second time:
    small_window = int(filt_window // 3)
    # Has to be odd:
    if small_window % 2 == 0:
        small_window = small_window + 1
    medfilt_baseline = np.reshape(
        sig.medfilt(interpolat_arr_baseline, small_window),
        (interpolat_arr_baseline.shape[0], 1),
    )

    # Boolean mask for entries where original data and baseline are zero:
    mask = np.where(medfilt_baseline == 0, data, 1) == 0
    # Normalize data:
    rel_power = data / medfilt_baseline
    # Set masked entries to 1:
    rel_power[mask] = 1.0

    return rel_power, medfilt_baseline
