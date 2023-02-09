"""Analysis data format"""

import warnings
import glob
from os import path
import posixpath
import re

import numpy as np
import h5py
from bitshuffle import h5

tmp = h5  # To appease linters who complain about unused imports.

# If the `caput` package is available, get `memh5` from there.  Otherwise, use
# the version of memh5 that ships with `ch_util`, eliminating the dependency.
try:
    from caput import memh5, tod
except ImportError:
    raise ImportError("Could not import memh5 or tod. Have you installed caput?")


ni_msg = "Ask Kiyo to implement this."


# Datasets in the Acq files whose shape is the same as the visibilities.
# Variable only used for legacy archive version 1.
ACQ_VIS_SHAPE_DATASETS = ("vis", "vis_flag", "vis_weight")

# Datasets in the Acq files that are visibilities or gated visibilities
ACQ_VIS_DATASETS = "^vis$|^gated_vis[0-9]$"

# Datasets in the HK files that are data.
HK_DATASET_NAMES = ("data", "^mux[0-9]{2}$")

# List of axes over which we can concatenate datasets.  To be concatenated, all
# datasets must have one and only one of these in their 'axes' attribute.
CONCATENATION_AXES = (
    "time",
    "gated_time0",
    "gated_time1",
    "gated_time2",
    "gated_time3",
    "gated_time4",
    "snapshot",
    "update_time",
    "station_time_blockhouse",
)

ANDATA_VERSION = "3.1.0"


# Main Class Definition
# ---------------------


class BaseData(tod.TOData):
    """CHIME data in analysis format.

    Inherits from :class:`caput.memh5.BasicCont`.

    This is intended to be the main data class for the post
    acquisition/real-time analysis parts of the pipeline. This class is laid
    out very similarly to how the data is stored in analysis format hdf5 files
    and the data in this class can be optionally stored in such an hdf5 file
    instead of in memory.

    Parameters
    ----------
    h5_data : h5py.Group, memh5.MemGroup or hdf5 filename, optional
        Underlying h5py like data container where data will be stored. If not
        provided a new :class:`caput.memh5.MemGroup` instance will be created.
    """

    time_axes = CONCATENATION_AXES
    distributed_axis = None

    # Datasets that we should convert into distribute ones
    _DIST_DSETS = [
        "vis",
        "vis_flag",
        "vis_weight",
        "gain",
        "gain_coeff",
        "frac_lost",
        "eval",
        "evec",
        "erms",
    ]

    # Convert strings to/from unicode on load and save
    convert_attribute_strings = True
    convert_dataset_strings = True

    def __new__(cls, h5_data=None, **kwargs):
        """Used to pick which subclass to instantiate based on attributes in
        data."""

        new_cls = subclass_from_obj(cls, h5_data)

        self = super(BaseData, new_cls).__new__(new_cls)
        return self

    def __init__(self, h5_data=None, **kwargs):
        super(BaseData, self).__init__(h5_data, **kwargs)
        if self._data.file.mode == "r+":
            self._data.require_group("cal")
            self._data.require_group("flags")
            self._data.require_group("reverse_map")
            self.attrs["andata_version"] = ANDATA_VERSION

    # - The main interface - #

    @property
    def datasets(self):
        """Stores hdf5 datasets holding all data.

        Each dataset can reference a calibration scheme in
        ``datasets[name].attrs['cal']`` which refers to an entry in
        :attr:`~BaseData.cal`.

        Do not try to add a new dataset by assigning to an item of this
        property. Use `create_dataset` instead.

        Returns
        -------
        datasets : read only dictionary
            Entries are :mod:`h5py` or :mod:`caput.memh5` datasets.

        """

        out = {}
        for name, value in self._data.items():
            if not memh5.is_group(value):
                out[name] = value
        return memh5.ro_dict(out)

    @property
    def flags(self):
        """Datasets representing flags and data weights.

        Returns
        -------
        flags : read only dictionary
            Entries are :mod:`h5py` or :mod:`caput.memh5` datasets.

        """

        try:
            g = self._data["flags"]
        except KeyError:
            return memh5.ro_dict({})

        out = {}
        for name, value in g.items():
            if not memh5.is_group(value):
                out[name] = value
        return memh5.ro_dict(out)

    @property
    def cal(self):
        """Stores calibration schemes for the datasets.

        Each entry is a calibration scheme which itself is a dict storing
        meta-data about calibration.

        Do not try to add a new entry by assigning to an element of this
        property. Use :meth:`~BaseData.create_cal` instead.

        Returns
        -------
        cal : read only dictionary
            Calibration schemes.

        """

        out = {}
        for name, value in self._data["cal"].items():
            out[name] = value.attrs
        return memh5.ro_dict(out)

    @property
    def reverse_map(self):
        """Stores the inverse mapping between axes.

        Do not try to add a new index_map by assigning to an item of this
        property. Use :meth:`~BaseData.create_reverse_map` instead.

        Returns
        -------
        index_map : read only dictionary
            Entries are 1D arrays used to interpret the axes of datasets.

        """

        out = {}
        try:
            g = self._data["reverse_map"]
        except KeyError:
            g = {}

        for name, value in g.items():
            out[name] = value[:]
        return memh5.ro_dict(out)

    # - Methods used by base class to control container structure. - #

    def dataset_name_allowed(self, name):
        """Permits datasets in the root and 'flags' groups."""

        parent_name, name = posixpath.split(name)
        return True if parent_name == "/" or parent_name == "/flags" else False

    def group_name_allowed(self, name):
        """Permits only the "flags" group."""

        return True if name == "/flags" else False

    # - Methods for manipulating and building the class. - #

    def create_cal(self, name, cal=None):
        """Create a new cal entry."""

        if cal is None:
            cal = {}
        self._data["cal"].create_group(name)
        for key, value in cal.items():
            self._data["cal"][name].attrs[key] = value

    def create_flag(self, name, *args, **kwargs):
        """Create a new flags dataset."""
        return self.create_dataset("flags/" + name, *args, **kwargs)

    def create_reverse_map(self, axis_name, reverse_map):
        """Create a new reverse map."""
        return self._data["reverse_map"].create_dataset(axis_name, data=reverse_map)

    def del_reverse_map(self, axis_name):
        """Delete a reverse map."""
        del self._data["reverse_map"][axis_name]

    # - These describe the various data axes. - #

    @property
    def ntime(self):
        """Length of the time axis of the visibilities."""

        return len(self.index_map["time"])

    @property
    def time(self):
        """The 'time' axis centres as Unix/POSIX time."""

        if (
            self.index_map["time"].dtype == np.float32
            # Already a calculated timestamp.
            or self.index_map["time"].dtype == np.float64
        ):
            return self.index_map["time"][:]

        else:
            time = _timestamp_from_fpga_cpu(
                self.index_map["time"]["ctime"], 0, self.index_map["time"]["fpga_count"]
            )
            # Shift from lower edge to centres.
            time += abs(np.median(np.diff(time)) / 2)
            return time

    @classmethod
    def _interpret_and_read(cls, acq_files, start, stop, datasets, out_group, **kwargs):
        """Read and concatenate the list of files. Keyword args may contain up to one axis selection."""
        # Save a reference to the first file to get index map information for
        # later.
        f_first = acq_files[0]

        # Handle axis selections
        sel = []
        for key in kwargs:
            if key[-4:] == "_sel":
                sel.append((key[:-4], kwargs[key]))
        if len(sel) > 1:
            raise ValueError("Cannot handle more than one axis selection.")
        elif len(sel) == 0:
            sel = None
        else:
            ax, sel = sel[0]

        if sel is None:
            andata_objs = [cls(d) for d in acq_files]
        else:
            andata_objs = [_read_axis_sel(cls, d, ax, sel) for d in acq_files]

        data = concatenate(
            andata_objs,
            out_group=out_group,
            start=start,
            stop=stop,
            datasets=datasets,
            convert_attribute_strings=cls.convert_attribute_strings,
            convert_dataset_strings=cls.convert_dataset_strings,
        )
        for k, v in f_first["index_map"].attrs.items():
            data.create_index_map(
                k, memh5.ensure_unicode(v) if cls.convert_dataset_strings else v
            )
        return data

    @classmethod
    def from_acq_h5(
        cls,
        acq_files,
        start=None,
        stop=None,
        datasets=None,
        out_group=None,
        distributed=False,
        comm=None,
        **kwargs,
    ):
        """Convert acquisition format hdf5 data to analysis data object.

        Reads hdf5 data produced by the acquisition system and converts it to
        analysis format in memory.

        Parameters
        ----------
        acq_files : filename, `h5py.File` or list there-of or filename pattern
            Files to convert from acquisition format to analysis format.
            Filename patterns with wild cards (e.g. "foo*.h5") are supported.
        start : integer, optional
            What frame to start at in the full set of files.
        stop : integer, optional
            What frame to stop at in the full set of files.
        datasets : list of strings
            Names of datasets to include from acquisition files. Default is to
            include all datasets found in the acquisition files.
        out_group : `h5py.Group`, hdf5 filename or `memh5.Group`
            Underlying hdf5 like container that will store the data for the
            BaseData instance.
        distributed : bool
            Read into a distributed array if `True`.
        comm : mpi4py.MPI.Comm
            MPI communicator to use.

        Examples
        --------
        Examples are analogous to those of :meth:`CorrData.from_acq_h5`.

        """

        if distributed:
            return cls._from_acq_h5_distributed(
                acq_files, start, stop, datasets, comm, **kwargs
            )
        else:
            return cls._from_acq_h5_single(
                acq_files, start, stop, datasets, out_group, **kwargs
            )

    @classmethod
    def _from_acq_h5_single(
        cls,
        acq_files,
        start=None,
        stop=None,
        datasets=None,
        out_group=None,
        **kwargs,
    ):
        """Load and concatenate the list of acquisition files into a local array.
        Axis selections may be supplied as keyword args, but the `BaseData` implementation
        only supports up to one axis selection.
        """

        # Make sure the input is a sequence and that we have at least one file.
        acq_files = tod.ensure_file_list(acq_files)
        if not acq_files:
            raise ValueError("Acquisition file list is empty.")

        to_close = [False] * len(acq_files)
        try:
            # Open the files while keeping track of this so that we can close
            # them later.
            _open_files(acq_files, to_close)

            # Now read them in: the functionality here is provided by the
            # overloaded method in the inherited class. If this method is
            # called on this base class, an exception will be raised.
            data = cls._interpret_and_read(
                acq_files=acq_files,
                start=start,
                stop=stop,
                datasets=datasets,
                out_group=out_group,
                **kwargs,
            )

        finally:
            # Close any files opened in this function.
            for ii in range(len(acq_files)):
                if len(to_close) > ii and to_close[ii]:
                    acq_files[ii].close()

        return data

    @classmethod
    def _from_acq_h5_distributed(
        cls,
        acq_files,
        start,
        stop,
        datasets,
        comm,
        **kwargs,
    ):
        """Load and concatenate the list of acquisition files into a distributed array.
        Axis selections may be supplied as keyword args, but the `BaseData` implementation
        only supports up to one axis selection, and it must match the distributed axis.
        """

        if cls.distributed_axis is None:
            raise RuntimeError(
                f"The container {cls} does not have a distributed axis "
                "defined but a distributed read was requested."
            )
        ax = cls.distributed_axis

        from mpi4py import MPI
        from caput import mpiutil, mpiarray, memh5

        # Turn into actual list of files
        files = tod.ensure_file_list(acq_files)

        # Construct communicator to use.
        if comm is None:
            comm = MPI.COMM_WORLD

        # Determine the size of the distributed axis
        ndist = None
        if comm.rank == 0:
            with h5py.File(files[0], "r") as f:
                ndist = len(f["index_map/" + ax][:])
        ndist = comm.bcast(ndist, root=0)

        # Handle selections along the distributed axis
        dist_sel = kwargs.get(ax + "_sel", None)

        # Calculate the global distributed selection
        dist_sel = _ensure_1D_selection(dist_sel)
        if isinstance(dist_sel, slice):
            dist_sel = list(range(*dist_sel.indices(ndist)))
        ndist = len(dist_sel)

        # Calculate the local selection
        n_local, d_start, d_end = mpiutil.split_local(ndist)
        local_dist_sel = _ensure_1D_selection(
            _convert_to_slice(dist_sel[d_start:d_end])
        )
        kwargs.update({ax + "_sel": local_dist_sel})

        # Load just the local part of the data.
        local_data = cls._from_acq_h5_single(
            acq_files=acq_files,
            start=start,
            stop=stop,
            datasets=datasets,
            out_group=None,
            **kwargs,
        )

        # Initialise distributed container
        data = cls(distributed=True, comm=comm)

        # Copy over the attributes
        memh5.copyattrs(
            local_data.attrs, data.attrs, convert_strings=cls.convert_attribute_strings
        )

        # Iterate over the datasets and copy them over
        for name, old_dset in local_data.datasets.items():

            # If this should be distributed, extract the sections and turn them into an MPIArray
            if name in cls._DIST_DSETS:
                dist_ind = list(old_dset.attrs["axis"]).index(ax)
                array = mpiarray.MPIArray.wrap(old_dset._data, axis=dist_ind, comm=comm)
            # Otherwise just copy copy out the old dataset
            else:
                array = old_dset[:]

            # Create the new dataset and copy over attributes
            new_dset = data.create_dataset(name, data=array)
            memh5.copyattrs(
                old_dset.attrs,
                new_dset.attrs,
                convert_strings=cls.convert_attribute_strings,
            )

        # Iterate over the flags and copy them over
        for name, old_dset in local_data.flags.items():

            # If this should be distributed, extract the sections and turn them into an MPIArray
            if name in cls._DIST_DSETS:
                dist_ind = list(old_dset.attrs["axis"]).index(ax)
                array = mpiarray.MPIArray.wrap(old_dset._data, axis=dist_ind, comm=comm)
            # Otherwise just copy copy out the old dataset
            else:
                array = old_dset[:]

            # Create the new dataset and copy over attributes
            new_dset = data.create_flag(name, data=array)
            memh5.copyattrs(
                old_dset.attrs,
                new_dset.attrs,
                convert_strings=cls.convert_attribute_strings,
            )

        # Copy over index maps
        for name, index_map in local_data.index_map.items():

            # Get reference to actual array
            index_map = index_map[:]

            # We need to explicitly stitch the distributed axis map back together
            if name == ax:

                # Gather onto all nodes and stich together
                dist_gather = comm.allgather(index_map)
                index_map = np.concatenate(dist_gather)

            # Create index map
            data.create_index_map(name, index_map)

        # Copy over reverse maps
        for name, reverse_map in local_data.reverse_map.items():

            # Get reference to actual array
            reverse_map = reverse_map[:]

            # Create index map
            data.create_reverse_map(name, reverse_map)

        return data

    @property
    def timestamp(self):
        """Deprecated name for :attr:`~BaseData.time`."""

        return self.time

    @staticmethod
    def convert_time(time):
        try:
            from .ephemeris import ensure_unix
        except ValueError:
            from .ephemeris import ensure_unix

        return ensure_unix(time)


class CorrData(BaseData):
    """Subclass of :class:`BaseData` for correlation data."""

    distributed_axis = "freq"

    @property
    def vis(self):
        """Convenience access to the visibilities array.

        Equivalent to `self.datasets['vis']`.
        """
        return self.datasets["vis"]

    @property
    def gain(self):
        """Convenience access to the gain dataset.

        Equivalent to `self.datasets['gain']`.
        """
        return self.datasets["gain"]

    @property
    def weight(self):
        """Convenience access to the visibility weight array.

        Equivalent to `self.flags['vis_weight']`.
        """
        return self.flags["vis_weight"]

    @property
    def input_flags(self):
        """Convenience access to the input flags dataset.

        Equivalent to `self.flags['inputs']`.
        """
        return self.flags["inputs"]

    @property
    def nprod(self):
        """Length of the prod axis."""
        return len(self.index_map["prod"])

    @property
    def prod(self):
        """The correlation product axis as channel pairs."""
        return self.index_map["prod"]

    @property
    def nfreq(self):
        """Length of the freq axis."""
        return len(self.index_map["freq"])

    @property
    def freq(self):
        """The spectral frequency axis as bin centres in MHz."""
        return self.index_map["freq"]["centre"]

    @property
    def ninput(self):
        return len(self.index_map["input"])

    @property
    def input(self):
        return self.index_map["input"]

    @property
    def nstack(self):
        return len(self.index_map["stack"])

    @property
    def stack(self):
        """The correlation product axis as channel pairs."""
        return self.index_map["stack"]

    @property
    def prodstack(self):
        """A pair of input indices representative of those in the stack.

        Note, these are correctly conjugated on return, and so calculations
        of the baseline and polarisation can be done without additionally
        looking up the stack conjugation.
        """
        if not self.is_stacked:
            return self.prod

        t = self.index_map["prod"][:][self.index_map["stack"]["prod"]]

        prodmap = t.copy()
        conj = self.stack["conjugate"]
        prodmap["input_a"] = np.where(conj, t["input_b"], t["input_a"])
        prodmap["input_b"] = np.where(conj, t["input_a"], t["input_b"])

        return prodmap

    @property
    def is_stacked(self):
        return "stack" in self.index_map and len(self.stack) != len(self.prod)

    @classmethod
    def _interpret_and_read(
        cls,
        acq_files,
        start,
        stop,
        datasets,
        out_group,
        stack_sel,
        prod_sel,
        input_sel,
        freq_sel,
        apply_gain,
        renormalize,
    ):
        # Selection defaults.
        freq_sel = _ensure_1D_selection(freq_sel)
        # If calculating the 'gain' dataset, ensure prerequisite datasets
        # are loaded.
        if datasets is not None and (
            ("vis" in datasets and apply_gain) or ("gain" in datasets)
        ):
            datasets = tuple(datasets) + ("gain", "gain_exp", "gain_coeff")
        # Always load packet loss dataset if available, so we can normalized
        # for it.
        if datasets is not None:
            norm_dsets = [d for d in datasets if re.match(ACQ_VIS_DATASETS, d)]
            if "vis_weight" in datasets:
                norm_dsets += ["vis_weight"]
            if len(norm_dsets):
                datasets = tuple(datasets) + ("flags/lost_packet_count",)

        # Inspect the header of the first file for version information.
        f = acq_files[0]
        try:
            archive_version = memh5.bytes_to_unicode(f.attrs["archive_version"])
        except KeyError:
            archive_version = "1.0.0"

        # Transform the dataset according to the version.
        if versiontuple(archive_version) < versiontuple("2.0.0"):
            # Nothing to do for input_sel as there is not input axis.
            if input_sel is not None:
                msg = (
                    "*input_sel* specified for archive version"
                    " 1.0 data which has no input axis."
                )
                raise ValueError(msg)
            prod_sel = _ensure_1D_selection(prod_sel)
            data = andata_from_acq1(
                acq_files, start, stop, prod_sel, freq_sel, datasets, out_group
            )
            input_sel = _ensure_1D_selection(input_sel)
        elif versiontuple(archive_version) >= versiontuple("2.0.0"):
            data, input_sel = andata_from_archive2(
                cls,
                acq_files,
                start,
                stop,
                stack_sel,
                prod_sel,
                input_sel,
                freq_sel,
                datasets,
                out_group,
            )

        # Generate the correct index_map/input for older files
        if versiontuple(archive_version) < versiontuple("2.1.0"):
            _remap_inputs(data)

        # Insert the gain dataset if requested, or datasets is not specified
        # For version 3.0.0 we don't need to do any of this
        if versiontuple(archive_version) < versiontuple("3.0.0") and (
            datasets is None or "gain" in datasets
        ):
            _insert_gains(data, input_sel)

            # Remove the FPGA applied gains (need to invert them first).
            if apply_gain and any(
                [re.match(ACQ_VIS_DATASETS, key) for key in data.datasets]
            ):
                from ch_util import tools

                gain = data.gain[:]

                # Create an array of safe-inverse gains.
                gain_inv = tools.invert_no_zero(gain)

                # Loop over datasets and apply inverse gains where appropriate
                for key, dset in data.datasets.items():
                    if (
                        re.match(ACQ_VIS_DATASETS, key)
                        and dset.attrs["axis"][1] == "prod"
                    ):
                        tools.apply_gain(
                            dset[:],
                            gain_inv,
                            out=dset[:],
                            prod_map=data.index_map["prod"],
                        )

        # Fix up wrapping of FPGA counts
        if versiontuple(archive_version) < versiontuple("2.4.0"):
            _unwrap_fpga_counts(data)

        # Renormalize for dropped packets
        # Not needed for > 3.0
        if (
            versiontuple(archive_version) < versiontuple("3.0.0")
            and renormalize
            and "lost_packet_count" in data.flags
        ):
            _renormalize(data)

        return data

    @classmethod
    def from_acq_h5(cls, acq_files, start=None, stop=None, **kwargs):
        """Convert acquisition format hdf5 data to analysis data object.

        This method overloads the one in BaseData.

        Changed Jan. 22, 2016: input arguments are now ``(acq_files, start,
        stop, **kwargs)`` instead of ``(acq_files, start, stop, prod_sel,
        freq_sel, datasets, out_group)``.

        Reads hdf5 data produced by the acquisition system and converts it to
        analysis format in memory.

        Parameters
        ----------
        acq_files : filename, `h5py.File` or list there-of or filename pattern
            Files to convert from acquisition format to analysis format.
            Filename patterns with wild cards (e.g. "foo*.h5") are supported.
        start : integer, optional
            What frame to start at in the full set of files.
        stop : integer, optional
            What frame to stop at in the full set of files.
        stack_sel : valid numpy index
            Used to select a subset of the stacked correlation products.
            Only one of *stack_sel*, *prod_sel*, and *input_sel* may be
            specified, with *prod_sel* preferred over *input_sel* and
            *stack_sel* proferred over both.
            :mod:`h5py` fancy indexing supported but to be used with caution
            due to poor reading performance.
        prod_sel : valid numpy index
            Used to select a subset of correlation products.
            Only one of *stack_sel*, *prod_sel*, and *input_sel* may be
            specified, with *prod_sel* preferred over *input_sel* and
            *stack_sel* proferred over both.
            :mod:`h5py` fancy indexing supported but to be used with caution
            due to poor reading performance.
        input_sel : valid numpy index
            Used to select a subset of correlator inputs.
            Only one of *stack_sel*, *prod_sel*, and *input_sel* may be
            specified, with *prod_sel* preferred over *input_sel* and
            *stack_sel* proferred over both.
            :mod:`h5py` fancy indexing supported but to be used with caution
            due to poor reading performance.
        freq_sel : valid numpy index
            Used to select a subset of frequencies.
            :mod:`h5py` fancy indexing supported but to be used with caution
            due to poor reading performance.
        datasets : list of strings
            Names of datasets to include from acquisition files. Default is to
            include all datasets found in the acquisition files.
        out_group : `h5py.Group`, hdf5 filename or `memh5.Group`
            Underlying hdf5 like container that will store the data for the
            BaseData instance.
        apply_gain : boolean, optional
            Whether to apply the inverse gains to the visibility datasets.
        renormalize : boolean, optional
            Whether to renormalize for dropped packets.
        distributed : boolean, optional
            Load data into a distributed dataset.
        comm : MPI.Comm
            Communicator to distributed over. Use MPI.COMM_WORLD if not set.

        Returns
        -------
        data : CorrData
            Loaded data object.

        Examples
        --------

        Suppose we have two acquisition format files (this test data is
        included in the ch_util repository):

        >>> import os
        >>> import glob
        >>> from . import test_andata
        >>> os.chdir(test_andata.data_path)
        >>> print(glob.glob('test_acq.h5*'))
        ['test_acq.h5.0001', 'test_acq.h5.0002']

        These can be converted into one big analysis format data object:

        >>> data = CorrData.from_acq_h5('test_acq.h5*')
        >>> print(data.vis.shape)
        (1024, 36, 31)

        If we only want a subset of the total frames (time bins) in these files
        we can supply start and stop indices.

        >>> data = CorrData.from_acq_h5('test_acq.h5*', start=5, stop=-3)
        >>> print(data.vis.shape)
        (1024, 36, 23)

        If we want a subset of the correlation products or spectral
        frequencies, specify the *prod_sel* or *freq_sel* respectively:

        >>> data = CorrData.from_acq_h5(
        ...     'test_acq.h5*',
        ...     prod_sel=[0, 8, 15, 21],
        ...     freq_sel=slice(5, 15),
        ...     )
        >>> print(data.vis.shape)
        (10, 4, 31)
        >>> data = CorrData.from_acq_h5('test_acq.h5*', prod_sel=1,
        ...                           freq_sel=slice(None, None, 10))
        >>> print(data.vis.shape)
        (103, 1, 31)

        The underlying hdf5-like container that holds the *analysis format*
        data can also be specified.

        >>> group = memh5.MemGroup()
        >>> data = CorrData.from_acq_h5('test_acq.h5*', out_group=group)
        >>> print(group['vis'].shape)
        (1024, 36, 31)
        >>> group['vis'] is data.vis
        True

        """

        stack_sel = kwargs.pop("stack_sel", None)
        prod_sel = kwargs.pop("prod_sel", None)
        input_sel = kwargs.pop("input_sel", None)
        freq_sel = kwargs.pop("freq_sel", None)
        datasets = kwargs.pop("datasets", None)
        out_group = kwargs.pop("out_group", None)
        apply_gain = kwargs.pop("apply_gain", True)
        renormalize = kwargs.pop("renormalize", True)
        distributed = kwargs.pop("distributed", False)
        comm = kwargs.pop("comm", None)

        if kwargs:
            msg = "Received unknown keyword arguments {}."
            raise ValueError(msg.format(kwargs.keys()))

        # If want a distributed file, just pass straight off to a private method
        if distributed:
            return cls._from_acq_h5_distributed(
                acq_files=acq_files,
                start=start,
                stop=stop,
                datasets=datasets,
                stack_sel=stack_sel,
                prod_sel=prod_sel,
                input_sel=input_sel,
                freq_sel=freq_sel,
                apply_gain=apply_gain,
                renormalize=renormalize,
                comm=comm,
            )

        return cls._from_acq_h5_single(
            acq_files=acq_files,
            start=start,
            stop=stop,
            datasets=datasets,
            out_group=out_group,
            stack_sel=stack_sel,
            prod_sel=prod_sel,
            input_sel=input_sel,
            freq_sel=freq_sel,
            apply_gain=apply_gain,
            renormalize=renormalize,
        )

    @classmethod
    def from_acq_h5_fast(cls, fname, comm=None, freq_sel=None, start=None, stop=None):
        """Efficiently read a CorrData file in a distributed fashion.

        This reads a single file from disk into a distributed container. In
        contrast to to `CorrData.from_acq_h5` it is more restrictive,
        allowing only contiguous slices of the frequency and time axes,
        and no down selection of the input/product/stack axis.

        Parameters
        ----------
        fname : str
            File name to read. Only supports one file at a time.
        comm : MPI.Comm, optional
            MPI communicator to distribute over. By default this will
            use `MPI.COMM_WORLD`.
        freq_sel : slice, optional
            A selection over the frequency axis. Only `slice` objects
            are supported. If not set, read all frequencies.
        start, stop : int, optional
            Start and stop indexes of the time selection.

        Returns
        -------
        data : andata.CorrData
            The CorrData container.
        """
        from mpi4py import MPI
        from caput import misc, mpiarray, memh5

        ## Datasets to read, if it's not listed here, it's not read at all
        # Datasets read by andata (should be small)
        DSET_CORE = ["flags/inputs", "flags/frac_lost", "flags/dataset_id"]
        # Datasets read directly and then inserted after the fact
        # (should have an input/product/stack axis, as axis=1)
        DSETS_DIRECT = ["vis", "gain", "flags/vis_weight"]

        if comm is None:
            comm = MPI.COMM_WORLD

        # Check the frequency selection
        if freq_sel is None:
            freq_sel = slice(None)
        if not isinstance(freq_sel, slice):
            raise ValueError("freq_sel must be a slice object, not %s" % repr(freq_sel))

        # Create the time selection
        time_sel = slice(start, stop)

        # Read the core dataset directly
        ad = cls.from_acq_h5(
            fname,
            datasets=DSET_CORE,
            distributed=True,
            comm=comm,
            freq_sel=freq_sel,
            start=start,
            stop=stop,
        )

        archive_version = memh5.bytes_to_unicode(ad.attrs["archive_version"])
        if versiontuple(archive_version) < versiontuple("3.0.0"):
            raise ValueError("Fast read not supported for files with version < 3.0.0")

        # Specify the selection to read from the file
        sel = (freq_sel, slice(None), time_sel)

        with misc.open_h5py_mpi(fname, "r", comm=comm) as fh:

            for ds_name in DSETS_DIRECT:

                if ds_name not in fh:
                    continue

                # Read dataset directly (distributed over input/product/stack axis) and
                # add to container
                arr = mpiarray.MPIArray.from_hdf5(
                    fh, ds_name, comm=comm, axis=1, sel=sel
                )
                arr = arr.redistribute(axis=0)
                dset = ad.create_dataset(ds_name, data=arr, distributed=True)

                # Copy over the attributes
                memh5.copyattrs(
                    fh[ds_name].attrs,
                    dset.attrs,
                    convert_strings=cls.convert_attribute_strings,
                )

        return ad


# For backwards compatibility.
AnData = CorrData


class HKData(BaseData):
    """Subclass of :class:`BaseData` for housekeeping data."""

    @property
    def atmel(self):
        """Get the ATMEL board that took these data.

        Returns
        -------
        comp : :obj:`layout.component`
            The ATMEL component that took these data.
        """
        try:
            from . import layout
        except ValueError:
            from . import layout

        sn = "ATMEGA" + "".join([str(i) for i in self.attrs["atmel_id"]])
        return layout.component.get(sn=sn)

    @property
    def mux(self):
        """Get the list of muxes in the data."""
        try:
            return self._mux
        except AttributeError:
            self._mux = []
            for dummy, d in self.datasets.items():
                self._mux.append(d.attrs["mux_address"][0])
            self._mux = np.sort(self._mux)
            return self._mux

    @property
    def nmux(self):
        """Get the number of muxes in the data."""
        return len(self.mux)

    def _find_mux(self, mux):
        for dummy, d in self.datasets.items():
            if d.attrs["mux_address"] == mux:
                return d
        raise ValueError("No dataset with mux = %d is present." % (mux))

    def chan(self, mux=-1):
        """Convenience access to the list of channels in a given mux.

        Parameters
        ----------
        mux : int
            A mux number. For housekeeping files with no multiplexing (e.g.,
            FLA's), leave this as ``-1``.

        Returns
        -------
        n : list
            The channels numbers.

        Raises
        ------
        :exc:`ValueError`
            Raised if **mux** does not exist.
        """
        try:
            self._chan
        except AttributeError:
            self._chan = dict()
        try:
            return self._chan[mux]
        except KeyError:
            ds = self._find_mux(mux)
            # chan_map = ds.attrs["axis"][0]
            self._chan[mux] = list(self.index_map[ds.attrs["axis"][0]])
            return self._chan[mux]

    def nchan(self, mux=-1):
        """Convenience access to the number of channels in a given mux.

        Parameters
        ----------
        mux : int
            A mux number. For housekeeping files with no multiplexing (e.g.,
            FLA's), leave this as ``-1``.

        Returns
        -------
        n : int
            The number of channels

        Raises
        ------
        :exc:`ValueError`
            Raised if **mux** does not exist.
        """
        return len(self.chan(mux))

    def tod(self, chan, mux=-1):
        """Convenience access to a single time-ordered datastream (TOD).

        Parameters
        ----------
        chan : int
            A channel number. (Generally, they should be in the range 0--7 for
            non-multiplexed data and 0--15 for multiplexed data.)
        mux : int
            A mux number. For housekeeping files with no multiplexing (e.g.,
            FLA's), leave this as ``-1``.

        Returns
        -------
        tod : :obj:`numpy.array`
            A 1D array of values for the requested channel/mux combination. Note
            that a reference to the data in the dataset is returned; this method
            does not make a copy.

        Raises
        ------
        :exc:`ValueError`
            Raised if one of **chan** or **mux** is not present in any dataset.
        """
        ds = self._find_mux(mux)
        chan_map = ds.attrs["axis"][0]
        try:
            idx = list(self.index_map[chan_map]).index(chan)
        except KeyError:
            raise ValueError("No channel %d exists for mux %d." % (chan, mux))

        # Return the data.
        return ds[idx, :]

    @classmethod
    def _interpret_and_read(cls, acq_files, start, stop, datasets, out_group):
        # Save a reference to the first file to get index map information for
        # later.
        f_first = acq_files[0]

        # Define dataset filter to do the transpose.
        def dset_filter(dataset):
            name = path.split(dataset.name)[1]
            match = False
            for regex in HK_DATASET_NAMES:
                if re.match(re.compile(regex), name):
                    match = True
            if match:
                # Do the transpose.
                data = np.empty((len(dataset[0]), len(dataset)), dtype=dataset[0].dtype)
                data = memh5.MemDatasetCommon.from_numpy_array(data)
                for i in range(len(dataset)):
                    for j in range(len(dataset[i])):
                        data[j, i] = dataset[i][j]
            memh5.copyattrs(
                dataset.attrs, data.attrs, convert_strings=cls.convert_attribute_strings
            )
            data.attrs["axis"] = (dataset.attrs["axis"][1], "time")
            return data

        andata_objs = [HKData(d) for d in acq_files]
        data = concatenate(
            andata_objs,
            out_group=out_group,
            start=start,
            stop=stop,
            datasets=datasets,
            dataset_filter=dset_filter,
            convert_attribute_strings=cls.convert_attribute_strings,
            convert_dataset_strings=cls.convert_dataset_strings,
        )

        # Some index maps saved as attributes, so convert to datasets.
        for k, v in f_first["index_map"].attrs.items():
            data.create_index_map(k, v)
        return data

    @classmethod
    def from_acq_h5(
        cls, acq_files, start=None, stop=None, datasets=None, out_group=None
    ):
        """Convert acquisition format hdf5 data to analysis data object.

        This method overloads the one in BaseData.

        Reads hdf5 data produced by the acquisition system and converts it to
        analysis format in memory.

        Parameters
        ----------
        acq_files : filename, `h5py.File` or list there-of or filename pattern
            Files to convert from acquisition format to analysis format.
            Filename patterns with wild cards (e.g. "foo*.h5") are supported.
        start : integer, optional
            What frame to start at in the full set of files.
        stop : integer, optional
            What frame to stop at in the full set of files.
        datasets : list of strings
            Names of datasets to include from acquisition files. Default is to
            include all datasets found in the acquisition files.
        out_group : `h5py.Group`, hdf5 filename or `memh5.Group`
            Underlying hdf5 like container that will store the data for the
            BaseData instance.

        Examples
        --------
        Examples are analogous to those of :meth:`CorrData.from_acq_h5`.
        """
        return super(HKData, cls).from_acq_h5(
            acq_files=acq_files,
            start=start,
            stop=stop,
            datasets=datasets,
            out_group=out_group,
        )


class HKPData(memh5.MemDiskGroup):
    """Subclass of :class:`BaseData` for housekeeping data."""

    # Convert strings to/from unicode on load and save
    convert_attribute_strings = True
    convert_dataset_strings = True

    @staticmethod
    def metrics(acq_files):
        """Get the names of the metrics contained within the files.

        Parameters
        ----------
        acq_files: list
            List of acquisition filenames.

        Returns
        -------
        metrics : list
        """

        import h5py

        metric_names = set()

        if isinstance(acq_files, str):
            acq_files = [acq_files]

        for fname in acq_files:
            with h5py.File(fname, "r") as fh:
                metric_names |= set(fh.keys())

        return metric_names

    @classmethod
    def from_acq_h5(
        cls, acq_files, start=None, stop=None, metrics=None, datasets=None, **kwargs
    ):
        """Load in the housekeeping files.

        Parameters
        ----------
        acq_files : list
            List of files to load.
        start, stop : datetime or float, optional
            Start and stop times for the range of data to load. Default is all.
        metrics : list
            Names of metrics to load. Default is all.
        datasets : list
            Synonym for metrics (the value of metrics will take precedence).


        Returns
        -------
        data : HKPData
        """

        from caput import time as ctime

        metrics = metrics if metrics is not None else datasets

        if "mode" not in kwargs:
            kwargs["mode"] = "r"
        if "ondisk" not in kwargs:
            kwargs["ondisk"] = True

        acq_files = [acq_files] if isinstance(acq_files, str) else acq_files
        files = [
            cls.from_file(
                f,
                convert_attribute_strings=cls.convert_attribute_strings,
                convert_dataset_strings=cls.convert_dataset_strings,
                **kwargs,
            )
            for f in acq_files
        ]

        def filter_time_range(dset):
            """Trim dataset to the specified time range."""
            data = dset[:]
            time = data["time"]

            mask = np.ones(time.shape, dtype=bool)

            if start is not None:
                tstart = ctime.ensure_unix(start)
                mask[:] *= time >= tstart

            if stop is not None:
                tstop = ctime.ensure_unix(stop)
                mask[:] *= time <= tstop

            return data[mask]

        def filter_file(f):
            """Filter a file's data down to the requested metrics
            and time range.
            """
            metrics_to_copy = set(f.keys())

            if metrics is not None:
                metrics_to_copy = metrics_to_copy & set(metrics)

            filtered_data = {}
            for dset_name in metrics_to_copy:
                filtered_data[dset_name] = filter_time_range(f[dset_name])
            return filtered_data

        def get_full_dtype(dset_name, filtered_data):
            """Returns a numpy.dtype object with the union of all columns
            from all files. Also returns the total length of the data set
            (metric) including all files.
            """

            length = 0
            all_columns = []
            all_types = []
            # review number of times and columns:
            for ii in range(len(filtered_data)):
                # If this file has this data set:
                if dset_name not in filtered_data[ii]:
                    continue
                # Increase the length of the data:
                length += len(filtered_data[ii][dset_name])
                # Add 'time' and 'value' columns first:
                if "time" not in all_columns:
                    all_columns.append("time")
                    all_types.append(filtered_data[ii][dset_name].dtype["time"])
                if "value" not in all_columns:
                    all_columns.append("value")
                    all_types.append(filtered_data[ii][dset_name].dtype["value"])
                # Add new column if any:
                for col in filtered_data[ii][dset_name].dtype.names:
                    if col not in all_columns:
                        all_columns.append(col)
                        all_types.append(filtered_data[ii][dset_name].dtype[col])

            data_dtype = np.dtype(
                [(all_columns[ii], all_types[ii]) for ii in range(len(all_columns))]
            )

            return data_dtype, length

        def get_full_attrs(dset_name, files):
            """Creates a 'full_attrs' dictionary of all attributes and all
            possible values they can take, from all the files, for a
            particular data set (metric). Also returns an 'index_remap'
            list of dictionaries to remap indices of values in different
            files.
            """

            full_attrs = {}  # Dictionary of attributes
            index_remap = []  # List of dictionaries (one per file)
            for ii, fl in enumerate(files):
                if dset_name not in fl:
                    continue
                index_remap.append({})  # List of dictionaries (one per file)
                for att, values in fl[dset_name].attrs.items():
                    # Reserve zeroeth entry for N/A
                    index_remap[ii][att] = np.zeros(len(values) + 1, dtype=int)
                    if att not in full_attrs:
                        full_attrs[att] = []
                    for idx, val in enumerate(values):
                        if val not in full_attrs[att]:
                            full_attrs[att] = np.append(full_attrs[att], val)
                        # Index of idx'th val in full_attrs[att]:
                        new_idx = np.where(full_attrs[att] == val)[0][0]
                        # zero is for N/A:
                        index_remap[ii][att][idx + 1] = new_idx + 1

            return full_attrs, index_remap

        def get_full_data(length, data_dtype, index_remap, filtered_data, dset_name):
            """Returns the full data matrix as a structured array. Values are
            modified when necessary acording to 'index_remap' to correspond
            to the final positions in the 'full_attrs'.
            """

            full_data = np.zeros(length, data_dtype)

            curr_ent = 0  # Current entry we are in the full data file
            for ii in range(len(filtered_data)):
                len_fl = len(filtered_data[ii][dset_name])
                curr_slice = np.s_[curr_ent : curr_ent + len_fl]
                if dset_name not in filtered_data[ii]:
                    continue
                for att in data_dtype.names:
                    # Length of this file:
                    if att in ["time", "value"]:
                        # No need to remap values:
                        full_data[att][curr_slice] = filtered_data[ii][dset_name][att]
                    elif att in index_remap[ii]:
                        # Needs remapping values
                        # (need to remove 1 beause indices are 1-based):
                        full_data[att][curr_slice] = index_remap[ii][att][
                            filtered_data[ii][dset_name][att]
                        ]
                    else:
                        # Column not in file. Fill with zeros:
                        full_data[att][curr_slice] = np.zeros(len_fl)
                # Update current entry value:
                curr_ent = curr_ent + len_fl

            return full_data

        hkp_data = cls()

        filtered_data = []
        for fl in files:
            filtered_data.append(filter_file(fl))

        for dset_name in metrics:
            data_dtype, length = get_full_dtype(dset_name, filtered_data)

            # Create the full dictionary of all attributes:
            full_attrs, index_remap = get_full_attrs(dset_name, files)

            # Populate the data here.( Need full attrs)
            full_data = get_full_data(
                length, data_dtype, index_remap, filtered_data, dset_name
            )
            new_dset = hkp_data.create_dataset(dset_name, data=full_data)

            # Populate attrs
            for att, values in full_attrs.items():
                new_dset.attrs[att] = memh5.bytes_to_unicode(values)

        return hkp_data

    def select(self, metric_name):
        """Return the metric as a pandas time-series DataFrame.

        Requires Pandas to be installed.

        Parameters
        ----------
        metric_name : string
            Name of metric to generate DataFrame for.

        Returns
        -------
        df : pandas.DataFrame
        """

        import pandas as pd

        dset = self[metric_name]

        fields = set(dset.dtype.fields.keys())
        time = pd.DatetimeIndex((dset["time"] * 1e9).astype("datetime64[ns]"))
        value = dset["value"]
        labels = fields - {"time", "value"}

        cols = {}
        cols["value"] = value
        cols["time"] = time

        for label_name in labels:
            label_ind = dset[label_name].astype(np.int16) - 1
            label_val = np.where(
                label_ind == -1, "-", dset.attrs[label_name][label_ind]
            )
            cols[label_name] = pd.Categorical(label_val)

        df = pd.DataFrame(data=cols)
        df.set_index("time", inplace=True)
        df.sort_index(inplace=True)

        return df

    def resample(self, metric_name, rule, how="mean", unstack=False, **kwargs):
        """Resample the metric onto a regular grid of time.

        This internally uses the Pandas resampling functionality so that
        documentation is a useful reference. This will return the metric with
        the labels as a series of multi-level columns.

        Parameters
        ----------
        metric_name : str
            Name of metric to resample.
        rule : str
            The set of times to resample onto (example '30S', '1Min', '2D'). See
            the pandas docs for a full description.
        how : str or callable, optional
            How should we combine samples to regrid the data? This takes any
            valid argument for the the pandas apply method. Useful options are
            `'mean'`, `'sum'`, `'min'`, `'max'` and `'std'`.
        unstack : bool, optional
            Unstack the data, i.e. return with the labels as hierarchial columns.
        kwargs
            Any remaining kwargs are passed to the `pandas.DataFrame.resample`
            method to give fine grained control of the resampling.

        Returns
        -------
        df : pandas.DataFrame
            A dataframe resampled onto a regular grid. Labels now appear as part
            of multi-level columns.
        """

        df = self.select(metric_name)

        group_columns = list(set(df.columns) - {"value"})

        resampled_df = df.groupby(group_columns).resample(rule).apply(how)

        if unstack:
            return resampled_df.unstack(group_columns)
        else:
            return resampled_df.reset_index(group_columns)


class WeatherData(BaseData):
    """Subclass of :class:`BaseData` for weather data."""

    @property
    def time(self):
        """Needs to be able to extrac times from both mingun_weather files
        and chime_weather files.
        """
        if "time" in self.index_map:
            return self.index_map["time"]
        else:
            return self.index_map["station_time_blockhouse"]

    @property
    def temperature(self):
        """For easy access to outside weather station temperature.
        Needs to be able to extrac temperatures from both mingun_weather files
        and chime_weather files.
        """
        if "blockhouse" in self.keys():
            return self["blockhouse"]["outTemp"]
        else:
            return self["outTemp"]

    def dataset_name_allowed(self, name):
        """Permits datasets in the root and 'blockhouse' groups."""

        parent_name, name = posixpath.split(name)
        return True if parent_name == "/" or parent_name == "/blockhouse" else False

    def group_name_allowed(self, name):
        """Permits only the "blockhouse" group."""

        return True if name == "/blockhouse" else False


class RawADCData(BaseData):
    """Subclass of :class:`BaseData` for raw ADC data."""

    @classmethod
    def _interpret_and_read(cls, acq_files, start, stop, datasets, out_group):

        # Define dataset filter to do the transpose.
        def dset_filter(dataset):
            if len(dataset.shape) == 2 and dataset.shape[1] == 1:
                data = dataset[:]
                data.shape = (dataset.shape[0],)
                data = memh5.MemDatasetCommon.from_numpy_array(data)
                memh5.copyattrs(
                    dataset.attrs,
                    data.attrs,
                    convert_strings=cls.convert_attribute_strings,
                )
            elif len(dataset.shape) == 2:
                data = dataset
            else:
                raise RuntimeError(
                    "Dataset (%s) has unexpected shape [%s]."
                    % (dataset.name, repr(dataset.shape))
                )
            return data

        andata_objs = [RawADCData(d) for d in acq_files]
        data = concatenate(
            andata_objs,
            out_group=out_group,
            start=start,
            stop=stop,
            datasets=datasets,
            dataset_filter=dset_filter,
            convert_attribute_strings=cls.convert_attribute_strings,
            convert_dataset_strings=cls.convert_dataset_strings,
        )
        return data


class GainFlagData(BaseData):
    """Subclass of :class:`BaseData` for gain, digitalgain, and flag input acquisitions.

    These acquisitions consist of a collection of updates to the real-time pipeline ordered
    chronologically.  In most cases the updates do not occur at a regular cadence.
    The time that each update occured can be accessed via `self.index_map['update_time']`.
    In addition, each update is given a unique update ID that can be accessed via
    `self.datasets['update_id']` and can be searched using the `self.search_update_id` method.
    """

    def resample(self, dataset, timestamp, transpose=False):
        """Return a dataset resampled at specific times.

        Parameters
        ----------
        dataset : string
            Name of the dataset to resample.
        timestamp : `np.ndarray`
            Unix timestamps.
        transpose : bool
            Tranpose the data such that time is the fastest varying axis.
            By default time will be the slowest varying axis.

        Returns
        -------
        data : np.ndarray
            The dataset resampled at the desired times and transposed if requested.
        """
        index = self.search_update_time(timestamp)
        dset = self.datasets[dataset][index]
        if transpose:
            dset = np.moveaxis(dset, 0, -1)

        return dset

    def search_update_time(self, timestamp):
        """Find the index into the `update_time` axis that is valid for specific times.

        For each time returns the most recent update the occured before that time.

        Parameters
        ----------
        timestamp : `np.ndarray` of unix timestamp
            Unix timestamps.

        Returns
        -------
        index : `np.ndarray` of `dtype = int`
            Index into the `update_time` axis that will yield values
            that are valid for the requested timestamps.
        """
        timestamp = np.atleast_1d(timestamp)

        if np.min(timestamp) < np.min(self.time):
            raise ValueError(
                "Cannot request timestamps before the earliest update_time."
            )

        dmax = np.max(timestamp) - np.max(self.time)
        if dmax > 0.0:
            msg = (
                "Requested timestamps are after the latest update_time "
                "by as much as %0.2f hours." % (dmax / 3600.0,)
            )
            warnings.warn(msg)

        index = np.digitize(timestamp, self.time, right=False) - 1

        return index

    def search_update_id(self, pattern, is_regex=False):
        """Find the index into the `update_time` axis corresponding to a particular `update_id`.

        Parameters
        ----------
        pattern : str
            The desired `update_id` or a glob pattern to search.
        is_regex : bool
            Set to True if `pattern` is a regular expression.

        Returns
        -------
        index : `np.ndarray` of `dtype = int`
            Index into the `update_time` axis that will yield all
            updates whose `update_id` matches the requested pattern.
        """
        import fnmatch

        ptn = pattern if is_regex else fnmatch.translate(pattern)
        regex = re.compile(ptn)
        index = np.array(
            [ii for ii, uid in enumerate(self.update_id[:]) if regex.match(uid)]
        )
        return index

    @property
    def time(self):
        """Aliases `index_map['update_time']` to `time` for `caput.tod` functionality."""
        return self.index_map["update_time"]

    @property
    def ntime(self):
        """Number of updates."""
        return len(self.index_map["update_time"])

    @property
    def input(self):
        """Correlator inputs."""
        return self.index_map["input"]

    @property
    def ninput(self):
        """Number of correlator inputs."""
        return len(self.index_map["input"])

    @property
    def update_id(self):
        """Aliases the `update_id` dataset."""
        return self.datasets["update_id"]


class FlagInputData(GainFlagData):
    """Subclass of :class:`GainFlagData` for flaginput acquisitions."""

    @property
    def flag(self):
        """Aliases the `flag` dataset."""
        return self.datasets["flag"]

    @property
    def source_flags(self):
        """Dictionary that allow look up of source flags based on source name."""
        if not hasattr(self, "_source_flags"):
            out = {}
            for kk, key in enumerate(self.index_map["source"]):
                out[key] = self.datasets["source_flags"][:, kk, :]

            self._source_flags = memh5.ro_dict(out)

        return self._source_flags

    def get_source_index(self, source_name):
        """Index into the `source` axis for a given source name."""
        return list(self.index_map["source"]).index(source_name)


class GainData(GainFlagData):
    """Subclass of :class:`GainFlagData` for gain and digitalgain acquisitions."""

    distributed_axis = "freq"

    @property
    def freq(self):
        """The spectral frequency axis as bin centres in MHz."""
        return self.index_map["freq"]["centre"]

    @property
    def nfreq(self):
        """Number of frequency bins."""
        return len(self.index_map["freq"])


class CalibrationGainData(GainData):
    """Subclass of :class:`GainData` for gain acquisitions."""

    distributed_axis = "freq"

    @property
    def source(self):
        """Names of the sources of gains."""
        return self.index_map["source"]

    @property
    def nsource(self):
        """Number of sources of gains."""
        return len(self.index_map["source"])

    @property
    def gain(self):
        """Aliases the `gain` dataset."""
        return self.datasets["gain"]

    @property
    def weight(self):
        """Aliases the `weight` dataset."""
        return self.datasets["weight"]

    @property
    def source_gains(self):
        """Dictionary that allows look up of source gains based on source name."""
        if not hasattr(self, "_source_gains"):
            out = {}
            for kk, key in enumerate(self.index_map["source"]):
                out[key] = self.datasets["source_gains"][:, kk, :]

            self._source_gains = memh5.ro_dict(out)

        return self._source_gains

    @property
    def source_weights(self):
        """Dictionary that allows look up of source weights based on source name."""
        if not hasattr(self, "_source_weights"):
            out = {}
            for kk, key in enumerate(self.index_map["source"]):
                out[key] = self.datasets["source_weights"][:, kk, :]

            self._source_weights = memh5.ro_dict(out)

        return self._source_weights

    def get_source_index(self, source_name):
        """Index into the `source` axis for a given source name."""
        return list(self.index_map["source"]).index(source_name)


class DigitalGainData(GainData):
    """Subclass of :class:`GainData` for digitalgain acquisitions."""

    distributed_axis = "freq"

    @property
    def gain_coeff(self):
        """The coefficient of the digital gain applied to the channelized data."""
        return self.datasets["gain_coeff"]

    @property
    def gain_exp(self):
        """The exponent of the digital gain applied to the channelized data."""
        return self.datasets["gain_exp"]

    @property
    def compute_time(self):
        """Unix timestamp indicating when the digital gain was computed."""
        return self.datasets["compute_time"]

    @property
    def gain(self):
        """The digital gain applied to the channelized data."""
        return self.datasets["gain_coeff"][:] * 2.0 ** (
            self.datasets["gain_exp"][:, np.newaxis, :]
        )


class BaseReader(tod.Reader):
    """Provides high level reading of CHIME data.

    You do not want to use this class, but rather one of its inherited classes
    (:class:`CorrReader`, :class:`HKReader`, :class:`WeatherReader`).

    Parses and stores meta-data from file headers allowing for the
    interpretation and selection of the data without reading it all from disk.

    Parameters
    ----------
    files : filename, `h5py.File` or list there-of or filename pattern
        Files containing data. Filename patterns with wild cards (e.g.
        "foo*.h5") are supported.
    """

    data_class = BaseData

    def __init__(self, files):

        # If files is a filename, or pattern, turn into list of files.
        if isinstance(files, str):
            files = sorted(glob.glob(files))

        self._data_empty = self.data_class.from_acq_h5(files, datasets=())

        # Fetch all meta data.
        time = self._data_empty.time
        datasets = _get_dataset_names(files[0])

        # Set the metadata attributes.
        self._files = tuple(files)
        self._time = time
        self._datasets = datasets
        # Set the default selections of the data.
        self.time_sel = (0, len(self.time))
        self.dataset_sel = datasets

    def select_time_range(self, start_time=None, stop_time=None):
        """Sets :attr:`~Reader.time_sel` to include a time range.

        The times from the samples selected will have bin centre timestamps
        that are bracketed by the given *start_time* and *stop_time*.

        Parameters
        ----------
        start_time : float or :class:`datetime.datetime`
            If a float, this is a Unix/POSIX time. Affects the first element of
            :attr:`~Reader.time_sel`.  Default leaves it unchanged.
        stop_time : float or :class:`datetime.datetime`
            If a float, this is a Unix/POSIX time. Affects the second element
            of :attr:`~Reader.time_sel`.  Default leaves it unchanged.

        """

        super(BaseReader, self).select_time_range(
            start_time=start_time, stop_time=stop_time
        )

    def read(self, out_group=None, distributed=False, comm=None):
        """Read the selected data.

        Parameters
        ----------
        out_group : `h5py.Group`, hdf5 filename or `memh5.Group`
            Underlying hdf5 like container that will store the data for the
            BaseData instance.
        distributed : bool
            Read into a distributed array if `True`.

        Returns
        -------
        data : :class:`BaseData`
            Data read from :attr:`~Reader.files` based on the selections given
            in :attr:`~Reader.time_sel`, :attr:`~Reader.prod_sel`, and
            :attr:`~Reader.freq_sel`.

        """

        return self.data_class.from_acq_h5(
            self.files,
            start=self.time_sel[0],
            stop=self.time_sel[1],
            datasets=self.dataset_sel,
            out_group=out_group,
            distributed=distributed,
            comm=comm,
        )


class CorrReader(BaseReader):
    """Subclass of :class:`BaseReader` for correlator data."""

    data_class = CorrData

    def __init__(self, files):
        super(CorrReader, self).__init__(files)
        data_empty = self._data_empty
        prod = data_empty.prod
        freq = data_empty.index_map["freq"]
        input = data_empty.index_map["input"]
        self._input = input
        self._prod = prod
        self._freq = freq
        self.prod_sel = None
        self.input_sel = None
        self.freq_sel = None
        # Create apply_gain and renormalize attributes,
        # which are passed to CorrData.from_acq_h5() when
        # the read() method is called.  This gives the
        # user the ability to turn off apply_gain and
        # renormalize when using Reader.
        self.apply_gain = True
        self.renormalize = True
        self.distributed = False
        # Insert virtual 'gain' dataset if required parent datasets are present.
        # We could be more careful about this, but I think this will always
        # work.
        datasets = self._datasets
        # if ('gain_coeff' in datasets and 'gain_exp' in datasets):
        datasets += ("gain",)
        self._datasets = datasets
        self.dataset_sel = datasets

    # Properties
    # ----------

    @property
    def prod(self):
        """Correlation products in data files."""
        return self._prod[:].copy()

    @property
    def input(self):
        """Correlator inputs in data files."""
        return self._input[:].copy()

    @property
    def freq(self):
        """Spectral frequency bin centres in data files."""
        return self._freq[:].copy()

    @property
    def prod_sel(self):
        """Which correlation products to read.

        Returns
        -------
        prod_sel : 1D data selection
            Valid numpy index for a 1D array, specifying what data to read
            along the correlation product axis.

        """
        return self._prod_sel

    @prod_sel.setter
    def prod_sel(self, value):
        if value is not None:
            # Check to make sure this is a valid index for the product axis.
            self.prod["input_a"][value]
            if self.input_sel is not None:
                msg = (
                    "*input_sel* is set and cannot specify both *prod_sel*"
                    " and *input_sel*."
                )
                raise ValueError(msg)
        self._prod_sel = value

    @property
    def input_sel(self):
        """Which correlator intputs to read.

        Returns
        -------
        input_sel : 1D data selection
            Valid numpy index for a 1D array, specifying what data to read
            along the correlation product axis.

        """
        return self._input_sel

    @input_sel.setter
    def input_sel(self, value):
        if value is not None:
            # Check to make sure this is a valid index for the product axis.
            self.input["chan_id"][value]
            if self.prod_sel is not None:
                msg = (
                    "*prod_sel* is set and cannot specify both *prod_sel*"
                    " and *input_sel*."
                )
                raise ValueError(msg)
        self._input_sel = value

    @property
    def freq_sel(self):
        """Which frequencies to read.

        Returns
        -------
        freq_sel : 1D data selection
            Valid numpy index for a 1D array, specifying what data to read
            along the frequency axis.

        """
        return self._freq_sel

    @freq_sel.setter
    def freq_sel(self, value):
        if value is not None:
            # Check to make sure this is a valid index for the frequency axis.
            self.freq["centre"][value]
        self._freq_sel = value

    # Data Selection Methods
    # ----------------------

    def select_prod_pairs(self, pairs):
        """Sets :attr:`~Reader.prod_sel` to include given product pairs.

        Parameters
        ----------
        pairs : list of integer pairs
            Input pairs to be included.

        """

        sel = []
        for input_a, input_b in pairs:
            for ii in range(len(self.prod)):
                p_input_a, p_input_b = self.prod[ii]
                if (input_a == p_input_a and input_b == p_input_b) or (
                    input_a == p_input_b and input_b == p_input_a
                ):
                    sel.append(ii)
        self.prod_sel = sel

    def select_prod_autos(self):
        """Sets :attr:`~Reader.prod_sel` to only auto-correlations."""

        sel = []
        for ii, prod in enumerate(self.prod):
            if prod[0] == prod[1]:
                sel.append(ii)
        self.prod_sel = sel

    def select_prod_by_input(self, input):
        """Sets :attr:`~Reader.prod_sel` to only products with given input.

        Parameters
        ----------
        input : integer
            Correlator input number.  All correlation products with
            this input as one of the pairs are selected.

        """

        sel = []
        for ii, prod in enumerate(self.prod):
            if prod[0] == input or prod[1] == input:
                sel.append(ii)
        self.prod_sel = sel

    def select_freq_range(self, freq_low=None, freq_high=None, freq_step=None):
        """Sets :attr:`~Reader.freq_sel` to given physical frequency range.

        Frequencies selected will have bin centres bracked by provided range.

        Parameters
        ----------
        freq_low : float
            Lower end of the frequency range in MHz.  Default is the lower edge
            of the band.
        freq_high : float
            Upper end of the frequency range in MHz.  Default is the upper edge
            of the band.
        freq_step : float
            How much bandwidth to skip over between samples in MHz. This value
            is approximate. Default is to include all samples in given range.

        """

        freq = self.freq["centre"]
        nfreq = len(freq)
        if freq_step is None:
            step = 1
        else:
            df = abs(np.mean(np.diff(freq)))
            step = int(freq_step // df)
        # Noting that frequencies are reverse ordered in datasets.
        if freq_low is None:
            stop = nfreq
        else:
            stop = np.where(freq < freq_low)[0][0]
        if freq_high is None:
            start = 0
        else:
            start = np.where(freq < freq_high)[0][0]
        # Slight tweak to behaviour if step is not unity, lining up edge on
        # freq_low instead of freq_high.
        start += (stop - start - 1) % step
        self.freq_sel = np.s_[start:stop:step]

    def select_freq_physical(self, frequencies):
        """Sets :attr:`~Reader.freq_sel` to include given physical frequencies.

        Parameters
        ----------
        frequencies : list of floats
            Frequencies to select. Physical frequencies are matched to indices
            on a best match basis.

        """

        freq_centre = self.freq["centre"]
        freq_width = self.freq["width"]
        frequencies = np.array(frequencies)
        n_sel = len(frequencies)
        diff_freq = abs(freq_centre - frequencies[:, None])
        match_mask = diff_freq < freq_width / 2
        freq_inds = []
        for ii in range(n_sel):
            matches = np.where(match_mask[ii, :])
            try:
                first_match = matches[0][0]
            except IndexError:
                msg = "No match for frequency %f MHz." % frequencies[ii]
                raise ValueError(msg)
            freq_inds.append(first_match)
        self.freq_sel = freq_inds

    # Data Reading
    # ------------

    def read(self, out_group=None):
        """Read the selected data.

        Parameters
        ----------
        out_group : `h5py.Group`, hdf5 filename or `memh5.Group`
            Underlying hdf5 like container that will store the data for the
            BaseData instance.

        Returns
        -------
        data : :class:`BaseData`
            Data read from :attr:`~Reader.files` based on the selections given
            in :attr:`~Reader.time_sel`, :attr:`~Reader.prod_sel`, and
            :attr:`~Reader.freq_sel`.

        """

        dsets = tuple(self.dataset_sel)

        # Add in virtual gain dataset
        # This is done in earlier now, in self.datasets.
        # if ('gain_coeff' in dsets and 'gain_exp' in dsets):
        #    dsets += ('gain',)

        return CorrData.from_acq_h5(
            self.files,
            start=self.time_sel[0],
            stop=self.time_sel[1],
            prod_sel=self.prod_sel,
            freq_sel=self.freq_sel,
            input_sel=self.input_sel,
            apply_gain=self.apply_gain,
            renormalize=self.renormalize,
            distributed=self.distributed,
            datasets=dsets,
            out_group=out_group,
        )


# For backwards compatibility.
Reader = CorrReader


class HKReader(BaseReader):
    """Subclass of :class:`BaseReader` for HK data."""

    data_class = HKData


class HKPReader(BaseReader):
    """Subclass of :class:`BaseReader` for HKP data."""

    data_class = HKPData


class WeatherReader(BaseReader):
    """Subclass of :class:`BaseReader` for weather data."""

    data_class = WeatherData


class FlagInputReader(BaseReader):
    """Subclass of :class:`BaseReader` for input flag data."""

    data_class = FlagInputData


class CalibrationGainReader(BaseReader):
    """Subclass of :class:`BaseReader` for calibration gain data."""

    data_class = CalibrationGainData


class DigitalGainReader(BaseReader):
    """Subclass of :class:`BaseReader` for digital gain data."""

    data_class = DigitalGainData


class RawADCReader(BaseReader):
    """Subclass of :class:`BaseReader` for raw ADC data."""

    data_class = RawADCData


class AnDataError(Exception):
    """Exception raised when something unexpected happens with the data."""

    pass


# Functions
# ---------

# In caput now.
concatenate = tod.concatenate


def subclass_from_obj(cls, obj):
    """Pick a subclass of :class:`BaseData` based on an input object.

    Parameters
    ----------
    cls : subclass of :class:`BaseData` (class, not an instance)
          Default class to return.
    obj : :class:`h5py.Group`, filename, :class:`memh5.Group` or
          :class:`BaseData` object from which to determine the appropriate
          subclass of :class:`AnData`.

    """
    # If obj is a filename, open it and recurse.
    if isinstance(obj, str):
        with h5py.File(obj, "r") as f:
            cls = subclass_from_obj(cls, f)
        return cls

    new_cls = cls
    acquisition_type = None
    try:
        acquisition_type = obj.attrs["acquisition_type"]
    except (AttributeError, KeyError):
        pass
    if acquisition_type == "corr":
        new_cls = CorrData
    elif acquisition_type == "hk":
        new_cls = HKData
    elif acquisition_type is None:
        if isinstance(obj, BaseData):
            new_cls = obj.__class__
    return new_cls


# Private Functions
# -----------------

# Utilities


def _open_files(files, opened):
    """Ensure that files are open, keeping a record of what was done.

    The arguments are modified in-place instead of returned, so that partial
    work is recorded in the event of an error.

    """

    for ii, this_file in enumerate(list(files)):
        # Sort out how to get an open hdf5 file.
        open_file, was_opened = memh5.get_h5py_File(this_file, mode="r")
        opened[ii] = was_opened
        files[ii] = open_file


def _ensure_1D_selection(selection):
    if isinstance(selection, tuple):
        if len(selection) != 1:
            msg = "Wrong number of indices."
            raise ValueError(msg)
        selection = selection[0]
    if selection is None:
        selection = np.s_[:]
    elif hasattr(selection, "__iter__"):
        selection = np.array(selection)
    elif isinstance(selection, slice):
        pass
    elif np.issubdtype(type(selection), np.integer):
        selection = np.s_[selection : selection + 1]
    else:
        raise ValueError("Cannont be converted to a 1D selection.")

    if isinstance(selection, np.ndarray):
        if selection.ndim != 1:
            msg = "Data selections may only be one dimensional."
            raise ValueError(msg)
        # The following is more efficient and solves h5py issue #425. Converts
        # to integer selection.
        if len(selection) == 1:
            return _ensure_1D_selection(selection[0])
        if np.issubdtype(selection.dtype, np.integer):
            if np.any(np.diff(selection) <= 0):
                raise ValueError("h5py requires sorted non-duplicate selections.")
        elif not np.issubdtype(selection.dtype, bool):
            raise ValueError("Array selections must be integer or boolean type.")
        elif np.issubdtype(selection.dtype, bool):
            # This is a workaround for h5py/h5py#1750
            selection = selection.nonzero()[0]

    return selection


def _convert_to_slice(selection):
    if hasattr(selection, "__iter__") and len(selection) > 1:

        uniq_step = np.unique(np.diff(selection))

        if (len(uniq_step) == 1) and uniq_step[0]:

            a = selection[0]
            b = selection[-1]
            b = b + (1 - (b < a) * 2)

            selection = slice(a, b, uniq_step[0])

    return selection


def _get_dataset_names(f):
    f, toclose = memh5.get_h5py_File(f, mode="r")
    try:
        dataset_names = ()
        for name in f.keys():
            if not memh5.is_group(f[name]):
                dataset_names += (name,)
        if "blockhouse" in f and memh5.is_group(f["blockhouse"]):
            # chime_weather datasets are inside group "blockhouse"
            for name in f["blockhouse"].keys():
                if not memh5.is_group(f["blockhouse"][name]):
                    dataset_names += ("blockhouse/" + name,)
        if "flags" in f and memh5.is_group(f["flags"]):
            for name in f["flags"].keys():
                if not memh5.is_group(f["flags"][name]):
                    dataset_names += ("flags/" + name,)
    finally:
        if toclose:
            f.close()
    return dataset_names


def _resolve_stack_prod_input_sel(
    stack_sel, stack_map, stack_rmap, prod_sel, prod_map, input_sel, input_map
):
    nsels = (stack_sel is not None) + (prod_sel is not None) + (input_sel is not None)
    if nsels > 1:
        raise ValueError(
            "Only one of *stack_sel*, *input_sel*, and *prod_sel* may be specified."
        )

    if nsels == 0:
        stack_sel = _ensure_1D_selection(stack_sel)
        prod_sel = _ensure_1D_selection(prod_sel)
        input_sel = _ensure_1D_selection(input_sel)
    else:
        if prod_sel is not None:
            prod_sel = _ensure_1D_selection(prod_sel)
            # Choose inputs involved in selected products.
            input_sel = _input_sel_from_prod_sel(prod_sel, prod_map)
            stack_sel = _stack_sel_from_prod_sel(prod_sel, stack_rmap)
        elif input_sel is not None:
            input_sel = _ensure_1D_selection(input_sel)
            prod_sel = _prod_sel_from_input_sel(input_sel, input_map, prod_map)
            stack_sel = _stack_sel_from_prod_sel(prod_sel, stack_rmap)
        else:  # stack_sel
            stack_sel = _ensure_1D_selection(stack_sel)
            prod_sel = _prod_sel_from_stack_sel(stack_sel, stack_map, stack_rmap)
            input_sel = _input_sel_from_prod_sel(prod_sel, prod_map)

        # Now we need to rejig the index maps for the subsets of the inputs,
        # prods.
        stack_inds = np.arange(len(stack_map), dtype=int)[stack_sel]
        # prod_inds = np.arange(len(prod_map), dtype=int)[prod_sel]  # never used
        input_inds = np.arange(len(input_map), dtype=int)[input_sel]

        stack_rmap = stack_rmap[prod_sel]
        stack_rmap["stack"] = _search_array(stack_inds, stack_rmap["stack"])

        # Remake stack map from scratch, since prod referenced in current stack
        # map may have dissapeared.
        stack_map = np.empty(len(stack_inds), dtype=stack_map.dtype)
        stack_map["prod"] = _search_array(
            stack_rmap["stack"], np.arange(len(stack_inds))
        )
        stack_map["conjugate"] = stack_rmap["conjugate"][stack_map["prod"]]

        prod_map = prod_map[prod_sel]
        pa = _search_array(input_inds, prod_map["input_a"])
        pb = _search_array(input_inds, prod_map["input_b"])
        prod_map["input_a"] = pa
        prod_map["input_b"] = pb
        input_map = input_map[input_sel]
    return stack_sel, stack_map, stack_rmap, prod_sel, prod_map, input_sel, input_map


def _npissorted(arr):
    return np.all(np.diff >= 0)


def _search_array(a, v):
    """Find the indeces in array `a` of values in array 'v'.

    Use algorithm that presorts `a`, efficient if `v` is long.

    """
    a_sort_inds = np.argsort(a, kind="mergesort")
    a_sorted = a[a_sort_inds]
    indeces_in_sorted = np.searchsorted(a_sorted, v)
    # Make sure values actually present.
    if not np.all(v == a_sorted[indeces_in_sorted]):
        raise ValueError("Element in 'v' not in 'a'.")
    return a_sort_inds[indeces_in_sorted]


def _input_sel_from_prod_sel(prod_sel, prod_map):
    prod_map = prod_map[prod_sel]
    input_sel = []
    for p0, p1 in prod_map:
        input_sel.append(p0)
        input_sel.append(p1)
    # ensure_1D here deals with h5py issue #425.
    input_sel = _ensure_1D_selection(sorted(list(set(input_sel))))
    return input_sel


def _prod_sel_from_input_sel(input_sel, input_map, prod_map):
    inputs = list(np.arange(len(input_map), dtype=int)[input_sel])
    prod_sel = []
    for ii, p in enumerate(prod_map):
        if p[0] in inputs and p[1] in inputs:
            prod_sel.append(ii)
    # ensure_1D here deals with h5py issue #425.
    prod_sel = _ensure_1D_selection(prod_sel)
    return prod_sel


def _stack_sel_from_prod_sel(prod_sel, stack_rmap):
    stack_sel = stack_rmap["stack"][prod_sel]
    stack_sel = _ensure_1D_selection(sorted(list(set(stack_sel))))
    return stack_sel


def _prod_sel_from_stack_sel(stack_sel, stack_map, stack_rmap):
    stack_inds = np.arange(len(stack_map))[stack_sel]
    stack_rmap_sort_inds = np.argsort(stack_rmap["stack"], kind="mergesort")
    stack_rmap_sorted = stack_rmap["stack"][stack_rmap_sort_inds]
    left_indeces = np.searchsorted(stack_rmap_sorted, stack_inds, side="left")
    right_indeces = np.searchsorted(stack_rmap_sorted, stack_inds, side="right")
    prod_sel = []
    for ii in range(len(stack_inds)):
        prod_sel.append(stack_rmap_sort_inds[left_indeces[ii] : right_indeces[ii]])
    prod_sel = np.concatenate(prod_sel)
    prod_sel = _ensure_1D_selection(sorted(list(set(prod_sel))))
    return prod_sel


def versiontuple(v):
    return tuple(map(int, (v.split("."))))


# Calculations from data.


def _renormalize(data):
    """Correct vis and vis_weight for lost packets."""
    from ch_util import tools

    # Determine the datasets that need to be renormalized
    datasets_to_renormalize = [
        key for key in data.datasets if re.match(ACQ_VIS_DATASETS, key)
    ]

    if not datasets_to_renormalize:
        return

    # Determine if we will correct vis_weight in addition to vis.
    adjust_weight = "vis_weight" in data.flags

    # Extract number of packets expected
    n_packets_expected = data.attrs["gpu.gpu_intergration_period"][0]

    # Loop over frequencies to limit memory usage
    for ff in range(data.nfreq):

        # Calculate the fraction of packets received
        weight_factor = 1.0 - data.flags["lost_packet_count"][ff] / float(
            n_packets_expected
        )

        # Multiply vis_weight by fraction of packets received
        if adjust_weight:
            data.flags["vis_weight"][ff] = np.round(
                data.flags["vis_weight"][ff] * weight_factor[None, :]
            )

        # Divide vis by fraction of packets received
        weight_factor = tools.invert_no_zero(weight_factor)

        for key in datasets_to_renormalize:
            data.datasets[key][ff] *= weight_factor[None, :]


def _unwrap_fpga_counts(data):
    """Unwrap 32-bit FPGA counts in a CorrData object."""

    import datetime

    time_map = data.index_map["time"][:]

    # If FPGA counts are already 64-bit then we don't need to unwrap
    if time_map["fpga_count"].dtype == np.uint64:
        return

    # Try and fetch out required attributes, if they are not there (which
    # happens in older files), fill in the usual values
    try:
        nfreq = data.attrs["n_freq"][0]
        samp_freq_MHz = data.attrs["fpga.samp_freq"][0]
    except KeyError:
        nfreq = 1024
        samp_freq_MHz = 800.0

    # Calculate the length of an FPGA count and the time it takes to wrap
    seconds_per_count = 2.0 * nfreq / (samp_freq_MHz * 1e6)
    wrap_time = 2**32.0 * seconds_per_count

    # Estimate the FPGA initial zero time from the timestamp in the acquisition
    # name, if the acq name is not there, or of the correct format just silently return
    try:
        acq_name = data.attrs["acquisition_name"]
        acq_dt = datetime.datetime.strptime(acq_name[:16], "%Y%m%dT%H%M%SZ")
    except (KeyError, ValueError):
        return
    acq_start = CorrData.convert_time(acq_dt)

    # Calculate the time that the count last wrapped
    last_wrap = time_map["ctime"] - time_map["fpga_count"] * seconds_per_count

    # Use this and the FPGA zero time to calculate the total number of wraps
    num_wraps = np.round((last_wrap - acq_start) / wrap_time).astype(np.uint64)

    # Correct the FPGA counts by adding on the counts lost by wrapping
    fpga_corrected = time_map["fpga_count"] + num_wraps * 2**32

    # Create an array to represent the new time dataset, and fill in the corrected values
    _time_dtype = [("fpga_count", np.uint64), ("ctime", np.float64)]
    new_time_map = np.zeros(time_map.shape, dtype=_time_dtype)
    new_time_map["fpga_count"] = fpga_corrected
    new_time_map["ctime"] = time_map["ctime"]

    # Replace the time input map
    data.del_index_map("time")
    data.create_index_map("time", new_time_map)


def _timestamp_from_fpga_cpu(cpu_s, cpu_us, fpga_counts):
    ntime = len(cpu_s)
    timestamp = np.empty(ntime, dtype=np.float64)
    timestamp[:] = cpu_s
    if cpu_us is not None:
        timestamp += cpu_us / 1.0e6
    # If we have the more precise fpga clock, use it.  Use the above to
    # calibrate.
    if fpga_counts is not None:
        timestamp_cpu = timestamp.copy()
        # Find discontinuities in the fpga_counts from wrapping.
        d_fpga_counts = np.diff(fpga_counts.astype(np.int64))
        (edge_inds,) = np.where(d_fpga_counts != np.median(d_fpga_counts))
        edge_inds = np.concatenate(([0], edge_inds + 1, [ntime]))
        # Calculate a global slope.
        slope_num = 0
        slope_den = 0
        for ii in range(len(edge_inds) - 1):
            sl = np.s_[edge_inds[ii] : edge_inds[ii + 1]]
            mean_cpu = np.mean(timestamp_cpu[sl])
            mean_fpga = np.mean(fpga_counts[sl])
            diff_cpu = timestamp_cpu[sl] - mean_cpu
            diff_fpga = fpga_counts[sl] - mean_fpga
            slope_num += np.sum(diff_cpu * diff_fpga)
            slope_den += np.sum(diff_fpga**2)
        slope = slope_num / slope_den
        # Calculate offset in each section.
        for ii in range(len(edge_inds) - 1):
            sl = np.s_[edge_inds[ii] : edge_inds[ii + 1]]
            mean_cpu = np.mean(timestamp_cpu[sl])
            mean_fpga = np.mean(fpga_counts[sl])
            offset = mean_cpu - slope * mean_fpga
            # Apply fit.
            timestamp[sl] = slope * fpga_counts[sl] + offset
    # XXX
    # The above provides integration ends, not centres.  Fix:
    # delta = np.median(np.diff(timestamp))
    # timestamp -= abs(delta) / 2.
    return timestamp


# IO for acquisition format 1.0


def _copy_dataset_acq1(
    dataset_name, acq_files, start, stop, out_data, prod_sel=None, freq_sel=None
):

    s_ind = 0
    ntime = stop - start
    for ii, acq in enumerate(acq_files):
        acq_dataset = acq[dataset_name]
        this_ntime = len(acq_dataset)
        if s_ind + this_ntime < start or s_ind >= stop:
            # No data from this file is included.
            s_ind += this_ntime
            continue
        # What data (time frames) are included in this file.
        # out_slice = np.s_[max(0, s_ind - start):s_ind - start + this_ntime]
        # acq_slice = np.s_[max(0, start - s_ind):min(this_ntime, stop - s_ind)]
        acq_slice, out_slice = tod._get_in_out_slice(start, stop, s_ind, this_ntime)
        # Split the fields of the dataset into separate datasets and reformat.
        split_dsets, split_dsets_cal = _format_split_acq_dataset_acq1(
            acq_dataset, acq_slice
        )
        if dataset_name == "vis":
            # Convert to 64 but complex.
            if set(split_dsets.keys()) != {"imag", "real"}:
                msg = (
                    "Visibilities should have fields 'real' and 'imag'"
                    " and instead have %s." % str(list(split_dsets.keys()))
                )
                raise ValueError(msg)
            vis_data = np.empty(split_dsets["real"].shape, dtype=np.complex64)
            vis_data.real[:] = split_dsets["real"]
            vis_data.imag[:] = split_dsets["imag"]

            split_dsets = {"": vis_data}
            split_dsets_cal = {}

        for split_dset_name, split_dset in split_dsets.items():
            if prod_sel is not None:  # prod_sel could be 0.
                # Do this in two steps to get around shape matching.
                split_dset = split_dset[freq_sel, :, :]
                split_dset = split_dset[:, prod_sel, :]
            if split_dset_name:
                full_name = dataset_name + "_" + split_dset_name
            else:
                full_name = dataset_name
            if start >= s_ind:
                # First file, initialize output dataset.
                shape = split_dset.shape[:-1] + (ntime,)
                if split_dset_name in split_dsets_cal:
                    attrs = {"cal": split_dsets_cal[split_dset_name]}
                else:
                    attrs = {}
                # Try to figure out the axis names.
                if prod_sel is not None:
                    # The shape of the visibilities.
                    attrs["axis"] = ("freq", "prod", "time")
                else:
                    ndim = len(shape)
                    attrs["axis"] = ("UNKNOWN",) * (ndim - 1) + ("time",)
                ds = out_data.create_dataset(
                    full_name, dtype=split_dset.dtype, shape=shape
                )

                # Copy over attributes
                for k, v in attrs.items():
                    ds.attrs[k] = v
            # Finally copy the data over.
            out_data.datasets[full_name][..., out_slice] = split_dset[:]
        s_ind += this_ntime


def _check_files_acq1(files):
    """Gets a list of open hdf5 file objects and checks their consistency.

    Checks that they all have the same datasets and that all datasets have
    consistent data types.

    Essential arguments are modified in-place instead of using return values.
    This keeps the lists as up to date as possible in the event that an
    exception is raised within this function.

    Non-essential information is returned such as the dtypes for all the
    datasets.

    """

    first_file = True
    for ii, open_file in enumerate(list(files)):
        # Sort out how to get an open hdf5 file.
        # Check that all files have the same datasets with the same dtypes
        # and consistent shape.
        # All datasets in the same file must be the same shape.
        # Between files, all datasets with the same name must have the same
        # dtype.
        this_dtypes = {}
        first_dset = True
        for key in open_file.keys():
            if not memh5.is_group(open_file[key]):
                this_dtypes[key] = open_file[key].dtype
                if first_dset:
                    this_dset_shape = open_file[key].shape
                    first_dset = False
                else:
                    if open_file[key].shape != this_dset_shape:
                        msg = "Datasets in a file do not all have same shape."
                        raise ValueError(msg)
        if first_file:
            dtypes = this_dtypes
            first_file = False
        else:
            if this_dtypes != dtypes:
                msg = "Files do not have compatible datasets."
                raise ValueError(msg)
    return dtypes


def _get_header_info_acq1(h5_file):
    # Right now only have to deal with one format.  In the future will need to
    # deal with all different kinds of data.
    header_info = _data_attrs_from_acq_attrs_acq1(h5_file.attrs)
    # Now need to calculate the time stamps.
    timestamp_data = h5_file["timestamp"]
    if not len(timestamp_data):
        msg = "Acquisition file contains zero frames"
        raise AnDataError(msg)
    time = np.empty(
        len(timestamp_data), dtype=[("fpga_count", "<u4"), ("ctime", "<f8")]
    )
    time_upper_edges = _timestamp_from_fpga_cpu(
        timestamp_data["cpu_s"], timestamp_data["cpu_us"], timestamp_data["fpga_count"]
    )
    time_lower_edges = time_upper_edges - np.median(np.diff(time_upper_edges))
    time["ctime"] = time_lower_edges
    time["fpga_count"] = timestamp_data["fpga_count"]
    header_info["time"] = time
    datasets = [key for key in h5_file.keys() if not memh5.is_group(h5_file[key])]
    header_info["datasets"] = tuple(datasets)
    return header_info


def _resolve_header_info_acq1(header_info):
    first_info = header_info[0]
    freq = first_info["freq"]
    prod = first_info["prod"]
    datasets = first_info["datasets"]
    time_list = [first_info["time"]]
    for info in header_info[1:]:
        if not np.allclose(info["freq"]["width"], freq["width"]):
            msg = "Files do not have consistent frequency bin widths."
            raise ValueError(msg)
        if not np.allclose(info["freq"]["centre"], freq["centre"]):
            msg = "Files do not have consistent frequency bin centres."
            raise ValueError(msg)
        if not np.all(info["prod"] == prod):
            msg = "Files do not have consistent correlation products."
            raise ValueError(msg)
        if not np.all(info["datasets"] == datasets):
            msg = "Files do not have consistent data sets."
            raise ValueError(msg)
        time_list.append(info["time"])
    time = np.concatenate(time_list)
    return time, prod, freq, datasets


def _get_files_frames_acq1(files, start, stop):
    """Counts the number of frames in each file and sorts out which frames to
    read."""

    dataset_name = "vis"  # For now just base everything off of 'vis'.
    n_times = []
    for this_file in files:
        # Make sure the dataset is 1D.
        if len(this_file[dataset_name].shape) != 1:
            raise ValueError("Expected 1D datasets.")
        n_times.append(len(this_file[dataset_name]))
    n_time_total = np.sum(n_times)
    return tod._start_stop_inds(start, stop, n_time_total)


def _format_split_acq_dataset_acq1(dataset, time_slice):
    """Formats a dataset from a acq h5 file into a more easily handled array.

    Completely reverses the order of all axes.

    """

    # Get shape information.
    ntime = len(dataset)
    ntime_out = len(np.arange(ntime)[time_slice])
    # If each record is an array, then get that shape.
    back_shape = dataset[0].shape
    # The shape of the output array.
    reversed_back_shape = list(back_shape)
    reversed_back_shape.reverse()
    out_shape = tuple(reversed_back_shape) + (ntime_out,)
    # Check if there are multiple data fields in this dataset.  If so they will
    # each end up in their own separate arrays.
    if dataset[0].dtype.fields is None:
        dtype = dataset[0].dtype
        out = np.empty(out_shape, dtype=dtype)
        for jj, ii in enumerate(np.arange(ntime)[time_slice]):
            # 1D case is trivial.
            if not back_shape:
                out[jj] = dataset[ii]
            elif len(back_shape) == 1:
                out[:, jj] = dataset[ii]
            else:
                raise NotImplementedError("Not done yet.")
                # Otherwise, loop over all dimensions except the last one.
                it = np.nditer(dataset[ii][..., 0], flags=["multi_index"], order="C")
                while not it.finished:
                    it.iternext()
        if "cal" in dataset.attrs:
            if len(dataset.attrs["cal"]) != 1:
                msg = "Mismatch between dataset and it's cal attribute."
                raise AttributeError(msg)
            out_cal = {"": dataset.attrs["cal"][0]}
        else:
            out_cal = {}
        return {"": out}, out_cal
    else:
        fields = list(dataset[0].dtype.fields.keys())
        # If there is a 'cal' attribute, make sure it's the right shape.
        if "cal" in dataset.attrs:
            if dataset.attrs["cal"].shape != (1,):
                msg = "'cal' attribute has more than one element."
                raise AttributeError(msg)
            if len(list(dataset.attrs["cal"].dtype.fields.keys())) != len(fields):
                msg = "'cal' attribute not compatible with dataset dtype."
                raise AttributeError(msg)
        out = {}
        out_cal = {}
        # Figure out what fields there are and allocate memory.
        for field in fields:
            dtype = dataset[0][field].dtype
            out_arr = np.empty(out_shape, dtype=dtype)
            out[field] = out_arr
            if "cal" in dataset.attrs:
                out_cal[field] = memh5.bytes_to_unicode(dataset.attrs["cal"][0][field])
        for jj, ii in enumerate(np.arange(ntime)[time_slice]):
            # Copy data for efficient read.
            record = dataset[ii]  # Copies to memory.
            for field in fields:
                if not back_shape:
                    out[field][jj] = record[field]
                elif len(back_shape) == 1:
                    out[field][:, jj] = record[field][:]
                else:
                    # Multidimensional, try to be more efficient.
                    it = np.nditer(record[..., 0], flags=["multi_index"], order="C")
                    while not it.finished:
                        # Reverse the multiindex for the out array.
                        ind = it.multi_index + (slice(None),)
                        ind_rev = list(ind)
                        ind_rev.reverse()
                        ind_rev = tuple(ind_rev) + (jj,)
                        out[field][ind_rev] = record[field][ind]
                        it.iternext()
        return out, out_cal


def _data_attrs_from_acq_attrs_acq1(acq_attrs):
    # The frequency axis.  In MHz.
    samp_freq = float(acq_attrs["system_sampling_frequency"]) / 1e6
    nfreq = int(acq_attrs["n_freq"])
    freq_width = samp_freq / 2 / nfreq
    freq_width_array = np.empty((nfreq,), dtype=np.float64)
    freq_width_array[:] = freq_width
    freq_centre = (
        samp_freq - np.cumsum(freq_width_array) + freq_width
    )  # This offset gives the correct channels
    freq = np.empty(nfreq, dtype=[("centre", np.float64), ("width", np.float64)])
    freq["centre"] = freq_centre
    freq["width"] = freq_width
    # The product axis.
    prod_channels = acq_attrs["chan_indices"]
    nprod = len(prod_channels)
    prod = np.empty(nprod, dtype=[("input_a", np.int64), ("input_b", np.int64)])
    # This raises a warning for some data, where the col names aren't exactly
    # 'input_a' and 'input_b'.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prod[:] = prod_channels
    for ii in range(nprod):
        prod[ii][0] = prod_channels[ii][0]
        prod[ii][1] = prod_channels[ii][1]
    # Populate the output.
    out = {}
    out["freq"] = freq
    out["prod"] = prod
    return out


def _get_index_map_from_acq1(acq_files, time_sel, prod_sel, freq_sel):
    data_headers = []
    for acq_file in acq_files:
        data_headers.append(_get_header_info_acq1(acq_file))
    time, prod, freq, tmp_dsets = _resolve_header_info_acq1(data_headers)
    # Populate output.
    out = {}
    out["time"] = time[time_sel[0] : time_sel[1]]
    out["prod"] = prod[prod_sel]
    out["freq"] = freq[freq_sel]
    return out


def andata_from_acq1(acq_files, start, stop, prod_sel, freq_sel, datasets, out_group):
    # First open all the files and collect necessary data for all of them.
    dtypes = _check_files_acq1(acq_files)
    # Figure how much of the total data to read.
    start, stop = _get_files_frames_acq1(acq_files, start, stop)
    # Initialize the output.
    data = CorrData(out_group)
    # Assume all meta-data are the same as in the first file and copy it
    # over.
    acq = acq_files[0]
    data.add_history("acq", memh5.attrs2dict(acq.attrs))
    data.history["acq"]["archive_version"] = "1.0.0"
    # Copy data attribute axis info.
    index_map = _get_index_map_from_acq1(acq_files, (start, stop), prod_sel, freq_sel)
    for axis_name, axis_values in index_map.items():
        data.create_index_map(axis_name, axis_values)
    # Set file format attributes.
    data.attrs["instrument_name"] = (
        "UNKNOWN"
        if "instrument_name" not in acq.attrs
        else acq.attrs["instrument_name"]
    )
    data.attrs["acquisition_name"] = "UNKNOWN"
    data.attrs["acquisition_type"] = "corr"
    # Copy over the cal information if there is any.
    if "cal" in acq:
        memh5.deep_group_copy(
            acq["cal"],
            data._data["cal"],
            convert_attribute_strings=CorrData.convert_attribute_strings,
            convert_dataset_strings=CorrData.convert_dataset_strings,
        )
    # Now copy the datasets.
    if datasets is None:
        datasets = list(dtypes.keys())
    # Start with the visibilities.
    vis_shape = ()
    for dataset_name in dtypes.keys():
        if dataset_name not in datasets:
            continue
            # msg = "No dataset named %s in Acq files." % dataset_name
            # raise ValueError(msg)

        if dataset_name in ACQ_VIS_SHAPE_DATASETS:
            # These datasets must all be the same shape.
            if not vis_shape:
                vis_shape = dtypes[dataset_name].shape
            elif dtypes[dataset_name].shape != vis_shape or len(vis_shape) != 2:
                msg = (
                    "Expected the following datasets to be"
                    " identically shaped and 3D in Acq files: %s."
                    % str(ACQ_VIS_SHAPE_DATASETS)
                )
                raise ValueError(msg)
            _copy_dataset_acq1(
                dataset_name, acq_files, start, stop, data, prod_sel, freq_sel
            )
        else:
            _copy_dataset_acq1(dataset_name, acq_files, start, stop, data)
    return data


# IO for archive format 2.0


def andata_from_archive2(
    cls,
    acq_files,
    start,
    stop,
    stack_sel,
    prod_sel,
    input_sel,
    freq_sel,
    datasets,
    out_group,
):

    # XXX For short term force to CorrData class.  Will be fixed once archive
    # files carry 'acquisition_type' attribute.
    # andata_objs = [ cls(d) for d in acq_files ]
    andata_objs = [CorrData(d) for d in acq_files]

    # Resolve input and prod maps
    first_imap = andata_objs[0].index_map
    first_rmap = andata_objs[0].reverse_map

    # Cannot use input/prod sel for stacked data
    if "stack" in first_imap:
        if input_sel:
            raise ValueError("Cannot give input_sel for a stacked dataset.")
        if prod_sel:
            raise ValueError("Cannot give prod_sel for a stacked dataset.")

    prod_map = first_imap["prod"][:].view(np.ndarray).copy()
    input_map = first_imap["input"][:].view(np.ndarray).copy()
    input_map = memh5.ensure_unicode(input_map)  # Convert string entries to unicode
    if "stack" in first_imap:
        stack_map = first_imap["stack"][:].view(np.ndarray).copy()
        stack_rmap = first_rmap["stack"][:].view(np.ndarray).copy()
    else:
        # Unstacked so the stack and prod axes are essentially the same.
        nprod = len(prod_map)
        stack_map = np.empty(nprod, dtype=[("prod", "<u4"), ("conjugate", "u1")])
        stack_map["conjugate"][:] = 0
        stack_map["prod"] = np.arange(nprod)
        stack_rmap = np.empty(nprod, dtype=[("stack", "<u4"), ("conjugate", "u1")])
        stack_rmap["conjugate"][:] = 0
        stack_rmap["stack"] = np.arange(nprod)
        # Efficiently slice prod axis, not stack axis.
        if stack_sel is not None:
            prod_sel = stack_sel
            stack_sel = None

    (
        stack_sel,
        stack_map,
        stack_rmap,
        prod_sel,
        prod_map,
        input_sel,
        input_map,
    ) = _resolve_stack_prod_input_sel(
        stack_sel, stack_map, stack_rmap, prod_sel, prod_map, input_sel, input_map
    )

    # Define dataset filter to convert vis datatype.
    def dset_filter(dataset, time_sel=None):
        # For compatibility with older caput.
        if time_sel is None:
            time_sel = slice(None)
        # A lot of the logic here is that h5py can only deal with one
        # *fancy* slice (that is 1 axis where the slice is an array).
        # Note that *time_sel* is always a normal slice, so don't have to worry
        # about it as much.
        attrs = getattr(dataset, "attrs", {})
        name = path.split(dataset.name)[-1]
        # Special treatement for pure sub-array dtypes, which get
        # modified by numpy to add dimensions when read.
        dtype = dataset.dtype
        if dtype.kind == "V" and not dtype.fields and dtype.shape:
            field_name = str(name.split("/")[-1])
            dtype = np.dtype([(field_name, dtype)])
            shape = dataset.shape
            # The datasets this effects are tiny, so just read them in.
            dataset = dataset[:].view(dtype)
            dataset.shape = shape

        axis = attrs["axis"]
        if axis[0] == "freq" and axis[1] in ("stack", "prod", "input"):
            # For large datasets, take great pains to down-select as
            # efficiently as possible.
            if axis[1] == "stack":
                msel = stack_sel
            elif axis[1] == "prod":
                msel = prod_sel
            else:
                msel = input_sel
            if isinstance(msel, np.ndarray) and isinstance(freq_sel, np.ndarray):
                nfsel = np.sum(freq_sel) if freq_sel.dtype == bool else len(freq_sel)
                npsel = np.sum(msel) if msel.dtype == bool else len(msel)
                nfreq = len(andata_objs[0].index_map["freq"])
                nprod = len(andata_objs[0].index_map["prod"])
                frac_fsel = float(nfsel) / nfreq
                frac_psel = float(npsel) / nprod

                if frac_psel < frac_fsel:
                    dataset = dataset[:, msel, time_sel][freq_sel, :, :]
                else:
                    dataset = dataset[freq_sel, :, time_sel][:, msel, :]
            else:
                # At least one of *msel* and *freq_sel* is an
                # integer or slice object and h5py can do the full read
                # efficiently.
                dataset = dataset[freq_sel, msel, time_sel]
        else:
            # Dynamically figure out the axis ordering.
            axis = memh5.bytes_to_unicode(attrs["axis"])
            ndim = len(dataset.shape)  # h5py datasets don't have ndim.
            if ("freq" in axis and isinstance(freq_sel, np.ndarray)) + (
                "stack" in axis and isinstance(stack_sel, np.ndarray)
            ) + ("prod" in axis and isinstance(prod_sel, np.ndarray)) + (
                "input" in axis and isinstance(input_sel, np.ndarray)
            ) > 1:
                # At least two array slices. Incrementally down select.
                # First freq.
                dataset_sel = [slice(None)] * ndim
                for ii in range(ndim):
                    if axis[ii] == "freq":
                        dataset_sel[ii] = freq_sel
                # Assume the time is the fastest varying index
                # and down select here.
                dataset_sel[-1] = time_sel
                dataset = dataset[tuple(dataset_sel)]
                # And again for stack.
                dataset_sel = [slice(None)] * ndim
                for ii in range(ndim):
                    if attrs["axis"][ii] == "stack":
                        dataset_sel[ii] = stack_sel
                dataset = dataset[tuple(dataset_sel)]
                # And again for prod.
                dataset_sel = [slice(None)] * ndim
                for ii in range(ndim):
                    if axis[ii] == "prod":
                        dataset_sel[ii] = prod_sel
                dataset = dataset[tuple(dataset_sel)]
                # And again for input.
                dataset_sel = [slice(None)] * ndim
                for ii in range(ndim):
                    if axis[ii] == "input":
                        dataset_sel[ii] = input_sel
                dataset = dataset[tuple(dataset_sel)]
            else:
                dataset_sel = [slice(None)] * ndim
                for ii in range(ndim):
                    if axis[ii] == "freq":
                        dataset_sel[ii] = freq_sel
                    elif axis[ii] == "stack":
                        dataset_sel[ii] = stack_sel
                    elif axis[ii] == "prod":
                        dataset_sel[ii] = prod_sel
                    elif axis[ii] == "input":
                        dataset_sel[ii] = input_sel
                    elif axis[ii] in CONCATENATION_AXES:
                        dataset_sel[ii] = time_sel
                dataset = dataset[tuple(dataset_sel)]

        # Change data type for the visibilities, if necessary.
        if re.match(ACQ_VIS_DATASETS, name) and dtype != np.complex64:
            data = dataset[:]
            dataset = np.empty(dataset.shape, dtype=np.complex64)
            dataset.real = data["r"]
            dataset.imag = data["i"]

        # Convert any string types to unicode. At the moment this should only effect
        # the dataset_id dataset
        dataset = memh5.ensure_unicode(dataset)

        return dataset

    # The actual read, file by file.
    data = concatenate(
        andata_objs,
        out_group=out_group,
        start=start,
        stop=stop,
        datasets=datasets,
        dataset_filter=dset_filter,
        convert_attribute_strings=cls.convert_attribute_strings,
        convert_dataset_strings=cls.convert_dataset_strings,
    )

    # Andata (or memh5) should already do the right thing.
    # Explicitly close up files
    # for ad in andata_objs:
    #     ad.close()

    # Rejig the index map according to prod_sel and freq_sel.
    # Need to use numpy arrays to avoid weird cyclic reference issues.
    # (https://github.com/numpy/numpy/issues/1601)
    fmap = data.index_map["freq"][freq_sel].view(np.ndarray).copy()
    # pmap = data.index_map['prod'][prod_sel].view(np.ndarray).copy()
    # imap = data.index_map['input'][input_sel].view(np.ndarray).copy()
    data.create_index_map("freq", fmap)
    data.create_index_map("stack", stack_map)
    data.create_reverse_map("stack", stack_rmap)
    data.create_index_map("prod", prod_map)
    data.create_index_map("input", input_map)
    return data, input_sel


# Routines for re-mapping the index_map/input to match up the order that is
# in the files, and the layout database


def _generate_input_map(serials, chans=None):
    # Generate an input map in the correct format. If chans is None, just
    # number from 0 upwards, otherwise use the channel numbers specified.

    # Define datatype of input map array
    # TODO: Python 3 string issues
    _imap_dtype = [
        ("chan_id", np.int64),
        ("correlator_input", "U32"),
    ]

    # Add in channel numbers correctly
    if chans is None:
        chan_iter = enumerate(serials)
    else:
        chan_iter = list(zip(chans, serials))

    imap = np.array(list(chan_iter), dtype=_imap_dtype)

    return imap


def _get_versiontuple(afile):
    if "acq" in afile.history:
        archive_version = afile.history["acq"]["archive_version"]
    else:
        archive_version = afile.attrs["archive_version"]

    archive_version = memh5.bytes_to_unicode(archive_version)

    return versiontuple(archive_version)


def _remap_stone_abbot(afile):
    # Generate an index_map/input for the old stone/abbot files

    # Really old files do not have an adc_serial attribute
    if "adc_serial" not in afile.history["acq"]:
        warnings.warn("Super old file. Cannot tell difference between stone and abbot.")
        serial = -1
    else:
        # Fetch and parse serial value
        serial = int(afile.history["acq"]["adc_serial"])

    # The serials are defined oddly in the files, use a dict to look them up
    serial_map = {1: "0003", 33: "0033", -1: "????"}  # Stone  # Abbot  # Unknown

    # Construct new array of index_map
    serial_pat = "29821-0000-%s-C%%i" % serial_map[serial]
    inputmap = _generate_input_map([serial_pat % ci for ci in range(8)])

    # Copy out old index_map/input if it exists
    if "input" in afile.index_map:
        afile.create_index_map("input_orig", np.array(afile.index_map["input"]))
        # del afile._data['index_map']._dict['input']
        afile.del_index_map("input")

    # Create new index map
    afile.create_index_map("input", inputmap)

    return afile


def _remap_blanchard(afile):
    # Remap a blanchard correlator file

    BPC_END = (
        1410586200.0  # 2014/09/13 05:30 UTC ~ when blanchard was moved into the crate
    )
    last_time = afile.time[-1]

    # Use time to check if blanchard was in the crate or not
    if last_time < BPC_END:

        # Find list of channels and adc serial using different methods depending on the archive file version
        if _get_versiontuple(afile) < versiontuple("2.0.0"):
            # The older files have no index_map/input so we need to guess/construct it.
            chanlist = list(range(16))
            adc_serial = afile.history["acq"]["adc_serial"][0]

        else:
            # The newer archive files have the index map, and so we can just parse this
            chanlist = afile.index_map["input"]["chan"]
            adc_serial = afile.index_map["input"]["adc_serial"][0]

        # Construct new array of index_map
        serial_pat = "29821-0000-%s-C%%02i" % adc_serial
        inputmap = _generate_input_map([serial_pat % ci for ci in chanlist])

    else:
        _remap_crate_corr(afile, 0)
        return afile

    # Copy out old index_map/input if it exists
    if "input" in afile.index_map:
        afile.create_index_map("input_orig", np.array(afile.index_map["input"]))
        # del afile._data['index_map']._dict['input']
        afile.del_index_map("input")

    # Create new index map
    afile.create_index_map("input", inputmap)

    return afile


def _remap_first9ucrate(afile):
    # Remap a first9ucrate file
    if _get_versiontuple(afile) < versiontuple("2.0.0"):
        warnings.warn("Remapping old format first9ucrate files is not supported.")
        return afile

    # Remap ignoring the fact that there was firt9ucrate data in the old format
    _remap_crate_corr(afile, 15)

    return afile


def _remap_slotX(afile):
    # Remap a slotXX correlator file

    # Figure out the slot number
    inst_name = afile.attrs["instrument_name"]
    slotnum = int(inst_name[4:])

    _remap_crate_corr(afile, slotnum)

    return afile


def _remap_crate_corr(afile, slot):
    # Worker routine for remapping the new style files for blanchard, first9ucrate and slotX

    if _get_versiontuple(afile) < versiontuple("2.0.0"):
        raise Exception("Only functions with archive 2.0.0 files.")

    CRATE_CHANGE = 1412640000.0  # The crate serial changed over for layout 60
    last_time = afile.time[-1]

    if last_time < CRATE_CHANGE:
        crate_serial = "K7BP16-0002"
    else:
        crate_serial = "K7BP16-0004"

    # Fetch and remap the channel list
    chanlist = afile.index_map["input"]["chan"]
    channel_remapping = np.array(
        [12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]
    )  # Channel order in new scheme
    chanlist = channel_remapping[chanlist]

    # The slot remapping function (i.e. C(c) from doclib/165/channel_standards)
    slot_remapping = [
        80,
        16,
        64,
        0,
        208,
        144,
        192,
        128,
        240,
        176,
        224,
        160,
        112,
        48,
        96,
        32,
    ]

    # Create new list of serials
    serial_pat = crate_serial + ("%02i%%02i" % int(slot))
    serials = [serial_pat % ci for ci in chanlist]

    # Create a list of channel ids (taking into account that they are
    # meaningless for the old crate)
    if last_time >= CRATE_CHANGE:
        chans = [slot_remapping[slot - 1] + ci for ci in chanlist]
    else:
        chans = chanlist

    inputmap = _generate_input_map(serials, chans)

    # Save and remove old index map
    afile.create_index_map("input_orig", np.array(afile.index_map["input"]))
    afile.del_index_map("input")

    # Create new index map
    afile.create_index_map("input", inputmap)

    return afile


def _remap_inputs(afile):
    # Master routine for remapping inputs. This tries to figure out which
    # instrument took the data, and then dispatch to the right routine to
    # generate the new index_map/input. This follows the logic in doclib:165

    # Eventually the change will be made in the correlator software and we
    # can stop remapping files after that time.

    # NOTE: need to be careful where you use afile.attrs versus
    # afile.history['acq'] for getting properties

    last_time = afile.time[-1]
    SA_END = 1397088000.0  # 2014/04/10 ~ last time stone and abbot were working

    # h5py should return a byte string for the attribute and so we need to decode
    # it
    inst_name = memh5.bytes_to_unicode(afile.attrs.get("instrument_name", b""))
    num_antenna = int(afile.history.get("acq", {}).get("n_antenna", "-1"))

    # Test if is abbot or stone
    if last_time < SA_END and num_antenna == 8:
        # Relies upon old files having the acq history
        _remap_stone_abbot(afile)

    elif inst_name == "blanchard":
        _remap_blanchard(afile)

    elif inst_name == "first9ucrate":
        _remap_first9ucrate(afile)

    elif inst_name[:4] == "slot":
        _remap_slotX(afile)

    else:
        warnings.warn("I don't know what this data is.")


def _insert_gains(data, input_sel):
    # Construct a full dataset for the gains and insert it into the CorrData
    # object
    # freq_sel is needed for selecting the relevant frequencies in old data

    # Input_sel is only used for pre archive_version 2.2, where there is no way
    # to know which header items to pull out.

    # For old versions the gains are stored in the attributes and need to be
    # extracted
    if ("archive_version" not in data.attrs) or versiontuple(
        memh5.bytes_to_unicode(data.attrs["archive_version"])
    ) < versiontuple("2.2.0"):

        # Hack to find the indices of the frequencies in the file
        fc = data.index_map["freq"]["centre"]
        fr = np.linspace(
            800, 400.0, 1024, endpoint=False
        )  # The should be the frequency channel

        # Compare with a tolerance (< 1e-4). Broken out into loop so we can deal
        # with the case where there are no matches
        fsel = []
        for freq in fc:
            fi = np.argwhere(np.abs(fr - freq) < 1e-4)

            if len(fi) == 1:
                fsel.append(fi[0, 0])

        # Initialise gains to one by default
        gain = np.ones((data.nfreq, data.ninput), dtype=np.complex64)

        try:
            ninput_orig = data.attrs["number_of_antennas"]
        except KeyError:
            ninput_orig = data.history["acq"]["number_of_antennas"]

        # In certain files this entry is a length-1 array, turn it into a scalar if it is not
        if isinstance(ninput_orig, np.ndarray):
            ninput_orig = ninput_orig[0]

        if ninput_orig <= 16:
            # For 16 channel or earlier data, each channel has a simple
            # labelling for its gains
            keylist = [
                (channel, "antenna_scaler_gain" + str(channel))
                for channel in range(ninput_orig)
            ]
        else:
            # For 256 channel data this is more complicated

            # Construct list of keys for all gain entries
            keylist = [key for key in data.attrs.keys() if key[:2] == "ID"]

            # Extract the channel id from each key
            chanid = [key.split("_")[1] for key in keylist]

            # Sort the keylist according to the channel ids, as the inputs
            # should be sorted by channel id.
            keylist = sorted(zip(chanid, keylist))
        # Down select keylist based on input_sel.
        input_sel_list = list(np.arange(ninput_orig, dtype=int)[input_sel])
        keylist = [keylist[ii] for ii in input_sel_list]

        if len(fsel) != data.nfreq:
            warnings.warn(
                "Could not match all frequency channels. Skipping gain calculation."
            )
        else:
            # Iterate over the keys and extract the gains
            for chan, key in keylist:

                # Try and find gain entry
                if key in data.attrs:
                    g_data = data.attrs[key]
                elif key in data.history["acq"]:
                    g_data = data.history["acq"][key]
                else:
                    warnings.warn(
                        "Cannot find gain entry [%s] for channel %i" % (key, chan)
                    )
                    continue

                # Unpack the gain values and construct the gain array
                g_real, g_imag = g_data[1:-1:2], g_data[2:-1:2]
                g_exp = g_data[-1]

                g_full = (g_real + 1.0j * g_imag) * 2**g_exp

                # Select frequencies that are loaded from the file
                g_sel = g_full[fsel]

                gain[:, input_sel_list.index(chan)] = g_sel

        # Gain array must be specified for all times, repeat along the time axis
        gain = np.tile(gain[:, :, np.newaxis], (1, 1, data.ntime))

    else:

        gain = np.ones((data.nfreq, data.ninput, data.ntime), dtype=np.complex64)

        # Check that the gain datasets have been loaded
        if ("gain_coeff" not in data.datasets) or ("gain_exp" not in data.datasets):
            warnings.warn(
                "Required gain datasets not loaded from file (> v2.2.0), using unit gains."
            )

        else:
            # Extract the gain datasets from the file
            gain_exp = data.datasets["gain_exp"][:]
            gain_coeff = data.datasets["gain_coeff"][:]

            # Turn into a single array
            if gain_coeff.dtype == np.complex64:
                gain *= gain_coeff
            else:
                gain.real[:] = gain_coeff["r"]
                gain.imag[:] = gain_coeff["i"]
            gain *= 2 ** gain_exp[np.newaxis, :, :]

    # Add gain dataset to object, and create axis attribute
    gain_dset = data.create_dataset("gain", data=gain)
    gain_dset.attrs["axis"] = np.array(["freq", "input", "time"])


def _read_axis_sel(cls, d, ax, ax_sel):
    """Read from a Group, making selections along a given axis."""
    # create selections dict
    def walk_tree(g):
        sel = {}
        for key in g:
            if isinstance(g[key], (h5py.Group, memh5.MemGroup)):
                sel.update(walk_tree(g[key]))
            axis = [
                a.decode() if isinstance(a, bytes) else a
                for a in g[key].attrs.get("axis", [])
            ]
            if ax in axis:
                ai = axis.index(ax)
                s = [slice(None)] * len(g[key].shape)
                s[ai] = ax_sel
                sel[key] = tuple(s)
        return sel

    sel = walk_tree(d)

    new = memh5.MemGroup()

    memh5.deep_group_copy(
        d,
        new,
        selections=sel,
        convert_dataset_strings=cls.convert_dataset_strings,
        convert_attribute_strings=cls.convert_attribute_strings,
    )

    return cls(new)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
