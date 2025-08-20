"""
Data Index Searcher for CHIME

Search routines for locating data withing the CHIME data index.

Data tables
===========

- :py:class:`DataFlag`
- :py:class:`DataFlagType`


Exceptions
==========

- :py:class:`DataFlagged`


High Level Index Searcher
=========================

- :py:class:`Finder`
- :py:class:`DataIntervalList`
- :py:class:`BaseDataInterval`
- :py:class:`CorrDataInterval`
- :py:class:`HKDataInterval`
- :py:class:`WeatherDataInterval`
- :py:class:`FlagInputDataInterval`
- :py:class:`CalibrationGainDataInterval`
- :py:class:`DigitalGainDataInterval`
- :py:class:`TimingDataInterval`


Routines
========

- :py:meth:`connect_database`
- :py:meth:`files_in_range`

"""

import logging
from os import path
import time
import socket
import peewee as pw
import tabulate

import caput.time as ctime

from ch_ephem.observers import chime

from chimedb.core.exceptions import CHIMEdbError
import chimedb.data_index as di
from chimedb.data_index.orm import file_info_table
from chimedb.dataflag import DataFlagType, DataFlag

from . import layout
from ._db_tables import connect_peewee_tables as connect_database
from .holography import HolographySource, HolographyObservation

# Module Constants
# ================

GF_REJECT = "gf_reject"
GF_RAISE = "gf_raise"
GF_WARN = "gf_warn"
GF_ACCEPT = "gf_accept"


# High level interface to the data index
# ======================================

# The following are the info tables that we use to join over when using the
# finder.
_acq_info_table = [di.CorrAcqInfo, di.HKAcqInfo, di.RawadcAcqInfo]


class Finder:
    """High level searching of the CHIME data index.

    This class gives a convenient way to search and filter data acquisitions
    as well as time ranges of data within acquisitions. Search results
    constitute a list of files within an acquisition as well as a time range for
    the data within these files. Convenient methods are provided for loading
    the precise time range of constituting a search result.

    This is intended to make the most common types of searches of CHIME data as
    convenient as possible.  However for very complex searches, it may be
    necessary to resort to the lower level interface.

    Searching the index
    ===================

    There are four ways that a search can be modified which may be combined in
    any way.

    #. You can restrict the types of acquisition that are under
       consideration, using methods whose names begin with ``only_``.
       In this way, one can consider only, say, housekeeping acquisitions.
    #. The second is to adjust the total time range under consideration.
       This is achieved by assigning to :attr:`~Finder.time_range` or calling
       methods beginning with ``set_time_range_``. The total time range affects
       acquisitions under consideration as well as the data time ranges within
       the acquisitions. Subsequent changes to the total time range under
       consideration may only become more restrictive.
    #. The data index may also be filtered by acquisition using methods whose
       names begin with ``filter_acqs``. Again subsequent filtering are always
       combined to become more restrictive.  The attribute :attr:`~Finder.acqs`
       lists the acquisitions currently included in the search for convenience
       when searching interactively.
    #. Time intervals within acquisitions are added using methods with names
       beginning with ``include_``.  Time intervals are defined in the
       :attr:`~Finder.time_intervals` attribute, and are inclusive (you can
       add as many as you want).
    #. Finally, upon calling :meth:``get_results`` or :meth:``get_results_acq``,
       one can pass an arbitrary condition on individual files, thereby
       returning only a subset of files from each acquisition.

    Getting results
    ===============

    Results of the search can be retrieved using methods whose names begin with
    ``get_results`` An individual search result is constituted of a list of file
    names and a time interval within these files. These can easily loaded into
    memory using helper functions (see :class:`BaseDataInterval` and
    :class:`DataIntervalList`).

    Parameters
    ----------
    acqs : list of :class:`chimedb.data_index.ArchiveAcq` objects
        Acquisitions to initially include in data search.  Default is to search
        all acquisitions.
    node_spoof : dictionary
        Normally, the DB will be queried to find which nodes are mounted on your
        host. If you are on a machine that is cross-mounted, though, you can
        enter a dictionary of "node_name": "mnt_root" pairs, specifying the
        nodes to search and where they are mounted on your host.

    Examples
    --------

    To find all the correlator data between two times.

    >>> from ch_util import finder
    >>> from datetime import datetime
    >>> f = finder.Finder()
    >>> f.only_corr()
    >>> f.set_time_range(datetime(2014,2,24), datetime(2014,2,25))
    >>> f.print_results_summary()
      #  acquisition                    start (s)    len (s)    files       MB
    ---  ---------------------------  -----------  ---------  -------  -------
      0  20140219T145849Z_abbot_corr       378053    86400         25  3166.08
      1  20140224T051212Z_stone_corr            0    67653.9       19  2406.78
    Total 154054 seconds,   5573 MB of data.

    Search for transits of a given source.

    >>> import ch_ephem.sources
    >>> f.include_transits(ch_ephem.sources.CasA, time_delta=3600)
    >>> f.print_results_summary()
      #  acquisition                    start (s)    len (s)    files       MB
    ---  ---------------------------  -----------  ---------  -------  -------
      0  20140219T145849Z_abbot_corr     452092         3600        2  253.286
      1  20140224T051212Z_stone_corr      55292.9       3600        2  253.346
    Total   7200 seconds,    507 MB of data.

    To read the data,

    >>> from ch_util import andata
    >>> results_list = f.get_results()
    >>> # Pick result number 1
    >>> result = results_list[0]
    >>> # Pick product number 0 (autocorrelation)
    >>> data = result.as_loaded_data(prod_sel=0)
    >>> print data.vis.shape
    (1024, 1, 360)

    More intricate filters on the acquisitions are possible.

    >>> import chimedb.data_index as di
    >>> f = finder.Finder()
    >>> # Find ALL 10ms cadence data correlated by 'stone' with 8 channels.
    >>> f.filter_acqs((di.CorrAcqInfo.integration < 0.011)
    ...               & (di.CorrAcqInfo.integration > 0.009)
    ...               & (di.CorrAcqInfo.nfreq == 1024)
    ...               & (di.CorrAcqInfo.nprod == 36)
    ...               & (di.ArchiveInst.name == 'stone'))
    >>> f.print_results_summary()
      #  acquisition                    start (s)    len (s)    files        MB
    ---  ---------------------------  -----------  ---------  -------  --------
      0  20140211T020307Z_stone_corr            0    391.764      108   13594.1
      1  20140128T135105Z_stone_corr            0   4165.22       104  131711
      2  20131208T070336Z_stone_corr            0   1429.78       377   47676.4
      3  20140212T014603Z_stone_corr            0   2424.43       660   83604.1
      4  20131210T060233Z_stone_corr            0   1875.32       511   64704.8
      5  20140210T021023Z_stone_corr            0    874.144      240   30286.4
    Total  11161 seconds, 371577 MB of data.


    Here is an example that uses node spoofing and also filters files within
    acquisitions to include only LNA housekeeping files:

    >>> f = finder.Finder(node_spoof={"scinet_hpss": "/dev/null"})
    >>> f.only_hk()
    >>> f.set_time_range(datetime(2014, 9, 1), datetime(2014, 10, 10))
    >>> f.print_results_summary()
      #  acquisition                start (s)           len (s)    files            MB
    ---  -----------------------  -----------  ----------------  -------  ------------
      0  20140830T005410Z_ben_hk       169549  419873                 47  2093
      1  20140905T203905Z_ben_hk            0   16969.1                2     0.0832596
      2  20140908T153116Z_ben_hk            0       1.11626e+06       56     4.45599
      3  20141009T222415Z_ben_hk            0    5744.8                2     0.191574
    Total 1558847 seconds,   2098 MB of data.
    >>> res = f.get_results(file_condition = (di.HKFileInfo.atmel_name == "LNA"))
    >>> for r in res:
    ...   print(f"No. files: {len(r[0])}")
    ...
    No. files: 8
    No. files: 1
    No. files: 19
    No. files: 1

    In the above example, the restriction to LNA housekeeping could also have
    been accomplished with the convenience method :meth:`Finder.set_hk_input`:

    >>> f.set_hk_input("LNA")
    >>> res = f.get_results()
    """

    # Constructors and setup
    # ----------------------

    def __init__(self, acqs=(), node_spoof=None):
        import copy

        # Which nodes do we have available?
        host = socket.gethostname().split(".")[0]
        self._my_node = []
        self._node_spoof = node_spoof

        connect_database()

        if not node_spoof:
            for n in (
                di.StorageNode.select()
                .where(di.StorageNode.host == host)
                .where(di.StorageNode.active)
            ):
                self._my_node.append(n)
        else:
            for key, val in node_spoof.items():
                self._my_node.append(di.StorageNode.get(name=key))

        if not len(self._my_node):
            raise RuntimeError(
                "No nodes found. Perhaps you need to pass a 'node_spoof' parameter?"
            )

        # Get list of join tables. We make a copy because the user may alter
        # this later through the only_XXX() methods.
        self._acq_info = copy.copy(_acq_info_table)
        self._file_info = copy.copy(file_info_table)

        if acqs:
            pass
        else:
            acqs = di.ArchiveAcq.select()
            for i in self._acq_info:
                acqs.join(i)
        self._acqs = list(acqs)
        self._time_range = (chime.lsd_start_day, time.time())
        self._time_intervals = None
        self._time_exclusions = []
        self._atmel_restrict = None
        self.min_interval = 240.0
        self._gf_mode = {"comment": GF_ACCEPT, "warning": GF_WARN, "severe": GF_REJECT}
        self._data_flag_types = []
        # The following line cuts any acquisitions with no files.
        # self.filter_acqs_by_files(True)
        # This is very similar to the above line, but takes ~.5s instead of
        # 12s.
        acq_ids = [acq.id for acq in self.acqs]
        if not acq_ids:
            # Nothing to do.
            return
        condition = (
            (di.ArchiveAcq.id << acq_ids)
            & (di.ArchiveFileCopy.node << self._my_node)
            & (di.ArchiveFileCopy.has_file == "Y")
        )
        selection = di.ArchiveAcq.select().join(di.ArchiveFile).join(di.ArchiveFileCopy)
        self._acqs = list(selection.where(condition).group_by(di.ArchiveAcq))

    @classmethod
    def offline(cls, acqs=()):
        """Initialize :class:`~Finder` when not working on a storage node.

        Normally only data that is available on the present host is searched,
        and as such :class:`~Finder` can't be used to browse the index when you
        don't have access to the acctual data. Initializing using this method
        spoofs the 'gong' and 'niedermayer' storage nodes (which should have a
        full copy of the archive) such that the data index can be search the
        full archive.

        """

        node_spoof = {}
        # for n in di.StorageNode.select():
        #    node_spoof[n.name] = ''
        # I think all the data live on at lease one of these -KM.
        node_spoof["gong"] = ""
        node_spoof["niedermayer"] = ""
        return cls(acqs, node_spoof=node_spoof)

    # Filters on the index
    # --------------------

    @property
    def acqs(self):
        """Acquisitions remaining in this search.

        Returns
        -------
        acqs : list of :class:`chimedb.data_index.ArchiveAcq` objects

        """

        return list(self._acqs)

    @property
    def time_range(self):
        """Time range to be included in search.

        Data files and acquisitions that do not overlap with this range are
        excluded. Assigning to this is equivalent to calling
        :meth:`~Finder.set_time_range`.

        Returns
        -------
        time_range : tuple of 2 floats
            Unix/POSIX beginning and end of the time range.

        """

        return self._time_range

    @property
    def time_intervals(self):
        """Periods in time to be included.

        Periods are combined with `OR` unless list is empty, in which case no
        filtering is performed.

        Returns
        -------
        time_intervals : list of pairs of floats
            Each entry is the Unix/POSIX beginning and end of the time interval
            to be included.

        """

        if self._time_intervals is None:
            return [self.time_range]

        return list(self._time_intervals)

    def _append_time_interval(self, interval):
        if self._time_intervals is None:
            time_intervals = []
        else:
            time_intervals = self._time_intervals
        time_intervals.append(interval)
        self._time_intervals = time_intervals

    @property
    def time_exclusions(self):
        """Periods in time to be excluded.

        Returns
        -------
        time_exclusions : list of pairs of floats
            Each entry is the Unix/POSIX beginning and end of the time interval
            to be excluded.

        """

        return list(self._time_exclusions)

    def _append_time_exclusion(self, interval):
        self._time_exclusions.append(interval)

    @property
    def min_interval(self):
        """Minimum length of a block of data to be considered.

        This can be set to any number.  The default is 240 seconds.

        Returns
        -------
        min_interval : float
            Length of time in seconds.

        """

        return self._min_interval

    @min_interval.setter
    def min_interval(self, value):
        self._min_interval = float(value)

    @property
    def global_flag_mode(self):
        """Global flag behaviour mode.

        Defines how global flags are treated when finding data. There are three
        severities of global flag: comment, warning, and severe.  There are
        four possible behaviours when a search result overlaps a global flag,
        represented by module constants:

        :GF_REJECT: Reject any data overlapping flag silently.
        :GF_RAISE: Raise an exception when retrieving data intervals.
        :GF_WARN: Send a warning when retrieving data intervals but proceed.
        :GF_ACCEPT: Accept the data silently, ignoring the flag.

        The behaviour for all three severities is represented by a dictionary.
        If no mode is set, then the default behaviour is
        `{'comment' : GF_ACCEPT, 'warning' : GF_WARN, 'severe' : GF_REJECT}`.

        This is modified using :meth:`Finder.update_global_flag_mode`.

        Returns
        -------
        global_flag_mode : dictionary with keys 'comment', 'warning', 'severe'.
            Specifies finder behaviour.

        """

        return dict(self._gf_mode)

    @property
    def data_flag_types(self):
        """Types of DataFlag to exclude from results."""
        return self._data_flag_types

    # Setting up filters on the data
    # ------------------------------

    def update_global_flag_mode(self, comment=None, warning=None, severe=None):
        """Update :attr:`Finder.global_flag_mode`, the global flag mode.

        Parameters
        ----------
        comment : One of *GF_REJECT*, *GF_RAISE*, *GF_WARN*, or *GF_ACCEPT*.
        warning : One of *GF_REJECT*, *GF_RAISE*, *GF_WARN*, or *GF_ACCEPT*.
        severe : One of *GF_REJECT*, *GF_RAISE*, *GF_WARN*, or *GF_ACCEPT*.

        """

        if comment:
            _validate_gf_value(comment)
            self._gf_mode["comment"] = comment
        if warning:
            _validate_gf_value(warning)
            self._gf_mode["warning"] = warning
        if severe:
            _validate_gf_value(severe)
            self._gf_mode["severe"] = severe

    def accept_all_global_flags(self):
        """Set global flag behaviour to accept all data."""

        self.update_global_flag_mode(
            comment=GF_ACCEPT, warning=GF_ACCEPT, severe=GF_ACCEPT
        )

    def only_corr(self):
        """Only include correlator acquisitions in this search."""
        self._acq_info = [di.CorrAcqInfo]
        self._file_info = [di.CorrFileInfo]
        self.filter_acqs(True)

    def only_hk(self):
        """Only include housekeeping acquisitions in this search."""
        self._acq_info = [di.HKAcqInfo]
        self._file_info = [di.HKFileInfo]
        self.filter_acqs(True)

    def only_rawadc(self):
        """Only include raw ADC acquisitions in this search."""
        self._acq_info = [di.RawadcAcqInfo]
        self._file_info = [di.RawadcFileInfo]
        self.filter_acqs(True)

    def only_hfb(self, compression=None):
        """Only include HFB acquisitions in this search.

        Parameters
        ----------
        compression : bool, optional
            If True or False, only select acqs with compressed/uncompressed
            files.  By default, this is None, and all acqs are selected.
        """
        self._acq_info = [di.HFBAcqInfo]
        self._file_info = [di.HFBFileInfo]
        if compression is True or compression is False:
            self.filter_acqs(di.HFBFileInfo.compressed == compression)
        else:
            self.filter_acqs(True)

    def only_weather(self):
        """Only include weather acquisitions in this search."""
        self._acq_info = []
        self._file_info = [di.WeatherFileInfo]
        self.filter_acqs(di.AcqType.name == "weather")

    def only_chime_weather(self):
        """Only include chime weather acquisitions in this search.
        This excludes the old format mingun-weather."""
        self._acq_info = []
        self._file_info = [di.WeatherFileInfo]
        self.filter_acqs(di.AcqType.name == "weather")
        self.filter_acqs(di.ArchiveInst.name == "chime")

    def only_hkp(self):
        """Only include Prometheus housekeeping data in this search"""
        self._acq_info = []
        self._file_info = [di.HKPFileInfo]
        self.filter_acqs(di.AcqType.name == "hkp")

    def only_digitalgain(self):
        """Only include digital gain data in this search"""
        self._acq_info = []
        self._file_info = [di.DigitalGainFileInfo]
        self.filter_acqs(di.AcqType.name == "digitalgain")

    def only_gain(self):
        """Only include calibration gain data in this search"""
        self._acq_info = []
        self._file_info = [di.CalibrationGainFileInfo]
        self.filter_acqs(di.AcqType.name == "gain")

    def only_timing(self):
        """Only include timing data in this search.

        **NB:** These are `chime_timing` files generated by the
        calibration broker, *not* `chimetiming_corr` files from
        the receiver.
        """
        self._acq_info = []
        self._file_info = [di.TimingCorrectionFileInfo]
        self.filter_acqs(di.AcqType.name == "timing")

    def only_flaginput(self):
        """Only include input flag data in this search"""
        self._acq_info = []
        self._file_info = [di.FlagInputFileInfo]
        self.filter_acqs(di.AcqType.name == "flaginput")

    def filter_acqs(self, condition):
        """Filter the acquisitions included in this search.

        Parameters
        ----------
        condition : :mod:`peewee` comparison
            Condition on any on :class:`chimedb.data_index.ArchiveAcq` or any
            class joined to :class:`chimedb.data_index.ArchiveAcq`: using the
            syntax from the :mod:`peewee` module [1]_.

        Examples
        --------

        >>> from ch_util import finder
        >>> import chimedb.data_index as di
        >>> f = finder.Finder()
        >>> f.filter_acqs(di.ArchiveInst.name == 'stone')
        >>> f.filter_acqs((di.AcqType == 'corr') & (di.CorrAcqInfo.nprod == 36))

        See Also
        --------

        :meth:`Finder.filter_acqs_by_files`


        References
        ----------

        .. [1] http://peewee.readthedocs.org/en/latest/peewee/querying.html

        """

        # Get the acquisitions currently included.
        acq_ids = [acq.id for acq in self.acqs]
        if not acq_ids:
            # Nothing to do.
            return
        # From these, only include those meeting the new condition.
        # XXX simpler?
        condition = (di.ArchiveAcq.id << acq_ids) & condition

        selection = di.ArchiveAcq.select().join(di.AcqType)
        for i in self._acq_info:
            selection = selection.switch(di.ArchiveAcq).join(i, pw.JOIN.LEFT_OUTER)
        selection = selection.switch(di.ArchiveAcq).join(di.ArchiveInst)
        self._acqs = list(selection.where(condition).group_by(di.ArchiveAcq))

    def filter_acqs_by_files(self, condition):
        """Filter the acquisitions by the properties of its files.

        Because each acquisition has many files, this filter should be
        significantly slower than :meth:`Finder.filter_acqs`.

        Parameters
        ----------
        condition : :mod:`peewee` comparison
            Condition on any on :class:`chimedb.data_index.ArchiveAcq`,
            :class:`chimedb.data_index.ArchiveFile` or any class joined to
            :class:`chimedb.data_index.ArchiveFile` using the syntax from the
            :mod:`peewee` module [2]_.

        See Also
        --------

        :meth:`Finder.filter_acqs`

        Examples
        --------

        References
        ----------

        .. [2] http://peewee.readthedocs.org/en/latest/peewee/querying.html

        """
        # Get the acquisitions currently included.
        acq_ids = [acq.id for acq in self.acqs]
        if not acq_ids:
            # Nothing to do.
            return
        condition = (
            (di.ArchiveAcq.id << acq_ids)
            & (di.ArchiveFileCopy.node << self._my_node)
            & (di.ArchiveFileCopy.has_file == "Y")
            & (condition)
        )
        selection = di.ArchiveAcq.select().join(di.ArchiveFile).join(di.ArchiveFileCopy)
        info_cond = False
        for i in self._file_info:
            selection = selection.switch(di.ArchiveFile).join(
                i, join_type=pw.JOIN.LEFT_OUTER
            )
            # The following ensures that at least _one_ of the info tables is
            # joined.
            info_cond |= ~(i.start_time >> None)
        self._acqs = list(
            selection.where(condition & info_cond).group_by(di.ArchiveAcq)
        )

    def set_time_range(self, start_time=None, end_time=None):
        """Restrict the time range of the search.

        This method updates the :attr:`~Index.time_range` property and also
        excludes any acquisitions that do not overlap with the new range. This
        method always narrows the time range under consideration, never expands
        it.

        Parameters
        ----------
        start_time : float or :class:`datetime.datetime`
            Unix/POSIX time or UTC start of desired time range. Optional.
        end_time : float or :class:`datetime.datetime`
            Unix/POSIX time or UTC end of desired time range. Optional.

        """
        # Update `self.time_range`.
        if start_time is None:
            start_time = 0
        if end_time is None:
            end_time = time.time()
        start_time = ctime.ensure_unix(start_time)
        end_time = ctime.ensure_unix(end_time)
        old_start_time, old_end_time = self.time_range
        start_time = max(start_time, old_start_time)
        end_time = min(end_time, old_end_time)
        if start_time >= end_time:
            msg = "No time spanned by search. start=%s, stop=%s"
            msg = msg % (start_time, end_time)
            raise ValueError(msg)

        # Delete any acquisitions that do not overlap with the new range.
        cond = True
        for i in self._file_info:
            cond &= (i.start_time >> None) | (
                (i.start_time < end_time) & (i.finish_time > start_time)
            )
        self.filter_acqs_by_files(cond)

        if self._time_intervals is not None:
            time_intervals = _trim_intervals_range(
                self.time_intervals, (start_time, end_time)
            )
            self._time_intervals = time_intervals
        time_exclusions = _trim_intervals_range(
            self.time_exclusions, (start_time, end_time)
        )
        self._time_exclusions = time_exclusions
        self._time_range = (start_time, end_time)

    def set_time_range_global_flag(self, flag):
        """Set time range to correspond to a global flag.

        Parameters
        ----------
        flag : integer or string
            Global flag ID or name, e.g. "run_pass1_a", or 11292.

        Notes
        -----

        Global flag ID numbers, names, and descriptions are listed at
        http://bao.phas.ubc.ca/layout/event.php?filt_event_type_id=7

        """

        start_time, end_time = _get_global_flag_times_by_name_event_id(flag)
        self.set_time_range(start_time, end_time)

    def set_time_range_season(self, year=None, season=None):
        """Set the time range by as specific part of a given year.

        NOT YET IMPLEMENTED

        Parameters
        ----------
        year : integer
            Calender year
        season : string
            Month name (3 letter abbreviations are acceptable) or one of
            'winter', 'spring', 'summer', or 'fall'.

        """
        raise NotImplementedError()

    def _format_time_interval(self, start_time, end_time):
        if start_time is None:
            start_time = 0
        if end_time is None:
            end_time = time.time()
        start_time = ctime.ensure_unix(start_time)
        end_time = ctime.ensure_unix(end_time)
        range_start, range_end = self.time_range
        start_time = max(start_time, range_start)
        end_time = min(end_time, range_end)
        if start_time < end_time:
            return (start_time, end_time)

        return None

    def include_time_interval(self, start_time, end_time):
        """Include a time interval.

        Examples
        --------

        First a certain layout is chosen

        >>> from ch_util import finder
        >>> f = finder.Finder()
        >>> f.set_time_range_layout(26)
        >>> f.print_results_summary()
        interval | acquisition | offset from start (s) | length (s) | N files
           1  |  20140311T192616Z_abbot_corr  |    16412.8  |      667.1  |  1
           2  |  20140312T001123Z_abbot_corr  |        0.0  |     1150.5  |  314
           3  |  20140312T003054Z_abbot_corr  |        0.0  |    79889.4  |  23
           4  |  20140312T224940Z_abbot_corr  |        0.0  |      591.0  |  4
           5  |  20140312T230108Z_abbot_corr  |        0.0  |   171909.0  |  48
           6  |  20140315T014330Z_abbot_corr  |        0.0  |    35119.7  |  10
           7  |  20140318T154959Z_abbot_corr  |        0.0  |    51739.6  |  15
           8  |  20140320T120437Z_abbot_corr  |        0.0  |   186688.6  |  52
           9  |  20140325T174231Z_abbot_corr  |        0.0  |    86019.3  |  24
          10  |  20140326T175440Z_abbot_corr  |        0.0  |   286487.7  |  80
          11  |  20140330T064125Z_abbot_corr  |        0.0  |     2998.6  |  1590
          12  |  20140330T102505Z_abbot_corr  |        0.0  |   626385.0  |  174
          13  |  20140403T000057Z_blanchard_corr  |        0.0  |    54912.3  |  16
          14  |  20140403T152314Z_blanchard_corr  |        0.0  |   340637.8  |  94
          15  |  20140408T222844Z_abbot_corr  |        0.0  |    75589.3  |  21
          16  |  20140409T184530Z_blanchard_corr  |        0.0  |     3795.0  |  2
          17  |  20140410T003326Z_blanchard_corr  |        0.0  |     2173.7  |  72
          18  |  20140409T165603Z_blanchard_corr  |        0.0  |     4952.7  |  2
        Total 2011706.304970 seconds of data.

        To find a specific day in that layout choose the functionality
        include_time_interval

        >>> from datetime import datetime
        >>> f.include_time_interval(datetime(2014,04,8), datetime(2014,04,9))
        >>> f.print_results_summary()
        interval | acquisition | offset from start (s) | length (s) | N files
           1  |  20140408T222844Z_abbot_corr  |        0.0  |     5465.1  |  2
        Total 5465.059670 seconds of data.

        """

        interval = self._format_time_interval(start_time, end_time)
        if interval:
            self._append_time_interval(interval)

    def exclude_time_interval(self, start_time, end_time):
        """Exclude a time interval.

        Examples
        --------

        >>> from ch_util import finder
        >>> from datetime import datetime
        >>> f = finder.Finder()
        >>> f.set_time_range(datetime(2014,04,04), datetime(2014,04,14))
        >>> # f.print_results_summary() will show all the files in this time range
        >>> # Now want to exclude all data from 04, 10 to 04, 11
        >>> f.exclude_time_interval(datetime(2014,04,10),datetime(2014,04,11))
        >>> f.print_results_summary()
        interval | acquisition | offset from start (s) | length (s) | N files
           1  |  20140330T102505Z_abbot_corr  |   394484.2  |   231900.8  |  65
           2  |  20140403T152314Z_blanchard_corr  |    30988.4  |   309649.3  |  86
           3  |  20140408T222844Z_abbot_corr  |        0.0  |    75589.3  |  21
           4  |  20140409T184530Z_blanchard_corr  |        0.0  |     3795.0  |  2
           5  |  20140409T165603Z_blanchard_corr  |        0.0  |     4952.7  |  2
           6  |  20140411T003404Z_blanchard_corr  |        0.0  |   161606.5  |  45
           7  |  20140411T000920Z_blanchard_corr  |        0.0  |     1080.4  |  36
           8  |  20140413T002319Z_blanchard_corr  |        0.0  |    84981.7  |  24
        Total 873555.739000 seconds of data.
        """

        interval = self._format_time_interval(start_time, end_time)
        if interval:
            self._append_time_exclusion(interval)

    def include_global_flag(self, flag):
        """Update :attr:`time_intervals` to include a global flag.

        Parameters
        ----------
        flag : integer or string
            Global flag ID or name, e.g. "run_pass1_a", or 11292.

        Notes
        -----

        Global flag ID numbers, names, and descriptions are listed at
        http://bao.phas.ubc.ca/layout/event.php?filt_event_type_id=7

        """

        start_time, end_time = _get_global_flag_times_by_name_event_id(flag)
        self.include_time_interval(start_time, end_time)

    def exclude_global_flag(self, flag):
        """Update :attr:`time_intervals` to exclude a global flag.

        Parameters
        ----------
        flag : integer or string
            Global flag ID or name, e.g. "run_pass1_a", or 65.

        See Also
        --------

        Look under :meth:`include_global_flag` for a very similar example.

        Notes
        -----

        Global flag ID numbers, names, and descriptions are listed at
        http://bao.phas.ubc.ca/layout/event.php?filt_event_type_id=7

        """

        start_time, end_time = _get_global_flag_times_by_name_event_id(flag)
        self.exclude_time_interval(start_time, end_time)

    def exclude_data_flag_type(self, flag_type):
        """Exclude times that overlap with DataFlags of this type.

        Parameters
        ----------
        flag_type : string or list of string
            Name of DataFlagType(s) to exclude from results, e.g. "rain".
        """

        if isinstance(flag_type, list):
            self.data_flag_types.extend(flag_type)
        else:
            self.data_flag_types.append(flag_type)

    def include_RA_interval(self, start_RA, end_RA):
        """Add time intervals to include passings of given right RA intervals

        Parameters
        ----------
        start_RA : float
            Starting right ascension in degrees.
        end_RA : float
            Ending right ascension in degrees.

        Examples
        --------
        >>> from ch_util import finder
        >>> from datetime import datetime
        >>> f = finder.Finder()
        >>> f.set_time_range(datetime(2014,04,04), datetime(2014,04,14))
        >>> f.include_RA_interval(90., 180.)
        >>> f.print_results_summary()
        interval | acquisition | offset from start (s) | length (s) | N files
           1  |  20140330T102505Z_abbot_corr  |   398689.9  |    21541.0  |  7
           2  |  20140330T102505Z_abbot_corr  |   484854.0  |    21541.0  |  7
           3  |  20140330T102505Z_abbot_corr  |   571018.1  |    21541.0  |  7
           4  |  20140403T152314Z_blanchard_corr  |    35194.1  |    21541.0  |  7
           5  |  20140403T152314Z_blanchard_corr  |   121358.2  |    21541.0  |  7
           6  |  20140403T152314Z_blanchard_corr  |   207522.3  |    21541.0  |  7
           7  |  20140403T152314Z_blanchard_corr  |   293686.4  |    21541.0  |  6
           8  |  20140408T222844Z_abbot_corr  |     8491.2  |    21541.0  |  7
           9  |  20140410T003326Z_blanchard_corr  |      754.5  |     1419.2  |  48
          10  |  20140410T031023Z_blanchard_corr  |        0.0  |     1376.5  |  46
          11  |  20140410T014136Z_blanchard_corr  |        0.0  |     2347.4  |  78
          12  |  20140411T003404Z_blanchard_corr  |      397.4  |    21541.0  |  7
          13  |  20140411T003404Z_blanchard_corr  |    86561.5  |    21541.0  |  7
          14  |  20140413T002319Z_blanchard_corr  |      664.1  |    21541.0  |  7
        Total 242094.394565 seconds of data.
        """

        delta_RA = (end_RA - start_RA) % 360
        mid_RA = (start_RA + delta_RA / 2.0) % 360
        time_delta = delta_RA * 4 * 60.0 * ctime.SIDEREAL_S
        self.include_transits(mid_RA, time_delta=time_delta)

    def exclude_RA_interval(self, start_RA, end_RA):
        """Add time intervals to exclude passings of given right RA
        intervals

        Parameters
        ----------
        start_RA : float
            Starting right ascension in degrees.
        end_RA : float
            Ending right ascension in degrees.

        Examples
        --------
        Look under include_RA_interval for very similar example.

        """

        delta_RA = (end_RA - start_RA) % 360
        mid_RA = (start_RA + delta_RA / 2.0) % 360
        time_delta = delta_RA * 4 * 60.0 * ctime.SIDEREAL_S
        self.exclude_transits(mid_RA, time_delta=time_delta)

    def include_transits(self, body, time_delta=None):
        """Add time intervals to include transits for given celestial body.

        Parameters
        ----------
        body : :class:`ephem.Body` or float
            Transiting celestial body.  If a float, interpret as a right
            ascension in degrees.
        time_delta : float
            Total amount of time to include surrounding the transit in
            seconds. Default is to use twice the value of
            :attr:`~Finder.min_interval`.

        Examples
        --------
        >>> from ch_util import finder
        >>> import ch_ephem.sources
        >>> from datetime import datetime
        >>> f = finder.Finder()
        >>> f.set_time_range(datetime(2014,02,20), datetime(2014,02,22))
        >>> f.include_transits(ch_ephem.sources.CasA, time_delta=3600)
        >>> f.print_results_summary()
        interval | acquisition | offset from start (s) | length (s) | N files
           1  |  20140219T145849Z_abbot_corr  |   107430.9  |     3600.0  |  2
           2  |  20140219T145849Z_abbot_corr  |   193595.0  |     3600.0  |  2
           3  |  20140220T213252Z_stone_corr  |        0.0  |      990.2  |  1
           4  |  20140220T213252Z_stone_corr  |    83554.3  |     3600.0  |  2
        Total 11790.181012 seconds of data.

        """

        if not time_delta:
            time_delta = self.min_interval * 2
        ttimes = chime.transit_times(body, *self.time_range)
        for ttime in ttimes:
            self.include_time_interval(
                ttime - time_delta / 2.0, ttime + time_delta / 2.0
            )

    def include_26m_obs(self, source, require_quality=True):
        """Add time intervals to include 26m observations of a source.

        Parameters
        ----------
        source : string
            Source observed. Has to match name on database exactly.
        require_quality : bool (default: True)
            Require the quality flag to be zero (ie that the 26 m
            pointing is trustworthy) or None

        Examples
        --------
        >>> from ch_util import finder
        >>> from datetime import datetime
        >>> f = finder.Finder()
        >>> f.only_corr()
        >>> f.set_time_range(datetime(2017,8,1,10), datetime(2017,8,2))
        >>> f.filter_acqs((di.ArchiveInst.name == 'pathfinder'))
        >>> f.include_26m_obs('CasA')
        >>> f.print_results_summary()
           # | acquisition                          |start (s)| len (s) |files |     MB
           0 | 20170801T063349Z_pathfinder_corr     |   12337 |   11350 |    2 | 153499
           1 | 20170801T131035Z_pathfinder_corr     |       0 |    6922 |    1 |  75911
        Total  18271 seconds, 229410 MB of data.

        """

        connect_database()
        sources = HolographySource.select()
        sources = sources.where(HolographySource.name == source)
        if len(sources) == 0:
            msg = (
                f"No sources found in the database that match: {source}\n"
                + "Returning full time range"
            )
            logging.warning(msg)
        obs = (
            HolographyObservation.select()
            .join(HolographySource)
            .where(HolographyObservation.source << sources)
        )
        if require_quality:
            obs = obs.select().where(
                (HolographyObservation.quality_flag == 0)
                | (HolographyObservation.quality_flag == None)  # noqa E712
            )

        found_obs = False
        for ob in obs:
            in_range = (self.time_range[1] > ob.start_time) and (
                self.time_range[0] < ob.finish_time
            )
            if in_range:
                found_obs = True
                self.include_time_interval(ob.start_time, ob.finish_time)
        if not found_obs:
            msg = (
                f"No observation of the source ({source}) "
                "was found within the time range. Returning full time range"
            )
            logging.warning(msg)

    def exclude_transits(self, body, time_delta):
        """Add time intervals to exclude transits for given celestial body.

        Parameters
        ----------
        body : :class:`ephem.Body` or float
            Transiting celestial body.  If a float, interpret as a right
            ascension in degrees.
        time_delta : float
            Total amount of time to include surrounding the transit in
            seconds. Default is to use twice the value of
            :attr:`~Finder.min_interval`.

        Examples
        --------
        >>> from ch_util import finder
        >>> from datetime import datetime
        >>> f = finder.Finder()
        >>> f.set_time_range(datetime(2014,02,20), datetime(2014,02,22))
        >>> import ephem
        >>> f.exclude_transits(ephem.Sun(), time_delta=43200)
        >>> f.print_results_summary()
        interval | acquisition | offset from start (s) | length (s) | N files
           1  |  20140219T145849Z_abbot_corr  |    32453.1  |    51128.4  |  15
           2  |  20140219T145849Z_abbot_corr  |   126781.5  |    43193.0  |  13
           3  |  20140219T145523Z_stone_corr  |    32662.5  |    18126.9  |  6
           4  |  20140220T213252Z_stone_corr  |    16740.8  |    43193.0  |  13
        Total 155641.231275 seconds of data.

        """

        if not time_delta:
            time_delta = self.min_interval * 2
        ttimes = chime.transit_times(body, *self.time_range)
        for ttime in ttimes:
            self.exclude_time_interval(
                ttime - time_delta / 2.0, ttime + time_delta / 2.0
            )

    def exclude_daytime(self):
        """Add time intervals to exclude all day time data."""

        rise_times = chime.solar_rising(
            self.time_range[0] - 24 * 3600.0, self.time_range[1]
        )

        for rise_time in rise_times:
            set_time = chime.solar_setting(rise_time)
            self.exclude_time_interval(rise_time, set_time)

    def exclude_nighttime(self):
        """Add time intervals to exclude all night time data."""

        set_times = chime.solar_setting(
            self.time_range[0] - 24 * 3600.0, self.time_range[1]
        )

        for set_time in set_times:
            rise_time = chime.solar_rising(set_time)
            self.exclude_time_interval(set_time, rise_time)

    def exclude_sun(self, time_delta=4000.0, time_delta_rise_set=4000.0):
        """Add time intervals to exclude sunrise, sunset, and sun transit.

        Parameters
        ----------
        time_delta : float
            Total amount of time to exclude surrounding the sun transit in
            seconds. Default is to use 4000.0 seconds.
        time_delta_rise_set : float
            Total amount of time to exclude after sunrise and before sunset
            in seconds.  Default is to use 4000.0 seconds.
        """

        # Sunrise
        rise_times = chime.solar_rising(
            self.time_range[0] - time_delta_rise_set, self.time_range[1]
        )
        for rise_time in rise_times:
            self.exclude_time_interval(rise_time, rise_time + time_delta_rise_set)

        # Sunset
        set_times = chime.solar_setting(
            self.time_range[0], self.time_range[1] + time_delta_rise_set
        )
        for set_time in set_times:
            self.exclude_time_interval(set_time - time_delta_rise_set, set_time)

        # Sun transit
        transit_times = chime.solar_transit(
            self.time_range[0] - time_delta / 2.0, self.time_range[1] + time_delta / 2.0
        )
        for transit_time in transit_times:
            self.exclude_time_interval(
                transit_time - time_delta / 2.0, transit_time + time_delta / 2.0
            )

    def set_hk_input(self, name):
        """Restrict files to only one HK input type.

        This is a shortcut for specifying
        ``file_condition = (chimedb.data_index.HKFileInfo.atmel_name == name)``
        in :meth:`get_results_acq`. Instead, one can simply call this function
        with **name** as, e.g., "LNA", "FLA", and calls to
        :meth:`get_results_acq` will be appropriately restricted.

        Parameters
        ----------
        name : str
            The name of the housekeeping input.
        """
        self._atmel_restrict = di.HKFileInfo.atmel_name == name

    def get_results_acq(self, acq_ind, file_condition=None):
        """Get search results restricted to a given acquisition.

        Parameters
        ----------
        acq_ind : int
            Index of :attr:`Finder.acqs` for the desired acquisition.
        file_condition : :mod:`peewee` comparison
            Any additional condition for filtering the files within the
            acquisition. In general, this should be a filter on one of the file
            information tables, e.g., :class:`CorrFileInfo`.

        Returns
        -------
        interval_list : :class:`DataIntervalList`
            Search results.

        """
        acq = self.acqs[acq_ind]
        acq_start = acq.start_time
        acq_finish = acq.finish_time
        time_intervals = _trim_intervals_range(
            self.time_intervals, (acq_start, acq_finish), self.min_interval
        )
        time_intervals = _trim_intervals_exclusions(
            time_intervals, self.time_exclusions, self.min_interval
        )
        # Deal with all global flags.
        for severity, mode in self.global_flag_mode.items():
            if mode is GF_ACCEPT:
                # Do nothing.
                continue

            # Need to actually get the flags.
            global_flags = layout.global_flags_between(acq_start, acq_finish, severity)
            global_flag_names = [gf.name for gf in global_flags]
            flag_times = []
            for f in global_flags:
                start, stop = layout.get_global_flag_times(f.id)
                if stop is None:
                    stop = time.time()
                start = ctime.ensure_unix(start)
                stop = ctime.ensure_unix(stop)
                flag_times.append((start, stop))
            overlap = _check_intervals_overlap(time_intervals, flag_times)

            if mode is GF_WARN:
                if overlap:
                    msg = (
                        f"Global flag with severity '{severity}' present in data"
                        " search results and warning requested."
                        " Global flag name: " + global_flag_names[overlap[1]]
                    )
                    logging.warning(msg)
            elif mode is GF_RAISE:
                if overlap:
                    msg = (
                        f"Global flag with severity '{severity}' present in data"
                        " search results and exception requested."
                        " Global flag name: " + global_flag_names[overlap[1]]
                    )
                    raise DataFlagged(msg)
            elif mode is GF_REJECT:
                if overlap:
                    time_intervals = _trim_intervals_exclusions(
                        time_intervals, flag_times, self.min_interval
                    )
            else:
                raise RuntimeError("Finder has invalid global_flag_mode.")
        # Do the same for Data flags
        if len(self.data_flag_types) > 0:
            df_types = [t.name for t in DataFlagType.select()]
            for dft in self.data_flag_types:
                if dft not in df_types:
                    raise RuntimeError(f"Could not find data flag type {dft}.")
                flag_times = []
                for f in DataFlag.select().where(
                    DataFlag.type == DataFlagType.get(name=dft)
                ):
                    start, stop = f.start_time, f.finish_time
                    if stop is None:
                        stop = time.time()
                    start = ctime.ensure_unix(start)
                    stop = ctime.ensure_unix(stop)
                    flag_times.append((start, stop))
                overlap = _check_intervals_overlap(time_intervals, flag_times)
                if overlap:
                    time_intervals = _trim_intervals_exclusions(
                        time_intervals, flag_times, self.min_interval
                    )
        data_intervals = []
        if self._atmel_restrict:
            if file_condition:
                file_condition &= self._atmel_restrict
            else:
                file_condition = self._atmel_restrict
        for time_interval in time_intervals:
            filenames = files_in_range(
                acq.id,
                time_interval[0],
                time_interval[1],
                self._my_node,
                file_condition,
                self._node_spoof,
            )
            filenames = sorted(filenames)

            tup = (filenames, time_interval)
            if acq.type == di.AcqType.corr():
                data_intervals.append(CorrDataInterval(tup))
            elif acq.type == di.AcqType.hk():
                data_intervals.append(HKDataInterval(tup))
            elif acq.type == di.AcqType.weather():
                data_intervals.append(WeatherDataInterval(tup))
            elif acq.type == di.AcqType.flaginput():
                data_intervals.append(FlagInputDataInterval(tup))
            elif acq.type == di.AcqType.gain():
                data_intervals.append(CalibrationGainDataInterval(tup))
            elif acq.type == di.AcqType.digitalgain():
                data_intervals.append(DigitalGainDataInterval(tup))
            elif acq.type == di.AcqType.timing():
                data_intervals.append(TimingDataInterval(tup))
            else:
                data_intervals.append(BaseDataInterval(tup))

        return DataIntervalList(data_intervals)

    def get_results(self, file_condition=None):
        """Get all search results.

        Parameters
        ----------
        file_condition : :mod:`peewee` comparison
            Any additional condition for filtering the files within the
            acquisition. In general, this should be a filter on one of the file
            information tables, e.g., chimedb.data_index.CorrFileInfo.

        Returns
        -------
        interval_list : :class:`DataIntervalList`
            Search results.
        cond : :mod:`peewee` comparison
            Any extra filters, particularly filters on individual files.

        """

        intervals = []
        for ii in range(len(self.acqs)):
            intervals += self.get_results_acq(ii, file_condition)
        return DataIntervalList(intervals)

    def print_acq_info(self):
        """Print the acquisitions included in this search and thier properties.

        This method is convenient when searching the data index interactively
        and you want to see what acquisitions remain after applying filters or
        restricting the time range.

        See Also
        --------
        :meth:`Finder.print_results_summary`

        """

        print("acquisition | name | start | length (hrs) | N files")
        row_proto = "%4d  |  %-36s  |  %s  |  %7.2f  |  %4d"
        for ii, acq in enumerate(self.acqs):
            start = acq.start_time
            finish = acq.finish_time
            length = (finish - start) / 3600.0
            start = ctime.unix_to_datetime(start)
            start = start.strftime("%Y-%m-%d %H:%M")
            name = acq.name
            n_files = acq.n_timed_files
            print(row_proto % (ii, name, start, length, n_files))

    def print_results_summary(self):
        """Print a summary of the search results."""

        total_data = 0.0
        total_size = 0.0
        interval_number = 0
        titles = ("#", "acquisition", "start (s)", "len (s)", "files", "MB")
        rows = []
        for ii, acq in enumerate(self.acqs):
            acq_start = acq.start_time
            intervals = self.get_results_acq(ii)
            for interval in intervals:
                offset = interval[1][0] - acq_start
                length = interval[1][1] - interval[1][0]
                n_files = len(interval[0])
                # Get total size of files by doing new query.
                cond = (
                    (di.ArchiveFile.acq == acq)
                    & (di.ArchiveFileCopy.node << self._my_node)
                    & (di.ArchiveFileCopy.has_file == "Y")
                )
                info_cond = False
                for i in self._file_info:
                    info_cond |= (i.finish_time >= interval[1][0]) & (
                        i.start_time <= interval[1][1]
                    )
                size_q = di.ArchiveFile.select(pw.fn.Sum(di.ArchiveFile.size_b)).join(
                    di.ArchiveFileCopy
                )
                for i in self._file_info:
                    size_q = size_q.switch(di.ArchiveFile).join(
                        i, join_type=pw.JOIN.LEFT_OUTER
                    )
                size_q = size_q.where(cond & info_cond)
                try:
                    s = float(size_q.scalar()) / 1024**2  # MB.
                except TypeError:
                    s = 0
                info = (interval_number, acq.name, offset, length, n_files, s)
                rows.append(info)
                total_data += length
                total_size += s
                interval_number += 1
        print(tabulate.tabulate(rows, headers=titles))
        print(f"Total {total_data:6.0f} seconds, {total_size:6.0f} MB of data.")


def _trim_intervals_range(intervals, time_range, min_interval=0.0):
    range_start, range_end = time_range
    out = []
    for start, end in intervals:
        start = max(start, range_start)
        end = min(end, range_end)
        if end <= start + min_interval:
            continue

        out.append((start, end))
    return out


def _trim_intervals_exclusions(intervals, exclusions, min_interval=0.0):
    for excl_start, excl_end in exclusions:
        tmp_intervals = []
        for start, end in intervals:
            if end <= excl_start or start >= excl_end:
                if start + min_interval <= end:
                    tmp_intervals.append((start, end))
                continue
            if end > excl_start and start + min_interval <= excl_start:
                tmp_intervals.append((start, excl_start))
            if start < excl_end and excl_end + min_interval <= end:
                tmp_intervals.append((excl_end, end))
        intervals = tmp_intervals
    return intervals


def _check_intervals_overlap(intervals1, intervals2):
    """Return the first pair of indexes that overlap."""
    for ii in range(len(intervals1)):
        start1, stop1 = intervals1[ii]
        for jj in range(len(intervals2)):
            start2, stop2 = intervals2[jj]
            if start1 < stop2 and start2 < stop1:
                return ii, jj
    return None


def _validate_gf_value(value):
    if value not in (GF_REJECT, GF_RAISE, GF_WARN, GF_ACCEPT):
        raise ValueError(
            "Global flag behaviour must be one of"
            " the *GF_REJECT*, *GF_RAISE*, *GF_WARN*, *GF_ACCEPT*"
            " constants from the finder module."
        )


def _get_global_flag_times_by_name_event_id(flag):
    if isinstance(flag, str):
        event = (
            layout.event.select()
            .where(layout.event.active == True)  # noqa: E712
            .join(
                layout.global_flag, on=(layout.event.graph_obj == layout.global_flag.id)
            )
            .where(layout.global_flag.name == flag)
            .get()
        )
    else:
        event = layout.event.get(id=flag)
    start = event.start.time
    try:
        end = event.end.time
    except pw.DoesNotExist:
        end = time.time()
    return start, end


class DataIntervalList(list):
    """A list of data index search results.

    Just a normal python list of :class:`DataInterval`-derived objects with
    some helper methods. Instances are created by calls to
    :meth:`Finder.get_results`.
    """

    def iter_reader(self):
        """Iterate over data intervals converting to :class:`andata.Reader`.

        Returns
        -------
        reader_iterator
            Iterator over data intervals as :class:`andata.Reader` instances.

        """

        for data_interval in self:
            yield data_interval.as_reader()

    def iter_loaded_data(self, **kwargs):
        """Iterate over data intervals loading as :class:`andata.AnData`.

        Parameters
        ----------
        **kwargs : argument list
            Pass any parameters accepted by the
            :class:`BaseDataInverval`-derived class that you are using.

        Returns
        -------
        loaded_data_iterator
            Iterator over data intervals loaded into memory as
            :class:`andata.BaseData`-derived instances.

        Examples
        --------

        Use this method to loop over data loaded into memory.

        >>> for data in interval_list.iter_loaded_data():
        ...     pass

        Data is loaded into memory on each iteration. To immediately load all
        data into memory, initialize a list using the iterator:

        >>> loaded_data_list = list(interval_list.iter_loaded_data())

        """

        for data_interval in self:
            yield data_interval.as_loaded_data(**kwargs)


class BaseDataInterval(tuple):
    """A single data index search result.

    Just a normal python tuple with some helper methods. Instances are created
    by calls to :meth:`Finder.get_results`.

    A data interval as two elements: a list of filenames and a time range within
    those files.

    You should generally only use the classes derived from this one (i.e.,
    :class:`CorrDataInterval`, etc.)
    """

    @property
    def _reader_class(self):
        # only dynamic imports of andata allowed in this module.
        from . import andata

        return andata.BaseReader

    def as_reader(self):
        """Get data interval as an :class:`andata.Reader` instance.

        The :class:`andata.Reader` is initialized with the filename list part
        of the data interval then the time range part of the data interval is
        used as an arguments to :meth:`andata.Reader.select_time_range`.

        Returns
        -------
        reader : :class:`andata.Reader`

        """
        rc = self._reader_class
        reader = rc(self[0])
        reader.select_time_range(self[1][0], self[1][1])
        return reader

    def as_loaded_data(self, **kwargs):
        """Load data interval to memory as an :class:`andata.AnData` instance.

        Parameters
        ----------
        datasets : list of strings
            Passed on to :meth:`andata.AnData.from_acq_h5`

        Returns
        -------
        data : :class:`andata.AnData`
            Data interval loaded into memory.

        """
        reader = self.as_reader()
        for k, v in kwargs.items():
            if v is not None:
                setattr(reader, k, v)
        return reader.read()


class CorrDataInterval(BaseDataInterval):
    """Derived class from :class:`BaseDataInterval` for correlator data."""

    @property
    def _reader_class(self):
        # only dynamic imports of andata allowed in this module.
        from . import andata

        return andata.CorrReader

    def as_loaded_data(self, prod_sel=None, freq_sel=None, datasets=None):
        """Load data interval to memory as an :class:`andata.CorrData` instance

        Parameters
        ----------
        prod_sel : valid numpy index
            Passed on to :meth:`andata.CorrData.from_acq_h5`
        freq_sel : valid numpy index
            Passed on to :meth:`andata.CorrData.from_acq_h5`
        datasets : list of strings
            Passed on to :meth:`andata.CorrData.from_acq_h5`

        Returns
        -------
        data : :class:`andata.CorrData`
            Data interval loaded into memory.

        """
        return super().as_loaded_data(
            prod_sel=prod_sel, freq_sel=freq_sel, datasets=datasets
        )


# Legacy.
DataInterval = CorrDataInterval


class HKDataInterval(BaseDataInterval):
    """Derived class from :class:`BaseDataInterval` for housekeeping data."""

    @property
    def _reader_class(self):
        # only dynamic imports of andata allowed in this module.
        from . import andata

        return andata.HKReader


class WeatherDataInterval(BaseDataInterval):
    """Derived class from :class:`BaseDataInterval` for weather data."""

    @property
    def _reader_class(self):
        # only dynamic imports of andata allowed in this module.
        from . import andata

        return andata.WeatherReader


class FlagInputDataInterval(BaseDataInterval):
    """Derived class from :class:`BaseDataInterval` for flag input data."""

    @property
    def _reader_class(self):
        # only dynamic imports of andata allowed in this module.
        from . import andata

        return andata.FlagInputReader


class DigitalGainDataInterval(BaseDataInterval):
    """Derived class from :class:`BaseDataInterval` for digital gain data."""

    @property
    def _reader_class(self):
        # only dynamic imports of andata allowed in this module.
        from . import andata

        return andata.DigitalGainReader


class CalibrationGainDataInterval(BaseDataInterval):
    """Derived class from :class:`BaseDataInterval` for calibration gain data."""

    @property
    def _reader_class(self):
        # only dynamic imports of andata allowed in this module.
        from . import andata

        return andata.CalibrationGainReader


class TimingDataInterval(BaseDataInterval):
    """Derived class from :class:`BaseDataInterval` for timing data."""

    @property
    def _reader_class(self):
        # only dynamic imports of andata allowed in this module.
        from . import timing

        return timing.TimingReader


# Query routines
# ==============


def files_in_range(
    acq, start_time, end_time, node_list, extra_cond=None, node_spoof=None
):
    """Get files for a given acquisition within a time range.

    Parameters
    ----------
    acq : string or int
        Which acquisition, by its name or id key.
    start_time : float
        POSIX/Unix time for the start or time range.
    end_time : float
        POSIX/Unix time for the end or time range.
    node_list : list of `chimedb.data_index.StorageNode` objects
        Only return files residing on the given nodes.
    extra_cond : :mod:`peewee` comparison
        Any additional expression for filtering files.

    Returns
    -------
    file_names : list of strings
        List of filenames, including the full path.

    """

    if isinstance(acq, str):
        acq_name = acq
        acq = di.ArchiveAcq.get(di.ArchiveAcq.name == acq).acq
    else:
        acq_name = di.ArchiveAcq.get(di.ArchiveAcq.id == acq).name

    cond = (
        (di.ArchiveFile.acq == acq)
        & (di.ArchiveFileCopy.node << node_list)
        & (di.ArchiveFileCopy.has_file == "Y")
    )
    info_cond = False
    for i in file_info_table:
        info_cond |= (i.finish_time >= start_time) & (i.start_time <= end_time)

    if extra_cond:
        cond &= extra_cond

    query = (
        di.ArchiveFileCopy.select(
            di.ArchiveFileCopy.node,
            di.ArchiveFile.name,
            di.StorageNode.root,
            di.StorageNode.name.alias("node_name"),
        )
        .join(di.StorageNode)
        .switch(di.ArchiveFileCopy)
        .join(di.ArchiveFile)
    )
    for i in file_info_table:
        query = query.switch(di.ArchiveFile).join(i, join_type=pw.JOIN.LEFT_OUTER)
    query = query.where(cond & info_cond).objects()

    if not node_spoof:
        return [path.join(af.root, acq_name, af.name) for af in query]

    return [path.join(node_spoof[af.node_name], acq_name, af.name) for af in query]


# Exceptions
# ==========


class DataFlagged(CHIMEdbError):
    """Raised when data is affected by a global flag."""


if __name__ == "__main__":
    import doctest

    doctest.testmod()
