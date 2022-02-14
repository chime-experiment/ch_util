"""
Holography observation tables.

This module defines the tables:

- :py:class:`HolographyObservation`
- :py:class:`HolographySource`

and the constants:

- :py:const:`QUALITY_GOOD`
- :py:const:`QUALITY_OFFSOURCE`
- :py:const:`ONSOURCE_DIST_TO_FLAG`

"""

import os
import warnings
import zipfile
import numpy as np
import peewee as pw
from chimedb.core.orm import base_model

from ch_util import ephemeris

# Global variables and constants.
# ================================

QUALITY_GOOD = 0
QUALITY_OFFSOURCE = 1
QUALITY_BADGATING = 2
QUALITY_NOISEOFF = 4
ONSOURCE_DIST_TO_FLAG = 0.1

# Tables in the for tracking Holography observations
# ==================================================


class HolographySource(base_model):
    """A peewee model for the Holography sources.

    Attributes
    ----------
    name : str
        Unique name for the source. Be careful to avoid duplicates.
    ra, dec : float
        ICRS co-ordinates of the source.
    """

    name = pw.CharField(max_length=128, unique=True)
    ra = pw.FloatField()
    dec = pw.FloatField()


class HolographyObservation(base_model):
    """
    A peewee model for the holographic observations.

    Attributes
    ----------
    source : foreign key
        The source that we were observing.
    start_time, finish_time : float
        Start and end times of the source observation (as UNIX times).
    notes : str
        Any free form notes about the observation.
    """

    source = pw.ForeignKeyField(HolographySource, backref="observations")
    start_time = pw.DoubleField()
    finish_time = pw.DoubleField()

    quality_flag = (
        pw.BitField()
    )  # maximum of 64 fields. If we need more, use BigBitField
    off_source = quality_flag.flag(QUALITY_OFFSOURCE)
    bad_gating = quality_flag.flag(QUALITY_BADGATING)
    noise_off = quality_flag.flag(QUALITY_NOISEOFF)

    notes = pw.TextField(null=True)

    @classmethod
    def from_lst(
        cls,
        source,
        start_day,
        start_lst,
        duration_lst,
        quality_flag=QUALITY_GOOD,
        notes=None,
    ):
        """Method to initialize a HolographyObservation from a start day,
        start LST, and a stop day, stop LST.

        Parameters
        ----------
        source : HolographySource
            An instance of HolographySource.
        start_day: string
            Of format YYYMMDD-ABT, ABT can be one of (UTC, PST, PDT)
        start_lst, duration: float
            Hours and fraction of hours on a scale from 0-24.
        quality_flag : int, default : 0
            Flag for poor quality data. Good data is zero.
            Sets a bitmask in the HolographyObservation instance.
        notes : string, optional
            Any notes on this observation.
        """

        start_time = ephemeris.lsa_to_unix(
            start_lst * 360 / 24,
            ephemeris.datetime_to_unix(ephemeris.parse_date(start_day)),
        )
        duration_unix = duration_lst * (3600.0) * ephemeris.SIDEREAL_S

        finish_time = start_time + duration_unix

        return cls.create(
            source=source,
            start_time=start_time,
            finish_time=finish_time,
            quality_flag=quality_flag,
            notes=notes,
        )

    # Aliases of source names in the spreadsheet to ones we use in the database
    # (hard-coded at initialization, but user should be able to edit)
    source_alias = {
        "B0329******": "B0329+54",
        "B0950*****": "B0950+08",
        "B1133+16*****": "B1133+16",
        "B1929+10*****": "B1929+10",
        "B0355+56": "B0355+54",
        "3C218": "HydraA",
        "C48": "3C48",
        "3C_58": "3C58",
        "3C348": "HerA",
        "3C144": "TauA",
        "PerB": "3C123",
        "B0531+21*****": "B0531+21",
        "B2016+28*****": "B2016+28",
        "B1133*****": "B1133+16",
        "B1937+21*****": "B1937+21",
        "B2016*****": "B2016+28",
        "B0950+08*****": "B0950+08",
        "FAN": "FanRegion1",
        "Fan Region 1": "FanRegion1",
        "FAN1": "FanRegion1",
        "Fan Region 2": "FanRegion2",
        "FAN2": "FanRegion2",
        "B0905*****": "B0905*****",
        "VIRA": "VirA",
        "3C274": "VirA",
        "3C405": "CygA",
        "3C461": "CasA",
        "NCP_20H": "NCP 20H",
        "NCP_4H": "NCP 4H",
    }

    # read the .POST_REPORT file and pull out source name, time, and observation
    # duration
    @classmethod
    def parse_post_report(cls, post_report_file):
        """
        read a .POST_REPORT file from the nsched program which controls the
        John Galt Telescope and extract the source name, estimated start time,
        DRAO sidereal day, commanded duration, and estimated finish time

        Parameters
        ----------
        post_report_file : str
            path to the .POST_REPORT file to read

        Returns
        -------
        output_params : dictionary
            output_params['src'] : HolographySource object or string
                If the source is a known source in the holography database,
                return the HolographySource object. If not, return the name
                of the source as a string
            output_params['SID'] : int
                DRAO sidereal day at the beginning of the observation
            output_params['start_time'] : skyfield time object
                UTC time at the beginning of the observation
            output_params['DURATION'] : float
                Commanded duration of the observation in sidereal hours
            output_params['finish_time'] : skyfield time object
                Calculated UTC time at the end of the observation
                Calculated as start_time + duration * ephemeris.SIDEREAL_S

        """
        import re

        ts = ephemeris.skyfield_wrapper.timescale

        output_params = {}

        with open(post_report_file, "r") as f:
            lines = [line for line in f]
            for l in lines:
                if (l.find("Source")) != -1:
                    srcnm = re.search("Source:\s+(.*?)\s+", l).group(1)
                    if srcnm in cls.source_alias:
                        srcnm = cls.source_alias[srcnm]
                if (l.find("DURATION")) != -1:
                    output_params["DURATION"] = float(
                        re.search("DURATION:\s+(.*?)\s+", l).group(1)
                    )

                # convert Julian Date to Skyfield time object
                if (l.find("JULIAN DATE")) != -1:
                    output_params["start_time"] = ts.ut1(
                        jd=float(re.search("JULIAN DATE:\s+(.*?)\s+", l).group(1))
                    )

                if l.find("SID:") != -1:
                    output_params["SID"] = int(re.search("SID:\s(.*?)\s+", l).group(1))
            try:
                output_params["src"] = HolographySource.get(name=srcnm)
            except pw.DoesNotExist:
                print("Missing", srcnm)
                output_params["src"] = srcnm

            output_params["finish_time"] = ephemeris.unix_to_skyfield_time(
                ephemeris.ensure_unix(output_params["start_time"])
                + output_params["DURATION"] * 3600.0 * ephemeris.SIDEREAL_S
            )

            output_params["quality_flag"] = QUALITY_GOOD

            return output_params

    @classmethod
    def create_from_ant_logs(
        cls,
        logs,
        verbose=False,
        onsource_dist=0.1,
        notes=None,
        quality_flag=0,
        **kwargs,
    ):
        """
        Read John Galt Telescope log files and create an entry in the
        holography database corresponding to the exact times on source

        Parameters
        ----------
        logs : list of strings
            log file archives (.zip files) to pass to parse_ant_logs()
        onsource_dist : float (default: 0.1)
            maximum angular distance at which to consider the Galt Telescope
            on source (in degrees)

        Returns
        -------
        none
        """

        from ch_util.ephemeris import sphdist
        from skyfield.positionlib import Angle

        ts = ephemeris.skyfield_wrapper.timescale
        DATE_FMT_STR = "%Y-%m-%d %H:%M:%S %z"

        pr_list, al_list = cls.parse_ant_logs(logs, return_post_report_params=True)

        for post_report_params, ant_log, curlog in zip(pr_list, al_list, logs):
            print(" ")
            if isinstance(post_report_params["src"], HolographySource):
                if verbose:
                    print(
                        "Processing {} from {}".format(
                            post_report_params["src"].name, curlog
                        )
                    )
                dist = sphdist(
                    Angle(degrees=post_report_params["src"].ra),
                    Angle(degrees=post_report_params["src"].dec),
                    ant_log["ra"],
                    ant_log["dec"],
                )
                if verbose:
                    print("onsource_dist = {:.2f} deg".format(onsource_dist))
                onsource = np.where(dist.degrees < onsource_dist)[0]

                if len(onsource) > 0:
                    stdoffset = np.std(dist.degrees[onsource[0] : onsource[-1]])
                    meanoffset = np.mean(dist.degrees[onsource[0] : onsource[-1]])
                    obs = {
                        "src": post_report_params["src"],
                        "start_time": ant_log["t"][onsource[0]],
                        "finish_time": ant_log["t"][onsource[-1]],
                        "quality_flag": QUALITY_GOOD,
                    }
                    noteout = "from .ANT log " + ts.now().utc_strftime(DATE_FMT_STR)
                    if notes is not None:
                        noteout = notes + " " + noteout
                    if stdoffset > 0.05 or meanoffset > ONSOURCE_DIST_TO_FLAG:
                        obs["quality_flag"] += QUALITY_OFFSOURCE
                        print(
                            (
                                "Mean offset: {:.4f}. Std offset: {:.4f}. "
                                "Setting quality flag to {}."
                            ).format(meanoffset, stdoffset, QUALITY_OFFSOURCE)
                        )
                        noteout = (
                            "Questionable on source. Mean, STD(offset) : "
                            "{:.3f}, {:.3f}. {}".format(meanoffset, stdoffset, noteout)
                        )
                    obs["quality_flag"] += quality_flag
                    if verbose:
                        print(
                            "Times in .ANT log    : {} {}".format(
                                ant_log["t"][onsource[0]].utc_strftime(DATE_FMT_STR),
                                ant_log["t"][onsource[-1]].utc_strftime(DATE_FMT_STR),
                            )
                        )
                        print(
                            "Times in .POST_REPORT: {} {}".format(
                                post_report_params["start_time"].utc_strftime(
                                    DATE_FMT_STR
                                ),
                                post_report_params["finish_time"].utc_strftime(
                                    DATE_FMT_STR
                                ),
                            )
                        )
                        print(
                            "Mean offset: {:.4f}. Std offset: {:.4f}.".format(
                                meanoffset, stdoffset
                            )
                        )

                    cls.create_from_dict(obs, verbose=verbose, notes=noteout, **kwargs)
                else:
                    print(
                        (
                            "No on source time found for {}\n{} {}\n"
                            "Min distance from source {:.1f} degrees"
                        ).format(
                            curlog,
                            post_report_params["src"].name,
                            post_report_params["start_time"].utc_strftime(
                                "%Y-%m-%d %H:%M"
                            ),
                            np.min(dist.degrees),
                        )
                    )
            else:
                print(
                    "{} is not a HolographySource; need to add to database?".format(
                        post_report_params["src"]
                    )
                )
                print("Doing nothing")

    @classmethod
    def create_from_dict(
        cls,
        dict,
        notes=None,
        start_tol=60.0,
        dryrun=True,
        replace_dup=False,
        verbose=False,
    ):
        """
        Create a holography database entry from a dictionary

        This routine checks for duplicates and overwrites duplicates if and
        only if `replace_dup = True`

        Parameters
        ----------
        dict : dict
            src : :py:class:`HolographySource`
                A HolographySource object for the source
            start_time
                Start time as a Skyfield Time object
            finish_time
                Finish time as a Skyfield Time object
        """
        DATE_FMT_STR = "%Y-%m-%d %H:%M:%S %Z"

        def check_for_duplicates(t, src, start_tol, ignore_src_mismatch=False):
            """
            Check for duplicate holography observations, comparing the given
            observation to the existing database

            Inputs
            ------
            t: Skyfield Time object
                beginning time of observation
            src: HolographySource
                target source
            start_tol: float
                Tolerance in seconds within which to search for duplicates
            ignore_src_mismatch: bool (default: False)
                If True, consider observations a match if the time matches
                but the source does not

            Outputs
            -------
            If a duplicate is found: :py:class:`HolographyObservation` object for the
            existing entry in the database

            If no duplicate is found: None
            """
            ts = ephemeris.skyfield_wrapper.timescale

            unixt = ephemeris.ensure_unix(t)

            dup_found = False

            existing_db_entry = cls.select().where(
                cls.start_time.between(unixt - start_tol, unixt + start_tol)
            )
            if len(existing_db_entry) > 0:
                if len(existing_db_entry) > 1:
                    print("Multiple entries found.")
                for entry in existing_db_entry:
                    tt = ts.utc(ephemeris.unix_to_datetime(entry.start_time))
                    # LST = GST + east longitude
                    ttlst = np.mod(tt.gmst + DRAO_lon, 24.0)

                    # Check if source name matches. If not, print a warning
                    # but proceed anyway.
                    if src.name.upper() == entry.source.name.upper():
                        dup_found = True
                        if verbose:
                            print("Observation is already in database.")
                    else:
                        if ignore_src_mismatch:
                            dup_found = True
                        print(
                            "** Observation at same time but with different "
                            + "sources in database: ",
                            src.name,
                            entry.source.name,
                            tt.utc_datetime().isoformat(),
                        )
                        # if the observations match in start time and source,
                        # call them the same observation. Not the most strict
                        # check possible.

                    if dup_found:
                        tf = ts.utc(ephemeris.unix_to_datetime(entry.finish_time))
                        print(
                            "Tried to add  :  {} {}; LST={:.3f}".format(
                                src.name, t.utc_datetime().strftime(DATE_FMT_STR), ttlst
                            )
                        )
                        print(
                            "Existing entry:  {} {}; LST={:.3f}".format(
                                entry.source.name,
                                tt.utc_datetime().strftime(DATE_FMT_STR),
                                ttlst,
                            )
                        )
            if dup_found:
                return existing_db_entry
            else:
                return None

        # DRAO longitude in hours
        DRAO_lon = ephemeris.chime.longitude * 24.0 / 360.0

        if verbose:
            print(" ")
        addtodb = True

        dup_entries = check_for_duplicates(dict["start_time"], dict["src"], start_tol)

        if dup_entries is not None:
            if replace_dup:
                if not dryrun:
                    for entry in dup_entries:
                        cls.delete_instance(entry)
                        if verbose:
                            print("Deleted observation from database and replacing.")
                elif verbose:
                    print("Would have deleted observation and replaced (dry run).")
                addtodb = True
            else:
                addtodb = False
                for entry in dup_entries:
                    print(
                        "Not replacing duplicate {} observation {}".format(
                            entry.source.name,
                            ephemeris.unix_to_datetime(entry.start_time).strftime(
                                DATE_FMT_STR
                            ),
                        )
                    )

        # we've appended this observation to obslist.
        # Now add to the database, if we're supposed to.
        if addtodb:
            string = "Adding to database: {} {} to {}"
            print(
                string.format(
                    dict["src"].name,
                    dict["start_time"].utc_datetime().strftime(DATE_FMT_STR),
                    dict["finish_time"].utc_datetime().strftime(DATE_FMT_STR),
                )
            )
            if dryrun:
                print("Dry run; doing nothing")
            else:
                cls.create(
                    source=dict["src"],
                    start_time=ephemeris.ensure_unix(dict["start_time"]),
                    finish_time=ephemeris.ensure_unix(dict["finish_time"]),
                    quality_flag=dict["quality_flag"],
                    notes=notes,
                )

    @classmethod
    def parse_ant_logs(cls, logs, return_post_report_params=False):
        """
        Unzip and parse .ANT log file output by nsched for John Galt Telescope
        observations

        Parameters
        ----------
        logs : list of strings
            .ZIP filenames. Each .ZIP archive should include a .ANT file and
            a .POST_REPORT file. This method unzips the archive, uses
            `parse_post_report` to read the .POST_REPORT file and extract
            the CHIME sidereal day corresponding to the DRAO sidereal day,
            and then reads the lines in the .ANT file to obtain the pointing
            history of the Galt Telescope during this observation.

            (The DRAO sidereal day is days since the clock in Ev Sheehan's
            office at DRAO was reset. This clock is typically only reset every
            few years, but it does not correspond to any defined date, so the
            date must be figured out from the .POST_REPORT file, which reports
            both the DRAO sidereal day and the UTC date and time.

            Known reset dates: 2017-11-21, 2019-3-10)

        Returns
        -------

        if output_params == False:
            ant_data: A dictionary consisting of lists containing the LST,
                hour angle, RA, and dec (all as Skyfield Angle objects),
                CHIME sidereal day, and DRAO sidereal day.

        if output_params == True
            output_params: dictionary returned by `parse_post_report`
            and
            ant_data: described above

        Files
        -----
        the .ANT and .POST_REPORT files in the input .zip archive are
        extracted into /tmp/26mlog/<loginname>/
        """

        from skyfield.positionlib import Angle
        from caput import time as ctime

        DRAO_lon = ephemeris.CHIMELONGITUDE * 24.0 / 360.0

        def sidlst_to_csd(sid, lst, sid_ref, t_ref):
            """
            Convert an integer DRAO sidereal day and LST to a float
            CHIME sidereal day

            Parameters
            ----------
            sid : int
                DRAO sidereal day
            lst : float, in hours
                local sidereal time
            sid_ref : int
                DRAO sidereal day at the reference time t_ref
            t_ref : skyfield time object, Julian days
                Reference time

            Returns
            -------
            output : float
                CHIME sidereal day
            """
            csd_ref = int(
                ephemeris.csd(ephemeris.datetime_to_unix(t_ref.utc_datetime()))
            )
            csd = sid - sid_ref + csd_ref
            return csd + lst / ephemeris.SIDEREAL_S / 24.0

        ant_data_list = []
        post_report_list = []

        for log in logs:
            doobs = True

            filename = log.split("/")[-1]
            basedir = "/tmp/26mlog/{}/".format(os.getlogin())
            basename, extension = filename.split(".")
            post_report_file = basename + ".POST_REPORT"
            ant_file = basename + ".ANT"

            if extension == "zip":
                try:
                    zipfile.ZipFile(log).extract(post_report_file, path=basedir)
                except:
                    print(
                        "Failed to extract {} into {}. Moving right along...".format(
                            post_report_file, basedir
                        )
                    )
                    doobs = False
                try:
                    zipfile.ZipFile(log).extract(ant_file, path=basedir)
                except:
                    print(
                        "Failed to extract {} into {}. Moving right along...".format(
                            ant_file, basedir
                        )
                    )
                    doobs = False

            if doobs:
                try:
                    post_report_params = cls.parse_post_report(
                        basedir + post_report_file
                    )

                    with open(os.path.join(basedir, ant_file), "r") as f:
                        lines = [line for line in f]
                        ant_data = {"sid": np.array([])}
                        lsth = []
                        lstm = []
                        lsts = []

                        hah = []
                        ham = []
                        has = []

                        decd = []
                        decm = []
                        decs = []

                        for l in lines:
                            arr = l.split()

                            try:
                                lst_hms = [float(x) for x in arr[2].split(":")]

                                # do last element first: if this is going to
                                # crash because a line in the log is incomplete,
                                # we don't want it to append to any of the lists

                                decs.append(float(arr[8].replace('"', "")))
                                decm.append(float(arr[7].replace("'", "")))
                                decd.append(float(arr[6].replace("D", "")))

                                has.append(float(arr[5].replace("S", "")))
                                ham.append(float(arr[4].replace("M", "")))
                                hah.append(float(arr[3].replace("H", "")))

                                lsts.append(float(lst_hms[2]))
                                lstm.append(float(lst_hms[1]))
                                lsth.append(float(lst_hms[0]))

                                ant_data["sid"] = np.append(
                                    ant_data["sid"], int(arr[1])
                                )
                            except:
                                print(
                                    "Failed in file {} for line \n{}".format(
                                        ant_file, l
                                    )
                                )
                                if len(ant_data["sid"]) != len(decs):
                                    print("WARNING: mismatch in list lengths.")

                        ant_data["lst"] = Angle(hours=(lsth, lstm, lsts))

                        ha = Angle(hours=(hah, ham, has))
                        dec = Angle(degrees=(decd, decm, decs))

                        ant_data["ha"] = Angle(
                            radians=ha.radians
                            - ephemeris.galt_pointing_model_ha(ha, dec).radians,
                            preference="hours",
                        )

                        ant_data["dec_cirs"] = Angle(
                            radians=dec.radians
                            - ephemeris.galt_pointing_model_dec(ha, dec).radians,
                            preference="degrees",
                        )

                        ant_data["csd"] = sidlst_to_csd(
                            np.array(ant_data["sid"]),
                            ant_data["lst"].hours,
                            post_report_params["SID"],
                            post_report_params["start_time"],
                        )

                    ant_data["t"] = ephemeris.unix_to_skyfield_time(
                        ephemeris.csd_to_unix(ant_data["csd"])
                    )

                    # Correct RA from equinox to CIRS coords (both in radians)
                    era = np.radians(
                        ctime.unix_to_era(ephemeris.ensure_unix(ant_data["t"]))
                    )
                    gast = ant_data["t"].gast * 2 * np.pi / 24.0

                    ant_data["ra_cirs"] = Angle(
                        radians=ant_data["lst"].radians
                        - ant_data["ha"].radians
                        + (era - gast),
                        preference="hours",
                    )

                    obs = ephemeris.Star_cirs(
                        ra=ant_data["ra_cirs"],
                        dec=ant_data["dec_cirs"],
                        epoch=ant_data["t"],
                    )

                    ant_data["ra"] = obs.ra
                    ant_data["dec"] = obs.dec

                    ant_data_list.append(ant_data)
                    post_report_list.append(post_report_params)
                except:
                    print("Parsing {} failed".format(post_report_file))

        if return_post_report_params:
            return post_report_list, ant_data_list
        return ant_data

    @classmethod
    def create_from_post_reports(
        cls,
        logs,
        start_tol=60.0,
        dryrun=True,
        replace_dup=False,
        verbose=True,
        notes=None,
    ):
        """Create holography database entry from .POST_REPORT log files
        generated by the nsched controller for the Galt Telescope.

        Parameters
        ----------
        logs : string
            list of paths to archives. Filenames should be, eg,
            01DEC17_1814.zip.  Must be only one period in the filename,
            separating the extension.

        start_tol : float (optional; default: 60.)
            Tolerance (in seconds) around which to search for duplicate
            operations.

        dryrun : boolean (optional; default: True)
            Dry run only; do not add entries to database

        replace_dup : boolean (optional; default: False)
            Delete existing duplicate entries and replace. Only has effect if
            dry_run == False

        notes : string or list of strings (optional; default: None)
            notes to be added. If a string, the same note will be added to all
            observations. If a list of strings (must be same length as logs),
            each element of the list will be added to the corresponding
            database entry.
            Nota bene: the text "Added by create_from_post_reports" with the
            current date and time will also be included in the notes database
            entry.

        Example
        -------
        from ch_util import holography as hl
        import glob

        obs = hl.HolographyObservation
        logs = glob.glob('/path/to/logs/*JUN18*.zip')
        obs_list, dup_obs_list, missing = obs.create_from_post_reports(logs, dryrun=False)
        """
        # check notes. Can be a string (in which case duplicate it), None (in
        # which case do nothing) or a list (in which case use it if same length
        # as logs, otherwise crash)
        if notes is None:
            print("Notes is None")
            notesarr = [None] * len(logs)
        elif isinstance(notes, str):
            notesarr = [notes] * len(logs)
        else:
            assert len(notes) == len(
                logs
            ), "notes must be a string or a list the same length as logs"
            notesarr = notes

        for log, note in zip(logs, notesarr):
            if verbose:
                print("Working on {}".format(log))
            filename = log.split("/")[-1]
            # basedir = '/'.join(log.split('/')[:-1]) + '/'
            basedir = "/tmp/"

            basename, extension = filename.split(".")

            post_report_file = basename + ".POST_REPORT"

            doobs = True
            if extension == "zip":
                try:
                    zipfile.ZipFile(log).extract(post_report_file, path=basedir)
                except Exception:
                    print(
                        "failed to find {}. Moving right along...".format(
                            post_report_file
                        )
                    )
                    doobs = False
            elif extension != "POST_REPORT":
                print(
                    "WARNING: extension should be .zip or .POST_REPORT; is ", extension
                )

            if doobs:

                # Read the post report file and pull out the HolographySource
                # object, start time (LST), and duration (in LST hours) of the
                # observation
                output_params = cls.parse_post_report(basedir + post_report_file)
                t = output_params["start_time"]
                src = output_params["src"]

                # if the source was found, src would be a HolographySource
                # object otherwise (ie the source is missing), it's a string
                if isinstance(src, str):
                    warnings.warn(
                        f"Source {src} was not found for observation at time {t}."
                    )
                else:
                    cls.create_from_dict(
                        output_params,
                        notes=notes,
                        start_tol=start_tol,
                        dryrun=dryrun,
                        replace_dup=replace_dup,
                        verbose=verbose,
                    )
