"""Track Holography observations with the 26m telescope."""

import peewee as pw
import click

from caput import time as ctime
from ch_ephem.observers import chime
import chimedb.core as db

from . import holography as hl

source_alias = {"3C218": "HydraA", "3C348": "HerA", "3C123": "PerB"}


@click.group()
def cli():
    pass


@cli.command()
@click.argument("name", type=str)
@click.argument("ra", type=float)
@click.argument("dec", type=float)
def new_source(name, ra, dec):
    """Create a new holography source with a unique NAME.

    RA and DEC should be given as degrees in ICRS co-ordinates. Note, to use
    negative values for RA and DEC, you must use a '--' separator like:

        alpenhorn_holography -- NAME RA DEC
    """

    db.connect(read_write=True)

    try:
        source = (
            hl.HolographySource.select()
            .where(
                hl.HolographySource.ra.between(ra - 0.1, ra + 0.1)
                & hl.HolographySource.dec.between(dec - 0.1, dec + 0.1)
            )
            .get()
        )
        print('Source "{source.name}" with this RA and DEC already exists in database!')

    except pw.DoesNotExist:
        try:
            source = (
                hl.HolographySource.select()
                .where(
                    hl.HolographySource.ra.between(ra - 1, ra + 1)
                    & hl.HolographySource.dec.between(dec - 1, dec + 1)
                )
                .get()
            )
            print(
                f'WARNING! Source "{source.name}" is within 1 degree in RA and DEC '
                "of the source you are entering into the database!"
            )
        except pw.DoesNotExist:
            pass

        hl.HolographySource.create(name=name, ra=ra, dec=dec)
        print("Your source was successfully logged into the database!")


@cli.command()
@click.argument("source", type=str)
@click.argument("start_day", type=str)
@click.argument("start_lst", type=float)
@click.argument("duration_lst", type=float)
@click.option("--notes", type=str)
@click.option(
    "--quality_flag",
    type=int,
    help="Flag for poor quality data. Good data (default) is zero.",
)
def new_obs(source, start_day, start_lst, duration_lst, notes, quality_flag):
    """Create a new entry for an observation of SOURCE.

    START_DAY must be given as a string of format YYYYMMDD-ABT,
    where ABT is one of UTC, PST or PDT.
    START_LST and DURATION_LST must be given as a float representing hours and
    fraction of hours.
    """

    db.connect(read_write=True)

    if source in source_alias:
        source = source_alias[source]

    # Get the source
    try:
        source = hl.HolographySource.get(name=source)
    except pw.DoesNotExist:
        raise click.BadParameter("Source was not found in the database")

    hl.HolographyObservation.from_lst(
        source=source,
        start_day=start_day,
        start_lst=start_lst,
        duration_lst=duration_lst,
        quality_flag=quality_flag,
        notes=notes,
    )


@cli.command()
@click.argument("files", type=str)
@click.option("--notes", type=str, default=None)
@click.option(
    "--replace_dup",
    type=bool,
    default=False,
    is_flag=True,
    help="replace duplicate observations",
)
@click.option("--dryrun", type=bool, default=False, is_flag=True)
@click.option(
    "--start_tol",
    type=float,
    default=60.0,
    help=(
        "Time range tolerance in seconds around which to match "
        "duplicate observations (default: 60)"
    ),
)
@click.option(
    "--onsource_dist",
    type=float,
    default=0.1,
    help=(
        "Distance in degrees beyond which to consider the telescope off source"
        " (default: 0.1) "
    ),
)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
@click.option(
    "--quality_flag",
    type=int,
    default=0,
    help="Flag for poor quality data. Good data (default) is zero.",
)
def new_obs_ant(
    files, notes, replace_dup, start_tol, dryrun, onsource_dist, verbose, quality_flag
):
    """Create new entries for observations in specified files using the .ANT
    log from the John Galt Telescope to identify when the telescope arrived
    on source and left the source. The log entry will be created based on the
    sidereal times in the .ANT file using the .POST_REPORT file to get the
    corresponding date.

    This function is a wrapper for

        ch_util.holography.HolographyObservation.create_from_ant_logs

    example: alpenhorn_holography new_obs_ant '/mnt/gong/26m_logs/post/*JUN18*.zip'

    Required input:

    files : str
        Path to zip archive(s) containing the log files
    """

    import glob

    db.connect(read_write=True)

    if files.split(".")[-1].lower() != "zip":
        print(f'WARNING: File specified is not ".zip". Was that intentional?\n{files}')

    filelist = glob.glob(files)

    obs = hl.HolographyObservation()
    obs.create_from_ant_logs(
        filelist,
        notes=notes,
        dryrun=dryrun,
        start_tol=start_tol,
        replace_dup=replace_dup,
        verbose=verbose,
        onsource_dist=onsource_dist,
        quality_flag=quality_flag,
    )


@cli.command()
@click.argument("files", type=str)
@click.option("--notes", type=str, default=None)
@click.option(
    "--replace_dup",
    type=bool,
    default=False,
    help="Replace duplicate observations in the database",
)
@click.option(
    "--start_tol",
    type=float,
    default=60.0,
    help=(
        "Time range tolerance in seconds around which to match "
        "duplicate observations (default: 60)"
    ),
)
def new_obs_post_reports(files, notes, replace_dup, start_tol):
    """Create new entries for observations in specified zip archives using the
    .POST_REPORT log from the John Galt Telescope. This file is created
    at the *beginning* of the observation and therefore does not know exactly
    when the 26 m reached the source or if the observation ran to completion.
    new_obs_ant will in most cases be more accurate.

    This function is a wrapper for:

        ch_util.holography.HolographyObservation.create_from_post_reports

    example: alpenhorn_holography new_obs_ant '/mnt/gong/26m_logs/post/*JUN18*.zip'
    """

    import glob

    print("command line options: ", files, notes, replace_dup, start_tol)

    db.connect(read_write=True)

    filelist = glob.glob(files)

    obs = hl.HolographyObservation()
    obs.create_from_post_reports(
        filelist,
        notes=notes,
        dryrun=False,
        start_tol=start_tol,
        replace_dup=replace_dup,
    )


@cli.command()
@click.argument("source", type=str)
@click.argument("start_time_utc", type=str)
@click.argument("duration_lst", type=float)
@click.option("--notes", type=str)
@click.option(
    "--quality_flag", type=int, help="Flag for poor quality data. Good data is zero."
)
def new_obs_utc(source, start_time_utc, duration_lst, notes, quality_flag):
    """Create a new entry for an observation of SOURCE.

    START_TIME_UTC must be given as a string of format YYYYMMDDTHHMMSS, so e.g.
    20150112T134600.
    DURATION_LST must be given as a float representing hours and
    fraction of hours.
    """

    db.connect(read_write=True)

    if source in source_alias:
        source = source_alias[source]

    # Get the source
    try:
        source = hl.HolographySource.get(name=source)
    except pw.DoesNotExist:
        raise click.BadParameter("Source was not found in the database")

    # Get the start time in unix
    start_time = ctime.ensure_unix(ctime.timestr_to_datetime(start_time_utc))
    # Get the duration in unix
    duration_unix = duration_lst * (3600.0) / ctime.SIDEREAL_S

    finish_time = start_time + duration_unix

    hl.HolographyObservation.create(
        source=source,
        start_time=start_time,
        finish_time=finish_time,
        quality_flag=quality_flag,
        notes=notes,
    )


@cli.command()
@click.option(
    "--sort", type=str, help='Sort order ("id", "name", "ra", or "dec")', default="id"
)
def list_sources(sort):
    """List sources."""

    db.connect()
    sources = hl.HolographySource.select()

    sort_options = {
        "id": hl.HolographySource.id,
        "name": hl.HolographySource.name,
        "ra": hl.HolographySource.ra,
        "dec": hl.HolographySource.dec,
    }

    try:
        sources = sources.order_by(sort_options[sort])
    except KeyError:
        print(f'Sort key "{sort}" not known.')

    print_sources(sources)


def print_sources(sources):
    from skyfield.api import Angle
    import tabulate

    print("Holography sources:")
    print(" ")

    source_info = []
    for s in sources:
        ra_hms = Angle(degrees=s.ra).signed_hms(warn=False)
        dec_dms = Angle(degrees=s.dec).signed_dms()

        source_info.append(
            [
                s.id,
                s.name,
                s.ra,
                s.dec,
                f"{ra_hms[0] * ra_hms[1]:2.0f}:{ra_hms[2]:02.0f}:{ra_hms[3]:05.2f}",
                (
                    f"{dec_dms[0] * dec_dms[1]:+3.0f}:"
                    f"{dec_dms[2]:02.0f}:{dec_dms[3]:05.2f}"
                ),
            ]
        )

    print(
        tabulate.tabulate(
            source_info,
            headers=["Id", "Name", "RA (deg)", "DEC (deg)", "RA (hms)", "dec (dms)"],
            colalign=["right", "left", "decimal", "decimal", "right", "right"],
        )
    )


@cli.command()
@click.option("--source", type=str, help="Show only the given source")
@click.option(
    "--sort",
    type=click.Choice(["name", "start"]),
    default="name",
    help="Sort by name of source or start_time of observation",
)
@click.option(
    "--tz",
    help=(
        'Specify time zone. "PDT": UTC-7. "PST": UTC-8. "EDT": UTC-4. '
        '"EST": UTC-5. "PT": "PST" or "PDT" depending on current date.'
    ),
)
@click.option(
    "--tzoffset",
    default=0,
    type=float,
    help=(
        "Time zone offset from UTC in hours (default: 0). "
        "--tz parameter takes priority."
    ),
)
@click.option(
    "--list_sources",
    default=False,
    is_flag=True,
    type=bool,
    help="List all known sources first",
)
@click.option(
    "--days", type=float, help="Number of days in the past to show", default=None
)
@click.option(
    "--recent", type=bool, is_flag=True, help="Alias for --days 30 --sort start --tz PT"
)
def list(source, sort, tzoffset=0, tz=None, list_sources=True, days=None, recent=False):
    """List sources and observations."""

    import tabulate
    import numpy as np
    import datetime
    import pytz

    sort_options = {
        "name": hl.HolographySource.name,
        "start": hl.HolographyObservation.start_time.desc(),
    }

    if recent:
        days = 30
        tz = "PT"
        sort = "start"

    db.connect()

    tzs = {
        "PDT": -7.0,
        "PST": -8.0,
        "EDT": -4.0,
        "EST": -5.0,
        "UTC": 0.0,
        "PT": pytz.timezone("Canada/Pacific")
        .utcoffset(datetime.datetime.now())
        .total_seconds()
        / 3600,
    }

    if tz is not None:
        try:
            tzoffset = tzs[tz.upper()]
        except KeyError:
            print(f"Time zone {tz} not known. Known time zones:")
            for key, value in tzs.items():
                print(key, value)
            print(f"Using UTC{tzoffset:+.1f}.")

    sources = hl.HolographySource.select()

    if source is not None:
        sources = sources.where(hl.HolographySource.name == source)

    obs = (
        hl.HolographyObservation.select(hl.HolographyObservation, hl.HolographySource)
        .join(hl.HolographySource)
        .where(hl.HolographyObservation.source << sources)
    )

    if days is not None:
        start_time = (
            ctime.datetime_to_unix(datetime.datetime.today()) - days * 24.0 * 3600.0
        )
        obs = obs.where(hl.HolographyObservation.start_time > start_time)

    sort_param = sort_options[sort]
    obs = obs.order_by(sort_param)

    if list_sources:
        print_sources(sources)
        print("\n")

    def _trim_notes(notes, wlen=10):
        if notes is None:
            return None
        words = notes.split()
        if len(words) < wlen:
            return notes
        words = words[:10] + ["..."]
        return " ".join(words)

    DRAO_lon = chime.longitude * 24.0 / 360.0
    ts = ctime.skyfield_wrapper.timescale

    obs_info = [
        [
            o.id,
            o.source.name,
            ctime.unix_to_datetime(o.start_time + tzoffset * 3600.0).strftime(
                "%Y-%m-%d %H:%M"
            ),
            np.mod(ts.utc(ctime.unix_to_datetime(o.start_time)).gmst + DRAO_lon, 24.0),
            ctime.unix_to_datetime(o.finish_time + tzoffset * 3600.0).strftime(
                "%Y-%m-%d %H:%M"
            ),
            np.mod(ts.utc(ctime.unix_to_datetime(o.finish_time)).gmst + DRAO_lon, 24.0),
            o.quality_flag,
            _trim_notes(o.notes),
        ]
        for o in obs
    ]

    obs_head = [
        "Id",
        "Name",
        "Start time UTC" + f"{tzoffset:+.1f}",
        "LST",
        "Finish time UTC" + f"{tzoffset:+.1f}",
        "LST",
        "Quality",
        "Notes",
    ]

    print("Holography observations:")
    print(" ")
    print(tabulate.tabulate(obs_info, headers=obs_head, floatfmt=".2f"))


@cli.command()
@click.argument("obs_id", type=int)
def delete_obs(obs_id):
    """Delete an entry for an observation with ID.

    You can find the ID of the observation you want to delete by using 'list'.
    """
    import tabulate

    db.connect(read_write=True)

    # Get the observation
    try:
        obs = hl.HolographyObservation.get(id=obs_id)
    except pw.DoesNotExist:
        raise click.BadParameter("Observation does not exist in database")

    obs_info = [
        [
            obs.id,
            obs.source.name,
            ctime.unix_to_datetime(obs.start_time),
            ctime.unix_to_datetime(obs.finish_time),
            obs.quality_flag,
        ]
    ]

    obs_head = ["Id", "Name", "Start time", "Finish time", "Quality"]

    print(" ")
    print(tabulate.tabulate(obs_info, headers=obs_head))
    print(" ")

    if click.confirm("Do you want to delete this Holography observation?"):
        obs.delete_instance()
        click.echo("Observation deleted ")


if __name__ == "__main__":
    cli()
