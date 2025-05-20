"""
Catalog of HFB test targets
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from fluxcat import FluxCatalog
from .tools import ensure_list

if TYPE_CHECKING:
    import skyfield.starlib.Star
    import caput.time.Observer

# Define the source collection(s) that should be loaded
# when `HFBCatalog` is first used.
# This is a catalog provided by `fluxcat`.
DEFAULT_COLLECTIONS = "hfb_target_list"


class HFBCatalog(FluxCatalog):
    """
    Class for cataloguing HFB targets.

    Attributes
    ----------
    fields : list
        List of attributes that are read-from and written-to the
        JSON catalog files.
    """

    fields = [
        "ra",
        "dec",
        "alternate_names",
        "freq_abs",
    ]

    def __init__(
        self,
        name,
        ra=None,
        dec=None,
        alternate_names=[],
        freq_abs=[],
        overwrite=0,
    ):
        """
        Instantiate an HFBCatalog object for an HFB target.

        Parameters
        ----------
        name : string
            Name of the source.

        ra : float
            Right Ascension in degrees.

        dec : float
            Declination in degrees.

        alternate_names : list of strings
            Alternate names for the source.

        freq_abs : list of floats
            Frequencies at which (the peaks of) absorption features are found.

        overwrite : int between 0 and 2
            Action to take in the event that this source is already in the catalog:
            - 0 - Return the existing entry.
            - 1 - Add the measurements to the existing entry.
            - 2 - Overwrite the existing entry.
            Default is 0.
            BUG: Currently, `freq_abs` is always overwritten.
        """

        super().__init__(
            name,
            ra=ra,
            dec=dec,
            alternate_names=alternate_names,
            overwrite=overwrite,
        )

        self.freq_abs = freq_abs


def get_doppler_shifted_freq(
    source: skyfield.starlib.Star | str,
    date: float | list,
    freq_rest: float | list = None,
    obs: caput.time.Observer | None = None,
) -> np.array:
    """Calculate Doppler shifted frequency of spectral feature with rest
    frequency `freq_rest`, seen towards source `source` at time `date`, due to
    Earth's motion and rotation, following the relativistic Doppler effect.

    Parameters
    ----------
    source
        Position(s) on the sky. If the input is a `str`, attempt to resolve this
        from `ch_util.hfbcat.HFBCatalog`.
    date
        Unix time(s) for which to calculate Doppler shift.
    freq_rest
        Rest frequency(ies) in MHz. If None, attempt to obtain rest frequency
        of absorption feature from `ch_util.hfbcat.HFBCatalog.freq_abs`.
    obs
        An Observer instance to use. If not supplied use `chime`. For many
        calculations changing from this default will make little difference.

    Returns
    -------
    freq_obs
        Doppler shifted frequencies in MHz. Array where rows correspond to the
        different input rest frequencies and columns correspond either to input
        times or to input sky positions (whichever contains multiple entries).

    Notes
    -----
    Only one of `source` and `date` can contain multiple entries.

    Example
    -------
    To get the Doppler shifted frequencies of a feature with a rest frequency
    of 600 MHz for two positions on the sky at a single point in time (Unix
    time 1717809534 = 2024-06-07T21:18:54+00:00), run:

    >>> from skyfield.starlib import Star
    >>> from skyfield.units import Angle
    >>> from ch_util.hfbcat import get_doppler_shifted_freq
    >>> coord = Star(ra=Angle(degrees=[100, 110]), dec=Angle(degrees=[45, 50]))
    >>> get_doppler_shifted_freq(coord, 1717809534, 600)
    """

    from scipy.constants import c as speed_of_light
    from ch_ephem.coord import get_range_rate

    if obs is None:
        from ch_ephem.observers import chime as obs

    # For source string inputs, get skyfield Star object from HFB catalog
    if isinstance(source, str):
        try:
            source = HFBCatalog[source].skyfield
        except KeyError:
            raise ValueError(f"Could not find source {source} in HFB catalog.")

    # Get rest frequency from HFB catalog
    if freq_rest is None:
        if not source.names or source.names not in HFBCatalog:
            raise ValueError(
                "Rest frequencies must be supplied unless source can be found "
                "in ch_util.hfbcat.HFBCatalog. "
                f"Got source {source} with names {source.names}"
            )

        freq_rest = HFBCatalog[source.names].freq_abs

    # Prepare rest frequencies for broadcasting
    freq_rest = np.asarray(ensure_list(freq_rest))[:, np.newaxis]

    # Get rate at which the distance between the observer and source changes
    # (positive for observer and source moving appart)
    range_rate = get_range_rate(source, date, obs)

    # Compute observed frequencies from rest frequencies
    # using relativistic Doppler effect
    beta = range_rate / speed_of_light
    return freq_rest * np.sqrt((1.0 - beta) / (1.0 + beta))
