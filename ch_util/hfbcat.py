"""
Catalog of HFB test targets
"""

import os
import warnings

from .fluxcat import FluxCatalog

# Define the source collection that should be loaded when this module is imported.
HFB_COLLECTION = os.path.join(
    os.path.dirname(__file__), "catalogs", "hfb_target_list.json"
)


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


# Load the HFB target list
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", r"The alternate name .* is already held by the source .*."
    )
    HFBCatalog.load(HFB_COLLECTION)
