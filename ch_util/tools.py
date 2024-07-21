"""
Tools for CHIME analysis

A collection of miscellaneous utility routines.


Correlator Inputs
=================

Query the layout database to find out what is ultimately connected at the end
of correlator inputs. This is done by calling the routine
:func:`get_correlator_inputs`, which returns a list of the inputs. Routines
such as :func:`get_feed_positions` operate on this list.

- :py:meth:`get_correlator_inputs`
- :py:meth:`get_feed_positions`
- :py:meth:`get_feed_polarisations`
- :py:meth:`is_array`
- :py:meth:`is_array_x`
- :py:meth:`is_array_y`
- :py:meth:`is_array_on`
- :py:meth:`is_chime`
- :py:meth:`is_pathfinder`
- :py:meth:`is_holographic`
- :py:meth:`is_noise_source`
- :py:meth:`reorder_correlator_inputs`
- :py:meth:`redefine_stack_index_map`
- :py:meth:`serial_to_id`
- :py:meth:`serial_to_location`
- :py:meth:`parse_chime_serial`
- :py:meth:`parse_pathfinder_serial`
- :py:meth:`parse_old_serial`
- :py:meth:`get_noise_source_index`
- :py:meth:`get_holographic_index`
- :py:meth:`change_pathfinder_location`
- :py:meth:`change_chime_location`

This can determine if we are connected to any of the following:

- :py:class:`HolographyAntenna`
- :py:class:`ArrayAntenna`
- :py:class:`PathfinderAntenna`
- :py:class:`CHIMEAntenna`
- :py:class:`RFIAntenna`
- :py:class:`NoiseSource`
- :py:class:`Blank`

Example
-------

Fetch the inputs for blanchard during layout 38::

    >>> from datetime import datetime
    >>> inputs = get_correlator_inputs(datetime(2016,05,23,00), correlator='pathfinder')
    >>> inputs[1]
    CHIMEAntenna(id=1, reflector=u'W_cylinder', antenna=u'ANT0123B', powered=True, pos=9.071800000000001, input_sn=u'K7BP16-00040401', pol=u'S', corr=u'K7BP16-0004', cyl=0)
    >>> print "NS position:", inputs[1].pos
    NS position: 9.0718
    >>> print "Polarisation:", inputs[1].pol
    Polarisation: S
    >>> inputs[3]
    CHIMEAntenna(id=3, reflector=u'W_cylinder', antenna=u'ANT0128B', powered=True, pos=9.681400000000002, input_sn=u'K7BP16-00040403', pol=u'S', corr=u'K7BP16-0004', cyl=0)

Housekeeping Inputs
===================

Functions
---------

- :py:meth:`antenna_to_lna`
- :py:meth:`calibrate_temperature`
- :py:meth:`hk_to_sensor`
- :py:meth:`lna_to_antenna`
- :py:meth:`sensor_to_hk`

Classes
-------

- :py:class:`HKInput`


Product Array Mapping
=====================

Tools for mapping between products stored in upper triangular format, and the
underlying pairs of inputs.

- :py:meth:`cmap`
- :py:meth:`icmap`
- :py:meth:`fast_pack_product_array`
- :py:meth:`pack_product_array`
- :py:meth:`unpack_product_array`


Matrix Factorisation
====================

A few useful routines for factorising matrices, usually for calibration.

- :py:meth:`eigh_no_diagonal`
- :py:meth:`rankN_approx`
- :py:meth:`normalise_correlations`
- :py:meth:`apply_gain`
- :py:meth:`subtract_rank1_signal`


Fringestopping
==============

Routines for undoing the phase rotation of a fixed celestial source. The
routine :func:`fringestop` is an easy to use routine for fringestopping data
given a list of the feeds in the data. For more advanced usage
:func:`fringestop_phase` can be used.

- :py:meth:`fringestop_phase`
- :py:meth:`fringestop`

Miscellaneous
=============

- :py:meth:`invert_no_zero`
- :py:meth:`ensure_list`
"""

import datetime
import numpy as np
import scipy.linalg as la
import re
from typing import Tuple

from caput import pfb
from caput.interferometry import projected_distance, fringestop_phase

from ch_util import ephemeris

# Currently the position between the Pathfinder and 26m have been
# calibrated with holography, but positions between CHIME and
# Pathfinder/26m have not (they were determined from high-res
# satellite images and are only approximate).  We need to
# use CHIME holography data to constrain distance [x, y, z] between
# CHIME and 26m.  I then recommend defining our coordinate system
# such that center of CHIME array is the origin (so leaving variable
# _CHIME_POS alone, and updating _PF_POS and _26M_POS appropriately.)

# CHIME geometry
_CHIME_POS = [0.0, 0.0, 0.0]
# CHIME rotation from north. Anti-clockwise looking at the ground (degrees).
# See DocLib #695 for more information.
_CHIME_ROT = -0.071

# 26m geometry
_26M_POS = [254.162124, 21.853934, 18.93]
_26M_B = 2.14  # m

# Pathfinder geometry
_PF_POS = [373.754961, -54.649866, 0.0]
_PF_ROT = 1.986  # Pathfinder rotation from north (towards west) in degrees
_PF_SPACE = 22.0  # Pathfinder cylinder spacing

# KKO geometry
_KKO_POS = [0.0, 0.0, 0.0]
_KKO_ROT = 0.6874
_KKO_ROLL = 0.5888
_PCO_POS = _KKO_POS
_PCO_ROT = _KKO_ROT  # Aliases for backwards-compatibility
# KKO_ROT = rotation of cylinder axis from North. Anti-clockwise looking at the ground (degrees).
# KKO_ROLL = roll of cylinder toward east from Vertical. Anti-clockwise looking North along the focal line.
# See Doclib #1530 and #1121 for more information.

# GBO geometry
_GBO_POS = [0.0, 0.0, 0.0]
_GBO_ROT = -27.3745
_GBO_ROLL = -30.0871

# HCO geometry
_HCO_POS = [0.0, 0.0, 0.0]
_HCO_ROT = -0.8023
_HCO_ROLL = 1.0556


# Lat/Lon
_LAT_LON = {
    "chime": [49.3207125, -119.623670],
    "pathfinder": [49.3202245, -119.6183635],
    "galt_26m": [49.320909, -119.620174],
    "gbo_tone": [38.4292962636, -79.8451625395],
}

# Classes
# =======


class HKInput(object):
    """A housekeeping input.

    Parameters
    ----------
    atmel : :obj:`layout.component`
       The ATMEL board.
    chan : int
        The channel number.
    mux : int
        The mux number; if this HK stream has no multiplexer, this will simply
        remain as :obj:`Null`

    Attributes
    ----------
    atmel : :obj:`layout.component`
       The ATMEL board.
    chan : int
        The channel number.
    mux : int
        The mux number; if this HK stream has no multiplexer, this will simply
        remain as :obj:`Null`
    """

    atmel = None
    chan = None
    mux = None

    def __init__(self, atmel=None, chan=None, mux=None):
        self.atmel = atmel
        self.chan = chan
        self.mux = mux

    def __repr__(self):
        ret = "<HKInput atmel=%s chan=%d " % (self.atmel.sn, self.chan)
        if self.mux:
            ret += "mux=%d>" % self.mux
        else:
            ret += "(no mux)>"
        return ret


class CorrInput(object):
    """Base class for describing a correlator input.

    Meant to be subclassed by actual types of inputs.

    Attributes
    ----------
    input_sn : str
        Unique serial number of input.
    corr : str
        Unique serial number of correlator.
        Set to `None` if no correlator is connected.
    corr_order : int
        Order of input for correlator internal datastream.
    crate : int
        Crate number within the correlator.
        Set to `None` if correlator consists of single crate.
    slot : int
        Slot number of the fpga motherboard within the crate.
        Ranges from 0 to 15, left to right.
        Set to `None` if correlator consists of single slot.
    sma : int
        SMA number on the fpga motherboard within the slot.
        Ranges from 0 to 15, bottom to top.
    """

    def __init__(self, **input_dict):
        import inspect

        for basecls in inspect.getmro(type(self))[::-1]:
            for k, attr in basecls.__dict__.items():
                if k[0] != "_":
                    if not isinstance(attr, property):
                        self.__dict__[k] = input_dict.get(k, None)

                    elif attr.fset is not None:
                        attr.fset(self, input_dict.get(k, None))

    def _attribute_strings(self):
        prop = [
            (k, getattr(self, k))
            for k in ["id", "crate", "slot", "sma", "corr_order", "delay"]
        ]

        kv = ["%s=%s" % (k, repr(v)) for k, v in prop if v is not None] + [
            "%s=%s" % (k, repr(v)) for k, v in self.__dict__.items() if k[0] != "_"
        ]

        return kv

    def __repr__(self):
        kv = self._attribute_strings()

        return "%s(%s)" % (self.__class__.__name__, ", ".join(kv))

    @property
    def id(self):
        """Channel ID. Automatically calculated from the serial number
        if id is not explicitly set.

        Returns
        -------
        id : int
            Channel id. Calculated from the serial.
        """
        if hasattr(self, "_id"):
            return self._id
        else:
            return serial_to_id(self.input_sn)

    @id.setter
    def id(self, val):
        if val is not None:
            self._id = val

    @property
    def corr_order(self):
        return serial_to_location(self.input_sn)[0]

    @property
    def crate(self):
        return serial_to_location(self.input_sn)[1]

    @property
    def slot(self):
        return serial_to_location(self.input_sn)[2]

    @property
    def sma(self):
        return serial_to_location(self.input_sn)[3]

    @property
    def delay(self):
        """The delay along the signal chain in seconds.

        Postive delay values mean signals arriving later than the nominal value.

        Note that these are always relative. Here CHIME inputs are chosen as
        the delay=0 reference.
        """
        return getattr(self, "_delay", 0)

    input_sn = None
    corr = None


class Blank(CorrInput):
    """Unconnected input."""

    pass


class Antenna(CorrInput):
    """An antenna input.

    Attributes
    ----------
    reflector : str
        The name of the reflector the antenna is on.
    antenna : str
        Serial number of the antenna.
    rf_thru : str
        Serial number of the RF room thru that
        the connection passes.
    """

    reflector = None
    antenna = None
    rf_thru = None


class RFIAntenna(Antenna):
    """RFI monitoring antenna"""

    pass


class NoiseSource(CorrInput):
    """Broad band noise calibration source."""

    pass


class ArrayAntenna(Antenna):
    """Antenna that is part of a cylindrical interferometric array.

    Attributes
    ----------
    cyl : int
        Index of the cylinder.
    pos : [x, y, z]
        Position of the antenna in meters in right-handed coordinates
        where x is eastward, y is northward, and z is upward.
    pol : str
        Orientation of the polarisation.
    flag : bool
        Flag indicating whether or not the antenna is good.
    """

    _rotation = 0.0
    _roll = 0.0
    _offset = [0.0] * 3

    cyl = None
    pol = None
    flag = None

    def _attribute_strings(self):
        kv = super(ArrayAntenna, self)._attribute_strings()
        if self.pos is not None:
            pos = ", ".join(["%0.2f" % pp for pp in self.pos])
            kv.append("pos=[%s]" % pos)
        return kv

    @property
    def pos(self):
        if hasattr(self, "_pos"):
            pos = self._pos

            if self._rotation:
                t = np.radians(self._rotation)
                c, s = np.cos(t), np.sin(t)

                pos = [c * pos[0] - s * pos[1], s * pos[0] + c * pos[1], pos[2]]

            if any(self._offset):
                pos = [pos[dim] + off for dim, off in enumerate(self._offset)]

            return pos

        else:
            return None

    @pos.setter
    def pos(self, val):
        if (val is not None) and hasattr(val, "__iter__") and (len(val) > 1):
            self._pos = [0.0] * 3
            for ind, vv in enumerate(val):
                self._pos[ind] = vv


class PathfinderAntenna(ArrayAntenna):
    """Antenna that is part of the Pathfinder.

    Attributes
    ----------
    powered : bool
        Flag indicating that the antenna is powered.
    """

    _rotation = _PF_ROT
    _offset = _PF_POS

    # The delay relative to other inputs isn't really known. Set to NaN so we
    # don't make any mistakes
    _delay = np.nan

    powered = None


class CHIMEAntenna(ArrayAntenna):
    """Antenna that is part of CHIME."""

    _rotation = _CHIME_ROT
    _offset = _CHIME_POS
    _delay = 0  # Treat CHIME antennas as defining the delay zero point


class KKOAntenna(ArrayAntenna):
    """KKO outrigger antenna for the CHIME/FRB project."""

    _rotation = _KKO_ROT
    _roll = _KKO_ROLL
    _offset = _KKO_POS
    _delay = np.nan


PCOAntenna = KKOAntenna  # Alias for backwards-compatibility


class GBOAntenna(ArrayAntenna):
    """GBO outrigger antenna for the CHIME/FRB project."""

    _rotation = _GBO_ROT
    _roll = _GBO_ROLL
    _offset = _GBO_POS
    _delay = np.nan


class HCOAntenna(ArrayAntenna):
    """HCRO outrigger antenna for the CHIME/FRB project."""

    _rotation = _HCO_ROT
    _roll = _HCO_ROLL
    _offset = _HCO_POS
    _delay = np.nan


class TONEAntenna(ArrayAntenna):
    """Antenna that is part of GBO/TONE Outrigger.
    Let's allow for a global rotation and offset.
    """

    _rotation = 0.00
    _offset = [0.00, 0.00, 0.00]
    _delay = np.nan


class HolographyAntenna(Antenna):
    """Antenna used for holography.

    Attributes
    ----------
    pos : [x, y, z]
        Position of the antenna in meters in right-handed coordinates
        where x is eastward, y is northward, and z is upward.
    pol : str
        Orientation of the polarisation.
    """

    pos = None
    pol = None
    _delay = 1.475e-6  # From doclib:1093


# Private Functions
# =================


def _ensure_graph(graph):
    from . import layout

    try:
        graph.sg_spec
    except:
        graph = layout.graph(graph)
    return graph


def _get_feed_position(lay, rfl, foc, cas, slt, slot_factor):
    """Calculate feed position from node properties.

    Parameters
    ----------
    lay : layout.graph
        Layout instance to search from.
    rfl : layout.component
        Reflector.
    foc : layout.component
        Focal line slot.
    cas : layout.component
        Cassette.
    slt : layout.component
        Cassette slot.
    slot_factor : float
        1.5 for CHIME, 0.5 for Outriggers

    Returns
    -------
    pos : list
        x,y,z coordinates of the feed relative to the centre of the focal line.
    """
    try:
        pos = [0.0] * 3

        for node in [rfl, foc, cas, slt]:
            prop = lay.node_property(node)

            for ind, dim in enumerate(["x_offset", "y_offset", "z_offset"]):
                if dim in prop:
                    pos[ind] += float(prop[dim].value)  # in metres

        if "y_offset" not in lay.node_property(slt):
            pos[1] += (float(slt.sn[-1]) - slot_factor) * 0.3048

    except:
        pos = None

    return pos


def _get_input_props(lay, corr_input, corr, rfl_path, rfi_antenna, noise_source):
    """Fetch all the required properties of an ADC channel or correlator input.

    Parameters
    ----------
    lay : layout.graph
        Layout instance to search from.
    corr_input : layout.component
        ADC channel or correlator input.
    corr : layout.component
        Correlator.
    rfl_path : [layout.component]
        Path from input to reflector, or None.
    rfi_antenna : layout.component
        Closest RFI antenna
    noise_source : layout.component
        Closest noise source.

    Returns
    -------
    channel : CorrInput
        An instance of `CorrInput` containing the channel properties.
    """

    if corr is not None:
        corr_sn = corr.sn
    else:
        corr_sn = None

    # Check if the correlator input component contains a chan_id property
    corr_prop = lay.node_property(corr_input)
    chan_id = int(corr_prop["chan_id"].value) if "chan_id" in corr_prop else None

    rfl = None
    cas = None
    slt = None
    ant = None
    pol = None
    rft = None
    if rfl_path is not None:
        rfl = rfl_path[-1]

        def find(name):
            f = [a for a in rfl_path[1:-1] if a.type.name == name]
            return f[0] if len(f) == 1 else None

        foc = find("focal line slot")
        cas = find("cassette")
        slt = find("cassette slot")
        ant = find("antenna")
        pol = find("polarisation")

        for rft_name in ["rf room thru", "RFT thru"]:
            rft = find(rft_name)
            if rft is not None:
                break

    # If the antenna does not exist, it might be the RFI antenna, the noise source, or empty
    if ant is None:
        if rfi_antenna is not None:
            rfl = lay.closest_of_type(
                rfi_antenna,
                "reflector",
                type_exclude=["correlator card slot", "ADC board"],
            )
            rfl_sn = rfl.sn if rfl is not None else None
            return RFIAntenna(
                id=chan_id,
                input_sn=corr_input.sn,
                corr=corr_sn,
                reflector=rfl_sn,
                antenna=rfi_antenna.sn,
            )

        # Check to see if it is a noise source
        if noise_source is not None:
            return NoiseSource(id=chan_id, input_sn=corr_input.sn, corr=corr_sn)

        # If we get to here, it's probably a blank input
        return Blank(id=chan_id, input_sn=corr_input.sn, corr=corr_sn)

    # Determine polarization from antenna properties
    try:
        keydict = {
            "H": "hpol_orient",
            "V": "vpol_orient",
            "1": "pol1_orient",
            "2": "pol2_orient",
        }

        pkey = keydict[pol.sn[-1]]
        pdir = lay.node_property(ant)[pkey].value

    except:
        pdir = None

    # Determine serial number of RF thru
    rft_sn = getattr(rft, "sn", None)

    # If the cassette does not exist, must be holography antenna
    if slt is None:
        return HolographyAntenna(
            id=chan_id,
            input_sn=corr_input.sn,
            corr=corr_sn,
            reflector=rfl.sn,
            pol=pdir,
            antenna=ant.sn,
            rf_thru=rft_sn,
            pos=_26M_POS,
        )

    # If we are still here, we are a CHIME/Pathfinder feed

    # Determine if the correlator input has been manually flagged as good or bad
    flag = (
        bool(int(corr_prop["manual_flag"].value))
        if "manual_flag" in corr_prop
        else True
    )

    # Map the cylinder name in the database into a number. This might
    # be worth changing, such that we could also map into letters
    # (i.e. A, B, C, D) to save confusion.
    pos_dict = {
        "W_cylinder": 0,
        "E_cylinder": 1,
        "cylinder_A": 2,
        "cylinder_B": 3,
        "cylinder_C": 4,
        "cylinder_D": 5,
        "pco_cylinder": 6,
        "gbo_cylinder": 7,
        "hcro_cylinder": 8,
    }

    cyl = pos_dict[rfl.sn]

    # Different conventions for CHIME, PCO, GBO, HCRO, and Pathfinder
    if cyl >= 2 and cyl <= 5:
        # Dealing with a CHIME feed

        # Determine position
        pos = _get_feed_position(
            lay=lay, rfl=rfl, foc=foc, cas=cas, slt=slt, slot_factor=1.5
        )

        # Return CHIMEAntenna object
        return CHIMEAntenna(
            id=chan_id,
            input_sn=corr_input.sn,
            corr=corr_sn,
            reflector=rfl.sn,
            cyl=cyl,
            pos=pos,
            pol=pdir,
            antenna=ant.sn,
            rf_thru=rft_sn,
            flag=flag,
        )

    elif cyl == 0 or cyl == 1:
        # Dealing with a pathfinder feed

        # Determine y_offset
        try:
            pos = [0.0] * 3

            pos[0] = cyl * _PF_SPACE

            cas_prop = lay.node_property(cas)
            slt_prop = lay.node_property(slt)

            d1 = float(cas_prop["dist_to_n_end"].value) / 100.0  # in metres
            d2 = float(slt_prop["dist_to_edge"].value) / 100.0  # in metres
            orient = cas_prop["slot_zero_pos"].value

            pos[1] = d1 + d2 if orient == "N" else d1 - d2

            # Turn into distance increasing from South to North.
            pos[1] = 20.0 - pos[1]

        except:
            pos = None

        # Try and determine if the FLA is powered or not. Paths without an
        # FLA (e.g. RFoF paths) are assumed to be powered on.
        pwd = True

        if rft is not None:
            rft_prop = lay.node_property(rft)

            if "powered" in rft_prop:
                pwd = rft_prop["powered"].value
                pwd = bool(int(pwd))

        # Return PathfinderAntenna object
        return PathfinderAntenna(
            id=chan_id,
            input_sn=corr_input.sn,
            corr=corr_sn,
            reflector=rfl.sn,
            cyl=cyl,
            pos=pos,
            pol=pdir,
            antenna=ant.sn,
            rf_thru=rft_sn,
            powered=pwd,
            flag=flag,
        )

    elif cyl == 6:
        # Dealing with an KKO feed

        # Determine position
        pos = _get_feed_position(
            lay=lay, rfl=rfl, foc=foc, cas=cas, slt=slt, slot_factor=0.5
        )

        # Return KKOAntenna object
        return KKOAntenna(
            id=chan_id,
            input_sn=corr_input.sn,
            corr=corr_sn,
            reflector=rfl.sn,
            cyl=cyl,
            pos=pos,
            pol=pdir,
            antenna=ant.sn,
            rf_thru=rft_sn,
            flag=flag,
        )

    elif cyl == 7:
        # Dealing with a GBO feed

        # Determine position
        pos = _get_feed_position(
            lay=lay, rfl=rfl, foc=foc, cas=cas, slt=slt, slot_factor=0.5
        )

        # Return GBOAntenna object
        return GBOAntenna(
            id=chan_id,
            input_sn=corr_input.sn,
            corr=corr_sn,
            reflector=rfl.sn,
            cyl=cyl,
            pos=pos,
            pol=pdir,
            antenna=ant.sn,
            rf_thru=rft_sn,
            flag=flag,
        )

    elif cyl == 8:
        # Dealing with a HCO feed

        # Determine position
        pos = _get_feed_position(
            lay=lay, rfl=rfl, foc=foc, cas=cas, slt=slt, slot_factor=0.5
        )

        # Return HCOAntenna object
        return HCOAntenna(
            id=chan_id,
            input_sn=corr_input.sn,
            corr=corr_sn,
            reflector=rfl.sn,
            cyl=cyl,
            pos=pos,
            pol=pdir,
            antenna=ant.sn,
            rf_thru=rft_sn,
            flag=flag,
        )


# Public Functions
# ================


def calibrate_temperature(raw):
    """Calibrate housekeeping temperatures.

    The offset used here is rough; the results are therefore not absolutely
    precise.

    Parameters
    ----------
    raw : numpy array
        The raw values.

    Returns
    -------
    t : numpy array
        The temperature in degrees Kelvin.
    """
    import numpy

    off = 150.0
    r_t = 2000.0 * (8320.0 / (raw - off) - 1.0)
    return 1.0 / (1.0 / 298.0 + numpy.log(r_t / 1.0e4) / 3950.0)


def antenna_to_lna(graph, ant, pol):
    """Find an LNA connected to an antenna.

    Parameters
    ----------
    graph : obj:`layout.graph` or :obj:`datetime.datetime`
        The graph in which to do the search. If you pass a time, then the graph
        will be constructed internally. (Note that the latter option will be
        quite slow if you do repeated calls!)
    ant : :obj:`layout.component`
        The antenna.
    pol : integer
        There can be up to two LNA's connected to the two polarisation outputs
        of an antenna. Select which by passing :obj:`1` or :obj:`2`. (Note that
        conversion to old-style naming 'A' and 'B' is done automatically.)

    Returns
    -------
    lna : :obj:`layout.component` or string
        The LNA.

    Raises
    ------
    :exc:`layout.NotFound`
        Raised if the polarisation connector could not be found in the graph.
    """
    from . import layout

    graph = _ensure_graph(graph)
    pol_obj = None
    for p in graph.neighbour_of_type(graph.component(comp=ant), "polarisation"):
        if p.sn[-1] == str(pol) or p.sn[-1] == chr(ord("A") + pol):
            pol_obj = p
            break
    if not pol_obj:
        raise layout.NotFound
    try:
        return graph.neighbour_of_type(pol_obj, "LNA")[0]
    except IndexError:
        return None


def lna_to_antenna(graph, lna):
    """Find an antenna connected to an LNA.

    Parameters
    ----------
    graph : obj:`layout.graph` or :obj:`datetime.datetime`
        The graph in which to do the search. If you pass a time, then the graph
        will be constructed internally. (Note that the latter option will be
        quite slow if you do repeated calls!)
    lna : :obj:`layout.component` or string
        The LNA.

    Returns
    -------
    antenna : :obj:`layout.component`
        The antenna.
    """
    graph = _ensure_graph(graph)
    return graph.closest_of_type(
        graph.component(comp=lna), "antenna", type_exclude="60m coax"
    )


def sensor_to_hk(graph, comp):
    """Find what housekeeping channel a component is connected to.

    Parameters
    ----------
    graph : obj:`layout.graph` or :obj:`datetime.datetime`
        The graph in which to do the search. If you pass a time, then the graph
        will be constructed internally. (Note that the latter option will be
        quite slow if you do repeated calls!)
    comp : :obj:`layout.component` or string
        The component to search for (you can pass by serial number if you wish).
        Currently, only components of type LNA, FLA and RFT thru are accepted.

    Returns
    -------
    inp : :obj:`HKInput`
        The housekeeping input channel the sensor is connected to.
    """
    graph = _ensure_graph(graph)
    comp = graph.component(comp=comp)

    if comp.type.name == "LNA":
        # Find the closest mux.
        mux = graph.closest_of_type(
            comp, "HK mux", type_exclude=["polarisation", "cassette", "60m coax"]
        )
        if not mux:
            return None
        try:
            hydra = graph.neighbour_of_type(comp, "HK hydra")[0]
        except IndexError:
            return None
        chan = int(hydra.sn[-1])
        if mux.sn[-1] == "B":
            chan += 8

        # Find the ATMEL board.
        atmel = graph.closest_of_type(
            hydra, "HK ATMega", type_exclude=["cassette", "antenna"]
        )

        return HKInput(atmel, chan, int(mux.sn[-2]))

    elif comp.type.name == "FLA" or comp.type.name == "RFT thru":
        if comp.type.name == "FLA":
            try:
                comp = graph.neighbour_of_type(comp, "RFT thru")[0]
            except IndexError:
                return None
        try:
            hydra = graph.neighbour_of_type(comp, "HK hydra")[0]
        except IndexError:
            return None

        # Find the ATMEL board.
        atmel = graph.closest_of_type(
            hydra, "HK ATMega", type_exclude=["RFT thru", "FLA", "SMA coax"]
        )

        return HKInput(atmel, int(hydra.sn[-1]), None)
    else:
        raise ValueError("You can only pass components of type LNA, FLA or RFT thru.")


def hk_to_sensor(graph, inp):
    """Find what component a housekeeping channel is connected to.

    This method is for finding either LNA or FLA's that your housekeeping
    channel is connected to. (It currently cannot find accelerometers, other
    novel housekeeping instruments that may later exist; nor will it work if the
    FLA/LNA is connected via a very non-standard chain of components.)

    Parameters
    ----------
    graph : obj:`layout.graph` or :obj:`datetime.datetime`
        The graph in which to do the search. If you pass a time, then the graph
        will be constructed internally. (Note that the latter option will be
        quite slow if you do repeated calls!)
    inp : :obj:`HKInput`
        The housekeeping input to search.

    Returns
    -------
    comp : :obj:`layout.component`
        The LNA/FLA connected to the specified channel; :obj:`None` is returned
        if none is found.

    Raises
    ------
    :exc:`ValueError`
        Raised if one of the channels or muxes passed in **hk_chan** is out of
        range.
    """

    from . import layout

    graph = _ensure_graph(graph)

    # Figure out what it is connected to.
    for thing in graph.neighbours(graph.component(comp=inp.atmel)):
        if thing.type.name == "HK preamp":
            # OK, this is a preamp going to FLA's.
            if inp.chan < 0 or inp.chan > 7:
                raise ValueError(
                    "For FLA housekeeping, the channel number "
                    "must be in the range [0, 7]."
                )
            for hydra in graph.neighbour_of_type(thing, "HK hydra"):
                if hydra.sn[-1] == str(inp.chan):
                    return graph.closest_of_type(hydra, "FLA", type_exclude="HK preamp")

        if thing.type.name == "HK mux box":
            # OK, this is a mux box going to LNA's.
            if inp.mux < 0 or inp.mux > 7:
                raise ValueError(
                    "For LNA housekeeping, the mux number must be "
                    "in the range [0, 7]."
                )
            if inp.chan < 0 or inp.chan > 15:
                raise ValueError(
                    "For LNA housekeeping, the channel number "
                    "must be in the range [0, 15]."
                )

            # Construct the S/N of the mux connector and get it.
            sn = "%s%d%s" % (thing.sn, inp.mux, "A" if inp.chan < 8 else "B")
            try:
                mux_card = graph.component(comp=sn)
            except layout.NotFound:
                return None

            # Find the closest preamp and the hydra cable corresponding to the
            # channel requested.
            preamp = graph.closest_of_type(
                mux_card, "HK preamp", type_exclude="HK mux box"
            )
            if not preamp:
                return None

            for hydra in graph.neighbour_of_type(preamp, "HK hydra"):
                if hydra.sn[-1] == str(inp.chan % 8):
                    try:
                        return graph.neighbour_of_type(hydra, "LNA")[0]
                    except IndexError:
                        return None
    return None


# Parse a serial number into crate, slot, and sma number
def parse_chime_serial(sn):
    mo = re.match("FCC(\d{2})(\d{2})(\d{2})", sn)

    if mo is None:
        raise RuntimeError(
            "Serial number %s does not match expected CHIME format." % sn
        )

    crate = int(mo.group(1))
    slot = int(mo.group(2))
    sma = int(mo.group(3))

    return crate, slot, sma


def parse_pathfinder_serial(sn):
    mo = re.match("(\w{6}\-\d{4})(\d{2})(\d{2})", sn)

    if mo is None:
        raise RuntimeError(
            "Serial number %s does not match expected Pathfinder format." % sn
        )

    crate = mo.group(1)
    slot = int(mo.group(2))
    sma = int(mo.group(3))

    return crate, slot, sma


def parse_old_serial(sn):
    mo = re.match("(\d{5}\-\d{4}\-\d{4})\-C(\d{1,2})", sn)

    if mo is None:
        raise RuntimeError(
            "Serial number %s does not match expected 8/16 channel format." % sn
        )

    slot = mo.group(1)
    sma = int(mo.group(2))

    return slot, sma


def serial_to_id(serial):
    """Get the channel ID corresponding to a correlator input serial number.

    Parameters
    ----------
    serial : string
        Correlator input serial number.

    Returns
    -------
    id : int
    """

    # Map a slot and SMA to channel id for Pathfinder
    def get_pathfinder_channel(slot, sma):
        c = [
            None,
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
        channel = c[slot] + sma if slot > 0 else sma
        return channel

    # Determine ID
    try:
        res = parse_chime_serial(serial)
        # CHIME chan_id is defined in layout database
        return -1
    except RuntimeError:
        pass

    try:
        res = parse_pathfinder_serial(serial)
        return get_pathfinder_channel(*(res[1:]))
    except RuntimeError:
        pass

    try:
        res = parse_old_serial(serial)
        return res[1]
    except RuntimeError:
        pass

    return -1


def serial_to_location(serial):
    """Get the internal correlator ordering and the
    crate, slot, and sma number from a correlator input serial number.

    Parameters
    ----------
    serial : string
        Correlator input serial number.

    Returns
    -------
    location : 4-tuple
        (corr_order, crate, slot, sma)
    """

    default = (None,) * 4
    if serial is None:
        return default

    # Map slot and sma to position within
    def get_crate_channel(slot, sma):
        sma_to_adc = [12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]
        return slot * 16 + sma_to_adc[sma]

    # Determine ID
    try:
        res = parse_chime_serial(serial)
        corr_id = res[0] * 256 + get_crate_channel(*res[1:])
        return (corr_id,) + res
    except RuntimeError:
        pass

    try:
        res = list(parse_pathfinder_serial(serial))
        # Use convention that slot number starts at 0 for consistency with CHIME
        res[1] -= 1
        corr_id = get_crate_channel(*res[1:])
        return (corr_id, None) + tuple(res[1:])
    except RuntimeError:
        pass

    try:
        res = parse_old_serial(serial)
        return (res[1], None, None, res[1])
    except RuntimeError:
        pass

    return default


def get_default_frequency_map_stream() -> Tuple[np.ndarray]:
    """Get the default CHIME frequency map stream.

    Level order is [shuffle, crate, slot, link].

    Returns
    -------
    stream
        [shuffle, crate, slot, link] for each frequency bin
    stream_id
        stream_id for each map combination
        shuffle*2**12 + crate*2**8 + slot*2**4 + link
    """
    stream = np.empty((1024, 4), dtype=np.int32)

    # shuffle
    stream[:, 0] = 3
    # crate
    stream[:, 1] = np.tile(np.arange(2).repeat(16), 32)
    # slot
    stream[:, 2] = np.tile(np.arange(16), 64)
    # link
    stream[:, 3] = np.tile(np.arange(8).repeat(32), 4)

    stream_id = (
        stream[:, 0] * 2**12 + stream[:, 1] * 2**12 + stream[:, 2] * 2**4 + stream[:, 3]
    ).astype(np.int64)

    return stream, stream_id


def order_frequency_map_stream(fmap: np.ndarray, stream_id: np.ndarray) -> np.ndarray:
    """Order stream_id components based on a frequency map.

    Level order is [shuffle, crate, slot, link]

    Parameters
    ----------
    fmap
        frequency map
    stream_id
        1-D array of stream_ids associated with each row in fmap

    Returns
    -------
    stream
        shuffle, crate, slot, link for each frequency
    """

    def decode_stream_id(sid: int) -> Tuple[int]:
        link = sid & 15
        slot = (sid >> 4) & 15
        crate = (sid >> 8) & 15
        shuffle = (sid >> 12) & 15

        return (shuffle, crate, slot, link)

    decoded_stream = [decode_stream_id(i) for i in stream_id[:]]
    x = [[] for _ in range(len(stream_id))]

    for ii, freqs in enumerate(fmap):
        for f in freqs:
            x[f].append(decoded_stream[ii])

    # TODO: maybe implement some checks here
    stream = np.array([i[0] for i in x], dtype=np.int32)

    return stream


def get_correlator_inputs(lay_time, correlator=None, connect=True):
    """Get the information for all channels in a layout.

    Parameters
    ----------
    lay_time : layout.graph or datetime
        layout.graph object, layout tag id, or datetime.
    correlator : str, optional
        Fetch only for specified correlator. Use the serial number in database,
        or `pathfinder` or `chime`, which will substitute the correct serial.
        If `None` return for all correlators.
        Option `tone` added for GBO 12 dish outrigger prototype array.
    connect : bool, optional
        Connect to database and set the user to Jrs65 prior to query.
        Default is True.

    Returns
    -------
    channels : list
        List of :class:`CorrInput` instances. Returns `None` for MPI ranks
        other than zero.
    """

    from ch_util import layout
    import networkx as nx
    from chimedb.core.connectdb import connect_this_rank

    coax_type = ["SMA coax", "3.25m SMA coax"]

    block = [
        "correlator card slot",
        "ADC board",
        "rf room bulkhead",
        "c-can bulkhead",
        "50m coax bundle",
        "HK hydra",
        "connector plate pol 1",
        "connector plate pol 2",
        "thermometer",
    ]

    # Replace 'pathfinder' or 'chime' with serial number
    if isinstance(correlator, str):
        if correlator.lower() == "pathfinder":
            correlator = "K7BP16-0004"
        elif correlator.lower() == "chime":
            correlator = "FCC"
        elif correlator.lower() == "pco":
            correlator = "FCA"
        elif correlator.lower() == "kko":
            correlator = "FCA"
        elif correlator.lower() == "gbo":
            correlator = "FCG"
        elif correlator.lower() == "tone":
            # A hack to return GBO correlator inputs
            correlator = "tone"
            connect = False
            laytime = 0
            return fake_tone_database()

    if not connect_this_rank():
        return None

    if connect:
        layout.connect_database(read_write=False)
        layout.set_user("Jrs65")

    # Fetch layout_tag start time if we received a layout num
    if isinstance(lay_time, int):
        raise ValueError("Layout IDs are no longer supported.")
    elif isinstance(lay_time, datetime.datetime):
        layout_graph = layout.graph.from_db(lay_time)
    elif isinstance(lay_time, layout.graph):
        layout_graph = lay_time
    else:
        raise ValueError("Unsupported argument lay_time=%s" % repr(lay_time))

    # Fetch all the input components
    inputs = []
    try:
        inputs += layout_graph.component(type="ADC channel")
    except layout.NotFound:
        pass

    try:
        inputs += layout_graph.component(type="correlator input")
    except layout.NotFound:
        pass

    # Restrict the inputs processed to only those directly connected to the
    # specified correlator
    if correlator is not None:
        try:
            corr = layout_graph.component(correlator)
        except layout.NotFound:
            raise ValueError("Unknown correlator %s" % correlator)

        # Cut out SMA coaxes so we don't go outside of the correlator
        sg = set(layout_graph.nodes())
        for coty in coax_type:
            try:
                comp_coty = layout_graph.component(type=coty)
            except layout.NotFound:
                pass
            else:
                sg -= set(comp_coty)
        sg = layout_graph.subgraph(sg)

        # Use only inputs that are connected to the correlator
        inputs = nx.node_connected_component(sg, corr) & set(inputs)

    inputs = sorted(inputs, key=lambda adc: adc.sn)

    # Perform nearly all the graph queries in one huge batcn to speed things up,
    # and pass the results into _get_input_props for further processing
    corrs = layout_graph.closest_of_type(inputs, "correlator", type_exclude=coax_type)

    rfls = layout_graph.shortest_path_to_type(inputs, "reflector", type_exclude=block)

    block.append("reflector")
    rfi_ants = layout_graph.closest_of_type(inputs, "RFI antenna", type_exclude=block)
    noise_sources = layout_graph.closest_of_type(
        inputs, "noise source", type_exclude=block
    )

    inputlist = [
        _get_input_props(layout_graph, *args)
        for args in zip(inputs, corrs, rfls, rfi_ants, noise_sources)
    ]

    # Filter to include only inputs attached to the given correlator. In theory
    # this shouldn't be necessary if the earlier filtering worked, but I think
    # it'll help catch some odd cases
    if correlator is not None:
        inputlist = [input_ for input_ in inputlist if input_.corr == correlator]

    # Sort by channel ID
    inputlist.sort(key=lambda input_: input_.id)

    return inputlist


def change_pathfinder_location(rotation=None, location=None, default=False):
    """Change the orientation or location of Pathfinder.

    Parameters
    ----------
    rotation : float
        Rotation of the telescope from true north in degrees.
    location: list
        [x, y, z] of the telescope in meters,
        where x is eastward, y is northward, and z is upward.
    default:  bool
        Set parameters back to default value.  Overides other keywords.
    """

    if default:
        rotation = _PF_ROT
        location = _PF_POS

    if rotation is not None:
        PathfinderAntenna._rotation = rotation

    if location is not None:
        offset = [location[ii] if ii < len(location) else 0.0 for ii in range(3)]
        PathfinderAntenna._offset = offset


def change_chime_location(rotation=None, location=None, default=False):
    """Change the orientation or location of CHIME.

    Parameters
    ----------
    rotation : float
        Rotation of the telescope from true north in degrees.
    location: list
        [x, y, z] of the telescope in meters,
        where x is eastward, y is northward, and z is upward.
    default: bool
        Set parameters back to default value.  Overides other keywords.
    """

    if default:
        rotation = _CHIME_ROT
        location = _CHIME_POS

    if rotation is not None:
        CHIMEAntenna._rotation = rotation

    if location is not None:
        offset = [location[ii] if ii < len(location) else 0.0 for ii in range(3)]
        CHIMEAntenna._offset = offset


def get_feed_positions(feeds, get_zpos=False):
    """Get the positions of the CHIME antennas.

    Parameters
    ----------
    feeds : list of CorrInput
        List of feeds to compute positions of.
    get_zpos: bool
        Return a third column with elevation information.

    Returns
    -------
    positions : np.ndarray[nfeed, 2]
        Array of feed positions. The first column is the E-W position
        (increasing to the E), and the second is the N-S position (increasing
        to the N). Non CHIME feeds get set to `NaN`.
    """

    # Extract positions for all array antennas or holographic antennas, fill other
    # inputs with NaNs
    pos = np.array(
        [
            feed.pos if (is_array(feed) or is_holographic(feed)) else [np.nan] * 3
            for feed in feeds
        ]
    )

    # Drop z coordinate if not explicitely requested
    if not get_zpos:
        pos = pos[:, 0:2]

    return pos


def fake_tone_database():
    positions_and_polarizations = [
        ("S", [15.08, -1.61]),
        ("E", [15.08, -1.61]),
        ("S", [-9.19, -15.24]),
        ("E", [-9.19, -15.24]),
        ("S", [7.02, 14.93]),
        ("E", [7.02, 14.93]),
        ("S", [9.01, -5.02]),
        ("E", [9.01, -5.02]),
        ("S", [2.8, 2.67]),
        ("E", [2.8, 2.67]),
        ("S", [-1.66, 10.38]),
        ("E", [-1.66, 10.38]),
        ("S", [-7.63, -0.79]),
        ("E", [-7.63, -0.79]),
        ("S", [-15.43, -5.33]),
        ("E", [-15.43, -5.33]),
    ]
    inputs = []
    for id, pol_ns_ew in enumerate(positions_and_polarizations):
        inputs.append(
            TONEAntenna(
                id=id,
                crate=0,
                slot=0,
                sma=0,
                corr_order=0,
                input_sn=f"TONE{id:04}",
                corr="tone",
                reflector=None,
                antenna=f"ANT{id//2:04}",
                rf_thru="N/A",
                cyl=0,
                pol=pol_ns_ew[0],
                flag=True,
                pos=[pol_ns_ew[1][0], pol_ns_ew[1][1], 0],
            )
        )
    return inputs


def get_feed_polarisations(feeds):
    """Get an array of the feed polarisations.

    Parameters
    ----------
    feeds : list of CorrInput
        List of feeds to compute positions of.

    Returns
    -------
    pol : np.ndarray
        Array of characters giving polarisation. If not an array feed returns '0'.
    """
    pol = np.array([(f.pol if is_array(f) else "0") for f in feeds])

    return pol


def is_array(feed):
    """Is this feed part of an array?

    Parameters
    ----------
    feed : CorrInput

    Returns
    -------
    isarr : bool
    """
    return isinstance(feed, ArrayAntenna)


def is_array_x(feed):
    """Is this an X-polarisation antenna in an array?"""
    return is_array(feed) and feed.pol == "E"


def is_array_y(feed):
    """Is this a Y-polarisation antenna in an array?"""
    return is_array(feed) and feed.pol == "S"


def is_chime(feed):
    """Is this feed a CHIME antenna?

    Parameters
    ----------
    feed : CorrInput

    Returns
    -------
    ischime : bool
    """
    return isinstance(feed, CHIMEAntenna)


def is_pathfinder(feed):
    """Is this feed a Pathfinder antenna?

    Parameters
    ----------
    feed : CorrInput

    Returns
    -------
    ispathfinder : bool
    """
    return isinstance(feed, PathfinderAntenna)


def is_holographic(feed):
    """Is this feed a holographic antenna?

    Parameters
    ----------
    feed : CorrInput

    Returns
    -------
    isholo : bool
    """
    return isinstance(feed, HolographyAntenna)


def get_holographic_index(inputs):
    """Find the indices of the holography antennas.

    Parameters
    ----------
    inputs : list of :class:`CorrInput`

    Returns
    -------
    ixholo : list of int
        Returns None if holographic antenna not found.
    """
    ixholo = [ix for ix, inp in enumerate(inputs) if is_holographic(inp)]
    return ixholo or None


def is_noise_source(inp):
    """Is this correlator input connected to a noise source?

    Parameters
    ----------
    inp : CorrInput

    Returns
    -------
    isnoise : bool
    """
    return isinstance(inp, NoiseSource)


def get_noise_source_index(inputs):
    """Find the indices of the noise sources.

    Parameters
    ----------
    inputs : list of :class:`CorrInput`

    Returns
    -------
    ixns : list of int
        Returns None if noise source not found.
    """
    ixns = [ix for ix, inp in enumerate(inputs) if is_noise_source(inp)]
    return ixns or None


def get_noise_channel(inputs):
    """Returns the index of the noise source with
    the lowest chan id (for backwards compatability).
    """
    noise_sources = get_noise_source_index(inputs)
    return (noise_sources or [None])[0]


def is_array_on(inputs, *args):
    """Check if inputs are attached to an array antenna AND powered on AND flagged as good.

    Parameters
    ----------
    inputs : CorrInput or list of CorrInput objects

    Returns
    -------
    pwds : boolean or list of bools.
        If list, it is the same length as inputs. Value is True if input is
        attached to a ArrayAntenna *and* powered-on and False otherwise
    """

    if len(args) > 0:
        raise RuntimeError("This routine no longer accepts a layout time argument.")

    # Treat scalar case
    if isinstance(inputs, CorrInput):
        return (
            is_array(inputs)
            and getattr(inputs, "powered", True)
            and getattr(inputs, "flag", True)
        )

    # Assume that the argument is a sequence otherwise
    else:
        return [is_array_on(inp) for inp in inputs]


# Create an is_chime_on alias for backwards compatibility
is_chime_on = is_array_on


def reorder_correlator_inputs(input_map, corr_inputs):
    """Sort a list of correlator inputs into the order given in input map.

    Parameters
    ----------
    input_map : np.ndarray
        Index map of correlator inputs.
    corr_inputs : list
        List of :class:`CorrInput` objects, e.g. the output from
        :func:`get_correlator_inputs`.

    Returns
    -------
    corr_input_list: list
        List of :class:`CorrInput` instances in the new order. Returns `None`
        where the serial number had no matching entry in parameter ``corr_inputs``.
    """
    serials = input_map["correlator_input"]

    sorted_inputs = []

    for serial in serials:
        for corr_input in corr_inputs:
            if serial == corr_input.input_sn:
                sorted_inputs.append(corr_input)
                break
        else:
            sorted_inputs.append(None)

    return sorted_inputs


def redefine_stack_index_map(input_map, prod, stack, reverse_stack):
    """Ensure that only baselines between array antennas are used to represent the stack.

    The correlator will have inputs that are not connected to array antennas.  These inputs
    are flagged as bad and are not included in the stack, however, products that contain
    their `chan_id` can still be used to represent a characteristic baseline in the `stack`
    index map.  This method creates a new `stack` index map that, if possible, only contains
    products between two array antennas.  This new `stack` index map should be used when
    calculating baseline distances to fringestop stacked data.

    Parameters
    ----------
    input_map : list of :class:`CorrInput`
        List describing the inputs as they are in the file, output from
        `tools.get_correlator_inputs`
    prod : np.ndarray[nprod,] of dtype=('input_a', 'input_b')
        The correlation products as pairs of inputs.
    stack : np.ndarray[nstack,] of dtype=('prod', 'conjugate')
        The index into the `prod` axis of a characteristic baseline included in the stack.
    reverse_stack :  np.ndarray[nprod,] of dtype=('stack', 'conjugate')
        The index into the `stack` axis that each `prod` belongs.

    Returns
    -------
    stack_new : np.ndarray[nstack,] of dtype=('prod', 'conjugate')
        The updated `stack` index map, where each element is an index to a product
        consisting of a pair of array antennas.
    stack_flag : np.ndarray[nstack,] of dtype=bool
        Boolean flag that is True if this element of the stack index map is now valid,
        and False if none of the baselines that were stacked contained array antennas.
    """
    feed_flag = np.array([is_array(inp) for inp in input_map])
    example_prod = prod[stack["prod"]]
    stack_flag = feed_flag[example_prod["input_a"]] & feed_flag[example_prod["input_b"]]

    stack_new = stack.copy()

    bad_stack_index = np.flatnonzero(~stack_flag)
    for ind in bad_stack_index:
        this_stack = np.flatnonzero(reverse_stack["stack"] == ind)
        for ts in this_stack:
            tp = prod[ts]
            if feed_flag[tp[0]] and feed_flag[tp[1]]:
                stack_new[ind]["prod"] = ts
                stack_new[ind]["conjugate"] = reverse_stack[ts]["conjugate"]
                stack_flag[ind] = True
                break

    return stack_new, stack_flag


def cmap(i, j, n):
    """Given a pair of feed indices, return the pair index.

    Parameters
    ----------
    i, j : integer
        Feed index.
    n : integer
        Total number of feeds.

    Returns
    -------
    pi : integer
        Pair index.
    """
    if i <= j:
        return (n * (n + 1) // 2) - ((n - i) * (n - i + 1) // 2) + (j - i)
    else:
        return cmap(j, i, n)


def icmap(ix, n):
    """Inverse feed map.

    Parameters
    ----------
    ix : integer
        Pair index.
    n : integer
        Total number of feeds.

    Returns
    -------
    fi, fj : integer
        Feed indices.
    """
    for ii in range(n):
        if cmap(ii, n - 1, n) >= ix:
            break

    i = ii
    j = ix - cmap(i, i, n) + i
    return i, j


def unpack_product_array(prod_arr, axis=1, feeds=None):
    """Expand packed products to correlation matrices.

    This turns an axis of the packed upper triangle set of products into the
    full correlation matrices. It replaces the specified product axis with two
    axes, one for each feed. By setting `feeds` this routine can also
    pull out a subset of feeds.

    Parameters
    ----------
    prod_arr : np.ndarray[..., nprod, :]
        Array containing products packed in upper triangle format.
    axis : int, optional
        Axis the products are contained on.
    feeds : list of int, optional
        Indices of feeds to include. If :obj:`None` (default) use all feeds.

    Returns
    -------
    corr_arr : np.ndarray[..., nfeed, nfeed, ...]
        Expanded array.
    """

    nprod = prod_arr.shape[axis]
    nfeed = int((2 * nprod) ** 0.5)

    if nprod != (nfeed * (nfeed + 1) // 2):
        raise Exception(
            "Product axis size does not look correct (not exactly n(n+1)/2)."
        )

    shape0 = prod_arr.shape[:axis]
    shape1 = prod_arr.shape[(axis + 1) :]

    # Construct slice objects representing the axes before and after the product axis
    slice0 = (np.s_[:],) * len(shape0)
    slice1 = (np.s_[:],) * len(shape1)

    # If no feeds specified use all of them
    feeds = list(range(nfeed)) if feeds is None else feeds

    outfeeds = len(feeds)

    exp_arr = np.zeros(shape0 + (outfeeds, outfeeds) + shape1, dtype=prod_arr.dtype)

    # Iterate over products and copy into correct location of expanded array
    # Use a python loop, but should be fast if other axes are large
    for ii, fi in enumerate(feeds):
        for ij, fj in enumerate(feeds):
            pi = cmap(fi, fj, nfeed)

            if fi <= fj:
                exp_arr[slice0 + (ii, ij) + slice1] = prod_arr[slice0 + (pi,) + slice1]
            else:
                exp_arr[slice0 + (ii, ij) + slice1] = prod_arr[
                    slice0 + (pi,) + slice1
                ].conj()

    return exp_arr


def pack_product_array(exp_arr, axis=1):
    """Pack full correlation matrices into upper triangular form.

    It replaces the two feed axes of the matrix, with a single upper triangle product axis.


    Parameters
    ----------
    exp_arr : np.ndarray[..., nfeed, nfeed, ...]
        Array of full correlation matrices.
    axis : int, optional
        Index of the first feed axis. The second feed axis must be the next one.

    Returns
    -------
    prod_arr : np.ndarray[..., nprod, ...]
        Array containing products packed in upper triangle format.
    """

    nfeed = exp_arr.shape[axis]
    nprod = nfeed * (nfeed + 1) // 2

    if nfeed != exp_arr.shape[axis + 1]:
        raise Exception("Does not look like correlation matrices (axes must be equal).")

    shape0 = exp_arr.shape[:axis]
    shape1 = exp_arr.shape[(axis + 2) :]

    slice0 = (np.s_[:],) * len(shape0)
    slice1 = (np.s_[:],) * len(shape1)

    prod_arr = np.zeros(shape0 + (nprod,) + shape1, dtype=exp_arr.dtype)

    # Iterate over products and copy from correct location of expanded array
    for pi in range(nprod):
        fi, fj = icmap(pi, nfeed)

        prod_arr[slice0 + (pi,) + slice1] = exp_arr[slice0 + (fi, fj) + slice1]

    return prod_arr


def fast_pack_product_array(arr):
    """
    Equivalent to ch_util.tools.pack_product_array(arr, axis=0),
    but 10^5 times faster for full CHIME!

    Currently assumes that arr is a 2D array of shape (nfeeds, nfeeds),
    and returns a 1D array of length (nfeed*(nfeed+1))/2.  This case
    is all we need for phase calibration, but pack_product_array() is
    more general.
    """

    assert arr.ndim == 2
    assert arr.shape[0] == arr.shape[1]

    nfeed = arr.shape[0]
    nprod = (nfeed * (nfeed + 1)) // 2

    ret = np.zeros(nprod, dtype=np.float64)
    iout = 0

    for i in range(nfeed):
        ret[iout : (iout + nfeed - i)] = arr[i, i:]
        iout += nfeed - i

    return ret


def rankN_approx(A, rank=1):
    """Create the rank-N approximation to the matrix A.

    Parameters
    ----------
    A : np.ndarray
        Matrix to approximate
    rank : int, optional

    Returns
    -------
    B : np.ndarray
        Low rank approximation.
    """

    N = A.shape[0]

    evals, evecs = la.eigh(A, eigvals=(N - rank, N - 1))

    return np.dot(evecs, evals * evecs.T.conj())


def eigh_no_diagonal(A, niter=5, eigvals=None):
    """Eigenvalue decomposition ignoring the diagonal elements.

    The diagonal elements are iteratively replaced with those from a rank=1 approximation.

    Parameters
    ----------
    A : np.ndarray[:, :]
        Matrix to decompose.
    niter : int, optional
        Number of iterations to perform.
    eigvals : (lo, hi), optional
        Indices of eigenvalues to select (inclusive).

    Returns
    -------
    evals : np.ndarray[:]
    evecs : np.ndarray[:, :]
    """

    Ac = A.copy()

    if niter > 0:
        Ac[np.diag_indices(Ac.shape[0])] = 0.0

        for i in range(niter):
            Ac[np.diag_indices(Ac.shape[0])] = rankN_approx(Ac).diagonal()

    return la.eigh(Ac, eigvals=eigvals)


def normalise_correlations(A, norm=None):
    """Normalise to make a correlation matrix from a covariance matrix.

    Parameters
    ----------
    A : np.ndarray[:, :]
        Matrix to normalise.
    norm : np.ndarray[:,:]
        Normalize by diagonals of norm.
        If None, then normalize by diagonals of A.

    Returns
    -------
    X : np.ndarray[:, :]
        Normalised correlation matrix.
    ach : np.ndarray[:]
        Array of the square root diagonal elements that normalise the matrix.
    """

    if norm is None:
        ach = A.diagonal() ** 0.5
    else:
        ach = norm.diagonal() ** 0.5

    aci = invert_no_zero(ach)

    X = A * np.outer(aci, aci.conj())

    return X, ach


def apply_gain(vis, gain, axis=1, out=None, prod_map=None):
    """Apply per input gains to a set of visibilities packed in upper
    triangular format.

    This allows us to apply the gains while minimising the intermediate
    products created.

    Parameters
    ----------
    vis : np.ndarray[..., nprod, ...]
        Array of visibility products.
    gain : np.ndarray[..., ninput, ...]
        Array of gains. One gain per input.
    axis : integer, optional
        The axis along which the inputs (or visibilities) are
        contained. Currently only supports axis=1.
    out : np.ndarray
        Array to place output in. If :obj:`None` create a new
        array. This routine can safely use `out = vis`.
    prod_map : ndarray of integer pairs
        Gives the mapping from product axis to input pairs. If not supplied,
        :func:`icmap` is used.

    Returns
    -------
    out : np.ndarray
        Visibility array with gains applied. Same shape as :obj:`vis`.

    """

    nprod = vis.shape[axis]
    ninput = gain.shape[axis]

    if prod_map is None and nprod != (ninput * (ninput + 1) // 2):
        raise Exception("Number of inputs does not match the number of products.")

    if prod_map is not None:
        if len(prod_map) != nprod:
            msg = "Length of *prod_map* does not match number of input products."
            raise ValueError(msg)
        # Could check prod_map contents as well, but the loop should give a
        # sensible error if this is wrong, and checking is expensive.
    else:
        prod_map = [icmap(pp, ninput) for pp in range(nprod)]

    if out is None:
        out = np.empty_like(vis)
    elif out.shape != vis.shape:
        raise Exception("Output array is wrong shape.")

    # Iterate over input pairs and set gains
    for pp in range(nprod):
        # Determine the inputs.
        ii, ij = prod_map[pp]

        # Fetch the gains
        gi = gain[:, ii]
        gj = gain[:, ij].conj()

        # Apply the gains and save into the output array.
        out[:, pp] = vis[:, pp] * gi * gj

    return out


def subtract_rank1_signal(vis, signal, axis=1, out=None, prod_map=None):
    """Subtract a rank 1 signal from a set of visibilities packed in upper
    triangular format.

    This allows us to subtract the noise injection solutions
    while minimising the intermediate products created.

    Parameters
    ----------
    vis : np.ndarray[..., nprod, ...]
        Array of visibility products.
    signal : np.ndarray[..., ninput, ...]
        Array of underlying signals. One signal per input.
    axis : integer, optional
        The axis along which the inputs (or visibilities) are
        contained. Currently only supports axis=1.
    out : np.ndarray
        Array to place output in. If :obj:`None` create a new
        array. This routine can safely use `out = vis`.
    prod_map : ndarray of integer pairs
        Gives the mapping from product axis to input pairs. If not supplied,
        :func:`icmap` is used.

    Returns
    -------
    out : np.ndarray
        Visibility array with signal subtracted. Same shape as :obj:`vis`.
    """

    nprod = vis.shape[axis]
    ninput = signal.shape[axis]

    if prod_map is None and nprod != (ninput * (ninput + 1) // 2):
        raise Exception("Number of inputs does not match the number of products.")

    if prod_map is not None:
        if len(prod_map) != nprod:
            msg = "Length of *prod_map* does not match number of input products."
            raise ValueError(msg)
        # Could check prod_map contents as well, but the loop should give a
        # sensible error if this is wrong, and checking is expensive.
    else:
        prod_map = [icmap(pp, ninput) for pp in range(nprod)]

    if out is None:
        out = np.empty_like(vis)
    elif out.shape != vis.shape:
        raise Exception("Output array is wrong shape.")

    # Iterate over input pairs and set signals
    for pp in range(nprod):
        # Determine the inputs.
        ii, ij = prod_map[pp]

        # Fetch the signals
        si = signal[:, ii]
        sj = signal[:, ij].conj()

        # Apply the signals and save into the output array.
        out[:, pp] = vis[:, pp] - si * sj

    return out


def fringestop_time(
    timestream,
    times,
    freq,
    feeds,
    src,
    wterm=False,
    bterm=True,
    prod_map=None,
    csd=False,
    inplace=False,
    static_delays=True,
    obs=ephemeris.chime,
):
    """Fringestop timestream data to a fixed source.

    Parameters
    ----------
    timestream : np.ndarray[nfreq, nprod, times]
        Array containing the visibility timestream.
    times : np.ndarray[times]
        The UNIX time of each sample, or (if csd=True), the CSD of each sample.
    freq : np.ndarray[nfreq]
        The frequencies in the array (in MHz).
    feeds : list of CorrInputs
        The feeds in the timestream.
    src : skyfield source
        skyfield.starlib.Star or skyfield.vectorlib.VectorSum or
        skyfield.jpllib.ChebyshevPosition body representing the source.
    wterm: bool, optional
        Include elevation information in the calculation.
    bterm: bool, optional
        Include a correction for baselines including the 26m Galt telescope.
    prod_map: np.ndarray[nprod]
        The products in the `timestream` array.
    csd: bool, optional
        Interpret the times parameter as CSDs.
    inplace: bool, optional
        Fringestop the visibilities in place. If not set, leave the originals intact.
    static_delays: bool, optional
        Correct for static cable delays in the system.

    Returns
    -------
    fringestopped_timestream : np.ndarray[nfreq, nprod, times]
    """

    # Check the shapes match
    nfeed = len(feeds)
    nprod = len(prod_map) if prod_map is not None else nfeed * (nfeed + 1) // 2
    expected_shape = (len(freq), nprod, len(times))

    if timestream.shape != expected_shape:
        raise ValueError(
            "The shape of the timestream %s does not match the expected shape %s"
            % (timestream.shape, expected_shape)
        )

    delays = delay(
        times,
        feeds,
        src,
        wterm=wterm,
        bterm=bterm,
        prod_map=prod_map,
        csd=csd,
        static_delays=static_delays,
        obs=obs,
    )

    # Set any non CHIME feeds to have zero phase
    delays = np.nan_to_num(delays, copy=False)

    # If modifying inplace, loop to try and save some memory on large datasets
    if inplace:
        for fi, fr in enumerate(freq):
            fs_phase = np.exp(2.0j * np.pi * delays * fr * 1e6)
            timestream[fi] *= fs_phase
        fs_timestream = timestream
    # Otherwise we might as well generate the entire phase array in onestop
    else:
        fs_timestream = 2.0j * np.pi * delays * freq[:, np.newaxis, np.newaxis] * 1e6
        fs_timestream = np.exp(fs_timestream, out=fs_timestream)
        fs_timestream *= timestream

    return fs_timestream


# Cache the PFB object
_chime_pfb = pfb.PFB(4, 2048)


def decorrelation(
    timestream,
    times,
    feeds,
    src,
    wterm=True,
    bterm=True,
    prod_map=None,
    csd=False,
    inplace=False,
    static_delays=True,
):
    """Apply the decorrelation corrections to a timestream from observing a source.

    Parameters
    ----------
    timestream : np.ndarray[nfreq, nprod, times]
        Array containing the timestream.
    times : np.ndarray[times]
        The UNIX time of each sample, or (if csd=True), the CSD of each sample.
    feeds : list of CorrInputs
        The feeds in the timestream.
    src : skyfield source
        skyfield.starlib.Star or skyfield.vectorlib.VectorSum or
        skyfield.jpllib.ChebyshevPosition body representing the source.
    wterm: bool, optional
        Include elevation information in the calculation.
    bterm: bool, optional
        Include a correction for baselines including the 26m Galt telescope.
    prod_map: np.ndarray[nprod]
        The products in the `timestream` array.
    csd: bool, optional
        Interpret the times parameter as CSDs.
    inplace: bool, optional
        Fringestop the visibilities in place. If not set, leave the originals intact.
    static_delays: bool, optional
        Correct for static cable delays in the system.

    Returns
    -------
    corrected_timestream : np.ndarray[nfreq, nprod, times]
    """

    # Check the shapes match
    nfeed = len(feeds)
    nprod = len(prod_map) if prod_map is not None else nfeed * (nfeed + 1) // 2
    expected_shape = (nprod, len(times))

    if timestream.shape[1:] != expected_shape:
        raise ValueError(
            "The shape of the timestream %s does not match the expected shape %s"
            % (timestream.shape, expected_shape)
        )

    delays = delay(
        times,
        feeds,
        src,
        wterm=wterm,
        bterm=bterm,
        prod_map=prod_map,
        csd=csd,
        static_delays=static_delays,
    )

    # Set any non CHIME feeds to have zero delay
    delays = np.nan_to_num(delays, copy=False)

    ratio_correction = invert_no_zero(
        _chime_pfb.decorrelation_ratio(delays * 800e6)[np.newaxis, ...]
    )

    if inplace:
        timestream *= ratio_correction
    else:
        timestream = timestream * ratio_correction

    return timestream


def delay(
    times,
    feeds,
    src,
    wterm=True,
    bterm=True,
    prod_map=None,
    csd=False,
    static_delays=True,
    obs=ephemeris.chime,
):
    """Calculate the delay in a visibilities observing a given source.

    This includes both the geometric delay and static (cable) delays.

    Parameters
    ----------
    times : np.ndarray[times]
        The UNIX time of each sample, or (if csd=True), the CSD of each sample.
    feeds : list of CorrInputs
        The feeds in the timestream.
    src : skyfield source
        skyfield.starlib.Star or skyfield.vectorlib.VectorSum or
        skyfield.jpllib.ChebyshevPosition body representing the source.
    wterm: bool, optional
        Include elevation information in the calculation.
    bterm: bool, optional
        Include a correction for baselines which include the 26m Galt telescope.
    prod_map: np.ndarray[nprod]
        The products in the `timestream` array.
    csd: bool, optional
        Interpret the times parameter as CSDs.
    static_delays: bool, optional
        If set the returned value includes both geometric and static delays.
        If `False` only geometric delays are included.

    Returns
    -------
    delay : np.ndarray[nprod, nra]
    """

    import scipy.constants

    ra = (times % 1.0) * 360.0 if csd else obs.unix_to_lsa(times)
    src_ra, src_dec = ephemeris.object_coords(src, times.mean(), obs=obs)
    ha = (np.radians(ra) - src_ra)[np.newaxis, :]
    latitude = np.radians(obs.latitude)
    # Get feed positions / c
    feedpos = get_feed_positions(feeds, get_zpos=wterm) / scipy.constants.c
    feed_delays = np.array([f.delay for f in feeds])
    # Calculate the geometric delay between the feed and the reference position
    delay_ref = -projected_distance(ha, latitude, src_dec, *feedpos.T[..., np.newaxis])

    # Add in the static delays
    if static_delays:
        delay_ref += feed_delays[:, np.newaxis]

    # Calculate baseline separations and pack into product array
    if prod_map is None:
        delays = fast_pack_product_array(
            delay_ref[:, np.newaxis] - delay_ref[np.newaxis, :]
        )
    else:
        delays = delay_ref[prod_map["input_a"]] - delay_ref[prod_map["input_b"]]

    # Add the b-term for baselines including the 26m Galt telescope
    if bterm:
        b_delay = _26M_B / scipy.constants.c * np.cos(src_dec)

        galt_feeds = get_holographic_index(feeds)

        galt_conj = np.where(np.isin(prod_map["input_a"], galt_feeds), -1, 0)
        galt_noconj = np.where(np.isin(prod_map["input_b"], galt_feeds), 1, 0)

        conj_flag = galt_conj + galt_noconj

        delays += conj_flag[:, np.newaxis] * b_delay

    return delays


def beam_index2number(beam_index):
    """Convert beam "index" (0-1023) to beam "number" (0-255, 1000-1255, etc.)

    The beam "number", with 1000s indicating the beam's East-West index and the
    remainder going from 0 through 255 indicating the beam's North-South index,
    is used in the CHIME/FRB beam_model package.

    Parameters
    ----------
    beam_index : int or np.ndarray of int
        The beam index or indices to be converted.

    Returns
    -------
    beam_number : same as beam_index
        The corresponding beam number or numbers.
    """
    beam_ew_index = beam_index // 256
    beam_ns_index = beam_index % 256
    beam_number = 1000 * beam_ew_index + beam_ns_index
    return beam_number


def invert_no_zero(*args, **kwargs):
    from caput import tools
    import warnings

    warnings.warn(
        f"Function invert_no_zero is deprecated - use 'caput.tools.invert_no_zero'",
        category=DeprecationWarning,
    )
    return tools.invert_no_zero(*args, **kwargs)


def ensure_list(obj, num=None):
    """Ensure `obj` is list-like, optionally with the length `num`.

    If `obj` not a string but is iterable, it is returned as-is,
    although a length different than `num`, if given, will result in a
    `ValueError`.

    If `obj` is a string or non-iterable, a new list is created with
    `num` copies of `obj` as elements.  In this case, if `num` is not
    given, it is taken to be 1.

    Parameters
    ----------
    obj
        The object to check.
    num: int, optional
        If given, also ensure that the list has `num` elements.


    Returns
    -------
    obj
        The input object, or the newly created list

    Raises
    ------
    ValueError:
        `obj` was iterable but did not have a length of `num`
    """
    if hasattr(obj, "__iter__") and not isinstance(obj, str):
        nnum = len(obj)
        if (num is not None) and (nnum != num):
            raise ValueError("Input list has wrong size.")
    else:
        if num is not None:
            obj = [obj] * num
        else:
            obj = [obj]

    return obj
