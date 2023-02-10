"""
Interface to the CHIME components and graphs

This module interfaces to the layout tables in the CHIME database.

The :mod:`peewee` module is used for the ORM to the MySQL database. Because the
layouts are event-driven, you should never attempt to enter events by raw
inserts to the :class:`event` or :class:`timestamp` tables, as you could create
inconsistencies. Rather, use the methods which are described in this document to
do such alterations robustly.

For most uses, you probably want to import the following:

>>> from datetime import datetime
>>> import logging
>>> logging.basicConfig(level = logging.INFO)
>>> import peewee
>>> import layout
>>> layout.connect_database()

.. note::
   The database must now be explicitly connected. This should not be done within
   an import statement.

.. note::
   The :mod:`logging` module can be set to the level of your preference, or not
   imported altogether if you don't want log messages from the :mod:`layout`
   module. Note that the :mod:`peewee` module sends a lot of messages to the
   DEBUG stream.

If you will be altering the layouts, you will need to register as a user:

>>> layout.set_user("Ahincks")

Use your CHIME wiki username here.  Make sure it starts with a capital letter.
Note that different users have different permissions, stored in the
:class:`user_permission` table. If you are simply reading from the layout,
there is no need to register as a user.

Choose Your Own Adventure
=========================

============================================= ==================================
If you want to ...                            ... then see
============================================= ==================================
retrieve and examine layout graphs            :class:`graph`
add components                                :meth:`component.add<ch_util._db_tables.component.add>`,
                                              :func:`add_component<ch_util._db_tables.add_component>`
remove components                             :meth:`component.remove<ch_util._db_tables.component.remove>`,
                                              :func:`remove_component<ch_util._db_tables.remove_component>`
make connexions                               :func:`make_connexion<ch_util._db_tables.make_connexion>`
sever connexions                              :func:`sever_connexion<ch_util._db_tables.sever_connexion>`
set component properties                      :meth:`component.set_property<ch_util._db_tables.component.set_property>`
                                              :func:`set_property<ch_util._db_tables.set_property>`
get component properties                      :meth:`component.get_property<ch_util._db_tables.component.get_property>`
make/sever many connexions and set many       :func:`enter_ltf`
component properties at the same time
add component history notes                   :meth:`component.add_history<ch_util._db_tables.component.add_history>`
add link to component documentation           :meth:`component.add_doc<ch_util._db_tables.component.add_doc>`
create a global flag                          :meth:`global_flag.start<ch_util._db_tables.global_flag.start>`
============================================= ==================================

Functions
=========

- :py:meth:`add_component<ch_util._db_tables.add_component>`
- :py:meth:`compare_connexion<ch_util._db_tables.compare_connexion>`
- :py:meth:`connect_database<ch_util._db_tables.connect_peewee_tables>`
- :py:meth:`enter_ltf`
- :py:meth:`make_connexion<ch_util._db_tables.make_connexion>`
- :py:meth:`remove_component<ch_util._db_tables.remove_component>`
- :py:meth:`set_user<ch_util._db_tables.set_user>`
- :py:meth:`sever_connexion<ch_util._db_tables.sever_connexion>`
- :py:meth:`global_flags_between`
- :py:meth:`get_global_flag_times`

Classes
=======

- :py:class:`subgraph_spec`
- :py:class:`graph`

Database Models
===============

- :py:class:`component<ch_util._db_tables.component>`
- :py:class:`component_history<ch_util._db_tables.component_history>`
- :py:class:`component_type<ch_util._db_tables.component_type>`
- :py:class:`component_type_rev<ch_util._db_tables.component_type_rev>`
- :py:class:`component_doc<ch_util._db_tables.component_doc>`
- :py:class:`connexion<ch_util._db_tables.connexion>`
- :py:class:`external_repo<ch_util._db_tables.external_repo>`
- :py:class:`event<ch_util._db_tables.event>`
- :py:class:`event_type<ch_util._db_tables.event_type>`
- :py:class:`graph_obj<ch_util._db_tables.graph_obj>`
- :py:class:`global_flag<ch_util._db_tables.global_flag>`
- :py:class:`predef_subgraph_spec<ch_util._db_tables.predef_subgraph_spec>`
- :py:class:`predef_subgraph_spec_param<ch_util._db_tables.predef_subgraph_spec_param>`
- :py:class:`property<ch_util._db_tables.property>`
- :py:class:`property_component<ch_util._db_tables.property_component>`
- :py:class:`property_type<ch_util._db_tables.property_type>`
- :py:class:`timestamp<ch_util._db_tables.timestamp>`
- :py:class:`user_permission<ch_util._db_tables.user_permission>`
- :py:class:`user_permission_type<ch_util._db_tables.user_permission_type>`

Exceptions
==========

- :py:class:`NoSubgraph<ch_util._db_tables.NoSubgraph>`
- :py:class:`BadSubgraph<ch_util._db_tables.BadSubgraph>`
- :py:class:`DoesNotExist<ch_util._db_tables.DoesNotExist>`
- :py:class:`UnknownUser<ch_util._db_tables.UnknownUser>`
- :py:class:`NoPermission<ch_util._db_tables.NoPermission>`
- :py:class:`LayoutIntegrity<ch_util._db_tables.LayoutIntegrity>`
- :py:class:`PropertyType<ch_util._db_tables.PropertyType>`
- :py:class:`PropertyUnchanged<ch_util._db_tables.PropertyUnchanged>`
- :py:class:`ClosestDraw<ch_util._db_tables.ClosestDraw>`
- :py:class:`NotFound<chimedb.core.NotFoundError>`

Constants
=========

- :py:const:`EVENT_AT`
- :py:const:`EVENT_BEFORE`
- :py:const:`EVENT_AFTER`
- :py:const:`EVENT_ALL`
- :py:const:`ORDER_ASC`
- :py:const:`ORDER_DESC`
"""

import datetime
import inspect
import logging
import networkx as nx
import os
import peewee as pw
import re

import chimedb.core

_property = property  # Do this since there is a class "property" in _db_tables.
from ._db_tables import (
    EVENT_AT,
    EVENT_BEFORE,
    EVENT_AFTER,
    EVENT_ALL,
    ORDER_ASC,
    ORDER_DESC,
    _check_fail,
    _plural,
    _are,
    AlreadyExists,
    NoSubgraph,
    BadSubgraph,
    DoesNotExist,
    UnknownUser,
    NoPermission,
    LayoutIntegrity,
    PropertyType,
    PropertyUnchanged,
    ClosestDraw,
    set_user,
    graph_obj,
    global_flag_category,
    global_flag,
    component_type,
    component_type_rev,
    external_repo,
    component,
    component_history,
    component_doc,
    connexion,
    property_type,
    property_component,
    property,
    event_type,
    timestamp,
    event,
    predef_subgraph_spec,
    predef_subgraph_spec_param,
    user_permission_type,
    user_permission,
    compare_connexion,
    add_component,
    remove_component,
    set_property,
    make_connexion,
    sever_connexion,
)

# Legacy name
from chimedb.core import NotFoundError as NotFound

os.environ["TZ"] = "UTC"

# Logging
# =======

# Set default logging handler to avoid "No handlers could be found for logger
# 'layout'" warnings.
from logging import NullHandler


# All peewee-generated logs are logged to this namespace.
logger = logging.getLogger("layout")
logger.addHandler(NullHandler())


# Layout!
# =======


class subgraph_spec(object):
    """Specifications for extracting a subgraph from a full graph.

    The subgraph specification can be created from scratch by passing the
    appropriate parameters. They can also be pulled from the database using the
    class method :meth:`FROM_PREDef`.

    The parameters can be passed as ID's, names of compoenet types or
    :obj:`component_type` instances.

    Parameters
    ----------
    start : integer, :obj:`component_type` or string
      The component type for the start of the subgraph.
    terminate : list of integers, of :obj:`component_type` or of strings
      Component type id's for terminating the subgraph.
    oneway : list of list of integer pairs, of :obj:`component_type` or of strings
      Pairs of component types for defining connexions that should only be
      traced one way when moving from the starting to terminating components.
    hide : list of integers, of :obj:`component_type` or of strings
      Component types for components that should be hidden and skipped over in
      the subgraph.

    Examples
    --------
    To look at subgraphs of components between the outer bulkhead and the
    correlator inputs, one could create the following specification:

    >>> import layout
    >>> from datetime import datetime
    >>> sg_spec = layout.subgraph_spec(start = "c-can thru",
                                       terminate = ["correlator input", "60m coax"],
                                       oneway = [],
                                       hide = ["60m coax", "SMA coax"])

    What did we do? We specified that the subgraph starts at the C-Can bulkhead.
    It terminates at the correlator input; in the other direction, it must also
    terminate at a 60 m coaxial cable plugged into the bulkhead. We hide the 60 m
    coaxial cable so that it doesn't show up in the subgraph. We also hide the SMA
    cables so that they will be skipped over.

    We can load all such subgraphs from the database now and see how many nodes
    they contain:

    >>> sg = layout.graph.from_db(datetime(2014, 10, 5, 12, 0), sg_spec)
    print [s.order() for s in sg]
    [903, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 903,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 903, 3, 3, 3, 903, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 903, 3, 1, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    903, 3, 903, 3, 3, 3, 3, 3, 3, 3, 903, 903, 5, 5]

    Most of them are as short as we would expect, but there are some
    complications. Let's look at that first one by printing out its LTF:

    >>> print sg[0].ltf
    # C-can thru to RFT thru.
    CANAD0B
    RFTA15B attenuation=10 therm_avail=ch7
    <BLANKLINE>
    # RFT thru to HK preamp.
    RFTA15B attenuation=10 therm_avail=ch7
    CHB036C7
    HPA0002A
    <BLANKLINE>
    # HK preamp to HK readout.
    HPA0002A
    ATMEGA49704949575721220150
    HKR00
    <BLANKLINE>
    # HK readout to HK ATMega.
    HKR00
    ATMEGA50874956504915100100
    etc...
    etc...
    # RFT thru to FLA.
    RFTA15B attenuation=10 therm_avail=ch7
    FLA0159B

    Some FLA's are connected to HK hydra cables and we need to terminate on these
    as well. It turns out that some outer bulkheads are connected to 200 m
    coaxial cables, and some FLA's are connected to 50 m delay cables, adding to
    the list of terminations. Let's exclude these as well:

    >>> sg_spec.terminate += ["200m coax", "HK hydra", "50m coax"]
    >>> sg_spec.hide += ["200m coax", "HK hydra", "50m coax"]
    >>> sg = layout.graph.from_db(datetime(2014, 10, 5, 12, 0), sg_spec)
    >>> print [s.order() for s in sg]
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 10, 10, 5, 5]

    The remaining subgraphs with more than three components actually turn out to
    be errors in the layout! Let's investigate the last one by removing any hidden
    components and printing its LTF.

    >>> sn = sg[-1].component(type = "C-can thru")[0].sn
    CANBL1B
    >>> sg_spec.hide = []
    >>> bad_sg = layout.graph.from_db(datetime(2014, 10, 5, 12, 0), sg_spec, sn)
    >>> print bad_sg.ltf()
    # C-can thru to c-can thru.
    CANBL1B
    CXS0017
    RFTQ00B
    CXS0016
    FLA0073B
    RFTQ01B attenuation=9
    CXS0015
    CANBL0B

    It appears that :code:`CXS0016` mistakenly connects :code:`RFTQ00B` to
    :code:`FLA0073B`. This is an error that should be investigated and fixed. But
    by way of illustration, let's cut this subgraph short by specifying a one-way
    connection, and not allowing the subgrapher to trace backwards from the inner
    bulkhead to an SMA cable:

    >>> sg_spec.oneway = [["SMA coax", "RFT thru"]]
    >>> bad_sg = layout.graph.from_db(datetime(2014, 10, 5, 12, 0), sg_spec, sn)
    >>> print bad_sg.ltf()
    # C-can thru to RFT thru.
    CANBL1B
    CXS0017
    RFTQ00B
    """

    def __init__(self, start, terminate, oneway, hide):
        self.start = start
        self.terminate = terminate
        self.oneway = oneway
        self.hide = hide

    @classmethod
    def from_predef(cls, predef):
        """Create a subgraph specification from a predefined version in the DB.

        Parameters
        ----------
        predef : :class:`predef_subgraph_spec`
          A predefined subgraph specification in the DB.
        """
        s = predef.start_type.id
        t = []
        o = []
        h = []
        for param in predef_subgraph_spec_param.select(
            predef_subgraph_spec_param.action,
            predef_subgraph_spec_param.type1.alias("type1_id"),
            predef_subgraph_spec_param.type2.alias("type2_id"),
        ).where(predef_subgraph_spec_param.predef_subgraph_spec == predef):
            if param.action == "T":
                t.append(param.type1_id)
            elif param.action == "O":
                o.append([param.type1_id, param.type2_id])
            elif param.action == "H":
                h.append(param.type1_id)
            else:
                raise RuntimeError('Unknown subgraph action type "%s".' % param.action)
        return cls(s, t, o, h)

    @_property
    def start(self):
        """The component type ID starting the subgraph."""
        return self._start

    @start.setter
    def start(self, val):
        self._start = _id_from_multi(component_type, val)

    @_property
    def terminate(self):
        """The component type ID(s) terminating the subgraph."""
        return self._terminate

    @terminate.setter
    def terminate(self, val):
        self._terminate = [_id_from_multi(component_type, tt) for tt in val]

    @_property
    def oneway(self):
        """Pairs of component type ID(s) for one-way tracing of the subgraph."""
        return self._oneway

    @oneway.setter
    def oneway(self, val):
        self._oneway = [
            [
                _id_from_multi(component_type, oo[0]),
                _id_from_multi(component_type, oo[1]),
            ]
            for oo in val
        ]

    @_property
    def hide(self):
        """The component type ID(s) that are skipped over in the subgraph."""
        return self._hide

    @hide.setter
    def hide(self, val):
        self._hide = [_id_from_multi(component_type, h) for h in val]


class graph(nx.Graph):
    """A graph of connexions.

    This class inherits the
    `networkx.Graph <http://networkx.github.io/documentation/networkx-1.9.1/>`_
    class and adds CHIME-specific functionality.

    Use the :meth:`from_db` class method to construct a graph from the database.

    Parameters
    ----------
    time : datetime.datetime
      The time at which the graph is valid. Default is now().

    Examples
    --------

    To load a graph from the database, use the :meth:`from_db` class method:

    >>> from ch_util import graph
    >>> from datetime import datetime
    >>> g = layout.graph.from_db(datetime(2014, 10, 5, 12, 0))

    You can now use any of the
    `networkx.Graph <http://networkx.github.io/documentation/networkx-1.9.1/>`_
    methods:

    >>> print g.order(), g.size()
    2483 2660

    There are some convenience methods for our implementation. For example, you
    can easily find components by component type:

    >>> print g.component(type = "reflector")
    [<layout.component object at 0x7fd1b2cda710>, <layout.component object at 0x7fd1b2cda810>, <layout.component object at 0x7fd1b2cfb7d0>]

    Note that the graph nodes are :obj:`component` objects. You can also use the
    :meth:`component` method to search for components by serial number:

    >>> ant = g.component(comp = "ANT0044B")

    Node properties are stored as per usual for :class:`networkx.Graph` objects:

    >>> print g.nodes[ant]
    {'_rev_id': 11L, '_type_id': 2L, u'pol1_orient': <layout.property object at 0x7f31ed323fd0>, '_type_name': u'antenna', '_id': 32L, u'pol2_orient': <layout.property object at 0x7f31ed2c8790>, '_rev_name': u'B'}

    Note, however, that there are some internally-used properties (starting with
    an underscore). The :meth:`node_property` returns a dictionary of properties
    without these private memebers:

    >>> for p in g.node_property(ant).values():
    ...   print "%s = %s %s" % (p.type.name, p.value, p.type.units if p.type.units else "")
    pol1_orient = S
    pol2_orient = E

    To search the graph for the closest component of a given type to a single
    component, using :meth:`closest_of_type`:

    >>> slt_type = layout.component_type.get(name = "cassette slot")
    >>> print g.closest_of_type(ant, slt_type).sn
    CSS004C0

    Use of :meth:`closest_of_type` can be subtle for components separated by long
    paths. See its documentation for more examples.

    Subgraphs can be created using a subgraph specification, encoded in a
    :class:`subgraph_spec` object. See the documentation for that class for
    details, but briefly, this allows one to create a smaller, more manageable
    graph containing only components and connexions you are interested in. Given a
    subgraph, the :meth:`ltf` method can be useful.
    """

    def __init__(self, time=datetime.datetime.now()):
        # Initialise the graph.
        nx.Graph.__init__(self)
        self._time = time
        self._sg_spec = None
        self._sg_spec_start = None
        self._sn_dict = dict()
        self._ctype_dict = dict()

        # We will cache all the component types, revisions and properties now,
        # since these will be used constantly by the graph.
        component_type.fill_cache()
        component_type_rev.fill_cache()
        property_type.fill_cache()

        # Aliases.
        self.neighbours = self.neighbors
        self.neighbor_of_type = self.neighbour_of_type

    @classmethod
    def from_db(cls, time=datetime.datetime.now(), sg_spec=None, sg_start_sn=None):
        """Create a new graph by reading the database.

        This method is designed to be efficient. It has customised SQL calls so that
        only a couple of queries are required. Doing this with the standard peewee
        functionality requires many more calls.

        This method will establish a connection to the database if it doesn't
        already exist.

        Parameters
        ----------
        time : datetime.datetime
          The time at which the graph is valid. Default is now().
        sg_spec : :obj:`subgraph_spec`
          The subgraph specificationto use; can be set to :obj:`None`.
        sg_start_sn : string
          If a serial number is specified, then only the subgraph starting with that
          component will be returned. This parameter is ignored if sg_spec is
          :obj:`None`.

        Returns
        -------
        :obj:`graph`
          If *sg_spec* is not :obj:`None`, and *sg_start_sn* is not specified, then
          a list of :obj:`graph` objects is returned instead.

        Raises
        ------
        If no graph is found, :exc:`NotFound` is raised.
        """

        # Initalise the database connections
        connect_database()

        g = cls(time)

        # Add the connexions.
        sql = (
            "SELECT c1.*, c2.*, pt.id "
            "FROM connexion c "
            "JOIN component c1 ON c1.sn = c.comp_sn1 "
            "JOIN event e1 ON e1.graph_obj_id = c1.id "
            "JOIN timestamp e1t1 ON e1.start_id = e1t1.id "
            "LEFT JOIN timestamp e1t2 ON e1.end_id = e1t2.id "
            "JOIN component c2 ON c2.sn = c.comp_sn2 "
            "JOIN event e2 ON e2.graph_obj_id = c2.id "
            "JOIN timestamp e2t1 ON e2.start_id = e2t1.id "
            "LEFT JOIN timestamp e2t2 ON e2.end_id = e2t2.id "
            "JOIN event e ON e.graph_obj_id = c.id "
            "JOIN event_type pt ON e.type_id = pt.id "
            "JOIN timestamp t1 ON e.start_id = t1.id "
            "LEFT JOIN timestamp t2 ON e.end_id = t2.id "
            "WHERE e.active = 1 AND e1.type_id = 1 AND e2.type_id = 1 AND "
            "e1t1.time <= '%s' AND "
            "(e1.end_id IS NULL OR e1t2.time > '%s') AND "
            "e2t1.time <= '%s' AND "
            "(e2.end_id IS NULL OR e2t2.time > '%s') AND "
            "t1.time <= '%s' AND "
            "(e.end_id IS NULL OR t2.time > '%s');"
            % (time, time, time, time, time, time)
        )
        # print sql
        conn_list = chimedb.core.proxy.execute_sql(sql)
        for r in conn_list:
            c1 = g._ensure_add(r[0], r[1], r[2], r[3])
            c2 = g._ensure_add(r[4], r[5], r[6], r[7])
            if r[8] == event_type.perm_connexion().id:
                perm = True
            else:
                perm = False
            g.add_edge(c1, c2, permanent=perm, hidden=False)

        # Add the properties.
        sql = (
            "SELECT p.*, c.*, pt.name "
            "FROM property p "
            "JOIN property_type pt ON p.type_id = pt.id "
            "JOIN component c ON p.comp_sn = c.sn "
            "JOIN event ce ON ce.graph_obj_id = c.id "
            "JOIN timestamp ct1 ON ce.start_id = ct1.id "
            "LEFT JOIN timestamp ct2 ON ce.end_id = ct2.id "
            "JOIN event e ON e.graph_obj_id = p.id "
            "JOIN timestamp t1 ON e.start_id = t1.id "
            "LEFT JOIN timestamp t2 ON e.end_id = t2.id "
            "WHERE e.active = 1 AND ce.type_id = 1 AND "
            "ct1.time <= '%s' AND "
            "(ce.end_id IS NULL OR ct2.time > '%s') AND "
            "t1.time <= '%s' AND "
            "(e.end_id IS NULL OR t2.time > '%s');" % (time, time, time, time)
        )
        prop_list = chimedb.core.proxy.execute_sql(sql)
        for r in prop_list:
            p = property(id=r[0], comp=r[1], type=r[2], value=r[3])
            p.type = property_type.from_id(r[2])
            c = g._ensure_add(r[4], r[5], r[6], r[7])
            g.nodes[c][r[8]] = p

        if sg_spec:
            return graph.from_graph(g, sg_spec, sg_start_sn)
        else:
            return g

    def _ensure_add(self, id, sn, type, rev):
        """Robustly add a component, avoiding duplication."""
        try:
            c = self.component(comp=sn)
        except NotFound:
            # Component ID is a foreign key to graph_obj, so we need to make an
            # instance of this for that.
            g = graph_obj(id=id)
            c = component(id=g, sn=sn, type=type, type_rev=rev)

            # We hydrate the component type and revision so that no further queries
            # need to be made. When the graph was initialised, all of the types and
            # revisions were cached, so the following requires no further queries.
            c.type = component_type.from_id(type)
            c.rev = component_type_rev.from_id(rev)
            self.add_node(c)
            self._sn_dict[sn] = c
            try:
                self._ctype_dict[type].append(c)
            except KeyError:
                self._ctype_dict[type] = [c]
        return c

    def node_property(self, n):
        """Return the properties of a node excluding internally used properties.

        If you iterate over a nodes properties, you will also get the
        internally-used properties (starting with an underscore). This method gets
        the dictionary of properties without these "private" properties.

        Parameters
        ----------
        node : node object
          The node for which to get the properties.

        Returns
        -------
        A dictionary of properties.

        Examples
        --------
        >>> from ch_util import graph
        >>> from datetime import datetime
        >>> g = layout.graph.from_db(datetime(2014, 10, 5, 12, 0))
        >>> rft = g.component(comp = "RFTK07B")
        >>> for p in g.node_property(rft).values():
        ...   print "%s = %s %s" % (p.type.name, p.value, p.type.units if p.type.units else "")
        attenuation = 10 dB
        therm_avail = ch1
        """
        ret = dict()
        for key, val in self.nodes[n].items():
            if key[0] != "_":
                ret[key] = val
        return ret

    def component(self, comp=None, type=None, sort_sn=False):
        """Return a component or list of components from the graph.

        The components exist as graph nodes. This method provides searchable access
        to them.

        Parameters
        ----------
        comp : string or :obj:`component`
          If not :obj:`None`, then return the component with this serial number, or
          :obj:`None` if it does not exist in the graph. If this parameter is set,
          then **type** is ignored. You can also pass a component object; the
          instance of that component with the same serial number will be returned if
          it exists in this graph.
        type : string or :class:`component_type`
          If not :obj:`None`, then only return components of this type. You may pass
          either the name of the component type or an object.

        Returns
        -------
        :class:`component` or list of such
          If the **sn** parameter is passed, a single :class:`component` object is
          returned. If the **type** parameter is passed, a list of
          :class:`component` objects is returned.

        Raises
        ------
        :exc:`NotFound`
          Raised if no component is found.

        Examples
        --------
        >>> from ch_util import graph
        >>> from datetime import datetime
        >>> g = layout.graph.from_db(datetime(2014, 10, 5, 12, 0))
        >>> print g.component("CXA0005A").type_rev.name
        B
        >>> for r in g.component(type = "reflector"):
        ...   print r.sn
        E_cylinder
        W_cylinder
        26m_dish

        """
        if comp:
            ret = None
            try:
                sn = comp.sn
            except AttributeError:
                sn = comp
            try:
                ret = self._sn_dict[sn]
            except KeyError:
                raise NotFound('Serial number "%s" is not in the graph.' % (sn))
        elif not type:
            ret = self.nodes()
        else:
            try:
                type_id = type.id
                type_name = type.name
            except AttributeError:
                type_id = component_type.from_name(type).id
                type_name = type
            try:
                ret = list(self._ctype_dict[type_id])
                if sort_sn:
                    ret.sort(key=lambda x: x.sn)
            except KeyError:
                raise NotFound(
                    'No components of type "%s" are in the graph.' % type_name
                )
        return ret

    def _subgraph_recurse(self, gr, comp1, sg, done, last_no_hide):
        if comp1.type.id in sg.hide:
            c1 = last_no_hide
            hidden = True
        else:
            c1 = gr._ensure_add(
                comp1.id, comp1.sn, comp1.type.id, comp1.rev.id if comp1.rev else None
            )
            if not last_no_hide:
                last_no_hide = c1
            for k, v in self.node_property(comp1).items():
                gr.nodes[c1][k] = v
            hidden = False
            if not last_no_hide:
                last_no_hide = c1

        done.append(comp1.sn)
        for comp2 in self.neighbors(comp1):
            # Watch for connexions in the wrong order.
            check = [comp2.type.id, comp1.type.id]
            if check in sg.oneway:
                continue

            if comp2.type.id not in sg.hide:
                c2 = gr._ensure_add(
                    comp2.id,
                    comp2.sn,
                    comp2.type.id,
                    comp2.rev.id if comp2.rev else None,
                )
                for k, v in self.node_property(comp2).items():
                    gr.nodes[c2][k] = v

                try:
                    gr.edges[c1, c2]
                except KeyError:
                    if c1.sn != c2.sn:
                        if hidden:
                            perm = False
                        else:
                            perm = self.edges[comp1, comp2]["permanent"]
                        gr.add_edge(c1, c2, permanent=perm, hidden=hidden, _head=c1)
                        last_no_hide = c2

            if comp2.type.id not in sg.terminate and comp2.sn not in done:
                self._subgraph_recurse(gr, comp2, sg, done, last_no_hide)
        return

    @classmethod
    def from_graph(cls, g, sg_spec=None, sg_start_sn=None):
        """Find subgraphs within this graph.

        Parameters
        ----------
        g : :obj:`graph`
          The graph from which to get the new graph.
        sg_spect : :obj:`subgraph_spec`
          The subgraph specification to use; can be set to :obj:`None`.

        Returns
        -------
        A list of :obj:`graph` objects, one for each subgraph found. If, however,
        *g* is set to :obj:`None`, a reference to the input graph is returned.
        """
        if sg_spec == None:
            return g
        if sg_spec.start in sg_spec.terminate:
            raise BadSubgraph(
                "You cannot terminate on the component type of the "
                "starting component of your subgraph."
            )
        if sg_spec.start in sg_spec.hide:
            raise BadSubgraph(
                "You cannot hide the component type of the "
                "starting component of a subgraph."
            )

        ret = []
        for start_comp in g.component(type=component_type.from_id(sg_spec.start)):
            if sg_start_sn:
                if start_comp.sn != sg_start_sn:
                    continue
            ret.append(cls(time=g.time))
            g._subgraph_recurse(ret[-1], start_comp, sg_spec, [], None)
            ret[-1]._sg_spec = sg_spec
            ret[-1]._sg_spec_start = ret[-1].component(comp=start_comp.sn)

        if len(ret) < 1:
            raise NotFound("No subgraph was found.")
        if sg_start_sn:
            return ret[-1]
        else:
            return ret

    def _print_chain(self, chain):
        if len(chain) <= 1:
            return ""

        ret = ""
        ctype1 = chain[0].type.name
        ctype2 = chain[-1].type.name
        ret = "# %s to %s.\n" % (ctype1[0].upper() + ctype1[1:], ctype2)
        for c in chain:
            ret += c.sn
            for prop, value in self.node_property(c).items():
                ret += " %s=%s" % (prop, value.value)
            ret += "\n"
        ret += "\n"

        return ret

    def _ltf_recurse(self, comp, done, last):
        ret = ""
        if last:
            chain = [last, comp]
        else:
            chain = [comp]
        done.append(comp)
        while 1:
            next_comp = list(set(self.neighbors(comp)) - set(done))
            if not len(next_comp) or comp.type.id in self.sg_spec.terminate:
                ret += self._print_chain(chain)
                break

            if len(next_comp) == 1:
                chain.append(next_comp[0])
                done.append(next_comp[0])
                comp = next_comp[0]
            elif len(next_comp) > 1:
                done_print = False
                for c in next_comp:
                    if not done_print:
                        ret += self._print_chain(chain)
                        done_print = True
                    done.append(c)
                    ret += self._ltf_recurse(c, done, chain[-1])
                break
            else:
                break

        return ret

    def ltf(self):
        """Get an LTF representation of the graph. The graph must be a subgraph,
        i.e., generated with a :obj:`predef_subgraph_spec`.

        Returns
        -------
        ltf : string
          The LTF representation of the graph.

        Raises
        ------
        :exc:`NoSubgraph`
        Raised if no subgraph specification is associate with this layout.

        Examples
        --------
        Get the LTF for a subgraph of antenna to HK.

        >>> import layout
        >>> from datetime import datetime
        >>> start = layout.component_type.get(name = "antenna").id
        >>> terminate = [layout.component_type.get(name = "reflector").id,
                         layout.component_type.get(name = "cassette slot").id,
                         layout.component_type.get(name = "correlator input").id,
                         layout.component_type.get(name = "HK preamp").id,
                         layout.component_type.get(name = "HK hydra").id]
        >>> hide = [layout.component_type.get(name = "reflector").id,
                    layout.component_type.get(name = "cassette slot").id,
                    layout.component_type.get(name = "HK preamp").id,
                    layout.component_type.get(name = "HK hydra").id]
        >>> sg_spec = layout.subgraph_spec(start, terminate, [], hide)
        >>> sg = layout.graph.from_db(datetime(2014, 11, 20, 12, 0), sg_spec, "ANT0108B")
        >>> print sg.ltf()
        # Antenna to correlator input.
        ANT0108B pol1_orient=S pol2_orient=E
        PL0108B1
        LNA0249B
        CXA0239C
        CANBJ6B
        CXS0042
        RFTG00B attenuation=10
        FLA0196B
        CXS0058
        K7BP16-00041606
        <BLANKLINE>
        # Antenna to correlator input.
        ANT0108B pol1_orient=S pol2_orient=E
        PL0108B2
        LNA0296B
        CXA0067B
        CANBG6B
        CXS0090
        RFTG01B attenuation=10
        FLA0269B
        CXS0266
        K7BP16-00041506
        """
        if not self._sg_spec:
            raise NoSubgraph(
                "This layout is not a subgraph. You can only create "
                "LTF representations of subgraphs generated from "
                "predef_subgraph_spec objects."
            )
        return self._ltf_recurse(self._sg_spec_start, [], None)

    def shortest_path_to_type(self, comp, type, type_exclude=None, ignore_draws=True):
        """Searches for the shortest path to a component of a given type.

        Sometimes the closest component is through a long, convoluted path that you
        do not wish to explore. You can cut out these cases by including a list of
        component types that will block the search along a path.

        The component may be passed by object or by serial number; similarly for
        component types.

        Parameters
        ----------
        comp : :obj:`component` or string or list of one of these
          The component(s) to search from.
        type : :obj:`component_type` or string
          The component type to find.
        type_exclude : list of :obj:`component_type` or strings
          Any components of this type will prematurely cut off a line of
          investigation.
        ignore_draws : boolean
          It is possible that there be more than one component of a given type the
          same distance from the starting component. If this parameter is set to
          :obj:`True`, then just return the first one that is found. If set to
          :obj:`False`, then raise an exception.

        Returns
        -------
        comp: :obj:`component` or list of such
          The closest component of the given type to **start**. If no path to a
          component of the specified type exists, return :obj:`None`.

        Raises
        ------
        :exc:`ClosestDraw`
          Raised if there is no unique closest component and **ignore_draws** is set
          to :obj:`False`.

        Examples
        --------
        See the examples for :meth:`closest_of_type`.
        """
        # Get the start node and the list of candidate end nodes.
        one = False
        if isinstance(comp, str) or isinstance(comp, component):
            comp = [comp]
            one = True

        start_list = [self.component(comp=c) for c in comp]

        # Find end_candidates. If there are none in this graph, return None.
        try:
            end_candidate = self.component(type=type)
        except NotFound:
            return None if one else [None] * len(comp)

        if end_candidate is None:
            return None if one else [None] * len(comp)

        # Get the list of components to exclude, based on the types in the
        # **type_exclude** parameter.
        exclude = []
        if type_exclude is not None:
            if not isinstance(type_exclude, list):
                type_exclude = [type_exclude]
            for t in type_exclude:
                try:
                    exclude += self.component(type=t)
                except NotFound:
                    pass

        # Construct a subgraph without the excluded nodes
        graph = self.subgraph(set(self.nodes()) - set(exclude)).copy()

        # Add a type marking node into the graph connected to all components of
        # the type we are looking for
        tn = "Type node marker"
        graph.add_node(tn)
        edges = [(tn, end) for end in end_candidate]
        graph.add_edges_from(edges)

        # Get the shortest path to type by searching for the shortest path from
        # the start to the type marker, the actual path is the same after
        # removing the type marker
        shortest = []
        for start in start_list:
            try:
                path = nx.shortest_path(graph, source=start, target=tn)[:-1]
            except (nx.NetworkXError, nx.NetworkXNoPath):
                path = None

            shortest.append(path)

        # Return the shortest path (or None if not found)
        if one:
            return shortest[0]
        else:
            return shortest

    def closest_of_type(self, comp, type, type_exclude=None, ignore_draws=True):
        """Searches for the closest connected component of a given type.

        Sometimes the closest component is through a long, convoluted path that you
        do not wish to explore. You can cut out these cases by including a list of
        component types that will block the search along a path.

        The component may be passed by object or by serial number; similarly for
        component types.

        Parameters
        ----------
        comp : :obj:`component` or string or list of such
          The component to search from.
        type : :obj:`component_type` or string
          The component type to find.
        type_exclude : list of :obj:`component_type` or strings
          Any components of this type will prematurely cut off a line of
          investigation.
        ignore_draws : boolean
          It is possible that there be more than one component of a given type the
          same distance from the starting component. If this parameter is set to
          :obj:`True`, then just return the first one that is found. If set to
          :obj:`False`, then raise an exception.

        Returns
        -------
        comp: :obj:`component` or list of such
          The closest component of the given type to **start**. If no component of
          type is found :obj:`None` is returned.

        Raises
        ------
        :exc:`ClosestDraw`
          Raised if there is no unique closest component and **ignore_draws** is set
          to :obj:`False`.

        Examples
        --------
        Find the cassette slot an antenna is plugged into:

        >>> import layout
        >>> from datetime import datetime
        >>> g = layout.graph.from_db(datetime(2014, 11, 5, 12, 0))
        >>> print g.closest_of_type("ANT0044B", "cassette slot").sn
        CSS004C0

        The example above is simple as the two components are adjacent:

        >>> print [c.sn for c in g.shortest_path_to_type("ANT0044B", "cassette slot")]
        [u'ANT0044B', u'CSS004C0']

        In general, though, you need to take care when
        using this method and make judicious use of the **type_exclude** parameter.
        For example, consider the following example:

        >>> print g.closest_of_type("K7BP16-00040112", "RFT thru").sn
        RFTB15B

        It seems OK on the surface, but the path it has used is probably not what
        you want:

        >>> print [c.sn for c in g.shortest_path_to_type("K7BP16-00040112", "RFT thru")]
        [u'K7BP16-00040112', u'K7BP16-000401', u'K7BP16-00040101', u'FLA0280B', u'RFTB15B']

        We need to block the searcher from going into the correlator card slot and
        then back out another input, which we can do like so:

        >>> print g.closest_of_type("K7BP16-00040112", "RFT thru", type_exclude = "correlator card slot").sn
        RFTQ15B

        The reason the first search went through the correlator card slot is because
        there are delay cables and splitters involved.

        >>> print [c.sn for c in g.shortest_path_to_type("K7BP16-00040112", "RFT thru", type_exclude = "correlator card slot")]
        [u'K7BP16-00040112', u'CXS0279', u'CXA0018A', u'CXA0139B', u'SPL001AP2', u'SPL001A', u'SPL001AP3', u'CXS0281', u'RFTQ15B']

        The shortest path really was through the correlator card slot, until we
        explicitly rejected such paths.
        """

        path = self.shortest_path_to_type(comp, type, type_exclude, ignore_draws)

        try:
            closest = [p[-1] if p is not None else None for p in path]
        except TypeError:
            closest = path[-1] if path is not None else None
        return closest

    def neighbour_of_type(self, n, type):
        """Get a list of neighbours of a given type.

        This is like the :meth:`networkx.Graph.neighbors` method, but selects only
        the neighbours of the specified type.

        Parameters
        ----------
        comp : :obj:`component`
          A node in the graph.
        type : :obj:`component_type` or string
          The component type to find.

        Returns
        -------
        nlist : A list of nodes of type **type** adjacent to **n**.

        Raises
        ------
        :exc:`networkx.NetworkXError`
          Raised if **n** is not in the graph.
        """
        ret = []
        try:
            type.name
        except AttributeError:
            type = component_type.from_name(type)
        for nn in self.neighbours(n):
            if nn.type == type:
                ret.append(nn)
        return ret

    @_property
    def time(self):
        """The time of the graph.

        Returns
        -------
        time : datetime.datetime
          The time at which this graph existed.
        """
        return self._time

    @_property
    def sg_spec(self):
        """The :obj:`subgraph_spec` (subgraph specification) used to get this graph.

        Returns
        -------
        The :obj:`subgraph_spec` used to get this graph, if any.
        """
        return self._sg_spec

    @_property
    def sg_spec_start(self):
        """The subgraph starting component.

        Returns
        -------
        The :obj:`component` that was used to begin the subgraph, if any.
        """
        return self._sg_spec_start


# Private Functions
# ==================


def _add_to_sever(sn1, sn2, sever, fail_comp):
    ok = True
    for sn in (sn1, sn2):
        try:
            component.get(sn=sn)
        except pw.DoesNotExist:
            fail_comp.append(sn)
            ok = False
    if ok:
        conn = connexion.from_pair(sn1, sn2)
        sever.append(conn)


def _add_to_chain(chain, sn, prop, sever, fail_comp):
    if sn == "//":
        if not len(chain):
            raise SyntaxError("Stray sever mark (//) in LTF.")
        if chain[-1] == "//":
            raise SyntaxError("Consecutive sever marks (//) in LTF.")
        chain.append("//")
        return

    if len(chain):
        if chain[-1] == "//":
            if len(chain) < 2:
                raise SyntaxError(
                    'Confused about chain ending in "%s". Is the '
                    "first serial number valid?" % (chain[-1])
                )
            try:
                _add_to_sever(chain[-2]["comp"].sn, sn, sever, fail_comp)
            except KeyError:
                pass
            del chain[-2]
            del chain[-1]

    chain.append(dict())
    try:
        chain[-1]["comp"] = component.get(sn=sn)
        for k in range(len(prop)):
            if len(prop[k].split("=")) != 2:
                raise SyntaxError('Confused by the property command "%s".' % prop[k])
            chain[-1][prop[k].split("=")[0]] = prop[k].split("=")[1]
    except pw.DoesNotExist:
        if not sn in fail_comp:
            fail_comp.append(sn)


def _id_from_multi(cls, o):
    if isinstance(o, int):
        return o
    elif isinstance(o, cls):
        return o.id
    else:
        return cls.get(name=o).id


# Public Functions
# ================

from ._db_tables import connect_peewee_tables as connect_database


def enter_ltf(ltf, time=datetime.datetime.now(), notes=None, force=False):
    """Enter an LTF into the database.

    This is a special mark-up language for quickly entering events. See the "help"
    box on the LTF page of the web interface for instructions.

    Parameters
    ----------
    ltf : string
      Pass either the path to a file containing the LTF, or a string containing
      the LTF.
    time : datetime.datetime
      The time at which to apply the LTF.
    notes : string
      Notes for the timestamp.
    force : bool
      If :obj:`True`, then do nothing when events that would damage database
      integrity are encountered; skip over them. If :obj:`False`, then a bad
      propsed event will raise the appropriate exception.
    """

    try:
        with open(ltf, "r") as myfile:
            ltf = myfile.readlines()
    except IOError:
        try:
            ltf = ltf.splitlines()
        except AttributeError:
            pass
    chain = []
    fail_comp = []
    multi_sn = None
    multi_prop = None
    chain.append([])
    sever = []
    i = 0
    for l in ltf:
        if len(l) and l[0] == "#":
            continue
        severed = False
        try:
            if l.split()[1] == "//":
                severed = True
        except IndexError:
            pass

        if not len(l) or l.isspace() or severed or l[0:2] == "$$":
            if severed:
                _add_to_sever(l.split()[0], l.split()[2], sever, fail_comp)
            if multi_sn:
                _add_to_chain(chain[i], multi_sn, prop, sever, fail_comp)
                multi_sn = False
            chain.append([])
            i += 1
            continue

        l = l.replace("\n", "")
        l = l.strip()

        sn = l.split()[0]
        prop = l.split()[1:]

        # Check to see if this is a multiple-line SN.
        if multi_sn:
            if sn[0] == "+":
                off = len(multi_sn) - len(sn)
            else:
                off = 0
            match = False
            if len(multi_sn) == len(sn) + off:
                for j in range(len(sn)):
                    if sn[j] != "." and sn[j] != "-" and sn[j] != "+":
                        if multi_sn[j + off] == "." or multi_sn[j + off] == "-":
                            match = True
                            multi_sn = (
                                multi_sn[: j + off] + sn[j] + multi_sn[j + off + 1 :]
                            )
                            multi_prop = prop
            if not match:
                _add_to_chain(chain[i], multi_sn, multi_prop, sever, fail_comp)
                _add_to_chain(chain[i], sn, prop, sever, fail_comp)
                multi_sn = None
                multi_prop = []
        else:
            if sn.find("+") >= 0 or sn.find("-") >= 0 or sn.find(".") >= 0:
                multi_sn = sn
                multi_prop = []
            else:
                _add_to_chain(chain[i], sn, prop, sever, fail_comp)

    if multi_sn:
        _add_to_chain(chain[i], multi_sn, multi_prop, sever, fail_comp)

    _check_fail(
        fail_comp,
        False,
        DoesNotExist,
        "The following component%s "
        "%s not in the DB and must be added first"
        % (_plural(fail_comp), _are(fail_comp)),
    )

    conn_list = []
    prop_list = []
    for c in chain:
        for i in range(1, len(c)):
            comp1 = c[i - 1]["comp"]
            comp2 = c[i]["comp"]
            if comp1.sn == comp2.sn:
                logger.info(
                    "Skipping auto connexion: %s <=> %s." % (comp1.sn, comp2.sn)
                )
            else:
                conn = connexion.from_pair(comp1, comp2)
                try:
                    if conn.is_permanent(time):
                        logger.info(
                            "Skipping permanent connexion: %s <=> %s."
                            % (comp1.sn, comp2.sn)
                        )
                    elif conn not in conn_list:
                        conn_list.append(conn)
                except pw.DoesNotExist:
                    conn_list.append(conn)
        for i in range(len(c)):
            comp = c[i]["comp"]
            for p in c[i].keys():
                if p == "comp":
                    continue
                try:
                    prop_list.append([comp, property_type.get(name=p), c[i][p]])
                except pw.DoesNotExist:
                    raise DoesNotExist('Property type "%s" does not exist.' % p)
    make_connexion(conn_list, time, False, notes, force)
    sever_connexion(sever, time, notes, force)
    for p in prop_list:
        p[0].set_property(p[1], p[2], time, notes)


def get_global_flag_times(flag):
    """Convenience function to get global flag times by id or name.

    Parameters
    ----------
    flag : integer or string
      If an integer, this is a global flag id, e.g. `64`. If a string this is the
      global flag's name e.g. 'run_pass0_e'.

    Returns
    -------
    start : :class:`datetime.datetime`
      Global flag start time (UTC).
    end : :class:`datetime.datetime` or `None`
      Global flag end time (UTC) or `None` if the flag hasn't ended.

    """

    if isinstance(flag, str):
        query_ = global_flag.select().where(global_flag.name == flag)
    else:
        query_ = global_flag.select().where(global_flag.id == flag)

    flag_ = query_.join(graph_obj).join(event).where(event.active == True).get()

    event_ = event.get(graph_obj=flag_.id, active=True)

    start = event_.start.time
    try:
        end = event_.end.time
    except pw.DoesNotExist:
        end = None
    return start, end


def global_flags_between(start_time, end_time, severity=None):
    """Find global flags that overlap a time interval.

    Parameters
    ----------
    start_time
    end_time
    severity : str
        One of 'comment', 'warning', 'severe', or None.

    Returns
    -------
    flags : list
        List of global_flag objects matching criteria.

    """

    from . import ephemeris

    start_time = ephemeris.ensure_unix(start_time)
    end_time = ephemeris.ensure_unix(end_time)

    query = global_flag.select()
    if severity:
        query = query.where(global_flag.severity == severity)
    query = query.join(graph_obj).join(event).where(event.active == True)

    # Set aliases for the join
    ststamp = timestamp.alias()
    etstamp = timestamp.alias()

    # Add constraint for the start time
    query = query.join(ststamp, on=event.start).where(
        ststamp.time < ephemeris.unix_to_datetime(end_time)
    )
    # Constrain the end time (being careful to deal with open events properly)
    query = (
        query.switch(event)
        .join(etstamp, on=event.end, join_type=pw.JOIN.LEFT_OUTER)
        .where(
            (etstamp.time > ephemeris.unix_to_datetime(start_time))
            | event.end.is_null()
        )
    )

    return list(query)
