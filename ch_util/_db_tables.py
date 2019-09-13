"""
Private module for defining the DB tables with the peewee ORM. These are
imported into the layout and finder modules.
"""
# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

from past.builtins import basestring
import datetime
import re

import chimedb.core
import chimedb.data_index

import peewee as pw
import numpy as np

# Logging
# =======

# Set default logging handler to avoid "No handlers could be found for logger
# 'layout'" warnings.
import logging

# All peewee-generated logs are logged to this namespace.
logger = logging.getLogger("_db_tables")
logger.addHandler(logging.NullHandler())


# Global variables and constants.
# ================================

_property = property  # Do this because we want a class named "property".
_user = None

#: Return events at the specified time.
EVENT_AT = 0
#: Return events before the specified time.
EVENT_BEFORE = 1
#: Return events after the specified time.
EVENT_AFTER = 2
#: Return all events (and ignore any specified time).
EVENT_ALL = 3
#: Order search results in ascending order.
ORDER_ASC = 0
#: Order search results in descending order.
ORDER_DESC = 1

# Exceptions
# ==========


class NoSubgraph(chimedb.core.CHIMEdbError):
    """Raise when a subgraph specification is missing."""


class BadSubgraph(chimedb.core.CHIMEdbError):
    """Raise when an error in subgraph specification is made."""


class DoesNotExist(chimedb.core.CHIMEdbError):
    """The event does not exist at the specified time."""


class UnknownUser(chimedb.core.CHIMEdbError):
    """The user requested is unknown."""


class NoPermission(chimedb.core.CHIMEdbError):
    """User does not have permission for a task."""


class LayoutIntegrity(chimedb.core.CHIMEdbError):
    """Action would harm the layout integrity."""


class PropertyType(chimedb.core.CHIMEdbError):
    """Bad property type."""


class PropertyUnchanged(chimedb.core.CHIMEdbError):
    """A property change was requested, but no change is needed."""


class ClosestDraw(chimedb.core.CHIMEdbError):
    """There is a draw for the shortest path to a given component type."""


# Helper classes for the peewee ORM
# =================================

from chimedb.core.orm import JSONDictField, EnumField, base_model, name_table


class event_table(base_model):
    """Baseclass for all models which are linked to the event class.

    Methods
    -------
    event
    """

    def event(
        self,
        time=datetime.datetime.now(),
        type=None,
        when=EVENT_ALL,
        order=None,
        active=True,
    ):
        """event(self, time = datetime.datetime.now(), type = None,\
                 when = EVENT_ALL, order = None, active = True)
           Find events associated with entries in this table.

        Parameters
        ----------
        time : datetime.datetime
          |event_time|
        type : :obj:`event_type`
          Only get events of the specified type.
        when : int
          |event_when|
        order : int or :obj:`None`
          |event_order|
        active : bool
          |event_active|

        Returns
        -------
        event : :obj:`peewee.SelectQuery`
        """
        ret = _graph_obj_iter(event, self.__class__, time, when, order, active).where(
            self.__class__.id == self.id
        )
        if type:
            try:
                dummy = iter(type)
                ret = ret.where(event.type << type)
            except TypeError:
                ret = ret.where(event.type == type)
        return ret


# Initializing connection to database.
# ====================================


def connect_peewee_tables(read_write=False, reconnect=False):
    """Initialize the connection to the CHIME data index database.

    This function uses the current database connector from
    :mod:`~chimedb.core` to establish a connection to the CHIME data
    index. It must be called if you change the connection method after
    importing this module. Or if you wish to connect with both read and write
    privileges.

    Parameters
    ----------
    read_write : bool
        Whether to connect with read and write privileges.
    reconnect : bool
        Force a reconnection.
    """

    chimedb.core.connect(read_write, reconnect)

    # Set the default, no-permissions user.
    set_user("Chime")


def set_user(u):
    """Identify yourself as a user, for record keeping.

    All events recorded in the database are associated with a user, and not all
    users have all permissions. You must call this function before making any
    changes to the database.

    Parameters
    ----------
    u : string or integer
      Your user identifier: the integer id, the username, or the full name.

    Raises
    ------
    UnknownUser
    """
    global _user

    _user = dict()

    # Find the user.
    if isinstance(u, int):
        q = chimedb.core.proxy.execute_sql(
            "SELECT user_id FROM chimewiki.user WHERE user_id = %d;" % u
        )
    else:
        q = chimedb.core.proxy.execute_sql(
            "SELECT user_id FROM chimewiki.user "
            "WHERE user_name = '%s' OR "
            "user_real_name = '%s';" % (u, u)
        )
    r = q.fetchone()
    if not r:
        raise UnknownUser("Could not find user.")
    _user["id"] = r[0]
    _user["perm"] = []

    # Get permissions.
    for r in (
        user_permission_type.select()
        .join(user_permission)
        .where(user_permission.user_id == _user["id"])
    ):
        _user["perm"].append(r.name)


def _check_user(perm):
    if not _user:
        raise UnknownUser(
            "You must call layout.set_user() before attempting to alter the DB."
        )
    if perm not in _user["perm"]:
        try:
            p = (
                user_permission_type.select()
                .where(user_permission_type.name == perm)
                .get()
            )
            raise NoPermission("You do not have the permissions to %s." % p.long_name)
        except pw.DoesNotExist:
            raise RuntimeError("Wow, code is broken.")


def _peewee_get_current_user():
    # Get the current user for peewee, working around the issues with creating
    # instances before the local user has been set. The particular issue here is
    # that peewee creates an instance with the default values before setting the
    # ones fetched from the database, so although it seems like you shouldn't
    # need the user to have been set, you do.

    if _user is None:
        return None

    return _user["id"]


# Tables in the DB pertaining to layouts
# ======================================


class graph_obj(base_model):
    """Parent table for any table that has events associated with it.
    This is a way to make the event table polymorphic. It points to this table,
    which shares (unique) primary keys with child tables (e.g., component). It
    only has one key: ID.

    Attributes
    ----------
    id
    """

    id = pw.AutoField()


class global_flag_category(base_model):
    """Categories for global flags.
    Examples of component types are antennas, 60m coaxial cables, and so on.

    Attributes
    ----------
    name : string
        The name of the category.
    notes : string
        An (optional) description of the category.
    """

    name = pw.CharField(max_length=255)
    notes = pw.CharField(max_length=65000, null=True)


class global_flag(event_table):
    """A simple flag index for global flags.

    Attributes
    ----------
    id : foreign key, primary key
        The ID shared with parent table graph_obj.
    category : foreign key
        The category of flag.
    severity  : enum('comment', 'warning', 'severe')
        An indication of how the data finder should react to this flag.
    inst : foreign key
        The acquisition instrument, if any, affected by this flag.
    name : string
        A short description of the flag.
    notes : string
        Notes about the global flag.
    """

    id = pw.ForeignKeyField(
        graph_obj, column_name="id", primary_key=True, backref="global_flag"
    )
    category = pw.ForeignKeyField(global_flag_category, backref="flag")
    severity = EnumField(["comment", "warning", "severe"])
    inst = pw.ForeignKeyField(chimedb.data_index.ArchiveInst, backref="flag", null=True)
    name = pw.CharField(max_length=255)
    notes = pw.CharField(max_length=65000, null=True)

    def start(self, time=datetime.datetime.now(), notes=None):
        """Start this global flag.

        Examples
        --------

        The following starts and ends a new global flag.

        >>> cat = layout.global_flag_category.get(name = "pass")
        >>> flag = layout.global_flag(category = cat, severity = "comment", name = "run_pass12_a").start(time = datetime.datetime(2015, 4, 1, 12))
        >>> flag.end(time = datetime.datetime(2015, 4, 5, 15, 30))

        Parameters
        ----------
        time : datetime.datetime
            The time at which the flag is to start.
        notes : string
            Any notes for the timestamp.

        Returns
        -------
        self : :obj:`global_flag`

        Raises
        ------
        :exc:`AlreadyExists` if the flag has already been started.
        """
        _check_user("global_flag")

        # Ensure that it has not been started before---i.e., that there is not an
        # active event associated with it.
        g = None
        try:
            g = global_flag.get(id=self.id)
        except pw.DoesNotExist:
            pass
        if g:
            try:
                g.event(time, event_type.global_flag()).get()
                raise AlreadyExists("This flag has already been started.")
            except pw.DoesNotExist:
                pass

        start = timestamp.create(time=time, notes=notes)
        if g:
            o = g.id
        else:
            o = graph_obj.create()
            self.id = o
            self.save(force_insert=True)
            g = self
        e = event.create(
            graph_obj=o, type=event_type.global_flag(), start=start, end=None
        )
        logger.info("Created global flag as event %d." % e.id)
        return g

    def end(self, time=datetime.datetime.now(), notes=None):
        """End this global flag.

        See :meth:`global_flag.start` for an example.

        Parameters
        ----------
        time : datetime.datetime
            The time at which the flag is to end.
        notes : string
            Any notes for the timestamp.

        Returns
        -------
        self : :obj:`global_flag`

        Raises
        ------
        :exc:`AlreadyExists` if the flag has already been ended; :exc:`DoesNotExist`
        if it has not been started.
        """
        _check_user("global_flag")

        # Ensure that it has been started but not ended.
        try:
            g = global_flag.get(id=self.id)
        except pw.DoesNotExist:
            raise DoesNotExist("This flag was never started.")
        try:
            e = g.event(time, event_type.global_flag()).get()
        except pw.DoesNotExist:
            raise DoesNotExist("This flag was never started.")
        try:
            e.end
            raise AlreadyExists("This event has already been ended.")
        except pw.DoesNotExist:
            pass

        end = timestamp.create(time=time, notes=notes)
        e.end = end
        e.save()
        logger.info("Ended global flag.")
        return self


class component_type(name_table):
    """A CHIME component type.
    Examples of component types are antennas, 60m coaxial cables, and so on.

    Attributes
    ----------
    name : string
        The name of the component type.
    notes : string
        An (optional) description of the component type.
    """

    name = pw.CharField(max_length=255)
    notes = pw.CharField(max_length=65000, null=True)


class component_type_rev(name_table):
    """A CHIME component type revision.

    Component types can, optionally, have revisions. For example, when an antenna
    design changes, a new revision is introduced.

    Attributes
    ----------
    type : foreign key
        The component type this revision applies to.
    name : string
        The name of the component type.
    notes : string
        An (optional) description of the component type.
    """

    type = pw.ForeignKeyField(component_type, backref="rev")
    name = pw.CharField(max_length=255)
    notes = pw.CharField(max_length=65000, null=True)


class external_repo(name_table):
    """Information on an external repository.

    Attributes
    ----------
    name : string
        The name of the repository.
    root : string
        Its location, e.g., a URL onto which individual paths to files can be
        appended.
    notes : string
        Any notes about this repository.
    """

    name = pw.CharField(max_length=255)
    root = pw.CharField(max_length=255)
    notes = pw.CharField(max_length=65000, null=True)


class component(event_table):
    """A CHIME component.

    To add or remove components, use the :meth:`add` and :meth:`remove` methods.
    There are also methods for getting and setting component properties, history
    and documents.

    Attributes
    ----------
    id : foreign key, primary key
        The ID shared with parent table graph_obj.
    sn : string, unique
        The unique serial number of the component.
    type : foreign key
        The component type.
    type_rev : foreign key
        The revision of this component.

    Methods
    -------
    get_connexion
    get_history
    get_doc
    add
    remove
    add_history
    add_doc
    get_property
    set_property
    """

    id = pw.ForeignKeyField(
        graph_obj, column_name="id", primary_key=True, backref="component"
    )
    sn = pw.CharField(max_length=255, unique=True, index=True)
    type = pw.ForeignKeyField(component_type, backref="component")
    type_rev = pw.ForeignKeyField(component_type_rev, backref="component", null=True)

    class Meta(object):
        indexes = (("sn"), True)

    def __hash__(self):
        # Reimplement the hash function. Peewee's default implementation, while
        # pretty sensible significantly slows down networkx when constructing a
        # graph as it is called a huge number of times
        return id(self)

    def get_connexion(
        self,
        comp=None,
        time=datetime.datetime.now(),
        when=EVENT_AT,
        order=ORDER_ASC,
        active=True,
    ):
        """Get connexions involving this component.

        Parameters
        ----------
        comp : :obj:`component`
            If this parameter is set, then search for connexions between this
            component and *comp*.
        time : datetime.datetime
            |event_time|
        when : int
            |event_when|
        order : int
            |event_order|
        active : bool
            |event_active|

        Returns
        -------
        A :obj:`peewee.SelectQuery` for :class:`connexion` entries.
        """
        c = _graph_obj_iter(connexion, event, time, when, order, active).where(
            (connexion.comp1 == self) | (connexion.comp2 == self)
        )
        if comp:
            return c.where((connexion.comp1 == comp) | (connexion.comp2 == comp)).get()
        else:
            return c

    def get_history(
        self, time=datetime.datetime.now(), when=EVENT_AT, order=ORDER_ASC, active=True
    ):
        """Get history items associated with this component.

        Parameters
        ----------
        time : datetime.datetime
            |event_time|
        when : int
            |event_when|
        order : int
            |event_order|
        active : bool
            |event_active|

        Returns
        -------
        A :obj:`peewee.SelectQuery` for :class:`history` entries.
        """
        return _graph_obj_iter(
            component_history, event, time, EVENT_AT, None, True
        ).where(component_history.comp == self)

    def get_doc(self, time=datetime.datetime.now()):
        """Get document pointers associated with this component.

        Parameters
        ----------
        time : datetime.datetime
            The time at which the document events should be active.

        Returns
        -------
        A :obj:`peewee.SelectQuery` for :class:`component_doc` entries.
        """
        return _graph_obj_iter(component_doc, event, time, EVENT_AT, None, True).where(
            component_doc.comp == self
        )

    def add(self, time=datetime.datetime.now(), notes=None, force=True):
        """Add this component.

        This triggers the "component available" event.

        To add many components at once, see :func:`add_component`.

        Examples
        --------

        The following makes a new LNA available:

        >>> lna_type = layout.component_type.get(name = "LNA")
        >>> lna_rev = lna_type.rev.where(layout.component_type_rev.name == "B").get()
        >>> comp = layout.component(sn = "LNA0000A", type = lna_type, rev = lna_type.rev).add()

        Parameters
        ----------
        time : datetime.datetime
            The time at which the component is to be made available.
        notes : string
            Any notes for the timestamp.
        force : bool
            If :obj:`False`, then raise :exc:`AlreadyExists` if this event creates a
            conflict; otherwise, do not add but ignore on conflict.

        Returns
        -------
        self : :obj:`component`
        """
        add_component(self, time, notes, force)
        return self

    def remove(self, time=datetime.datetime.now(), notes=None, force=False):
        """Remove this component.

        This ends the "component available" event.

        To remove many components at once, see :func:`remove_component`.

        Parameters
        ----------
        time : datetime.datetime
            The time at which the component is to be removed.
        notes : string
            Any notes for the timestamp.
        force : bool
            If :obj:`False`, then raise :exc:`DoesNotExist` if this event creates a
            conflict; otherwise, do not add but ignore on conflict.
        """
        remove_component(self, time, notes, force)

    def add_history(self, notes, time=datetime.datetime.now(), timestamp_notes=None):
        """Add a history item for this component.

        Parameters
        ----------
        notes : string
            The history note.
        time : datetime.datetime
            The time at which the history is to be added.
        timestamp_notes : string
            Any notes for the timestamp.

        Returns
        -------
        history : :obj:`component_history`
            The newly-created component history object.
        """
        _check_user("comp_info")
        o = graph_obj.create()
        h = component_history.create(id=o, comp=self, notes=notes)
        t_stamp = timestamp.create(time=time, notes=timestamp_notes)
        e = event.create(graph_obj=o, type=event_type.comp_history(), start=t_stamp)
        logger.info("Added component history as event %d." % e.id)
        return h

    def add_doc(self, repo, ref, time=datetime.datetime.now(), notes=None):
        """Add a document pointer for this component.

        Parameters
        ----------
        repo : :obj:`external_repo`
            The place where the document is.
        ref : string
            A path or similar pointer, relevative to the root of *repo*.
        time : datetime.datetime
            The time at which the document pointer is to be added.
        notes : string
            Any notes for the timestamp.

        Returns
        -------
        history : :obj:`component_doc`
            The newly-created document pointer object.
        """

        _check_user("comp_info")
        try:
            external_repo.get(id=repo.id)
        except pw.DoesNotExist:
            raise DoesNotExist("Repository does not exist in the DB. Create it first.")
        o = graph_obj.create()
        d = component_doc.create(id=o, comp=self, repo=repo, ref=ref)
        t_stamp = timestamp.create(time=time, notes=notes)
        e = event.create(graph_obj=o, type=event_type.comp_doc(), start=t_stamp)
        logger.info("Added component document as event %d." % e.id)
        return d

    def get_property(self, type=None, time=datetime.datetime.now()):
        """Get a property for this component.

        Parameters
        ----------
        type : :obj:`property_type`
            The property type to search for.
        time : obj:`datetime.datetime`
            The time at which to get the property.

        Returns
        -------
        property : string
            If no property is set, then :obj:`None` is returned.
        """
        p = _graph_obj_iter(property, event, time, EVENT_AT, None, True).where(
            property.comp == self
        )
        if type:
            return p.where(property.type == type).get()
        else:
            return p

    def set_property(self, type, value, time=datetime.datetime.now(), notes=None):
        """Set a property for this component.

        Parameters
        ----------
        type : :obj:`property_type`
            The property type to search for.
        value : string
            The value to set.
        time : obj:`datetime.datetime`
            The time at which to get the property.
        notes : string
            Notes for the timestamp.

        Raises
        ------
        :exc:ValueError:, if *value* does not conform to the property regular
        expression.
        """
        set_property(self, type, value, time=time, notes=notes, force=True)

    def __repr__(self):
        # Format a representation of the object
        # At the moment, cannot include the type info as it generates another query
        fmt = "<component serial='%s'>"
        return fmt % self.sn


class component_history(event_table):
    """For providing history information on a component.

    Attributes
    ----------
    id : foreign key, primary key
        The ID shared with parent table graph_obj.
    comp : foreign key
        The component linked to the history.
    notes : string
        The history information.
    """

    id = pw.ForeignKeyField(
        graph_obj, column_name="id", primary_key=True, backref="history"
    )
    comp = pw.ForeignKeyField(
        component, column_name="comp_sn", field="sn", backref="history"
    )
    notes = pw.CharField(max_length=65000)


class component_doc(event_table):
    """For linking a component to a document in an external repository.

    Attributes
    ----------
    id : foreign key, primary key
        The ID shared with parent table graph_obj.
    comp : foreign key
        The component linked to the document.
    repo : foreign key
        The repository holding the document.
    ref : string
        The location of the document within the repository (e.g., a filename).
    """

    id = pw.ForeignKeyField(
        graph_obj, column_name="id", primary_key=True, backref="doc"
    )
    comp = pw.ForeignKeyField(
        component, column_name="comp_sn", field="sn", backref="doc"
    )
    repo = pw.ForeignKeyField(external_repo, backref="doc")
    ref = pw.CharField(max_length=65000)


class connexion(event_table):
    """A connexion between two components.

    This should always be instatiated using the from_pair() method.

    Attributes
    ----------
    id : foreign key, primary key
        The ID shared with parent table graph_obj.
    comp1 : foreign key
        The first component in the connexion.
    comp2 : foreign key
        The second component in the connexion.

    Methods
    -------
    from_pair
    is_connected
    is_permanent
    make
    other_comp
    sever
    """

    id = pw.ForeignKeyField(
        graph_obj, column_name="id", primary_key=True, backref="connexion"
    )
    comp1 = pw.ForeignKeyField(
        component, column_name="comp_sn1", field="sn", backref="conn1"
    )
    comp2 = pw.ForeignKeyField(
        component, column_name="comp_sn2", field="sn", backref="conn2"
    )

    class Meta(object):
        indexes = (("component_sn1", "component_sn2"), True)

    @classmethod
    def from_pair(cls, comp1, comp2, allow_new=True):
        """Get a :obj:`connexion` given a pair of components.

        Parameters
        ----------
        comp1 : str or :obj:`component`
            Pass either the serial number or a :obj:`component` object.
        comp2 : str or :obj:`component`
            Pass either the serial number or a :obj:`component` object.
        allow_new : bool
            If :obj:`False`, then raise :exc:`peewee.DoesNotExist` if the connexion
            does not exist at all in the database.

        Returns
        -------
        connexion : :obj:`connexion`
        """
        pair = []
        for comp in (comp1, comp2):
            if isinstance(comp, component):
                comp = comp.sn
            try:
                pair.append(component.get(sn=comp))
            except pw.DoesNotExist:
                raise DoesNotExist("Component %s does not exist." % comp)
        q = cls.select().where(
            ((cls.comp1 == pair[0]) & (cls.comp2 == pair[1]))
            | ((cls.comp1 == pair[1]) & (cls.comp2 == pair[0]))
        )
        if allow_new:
            try:
                return q.get()
            except:
                return cls(comp1=pair[0], comp2=pair[1])
        else:
            return q.get()

    def is_connected(self, time=datetime.datetime.now()):
        """See if a connexion exists.

        Connexions whose events have been deactivated are not included.

        Parameters
        ----------
        time : datetime
            The time at which to check whether the connexion exists.

        Returns
        -------
        connected : bool
            :obj:`True` if there is a connexion, otherwise :obj:`False`.

        Raises
        ------
        peewee.DoesNotExist
            Raised if one or both of the components does not exist.
        """
        try:
            self.event(
                time=time,
                type=(event_type.connexion(), event_type.perm_connexion()),
                when=EVENT_AT,
            ).get()
            return True
        except pw.DoesNotExist:
            return False

    def is_permanent(self, time=datetime.datetime.now()):
        """See if a permenant connexion exists.

        Connexions whose events have been deactivated are not included.

        Parameters
        ----------
        time : datetime
            The time at which to check whether the connexion exists.

        Returns
        -------
        connected : bool
            :obj:`True` if there is a permanent connexion, otherwise :obj:`False`.

        Raises
        ------
        peewee.DoesNotExist
            Raised if one or both of the components does not exist.
        """
        try:
            self.event(time=time, type=event_type.perm_connexion(), when=EVENT_AT).get()
            return True
        except pw.DoesNotExist:
            return False

    def other_comp(self, comp):
        """Given one component in the connexion, return the other.

        Parameters
        ----------
        comp : :obj:`component`
            The component you know in the connexion.

        Returns
        -------
        :obj:`component`
            The other component in the connexion, i.e., the one that isn't *comp*.

        Raises
        ------
        :exc:`DoesNotExist`
            If *comp* is not part of this connexion, an exception occurs.
        """
        if self.comp1 == comp:
            return self.comp2
        elif self.comp2 == comp:
            return self.comp1
        else:
            raise DoesNotExist(
                "The component you passed is not part of this connexion."
            )

    def make(
        self, time=datetime.datetime.now(), permanent=False, notes=None, force=False
    ):
        """Create a connexion.
        This method begins a connexion event at the specified time.

        Parameters
        ----------
        time : datetime.datetime
            The time at which to begin the connexion event.
        permanent : bool
            If :obj:`True`, then make this a permanent connexion.
        notes : string
            Any notes for the timestamp.
        force : bool
            If :obj:`False`, then :exc:`AlreadyExists` will be raised if the connexion
            already exists; otherwise, conflicts will be ignored and nothing will be
            done.

        Returns
        -------
        connexion : :obj:`connexion`
        """
        make_connexion(self, time, permanent, notes, force)
        return connexion.from_pair(self.comp1, self.comp2)

    def sever(self, time=datetime.datetime.now(), notes=None, force=False):
        """Sever a connexion.
        This method ends a connexion event at the specified time.

        Parameters
        ----------
        time : datetime.datetime
            The time at which to end the connexion event.
        notes : string
            Any notes for the timestamp.
        force : bool
            If :obj:`False`, then :exc:`DoesNotExists` will be raised if the connexion
            does not exist; otherwise, conflicts will be ignored and nothing will be
            done.
        """
        sever_connexion(self, time, notes, force)


class property_type(name_table):
    """A component property type.

    Attributes
    ----------
    name : string
        The name of the property type (e.g., "attenuation").
    units : string
        The (optional) units of the property (e.g., "dB").
    regex : string
        An (optional) regular expression for controlling allowed property values.
    notes : string
        Any (optional) notes further explaining the property.
    """

    name = pw.CharField(max_length=255)
    units = pw.CharField(max_length=255, null=True)
    regex = pw.CharField(max_length=255, null=True)
    notes = pw.CharField(max_length=65000, null=True)


class property_component(base_model):
    """A list associating property types with components.
    A property can be for one or more component types. For example,
    "dist_from_n_end" is only a property of cassettes, but "termination" may be a
    property of LNA's, FLA's and so on. This is simply a table for matching
    property types to component types.

    Attributes
    ----------
    prop_type : foreign key
        The property type to be mapped.
    comp_type : foreign key
        The component type to be mapped.
    """

    prop_type = pw.ForeignKeyField(property_type, backref="property_component")
    comp_type = pw.ForeignKeyField(component_type, backref="property_component")

    class Meta(object):
        indexes = (("prop_type", "comp_type"), True)


class property(event_table):
    """A property associated with a particular component.

    Attributes
    ----------
    id : foreign key, primary key
        The ID shared with parent table graph_obj.
    comp : foreign key
        The component to which this property belongs.
    type : foreign key
        The property type.
    value : string
        The actual property.
    """

    id = pw.ForeignKeyField(
        graph_obj, column_name="id", primary_key=True, backref="property"
    )
    comp = pw.ForeignKeyField(
        component, column_name="comp_sn", backref="property", field="sn"
    )
    type = pw.ForeignKeyField(property_type, backref="property")
    value = pw.CharField(max_length=255)

    class Meta(object):
        indexes = (("comp_sn, type_id"), False)


class event_type(name_table):
    """For differentiating event types.

    The class methods :meth:`comp_avail`, :meth:`connexion` and so on return
    event type instances, and internally store the result. Thus, subsequent calls
    do not generate more database queries. This can reduce overhead.

    Attributes
    ----------
    name : string
        The name of the event type.
    human_name : string
        A proper, English name.
    assoc_table : string
        The (optional) table that this event is about; it should be a child of
        graph_obj.
    no_end : enum('Y', 'N')
        If 'Y', then this is an "instantaneous" event, i.e., there will never be
        recorded an end.
    require_notes : enum('Y', 'N')
        If 'Y', then the notes of the event _must_ be set.
    notes : string
        Any notes about this event type.
    """

    name = pw.CharField(max_length=255)
    human_name = pw.CharField(max_length=255)
    assoc_table = pw.CharField(max_length=255, null=True)
    no_end = EnumField(["Y", "N"], default="N")
    require_notes = EnumField(["Y", "N"], default="N")
    notes = pw.CharField(max_length=65000, null=True)

    @classmethod
    def comp_avail(cls):
        """For getting the component available event type."""
        return cls.from_name("comp_avail")

    @classmethod
    def connexion(cls):
        """For getting the connexion event type."""
        return cls.from_name("connexion")

    @classmethod
    def property(cls):
        """For getting the property event type."""
        return cls.from_name("property")

    @classmethod
    def perm_connexion(cls):
        """For getting the permanent connexion event type."""
        return cls.from_name("perm_connexion")

    @classmethod
    def comp_history(cls):
        """For getting the component history event type."""
        return cls.from_name("comp_history")

    @classmethod
    def comp_doc(cls):
        """For getting the component document event type."""
        return cls.from_name("comp_doc")

    @classmethod
    def global_flag(cls):
        return cls.from_name("global_flag")


class timestamp(base_model):
    """A timestamp.

    Attributes
    ----------
    time : datetime
        The timestamp.
    entry_time : datetime
        The creation time of the timestamp.
    user_id : foreign key
        In the actual DB, this is a foreign key to chimewiki.user(user_id), but
        peewee doesn't support foreign keys to different schemas.
    notes : string
        Any (optional) notes about the timestamp.
    """

    # Removed problematic constructor and replaced functionality with
    # the fact that peewee supports callables as default arguments.

    time = pw.DateTimeField(default=datetime.datetime.now)
    entry_time = pw.DateTimeField(default=datetime.datetime.now)
    user_id = pw.IntegerField(default=_peewee_get_current_user)
    notes = pw.CharField(max_length=65000, null=True)


class event(base_model):
    """An event, or timestamp, for something graphy.

    *Never* manually create, delete or alter events! Doing so can damage the
    integrity of the database.

    To interact with events, use:

    * :meth:`component.add`, :func:`add_component`, :meth:`component.remove` and
      :func:`remove_component` for starting and ending component
    * :meth:`connexion.make` and :meth:`connexion.sever` for making and severing
      connexions
    * :meth:`component.set_property` for starting/ending component properties
    * :meth:`component.add_history` and :meth:`component.add_doc` for adding
      component history and documents.
    * :meth:`global_flag.start`, :meth:`global_flag.end` to set a global flag.

    You can safely deactivate an event using :meth:`deactivate`; this method only
    allows deactivation if it will not damage the database integrity.

    Attributes
    ----------
    active : bool
        Is this event active? (Instead of deleting events, we deactivate them.)
    replaces : foreign key
        Instead of editing events, we replace them, so that we have a history of
        event edits. This key indicates which event (if any) this event replaces.
    graph_obj : foreign key
        Which graph object is this event about?
    type : foreign key
        What kind of event is it?
    start : foreign key
        The timestamp for the event start.
    end : foreign key
        The timestamp for the event end.

    Methods
    -------
    deactivate
    """

    active = pw.BooleanField(default=True)
    replaces = pw.ForeignKeyField("self", null=True, backref="replacement")
    graph_obj = pw.ForeignKeyField(graph_obj, backref="event")
    type = pw.ForeignKeyField(event_type, backref="event")
    start = pw.ForeignKeyField(timestamp, backref="event_start")
    end = pw.ForeignKeyField(timestamp, backref="event_end")

    class Meta(object):
        indexes = ((("type_id"), False), (("start", "end"), False))

    def _event_permission(self):
        t = self.type
        if t == event_type.comp_avail():
            _check_user("comp_avail")
        elif t == event_type.connexion() or t == event_type.perm_connexion():
            _check_user("connexion")
        elif t == event_type.comp_history() or t == event_type.comp_doc():
            _check_user("comp_info")
        elif t == event_type.property():
            _check_user("property")
        elif t == event_type.global_flag():
            _check_user("global_flag")
        # Layout notes need to be reworked.
        # elif t == event_type.layout_note():
        #  _check_user("layout_note")
        else:
            raise NoPermission("This layout type cannot be deactivated.")

    def deactivate(self):
        """Deactivate an event.

        Events are never deleted; rather, the :attr:`active` flag is switched off.
        This method first checks to see whether doing so would break database
        integrity, and only deactivates if it will not.

        Raises
        ------
        :exc:LayoutIntegrity: if deactivating will compromise layout integrity.
        """
        self._event_permission()
        fail = []

        if not self.active:
            logger.info("Event %d is already deactivated." % (self.id))
            return

        # If this is about component availability, do not deactivate if it is
        # connected, or if it has any properties, history or documents.
        if self.type == event_type.comp_avail():
            comp = self.graph_obj.component.get()

            # Check history.
            for e in _graph_obj_iter(
                event, component_history, None, EVENT_ALL, None, True
            ).where(component_history.comp == comp):
                fail.append(str(e.id))
            _check_fail(
                fail,
                False,
                LayoutIntegrity,
                "Cannot deactivate because "
                "the following history event%s %s set for this "
                "component" % (_plural(fail), _are(fail)),
            )

            # Check documents.
            for e in _graph_obj_iter(
                event, component_doc, None, EVENT_ALL, None, True
            ).where(component_doc.comp == comp):
                fail.append(str(e.id))
            _check_fail(
                fail,
                False,
                LayoutIntegrity,
                "Cannot deactivate because "
                "the following document event%s %s set for this "
                "component" % (_plural(fail), _are(fail)),
            )

            # Check properties.
            for e in _graph_obj_iter(
                event, property, None, EVENT_ALL, None, True
            ).where(property.comp == comp):
                fail.append(str(e.id))
            _check_fail(
                fail,
                False,
                LayoutIntegrity,
                "Cannot deactivate because "
                "the following property event%s %s set for this "
                "component" % (_plural(fail), _are(fail)),
            )

            # Check connexions.
            for conn in comp.get_connexion(when=EVENT_ALL):
                fail.append("%s<->%s" % (conn.comp1.sn, conn.comp2.sn))
            _check_fail(
                fail,
                False,
                LayoutIntegrity,
                "Cannot deactivate because "
                "the following component%s are connected" % (_plural(fail)),
            )

        self.active = False
        self.save()
        logger.info("Deactivated event %d." % self.id)

    def _replace(self, start=None, end=None, force_end=False):
        """Replace one or both timestamps for an event.

        Currently, the following is not supported and will raise a
        :exc:`RuntimeError` exception:

        - replacing the end time of a component availability event;
        - replacing the start time of a component availability event if the start
          time is *later* than the current start time.

        Parameters
        ----------
        start : :obj:`timestamp`
            The new starting timestamp. If :obj:`None`, then the starting timestamp is
            not altered.
        end : :obj:`timestamp`
            The new end timestamp. If this is set to :obj:`None` *and* **force_end**
            is :obj:`True`, then the event will be set with no end time.
        force_end : bool
            If :obj:`True`, then a value of :obj:`None` for **end** is interpreted as
            having no end time; otherwise, :obj:`None` for **end** will not change the
            end timestamp.

        Returns
        -------
        event : :obj:`event`
            The modified event.
        """
        self._event_permission()
        if self.type == event_type.comp_avail():
            if end or force_end:
                raise RuntimeError(
                    "This method does not currently support ending "
                    "component availability events."
                )
            if start.time > self.start.time:
                raise RuntimeError(
                    "This method does not currently support moving a "
                    "component availability event later."
                )
        if start == None:
            start = self.start
        else:
            try:
                timestamp.get(id=start.id)
            except pw.DoesNotExist:
                start.save()
        if end == None:
            if not force_end:
                end = _pw_getattr(self, "end", None)
        else:
            if end.time < start.time:
                raise LayoutIntegrity("End time cannot be earlier than start time.")
            try:
                timestamp.get(id=end.id)
            except pw.DoesNotExist:
                end.save()
        self.active = False
        self.save()

        new = event.create(
            replaces=self,
            graph_obj=self.graph_obj,
            type=self.type,
            start=start,
            end=end,
        )
        self = new
        return self


class predef_subgraph_spec(name_table):
    """A specification for a subgraph of a full graph.

    Attributes
    ----------
    name : string
        The name of this subgraph specification.
    start_type : foreign key
        The starting component type.
    notes : string
        Optional notes about this specification.
    """

    name = pw.CharField(max_length=255)
    start_type = pw.ForeignKeyField(
        component_type, backref="predef_subgraph_spec_start"
    )
    notes = pw.CharField(max_length=65000, null=True)


class predef_subgraph_spec_param(base_model):
    """Parameters for a subgraph specification.

    Attributes
    ----------
    predef_subgraph_spec : foreign key
        The subgraph which this applies.
    type1 : foreign key
        A component type.
    type2 : foreign key
        A component type.
    action : enum('T', 'H', 'O')
        The role of this component type:
        - T: terminate at type1 (type2 is left NULL).
        - H: hide type1 (type2 is left NULL).
        - O: only draw connexions one way between type1 and type2.
    """

    predef_subgraph_spec = pw.ForeignKeyField(predef_subgraph_spec, backref="param")
    type1 = pw.ForeignKeyField(component_type, backref="subgraph_param1")
    type2 = pw.ForeignKeyField(component_type, backref="subgraph_param2", null=True)
    action = EnumField(["T", "H", "O"])

    class Meta(object):
        indexes = (("predef_subgraph_spec", "type", "action"), False)


class user_permission_type(name_table):
    """Defines permissions for the DB interface.

    Attributes
    ----------
    name : string
        The name of the permission.
    notes : string
        An (optional) description of the permission.
        peewee doesn't support foreign keys to different schemas.
    """

    name = pw.CharField(max_length=255)
    long_name = pw.CharField(max_length=65000)


class user_permission(base_model):
    """Specifies users' permissions.

    Attributes
    ----------
    user_id : foreign key
        In the actual DB, this is a foreign key to chimewiki.user(user_id), but
        peewee doesn't support foreign keys to different schemas.
    type : foreign key
        The permission type to grant to the user.
    """

    user_id = pw.IntegerField()
    type = pw.ForeignKeyField(user_permission_type, backref="user")

    class Meta(object):
        indexes = (("user_id", "type"), False)


class DataFlagType(base_model):
    """The type of flag that we are using.

    Attributes
    ----------
    name : string
        Name of the type of flag.
    description : string
        A long description of the flag type.
    metadata : dict
        An optional JSON object describing how this flag type is being generated.
    """

    name = pw.CharField(max_length=64)
    description = pw.TextField(null=True)
    metadata = JSONDictField(null=True)


class DataFlag(base_model):
    """A flagged range of data.

    Attributes
    ----------
    type : DataFlagType
        The type of flag.
    start_time, finish_time : double
        The start and end times as UNIX times.
    metadata : dict
        A JSON object with extended metadata. See below for guidelines.

    Notes
    -----
    To ensure that the added metadata is easily parseable, it should adhere
    to a rough schema. The following common fields may be present:

    `instrument` : optional
        The name of the instrument that the flags applies to. If not set,
        assumed to apply to all instruments.
    `freq` : optional
        A list of integer frequency IDs that the flag applies to. If not
        present the flag is assumed to apply to *all* frequencies.
    `inputs` : optional
        A list of integer feed IDs (in cylinder order) that the flag applies
        to. If not present the flag is assumed to apply to *all* inputs. For
        this to make sense an `instrument` field is also required.

    Any other useful metadata can be put straight into the metadata field,
    though it must be accessed directly.
    """

    type = pw.ForeignKeyField(DataFlagType, backref="flags")

    start_time = pw.DoubleField()
    finish_time = pw.DoubleField()

    metadata = JSONDictField(null=True)

    @_property
    def instrument(self):
        """The instrument the flag applies to."""
        if self.metadata is not None:
            return self.metadata.get("instrument", None)
        else:
            return None

    @_property
    def freq(self):
        """The list of inputs the flag applies to. `None` if not set.
        """
        if self.metadata is not None:
            return self.metadata.get("freq", None)
        else:
            return None

    @_property
    def freq_mask(self):
        """An array for the frequencies flagged (`True` if the flag applies).
        """

        # TODO: hard coded for CHIME
        mask = np.ones(1024, dtype=np.bool)

        if self.freq is not None:
            mask[self.freq] = False
            mask = ~mask

        return mask

    @_property
    def inputs(self):
        """The list of inputs the flag applies to. `None` if not set.
        """
        if self.metadata is not None:
            return self.metadata.get("inputs", None)
        else:
            return None

    @_property
    def input_mask(self):
        """An array for the inputs flagged (`True` if the flag applies).
        """
        if self.instrument is None:
            return None

        inp_dict = {"chime": 2048, "pathfinder": 256}
        mask = np.ones(inp_dict[self.instrument], dtype=np.bool)

        if self.inputs is not None:
            mask[self.inputs] = False
            mask = ~mask

        return mask

    @classmethod
    def create_flag(
        cls,
        flagtype,
        start_time,
        finish_time,
        freq=None,
        instrument="chime",
        inputs=None,
        metadata=None,
    ):
        """Create a flag entry.

        Parameters
        ----------
        flagtype : string
            Name of flag type. Must already exist in database.
        start_time, end_time : float
            Start and end of flagged time.
        freq : list, optional
            List of affected frequencies.
        instrument : string, optional
            Affected instrument.
        inputs : list, optional
            List of affected inputs.
        metadata : dict
            Extra metadata to go with the flag entry.

        Returns
        -------
        flag : PipelineFlag
            The flag instance.
        """

        table_metadata = {}

        if freq is not None:
            if not isinstance(freq, list):
                raise ValueError("freq argument (%s) must be list.", freq)
            table_metadata["freq"] = freq

        if instrument is not None:
            table_metadata["instrument"] = instrument

        if inputs is not None:
            if not isinstance(inputs, list):
                raise ValueError("inputs argument (%s) must be list.", inputs)
            table_metadata["inputs"] = inputs

        if metadata is not None:
            if not isinstance(metadata, dict):
                raise ValueError("metadata argument (%s) must be dict.", metadata)
            table_metadata.update(metadata)

        # Get the flag
        type_ = DataFlagType.get(name=flagtype)

        flag = cls.create(
            type=type_,
            start_time=start_time,
            finish_time=finish_time,
            metadata=table_metadata,
        )

        return flag


# Functions for manipulating layout tables in a robust way.
# =========================================================


def _graph_obj_iter(sel, obj, time, when, order, active):
    ret = sel.select()
    if sel.__name__ == "event":
        ret = ret.join(obj, on=(sel.graph_obj == obj.id))
    else:
        ret = ret.join(obj, on=(sel.id == obj.graph_obj))

    if when == EVENT_AT:
        start = timestamp.alias()
        end = timestamp.alias()
        ret = (
            ret.switch(event)
            .join(start, on=(start.id == event.start))
            .switch(event)
            .join(end, on=(end.id == event.end), join_type=pw.JOIN.LEFT_OUTER)
            .where((start.time <= time) & ((end.time >> None) | (end.time > time)))
        )
    elif when == EVENT_BEFORE:
        ret = (
            ret.switch(event)
            .join(timestamp, on=(timestamp.id == event.end))
            .where(timestamp.time < time)
        )
    elif when == EVENT_AFTER:
        ret = (
            ret.switch(event)
            .join(timestamp, on=(timestamp.id == event.start))
            .where(timestamp.time > time)
        )

    if active:
        ret = ret.where(event.active == True)

    if (not when == EVENT_AT) and order:
        if order == ORDER_ASC:
            ret = ret.order_by(timestamp.time.asc())
        elif order == ORDER_DESC:
            ret = ret.order_by(timestamp.time.desc())
        else:
            raise ValueError("Unknown value of 'when' passed (%d)." % when)

    return ret


def _pw_getattr(obj, attr, default):
    try:
        return getattr(obj, attr)
    except pw.DoesNotExist:
        return default


def _check_property_type(ptype, ctype):
    try:
        property_type.get(id=ptype.id)
    except pw.DoesNotExist:
        raise DoesNotExist("Property type does not exist in the DB.")
    try:
        property_type.select().where(property_type.id == ptype).join(
            property_component
        ).where(property_component.comp_type == ctype).get().name
    except pw.DoesNotExist:
        raise PropertyType(
            'Property type "%s" cannot be used for component '
            'type "%s".' % (ptype.name, ctype.name)
        )


def _check_fail(fail, force, exception, msg):
    if len(fail):
        msg = "%s: %s" % (msg, ", ".join(fail))
        if force:
            logger.debug(msg)
        else:
            raise exception(msg)


def _conj(l):
    if len(l) == 1:
        return "s"
    else:
        return ""


def _plural(l):
    if len(l) == 1:
        return ""
    else:
        return "s"


def _does(l):
    if len(l) == 1:
        return "does"
    else:
        return "do"


def _are(l):
    if len(l) == 1:
        return "is"
    else:
        return "are"


def compare_connexion(conn1, conn2):
    """See if two connexions are the same.
    Because the :class:`connexion` could store the two components in different
    orders, or have different instances of the same component object, direct
    comparison may fail. This function explicitly compares both possible
    combinations of serial numbers.

    Parameters
    ----------
    conn1 : :obj:`connexion`
        The first connexion object.
    conn2 : :obj:`connexion`
        The second connexion object.

    Returns
    -------
    :obj:`True` if the connexions are the same, :obj:`False` otherwise.
    """
    sn11 = conn1.comp1.sn
    sn12 = conn1.comp2.sn
    sn21 = conn2.comp1.sn
    sn22 = conn2.comp2.sn

    if (sn11 == sn21 and sn12 == sn22) or (sn11 == sn22 and sn12 == sn21):
        return True
    else:
        return False


def add_global_flag(
    name,
    start_time=datetime.datetime.now(),
    end_time=None,
    notes=None,
    start_notes=None,
    end_notes=None,
):
    """Add a global flag.

    A global flag is an event that labels the configuration over a specified time
    for future reference.

    Parameters
    ----------
    name : string
        The name for the flag.
    start_time : datetime.datetime
        The time at which the flag starts.
    end_time : datetime.datetime
        The time at which the flag ends. It can be set to :obj:`None` to leave the
        flag open-ended.
    notes : string
        Any notes for the flag.
    start_notes : string
        Any notes for the start timestamp.
    end_notes : string
        Any notes for the end timestamp.

    Returns
    -------
    The global flag object :obj:`global_flag`.
    """
    return t


def add_component(comp, time=datetime.datetime.now(), notes=None, force=False):
    """Make one or more components available with a common timestamp.

    If you are adding only one component, this function is equivalent to calling
    :meth:`component.add`. However, multiple calls to :meth:`component.add`
    generate a unique timestamp per call. To assign a single timestamp to many
    additions at once, use this function.

    Examples
    --------
    >>> lna_type = layout.component_type.get(name = "LNA")
    >>> lna_rev = lna_type.rev.where(layout.component_type_rev.name == "B").get()
    >>> c = []
    >>> for i in range(0, 10):
    ...   c.append(layout.component(sn = "LNA%04dB" % (i), type = lna_type, rev = lna_rev))
    >>> layout.add_component(c, time = datetime(2014, 10, 10, 11), notes = "Adding many at once.")

    Parameters
    ----------
    comp : list of :obj:`component` objects
        The components to make available.
    time : datetime.datetime
        The time at which to make the components available.
    notes : string
        Any notes for the timestamp.
    force : bool
        If :obj:`True`, then add any components that can be added, while doing
        nothing (except making note of such in the logger) for components whose
        addition would violate database integrity. If :obj:`False`,
        :exc:`AlreadyExists` is raised for any addition that violates database
        integrity.
    """
    import copy

    _check_user("comp_avail")
    if isinstance(comp, component):
        comp_list = [comp]
    else:
        comp_list = comp

    # First check to see that the component does not already exist at this time.
    fail = []
    to_add = []
    to_add_sn = []
    for comp in comp_list:
        try:
            c = component.get(sn=comp.sn)
        except pw.DoesNotExist:
            to_add.append(comp)
            to_add_sn.append(comp.sn)
            continue
        try:
            c.event(time, event_type.comp_avail(), EVENT_AT).get()
            fail.append(c.sn)
        except:
            to_add.append(comp)
            to_add_sn.append(comp.sn)

    # Also add permanently connected components.
    done = copy.deepcopy(to_add)
    add = []
    sn = []
    for c in to_add:
        try:
            component.get(sn=c.sn)
            this_add, this_sn = _get_perm_connexion_recurse(c, time, done)
            add += this_add
            sn += this_sn
            to_add += add
            to_add_sn += sn
        except pw.DoesNotExist:
            # If the component doesn't exist in the DB, then it can't have any
            # permanent connexions.
            pass

    _check_fail(
        fail,
        force,
        AlreadyExists,
        "Aborting because the following "
        "component%s %s already available at that time" % (_plural(fail), _are(fail)),
    )

    if len(to_add):
        t_stamp = timestamp.create(time=time, notes=notes)
    for comp in to_add:
        try:
            comp = component.get(sn=comp.sn)
        except pw.DoesNotExist:
            o = graph_obj.create()
            comp.id = o
            comp.save(force_insert=True)

        try:
            # If the component is already available after this time, replace it with
            # an event starting at this new time.
            e_old = comp.event(
                time, event_type.comp_avail(), EVENT_AFTER, ORDER_ASC
            ).get()
            e_old._replace(start=t_stamp)
            logger.debug(
                "Added %s by replacing previous event %d." % (comp.sn, e_old.id)
            )
        except pw.DoesNotExist:
            e = event.create(
                graph_obj=comp.id, type=event_type.comp_avail(), start=t_stamp
            )
            logger.debug("Added %s with new event %d." % (comp.sn, e.id))
    if len(to_add):
        logger.info(
            "Added %d new component%s: %s"
            % (len(to_add), _plural(to_add), ", ".join(to_add_sn))
        )
    else:
        logger.info("Added no new component.")


def _get_perm_connexion_recurse(comp, time, done=[]):
    add = []
    add_sn = []

    if comp not in done:
        add.append(comp)
        add_sn.append(comp.sn)
    done.append(comp)

    for conn in comp.get_connexion(time=time):
        if conn.is_permanent():
            c2 = conn.other_comp(comp)
            if c2 not in done:
                a, s = _get_perm_connexion_recurse(c2, time, done)
                add += a
                add_sn += s

    return add, add_sn


def _check_perm_connexion_recurse(comp, time, done=[]):
    fail = []
    ev = []
    ev_sn = []
    done.append(comp)

    for conn in comp.get_connexion(time=time):
        if conn.is_permanent():
            c2 = conn.other_comp(comp)
            if c2 not in done:
                e, s, f = _check_perm_connexion_recurse(c2, time, done)
                fail += f
                ev += e
                ev_sn += s
                done.append(c2)
        else:
            fail.append("%s<->%s" % (conn.comp1.sn, conn.comp2.sn))

    ev.append(comp.event(time, event_type.comp_avail(), EVENT_AT).get())
    ev_sn.append(comp.sn)

    return ev, ev_sn, fail


def remove_component(comp, time=datetime.datetime.now(), notes=None, force=False):
    """End availability of one or more components with a common timestamp.

    If you are adding only one component, this function is equivalent to calling
    :meth:`component.remove`. However, multiple calls to :meth:`component.remove`
    generate a unique timestamp per call. To assign a single timestamp to many
    additions at once, use this function.

    Parameters
    ----------
    comp : list of :obj:`component` objects
        The components to end availability of.
    time : datetime.datetime
        The time at which to end availability.
    notes : string
        Any notes for the timestamp.
    force : bool
        If :obj:`True`, then remove any components that can be removed, while doing
        nothing (except making note of such in the logger) for components whose
        removal would violate database integrity. If :obj:`False`,
        :exc:`DoesNotExist` is raised for any addition that violates database
        integrity.
    """
    _check_user("comp_avail")
    if isinstance(comp, component) or isinstance(comp, str):
        comp_list = [comp]
    else:
        comp_list = comp

    # First check to see that the component already exists at this time; also
    # ensure that it is not connected to anything.
    fail_avail = []
    fail_conn = []
    fail_perm_conn = []
    ev = []
    ev_comp_sn = []
    for comp in comp_list:
        try:
            c = component.get(sn=comp.sn)
            e = c.event(time, event_type.comp_avail(), EVENT_AT).get()

            found_conn = False
            for conn in c.get_connexion(time=time):
                if not conn.is_permanent():
                    fail_conn.append("%s<->%s" % (conn.comp1.sn, conn.comp2.sn))
                    found_conn = True

            perm_ev, perm_ev_sn, perm_fail = _check_perm_connexion_recurse(c, time)
            if len(perm_fail):
                fail_perm_conn += perm_fail
                found_conn = True
            elif len(perm_ev):
                ev += perm_ev
                ev_comp_sn += perm_ev_sn
                found_conn = True

            if not found_conn:
                ev.append(c.event(time, event_type.comp_avail(), EVENT_AT).get())
                ev_comp_sn.append(c.sn)
        except pw.DoesNotExist:
            fail_avail.append(c.sn)
            pass

    _check_fail(
        fail_avail,
        force,
        LayoutIntegrity,
        "The following component%s "
        "%s not available at that time, or you have specified an "
        "end time earlier than %s start time%s"
        % (
            _plural(fail_avail),
            _are(fail_avail),
            "its" if len(fail_avail) == 1 else "their",
            _plural(fail_avail),
        ),
    )
    _check_fail(
        fail_conn,
        force,
        LayoutIntegrity,
        "Cannot remove because the "
        "following component%s %s connected" % (_plural(fail_conn), _are(fail_conn)),
    )
    _check_fail(
        fail_perm_conn,
        force,
        LayoutIntegrity,
        "Cannot remove because "
        "the following component%s %s connected (via permanent "
        "connexions)" % (_plural(fail_perm_conn), _are(fail_perm_conn)),
    )

    t_stamp = timestamp.create(time=time, notes=notes)
    for e in ev:
        e.end = t_stamp
        e.save()
        logger.debug("Removed component by ending event %d." % e.id)
    if len(ev):
        logger.info(
            "Removed %d component%s: %s."
            % (len(ev), _plural(ev), ", ".join(ev_comp_sn))
        )
    else:
        logger.info("Removed no component.")


def set_property(
    comp, type, value, time=datetime.datetime.now(), notes=None, force=False
):
    """Set a property value for one or more components with a common timestamp.

    Passing :obj:`None` for the property value erases that property from the
    component.

    If you altering only one component, this function is equivalent to calling
    :meth:`component.set_property`. However, multiple calls to
    :meth:`component.set_property` generate a unique timestamp per call. To
    assign a single timestamp to many additions at once, use this function.

    Parameters
    ----------
    comp : list of :obj:`component` objects
        The components to assign the property to.
    type : :obj:`property_type`
        The property type.
    value : str
        The property value to assign.
    time : datetime.datetime
        The time at which to end availability.
    notes : string
        Any notes for the timestamp.
    force : bool
        If :obj:`False`, then complain if altering the property does nothing (e.g.,
        because the property value would be unchanged for a certain component);
        otherwise, ignore such situations and merely issue logging information on
        them.

    Raises
    ------
    :exc:ValueError:, if *value* does not conform to the property type's regular
    expression; :exc:PropertyUnchanged: if *force* is :obj:`False`: and a
    component's property value would remain unaltered.
    """
    _check_user("property")
    if isinstance(comp, component) or isinstance(comp, str):
        comp_list = [comp]
    else:
        comp_list = comp
    for comp in comp_list:
        _check_property_type(type, comp.type)
    if type.regex and value != None:
        if not re.match(re.compile(type.regex), value):
            raise ValueError(
                'Value "%s" does not conform to regular '
                "expression %s." % (value, type.regex)
            )

    fail = []
    to_end = []
    to_end_sn = []
    to_set = []
    to_set_sn = []
    for comp in comp_list:
        try:
            # If this property type is already set, then end it---unless the value is
            # exactly the same, in which case don't do anything.
            p = (
                _graph_obj_iter(property, event, time, EVENT_AT, None, True)
                .where((property.comp == comp) & (property.type == type))
                .get()
            )
            if p.value == value:
                fail.append(comp.sn)
            else:
                to_end.append(p)
                to_end_sn.append(comp.sn)
                to_set.append(comp)
                to_set_sn.append(comp.sn)
        except pw.DoesNotExist:
            if not value:
                fail.append(comp.sn)
            to_set.append(comp)
            to_set_sn.append(comp.sn)

    _check_fail(
        fail,
        force,
        PropertyUnchanged,
        "The following component%s "
        "property does not change" % ("'s" if len(fail) == 1 else "s'"),
    )

    # End any events that need to be ended.
    if len(to_end):
        t_stamp = timestamp.create(time=time)
        for p in to_end:
            e = p.event(time, event_type.property(), EVENT_AT).get()
            e.end = t_stamp
            e.save()

    # If no value was passed, then we are done.
    if not value:
        logger.info(
            "Removed property %s from the following %d component%s: %s."
            % (type.name, len(to_end), _plural(to_end), ", ".join(to_end_sn))
        )
        return

    # Start the event with a common timestamp.
    if len(to_set):
        t_stamp = timestamp.create(time=time, notes=notes)
        for comp in to_set:
            o = graph_obj.create()
            p = property.create(id=o, comp=comp, type=type, value=value)
            e = event.create(graph_obj=o, type=event_type.property(), start=t_stamp)
        logger.info(
            "Added property %s=%s to the following component%s: %s."
            % (type.name, value, _plural(to_set), ", ".join(to_set_sn))
        )
    else:
        logger.info("No component property was changed.")


def make_connexion(
    conn, time=datetime.datetime.now(), permanent=False, notes=None, force=False
):
    """Connect one or more component pairs with a common timestamp.

    If you are connecting only one pair, this function is equivalent to calling
    :meth:`connexion.make`. However, multiple calls to :meth:`connexion.make`
    generate a unique timestamp per call. To assign a single timestamp to many
    connexions at once, use this function.

    Examples
    --------
    >>> conn = []
    >>> for i in range(0, 10):
    ...  comp1 = layout.component.get(sn = "LNA%04dB" % (i))
    ...  comp2 = layout.component.get(sn = "CXA%04dB"% (i))
    ...  conn.append(layout.connexion.from_pair(comp1, comp2))
    >>> layout.make_connexion(conn, time = datetime(2013, 10, 11, 23, 15), notes = "Making multiple connexions at once.")

    Parameters
    ----------
    comp : list of :obj:`connexion` objects
        The connexions to make.
    time : datetime.datetime
        The time at which to end availability.
    notes : string
        Any notes for the timestamp.
    force : bool
        If :obj:`True`, then remove any components that can be removed, while doing
        nothing (except making note of such in the logger) for components whose
        removal would violate database integrity. If :obj:`False`,
        :exc:`DoesNotExist` is raised for any addition that violates database
        integrity.
    """
    _check_user("connexion")
    if isinstance(conn, connexion):
        conn = [conn]

    # Check that the connexions do not yet exist.
    fail = []
    to_conn = []
    to_conn_sn = []
    for c in conn:
        if c.is_connected(time):
            fail.append("%s<=>%s" % (c.comp1.sn, c.comp2.sn))
        else:
            to_conn.append(c)
            to_conn_sn.append("%s<=>%s" % (c.comp1.sn, c.comp2.sn))
    if len(fail):
        _check_fail(
            fail,
            force,
            AlreadyExists,
            "Cannot connect because the following connexions already exist",
        )

    t_stamp = timestamp.create(time=time, notes=notes)
    for c in to_conn:
        try:
            # If there is a connexion after this time, replace it with an event
            # starting at this new time.
            e_old = c.event(time, EVENT_AFTER, ORDER_ASC).get()
            e_old._replace(start=t_stamp)
            logger.debug("Added connexion by replacing previous event %d." % e_old.id)
        except pw.DoesNotExist:
            try:
                conn = connexion.from_pair(c.comp1, c.comp2, allow_new=False)
                o = conn.id
            except:
                o = graph_obj.create()
                conn = connexion.create(id=o, comp1=c.comp1, comp2=c.comp2)
            if permanent:
                e_type = event_type.perm_connexion()
            else:
                e_type = event_type.connexion()
            e = event.create(graph_obj=o, type=e_type, start=t_stamp)
            logger.debug("Added connexion with new event %d." % e.id)
    if len(to_conn):
        logger.info(
            "Added %d new connexion%s: %s"
            % (len(to_conn), _plural(to_conn), ", ".join(to_conn_sn))
        )
    else:
        logger.info("Added no new connexions.")


def sever_connexion(conn, time=datetime.datetime.now(), notes=None, force=False):
    """Sever one or more component pairs with a common timestamp.

    If you are severing only one pair, this function is equivalent to calling
    :meth:`connexion.sever`. However, multiple calls to :meth:`connexion.sever`
    generate a unique timestamp per call. To assign a single timestamp to many
    connexion severances at once, use this function.

    Examples
    --------
    >>> conn = []
    >>> for i in range(0, 10):
    ...  comp1 = layout.component.get(sn = "LNA%04dB" % (i))
    ...  comp2 = layout.component.get(sn = "CXA%04dB"% (i))
    ...  conn.append(layout.connexion.from_pair(comp1, comp2))
    >>> layout.sever_connexion(conn, time = datetime(2014, 10, 11, 23, 15), notes = "Severing multiple connexions at once.")

    Parameters
    ----------
    comp : list of :obj:`connexion` objects
        The connexions to sever.
    time : datetime.datetime
        The time at which to end availability.
    notes : string
        Any notes for the timestamp.
    force : bool
        If :obj:`True`, then sever any connexions that can be severed, while doing
        nothing (except making note of such in the logger) for connexions whose
        severence would violate database integrity. If :obj:`False`,
        :exc:`DoesNotExist` is raised for any severence that violates database
        integrity.
    """
    _check_user("connexion")
    if isinstance(conn, connexion):
        conn = [conn]

    # Check that the connexions actually exist.
    fail_conn = []
    fail_perm = []
    ev = []
    ev_conn_sn = []
    for c in conn:
        try:
            ev.append(
                c.event(time=time, type=event_type.connexion(), when=EVENT_AT).get()
            )
            ev_conn_sn.append("%s<=>%s" % (c.comp1.sn, c.comp2.sn))
        except pw.DoesNotExist:
            try:
                c.event(
                    time=time, type=event_type.perm_connexion(), when=EVENT_AT
                ).get()
                fail_perm.append("%s<=>%s" % (c.comp1.sn, c.comp2.sn))
            except pw.DoesNotExist:
                fail_conn.append("%s<=>%s" % (c.comp1.sn, c.comp2.sn))
    _check_fail(
        fail_conn,
        force,
        AlreadyExists,
        "Cannot disconnect because "
        "the following connexion%s %s not exist at that time"
        % (_plural(fail_conn), _does(fail_conn)),
    )
    _check_fail(
        fail_perm, force, LayoutIntegrity, "Cannot disconnect permanent connexions"
    )

    t_stamp = timestamp.create(time=time, notes=notes)
    for e in ev:
        e.end = t_stamp
        e.save()
        logger.debug("Severed connexion by ending event %d." % e.id)
    if len(ev):
        logger.info(
            "Severed %d connexion%s: %s."
            % (len(ev), _plural(ev), ", ".join(ev_conn_sn))
        )
    else:
        logger.info("Severed no connexion.")
