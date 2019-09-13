# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
from ch_util import layout
import peewee as pw

# DO NOT RUN THESE TESTS. THEY MESS WITH THE LAYOUT DATABASE. -KM
if False:
    layout.set_user("Adam Hincks")

    # Add a bunch of components wholesale
    ctype = layout.component_type.get(name="LNA")
    lna = []
    for i in range(5):
        lna.append(layout.component(sn="a%02d" % i, type=ctype))
    layout.add_component(lna, time=datetime.now(), force=True)

    ctype = layout.component_type.get(name="FLA")
    fla = []
    for i in range(5):
        fla.append(layout.component(sn="b%02d" % i, type=ctype))
    layout.add_component(fla, time=datetime.now(), notes="FLAs!!!!!!", force=True)

    ctype = layout.component_type.get(name="60m coax")
    c = []
    for i in range(5):
        c.append(layout.component(sn="c%02d" % i, type=ctype))
    layout.add_component(c, time=datetime.now(), force=True)

    # Add a bunch of connexions, wholesale
    conn1 = []
    conn2 = []
    for i in range(5):
        conn1.append(layout.connexion.from_pair("a%02d" % i, "b%02d" % i))
        conn2.append(layout.connexion.from_pair("b%02d" % i, "c%02d" % i))
    layout.make_connexion(conn1, permanent=True, force=True)
    layout.make_connexion(conn2, force=True)

    # Break a connexion.
    layout.connexion.from_pair("b02", "c02").sever()

    # Remove a component.
    layout.component.get(sn="c02").remove()
