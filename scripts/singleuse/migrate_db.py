# === Start Python 2/3 compatibility
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614
# === End Python 2/3 compatibility

import datetime
import logging
logging.basicConfig(level = logging.INFO)
import layout
import MySQLdb
import peewee as pw

def conn_list(lay_id, db):
  ret = []
  cur = db.cursor()
  cur.execute("SELECT comp_sn1, comp_sn2 FROM layout_conn WHERE " \
              "layout_id = " + str(lay_id) + ";")
  for c in cur.fetchall():
    sn1 = c[0]
    sn2 = c[1]
    if sn1[0:3] == "POL":
      sn1 = "PL0" + sn1[3:]
    if sn2[0:3] == "POL":
      sn2 = "PL0" + sn2[3:]
    ret.append([sn1, sn2])
    ret[-1].sort()
  return ret

def prop_list(lay_id, db):
  ret = []
  cur = db.cursor()
  cur.execute("SELECT comp_sn, name, value FROM layout_prop JOIN prop_type ON "
              "layout_prop.prop_type_id = prop_type.id WHERE " \
              "layout_id = " + str(lay_id) + ";")
  for c in cur.fetchall():
    ret.append(c)
  return ret

db = MySQLdb.connect("localhost", "***REMOVED***", "***REMOVED***", "***REMOVED***")
cur_lay = db.cursor()
cur = db.cursor()

last_user = ""
def set_user(id):
  global last_user

  # A hack: we want to transfer the user information over.
  id = int(id)
  if id != last_user:
    layout.set_user(id)
    print("Changed to user %d." % id)
    last_user = id

# Add components!
if True:
  cur.execute("SELECT * FROM comp ORDER BY parent_sn, sn;")
  for row in cur.fetchall():
    set_user(row[4])
    try:
      layout.component.get(sn = row[0])
    except pw.DoesNotExist:
      t = layout.component_type.get(id = row[2])
      if row[3]:
        r = layout.component_type_rev.get(id = row[3])
      else:
        r = None
      layout.component(sn = row[0], type = t, type_rev = r).add(time = row[5])
      if row[1]:
        layout.connexion.from_pair(row[0], row[1]) \
              .make(time = row[5], permanent = True)

# Add permanent properties!
if True:
  cur.execute("SELECT p.*, t.name FROM layout_prop p "
              "JOIN prop_type t ON p.prop_type_id = t.id WHERE layout_id = 0;")
  for row in cur.fetchall():
    set_user(row[3])
    comp = layout.component.get(sn = row[2])
    e = comp.event(type = layout.event_type.comp_avail()).get()
    comp.set_property(layout.property_type.get(name = row[7]), row[5], 
                      time = e.start.time, notes = row[6])

# Add documents!
if True:
  repo = layout.external_repo.get(name = "hyper")
  cur.execute("SELECT * FROM comp_testdata WHERE repo_path LIKE " \
              "'http%';")
  for row in cur.fetchall():
    set_user(row[2])
    layout.component.get(sn = row[1]).add_doc(repo, row[4][30:], time = row[3])

# Add history!
if True:
  # There were a couple of items improperly put in the testdata table.
  cur.execute("SELECT * from comp_testdata WHERE id IN (509, 510);")
  for row in cur.fetchall():
    set_user(row[2])
    layout.component.get(sn = row[1]).add_history(row[4], time = row[3])
 
  # Now the actual history.
  cur.execute("SELECT * from comp_history;")
  for row in cur.fetchall():
    set_user(row[2])
    h = row[4]
    if row[5]:
      h += " " + row[5]
    layout.component.get(sn = row[1]).add_history(row[4], time = row[3])

# Add layouts!
if True:
  last_conn = None
  last_prop = None
  last_lay_id = None
  last_time = None
  cur_lay.execute("SELECT l.id, l.user_id, change_start, change_end, l.name, " \
                  "l.notes FROM "
                  "layout l JOIN layout_timestamp t ON t.layout_id = l.id " \
                  "ORDER BY t.change_end;")
  for lay_row in cur_lay.fetchall():
    if lay_row[0] == 32:
      # This layout was "revised" by layout 33; skip over it.
      continue
    set_user(lay_row[1])
    curr_conn = conn_list(lay_row[0], db)
    curr_prop = prop_list(lay_row[0], db)
    make_conn = []
    sever_conn = []
    start_prop = []
    stop_prop = []

    if last_time:
      layout.add_layout_tag(last_name, start_time = last_time, 
                            end_time = lay_row[2], notes = last_notes)
    last_time = lay_row[3]
    last_name = lay_row[4]
    last_notes = lay_row[5]

    if last_conn:
      for l in last_conn:
        try:
          i = curr_conn.index(l)
        except:
          sever_conn.append(l)
      for l in curr_conn:
        try:
          i = last_conn.index(l)
        except:
          make_conn.append(l)
  
      for p in last_prop:
        try:
          i = curr_prop.index(p)
        except:
          found = False
          for p2 in curr_prop:
            if p[0] == p2[0] and p[1] == p2[1]:
              found = True
          if not found:
            stop_prop.append(p)
      for p in curr_prop:
        try:
          i = last_prop.index(p)
        except:
          start_prop.append(p)
    else:
      make_conn = curr_conn
      start_prop = curr_prop
  
    conn_make_conn = []
    for c in make_conn:
      conn_make_conn.append(layout.connexion.from_pair(c[0], c[1]))
    conn_sever_conn = []
    for c in sever_conn:
      conn_sever_conn.append(layout.connexion.from_pair(c[0], c[1]))

    print("Layout %d\n---------" % (lay_row[0]))
    layout.make_connexion(conn_make_conn, time = lay_row[3])
    layout.sever_connexion(conn_sever_conn, time = lay_row[2])
    for p in start_prop:
      print(p)
      t = layout.property_type.get(name = p[1])
      layout.component.get(sn = p[0]).set_property(t, p[2], time = lay_row[3])
    for p in stop_prop:
      print(p)
      t = layout.property_type.get(name = p[1])
      layout.component.get(sn = p[0]).set_property(t, None, time = lay_row[2])
    print()
    print()
  
    last_conn = curr_conn
    last_prop = curr_prop
    last_lay_id = lay_row[0]

  layout.add_layout_tag(lay_row[4], start_time = lay_row[3], notes = lay_row[5])
