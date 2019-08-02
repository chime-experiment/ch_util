#!/usr/bin/env python

# === Start Python 2/3 compatibility
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614
# === End Python 2/3 compatibility

import datetime
import re
import sys

CMD_DONE        = "DONE"
CMD_DONE_QUIT   = "DONE-QUIT"
CMD_FREEFORM    = "FREEFORM"
CMD_OVERRIDE    = "OVERRIDE"
CMD_CANCEL      = "CANCEL"

# A global variable.
overridden = False

# For echoing stdout to a file.
class tee_stdout(object):
  def __init__(self, stdout, fp):
    self.stdout = stdout
    self.fp = fp

  def readline(self):
    ret = self.stdout.readline()
    self.fp.write(ret)
    self.fp.flush()
    return ret

  def write(self, txt):
    self.stdout.write(txt)
    self.stdout.flush()
    self.fp.write(txt)
    self.fp.flush()

def parse_cmd(x):
  if not x:
    return False
  if not re.match(r"\$CMD\$.+", x):
    return False
  else:
    return x[5:]

def test_cmd_done(x):
  return bool(parse_cmd(x) == "DONE")

def test_adc_head(x):
  return bool(re.match(r"C\.", x[-2:]))

def test_adc_chan(x):
  return bool(re.match(r"\*[0-9]", x))

def test_sma(x):
  return bool(re.match(r"CXS[0-9][0-9][0-9][0-9]", x))

def test_60m_coax(x):
  return bool(re.match(r"CXA[0-9][0-9][0-9][0-9][A-Z]", x))

def test_fla(x):
  return bool(re.match(r"FLA[0-9][0-9][0-9][0-9][A-Z]", x))

def test_rft_head(x):
  return bool(re.match(r"RFT---[A-Z]", x))

def test_can_head(x):
  return bool(re.match(r"CAN---[A-Z]", x))

def test_row(x):
  return bool(re.match(r"---[A-Z]---", x))

def test_col(x):
  return bool(re.match(r"----[0-9][0-9]-", x))

def get_barcode(msg, test = None, fmt = "", override = True):
  global overridden
  overridden = False

  for i in range(0, 3):
    x = input("    Scan %s barcode: " % msg)
    if parse_cmd(x):
      if parse_cmd(x) == CMD_OVERRIDE:
        return input("   Override in effect. Enter any barcode: ")
      else:
        return x
    elif test:
      if test(x):
        return x
      else:
        x1 = x
        print("    Was expecting format \"%s\"!" % fmt)
        if override:
          x = input("    Same barcode to over-ride, or correct: ")
          if x == x1:
            x = input("    Confirm over-ride by entering once more: ")
            if x == x1:
              print("    Over-ride successful.")
              overridden = True
              return x
            elif test(x):
              return x
            else:
              print("    Bad over-ride and barcode not of format " \
                       "\"%s\". Try again." % fmt)
          elif test(x):
            return x
          else:
            print("    Bad over-ride and barcode not of format " \
                     "\"%s\". Try again." % fmt)
    else:
      return x
  print("    Giving up and cancelling current scan.")
  return "$CMD$" + CMD_CANCEL

def get_adc_chan():
  print("  ADC channel -- two scans required.")
  base = get_barcode("ADC channel head", test_adc_head, "XXXX-XXXX-XXXX-C.")
  if parse_cmd(base):
    return False
  if not test_adc_head(base):   # In case user has overridden.
    return base
  num = get_barcode("channel number", test_adc_chan, "*X", override = False)
  if parse_cmd(base):
    return False
  return base[:-1] + num[1]

def get_sma():
  print("  SMA coax cable -- one scan required.")
  return get_barcode("SMA", test_sma, "CXS0000")

def get_60m_coax():
  print("  60m coax cable -- one scan required.")
  return get_barcode("60m coax", test_60m_coax, "CXA0000A")

def get_fla():
  print("  Filter/amplifier -- one scan required.")
  return get_barcode("FLA", test_fla, "FLA0000A")

def get_rft():
  print("  RFT thru -- three scans required.")
  base = get_barcode("RFT thru head", test_rft_head, "RFT---A")
  if parse_cmd(base):
    return False
  if not test_rft_head(base):
    return base
  row = get_barcode("row", test_row, "---R---", override = False)
  if parse_cmd(row):
    return False
  col = get_barcode("column", test_col, "----CC-", override = False)
  if parse_cmd(col):
    return False
  return base[0:3] + row[3] + col[4:6] + base[6:7]

def get_can():
  print("  C-Can thru -- three scans required.")
  base = get_barcode("C-Can thru head", test_can_head, "CAN---A")
  if parse_cmd(base):
    return False
  if not test_can_head(base):
    return base
  row = get_barcode("row", test_row, "---R---", override = False)
  if parse_cmd(row):
    return False
  col = get_barcode("column", test_col, "----CC-", override = False)
  if parse_cmd(col):
    return False
  return base[0:3] + row[3] + col[4:6] + base[6:7]

def commit_chain(fp, cmd, chain, name):
  if not parse_cmd(cmd) == CMD_CANCEL:
    fp.write("# Chain type: %s. Scanned: %s.\n" % (name,\
             datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    for x in chain:
      fp.write("%s\n" % x)
    fp.write("\n")
    fp.flush()
    print("  Chain written to disc.")
  else:
    print("  Aborted. Nothing written to disc.")
  print("-" * 80)

def get_freeform_chain(fp):
  chain = []
  print("Starting free-form chain. Enter last component thrice to signify " \
        "the end, or")
  print("enter the DONE command to end.")
  print("-" * 80)
  while 1:
    x = get_barcode("")
    if parse_cmd(x):
      if parse_cmd(x) == CMD_DONE or parse_cmd(x) == CMD_DONE_QUIT:
        commit_chain(fp, x, chain, "free form")
        break
      else:
        print("  Cancelling.")
        return
    if len(chain):
      if x == chain[-1]:
        print("  Component entered twice. Modify, or enter it again to " \
                "finish the chain.")
        x1 = x
        x = get_barcode("")
        if x == x1:
          x = input("  Chain complete. Enter anything to commit, or " \
                          "CANCEL: ")
          commit_chain(fp, x, chain, "free form")
          break
        else:
          chain.append(x)
      else:
        chain.append(x)
    else:
      chain.append(x)
    print("  Chain so far: ", chain)
    print()

def get_chain(fp, get_list, name):
  chain = []
  print("Starting chain of type: %s." % name)
  print("-" * 80)
  for l in get_list:
    while 1:
      x = l()
      if parse_cmd(x) or not x:
        if parse_cmd(x) == CMD_DONE or parse_cmd(x) == CMD_DONE_QUIT:
          print("  Preset chains cannot be fininshed early. Finish or cancel.")
          continue
        else:
          print("  Cancelling chain.")
          return False
      chain.append(x)
      print("  Chain so far: ", chain)
      print()
      if overridden == False:
        break
  x = input("  Chain complete. Enter anything to commit, or CANCEL: ")
  commit_chain(fp, x, chain, name)

if len(sys.argv) != 2:
  print("Usage: %s <OUTPUT FILE>" % sys.argv[0])
  exit()

# Predefined chains.
chain_adc_to_rft    = [get_adc_chan, get_sma, get_fla, get_rft]
chain_adc_sma = [get_adc_chan, get_sma]
chain_rft_in = [get_rft, get_fla, get_sma]
chain_rft_out = [get_rft, get_sma]
chain_can_in = [get_can, get_sma]
chain_can_out = [get_can, get_60m_coax]

fp = open(sys.argv[1], "a")
fp_log = open(sys.argv[1] + ".log", "a")

#sys.stdout = tee_stdout(sys.stdout, fp_log)
sys.stdin = tee_stdout(sys.stdin, fp_log)

fp.write("USER %s\n" % input("Please enter/scan your name: "))
fp.flush()

while 1:
  x = input("Enter a command: ")
  if parse_cmd(x):
    if parse_cmd(x) == "CHAIN-ADC-RFT":
      get_chain(fp, chain_adc_to_rft, "ADC to RFT")
    if parse_cmd(x) == "CHAIN-ADC-SMA":
      get_chain(fp, chain_adc_sma, "ADC SMA's")
    if parse_cmd(x) == "CHAIN-RFT-IN":
      get_chain(fp, chain_rft_in, "RFT bulkhead inside")
    if parse_cmd(x) == "CHAIN-RFT-OUT":
      get_chain(fp, chain_rft_out, "RFT bulkhead outside")
    if parse_cmd(x) == "CHAIN-CAN-IN":
      get_chain(fp, chain_can_in, "C-Can bulkhead inside")
    if parse_cmd(x) == "CHAIN-CAN-OUT":
      get_chain(fp, chain_can_out, "C-Can bulkhead outside")
    if parse_cmd(x) == "FREEFORM":
      get_freeform_chain(fp)
  else:
    x1 = x
    print("That command is not recognised.")
    x = input("Enter the same thing for a free-form command: ")
    if x == x1:
      get_freeform_chain(fp)
    
