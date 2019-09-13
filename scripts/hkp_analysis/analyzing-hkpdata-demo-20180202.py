# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility


# coding: utf-8

# # Prerequisites
#
# For this demo we really just need to import `andata`, but the other two will probably come in handy during your analyses):

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from ch_util.andata import HKPData


# # Loading the data
# Most of the functionality in `HKPData` currently is just loading the data file and converting it to a Pandas `DataFrame`. You do this by first opening the file with `from_acq_h5` (narrowing it to just the metrics of interest will spead-up your loading time substantially), and then `select`ing the metric by name:

# In[15]:


f = HKPData.from_acq_h5(
    "/Users/davor/projects/ch_prometheus/archiving/test_data/hkp_prom_20180122.h5",
    metrics=["ext_sensor_value"],
)

m = f.select("ext_sensor_value")
m.head()


# You can't assume a metric name doesn't appear in multiple jobs. All of them get merged into a single dataset, as you can see from the "job" label:

# In[18]:


m.job.unique()


# This also means that you have to be careful about plotting samples from a metric, as multiple jobs probably mean that they are measuring different things. So one of the things you'll always want to do is filter the metric by job of interest with `query`:

# In[5]:


flas = m.query('job=="fla"')
flas.head()


# At this point, we could further filter by the location of interest. E.g., the `hut` label would select FLA temperatures for an entire hut. To view just a quadrant of a bulkhead, which is each allocated a separate Enviromux-1W unit, we'll filter by `instance`:

# In[19]:


f7 = flas.query('instance=="fla-enviromux7"')
f7.head()


# Since we want to plot a line for each FLA's time series, we first group the data by the `device` label:

# In[8]:


fs = f7.groupby("device")


# The following bit is annoying, but a consequence of having the `device` "category" (in the Pandas sense) still include values which have been filtered out, and have groups of size zero that would cause an error to plot, we'll first get the set of device IDs that are actually connected to `fla-enviromux7` (there is almost certainly a more Pandas-ic way to do it, let us know if you know it):

# In[9]:


fs_devices = [k for k, v in fs.groups.items() if len(v) > 0]
fs_devices


# Now we just do the usual Matplotlib plotting. The time indices are available in the `index` property (not really a column in the data, from Pandas's viewpoint), while the value is in the `value` column:

# In[21]:


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
for d in fs_devices:
    z = fs.get_group(d)
    ax.plot(z.index, z.value / 10, label=d)
fig.legend()
