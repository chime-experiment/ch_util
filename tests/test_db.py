"""Read only tests of database functionality.

These are not particularly exhaustive.
"""

import unittest
import datetime

from ch_util import data_index as di
from ch_util import layout


class TestDataIndex(unittest.TestCase):
    def setUp(self):
        try:
            di.connect_database(read_write=False)
        except:
            raise unittest.SkipTest("Skipping test as couldn't connect to db.")

    def test_group_query(self):
        sgnames = [sg.name for sg in di.StorageGroup.select()]

        self.assertIn("cedar_online", sgnames)
        self.assertLess(3, len(sgnames))  # 3 is an arbitrary low num

    def test_acq_type(self):
        atnames = [at.name for at in di.AcqType.select()]

        self.assertIn("corr", atnames)
        self.assertIn("hkp", atnames)


class TestLayout(unittest.TestCase):
    def setUp(self):
        try:
            layout.connect_database(read_write=False)
            layout.set_user("Jrs65")
            self.graph = layout.graph.from_db(datetime.datetime.now())
        except:
            raise unittest.SkipTest("Skipping test as couldn't connect to db.")

    def test_component_query(self):
        reflectors = [c.sn for c in self.graph.component(type="reflector")]
        self.assertIn("cylinder_A", reflectors)

    def test_closest(self):
        ant = self.graph.component(type="antenna")[100]

        comp = self.graph.closest_of_type(ant, "reflector")
        self.assertIsNotNone(comp)
