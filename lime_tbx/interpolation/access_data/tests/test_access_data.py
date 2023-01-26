"""Tests for classname module"""

"""___Built-In Modules___"""
# import here

"""___Third-Party Modules___"""
import unittest
import numpy as np

"""___LIME TBX Modules___"""
from ..access_data import get_asd_data

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "26/01/2023"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


class TestAccessData(unittest.TestCase):
    def test_get_asd_data_ok(self):
        asd_data = get_asd_data(19)
        self.assertIsNotNone(asd_data)

    def test_get_asd_data_not_mpa_absolute(self):
        asd_data_neg = get_asd_data(-80)
        asd_data_pos = get_asd_data(80)
        self.assertTrue(np.not_equal(asd_data_neg.data, asd_data_pos.data).any())

    def test_get_asd_data_consistent(self):
        asd_data0 = get_asd_data(7.2)
        asd_data1 = get_asd_data(7.2)
        self.assertIsNotNone(asd_data0)
        np.testing.assert_array_equal(asd_data0.data, asd_data1.data)


if __name__ == "__main__":
    unittest.main()
