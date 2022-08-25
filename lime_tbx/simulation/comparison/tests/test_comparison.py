"""Tests for comparison module"""

"""___Built-In Modules___"""
# import here

"""___Third-Party Modules___"""
import unittest

"""___LIME_TBX Modules___"""
from .. import comparison

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "25/08/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"

KERNELS_PATH = "./kernels"
EOCFI_PATH = "./eocfi_data"


class TestComparison(unittest.TestCase):
    # Function to_llh
    def test_to_llh_ok(self):
        lat, lon, h = comparison.to_llh(3196669.145, 3196669.145, 4490530.3894655)
        self.assertAlmostEqual(lat, 45)
        self.assertAlmostEqual(lon, 45)
        self.assertAlmostEqual(h, 4500, delta=6)


if __name__ == "__main__":
    unittest.main()
