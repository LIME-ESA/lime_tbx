"""Tests for classname module"""

"""___Built-In Modules___"""
import unittest
from datetime import datetime

"""___Third-Party Modules___"""
import numpy as np

"""___NPL Modules___"""
from ..regular_simulation import RegularSimulation
from ....datatypes.datatypes import SpectralResponseFunction, SurfacePoint
from ....coefficients.access_data import access_data

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "2022/05/19"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"

KERNELS_PATH = "./kernels"
VALL_LAT = 41.6636
VALL_LON = -4.70583
VALL_ALT = 705
JAN_FULL_MOON_00 = datetime(2022, 1, 17, 0, 0, 0)
JAN_FULL_MOON_17 = datetime(2022, 1, 17, 17, 0, 0)
FEB_NEW_MOON_00 = datetime(2022, 2, 2, 0, 0, 0)
DEFAULT_PROP_ERROR = 10  # 10% difference from AEMET's RIMO is allowed
VALL_NAME = "VALLADOLID"
COEFFS = access_data._get_default_irradiance_coefficients()

rs = RegularSimulation()
srf = SpectralResponseFunction("", {i: 1.0 for i in np.arange(380, 2500, 2)})

# Surface point test
def _test_valladolid_no_corr(ts: unittest.TestCase, wavelength, expected, date):
    sp = SurfacePoint(VALL_LAT, VALL_LON, VALL_ALT, date)
    res = rs.get_eli_from_surface(
        SpectralResponseFunction("", {wavelength: 1.0}), sp, COEFFS, KERNELS_PATH
    )
    ts.assertAlmostEqual(res[0], expected, delta=expected * DEFAULT_PROP_ERROR)


class TestRegularSimulation(unittest.TestCase):

    # Surface point tests
    def test_get_eli_Valladolid(self):
        sp = SurfacePoint(VALL_LAT, VALL_LON, VALL_ALT, datetime(2022, 1, 17, 2, 30, 0))
        res = rs.get_eli_from_surface(
            SpectralResponseFunction("", {400: 1.0}), sp, COEFFS, KERNELS_PATH
        )
        self.assertGreater(res[0], 0, "Should be greater than 0")

    def test_eli336_uncorrected_Valladolid_20220117_00(self):
        _test_valladolid_no_corr(self, 336, 9.1239e-07, JAN_FULL_MOON_00)

    def test_eli380_uncorrected_Valladolid_20220117_00(self):
        _test_valladolid_no_corr(self, 380, 1.3348e-06, JAN_FULL_MOON_00)

    def test_eli440_uncorrected_Valladolid_20220117_00(self):
        _test_valladolid_no_corr(self, 440, 2.4528e-06, JAN_FULL_MOON_00)

    def test_eli500_uncorrected_Valladolid_20220117_00(self):
        _test_valladolid_no_corr(self, 500, 2.9900e-06, JAN_FULL_MOON_00)

    def test_eli862_uncorrected_Valladolid_20220117_00(self):
        _test_valladolid_no_corr(self, 862, 2.2911e-06, JAN_FULL_MOON_00)

    def test_eli1011_uncorrected_Valladolid_20220117_00(self):
        _test_valladolid_no_corr(self, 1011, 1.8330e-06, JAN_FULL_MOON_00)

    def test_eli1662_uncorrected_Valladolid_20220117_00(self):
        _test_valladolid_no_corr(self, 1662, 8.4489e-07, JAN_FULL_MOON_00)

    def test_eli338_uncorrected_Valladolid_20220117_17(self):
        _test_valladolid_no_corr(self, 338, 1.2293e-06, JAN_FULL_MOON_17)

    def test_eli385_uncorrected_Valladolid_20220117_17(self):
        _test_valladolid_no_corr(self, 385, 1.5392e-06, JAN_FULL_MOON_17)

    def test_eli481_uncorrected_Valladolid_20220117_17(self):
        _test_valladolid_no_corr(self, 481, 4.0048e-06, JAN_FULL_MOON_17)

    def test_eli540_uncorrected_Valladolid_20220117_17(self):
        _test_valladolid_no_corr(self, 540, 3.8516e-06, JAN_FULL_MOON_17)

    def test_eli879_uncorrected_Valladolid_20220117_17(self):
        _test_valladolid_no_corr(self, 879, 2.7428e-06, JAN_FULL_MOON_17)

    def test_eli1020_uncorrected_Valladolid_20220117_17(self):
        _test_valladolid_no_corr(self, 1020, 2.2381e-06, JAN_FULL_MOON_17)

    def test_eli1654_uncorrected_Valladolid_20220117_17(self):
        _test_valladolid_no_corr(self, 1654, 1.0285e-06, JAN_FULL_MOON_17)

    def test_eli336_uncorrected_Valladolid_20220202_00(self):
        _test_valladolid_no_corr(self, 336, 5.4519e-10, FEB_NEW_MOON_00)

    def test_eli380_uncorrected_Valladolid_20220202_00(self):
        _test_valladolid_no_corr(self, 380, 1.0571e-09, FEB_NEW_MOON_00)

    def test_eli440_uncorrected_Valladolid_20220202_00(self):
        _test_valladolid_no_corr(self, 440, 1.8795e-09, FEB_NEW_MOON_00)

    def test_eli500_uncorrected_Valladolid_20220202_00(self):
        _test_valladolid_no_corr(self, 500, 2.7575e-09, FEB_NEW_MOON_00)

    def test_eli862_uncorrected_Valladolid_20220202_00(self):
        _test_valladolid_no_corr(self, 862, 2.1584e-09, FEB_NEW_MOON_00)

    def test_eli1011_uncorrected_Valladolid_20220202_00(self):
        _test_valladolid_no_corr(self, 1011, 7.5293e-10, FEB_NEW_MOON_00)

    def test_eli1662_uncorrected_Valladolid_20220202_00(self):
        _test_valladolid_no_corr(self, 1662, 7.9724e-10, FEB_NEW_MOON_00)


if __name__ == "__main__":
    unittest.main()
