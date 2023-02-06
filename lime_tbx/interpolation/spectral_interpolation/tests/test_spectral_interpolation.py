"""Tests for spectral_interpolation module"""

"""___Built-In Modules___"""
# import here

"""___Third-Party Modules___"""
import unittest

"""___LIME TBX Modules___"""
from ..spectral_interpolation import SpectralInterpolation, ISpectralInterpolation
from ....datatypes.datatypes import KernelsPath, MoonData

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "26/01/2023"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"

KP = KernelsPath("./kernels", "./kernels")
MD1 = MoonData(
    0.9863676197729848,
    399227.54900652857,
    0.1343504656066533,
    -4.658809009228347,
    -3.139429310609046,
    11.317038213996295,
    -11.317038213996295,
)
MD2 = MoonData(
    0.9904106311343145,
    389941.3911970312,
    0.049851687502014026,
    -6.15147366076081,
    -5.052383178500747,
    9.11726084520294,
    -9.11726084520294,
)


def get_interpolator() -> ISpectralInterpolation:
    return SpectralInterpolation()


class TestSpectralInterpolation(unittest.TestCase):
    def test_get_asd_references_ok(self):
        ip = get_interpolator()
        asd_ref = ip.get_best_interp_reference(MD1)
        pol_asd_ref = ip.get_best_polar_interp_reference(MD1)
        self.assertIsNotNone(asd_ref)
        self.assertIsNotNone(pol_asd_ref)
        asd_ref = ip.get_best_interp_reference(MD2)
        pol_asd_ref = ip.get_best_polar_interp_reference(MD2)
        self.assertIsNotNone(asd_ref)
        self.assertIsNotNone(pol_asd_ref)


if __name__ == "__main__":
    unittest.main()
