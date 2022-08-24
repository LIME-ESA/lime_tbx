"""Tests for spectral_integration module"""

"""___Built-In Modules___"""
# import here

"""___Third-Party Modules___"""
import unittest

"""___LIME_TBX Modules___"""
from ..spectral_integration import ISpectralIntegration, SpectralIntegration

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "24/08/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


def get_spectral_integrator() -> ISpectralIntegration:
    return SpectralIntegration()


class TestSpectralIntegration(unittest.TestCase):
    def test_function1(self):
        pass


if __name__ == "__main__":
    unittest.main()
