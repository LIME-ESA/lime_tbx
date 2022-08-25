"""Tests for spectral_integration module"""

"""___Built-In Modules___"""
# import here

"""___Third-Party Modules___"""
import unittest
import numpy as np

"""___LIME_TBX Modules___"""
from ..spectral_integration import ISpectralIntegration, SpectralIntegration
from ...datatypes.datatypes import SRFChannel, SpectralResponseFunction

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "24/08/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"

CH_WLENS = np.array([350, 400, 450, 500])
CH_SRF = np.array([0.2, 0.2, 0.3, 0.3])
CH_ELIS = np.array([0.005, 0.0002, 0.3, 0.0001])


def get_spectral_integrator() -> ISpectralIntegration:
    return SpectralIntegration()


def get_srf() -> SpectralResponseFunction:
    spectral_response = {CH_SRF[i]: CH_WLENS[i] for i in range(len(CH_SRF))}
    ch = SRFChannel((CH_WLENS[-1] - CH_WLENS[0]) / 2, "Default", spectral_response)
    return SpectralResponseFunction("default", [ch])


class TestSpectralIntegration(unittest.TestCase):

    # Function _convolve_srf
    def test__convolve_srf_ok(self):
        si: SpectralIntegration = get_spectral_integrator()
        signal = si._convolve_srf(CH_WLENS, CH_SRF, CH_ELIS)
        ch_signal = np.trapz(CH_SRF * CH_ELIS, CH_WLENS)
        self.assertEqual(signal, ch_signal)

    # Function integrate_elis
    def test_integrate_elis_ok(self):
        si = get_spectral_integrator()
        signals = si.integrate_elis(get_srf(), CH_ELIS)
        self.assertTrue(isinstance(signals, list))

    # Function u_integrate_elis
    def test_u_integrate_elis_ok(self):
        si = get_spectral_integrator()
        uncertainties = si.u_integrate_elis(get_srf(), CH_ELIS)
        self.assertTrue(isinstance(uncertainties, list))


if __name__ == "__main__":
    unittest.main()
