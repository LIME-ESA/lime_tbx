"""Tests for spectral_integration module"""

"""___Built-In Modules___"""
# import here

"""___Third-Party Modules___"""
import unittest
import numpy as np

"""___LIME_TBX Modules___"""
from ..spectral_integration import ISpectralIntegration, SpectralIntegration
from ...datatypes.datatypes import SRFChannel, SpectralData, SpectralResponseFunction
from lime_tbx.business.lime_algorithms.lime.tsis_irradiance import _get_tsis_data

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "24/08/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"

CH_WLENS = np.array([350, 400, 450, 500])
CH_SRF = np.array([0.2, 0.2, 0.3, 0.3])
CH_ELIS = np.array([0.005, 0.0002, 0.3, 0.0001])
CH_CORR = np.zeros((4, 4))
np.fill_diagonal(CH_CORR, 1)

ELIS_LIME = SpectralData(
    CH_WLENS,
    CH_ELIS,
    CH_ELIS * 0.01,
    SpectralData.make_irradiance_ds(CH_WLENS, CH_ELIS, CH_ELIS * 0.01, CH_CORR),
)


def get_spectral_integrator() -> ISpectralIntegration:
    return SpectralIntegration()


def get_srf() -> SpectralResponseFunction:
    spectral_response = {CH_WLENS[i]: CH_SRF[i] for i in range(len(CH_SRF))}
    ch = SRFChannel((CH_WLENS[-1] - CH_WLENS[0]) / 2, "Default", spectral_response)
    return SpectralResponseFunction("default", [ch])


class TestSpectralIntegration(unittest.TestCase):
    # Function _convolve_srf
    def test_convolve_srf_ok(self):
        si: SpectralIntegration = get_spectral_integrator()
        signal = si._convolve_srf(CH_WLENS, CH_SRF, CH_ELIS)
        ch_signal = np.trapz(CH_SRF * CH_ELIS, CH_WLENS) / np.trapz(CH_SRF, CH_WLENS)
        self.assertEqual(signal, ch_signal)

    # Function integrate_elis
    def test_integrate_elis_ok(self):
        si = get_spectral_integrator()
        signals = si.integrate_elis(get_srf(), ELIS_LIME)
        self.assertEqual(signals, [0.12073999999999999])

    # Function u_integrate_elis
    def test_u_integrate_elis_ok(self):
        si = get_spectral_integrator()
        uncertainties = si.u_integrate_elis(get_srf(), ELIS_LIME)
        self.assertIsInstance(uncertainties, np.ndarray)
        self.assertAlmostEqual(uncertainties[0], 0.00118, 3)

    def test_integrate_cimel_ok(self):
        solar_data = _get_tsis_data()
        solar_x = np.array(list(solar_data.keys()))
        solar_y = np.array(list(map(lambda x: x[0], solar_data.values())))
        cimel_wavs = [440, 500, 675, 870, 1020, 1640]
        si = get_spectral_integrator()
        esi_cimel = si.integrate_cimel(solar_y, solar_x, cimel_wavs)
        np.testing.assert_allclose(
            esi_cimel, [1.862, 1.960, 1.515, 0.9309, 0.7016, 0.2278], rtol=0.01
        )
        zeros_cimel = si.integrate_cimel(
            np.zeros(3000), np.arange(0, 3000, 1), cimel_wavs
        )
        np.testing.assert_allclose(zeros_cimel, np.zeros_like(esi_cimel), atol=0.01)
        ones_cimel = si.integrate_cimel(
            np.ones(3000), np.arange(0, 3000, 1), cimel_wavs
        )
        np.testing.assert_allclose(ones_cimel, np.ones_like(ones_cimel), rtol=0.01)


if __name__ == "__main__":
    unittest.main()
