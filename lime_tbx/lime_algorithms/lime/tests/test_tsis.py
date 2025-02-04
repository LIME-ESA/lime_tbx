"""Tests for tsis module"""

"""___Built-In Modules___"""
import os

"""___Third-Party Modules___"""
import unittest
import numpy as np
import numpy.testing as npt

"""___NPL Modules___"""

"""___LIME_TBX Modules"""
from lime_tbx.lime_algorithms.lime.tsis_irradiance import (
    _get_tsis_data,
    tsis_cimel,
    tsis_asd,
    tsis_fwhm,
    _gen_files,
)
from lime_tbx.spectral_integration.spectral_integration import SpectralIntegration

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2022"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


class Test_TSIS(unittest.TestCase):
    def test_tsis_irradiance(self):
        _current_dir = os.path.dirname(os.path.abspath(__file__))
        solar_data = _get_tsis_data()
        solar_x = np.array(list(solar_data.keys()))
        solar_y = np.array(list(map(lambda x: x[0], solar_data.values())))
        u_solar_y = np.array(list(map(lambda x: x[1], solar_data.values())))
        cimel_wavs, cimel_esi, u_cimel_esi = tsis_cimel(
            solar_y, solar_x, u_solar_y, mc_steps=10
        )
        # asd_wavs,asd_esi,u_asd_esi=tsis_asd(solar_y,solar_x,u_solar_y,MCsteps=2)
        dat = np.genfromtxt(
            os.path.join(_current_dir, "../assets/tsis_asd.csv"), delimiter=","
        )
        asd_wavs, asd_esi, u_asd_esi = dat[:, 0], dat[:, 1], dat[:, 2]
        dat = np.genfromtxt(
            os.path.join(_current_dir, "../assets/tsis_fwhm_3_1_gaussian.csv"),
            delimiter=",",
        )
        gauss_wavs, gauss_esi, u_gauss_esi = dat[:, 0], dat[:, 1], dat[:, 2]
        dat = np.genfromtxt(
            os.path.join(_current_dir, "../assets/tsis_fwhm_1_1_triangle.csv"),
            delimiter=",",
        )
        tria_wavs, tria_esi, u_tria_esi = dat[:, 0], dat[:, 1], dat[:, 2]
        si = SpectralIntegration()
        cimel_esi_asd = si.integrate_cimel(asd_esi, asd_wavs, cimel_wavs)
        cimel_esi_gauss = si.integrate_cimel(gauss_esi, gauss_wavs, cimel_wavs)
        cimel_esi_tria = si.integrate_cimel(tria_esi, tria_wavs, cimel_wavs)
        npt.assert_allclose(cimel_esi, cimel_esi_asd, rtol=0.01, atol=0.01)
        npt.assert_allclose(cimel_esi, cimel_esi_gauss, rtol=0.01, atol=0.01)
        npt.assert_allclose(cimel_esi, cimel_esi_tria, rtol=0.01, atol=0.01)


if __name__ == "__main__":
    unittest.main()
