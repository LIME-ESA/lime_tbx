"""Tests for tsis module"""

"""___Built-In Modules___"""
from lime_tbx.lime_algorithms.rolo.tsis_irradiance import (
    _get_tsis_data,
    tsis_cimel,
    tsis_asd,
    tsis_intp,
)
from lime_tbx.spectral_integration.spectral_integration import SpectralIntegration

"""___Third-Party Modules___"""
import unittest
import numpy as np
import numpy.testing as npt

"""___NPL Modules___"""

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2022"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


class Test_TSIS(unittest.TestCase):
    def test_tsis_irradiance(self):
        # TODO what is this
        self.skipTest("This tsis doesn't work")
        solar_data = _get_tsis_data()
        solar_x = np.array(list(solar_data.keys()))
        solar_y = np.array(list(map(lambda x: x[0], solar_data.values())))
        u_solar_y = np.array(list(map(lambda x: x[1], solar_data.values())))
        cimel_wavs, cimel_esi, u_cimel_esi = tsis_cimel(
            solar_y, solar_x, u_solar_y, MCsteps=10
        )
        # asd_wavs,asd_esi,u_asd_esi=tsis_asd(solar_y,solar_x,u_solar_y,MCsteps=2)
        dat = np.genfromtxt("../assets/tsis_asd.csv", delimiter=",")
        asd_wavs, asd_esi, u_asd_esi = dat[:, 0], dat[:, 1], dat[:, 2]
        dat = np.genfromtxt("../assets/tsis_intp.csv", delimiter=",")
        intp_wavs, intp_esi, u_intp_esi = dat[:, 0], dat[:, 1], dat[:, 2]
        si = SpectralIntegration()
        cimel_esi_asd = si.integrate_cimel(asd_esi, asd_wavs)
        cimel_esi_intp = si.integrate_cimel(intp_esi, intp_wavs)
        npt.assert_allclose(cimel_esi, cimel_esi_asd, rtol=0.01, atol=0.01)
        npt.assert_allclose(cimel_esi, cimel_esi_intp, rtol=0.01, atol=0.01)


if __name__ == "__main__":
    unittest.main()
