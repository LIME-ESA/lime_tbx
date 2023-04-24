"""Tests for spectral_interpolation module"""

"""___Built-In Modules___"""
# import here

"""___Third-Party Modules___"""
import unittest
import numpy as np
import comet_maths as cm

"""___LIME_TBX Modules___"""
from ..spectral_interpolation import SpectralInterpolation, ISpectralInterpolation
from ...interp_data import interp_data as idata
from ....datatypes.datatypes import KernelsPath, MoonData
from lime_tbx.interpolation.spectral_interpolation.spectral_interpolation import (
    SpectralInterpolation,
)
from lime_tbx.lime_algorithms.lime.tsis_irradiance import (
    _get_tsis_data,
    tsis_cimel,
    tsis_fwhm,
)


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

CIMEL_WAV = np.array([440, 500, 675, 870, 1020, 1640])
CIMEL_DATA = np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
ASD_WAV = np.array([v for v in range(300, 1000)])
ASD_DATA = np.array([1 for v in range(300, 1000)])
U_CIMEL = np.array([0.1 * v for v in CIMEL_DATA])
U_ASD = np.array([0.1 * v for v in ASD_DATA])


def get_interpolator() -> ISpectralInterpolation:
    return SpectralInterpolation()


class TestSpectralInterpolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        idata.set_interpolation_spectrum_name("ASD")

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

    def test_get_asd_references_same_samemd(self):
        ip = get_interpolator()
        asd_ref = ip.get_best_interp_reference(MD1)
        asd_ref2 = ip.get_best_interp_reference(MD1)
        np.testing.assert_array_equal(asd_ref.wlens, asd_ref2.wlens)
        np.testing.assert_array_equal(asd_ref.data, asd_ref2.data)
        np.testing.assert_array_equal(asd_ref.uncertainties, asd_ref2.uncertainties)
        pol_asd_ref = ip.get_best_polar_interp_reference(MD1)
        pol_asd_ref2 = ip.get_best_polar_interp_reference(MD1)
        np.testing.assert_array_equal(pol_asd_ref.wlens, pol_asd_ref2.wlens)
        np.testing.assert_array_equal(pol_asd_ref.data, pol_asd_ref2.data)
        np.testing.assert_array_equal(
            pol_asd_ref.uncertainties, pol_asd_ref2.uncertainties
        )

    def test_get_interp_ref(self):
        ip = get_interpolator()
        wlens = np.array([i for i in range(400, 801, 50)])
        irefl = ip.get_interpolated_refl(
            CIMEL_WAV, CIMEL_DATA, ASD_WAV, ASD_DATA, wlens
        )
        supposed_irefl = np.interp(wlens, CIMEL_WAV, CIMEL_DATA)
        np.testing.assert_array_almost_equal(supposed_irefl, irefl)

    def test_get_interp_ref_unc(self):
        ip = get_interpolator()
        iunc = ip.get_interpolated_refl_unc(
            CIMEL_WAV, CIMEL_DATA, ASD_WAV, ASD_DATA, CIMEL_WAV, U_CIMEL, U_ASD
        )

    def test_interpolate_refl(self):
        sin = SpectralInterpolation()
        solar_data = _get_tsis_data()
        solar_x = np.array(list(solar_data.keys()))
        solar_y = np.array(list(map(lambda x: x[0], solar_data.values())))
        u_solar_y = np.array(list(map(lambda x: x[1], solar_data.values())))
        wlens = np.arange(350, 2500, 1)
        si = ()
        cimel_wavs, cimel_esi, u_cimel_esi = tsis_cimel(solar_y, solar_x, u_solar_y)

        esi_intp = cm.interpolate_1d(solar_x, solar_y, wlens)
        modified_solar_y = solar_y / np.arange(1, 5, 4 / len(solar_y))
        elrefs_intp, u_elrefs_intp, corr_elrefs_intp = sin.get_interpolated_refl_unc(
            cimel_wavs,
            cimel_esi,
            solar_x,
            modified_solar_y,
            wlens,
            u_cimel_esi,
            u_solar_y,
            "syst",
            "syst",
        )
        np.testing.assert_allclose(elrefs_intp, esi_intp, rtol=0.10, atol=0.10)


if __name__ == "__main__":
    unittest.main()
