"""Tests for dolp module"""

"""___Built-In Modules___"""
# import here

"""___Third-Party Modules___"""
import unittest
import numpy as np

"""___NPL Modules___"""
from ..dolp import DOLP, IDOLP
from lime_tbx.datatypes.datatypes import PolarizationCoefficients

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "01/02/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"

_WLENS = [440, 500, 675, 870, 1020, 1640]

_POS_COEFFS = [
    (0.003008799098, 0.000177889155, 0.000002581092, 0.000000012553),
    (0.002782607290, 0.000161111675, 0.000002331213, 0.000000011175),
    (0.002467126521, 0.000140139814, 0.000002021823, 0.000000009468),
    (0.002536989960, 0.000150448307, 0.000002233876, 0.000000010661),
    (0.002481149030, 0.000149814043, 0.000002238987, 0.000000010764),
    (0.002135380897, 0.000126059235, 0.000001888331, 0.000000009098),
]

_NEG_COEFFS = [
    (-0.003328093061, 0.000221328429, -0.000003441781, 0.000000018163),
    (-0.002881735316, 0.000186855017, -0.000002860010, 0.000000014778),
    (-0.002659373268, 0.000170314209, -0.000002652223, 0.000000013710),
    (-0.002521475080, 0.000157719602, -0.000002452656, 0.000000012597),
    (-0.002546369943, 0.000158157867, -0.000002469036, 0.000000012675),
    (-0.002726077195, 0.000171190004, -0.000002850707, 0.000000015473),
]

_NO_UNCS = [(0, 0, 0, 0) for _ in range(6)]

POL_COEFFS = PolarizationCoefficients(
    _WLENS,
    _POS_COEFFS,
    _NO_UNCS,
    _NEG_COEFFS,
    _NO_UNCS,
)


def get_dolp() -> IDOLP:
    return DOLP()


class TestDolp(unittest.TestCase):
    # TODO Add the uncertainties comparation when implemented

    def test_get_polarized_cimel_10(self):
        d = get_dolp()
        vals = d.get_polarized(10, POL_COEFFS)
        np.testing.assert_array_equal(vals.wlens, _WLENS)
        np.testing.assert_array_almost_equal(
            vals.data, [-0.014408, -0.012844, -0.012077, -0.011769, -0.01199, -0.012838]
        )
        # np.testing.assert_array_almost_equal(vals.uncertainties, [0, 0, 0, 0, 0, 0])

    def test_get_polarized_cimel_minus10(self):
        d = get_dolp()
        vals = d.get_polarized(-10, POL_COEFFS)
        np.testing.assert_array_equal(vals.wlens, _WLENS)
        np.testing.assert_array_almost_equal(
            vals.data,
            [-0.014755, -0.013934, -0.012584, -0.012452, -0.011961, -0.010545],
        )
        # np.testing.assert_array_almost_equal(vals.uncertainties, [0, 0, 0, 0, 0, 0])

    def test_get_polarized_cimel_30(self):
        d = get_dolp()
        vals = d.get_polarized(30, POL_COEFFS)
        np.testing.assert_array_equal(vals.wlens, _WLENS)
        np.testing.assert_array_almost_equal(
            vals.data, [0.021137, 0.016467, 0.012997, 0.010285, 0.009554, 0.007853]
        )
        # np.testing.assert_array_almost_equal(vals.uncertainties, [0, 0, 0, 0, 0, 0])

    def test_get_polarized_cimel_60(self):
        d = get_dolp()
        vals = d.get_polarized(60, POL_COEFFS)
        np.testing.assert_array_equal(vals.wlens, _WLENS)
        np.testing.assert_array_almost_equal(
            vals.data, [0.089065, 0.073535, 0.05837, 0.049985, 0.047542, 0.037497]
        )
        # np.testing.assert_array_almost_equal(vals.uncertainties, [0, 0, 0, 0, 0, 0])

    def test_get_polarized_cimel_90(self):
        d = get_dolp()
        vals = d.get_polarized(90, POL_COEFFS)
        np.testing.assert_array_equal(vals.wlens, _WLENS)
        np.testing.assert_array_almost_equal(
            vals.data, [0.175848, 0.138807, 0.106244, 0.089099, 0.083585, 0.07831]
        )
        # np.testing.assert_array_almost_equal(vals.uncertainties, [0, 0, 0, 0, 0, 0])

    def test_get_polarized_cimel_minus40(self):
        d = get_dolp()
        vals = d.get_polarized(-40, POL_COEFFS)
        np.testing.assert_array_equal(vals.wlens, _WLENS)
        np.testing.assert_array_almost_equal(
            vals.data, [0.031216, 0.025885, 0.02038, 0.023562, 0.024717, 0.018717]
        )
        # np.testing.assert_array_almost_equal(vals.uncertainties, [0, 0, 0, 0, 0, 0])

    def test_get_polarized_cimel_minus80(self):
        d = get_dolp()
        vals = d.get_polarized(-80, POL_COEFFS)
        np.testing.assert_array_equal(vals.wlens, _WLENS)
        np.testing.assert_array_almost_equal(
            vals.data, [0.090438, 0.072653, 0.052161, 0.05284, 0.05485, 0.041777]
        )
        # np.testing.assert_array_almost_equal(vals.uncertainties, [0, 0, 0, 0, 0, 0])

    def test_get_polarized_cimel_minus90(self):
        d = get_dolp()
        vals = d.get_polarized(-90, POL_COEFFS)
        np.testing.assert_array_equal(vals.wlens, _WLENS)
        np.testing.assert_array_almost_equal(
            vals.data, [0.112096, 0.088307, 0.060378, 0.061275, 0.064195, 0.049222]
        )
        # np.testing.assert_array_almost_equal(vals.uncertainties, [0, 0, 0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
