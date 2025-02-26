"""Tests for dolp module"""

"""___Built-In Modules___"""
# import here

"""___Third-Party Modules___"""
import unittest
import numpy as np

"""___NPL Modules___"""
from ..dolp import DOLP, IDOLP
from lime_tbx.common.datatypes import PolarisationCoefficients

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

_UNC_DEF_VAL = 0.000000001

_NO_UNCS = [(_UNC_DEF_VAL, _UNC_DEF_VAL, _UNC_DEF_VAL, _UNC_DEF_VAL) for _ in range(6)]

_nounc_size = len(_NO_UNCS) * len(_NO_UNCS[0])

_NO_ERR_CORR = np.zeros((_nounc_size, _nounc_size))
np.fill_diagonal(_NO_ERR_CORR, 1)

POL_COEFFS = PolarisationCoefficients(
    _WLENS,
    _POS_COEFFS,
    _NO_UNCS,
    _NO_ERR_CORR,
    _NEG_COEFFS,
    _NO_UNCS,
    _NO_ERR_CORR,
)


def get_dolp() -> IDOLP:
    return DOLP()


class TestDolp(unittest.TestCase):
    # TODO Add the uncertainties comparation when implemented
    # TODO Add comparisons to externally calculated data with the DOLP model.

    def test_get_polarized_cimel_10(self):
        d = get_dolp()
        vals = d.get_polarized(10, POL_COEFFS)
        np.testing.assert_array_equal(vals.wlens, _WLENS)
        np.testing.assert_array_almost_equal(
            vals.data,
            [0.050584, 0.04638, 0.040802, 0.042755, 0.04214, 0.035939],
        )
        # np.testing.assert_array_almost_equal(vals.uncertainties, [0, 0, 0, 0, 0, 0])

    def test_get_polarized_cimel_minus10(self):
        d = get_dolp()
        vals = d.get_polarized(-10, POL_COEFFS)
        np.testing.assert_array_equal(vals.wlens, _WLENS)
        np.testing.assert_array_almost_equal(
            vals.data,
            [0.059037, 0.050511, 0.046414, 0.043565, 0.043875, 0.047385],
        )
        # np.testing.assert_array_almost_equal(vals.uncertainties, [0, 0, 0, 0, 0, 0])

    def test_get_polarized_cimel_30(self):
        d = get_dolp()
        vals = d.get_polarized(30, POL_COEFFS)
        np.testing.assert_array_equal(vals.wlens, _WLENS)
        np.testing.assert_array_almost_equal(
            vals.data,
            [0.330222, 0.300473, 0.262398, 0.280463, 0.278439, 0.235869],
        )
        # np.testing.assert_array_almost_equal(vals.uncertainties, [0, 0, 0, 0, 0, 0])

    def test_get_polarized_cimel_60(self):
        d = get_dolp()
        vals = d.get_polarized(60, POL_COEFFS)
        np.testing.assert_array_equal(vals.wlens, _WLENS)
        np.testing.assert_array_almost_equal(
            vals.data, [1.541132, 1.395328, 1.21195, 1.314517, 1.311322, 1.107726]
        )
        # np.testing.assert_array_almost_equal(vals.uncertainties, [0, 0, 0, 0, 0, 0])

    def test_get_polarized_cimel_90(self):
        d = get_dolp()
        vals = d.get_polarized(90, POL_COEFFS)
        np.testing.assert_array_equal(vals.wlens, _WLENS)
        np.testing.assert_array_almost_equal(
            vals.data,
            [4.416912, 3.988085, 3.452278, 3.774924, 3.775245, 3.186777],
        )
        # np.testing.assert_array_almost_equal(vals.uncertainties, [0, 0, 0, 0, 0, 0])

    def test_get_polarized_cimel_minus40(self):
        d = get_dolp()
        vals = d.get_polarized(-40, POL_COEFFS)
        np.testing.assert_array_equal(vals.wlens, _WLENS)
        np.testing.assert_array_almost_equal(
            vals.data,
            [0.75402, 0.63511, 0.583718, 0.542429, 0.545374, 0.605003],
        )
        # np.testing.assert_array_almost_equal(vals.uncertainties, [0.0025, 0.0025, 0.0024, 0.0023, 0.0026, 0.0025], 3)

    def test_get_polarized_cimel_minus80(self):
        d = get_dolp()
        vals = d.get_polarized(-80, POL_COEFFS)
        np.testing.assert_array_equal(vals.wlens, _WLENS)
        np.testing.assert_array_almost_equal(
            vals.data,
            [4.188898, 3.496043, 3.222261, 2.982856, 2.999234, 3.407038],
        )
        # np.testing.assert_array_almost_equal(vals.uncertainties, [0.04, 0.04, 0.04, 0.045, 0.035, 0.04], 2)

    def test_get_polarized_cimel_minus90(self):
        d = get_dolp()
        vals = d.get_polarized(-90, POL_COEFFS)
        np.testing.assert_array_equal(vals.wlens, _WLENS)
        np.testing.assert_array_almost_equal(
            vals.data,
            [5.793021, 4.827414, 4.451872, 4.118937, 4.141786, 4.725335],
        )
        # np.testing.assert_array_almost_equal(vals.uncertainties, [0.065, 0.062, 0.068, 0.065, 0.067, 0.066], 2)


if __name__ == "__main__":
    unittest.main()
