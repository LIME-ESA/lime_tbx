"""Tests for aolp"""

import unittest
import numpy as np

from ..aolp import AOLP
from lime_tbx.common.datatypes import AOLPCoefficients

_WLENS = [440, 500, 675, 870, 1020, 1640]

_COEFFS = np.array(
    [
        [
            3.37553350e00,
            5.21079393e-02,
            -1.50838049e-03,
            -2.74490746e-06,
            1.68191819e-07,
            -5.92104497e-11,
        ],
        [
            3.33275616e00,
            5.08911311e-02,
            -1.22716312e-03,
            -3.96853068e-06,
            1.30779508e-07,
            1.83423913e-11,
        ],
        [
            2.89868681e00,
            5.46717608e-02,
            -7.85621158e-04,
            -3.59373404e-06,
            7.04687580e-08,
            -1.55044654e-10,
        ],
        [
            1.80322512e00,
            5.50918459e-02,
            -1.18566712e-03,
            -1.92807409e-05,
            1.77604006e-07,
            1.29196990e-09,
        ],
        [
            6.80367016e-03,
            4.93838281e-02,
            -5.38136447e-04,
            -2.60183347e-05,
            1.33190297e-07,
            2.20567801e-09,
        ],
        [
            -3.13621424e-01,
            4.77244168e-02,
            6.51419700e-04,
            -1.37190686e-05,
            -8.90612818e-08,
            6.75202931e-10,
        ],
    ]
)

_UNC_DEF_VAL = 0.000000001

_NO_UNCS = [[_UNC_DEF_VAL for _ in range(6)] for _ in range(6)]

_nounc_size = len(_NO_UNCS) * len(_NO_UNCS[0])

_NO_ERR_CORR = np.zeros((_nounc_size, _nounc_size))
np.fill_diagonal(_NO_ERR_CORR, 1)

POL_COEFFS = AOLPCoefficients(
    _WLENS,
    _COEFFS,
    _NO_UNCS,
    _NO_ERR_CORR,
)


def get_aolp() -> AOLP:
    return AOLP()


class TestDolp(unittest.TestCase):
    # TODO Add the uncertainties comparation when implemented
    # TODO Add comparisons to externally calculated data with the AOLP model.

    def test_get_aolp_cimel_10(self):
        d = get_aolp()
        vals = d.get_aolp(10, POL_COEFFS)
        np.testing.assert_array_equal(vals.wlens, _WLENS)
        np.testing.assert_array_almost_equal(
            vals.data,
            [3.744706, 3.716292, 3.363938, 2.218201, 0.422362, 0.214223],
        )
        # np.testing.assert_array_almost_equal(vals.uncertainties, [0, 0, 0, 0, 0, 0])

    def test_get_aolp_cimel_minus10(self):
        d = get_aolp()
        vals = d.get_aolp(-10, POL_COEFFS)
        np.testing.assert_array_equal(vals.wlens, _WLENS)
        np.testing.assert_array_almost_equal(
            vals.data,
            [2.708049, 2.706403, 2.277721, 1.154668, -0.513719, -0.712963],
        )
        # np.testing.assert_array_almost_equal(vals.uncertainties, [0, 0, 0, 0, 0, 0])

    def test_get_aolp_cimel_30(self):
        d = get_aolp()
        vals = d.get_aolp(30, POL_COEFFS)
        np.testing.assert_array_equal(vals.wlens, _WLENS)
        np.testing.assert_array_almost_equal(
            vals.data,
            [3.641913, 3.75427, 3.788062, 2.043554, 0.462983, 1.278242],
        )
        # np.testing.assert_array_almost_equal(vals.uncertainties, [0, 0, 0, 0, 0, 0])

    def test_get_aolp_cimel_60(self):
        d = get_aolp()
        vals = d.get_aolp(60, POL_COEFFS)
        np.testing.assert_array_equal(vals.wlens, _WLENS)
        np.testing.assert_array_almost_equal(
            vals.data, [2.612664, 2.8204, 3.367222, -0.017922, -1.146137, 1.302439]
        )
        # np.testing.assert_array_almost_equal(vals.uncertainties, [0, 0, 0, 0, 0, 0])

    def test_get_aolp_cimel_90(self):
        d = get_aolp()
        vals = d.get_aolp(90, POL_COEFFS)
        np.testing.assert_array_equal(vals.wlens, _WLENS)
        np.testing.assert_array_almost_equal(
            vals.data,
            [4.531762, 3.768631, 2.543714, 2.383479, 2.888, -2.59943],
        )
        # np.testing.assert_array_almost_equal(vals.uncertainties, [0, 0, 0, 0, 0, 0])

    def test_get_aolp_cimel_minus40(self):
        d = get_aolp()
        vals = d.get_aolp(-40, POL_COEFFS)
        np.testing.assert_array_equal(vals.wlens, _WLENS)
        np.testing.assert_array_almost_equal(
            vals.data,
            [-0.509885, -0.079447, -0.118902, -0.74118, -1.049289, -0.599444],
        )
        # np.testing.assert_array_almost_equal(vals.uncertainties, [0.0025, 0.0025, 0.0024, 0.0023, 0.0026, 0.0025], 3)

    def test_get_aolp_cimel_minus80(self):
        d = get_aolp()
        vals = d.get_aolp(-80, POL_COEFFS)
        np.testing.assert_array_equal(vals.wlens, _WLENS)
        np.testing.assert_array_almost_equal(
            vals.data,
            [-1.958186, -1.263866, -1.268587, 2.72048, 4.16132, 1.201219],
        )
        # np.testing.assert_array_almost_equal(vals.uncertainties, [0.04, 0.04, 0.04, 0.045, 0.035, 0.04], 2)

    def test_get_aolp_cimel_minus90(self):
        d = get_aolp()
        vals = d.get_aolp(-90, POL_COEFFS)
        np.testing.assert_array_equal(vals.wlens, _WLENS)
        np.testing.assert_array_almost_equal(
            vals.data,
            [-0.146328, 0.177725, -0.226493, 5.320361, 5.885027, 0.838565],
        )
        # np.testing.assert_array_almost_equal(vals.uncertainties, [0.065, 0.062, 0.068, 0.065, 0.067, 0.066], 2)


if __name__ == "__main__":
    unittest.main()
