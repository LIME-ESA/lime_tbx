"""Tests for the elref module"""

"""___Built-In Modules___"""
# import here

"""___Third-Party Modules___"""
import unittest
import numpy as np
import xarray as xr

"""___NPL Modules"""
import obsarray

"""___LIME_TBX Modules___"""
from .. import elref
from lime_tbx.common.datatypes import (
    ReflectanceCoefficients,
    MoonData,
)
from lime_tbx.common.templates import TEMPLATE_CIMEL

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "16/02/2023"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


WLENS = [440, 500, 675, 870, 1020, 1640]
_COEFFS = np.array(
    [
        [
            -2.263172432,
            -1.953409783,
            0.691585146,
            -0.301894577,
            0.052456211,
            0.008714468,
            -0.004148856,
            -0.000358287,
            0.001216634,
            0.000732067,
            0.001610105,
            -0.092938476,
            2.000625563,
            -0.005710425,
            1.354459689,
            1.314673623,
            9.324088764,
            9.596769204,
        ],
        [
            -2.150477824,
            -1.828162868,
            0.596750118,
            -0.279326293,
            0.050077725,
            0.010694765,
            -0.003817844,
            -0.000414510,
            0.001116700,
            0.000944686,
            0.001779578,
            12.966528574,
            -12.421979227,
            -0.002729971,
            1.354459689,
            1.314673623,
            9.324088764,
            9.596769204,
        ],
        [
            -1.914524626,
            -1.722984816,
            0.562314803,
            -0.276204745,
            0.047094444,
            0.012212050,
            -0.004842774,
            -0.000426178,
            0.001112617,
            0.000936428,
            0.001710380,
            9.886489331,
            -9.752388729,
            -0.005938836,
            1.354459689,
            1.314673623,
            9.324088764,
            9.596769204,
        ],
        [
            -1.816469623,
            -1.590599221,
            0.465802547,
            -0.248147989,
            0.046822590,
            0.018782172,
            -0.007000712,
            -0.000374185,
            0.001153363,
            0.000894613,
            0.001881709,
            10.478132200,
            -10.363729899,
            -0.003423694,
            1.354459689,
            1.314673623,
            9.324088764,
            9.596769204,
        ],
        [
            -1.752793600,
            -1.505016569,
            0.401689482,
            -0.229885748,
            0.052411689,
            0.021768424,
            -0.008638112,
            -0.000449685,
            0.001044300,
            0.000837371,
            0.001817379,
            11.936277597,
            -11.815427729,
            -0.002552853,
            1.354459689,
            1.314673623,
            9.324088764,
            9.596769204,
        ],
        [
            -1.474382296,
            -1.217779712,
            0.189072862,
            -0.168372041,
            0.047554853,
            0.011998950,
            -0.004867793,
            -0.000489541,
            0.000944675,
            0.001093331,
            0.001732148,
            14.326729557,
            -14.410231212,
            0.000003484,
            1.354459689,
            1.314673623,
            9.324088764,
            9.596769204,
        ],
    ]
).T
_UNC_DATA = np.array(
    [
        [
            9.3686983712e-03,
            1.2017105711e-08,
            1.4446094233e-08,
            5.0688276442e-09,
            5.4710503849e-11,
            1.0713699068e-10,
            4.2666117090e-11,
            3.9586194983e-13,
            9.4897416668e-14,
            2.1961497567e-12,
            1.3962709717e-12,
            1.4488250162e-05,
            1.4521860048e-05,
            4.1449084817e-10,
            5.3081256301e-10,
            4.0658906933e-07,
            1.7771204867e-08,
            3.2858296080e-09,
        ],
        [
            9.1701494391e-03,
            7.9979534997e-09,
            9.6866858002e-09,
            3.4138426785e-09,
            2.7425460246e-11,
            5.9872382562e-11,
            2.5426128533e-11,
            1.0950806502e-13,
            4.4935480469e-13,
            8.9862424048e-13,
            8.1822714222e-13,
            1.3827679935e-05,
            1.3298909016e-05,
            2.9113789444e-10,
            5.3081256301e-10,
            4.0658906933e-07,
            1.7771204867e-08,
            3.2858296080e-09,
        ],
        [
            8.2767133073e-03,
            2.4885950951e-08,
            3.0071565504e-08,
            1.0589573703e-08,
            9.1741417556e-11,
            1.8259631887e-10,
            7.4154366859e-11,
            1.9820091515e-12,
            6.8473366971e-13,
            3.3832913074e-12,
            2.4629467473e-12,
            1.9976506903e-05,
            1.9902730325e-05,
            9.0119193403e-10,
            5.3081256301e-10,
            4.0658906933e-07,
            1.7771204867e-08,
            3.2858296080e-09,
        ],
        [
            8.6737894428e-03,
            2.8697757008e-08,
            3.4658326743e-08,
            1.2200170426e-08,
            9.1943111739e-11,
            1.9479558250e-10,
            8.1043308234e-11,
            1.9953129399e-12,
            1.4641205529e-14,
            4.2183521128e-12,
            2.6903374935e-12,
            2.4371662406e-05,
            2.4276361935e-05,
            1.0384188439e-09,
            5.3081256301e-10,
            4.0658906933e-07,
            1.7771204867e-08,
            3.2858296080e-09,
        ],
        [
            7.0855452522e-03,
            3.0140585912e-08,
            3.6316596916e-08,
            1.2769906654e-08,
            1.0508354778e-10,
            2.2293640556e-10,
            9.2973394180e-11,
            1.8178554397e-12,
            1.4556234568e-13,
            3.6788795025e-12,
            2.1600561063e-12,
            3.1860960325e-05,
            3.1715350334e-05,
            1.0683631968e-09,
            5.3081256301e-10,
            4.0658906933e-07,
            1.7771204867e-08,
            3.2858296080e-09,
        ],
        [
            7.0855453006e-03,
            3.7693986894e-08,
            4.5422169867e-08,
            1.5961734816e-08,
            1.2951906160e-10,
            2.7354352617e-10,
            1.1234950024e-10,
            2.5526999002e-12,
            9.5620922209e-14,
            5.7168439035e-12,
            3.9907505473e-12,
            3.8894128414e-05,
            3.8719434036e-05,
            1.3463844571e-09,
            5.3081256301e-10,
            4.0658906933e-07,
            1.7771204867e-08,
            3.2858296080e-09,
        ],
    ]
).T
_ERR_CORR_SIZE = len(WLENS) * len(_COEFFS)
_ERR_CORR = np.zeros((_ERR_CORR_SIZE, _ERR_CORR_SIZE))
np.fill_diagonal(_ERR_CORR, 1)

ELREF_CHECK_DATA = np.array(
    [
        0.030732903677862477,
        0.03648929661334069,
        0.0492340672567728,
        0.05826947690460127,
        0.0642498981188091,
        0.09576452090114816,
    ]
)
CHECK_MD = MoonData(1, 400000, 1, 40, 30, 50, 50)
CHECK_UNCS = np.array(
    [6.2591e-06, 7.4265e-06, 7.7568e-06, 9.4509e-06, 8.1534e-06, 1.0008e-05]
)


def get_coeffs() -> ReflectanceCoefficients:
    dim_sizes = {
        "wavelength": len(WLENS),
        "i_coeff": len(_COEFFS),
        "i_coeff.wavelength": len(WLENS) * len(_COEFFS),
    }
    data = _COEFFS
    u_data = _UNC_DATA * np.sign(_COEFFS)
    err_corr_coeff = _ERR_CORR
    # create dataset
    ds_cimel: xr.Dataset = obsarray.create_ds(TEMPLATE_CIMEL, dim_sizes)
    ds_cimel = ds_cimel.assign_coords(wavelength=WLENS)
    ds_cimel.coeff.values = data
    ds_cimel.u_coeff.values = u_data
    ds_cimel.err_corr_coeff.values = err_corr_coeff
    rf = ReflectanceCoefficients(ds_cimel)
    return rf


class TestELRef(unittest.TestCase):
    # TODO Add tests with values calculated outside this implementation.

    def test_measurement_func_elref_all_wlens(self):
        cf = get_coeffs()
        cfc = cf.coeffs
        elref_val = elref._measurement_func_elref(
            cfc._coeffs,
            CHECK_MD.long_sun_radians,
            CHECK_MD.long_obs,
            CHECK_MD.lat_obs,
            CHECK_MD.mpa_degrees,
        )
        np.testing.assert_array_equal(elref_val, ELREF_CHECK_DATA)

    def test_measurement_func_elref_one_wlen(self):
        cf = get_coeffs()
        cfc = cf.coeffs
        elref_val = elref._measurement_func_elref(
            cfc._coeffs[:, 0:1],
            CHECK_MD.long_sun_radians,
            CHECK_MD.long_obs,
            CHECK_MD.lat_obs,
            CHECK_MD.mpa_degrees,
        )
        self.assertEqual(elref_val, ELREF_CHECK_DATA[0])

    def test_measurement_func_elref_one_multiple_same(self):
        cf = get_coeffs()
        cfc = cf.coeffs
        elref_val_multiple = elref._measurement_func_elref(
            cfc._coeffs,
            CHECK_MD.long_sun_radians,
            CHECK_MD.long_obs,
            CHECK_MD.lat_obs,
            CHECK_MD.mpa_degrees,
        )
        elref_val = np.array(
            [
                elref._measurement_func_elref(
                    cfc._coeffs[:, i : i + 1],
                    CHECK_MD.long_sun_radians,
                    CHECK_MD.long_obs,
                    CHECK_MD.lat_obs,
                    CHECK_MD.mpa_degrees,
                )
                for i in range(len(cfc._coeffs[0]))
            ]
        ).T[0]
        np.testing.assert_array_equal(elref_val, elref_val_multiple)

    def test_calculate_elref(self):
        cf = get_coeffs()
        elrefs = elref.calculate_elref(cf, CHECK_MD)
        np.testing.assert_array_equal(elrefs, ELREF_CHECK_DATA)

    def test_calculate_elref_unc(self):
        cf = get_coeffs()
        unc_elrefs, corr = elref.calculate_elref_unc(cf, CHECK_MD)
        np.testing.assert_array_almost_equal(unc_elrefs, CHECK_UNCS, 4)


if __name__ == "__main__":
    unittest.main()
