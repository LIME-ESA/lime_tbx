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
from lime_tbx.datatypes.datatypes import (
    ReflectanceCoefficients,
    MoonData,
)
from lime_tbx.datatypes.templates import TEMPLATE_CIMEL

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
            0.001216634,
            -0.000358287,
            0.001610105,
            0.000732067,
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
            0.001116700,
            -0.000414510,
            0.001779578,
            0.000944686,
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
            0.001112617,
            -0.000426178,
            0.001710380,
            0.000936428,
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
            0.001153363,
            -0.000374185,
            0.001881709,
            0.000894613,
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
            0.001044300,
            -0.000449685,
            0.001817379,
            0.000837371,
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
            0.000944675,
            -0.000489541,
            0.001732148,
            0.001093331,
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
_UNC_DATA = np.zeros(_COEFFS.shape)
_ERR_CORR_SIZE = len(WLENS) * len(_COEFFS)
_ERR_CORR = np.zeros((_ERR_CORR_SIZE, _ERR_CORR_SIZE))

ELREF_CHECK_DATA = np.array(
    [
        0.030732903677862477,
        0.036489296613340674,
        0.0492340672567728,
        0.05826947690460127,
        0.0642498981188091,
        0.09576452090114816,
    ]
)
CHECK_MD = MoonData(1, 400000, 1, 40, 30, 50, 50)
CHECK_UNCS = np.array(
    [6.9388939039072284e-18, 1.3877787807814457e-17, 1.3877787807814457e-17, 0, 0, 0]
)


def get_coeffs() -> ReflectanceCoefficients:
    dim_sizes = {
        "wavelength": len(WLENS),
        "i_coeff": len(_COEFFS),
        "i_coeff.wavelength": len(WLENS) * len(_COEFFS),
    }
    data = _COEFFS
    u_data = _UNC_DATA
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
            cfc.a_coeffs,
            cfc.b_coeffs,
            cfc.c_coeffs,
            cfc.d_coeffs,
            cfc.p_coeffs,
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
            cfc.a_coeffs.T[0],
            cfc.b_coeffs.T[0],
            cfc.c_coeffs.T[0],
            cfc.d_coeffs.T[0],
            cfc.p_coeffs.T[0],
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
            cfc.a_coeffs,
            cfc.b_coeffs,
            cfc.c_coeffs,
            cfc.d_coeffs,
            cfc.p_coeffs,
            CHECK_MD.long_sun_radians,
            CHECK_MD.long_obs,
            CHECK_MD.lat_obs,
            CHECK_MD.mpa_degrees,
        )
        elref_val = np.array(
            [
                elref._measurement_func_elref(
                    cfc.a_coeffs.T[i],
                    cfc.b_coeffs.T[i],
                    cfc.c_coeffs.T[i],
                    cfc.d_coeffs.T[i],
                    cfc.p_coeffs.T[i],
                    CHECK_MD.long_sun_radians,
                    CHECK_MD.long_obs,
                    CHECK_MD.lat_obs,
                    CHECK_MD.mpa_degrees,
                )
                for i in range(len(cfc.a_coeffs.T))
            ]
        )
        np.testing.assert_array_equal(elref_val, elref_val_multiple)

    def test_calculate_elref(self):
        cf = get_coeffs()
        elrefs = elref.calculate_elref(cf, CHECK_MD)
        np.testing.assert_array_equal(elrefs, ELREF_CHECK_DATA)

    def test_calculate_elref_unc(self):
        cf = get_coeffs()
        unc_elrefs = elref.calculate_elref_unc(cf, CHECK_MD)
        np.testing.assert_array_equal(unc_elrefs, CHECK_UNCS)


if __name__ == "__main__":
    unittest.main()
