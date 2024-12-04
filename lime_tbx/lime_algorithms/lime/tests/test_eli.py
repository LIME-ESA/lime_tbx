"""Tests for the eli module"""

"""___Built-In Modules___"""
# import here

"""___Third-Party Modules___"""
import unittest
import numpy as np

"""___LIME_TBX Modules___"""
from .. import eli
from lime_tbx.datatypes.datatypes import (
    MoonData,
    SpectralData,
)

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "16/02/2023"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


WLENS = np.array([440, 500, 675, 870, 1020, 1640])
ELREF_DATA = np.array(
    [
        0.030732903677862477,
        0.036489296613340674,
        0.0492340672567728,
        0.05826947690460127,
        0.0642498981188091,
        0.09576452090114816,
    ]
)
ELREF_UNCS = np.array(
    [0.00028234, 0.00033419, 0.00041053, 0.00050497, 0.00045305, 0.00069454]
)
ESI_DATA = np.array(
    [1.771, 1.9155, 1.512, 0.97635, 0.7121999999999999, 0.23554999999999998]
)
MD = MoonData(1, 400000, 1, 40, 30, 50, 50)
SOLID_ANGLE_MOON: float = 6.4177e-05
DIST_EARTH_MOON_KM: int = 384400

ELIS_CHECK_DATA_WEHRLI = np.array(
    [
        1.0268298152854062e-06,
        1.3186330679696958e-06,
        1.4044096994475665e-06,
        1.0733045351503149e-06,
        8.6327810688338977e-07,
        4.2556319749134344e-07,
    ]
)
ELIS_CHECK_DATA = np.array(
    [
        1.0797115270270405e-06,
        1.3501265797219511e-06,
        1.4073660680104437e-06,
        1.0233894777590921e-06,
        8.503963982841415e-07,
        4.1148031586953454e-07,
    ]
)

ELIS_CHECK_UNCS = np.array(
    [
        9.92302678e-09,
        1.23667473e-08,
        1.17368027e-08,
        8.86998430e-09,
        5.99776235e-09,
        2.98459689e-09,
    ]
)

ELREF_CORR = np.zeros((len(ELIS_CHECK_UNCS), len(ELIS_CHECK_UNCS)))
np.fill_diagonal(ELREF_CORR, 1)

ELREF_DS = SpectralData.make_reflectance_ds(WLENS, ELREF_DATA, ELREF_UNCS, ELREF_CORR)


class TestELRef(unittest.TestCase):
    # TODO Add tests with values calculated outside this implementation.

    def test_measurement_func_eli_all_wlens(self):
        elis = eli._measurement_func_eli(
            ELREF_DATA,
            SOLID_ANGLE_MOON,
            ESI_DATA,
            MD.distance_sun_moon,
            DIST_EARTH_MOON_KM,
            MD.distance_observer_moon,
            MD.geom_factor,
        )
        np.testing.assert_array_equal(elis, ELIS_CHECK_DATA_WEHRLI)

    def test_measurement_func_eli_one_wlen(self):
        eli_val = eli._measurement_func_eli(
            ELREF_DATA[0],
            SOLID_ANGLE_MOON,
            ESI_DATA[0],
            MD.distance_sun_moon,
            DIST_EARTH_MOON_KM,
            MD.distance_observer_moon,
            MD.geom_factor,
        )
        self.assertEqual(eli_val, ELIS_CHECK_DATA_WEHRLI[0])

    def test_measurement_func_eli_one_multiple_same(self):
        elis = eli._measurement_func_eli(
            ELREF_DATA,
            SOLID_ANGLE_MOON,
            ESI_DATA,
            MD.distance_sun_moon,
            DIST_EARTH_MOON_KM,
            MD.distance_observer_moon,
            MD.geom_factor,
        )
        eli_ones = np.array(
            [
                eli._measurement_func_eli(
                    ELREF_DATA[i],
                    SOLID_ANGLE_MOON,
                    ESI_DATA[i],
                    MD.distance_sun_moon,
                    DIST_EARTH_MOON_KM,
                    MD.distance_observer_moon,
                    MD.geom_factor,
                )
                for i in range(len(ELREF_DATA))
            ]
        )
        np.testing.assert_array_equal(eli_ones, elis)

    def test_calculate_eli_from_elref(self):
        elis = eli.calculate_eli_from_elref(WLENS, MD, ELREF_DATA, "cimel")
        np.testing.assert_array_equal(elis, ELIS_CHECK_DATA)

    def test_calculate_eli_unc(self):
        elref_spectrum = SpectralData(WLENS, ELREF_DATA, ELREF_UNCS, ELREF_DS)
        u_elis, corr = eli.calculate_eli_from_elref_unc(elref_spectrum, MD, "cimel")
        np.testing.assert_array_almost_equal(u_elis, ELIS_CHECK_UNCS)


if __name__ == "__main__":
    unittest.main()
