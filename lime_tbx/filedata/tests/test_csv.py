"""Tests for csv module"""

"""___Built-In Modules___"""
from datetime import datetime, timezone
import filecmp


"""___Third-Party Modules___"""
import unittest
import numpy as np

"""___LIME_TBX Modules___"""
from ...datatypes.datatypes import (
    CustomPoint,
    SRFChannel,
    SatellitePoint,
    SpectralData,
    SpectralResponseFunction,
    SurfacePoint,
)
from ..csv import (
    export_csv,
    export_csv_comparation,
    export_csv_integrated_irradiance,
    read_datetimes,
    read_refl_coefficients,
)

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "30/08/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"

WLENS = [350, 370, 390, 410, 430, 450]
DATA = [0.0001, 0.0002, 0.0001, 0.0002, 0.0001, 0.0003]
DATA2 = [0.0002, 0.0001, 0.0002, 0.0002, 0.0003, 0.0003]
UNCS = [0, 0, 0, 0.0000000001, 0, 0]
UNCS2 = [0.000002, 0, 0, 0.0000000001, 0, 0]
DT1 = datetime(2018, 2, 27, 2, tzinfo=timezone.utc)
DT2 = datetime(2019, 2, 23, 2, tzinfo=timezone.utc)
SPOINT = SurfacePoint(43, 45, 4500, DT1)
SPOINT2 = SurfacePoint(42, 45, 4500, [DT1, DT2])
SPOINT3 = SurfacePoint(43, 45, 4500, DT2)
CPOINT = CustomPoint(0.9, 450000, 30, 30, 1, 11.2, -11.2)
SATPOINT = SatellitePoint("BIOMASS", DT1)
SATPOINT2 = SatellitePoint("BIOMASS", [DT1, DT2])

SD1 = SpectralData(WLENS, DATA, UNCS, None)
SD2 = SpectralData(WLENS, DATA2, UNCS2, None)

DTS = [
    datetime(2022, 1, 17, 2, 30, tzinfo=timezone.utc),
    datetime(2022, 1, 26, 3, 25, 14, tzinfo=timezone.utc),
    datetime(2022, 1, 11, 3, 21, 4, tzinfo=timezone.utc),
    datetime(2022, 1, 13, 5, 29, 34, tzinfo=timezone.utc),
    datetime(2022, 1, 15, 7, 28, 14, tzinfo=timezone.utc),
    datetime(2022, 1, 16, 7, 23, 4, tzinfo=timezone.utc),
    datetime(2022, 1, 13, 3, 26, 34, tzinfo=timezone.utc),
    datetime(2022, 1, 14, 11, 22, 4, tzinfo=timezone.utc),
]


def get_srf() -> SpectralResponseFunction:
    spectral_response = {WLENS[i]: DATA[i] for i in range(len(DATA))}
    ch = SRFChannel(
        WLENS[0] + ((WLENS[-1] - WLENS[0]) / 2), "Default", spectral_response
    )
    ch2 = SRFChannel(
        WLENS[1] + ((WLENS[-1] - WLENS[1]) / 2), "Secpnd", spectral_response
    )
    ch3 = SRFChannel(WLENS[2] + ((WLENS[-1] - WLENS[2]) / 2), "Drai", spectral_response)
    return SpectralResponseFunction("default", [ch, ch2, ch3])


class TestCSV(unittest.TestCase):
    def test_export_csv_1(self):
        path = "./test_files/csv/export_1.test.csv"
        export_csv(SD1, "Wavelength", "Irradiance", SPOINT, path, "test")
        self.assertTrue(filecmp.cmp(path, "./test_files/csv/export_1.csv"))

    def test_export_csv_2(self):
        path = "./test_files/csv/export_2.test.csv"
        export_csv(SD1, "Wavelength", "Irradiance", CPOINT, path, "test")
        self.assertTrue(filecmp.cmp(path, "./test_files/csv/export_2.csv"))

    def test_export_csv_3(self):
        path = "./test_files/csv/export_3.test.csv"
        export_csv(SD1, "Wavelength", "Irradiance", SATPOINT, path, "test")
        self.assertTrue(filecmp.cmp(path, "./test_files/csv/export_3.csv"))

    def test_export_csv_4(self):
        data = [SD1, SD2]
        path = "./test_files/csv/export_4.test.csv"
        export_csv(data, "Wavelength", "Irradiance", SPOINT2, path, "test")
        self.assertTrue(filecmp.cmp(path, "./test_files/csv/export_4.csv"))

    def test_export_csv_5(self):
        data = [SD1, SD2]
        path = "./test_files/csv/export_5.test.csv"
        export_csv(data, "Wavelength", "Irradiance", SATPOINT2, path, "test")
        self.assertTrue(filecmp.cmp(path, "./test_files/csv/export_5.csv"))

    def test_export_csv_comparation_1(self):
        data = [
            SpectralData([350, 350], [0.02, 0.03], [0, 0.005], None),
            SpectralData([350, 350], [0.03, 0.03], [0, 0], None),
        ]
        path = "./test_files/csv/export_comp_1.test.csv"
        export_csv_comparation(data, "Signal", [SPOINT, SPOINT3], path, "test")
        self.assertTrue(filecmp.cmp(path, "./test_files/csv/export_comp_1.csv"))

    def test_export_csv_integrated_irradiance(self):
        srf = get_srf()
        signals = np.array(
            [np.array(DATA)[0:2] * (i + 1) for i in range(len(srf.channels))]
        )
        uncs = np.array(
            [np.array(UNCS)[0:2] * (i + 1) for i in range(len(srf.channels))]
        )
        data = SpectralData(WLENS, signals, uncs, None)
        path = "./test_files/csv/export_intirr_1.test.csv"
        export_csv_integrated_irradiance(srf, data, path, SPOINT2, "test")
        self.assertTrue(filecmp.cmp(path, "./test_files/csv/export_intirr_1.csv"))

    def test_read_datetimes(self):
        dts = read_datetimes("./test_files/csv/timeseries.csv")
        self.assertEqual(len(dts), len(DTS))
        for i, dt in enumerate(dts):
            self.assertEqual(dt, DTS[i])

    def test_read_refl_coefficients(self):
        rf = read_refl_coefficients("./coeff_data/versions/23_01_01.csv")
        version = "23.01.01"
        data = [
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
        u_data = [
            [
                9.3686983712e-03,
                1.2017105711e-08,
                1.4446094233e-08,
                5.0688276442e-09,
                5.4710503849e-11,
                1.0713699068e-10,
                4.2666117090e-11,
                9.4897416668e-14,
                3.9586194983e-13,
                1.3962709717e-12,
                2.1961497567e-12,
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
                4.4935480469e-13,
                1.0950806502e-13,
                8.1822714222e-13,
                8.9862424048e-13,
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
                6.8473366971e-13,
                1.9820091515e-12,
                2.4629467473e-12,
                3.3832913074e-12,
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
                1.4641205529e-14,
                1.9953129399e-12,
                2.6903374935e-12,
                4.2183521128e-12,
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
                1.4556234568e-13,
                1.8178554397e-12,
                2.1600561063e-12,
                3.6788795025e-12,
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
                9.5620922209e-14,
                2.5526999002e-12,
                3.9907505473e-12,
                5.7168439035e-12,
                3.8894128414e-05,
                3.8719434036e-05,
                1.3463844571e-09,
                5.3081256301e-10,
                4.0658906933e-07,
                1.7771204867e-08,
                3.2858296080e-09,
            ],
        ]
        self.assertEqual(rf.version, version)
        for i, arr in enumerate(rf._ds.coeff.values.T):
            for j, val in enumerate(arr):
                self.assertEqual(val, data[i][j])
        for i, arr in enumerate(rf._ds.u_coeff.values.T):
            for j, val in enumerate(arr):
                self.assertEqual(val, u_data[i][j])


if __name__ == "__main__":
    unittest.main()
