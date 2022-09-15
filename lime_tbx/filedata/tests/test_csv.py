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
        export_csv(SD1, "Wavelength", "Irradiance", SPOINT, path)
        self.assertTrue(filecmp.cmp(path, "./test_files/csv/export_1.csv"))

    def test_export_csv_2(self):
        path = "./test_files/csv/export_2.test.csv"
        export_csv(SD1, "Wavelength", "Irradiance", CPOINT, path)
        self.assertTrue(filecmp.cmp(path, "./test_files/csv/export_2.csv"))

    def test_export_csv_3(self):
        path = "./test_files/csv/export_3.test.csv"
        export_csv(SD1, "Wavelength", "Irradiance", SATPOINT, path)
        self.assertTrue(filecmp.cmp(path, "./test_files/csv/export_3.csv"))

    def test_export_csv_4(self):
        data = [SD1, SD2]
        path = "./test_files/csv/export_4.test.csv"
        export_csv(data, "Wavelength", "Irradiance", SPOINT2, path)
        self.assertTrue(filecmp.cmp(path, "./test_files/csv/export_4.csv"))

    def test_export_csv_5(self):
        data = [SD1, SD2]
        path = "./test_files/csv/export_5.test.csv"
        export_csv(data, "Wavelength", "Irradiance", SATPOINT2, path)
        self.assertTrue(filecmp.cmp(path, "./test_files/csv/export_5.csv"))

    def test_export_csv_comparation_1(self):
        data = [
            SpectralData([350, 350], [0.02, 0.03], [0, 0.005], None),
            SpectralData([350, 350], [0.03, 0.03], [0, 0], None),
        ]
        path = "./test_files/csv/export_comp_1.test.csv"
        export_csv_comparation(data, "Signal", [SPOINT, SPOINT3], path)
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
        export_csv_integrated_irradiance(srf, data, path, SPOINT2)
        self.assertTrue(filecmp.cmp(path, "./test_files/csv/export_intirr_1.csv"))

    def test_read_datetimes(self):
        dts = read_datetimes("./test_files/csv/timeseries.csv")
        self.assertEqual(len(dts), len(DTS))
        for i, dt in enumerate(dts):
            self.assertEqual(dt, DTS[i])


if __name__ == "__main__":
    unittest.main()
