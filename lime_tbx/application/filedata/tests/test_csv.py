"""Tests for csv module"""

"""___Built-In Modules___"""
from datetime import datetime, timezone
import filecmp
from typing import Tuple, List


"""___Third-Party Modules___"""
import unittest
import numpy as np

"""___LIME_TBX Modules___"""
from lime_tbx.common.datatypes import (
    CustomPoint,
    SRFChannel,
    SatellitePoint,
    SpectralData,
    SpectralResponseFunction,
    SurfacePoint,
    ComparisonData,
    MoonData,
)
from lime_tbx.common.constants import CompFields
from ..csv import (
    export_csv_simulation,
    export_csv_comparison,
    export_csv_comparison_bywlen,
    export_csv_integrated_irradiance,
    export_csv_srf,
    read_datetimes,
)
from lime_tbx.presentation.gui.settings import SettingsManager

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
MD1 = MoonData(1, 400000, 1.1, 45.10, 44.1, 15.2, -15.2)
MD2 = MoonData(1.000001, 400100, 1.15, 45.60, 42.1, 12.2, -12.2)
MD3 = MoonData(0.98128, 387282, 0.9912, 23.99, -55.2, 30.123, 30.123)
MD4 = MoonData(0.9, 450000, 1, 30, 30, 11.2, -11.2)
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

DTS2 = [
    datetime(2022, 1, 1, 1, 1, 1, 1234, tzinfo=timezone.utc),
    datetime(2022, 1, 1, 2, 1, 1, 1234, tzinfo=timezone.utc),
    datetime(2022, 1, 1, 3, 1, 1, 1234, tzinfo=timezone.utc),
    datetime(2022, 1, 1, 4, 1, 1, 1234, tzinfo=timezone.utc),
]

CIMEL_DATA = SpectralData([314, 420], [0.01, 0.01], [0.0000001, 0], None)


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
        export_csv_simulation(
            SD1,
            "Wavelength",
            "Irradiance",
            SPOINT,
            path,
            "test",
            True,
            "ASD",
            False,
            CIMEL_DATA,
            MD3,
        )
        self.assertTrue(filecmp.cmp(path, "./test_files/csv/export_1.csv"))

    def test_export_csv_2(self):
        path = "./test_files/csv/export_2.test.csv"
        export_csv_simulation(
            SD1,
            "Wavelength",
            "Irradiance",
            CPOINT,
            path,
            "test",
            True,
            "ASD",
            False,
            CIMEL_DATA,
            MD4,
        )
        self.assertTrue(filecmp.cmp(path, "./test_files/csv/export_2.csv"))

    def test_export_csv_3(self):
        path = "./test_files/csv/export_3.test.csv"
        export_csv_simulation(
            SD1,
            "Wavelength",
            "Irradiance",
            SATPOINT,
            path,
            "test",
            False,
            "ASD",
            False,
            CIMEL_DATA,
            MD2,
        )
        self.assertTrue(filecmp.cmp(path, "./test_files/csv/export_3.csv"))

    def test_export_csv_4(self):
        data = [SD1, SD2]
        path = "./test_files/csv/export_4.test.csv"
        export_csv_simulation(
            data,
            "Wavelength",
            "Irradiance",
            SPOINT2,
            path,
            "test",
            [True, False],
            "ASD",
            False,
            CIMEL_DATA,
            None,
        )
        self.assertTrue(filecmp.cmp(path, "./test_files/csv/export_4.csv"))

    def test_export_csv_5(self):
        data = [SD1, SD2]
        path = "./test_files/csv/export_5.test.csv"
        export_csv_simulation(
            data,
            "Wavelength",
            "Irradiance",
            SATPOINT2,
            path,
            "test",
            [True, True],
            "ASD",
            False,
            CIMEL_DATA,
            None,
        )
        self.assertTrue(filecmp.cmp(path, "./test_files/csv/export_5.csv"))

    def test_export_csv_comparison_1(self):
        data = [
            SpectralData(
                [MD1.mpa_degrees, MD2.mpa_degrees], [0.02, 0.03], [0, 0.005], None
            ),
            SpectralData(
                [MD1.mpa_degrees, MD2.mpa_degrees], [0.03, 0.03], [0, 0], None
            ),
        ]
        path = "./test_files/csv/export_comp_1.test.csv"
        comp_data = ComparisonData(
            data[0],
            data[1],
            data[1],
            0.0,
            0.1,
            0.2,
            2,
            [DT1, DT2],
            [SPOINT, SPOINT3],
            [True, False],
            data[1],
            0.1,
            [MD1, MD2],
        )
        xlabel = "MPA"
        export_csv_comparison(
            comp_data,
            xlabel,
            ["Observed Signal", "Simulated Signal"],
            path,
            "test",
            "ASD",
            False,
            CompFields.DIFF_REL,
        )
        self.assertTrue(filecmp.cmp(path, "./test_files/csv/export_comp_1.csv"))

    def test_export_csv_comparison_wlen(self):
        data = [
            SpectralData([DT1, DT2], [0.02, 0.03], [0, 0.005], None),
            SpectralData([DT1, DT2], [0.03, 0.03], [0, 0], None),
        ]
        path = "./test_files/csv/export_comp_wlen.test.csv"
        comp_data = ComparisonData(
            data[0],
            data[1],
            data[1],
            0.0,
            0.1,
            0.2,
            2,
            [DT1, DT2],
            [SPOINT, SPOINT3],
            [True, False],
            data[1],
            0.1,
            [MD1, MD2],
        )
        data2 = [
            SpectralData([DT1, DT2], [0.05, 0.03], [0, 0.005], None),
            SpectralData([DT1, DT2], [0.08, 0.03], [0, 0], None),
        ]
        comp_data2 = ComparisonData(
            data2[0],
            data2[1],
            data2[1],
            0.0,
            0.1,
            0.2,
            2,
            [DT1, DT2],
            [SPOINT, SPOINT3],
            [True, False],
            data[1],
            0.1,
            [MD1, MD2],
        )
        export_csv_comparison_bywlen(
            [comp_data, comp_data2],
            [350, 450],
            "Wavelength",
            ["Observed Signal", "Simulated Signal"],
            path,
            "test",
            "ASD",
            False,
            CompFields.DIFF_REL,
        )
        self.assertTrue(filecmp.cmp(path, "./test_files/csv/export_comp_wlen.csv"))

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
        export_csv_integrated_irradiance(
            srf,
            data,
            path,
            SPOINT2,
            "test",
            [True, True],
            "ASD",
            False,
            30,
        )
        self.assertTrue(filecmp.cmp(path, "./test_files/csv/export_intirr_1.csv"))

    def test_export_csv_srf(self):
        srf = SettingsManager().get_default_srf()
        data = SpectralData(srf.get_wavelengths(), srf.get_values(), None, None)
        path = "./test_files/csv/export_srf_1.test.csv"
        export_csv_srf(
            data,
            srf.get_channels_names(),
            "Wavelengths (nm)",
            "Intensity (Fractions of unity)",
            path,
        )
        self.assertTrue(filecmp.cmp(path, "./test_files/csv/export_srf_1.csv"))

    def test_read_datetimes(self):
        dts = read_datetimes("./test_files/csv/timeseries.csv")
        self.assertEqual(len(dts), len(DTS))
        for i, dt in enumerate(dts):
            self.assertEqual(dt, DTS[i])

    def test_read_datetimes_complex(self):
        dts = read_datetimes("./test_files/csv/complex_times.csv")
        self.assertEqual(len(dts), len(DTS2))
        for i, dt in enumerate(dts):
            self.assertEqual(dt, DTS2[i])


if __name__ == "__main__":
    unittest.main()
