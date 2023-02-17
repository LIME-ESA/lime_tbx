"""This tests compares the extraterrestrial lunar irradiance output with the one calculated with AEMET's RimoAPP.

Cases based on Valladolid coordinates.
Using RimoApp data so the coefficients are the ROLO ones. The RimoApp output contains only some wavelengths,
and the only ones that are in the output that are also in the coefficients are 405 and 544 nm, so those are used.
Before comparison the RimoApp output is divided by the apollo coefficients, so the correct data is compared.

AEMET's RimoApp: https://testbed.aemet.es/rimoapp/
"""

"""___Built-In Modules___"""
from datetime import datetime, timezone
from typing import List, Union
import csv

"""___Third-Party Modules___"""
import unittest
import numpy as np
import xarray as xr

"""___NPL Modules___"""
import obsarray

"""___LIME_TBX Modules___"""
from .. import rolo, eli, elref
from lime_tbx.datatypes.datatypes import (
    MoonData,
    KernelsPath,
    SpectralData,
    ReflectanceCoefficients,
)
from lime_tbx.datatypes.templates import TEMPLATE_CIMEL
from lime_tbx.spice_adapter.spice_adapter import SPICEAdapter


VALL_LAT = 41.6636
VALL_LON = -4.70583
VALL_ALT = 705
KERNELS_PATH = KernelsPath("./kernels", "./kernels")

ROLO_WLENS = [
    405,
    544,
]  # only integer wavelengths that are present in the rimoapp output
ROLO_FULL_WLENS = np.array(
    [
        350.0,
        355.1,
        405.0,
        412.3,
        414.4,
        441.6,
        465.8,
        475.0,
        486.9,
        544.0,
        549.1,
        553.8,
        665.1,
        693.1,
        703.6,
        745.3,
        763.7,
        774.8,
        865.3,
        872.6,
        882.0,
        928.4,
        939.3,
        942.1,
        1059.5,
        1243.2,
        1538.7,
        1633.6,
        1981.5,
        2126.3,
        2250.9,
        2383.6,
    ]
)
ROLO_APOLLO_COEFFS = np.array(
    [
        1.0301,
        1.0970,
        0.9325,
        0.9466,
        1.0225,
        1.0157,
        1.0470,
        1.0084,
        1.0100,
        1.0148,
        0.9843,
        1.0134,
        0.9329,
        0.9849,
        0.9994,
        0.9957,
        1.0059,
        0.9618,
        0.9561,
        0.9796,
        0.9568,
        0.9873,
        1.0575,
        1.0108,
        0.9743,
        1.0386,
        1.0338,
        1.0577,
        1.0650,
        1.0815,
        0.8945,
        0.9689,
    ]
)
_C_COEFF = [0.00034115, -0.0013425, 0.00095906, 0.00066229]
_P_COEFF = [4.06054, 12.8802, -30.5858, 16.7498]
_ROLO_COEFFS = np.array(
    [
        [
            -2.35754,
            -1.72134,
            0.40337,
            -0.21105,
            0.03505,
            0.01043,
            -0.00341,
            *_C_COEFF,
            0.35235,
            -0.03818,
            -0.00006,
            *_P_COEFF,
        ],
        [
            -2.13864,
            -1.60613,
            0.27886,
            -0.16426,
            0.03833,
            0.01189,
            -0.00390,
            *_C_COEFF,
            0.37190,
            -0.10629,
            0.01428,
            *_P_COEFF,
        ],
    ]
).T
_ROLO_UNCS = np.zeros(_ROLO_COEFFS.shape)
_ERR_CORR_SIZE = len(ROLO_WLENS) * len(_ROLO_COEFFS)
_ERR_CORR = np.zeros((_ERR_CORR_SIZE, _ERR_CORR_SIZE))


def read_rimoapp(path: str):
    rimodata = np.loadtxt(path, delimiter="\t", dtype=str)
    indexes = np.where(np.in1d(rimodata[0], np.array(ROLO_WLENS, dtype=str)))[0]
    rolo_indexes = np.where(np.in1d(ROLO_FULL_WLENS, np.array(ROLO_WLENS)))[0]
    data = rimodata[1:]
    cimel_rimodata = {}
    for line in data:
        dt = datetime(*list(map(int, line[:6])), tzinfo=timezone.utc)
        cimel_rimodata[dt] = (
            line[indexes].astype(float) / ROLO_APOLLO_COEFFS[rolo_indexes]
        )
    return cimel_rimodata


def get_coeffs() -> ReflectanceCoefficients:
    dim_sizes = {
        "wavelength": len(ROLO_WLENS),
        "i_coeff": len(_ROLO_COEFFS),
        "i_coeff.wavelength": len(ROLO_WLENS) * len(_ROLO_COEFFS),
    }
    data = _ROLO_COEFFS
    u_data = _ROLO_UNCS
    err_corr_coeff = _ERR_CORR
    # create dataset
    ds_cimel: xr.Dataset = obsarray.create_ds(TEMPLATE_CIMEL, dim_sizes)
    ds_cimel = ds_cimel.assign_coords(wavelength=ROLO_WLENS)
    ds_cimel.coeff.values = data
    ds_cimel.u_coeff.values = u_data
    ds_cimel.err_corr_coeff.values = err_corr_coeff
    rf = ReflectanceCoefficients(ds_cimel)
    return rf


RIMODATA_JAN_FULL_MOON_00 = read_rimoapp("test_files/rimoapp/rimoapp_2022_01_17.csv")
RIMODATA_FEB_2022 = read_rimoapp("test_files/rimoapp/rimoapp_2022_02.csv")


def create_vall_moondata(
    dt: Union[datetime, List[datetime]]
) -> Union[MoonData, List[MoonData]]:
    sp = SPICEAdapter()
    return sp.get_moon_data_from_earth(VALL_LAT, VALL_LON, VALL_ALT, dt, KERNELS_PATH)


REL_DIFF_MAX = 1e-4  # lower number doesnt make sense, RimoApp output is truncated to 5 significant numbers.


class TestIrrApollo(unittest.TestCase):
    def test_Valladolid_20220117(self):
        rimodata = RIMODATA_JAN_FULL_MOON_00
        rl = rolo.ROLO()
        rc = get_coeffs()
        dts = list(rimodata.keys())
        mds = create_vall_moondata(dts)
        for irrs, md in zip(rimodata.values(), mds):
            elrefs = rl.get_elrefs(rc, md)
            elis = rl.get_elis_from_elrefs(elrefs, md)
            np.testing.assert_allclose(elis.data, irrs, REL_DIFF_MAX)

    def test_Valladolid_202202_fullmonth(self):
        rimodata = RIMODATA_FEB_2022
        rl = rolo.ROLO()
        rc = get_coeffs()
        dts = list(rimodata.keys())
        mds = create_vall_moondata(dts)
        for irrs, md in zip(rimodata.values(), mds):
            elrefs = rl.get_elrefs(rc, md)
            elis = rl.get_elis_from_elrefs(elrefs, md)
            np.testing.assert_allclose(elis.data, irrs, REL_DIFF_MAX)


if __name__ == "__main__":
    unittest.main()
