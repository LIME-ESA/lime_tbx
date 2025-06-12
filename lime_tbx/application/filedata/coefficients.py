import warnings

import numpy as np
import xarray as xr
import obsarray
from packaging import version

from lime_tbx.common.datatypes import (
    LimeCoefficients,
    ReflectanceCoefficients,
    PolarisationCoefficients,
    AOLPCoefficients,
)
from lime_tbx.common.templates import TEMPLATE_CIMEL
from lime_tbx.common import constants

TEMPLATE_COEFFS = {
    "coeff": {
        "dtype": np.float32,
        "dim": ["i_coeff", "wavelength"],
        "attributes": {
            "standard_name": "LIME model coefficients",
            "units": [],
            "u_components": ["u_coeff"],
        },
    },
    "u_coeff": {
        "dtype": np.float32,
        "dim": ["i_coeff", "wavelength"],
        "attributes": {
            "units": "%",
            "err_corr": [
                {
                    "dim": ["i_coeff", "wavelength"],
                    "form": "err_corr_matrix",
                    "params": ["err_corr_coeff"],
                    "units": [],
                }
            ],
        },
    },
    "err_corr_coeff": {
        "dim": ["i_coeff.wavelength", "i_coeff.wavelength"],
        "dtype": np.float32,
    },
    "dolp_coeff_pos": {
        "dtype": np.float32,
        "dim": ["j_coeff", "wavelength"],
        "attributes": {
            "standard_name": "Polynomial coefficients for degree of linear polarisation (DOLP) for positivepolarisation",
            "units": [],
            "u_components": ["u_dolp_coeff_pos"],
        },
    },
    "u_dolp_coeff_pos": {
        "dtype": np.float32,
        "dim": ["j_coeff", "wavelength"],
        "attributes": {
            "units": "%",
            "err_corr": [
                {
                    "dim": ["j_coeff", "wavelength"],
                    "form": "err_corr_matrix",
                    "params": ["err_corr_dolp_coeff_pos"],
                    "units": [],
                }
            ],
        },
    },
    "err_corr_dolp_coeff_pos": {
        "dim": ["j_coeff.wavelength", "j_coeff.wavelength"],
        "dtype": np.float32,
    },
    "dolp_coeff_neg": {
        "dtype": np.float32,
        "dim": ["j_coeff", "wavelength"],
        "attributes": {
            "standard_name": "Polynomial coefficients for degree of linear polarisation (DOLP) for negative polarisation",
            "units": [],
            "u_components": ["u_dolp_coeff_neg"],
        },
    },
    "u_dolp_coeff_neg": {
        "dtype": np.float32,
        "dim": ["j_coeff", "wavelength"],
        "attributes": {
            "units": "%",
            "err_corr": [
                {
                    "dim": ["j_coeff", "wavelength"],
                    "form": "err_corr_matrix",
                    "params": ["err_corr_dolp_coeff_neg"],
                    "units": [],
                }
            ],
        },
    },
    "err_corr_dolp_coeff_neg": {
        "dim": ["j_coeff.wavelength", "j_coeff.wavelength"],
        "dtype": np.float32,
    },
}


def read_coeff_nc(path: str) -> LimeCoefficients:
    warnings.filterwarnings(
        "ignore",
        "Duplicate dimension names present: dimensions {",
        UserWarning,
        "xarray",
    )
    ds = xr.open_dataset(path)
    if "tbx_version_required" in ds.attrs:
        tbx_minv = ds.tbx_version_required
        tbx_v = constants.VERSION_NAME
        if version.parse(tbx_v) < version.parse(tbx_minv):
            # Coefficients won't work for a version prior to tbx_minv
            ds.close()
            raise Exception(
                f"Coefficient file {path} has a minimum LIME TBX version specified. "
                f"Required version: >={tbx_minv}. Current TBX version: {tbx_v}."
            )
    file_version = ds.file_version
    # creation_date = ds.creation_date
    release_date = ds.release_date
    # software_version = ds.software_version
    # data_origin = ds.data_origin
    # data_origin_release_date = ds.data_origin_release_date
    wlens = ds.wavelength.data
    # define dim_size_dict to specify size of arrays
    dim_sizes = {
        "wavelength": len(ds.wavelength),
        "i_coeff": len(ds.i_coeff),
        "i_coeff.wavelength": len(ds.wavelength) * len(ds.i_coeff),
    }
    version_name = f"{release_date}_v{file_version}"
    data = np.array(ds["coeff"].values)
    u_data = np.array(ds["u_coeff"].values)
    err_corr_coeff = np.array(ds["err_corr_coeff"].values)
    # create dataset
    ds_cimel: xr.Dataset = obsarray.create_ds(TEMPLATE_CIMEL, dim_sizes)
    ds_cimel = ds_cimel.assign_coords(wavelength=wlens)
    ds_cimel.coeff.values = data
    ds_cimel.u_coeff.values = u_data
    ds_cimel.err_corr_coeff.values = err_corr_coeff

    rf = ReflectanceCoefficients(ds_cimel)

    p_pos_data = np.array(ds.dolp_coeff_pos.T).astype(float)
    p_pos_u_data = np.array(ds.u_dolp_coeff_pos.T).astype(float)
    p_pos_err_corr_data = np.nan_to_num(
        np.array(ds.err_corr_dolp_coeff_pos).astype(float)
    )
    np.fill_diagonal(p_pos_err_corr_data, 1)
    p_neg_data = np.array(ds.dolp_coeff_neg.T).astype(float)
    p_neg_u_data = np.array(ds.u_dolp_coeff_neg.T).astype(float)
    p_neg_err_corr_data = np.nan_to_num(
        np.array(ds.err_corr_dolp_coeff_neg).astype(float)
    )
    np.fill_diagonal(p_neg_err_corr_data, 1)
    pol = PolarisationCoefficients(
        wlens,
        p_pos_data,
        p_pos_u_data,
        p_pos_err_corr_data,
        p_neg_data,
        p_neg_u_data,
        p_neg_err_corr_data,
    )
    aolp_coeff = np.array([[np.nan for _ in wlens] for _ in range(6)])
    unc_aolp = np.array([[np.nan for _ in wlens] for _ in range(6)])
    err_corr_aolp = np.ones((len(unc_aolp), len(unc_aolp)))
    if "aolp_coeff" in ds.variables:
        aolp_coeff = np.array(ds.aolp_coeff.T).astype(float)
        unc_aolp = np.array(ds.u_aolp_coeff.T).astype(float)
        err_corr_aolp = np.array(ds.err_corr_aolp_coeff).astype(float)
    aolp = AOLPCoefficients(wlens, aolp_coeff, unc_aolp, err_corr_aolp)
    ds.close()
    return LimeCoefficients(rf, pol, aolp, version_name)
