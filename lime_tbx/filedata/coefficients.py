import numpy as np
import xarray as xr
import obsarray

from lime_tbx.datatypes.datatypes import (
    LimeCoefficients,
    ReflectanceCoefficients,
    PolarizationCoefficients,
    LimeException,
)
from lime_tbx.datatypes.templates_digital_effects_table import TEMPLATE_CIMEL

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
    ds = xr.open_dataset(path)
    file_version = ds.file_version
    creation_date = ds.creation_date
    release_date = ds.release_date
    software_version = ds.software_version
    data_origin = ds.data_origin
    data_origin_release_date = ds.data_origin_release_date
    # define dim_size_dict to specify size of arrays
    dim_sizes = {
        "wavelength": len(ds.wavelength),
        "i_coeff": len(ds.i_coeff),
        "i_coeff.wavelength": len(ds.wavelength) * len(ds.i_coeff),
    }
    wlens = [440, 500, 675, 870, 1020, 1640]
    version_name = release_date
    data = np.array(ds.coeff.values)
    u_data = np.array(ds.u_coeff.values)
    err_corr_coeff = np.array(ds.err_corr_coeff.values)
    # create dataset
    ds_cimel: xr.Dataset = obsarray.create_ds(TEMPLATE_CIMEL, dim_sizes)
    ds_cimel = ds_cimel.assign_coords(wavelength=wlens)
    ds_cimel.coeff.values = data
    ds_cimel.u_coeff.values = u_data
    ds_cimel.err_corr_coeff.values = err_corr_coeff

    rf = ReflectanceCoefficients(ds_cimel)

    p_pos_data = np.array(ds.dolp_coeff_pos.T).astype(float)
    p_pos_u_data = np.array(ds.u_dolp_coeff_pos.T).astype(float)
    p_pos_err_corr_data = np.array(ds.err_corr_dolp_coeff_pos).astype(float)
    p_neg_data = np.array(ds.dolp_coeff_neg.T).astype(float)
    p_neg_u_data = np.array(ds.u_dolp_coeff_neg.T).astype(float)
    p_neg_err_corr_data = np.array(ds.err_corr_dolp_coeff_neg).astype(float)
    pol = PolarizationCoefficients(
        wlens,
        p_pos_data,
        p_pos_u_data,
        p_pos_err_corr_data,
        p_neg_data,
        p_neg_u_data,
        p_neg_err_corr_data,
    )
    return LimeCoefficients(rf, pol, version_name)
