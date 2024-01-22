"""This module contains constant templates for the creation of datasets with obsarray"""

"""___Built-In Modules___"""
# import here

"""___Third-Party Modules___"""
import numpy as np

"""___LIME_TBX Modules___"""
# import here

# define ds variables
TEMPLATE_CIMEL = {
    "coeff": {
        "dtype": np.float32,
        "dim": ["i_coeff", "wavelength"],
        "attrs": {"units": [], "u_components": ["u_coeff"]},
    },
    "u_coeff": {
        "dtype": np.float32,
        "dim": ["i_coeff", "wavelength"],
        "attrs": {"units": "%"},
        "err_corr": [
            {
                "dim": ["i_coeff", "wavelength"],
                "form": "err_corr_matrix",
                "params": ["err_corr_coeff"],
                "units": [],
            }
        ],
    },
    "err_corr_coeff": {
        "dim": ["i_coeff.wavelength", "i_coeff.wavelength"],
        "dtype": np.float32,
    },
}


TEMPLATE_REFL = {
    "reflectance": {
        "dtype": np.float32,
        "dim": ["wavelength"],
        "attrs": {"units": [], "u_components": ["u_reflectance"]},
    },
    "u_reflectance": {
        "dtype": np.float32,
        "dim": ["wavelength"],
        "attrs": {"units": "%"},
        "err_corr": [
            {
                "dim": "wavelength",
                "form": "err_corr_matrix",
                "params": ["err_corr_reflectance_wavelength"],
                "units": [],
            }
        ],
    },
    "err_corr_reflectance_wavelength": {
        "dim": ["wavelength", "wavelength"],
        "dtype": np.float32,
    },
}

TEMPLATE_IRR = {
    "irradiance": {
        "dtype": np.float32,
        "dim": ["wavelength"],
        "attrs": {"units": [], "u_components": ["u_irradiance"]},
    },
    "u_irradiance": {
        "dtype": np.float32,
        "dim": ["wavelength"],
        "attrs": {"units": "%"},
        "err_corr": [
            {
                "dim": "wavelength",
                "form": "err_corr_matrix",
                "params": ["err_corr_irradiance"],
                "units": [],
            }
        ],
    },
    "err_corr_irradiance": {
        "dim": ["wavelength", "wavelength"],
        "dtype": np.float32,
    },
}

TEMPLATE_POL = {
    "polarisation": {
        "dtype": np.float32,
        "dim": ["wavelength"],
        "attrs": {"units": [], "u_components": ["u_polarisation"]},
    },
    "u_polarisation": {
        "dtype": np.float32,
        "dim": ["wavelength"],
        "attrs": {"units": "%"},
        "err_corr": [
            {
                "dim": "wavelength",
                "form": "err_corr_matrix",
                "params": ["err_corr_polarisation"],
                "units": [],
            }
        ],
    },
    "err_corr_polarisation": {
        "dim": ["wavelength", "wavelength"],
        "dtype": np.float32,
    },
}

TEMPLATE_SIGNALS = {
    "signals": {
        "dtype": np.float32,
        "dim": ["channels", "dts"],
        "attrs": {"units": [], "u_components": ["u_signals"]},
    },
    "u_signals": {
        "dtype": np.float32,
        "dim": ["channels", "dts"],
        "attrs": {"units": "%"},
        "err_corr": [
            {
                "dim": "channels",
                "form": "err_corr_matrix",
                "params": ["err_corr_signals_channels"],
                "units": [],
            },
            {
                "dim": "dts",
                "form": "err_corr_matrix",
                "params": ["err_corr_signals_dts"],
                "units": [],
            },
        ],
    },
    "err_corr_signals_channels": {
        "dim": ["channels", "channels"],
        "dtype": np.float32,
    },
    "err_corr_signals_dts": {
        "dim": ["dts", "dts"],
        "dtype": np.float32,
    },
}
