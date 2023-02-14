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
            {"dim": "wavelength", "form": "random", "params": [], "units": []}
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
        "attrs": {"units": [], "u_components": ["u_coeff"]},
    },
    "u_ran_reflectance": {
        "dtype": np.float32,
        "dim": ["wavelength"],
        "attrs": {"units": "%"},
        "err_corr": [
            {"dim": "wavelength", "form": "random", "params": [], "units": []}
        ],
    },
    "u_sys_reflectance": {
        "dtype": np.float32,
        "dim": ["wavelength"],
        "attrs": {"units": "%"},
        "err_corr": [
            {"dim": "wavelength", "form": "systematic", "params": [], "units": []}
        ],
    },
}

TEMPLATE_IRR = {
    "irradiance": {
        "dtype": np.float32,
        "dim": ["wavelength"],
        "attrs": {"units": [], "u_components": ["u_coeff"]},
    },
    "u_ran_irradiance": {
        "dtype": np.float32,
        "dim": ["wavelength"],
        "attrs": {"units": "%"},
        "err_corr": [
            {"dim": "wavelength", "form": "random", "params": [], "units": []}
        ],
    },
    "u_sys_irradiance": {
        "dtype": np.float32,
        "dim": ["wavelength"],
        "attrs": {"units": "%"},
        "err_corr": [
            {"dim": "wavelength", "form": "systematic", "params": [], "units": []}
        ],
    },
}

TEMPLATE_POL = {
    "polarization": {
        "dtype": np.float32,
        "dim": ["wavelength"],
        "attrs": {"units": [], "u_components": ["u_coeff"]},
    },
    "u_ran_polarization": {
        "dtype": np.float32,
        "dim": ["wavelength"],
        "attrs": {"units": "%"},
        "err_corr": [
            {"dim": "wavelength", "form": "random", "params": [], "units": []}
        ],
    },
    "u_sys_polarization": {
        "dtype": np.float32,
        "dim": ["wavelength"],
        "attrs": {"units": "%"},
        "err_corr": [
            {"dim": "wavelength", "form": "systematic", "params": [], "units": []}
        ],
    },
}

TEMPLATE_SIGNALS = {
    "signals": {
        "dtype": np.float32,
        "dim": ["channels", "dts"],
        "attrs": {"units": [], "u_components": ["u_coeff"]},
    },
    "u_ran_signals": {
        "dtype": np.float32,
        "dim": ["channels", "dts"],
        "attrs": {"units": "%"},
        "err_corr": [{"dim": "channels", "form": "random", "params": [], "units": []}],
    },
    "u_sys_signals": {
        "dtype": np.float32,
        "dim": ["channels", "dts"],
        "attrs": {"units": "%"},
        "err_corr": [
            {"dim": "channels", "form": "systematic", "params": [], "units": []}
        ],
    },
}
