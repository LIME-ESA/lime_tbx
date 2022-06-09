import numpy as np

# define ds variables
template_cimel = {
    "coeff": {
        "dtype": np.float32,
        "dim": ["i_coeff","wavelength"],
        "attrs": {
            "units": [],
            "u_components": ["u_coeff"]
        }
    },
    "u_coeff": {
        "dtype": np.float32,
        "dim": ["i_coeff","wavelength"],
        "attrs": {"units": "%"},
        "err_corr": [
            {
                "dim": "wavelength",
                "form": "random",
                "params": [],
                "units": []
            }
        ]
    },
}

template_refl = {
    "reflectance": {
        "dtype": np.float32,
        "dim": ["wavelength"],
        "attrs": {
            "units": [],
            "u_components": ["u_coeff"]
        }
    },
    "u_ran_reflectance": {
        "dtype": np.float32,
        "dim": ["wavelength"],
        "attrs": {"units": "%"},
        "err_corr": [
            {
                "dim": "wavelength",
                "form": "random",
                "params": [],
                "units": []
            }
        ]
    },
    "u_sys_reflectance": {
        "dtype": np.float32,
        "dim": ["wavelength"],
        "attrs": {"units": "%"},
        "err_corr": [
            {
                "dim": "wavelength",
                "form": "systematic",
                "params": [],
                "units": []
            }
        ]
    },
}

template_irr = {
    "irradiance": {
        "dtype": np.float32,
        "dim": ["wavelength"],
        "attrs": {
            "units": [],
            "u_components": ["u_coeff"]
        }
    },
    "u_ran_irradiance": {
        "dtype": np.float32,
        "dim": ["wavelength"],
        "attrs": {"units": "%"},
        "err_corr": [
            {
                "dim": "wavelength",
                "form": "random",
                "params": [],
                "units": []
            }
        ]
    },
    "u_sys_irradiance": {
        "dtype": np.float32,
        "dim": ["wavelength"],
        "attrs": {"units": "%"},
        "err_corr": [
            {
                "dim": "wavelength",
                "form": "systematic",
                "params": [],
                "units": []
            }
        ]
    },
}

template_pol = {
    "polarization": {
        "dtype": np.float32,
        "dim": ["wavelength"],
        "attrs": {
            "units": [],
            "u_components": ["u_coeff"]
        }
    },
    "u_ran_polarization": {
        "dtype": np.float32,
        "dim": ["wavelength"],
        "attrs": {"units": "%"},
        "err_corr": [
            {
                "dim": "wavelength",
                "form": "random",
                "params": [],
                "units": []
            }
        ]
    },
    "u_sys_polarization": {
        "dtype": np.float32,
        "dim": ["wavelength"],
        "attrs": {"units": "%"},
        "err_corr": [
            {
                "dim": "wavelength",
                "form": "systematic",
                "params": [],
                "units": []
            }
        ]
    },
}
