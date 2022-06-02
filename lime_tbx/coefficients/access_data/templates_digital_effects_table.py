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
        "dtype": np.int16,
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

