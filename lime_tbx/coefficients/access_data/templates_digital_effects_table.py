import numpy as np

# define ds variables
template_cimel = {
    "coeff": {
        "dtype": np.float32,
        "dim": ["wavelength","i_coeff"],
        "attrs": {
            "units": [],
            "u_components": ["u_coeff"]
        }
    },
    "u_coeff": {
        "dtype": np.int16,
        "dim": ["wavelength","i_coeff"],
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

