"""describe class"""

"""___Built-In Modules___"""
from typing import Union, List
from datetime import datetime

"""___Third-Party Modules___"""
import netCDF4 as nc
import numpy as np

"""___NPL Modules___"""
from ..datatypes.datatypes import (
    SpectralResponseFunction,
)

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "20/05/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


def read_srf(filepath: str) -> SpectralResponseFunction:
    """
    Read a glod-formatted netcdf srf file and create a SpectralResponseFunction data object
    from it.

    Parameters
    ----------
    filepath: str
        Path where the file is located.

    Returns
    -------
    srf: SpectralResponseFunction
        Generated SpectralResponseFunction data object.
    """
    ds = nc.Dataset(filepath)
    wvlens = []
    factors = []
    wlen_units: str = ds["wavelength"].units
    if wlen_units == "m":
        f_to_nm = 1000000000
    elif wlen_units == "mm":
        f_to_nm = 1000000
    elif wlen_units == "um" or wlen_units == "μm":
        f_to_nm = 1000
    else:  # wlen_units == 'nm':
        f_to_nm = 1
    for wvlen_arr in ds["wavelength"]:
        for wvlen in wvlen_arr:
            if not isinstance(wvlen, np.ma.core.MaskedConstant):
                wvlens.append(wvlen * f_to_nm)
    for factor_arr in ds["srf"]:
        for factor in factor_arr:
            if not isinstance(factor, np.ma.core.MaskedConstant):
                factors.append(factor)
    spectral_response = dict(zip(wvlens, factors))
    return SpectralResponseFunction(spectral_response)