"""
This module contains the functionality that read Spectral Response Function files
from GLOD format files.

It exports the following functions:
    * read_srf - Read a glod-formatted netcdf srf file.
"""

"""___Built-In Modules___"""
import os
import logging

"""___Third-Party Modules___"""
import netCDF4 as nc
import numpy as np

"""___NPL Modules___"""
from ..datatypes.datatypes import (
    SRFChannel,
    SpectralResponseFunction,
)

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "20/05/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"

_READ_FILE_ERROR_STR = (
    "There was a problem while loading the file. See log for details."
)


def _calc_factor_to_nm(units: str) -> float:
    if units == "m":
        f_to_nm = 1000000000
    elif units == "mm":
        f_to_nm = 1000000
    elif units == "um" or units == "μm":
        f_to_nm = 1000
    else:  # wlen_units == 'nm':
        f_to_nm = 1
    return f_to_nm


def _append_if_not_masked(l: list, value: float, factor: float = None):
    if not isinstance(value, np.ma.core.MaskedConstant):
        if factor:
            value = value * factor
        l.append(value)


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
    try:
        ds = nc.Dataset(filepath)
        n_channels = len(ds["channel"])
        wvlens = [[] for _ in range(n_channels)]
        factors = [[] for _ in range(n_channels)]
        wlen_units: str = ds["wavelength"].units
        f_to_nm = _calc_factor_to_nm(wlen_units)
        for wvlen_arr in ds["wavelength"]:
            for i in range(len(wvlen_arr)):
                _append_if_not_masked(wvlens[i], wvlen_arr[i], f_to_nm)
        for factor_arr in ds["srf"]:
            for i in range(len(factor_arr)):
                _append_if_not_masked(factors[i], factor_arr[i])
        channels = []
        ch_units: str = ds["channel"].units
        ch_f_to_nm = _calc_factor_to_nm(ch_units)
        for i in range(n_channels):
            channel_spec_resp = dict(zip(wvlens[i], factors[i]))
            center = float(ds["channel"][i].data) * ch_f_to_nm
            c_id = ds["channel_id"][i]
            channel = SRFChannel(center, c_id, channel_spec_resp)
            channels.append(channel)
        name = os.path.basename(filepath)
        return SpectralResponseFunction(name, channels)
    except Exception as e:
        logging.critical(e)
        raise Exception(_READ_FILE_ERROR_STR)
