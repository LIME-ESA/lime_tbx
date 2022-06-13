"""
This module contains the functionality that read moon observations file from GLOD format files.

It exports the following functions:
    * read_moon_obs - Read a glod-formatted netcdf moon observations file.
"""

"""___Built-In Modules___"""
import os
from datetime import datetime
from typing import List

"""___Third-Party Modules___"""
import netCDF4 as nc
import numpy as np

"""___NPL Modules___"""
from ..datatypes.datatypes import (
    LunarObservation,
    SatellitePosition,
)
from ..datatypes import constants

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "20/05/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


def _calc_divisor_to_nm(units: str) -> float:
    if units == "W m-2 m-1":
        d_to_nm = 1000000000
    elif units == "W m-2 mm-1":
        d_to_nm = 1000000
    elif units == "W m-2 um-1" or units == "W m-2 μm-1":
        d_to_nm = 1000
    else:  # units == 'W m-2 nm-1':
        d_to_nm = 1
    return d_to_nm


def read_moon_obs(path: str):
    """
    Read a glod-formatted netcdf moon observations file and create a data object
    from it.

    Parameters
    ----------
    filepath: str
        Path where the file is located.

    Returns
    -------
    moon_obs: MoonObservation
        Generated MoonObservation from the given datafile
    """
    ds = nc.Dataset(path)
    n_channels = len(ds["channel_name"])
    ch_names = []
    ch_irrs = {}
    for i in range(n_channels):
        is_full = isinstance(ds["channel_name"][i].mask, np.bool_)
        ch_name = str(ds["channel_name"][i].data, "utf-8")
        if not is_full:
            end = list(ds["channel_name"][i].mask).index(True)
            ch_name = ch_name[:end]
        ch_names.append(ch_name)
    dt = datetime.fromtimestamp(float(ds["date"][0].data))
    sat_pos_ref = str(ds["sat_pos_ref"][:].data, "utf-8")
    sat_pos = SatellitePosition(*list(map(float, ds["sat_pos"][:][:].data)))
    irr_obs = ds["irr_obs"][:]
    irr_obs_units: str = ds["irr_obs"].units
    d_to_nm = _calc_divisor_to_nm(irr_obs_units)
    for i, ch_irr in enumerate(irr_obs):
        if not isinstance(ch_irr, np.ma.core.MaskedConstant):
            ch_irrs[ch_names[i]] = float(ch_irr) / d_to_nm
    return LunarObservation(ch_names, sat_pos_ref, ch_irrs, dt, sat_pos)
