"""describe class"""

"""___Built-In Modules___"""
import os
from datetime import datetime


"""___Third-Party Modules___"""
import netCDF4 as nc
import numpy as np

"""___NPL Modules___"""
from ..datatypes.datatypes import (
    MoonCapture,
    MoonObservation,
    SatellitePosition,
)
from ..datatypes import constants

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "20/05/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


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
    ch_irrs = []
    for i in range(n_channels):
        is_full = isinstance(ds["channel_name"][i].mask, np.bool_)
        ch_name = str(ds["channel_name"][i].data, "utf-8")
        if not is_full:
            end = list(ds["channel_name"][i].mask).index(True)
            ch_name = ch_name[:end]
        ch_names.append(ch_name)
        ch_irrs.append([])
    n_dates = len(ds["date"])
    dates = []
    for i in range(n_dates):
        dates.append(datetime.fromtimestamp(float(ds["date"][i].data)))
    sat_pos_ref = str(ds["sat_pos_ref"][:].data, "utf-8")
    sat_positions = ds["sat_pos"][:]
    if len(ds["sat_pos"].shape) == 1:
        sat_positions = [sat_positions]
    irr_obs = ds["irr_obs"][:]
    if n_dates == 1:
        irr_obs = [irr_obs]
    positions = []
    for i, dt_irrs in enumerate(irr_obs):
        satpos = SatellitePosition(*list(map(float, sat_positions[i][:].data)))
        positions.append(satpos)
        dt = dates[i]
        for j, ch_irr in enumerate(dt_irrs):
            ch_irrs[j].append(MoonCapture(ch_irr, dt, satpos))
    return MoonObservation(ch_names, sat_pos_ref, ch_irrs, dates, positions)