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
    LunarObservationWrite,
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


def read_moon_obs(path: str) -> LunarObservation:
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
    ds.close()
    return LunarObservation(ch_names, sat_pos_ref, ch_irrs, dt, sat_pos)


_DT_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def write_obs(obs: List[LunarObservationWrite], path: str, dt: datetime):
    ds = nc.Dataset(path, "w", format="NETCDF4")
    ds.Conventions = "CF-1.6"
    ds.Metadata_Conventions = "Unidata Dataset Discovery v1.0"
    ds.standard_name_vocabulary = "CF Standard Name Table (Version 21, 12 January 2013)"
    ds.project = "LIME 2"
    ds.title = "LIME simulation lunar observation file"
    ds.summary = "Lunar observation file"
    ds.keywords = "GSICS, satellites, lunar, moon, observation, visible, LIME"
    ds.references = "TBD"
    ds.institution = "ESA"
    ds.licence = ""
    ds.creator_name = ""
    ds.creator_email = ""
    ds.creator_url = ""
    ds.instrument = "LIME TBX"
    ds.instrument_wmo_code = "N/A"
    ds.data_source = "N/A"
    dt_str = dt.strftime(_DT_FORMAT)
    ds.date_created = dt_str
    ds.date_modified = dt_str
    ds.history = "TBD"
    ds.id = os.path.basename(path)
    ds.wmo_data_category = 101
    ds.wmo_international_data_subcategory = 0
    ds.processing_level = "v1.0.0"
    ds.doc_url = "N/A"
    ds.doc_doi = "N/A"
    ds.time_coverage_start = min(obs, key=lambda o: o.dt).dt.strftime(_DT_FORMAT)
    ds.time_coverage_end = max(obs, key=lambda o: o.dt).dt.strftime(_DT_FORMAT)
    ds.reference_model = "LIME2 coefficients v0"  # TODO add coefficients version
    # DIMENSIONS
    chan = ds.createDimension("chan", len(obs[0].ch_names))
    chan_strlen = ds.createDimension("chan_strlen", 6)
    date = ds.createDimension("date", len(obs))
    sat_ref_strlen = ds.createDimension("sat_ref_strlen", 6)
    col = ds.createDimension("col", 0)
    row = ds.createDimension("row", 0)
    sat_xyz = ds.createDimension("sat_xyz", 3)
    wlens_dim = ds.createDimension("wlens", len(obs[0].irrs.wlens))
    # VARIABLES
    dates = ds.createVariable("date", "f4", ("date",))
    dates.standard_name = "time"
    dates.long_name = "time of lunar observation"
    dates.units = "seconds since 1970-01-01T00:00:00Z"
    dates.calendar = "gregorian"
    dates[:] = np.array([o.dt.timestamp() for o in obs])
    channel_name = ds.createVariable("channel_name", "S1", ("chan", "chan_strlen"))
    channel_name.standard_name = "sensor_band_identifier"
    channel_name.long_name = "channel identifier"
    chn = np.array([np.array([ch], "S4") for ch in obs[0].ch_names])
    channel_name[:] = chn
    sat_pos = ds.createVariable("sat_pos", "f4", ("date", "sat_xyz"))
    sat_pos.long_name = "satellite position x y z in sat_pos_ref"
    sat_pos.units = "km"
    sat_pos.valid_min = 0.0
    sat_pos.valid_max = 999999995904.0
    sat_pos[:] = np.array(
        [np.array([o.sat_pos.x, o.sat_pos.y, o.sat_pos.z]) for o in obs]
    )
    sat_pos_ref = ds.createVariable("sat_pos_ref", "S1", ("sat_ref_strlen",))
    sat_pos_ref.long_name = "reference frame of satellite position"
    sat_pos_ref[:] = obs[0].sat_pos_ref
    irr_obs = ds.createVariable("irr_obs", "f4", ("date", "chan"))
    irr_obs.units = "W m-2 nm-1"
    irr_obs.long_name = "observed lunar irradiance"
    irr_obs.valid_min = 0.0
    irr_obs.valid_max = 1000000.0
    irr_obs[:] = np.array([o.ch_irrs[ch] for ch in obs[0].ch_names for o in obs])
    irr_obs_unc = ds.createVariable("irr_obs_unc", "f4", ("date", "chan"))
    irr_obs_unc.units = "W m-2 nm-1"
    irr_obs_unc.long_name = "uncertainties of the observed lunar irradiance"
    irr_obs_unc.valid_min = 0.0
    irr_obs_unc.valid_max = 1000000.0
    irr_obs_unc[:] = np.array(
        [np.array([o.signals_uncs[i] for i in range(len(o.signals_uncs))]) for o in obs]
    )
    # Spectral irr refl and obs
    wlens = ds.createVariable("wlens", "f4", ("wlens",))
    wlens.units = "nm"
    wlens.long_name = "Wavelengths for irr_spectrum, refl_spectrum and polar_spectrum"
    wlens.valid_min = 0.0
    wlens.valid_max = 1000000.0
    wlens[:] = obs[0].irrs.wlens
    irr_spectrum = ds.createVariable("irr_spectrum", "f4", ("date", "wlens"))
    irr_spectrum.units = "W m-2 nm-1"
    irr_spectrum.long_name = "simulated lunar irradiance per wavelength"
    irr_spectrum.valid_min = 0.0
    irr_spectrum.valid_max = 1000000.0
    irr_spectrum[:] = np.array(
        [np.array([o.irrs.data[i] for i in range(len(obs[0].irrs.wlens))]) for o in obs]
    )
    irr_spectrum_unc = ds.createVariable("irr_spectrum_unc", "f4", ("date", "wlens"))
    irr_spectrum_unc.units = "W m-2 nm-1"
    irr_spectrum_unc.long_name = (
        "uncertainties of the simulated lunar irradiance per wavelength"
    )
    irr_spectrum_unc.valid_min = 0.0
    irr_spectrum_unc.valid_max = 1000000.0
    irr_spectrum_unc[:] = np.array(
        [
            np.array([o.irrs.uncertainties[i] for i in range(len(obs[0].irrs.wlens))])
            for o in obs
        ]
    )
    refl_spectrum = ds.createVariable("refl_spectrum", "f4", ("date", "wlens"))
    refl_spectrum.units = "Fractions of unity"
    refl_spectrum.long_name = "simulated lunar degree of reflectance per wavelength"
    refl_spectrum.valid_min = 0.0
    refl_spectrum.valid_max = 1.0
    refl_spectrum[:] = np.array(
        [
            np.array([o.refls.data[i] for i in range(len(obs[0].refls.wlens))])
            for o in obs
        ]
    )
    refl_spectrum_unc = ds.createVariable("refl_spectrum_unc", "f4", ("date", "wlens"))
    refl_spectrum_unc.units = "Fractions of unity"
    refl_spectrum_unc.long_name = (
        "uncertainties of the simulated lunar degree of reflectance per wavelength"
    )
    refl_spectrum_unc.valid_min = 0.0
    refl_spectrum_unc.valid_max = 1.0
    refl_spectrum_unc[:] = np.array(
        [
            np.array([o.refls.uncertainties[i] for i in range(len(obs[0].refls.wlens))])
            for o in obs
        ]
    )
    polar_spectrum = ds.createVariable("polar_spectrum", "f4", ("date", "wlens"))
    polar_spectrum.units = "Fractions of unity"
    polar_spectrum.long_name = "simulated lunar degree of polarization per wavelength"
    polar_spectrum.valid_min = -1.0
    polar_spectrum.valid_max = 1.0
    polar_vals = np.array(
        [
            np.array([o.polars.data[i] for i in range(len(obs[0].polars.wlens))])
            for o in obs
        ]
    )
    polar_spectrum[:] = polar_vals
    polar_spectrum_unc = ds.createVariable(
        "polar_spectrum_unc", "f4", ("date", "wlens")
    )
    polar_spectrum_unc.units = "Fractions of unity"
    polar_spectrum_unc.long_name = (
        "uncertainties of the simulated lunar degree of polarization per wavelength"
    )
    polar_spectrum_unc.valid_min = -1.0
    polar_spectrum_unc.valid_max = 1.0
    polar_spectrum_unc[:] = np.array(
        [
            np.array(
                [o.polars.uncertainties[i] for i in range(len(obs[0].polars.wlens))]
            )
            for o in obs
        ]
    )
    ds.close()
