"""
This module contains the functionality that read moon observations file from GLOD format files.

It exports the following functions:
    * read_moon_obs - Read a glod-formatted netcdf moon observations file.
"""

"""___Built-In Modules___"""
from dataclasses import dataclass
from datetime import datetime, timezone, tzinfo
import os
from typing import List, Union, Tuple


"""___Third-Party Modules___"""
import netCDF4 as nc
import numpy as np

"""___NPL Modules___"""
from ..datatypes.datatypes import (
    LGLODComparisonData,
    LGLODData,
    LunarObservation,
    LunarObservationWrite,
    SatellitePosition,
    SelenographicDataWrite,
    SpectralData,
    SpectralResponseFunction,
)
from lime_tbx.datatypes import constants

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
    dt = datetime.fromtimestamp(float(ds["date"][0].data), tz=timezone.utc)
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


def _write_start_dataset(
    lglod: LGLODData, path: str, dt: datetime, coefficients_version: str
):
    obs = lglod.observations
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
    quant_dates = len(obs)
    if quant_dates == 1 and obs[0].dt == None:
        quant_dates = 0
    if quant_dates > 0:
        ds.time_coverage_start = min(obs, key=lambda o: o.dt).dt.strftime(_DT_FORMAT)
        ds.time_coverage_end = max(obs, key=lambda o: o.dt).dt.strftime(_DT_FORMAT)
    else:
        ds.time_coverage_start = ""
        ds.time_coverage_end = ""
    ds.reference_model = "LIME2 coefficients version {}".format(
        coefficients_version
    )  # TODO coefficients
    ds.not_default_srf = int(lglod.not_default_srf)
    return ds


@dataclass
class _NormalSimulationData:
    quant_dates: int
    coefficients_version: str
    ch_names: List[str]
    sat_pos_ref: str
    sat_names: List[str]
    sat_pos: List[SatellitePosition]
    dates: List[datetime]


def _write_normal_simulations(
    lglod: Union[LGLODData, LGLODComparisonData],
    path: str,
    dt: datetime,
    sim_data: _NormalSimulationData,
):
    ds = _write_start_dataset(lglod, path, dt, sim_data.coefficients_version)
    # DIMENSIONS
    max_len_strlen = len(max(sim_data.ch_names, key=len))
    chan_st_type = "S{}".format(max_len_strlen)
    max_len_sat_pos_ref = len(max(sim_data.sat_pos_ref, key=len))
    sat_pos_ref_st_type = "S{}".format(max_len_sat_pos_ref)
    max_len_sat_name = len(max(sim_data.sat_names, key=len))
    sat_name_st_type = "S{}".format(max_len_sat_name)
    chan = ds.createDimension("chan", len(sim_data.ch_names))
    chan_strlen = ds.createDimension("chan_strlen", max_len_strlen)
    date = ds.createDimension("date", sim_data.quant_dates)
    number_obs = ds.createDimension("number_obs", len(sim_data.sat_pos))
    sat_ref_strlen = ds.createDimension("sat_ref_strlen", max_len_sat_pos_ref)
    sat_name_strlen = ds.createDimension("sat_name_strlen", max_len_sat_name)
    col = ds.createDimension("col", 0)
    row = ds.createDimension("row", 0)
    sat_xyz = ds.createDimension("sat_xyz", 3)
    # VARIABLES
    dates = ds.createVariable("date", "f4", ("date",))
    dates.standard_name = "time"
    dates.long_name = "time of lunar observation"
    dates.units = "seconds since 1970-01-01T00:00:00Z"
    dates.calendar = "gregorian"
    if sim_data.quant_dates > 0:
        dates[:] = np.array([dt.timestamp() for dt in sim_data.dates])
    else:
        dates[:] = np.array([])
    channel_name = ds.createVariable("channel_name", "S1", ("chan", "chan_strlen"))
    channel_name.standard_name = "sensor_band_identifier"
    channel_name.long_name = "channel identifier"
    channel_name[:] = np.array(
        [nc.stringtochar(np.array([ch], chan_st_type)) for ch in sim_data.ch_names]
    )
    sat_pos = ds.createVariable("sat_pos", "f4", ("number_obs", "sat_xyz"))
    sat_pos.long_name = "satellite position x y z in sat_pos_ref"
    sat_pos.units = "km"
    sat_pos.valid_min = 0.0
    sat_pos.valid_max = 999999995904.0
    sat_pos[:] = np.array(
        [np.array([sat_pos.x, sat_pos.y, sat_pos.z]) for sat_pos in sim_data.sat_pos]
    )
    sat_pos_ref = ds.createVariable("sat_pos_ref", "S1", ("sat_ref_strlen",))
    sat_pos_ref.long_name = "reference frame of satellite position"
    sat_pos_ref[:] = nc.stringtochar(
        np.array([sim_data.sat_pos_ref], sat_pos_ref_st_type)
    )
    sat_name = ds.createVariable("sat_name", "S1", ("sat_name_strlen",))
    sat_name.long_name = "Name of the satellite (or empty if it wasn't a satellite)"
    sat_name[:] = nc.stringtochar(np.array([sim_data.sat_names[0]], sat_name_st_type))

    return ds


def write_obs(
    lglod: LGLODData,
    path: str,
    dt: datetime,
    coefficients_version: str = "Mock coefficients",
):
    obs = lglod.observations
    quant_dates = len(obs)
    if quant_dates == 1 and obs[0].dt == None:
        quant_dates = 0
    sim_data = _NormalSimulationData(
        quant_dates,
        coefficients_version,
        obs[0].ch_names,
        obs[0].sat_pos_ref,
        [o.sat_name for o in obs],
        [o.sat_pos for o in obs],
        [o.dt for o in obs],
    )
    ds = _write_normal_simulations(lglod, path, dt, sim_data)
    # dims
    wlens_dim = ds.createDimension("wlens", len(obs[0].irrs.wlens))
    wlens_cimel = ds.createDimension("wlens_cimel", len(lglod.elis_cimel[0].wlens))
    # vals
    if quant_dates == 0:
        distance_sun_moon = ds.createVariable(
            "distance_sun_moon", "f4", ("number_obs",)
        )
        distance_sun_moon.long_name = "Distance between the Sun and the Moon."
        distance_sun_moon.units = "AU"
        distance_sun_moon[:] = np.array(
            [o.selenographic_data.distance_sun_moon for o in obs]
        )
        selen_sun_lon_rad = ds.createVariable(
            "selen_sun_lon_rad", "f4", ("number_obs",)
        )
        selen_sun_lon_rad.long_name = "Selenographic longitude of the Sun"
        selen_sun_lon_rad.units = "Radians"
        selen_sun_lon_rad[:] = np.array(
            [o.selenographic_data.selen_sun_lon_rad for o in obs]
        )
        mpa_degrees = ds.createVariable("mpa_degrees", "f4", ("number_obs",))
        mpa_degrees.long_name = "Moon phase angle"
        mpa_degrees.units = "Decimal degrees"
        mpa_degrees[:] = np.array([o.selenographic_data.mpa_degrees for o in obs])
    irr_obs = ds.createVariable("irr_obs", "f4", ("number_obs", "chan"))
    irr_obs.units = "W m-2 nm-1"
    irr_obs.long_name = "observed lunar irradiance for each channel"
    irr_obs.valid_min = 0.0
    irr_obs.valid_max = 1000000.0
    irr_obs[:] = lglod.signals.data
    irr_obs_unc = ds.createVariable("irr_obs_unc", "f4", ("number_obs", "chan"))
    irr_obs_unc.units = "W m-2 nm-1"
    irr_obs_unc.long_name = (
        "uncertainties of the observed lunar irradiance for each channel"
    )
    irr_obs_unc.valid_min = 0.0
    irr_obs_unc.valid_max = 1000000.0
    irr_obs_unc[:] = lglod.signals.uncertainties
    wlens = ds.createVariable("wlens", "f4", ("wlens",))
    wlens.units = "nm"
    wlens.long_name = "Wavelengths for irr_spectrum, refl_spectrum and polar_spectrum"
    wlens.valid_min = 0.0
    wlens.valid_max = 1000000.0
    wlens[:] = obs[0].irrs.wlens
    irr_spectrum = ds.createVariable("irr_spectrum", "f4", ("number_obs", "wlens"))
    irr_spectrum.units = "W m-2 nm-1"
    irr_spectrum.long_name = "simulated lunar irradiance per wavelength"
    irr_spectrum.valid_min = 0.0
    irr_spectrum.valid_max = 1000000.0
    irr_spectrum[:] = np.array(
        [np.array([o.irrs.data[i] for i in range(len(obs[0].irrs.wlens))]) for o in obs]
    )
    irr_spectrum_unc = ds.createVariable(
        "irr_spectrum_unc", "f4", ("number_obs", "wlens")
    )
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
    refl_spectrum = ds.createVariable("refl_spectrum", "f4", ("number_obs", "wlens"))
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
    refl_spectrum_unc = ds.createVariable(
        "refl_spectrum_unc", "f4", ("number_obs", "wlens")
    )
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
    polar_spectrum = ds.createVariable("polar_spectrum", "f4", ("number_obs", "wlens"))
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
        "polar_spectrum_unc", "f4", ("number_obs", "wlens")
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
    cimel_wlens = ds.createVariable("cimel_wlens", "f4", ("wlens_cimel",))
    cimel_wlens.units = "nm"
    cimel_wlens.long_name = "Cimel wavelengths"
    cimel_wlens[:] = lglod.elis_cimel[0].wlens
    irr_cimel = ds.createVariable("irr_cimel", "f4", ("number_obs", "wlens_cimel"))
    irr_cimel.units = "W m-2 nm-1"
    irr_cimel.long_name = "Simulated irradiance for the cimel wavelengths."
    irr_cimel[:] = np.array([cimel.data for cimel in lglod.elis_cimel])
    irr_cimel_unc = ds.createVariable(
        "irr_cimel_unc", "f4", ("number_obs", "wlens_cimel")
    )
    irr_cimel_unc.units = "W m-2 nm-1"
    irr_cimel_unc.long_name = (
        "Uncertainties for the simulated irradiance for the cimel wavelengths."
    )
    irr_cimel_unc[:] = np.array([cimel.uncertainties for cimel in lglod.elis_cimel])
    refl_cimel = ds.createVariable("refl_cimel", "f4", ("number_obs", "wlens_cimel"))
    refl_cimel.units = "Fractions of unity"
    refl_cimel.long_name = "Simulated reflectance for the cimel wavelengths."
    refl_cimel[:] = np.array([cimel.data for cimel in lglod.elrefs_cimel])
    refl_cimel_unc = ds.createVariable(
        "refl_cimel_unc", "f4", ("number_obs", "wlens_cimel")
    )
    refl_cimel_unc.units = "Fractions of unity"
    refl_cimel_unc.long_name = (
        "Uncertainties for the simulated reflectance for the cimel wavelengths."
    )
    refl_cimel_unc[:] = np.array([cimel.uncertainties for cimel in lglod.elrefs_cimel])
    ds.close()


def read_lime_glod(path: str) -> LGLODData:
    ds = nc.Dataset(path)
    not_default_srf = bool(ds.not_default_srf)
    datetimes = list(
        map(
            lambda x: datetime.fromtimestamp(x, tz=timezone.utc),
            map(int, ds.variables["date"][:]),
        )
    )
    channel_names_0 = [
        chn.tobytes().decode("utf-8") for chn in ds.variables["channel_name"][:].data
    ]
    lambda_to_satpos = lambda xyz: SatellitePosition(*xyz)
    sat_poss = list(map(lambda_to_satpos, ds.variables["sat_pos"][:].data))
    sat_pos_ref_0 = ds.variables["sat_pos_ref"][:].data.tobytes().decode("utf-8")
    signals_data = np.array(ds.variables["irr_obs"][:].data)
    signals_uncs = np.array(ds.variables["irr_obs_unc"][:].data)
    signals = SpectralData(
        np.array(channel_names_0), signals_data, np.array(signals_uncs), None
    )
    wlens = list(map(float, ds.variables["wlens"][:].data))
    lambda_to_satname = lambda data: data.tobytes().decode("utf-8").replace("\x00", "")
    sat_name_0 = lambda_to_satname(ds.variables["sat_name"][:].data)
    irr_spectrum = [
        list(map(float, data)) for data in ds.variables["irr_spectrum"][:].data
    ]
    irr_spectrum_unc = [
        list(map(float, data)) for data in ds.variables["irr_spectrum_unc"][:].data
    ]
    refl_spectrum = [
        list(map(float, data)) for data in ds.variables["refl_spectrum"][:].data
    ]
    refl_spectrum_unc = [
        list(map(float, data)) for data in ds.variables["refl_spectrum_unc"][:].data
    ]
    polar_spectrum = [
        list(map(float, data)) for data in ds.variables["polar_spectrum"][:].data
    ]
    polar_spectrum_unc = [
        list(map(float, data)) for data in ds.variables["polar_spectrum_unc"][:].data
    ]
    cimel_wlens = np.array(ds.variables["cimel_wlens"][:].data)
    irr_cimel = [list(map(float, data)) for data in ds.variables["irr_cimel"][:].data]
    irr_cimel_unc = [
        list(map(float, data)) for data in ds.variables["irr_cimel_unc"][:].data
    ]
    refl_cimel = [list(map(float, data)) for data in ds.variables["refl_cimel"][:].data]
    refl_cimel_unc = [
        list(map(float, data)) for data in ds.variables["refl_cimel_unc"][:].data
    ]
    if len(datetimes) == 0:
        mpa_degrees = ds.variables["mpa_degrees"][:].data
        distance_sun_moon = ds.variables["distance_sun_moon"][:].data
        selen_sun_lon_rad = ds.variables["selen_sun_lon_rad"][:].data
    obss = []
    ds.close()
    for i in range(len(sat_poss)):
        irrs = SpectralData(
            np.array(wlens),
            np.array(irr_spectrum[i]),
            np.array(irr_spectrum_unc[i]),
            None,
        )
        refls = SpectralData(
            np.array(wlens),
            np.array(refl_spectrum[i]),
            np.array(refl_spectrum_unc[i]),
            None,
        )
        polars = SpectralData(
            np.array(wlens),
            np.array(polar_spectrum[i]),
            np.array(polar_spectrum_unc[i]),
            None,
        )
        dt = None
        selenographic_data = None
        if len(datetimes) > 0:
            dt = datetimes[i]
        else:
            selenographic_data = SelenographicDataWrite(
                distance_sun_moon[i], selen_sun_lon_rad[i], mpa_degrees[i]
            )
        obs = LunarObservationWrite(
            channel_names_0,
            sat_pos_ref_0,
            dt,
            sat_poss[i],
            irrs,
            refls,
            polars,
            sat_name_0,
            selenographic_data,
        )
        obss.append(obs)
    elis_cimel = [
        SpectralData(
            cimel_wlens, np.array(irr_cimel[i]), np.array(irr_cimel_unc[i]), None
        )
        for i in range(len(irr_cimel))
    ]
    elrefs_cimel = [
        SpectralData(
            cimel_wlens, np.array(refl_cimel[i]), np.array(refl_cimel_unc[i]), None
        )
        for i in range(len(refl_cimel))
    ]
    return LGLODData(obss, signals, not_default_srf, elis_cimel, elrefs_cimel)


def write_comparison(
    lglod: LGLODComparisonData, path: str, dt: datetime, coefficients_version: str
):
    quant_dates = len(lglod.comparisons)
    ch_names = lglod.ch_names
    sat_pos_ref = lglod.sat_pos_ref
    sat_names = [lglod.sat_name for _ in range(quant_dates)]
    sat_pos = lglod.sat_pos
    dates = lglod.comparisons[0].dts
    sim_data = _NormalSimulationData(
        quant_dates,
        coefficients_version,
        ch_names,
        sat_pos_ref,
        sat_names,
        sat_pos,
        dates,
    )
    ds = _write_normal_simulations(lglod, path, dt, sim_data)
    # DIMENSIONS
