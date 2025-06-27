"""
This module contains the functionality that reads moon observations from GLOD format files.

It exports the following functions:
    * read_moon_obs - Read a GLOD-formatted netcdf moon observations file.
"""

"""___Built-In Modules___"""
from datetime import datetime, timezone

"""___Third-Party Modules___"""
import numpy as np
import xarray as xr
from xarray_schema import DatasetSchema, DataArraySchema, SchemaError
from xarray_schema.components import AttrsSchema

"""___LIME_TBX Modules___"""
from lime_tbx.common.datatypes import (
    KernelsPath,
    LunarObservation,
    SatellitePosition,
    EocfiPath,
    MoonData,
)
from lime_tbx.common import logger
from lime_tbx.business.spice_adapter.spice_adapter import SPICEAdapter
from lime_tbx.business.lime_algorithms.lime.eli import DIST_EARTH_MOON_KM
from lime_tbx.business.eocfi_adapter.eocfi_adapter import EOCFIConverter
from lime_tbx.application.filedata.netcdfcommon import (
    xr_open_dataset,
    validate_schema,
    AttrSchema,
    get_length_conversion_factor,
)

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "20/05/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Production"


_READ_FILE_ERROR_STR = (
    "There was a problem while loading the file. See log for details."
)


def _calc_divisor_to_nm(units: str) -> float:
    """
    Get the conversion divisor factor to nanometers for spectral irradiance units.

    This function returns the divisor needed to convert spectral irradiance units
    from their given wavelength denominator (meters, millimeters, micrometers, or nanometers)
    to a per-nanometer (`nm-1`) basis.

    Parameters
    ----------
    units : str
        The spectral irradiance unit to be converted.

    Returns
    -------
    float
        The divisor to convert the input unit to a per-nanometer basis.
    """
    if units == "W m-2 m-1":
        d_to_nm = 1000000000
    elif units == "W m-2 mm-1":
        d_to_nm = 1000000
    elif units == "W m-2 um-1" or units == "W m-2 μm-1":
        d_to_nm = 1000
    else:  # units == 'W m-2 nm-1':
        d_to_nm = 1
    return d_to_nm


def _validate_schema_regular_moonobs(ds: xr.Dataset):
    """Validates that a xarray dataset follows the Basic Moon Observation schema.

    The dataset structure is:

    coordinates: 'date': float or datetime
    dims without coordinates: 'chan' and 'sat_xyz'
    data_vars: 'channel_name': str, 'sat_pos': float, 'sat_pos_ref': str, 'irr_obs': float

    Parameters
    ----------
    ds: xr.Dataset
        Dataset to validate
    """
    data_vars = {
        "channel_name": DataArraySchema(np.generic, dims=["chan"]),
        "sat_pos": DataArraySchema(np.floating, dims=["sat_xyz"]),
        "sat_pos_ref": DataArraySchema(np.generic, dims=[]),
        "irr_obs": DataArraySchema(np.floating, dims=["chan"]),
    }
    coords = {"date": DataArraySchema(np.floating)}
    attrs = AttrsSchema({"data_source": AttrSchema(np.generic)})

    def _check_dim_lengths(ds: xr.Dataset):
        if len(ds["sat_xyz"]) != 3:
            raise SchemaError("Dimension 'sat_xyz' should be of length=3.")
        if len(ds["date"]) != 1:
            raise SchemaError("Dimension 'date' should be of length=1.")

    checks = [_check_dim_lengths]
    dss = DatasetSchema(
        data_vars=data_vars,
        coords=coords,
        attrs=attrs,
        checks=checks,
    )
    date_datetime = {"date": DataArraySchema(np.datetime64)}
    odss = DatasetSchema(
        data_vars=data_vars,
        coords=date_datetime,
        attrs=attrs,
        checks=checks,
    )
    validate_schema(dss, ds, [odss])


def _get_moondata_from_moon_obs(
    ds: xr.Dataset, dt: datetime, kernels_path: KernelsPath, eocfi_path: EocfiPath
) -> MoonData:
    # sun vars
    sun_vars = {"distance_sun_moon": None, "sun_sel_lon": None}  # 'sun_sel_lat'
    sun_var_smd_eq = {
        "distance_sun_moon": "dist_sun_moon_au",
        "sun_sel_lon": "lon_sun_rad",
    }
    smd = None
    for suva in sun_vars:
        if suva in ds.variables:
            sun_vars[suva] = ds[suva].values[0]
        else:
            if smd is None:
                smd = SPICEAdapter.get_solar_moon_datas(
                    [dt], kernels_path.main_kernels_path
                )[0]
                smd.lon_sun_rad = np.degrees(smd.lon_sun_rad)
            sun_vars[suva] = getattr(smd, sun_var_smd_eq[suva])
    sun_sel_lon = sun_vars["sun_sel_lon"]
    dsm = sun_vars["distance_sun_moon"]
    # sat vars
    sat_vars = {
        "distance_sat_moon": None,
        "sat_sel_lon": None,
        "sat_sel_lat": None,
        "phase_angle": None,
    }
    sat_var_md_eq = {
        "distance_sat_moon": "distance_observer_moon",
        "sat_sel_lon": "long_obs",
        "sat_sel_lat": "lat_obs",
        "phase_angle": "mpa_degrees",
    }
    mdeo = None
    for sava in sat_vars:
        if sava in ds.variables:
            sat_vars[sava] = ds[sava].values[0]
        else:
            if mdeo is None:
                eo = EOCFIConverter(eocfi_path, kernels_path)
                _sname = ds["sat_name"].values
                if str(_sname.dtype.str).startswith("|S"):
                    sat_name = str(_sname, "utf-8")
                else:
                    sat_name = _sname[0]
                xyzs = eo.get_satellite_position_rectangular(sat_name, [dt])
                mdeo = SPICEAdapter.get_moon_datas_from_rectangular_multiple(
                    xyzs, [dt], kernels_path, "ITRF93"
                )[0]
            sat_vars[sava] = getattr(mdeo, sat_var_md_eq[sava])
    dom = sat_vars["distance_sat_moon"]
    sat_sel_lon = sat_vars["sat_sel_lon"]
    sat_sel_lat = sat_vars["sat_sel_lat"]
    mpa = sat_vars["phase_angle"]
    geom_factor = None
    geom_factor_names = ["geom_factor", "geom_const"]
    for gfn in geom_factor_names:
        if gfn in ds.variables:
            geom_factor = ds[gfn].values[0]
            break
    md = MoonData(
        dsm,
        dom,
        np.radians(sun_sel_lon),
        sat_sel_lat,
        sat_sel_lon,
        abs(mpa),
        mpa,
        geom_factor,
    )
    return md


def read_moon_obs(
    path: str, kernels_path: KernelsPath, eocfi_path: EocfiPath
) -> LunarObservation:
    """
    Read a glod-formatted netcdf moon observations file and create a data object
    from it.

    Parameters
    ----------
    path: str
        Path where the file is located.

    Returns
    -------
    moon_obs: MoonObservation
        Generated MoonObservation from the given datafile
    """
    try:
        ds = xr_open_dataset(path, mask_limits={"irr_obs": True, "__default__": False})
        _validate_schema_regular_moonobs(ds)
        ch_names = np.char.rstrip(
            np.char.decode(ds["channel_name"].to_numpy(), "utf-8"), "\x00"
        )
        dt = ds["date"].to_numpy().item()
        if isinstance(dt, np.datetime64):
            dt = (
                dt.astype("datetime64[us]")
                .astype(datetime)
                .replace(tzinfo=timezone.utc)
            )
        else:
            dt = datetime.fromtimestamp(dt, tz=timezone.utc)
        sat_pos_ref = ds["sat_pos_ref"].to_numpy()
        if isinstance(sat_pos_ref, np.ndarray):
            sat_pos_ref = sat_pos_ref.item()
        if isinstance(sat_pos_ref, (bytes, bytearray)):
            sat_pos_ref = sat_pos_ref.decode("utf-8", errors="replace")
        else:
            sat_pos_ref = str(sat_pos_ref)
        sat_pos_vals = ds["sat_pos"].to_numpy()
        has_sat_pos = ~np.isnan(sat_pos_vals).any()
        if has_sat_pos:
            d_to_m = get_length_conversion_factor(ds["sat_pos"].units, "m")
            sat_pos = SatellitePosition(*sat_pos_vals * d_to_m)
            md = None
            corr_dist = False
        else:
            sat_pos = None
            md = _get_moondata_from_moon_obs(ds, dt, kernels_path, eocfi_path)
            corr_dist = bool(int(ds.attrs.get("to_correct_distance", 0)))
        irr_obs = ds["irr_obs"].to_numpy()
        d_to_nm = _calc_divisor_to_nm(ds["irr_obs"].units)
        data_source = ds.attrs.get("data_source")
        ds.close()
        if corr_dist:
            irr_obs *= (1 / md.distance_sun_moon) ** 2 * (
                DIST_EARTH_MOON_KM / md.distance_observer_moon
            ) ** 2
        irr_obs /= d_to_nm
        valid = ~np.isnan(irr_obs)
        ch_irrs = dict(zip(ch_names[valid], irr_obs[valid].astype(float)))
        return LunarObservation(
            list(ch_names), sat_pos_ref, ch_irrs, dt, sat_pos, data_source, md
        )
    except SchemaError as e:
        raise e
    except Exception as e:
        logger.get_logger().exception(e)
        raise Exception(_READ_FILE_ERROR_STR)
