"""
This module contains the functionality that read Spectral Response Function files
from GLOD format files.

It exports the following functions:
    * read_srf - Read a glod-formatted netcdf srf file.
"""

"""___Built-In Modules___"""
import os

"""___Third-Party Modules___"""
import numpy as np
import xarray as xr
from xarray_schema import DatasetSchema, DataArraySchema, SchemaError

"""___LIME_TBX Modules___"""
from lime_tbx.common.datatypes import (
    SRFChannel,
    SpectralResponseFunction,
)
from lime_tbx.application.filedata.netcdfcommon import (
    validate_schema,
    xr_open_dataset,
    get_length_conversion_factor,
    DTypeSchema,
)
from lime_tbx.common import logger

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "20/05/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Production"

_READ_FILE_ERROR_STR = (
    "There was a problem while loading the SRF file. See log for details."
)


def _validate_schema_srf(ds: xr.Dataset):
    """Validates that a xarray dataset follows the SRF schema.

    The dataset structure is:

    coordinates: 'channel': float
    dims without coordinates: Any name, for example 'sample'.
    data_vars: 'srf': float, 'wavelength': float, 'channel_id': str

    Parameters
    ----------
    ds: xr.Dataset
        Dataset to validate
    """

    def get_data_vars(channel_dim: str = "channel"):
        sample_dim = None
        data_vars = {
            "srf": DataArraySchema(np.floating, dims=[sample_dim, channel_dim]),
            "wavelength": DataArraySchema(np.floating, dims=[sample_dim, channel_dim]),
        }
        return data_vars

    coords = {"channel": DataArraySchema(np.floating)}
    channel_id = {"channel_id": DataArraySchema(DTypeSchema(np.character))}
    # channel_id can actually be either a variable or a coordinate, preferably a variable.
    dss = DatasetSchema(
        data_vars=get_data_vars() | channel_id,
        coords=coords,
    )
    odss = DatasetSchema(
        data_vars=get_data_vars(),
        coords=coords | channel_id,
    )
    # channel can be a variable, then there are no coordinates
    flex_dss = DatasetSchema(
        data_vars=get_data_vars(None) | channel_id | coords,
    )
    validate_schema(dss, ds, [odss, flex_dss])


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
        ds = xr_open_dataset(filepath)
        _validate_schema_srf(ds)
        n_channels = len(ds["channel"])
        wvlens = [[] for _ in range(n_channels)]
        factors = [[] for _ in range(n_channels)]
        wlen_units = "nm"
        if hasattr(ds["wavelength"], "units"):
            wlen_units: str = ds["wavelength"].units
        f_to_nm = get_length_conversion_factor(wlen_units, "nm")
        if ds["wavelength"].ndim == 1:
            same_arr = ds["wavelength"].values * f_to_nm
            for i in range(n_channels):
                wvlens[i] = same_arr[~np.isnan(same_arr)]
        else:
            wvlen_arr = (ds["wavelength"].values * f_to_nm).T
            for i in range(n_channels):
                wvlens[i] = wvlen_arr[i][~np.isnan(wvlen_arr[i])]
        factor_arr = ds["srf"].values.T
        for i in range(n_channels):
            factors[i] = factor_arr[i][~np.isnan(factor_arr[i])]
        channels = []
        ch_units = "nm"
        if hasattr(ds["channel"], "units"):
            ch_units: str = ds["channel"].units
        ch_f_to_nm = get_length_conversion_factor(ch_units, "nm")
        for i in range(n_channels):
            channel_spec_resp = dict(zip(wvlens[i], factors[i]))
            center = float(ds["channel"][i].data) * ch_f_to_nm
            c_id = str(ds["channel_id"][i].values.astype(str))
            channel = SRFChannel(center, c_id, channel_spec_resp)
            channels.append(channel)
        name = os.path.basename(filepath)
        srf = SpectralResponseFunction(name, channels)
    except SchemaError as e:
        raise e
    except Exception as e:
        logger.get_logger().exception(e)
        raise Exception(_READ_FILE_ERROR_STR) from e
    finally:
        ds.close()
    return srf
