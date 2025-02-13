from typing import Iterable

import numpy as np
import xarray as xr
from xarray_schema import DatasetSchema, SchemaError


def xr_open_dataset(
    filepath: str, mask_fillvalue: bool = True, mask_limits: bool = True
) -> xr.Dataset:
    """Open a netCDF dataset as an xarray Dataset

    Read the netCDF dataset masking the appropiate values, checking
    against the attributes `valid_min` and `valid_max`.

    Parameters
    ----------
    filepath: str
        Path where the netCDF file is located

    Returns
    -------
    ds: xr.Dataset
        Dataset with the information of the netCDF file.
    """
    ds = xr.open_dataset(filepath, mask_and_scale=mask_fillvalue)
    if mask_limits:
        for vname in list(ds.data_vars.keys()):
            values = ds[vname].values
            if hasattr(ds[vname], "valid_min"):
                valid_min = ds[vname].valid_min
                values = np.where(values >= valid_min, values, np.nan)
            if hasattr(ds[vname], "valid_max"):
                valid_max = ds[vname].valid_max
                values = np.where(values <= valid_max, values, np.nan)
            ds[vname].values = values
    return ds


def _validate_schema(dss: DatasetSchema, ds: xr.Dataset):
    """Validates the dataset `ds` against the schema `dss`.

    Parameters
    ----------
    dss: xarray_schema.DatasetSchema
        Schema that the dataset is going to be validated against.
    ds: xarray.Dataset
        Dataset to validate.

    Raises
    ------
    SchemaError
        Error indicating that the dataset is not following the schema.
    """
    try:
        dss.validate(ds)
    except NotImplementedError:
        pass
    if dss.coords is not None:
        for key, da_schema in dss.coords.items():
            if da_schema is not None:
                if key not in ds.coords:
                    raise SchemaError(f"Coordinate '{key}' not in dataset")
                try:
                    da_schema.validate(ds.coords[key])
                except SchemaError as e:
                    raise SchemaError(f"Error in coordinate '{key}': {e}") from e


def validate_schema(
    dss: DatasetSchema, ds: xr.Dataset, other_schemas: Iterable[DatasetSchema] = None
):
    """Validates the dataset `ds` against the schema `dss`.

    Parameters
    ----------
    dss: xarray_schema.DatasetSchema
        Schema that the dataset is going to be validated against.
    ds: xarray.Dataset
        Dataset to validate.
    fallback_dss: Iterable[xarray_schema.DatasetSchema]
        If the dataset can have multiple schemas, it will be validated
        against those before the canonical one. Once it succeeds in a validation,
        it's understood as validated and finishes the validation.

    Raises
    ------
    SchemaError
        Error indicating that the dataset is not following the schema.
    """
    if other_schemas is not None:
        for odss in other_schemas:
            try:
                _validate_schema(odss, ds)
                return
            except SchemaError:
                pass
    _validate_schema(dss, ds)
