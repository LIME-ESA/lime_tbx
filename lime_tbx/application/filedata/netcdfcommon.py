"""Common functionalities for reading netCDF files with `xarray`."""

"""___Built-In Modules___"""
from typing import Iterable, Mapping, Any, Union

"""___Third-Party Modules___"""
import numpy as np
import xarray as xr
from xarray_schema import DatasetSchema, SchemaError
from xarray_schema.components import (
    AttrSchema as BrAttrSchema,
    DTypeSchema as BrDTypeSchema,
    DTypeLike,
)

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "13/02/2025"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Production"


def get_length_conversion_factor(from_unit: str, to_unit: str) -> float:
    """
    Get the conversion factor between two metric length units.

    This function calculates the factor needed to convert a length measurement
    from one metric unit to another. It supports various unit spellings, including
    American and British English variations.

    Parameters
    ----------
    from_unit : str
        The original unit of measurement. Supported units include metres (m),
        centimetres (cm), millimeters (mm), micrometres (μm / um), and nanometres (nm).
    to_unit : str
        The target unit of measurement. Must be one of the supported values listed above.

    Returns
    -------
    float
        The multiplication factor to convert from `from_unit` to `to_unit`.

    Raises
    ------
    ValueError
        If either `from_unit` or `to_unit` is not a recognized metric length unit.
    """

    def _gen_unit_variations(basename: str) -> list[str]:
        """Generate common variations of a metric unit (American & British spellings)."""
        return [
            f"{basename}metre",  # "metre" (British English)
            f"{basename}meter",  # "meter" (American English)
            f"{basename}metres",  # "metres" (plural British)
            f"{basename}meters",  # "meters" (plural American)
        ]

    # Build the metric factors dictionary efficiently
    metric_factors = {
        **dict.fromkeys(("nm", *_gen_unit_variations("nano")), -9),
        **dict.fromkeys(("um", "μm", *_gen_unit_variations("micro")), -6),
        **dict.fromkeys(("mm", *_gen_unit_variations("milli")), -3),
        **dict.fromkeys(("cm", *_gen_unit_variations("centi")), -2),
        **dict.fromkeys(("m", *_gen_unit_variations("")), 0),
        **dict.fromkeys(("km", *_gen_unit_variations("kilo")), 3),
    }
    if from_unit not in metric_factors:
        raise ValueError(
            f"Unsupported unit: {from_unit} or {to_unit}. Supported units: {list(metric_factors.keys())}"
        )
    factor = metric_factors[from_unit] - metric_factors[to_unit]
    factor = 10**factor
    return factor


class AttrSchema(BrAttrSchema):
    """
    Re-implementation of `xarray_schema.AttrSchema` but correctly validating and raising `SchemaError`s
    """

    def validate(self, attr: Any):
        """Validate that `attr` follows the schema.

        Parameters
        ----------
        attr: Any
            xarray dataset attribute that is going to be validated against the schema.

        Raises
        ------
        SchemaError
            Error indicating that `attr` is not following the schema.
        """
        if self.type is not None:
            if not np.issubdtype(type(attr), self.type):
                raise SchemaError(
                    f"Error in attribute {attr}: is not of type {self.type}"
                )
        if self.value is not None:
            if self.value is not None and self.value != attr:
                raise SchemaError(f"name {attr} != {self.value}")


class DTypeSchema(BrDTypeSchema):
    def __init__(self, dtype: DTypeLike) -> None:
        if dtype in [
            np.floating,
            np.integer,
            np.signedinteger,
            np.unsignedinteger,
            np.generic,
            np.character,
        ]:
            self.dtype = dtype
        else:
            self.dtype = np.dtype(dtype)


def xr_open_dataset(
    filepath: str,
    mask_fillvalue: Union[bool, Mapping[str, bool]] = True,
    mask_limits: Union[bool, Mapping[str, bool]] = True,
) -> xr.Dataset:
    """Open a netCDF dataset as an xarray Dataset

    Read the netCDF dataset masking the appropiate values, checking
    against the variable attributes `_FillValue`, `valid_min` and `valid_max`.

    Parameters
    ----------
    filepath: str
        Path where the netCDF file is located
    mask_fill_value: bool | Mapping[str, bool]
        If True, tell `xarray` replace array values equal to `_FillValue` with `NA` and scale values.
        Pass a mapping, e.g. {"my_variable": False}, to toggle this feature per-variable individually.
        If a mapping is passed and a variable is missing, it's understood that the variable is mapped to True by default.
    mask_limits: bool | Mapping[str, bool]
        If True, replace array values lower than `valid_min` or higher than `valid_max` with `NA`, if present.
        Pass a mapping, e.g. {"my_variable": False}, to toggle this feature per-variable individually.
        If a mapping is passed and a variable is missing, it's understood that the variable is mapped to True by default.

    Returns
    -------
    ds: xr.Dataset
        Dataset with the information of the netCDF file.
    """
    ds = xr.open_dataset(filepath, mask_and_scale=mask_fillvalue)
    for vname in list(ds.data_vars.keys()):
        mask_limits_var = False
        if isinstance(mask_limits, bool):
            mask_limits_var = mask_limits
        else:
            mask_limits_var = vname not in mask_limits or mask_limits[vname]
        if mask_limits_var:
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
    other_schemas: Iterable[xarray_schema.DatasetSchema]
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
