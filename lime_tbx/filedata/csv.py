"""
This module contains the functionality that lets read and write LIME data from/to a CSV file.

It exports the following functions:
    * export_csv - Export a graph to a csv file.
    * export_csv_comparation - Export a channel comparison to a CSV file.
    * export_csv_integrated_irradiance - Export a integrated irradiance table to a CSV file.
    * read_datetimes - Read a time-series CSV file.
"""

"""___Built-In Modules___"""
from typing import Union, List, Iterable
from datetime import datetime, timezone
import csv

"""___Third-Party Modules___"""
import numpy as np
import xarray
import obsarray

"""___NPL Modules___"""
from ..datatypes.datatypes import (
    LimeCoefficients,
    Point,
    PolarizationCoefficients,
    SpectralData,
    SpectralResponseFunction,
    SpectralValidity,
    SurfacePoint,
    CustomPoint,
    LimeException,
)
from ..datatypes import logger
from ..datatypes.datatypes import ReflectanceCoefficients
from lime_tbx.datatypes.templates_digital_effects_table import TEMPLATE_CIMEL

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "15/05/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"

_EXPORT_ERROR_STR = "Error while exporting as CSV. See log for details."
_READ_FILE_ERROR_STR = (
    "There was a problem while loading the file. See log for details."
)
_WARN_OUT_MPA_RANGE = "The LIME can only give a reliable simulation \
for absolute moon phase angles between 2° and 90°"


def _write_point(writer, point: Union[Point, None]):
    if point is not None:
        if isinstance(point, SurfacePoint):
            writer.writerow(["latitude", point.latitude])
            writer.writerow(["longitude", point.longitude])
            writer.writerow(["altitude(m)", point.altitude])
            dt: Union[datetime, List[datetime]] = point.dt
            if isinstance(dt, list):
                str_dt = map(str, dt)
                writer.writerow(["datetimes", *str_dt])
            else:
                writer.writerow(["datetime", str(dt)])
        elif isinstance(point, CustomPoint):
            writer.writerow(["absolute moon phase angle", point.abs_moon_phase_angle])
            writer.writerow(["distance observer moon", point.distance_observer_moon])
            writer.writerow(["distance sun moon", point.distance_sun_moon])
            writer.writerow(["selenographic latitude of observer", point.selen_obs_lat])
            writer.writerow(
                ["selenographic longitude of observer", point.selen_obs_lon]
            )
            writer.writerow(["selenographic longitude of sun", point.selen_sun_lon])
        else:
            writer.writerow(["satellite", point.name])
            dt: Union[datetime, List[datetime]] = point.dt
            if isinstance(dt, list):
                str_dt = map(str, dt)
                writer.writerow(["datetimes", *str_dt])
            else:
                writer.writerow(["datetime", str(dt)])


def export_csv(
    data: Union[SpectralData, List[SpectralData]],
    xlabel: str,
    ylabel: str,
    point: Union[Point, None],
    name: str,
    coeff_version: str,
    inside_mpa_range: Union[bool, List[bool]],
):
    """
    Export the given data to a csv file

    Parameters
    ----------
    data: SpectralData | list of SpectralData
        Data that will be exported
    xlabel: str
        Label of the x_data
    ylabel: str
        Label of the y_data
    point: Point
        Point from which the data is generated. In case it's None, no metadata will be printed.
    name: str
        CSV file path
    coeff_version: str
        Version of the CIMEL coefficients used for calculating the data
    inside_mpa_range: bool | list of bool
        Indication if the point moon phase angle/s were inside the valid LIME range.
    """
    try:
        with open(name, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["LIME2 coefficients version", coeff_version])
            some_out_mpa_range = (
                not inside_mpa_range
                if not isinstance(inside_mpa_range, list)
                else False in inside_mpa_range
            )
            if some_out_mpa_range:
                writer.writerow(["**", _WARN_OUT_MPA_RANGE])
            _write_point(writer, point)
            ylabels = []
            if not isinstance(point, CustomPoint) and point != None:
                dts = point.dt
                if not isinstance(dts, list):
                    dts = [dts]
                    inside_mpa_range = [inside_mpa_range]
                for dt, inside_mpa in zip(dts, inside_mpa_range):
                    warn_out_mpa_range = ""
                    if not inside_mpa:
                        warn_out_mpa_range = " **"
                    ylabels.append(
                        "{} {}{}".format(str(dt), ylabel, warn_out_mpa_range)
                    )
                    ylabels.append(
                        "{} uncertainties{}".format(str(dt), warn_out_mpa_range)
                    )
            else:
                warn_out_mpa_range = ""
                if some_out_mpa_range:
                    warn_out_mpa_range = " **"
                ylabels.append(f"{ylabel}{warn_out_mpa_range}")
            writer.writerow([xlabel, *ylabels])
            if not isinstance(data, list) and not isinstance(data, np.ndarray):
                data = [data]
            x_data = data[0].wlens
            y_data = []
            for i in range(len(x_data)):
                yd = []
                for datum in data:
                    yd.append(datum.data[i])
                    yd.append(datum.uncertainties[i])
                y_data.append(yd)
            for i in range(len(x_data)):
                writer.writerow([x_data[i], *y_data[i]])
    except Exception as e:
        logger.get_logger().exception(e)
        raise Exception(_EXPORT_ERROR_STR)


def export_csv_comparation(
    data: List[SpectralData],
    ylabel: str,
    points: List[SurfacePoint],
    name: str,
    coeff_version: str,
    x_datetime: bool = True,
):
    """
    Export the given data to a csv file

    Parameters
    ----------
    x_data: list of float
        Data from the x axis, which would correspond to the key, of the key-value pair
    y_data: tuple of two list of float
        Data from the y axis, which would correspond to the value, of the key-value pair.
        In the comparation it is the observed irradiance and the simulated one, in that exact order.
    ylabel: str
        Label of the y_data
    points: list of SurfacePoint
        Points from which the data is generated. In case it's None, no metadata will be printed.
    name: str
        CSV file path
    coeff_version: str
        Version of the CIMEL coefficients used for calculating the data
    x_datetime: bool
        True if it used datetimes as the x_axis, False if it used mpa
    """
    x_label = "UTC datetime"
    if not x_datetime:
        x_label = "Moon phase angle (degrees)"
    try:
        with open(name, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["LIME2 coefficients version", coeff_version])
            writer.writerow(
                [
                    x_label,
                    "latitude",
                    "longitude",
                    "altitude(m)",
                    "Observed {}".format(ylabel),
                    "Simulated {}".format(ylabel),
                    "Observation uncertainties",
                    "Simulation uncertainties",
                ]
            )
            x_data = data[0].wlens
            for i in range(len(x_data)):
                pt = points[i]
                if x_datetime:
                    x_val = pt.dt.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    x_val = x_data[i]
                writer.writerow(
                    [
                        x_val,
                        pt.latitude,
                        pt.longitude,
                        pt.altitude,
                        data[0].data[i],
                        data[1].data[i],
                        data[0].uncertainties[i],
                        data[1].uncertainties[i],
                    ]
                )
    except Exception as e:
        logger.get_logger().exception(e)
        raise Exception(_EXPORT_ERROR_STR)


def export_csv_integrated_irradiance(
    srf: SpectralResponseFunction,
    signals: SpectralData,
    name: str,
    point: Point,
    coeff_version: str,
    inside_mpa_range: Union[bool, List[bool]],
):
    """
    Export the given integrated signal data to a csv file

    Parameters
    ----------
    srf: SpectralResponseFunction
        Spectral response function that contains the channels of the integrated signal data.
    signals: list of SpectralData
        List of irradiances of each channel, in order.
    name: str
        CSV file path
    point: Point
        Point from which the data is generated. In case it's None, no metadata will be printed.
    coeff_version: str
        Version of the CIMEL coefficients used for calculating the data
    inside_mpa_range: bool | list of bool
        Indication if the point moon phase angle/s were inside the valid LIME range.
    """
    try:
        with open(name, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["LIME2 coefficients version", coeff_version])
            some_out_mpa_range = (
                not inside_mpa_range
                if not isinstance(inside_mpa_range, list)
                else False in inside_mpa_range
            )
            if some_out_mpa_range:
                writer.writerow(["**", _WARN_OUT_MPA_RANGE])
            _write_point(writer, point)
            writer.writerow(["srf name", srf.name])
            irr_titles = []
            if not isinstance(point, CustomPoint) and point != None:
                dts = point.dt
                if not isinstance(dts, list):
                    dts = [dts]
                    inside_mpa_range = [inside_mpa_range]
                for dt, inside_mpa in zip(dts, inside_mpa_range):
                    warn_out_mpa_range = ""
                    if not inside_mpa:
                        warn_out_mpa_range = " **"
                    irr_titles.append(
                        "{} irradiances (Wm⁻²nm⁻¹){}".format(
                            str(dt), warn_out_mpa_range
                        )
                    )
                    irr_titles.append(
                        "{} uncertainties{}".format(str(dt), warn_out_mpa_range)
                    )
            else:
                irr_titles.append("irradiances (Wm⁻²nm⁻¹)")
                irr_titles.append("uncertainties")
            writer.writerow(["id", "center (nm)", "inside LIME range", *irr_titles])
            for i, ch in enumerate(srf.channels):
                if ch.valid_spectre == SpectralValidity.VALID:
                    validity = "In"
                elif ch.valid_spectre == SpectralValidity.PARTLY_OUT:
                    validity = "Partially"
                else:
                    validity = "Out"
                print_data = []
                for j in range(len(signals.data[i])):
                    print_data.append(signals.data[i][j])
                    print_data.append(signals.uncertainties[i][j])

                writer.writerow([ch.id, ch.center, validity, *print_data])
    except Exception as e:
        logger.get_logger().exception(e)
        raise Exception(_EXPORT_ERROR_STR)


def read_datetimes(path: str) -> List[datetime]:
    """
    Read a time-series CSV file.

    Parameters
    ----------
    path: str
        Path where the file is stored.

    Returns
    -------
    dts: list of datetime
        Datetimes that where stored in that file.
    """
    try:
        with open(path, "r") as file:
            reader = csv.reader(file)
            datetimes = []
            for row in reader:
                irow = map(int, row)
                dt = datetime(*irow, tzinfo=timezone.utc)
                datetimes.append(dt)
            return datetimes
    except Exception as e:
        logger.get_logger().exception(e)
        raise Exception(_READ_FILE_ERROR_STR)


def read_lime_coefficients_from_stream(
    stream: Iterable[str],
) -> LimeCoefficients:
    # define dim_size_dict to specify size of arrays
    dim_sizes = {
        "wavelength": 6,
        "i_coeff": 18,
    }
    reader = csv.reader(stream)
    rows = []
    for row in reader:
        if len(row) > 0 and len(row[0]) > 0 and row[0][0] != "#":
            rows.append(row)
    if len(rows) != 37:
        raise LimeException(
            f"Wrong format in the coefficients update file. There should be 37 uncommented lines, found {len(rows)}."
        )
    wlens = [440, 500, 675, 870, 1020, 1640]
    version_name = rows[0][0]
    data = np.array(rows[1:7]).astype(float)
    u_data = np.array(rows[7:13]).astype(float)
    # create dataset
    ds_cimel: xarray.Dataset = obsarray.create_ds(TEMPLATE_CIMEL, dim_sizes)
    ds_cimel = ds_cimel.assign_coords(wavelength=wlens)
    ds_cimel.coeff.values = data.T
    ds_cimel.u_coeff.values = u_data.T

    rf = ReflectanceCoefficients(ds_cimel)

    p_pos_data = np.array(rows[13:19]).astype(float)
    p_pos_u_data = np.array(rows[19:25]).astype(float)
    p_neg_data = np.array(rows[25:31]).astype(float)
    p_neg_u_data = np.array(rows[31:37]).astype(float)
    pol = PolarizationCoefficients(
        wlens, p_pos_data, p_pos_u_data, p_neg_data, p_neg_u_data
    )
    return LimeCoefficients(rf, pol, version_name)


def read_lime_coefficients(path: str) -> LimeCoefficients:
    """
    Read a Reflectance Coefficients CSV file.

    Parameters
    ----------
    path: str
        Path where the file is stored.

    Returns
    -------
    lc: LimeCoefficients
        LimeCoefficients read.
    """
    try:
        with open(path, "r") as file:
            return read_lime_coefficients_from_stream(file)
    except Exception as e:
        logger.get_logger().exception(e)
        raise Exception(_READ_FILE_ERROR_STR)
