"""
This module contains the functionality that lets read and write LIME data from/to a CSV file.

It exports the following functions:
    * export_csv - Export a graph to a csv file.
    * export_csv_comparison - Export a channel comparison to a CSV file.
    * export_csv_integrated_irradiance - Export a integrated irradiance table to a CSV file.
    * read_datetimes - Read a time-series CSV file.
"""

"""___Built-In Modules___"""
from typing import Union, List
from datetime import datetime, timezone
import csv
import os

"""___Third-Party Modules___"""
import numpy as np
import pandas as pd

"""___NPL Modules___"""
from lime_tbx.datatypes.datatypes import (
    Point,
    SpectralData,
    SpectralResponseFunction,
    SpectralValidity,
    SurfacePoint,
    CustomPoint,
    ComparisonData,
)
from lime_tbx.datatypes import logger
from lime_tbx.datatypes.constants import CompFields

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "15/05/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"

_EXPORT_ERROR_STR = "Error while exporting as CSV. See log for details."
_READ_FILE_DTS_ERROR_STR = "There was a problem while loading datetimes csv file.\n\
Check that every row has the correct format and see log for details."
_WARN_OUT_MPA_RANGE = "The LIME can only give a reliable simulation \
for absolute moon phase angles between 2° and 90°"


def _write_point(writer, point: Union[Point, None]):
    if point is not None:
        if isinstance(point, SurfacePoint):
            writer.writerow(["latitude (deg)", point.latitude])
            writer.writerow(["longitude (deg)", point.longitude])
            writer.writerow(["altitude (m)", point.altitude])
            dt: Union[datetime, List[datetime]] = point.dt
            if isinstance(dt, list):
                str_dt = map(
                    lambda dti: dti.isoformat(sep=" ", timespec="milliseconds"), dt
                )
                writer.writerow(["datetimes", *str_dt])
            else:
                writer.writerow(["datetime", str(dt)])
        elif isinstance(point, CustomPoint):
            writer.writerow(["moon phase angle (deg)", point.moon_phase_angle])
            writer.writerow(
                ["distance observer moon (km)", point.distance_observer_moon]
            )
            writer.writerow(["distance sun moon (AU)", point.distance_sun_moon])
            writer.writerow(
                ["selenographic latitude of observer (deg)", point.selen_obs_lat]
            )
            writer.writerow(
                ["selenographic longitude of observer (deg)", point.selen_obs_lon]
            )
            writer.writerow(
                ["selenographic longitude of sun (rad)", point.selen_sun_lon]
            )
        else:
            writer.writerow(["satellite", point.name])
            dt: Union[datetime, List[datetime]] = point.dt
            if isinstance(dt, list):
                str_dt = map(str, dt)
                writer.writerow(["datetimes", *str_dt])
            else:
                writer.writerow(["datetime", str(dt)])


def export_csv_srf(
    data: Union[SpectralData, List[SpectralData]],
    ch_names: List[str],
    xlabel: str,
    ylabel: str,
    name: str,
):
    """
    Export the given data to a csv file

    Parameters
    ----------
    data: SpectralData | list of SpectralData
        Data that will be exported
    ch_names: list of str
        Ordered list of names of the srf channels
    xlabel: str
        Label of the x_data
    ylabel: str
        Label of the y_data
    name: str
        CSV file path
    """
    try:
        with open(name, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            ylabels = []
            ylabels.append(f"{ylabel}")
            writer.writerow(["Channel ID", xlabel, *ylabels])
            if not isinstance(data, list) and not isinstance(data, np.ndarray):
                data = [data]
            for i, spd in enumerate(data):
                wlens = spd.wlens
                vals = spd.data
                for wlen, val in zip(wlens, vals):
                    writer.writerow([ch_names[i], wlen, val])
    except Exception as e:
        logger.get_logger().exception(e)
        raise Exception(_EXPORT_ERROR_STR)


def export_csv_simulation(
    data: Union[SpectralData, List[SpectralData]],
    xlabel: str,
    ylabel: str,
    point: Union[Point, None],
    name: str,
    coeff_version: str,
    inside_mpa_range: Union[bool, List[bool]],
    interp_spectrum_name: str,
    skip_uncs: bool,
    cimel_data: Union[SpectralData, List[SpectralData]],
    mpa: Union[float, None],
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
    interp_spectrum_name: str
        Name of the spectrum used for interpolation.
    """
    try:
        with open(name, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["LIME coefficients version", coeff_version])
            writer.writerow(["Interpolation spectrum", interp_spectrum_name])
            if mpa is not None and not isinstance(point, CustomPoint):
                writer.writerow(["moon phase angle (deg)", mpa])
            some_out_mpa_range = (
                not inside_mpa_range
                if not isinstance(inside_mpa_range, list)
                else False in inside_mpa_range
            )
            if some_out_mpa_range:
                writer.writerow(["**", _WARN_OUT_MPA_RANGE])
            _write_point(writer, point)
            ylabels = []
            cimel_ylabels = []
            if not isinstance(point, CustomPoint) and point != None:
                dts = point.dt
                if not isinstance(dts, list):
                    dts = [dts]
                    inside_mpa_range = [inside_mpa_range]
                for dt, inside_mpa in zip(dts, inside_mpa_range):
                    warn_out_mpa_range = ""
                    if not inside_mpa:
                        warn_out_mpa_range = " **"
                    ylab = f"{dt.isoformat(sep=' ', timespec='milliseconds')} {ylabel}{warn_out_mpa_range}"
                    ylabels.append(ylab)
                    cimel_ylabels.append(ylab)
                    if not skip_uncs:
                        halfy2 = f"{dt.isoformat(sep=' ', timespec='milliseconds')} uncertainties"
                        ylabels.append(f"{halfy2}{warn_out_mpa_range}")
                        cimel_ylabels.append(f"{halfy2} (k=2){warn_out_mpa_range}")
            else:
                warn_out_mpa_range = ""
                if some_out_mpa_range:
                    warn_out_mpa_range = " **"
                ylab = f"{ylabel}{warn_out_mpa_range}"
                ylabels.append(ylab)
                cimel_ylabels.append(ylab)
                if not skip_uncs:
                    ylabels.append(f"uncertainties{warn_out_mpa_range}")
                    cimel_ylabels.append(f"uncertainties (k=2){warn_out_mpa_range}")
            if cimel_data:
                writer.writerow([f"CIMEL {xlabel}", *cimel_ylabels])
                if not isinstance(cimel_data, list) and not isinstance(
                    cimel_data, np.ndarray
                ):
                    cimel_data = [cimel_data]
                x_data = cimel_data[0].wlens
                for i, cimel_w in enumerate(x_data):
                    y_data = []
                    for cdata in cimel_data:
                        y_data.append(cdata.data[i])
                        if not skip_uncs:
                            y_data.append(cdata.uncertainties[i])
                    writer.writerow([cimel_w, *y_data])
            writer.writerow([xlabel, *ylabels])
            if not isinstance(data, list) and not isinstance(data, np.ndarray):
                data = [data]
            x_data = data[0].wlens
            y_data = []
            for i in range(len(x_data)):
                yd = []
                for datum in data:
                    yd.append(datum.data[i])
                    if not skip_uncs:
                        yd.append(datum.uncertainties[i])
                y_data.append(yd)
            for i in range(len(x_data)):
                writer.writerow([x_data[i], *y_data[i]])
    except Exception as e:
        logger.get_logger().exception(e)
        raise Exception(_EXPORT_ERROR_STR)


def export_csv_comparison(
    data: ComparisonData,
    xlabel: str,
    ylabels: List[str],
    name: str,
    coeff_version: str,
    interp_spectrum_name: str,
    skip_uncs: bool,
    chosen_diffs: CompFields,
):
    """
    Export the given data to a csv file

    Parameters
    ----------
    data: ComparisonData
        Comparison data.
    xlabel: str
        Label of the x axis data
    ylabels: list of str
        Labels of the y axis data
    name: str
        CSV file path
    coeff_version: str
        Version of the CIMEL coefficients used for calculating the data
    interp_spectrum_name: str
        Name of the spectrum used for interpolation.
    chosen_diffs: CompFields
        Type of difference chosen to be shown for this comparison.
    """
    ampa_valid_range = data.ampa_valid_range
    try:
        with open(name, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["LIME coefficients version", coeff_version])
            writer.writerow(["Interpolation spectrum", interp_spectrum_name])
            writer.writerow(
                [
                    "MRD (Mean Relative Difference %)",
                    data.mean_relative_difference,
                ]
            )
            writer.writerow(
                [
                    "STD-RD (Standard deviation of Relative Difference %)",
                    data.standard_deviation_mrd,
                ]
            )
            writer.writerow(
                [
                    "MARD (Mean of the Absolutes of the Relative Differences %)",
                    data.mean_absolute_relative_difference,
                ]
            )
            writer.writerow(
                [
                    "MPD (Mean Percentage Difference %)",
                    data.mean_perc_difference,
                ]
            )
            if False in ampa_valid_range:
                writer.writerow(["**", _WARN_OUT_MPA_RANGE])
            if data.points:  # When exporting by wavelength using means, points are None
                is_surface = isinstance(data.points[0], SurfacePoint)
                if is_surface:
                    header_coords = ["latitude", "longitude", "altitude(m)"]
                else:  # CustomPoint
                    header_coords = [
                        "moon phase angle (deg)",
                        "selenographic latitude (deg)",
                        "selenographic longitude (deg)",
                        "solar selenographic longitude (rad)",
                        "distance sun moon (AU)",
                        "distance observer moon (km)",
                    ]
            else:
                header_coords = []
            diffdata, difflabel = data.get_diffs_and_label(chosen_diffs)
            header = [xlabel] + header_coords + [ylabels[0], ylabels[1]]
            if difflabel:
                header += [difflabel]
            if not skip_uncs:
                header += [
                    f"{ylabels[0]} uncertainties",
                    f"{ylabels[1]} uncertainties",
                ]
                if difflabel:
                    header += [f"{difflabel} uncertainties"]
            writer.writerow(header)
            xdata = data.observed_signal.wlens
            if isinstance(xdata[0], datetime):
                xdata = [x.isoformat(sep=" ", timespec="milliseconds") for x in xdata]
            for i, x_val in enumerate(xdata):
                warn_out_mpa_range = ""
                if not ampa_valid_range[i]:
                    warn_out_mpa_range = " **"
                x_val = f"{x_val}{warn_out_mpa_range}"
                pt_datarow = []
                if data.points:
                    pt = data.points[i]
                    if is_surface:
                        pt_datarow = [
                            pt.latitude,
                            pt.longitude,
                            pt.altitude,
                        ]
                    else:
                        pt_datarow = [
                            pt.moon_phase_angle,
                            pt.selen_obs_lat,
                            pt.selen_obs_lon,
                            pt.selen_sun_lon,
                            pt.distance_sun_moon,
                            pt.distance_observer_moon,
                        ]
                datarow = (
                    [x_val]
                    + pt_datarow
                    + [data.observed_signal.data[i], data.simulated_signal.data[i]]
                )
                if diffdata:
                    datarow.append(diffdata.data[i])
                if not skip_uncs:
                    datarow += [
                        data.observed_signal.uncertainties[i],
                        data.simulated_signal.uncertainties[i],
                    ]
                    if diffdata:
                        datarow.append(diffdata.uncertainties[i])
                writer.writerow(datarow)
    except Exception as e:
        logger.get_logger().exception(e)
        raise Exception(_EXPORT_ERROR_STR)


def export_csv_comparison_bywlen(
    data: List[ComparisonData],
    wlens: List[float],
    xlabel: str,
    ylabels: List[str],
    name: str,
    coeff_version: str,
    interp_spectrum_name: str,
    skip_uncs: bool,
    chosen_diffs: CompFields,
):
    wlens = [w for w, d in zip(wlens, data) if d is not None]
    data = [d for d in data if d is not None]
    if data:
        try:
            with open(name, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                mpd = np.ma.masked_invalid(
                    [sd.mean_perc_difference for sd in data]
                ).mean()
                mrd = np.ma.masked_invalid(
                    [sd.mean_relative_difference for sd in data]
                ).mean()
                stdrd = np.ma.masked_invalid(
                    [sd.standard_deviation_mrd for sd in data]
                ).mean()
                mard = np.ma.masked_invalid(
                    [sd.mean_absolute_relative_difference for sd in data]
                ).mean()
                writer.writerow(["LIME coefficients version", coeff_version])
                writer.writerow(["Interpolation spectrum", interp_spectrum_name])
                writer.writerow(["MRD (Mean Relative Difference %)", mrd])
                writer.writerow(
                    ["STD-RD (Standard deviation of Relative Difference %)", stdrd]
                )
                writer.writerow(
                    ["MARD (Mean of the Absolutes of the Relative Differences %)", mard]
                )
                writer.writerow(["MPD (Mean Percentage Difference %)", mpd])
                if False in [not np.all(d.ampa_valid_range) for d in data]:
                    writer.writerow(["**", _WARN_OUT_MPA_RANGE])
            # Now with pandas the rest
            _, diff_name = data[0].get_diffs_and_label(chosen_diffs)
            diff_name = [diff_name] if diff_name is not None else []
            arr_names = ylabels + diff_name
            stat_names = ["Q1", "Median", "Q3", "Mean"]
            stat_names = [f"{arn} {st}" for arn in arr_names for st in stat_names]
            vals = []
            for d, w in zip(data, wlens):
                diff_arr, _ = d.get_diffs_and_label(chosen_diffs)
                diff_arr = [diff_arr.data] if diff_arr is not None else []
                arrs = [d.observed_signal.data, d.simulated_signal.data] + diff_arr
                qs = np.quantile(arrs, [0.25, 0.5, 0.75], axis=1)
                m = np.mean(arrs, axis=1)
                vals.append(
                    np.concatenate([[w], np.concatenate([qs, [m]]).T.flatten()])
                )
            columns = [xlabel] + stat_names
            df = pd.DataFrame(vals, columns=columns)
            df.set_index(xlabel).to_csv(name, mode="a")
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
    interp_spectrum_name: str,
    skip_uncs: bool,
    mpa: Union[float, None],
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
    interp_spectrum_name: str
        Name of the spectrum used for interpolation.
    """
    try:
        with open(name, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["LIME coefficients version", coeff_version])
            writer.writerow(["Interpolation spectrum", interp_spectrum_name])
            if mpa is not None and not isinstance(point, CustomPoint):
                writer.writerow(["moon phase angle (deg)", mpa])
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
                            dt.isoformat(sep=" ", timespec="milliseconds"),
                            warn_out_mpa_range,
                        )
                    )
                    if not skip_uncs:
                        irr_titles.append(
                            f"{dt.isoformat(sep=' ', timespec='milliseconds')} uncertainties{warn_out_mpa_range}"
                        )
            else:
                irr_titles.append("irradiances (Wm⁻²nm⁻¹)")
                if not skip_uncs:
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
                    if not skip_uncs:
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
                if not row:
                    continue
                irow = map(int, row)
                dt = datetime(*irow, tzinfo=timezone.utc)
                datetimes.append(dt)
            return datetimes
    except Exception as e:
        logger.get_logger().exception(e)
        raise Exception(_READ_FILE_DTS_ERROR_STR)
