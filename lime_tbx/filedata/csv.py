"""
This module contains the functionality that lets read and write LIME data from/to a CSV file.

It exports the following functions:
    * export_csv - Export a graph to a csv file.
    * export_csv_comparation - Export a channel comparison to a CSV file.
    * export_csv_integrated_irradiance - Export a integrated irradiance table to a CSV file.
    * read_datetimes - Read a time-series CSV file.
"""

"""___Built-In Modules___"""
from typing import Union, List, Tuple
from datetime import datetime
import csv

"""___Third-Party Modules___"""
import numpy as np

"""___NPL Modules___"""
from ..datatypes.datatypes import (
    Point,
    SatellitePoint,
    SpectralData,
    SpectralResponseFunction,
    SpectralValidity,
    SurfacePoint,
    CustomPoint,
)

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "15/05/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


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
    """
    with open(name, "w") as file:
        writer = csv.writer(file)
        _write_point(writer, point)
        ylabels = []
        if not isinstance(point, CustomPoint) and point != None:
            dts = point.dt
            if not isinstance(dts, list):
                dts = [dts]
            for dt in dts:
                ylabels.append("{} {}".format(str(dt), ylabel))
                ylabels.append("{} uncertainties".format(str(dt)))
        else:
            ylabels.append(ylabel)
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


def export_csv_comparation(
    data: Union[SpectralData, List[SpectralData]],
    xlabel: str,
    ylabel: str,
    points: List[SurfacePoint],
    name: str,
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
    xlabel: str
        Label of the x_data
    ylabel: str
        Label of the y_data
    points: list of SurfacePoint
        Points from which the data is generated. In case it's None, no metadata will be printed.
    name: str
        CSV file path
    """

    with open(name, "w") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "UTC datetime",
                "latitude",
                "longitude",
                "altitude(m)",
                "Observed {}".format(ylabel),
                "Simulated {}".format(ylabel),
                "Uncertainties",
            ]
        )
        x_data = data[0].wlens
        for i in range(len(x_data)):
            pt = points[i]
            writer.writerow(
                [
                    pt.dt.strftime("%Y-%m-%d %H:%M:%S"),
                    pt.latitude,
                    pt.longitude,
                    pt.altitude,
                    data[0].data[i],
                    data[1].data[i],
                    data[1].uncertainties[i],
                ]
            )


def export_csv_integrated_irradiance(
    srf: SpectralResponseFunction,
    signals: SpectralData,
    name: str,
    point: Point,
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
    """
    with open(name, "w") as file:
        writer = csv.writer(file)
        _write_point(writer, point)
        writer.writerow(["srf name", srf.name])
        irr_titles = []
        if not isinstance(point, CustomPoint) and point != None:
            dts = point.dt
            if not isinstance(dts, list):
                dts = [dts]
            for dt in dts:
                irr_titles.append("{} irradiances (Wm⁻²nm⁻¹)".format(str(dt)))
                irr_titles.append("{} uncertainties".format(str(dt)))
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
    with open(path, "r") as file:
        reader = csv.reader(file)
        datetimes = []
        for row in reader:
            irow = map(int, row)
            dt = datetime(*irow)
            datetimes.append(dt)
        return datetimes
