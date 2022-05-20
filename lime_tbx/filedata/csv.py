"""describe class"""

"""___Built-In Modules___"""
from typing import Union, List
from datetime import datetime
import csv

"""___Third-Party Modules___"""
# import here

"""___NPL Modules___"""
from ..datatypes.datatypes import (
    SurfacePoint,
    CustomPoint,
)

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "15/05/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


def export_csv(
    x_data: List[float],
    y_data: List[float],
    xlabel: str,
    ylabel: str,
    point: Union[SurfacePoint, CustomPoint],
    name: str,
):
    """
    Export the given data to a csv file

    Parameters
    ----------
    x_data: list of float
        Data from the x axis, which would correspond to the key, of the key-value pair
    y_data: list of float
        Data from the y axis, which would correspond to the value, of the key-value pair
    xlabel: str
        Label of the x_data
    ylabel: str
        Label of the y_data
    point: SurfacePoint | CustomPoint
        Point from which the data is generated
    name: str
        CSV file path
    """
    with open(name, "w") as file:
        writer = csv.writer(file)
        if isinstance(point, SurfacePoint):
            writer.writerow(["latitude", point.latitude])
            writer.writerow(["longitude", point.longitude])
            writer.writerow(["altitude(m)", point.altitude])
            dt: datetime = point.dt  # dt cant be a list of datetime in this function
            writer.writerow(["datetime", str(dt)])
        else:
            writer.writerow(["absolute moon phase angle", point.abs_moon_phase_angle])
            writer.writerow(["distance observer moon", point.distance_observer_moon])
            writer.writerow(["distance sun moon", point.distance_sun_moon])
            writer.writerow(["selenographic latitude of observer", point.selen_obs_lat])
            writer.writerow(
                ["selenographic longitude of observer", point.selen_obs_lon]
            )
            writer.writerow(["selenographic longitude of sun", point.selen_sun_lon])
        writer.writerow([xlabel, ylabel])
        for i in range(len(x_data)):
            writer.writerow([x_data[i], y_data[i]])
