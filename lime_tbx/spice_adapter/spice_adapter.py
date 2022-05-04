"""describe class"""

"""___Built-In Modules___"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Union

"""___Third-Party Modules___"""
import spicedmoon

"""___NPL Modules___"""
# import here

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2022"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


class ISPICEAdapter(ABC):
    @staticmethod
    @abstractmethod
    def get_moon_data_from_earth(
        latitude: float,
        longitude: float,
        altitude: float,
        dt: Union[datetime, List[datetime]],
        kernels_path: str,
    ) -> Union[spicedmoon.MoonData, List[spicedmoon.MoonData]]:
        """
        Calculate lunar data for a position on earth surface at a concrete datetime.

        Parameters
        ----------
        latitude: float
            Geographic latitude in decimal degrees.
        longitude: float
            Geographic longitude in decimal degrees.
        altitude: float
            Altitude over the sea level in meters.
        dt: datetime | list of datetimes
            Time or time series at which the lunar data will be calculated.
        kernels_path: str
            Path where the needed SPICE kernels are located.
            The user must have write access to that directory.

        Returns
        -------
        md: MoonData | list of MoonData
            Lunar data for the given parameters. If the parameter dt was a list,
            this will be a list. Otherwise not.
        """
        pass


class SPICEAdapter(ISPICEAdapter):
    @staticmethod
    def get_moon_data_from_earth(
        latitude: float,
        longitude: float,
        altitude: float,
        dt: Union[datetime, List[datetime]],
        kernels_path: str,
    ) -> Union[spicedmoon.MoonData, List[spicedmoon.MoonData]]:
        if isinstance(dt, list):
            return spicedmoon.get_moon_datas(
                latitude, longitude, altitude, dt, kernels_path
            )
        return spicedmoon.get_moon_datas(
            latitude, longitude, altitude, [dt], kernels_path
        )[0]
