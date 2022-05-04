"""describe class"""

"""___Built-In Modules___"""
from abc import ABC, abstractmethod
from datetime import datetime

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
        dt: datetime,
        kernels_path: str,
    ) -> spicedmoon.MoonData:
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
        dt: datetime
            Time at which the lunar data will be calculated.
        kernels_path: str
            Path where the needed SPICE kernels are located.
            The user must have write access to that directory.

        Returns
        -------
        md: MoonData
            Lunar data for the given parameters.
        """
        pass


class SPICEAdapter(ISPICEAdapter):
    @staticmethod
    def get_moon_data_from_earth(
        latitude: float,
        longitude: float,
        altitude: float,
        dt: datetime,
        kernels_path: str,
    ) -> spicedmoon.MoonData:
        return spicedmoon.get_moon_datas(
            latitude, longitude, altitude, [dt], kernels_path
        )[0]
