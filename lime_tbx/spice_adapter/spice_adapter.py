"""
This module acts as an interface/adapter with NASA's SPICE software.

It exports the following classes:
    * ISPICEAdapter - Interface that contains the methods of this module.
    * SPICEAdapter - Class that implements the methods exported by this module.
"""

"""___Built-In Modules___"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Union

"""___Third-Party Modules___"""
import spicedmoon

"""___LIME_TBX Modules___"""
from ..datatypes import datatypes

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "01/03/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


class ISPICEAdapter(ABC):
    @staticmethod
    @abstractmethod
    def get_moon_data_from_earth(
        latitude: float,
        longitude: float,
        altitude: float,
        dt: Union[datetime, List[datetime]],
        kernels_path: datatypes.KernelsPath,
    ) -> Union[datatypes.MoonData, List[datatypes.MoonData]]:
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
        kernels_path: datatypes.KernelsPath,
    ) -> Union[datatypes.MoonData, List[datatypes.MoonData]]:
        if isinstance(dt, list):
            mds = spicedmoon.get_moon_datas(
                latitude,
                longitude,
                altitude,
                dt,
                kernels_path.main_kernels_path,
                custom_kernel_path=kernels_path.custom_kernel_path,
            )
            mds2 = []
            for md in mds:
                md2 = datatypes.MoonData(
                    md.dist_sun_moon_au,
                    md.dist_obs_moon,
                    md.lon_sun_rad,
                    md.lat_obs,
                    md.lon_obs,
                    abs(md.mpa_deg),
                    md.mpa_deg,
                )
                mds2.append(md2)
            return mds2
        md = spicedmoon.get_moon_datas(
            latitude,
            longitude,
            altitude,
            [dt],
            kernels_path.main_kernels_path,
            custom_kernel_path=kernels_path.custom_kernel_path,
        )[0]
        return datatypes.MoonData(
            md.dist_sun_moon_au,
            md.dist_obs_moon,
            md.lon_sun_rad,
            md.lat_obs,
            md.lon_obs,
            abs(md.mpa_deg),
            md.mpa_deg,
        )
