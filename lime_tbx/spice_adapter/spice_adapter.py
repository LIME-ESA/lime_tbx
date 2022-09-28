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
import os

"""___Third-Party Modules___"""
import spicedmoon
import spiceypy as spice
import numpy as np

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
                #custom_kernel_path=kernels_path.custom_kernel_path,
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
            #custom_kernel_path=kernels_path.custom_kernel_path,
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

    @staticmethod
    def _load_kernels(kernels_path: str):
        kernels = [
            "moon_pa_de421_1900-2050.bpc",
            "moon_080317.tf",
            "pck00010.tpc",
            "naif0011.tls",
            "de421.bsp",
            "earth_assoc_itrf93.tf",
            "earth_latest_high_prec.bpc",
            "earth_070425_370426_predict.bpc",
        ]
        for kernel in kernels:
            k_path = os.path.join(kernels_path, kernel)
            spicedmoon.spicedmoon._furnsh_safer(k_path)

    @staticmethod
    def _clear_kernels():
        spice.kclear()

    @staticmethod
    def to_rectangular(
        lat: float, lon: float, alt_meters: float, body: str, kernels_path: str
    ):
        SPICEAdapter._load_kernels(kernels_path)
        _, radios = spice.bodvrd(body, "RADII", 3)
        eq_rad = radios[0]  # Equatorial Radius
        pol_rad = radios[2]  # Polar radius
        flattening = (eq_rad - pol_rad) / eq_rad
        alt_km = alt_meters / 1000
        pos_iau = list(
            map(
                lambda n: n * 1000,
                spice.pgrrec(
                    body,
                    spice.rpd() * lon,
                    spice.rpd() * lat,
                    alt_km,
                    eq_rad,
                    flattening,
                ),
            )
        )
        SPICEAdapter._clear_kernels()
        return pos_iau[0], pos_iau[1], pos_iau[2]  # in meters

    @staticmethod
    def to_planetographic(
        x: float, y: float, z: float, body: str, kernels_path: str
    ):  # in meters
        SPICEAdapter._load_kernels(kernels_path)
        _, radios = spice.bodvrd(body, "RADII", 3)
        eq_rad = radios[0]  # Equatorial Radius
        pol_rad = radios[2]  # Polar radius
        flattening = (eq_rad - pol_rad) / eq_rad
        pos_iau = np.array(list(map(lambda n: n / 1000, [x, y, z])))
        lat, lon, alt_km = spice.recpgr(body, pos_iau, eq_rad, flattening)
        SPICEAdapter._clear_kernels()
        return lat * spice.dpr(), lon * spice.dpr(), alt_km * 1000
