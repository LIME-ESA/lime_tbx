"""
This module acts as an interface/adapter with NASA's SPICE software.

It exports the following classes:
    * ISPICEAdapter - Interface that contains the methods of this module.
    * SPICEAdapter - Class that implements the methods exported by this module.
"""

"""___Built-In Modules___"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Union, Tuple, Callable
import os

"""___Third-Party Modules___"""
import spicedmoon
import spiceypy as spice
import numpy as np

"""___LIME_TBX Modules___"""
from lime_tbx.datatypes import datatypes

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "01/03/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


class ISPICEAdapter(ABC):
    """Interface that contains the methods of this module.

    It exports the following functions:
        * get_moon_data_from_earth() - Calculate lunar data for a position on earth
            surface at a concrete datetime.
        * get_moon_data_from_moon() - Calculate lunar data for a position on Moon's
            surface at a concrete datetime.
        * to_rectangular() - Transforms planetographic coordinates to rectangular coordinates.
        * to_planetographic() - Transforms rectangular coordinates to planetographic coordinates.
        * to_planetographic_multiple() - Transforms multiple rectangular coordinates
            to planetographic coordinates.
    """

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

    @staticmethod
    def get_moon_datas_from_earth_rectangular_multiple(
        xyzs: List[Tuple[float, float, float]],
        dts: List[datetime],
        kernels_path: datatypes.KernelsPath,
    ) -> List[datatypes.MoonData]:
        """
        Calculate lunar data for the given rectangular coordinates.
        Faster execution for some cases

        Returns:
        --------
        list of MoonData
            Lunar data for the given parameters.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_moon_data_from_moon(
        latitude: float,
        longitude: float,
        altitude: float,
        dt: Union[datetime, List[datetime]],
        kernels_path: datatypes.KernelsPath,
    ) -> Union[datatypes.MoonData, List[datatypes.MoonData]]:
        """
        Calculate lunar data for a position on moon surface at a concrete datetime.

        Parameters
        ----------
        latitude: float
            Selenographic latitude in decimal degrees.
        longitude: float
            Selenographic longitude in decimal degrees.
        altitude: float
            Altitude in meters.
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

    @staticmethod
    @abstractmethod
    def to_rectangular(
        lat: float, lon: float, alt_meters: float, body: str, kernels_path: str
    ) -> Tuple[float, float, float]:
        """Transforms planetographic coordinates to rectangular coordinates.

        Parameters
        ----------
        lat: float
            Latitude in degrees.
        lon: float
            Longitude in degrees.
        alt_meters: float
            Altitude in meters.
        body: str
            Name of the body. For example 'MOON' or 'EARTH'.
        kernels_path: str
            Path to the directory of the main kernels.

        Returns
        -------
        xyz: tuple of 3 floats
            Rectangular coordinates in meters
        """
        pass

    @staticmethod
    @abstractmethod
    def to_planetographic(
        x: float, y: float, z: float, body: str, kernels_path: str
    ) -> Tuple[float, float, float]:
        """Transforms rectangular coordinates to planetographic coordinates.

        Parameters
        ----------
        x: float
            x coordinate in meters.
        y: float
            y coordinate in meters.
        z: float
            z coordinate in meters.
        body: str
            Name of the body. For example 'MOON' or 'EARTH'.
        kernels_path: str
            Path to the directory of the main kernels.

        Returns
        -------
        lat: float
            Latitude in degrees.
        lon: float
            Longitude in degrees.
        alt: float
            Altitude in meters.
        """

    @staticmethod
    @abstractmethod
    def to_planetographic_multiple(
        xyz_list: List[Tuple[float]], body: str, kernels_path: str
    ) -> List[Tuple[float, float, float]]:
        """Transforms multiple rectangular coordinates to planetographic coordinates.

        Parameters
        ----------
        xyz_list: list of tuples of 3 floats
            List of xyz coordinates in meters.
        body: str
            Name of the body. For example 'MOON' or 'EARTH'.
        kernels_path: str
            Path to the directory of the main kernels.

        Returns
        -------
        llhs: list of tuples of 3 floats
            List of planetographic coordinates, in lat (deg), lon (deg), alt (meters) form.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_solar_moon_datas(
        dts: List[datetime],
        kernels_path: str,
    ):
        """
        Obtain solar selenographic coordinates of at multiple times.

        times : list of str | list of datetime
            Times at which the lunar data will be calculated.
            If they are str, they must be in a valid UTC format allowed by SPICE, such as
            %Y-%m-%d %H:%M:%S.
            If they are datetimes they must be timezone aware, or they will be understood
            as computer local time.
        kernels_path : str
            Path where the SPICE kernels are stored
        """
        pass


class SPICEAdapter(ISPICEAdapter):
    """Class that implements the methods of this module.

    It exports the following functions:
        * get_moon_data_from_earth() - Calculate lunar data for a position on earth
            surface at a concrete datetime.
        * get_moon_data_from_moon() - Calculate lunar data for a position on moon's
            surface at a concrete datetime.
        * to_rectangular() - Transforms planetographic coordinates to rectangular coordinates.
        * to_planetographic() - Transforms rectangular coordinates to planetographic coordinates.
        * to_planetographic_multiple() - Transforms multiple rectangular coordinates
            to planetographic coordinates.
    """

    @staticmethod
    def _get_moon_data_from_callback(
        latitude: float,
        longitude: float,
        altitude: float,
        dt: Union[datetime, List[datetime]],
        kernels_path: datatypes.KernelsPath,
        source_frame: str,
        target_frame: str,
        callback: Callable,
    ) -> Union[datatypes.MoonData, List[datatypes.MoonData]]:
        was_list = True
        if not isinstance(dt, list):
            was_list = False
            dt = [dt]
        mds = callback(
            latitude,
            longitude,
            altitude,
            dt,
            kernels_path.main_kernels_path,
            True,
            target_frame,
            custom_kernel_path=kernels_path.custom_kernel_path,
            ignore_bodvrd=False,
            source_frame=source_frame,
            target_frame=target_frame,
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
        if not was_list:
            mds2 = mds2[0]
        return mds2

    @staticmethod
    def get_moon_data_from_earth(
        latitude: float,
        longitude: float,
        altitude: float,
        dt: Union[datetime, List[datetime]],
        kernels_path: datatypes.KernelsPath,
    ) -> Union[datatypes.MoonData, List[datatypes.MoonData]]:
        return SPICEAdapter._get_moon_data_from_callback(
            latitude,
            longitude,
            altitude,
            dt,
            kernels_path,
            "J2000",
            "MOON_ME",
            spicedmoon.get_moon_datas,
        )

    @staticmethod
    def get_moon_datas_from_earth_rectangular_multiple(
        xyzs: List[Tuple[float, float, float]],
        dts: List[datetime],
        kernels_path: datatypes.KernelsPath,
    ) -> List[datatypes.MoonData]:
        xyzs = [(x / 1000, y / 1000, z / 1000) for x, y, z in xyzs]
        times = list(map(lambda dt: dt.strftime("%Y-%m-%d %H:%M:%S"), dts))
        mds = spicedmoon.spicedmoon.get_moon_datas_xyzs_no_zenith_azimuth(
            xyzs, times, kernels_path.main_kernels_path
        )
        return [
            datatypes.MoonData(
                md.dist_sun_moon_au,
                md.dist_obs_moon,
                md.lon_sun_rad,
                md.lat_obs,
                md.lon_obs,
                abs(md.mpa_deg),
                md.mpa_deg,
            )
            for md in mds
        ]

    @staticmethod
    def get_moon_data_from_moon(
        latitude: float,
        longitude: float,
        altitude: float,
        dt: Union[datetime, List[datetime]],
        kernels_path: datatypes.KernelsPath,
    ) -> Union[datatypes.MoonData, List[datatypes.MoonData]]:
        return SPICEAdapter._get_moon_data_from_callback(
            latitude,
            longitude,
            altitude,
            dt,
            kernels_path,
            "MOON_ME",
            "MOON_ME",
            spicedmoon.get_moon_datas_from_moon,
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
                spice.georec(
                    # spice.pgrrec(
                    #    body,
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
        lon, lat, alt_km = spice.recgeo(
            pos_iau, eq_rad, flattening
        )  # Changing this because it's the funtion that Stefan uses.
        # lon, lat, alt_km = spice.recpgr(body, pos_iau, eq_rad, flattening)
        SPICEAdapter._clear_kernels()
        lat = lat * spice.dpr()
        lon = lon * spice.dpr()
        alt = alt_km * 1000
        while lon < -180:
            lon += 360
        while lon > 180:
            lon -= 360
        return lat, lon, alt

    @staticmethod
    def to_planetographic_multiple(
        xyz_list: List[Tuple[float]], body: str, kernels_path: str
    ):  # in meters
        SPICEAdapter._load_kernels(kernels_path)
        _, radios = spice.bodvrd(body, "RADII", 3)
        eq_rad = radios[0]  # Equatorial Radius
        pol_rad = radios[2]  # Polar radius
        flattening = (eq_rad - pol_rad) / eq_rad
        llh_list = []  # alt km
        for xyz in xyz_list:
            pos_iau = np.array(list(map(lambda n: n / 1000, xyz)))
            llh = spice.recgeo(pos_iau, eq_rad, flattening)
            # llh = spice.recpgr(body, pos_iau, eq_rad, flattening)
            llh_list.append(llh)
        SPICEAdapter._clear_kernels()
        for i, llh in enumerate(llh_list):
            lat = llh[1] * spice.dpr()
            lon = llh[0] * spice.dpr()
            alt = llh[2] * 1000
            while lon < -180:
                lon += 360
            while lon > 180:
                lon -= 360
            llh_list[i] = (lat, lon, alt)
        return llh_list

    @staticmethod
    def get_solar_moon_datas(
        dts: List[datetime],
        kernels_path: str,
    ):
        return spicedmoon.get_sun_moon_datas(dts, kernels_path)
