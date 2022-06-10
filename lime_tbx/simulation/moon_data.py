"""describe class"""

"""___Built-In Modules___"""
from abc import ABC, abstractmethod
from typing import List, Union

"""___Third-Party Modules___"""
# import here
import punpy

"""___NPL Modules___"""
from lime_tbx.datatypes.datatypes import (
    IrradianceCoefficients,
    MoonData,
    PolarizationCoefficients,
    SpectralResponseFunction,
    SurfacePoint,
    CustomPoint,
    SatellitePoint,
    SpectralData,
    CimelCoef
)

from lime_tbx.spice_adapter.spice_adapter import SPICEAdapter
from lime_tbx.eocfi_adapter.eocfi_adapter import EOCFIConverter


"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2022"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


class MoonDataFactory():
    """
        Class for running the main lime-tbx functionality
        """

    @staticmethod
    def get_md(point: Union[SurfacePoint, CustomPoint, SatellitePoint], eocfi_path: str,
               kernels_path: str,) -> MoonData:
        if isinstance(point, SurfacePoint):
            md=MoonDataFactory.get_md_from_surface(point,kernels_path)

        elif isinstance(point, CustomPoint):
            md=MoonDataFactory.get_md_from_custom(point)
        else:
            md= MoonDataFactory.get_md_from_satellite(
                point, eocfi_path, kernels_path
            )
        return md

    @staticmethod
    def get_md_from_surface(
        sp: SurfacePoint,
        kernels_path: str,
    ) -> MoonData:
        md = SPICEAdapter().get_moon_data_from_earth(sp.latitude,sp.longitude,
            sp.altitude,sp.dt,kernels_path)
        return md

    @staticmethod
    def get_md_from_custom(
        cp: CustomPoint,
    ) -> MoonData:
        md = MoonData(
            cp.distance_sun_moon,
            cp.distance_observer_moon,
            cp.selen_sun_lon,
            cp.selen_obs_lat,
            cp.selen_obs_lon,
            cp.abs_moon_phase_angle,
            cp.moon_phase_angle,
        )
        return md

    @staticmethod
    def get_md_from_satellite(
            sp: SatellitePoint,
            eocfi_path: str,
            kernels_path: str,)-> MoonData:

        eocfi = EOCFIConverter(eocfi_path)
        dts = sp.dt
        if not isinstance(dts, list):
            dts = [dts]

        mds=[]
        for dt in dts:
            lat, lon, height = eocfi.get_satellite_position(sp.name, dt)
            srp = SurfacePoint(lat, lon, height, dt)
            mds.append(MoonDataFactory.get_md_from_surface(srp, kernels_path))

        return mds