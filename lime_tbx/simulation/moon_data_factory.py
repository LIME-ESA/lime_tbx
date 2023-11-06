"""Module containing the factory class for MoonData."""

"""___Built-In Modules___"""
from typing import List, Union, Tuple

"""___Third-Party Modules___"""
# import here

"""___LIME_TBX Modules___"""
from lime_tbx.datatypes.datatypes import (
    MoonData,
    Point,
    SurfacePoint,
    CustomPoint,
    SatellitePoint,
    KernelsPath,
)
from lime_tbx.spice_adapter.spice_adapter import SPICEAdapter
from lime_tbx.eocfi_adapter.eocfi_adapter import EOCFIConverter, IEOCFIConverter


"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2022"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


class MoonDataFactory:
    """
    Class for creating MoonData objects from points.
    """

    @staticmethod
    def get_md(
        point: Point,
        eocfi_path: str,
        kernels_path: KernelsPath,
    ) -> Union[MoonData, List[MoonData]]:
        """
        Create a MoonData from a generic point, whatever subclass it is.

        Parameters
        ----------
        point: Point
            Point from which to create the MoonData.
        eocfi_path: str
            Path to the folder with the needed eocfi data files.
        kernels_path: KernelsPath
            Path to the folder with the needed SPICE kernel files.

        Returns
        -------
        md: MoonData | list of MoonData
            MoonData generated from the given data. If parameter point.dt exists and it is a list,
            it will be a list. Otherwise not.
        """
        if isinstance(point, SurfacePoint):
            md = MoonDataFactory.get_md_from_surface(point, kernels_path)

        elif isinstance(point, CustomPoint):
            md = MoonDataFactory.get_md_from_custom(point)
        else:
            md = MoonDataFactory.get_md_from_satellite(point, eocfi_path, kernels_path)
        return md

    @staticmethod
    def get_md_from_surface(
        sp: SurfacePoint,
        kernels_path: KernelsPath,
    ) -> Union[MoonData, List[MoonData]]:
        """
        Create a MoonData from a surface point.

        Parameters
        ----------
        sp: SurfacePoint
            SurfacePoint from which to create the MoonData.
        kernels_path: KernelsPath
            Path to the folder with the needed SPICE kernel files.

        Returns
        -------
        md: MoonData | list of MoonData
            MoonData generated from the given data. If the parameter dt was a list,
            this will be a list. Otherwise not.
        """
        md = SPICEAdapter().get_moon_data_from_earth(
            sp.latitude, sp.longitude, sp.altitude, sp.dt, kernels_path
        )
        return md

    @staticmethod
    def get_md_from_custom(
        cp: CustomPoint,
    ) -> MoonData:
        """
        Create a MoonData from a custom point.

        Parameters
        ----------
        cp: CustomPoint
            CustomPoint from which to create the MoonData.

        Returns
        -------
        md: MoonData
            MoonData generated from the given data.
        """
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
        kernels_path: KernelsPath,
    ) -> Union[MoonData, List[MoonData]]:
        """
        Create a MoonData from a satellite point.

        Parameters
        ----------
        sp: SatellitePoint
            SatellitePoint from which to create the MoonData.
        eocfi_path: str
            Path to the folder with the needed eocfi data files.
        kernels_path: KernelsPath
            Path to the folder with the needed SPICE kernel files.

        Returns
        -------
        md: MoonData | list of MoonData
            MoonData generated from the given data. It will be a list if sp.dt is a list.
        """
        return MoonDataFactory.get_md_and_surfaces_from_satellite(
            sp, eocfi_path, kernels_path
        )[0]

    @staticmethod
    def get_md_and_surfaces_from_satellite(
        sp: SatellitePoint,
        eocfi_path: str,
        kernels_path: KernelsPath,
    ) -> Union[
        Tuple[MoonData, SurfacePoint], Tuple[List[MoonData], List[SurfacePoint]]
    ]:
        """
        Create a MoonData from a satellite point.

        Parameters
        ----------
        sp: SatellitePoint
            SatellitePoint from which to create the MoonData.
        eocfi_path: str
            Path to the folder with the needed eocfi data files.
        kernels_path: KernelsPath
            Path to the folder with the needed SPICE kernel files.

        Returns
        -------
        md: MoonData | list of MoonData
            MoonData generated from the given data. It will be a list if sp.dt is a list.
        srp: SurfacePoint | list of SurfacePoint
            SurfacePoint generated from the position of the SatellitePoint at the given datetimes.
        """
        eocfi: IEOCFIConverter = EOCFIConverter(eocfi_path, kernels_path)
        dts = sp.dt
        if not isinstance(dts, list):
            dts = [dts]

        mds: List[MoonData] = []
        srps: List[SurfacePoint] = []
        xyzs = eocfi.get_satellite_position_rectangular(sp.name, dts)
        llhs = SPICEAdapter.to_planetographic_multiple(
            xyzs, "EARTH", kernels_path.main_kernels_path, dts, "ITRF93"
        )
        for i, llh in enumerate(llhs):
            srp = SurfacePoint(llh[0], llh[1], llh[2], dts[i])
            srps.append(srp)
        mds = SPICEAdapter.get_moon_datas_from_rectangular_multiple(
            xyzs, dts, kernels_path, "ITRF93"  # EOCFI uses ITRF93
        )

        if not isinstance(sp.dt, list):
            return mds[0], srps[0]

        return mds, srps
