"""describe class"""

"""___Built-In Modules___"""
from typing import List, Union, Tuple
from abc import ABC, abstractmethod

"""___Third-Party Modules___"""
# import here

"""___NPL Modules___"""
from ...datatypes.datatypes import (
    CimelData,
    IrradianceCoefficients,
    PolarizationCoefficients,
    SatellitePoint,
    SpectralResponseFunction,
    SurfacePoint,
    UncertaintyData,
)
from ...eocfi_adapter.eocfi_adapter import EOCFIConverter
from ..regular_simulation.regular_simulation import RegularSimulation

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2022"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


class IESASatellites(ABC):
    @staticmethod
    @abstractmethod
    def get_eli_from_satellite(
        srf: SpectralResponseFunction,
        sp: SatellitePoint,
        coefficients: IrradianceCoefficients,
        kernels_path: str,
        eocfi_path: str,
        cimel_data: CimelData = None,
        calc_uncertainty: bool = False,
    ) -> Tuple[Union[List[float], List[List[float]]], Union[UncertaintyData, List[UncertaintyData]]]:
        """
        Simulate the extraterrestrial lunar irradiance for a satellite point.

        Returns the data in Wm⁻²/nm

        Parameters
        ----------
        srf: SpectralResponseFunction
            Spectral Response Function that the simulation will be computed for.
        sp: SatellitePoint
            Satellite point for which the simulation will be computed for.
        coefficients: IrradianceCoefficients
            Values of the chosen coefficients for the ROLO algorithm.
        kernels_path: str
            Path where the needed SPICE kernels are located.
            The user must have write access to that directory.
        eocfi_path: str
            Path where the needed EOCFI datafiles are located.

        Returns
        -------
        elis: list of float
            Extraterrestrial lunar irradiances for the given srf at the specified point.
        uncerts: UncertaintyData or list of UncertaintyData
            The uncertaintie/s related to the elis
        """
        pass

    @staticmethod
    @abstractmethod
    def get_elref_from_satellite(
        srf: SpectralResponseFunction,
        sp: SatellitePoint,
        coefficients: IrradianceCoefficients,
        kernels_path: str,
        eocfi_path: str,
        cimel_data: CimelData = None,
    ) -> Tuple[Union[List[float], List[List[float]]], Union[UncertaintyData, List[UncertaintyData]]]:
        """
        Simulate the extraterrestrial lunar reflectance for a satellite point.

        Returns the data in fracions of unity.

        Parameters
        ----------
        srf: SpectralResponseFunction
            Spectral Response Function that the simulation will be computed for.
        sp: SatellitePoint
            Satellite point for which the simulation will be computed for.
        coefficients: IrradianceCoefficients
            Values of the chosen coefficients for the ROLO algorithm.
        kernels_path: str
            Path where the needed SPICE kernels are located.
            The user must have write access to that directory.
        eocfi_path: str
            Path where the needed EOCFI datafiles are located.

        Returns
        -------
        elrefs: list of float
            Extraterrestrial lunar reflectanes for the given srf at the specified point.
        uncerts: UncertaintyData or list of UncertaintyData
            The uncertaintie/s related to the elis
        """
        pass

    @staticmethod
    @abstractmethod
    def get_polarized_from_satellite(
        srf: SpectralResponseFunction,
        sp: SatellitePoint,
        coefficients: PolarizationCoefficients,
        kernels_path: str,
        eocfi_path: str,
    ) -> Union[List[float], List[List[float]]]:
        """
        Simulate the extraterrestrial lunar polarization for a satellite point.

        Returns the data in fracions of unity.

        Parameters
        ----------
        srf: SpectralResponseFunction
            Spectral Response Function that the simulation will be computed for.
        sp: SatellitePoint
            Satellite point for which the simulation will be computed for.
        coefficients: PolarizationCoefficients
            Values of the chosen coefficients for the DOLP algorithm.
        kernels_path: str
            Path where the needed SPICE kernels are located.
            The user must have write access to that directory.
        eocfi_path: str
            Path where the needed EOCFI datafiles are located.

        Returns
        -------
        polarizations: list of float
            Extraterrestrial lunar polarizations for the given srf at the specified point.
        """
        pass


class ESASatellites(IESASatellites):
    @staticmethod
    def get_eli_from_satellite(
        srf: SpectralResponseFunction,
        sp: SatellitePoint,
        coefficients: IrradianceCoefficients,
        kernels_path: str,
        eocfi_path: str,
        cimel_data: CimelData = None,
        calc_uncertainty: bool = False,
    ) -> Tuple[Union[List[float], List[List[float]]], Union[UncertaintyData, List[UncertaintyData]]]:
        eocfi = EOCFIConverter(eocfi_path)
        dts = sp.dt
        wasnt_list = False
        if not isinstance(dts, list):
            wasnt_list = True
            dts = [dts]
        elis = []
        uncerts = []
        for dt in dts:
            lat, lon, height = eocfi.get_satellite_position(sp.name, dt)
            srp = SurfacePoint(lat, lon, height, dt)
            new_eli, uncert = RegularSimulation.get_eli_from_surface(
                    srf, srp, coefficients, kernels_path, cimel_data, calc_uncertainty)
            elis.append(new_eli)
            uncerts.append(uncert)
        if wasnt_list:
            elis = elis[0]
            uncerts = uncerts[0]
        return elis, uncerts

    @staticmethod
    def get_elref_from_satellite(
        srf: SpectralResponseFunction,
        sp: SatellitePoint,
        coefficients: IrradianceCoefficients,
        kernels_path: str,
        eocfi_path: str,
        cimel_data: CimelData = None,
    ) -> Tuple[Union[List[float], List[List[float]]], Union[UncertaintyData, List[UncertaintyData]]]:
        eocfi = EOCFIConverter(eocfi_path)
        dts = sp.dt
        wasnt_list = False
        if not isinstance(dts, list):
            wasnt_list = True
            dts = [dts]
        elrefs = []
        uncerts = []
        for dt in dts:
            lat, lon, height = eocfi.get_satellite_position(sp.name, dt)
            srp = SurfacePoint(lat, lon, height, dt)
            new_elref, uncert = RegularSimulation.get_elref_from_surface(
                srf, srp, coefficients, kernels_path, cimel_data)
            elrefs.append(new_elref)
            uncerts.append(uncert)
        if wasnt_list:
            elrefs = elrefs[0]
            uncerts = uncerts[0]
        return elrefs, uncerts

    @staticmethod
    def get_polarized_from_satellite(
        srf: SpectralResponseFunction,
        sp: SatellitePoint,
        coefficients: PolarizationCoefficients,
        kernels_path: str,
        eocfi_path: str,
    ) -> Union[List[float], List[List[float]]]:
        eocfi = EOCFIConverter(eocfi_path)
        dts = sp.dt
        wasnt_list = False
        if not isinstance(dts, list):
            wasnt_list = True
            dts = [dts]
        polars = []
        for dt in dts:
            lat, lon, height = eocfi.get_satellite_position(sp.name, dt)
            srp = SurfacePoint(lat, lon, height, dt)
            polars.append(
                RegularSimulation.get_polarized_from_surface(
                    srf, srp, coefficients, kernels_path
                )
            )
        if wasnt_list:
            polars = polars[0]
        return polars
