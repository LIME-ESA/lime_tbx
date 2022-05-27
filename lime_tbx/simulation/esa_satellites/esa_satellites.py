"""describe class"""

"""___Built-In Modules___"""
from typing import List
from abc import ABC, abstractmethod

"""___Third-Party Modules___"""
# import here

"""___NPL Modules___"""
from ...datatypes.datatypes import (
    IrradianceCoefficients,
    PolarizationCoefficients,
    SatellitePoint,
    SpectralResponseFunction,
    SurfacePoint,
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
    ) -> List[float]:
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
    ) -> List[float]:
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
    ) -> List[float]:
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
    ) -> List[float]:
        eocfi = EOCFIConverter(eocfi_path)
        lat, lon, height = eocfi.get_satellite_position(sp.name, sp.dt)
        srp = SurfacePoint(lat, lon, height, sp.dt)
        return RegularSimulation.get_eli_from_surface(
            srf, srp, coefficients, kernels_path
        )

    @staticmethod
    def get_elref_from_satellite(
        srf: SpectralResponseFunction,
        sp: SatellitePoint,
        coefficients: IrradianceCoefficients,
        kernels_path: str,
        eocfi_path: str,
    ) -> List[float]:
        eocfi = EOCFIConverter(eocfi_path)
        lat, lon, height = eocfi.get_satellite_position(sp.name, sp.dt)
        srp = SurfacePoint(lat, lon, height, sp.dt)
        return RegularSimulation.get_elref_from_surface(
            srf, srp, coefficients, kernels_path
        )

    @staticmethod
    def get_polarized_from_satellite(
        srf: SpectralResponseFunction,
        sp: SatellitePoint,
        coefficients: PolarizationCoefficients,
        kernels_path: str,
        eocfi_path: str,
    ) -> List[float]:
        eocfi = EOCFIConverter(eocfi_path)
        lat, lon, height = eocfi.get_satellite_position(sp.name, sp.dt)
        srp = SurfacePoint(lat, lon, height, sp.dt)
        return RegularSimulation.get_polarized_from_surface(
            srf, srp, coefficients, kernels_path
        )
