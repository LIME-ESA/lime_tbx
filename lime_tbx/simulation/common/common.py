"""describe class"""

"""___Built-In Modules___"""
from typing import Union, List
from abc import ABC, abstractmethod

"""___Third-Party Modules___"""
# import here

"""___NPL Modules___"""
from ...datatypes.datatypes import (
    IrradianceCoefficients,
    MoonData,
    PolarizationCoefficients,
    SpectralResponseFunction,
)
from ...lime_algorithms.rolo import rolo
from ...lime_algorithms.dolp import dolp

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2022"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


class ICommonSimulation(ABC):
    @staticmethod
    @abstractmethod
    def get_eli_from_md(
        srf: SpectralResponseFunction,
        md: Union[MoonData, List[MoonData]],
        coefficients: IrradianceCoefficients,
    ) -> Union[List[float], List[List[float]]]:
        """
        Obtain the irradiance from the MoonData data structure.

        Parameters
        ----------
        srf: SpectralResponseFunction
            Spectral Response Function that the simulation will be computed for.
        md: MoonData | list of MoonData
            MoonData for one observation, or for multiple observations in case that
            it's a list.
        coefficients: IrradianceCoefficients
            Values of the chosen coefficients for the ROLO algorithm.

        Returns
        -------
        irradiances: list of float | list of list of float
            Extraterrestrial lunar irradiances for the given srf at the specified point.
            It will be a list of lists of float if the parameter md is a list. Otherwise it
            will only be a list of float.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_elref_from_md(
        srf: SpectralResponseFunction,
        md: Union[MoonData, List[MoonData]],
        coefficients: IrradianceCoefficients,
    ) -> Union[List[float], List[List[float]]]:
        """
        Obtain the reflectance from the MoonData data structure.

        Parameters
        ----------
        srf: SpectralResponseFunction
            Spectral Response Function that the simulation will be computed for.
        md: MoonData | list of MoonData
            MoonData for one observation, or for multiple observations in case that
            it's a list.
        coefficients: IrradianceCoefficients
            Values of the chosen coefficients for the ROLO algorithm.

        Returns
        -------
        reflectances: list of float | list of list of float
            Extraterrestrial lunar reflectances for the given srf at the specified point.
            It will be a list of lists of float if the parameter md is a list. Otherwise it
            will only be a list of float.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_polar_from_md(
        srf: SpectralResponseFunction,
        md: Union[MoonData, List[MoonData]],
        coefficients: PolarizationCoefficients,
    ) -> Union[List[float], List[List[float]]]:
        """
        Obtain the polarization from the MoonData data structure.

        Parameters
        ----------
        srf: SpectralResponseFunction
            Spectral Response Function that the simulation will be computed for.
        md: MoonData | list of MoonData
            MoonData for one observation, or for multiple observations in case that
            it's a list.
        coefficients: PolarizationCoefficients
            Values of the chosen coefficients for the DOLP algorithm.

        Returns
        -------
        polarizations: list of float | list of list of float
            Extraterrestrial lunar polarizations for the given srf at the specified point.
            It will be a list of lists of float if the parameter md is a list. Otherwise it
            will only be a list of float.
        """
        pass


class CommonSimulation(ICommonSimulation):
    @staticmethod
    def _get_eli_from_md(
        srf: SpectralResponseFunction,
        md: Union[MoonData, List[MoonData]],
        coefficients: IrradianceCoefficients,
    ) -> Union[List[float], List[List[float]]]:
        rl = rolo.ROLO()
        wlens = srf.get_wavelengths()
        if not isinstance(md, list):
            irradiances = rl.get_eli(wlens, md, coefficients)
            for i, w in enumerate(wlens):
                irradiances[i] = irradiances[i]
            return irradiances
        times_irr = []
        for m in md:
            irradiances = rl.get_eli(wlens, m, coefficients)
            for i, w in enumerate(wlens):
                irradiances[i] = irradiances[i]
            times_irr.append(irradiances)
        return times_irr

    @staticmethod
    def _get_elref_from_md(
        srf: SpectralResponseFunction,
        md: Union[MoonData, List[MoonData]],
        coefficients: IrradianceCoefficients,
    ) -> Union[List[float], List[List[float]]]:
        rl = rolo.ROLO()
        wlens = srf.get_wavelengths()
        if not isinstance(md, list):
            reflectances = rl.get_elref(wlens, md, coefficients)
            return reflectances
        times_refl = []
        for m in md:
            reflectances = rl.get_elref(wlens, m, coefficients)
            times_refl.append(reflectances)
        return times_refl

    @staticmethod
    def _get_polar_from_md(
        srf: SpectralResponseFunction,
        md: Union[MoonData, List[MoonData]],
        coefficients: PolarizationCoefficients,
    ) -> Union[List[float], List[List[float]]]:
        dl = dolp.DOLP()
        wlens = srf.get_wavelengths()
        if not isinstance(md, list):
            polarizations = dl.get_polarized(wlens, md.mpa_degrees, coefficients)
            return polarizations
        times_polar = []
        for m in md:
            polarizations = dl.get_polarized(wlens, m.mpa_degrees, coefficients)
            times_polar.append(polarizations)
        return times_polar
