"""describe class"""

"""___Built-In Modules___"""
from typing import Union, List, Tuple
from abc import ABC, abstractmethod

"""___Third-Party Modules___"""
# import here

"""___NPL Modules___"""
from ...datatypes.datatypes import (
    CimelData,
    IrradianceCoefficients,
    MoonData,
    PolarizationCoefficients,
    SpectralResponseFunction,
    UncertaintyData,
)
from ...lime_algorithms.rolo import rolo, eli, elref
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
        cimel_data: CimelData,
        calc_uncertainty: bool = False,
    ) -> Tuple[Union[List[float], List[List[float]]], UncertaintyData]:
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
        cimel_data: CimelData
            Cimel uncertainty data that will be used to calculate the uncertainties
        calc_uncertainty: bool
            Flag that if False, the uncertainties wont be calculated, and will return a None
            instead of a UncertaintyData.

        Returns
        -------
        irradiances: list of float | list of list of float
            Extraterrestrial lunar irradiances for the given srf at the specified point.
            It will be a list of lists of float if the parameter md is a list. Otherwise it
            will only be a list of float.
        uncertainty_data: UncertaintyData
            Uncertainty data calculated, in case that calc_uncertainty was True
        """
        pass

    @staticmethod
    @abstractmethod
    def get_elref_from_md(
        srf: SpectralResponseFunction,
        md: Union[MoonData, List[MoonData]],
        coefficients: IrradianceCoefficients,
        cimel_data: CimelData = None,
        calc_uncertainty: bool = False,
    ) -> Tuple[Union[List[float], List[List[float]]], UncertaintyData]:
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
        cimel_data: CimelData
            Cimel uncertainty data that will be used to calculate the uncertainties
        calc_uncertainty: bool
            Flag that if False, the uncertainties wont be calculated, and will return a None
            instead of a UncertaintyData.

        Returns
        -------
        reflectances: list of float | list of list of float
            Extraterrestrial lunar reflectances for the given srf at the specified point.
            It will be a list of lists of float if the parameter md is a list. Otherwise it
            will only be a list of float.
        uncertainty_data: UncertaintyData
            Uncertainty data calculated, in case that calc_uncertainty was True
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
    def _get_uncertainty_eli(md: MoonData, cimel_data: CimelData, calc_uncertainty: bool):
        uncertainty_data = None
        if calc_uncertainty:
            elis_cimel = eli.calculate_eli_band(cimel_data, md)
            u_elis_cimel = eli.calculate_eli_band_unc(cimel_data, md)
            uncertainty_data = UncertaintyData(cimel_data.wavelengths, elis_cimel, u_elis_cimel)
        return uncertainty_data
    
    @staticmethod
    def _get_uncertainty_elref(md: MoonData, cimel_data: CimelData):
        uncertainty_data = None
        if cimel_data:
            elrefs_cimel = elref.band_moon_disk_reflectance(cimel_data, md)
            u_elrefs_cimel = elref.band_moon_disk_reflectance_unc(cimel_data, md)
            uncertainty_data = UncertaintyData(cimel_data.wavelengths, elrefs_cimel, u_elrefs_cimel)
        return uncertainty_data

    @staticmethod
    def get_eli_from_md(
        srf: SpectralResponseFunction,
        md: Union[MoonData, List[MoonData]],
        coefficients: IrradianceCoefficients,
        cimel_data: CimelData = None,
        calc_uncertainty: bool = False,
    ) -> Tuple[Union[List[float], List[List[float]]], UncertaintyData]:
        rl = rolo.ROLO()
        wlens = srf.get_wavelengths()
        if not isinstance(md, list):
            irradiances = rl.get_eli(wlens, md, coefficients)
            for i, w in enumerate(wlens):
                irradiances[i] = irradiances[i]
            return irradiances, CommonSimulation._get_uncertainty_eli(md, cimel_data, calc_uncertainty)
        times_irr = []
        uncertainties = []
        for m in md:
            irradiances = rl.get_eli(wlens, m, coefficients)
            for i, w in enumerate(wlens):
                irradiances[i] = irradiances[i]
            uncertainties.append(CommonSimulation._get_uncertainty_eli(m, cimel_data, calc_uncertainty))
            times_irr.append(irradiances)
        return times_irr, uncertainties

    @staticmethod
    def get_elref_from_md(
        srf: SpectralResponseFunction,
        md: Union[MoonData, List[MoonData]],
        coefficients: IrradianceCoefficients,
        cimel_data: CimelData = None,
    ) -> Tuple[Union[List[float], List[List[float]]], UncertaintyData]:
        rl = rolo.ROLO()
        wlens = srf.get_wavelengths()
        if not isinstance(md, list):
            reflectances = rl.get_elref(wlens, md, coefficients)
            return reflectances, CommonSimulation._get_uncertainty_elref(md, cimel_data)
        times_refl = []
        uncertainties = []
        for m in md:
            reflectances = rl.get_elref(wlens, m, coefficients)
            times_refl.append(reflectances)
            uncertainties.append(CommonSimulation._get_uncertainty_elref(m, cimel_data))
        return times_refl, uncertainties

    @staticmethod
    def get_polar_from_md(
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
