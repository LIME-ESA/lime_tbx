"""describe class"""

"""___Built-In Modules___"""
from abc import ABC, abstractmethod
from typing import List
import os

"""___Third-Party Modules___"""
import numpy as np

"""___NPL Modules___"""
from lime_tbx.datatypes.datatypes import (
    LimeCoefficients,
    PolarizationCoefficients,
    SRFChannel,
    SpectralResponseFunction,
    ReflectanceCoefficients,
)
from lime_tbx.datatypes import constants
from lime_tbx.coefficients.access_data import access_data
from lime_tbx.interpolation.interp_data import interp_data

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "05/05/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


class ISettingsManager(ABC):
    """
    Object that manages the current settings chosen by the user.
    """

    @abstractmethod
    def get_default_srf(self) -> SpectralResponseFunction:
        """Obtain the default SRF"""
        pass

    @abstractmethod
    def get_srf(self) -> SpectralResponseFunction:
        """Obtain the current Spectral Response Function chosen by the user."""
        pass

    @abstractmethod
    def get_polar_coef(self) -> PolarizationCoefficients:
        """Obtain the current PolarizationCoefficients chosen by the user."""
        pass

    @abstractmethod
    def load_srf(self, srf: SpectralResponseFunction) -> None:
        """Add the srf to the available srf list."""
        pass

    @abstractmethod
    def select_srf(self, index: int) -> None:
        """Select the srf from the srf list at the given position."""
        pass

    @abstractmethod
    def get_available_srfs(self) -> List[SpectralResponseFunction]:
        """Obtain a list with all the SRFS the user can choose"""
        pass

    @abstractmethod
    def get_cimel_coef(self) -> ReflectanceCoefficients:
        """Obtain the CimelCoef the CIMEL coefficients and uncertainties"""
        pass

    @abstractmethod
    def get_lime_coef(self) -> LimeCoefficients:
        """Obtain the LimeCoef coefficients and uncertainties"""
        pass

    @abstractmethod
    def get_available_coeffs(self) -> List[LimeCoefficients]:
        """Obtain a list with all the lime coefficients available"""
        pass

    @abstractmethod
    def select_lime_coeff(self, index: int) -> None:
        """Select the cimel_coeff from the cimel coefficients list at the given position."""
        pass

    @abstractmethod
    def reload_coeffs(self) -> None:
        """Reload the cimel coefficients from file to the logic instance"""
        pass

    @abstractmethod
    def get_available_spectra_names(self) -> List[str]:
        """Obtain a list with all the available spectra names"""
        pass

    @abstractmethod
    def get_selected_spectrum_name(self) -> str:
        """Obtain the currently selected interpolation spectrum name"""
        pass

    @abstractmethod
    def select_interp_spectrum(self, name: str):
        """Select the interpolation spectrum to use"""
        pass

    @abstractmethod
    def can_perform_polarization(self) -> bool:
        """Checks if the current settings allow for the performance of polarization."""
        pass


DEF_SRF_STEP = 2


class SettingsManager(ISettingsManager):
    def __init__(self, previous_coeff_name: str = None):
        # generate an arbitrary default srf
        self.srfs = [self.get_default_srf()]
        self.srf = self.srfs[0]
        self.coeffs = access_data.AccessData().get_all_coefficients()
        index = -1
        if previous_coeff_name != None:
            versions = [coef.version for coef in self.coeffs]
            if previous_coeff_name in versions:
                index = versions.index(previous_coeff_name)
        self.coeff = self.coeffs[index]
        self.cimel_coeff = self.coeffs[index].reflectance
        self.polar_coeff = self.coeffs[index].polarization

    def get_default_srf(self) -> SpectralResponseFunction:
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        dir_path = os.path.join(dir_path, "spectral_integration", "assets")
        data = np.genfromtxt(
            os.path.join(dir_path, "interpolated_model_fwhm.csv"), delimiter=","
        )
        wavs = data[:, 0]
        spectral_response = {i: 1.0 for i in wavs}
        ch = SRFChannel(
            (constants.MAX_WLEN - constants.MIN_WLEN) / 2,
            constants.DEFAULT_SRF_NAME,
            spectral_response,
        )
        srf = SpectralResponseFunction(constants.DEFAULT_SRF_NAME, [ch])
        return srf

    def get_srf(self) -> SpectralResponseFunction:
        return self.srf

    def get_polar_coef(self) -> PolarizationCoefficients:
        return self.polar_coeff

    def load_srf(self, srf: SpectralResponseFunction):
        self.srfs.append(srf)

    def select_srf(self, index: int):
        self.srf = self.srfs[index]

    def get_available_srfs(self) -> List[SpectralResponseFunction]:
        return self.srfs

    def get_cimel_coef(self) -> ReflectanceCoefficients:
        return self.cimel_coeff

    def get_lime_coef(self) -> LimeCoefficients:
        return self.coeff

    def get_available_coeffs(self) -> List[LimeCoefficients]:
        return self.coeffs

    def select_lime_coeff(self, index: int) -> None:
        self.coeff = self.coeffs[index]
        self.cimel_coeff = self.coeffs[index].reflectance
        self.polar_coeff = self.coeffs[index].polarization
        access_data.AccessData().set_previusly_selected_version(self.coeff.version)

    def reload_coeffs(self) -> None:
        self.coeffs = access_data.AccessData().get_all_coefficients()
        self.coeff = self.coeffs[-1]
        self.cimel_coeff = self.coeffs[-1].reflectance
        self.polar_coeff = self.coeffs[-1].polarization

    def get_available_spectra_names(self) -> List[str]:
        return interp_data.get_available_spectra_names()

    def get_selected_spectrum_name(self) -> str:
        return interp_data.get_interpolation_spectrum_name()

    def select_interp_spectrum(self, name: str):
        interp_data.set_interpolation_spectrum_name(name)

    def can_perform_polarization(self) -> bool:
        return interp_data.can_perform_polarization()
