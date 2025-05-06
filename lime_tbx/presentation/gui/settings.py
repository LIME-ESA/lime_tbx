"""describe class"""

"""___Built-In Modules___"""
from abc import ABC, abstractmethod
from typing import List, Union
import os

"""___Third-Party Modules___"""
import numpy as np

"""___NPL Modules___"""
from lime_tbx.common.datatypes import (
    LimeCoefficients,
    PolarisationCoefficients,
    AOLPCoefficients,
    SRFChannel,
    SpectralResponseFunction,
    ReflectanceCoefficients,
)
from lime_tbx.common import constants
from lime_tbx.application.coefficients import access_data
from lime_tbx.business.interpolation.interp_data import interp_data
from lime_tbx.business.spectral_integration.spectral_integration import get_default_srf

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
    def get_polar_coef(self) -> PolarisationCoefficients:
        """Obtain the current PolarisationCoefficients chosen by the user."""
        pass

    @abstractmethod
    def get_aolp_coef(self) -> AOLPCoefficients:
        """Obtain the current AOLPCoefficients chosen by the user."""
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
    def get_available_dolp_spectra_names(self) -> List[str]:
        """Obtain a list with all the available dolp spectra names"""
        pass

    @abstractmethod
    def get_selected_spectrum_name(self) -> str:
        """Obtain the currently selected interpolation spectrum name"""
        pass

    @abstractmethod
    def get_selected_polar_spectrum_name(self) -> str:
        """Obtain the currently selected polarisation interpolation spectrum name"""
        pass

    @abstractmethod
    def select_interp_spectrum(self, name: str):
        """Select the interpolation spectrum to use"""
        pass

    @abstractmethod
    def select_interp_polar_spectrum(self, name: str):
        """Select the polarisation interpolation spectrum to use"""
        pass

    @abstractmethod
    def get_available_interp_SRFs(self) -> List[str]:
        """Obtain a list with all the available spectra names"""
        pass

    @abstractmethod
    def get_selected_interp_SRF(self) -> str:
        """Obtain the currently selected interpolation SRF name"""
        pass

    @abstractmethod
    def select_interp_SRF(self, name: str):
        """Select the interpolation spectrum to use as SRF"""
        pass

    @abstractmethod
    def is_show_interp_spectrum(self) -> bool:
        """Checks if the UI should show the spectrum used for interpolation.

        Returns
        -------
        should_show: bool
            True if the spectrum should be shown.
        """
        pass

    @abstractmethod
    def set_show_interp_spectrum(self, show: bool):
        """Sets the interpolation spectrum visibility as <show>.

        Parameters
        ----------
        show: bool
            Visibility of the interpolation spectrum.
        """
        pass

    @abstractmethod
    def is_show_cimel_points(self) -> bool:
        """Checks if the UI should show the CIMEL points used for interpolation.

        Returns
        -------
        should_show: bool
            True if the CIMEL points should be shown.
        """
        pass

    @abstractmethod
    def set_show_cimel_points(self, show: bool):
        """Sets the CIMEL points visibility as <show>.

        Parameters
        ----------
        show: bool
            Visibility of the CIMEL points.
        """
        pass

    @abstractmethod
    def is_skip_uncertainties(self) -> bool:
        """Checks if the user settings are set to skip the uncertainties calculation.

        Returns
        -------
        should_skip: bool
            True if the uncertainties calculation should be skipped."""
        pass

    @abstractmethod
    def set_skip_uncertainties(self, skip: bool):
        """Sets if the uncertainties should be calculated.

        Parameters
        ----------
        skip: bool
            True if the uncertainties should be skipped.
        """
        pass

    @abstractmethod
    def get_coef_version_name(self) -> str:
        """Gets the current coefficient version of the current simulation.
        (If the simulation is loaded, it won't be the same as of the settings)

        Returns
        ----------
        coef_version_name: str
            Coefficients version."""
        pass

    @abstractmethod
    def set_coef_version_name(self, coef_version_name: str):
        """Sets the current coefficient version of the current simulation status (not of the settings).

        Parameters
        ----------
        coef_version_name: str
            Coefficients version. None if reset.
        """
        pass

    @abstractmethod
    def get_use_wehrli_for_esi(self) -> bool:
        """
        Checks if the user/dev settings are set to use the wehrli spectrum as source for the ESI
        or if the TSIS will be used instead.

        Returns
        -------
        use_wehrli: bool
            Boolean indicating if the Wehrli spectrum will be used or not.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_intermediate_results_path() -> Union[str, None]:
        """
        Checks if the user is debugging the intermediate results and checks for the path they
        want the data to be stored at.

        Returns
        -------
        str or None
            Path selected by the user, or None in case that the user is not debugging the
            intermediate results.
        """
        pass


DEF_SRF_STEP = 2


class SettingsManager(ISettingsManager):
    def __init__(self, previous_coeff_name: str = None):
        # generate an arbitrary default srf
        self.srfs = [self.get_default_srf()]
        self.srf = self.srfs[0]
        self.coeffs = access_data.get_all_coefficients()
        index = -1
        if previous_coeff_name != None:
            versions = [coef.version for coef in self.coeffs]
            if previous_coeff_name in versions:
                index = versions.index(previous_coeff_name)
        self.coeff = self.coeffs[index]
        self.cimel_coeff = self.coeffs[index].reflectance
        self.polar_coeff = self.coeffs[index].polarisation
        self.aolp_coeff = self.coeffs[index].aolp
        self.coef_version_name = None

    def get_default_srf(self) -> SpectralResponseFunction:
        return get_default_srf()

    def get_srf(self) -> SpectralResponseFunction:
        return self.srf

    def get_polar_coef(self) -> PolarisationCoefficients:
        return self.polar_coeff

    def get_aolp_coef(self) -> AOLPCoefficients:
        return self.aolp_coeff

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
        self.polar_coeff = self.coeffs[index].polarisation
        self.aolp_coeff = self.coeffs[index].aolp
        access_data.set_previously_selected_version(self.coeff.version)

    def reload_coeffs(self) -> None:
        self.coeffs = access_data.get_all_coefficients()
        self.coeff = self.coeffs[-1]
        self.cimel_coeff = self.coeffs[-1].reflectance
        self.polar_coeff = self.coeffs[-1].polarisation
        self.aolp_coeff = self.coeffs[-1].aolp

    def get_available_interp_SRFs(self) -> List[str]:
        return interp_data.get_available_interp_SRFs()

    def get_selected_interp_SRF(self) -> str:
        return interp_data.get_interpolation_SRF()

    def select_interp_SRF(self, name: str):
        interp_data.set_interpolation_SRF(name)

    def get_available_spectra_names(self) -> List[str]:
        return interp_data.get_available_spectra_names()

    def get_available_dolp_spectra_names(self) -> List[str]:
        return interp_data.get_available_dolp_spectra_names()

    def get_selected_spectrum_name(self) -> str:
        return interp_data.get_interpolation_spectrum_name()

    def get_selected_polar_spectrum_name(self) -> str:
        return interp_data.get_dolp_interpolation_spectrum_name()

    def select_interp_spectrum(self, name: str):
        interp_data.set_interpolation_spectrum_name(name)

    def select_interp_polar_spectrum(self, name: str):
        interp_data.set_dolp_interpolation_spectrum_name(name)

    def is_show_interp_spectrum(self) -> bool:
        return interp_data.is_show_interpolation_spectrum()

    def set_show_interp_spectrum(self, show: bool):
        interp_data.set_show_interpolation_spectrum(show)

    def is_show_cimel_points(self) -> bool:
        return interp_data.is_show_cimel_points()

    def set_show_cimel_points(self, show: bool):
        interp_data.set_show_cimel_points(show)

    def is_skip_uncertainties(self) -> bool:
        return interp_data.is_skip_uncertainties()

    def set_skip_uncertainties(self, skip: bool):
        interp_data.set_skip_uncertainties(skip)

    def get_coef_version_name(self) -> str:
        if self.coef_version_name is None:
            return self.get_lime_coef().version
        return self.coef_version_name

    def set_coef_version_name(self, coef_version_name: str):
        self.coef_version_name = coef_version_name

    def get_use_wehrli_for_esi(self) -> bool:
        return interp_data.get_use_wehrli_for_esi()

    @staticmethod
    def get_intermediate_results_path() -> Union[str, None]:
        if constants.DEBUG_INTERMEDIATE_RESULTS_PATH in os.environ:
            return os.environ[constants.DEBUG_INTERMEDIATE_RESULTS_PATH]
        return None
