"""describe class"""

"""___Built-In Modules___"""
from abc import ABC, abstractmethod
from typing import List

"""___Third-Party Modules___"""
import numpy as np

"""___NPL Modules___"""
from ..datatypes.datatypes import (
    PolarizationCoefficients,
    SRFChannel,
    SpectralResponseFunction,
    ApolloIrradianceCoefficients,
    ReflectanceCoefficients,
)
from ..datatypes import constants
from ..coefficients.access_data import access_data

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
    def get_irr_coeffs(self) -> ApolloIrradianceCoefficients:
        """Obtain the current IrradianceCoefficients chosen by the user."""
        pass

    @abstractmethod
    def get_polar_coeffs(self) -> PolarizationCoefficients:
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
    def get_available_cimel_coeffs(self) -> List[ReflectanceCoefficients]:
        """Obtain a list with all the cimel coefficients available"""
        pass

    @abstractmethod
    def select_cimel_coeff(self, index: int) -> None:
        """Select the cimel_coeff from the cimel coefficients list at the given position."""
        pass

    @abstractmethod
    def reload_coeffs(self) -> None:
        """Reload the cimel coefficients from file to the logic instance"""
        pass


class MockSettingsManager(ISettingsManager):
    def __init__(self):
        # generate an arbitrary default srf
        self.srfs = [self.get_default_srf()]
        self.srf = self.srfs[0]
        self.cimel_coeffs = (
            access_data.get_all_cimel_coefficients()
        )  # [access_data.get_default_cimel_coeffs()]
        self.cimel_coeff = self.cimel_coeffs[-1]

    def get_default_srf(self) -> SpectralResponseFunction:
        spectral_response = {
            i: 1.0 for i in np.arange(constants.MIN_WLEN, constants.MAX_WLEN, 2)
        }
        ch = SRFChannel(
            (constants.MAX_WLEN - constants.MIN_WLEN) / 2,
            constants.DEFAULT_SRF_NAME,
            spectral_response,
        )
        srf = SpectralResponseFunction(constants.DEFAULT_SRF_NAME, [ch])
        return srf

    def get_srf(self) -> SpectralResponseFunction:
        return self.srf

    def get_irr_coeffs(self) -> ApolloIrradianceCoefficients:
        return access_data._get_default_irradiance_coefficients()

    def get_polar_coeffs(self) -> PolarizationCoefficients:
        return access_data._get_default_polarization_coefficients()

    def load_srf(self, srf: SpectralResponseFunction):
        self.srfs.append(srf)

    def select_srf(self, index: int):
        self.srf = self.srfs[index]

    def get_available_srfs(self) -> List[SpectralResponseFunction]:
        return self.srfs

    def get_cimel_coef(self) -> ReflectanceCoefficients:
        return self.cimel_coeff

    def get_available_cimel_coeffs(self) -> List[ReflectanceCoefficients]:
        return self.cimel_coeffs

    def select_cimel_coeff(self, index: int) -> None:
        self.cimel_coeff = self.cimel_coeffs[index]

    def reload_coeffs(self) -> None:
        self.cimel_coeffs = access_data.get_all_cimel_coefficients()
        self.cimel_coeff = self.cimel_coeffs[-1]
