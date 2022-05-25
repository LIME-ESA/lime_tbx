"""describe class"""

"""___Built-In Modules___"""
from abc import ABC, abstractmethod
from typing import List

"""___Third-Party Modules___"""
from PySide2 import QtWidgets, QtCore, QtGui
import numpy as np

"""___NPL Modules___"""
from ..datatypes.datatypes import (
    PolarizationCoefficients,
    SRFChannel,
    SpectralResponseFunction,
    IrradianceCoefficients,
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
    def get_srf(self) -> SpectralResponseFunction:
        """Obtain the current Spectral Response Function chosen by the user."""
        pass

    @abstractmethod
    def get_irr_coeffs(self) -> IrradianceCoefficients:
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


class MockSettingsManager(ISettingsManager):
    def __init__(self):
        # generate an arbitrary default srf
        spectral_response = {
            i: 1.0 for i in np.arange(constants.MIN_WLEN, constants.MAX_WLEN, 2)
        }
        ch = SRFChannel(
            (constants.MAX_WLEN - constants.MIN_WLEN) / 2, "Default", spectral_response
        )
        self.srfs = [SpectralResponseFunction("Default", [ch])]
        self.srf = self.srfs[0]

    def get_srf(self) -> SpectralResponseFunction:
        return self.srf

    def get_irr_coeffs(self) -> IrradianceCoefficients:
        return access_data._get_default_irradiance_coefficients()

    def get_polar_coeffs(self) -> PolarizationCoefficients:
        return access_data._get_default_polarization_coefficients()

    def load_srf(self, srf: SpectralResponseFunction):
        self.srfs.append(srf)

    def select_srf(self, index: int):
        self.srf = self.srfs[index]

    def get_available_srfs(self) -> List[SpectralResponseFunction]:
        return self.srfs
