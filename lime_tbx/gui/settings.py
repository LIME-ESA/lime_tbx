"""describe class"""

"""___Built-In Modules___"""
from abc import ABC, abstractmethod

"""___Third-Party Modules___"""
from PySide2 import QtWidgets, QtCore, QtGui
import numpy as np

"""___NPL Modules___"""
from ..datatypes.datatypes import SpectralResponseFunction, IrradianceCoefficients
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


class MockSettingsManager(ISettingsManager):
    def get_srf(self) -> SpectralResponseFunction:
        # generate an arbitrary default srf
        spectral_response = {i: 1.0 for i in np.arange(380, 2500, 2)}
        return SpectralResponseFunction(spectral_response)

    def get_irr_coeffs(self) -> IrradianceCoefficients:
        return access_data._get_default_irradiance_coefficients()
