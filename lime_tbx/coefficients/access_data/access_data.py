"""describe class"""

"""___Built-In Modules___"""
#import here

"""___Third-Party Modules___"""
#import here

"""___NPL Modules___"""
#import here

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2022"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"

from abc import ABC, abstractmethod

class IAccessData(ABC):

    @abstractmethod
    def get_all_coefficients_irradiance() -> list:
        pass
    @abstractmethod
    def get_all_coefficients_polarization() -> list:
        pass

class AccessData(IAccessData):

    def get_all_coefficients_irradiance() -> list:
        pass
    def get_all_coefficients_polarization() -> list:
        pass
