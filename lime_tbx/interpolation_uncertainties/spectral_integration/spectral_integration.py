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

class ISpectralIntegration(ABC):
    @abstractmethod
    def convolve_srf(srf, asd_irradiances):
        pass

class SpectralIntegration(ISpectralIntegration):

    def convolve_srf(srf, asd_irradiances):
        pass
        
