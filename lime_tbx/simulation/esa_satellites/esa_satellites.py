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

class IESASatellites(ABC):
    @abstractmethod
    def get_eli_from_satellite(srf, satellite, datetime) -> list:
        pass
    @abstractmethod
    def get_polarized_from_satellite(srf, satellite, datetime) -> list:
        pass

class ESASatellites(IESASatellites):
    def get_eli_from_satellite(srf, satellite, datetime) -> list:
        pass
    def get_polarized_from_satellite(srf, satellite, datetime) -> list:
        pass
        
