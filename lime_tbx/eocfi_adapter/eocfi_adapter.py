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

class IEOCFIConverter(ABC):
    @abstractmethod
    def get_satellite_location(satellite, datetime):
        pass

class EOCFIConverter(IEOCFIConverter): 

    def get_satellite_location(satellite, datetime):
        pass
        