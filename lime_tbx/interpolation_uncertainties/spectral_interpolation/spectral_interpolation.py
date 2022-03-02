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

class ISpectralInterpolation(ABC):
    @abstractmethod
    def get_best_asd_reference(selen_obs_lat, selen_obs_lon, selen_sun_lon, abs_moon_phase_angle):
        pass
    @abstractmethod
    def get_interpolated_refl(cimel_refl,asd_reference):
        pass

class SpectralInterpolation(ISpectralInterpolation):

    def get_best_asd_reference(selen_obs_lat, selen_obs_lon, selen_sun_lon, abs_moon_phase_angle):
        pass
    def get_interpolated_refl(cimel_refl,asd_reference):
        pass
        
