"""describe class"""

"""___Built-In Modules___"""
from lime_tbx.interpolation.access_data.access_data import _get_default_asd_data
"""___Third-Party Modules___"""


"""___NPL Modules___"""
import punpy
from comet_maths.interpolation.interpolation import Interpolator

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2022"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"

from abc import ABC, abstractmethod
from ...datatypes.datatypes import MoonData, IrradianceCoefficients

class ISpectralInterpolation(ABC):
    @abstractmethod
    def get_best_asd_reference(selen_obs_lat, selen_obs_lon, selen_sun_lon, abs_moon_phase_angle):
        pass
    @abstractmethod
    def get_interpolated_refl(cimel_refl,asd_reference):
        pass

class SpectralInterpolation(ISpectralInterpolation):
    def __init__(self,relative=True,method_main="cubic",method_hr="cubic"):
        self.intp = Interpolator(relative=relative,method_main=method_main,method_hr=method_hr,
                            min_scale=0.3)
        self.prop = punpy.MCPropagation(1000)

    def get_best_asd_reference(self,moon_data: MoonData):
        return _get_default_asd_data()

    def get_interpolated_refl(self,cimel_wav,cimel_refl,asd_wav,asd_refl,final_wav):
        yy = self.intp.interpolate_1d_along_example(cimel_wav,cimel_refl,asd_wav,asd_refl,final_wav)
        print("here4",cimel_wav,cimel_refl,asd_wav,asd_refl,final_wav)
        return yy

    def get_interpolated_refl_unc(self,cimel_wav,cimel_refl,asd_wav,asd_refl,final_wav,u_cimel_refl,u_asd_refl,corr_cimel_refl=None,corr_asd_refl=None):
        u_yy,corr_yy = self.prop.propagate_random(self.intp.interpolate_1d_along_example,
                                               [cimel_wav,cimel_refl,asd_wav,asd_refl,final_wav],
                                               [None,u_cimel_refl,None,u_asd_refl,None],
                                               corr_x=[None,corr_cimel_refl,None,corr_asd_refl,None],
                                               return_corr=True)
        return u_yy,corr_yy
