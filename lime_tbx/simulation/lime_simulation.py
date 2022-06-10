"""describe class"""

"""___Built-In Modules___"""
from abc import ABC, abstractmethod
from typing import List, Union

"""___Third-Party Modules___"""
# import here
import punpy

"""___NPL Modules___"""
from lime_tbx.datatypes.datatypes import (
    IrradianceCoefficients,
    MoonData,
    PolarizationCoefficients,
    SpectralResponseFunction,
    SurfacePoint,
    CustomPoint,
    SpectralData,
    CimelCoef
)

from lime_tbx.lime_algorithms.rolo import eli, elref, esi
from lime_tbx.lime_algorithms.dolp import dolp
from lime_tbx.interpolation.spectral_interpolation.spectral_interpolation import SpectralInterpolation
from lime_tbx.spice_adapter.spice_adapter import SPICEAdapter
from lime_tbx.simulation.moon_data import MoonDataFactory

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2022"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


class LimeSimulation():
    """
        Class for running the main lime-tbx functionality


        """

    def __init__(self,
                 eocfi_path: str,
                 kernels_path: str,
                 ):
        """
        Constructor method
        """
        self.kernels_path=kernels_path
        self.eocfi_path = eocfi_path

        self.moondata = []
        self.wlens = []
        self.elref = None
        self.elis = None
        self.elref_cimel = None
        self.elref_asd = None
        self.elis_cimel = None
        self.elis_asd = None
        self.polars = None
        self.refl_uptodate = False
        self.irr_uptodate = False
        self.pol_uptodate = False
        self.intp = SpectralInterpolation()


    def update_model_refl(self,point,cimel_coeff):
        if not self.refl_uptodate:
            md=MoonDataFactory.get_md(point,self.eocfi_path,self.kernels_path)

            cimel_data,asd_data,elref_data = self.calculate_elref(md,cimel_coeff)

            cimel_data = self._get_data_elref_cimel(md,cimel_coeff,True)
            asd_data = self.intp.get_best_asd_reference(md)
            intp_data = self.interpolate_refl(asd_data,cimel_data)

            self.elref = intp_data
            self.elref_cimel = cimel_data
            self.elref_asd = asd_data
            self.refl_uptodate=True

    def update_model_irr(self,point,cimel_coeff):
        md = MoonDataFactory.get_md(point,self.eocfi_path,self.kernels_path)
        if not self.refl_uptodate:
            cimel_data,asd_data,elref_data = self.calculate_elref(md,cimel_coeff)

            cimel_data = self._get_data_elref_cimel(md,cimel_coeff,True)
            asd_data = self.intp.get_best_asd_reference(md)
            intp_data = self.interpolate_refl(asd_data,cimel_data)

            self.elref = intp_data
            self.elref_cimel = cimel_data
            self.elref_asd = asd_data
            self.refl_uptodate=True

        if not self.irr_uptodate:
            self.elis = self.calculate_eli_from_elref(md,self.elref)
            self.elis_cimel = self.calculate_eli_from_elref(md, self.elref_cimel )
            self.elis_asd = self.calculate_eli_from_elref(md,self.elref_asd )
            self.irr_uptodate=True

    def update_model_pol(self,point,polar_coeff):
        md = MoonDataFactory.get_md(point,self.eocfi_path,self.kernels_path)
        if not self.pol_uptodate:
            self.polars = self.calculate_polar(md,polar_coeff)

    def calculate_elref(self,
            md: MoonData,
            cimel_coeff: CimelCoef,
            ) -> SpectralData:
        """Callback that performs the Reflectance operations.

        Parameters
        ----------
        srf: SpectralResponseFunction
            SRF that will be used to calculate the graph
        point: Union[SurfacePoint, CustomPoint, SatellitePoint]
            Point used
        coeffs: IrradianceCoefficients
            Coefficients used by the algorithms in order to calculate the irradiance or reflectance.
        cimel_coeff: CimelCoef
            CimelCoef with the CIMEL coefficients and uncertainties.
        kernels_path: str
            Path where the directory with the SPICE kernels is located.
        eocfi_path: str
            Path where the directory with the needed EOCFI data files is located.

        Returns
        -------
        wlens: list of float
            Wavelengths of def_srf
        elrefs: list of float
            Reflectances related to srf
        point: Union[SurfacePoint, CustomPoint, SatellitePoint]
            Point that was used in the calculations.
        uncertainty_data: UncertaintyData or list of UncertaintyData
            Calculated uncertainty data.
        """

        cimel_data = self._get_data_elref_cimel(md, cimel_coeff, True)
        asd_data = self.intp.get_best_asd_reference(md)
        intp_data = self.interpolate_refl(asd_data,cimel_data)

        return cimel_data, asd_data, intp_data

    def interpolate_refl(self,
            asd_data: SpectralData,
            cimel_coeff: SpectralData,
            calc_uncertainty: bool =True
            ) -> SpectralData:
        
        
        elrefs_intp = self.intp.get_interpolated_refl(cimel_coeff.wlen,cimel_coeff.data,
                                                 asd_data.wlen,asd_data.data,self.wlens)
        u_elrefs_intp = None
        if calc_uncertainty:
            u_elrefs_intp = elrefs_intp*0.01  # intp.get_interpolated_refl_unc(wlen_cimel,elrefs_cimel,wlen_asd,elrefs_asd,wlens,u_elrefs_cimel,u_elrefs_asd)

        ds_intp = SpectralData.make_reflectance_ds(self.wlens,elrefs_intp,u_elrefs_intp)

        spectral_data = SpectralData(cimel_coeff.wlen,elrefs_intp,u_elrefs_intp,ds_intp)
        return spectral_data

    def calculate_eli_from_elref(self, moon_data: MoonData,
            elref: SpectralData) -> SpectralData:
        """Calculation of Extraterrestrial Lunar Irradiance following Eq 3 in Roman et al., 2020

        Simulates a lunar observation for a wavelength for any observer/solar selenographic
        latitude and longitude. The irradiance is calculated in Wm⁻²/nm.

        Parameters
        ----------
        wavelength_nm : float
            Wavelength (in nanometers) of which the extraterrestrial lunar irradiance will be
            calculated.
        moon_data : MoonData
            Moon data needed to calculate Moon's irradiance
        coefficients : IrradianceCoefficients
            Needed coefficients for the simulation.

        Returns
        -------
        float
            The extraterrestrial lunar irradiance calculated
        """
        solid_angle_moon: float = 6.4177e-05
        omega = solid_angle_moon
        esk = [esi.get_esi_per_nm(wav) for wav in elref.wlen]
        dsm = moon_data.distance_sun_moon
        dom = moon_data.distance_observer_moon
        distance_earth_moon_km: int = 384400

        lunar_irr = eli.measurement_func_eli(elref.data,omega,esk,dsm,distance_earth_moon_km,dom)

        prop = punpy.MCPropagation(1000)

        unc = prop.propagate_random(eli.measurement_func_eli,
                                    [elref.data,omega,esk,dsm,distance_earth_moon_km,dom],
                                    [elref.uncertainties,None,None,None,None,None])

        return lunar_irr

    def calculate_polar(self,
            md: MoonData,
            polar_coeff: CimelCoef,
            ) -> SpectralData:
        dl = dolp.DOLP()
        if not isinstance(md,list):
            polarizations = dl.get_polarized(self.wlens,md.mpa_degrees,polar_coeff)

        else:
            polarizations = [dl.get_polarized(self.wlens,m.mpa_degrees,polar_coeff) for m in md]

        ds_intp = SpectralData.make_polarization_ds(self.wlens,polarizations,
                                                    None)

        spectral_data = SpectralData(polar_coeff.wlen,polarizations,
                                     None,ds_intp)

        return spectral_data


    @staticmethod
    def _get_data_eli_cimel(md: MoonData,cimel_coeff: CimelCoef,
                            calc_uncertainty: bool = True):
        elis_cimel = eli.calculate_eli_band(cimel_coeff,md)
        u_elis_cimel = None
        if calc_uncertainty:
            u_elis_cimel = eli.calculate_eli_band_unc(cimel_coeff,md)
        spectral_data = SpectralData(cimel_coeff.wlen,elis_cimel,u_elis_cimel)
        return spectral_data


    @staticmethod
    def _get_data_elref_cimel(md: MoonData,cimel_coeff: CimelCoef,
                              calc_uncertainty: bool = True):
        """

        :param md:
        :type md:
        :param cimel_coeff:
        :type cimel_coeff:
        :param calc_uncertainty:
        :type calc_uncertainty:
        :return:
        :rtype:
        """

        if not isinstance(md,list):
            elrefs_cimel = elref.band_moon_disk_reflectance(cimel_coeff,md)
            u_elrefs_cimel = None
            if calc_uncertainty:
                u_elrefs_cimel = elref.band_moon_disk_reflectance_unc(cimel_coeff,md)
        else:
            elrefs_cimel = [elref.band_moon_disk_reflectance(cimel_coeff,m) for m in md]
            u_elrefs_cimel = None
            if calc_uncertainty:
                u_elrefs_cimel = [elref.band_moon_disk_reflectance_unc(cimel_coeff,m) for m in md]

        ds_cimel = SpectralData.make_reflectance_ds(cimel_coeff.wlen,elrefs_cimel,u_elrefs_cimel)

        spectral_data = SpectralData(cimel_coeff.wlen,elrefs_cimel,u_elrefs_cimel,ds_cimel)

        return spectral_data