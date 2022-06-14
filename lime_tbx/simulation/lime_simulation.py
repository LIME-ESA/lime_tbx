"""Module containing the class that contains the state of the simulation."""

"""___Built-In Modules___"""
from typing import List, Union

"""___Third-Party Modules___"""
import punpy
import numpy as np

"""___NPL Modules___"""
from lime_tbx.datatypes.datatypes import (
    ApolloIrradianceCoefficients,
    MoonData,
    Point,
    PolarizationCoefficients,
    SpectralResponseFunction,
    SurfacePoint,
    CustomPoint,
    SpectralData,
    ReflectanceCoefficients,
)

from lime_tbx.lime_algorithms.rolo import eli, elref, rolo
from lime_tbx.lime_algorithms.dolp import dolp
from lime_tbx.interpolation.spectral_interpolation.spectral_interpolation import (
    SpectralInterpolation,
)
from lime_tbx.spice_adapter.spice_adapter import SPICEAdapter
from lime_tbx.simulation.moon_data_factory import MoonDataFactory
from lime_tbx.spectral_integration.spectral_integration import SpectralIntegration

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/06/2022"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


class LimeSimulation:
    """
    Class for running the main lime-tbx functionality

    Contains the state of the simulation, so it can be implemented efficiently.
    """

    def __init__(
        self,
        eocfi_path: str,
        kernels_path: str,
    ):
        """
        Parameters
        ----------
        eocfi_path: str
            Path where the folder with the needed EOCFI data files is located.
        kernels_path: str
            Path where the folder with the needed SPICE kernel files is located.
        """
        self.kernels_path = kernels_path
        self.eocfi_path = eocfi_path

        self.mds: Union[MoonData, List[MoonData]] = []
        self.wlens: List[float] = []
        self.elref: SpectralData = None
        self.elis: SpectralData = None
        self.signals: SpectralData = None
        self.elref_cimel: SpectralData = None
        self.elref_asd: SpectralData = None
        self.elis_cimel: SpectralData = None
        self.elis_asd: SpectralData = None
        self.polars: SpectralData = None
        self.srf: SpectralResponseFunction = None
        self.refl_uptodate = False
        self.irr_uptodate = False
        self.pol_uptodate = False
        self.signals_uptodate = False
        self.srf_updtodate = False
        self.mds_uptodate = False
        self.intp = SpectralInterpolation()

    def set_simulation_changed(self):
        """
        Marks the current data as not valid. It should be updated.
        """
        self.refl_uptodate = False
        self.irr_uptodate = False
        self.pol_uptodate = False
        self.signals_uptodate = False
        self.srf_updtodate = False
        self.mds_uptodate = False

    def _save_parameters(self, srf: SpectralResponseFunction, point: Point):
        if not self.mds_uptodate:
            self.mds = MoonDataFactory.get_md(point, self.eocfi_path, self.kernels_path)
            self.mds_uptodate = True
        if not self.srf_updtodate:
            self.srf = srf
            self.wlens = srf.get_wavelengths()
            self.srf_updtodate = True

    def update_reflectance(
        self,
        srf: SpectralResponseFunction,
        point: Point,
        cimel_coeff: ReflectanceCoefficients,
    ):
        self._save_parameters(srf, point)
        if not self.refl_uptodate:
            self.elref_cimel = self._calculate_elref(cimel_coeff)
            self.elref_asd = self.intp.get_best_asd_reference(self.mds)
            self.elref = self._interpolate_refl(self.elref_asd, self.elref_cimel)
            self.refl_uptodate = True

    def update_irradiance(
        self,
        srf: SpectralResponseFunction,
        point: Point,
        cimel_coeff: ReflectanceCoefficients,
    ):
        self._save_parameters(srf, point)
        if not self.refl_uptodate:
            self.update_reflectance(self.srf, point, cimel_coeff)

        if not self.irr_uptodate:
            self.elis = self._calculate_eli_from_elref(self.elref)
            self.elis_cimel = self._calculate_eli_from_elref(self.elref_cimel)
            self.elis_asd = self._calculate_eli_from_elref(self.elref_asd)
            self.irr_uptodate = True

        if not self.signals_uptodate:
            self.signals = self._calculate_signals(self.srf)
            self.signals_uptodate = True

    def update_polarization(
        self,
        srf: SpectralResponseFunction,
        point: Point,
        polar_coeff: PolarizationCoefficients,
    ):
        self._save_parameters(srf, point)
        if not self.pol_uptodate:
            self.polars = self._calculate_polar(polar_coeff)
            self.pol_uptodate = True

    def _interpolate_refl(
        self,
        asd_data: SpectralData,
        cimel_coeff: SpectralData,
    ) -> SpectralData:

        elrefs_intp = self.intp.get_interpolated_refl(
            cimel_coeff.wlens,
            cimel_coeff.data,
            asd_data.wlens,
            asd_data.data,
            self.wlens,
        )
        u_elrefs_intp = None
        u_elrefs_intp = (
            elrefs_intp * 0.01
        )  # intp.get_interpolated_refl_unc(wlen_cimel,elrefs_cimel,wlen_asd,elrefs_asd,wlens,u_elrefs_cimel,u_elrefs_asd)

        ds_intp = SpectralData.make_reflectance_ds(
            self.wlens, elrefs_intp, u_elrefs_intp
        )

        spectral_data = SpectralData(self.wlens, elrefs_intp, u_elrefs_intp, ds_intp)
        return spectral_data

    def _calculate_elref(
        self, cimel_coeff: ReflectanceCoefficients
    ) -> Union[SpectralData, List[SpectralData]]:
        """ """
        rl = rolo.ROLO()
        if not isinstance(self.mds, list):
            return rl.get_elrefs(cimel_coeff, self.mds)
        specs = []
        for m in self.mds:
            specs.append(rl.get_elrefs(cimel_coeff, m))
        return specs

    def _calculate_eli_from_elref(
        self, elref: SpectralData
    ) -> Union[SpectralData, List[SpectralData]]:
        """Calculation of Extraterrestrial Lunar Irradiance following Eq 3 in Roman et al., 2020

        Simulates a lunar observation for a wavelength for any observer/solar selenographic
        latitude and longitude. The irradiance is calculated in Wm⁻²/nm.

        Parameters
        ----------
        wavelength_nm : float
            Wavelength (in nanometers) of which the extraterrestrial lunar irradiance will be
            calculated.
        coefficients : IrradianceCoefficients
            Needed coefficients for the simulation.

        Returns
        -------
        elis: SpectralData | list of SpectralData
            The extraterrestrial lunar irradiance calculated, a list if moon_data is a list
        """
        rl = rolo.ROLO()
        if not isinstance(self.mds, list):
            return rl.get_elis_from_elrefs(elref, self.mds)
        specs = []
        for m in self.mds:
            specs.append(rl.get_elis_from_elrefs(elref, m))
        return specs

    def _calculate_polar(
        self,
        polar_coeff: PolarizationCoefficients,
    ) -> Union[SpectralData, List[SpectralData]]:
        dl = dolp.DOLP()
        if not isinstance(self.mds, list):
            polarizations = np.array(
                dl.get_polarized(self.wlens, self.mds.mpa_degrees, polar_coeff)
            )
            ds_pol = SpectralData.make_polarization_ds(self.wlens, polarizations, None)
            return SpectralData(self.wlens, polarizations, None, ds_pol)
        else:
            specs = []
            for m in self.mds:
                pol = dl.get_polarized(self.wlens, m.mpa_degrees, polar_coeff)
                ds_pol = SpectralData.make_polarization_ds(
                    self.wlens, polarizations, None
                )
                spectral_data = SpectralData(self.wlens, pol, None, ds_pol)
                specs.append(spectral_data)
        return specs

    def _calculate_signals(self, srf):
        signal = np.array(SpectralIntegration.integrate_elis(srf, self.elis.data))

        channel_ids = [srf.channels[i].id for i in range(len(srf.channels))]
        ds_pol = SpectralData.make_irradiance_ds(channel_ids, signal, None)

        spectral_data = SpectralData(channel_ids, signal, None, ds_pol)

        return spectral_data
