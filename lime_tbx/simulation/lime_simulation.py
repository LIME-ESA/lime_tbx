"""Module containing the class that contains the state of the simulation."""

"""___Built-In Modules___"""
from typing import List, Union, Tuple

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
        self.elref: Union[SpectralData, List[SpectralData]] = None
        self.elis: Union[SpectralData, List[SpectralData]] = None
        self.signals: Union[SpectralData, List[SpectralData]] = None
        self.elref_cimel: Union[SpectralData, List[SpectralData]] = None
        self.elref_asd: Union[SpectralData, List[SpectralData]] = None
        self.elis_cimel: Union[SpectralData, List[SpectralData]] = None
        self.elis_asd: Union[SpectralData, List[SpectralData]] = None
        self.polars: Union[SpectralData, List[SpectralData]] = None
        self.srf: SpectralResponseFunction = None
        self.refl_uptodate = False
        self.irr_uptodate = False
        self.pol_uptodate = False
        self.signals_uptodate = False
        self.srf_updtodate = False
        self.mds_uptodate = False
        self.intp = SpectralInterpolation()
        self.int = SpectralIntegration()

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

    @staticmethod
    def _get_reflectances_values(
        cimel_coeff: ReflectanceCoefficients,
        mds: Union[MoonData, List[MoonData]],
        intp: SpectralInterpolation,
        wlens: List[float],
    ) -> Tuple[
        Union[SpectralData, List[SpectralData]],
        Union[SpectralData, List[SpectralData]],
        Union[SpectralData, List[SpectralData]],
    ]:
        elref_cimel = LimeSimulation._calculate_elref(mds, cimel_coeff)
        if isinstance(mds, list):
            elref_asd = [intp.get_best_asd_reference(md) for md in mds]
        else:
            elref_asd = intp.get_best_asd_reference(mds)
        elref = LimeSimulation._interpolate_refl(elref_asd, elref_cimel, intp, wlens)
        return elref_cimel, elref_asd, elref

    def update_reflectance(
        self,
        srf: SpectralResponseFunction,
        point: Point,
        cimel_coeff: ReflectanceCoefficients,
    ):
        self._save_parameters(srf, point)
        if not self.refl_uptodate:
            (
                self.elref_cimel,
                self.elref_asd,
                self.elref,
            ) = LimeSimulation._get_reflectances_values(
                cimel_coeff, self.mds, self.intp, self.wlens
            )
            self.refl_uptodate = True

    @staticmethod
    def _get_irradiances_values(
        mds: Union[MoonData, List[MoonData]], elrefs, elref_cimel, elref_asd
    ):
        elis = LimeSimulation._calculate_eli_from_elref(mds, elrefs)
        elis_cimel = LimeSimulation._calculate_eli_from_elref(mds, elref_cimel)
        elis_asd = LimeSimulation._calculate_eli_from_elref(mds, elref_asd)
        return elis, elis_cimel, elis_asd

    def update_irradiance(
        self,
        srf: SpectralResponseFunction,
        signals_srf: SpectralResponseFunction,
        point: Point,
        cimel_coeff: ReflectanceCoefficients,
    ):
        self._save_parameters(srf, point)
        if not self.refl_uptodate:
            self.update_reflectance(self.srf, point, cimel_coeff)

        if not self.irr_uptodate:
            (
                self.elis,
                self.elis_cimel,
                self.elis_asd,
            ) = LimeSimulation._get_irradiances_values(
                self.mds, self.elref, self.elref_cimel, self.elref_asd
            )
            self.irr_uptodate = True

        if not self.signals_uptodate:
            self.signals = self._calculate_signals(signals_srf, cimel_coeff)
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

    @staticmethod
    def _interpolate_refl(
        asd_data: Union[SpectralData, List[SpectralData]],
        cimel_coeff: Union[SpectralData, List[SpectralData]],
        intp: SpectralInterpolation,
        wlens: List[float],
    ) -> Union[SpectralData, List[SpectralData]]:

        is_list = isinstance(cimel_coeff, list)
        if not is_list:
            cimel_coeff = [cimel_coeff]
            asd_data = [asd_data]  # both same length
        specs: Union[SpectralData, List[SpectralData]] = []
        for i, cf in enumerate(cimel_coeff):
            wlens_valid = [
                wlen
                for wlen in wlens
                if wlen >= min(cf.wlens) and wlen <= max(cf.wlens)
            ]
            inf_wlens = [wlen for wlen in wlens if wlen < min(cf.wlens)]
            sup_wlens = [wlen for wlen in wlens if wlen > max(cf.wlens)]
            elrefs_intp = intp.get_interpolated_refl(
                cf.wlens,
                cf.data,
                asd_data[i].wlens,
                asd_data[i].data,
                wlens_valid,
            )
            # 0s when the points cant be interpolated
            elrefs_intp = np.concatenate(
                [
                    np.zeros(len(inf_wlens), np.float64),
                    elrefs_intp,
                    np.zeros(len(sup_wlens), np.float64),
                ]
            )

            u_elrefs_intp = None
            u_elrefs_intp = (
                elrefs_intp * 0.01
            )  # intp.get_interpolated_refl_unc(wlen_cimel,elrefs_cimel,wlen_asd,elrefs_asd,wlens,u_elrefs_cimel,u_elrefs_asd)

            ds_intp = SpectralData.make_reflectance_ds(
                wlens, elrefs_intp, u_elrefs_intp
            )

            specs.append(SpectralData(wlens, elrefs_intp, u_elrefs_intp, ds_intp))
        if not is_list:
            specs = specs[0]
        return specs

    @staticmethod
    def _calculate_elref(
        mds: Union[MoonData, List[MoonData]], cimel_coeff: ReflectanceCoefficients
    ) -> Union[SpectralData, List[SpectralData]]:
        """ """
        rl = rolo.ROLO()
        if not isinstance(mds, list):
            return rl.get_elrefs(cimel_coeff, mds)
        specs = []
        for m in mds:
            specs.append(rl.get_elrefs(cimel_coeff, m))
        return specs

    @staticmethod
    def _calculate_eli_from_elref(
        mds: Union[MoonData, List[MoonData]],
        elrefs: Union[SpectralData, List[SpectralData]],
    ) -> Union[SpectralData, List[SpectralData]]:
        """Calculation of Extraterrestrial Lunar Irradiance following Eq 3 in Roman et al., 2020

        Simulates a lunar observation for a wavelength for any observer/solar selenographic
        latitude and longitude. The irradiance is calculated in Wm⁻²/nm.

        Parameters
        ----------
        mds: MoonData | list of MoonData
            moon datas
        elrefs: SpectralData | list of SpectralData
            elrefs previously calculated

        Returns
        -------
        elis: SpectralData | list of SpectralData
            The extraterrestrial lunar irradiance calculated, a list if self.mds is a list
        """
        rl = rolo.ROLO()
        if not isinstance(mds, list):
            return rl.get_elis_from_elrefs(elrefs, mds)
        specs = []
        for i, m in enumerate(mds):
            specs.append(rl.get_elis_from_elrefs(elrefs[i], m))
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
            return SpectralData(
                self.wlens, polarizations, ds_pol.u_ran_reflectance.values, ds_pol
            )
        else:
            specs = []
            for m in self.mds:
                polarizations = np.array(
                    dl.get_polarized(self.wlens, m.mpa_degrees, polar_coeff)
                )
                ds_pol = SpectralData.make_polarization_ds(
                    self.wlens, polarizations, None
                )
                spectral_data = SpectralData(
                    self.wlens, polarizations, ds_pol.u_ran_reflectance.values, ds_pol
                )
                specs.append(spectral_data)
        return specs

    def _calculate_signals(
        self,
        srf: SpectralResponseFunction,
        cimel_coeff: Union[SpectralData, List[SpectralData]],
    ) -> SpectralData:
        rl = rolo.ROLO()
        _, _, elrefs = LimeSimulation._get_reflectances_values(
            cimel_coeff, self.mds, self.intp, srf.get_wavelengths()
        )
        elis_signals = LimeSimulation._calculate_eli_from_elref(self.mds, elrefs)
        channel_ids = [srf.channels[i].id for i in range(len(srf.channels))]
        if not isinstance(elis_signals, list):
            elis_signals = [elis_signals]
        signals_list = []
        uncs_list = []
        for irr in elis_signals:
            signals_list.append(self.int.integrate_elis(srf, irr))
            uncs_list.append(self.int.u_integrate_elis(srf, irr))
        signals = np.array(signals_list).T
        uncs = np.array(uncs_list).T
        ds_pol = SpectralData.make_signals_ds(channel_ids, signals, uncs)
        sp_d = SpectralData(channel_ids, signals, uncs, ds_pol)
        return sp_d
