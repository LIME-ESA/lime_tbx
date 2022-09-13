"""Module containing the class that contains the state of the simulation."""

"""___Built-In Modules___"""
from typing import List, Union, Tuple
from abc import ABC, abstractmethod

"""___Third-Party Modules___"""
import numpy as np

"""___LIME_TBX Modules___"""
from lime_tbx.datatypes.datatypes import (
    LGLODData,
    LunarObservationWrite,
    MoonData,
    Point,
    PolarizationCoefficients,
    SatellitePoint,
    SpectralResponseFunction,
    SpectralData,
    ReflectanceCoefficients,
    KernelsPath,
    SurfacePoint,
)

from lime_tbx.lime_algorithms.rolo import rolo
from lime_tbx.lime_algorithms.dolp import dolp
from lime_tbx.interpolation.spectral_interpolation.spectral_interpolation import (
    SpectralInterpolation,
)
from lime_tbx.simulation.comparison import comparison
from lime_tbx.simulation.moon_data_factory import MoonDataFactory
from lime_tbx.spectral_integration.spectral_integration import SpectralIntegration

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/06/2022"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


class ILimeSimulation(ABC):
    """
    Interface for running the main lime-tbx functionality

    Should contain the state of the simulation, so it can be implemented efficiently.
    """

    @abstractmethod
    def set_simulation_changed(self):
        """
        Marks the current data as not valid. It should be updated.
        If it is marked as valid, it might not be updated even when instructed to.
        """
        pass

    @abstractmethod
    def update_reflectance(
        self,
        srf: SpectralResponseFunction,
        point: Point,
        cimel_coeff: ReflectanceCoefficients,
    ):
        """
        Updates the reflectance values if the stored values are not valid, using the given parameters.

        Parameters
        ----------
        srf: SpectralResponseFunction
            SRF for which the reflectance will be calculated.
        point: Point
            Point (location) for which the reflectance will be calculated.
        cimel_coeff: ReflectanceCoefficients
            Cimel Coefficients (and maybe more coeffs) used for the calculation of reflectance.
        """
        pass

    @abstractmethod
    def update_irradiance(
        self,
        srf: SpectralResponseFunction,
        signals_srf: SpectralResponseFunction,
        point: Point,
        cimel_coeff: ReflectanceCoefficients,
    ):
        """
        Updates the irradiance values if the stored value are not valid, using the given parameters.

        Parameters
        ----------
        srf: SpectralResponseFunction
            SRF for which the reflectance and irradiance will be calculated.
        signals_srf: SpectralResponseFunction
            SRF for which the integrated signal will be calculated.
        point: Point
            Point (location) for which the irradiance will be calculated.
        cimel_coeff: ReflectanceCoefficients
            Cimel Coefficients (and maybe more coeffs) used for the calculation of reflectance (needed for irradiance).
        """
        pass

    @abstractmethod
    def update_polarization(
        self,
        srf: SpectralResponseFunction,
        point: Point,
        polar_coeff: PolarizationCoefficients,
    ):
        """
        Updates the polarization values if the stored values are not valid, using the given parameters.

        Parameters
        ----------
        srf: SpectralResponseFunction
            SRF for which the polarization will be calculated.
        point: Point
            Point (location) for which the polarization will be calculated.
        polar_coeff: PolarizationCoefficients
            Coefficients used for the calculation of polarization.
        """
        pass

    @abstractmethod
    def get_elrefs(self) -> Union[SpectralData, List[SpectralData]]:
        """
        Returns the stored value for extraterrestrial lunar reflectance

        Returns
        -------
        elrefs: SpectralData | list of SpectralData
            Previously calculated reflectance/s
        """
        pass

    @abstractmethod
    def get_elis(self) -> Union[SpectralData, List[SpectralData]]:
        """
        Returns the stored value for extraterrestrial lunar irradiance

        Returns
        -------
        elrefs: SpectralData | list of SpectralData
            Previously calculated irradiance/s
        """
        pass

    @abstractmethod
    def get_signals(self) -> SpectralData:
        """
        Returns the stored value for integrated signals

        Returns
        -------
        signals: SpectralData
            Previously calculated integrated signal/s
        """
        pass

    @abstractmethod
    def get_elrefs_cimel(self) -> Union[SpectralData, List[SpectralData]]:
        """
        Returns the stored value for extraterrestrial lunar reflectance for the cimel wavelengths.

        Returns
        -------
        elrefs: SpectralData | list of SpectralData
            Previously calculated reflectance/s
        """
        pass

    @abstractmethod
    def get_elis_cimel(self) -> Union[SpectralData, List[SpectralData]]:
        """
        Returns the stored value for extraterrestrial lunar irradiance for the cimel wavelengths.

        Returns
        -------
        elrefs: SpectralData | list of SpectralData
            Previously calculated irradiance/s
        """
        pass

    @abstractmethod
    def get_elrefs_asd(self) -> Union[SpectralData, List[SpectralData]]:
        """
        Returns the stored value for extraterrestrial lunar reflectance for the asd spectrum.

        Returns
        -------
        elrefs: SpectralData | list of SpectralData
            Previously calculated reflectance/s
        """
        pass

    @abstractmethod
    def get_elis_asd(self) -> Union[SpectralData, List[SpectralData]]:
        """
        Returns the stored value for extraterrestrial lunar irradiance for the asd spectrum.

        Returns
        -------
        elrefs: SpectralData | list of SpectralData
            Previously calculated irradiance/s
        """
        pass

    @abstractmethod
    def get_polars(self) -> Union[SpectralData, List[SpectralData]]:
        """
        Returns the stored value for lunar polarization degree

        Returns
        -------
        elrefs: SpectralData | list of SpectralData
            Previously calculated polarization/s
        """
        pass

    @abstractmethod
    def get_surfacepoints(self) -> Union[SurfacePoint, List[SurfacePoint], None]:
        """
        Returns the Satellites points converted to the equivalent of surface points.
        In case they weren't Satellite Points, the behaviour is not defined.

        Returns
        -------
        surface_points: SurfacePoint | list of SurfacePoint | None
            Equivalent surface points
        """
        pass

    @abstractmethod
    def get_point(self) -> Point:
        """
        Returns the point that is being used in the simulation

        Returns
        -------
        point: Point
            The point used in the simulation.
        """
        pass


class LimeSimulation(ILimeSimulation):
    """
    Class for running the main lime-tbx functionality

    Contains the state of the simulation, so it can be implemented efficiently.
    """

    def __init__(
        self,
        eocfi_path: str,
        kernels_path: KernelsPath,
        MCsteps: int = 100,
        verbose: bool = True,
    ):
        """
        Parameters
        ----------
        eocfi_path: str
            Path where the folder with the needed EOCFI data files is located.
        kernels_path: KernelsPath
            Path where the folder with the needed SPICE kernel files is located.
        """
        self.kernels_path = kernels_path
        self.eocfi_path = eocfi_path

        self.mds: Union[MoonData, List[MoonData]] = []
        self.wlens: List[float] = []
        self.elref: Union[SpectralData, List[SpectralData]] = None
        self.elis: Union[SpectralData, List[SpectralData]] = None
        self.signals: SpectralData = None
        self.elref_cimel: Union[SpectralData, List[SpectralData]] = None
        self.elref_asd: Union[SpectralData, List[SpectralData]] = None
        self.elis_cimel: Union[SpectralData, List[SpectralData]] = None
        self.elis_asd: Union[SpectralData, List[SpectralData]] = None
        self.polars: Union[SpectralData, List[SpectralData]] = None
        self.srf: SpectralResponseFunction = None
        self.surfaces_of_sat: Tuple[SurfacePoint, List[SurfacePoint], None] = None
        self.point: Point = None
        self.refl_uptodate = False
        self.irr_uptodate = False
        self.pol_uptodate = False
        self.signals_uptodate = False
        self.srf_updtodate = False
        self.mds_uptodate = False
        self.intp = SpectralInterpolation(MCsteps=MCsteps)
        self.int = SpectralIntegration(MCsteps=MCsteps)
        self.verbose = verbose

    def set_simulation_changed(self):
        self.refl_uptodate = False
        self.irr_uptodate = False
        self.pol_uptodate = False
        self.signals_uptodate = False
        self.srf_updtodate = False
        self.mds_uptodate = False

    def _save_parameters(self, srf: SpectralResponseFunction, point: Point):
        if not self.mds_uptodate:
            if isinstance(point, SatellitePoint):
                (
                    self.mds,
                    self.surfaces_of_sat,
                ) = MoonDataFactory.get_md_and_surfaces_from_satellite(
                    point, self.eocfi_path, self.kernels_path
                )
            else:
                self.mds = MoonDataFactory.get_md(
                    point, self.eocfi_path, self.kernels_path
                )
            self.point = point
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
            if self.verbose:
                print("starting reflectance update")

            (
                self.elref_cimel,
                self.elref_asd,
                self.elref,
            ) = LimeSimulation._calculate_reflectances_values(
                cimel_coeff, self.mds, self.intp, self.wlens
            )
            self.refl_uptodate = True
            if self.verbose:
                print("reflectance update done")

    @staticmethod
    def _calculate_reflectances_values(
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

    @staticmethod
    def _calculate_irradiances_values(
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
            if self.verbose:
                print("starting irradiance update")
            (
                self.elis,
                self.elis_cimel,
                self.elis_asd,
            ) = LimeSimulation._calculate_irradiances_values(
                self.mds, self.elref, self.elref_cimel, self.elref_asd
            )
            self.irr_uptodate = True
            if self.verbose:
                print("irradiance update done")

        if not self.signals_uptodate:
            self.signals = self._calculate_signals(signals_srf)
            self.signals_uptodate = True
            if self.verbose:
                print("signals update done")

    def update_polarization(
        self,
        srf: SpectralResponseFunction,
        point: Point,
        polar_coeff: PolarizationCoefficients,
    ):
        self._save_parameters(srf, point)
        if not self.pol_uptodate:
            if self.verbose:
                print("starting polarisation update")
            self.polars = self._calculate_polar(polar_coeff)
            self.pol_uptodate = True
            if self.verbose:
                print("polarisation update done")

    @staticmethod
    def _interpolate_refl(
        asd_data: Union[SpectralData, List[SpectralData]],
        cimel_data: Union[SpectralData, List[SpectralData]],
        intp: SpectralInterpolation,
        wlens: List[float],
    ) -> Union[SpectralData, List[SpectralData]]:

        is_list = isinstance(cimel_data, list)
        if not is_list:
            cimel_data = [cimel_data]
            asd_data = [asd_data]  # both same length
        specs: Union[SpectralData, List[SpectralData]] = []
        for i, cf in enumerate(cimel_data):
            # wlens_valid = [
            #     wlen
            #     for wlen in wlens
            #     if wlen >= min(cf.wlens) and wlen <= max(cf.wlens)
            # ]
            # inf_wlens = [wlen for wlen in wlens if wlen < min(cf.wlens)]
            # sup_wlens = [wlen for wlen in wlens if wlen > max(cf.wlens)]
            elrefs_intp = intp.get_interpolated_refl(
                cf.wlens,
                cf.data,
                asd_data[i].wlens,
                asd_data[i].data,
                wlens,
            )
            # # 0s when the points cant be interpolated
            # elrefs_intp = np.concatenate(
            #     [
            #         np.zeros(len(inf_wlens), np.float64),
            #         elrefs_intp,
            #         np.zeros(len(sup_wlens), np.float64),
            #     ]
            # )

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
                self.wlens, polarizations, ds_pol.u_ran_polarization.values, ds_pol
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
                    self.wlens, polarizations, ds_pol.u_ran_polarization.values, ds_pol
                )
                specs.append(spectral_data)
        return specs

    def _calculate_signals(
        self,
        srf: SpectralResponseFunction,
    ) -> SpectralData:
        elis_signals = self.elis
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

    def set_observations(self, lglod: LGLODData, srf: SpectralResponseFunction):
        obss = lglod.observations
        self.elref = [obs.refls for obs in obss]
        self.elis = [obs.irrs for obs in obss]
        self.polars = [obs.polars for obs in obss]
        dts = [obs.dt for obs in obss]
        signals = lglod.signals
        ds_sign = SpectralData.make_signals_ds(
            signals.wlens,
            signals.data.T,
            signals.uncertainties.T,
        )
        self.signals = SpectralData(
            signals.wlens,
            signals.data.T,
            signals.uncertainties.T,
            ds_sign,
        )
        if not obss[0].sat_name or obss[0].sat_name == "":
            point = SurfacePoint(
                *comparison.to_llh(
                    obss[0].sat_pos.x, obss[0].sat_pos.y, obss[0].sat_pos.z
                ),
                dts
            )
        else:
            point = SatellitePoint(obss[0].sat_name, dts)
        self._save_parameters(srf, point)
        self.refl_uptodate = True
        self.irr_uptodate = True
        self.pol_uptodate = True
        self.signals_uptodate = True
        if self.verbose:
            print("observations loaded")

    def get_elrefs(self) -> Union[SpectralData, List[SpectralData]]:
        return self.elref

    def get_elis(self) -> Union[SpectralData, List[SpectralData]]:
        return self.elis

    def get_signals(self) -> SpectralData:
        return self.signals

    def get_elrefs_cimel(self) -> Union[SpectralData, List[SpectralData]]:
        return self.elref_cimel

    def get_elis_cimel(self) -> Union[SpectralData, List[SpectralData]]:
        return self.elis_cimel

    def get_elrefs_asd(self) -> Union[SpectralData, List[SpectralData]]:
        return self.elref_asd

    def get_elis_asd(self) -> Union[SpectralData, List[SpectralData]]:
        return self.elis_asd

    def get_polars(self) -> Union[SpectralData, List[SpectralData]]:
        return self.polars

    def get_surfacepoints(self) -> Union[SurfacePoint, List[SurfacePoint], None]:
        return self.surfaces_of_sat

    def get_point(self) -> Point:
        return self.point
