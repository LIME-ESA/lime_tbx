"""Module containing the class that contains the state of the simulation."""

"""___Built-In Modules___"""
from typing import List, Union, Tuple, Iterable, Callable
from abc import ABC, abstractmethod
import gc

"""___Third-Party Modules___"""
import numpy as np

"""___LIME_TBX Modules___"""
from lime_tbx.common.datatypes import (
    CustomPoint,
    LGLODData,
    MoonData,
    Point,
    PolarisationCoefficients,
    SatellitePoint,
    SpectralResponseFunction,
    SpectralData,
    ReflectanceCoefficients,
    KernelsPath,
    SurfacePoint,
    EocfiPath,
    AOLPCoefficients,
    MultipleCustomPoint,
)
from lime_tbx.common import constants
from lime_tbx.business.lime_algorithms.lime import lime
from lime_tbx.business.lime_algorithms import polar
from lime_tbx.business.interpolation.spectral_interpolation.spectral_interpolation import (
    SpectralInterpolation,
)
from lime_tbx.business.interpolation.interp_data import interp_data
from lime_tbx.application.simulation.moon_data_factory import MoonDataFactory
from lime_tbx.business.spectral_integration.spectral_integration import (
    SpectralIntegration,
)
from lime_tbx.business.spice_adapter.spice_adapter import SPICEAdapter
from lime_tbx.presentation.gui.settings import ISettingsManager

"""___Authorship___"""
__author__ = "Pieter De Vis, Jacob Fahy, Javier Gatón Herguedas, Ramiro González Catón, Carlos Toledano"
__created__ = "01/06/2022"
__maintainer__ = "Pieter De Vis, Javier Gatón Herguedas"
__email__ = "pieter.de.vis@npl.co.uk, gaton@goa.uva.es"
__status__ = "Development"


def is_ampa_valid_range(ampa: float) -> bool:
    """
    Checks if the value of the absolute moon phase angle is inside the valid range for the simulation.

    Parameters
    ----------
    ampa: float
        Absolute moon phase angle in degrees

    Returns
    -------
    valid_range: bool
        True if the angle is inside the valid range for the simulation.
    """
    return 2 <= ampa <= 90


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
    def clear_srf(self):
        """
        Dereference the SRF object from the LimeSimulation, allowing the garbage collector to remove it.
        """
        pass

    @abstractmethod
    def is_skipping_uncs(self) -> bool:
        """
        Checks if the current simulation instance is skipping the uncertainties calculation.

        Returns
        -------
        skip: bool
            True if the uncertainties are being skipped.
        """
        pass

    @abstractmethod
    def update_reflectance(
        self,
        srf: SpectralResponseFunction,
        point: Point,
        cimel_coeff: ReflectanceCoefficients,
        callback_observation: Callable = None,
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
        callback_observation: Callable
            Callback that will emit a signal when a simulation is done
        """
        pass

    @abstractmethod
    def update_irradiance(
        self,
        srf: SpectralResponseFunction,
        signals_srf: SpectralResponseFunction,
        point: Point,
        cimel_coeff: ReflectanceCoefficients,
        callback_observation: Callable = None,
        mda_precalculated: MoonData = None,
    ):
        """
        Updates the irradiance values if the stored value are not valid, using the given parameters.

        Parameters
        ----------
        srf: SpectralResponseFunction
            SRF for which the reflectance and irradiance will be calculated.
        srf: SpectralResponseFunction
            SRF for which the integrated signal will be calculated.
        point: Point
            Point (location) for which the irradiance will be calculated.
        cimel_coeff: ReflectanceCoefficients
            Cimel Coefficients (and maybe more coeffs) used for the calculation of reflectance (needed for irradiance).
        """
        pass

    @abstractmethod
    def update_polarisation(
        self,
        srf: SpectralResponseFunction,
        point: Point,
        polar_coeff: PolarisationCoefficients,
        callback_observation: Callable = None,
    ):
        """
        Updates the polarisation values if the stored values are not valid, using the given parameters.

        Parameters
        ----------
        srf: SpectralResponseFunction
            SRF for which the polarisation will be calculated.
        point: Point
            Point (location) for which the polarisation will be calculated.
        polar_coeff: PolarisationCoefficients
            Coefficients used for the calculation of polarisation.
        """
        pass

    @abstractmethod
    def update_aolp(
        self,
        srf: SpectralResponseFunction,
        point: Point,
        aolp_coeff: AOLPCoefficients,
        callback_observation: Callable = None,
    ):
        """
        Updates the aolp values if the stored values are not valid, using the given parameters.

        Parameters
        ----------
        srf: SpectralResponseFunction
            SRF for which the aolp will be calculated.
        point: Point
            Point (location) for which the aolp will be calculated.
        aolp_coeff: AOLPCoefficients
            Coefficients used for the calculation of aolp.
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
        Returns the stored value for lunar polarisation degree

        Returns
        -------
        polars: SpectralData | list of SpectralData
            Previously calculated polarisation/s
        """
        pass

    @abstractmethod
    def get_polars_cimel(self) -> Union[SpectralData, List[SpectralData]]:
        """
        Returns the stored value for lunar polarisation degree for the cimel wavelengths

        Returns
        -------
        polars_cimel: SpectralData | list of SpectralData
            Previously calculated polarisation/s
        """
        pass

    @abstractmethod
    def get_polars_asd(self) -> Union[SpectralData, List[SpectralData]]:
        """
        Returns the stored value for lunar polarisation degree for the asd spectrum.

        Returns
        -------
        polars_asd: SpectralData | list of SpectralData
            Previously calculated polarisation/s
        """
        pass

    @abstractmethod
    def get_aolp(self) -> Union[SpectralData, List[SpectralData]]:
        """
        Returns the stored value for lunar angle of linear polarisation

        Returns
        -------
        aolp: SpectralData | list of SpectralData
            Previously calculated polarisation angle/s
        """
        pass

    @abstractmethod
    def get_aolp_cimel(self) -> Union[SpectralData, List[SpectralData]]:
        """
        Returns the stored value for lunar angle of linear polarisation for the cimel wavelengths

        Returns
        -------
        aolp_cimel: SpectralData | list of SpectralData
            Previously calculated polarisation angle/s
        """
        pass

    @abstractmethod
    def get_aolp_asd(self) -> Union[SpectralData, List[SpectralData]]:
        """
        Returns the stored value for lunar angle of linear polarisation for the asd spectrum.

        Returns
        -------
        polars_asd: SpectralData | list of SpectralData
            Previously calculated polarisation angle/s
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

    @abstractmethod
    def get_moon_datas(self) -> Union[MoonData, List[MoonData]]:
        """
        Returns the moon datas for the current point

        Returns
        -------
        mds: MoonData | list of MoonData
            MoonData/s of the current point.
        """
        pass

    @abstractmethod
    def are_mpas_inside_mpa_range(self) -> Union[bool, List[bool]]:
        """
        Returns a list of values indicating if the moon datas contain a valid mpa range.

        Returns
        -------
        are_valid: bool | list of bool
            True/s if the MoonData/s contains mpa values in the valid range.
        """
        pass

    @abstractmethod
    def set_observations(self, lglod: LGLODData, srf: SpectralResponseFunction):
        """
        Loads a set of observations and the relative SRF into the lime simulation

        Parameters
        ----------
        lglod: LGLODData
            Observations to be loaded
        srf: SpectralResponseFunction
            SRF related to those observations
        """
        pass

    @abstractmethod
    def will_irradiance_calculate_reflectance_previously(self, pt: Point) -> bool:
        """Returns if the irradiance calculation function would perform the reflectance
        calculation monolitically as a previous step or not given the current state.

        Parameters
        ----------
        pt: Point
            Point for which the information wants to be known.

        Returns
        -------
        will_calculate: bool
            True if the reflectance would be calculated previously.
        """
        pass

    @abstractmethod
    def will_irradiance_calculate_reflectance_simultaneously(
        self, pt: Point = None
    ) -> bool:
        """Returns if the irradiance calculation function would perform the reflectance
        calculation simultaneously avoiding storing the error correlation matrices,
        given the current state.

        Parameters
        ----------
        pt: Point
            Point for which the information wants to be known.
            If not specified the currently loaded one will be used.

        Returns
        -------
        will_calculate: bool
            True if the reflectance would be calculated simultaneously.
        """
        pass

    @abstractmethod
    def is_polarisation_updated(self) -> bool:
        """Returns if the polarisation has been updated. If not, or nothing has been executed,
        or the spectrum doesn't contain polarisation values.

        Returns
        -------
        pol_uptodate: bool
            True if the polarisation is updated
        """
        pass

    @abstractmethod
    def is_aolp_updated(self) -> bool:
        """Returns if the AoLP has been updated. If not, or nothing has been executed,
        or the spectrum doesn't contain AoLP values.

        Returns
        -------
        aolp_uptodate: bool
            True if the v is updated
        """
        pass


class LimeSimulation(ILimeSimulation):
    """
    Class for running the main lime-tbx functionality

    Contains the state of the simulation, so it can be implemented efficiently.
    """

    def __init__(
        self,
        eocfi_path: EocfiPath,
        kernels_path: KernelsPath,
        settings_manager: ISettingsManager,
        MCsteps: int = 100,
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        eocfi_path: EocfiPath
            Path where the folder with the needed EOCFI data files is located.
        kernels_path: KernelsPath
            Path where the folder with the needed SPICE kernel files is located.
        """
        self.kernels_path = kernels_path
        self.eocfi_path = eocfi_path
        self.settings_manager = settings_manager

        self.mds: Union[MoonData, List[MoonData]] = []
        self.wlens: List[float] = []
        self.elref: Union[SpectralData, List[SpectralData]] = None
        self.elis: Union[SpectralData, List[SpectralData]] = None
        self.polars: Union[SpectralData, List[SpectralData]] = None
        self.aolp: Union[SpectralData, List[SpectralData]] = None
        self.signals: SpectralData = None
        self.elref_cimel: Union[SpectralData, List[SpectralData]] = None
        self.elref_asd: Union[SpectralData, List[SpectralData]] = None
        self.elis_cimel: Union[SpectralData, List[SpectralData]] = None
        self.elis_asd: Union[SpectralData, List[SpectralData]] = None
        self.polars_cimel: Union[SpectralData, List[SpectralData]] = None
        self.polars_asd: Union[SpectralData, List[SpectralData]] = None
        self.aolp_cimel: Union[SpectralData, List[SpectralData]] = None
        self.aolp_asd: Union[SpectralData, List[SpectralData]] = None
        self.srf: SpectralResponseFunction = None
        self.surfaces_of_sat: Tuple[SurfacePoint, List[SurfacePoint], None] = None
        self.point: Point = None
        self.refl_uptodate = False
        self.irr_uptodate = False
        self.pol_uptodate = False
        self.aolp_uptodate = False
        self.signals_uptodate = False
        self.srf_updtodate = False
        self.mds_uptodate = False
        self.intp = SpectralInterpolation(MCsteps=MCsteps)
        self.int = SpectralIntegration(MCsteps=MCsteps)
        self.verbose = verbose
        self._skip_uncs: bool = None
        self._interp_srf_name: str = None

    def set_simulation_changed(self):
        self.refl_uptodate = False
        self.irr_uptodate = False
        self.pol_uptodate = False
        self.aolp_uptodate = False
        self.signals_uptodate = False
        self.srf_updtodate = False
        self.mds_uptodate = False
        self._skip_uncs = None
        self._interp_srf_name = None
        self.settings_manager.set_coef_version_name(None)

    def clear_srf(self):
        self.srf = None
        self.set_simulation_changed()

    def _save_parameters(
        self,
        srf: SpectralResponseFunction,
        point: Point,
        mda_precalculated: MoonData = None,
    ):
        if not self.mds_uptodate:
            if isinstance(point, SatellitePoint):
                (
                    self.mds,
                    self.surfaces_of_sat,
                ) = MoonDataFactory.get_md_and_surfaces_from_satellite(
                    point, self.eocfi_path, self.kernels_path
                )
            else:
                if mda_precalculated:
                    self.mds = mda_precalculated
                else:
                    self.mds = MoonDataFactory.get_md(
                        point, self.eocfi_path, self.kernels_path
                    )
            self.point = point
            self.mds_uptodate = True
        if not self.srf_updtodate:
            self.srf = srf
            self.wlens = [*set(srf.get_wavelengths())]
            self.wlens = [
                x
                for x in self.wlens
                if x >= constants.MIN_WLEN and x <= constants.MAX_WLEN
            ]
            self.wlens.sort()
            self.srf_updtodate = True

    def is_skipping_uncs(self) -> bool:
        if self._skip_uncs is None:
            self._skip_uncs = interp_data.is_skip_uncertainties()
        return self._skip_uncs

    def get_interp_srf_name(self) -> str:
        if self._interp_srf_name is None:
            self._interp_srf_name = interp_data.get_interpolation_srf_as_srf_type()
        return self._interp_srf_name

    def update_reflectance(
        self,
        srf: SpectralResponseFunction,
        point: Point,
        cimel_coeff: ReflectanceCoefficients,
        callback_observation: Callable = None,
    ):
        self._save_parameters(srf, point)
        skip_uncs = self.is_skipping_uncs()
        if not self.refl_uptodate:
            if self.verbose:
                print("starting reflectance update")
            self.elref_asd = self.elref_cimel = self.elref = None
            keep_err_corr_mats = (
                not self.will_irradiance_calculate_reflectance_simultaneously()
            )
            (
                self.elref_cimel,
                self.elref_asd,
                self.elref,
            ) = LimeSimulation._calculate_reflectances_values(
                cimel_coeff,
                self.mds,
                self.intp,
                self.wlens,
                skip_uncs,
                callback_observation,
                keep_err_corr_mats,
            )
            interm_res_path = self.settings_manager.get_intermediate_results_path()
            if interm_res_path:
                np.savetxt(
                    f"{interm_res_path}/refl_cimel.csv",
                    self.elref_cimel.data,
                    delimiter=",",
                )
                np.savetxt(
                    f"{interm_res_path}/refl_interp_spectrum.csv",
                    np.array([self.elref.wlens, self.elref.data]).T,
                    fmt=["%.2f", "%e"],
                    delimiter=",",
                )
            self.refl_uptodate = True
            if self.verbose:
                print("reflectance update done")

    @staticmethod
    def _calculate_elref_eli_and_integrate(
        mds: Union[MoonData, List[MoonData]],
        cimel_coeff: ReflectanceCoefficients,
        wlens: List[float],
        interp_srf_name: str,
        intp: SpectralInterpolation,
        int: SpectralIntegration,
        signals_srf: SpectralResponseFunction,
        skip_uncs: bool,
        callback_observation: Callable,
        show_interp_spectrum: bool,
        use_wehrli: bool = False,
    ) -> Tuple[
        Union[SpectralData, List[SpectralData]],
        SpectralData,
        Union[SpectralData, List[SpectralData]],
        Union[SpectralData, List[SpectralData]],
        SpectralData,
    ]:
        elref_cimel = LimeSimulation._calculate_elref(mds, cimel_coeff, skip_uncs, True)
        if isinstance(mds, list):
            elref_asd = (intp.get_best_interp_reference(md) for md in mds)
            ret_asd = intp.get_best_interp_reference(mds[0])
        else:
            elref_asd = intp.get_best_interp_reference(mds)
            ret_asd = elref_asd
        elis_cimel, elis_asd = LimeSimulation._calculate_irradiances_values(
            mds,
            elref_cimel,
            ret_asd,
            skip_uncs,
            show_interp_spectrum,
            use_wehrli,
        )
        if use_wehrli:
            interp_srf_name = "asd_wehrli"
        elref, elis, signals = LimeSimulation._interpolate_refl_calc_irr_signal(
            elref_asd,
            elref_cimel,
            intp,
            wlens,
            mds,
            interp_srf_name,
            signals_srf,
            int,
            skip_uncs,
            callback_observation,
        )
        return elref_cimel, ret_asd, elref, elis_cimel, elis_asd, elis, signals

    @staticmethod
    def _calculate_reflectances_values(
        cimel_coeff: ReflectanceCoefficients,
        mds: Union[MoonData, List[MoonData]],
        intp: SpectralInterpolation,
        wlens: List[float],
        skip_uncs: bool,
        callback_observation: Callable,
        keep_err_corr_mats: bool,
    ) -> Tuple[
        Union[SpectralData, List[SpectralData]],
        SpectralData,
        Union[SpectralData, List[SpectralData]],
    ]:
        elref_cimel = LimeSimulation._calculate_elref(
            mds, cimel_coeff, skip_uncs, keep_err_corr_mats
        )
        if isinstance(mds, list):
            elref_asd = (intp.get_best_interp_reference(md) for md in mds)
            ret_asd = intp.get_best_interp_reference(mds[0])
        else:
            elref_asd = intp.get_best_interp_reference(mds)
            ret_asd = elref_asd
        elref = LimeSimulation._interpolate_refl(
            elref_asd,
            elref_cimel,
            intp,
            wlens,
            skip_uncs,
            callback_observation,
            keep_err_corr_mats,
        )
        return elref_cimel, ret_asd, elref

    @staticmethod
    def _calculate_irradiances_values(
        mds: Union[MoonData, List[MoonData]],
        elref_cimel,
        elref_asd,
        skip_uncs: bool,
        show_interp_spectrum: bool,
        use_wehrli: bool = False,
    ) -> Tuple[Union[SpectralData, List[SpectralData]], SpectralData,]:
        _cimel_srf = "cimel" if not use_wehrli else "cimel_wehrli"
        elis_cimel = LimeSimulation._calculate_eli_from_elref(
            mds, elref_cimel, _cimel_srf, skip_uncs
        )
        elis_asd = None
        if show_interp_spectrum:
            md = mds
            if isinstance(md, list):
                md = md[0]
            elis_asd = LimeSimulation._calculate_eli_from_elref(
                md, elref_asd, "asd", skip_uncs
            )
        return elis_cimel, elis_asd

    @staticmethod
    def _calculate_polarisation_values(
        mds: Union[MoonData, List[MoonData]],
        polar_coeff: PolarisationCoefficients,
        intp: SpectralInterpolation,
        wlens: List[float],
        skip_uncs: bool,
        callback_observation: Callable,
    ) -> Tuple[
        Union[SpectralData, List[SpectralData]],
        SpectralData,
        Union[SpectralData, List[SpectralData]],
    ]:
        polar_cimel = LimeSimulation._calculate_polar(mds, polar_coeff, skip_uncs)
        if isinstance(mds, list):
            polar_asd = (intp.get_best_polar_interp_reference(md) for md in mds)
            ret_polar_asd = intp.get_best_polar_interp_reference(mds[0])
        else:
            polar_asd = intp.get_best_polar_interp_reference(mds)
            ret_polar_asd = polar_asd
        polar = LimeSimulation._interpolate_polar(
            polar_asd,
            polar_cimel,
            intp,
            wlens,
            True,
            callback_observation,
        )
        return polar_cimel, ret_polar_asd, polar

    @staticmethod
    def _calculate_aolp_values(
        mds: Union[MoonData, List[MoonData]],
        aolp_coeff: AOLPCoefficients,
        intp: SpectralInterpolation,
        wlens: List[float],
        skip_uncs: bool,
        callback_observation: Callable,
    ) -> Tuple[
        Union[SpectralData, List[SpectralData]],
        SpectralData,
        Union[SpectralData, List[SpectralData]],
    ]:
        aolp_cimel = LimeSimulation._calculate_aolp(mds, aolp_coeff, skip_uncs)
        if isinstance(mds, list):
            aolp_asd = (intp.get_best_aolp_interp_reference(md) for md in mds)
            ret_aolp_asd = intp.get_best_aolp_interp_reference(mds[0])
        else:
            aolp_asd = intp.get_best_aolp_interp_reference(mds)
            ret_aolp_asd = aolp_asd
        aolp = LimeSimulation._interpolate_aolp(
            aolp_asd,
            aolp_cimel,
            intp,
            wlens,
            True,
            callback_observation,
        )
        return aolp_cimel, ret_aolp_asd, aolp

    def _update_irradiance_and_reflectance(
        self,
        srf: SpectralResponseFunction,
        signals_srf: SpectralResponseFunction,
        point: Point,
        cimel_coeff: ReflectanceCoefficients,
        callback_observation: Callable = None,
    ):
        self._save_parameters(srf, point)
        skip_uncs = self.is_skipping_uncs()
        interp_srf_name = self.get_interp_srf_name()
        if self.verbose:
            print("Starting irradiance & reflectance update")
        (
            self.elref_cimel,
            self.elref_asd,
            self.elref,
            self.elis_cimel,
            self.elis_asd,
            self.elis,
            self.signals,
        ) = LimeSimulation._calculate_elref_eli_and_integrate(
            self.mds,
            cimel_coeff,
            self.wlens,
            interp_srf_name,
            self.intp,
            self.int,
            signals_srf,
            skip_uncs,
            callback_observation,
            self.settings_manager.is_show_interp_spectrum(),
            self.settings_manager.get_use_wehrli_for_esi(),
        )
        if self.verbose:
            print("Irradiance, reflectance and signal update done")
        self.irr_uptodate = True
        self.signals_uptodate = True
        self.refl_uptodate = True

    def update_irradiance(
        self,
        srf: SpectralResponseFunction,
        signals_srf: SpectralResponseFunction,
        point: Point,
        cimel_coeff: ReflectanceCoefficients,
        callback_observation: Callable = None,
        mda_precalculated: MoonData = None,
    ):
        self._save_parameters(srf, point, mda_precalculated)
        if self.will_irradiance_calculate_reflectance_simultaneously():
            if not self.irr_uptodate or not self.signals_uptodate:
                self._update_irradiance_and_reflectance(
                    srf, signals_srf, point, cimel_coeff, callback_observation
                )
        else:
            skip_uncs = self.is_skipping_uncs()
            interp_srf_name = self.get_interp_srf_name()
            if not self.irr_uptodate or not self.signals_uptodate:
                self.elis = self.signals = None
            if not self.refl_uptodate:
                self.update_reflectance(
                    self.srf, point, cimel_coeff, callback_observation
                )
            if not self.irr_uptodate or not self.signals_uptodate:
                if self.verbose:
                    print("starting irradiance update")
                use_wehrli = self.settings_manager.get_use_wehrli_for_esi()
                (
                    self.elis_cimel,
                    self.elis_asd,
                ) = LimeSimulation._calculate_irradiances_values(
                    self.mds,
                    self.elref_cimel,
                    self.elref_asd,
                    skip_uncs,
                    self.settings_manager.is_show_interp_spectrum(),
                    use_wehrli,
                )
                if use_wehrli:
                    interp_srf_name = "asd_wehrli"
                self.irr_uptodate = True
                if self.verbose:
                    print("auxiliar irradiance update done")
                (
                    self.elis,
                    self.signals,
                ) = self._calculate_eli_from_elref_and_integrate(
                    self.mds,
                    self.elref,
                    interp_srf_name,
                    signals_srf,
                    skip_uncs,
                    callback_observation,
                )
                interm_res_path = self.settings_manager.get_intermediate_results_path()
                if interm_res_path:
                    np.savetxt(
                        f"{interm_res_path}/irr_from_refl_cimel.csv",
                        self.elis_cimel.data,
                        delimiter=",",
                    )
                    np.savetxt(
                        f"{interm_res_path}/irr_from_interp_spectrum.csv",
                        np.array([self.elis.wlens, self.elis.data]).T,
                        fmt=["%.2f", "%e"],
                        delimiter=",",
                    )
                    np.savetxt(
                        f"{interm_res_path}/signals_from_irr_interp_srf_integrated.csv",
                        self.signals.data,
                        delimiter=",",
                    )
                # Free up space
                elrefclearer = self.elref
                if not isinstance(elrefclearer, list):
                    elrefclearer = [elrefclearer]
                for elr in elrefclearer:
                    elr.clear_err_corr()
                #
                self.signals_uptodate = True
                if self.verbose:
                    print("irradiance & signals update done")

    def will_irradiance_calculate_reflectance_previously(self, pt: Point) -> bool:
        return (
            not self.refl_uptodate
            and not self.will_irradiance_calculate_reflectance_simultaneously(pt)
        )

    def will_irradiance_calculate_reflectance_simultaneously(
        self, pt: Point = None
    ) -> bool:
        if pt is None:
            pt = self.point
        lenvals = 1
        if not self.is_skipping_uncs():
            if isinstance(pt, (SurfacePoint, SatellitePoint)):
                if isinstance(pt.dt, list):
                    lenvals = len(pt.dt)
            elif isinstance(pt, MultipleCustomPoint):
                lenvals = len(pt.pts)
            if lenvals > constants.MAX_LIMIT_REFL_ERR_CORR_ARE_STORED:
                return True
        return False

    def is_polarisation_updated(self) -> bool:
        return self.pol_uptodate

    def is_aolp_updated(self) -> bool:
        return self.aolp_uptodate

    def update_polarisation(
        self,
        srf: SpectralResponseFunction,
        point: Point,
        polar_coeff: PolarisationCoefficients,
        callback_observation: Callable = None,
    ):
        skip_uncs = self.is_skipping_uncs()
        self._save_parameters(srf, point)
        if not self.pol_uptodate:
            if self.verbose:
                print("starting polarisation update")
            self.polars = self.polars_asd = self.polars_cimel = None
            (
                self.polars_cimel,
                self.polars_asd,
                self.polars,
            ) = LimeSimulation._calculate_polarisation_values(
                self.mds,
                polar_coeff,
                self.intp,
                self.wlens,
                skip_uncs,
                callback_observation,
            )
            self.pol_uptodate = True
            if self.verbose:
                print("polarisation update done")

    def update_aolp(
        self,
        srf: SpectralResponseFunction,
        point: Point,
        aolp_coeff: AOLPCoefficients,
        callback_observation: Callable = None,
    ):
        skip_uncs = self.is_skipping_uncs()
        self._save_parameters(srf, point)
        if not self.aolp_uptodate:
            if self.verbose:
                print("starting aolp_asd update")
            self.aolp = self.aolp_asd = self.aolp_cimel = None
            (
                self.aolp_cimel,
                self.aolp_asd,
                self.aolp,
            ) = LimeSimulation._calculate_aolp_values(
                self.mds,
                aolp_coeff,
                self.intp,
                self.wlens,
                skip_uncs,
                callback_observation,
            )
            self.aolp_uptodate = True
            if self.verbose:
                print("aolp update done")

    @staticmethod
    def _interpolate_refl_calc_irr_signal(
        asd_data: Union[SpectralData, Iterable[SpectralData]],
        cimel_data: Union[SpectralData, List[SpectralData]],
        intp: SpectralInterpolation,
        wlens: List[float],
        mds: Union[MoonData, List[MoonData]],
        srf_type: str,
        srf: SpectralResponseFunction,
        int: SpectralIntegration,
        skip_uncs: bool,
        callback_observation: Callable,
    ) -> Union[SpectralData, List[SpectralData]]:
        wlens = np.array(wlens)
        is_list = isinstance(cimel_data, list)
        if not is_list:
            cimel_data = [cimel_data]
            asd_data = [asd_data]  # both same length
            mds = [mds]
        elref_specs: Union[SpectralData, List[SpectralData]] = []

        rl = lime.LIME()
        channel_ids = [srf.channels[i].id for i in range(len(srf.channels))]

        elis_specs = []
        signals = []
        uncs = []
        for cf, asdd, md in zip(cimel_data, asd_data, mds):
            if callback_observation:
                callback_observation()
            ds_intp = None
            if not skip_uncs:
                (
                    elrefs_intp,
                    u_elrefs_intp,
                    corr_elrefs_intp,
                ) = intp.get_interpolated_refl_unc(
                    cf.wlens,
                    cf.data,
                    asdd.wlens,
                    asdd.data,
                    wlens,
                    cf.uncertainties,
                    asdd.uncertainties,
                    cf.err_corr,
                    asdd.err_corr,
                )
                ds_intp = SpectralData.make_reflectance_ds(
                    wlens, elrefs_intp, u_elrefs_intp, corr_elrefs_intp
                )
            else:
                elrefs_intp = intp.get_interpolated_refl(
                    cf.wlens,
                    cf.data,
                    asdd.wlens,
                    asdd.data,
                    wlens,
                )
                u_elrefs_intp = np.zeros(elrefs_intp.shape)
            elref_spec = SpectralData(wlens, elrefs_intp, u_elrefs_intp, ds_intp)
            spec, sig, sigunc = rl.get_elis_from_elrefs_and_integrate(
                elref_spec, md, srf_type, int, srf, skip_uncs
            )
            elis_specs.append(spec)
            signals.append(sig)
            uncs.append(sigunc)
            elref_specs.append(SpectralData(wlens, elrefs_intp, u_elrefs_intp, None))
        if callback_observation:
            callback_observation()
        signals = np.array(signals).T
        uncs = np.array(uncs).T
        sp_d = SpectralData(channel_ids, signals, uncs, None)
        if not is_list:
            elref_specs = elref_specs[0]
            elis_specs = elis_specs[0]
        return elref_specs, elis_specs, sp_d

    @staticmethod
    def _interpolate_refl(
        asd_data: Union[SpectralData, Iterable[SpectralData]],
        cimel_data: Union[SpectralData, List[SpectralData]],
        intp: SpectralInterpolation,
        wlens: List[float],
        skip_uncs: bool,
        callback_observation: Callable,
        keep_err_corr_mats: bool,
    ) -> Union[SpectralData, List[SpectralData]]:
        wlens = np.array(wlens)
        is_list = isinstance(cimel_data, list)
        if not is_list:
            cimel_data = [cimel_data]
            asd_data = [asd_data]  # both same length
        specs: Union[SpectralData, List[SpectralData]] = []
        for cf, asdd in zip(cimel_data, asd_data):
            if callback_observation:
                callback_observation()
            ds_intp = None
            if not skip_uncs:
                (
                    elrefs_intp,
                    u_elrefs_intp,
                    corr_elrefs_intp,
                ) = intp.get_interpolated_refl_unc(
                    cf.wlens,
                    cf.data,
                    asdd.wlens,
                    asdd.data,
                    wlens,
                    cf.uncertainties,
                    asdd.uncertainties,
                    cf.err_corr,
                    asdd.err_corr,
                )
                if keep_err_corr_mats:
                    ds_intp = SpectralData.make_reflectance_ds(
                        wlens, elrefs_intp, u_elrefs_intp, corr_elrefs_intp
                    )
            else:
                elrefs_intp = intp.get_interpolated_refl(
                    cf.wlens,
                    cf.data,
                    asdd.wlens,
                    asdd.data,
                    wlens,
                )
                u_elrefs_intp = np.zeros(elrefs_intp.shape)
            specs.append(SpectralData(wlens, elrefs_intp, u_elrefs_intp, ds_intp))
        if callback_observation:
            callback_observation()
        if not is_list:
            specs = specs[0]
        return specs

    @staticmethod
    def _interpolate_polar(
        asd_data: Union[SpectralData, Iterable[SpectralData]],
        cimel_data: Union[SpectralData, List[SpectralData]],
        intp: SpectralInterpolation,
        wlens: List[float],
        skip_uncs: bool,
        callback_observation: Callable,
    ) -> Union[SpectralData, List[SpectralData]]:
        is_list = isinstance(cimel_data, list)
        if not is_list:
            cimel_data = [cimel_data]
            asd_data = [asd_data]  # both same length
        specs: Union[SpectralData, List[SpectralData]] = []
        for cf, asdd in zip(cimel_data, asd_data):
            if callback_observation:
                callback_observation()
            if not skip_uncs:
                (polars_intp, u_polars_intp, _,) = intp.get_interpolated_refl_unc(
                    cf.wlens,
                    cf.data,
                    asdd.wlens,
                    asdd.data,
                    np.array(wlens),
                    cf.uncertainties,
                    asdd.uncertainties,
                    cf.err_corr,
                    asdd.err_corr,
                )
            else:
                polars_intp = intp.get_interpolated_refl(
                    cf.wlens,
                    cf.data,
                    asdd.wlens,
                    asdd.data,
                    np.array(wlens),
                )
                u_polars_intp = np.zeros(polars_intp.shape)
            # The Dataset is not needed later, and the corr_matrix is very big so we ignore it to save up space
            specs.append(SpectralData(wlens, polars_intp, u_polars_intp, None))
        if callback_observation:
            callback_observation()
        if not is_list:
            specs = specs[0]
        return specs

    @staticmethod
    def _interpolate_aolp(
        asd_data: Union[SpectralData, Iterable[SpectralData]],
        cimel_data: Union[SpectralData, List[SpectralData]],
        intp: SpectralInterpolation,
        wlens: List[float],
        skip_uncs: bool,
        callback_observation: Callable,
    ) -> Union[SpectralData, List[SpectralData]]:
        is_list = isinstance(cimel_data, list)
        if not is_list:
            cimel_data = [cimel_data]
            asd_data = [asd_data]  # both same length
        specs: Union[SpectralData, List[SpectralData]] = []
        for cf, asdd in zip(cimel_data, asd_data):
            if callback_observation:
                callback_observation()
            if not skip_uncs:
                (aolp_intp, u_aolp_intp, _,) = intp.get_interpolated_refl_unc(
                    cf.wlens,
                    cf.data,
                    asdd.wlens,
                    asdd.data,
                    np.array(wlens),
                    cf.uncertainties,
                    asdd.uncertainties,
                    cf.err_corr,
                    asdd.err_corr,
                )
            else:
                aolp_intp = intp.get_interpolated_refl(
                    cf.wlens,
                    cf.data,
                    asdd.wlens,
                    asdd.data,
                    np.array(wlens),
                )
                u_aolp_intp = np.zeros(aolp_intp.shape)
            # The Dataset is not needed later, and the corr_matrix is very big so we ignore it to save up space
            specs.append(SpectralData(wlens, aolp_intp, u_aolp_intp, None))
        if callback_observation:
            callback_observation()
        if not is_list:
            specs = specs[0]
        return specs

    @staticmethod
    def _calculate_elref(
        mds: Union[MoonData, List[MoonData]],
        cimel_coeff: ReflectanceCoefficients,
        skip_uncs: bool,
        keep_err_corr_mats: bool,
    ) -> Union[SpectralData, List[SpectralData]]:
        """ """
        rl = lime.LIME()
        if not isinstance(mds, list):
            return rl.get_elrefs(cimel_coeff, mds, skip_uncs, keep_err_corr_mats)
        specs = []
        for m in mds:
            specs.append(rl.get_elrefs(cimel_coeff, m, skip_uncs, keep_err_corr_mats))
        return specs

    @staticmethod
    def _calculate_eli_from_elref(
        mds: Union[MoonData, List[MoonData]],
        elrefs: Union[SpectralData, List[SpectralData]],
        srf_type: str,
        skip_uncs: bool,
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
        srf_type: str
            SRF type that is going to be used. Can be 'cimel', 'asd', 'interpolated_gaussian' or 'interpolated_triangle'
        skip_uncs: bool

        Returns
        -------
        elis: SpectralData | list of SpectralData
            The extraterrestrial lunar irradiance calculated, a list if self.mds is a list
        """
        rl = lime.LIME()
        if not isinstance(mds, list):
            return rl.get_elis_from_elrefs(elrefs, mds, srf_type, skip_uncs)
        specs = []
        for i, m in enumerate(mds):
            specs.append(rl.get_elis_from_elrefs(elrefs[i], m, srf_type, skip_uncs))
        return specs

    @staticmethod
    def _calculate_polar(
        mds: Union[MoonData, List[MoonData]],
        polar_coeff: PolarisationCoefficients,
        skip_uncs: bool,
    ) -> Union[SpectralData, List[SpectralData]]:
        dl = polar.dolp.DOLP()
        if not isinstance(mds, list):
            return dl.get_polarized(mds.mpa_degrees, polar_coeff, skip_uncs)
        else:
            specs = []
            for m in mds:
                spectral_data = dl.get_polarized(m.mpa_degrees, polar_coeff, skip_uncs)
                specs.append(spectral_data)
        return specs

    @staticmethod
    def _calculate_aolp(
        mds: Union[MoonData, List[MoonData]],
        aolp_coeff: AOLPCoefficients,
        skip_uncs: bool,
    ) -> Union[SpectralData, List[SpectralData]]:
        al = polar.aolp.AOLP()
        if not isinstance(mds, list):
            return al.get_aolp(mds.mpa_degrees, aolp_coeff, skip_uncs)
        else:
            specs = []
            for m in mds:
                spectral_data = al.get_aolp(m.mpa_degrees, aolp_coeff, skip_uncs)
                specs.append(spectral_data)
        return specs

    def _calculate_eli_from_elref_and_integrate(
        self,
        mds: Union[MoonData, List[MoonData]],
        elrefs: Union[SpectralData, List[SpectralData]],
        srf_type: str,
        srf: SpectralResponseFunction,
        skip_uncs: bool,
        callback_observation: Callable,
    ):
        rl = lime.LIME()
        channel_ids = [srf.channels[i].id for i in range(len(srf.channels))]
        if not isinstance(mds, list):
            spec, sig, sigunc = rl.get_elis_from_elrefs_and_integrate(
                elrefs, mds, srf_type, self.int, srf, skip_uncs
            )
            signals = np.array([sig]).T
            uncs = np.array([sigunc]).T
            sp_d = SpectralData(channel_ids, signals, uncs, None)
            return spec, sp_d
        specs = []
        signals = []
        uncs = []
        for i, m in enumerate(mds):
            if callback_observation:
                callback_observation()
            spec, sig, sigunc = rl.get_elis_from_elrefs_and_integrate(
                elrefs[i], m, srf_type, self.int, srf, skip_uncs
            )
            specs.append(spec)
            signals.append(sig)
            uncs.append(sigunc)
        if callback_observation:
            callback_observation()
        signals = np.array(signals).T
        uncs = np.array(uncs).T
        sp_d = SpectralData(channel_ids, signals, uncs, None)
        return specs, sp_d

    def set_observations(self, lglod: LGLODData, srf: SpectralResponseFunction):
        obss = lglod.observations
        self.settings_manager.set_coef_version_name(lglod.version)
        self.elref = [obs.refls for obs in obss]
        self.elref_cimel = lglod.elrefs_cimel
        self.elref_asd = None
        self.elis = [obs.irrs for obs in obss]
        self.elis_cimel = lglod.elis_cimel
        self.elis_asd = None
        self.polars = [obs.polars for obs in obss]
        self.polars_cimel = lglod.polars_cimel
        self.polars_asd = None
        self.aolp = [obs.aolp for obs in obss]
        self.aolp_cimel = lglod.aolp_cimel
        self.aolp_asd = None
        self._skip_uncs = lglod.skipped_uncs
        if obss[0].dt == None:
            dts = []
        else:
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
        if not dts:
            sel = obss[0].selenographic_data
            lat, lon, alt = SPICEAdapter.to_planetographic_same_frame(
                [(obss[0].sat_pos.x, obss[0].sat_pos.y, obss[0].sat_pos.z)],
                "MOON",
                self.kernels_path.main_kernels_path,
            )[0]
            point = CustomPoint(
                sel.distance_sun_moon,
                alt / 1000,
                lat,
                lon,
                sel.selen_sun_lon_rad,
                abs(sel.mpa_degrees),
                sel.mpa_degrees,
            )
        elif not obss[0].sat_name or obss[0].sat_name == "":
            point = SurfacePoint(
                *SPICEAdapter.to_planetographic_multiple(
                    [(obss[0].sat_pos.x, obss[0].sat_pos.y, obss[0].sat_pos.z)],
                    "EARTH",
                    self.kernels_path.main_kernels_path,
                    [dts[0]],
                    obss[0].sat_pos_ref,
                )[0],
                dts,
            )
        else:
            point = SatellitePoint(obss[0].sat_name, dts)
            surfaces_of_sat = []
            mds = []
            planetographics = SPICEAdapter.to_planetographic_multiple(
                [(obs.sat_pos.x, obs.sat_pos.y, obs.sat_pos.z) for obs in obss],
                "EARTH",
                self.kernels_path.main_kernels_path,
                dts,
                obss[0].sat_pos_ref,
            )
            for i, dt in enumerate(dts):
                sp = SurfacePoint(
                    *planetographics[i],
                    dt,
                )
                md = MoonDataFactory.get_md(sp, self.eocfi_path, self.kernels_path)
                surfaces_of_sat.append(sp)
                mds.append(md)
            self.surfaces_of_sat = surfaces_of_sat
            self.mds = mds
            self.point = point
            self.mds_uptodate = True
        self._save_parameters(srf, point)
        self.refl_uptodate = True
        self.irr_uptodate = True
        self.pol_uptodate = True
        self.aolp_uptodate = True
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
        if not self.settings_manager.is_show_interp_spectrum():
            return None
        return self.elref_asd

    def get_elis_asd(self) -> Union[SpectralData, List[SpectralData]]:
        if not self.settings_manager.is_show_interp_spectrum():
            return None
        return self.elis_asd

    def get_polars(self) -> Union[SpectralData, List[SpectralData]]:
        return self.polars

    def get_polars_cimel(self) -> Union[SpectralData, List[SpectralData]]:
        return self.polars_cimel

    def get_polars_asd(self) -> Union[SpectralData, List[SpectralData]]:
        if not self.settings_manager.is_show_interp_spectrum():
            return None
        return self.polars_asd

    def get_aolp(self) -> Union[SpectralData, List[SpectralData]]:
        return self.aolp

    def get_aolp_cimel(self) -> Union[SpectralData, List[SpectralData]]:
        return self.aolp_cimel

    def get_aolp_asd(self) -> Union[SpectralData, List[SpectralData]]:
        if not self.settings_manager.is_show_interp_spectrum():
            return None
        return self.aolp_asd

    def get_surfacepoints(self) -> Union[SurfacePoint, List[SurfacePoint], None]:
        return self.surfaces_of_sat

    def get_point(self) -> Point:
        return self.point

    def get_moon_datas(self) -> Union[MoonData, List[MoonData]]:
        return self.mds

    def are_mpas_inside_mpa_range(self) -> Union[bool, List[bool]]:
        l = []
        if isinstance(self.mds, list):
            for md in self.mds:
                l.append(is_ampa_valid_range(md.absolute_mpa_degrees))
            return l
        return is_ampa_valid_range(self.mds.absolute_mpa_degrees)
