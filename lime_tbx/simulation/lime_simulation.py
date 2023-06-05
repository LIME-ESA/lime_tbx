"""Module containing the class that contains the state of the simulation."""

"""___Built-In Modules___"""
from typing import List, Union, Tuple, Iterable
from abc import ABC, abstractmethod

"""___Third-Party Modules___"""
import numpy as np

"""___LIME_TBX Modules___"""
from lime_tbx.datatypes.datatypes import (
    CustomPoint,
    LGLODData,
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
from lime_tbx.datatypes import constants
from lime_tbx.lime_algorithms.lime import lime
from lime_tbx.lime_algorithms.dolp import dolp
from lime_tbx.interpolation.spectral_interpolation.spectral_interpolation import (
    SpectralInterpolation,
)
from lime_tbx.interpolation.interp_data import interp_data
from lime_tbx.simulation.moon_data_factory import MoonDataFactory
from lime_tbx.spectral_integration.spectral_integration import SpectralIntegration
from lime_tbx.spice_adapter.spice_adapter import SPICEAdapter
from lime_tbx.gui.settings import ISettingsManager

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
        srf: SpectralResponseFunction
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
        polars: SpectralData | list of SpectralData
            Previously calculated polarization/s
        """
        pass

    @abstractmethod
    def get_polars_cimel(self) -> Union[SpectralData, List[SpectralData]]:
        """
        Returns the stored value for lunar polarization degree for the cimel wavelengths

        Returns
        -------
        polars_cimel: SpectralData | list of SpectralData
            Previously calculated polarization/s
        """
        pass

    @abstractmethod
    def get_polars_asd(self) -> Union[SpectralData, List[SpectralData]]:
        """
        Returns the stored value for lunar polarization degree for the asd spectrum.

        Returns
        -------
        polars_asd: SpectralData | list of SpectralData
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
    def is_polarization_updated(self) -> bool:
        """Returns if the polarization has been updated. If not, or nothing has been executed,
        or the spectrum doesn't contain polarization values.

        Returns
        -------
        pol_uptodate: bool
            True if the polarization is updated
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
        settings_manager: ISettingsManager,
        MCsteps: int = 100,
        verbose: bool = False,
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
        self.settings_manager = settings_manager

        self.mds: Union[MoonData, List[MoonData]] = []
        self.wlens: List[float] = []
        self.elref: Union[SpectralData, List[SpectralData]] = None
        self.elis: Union[SpectralData, List[SpectralData]] = None
        self.polars: Union[SpectralData, List[SpectralData]] = None
        self.signals: SpectralData = None
        self.elref_cimel: Union[SpectralData, List[SpectralData]] = None
        self.elref_asd: Union[SpectralData, List[SpectralData]] = None
        self.elis_cimel: Union[SpectralData, List[SpectralData]] = None
        self.elis_asd: Union[SpectralData, List[SpectralData]] = None
        self.polars_cimel: Union[SpectralData, List[SpectralData]] = None
        self.polars_asd: Union[SpectralData, List[SpectralData]] = None
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
        self._skip_uncs: bool = None
        self._interp_srf_name: str = None

    def set_simulation_changed(self):
        self.refl_uptodate = False
        self.irr_uptodate = False
        self.pol_uptodate = False
        self.signals_uptodate = False
        self.srf_updtodate = False
        self.mds_uptodate = False
        self._skip_uncs = None
        self._interp_srf_name = None
        self.settings_manager.set_coef_version_name(None)

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
    ):
        self._save_parameters(srf, point)
        skip_uncs = self.is_skipping_uncs()
        if not self.refl_uptodate:
            if self.verbose:
                print("starting reflectance update")

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
        skip_uncs: bool,
    ) -> Tuple[
        Union[SpectralData, List[SpectralData]],
        SpectralData,
        Union[SpectralData, List[SpectralData]],
    ]:
        elref_cimel = LimeSimulation._calculate_elref(mds, cimel_coeff, skip_uncs)
        if isinstance(mds, list):
            elref_asd = (intp.get_best_interp_reference(md) for md in mds)
            ret_asd = intp.get_best_interp_reference(mds[0])
        else:
            elref_asd = intp.get_best_interp_reference(mds)
            ret_asd = elref_asd
        elref = LimeSimulation._interpolate_refl(
            elref_asd, elref_cimel, intp, wlens, skip_uncs
        )
        return elref_cimel, ret_asd, elref

    @staticmethod
    def _calculate_irradiances_values(
        mds: Union[MoonData, List[MoonData]],
        elrefs,
        elref_cimel,
        elref_asd,
        skip_uncs: bool,
        interp_srf_type: str,
    ) -> Tuple[
        Union[SpectralData, List[SpectralData]],
        Union[SpectralData, List[SpectralData]],
        Union[SpectralData, List[SpectralData]],
    ]:
        elis = LimeSimulation._calculate_eli_from_elref(
            mds, elrefs, interp_srf_type, skip_uncs
        )
        elis_cimel = LimeSimulation._calculate_eli_from_elref(
            mds, elref_cimel, "cimel", skip_uncs
        )
        elis_asd = None
        if interp_data.is_show_interpolation_spectrum():
            md = mds
            if isinstance(md, list):
                md = md[0]
            elis_asd = LimeSimulation._calculate_eli_from_elref(
                md, elref_asd, "asd", skip_uncs
            )
        return elis, elis_cimel, elis_asd

    @staticmethod
    def _calculate_polarization_values(
        mds: Union[MoonData, List[MoonData]],
        polar_coeff: PolarizationCoefficients,
        intp: SpectralInterpolation,
        wlens: List[float],
        skip_uncs: bool,
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
            polar_asd, polar_cimel, intp, wlens, True
        )
        # TODO confirm that we should skip polar uncertainties in interpolation if we do always linear
        return polar_cimel, ret_polar_asd, polar

    def update_irradiance(
        self,
        srf: SpectralResponseFunction,
        signals_srf: SpectralResponseFunction,
        point: Point,
        cimel_coeff: ReflectanceCoefficients,
    ):
        self._save_parameters(srf, point)
        skip_uncs = self.is_skipping_uncs()
        if not self.refl_uptodate:
            self.update_reflectance(self.srf, point, cimel_coeff)
        interp_srf_name = self.get_interp_srf_name()
        if not self.irr_uptodate:
            if self.verbose:
                print("starting irradiance update")
            (
                self.elis,
                self.elis_cimel,
                self.elis_asd,
            ) = LimeSimulation._calculate_irradiances_values(
                self.mds,
                self.elref,
                self.elref_cimel,
                self.elref_asd,
                skip_uncs,
                interp_srf_name,
            )
            self.irr_uptodate = True
            if self.verbose:
                print("irradiance update done")

        if not self.signals_uptodate:
            self.signals = self._calculate_signals(signals_srf, skip_uncs)
            self.signals_uptodate = True
            if self.verbose:
                print("signals update done")

    def is_polarization_updated(self) -> bool:
        return self.pol_uptodate

    def update_polarization(
        self,
        srf: SpectralResponseFunction,
        point: Point,
        polar_coeff: PolarizationCoefficients,
    ):
        skip_uncs = self.is_skipping_uncs()
        self._save_parameters(srf, point)
        if not self.pol_uptodate:
            if self.verbose:
                print("starting polarisation update")
            (
                self.polars_cimel,
                self.polars_asd,
                self.polars,
            ) = LimeSimulation._calculate_polarization_values(
                self.mds,
                polar_coeff,
                self.intp,
                self.wlens,
                skip_uncs,
            )
            self.pol_uptodate = True
            if self.verbose:
                print("polarisation update done")

    @staticmethod
    def _interpolate_refl(
        asd_data: Union[SpectralData, Iterable[SpectralData]],
        cimel_data: Union[SpectralData, List[SpectralData]],
        intp: SpectralInterpolation,
        wlens: List[float],
        skip_uncs: bool,
    ) -> Union[SpectralData, List[SpectralData]]:

        wlens = np.array(wlens)
        is_list = isinstance(cimel_data, list)
        if not is_list:
            cimel_data = [cimel_data]
            asd_data = [asd_data]  # both same length
        specs: Union[SpectralData, List[SpectralData]] = []
        for cf, asdd in zip(cimel_data, asd_data):
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
                    cf.ds.err_corr_reflectance.values,
                    asdd.ds.err_corr_reflectance.values,
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
                err_corr_side = len(u_elrefs_intp)
                corr_elrefs_intp = np.zeros((err_corr_side, err_corr_side))
                np.fill_diagonal(corr_elrefs_intp, 1)

            ds_intp = SpectralData.make_reflectance_ds(
                wlens, elrefs_intp, u_elrefs_intp, corr_elrefs_intp
            )
            specs.append(SpectralData(wlens, elrefs_intp, u_elrefs_intp, ds_intp))

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
    ) -> Union[SpectralData, List[SpectralData]]:

        is_list = isinstance(cimel_data, list)
        if not is_list:
            cimel_data = [cimel_data]
            asd_data = [asd_data]  # both same length
        specs: Union[SpectralData, List[SpectralData]] = []
        for cf, asdd in zip(cimel_data, asd_data):
            if not skip_uncs:
                (
                    polars_intp,
                    u_polars_intp,
                    corr_polars_intp,
                ) = intp.get_interpolated_refl_unc(
                    cf.wlens,
                    cf.data,
                    asdd.wlens,
                    asdd.data,
                    np.array(wlens),
                    cf.uncertainties,
                    asdd.uncertainties,
                    cf.ds.err_corr_polarization.values,
                    asdd.ds.err_corr_polarization.values,
                )
            else:
                polars_intp = intp.get_interpolated_refl(
                    cf.wlens,
                    cf.data,
                    asdd.wlens,
                    asdd.data,
                    wlens,
                )
                u_polars_intp = np.zeros(polars_intp.shape)
                corr_polars_intp = np.zeros((len(polars_intp), len(polars_intp)))
                np.fill_diagonal(corr_polars_intp, 1)

            ds_intp = SpectralData.make_reflectance_ds(
                wlens,
                polars_intp,
                u_polars_intp,
                corr_polars_intp,
            )

            specs.append(SpectralData(wlens, polars_intp, u_polars_intp, ds_intp))
        if not is_list:
            specs = specs[0]
        return specs

    @staticmethod
    def _calculate_elref(
        mds: Union[MoonData, List[MoonData]],
        cimel_coeff: ReflectanceCoefficients,
        skip_uncs: bool,
    ) -> Union[SpectralData, List[SpectralData]]:
        """ """
        rl = lime.LIME()
        if not isinstance(mds, list):
            return rl.get_elrefs(cimel_coeff, mds, skip_uncs)
        specs = []
        for m in mds:
            specs.append(rl.get_elrefs(cimel_coeff, m, skip_uncs))
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
            SRF type that is going to be used. Can be 'cimel', 'asd' or 'interpolated'.
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
        polar_coeff: PolarizationCoefficients,
        skip_uncs: bool,
    ) -> Union[SpectralData, List[SpectralData]]:
        dl = dolp.DOLP()
        if not isinstance(mds, list):
            return dl.get_polarized(mds.mpa_degrees, polar_coeff, skip_uncs)
        else:
            specs = []
            for m in mds:
                spectral_data = dl.get_polarized(m.mpa_degrees, polar_coeff, skip_uncs)
                specs.append(spectral_data)
        return specs

    def _calculate_signals(
        self,
        srf: SpectralResponseFunction,
        skip_uncs: bool,
    ) -> SpectralData:
        elis_signals = self.elis
        channel_ids = [srf.channels[i].id for i in range(len(srf.channels))]
        if not isinstance(elis_signals, list):
            elis_signals = [elis_signals]
        signals_list = []
        uncs_list = []
        for irr in elis_signals:
            sigs = self.int.integrate_elis(srf, irr)
            signals_list.append(sigs)
            if not skip_uncs:
                siguncs = self.int.u_integrate_elis(srf, irr)
            else:
                siguncs = np.zeros(np.array(sigs).shape)
            uncs_list.append(siguncs)
        signals = np.array(signals_list).T
        uncs = np.array(uncs_list).T
        ds_sig = SpectralData.make_signals_ds(channel_ids, signals, uncs)
        sp_d = SpectralData(channel_ids, signals, uncs, ds_sig)
        return sp_d

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
            lat, lon, alt = SPICEAdapter.to_planetographic(
                obss[0].sat_pos.x,
                obss[0].sat_pos.y,
                obss[0].sat_pos.z,
                "MOON",
                self.kernels_path.main_kernels_path,
            )
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
                *SPICEAdapter.to_planetographic(
                    obss[0].sat_pos.x,
                    obss[0].sat_pos.y,
                    obss[0].sat_pos.z,
                    "EARTH",
                    self.kernels_path.main_kernels_path,
                ),
                dts
            )
        else:
            point = SatellitePoint(obss[0].sat_name, dts)
            surfaces_of_sat = []
            mds = []
            for i, dt in enumerate(dts):
                sp = SurfacePoint(
                    *SPICEAdapter.to_planetographic(
                        obss[i].sat_pos.x,
                        obss[i].sat_pos.y,
                        obss[i].sat_pos.z,
                        "EARTH",
                        self.kernels_path.main_kernels_path,
                    ),
                    dt
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
        if not interp_data.is_show_interpolation_spectrum():
            return None
        return self.elref_asd

    def get_elis_asd(self) -> Union[SpectralData, List[SpectralData]]:
        if not interp_data.is_show_interpolation_spectrum():
            return None
        return self.elis_asd

    def get_polars(self) -> Union[SpectralData, List[SpectralData]]:
        return self.polars

    def get_polars_cimel(self) -> Union[SpectralData, List[SpectralData]]:
        return self.polars_cimel

    def get_polars_asd(self) -> Union[SpectralData, List[SpectralData]]:
        if not interp_data.is_show_interpolation_spectrum():
            return None
        return self.polars_asd

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
