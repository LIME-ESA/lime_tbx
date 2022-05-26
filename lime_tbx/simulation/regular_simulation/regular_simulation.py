"""describe class"""

"""___Built-In Modules___"""
from datetime import datetime
from typing import List, Union

"""___Third-Party Modules___"""
# import here

"""___NPL Modules___"""
from ...spice_adapter.spice_adapter import SPICEAdapter
from ...datatypes.datatypes import (
    IrradianceCoefficients,
    MoonData,
    PolarizationCoefficients,
    SatellitePoint,
    SpectralResponseFunction,
    SurfacePoint,
    CustomPoint,
)
from ...lime_algorithms.rolo import rolo
from ...lime_algorithms.dolp import dolp
from ...eocfi_adapter.eocfi_adapter import EOCFIConverter

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2022"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"

from abc import ABC, abstractmethod


class IRegularSimulation(ABC):
    @staticmethod
    @abstractmethod
    def get_eli_from_surface(
        srf: SpectralResponseFunction,
        sp: SurfacePoint,
        coefficients: IrradianceCoefficients,
        kernels_path: str,
    ) -> Union[List[float], List[List[float]]]:
        """
        Simulate the extraterrestrial lunar irradiance for a geographic point.

        Returns the data in Wm⁻²/nm

        Parameters
        ----------
        srf: SpectralResponseFunction
            Spectral Response Function that the simulation will be computed for.
        sp: SurfacePoint
            Earth's surface point for which the simulation will be computed for.
        coefficients: IrradianceCoefficients
            Values of the chosen coefficients for the ROLO algorithm.
        kernels_path: str
            Path where the needed SPICE kernels are located.
            The user must have write access to that directory.

        Returns
        -------
        elis: list of float | list of list of float
            Extraterrestrial lunar irradiances for the given srf at the specified point.
            It will be a list of lists of float if the parameter dt is a list. Otherwise it
            will only be a list of float.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_elref_from_surface(
        srf: SpectralResponseFunction,
        sp: SurfacePoint,
        coefficients: IrradianceCoefficients,
        kernels_path: str,
    ) -> Union[List[float], List[List[float]]]:
        """
        Simulate the extraterrestrial lunar reflectance for a geographic point.

        Returns the data in fractions of unity.

        Parameters
        ----------
        srf: SpectralResponseFunction
            Spectral Response Function that the simulation will be computed for.
        sp: SurfacePoint
            Earth's surface point for which the simulation will be computed for.
        coefficients: IrradianceCoefficients
            Values of the chosen coefficients for the ROLO algorithm.
        kernels_path: str
            Path where the needed SPICE kernels are located.
            The user must have write access to that directory.

        Returns
        -------
        elrefs: list of float | list of list of float
            Extraterrestrial lunar reflectances for the given srf at the specified point.
            It will be a list of lists of float if the parameter dt is a list. Otherwise it
            will only be a list of float.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_polarized_from_surface(
        srf: SpectralResponseFunction,
        sp: SurfacePoint,
        coefficients: PolarizationCoefficients,
        kernels_path: str,
    ) -> List[float]:
        """
        Simulate the lunar polarization for custom lunar parameters.

        Returns the data in fractions of unity.

        Parameters
        ----------
        srf: SpectralResponseFunction
            Spectral Response Function that the simulation will be computed for.
        sp: SurfacePoint
            Earth's surface point for which the simulation will be computed for.
        coefficients: PolarizationCoefficients
            Values of the chosen coefficients for the DOLP algorithm.
        kernels_path: str
            Path where the needed SPICE kernels are located.
            The user must have write access to that directory.

        Returns
        -------
        polarizations: list of float
            Lunar polarizations for the given srf and the specified parameters.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_eli_from_custom(
        srf: SpectralResponseFunction,
        cp: CustomPoint,
        coefficients: IrradianceCoefficients,
    ) -> List[float]:
        """
        Simulate the extraterrestrial lunar irradiance for custom lunar parameters.

        Returns the data in Wm⁻²/nm

        Parameters
        ----------
        srf: SpectralResponseFunction
            Spectral Response Function that the simulation will be computed for.
        cp: CustomPoint
            Custom point with custom lunar data for which the simulation will be computed for.
        coefficients: IrradianceCoefficients
            Values of the chosen coefficients for the ROLO algorithm.

        Returns
        -------
        elis: list of float
            Extraterrestrial lunar irradiances for the given srf and the specified parameters.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_elref_from_custom(
        srf: SpectralResponseFunction,
        cp: CustomPoint,
        coefficients: IrradianceCoefficients,
    ) -> List[float]:
        """
        Simulate the extraterrestrial lunar reflectance for custom lunar parameters.

        Returns the data in fractions of unity.

        Parameters
        ----------
        srf: SpectralResponseFunction
            Spectral Response Function that the simulation will be computed for.
        cp: CustomPoint
            Custom point with custom lunar data for which the simulation will be computed for.
        coefficients: IrradianceCoefficients
            Values of the chosen coefficients for the ROLO algorithm.

        Returns
        -------
        elrefs: list of float
            Extraterrestrial lunar reflectances for the given srf and the specified parameters.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_polarized_from_custom(
        srf: SpectralResponseFunction,
        cp: CustomPoint,
        coefficients: PolarizationCoefficients,
    ) -> List[float]:
        """
        Simulate the lunar polarization for custom lunar parameters.

        Returns the data in fractions of unity.

        Parameters
        ----------
        srf: SpectralResponseFunction
            Spectral Response Function that the simulation will be computed for.
        cp: CustomPoint
            Custom point with custom lunar data for which the simulation will be computed for.
        coefficients: PolarizationCoefficients
            Values of the chosen coefficients for the DOLP algorithm.

        Returns
        -------
        polarizations: list of float
            Lunar polarizations for the given srf and the specified parameters.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_eli_from_satellite(
        srf: SpectralResponseFunction,
        sp: SatellitePoint,
        coefficients: IrradianceCoefficients,
        kernels_path: str,
        eocfi_path: str,
    ) -> List[float]:
        """
        Simulate the extraterrestrial lunar irradiance for a satellite point.

        Returns the data in Wm⁻²/nm

        Parameters
        ----------
        srf: SpectralResponseFunction
            Spectral Response Function that the simulation will be computed for.
        sp: SatellitePoint
            Satellite point for which the simulation will be computed for.
        coefficients: IrradianceCoefficients
            Values of the chosen coefficients for the ROLO algorithm.
        kernels_path: str
            Path where the needed SPICE kernels are located.
            The user must have write access to that directory.
        eocfi_path: str
            Path where the needed EOCFI datafiles are located.

        Returns
        -------
        elis: list of float
            Extraterrestrial lunar irradiances for the given srf at the specified point.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_elref_from_satellite(
        srf: SpectralResponseFunction,
        sp: SatellitePoint,
        coefficients: IrradianceCoefficients,
        kernels_path: str,
        eocfi_path: str,
    ) -> List[float]:
        """
        Simulate the extraterrestrial lunar reflectance for a satellite point.

        Returns the data in fracions of unity.

        Parameters
        ----------
        srf: SpectralResponseFunction
            Spectral Response Function that the simulation will be computed for.
        sp: SatellitePoint
            Satellite point for which the simulation will be computed for.
        coefficients: IrradianceCoefficients
            Values of the chosen coefficients for the ROLO algorithm.
        kernels_path: str
            Path where the needed SPICE kernels are located.
            The user must have write access to that directory.
        eocfi_path: str
            Path where the needed EOCFI datafiles are located.

        Returns
        -------
        elrefs: list of float
            Extraterrestrial lunar reflectanes for the given srf at the specified point.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_polarized_from_satellite(
        srf: SpectralResponseFunction,
        sp: SatellitePoint,
        coefficients: PolarizationCoefficients,
        kernels_path: str,
        eocfi_path: str,
    ) -> List[float]:
        """
        Simulate the extraterrestrial lunar polarization for a satellite point.

        Returns the data in fracions of unity.

        Parameters
        ----------
        srf: SpectralResponseFunction
            Spectral Response Function that the simulation will be computed for.
        sp: SatellitePoint
            Satellite point for which the simulation will be computed for.
        coefficients: PolarizationCoefficients
            Values of the chosen coefficients for the DOLP algorithm.
        kernels_path: str
            Path where the needed SPICE kernels are located.
            The user must have write access to that directory.
        eocfi_path: str
            Path where the needed EOCFI datafiles are located.

        Returns
        -------
        polarizations: list of float
            Extraterrestrial lunar polarizations for the given srf at the specified point.
        """
        pass

    @staticmethod
    @abstractmethod
    def integrate_elis(srf: SpectralResponseFunction, elis: List[float]) -> List[float]:
        """
        Integrate the irradiance values for all SRF channels.

        Parameters
        ----------
        srf: SpectralResponseFunction
            SRF used for the integration, containing the channels.
        elis: list of float
            The corresponding irradiance values for the wlens. The order must correspond with
            the wavelengths order in the srf, channel by channel.

        Returns
        -------
        integrated_irradiances: list of float
            List of all the integrated irradiance values for each channel, in order.
        """
        pass


class RegularSimulation(IRegularSimulation):
    @staticmethod
    def _get_eli_from_md(
        srf: SpectralResponseFunction,
        md: Union[MoonData, List[MoonData]],
        coefficients: IrradianceCoefficients,
    ) -> Union[List[float], List[List[float]]]:
        """
        Parameters
        ----------
        srf: SpectralResponseFunction
            Spectral Response Function that the simulation will be computed for.
        md: MoonData | list of MoonData
            MoonData for one observation, or for multiple observations in case that
            it's a list.
        coefficients: IrradianceCoefficients
            Values of the chosen coefficients for the ROLO algorithm.

        Returns
        -------
        irradiances: list of float | list of list of float
            Extraterrestrial lunar irradiances for the given srf at the specified point.
            It will be a list of lists of float if the parameter md is a list. Otherwise it
            will only be a list of float.
        """
        rl = rolo.ROLO()
        wlens = srf.get_wavelengths()
        if not isinstance(md, list):
            irradiances = rl.get_eli(wlens, md, coefficients)
            for i, w in enumerate(wlens):
                irradiances[i] = irradiances[i]  # * srf.get_spectral_response(w)
            return irradiances
        times_irr = []
        for m in md:
            irradiances = rl.get_eli(wlens, m, coefficients)
            for i, w in enumerate(wlens):
                irradiances[i] = irradiances[i]  # * srf.get_spectral_response(w)
            times_irr.append(irradiances)
        return times_irr

    @staticmethod
    def _get_elref_from_md(
        srf: SpectralResponseFunction,
        md: Union[MoonData, List[MoonData]],
        coefficients: IrradianceCoefficients,
    ) -> Union[List[float], List[List[float]]]:
        """
        Parameters
        ----------
        srf: SpectralResponseFunction
            Spectral Response Function that the simulation will be computed for.
        md: MoonData | list of MoonData
            MoonData for one observation, or for multiple observations in case that
            it's a list.
        coefficients: IrradianceCoefficients
            Values of the chosen coefficients for the ROLO algorithm.

        Returns
        -------
        reflectances: list of float | list of list of float
            Extraterrestrial lunar reflectances for the given srf at the specified point.
            It will be a list of lists of float if the parameter md is a list. Otherwise it
            will only be a list of float.
        """
        rl = rolo.ROLO()
        wlens = srf.get_wavelengths()
        if not isinstance(md, list):
            reflectances = rl.get_elref(wlens, md, coefficients)
            return reflectances
        times_refl = []
        for m in md:
            reflectances = rl.get_elref(wlens, m, coefficients)
            times_refl.append(reflectances)
        return times_refl

    @staticmethod
    def _get_polar_from_md(
        srf: SpectralResponseFunction,
        md: Union[MoonData, List[MoonData]],
        coefficients: PolarizationCoefficients,
    ) -> Union[List[float], List[List[float]]]:
        """
        Parameters
        ----------
        srf: SpectralResponseFunction
            Spectral Response Function that the simulation will be computed for.
        md: MoonData | list of MoonData
            MoonData for one observation, or for multiple observations in case that
            it's a list.
        coefficients: PolarizationCoefficients
            Values of the chosen coefficients for the DOLP algorithm.

        Returns
        -------
        polarizations: list of float | list of list of float
            Extraterrestrial lunar polarizations for the given srf at the specified point.
            It will be a list of lists of float if the parameter md is a list. Otherwise it
            will only be a list of float.
        """
        dl = dolp.DOLP()
        wlens = srf.get_wavelengths()
        if not isinstance(md, list):
            polarizations = dl.get_polarized(wlens, md.mpa_degrees, coefficients)
            return polarizations
        times_polar = []
        for m in md:
            polarizations = dl.get_polarized(wlens, m.mpa_degrees, coefficients)
            times_polar.append(polarizations)
        return times_polar

    @staticmethod
    def get_eli_from_surface(
        srf: SpectralResponseFunction,
        sp: SurfacePoint,
        coefficients: IrradianceCoefficients,
        kernels_path: str,
    ) -> Union[List[float], List[List[float]]]:
        md = SPICEAdapter().get_moon_data_from_earth(
            sp.latitude, sp.longitude, sp.altitude, sp.dt, kernels_path
        )
        return RegularSimulation._get_eli_from_md(srf, md, coefficients)

    @staticmethod
    def get_elref_from_surface(
        srf: SpectralResponseFunction,
        sp: SurfacePoint,
        coefficients: IrradianceCoefficients,
        kernels_path: str,
    ) -> Union[List[float], List[List[float]]]:
        md = SPICEAdapter().get_moon_data_from_earth(
            sp.latitude, sp.longitude, sp.altitude, sp.dt, kernels_path
        )
        return RegularSimulation._get_elref_from_md(srf, md, coefficients)

    @staticmethod
    def get_polarized_from_surface(
        srf: SpectralResponseFunction,
        sp: SurfacePoint,
        coefficients: IrradianceCoefficients,
        kernels_path: str,
    ) -> List[float]:
        md = SPICEAdapter().get_moon_data_from_earth(
            sp.latitude, sp.longitude, sp.altitude, sp.dt, kernels_path
        )
        return RegularSimulation._get_polar_from_md(srf, md, coefficients)

    @staticmethod
    def get_eli_from_custom(
        srf: SpectralResponseFunction,
        cp: CustomPoint,
        coefficients: IrradianceCoefficients,
    ) -> List[float]:
        md = MoonData(
            cp.distance_sun_moon,
            cp.distance_observer_moon,
            cp.selen_sun_lon,
            cp.selen_obs_lat,
            cp.selen_obs_lon,
            cp.abs_moon_phase_angle,
            cp.moon_phase_angle,
        )
        return RegularSimulation._get_eli_from_md(srf, md, coefficients)

    @staticmethod
    def get_elref_from_custom(
        srf: SpectralResponseFunction,
        cp: CustomPoint,
        coefficients: IrradianceCoefficients,
    ) -> List[float]:
        md = MoonData(
            cp.distance_sun_moon,
            cp.distance_observer_moon,
            cp.selen_sun_lon,
            cp.selen_obs_lat,
            cp.selen_obs_lon,
            cp.abs_moon_phase_angle,
            cp.moon_phase_angle,
        )
        return RegularSimulation._get_elref_from_md(srf, md, coefficients)

    @staticmethod
    def get_polarized_from_custom(
        srf: SpectralResponseFunction,
        cp: CustomPoint,
        coefficients: PolarizationCoefficients,
    ) -> List[float]:
        md = MoonData(
            cp.distance_sun_moon,
            cp.distance_observer_moon,
            cp.selen_sun_lon,
            cp.selen_obs_lat,
            cp.selen_obs_lon,
            cp.abs_moon_phase_angle,
            cp.moon_phase_angle,
        )
        return RegularSimulation._get_polar_from_md(srf, md, coefficients)

    @staticmethod
    def get_eli_from_satellite(
        srf: SpectralResponseFunction,
        sp: SatellitePoint,
        coefficients: IrradianceCoefficients,
        kernels_path: str,
        eocfi_path: str,
    ) -> List[float]:
        eocfi = EOCFIConverter(eocfi_path)
        lat, lon, height = eocfi.get_satellite_position(sp.name, sp.dt)
        srp = SurfacePoint(lat, lon, height, sp.dt)
        return RegularSimulation.get_eli_from_surface(
            srf, srp, coefficients, kernels_path
        )

    @staticmethod
    def get_elref_from_satellite(
        srf: SpectralResponseFunction,
        sp: SatellitePoint,
        coefficients: IrradianceCoefficients,
        kernels_path: str,
        eocfi_path: str,
    ) -> List[float]:
        eocfi = EOCFIConverter(eocfi_path)
        lat, lon, height = eocfi.get_satellite_position(sp.name, sp.dt)
        srp = SurfacePoint(lat, lon, height, sp.dt)
        return RegularSimulation.get_elref_from_surface(
            srf, srp, coefficients, kernels_path
        )

    @staticmethod
    def get_polarized_from_satellite(
        srf: SpectralResponseFunction,
        sp: SatellitePoint,
        coefficients: PolarizationCoefficients,
        kernels_path: str,
        eocfi_path: str,
    ) -> List[float]:
        eocfi = EOCFIConverter(eocfi_path)
        lat, lon, height = eocfi.get_satellite_position(sp.name, sp.dt)
        srp = SurfacePoint(lat, lon, height, sp.dt)
        return RegularSimulation.get_polarized_from_surface(
            srf, srp, coefficients, kernels_path
        )

    @staticmethod
    def integrate_elis(srf: SpectralResponseFunction, elis: List[float]) -> List[float]:
        signals = []
        wlens = srf.get_wavelengths()
        for ch in srf.channels:
            tot_eli = 0
            ch_wlens = list(ch.spectral_response.keys())
            dividend = 0
            for i, wl in enumerate(ch_wlens):
                interval = 0
                if i > 0:
                    interval += (wl - ch_wlens[i - 1]) / 2
                if i < len(ch_wlens) - 1:
                    interval += (ch_wlens[i + 1] - wl) / 2
                eli = elis[wlens.index(wl)]
                tot_eli += ch.spectral_response[wl] * eli * interval
                dividend += ch.spectral_response[wl] * interval
            signal = tot_eli / dividend
            signals.append(signal)
        return signals
