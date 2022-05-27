"""describe class"""

"""___Built-In Modules___"""
from abc import ABC, abstractmethod
from typing import List, Union

"""___Third-Party Modules___"""
# import here

"""___NPL Modules___"""
from ...spice_adapter.spice_adapter import SPICEAdapter
from ...datatypes.datatypes import (
    IrradianceCoefficients,
    MoonData,
    PolarizationCoefficients,
    SpectralResponseFunction,
    SurfacePoint,
    CustomPoint,
)
from ..common.common import CommonSimulation

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2022"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


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
    def integrate_elis(
        srf: SpectralResponseFunction, elis: Union[List[float], List[List[float]]]
    ) -> Union[List[float], List[List[float]]]:
        """
        Integrate the irradiance values for all SRF channels.

        Parameters
        ----------
        srf: SpectralResponseFunction
            SRF used for the integration, containing the channels.
        elis: list of float | list of list of float
            The corresponding irradiance values for the wlens. The order must correspond with
            the wavelengths order in the srf, channel by channel.
            It can be a list of lists in case multiple cases are passed together.

        Returns
        -------
        integrated_irradiances: list of float | list of list of float
            List of all the integrated irradiance values for each channel, in order.
            It will be a list of lists if "elis" was a list of lists.
        """
        pass


class RegularSimulation(IRegularSimulation):
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
        return CommonSimulation.get_eli_from_md(srf, md, coefficients)

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
        return CommonSimulation.get_elref_from_md(srf, md, coefficients)

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
        return CommonSimulation.get_polar_from_md(srf, md, coefficients)

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
        return CommonSimulation.get_eli_from_md(srf, md, coefficients)

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
        return CommonSimulation.get_elref_from_md(srf, md, coefficients)

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
        return CommonSimulation.get_polar_from_md(srf, md, coefficients)

    @staticmethod
    def integrate_elis(
        srf: SpectralResponseFunction, elis: Union[List[float], List[List[float]]]
    ) -> Union[List[float], List[List[float]]]:
        signals = []
        wlens = srf.get_wavelengths()
        if len(elis) == 0:
            return []
        wasnt_lists = False
        if not isinstance(elis[0], list):
            wasnt_lists = True
            elis = [elis]
        for ch in srf.channels:
            tots_eli = [0 for _ in range(len(elis))]
            ch_wlens = list(ch.spectral_response.keys())
            dividends = [0 for _ in range(len(elis))]
            for i, wl in enumerate(ch_wlens):
                interval = 0
                if i > 0:
                    interval += (wl - ch_wlens[i - 1]) / 2
                if i < len(ch_wlens) - 1:
                    interval += (ch_wlens[i + 1] - wl) / 2
                extra_dividend = ch.spectral_response[wl] * interval
                for i, sub_elis in enumerate(elis):
                    eli = sub_elis[wlens.index(wl)]
                    tots_eli[i] += extra_dividend * eli
                    dividends[i] += extra_dividend
            ch_signals = []
            for i in range(len(tots_eli)):
                signal = tots_eli[i] / dividends[i]
                ch_signals.append(signal)
            signals.append(ch_signals)
        if wasnt_lists:
            signals = [s[0] for s in signals]
        return signals
