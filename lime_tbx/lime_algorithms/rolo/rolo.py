"""
This module calculates the extra-terrestrial lunar disk irradiance.

It exports the following classes:
    * IROLO - Interface that contains the methods of this module.
    * ROLO - Class that implements the methods exported by this module.

It follows equations described in the following papers:
- Kieffer and Stone, 2005: The spectral irradiance of the Moon.
- Barreto et al., 2019: Evaluation of night-time aerosols measurements and lunar irradiance
models in the frame of the first multi-instrument nocturnal intercomparison campaign.
- Roman et al., 2020: Correction of a lunar-irradiance model for aerosol optical depth
retrieval and comparison with a star photometer.
"""

"""___Built-In Modules___"""
from abc import ABC, abstractmethod
from typing import List, Union

"""___Third-Party Modules___"""
import numpy as np

"""___LIME Modules___"""
from . import eli, elref
from ...datatypes.datatypes import (
    MoonData,
    ApolloIrradianceCoefficients,
    ReflectanceCoefficients,
    SpectralData,
)

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "2022/03/02"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


class IROLO(ABC):
    """
    Interface that contains the methods of this module.

    It exports the following functions:
        * get_elis_from_elrefs: Calculates the extra-terrestrial lunar irradiance in Wm⁻²/nm
            from reflectance data.
        * get_elrefs: Calculates the extra-terrestrial lunar reflectance in fractions of unity for some
            given parameters.
        * get_eli_apollo: Calculates the extra-terrestrial lunar irradiance in Wm⁻²/nm for some given
            parameters, using the apollo coefficients.
        * get_elref: Calculates the extra-terrestrial lunar reflectance in fractions of unity for some
            given parameters, using the apollo coefficients.
    """

    @staticmethod
    @abstractmethod
    def get_elis_from_elrefs(
        elref_spectrum: SpectralData, moon_data: MoonData
    ) -> SpectralData:
        """Calculation of Extraterrestrial Lunar Irradiance following Eq 3 in Roman et al., 2020,
        without using the Apollo Coefficients.

        Allow users to simulate lunar observation for any observer/solar selenographic
        latitude and longitude.

        Returns the data in Wm⁻²/nm

        Parameters
        ----------
        elref_spectrum : SpectralData
            Reflectance data.
        moon_data : MoonData
            Moon data needed to calculate Moon's irradiance.

        Returns
        -------
        elis: SpectralData
            The extraterrestrial lunar irradiances calculated.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_elrefs(
        coefficients: ReflectanceCoefficients,
        moon_data: MoonData,
    ) -> SpectralData:
        """Calculation of Extraterrestrial Lunar Reflectance following Eq 3 in Roman et al., 2020
        for the calculation of the irradiance, without using the Apollo Coefficients.

        Allow users to simulate lunar observation for any observer/solar selenographic
        latitude and longitude.

        Returns the data in fractions of unity.

        Parameters
        ----------
        coefficients : ReflectanceCoefficients
            Needed coefficients for the simulation
        moon_data : MoonData
            Moon data needed to calculate Moon's irradiance..

        Returns
        -------
        SpectralData
            The extraterrestrial lunar reflectances calculated.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_eli_apollo(
        wavelengths: Union[float, List[float]],
        moon_data: MoonData,
        coefficients: ApolloIrradianceCoefficients,
    ) -> np.ndarray:
        """Calculation of Extraterrestrial Lunar Irradiance following Eq 3 in Roman et al., 2020,
        using the Apollo Coefficients.

        Allow users to simulate lunar observation for any observer/solar selenographic
        latitude and longitude.

        Returns the data in Wm⁻²/nm

        Parameters
        ----------
        wavelengths : float | list of float
            Wavelength/s (in nanometers) of which the extraterrestrial lunar irradiance will be
            calculated.
        moon_data : MoonData
            Moon data needed to calculate Moon's irradiance.
        coefficients : IrradianceCoefficients
            Needed coefficients for the simulation.

        Returns
        -------
        np.ndarray of float
            The extraterrestrial lunar irradiances calculated.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_elref_apollo(
        wavelengths: Union[float, List[float]],
        moon_data: MoonData,
        coefficients: ApolloIrradianceCoefficients,
    ) -> np.ndarray:
        """Calculation of Extraterrestrial Lunar Reflectance following Eq 3 in Roman et al., 2020
        for the calculation of the irradiance, using the Apollo Coefficients.

        Allow users to simulate lunar observation for any observer/solar selenographic
        latitude and longitude.

        Returns the data in fractions of unity.

        Parameters
        ----------
        wavelengths : float | list of float
            Wavelength/s (in nanometers) of which the extraterrestrial lunar irradiance will be
            calculated.
        moon_data : MoonData
            Moon data needed to calculate Moon's irradiance.
        coefficients : IrradianceCoefficients
            Needed coefficients for the simulation.

        Returns
        -------
        np.ndarray of float
            The extraterrestrial lunar reflectances calculated.
        """
        pass


class ROLO(IROLO):
    """
    Class that implements the methods of this module.
    """

    @staticmethod
    def get_elis_from_elrefs(
        elref_spectrum: SpectralData, moon_data: MoonData
    ) -> SpectralData:
        wlens = elref_spectrum.wlens
        elis = eli.calculate_eli_from_elref(wlens, moon_data, elref_spectrum.data)
        unc = eli.calculate_eli_from_elref_unc(elref_spectrum, moon_data)
        ds_eli = SpectralData.make_irradiance_ds(wlens, elis, unc_rand=unc)
        return SpectralData(wlens, elis, unc, ds_eli)

    @staticmethod
    def get_elrefs(
        coefficients: ReflectanceCoefficients, moon_data: MoonData
    ) -> SpectralData:
        wlens = coefficients.wlens
        elrefs = elref.calculate_elref(coefficients, moon_data)
        unc = elref.calculate_elref_unc(coefficients, moon_data)
        ds = SpectralData.make_reflectance_ds(wlens, elrefs, unc_rand=unc)
        return SpectralData(wlens, elrefs, unc, ds)

    @staticmethod
    def get_eli_apollo(
        wavelengths: Union[float, List[float]],
        moon_data: MoonData,
        coefficients: ApolloIrradianceCoefficients,
    ) -> np.ndarray:
        """Calculation of Extraterrestrial Lunar Irradiance following Eq 3 in Roman et al., 2020

        Allow users to simulate lunar observation for any observer/solar selenographic
        latitude and longitude.

        Returns the data in Wm⁻²/nm

        Parameters
        ----------
        wavelengths : float | list of float
            Wavelength/s (in nanometers) of which the extraterrestrial lunar irradiance will be
            calculated.
        moon_data : MoonData
            Moon data needed to calculate Moon's irradiance.
        coefficients : IrradianceCoefficients
            Needed coefficients for the simulation.

        Returns
        -------
        np.ndarray of float
            The extraterrestrial lunar irradiances calculated.
        """
        if not isinstance(wavelengths, list):
            wavelengths = [wavelengths]
        return eli.calculate_elis_apollo(wavelengths, moon_data, coefficients)

    @staticmethod
    def get_elref_apollo(
        wavelengths: Union[float, List[float]],
        moon_data: MoonData,
        coefficients: ApolloIrradianceCoefficients,
    ) -> np.ndarray:
        """Calculation of Extraterrestrial Lunar Reflectance following Eq 3 in Roman et al., 2020
        for the calculation of the irradiance.

        Allow users to simulate lunar observation for any observer/solar selenographic
        latitude and longitude.

        Returns the data in fractions of unity.

        Parameters
        ----------
        wavelengths : float | list of float
            Wavelength/s (in nanometers) of which the extraterrestrial lunar irradiance will be
            calculated.
        moon_data : MoonData
            Moon data needed to calculate Moon's irradiance.
        coefficients : IrradianceCoefficients
            Needed coefficients for the simulation.

        Returns
        -------
        np.ndarray of float
            The extraterrestrial lunar reflectances calculated.
        """
        if not isinstance(wavelengths, list):
            wavelengths = [wavelengths]
        return elref.calculate_elref_apollo(wavelengths, moon_data, coefficients)
