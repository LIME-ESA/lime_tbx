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
# import here

"""___LIME Modules___"""
from . import eli, elref
from ...datatypes.datatypes import MoonData, IrradianceCoefficients

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
        * get_eli: Calculates the extra-terrestrial lunar irradiance in Wm⁻²/nm for some given parameters.
        * get_elref: Calculates the extra-terrestrial lunar reflectance in fractions of unity for some
            given parameters.
    """

    @staticmethod
    @abstractmethod
    def get_eli(
        wavelengths: Union[float, List[float]],
        moon_data: MoonData,
        coefficients: IrradianceCoefficients,
    ) -> Union[float, List[float]]:
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
        float | list of float
            The extraterrestrial lunar irradiance/s calculated. It will be a list if parameter
            "wavelengths" was a list.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_elref(
        wavelengths: Union[float, List[float]],
        moon_data: MoonData,
        coefficients: IrradianceCoefficients,
    ) -> Union[float, List[float]]:
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
        float | list of float
            The extraterrestrial lunar reflectance/s calculated. It will be a list if parameter
            "wavelengths" was a list.
        """
        pass


class ROLO(IROLO):
    """
    Class that implements the methods of this module.

    It exports the following functions:
        * get_eli: Calculates the extra-terrestrial lunar irradiance in Wm⁻²/nm for some given parameters.
    """

    @staticmethod
    def get_eli(
        wavelengths: Union[float, List[float]],
        moon_data: MoonData,
        coefficients: IrradianceCoefficients,
    ) -> Union[float, List[float]]:
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
        float | list of float
            The extraterrestrial lunar irradiance/s calculated. It will be a list if parameter
            "wavelengths" was a list.
        """
        if isinstance(wavelengths, list):
            elis = []
            for wlen in wavelengths:
                elis.append(eli.calculate_eli(wlen, moon_data, coefficients))
            return elis
        return eli.calculate_eli(wavelengths, moon_data, coefficients)

    @staticmethod
    def get_elref(
        wavelengths: Union[float, List[float]],
        moon_data: MoonData,
        coefficients: IrradianceCoefficients,
    ) -> Union[float, List[float]]:
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
        float | list of float
            The extraterrestrial lunar reflectance/s calculated. It will be a list if parameter
            "wavelengths" was a list.
        """
        if isinstance(wavelengths, list):
            elrefs = []
            for wlen in wavelengths:
                elrefs.append(elref.calculate_elref(wlen, moon_data, coefficients))
            return elrefs
        return elref.calculate_elref(wavelengths, moon_data, coefficients)
