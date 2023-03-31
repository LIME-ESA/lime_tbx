"""
This module calculates the extra-terrestrial lunar disk degree of polarization.

It exports the following classes:
    * IDOLP - Interface that contains the methods of this module.
    * DOLP - Class that implements the methods exported by this module.

It follows equations described in the following documents:
- Lunar irradiance model algorithm and theoretical basis document D7.
"""

"""___Built-In Modules___"""
from typing import List
import math
from abc import ABC, abstractmethod

"""___Third-Party Modules___"""
import numpy as np

"""___NPL Modules___"""
from lime_tbx.datatypes.datatypes import PolarizationCoefficients, SpectralData

"""___Authorship___"""
__author__ = "Pieter De Vis, Jacob Fahy, Javier Gatón Herguedas, Ramiro González Catón, Carlos Toledano"
__created__ = "01/02/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


class IDOLP(ABC):
    """
    Interface that contains the methods of this module.

    It exports the following functions:
        * get_polarized: Calculates the extra-terrestrial lunar polarization in fractions of unity for some
            given parameters.
    """

    @abstractmethod
    def get_polarized(
        self,
        mpa_degrees: float,
        coefficients: PolarizationCoefficients,
        skip_uncs: bool = False,
    ) -> SpectralData:
        """
        Calculation of the degree of linear polarization.

        Parameters
        ----------
        mpa_degrees: float
            Moon phase angle in degrees.
        coefficients: PolarizationCoefficients
            Coefficients needed in the dolp algorithm
        skip_uncs: bool
            Flag for skipping the calculation of uncertainties.

        Returns
        -------
        polarizations: SpectralData
            SpectralData with the degrees of polarization for each given wavelength.
        """
        pass


class DOLP(IDOLP):
    def _get_direct_polarized(
        self,
        wlen: float,
        mpa: float,
        coefficients: PolarizationCoefficients,
    ) -> float:
        if mpa < 0:
            a_coeffs = coefficients.get_coefficients_positive(wlen)
        else:
            a_coeffs = coefficients.get_coefficients_negative(wlen)
        result = (
            a_coeffs[0] * mpa
            + a_coeffs[1] * mpa**2
            + a_coeffs[2] * mpa**3
            + a_coeffs[3] * mpa**4
        )
        return result

    def get_polarized(
        self,
        mpa_degrees: float,
        coefficients: PolarizationCoefficients,
        skip_uncs: bool = False,
    ) -> SpectralData:
        """
        Calculation of the degree of linear polarization.

        Parameters
        ----------
        mpa_degrees: float
            Moon phase angle in degrees.
        coefficients: PolarizationCoefficients
            Coefficients needed in the dolp algorithm
        skip_uncs: bool
            Flag for skipping the calculation of uncertainties.

        Returns
        -------
        polarizations: SpectralData
            List with the degrees of polarization for each given wavelength.
        """
        polarizations = []
        mpa = mpa_degrees
        wavelengths = coefficients.get_wavelengths()
        for wlen in wavelengths:
            result = self._get_direct_polarized(wlen, mpa, coefficients)
            polarizations.append(result)
        polarizations = np.array(polarizations)
        ds_pol = SpectralData.make_polarization_ds(wavelengths, polarizations, None, None)
        if not skip_uncs:
            uncs = ds_pol.u_polarization.values
        else:
            uncs = np.zeros(polarizations.shape)
        return SpectralData(
            np.array(wavelengths),
            polarizations,
            uncs,
            ds_pol,
        )
