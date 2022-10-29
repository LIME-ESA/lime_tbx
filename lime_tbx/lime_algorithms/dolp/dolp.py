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
from ...datatypes.datatypes import PolarizationCoefficients

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2022"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
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
        wavelengths: List[float],
        mpa_degrees: float,
        coefficients: PolarizationCoefficients,
    ) -> List[float]:
        """
        Calculation of the degree of linear polarization.

        Parameters
        ----------
        wavelengths: list of float
            List of wavelengths from which the polarization needs to be calculated.
        mpa_degrees: float
            Moon phase angle in degrees.
        coefficients: PolarizationCoefficients
            Coefficients needed in the dolp algorithm

        Returns
        -------
        polarizations: list of float
            List with the degrees of polarization for each given wavelength.
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

    def _get_interpolated_polarized(
        self,
        wlen: float,
        mpa: float,
        coefficients: PolarizationCoefficients,
    ) -> float:
        wvlens = coefficients.get_wavelengths()
        if wlen < wvlens[0]:
            # The extrapolation done is "nearest"
            return self._get_direct_polarized(wvlens[0], mpa, coefficients)
        if wlen > wvlens[-1]:
            # The extrapolation done is "nearest"
            return self._get_direct_polarized(wvlens[-1], mpa, coefficients)
        if wlen in wvlens:
            return self._get_direct_polarized(wlen, mpa, coefficients)
        near_left = -math.inf
        near_right = math.inf
        for wvlen in wvlens:
            if near_left < wvlen < wlen:
                near_left = wvlen
            elif wlen < wvlen < near_right:
                near_right = wvlen
        x_values = [near_left, near_right]
        y_values = []
        y_values.append(self._get_direct_polarized(x_values[0], mpa, coefficients))
        y_values.append(self._get_direct_polarized(x_values[1], mpa, coefficients))
        return np.interp(wlen, x_values, y_values)

    def get_polarized(
        self,
        wavelengths: List[float],
        mpa_degrees: float,
        coefficients: PolarizationCoefficients,
    ) -> List[float]:
        """
        Calculation of the degree of linear polarization.

        Parameters
        ----------
        wavelengths: list of float
            List of wavelengths from which the polarization needs to be calculated.
        mpa_degrees: float
            Moon phase angle in degrees.
        coefficients: PolarizationCoefficients
            Coefficients needed in the dolp algorithm

        Returns
        -------
        polarizations: list of float
            List with the degrees of polarization for each given wavelength.
        """
        polarizations = []
        mpa = mpa_degrees
        for wlen in wavelengths:
            result = self._get_interpolated_polarized(wlen, mpa, coefficients)
            polarizations.append(result)
        return polarizations
