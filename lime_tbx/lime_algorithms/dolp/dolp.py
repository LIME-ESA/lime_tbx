"""
This module calculates the extra-terrestrial lunar disk degree of polarization.

It exports the following classes:
    * IDOLP - Interface that contains the methods of this module.
    * DOLP - Class that implements the methods exported by this module.

It follows equations described in the following documents:
- Lunar irradiance model algorithm and theoretical basis document D7.
"""

"""___Built-In Modules___"""
from typing import List, Tuple
import math
from abc import ABC, abstractmethod

"""___Third-Party Modules___"""
import numpy as np

"""___NPL Modules"""
import punpy

"""___LIME TBX Modules___"""
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


def _measurement_func_polarization(mpa: float, a_coeffs: np.ndarray) -> np.ndarray:
    result = (
        a_coeffs[:, 0] * mpa
        + a_coeffs[:, 1] * mpa**2
        + a_coeffs[:, 2] * mpa**3
        + a_coeffs[:, 3] * mpa**4
    )
    return result


class DOLP(IDOLP):
    def _get_direct_polarized(
        self,
        mpa: float,
        coefficients: PolarizationCoefficients,
    ) -> float:
        if mpa > 0:  # is this sign ok?
            a_coeffs = np.array(coefficients.pos_coeffs)
        else:
            a_coeffs = np.array(coefficients.neg_coeffs)
        return _measurement_func_polarization(mpa, a_coeffs)

    def _get_direct_polarized_individual(
        self,
        wlen: float,
        mpa: float,
        coefficients: PolarizationCoefficients,
    ) -> float:
        if mpa > 0:  # is this sign ok?
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

    def _calculate_polar_unc(
        self,
        mpa_degrees: float,
        coefficients: PolarizationCoefficients,
    ) -> Tuple[np.ndarray, np.ndarray]:
        prop = punpy.MCPropagation(100, MCdimlast=True)
        if mpa_degrees > 0:  # is this sign ok?
            a_coeffs = np.array(coefficients.pos_coeffs)
            unc_coeffs = np.array(coefficients.pos_unc)
            corr_coeffs = np.array(coefficients.p_pos_err_corr_data)
        else:
            a_coeffs = np.array(coefficients.neg_coeffs)
            unc_coeffs = np.array(coefficients.neg_unc)
            corr_coeffs = np.array(coefficients.p_neg_err_corr_data)
        unc, corr = prop.propagate_random(
            _measurement_func_polarization,
            [
                mpa_degrees,
                a_coeffs,
            ],
            [
                None,
                unc_coeffs,
            ],
            corr_x=[
                None,
                corr_coeffs,
            ],
            return_corr=True,
            allow_some_nans=False,
        )

        return unc, corr

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
        wavelengths = coefficients.get_wavelengths()
        polarizations = self._get_direct_polarized(mpa_degrees, coefficients)
        # ds_pol = SpectralData.make_polarization_ds(wavelengths, polarizations, None, None)
        if not skip_uncs:
            uncs, corr = self._calculate_polar_unc(mpa_degrees, coefficients)
            # uncs = ds_pol.u_polarization.values
        else:
            uncs = np.zeros(polarizations.shape)
        print(uncs)
        ds_pol = SpectralData.make_polarization_ds(
            wavelengths, polarizations, uncs, corr
        )
        return SpectralData(
            np.array(wavelengths),
            polarizations,
            uncs,
            ds_pol,
        )
