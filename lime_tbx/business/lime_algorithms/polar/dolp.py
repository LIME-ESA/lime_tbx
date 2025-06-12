"""
This module calculates the extra-terrestrial lunar disk degree of polarisation.

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
from lime_tbx.common.datatypes import (
    PolarisationCoefficients,
    SpectralData,
    AOLPCoefficients,
)
from lime_tbx.common import constants


class IDOLP(ABC):
    """
    Interface that contains the methods of this module.

    It exports the following functions:
        * get_polarized: Calculates the extra-terrestrial lunar polarisation in fractions of unity for some
            given parameters.
    """

    @abstractmethod
    def get_polarized(
        self,
        mpa_degrees: float,
        coefficients: PolarisationCoefficients,
        skip_uncs: bool = False,
    ) -> SpectralData:
        """
        Calculation of the degree of linear polarisation.

        Parameters
        ----------
        mpa_degrees: float
            Moon phase angle in degrees.
        coefficients: PolarisationCoefficients
            Coefficients needed in the dolp algorithm
        skip_uncs: bool
            Flag for skipping the calculation of uncertainties.

        Returns
        -------
        polarisations: SpectralData
            SpectralData with the degrees of polarisation for each given wavelength.
        """
        pass


def _measurement_func_polarisation(mpa: float, a_coeffs: np.ndarray) -> np.ndarray:
    quant_coeffs = len(a_coeffs[0])
    if quant_coeffs == 4:
        result = sum(a_coeffs[:, i] * mpa ** (i + 1) for i in range(quant_coeffs))
    else:  # elif quant_coeffs == 5:
        result = sum(a_coeffs[:, i] * mpa**i for i in range(quant_coeffs))
    return result


class DOLP(IDOLP):
    def _get_direct_polarized(
        self,
        mpa: float,
        coefficients: PolarisationCoefficients,
    ) -> float:
        if mpa > 0:  # is this sign ok?
            a_coeffs = np.array(coefficients.pos_coeffs)
        else:
            a_coeffs = np.array(coefficients.neg_coeffs)
        return _measurement_func_polarisation(mpa, a_coeffs)

    def _get_direct_polarized_individual(
        self,
        wlen: float,
        mpa: float,
        coefficients: PolarisationCoefficients,
    ) -> float:
        if mpa > 0:  # is this sign ok?
            a_coeffs = coefficients.get_coefficients_positive(wlen)
        else:
            a_coeffs = coefficients.get_coefficients_negative(wlen)
        quant_coeffs = len(a_coeffs[0])
        result = sum(a_coeffs[:, i] * mpa ** (i + 1) for i in range(quant_coeffs))
        return result

    def _calculate_polar_unc(
        self,
        mpa_degrees: float,
        coefficients: PolarisationCoefficients,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # it fails if unc_coeffs == zeros
        prop = punpy.MCPropagation(100, 1, MCdimlast=True)
        if mpa_degrees > 0:  # is this sign ok?
            a_coeffs = np.array(coefficients.pos_coeffs)
            unc_coeffs = np.array(coefficients.pos_unc)
            corr_coeffs = np.array(coefficients.p_pos_err_corr_data)
        else:
            a_coeffs = np.array(coefficients.neg_coeffs)
            unc_coeffs = np.array(coefficients.neg_unc)
            corr_coeffs = np.array(coefficients.p_neg_err_corr_data)
        unc, corr = prop.propagate_random(
            _measurement_func_polarisation,
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
        coefficients: PolarisationCoefficients,
        skip_uncs: bool = False,
    ) -> SpectralData:
        """
        Calculation of the degree of linear polarisation.

        Parameters
        ----------
        mpa_degrees: float
            Moon phase angle in degrees.
        coefficients: PolarisationCoefficients
            Coefficients needed in the dolp algorithm
        skip_uncs: bool
            Flag for skipping the calculation of uncertainties.

        Returns
        -------
        polarisations: SpectralData
            List with the degrees of polarisation for each given wavelength.
        """
        polarisations = []
        wavelengths = coefficients.get_wavelengths()
        polarisations = self._get_direct_polarized(mpa_degrees, coefficients)
        if not skip_uncs:
            uncs, corr = self._calculate_polar_unc(mpa_degrees, coefficients)
            ds_pol = SpectralData.make_polarisation_ds(
                wavelengths, polarisations, uncs, corr
            )
        else:
            uncs = np.zeros(polarisations.shape)
            ds_pol = None
        return SpectralData(
            np.array(wavelengths),
            polarisations,
            uncs,
            ds_pol,
        )
