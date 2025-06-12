"""
This module calculates the extra-terrestrial lunar disk angle of polarisation.

It exports the following class:
    * AOLP - Class that implements the methods exported by this module.

It follows equations described in the following documents:
- Lunar irradiance model algorithm and theoretical basis document D5.
"""

"""___Built-In Modules___"""
from typing import Tuple

"""___Third-Party Modules___"""
import numpy as np

"""___NPL Modules"""
import punpy

"""___LIME TBX Modules___"""
from lime_tbx.common.datatypes import (
    SpectralData,
    AOLPCoefficients,
)


def _measurement_func_aolp(mpa: float, coeffs: np.ndarray) -> np.ndarray:
    return sum(coeffs[:, i] * mpa**i for i in range(len(coeffs[0])))


class AOLP:
    def _get_direct_aolp(
        self,
        mpa: float,
        coefficients: AOLPCoefficients,
    ) -> float:
        return _measurement_func_aolp(mpa, coefficients.aolp_coeff)

    def _calculate_aolp_unc(
        self,
        mpa_degrees: float,
        coefficients: AOLPCoefficients,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # it fails if unc_coeffs == zeros
        prop = punpy.MCPropagation(100, 1, MCdimlast=True)
        coeffs = np.array(coefficients.aolp_coeff)
        unc_coeffs = np.array(coefficients.unc_coeff)
        corr_coeffs = np.array(coefficients.err_corr_data)
        unc, corr = prop.propagate_random(
            _measurement_func_aolp,
            [
                mpa_degrees,
                coeffs,
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

    def get_aolp(
        self,
        mpa_degrees: float,
        coefficients: AOLPCoefficients,
        skip_uncs: bool = False,
    ) -> SpectralData:
        """
        Calculation of the angle of linear polarisation.

        Parameters
        ----------
        mpa_degrees: float
            Moon phase angle in degrees.
        coefficients: AOLPCoefficients
            Coefficients needed in the aolp algorithm
        skip_uncs: bool
            Flag for skipping the calculation of uncertainties.

        Returns
        -------
        aolps: SpectralData
            List with the angle of polarisation for each given wavelength.
        """
        aolps = []
        wavelengths = coefficients.get_wavelengths()
        aolps = self._get_direct_aolp(mpa_degrees, coefficients)
        if not skip_uncs:
            uncs, corr = self._calculate_aolp_unc(mpa_degrees, coefficients)
            ds_pol = SpectralData.make_polarisation_ds(wavelengths, aolps, uncs, corr)
        else:
            uncs = np.zeros(aolps.shape)
            ds_pol = None
        return SpectralData(
            np.array(wavelengths),
            aolps,
            uncs,
            ds_pol,
        )
