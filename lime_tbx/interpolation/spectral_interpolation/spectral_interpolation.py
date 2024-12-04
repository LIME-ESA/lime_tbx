"""Module in charge of performing spectral interpolation.

It exports the following classes:
    * ISpectralInterpolation - Interface that contains all the abstract functions
        exported by this module.
    * SpectralInterpolation - Class that implements all the functions exported by
        this module.
"""

"""___Built-In Modules___"""
from abc import ABC, abstractmethod

"""___Third-Party Modules___"""
import numpy as np
from numpy.typing import NDArray
import comet_maths as cm

"""___NPL Modules___"""
# import here

"""___LIME_TBX Modules___"""
import lime_tbx.interpolation.interp_data.interp_data as idata
from lime_tbx.datatypes.datatypes import MoonData, SpectralData
from lime_tbx.spectral_integration.spectral_integration import (
    SpectralIntegration,
    ISpectralIntegration,
)


"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2022"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


class ISpectralInterpolation(ABC):
    """Interface that contains all the abstract functions exported by this module.

    It exports the following functions:
        * get_best_interp_reference() - Get the best reflectance interpolation reference
            for the given data for the currently selected interpolation spectrum.
        * get_best_polar_interp_reference() - Get the best polarisation interpolation
            reference for the given data for the currently selected interpolation spectrum.
        * get_interpolated_refl() - Interpolates the cimel values to final_wav using the
            given spectrum data as reference.
        * get_interpolated_refl_unc() - Calculate the uncertainties of the interpolation of
            the cimel_refl values to final_wav using the given interpolation spectrum data
            as reference.
    """

    @abstractmethod
    def get_best_interp_reference(self, moon_data: MoonData) -> SpectralData:
        """
        Get the best reflectance interpolation reference for the given data for the currently
        selected interpolation spectrum.

        Parameters
        ----------
        moon_data: MoonData
            Moon data for which the best interpolation reference will be returned.

        Returns
        -------
        interp_reference: SpectralData
            Best interpolation reference for the given data.
        """
        pass

    @abstractmethod
    def get_best_polar_interp_reference(self, moon_data: MoonData) -> SpectralData:
        """
        Get the best polarisation interpolation reference for the given data for the currently
        selected interpolation spectrum.

        Parameters
        ----------
        moon_data: MoonData
            Moon data for which the best interpolation reference will be returned.

        Returns
        -------
        interp_reference: SpectralData
            Best interpolation reference for the given data.
        """
        pass

    @abstractmethod
    def get_interpolated_refl(
        self,
        cimel_wav: NDArray[np.float_],
        cimel_refl: NDArray[np.float_],
        asd_wav: NDArray[np.float_],
        asd_refl: NDArray[np.float_],
        final_wav: NDArray[np.float_],
    ) -> NDArray[np.float_]:
        """Interpolates the cimel_refl values to final_wav using the given interpolation spectrum data as reference.

        Parameters
        ----------
        cimel_wav: np.ndarray of float
            Cimel wavelengths.
        cimel_ref: np.ndarray of float
            Cimel data values.
        asd_wav: np.ndarray of float
            Interpolation spectrum wavelengths.
        asd_refl: np.ndarray of float
            Interpolation spectrum data values.
        final_wav: np.ndarray of float
            Wavelengths at wich the data will be interpolated.

        Returns
        -------
        interp_refl: np.ndarray of float
            Interpolated data values for the final_wav wavelengths.
        """
        pass

    @abstractmethod
    def get_interpolated_refl_unc(
        self,
        cimel_wav: NDArray[np.float_],
        cimel_refl: NDArray[np.float_],
        asd_wav: NDArray[np.float_],
        asd_refl: NDArray[np.float_],
        final_wav: NDArray[np.float_],
        u_cimel_refl: NDArray[np.float_],
        u_asd_refl: NDArray[np.float_],
        corr_cimel_refl=None,
        corr_asd_refl=None,
    ) -> NDArray[np.float_]:
        """
        Calculate the uncertainties of the interpolation of the cimel_refl values to final_wav
        using the given interpolation spectrum data as reference.

        Parameters
        ----------
        cimel_wav: np.ndarray of float
            Cimel wavelengths.
        cimel_ref: np.ndarray of float
            Cimel data values.
        asd_wav: np.ndarray of float
            Interpolation spectrum wavelengths.
        asd_refl: np.ndarray of float
            Interpolation spectrum data values.
        final_wav: np.ndarray of float
            Wavelengths at wich the data would be interpolated.
        u_cimel_refl: np.ndarray of float
            Uncertainties of the cimel data.
        u_asd_refl: np.ndarray of float
            Uncertainties of the interpolation spectrum data.
        corr_cimel_refl: np.ndarray of float
            Error correlation of the cimel data
        corr_asd_refl: np.ndarray of float
            Error correlation of the interpolation spectrum data

        Returns
        -------
        interp_refl_unc: np.ndarray of float
            Uncertainties of the interpolated data values for the final_wav wavelengths.
        """
        pass


class SpectralInterpolation(ISpectralInterpolation):
    """Class that implements all the functions exported by this module."""

    def __init__(self, MCsteps=100):
        self.si: ISpectralIntegration = SpectralIntegration()
        self.MCsteps = MCsteps

    def _get_mock_asd_reference(self, moon_data: MoonData):
        mock = idata.get_best_asd_data(moon_data.mpa_degrees)
        mock.data.fill(1)
        return mock

    def get_best_interp_reference(self, moon_data: MoonData):
        name = idata.get_interpolation_spectrum_name()
        if name == idata.SPECTRUM_NAME_APOLLO16:
            return idata.get_apollo16_data()
        elif name == idata.SPECTRUM_NAME_BRECCIA:
            return idata.get_breccia_data()
        elif name == idata.SPECTRUM_NAME_COMPOSITE:
            return idata.get_composite_data()
        else:
            return idata.get_best_asd_data(moon_data.mpa_degrees)

    def get_best_polar_interp_reference(self, moon_data: MoonData):
        return idata.get_linear_polar_data()

    def get_interpolated_refl(
        self,
        cimel_wav: NDArray[np.float_],
        cimel_refl: NDArray[np.float_],
        asd_wav: NDArray[np.float_],
        asd_refl: NDArray[np.float_],
        final_wav: NDArray[np.float_],
    ) -> NDArray[np.float_]:
        integr_cimel = self.si.integrate_cimel(asd_refl, asd_wav, cimel_wav)
        interp_asd_cimel = cm.interpolate_1d(
            asd_wav, asd_refl, cimel_wav, method="linear"
        )
        corr_srf_cimel = integr_cimel - interp_asd_cimel
        return cm.interpolate_1d_along_example(
            cimel_wav,
            cimel_refl - corr_srf_cimel,
            asd_wav,
            asd_refl,
            final_wav,
            method="linear",
            method_hr="linear",
        )
        # from scipy import interpolate
        # f = interpolate.interp1d(cimel_wav, cimel_refl, fill_value="extrapolate")
        # yy = f(final_wav)
        # return yy

    def get_interpolated_refl_unc(
        self,
        cimel_wav: NDArray[np.float_],
        cimel_refl: NDArray[np.float_],
        asd_wav: NDArray[np.float_],
        asd_refl: NDArray[np.float_],
        final_wav: NDArray[np.float_],
        u_cimel_refl: NDArray[np.float_],
        u_asd_refl: NDArray[np.float_],
        corr_cimel_refl: NDArray[np.float_] = None,
        corr_asd_refl: NDArray[np.float_] = None,
    ) -> NDArray[np.float_]:
        corr_srf_cimel = self.si.integrate_cimel(
            asd_refl, asd_wav, cimel_wav
        ) - cm.interpolate_1d(asd_wav, asd_refl, cimel_wav, method="linear")
        return cm.interpolate_1d_along_example(
            cimel_wav,
            cimel_refl - corr_srf_cimel,
            asd_wav,
            asd_refl,
            final_wav,
            method="linear",
            method_hr="linear",
            u_y_i=u_cimel_refl,
            u_y_hr=u_asd_refl,
            unc_methods=["linear", "quadratic"],
            corr_y_i=corr_cimel_refl,
            corr_y_hr=corr_asd_refl,
            return_uncertainties=True,
            return_corr=True,
            parallel_cores=1,
            MCsteps=self.MCsteps,
        )
