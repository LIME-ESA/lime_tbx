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

"""___NPL Modules___"""
import punpy
from comet_maths.interpolation.interpolation import Interpolator

"""___LIME_TBX Modules___"""
import lime_tbx.interpolation.interp_data.interp_data as idata
from lime_tbx.datatypes.datatypes import MoonData, SpectralData


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
        * get_best_polar_interp_reference() - Get the best polarization interpolation
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
        Get the best polarization interpolation reference for the given data for the currently
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
        corr_cimel_refl:
            TODO
            Error correlation of the cimel data
        corr_asd_refl:
            TODO
            Error correlation of the interpolation spectrum data

        Returns
        -------
        interp_refl_unc: np.ndarray of float
            Uncertainties of the interpolated data values for the final_wav wavelengths.
        """
        pass


class SpectralInterpolation(ISpectralInterpolation):
    """Class that implements all the functions exported by this module."""

    def __init__(
        self, relative=True, method_main="linear", method_hr="linear", MCsteps=1000
    ):
        self.intp = Interpolator(
            relative=relative,
            method=method_main,
            method_hr=method_hr,
            min_scale=0.3,
            # plot_residuals=True,
        )
        self.prop = punpy.MCPropagation(MCsteps)

    def _get_best_polar_asd_reference(self, moon_data: MoonData):
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
        return self._get_best_polar_asd_reference(moon_data)

    def get_interpolated_refl(
        self,
        cimel_wav: NDArray[np.float_],
        cimel_refl: NDArray[np.float_],
        asd_wav: NDArray[np.float_],
        asd_refl: NDArray[np.float_],
        final_wav: NDArray[np.float_],
    ) -> NDArray[np.float_]:
        # from scipy import interpolate
        # f = interpolate.interp1d(cimel_wav, cimel_refl, fill_value="extrapolate")
        # yy = f(final_wav)
        # return yy
        yy = self.intp.interpolate_1d_along_example(
            cimel_wav, cimel_refl, asd_wav, asd_refl, final_wav
        )
        return yy

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
        u_yy, corr_yy = self.prop.propagate_random(
            self.intp.interpolate_1d_along_example,
            [cimel_wav, cimel_refl, asd_wav, asd_refl, final_wav],
            [None, u_cimel_refl, None, u_asd_refl, None],
            corr_x=[None, corr_cimel_refl, None, corr_asd_refl, None],
            return_corr=True,
        )
        return u_yy, corr_yy
