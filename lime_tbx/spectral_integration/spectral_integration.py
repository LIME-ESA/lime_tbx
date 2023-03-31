"""
This module performs spectral integrations over spectral response functions and
more data.

It exports the following classes:
    * ISpectralIntegration - Interface that contains the methods of this module.
    * SpectralIntegration - Class that implements the methods exported by this module.
"""

"""___Built-In Modules___"""
from typing import List, Union
from abc import ABC, abstractmethod
import os

"""___Third-Party Modules___"""
import numpy as np
import punpy
from matheo.band_integration import band_integration

"""___LIME_TBX Modules___"""
from lime_tbx.datatypes.datatypes import (
    SpectralResponseFunction,
    SpectralData,
    SRF_fwhm,
    SRFChannel,
)
from lime_tbx.datatypes import constants

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2022"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


_ASD_FILE = "assets/asd_fwhm.csv"
_CIMEL_FILE = "assets/responses_1088_13112020.txt"
_INTERPOLATED_FILE = "../../coeff_data/interpolated_model_fwhm.csv"


def get_default_srf():
    dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    dir_path = os.path.join(dir_path, "coeff_data")
    # dir_path = os.path.join(dir_path, "lime_tbx", "spectral_integration", "assets")
    data = np.genfromtxt(
        os.path.join(dir_path, "interpolated_model_fwhm.csv"), delimiter=","
    )
    wavs = data[:, 0]
    spectral_response = {i: 1.0 for i in wavs}
    ch = SRFChannel(
        (constants.MAX_WLEN - constants.MIN_WLEN) / 2,
        constants.DEFAULT_SRF_NAME,
        spectral_response,
    )
    srf = SpectralResponseFunction(constants.DEFAULT_SRF_NAME, [ch])
    return srf


class ISpectralIntegration(ABC):
    """Interface that contains the methods of this module.

    It exports the following functions:
        * integrate_elis() - Integrate the irradiances and obtain the signal.
        * u_integrate_elis() - Calculate the uncertainties of the irradiances integration.
    """

    @abstractmethod
    def integrate_elis(
        self, srf: SpectralResponseFunction, elis_lime: SpectralData
    ) -> Union[List[float], List[List[float]]]:
        """
        Integrate the irradiances and obtain the signal.

        Parameters
        ----------
        srf: SpectralResponseFunction
            Spectral response function to integrate to
        elis_lime: SpectralData
            Data to integrate to the srf

        Returns
        -------
        integr_data: list of float or list of list of float
            Integrated signals. List of float if elis_lime.data values are floats,
            list of list of float if they are lists or ndarrays.
        """
        pass

    @abstractmethod
    def u_integrate_elis(
        self, srf: SpectralResponseFunction, elis_lime: SpectralData
    ) -> Union[List[float], List[List[float]]]:
        """
        Calculate the uncertainties of the irradiances integration.

        Parameters
        ----------
        srf: SpectralResponseFunction
            Spectral response function to integrate to
        elis_lime: SpectralData
            Data to integrate to the srf

        Returns
        -------
        u_integr_data: list of float or list of list of float
            Uncertainties of the integrated signals. List of float if elis_lime.data
            values are floats, list of list of float if they are lists or ndarrays.
        """
        pass


class SpectralIntegration(ISpectralIntegration):
    """Class that implements the methods exported by this module"""

    def __init__(self, MCsteps=1000):
        self.prop = punpy.MCPropagation(MCsteps, parallel_cores=1)
        self.asd_srf = self._read_srf_asd()
        self.cimel_srf = self._read_srf_cimel()
        self.interpolated_srf = self._read_default_srf_interpolated()

    def _read_srf_asd(self) -> SpectralResponseFunction:
        """
        read asd fwhm and make SRF
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        data = np.genfromtxt(os.path.join(dir_path, _ASD_FILE), delimiter=",")
        srf = SRF_fwhm("asd", data[:, 0], data[:, 1], "gaussian")
        return srf

    def _read_default_srf_interpolated(self) -> SpectralResponseFunction:
        """
        read asd fwhm and make SRF
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        data = np.genfromtxt(os.path.join(dir_path, _INTERPOLATED_FILE), delimiter=",")
        srf = SRF_fwhm("interpolated", data[:, 0], data[:, 1], "gaussian")
        return srf

    def set_srf_interpolated(self, fwhm, sampling, shape, write=False) -> SpectralResponseFunction:
        """
        read asd fwhm and make SRF
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        wavs = np.arange(constants.MIN_WLEN, constants.MAX_WLEN+sampling, sampling)
        fwhms = fwhm * np.ones_like(wavs)
        if write:
            id_str=("%s_%s_%s"% (fwhm,sampling, shape)).replace(".","p")
            with open(os.path.join(dir_path,"assets/interpolated_model_fwhm_"+id_str+".csv"), "w") as f:
                for i in range(len(wavs)):
                    f.write("%s,%s\n" % (wavs[i], fwhms[i]))
        self.interpolated_srf = SRF_fwhm("interpolated", wavs, fwhms, shape)

    def _read_srf_cimel(self) -> SpectralResponseFunction:
        """
        read asd fwhm and make SRF
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        data = np.genfromtxt(os.path.join(dir_path, _CIMEL_FILE), delimiter=",")
        cimel_wavs = [440, 500, 675, 870, 1020, 1640]
        srflist = [None] * len(cimel_wavs)

        for i in range(len(cimel_wavs)):
            srf = {
                data[j, 2 * i]: data[j, 2 * i + 1]
                for j in range(len(data[:, 2 * i + 1]))
                if data[j, 2 * i] > 0
            }
            channel = SRFChannel(cimel_wavs[i], str(cimel_wavs[i]), srf)
            srflist[i] = channel

        return SpectralResponseFunction("cimel", srflist)

    def integrate_cimel(self, data: np.ndarray, wlens: np.ndarray) -> np.ndarray:
        y = self.integrate_elis_xy(self.cimel_srf, data, wlens)
        return y

    def integrate_solar_asd(self, data: np.ndarray, wlens: np.ndarray, ) -> np.ndarray:
        return band_integration.pixel_int(
            data,
            wlens,
            self.asd_srf.get_wavelengths(),
            (self.asd_srf.get_values() ** 2 - 0.01) ** 0.5,
            band_shape=self.asd_srf.get_shape()
        )

    def integrate_solar_interpolated_default(
        self, data: np.ndarray, wlens: np.ndarray
    ) -> np.ndarray:
        return band_integration.pixel_int(
            data,
            wlens,
            self.interpolated_srf.get_wavelengths(),
            (self.interpolated_srf.get_values() ** 2 - 0.01) ** 0.5,
            band_shape=self.interpolated_srf.get_shape()
        )

    def integrate_solar_interpolated_gaussian(
            self, data: np.ndarray, wlens: np.ndarray
    ) -> np.ndarray:
        # subtracting a fwhm of 0.1 in quadrature to account for fwhm of TSIS solar irradiance
        return band_integration.pixel_int(
            data,
            wlens,
            self.interpolated_srf.get_wavelengths(),
            (self.interpolated_srf.get_values() ** 2 - 0.01) ** 0.5,
            band_shape="gaussian"
        )

    def integrate_solar_interpolated_triangle(
        self, data: np.ndarray, wlens: np.ndarray
    ) -> np.ndarray:
        return band_integration.pixel_int(
            data,
            wlens,
            self.interpolated_srf.get_wavelengths(),
            (self.interpolated_srf.get_values() ** 2 - 0.01) ** 0.5,
            band_shape="triangle"
        )

    def integrate_elis_xy(
        self, srf: SpectralResponseFunction, data: np.ndarray, wlens: np.ndarray
    ) -> np.ndarray:
        """

        :param srf:
        :type srf:
        :param data:
        :type data:
        :param wlens:
        :type wlens:
        :return:
        :rtype:
        """
        signals = np.zeros(len(srf.channels))
        for ich, ch in enumerate(srf.channels):
            ch_wlens = np.array([w for w in ch.spectral_response.keys()])
            ch_srf = np.array([ch.spectral_response[k] for k in ch_wlens])
            signals[ich] = band_integration.band_int(data, wlens, ch_srf, ch_wlens)
        return signals

    def _convolve_srf(
        self, ch_wlens: np.ndarray, ch_srf: np.ndarray, ch_elis: np.ndarray
    ) -> Union[float, np.ndarray]:
        """
        Convolve ch_srf and ch_elis over ch_wlens, generating the signal.
        """
        divider = np.trapz(ch_srf, ch_wlens)
        if isinstance(divider, np.ndarray):
            divider[divider == 0] = 1
        elif divider == 0:
            divider = 1
        ch_signal = np.trapz(ch_srf * ch_elis, ch_wlens) / divider
        return ch_signal

    def _interpolate_and_convolve_srf(
        self,
        ch_wlens: np.ndarray,
        ch_srf: np.ndarray,
        elis_wlens: np.ndarray,
        elis: np.ndarray,
    ) -> Union[float, np.ndarray]:
        ch_elis = np.interp(ch_wlens, elis_wlens, elis)
        return self._convolve_srf(ch_wlens, ch_srf, ch_elis)

    def integrate_elis(
        self, srf: SpectralResponseFunction, elis_lime: SpectralData
    ) -> Union[List[float], List[List[float]]]:
        signals = []
        wlens = elis_lime.wlens
        elis = elis_lime.data
        if len(elis) == 0:
            return []
        wasnt_lists = False
        if not isinstance(elis[0], list) or not isinstance(elis[0], np.ndarray):
            wasnt_lists = True
            elis = np.array([elis])
        for ch in srf.channels:
            ch_wlens = np.array(
                [
                    w
                    for w in ch.spectral_response.keys()
                    if min(wlens) <= w <= max(wlens)
                ]
            )
            ch_srf = np.array([ch.spectral_response[k] for k in ch_wlens])
            elis_ids = np.where(np.in1d(wlens, ch_wlens))[0]
            ch_signals = []
            for subelis in elis:
                ch_elis = subelis[elis_ids]
                ch_signals.append(
                    self._interpolate_and_convolve_srf(ch_wlens, ch_srf, wlens, subelis)
                )
                # ch_signals.append(self._convolve_srf(ch_wlens, ch_srf, ch_elis))
            signals.append(ch_signals)
        if wasnt_lists:
            signals = [s[0] for s in signals]
        return signals

    def u_integrate_elis(
        self, srf: SpectralResponseFunction, elis_lime: SpectralData
    ) -> Union[List[float], List[List[float]]]:
        u_signals = []
        wlens = elis_lime.wlens
        elis = elis_lime.data
        u_elis = elis_lime.uncertainties
        if len(elis) == 0:
            return []
        wasnt_lists = False
        if not isinstance(elis[0], list) or not isinstance(elis[0], np.ndarray):
            wasnt_lists = True
            elis = np.array([elis])
            u_elis = np.array([u_elis])
        for ch in srf.channels:
            ch_wlens = np.array(
                [
                    w
                    for w in ch.spectral_response.keys()
                    if min(wlens) <= w <= max(wlens)
                ]
            )
            ch_srf = np.array([ch.spectral_response[k] for k in ch_wlens])
            elis_ids = np.where(np.in1d(wlens, ch_wlens))[0]
            u_ch_signals = []
            for i, subelis in enumerate(elis):
                ch_elis = subelis[elis_ids]
                u_ch_elis = u_elis[i][elis_ids]
                u_ch_signal = np.array([])
                if len(ch_wlens) > 0:
                    u_ch_signal = self.prop.propagate_random(
                        self._interpolate_and_convolve_srf,
                        [ch_wlens, ch_srf, wlens, subelis],
                        [None, None, None, u_elis[i]],
                        corr_x=[None, None, None, None],
                    )
                u_ch_signals.append(u_ch_signal)
            u_signals.append(u_ch_signals)
        if wasnt_lists:
            u_signals = [s[0] for s in u_signals]
        return u_signals


"""
    @staticmethod
    def integrate_elis_old(
        srf: SpectralResponseFunction, elis: Union[List[float], List[List[float]]]
    ) -> Union[List[float], List[List[float]]]:
        signals = []
        wlens = srf.get_wavelengths()
        if len(elis) == 0:
            return []
        wasnt_lists = False
        if not isinstance(elis[0], list) or not isinstance(elis[0], np.ndarray):
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
        return signals"""
