"""
This module performs spectral intetgrations over spectral response functions and
more data.

It exports the following classes:
    * ISpectralIntegration - Interface that contains the methods of this module.
    * SpectralIntegration - Class that implements the methods exported by this module.
"""

"""___Built-In Modules___"""
from typing import List, Union
from abc import ABC, abstractmethod

"""___Third-Party Modules___"""
import numpy as np
import punpy

"""___LIME_TBX Modules___"""
from lime_tbx.datatypes.datatypes import (
    SRFChannel,
    SpectralResponseFunction,
    SpectralData,
)

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2022"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


class ISpectralIntegration(ABC):
    @abstractmethod
    def integrate_elis(
        self, srf: SpectralResponseFunction, elis_lime: SpectralData
    ) -> Union[List[float], List[List[float]]]:
        """
        Integrate the irradiances and obtain the signal.
        """
        pass

    @abstractmethod
    def u_integrate_elis(
        self, srf: SpectralResponseFunction, elis_lime: SpectralData
    ) -> Union[List[float], List[List[float]]]:
        """
        Calculate the uncertainties of the irradiances integration.
        """
        pass


class SpectralIntegration(ISpectralIntegration):
    def __init__(self, MCsteps=1000):
        self.prop = punpy.MCPropagation(MCsteps)

    def _convolve_srf(
        self, ch_wlens: np.ndarray, ch_srf: np.ndarray, ch_elis: np.ndarray
    ) -> Union[float, np.ndarray]:
        """
        Convolve ch_srf and ch_elis over ch_wlens, generating the signal.
        """
        ch_signal = np.trapz(ch_srf * ch_elis, ch_wlens)
        return ch_signal

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
            ch_wlens = np.array([w for w in ch.spectral_response.keys() if w in wlens])
            ch_srf = np.array([ch.spectral_response[k] for k in ch_wlens])
            elis_ids = np.where(np.in1d(wlens, ch_wlens))[0]
            ch_signals = []
            for subelis in elis:
                ch_elis = subelis[elis_ids]
                ch_signals.append(self._convolve_srf(ch_wlens, ch_srf, ch_elis))
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
            ch_wlens = np.array([w for w in ch.spectral_response.keys() if w in wlens])
            ch_srf = np.array([ch.spectral_response[k] for k in ch_wlens])
            elis_ids = np.where(np.in1d(wlens, ch_wlens))[0]
            u_ch_signals = []
            for i, subelis in enumerate(elis):
                ch_elis = subelis[elis_ids]
                u_ch_elis = u_elis[i][elis_ids]
                u_ch_signals.append(
                    self.prop.propagate_random(
                        self._convolve_srf,
                        [ch_wlens, ch_srf, ch_elis],
                        [None, None, u_ch_elis],
                        corr_x=[None, None, None],
                    )
                )
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
