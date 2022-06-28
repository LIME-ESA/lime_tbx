"""describe class"""

"""___Built-In Modules___"""
from typing import List, Union, Tuple

"""___Third-Party Modules___"""
import numpy as np
import punpy

"""___NPL Modules___"""
from lime_tbx.datatypes.datatypes import (
    ReflectanceCoefficients,
    ApolloIrradianceCoefficients,
    MoonData,
    PolarizationCoefficients,
    SpectralResponseFunction,
    SurfacePoint,
    CustomPoint,
    SpectralData,
)

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2022"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"

from abc import ABC, abstractmethod


class ISpectralIntegration(ABC):
    @abstractmethod
    def convolve_srf(ch_wlens, ch_srf, wlens, elis):
        pass

    @staticmethod
    def integrate_elis(
            srf: SpectralResponseFunction, elis_lime: SpectralData
    ) -> Union[List[float], List[List[float]]]:
        pass

    @staticmethod
    def u_integrate_elis(
            srf: SpectralResponseFunction, elis_lime: SpectralData
    ) -> Union[List[float], List[List[float]]]:
        pass

class SpectralIntegration(ISpectralIntegration):
    def __init__(self, MCsteps=1000):
        self.prop = punpy.MCPropagation(MCsteps)

    def convolve_srf(self, ch_wlens, ch_srf, wlens, elis):
        elis_ids=[wlens.index(wl) for wl in ch_wlens]
        print(ch_srf,elis[0])
        ch_signal=[np.trapz(ch_wlens,ch_srf*subelis[elis_ids]) for subelis in elis]

    def integrate_elis(self,
            srf: SpectralResponseFunction, elis_lime: SpectralData
    ) -> Union[List[float], List[List[float]]]:
        signals = []
        wlens = srf.get_wavelengths()
        elis=elis_lime.data
        if len(elis) == 0:
            return []
        wasnt_lists = False
        if not isinstance(elis[0], list) or not isinstance(elis[0], np.ndarray):
            wasnt_lists = True
            elis = [elis]
        for ch in srf.channels:
            ch_wlens = ch.spectral_response.keys()
            ch_srf = ch.spectral_response.values()
            ch_signal=self.convolve_srf(ch_wlens,ch_srf, wlens, elis)
            signals.append(ch_signal)
        if wasnt_lists:
            signals = [s[0] for s in signals]
        return signals

    def u_integrate_elis(self,
            srf: SpectralResponseFunction, elis_lime: SpectralData
        ) -> Union[List[float], List[List[float]]]:
        u_signals = []
        wlens = srf.get_wavelengths()
        elis=elis_lime.data
        u_elis=elis_lime.uncertainties
        if len(elis) == 0:
            return []
        wasnt_lists = False
        if not isinstance(elis[0], list) or not isinstance(elis[0], np.ndarray):
            wasnt_lists = True
            elis = [elis]
        for ch in srf.channels:
            ch_wlens = ch.spectral_response.keys()
            ch_srf = ch.spectral_response.values()

            u_ch_signal = self.prop.propagate_random(
                    self.convolve_srf,
                    [ch_wlens, ch_srf, wlens, elis],
                    [None, None, None, u_elis],
                    corr_x=[None, None, None, None],
                )

            u_signals.append(u_ch_signal)
        if wasnt_lists:
            u_signals = [s[0] for s in u_signals]
        return u_signals

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
        return signals
