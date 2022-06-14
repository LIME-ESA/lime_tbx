"""describe class"""

"""___Built-In Modules___"""
from typing import List, Union, Tuple

"""___Third-Party Modules___"""
# import here

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
    def convolve_srf(srf, asd_irradiances):
        pass


class SpectralIntegration(ISpectralIntegration):
    def convolve_srf(srf, asd_irradiances):
        pass

    @staticmethod
    def integrate_elis(
        srf: SpectralResponseFunction, elis: Union[List[float], List[List[float]]]
    ) -> Union[List[float], List[List[float]]]:
        signals = []
        wlens = srf.get_wavelengths()
        if len(elis) == 0:
            return []
        wasnt_lists = False
        if not isinstance(elis[0], list):
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
