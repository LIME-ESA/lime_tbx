"""
This module abstracts and encapsulates utilities related to performing
and showing comparisons between user measurements and the model output.

It exports the following methods:
    * average_comparisons - Returns a ComparisonData based on the average
        of a list of ComparisonData, now classified by channel.
    * sort_by_mpa - Returns a copy of the given list of ComparisonData
        but sorted by moon phase angle.
"""

"""___Built-In Modules___"""
from typing import List

"""___Third-Party Modules___"""
import numpy as np

"""___NPL Modules___"""
from lime_tbx.simulation.lime_simulation import is_ampa_valid_range
from lime_tbx.datatypes.datatypes import (
    ComparisonData,
    SpectralData,
    MoonData,
)

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "29/10/2024"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Production"


def sort_by_mpa(comparisons: List[ComparisonData]) -> List[ComparisonData]:
    """Returns a copy of the given list of ComparisonData but sorted by moon phase angle.

    Parameters
    ----------
    comparisons: list of ComparisonData
        List of ComparisonData that will be sorted by mpa.

    Returns
    -------
    sorted_comparisons: list of ComparisonData
        List with the same ComparisonData instances but sorted by the moon phase angle.
    """
    new_comparisons = []
    for c in comparisons:
        if c.observed_signal == None:
            new_comparisons.append(c)
            continue
        spectrals = [
            c.observed_signal,
            c.simulated_signal,
            c.diffs_signal,
            c.perc_diffs,
        ]
        sp_vals = []
        for spectr in spectrals:
            sp_vals.append(spectr.wlens)
            sp_vals.append(spectr.data)
            sp_vals.append(spectr.uncertainties)
        vals = list(zip(*sp_vals, c.dts, c.points, c.mdas))
        vals.sort(key=lambda v: v[-1].mpa_degrees)
        mdas: List[MoonData] = [v[-1] for v in vals]
        mpas = [mda.mpa_degrees for mda in mdas]
        ampa_valid_range = [is_ampa_valid_range(abs(mpa)) for mpa in mpas]
        new_spectrals = []
        index = 0
        for i, spectr in enumerate(spectrals):
            index = i * 3
            wlens = np.array(mpas)  # [v[index].mpa_degrees for v in vals]
            data = np.array([v[index + 1] for v in vals])
            uncertainties = np.array([v[index + 2] for v in vals])
            new_spectrals.append(SpectralData(wlens, data, uncertainties, None))
        mrd = c.mean_relative_difference
        mard = c.mean_absolute_relative_difference
        mpd = c.mean_perc_difference
        std = c.standard_deviation_mrd
        nsamp = c.number_samples
        dts = [v[-3] for v in vals]
        points = [v[-2] for v in vals]
        nc = ComparisonData(
            new_spectrals[0],
            new_spectrals[1],
            new_spectrals[2],
            mrd,
            mard,
            std,
            nsamp,
            dts,
            points,
            ampa_valid_range,
            new_spectrals[3],
            mpd,
            mdas,
        )
        new_comparisons.append(nc)
    return new_comparisons


def average_comparisons(
    wlens: List[float], comps: List[ComparisonData]
) -> ComparisonData:
    """Returns a ComparisonData based on the average of a list of ComparisonData,
    now classified by channel.

    Parameters
    ----------
    comparisons: list of ComparisonData
        List of ComparisonData that will be merged and averaged.

    Returns
    -------
    avg_comparison: ComparisonData
        ComparisonData based on the average data of the given comparisons.
    """
    obs = SpectralData(
        wlens,
        np.array([np.mean(c.observed_signal.data) for c in comps]),
        np.array([np.mean(c.observed_signal.uncertainties) for c in comps]),
        None,
    )
    sim = SpectralData(
        wlens,
        np.array([np.mean(c.simulated_signal.data) for c in comps]),
        np.array([np.mean(c.simulated_signal.uncertainties) for c in comps]),
        None,
    )
    diffs = SpectralData(
        wlens,
        np.array([np.mean(c.diffs_signal.data) for c in comps]),
        np.array([np.mean(c.diffs_signal.uncertainties) for c in comps]),
        None,
    )
    mrd = np.ma.masked_invalid(diffs.data).mean()
    mard = np.ma.masked_invalid(
        [c.mean_absolute_relative_difference for c in comps]
    ).mean()
    stdrd = np.ma.masked_invalid([c.standard_deviation_mrd for c in comps]).mean()
    ns = np.mean([c.number_samples for c in comps])
    ampavr = np.array([np.all(c.ampa_valid_range) for c in comps])
    perc_diffs = SpectralData(
        wlens,
        np.array([np.mean(c.perc_diffs.data) for c in comps]),
        np.array([np.mean(c.perc_diffs.uncertainties) for c in comps]),
        None,
    )
    mpd = np.ma.masked_invalid(perc_diffs.data).mean()
    c = ComparisonData(
        obs,
        sim,
        diffs,
        mrd,
        mard,
        stdrd,
        ns,
        None,
        None,
        ampavr,
        perc_diffs,
        mpd,
        None,
    )
    return c
