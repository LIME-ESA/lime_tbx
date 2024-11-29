"""Tests for comparison.utils module"""

"""___Built-In Modules___"""
from datetime import datetime, timezone
import random
from typing import Tuple

"""___Third-Party Modules___"""
import unittest
import numpy as np

"""___LIME_TBX Modules___"""
from ....datatypes.datatypes import (
    ComparisonData,
    SpectralData,
)
from ..utils import sort_by_mpa, average_comparisons


"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "29/10/2024"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Production"


def get_random_spectral_data_dts(
    n_elems: int = 30,
) -> Tuple[SpectralData, SpectralData]:
    dts = list(
        map(datetime.fromtimestamp, sorted(random.sample(range(1700000000), n_elems)))
    )
    data = [random.random() for _ in range(n_elems)]
    uncerts = [random.random() / 10 for _ in range(n_elems)]
    data2 = [random.random() for _ in range(n_elems)]
    uncerts2 = [random.random() / 10 for _ in range(n_elems)]
    s0 = SpectralData(dts, data, uncerts, None)
    s1 = SpectralData(dts, data2, uncerts2, None)
    return s0, s1


class TestCompUtils(unittest.TestCase):
    # Function sort by mpa
    def test_sort_by_mpa_ok(self):
        n_elems = 3
        comps = []
        mpas = [
            [10, 20, 30],
            [30, 20, 10],
            [120, 100, 110],
            [115, 150, -5],
            [-1, -2, -80],
        ]
        for i in range(len(mpas)):
            sds = get_random_spectral_data_dts(n_elems)
            comp = ComparisonData(
                sds[0],
                sds[1],
                sds[1],
                0,
                0,
                0,
                3,
                sds[0].wlens,
                [None, None, None],
                mpas[i],
                [True, True, True],
                sds[1],
                0,
            )
            comps.append(comp)
        new_comps = sort_by_mpa(comps)
        mpas_indexes = np.argsort(mpas)
        for i in range(len(mpas_indexes)):
            for j in range(len(mpas_indexes[i])):
                id = mpas_indexes[i][j]
                self.assertEqual(
                    comps[i].observed_signal.data[id],
                    new_comps[i].observed_signal.data[j],
                )
                self.assertEqual(
                    comps[i].simulated_signal.data[id],
                    new_comps[i].simulated_signal.data[j],
                )
                self.assertEqual(
                    comps[i].diffs_signal.data[id], new_comps[i].diffs_signal.data[j]
                )

    def test_sort_by_mpa_already_ordered(self):
        n_elems = 3
        comps = []
        mpas = [
            [10, 20, 30],
            [10, 20, 30],
            [100, 110, 120],
            [-5, 115, 150],
            [-80, -2, -1],
        ]
        for i in range(len(mpas)):
            sds = get_random_spectral_data_dts(n_elems)
            comp = ComparisonData(
                sds[0],
                sds[1],
                sds[1],
                0,
                0,
                0,
                3,
                sds[0].wlens,
                [None, None, None],
                mpas[i],
                [True, True, True],
                sds[1],
                0,
            )
            comps.append(comp)
        new_comps = sort_by_mpa(comps)
        mpas_indexes = np.argsort(mpas)
        for i in range(len(mpas_indexes)):
            for j in range(len(mpas_indexes[i])):
                id = mpas_indexes[i][j]
                self.assertEqual(
                    comps[i].observed_signal.data[id],
                    new_comps[i].observed_signal.data[j],
                )
                self.assertEqual(
                    comps[i].simulated_signal.data[id],
                    new_comps[i].simulated_signal.data[j],
                )
                self.assertEqual(
                    comps[i].diffs_signal.data[id], new_comps[i].diffs_signal.data[j]
                )

    def test_sort_by_mpa_empty(self):
        comps = []
        new_comps = sort_by_mpa(comps)
        self.assertEqual(len(new_comps), 0)

    def test_sort_by_mpa_repeated(self):
        n_elems = 3
        comps = []
        mpas = [
            [10, 10, 10],
            [20, 20, 20],
            [110, 110, 110],
            [115, 115, 115],
            [-2, -2, -2],
        ]
        for i in range(len(mpas)):
            sds = get_random_spectral_data_dts(n_elems)
            comp = ComparisonData(
                sds[0],
                sds[1],
                sds[1],
                0,
                0,
                0,
                3,
                sds[0].wlens,
                [None, None, None],
                mpas[i],
                [True, True, True],
                sds[1],
                0,
            )
            comps.append(comp)
        new_comps = sort_by_mpa(comps)
        mpas_indexes = np.argsort(mpas)
        for i in range(len(mpas_indexes)):
            for j in range(len(mpas_indexes[i])):
                id = mpas_indexes[i][j]
                self.assertEqual(
                    comps[i].observed_signal.data[id],
                    new_comps[i].observed_signal.data[j],
                )
                self.assertEqual(
                    comps[i].simulated_signal.data[id],
                    new_comps[i].simulated_signal.data[j],
                )
                self.assertEqual(
                    comps[i].diffs_signal.data[id], new_comps[i].diffs_signal.data[j]
                )

    # Function average comparisons
    def test_average_comparisons_ok(self):
        n_elems = 3
        comps = []
        mpas = [
            [10, 20, 30],
            [30, 20, 10],
            [120, 100, 110],
            [115, 150, -5],
            [-1, -2, -80],
        ]
        wlcs = [440, 600, 750.0, 800.0, 1566]
        for i in range(len(wlcs)):
            sds = get_random_spectral_data_dts(n_elems)
            comp = ComparisonData(
                sds[0],
                sds[1],
                sds[1],
                0,
                0,
                0,
                3,
                sds[0].wlens,
                [None, None, None],
                mpas[i],
                [True, True, True],
                sds[1],
                0,
            )
            comps.append(comp)
        comp = average_comparisons(wlcs, comps)
        omeans = np.array([np.mean(c.observed_signal.data) for c in comps])
        np.testing.assert_array_equal(comp.observed_signal.data, omeans)
        smeans = np.array([np.mean(c.simulated_signal.data) for c in comps])
        np.testing.assert_array_equal(comp.simulated_signal.data, smeans)
        dmeans = np.array([np.mean(c.diffs_signal.data) for c in comps])
        np.testing.assert_array_equal(comp.diffs_signal.data, dmeans)


if __name__ == "__main__":
    unittest.main()
