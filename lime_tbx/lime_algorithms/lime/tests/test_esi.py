"""Tests for the esi module"""

"""___Built-In Modules___"""
# import here

"""___Third-Party Modules___"""
import unittest
import numpy as np

"""___LIME_TBX Modules___"""
from .. import esi

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "01/02/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"

WLENS0 = np.array([350, 370, 390, 410, 430, 450])
WLENS1 = np.array([i for i in range(350, 2501)])
WLENS2 = np.array([i for i in range(0, 1000)])


class TestESI(unittest.TestCase):
    def test_esi_equal(self):
        esi0 = esi.get_esi_per_nms(WLENS0)
        esi1 = esi.get_esi_per_nms(WLENS0)
        np.testing.assert_array_equal(esi0, esi1)

    def test_esi_wehrli_pernms_equal_diff_arrays(self):
        esi0 = esi.get_esi_per_nms(WLENS0)
        esi1 = esi.get_esi_per_nms(WLENS1)
        self.assertEqual(esi0[0], esi1[0])
        self.assertEqual(esi0[1], esi1[20])
        self.assertEqual(esi0[2], esi1[40])
        self.assertEqual(esi0[3], esi1[60])
        self.assertEqual(esi0[4], esi1[80])
        self.assertEqual(esi0[5], esi1[100])

    def test_esi_wehrli_pernms_outrange_ok(self):
        esi0 = esi.get_esi_per_nms(WLENS2)
        self.assertGreater(esi0[0], 0)
        self.assertEqual(esi0[0], esi0[1])

    def test_esi_get_wehrli_data_ok(self):
        wehrli_first_vals = {
            330.5: (1.006, 332.483),
            331.5: (0.9676, 320.7594),
            332.5: (0.9207, 306.13275),
            333.5: (0.9047, 301.71745),
            334.5: (0.9397, 314.32965),
            335.5: (0.9816, 329.3268),
            336.5: (0.7649, 257.38885),
            337.5: (0.8658, 292.2075),
            338.5: (0.9157, 309.96445),
            339.5: (0.9367, 318.00964999999997),
            340.5: (0.9916, 337.63980000000004),
        }
        wehrli = esi._get_wehrli_data()
        for wlen in wehrli_first_vals:
            self.assertEqual(wehrli_first_vals[wlen], wehrli[wlen])

    def test_get_esi_cimel(self):
        irr = esi.get_esi("cimel")[:, 1]
        cimel_irr = np.array(
            [
                1.8622064060781873,
                1.9612487495699018,
                1.5151828550890423,
                0.930943917032395,
                0.7015726794219673,
                0.227755098787054,
                0.08441416554330991,
            ]
        )
        np.testing.assert_array_almost_equal(irr, cimel_irr)

    def test_get_esi_asd(self):
        irr = esi.get_esi("asd")[:, 1]
        asd_ini = np.array(
            [
                0.9675921954269933,
                0.970759294326014,
                0.973078759195136,
                0.975807920888138,
                0.979593018094382,
                0.9812707933408101,
                0.9800621109123312,
                0.980048563361564,
                0.9843300860944322,
                0.9929400675390667,
                1.0061298816924482,
                1.0185612593403701,
                1.0351543393109122,
                1.0537738063761408,
                1.0711350333500396,
                1.0882226560590438,
                1.1053820602548339,
                1.1183101598602414,
                1.1261955176347087,
                1.1318576044677169,
            ]
        )
        asd_end = np.array(
            [
                0.05156687582778551,
                0.051452273600044895,
                0.05136368000595427,
                0.05130439826659359,
                0.051246616035855744,
                0.0511898234360733,
                0.05114608065724496,
                0.05114796789873246,
                0.05115119617112681,
                0.05116010796450618,
                0.05117104539837192,
                0.05117901674451523,
                0.05117841404298195,
                0.051168288151997435,
                0.05115145582102398,
                0.0511143346633173,
                0.05107505412802933,
                0.05103625634213393,
                0.050991633770290215,
                0.05091800369995822,
            ]
        )
        np.testing.assert_array_almost_equal(irr[:20], asd_ini)
        np.testing.assert_array_almost_equal(irr[-20:], asd_end)

    def test_get_esi_int_gauss(self):
        irr = esi.get_esi("interpolated_gaussian")[:, 1]
        intgas_ini = np.array(
            [
                0.9747955034746013,
                0.9917269779946136,
                1.002482579329606,
                1.0223746169832748,
                1.0656974670528094,
                1.0644895940342705,
                1.0018085884149932,
                0.8990935638226655,
                0.8658827458632903,
                0.895511920396254,
                0.9593961944311551,
                0.9804221465324595,
                0.9942404083971317,
                1.0174684953081385,
                1.0705832203122974,
                1.127488519708817,
                1.1925167253627424,
                1.2063412804396731,
                1.2067920130569851,
                1.196229066674803,
            ]
        )
        np.testing.assert_array_almost_equal(irr[:20], intgas_ini)

    def test_get_esi_int_triang(self):
        irr = esi.get_esi("interpolated_triangle")[:, 1]
        intgas_ini = np.array(
            [
                0.9747955034746013,
                0.9917269779946136,
                1.002482579329606,
                1.0223746169832748,
                1.0656974670528094,
                1.0644895940342705,
                1.0018085884149932,
                0.8990935638226655,
                0.8658827458632903,
                0.895511920396254,
                0.9593961944311551,
                0.9804221465324595,
                0.9942404083971317,
                1.0174684953081385,
                1.0705832203122974,
                1.127488519708817,
                1.1925167253627424,
                1.2063412804396731,
                1.2067920130569851,
                1.196229066674803,
            ]
        )
        np.testing.assert_array_almost_equal(irr[:20], intgas_ini)

    def test_get_esi_cimel_wehrli(self):
        irr = esi.get_esi("cimel_wehrli")[:, 1]
        cimel_irr = np.array([1.771, 1.9155, 1.512, 0.97635, 0.7122, 0.23555])
        np.testing.assert_array_almost_equal(irr, cimel_irr)

    def test_get_esi_asd_wehrli(self):
        irr = esi.get_esi("asd_wehrli")[:, 1]
        asd_ini = np.array(
            [
                9.919000e-01,
                1.055800e00,
                9.317000e-01,
                9.929000e-01,
                1.124000e00,
                1.095500e00,
                9.978500e-01,
                9.142500e-01,
                7.589500e-01,
                8.815500e-01,
                1.057300e00,
                9.361500e-01,
                1.034350e00,
                1.066350e00,
                9.863500e-01,
                1.139000e00,
                1.256000e00,
                1.231500e00,
                1.151000e00,
                1.209500e00,
            ]
        )
        np.testing.assert_array_almost_equal(irr[:20], asd_ini)

    def test_get_esi_unc_wehrlis(self):
        uncs = esi.get_u_esi("asd_wehrli")[:, 1]
        self.assertFalse(uncs.any())
        uncs = esi.get_u_esi("cimel_wehrli")[:, 1]
        self.assertFalse(uncs.any())

    def test_get_u_esi_cimel(self):
        irr = esi.get_u_esi("cimel")[:, 1]
        cimel_irr = np.array(
            [
                0.00047544757316708713,
                0.00028073022897993964,
                0.00021615839045347938,
                0.00013119794464804796,
                0.0001030192076809563,
                2.3519874638426612e-05,
                5.292995762187002e-06,
            ]
        )
        np.testing.assert_array_almost_equal(irr, cimel_irr)

    def test_get_u_esi_asd(self):
        irr = esi.get_u_esi("asd")[:, 1]
        asd_ini = np.array(
            [
                0.013035703319541353,
                0.013078371462092368,
                0.013109620017721643,
                0.013146388136802849,
                0.013197382146550066,
                0.013219985657636645,
                0.013203701910039903,
                0.01320351939036405,
                0.013261201395454593,
                0.013377197743982718,
                0.013554895021663432,
                0.013722374407327268,
                0.013945921540992768,
                0.014196768789676973,
                0.01443066463648309,
                0.014660874402927704,
                0.014892051234923698,
                0.015066222616840586,
                0.01517245661111303,
                0.01524873800659774,
            ]
        )
        asd_end = np.array(
            [
                0.0006947250055847777,
                0.0006931810487061084,
                0.000691987488349995,
                0.0006911888264599582,
                0.0006904103664445666,
                0.0006896452389691069,
                0.00068905592266741,
                0.0006890813482016078,
                0.000689124840491214,
                0.0006892449029941513,
                0.0006893922555062029,
                0.0006894996479767067,
                0.0006894915282384113,
                0.0006893551090744026,
                0.0006891283385783507,
                0.000688628230656488,
                0.000688099030889139,
                0.0006875763347902382,
                0.000686975165392589,
                0.0006859831979781749,
            ]
        )
        np.testing.assert_array_almost_equal(irr[:20], asd_ini)
        np.testing.assert_array_almost_equal(irr[-20:], asd_end)

    def test_get_u_esi_int_gauss(self):
        irr = esi.get_u_esi("interpolated_gaussian")[:, 1]
        intgas_ini = np.array(
            [
                0.011265766752541512,
                0.01146144477795866,
                0.011585747896549647,
                0.01181564133077349,
                0.012316325977491473,
                0.012302366520638503,
                0.011577958564592204,
                0.010390875184307428,
                0.01000705586294395,
                0.01034948190711767,
                0.011087796078246743,
                0.011330794206523845,
                0.011490492644821116,
                0.011758940950903272,
                0.012372790813537178,
                0.013030448579399072,
                0.013781983244515513,
                0.013941754410825612,
                0.013946963528223947,
                0.013824886929890976,
            ]
        )
        np.testing.assert_array_almost_equal(irr[:20], intgas_ini)

    def test_get_u_esi_int_triang(self):
        irr = esi.get_u_esi("interpolated_triangle")[:, 1]
        intgas_ini = np.array(
            [
                0.011265766752541512,
                0.01146144477795866,
                0.011585747896549647,
                0.01181564133077349,
                0.012316325977491473,
                0.012302366520638503,
                0.011577958564592204,
                0.010390875184307428,
                0.01000705586294395,
                0.01034948190711767,
                0.011087796078246743,
                0.011330794206523845,
                0.011490492644821116,
                0.011758940950903272,
                0.012372790813537178,
                0.013030448579399072,
                0.013781983244515513,
                0.013941754410825612,
                0.013946963528223947,
                0.013824886929890976,
            ]
        )
        np.testing.assert_array_almost_equal(irr[:20], intgas_ini)


if __name__ == "__main__":
    unittest.main()
