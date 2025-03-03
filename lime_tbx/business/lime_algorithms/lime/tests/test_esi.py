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
                1.9603369500934011,
                1.5155354495830629,
                0.930943917032395,
                0.7015726794219683,
                0.227755098787054,
                0.09141864990787517,
            ]
        )
        np.testing.assert_array_almost_equal(irr, cimel_irr)

    def test_get_esi_asd(self):
        irr = esi.get_esi("asd")[:, 1]
        asd_ini = np.array(
            [
                0.9687130138405707,
                0.9735964428359197,
                0.9775471891626081,
                0.980178145070853,
                0.9814245744819013,
                0.9816369192880028,
                0.9815836915701008,
                0.9823493284416859,
                0.9851395118278135,
                0.9910338891585021,
                1.000743324239325,
                1.0144288315930214,
                1.0316224502140092,
                1.0512640988682194,
                1.0718435459826474,
                1.091620498419502,
                1.108889073978433,
                1.1222505790366053,
                1.1308550573488139,
                1.134566580104952,
            ]
        )
        asd_end = np.array(
            [
                0.05157908377205688,
                0.051455659919497874,
                0.051354177062063536,
                0.05127599530110452,
                0.05122072791048691,
                0.05118633848845873,
                0.05116942471962701,
                0.051165634984576575,
                0.05117015193937136,
                0.051178175975738066,
                0.051185348576420625,
                0.05118806906298312,
                0.051183676753483336,
                0.05117049231452313,
                0.05114773404750597,
                0.05111534293695586,
                0.05107376077020051,
                0.0510237066171394,
                0.05096598917088341,
                0.050901378864909995,
            ]
        )
        np.testing.assert_array_almost_equal(irr[:20], asd_ini)
        np.testing.assert_array_almost_equal(irr[-20:], asd_end)

    def test_get_esi_int_gauss(self):
        irr = esi.get_esi("interpolated_gaussian")[:, 1]
        intgas_ini = np.array(
            [
                0.9743046913294741,
                0.9949424162818928,
                0.998847282759393,
                1.0216129712075723,
                1.060168234545524,
                1.0603733102139543,
                0.997380953289867,
                0.9096858009110536,
                0.8668283967710058,
                0.9020198694855773,
                0.9590003414726922,
                0.9801079257818487,
                0.9933879247362412,
                1.0245530560212333,
                1.0672139516860628,
                1.1267913258239437,
                1.1868535285438226,
                1.208613351580522,
                1.2002829954812528,
                1.1987332095943033,
            ]
        )
        np.testing.assert_array_almost_equal(irr[:20], intgas_ini)

    def test_get_esi_int_triang(self):
        irr = esi.get_esi("interpolated_triangle")[:, 1]
        intgas_ini = np.array(
            [
                1.0008732360178756,
                1.0380626965048247,
                0.9724665283074009,
                0.9582533204930623,
                1.1223692815452744,
                1.1292608031472011,
                1.0388147028083163,
                0.8510255159056774,
                0.7770369559616941,
                0.8329042157794957,
                1.1305504537218685,
                0.9181864540934498,
                0.9370409244980262,
                1.0681510469259312,
                1.0633676024094831,
                1.0467207130098573,
                1.2949183989680058,
                1.2369100888709692,
                1.1564467934567226,
                1.1730777700020405,
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
                0.0004802881850563571,
                0.00027742777629843624,
                0.00020691407372330675,
                0.00012733282727139363,
                0.00010319296443309817,
                2.1941713306337837e-05,
                5.768660295644467e-06,
            ]
        )
        np.testing.assert_array_almost_equal(irr, cimel_irr)

    def test_get_u_esi_asd(self):
        irr = esi.get_u_esi("asd")[:, 1]
        asd_ini = np.array(
            [
                0.0004859056003244296,
                0.000493964798940372,
                0.0005028173724653659,
                0.0005105921111319053,
                0.0005160638516936109,
                0.0005190806076340028,
                0.0005205331411978351,
                0.0005218751809784779,
                0.0005244162260497669,
                0.000528801592833928,
                0.0005350132241602241,
                0.000542817934743524,
                0.0005522363189830605,
                0.0005636592926497497,
                0.0005775610477440668,
                0.0005939811650769961,
                0.0006119828148914025,
                0.0006293238670075232,
                0.0006426470930261391,
                0.0006483780611816747,
            ]
        )
        asd_end = np.array(
            [
                2.5098405514711267e-05,
                2.4952850029271083e-05,
                2.4900819159821008e-05,
                2.493034200297332e-05,
                2.499588419071003e-05,
                2.503458528896218e-05,
                2.498758728538586e-05,
                2.4818750665887386e-05,
                2.452577925605859e-05,
                2.4141643563484457e-05,
                2.3726065092091943e-05,
                2.3348770466337896e-05,
                2.306928486844169e-05,
                2.2920962189790087e-05,
                2.2906240825844483e-05,
                2.3004026024269503e-05,
                2.318264602642898e-05,
                2.3409539022535163e-05,
                2.365333428415256e-05,
                2.3880812800224697e-05,
            ]
        )
        np.testing.assert_array_almost_equal(irr[:20], asd_ini)
        np.testing.assert_array_almost_equal(irr[-20:], asd_end)

    def test_get_u_esi_int_gauss(self):
        irr = esi.get_u_esi("interpolated_gaussian")[:, 1]
        intgas_ini = np.array(
            [
                0.0010202560849398542,
                0.0010379718142929518,
                0.0009702329163378632,
                0.0009757490962219266,
                0.0011235582366421668,
                0.001120854564699808,
                0.0009526954388076576,
                0.0008540340617372452,
                0.0008444244407943919,
                0.0010039776719297025,
                0.0011112018371800924,
                0.0009671424074240135,
                0.0009082795047851564,
                0.001009175381559155,
                0.0010599682130411137,
                0.0011155725936095476,
                0.0011628480872142037,
                0.0010910072806692643,
                0.0011169118944969548,
                0.0012036475598076455,
            ]
        )
        np.testing.assert_array_almost_equal(irr[:20], intgas_ini)

    def test_get_u_esi_int_triang(self):
        irr = esi.get_u_esi("interpolated_triangle")[:, 1]
        intgas_ini = np.array(
            [
                0.0017338982059582886,
                0.0016622193874714548,
                0.0017575657759426168,
                0.001477499492071497,
                0.0019290803464244212,
                0.0019655872488971494,
                0.0018056045359769208,
                0.0013859904160320965,
                0.001452373282944528,
                0.00140263364115238,
                0.0018044410661434236,
                0.0013598263861674964,
                0.0014340060569939972,
                0.0018778187535192015,
                0.0017472076187685507,
                0.00170962047506041,
                0.00254558725442235,
                0.00207565687229181,
                0.0019778091301057977,
                0.002024787477165641,
            ]
        )
        np.testing.assert_array_almost_equal(irr[:20], intgas_ini)


if __name__ == "__main__":
    unittest.main()
