"""Tests for srf module"""

"""___Built-In Modules___"""
# import here

"""___Third-Party Modules___"""
import unittest

"""___LIME_TBX Modules___"""
from ..srf import read_srf

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "30/08/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


class TestSRF(unittest.TestCase):
    def test_read_srf(self):
        srf = read_srf(
            "./test_files/srf/W_XX-EUMETSAT-Darmstadt_VIS+IR+SRF_MSG3+SEVIRI_C_EUMG.nc"
        )
        self.assertEqual(
            srf.name, "W_XX-EUMETSAT-Darmstadt_VIS+IR+SRF_MSG3+SEVIRI_C_EUMG.nc"
        )
        ch_names = [
            "VIS006",
            "HRVIS",
            "VIS008",
            "NIR016",
            "IR039",
            "IR062",
            "IR073",
            "IR087",
            "IR097",
            "IR108",
            "IR120",
            "IR134",
        ]
        for i, name in enumerate(srf.get_channels_names()):
            self.assertEqual(name, ch_names[i])
        for i, wlen in enumerate(srf.get_wavelengths()):
            self.assertEqual(wlen, WLENS[i])
        srf_vis006_sr = srf.get_channel_from_name("VIS006").spectral_response
        for key in srf_vis006_sr:
            self.assertEqual(srf_vis006_sr[key], VIS006_SR[key])


if __name__ == "__main__":
    unittest.main()

VIS006_SR = {
    485.0: 2.6522e-06,
    488.0: 5.95293e-06,
    491.0: 1.47129e-05,
    494.0: 1.00483e-05,
    497.0: 6.19717e-06,
    500.0: 3.67207e-06,
    503.0: 1.45878e-05,
    506.0: 1.66717e-05,
    509.0: 7.55848e-06,
    512.0: 2.94164e-06,
    515.0: 4.65267e-06,
    518.0: 1.38388e-05,
    521.0: 1.32082e-05,
    524.0: 8.11409e-06,
    527.0: 6.42834e-06,
    530.0: 3.14513e-06,
    533.0: 1.07178e-05,
    536.0: 1.25477e-05,
    539.0: 5.47142e-06,
    542.0: 2.44176e-06,
    545.0: 3.28877e-06,
    548.0: 7.52075e-06,
    551.0: 2.98575e-05,
    554.0: 1.96976e-05,
    557.0: 8.68243e-06,
    560.0: 2.58383e-05,
    563.0: 6.92602e-05,
    566.0: 0.000105476,
    569.0: 0.000322538,
    572.0: 0.000447837,
    575.0: 0.000816178,
    578.0: 0.001215542,
    581.0: 0.002351759,
    584.0: 0.002811187,
    587.0: 0.008036015,
    590.0: 0.020341649,
    593.0: 0.054117202,
    596.0: 0.161864241,
    599.0: 0.445153154,
    602.0: 0.815600683,
    605.0: 0.926814725,
    608.0: 0.882559869,
    611.0: 0.859840338,
    614.0: 0.865338218,
    617.0: 0.870957029,
    620.0: 0.87383779,
    623.0: 0.880714697,
    626.0: 0.878150901,
    629.0: 0.864465615,
    632.0: 0.865133245,
    635.0: 0.893400936,
    638.0: 0.93779448,
    641.0: 0.965932408,
    644.0: 0.965219979,
    647.0: 0.943432267,
    650.0: 0.912372258,
    653.0: 0.879205506,
    656.0: 0.859924209,
    659.0: 0.868651928,
    662.0: 0.907969032,
    665.0: 0.967159154,
    668.0: 1.0,
    671.0: 0.908268012,
    674.0: 0.636302823,
    677.0: 0.348118716,
    680.0: 0.174211538,
    683.0: 0.08728506,
    686.0: 0.046083283,
    689.0: 0.027590773,
    692.0: 0.015661789,
    695.0: 0.008128369,
    698.0: 0.007100068,
    701.0: 0.005108841,
    704.0: 0.001155187,
    707.0: 0.000247302,
    710.0: 0.0002081,
    713.0: 0.000281591,
    716.0: 9.88057e-05,
    719.0: 0.00011737,
    722.0: 4.80924e-05,
    725.0: 2.61375e-05,
    728.0: 2.28296e-05,
    731.0: 3.40499e-05,
    734.0: 5.41527e-05,
    737.0: 5.38275e-05,
    740.0: 1.71645e-05,
    743.0: 2.0145e-05,
    746.0: 2.56509e-05,
    749.0: 8.66205e-05,
    752.0: 6.02616e-05,
    755.0: 1.76063e-05,
    758.0: 2.45098e-05,
    761.0: 0.000119611,
    764.0: 0.000156306,
    767.0: 0.000130439,
    770.0: 5.32675e-05,
    773.0: 1.64151e-05,
    776.0: 9.18038e-05,
    779.0: 0.000121569,
    782.0: 0.000142643,
    785.0: 7.95033e-05,
}

WLENS = [
    485.0,
    488.0,
    491.0,
    494.0,
    497.0,
    500.0,
    503.0,
    506.0,
    509.0,
    512.0,
    515.0,
    518.0,
    521.0,
    524.0,
    527.0,
    530.0,
    533.0,
    536.0,
    539.0,
    542.0,
    545.0,
    548.0,
    551.0,
    554.0,
    557.0,
    560.0,
    563.0,
    566.0,
    569.0,
    572.0,
    575.0,
    578.0,
    581.0,
    584.0,
    587.0,
    590.0,
    593.0,
    596.0,
    599.0,
    602.0,
    605.0,
    608.0,
    611.0,
    614.0,
    617.0,
    620.0,
    623.0,
    626.0,
    629.0,
    632.0,
    635.0,
    638.0,
    641.0,
    644.0,
    647.0,
    650.0,
    653.0,
    656.0,
    659.0,
    662.0,
    665.0,
    668.0,
    671.0,
    674.0,
    677.0,
    680.0,
    683.0,
    686.0,
    689.0,
    692.0,
    695.0,
    698.0,
    701.0,
    704.0,
    707.0,
    710.0,
    713.0,
    716.0,
    719.0,
    722.0,
    725.0,
    728.0,
    731.0,
    734.0,
    737.0,
    740.0,
    743.0,
    746.0,
    749.0,
    752.0,
    755.0,
    758.0,
    761.0,
    764.0,
    767.0,
    770.0,
    773.0,
    776.0,
    779.0,
    782.0,
    785.0,
    300.0,
    306.0,
    312.0,
    318.0,
    324.0,
    330.0,
    336.0,
    342.0,
    348.0,
    354.0,
    360.0,
    366.0,
    372.0,
    378.0,
    384.0,
    390.0,
    396.0,
    402.0,
    408.0,
    414.0,
    420.0,
    426.0,
    432.0,
    438.0,
    444.0,
    450.0,
    456.0,
    462.0,
    468.0,
    474.0,
    480.0,
    486.0,
    492.0,
    498.0,
    504.0,
    510.0,
    516.0,
    522.0,
    528.0,
    534.0,
    540.0,
    546.0,
    552.0,
    558.0,
    564.0,
    570.0,
    576.0,
    582.0,
    588.0,
    594.0,
    600.0,
    606.0,
    612.0,
    618.0,
    624.0,
    630.0,
    636.0,
    642.0,
    648.0,
    654.0,
    660.0,
    666.0,
    672.0,
    678.0,
    684.0,
    690.0,
    696.0,
    702.0,
    708.0,
    714.0,
    720.0,
    726.0,
    732.0,
    738.0,
    744.0,
    750.0,
    756.0,
    762.0,
    768.0,
    774.0,
    780.0,
    786.0,
    792.0,
    798.0,
    804.0,
    810.0,
    816.0,
    822.0,
    828.0,
    834.0,
    840.0,
    846.0,
    852.0,
    858.0,
    864.0,
    870.0,
    876.0,
    882.0,
    888.0,
    894.0,
    900.0,
    906.0,
    912.0,
    918.0,
    924.0,
    930.0,
    936.0,
    942.0,
    948.0,
    954.0,
    960.0,
    966.0,
    972.0,
    978.0,
    984.0,
    990.0,
    996.0,
    1002.0,
    1008.0,
    1014.0,
    1020.0,
    1026.0,
    1032.0,
    1038.0,
    1044.0,
    1050.0,
    1056.0,
    1062.0,
    1068.0,
    1074.0,
    1080.0,
    1086.0,
    1092.0,
    1098.0,
    1104.0,
    1110.0,
    1116.0,
    1122.0,
    1128.0,
    1134.0,
    1140.0,
    1146.0,
    1152.0,
    1158.0,
    1164.0,
    1170.0,
    1176.0,
    1182.0,
    1188.0,
    1194.0,
    1200.0,
    1206.0,
    1212.0,
    1218.0,
    1224.0,
    1230.0,
    1236.0,
    1242.0,
    1248.0,
    1254.0,
    1260.0,
    1266.0,
    1272.0,
    1278.0,
    1284.0,
    1290.0,
    1296.0,
    1302.0,
    670.0,
    672.8,
    675.6,
    678.4,
    681.2,
    684.0,
    686.8,
    689.6,
    692.4,
    695.2,
    698.0,
    700.8,
    703.6,
    706.4,
    709.2,
    712.0,
    714.8,
    717.6,
    720.4000000000001,
    723.1999999999999,
    726.0,
    728.8,
    731.6,
    734.4000000000001,
    737.1999999999999,
    740.0,
    742.8000000000001,
    745.6,
    748.4,
    751.1999999999999,
    754.0,
    756.8000000000001,
    759.6,
    762.4,
    765.2,
    768.0,
    770.8000000000001,
    773.5999999999999,
    776.4,
    779.2,
    782.0,
    784.8000000000001,
    787.6,
    790.4,
    793.2,
    796.0,
    798.8,
    801.6,
    804.4,
    807.2,
    810.0,
    812.8,
    815.6,
    818.4,
    821.2,
    824.0,
    826.8,
    829.6,
    832.4,
    835.2,
    838.0,
    840.8,
    843.6,
    846.4000000000001,
    849.1999999999999,
    852.0,
    854.8,
    857.6,
    860.4000000000001,
    863.1999999999999,
    866.0,
    868.8000000000001,
    871.6,
    874.4,
    877.1999999999999,
    880.0,
    882.8000000000001,
    885.6,
    888.4,
    891.2,
    894.0,
    896.8000000000001,
    899.5999999999999,
    902.4,
    905.2,
    908.0,
    910.8000000000001,
    913.6,
    916.4,
    919.2,
    922.0,
    924.8,
    927.6,
    930.4,
    933.2,
    936.0,
    938.8,
    941.6,
    944.4,
    947.2,
    950.0,
    1360.0,
    1365.6,
    1371.2,
    1376.8,
    1382.4,
    1388.0,
    1393.6,
    1399.2,
    1404.8,
    1410.4,
    1416.0,
    1421.6,
    1427.2,
    1432.8000000000002,
    1438.3999999999999,
    1444.0,
    1449.6,
    1455.2,
    1460.8000000000002,
    1466.3999999999999,
    1472.0,
    1477.6000000000001,
    1483.2,
    1488.8,
    1494.3999999999999,
    1500.0,
    1505.6000000000001,
    1511.2,
    1516.8,
    1522.3999999999999,
    1528.0,
    1533.6000000000001,
    1539.1999999999998,
    1544.8,
    1550.4,
    1556.0,
    1561.6000000000001,
    1567.1999999999998,
    1572.8,
    1578.4,
    1584.0,
    1589.6,
    1595.2,
    1600.8,
    1606.4,
    1612.0,
    1617.6,
    1623.2,
    1628.8,
    1634.4,
    1640.0,
    1645.6,
    1651.2,
    1656.8,
    1662.4,
    1668.0,
    1673.6,
    1679.2,
    1684.8000000000002,
    1690.3999999999999,
    1696.0,
    1701.6,
    1707.2,
    1712.8000000000002,
    1718.3999999999999,
    1724.0,
    1729.6000000000001,
    1735.2,
    1740.8,
    1746.3999999999999,
    1752.0,
    1757.6000000000001,
    1763.2,
    1768.8,
    1774.4,
    1780.0,
    1785.6000000000001,
    1791.1999999999998,
    1796.8,
    1802.4,
    1808.0,
    1813.6000000000001,
    1819.1999999999998,
    1824.8,
    1830.4,
    1836.0,
    1841.6,
    1847.2,
    1852.8,
    1858.4,
    1864.0,
    1869.6,
    1875.2,
    1880.8,
    1886.4,
    1892.0,
    1897.6,
    1903.2,
    1908.8,
    1914.4,
    1920.0,
    3040.0,
    3057.6,
    3075.2000000000003,
    3092.8,
    3110.3999999999996,
    3128.0,
    3145.6,
    3163.2,
    3180.8,
    3198.4,
    3216.0,
    3233.6,
    3251.2,
    3268.8,
    3286.4,
    3304.0,
    3321.6,
    3339.2,
    3356.7999999999997,
    3374.4,
    3392.0,
    3409.6000000000004,
    3427.2,
    3444.7999999999997,
    3462.4,
    3480.0,
    3497.6,
    3515.2000000000003,
    3532.7999999999997,
    3550.3999999999996,
    3568.0,
    3585.6,
    3603.2000000000003,
    3620.8,
    3638.3999999999996,
    3656.0,
    3673.6,
    3691.2,
    3708.8,
    3726.4,
    3744.0,
    3761.6,
    3779.2,
    3796.8,
    3814.4,
    3832.0,
    3849.6,
    3867.2,
    3884.7999999999997,
    3902.4,
    3920.0,
    3937.6000000000004,
    3955.2000000000003,
    3972.7999999999997,
    3990.4,
    4008.0,
    4025.6,
    4043.2,
    4060.8,
    4078.4,
    4096.0,
    4113.6,
    4131.2,
    4148.799999999999,
    4166.400000000001,
    4184.0,
    4201.6,
    4219.2,
    4236.799999999999,
    4254.400000000001,
    4272.0,
    4289.6,
    4307.2,
    4324.8,
    4342.4,
    4360.0,
    4377.6,
    4395.2,
    4412.8,
    4430.4,
    4448.0,
    4465.6,
    4483.2,
    4500.8,
    4518.4,
    4536.0,
    4553.6,
    4571.2,
    4588.8,
    4606.4,
    4624.0,
    4641.6,
    4659.2,
    4676.8,
    4694.4,
    4712.0,
    4729.599999999999,
    4747.200000000001,
    4764.8,
    4782.4,
    4800.0,
    4450.0,
    4486.0,
    4522.0,
    4558.0,
    4594.0,
    4630.0,
    4666.0,
    4702.0,
    4738.0,
    4774.0,
    4810.0,
    4846.0,
    4882.0,
    4918.0,
    4954.0,
    4990.0,
    5026.0,
    5062.0,
    5098.0,
    5134.0,
    5170.0,
    5206.0,
    5242.0,
    5278.0,
    5314.0,
    5350.0,
    5386.0,
    5422.0,
    5458.0,
    5494.0,
    5530.0,
    5566.0,
    5602.0,
    5638.0,
    5674.0,
    5710.0,
    5746.0,
    5782.0,
    5818.0,
    5854.0,
    5890.0,
    5926.0,
    5962.0,
    5998.0,
    6034.0,
    6070.0,
    6106.0,
    6142.0,
    6178.0,
    6214.0,
    6250.0,
    6286.0,
    6322.0,
    6358.0,
    6394.0,
    6430.0,
    6466.0,
    6502.0,
    6538.0,
    6574.0,
    6610.0,
    6646.0,
    6682.0,
    6718.0,
    6754.0,
    6790.0,
    6826.0,
    6862.0,
    6898.0,
    6934.0,
    6970.0,
    7006.0,
    7042.0,
    7078.0,
    7114.0,
    7150.0,
    7186.0,
    7222.0,
    7258.0,
    7294.0,
    7330.0,
    7366.0,
    7402.0,
    7438.0,
    7474.0,
    7510.0,
    7546.0,
    7582.0,
    7618.0,
    7654.0,
    7690.0,
    7726.0,
    7762.0,
    7798.0,
    7834.0,
    7870.0,
    7906.0,
    7942.0,
    7978.0,
    8013.999999999999,
    8050.000000000001,
    6350.0,
    6370.0,
    6390.0,
    6410.0,
    6430.0,
    6450.0,
    6470.0,
    6490.0,
    6510.0,
    6530.0,
    6550.0,
    6570.0,
    6590.0,
    6610.0,
    6630.0,
    6650.0,
    6670.0,
    6690.0,
    6710.0,
    6730.0,
    6750.0,
    6770.0,
    6790.0,
    6810.0,
    6830.0,
    6850.0,
    6870.0,
    6890.0,
    6910.0,
    6930.0,
    6950.0,
    6970.0,
    6990.0,
    7010.0,
    7030.0,
    7050.0,
    7070.0,
    7090.0,
    7110.0,
    7130.0,
    7150.0,
    7170.0,
    7190.0,
    7210.0,
    7230.0,
    7250.0,
    7270.0,
    7290.0,
    7310.0,
    7330.0,
    7350.0,
    7370.0,
    7390.0,
    7410.0,
    7430.0,
    7450.0,
    7470.0,
    7490.0,
    7510.0,
    7530.0,
    7550.0,
    7570.0,
    7590.0,
    7610.0,
    7630.0,
    7650.0,
    7670.0,
    7690.0,
    7710.0,
    7730.0,
    7750.0,
    7770.0,
    7790.0,
    7810.0,
    7830.0,
    7850.0,
    7870.0,
    7890.0,
    7910.0,
    7930.0,
    7950.0,
    7970.0,
    7990.0,
    8010.0,
    8029.999999999999,
    8050.000000000001,
    8070.0,
    8090.0,
    8109.999999999999,
    8130.000000000001,
    8150.0,
    8170.0,
    8189.999999999999,
    8210.0,
    8230.0,
    8250.0,
    8270.0,
    8290.0,
    8310.0,
    8330.0,
    8350.0,
    7900.0,
    7916.0,
    7932.0,
    7948.0,
    7964.0,
    7980.0,
    7996.0,
    8012.0,
    8028.000000000001,
    8044.000000000001,
    8060.000000000001,
    8076.000000000001,
    8092.000000000001,
    8108.000000000001,
    8124.000000000001,
    8140.000000000001,
    8156.000000000001,
    8172.000000000001,
    8188.000000000001,
    8204.0,
    8220.0,
    8236.0,
    8252.0,
    8268.0,
    8284.0,
    8300.0,
    8316.0,
    8332.0,
    8348.0,
    8364.0,
    8380.0,
    8396.0,
    8412.0,
    8428.0,
    8444.0,
    8460.0,
    8476.0,
    8492.0,
    8508.0,
    8524.0,
    8540.0,
    8556.0,
    8572.0,
    8588.0,
    8604.0,
    8620.0,
    8636.0,
    8652.0,
    8668.0,
    8684.0,
    8700.0,
    8716.0,
    8732.0,
    8748.0,
    8764.0,
    8780.0,
    8796.0,
    8812.0,
    8828.0,
    8844.0,
    8860.0,
    8876.0,
    8892.0,
    8908.0,
    8924.0,
    8940.0,
    8956.0,
    8972.0,
    8988.0,
    9004.0,
    9020.0,
    9036.0,
    9052.0,
    9068.0,
    9084.0,
    9100.0,
    9116.0,
    9132.0,
    9148.0,
    9164.0,
    9180.0,
    9196.0,
    9212.0,
    9228.0,
    9244.0,
    9260.0,
    9276.0,
    9292.0,
    9308.0,
    9324.0,
    9340.0,
    9356.0,
    9372.0,
    9388.0,
    9404.0,
    9420.0,
    9436.0,
    9452.0,
    9468.0,
    9484.0,
    9500.0,
    9100.0,
    9111.2,
    9122.400000000001,
    9133.6,
    9144.8,
    9156.0,
    9167.199999999999,
    9178.4,
    9189.6,
    9200.8,
    9212.0,
    9223.2,
    9234.400000000001,
    9245.6,
    9256.8,
    9268.0,
    9279.199999999999,
    9290.4,
    9301.6,
    9312.8,
    9324.0,
    9335.2,
    9346.4,
    9357.6,
    9368.800000000001,
    9380.0,
    9391.199999999999,
    9402.4,
    9413.6,
    9424.8,
    9436.0,
    9447.2,
    9458.4,
    9469.6,
    9480.800000000001,
    9492.0,
    9503.199999999999,
    9514.4,
    9525.6,
    9536.8,
    9548.0,
    9559.2,
    9570.4,
    9581.6,
    9592.800000000001,
    9604.0,
    9615.199999999999,
    9626.4,
    9637.6,
    9648.8,
    9660.0,
    9671.2,
    9682.4,
    9693.6,
    9704.800000000001,
    9716.0,
    9727.2,
    9738.4,
    9749.599999999999,
    9760.8,
    9772.0,
    9783.2,
    9794.4,
    9805.6,
    9816.800000000001,
    9828.0,
    9839.2,
    9850.4,
    9861.599999999999,
    9872.8,
    9884.0,
    9895.2,
    9906.4,
    9917.6,
    9928.800000000001,
    9940.0,
    9951.2,
    9962.400000000001,
    9973.599999999999,
    9984.8,
    9996.0,
    10007.199999999999,
    10018.4,
    10029.6,
    10040.800000000001,
    10052.0,
    10063.2,
    10074.400000000001,
    10085.599999999999,
    10096.8,
    10108.0,
    10119.199999999999,
    10130.4,
    10141.6,
    10152.8,
    10164.0,
    10175.2,
    10186.400000000001,
    10197.6,
    10208.8,
    10220.0,
    8800.0,
    8840.0,
    8880.0,
    8920.0,
    8960.0,
    9000.0,
    9040.0,
    9080.0,
    9120.0,
    9160.0,
    9200.0,
    9240.0,
    9280.0,
    9320.0,
    9360.0,
    9400.0,
    9440.0,
    9480.0,
    9520.0,
    9560.0,
    9600.0,
    9640.0,
    9680.0,
    9720.0,
    9760.0,
    9800.0,
    9840.0,
    9880.0,
    9920.0,
    9960.0,
    10000.0,
    10040.0,
    10080.0,
    10120.0,
    10160.0,
    10200.0,
    10240.0,
    10280.0,
    10320.0,
    10360.0,
    10400.0,
    10440.0,
    10480.0,
    10520.0,
    10560.0,
    10600.0,
    10640.0,
    10680.0,
    10720.0,
    10760.0,
    10800.0,
    10840.0,
    10880.0,
    10920.0,
    10960.0,
    11000.0,
    11040.0,
    11080.0,
    11120.0,
    11160.0,
    11200.0,
    11240.0,
    11280.0,
    11320.0,
    11360.0,
    11400.0,
    11440.0,
    11480.0,
    11520.0,
    11560.0,
    11600.0,
    11640.0,
    11680.0,
    11720.0,
    11760.0,
    11800.0,
    11840.0,
    11880.0,
    11920.0,
    11960.0,
    12000.0,
    12040.0,
    12080.0,
    12120.0,
    12160.0,
    12200.0,
    12240.0,
    12280.0,
    12320.0,
    12360.0,
    12400.0,
    12440.0,
    12480.0,
    12520.0,
    12560.0,
    12600.0,
    12640.0,
    12680.0,
    12720.0,
    12760.0,
    12800.0,
    10000.0,
    10040.0,
    10080.0,
    10120.0,
    10160.0,
    10200.0,
    10240.0,
    10280.0,
    10320.0,
    10360.0,
    10400.0,
    10440.0,
    10480.0,
    10520.0,
    10560.0,
    10600.0,
    10640.0,
    10680.0,
    10720.0,
    10760.0,
    10800.0,
    10840.0,
    10880.0,
    10920.0,
    10960.0,
    11000.0,
    11040.0,
    11080.0,
    11120.0,
    11160.0,
    11200.0,
    11240.0,
    11280.0,
    11320.0,
    11360.0,
    11400.0,
    11440.0,
    11480.0,
    11520.0,
    11560.0,
    11600.0,
    11640.0,
    11680.0,
    11720.0,
    11760.0,
    11800.0,
    11840.0,
    11880.0,
    11920.0,
    11960.0,
    12000.0,
    12040.0,
    12080.0,
    12120.0,
    12160.0,
    12200.0,
    12240.0,
    12280.0,
    12320.0,
    12360.0,
    12400.0,
    12440.0,
    12480.0,
    12520.0,
    12560.0,
    12600.0,
    12640.0,
    12680.0,
    12720.0,
    12760.0,
    12800.0,
    12840.0,
    12880.0,
    12920.0,
    12960.0,
    13000.0,
    13040.0,
    13080.0,
    13120.0,
    13160.0,
    13200.0,
    13240.0,
    13280.0,
    13320.0,
    13360.0,
    13400.0,
    13440.0,
    13480.0,
    13520.0,
    13560.0,
    13600.0,
    13640.0,
    13680.0,
    13720.0,
    13760.0,
    13800.0,
    13840.0,
    13880.0,
    13920.0,
    13960.0,
    14000.0,
    11400.0,
    11440.0,
    11480.0,
    11520.0,
    11560.0,
    11600.0,
    11640.0,
    11680.0,
    11720.0,
    11760.0,
    11800.0,
    11840.0,
    11880.0,
    11920.0,
    11960.0,
    12000.0,
    12040.0,
    12080.0,
    12120.0,
    12160.0,
    12200.0,
    12240.0,
    12280.0,
    12320.0,
    12360.0,
    12400.0,
    12440.0,
    12480.0,
    12520.0,
    12560.0,
    12600.0,
    12640.0,
    12680.0,
    12720.0,
    12760.0,
    12800.0,
    12840.0,
    12880.0,
    12920.0,
    12960.0,
    13000.0,
    13040.0,
    13080.0,
    13120.0,
    13160.0,
    13200.0,
    13240.0,
    13280.0,
    13320.0,
    13360.0,
    13400.0,
    13440.0,
    13480.0,
    13520.0,
    13560.0,
    13600.0,
    13640.0,
    13680.0,
    13720.0,
    13760.0,
    13800.0,
    13840.0,
    13880.0,
    13920.0,
    13960.0,
    14000.0,
    14040.0,
    14080.0,
    14120.0,
    14160.0,
    14200.0,
    14240.0,
    14280.0,
    14320.0,
    14360.0,
    14400.0,
    14440.0,
    14480.0,
    14520.0,
    14560.0,
    14600.0,
    14640.0,
    14680.0,
    14720.0,
    14760.0,
    14800.0,
    14840.0,
    14880.0,
    14920.0,
    14960.0,
    15000.0,
    15040.0,
    15080.0,
    15120.0,
    15160.0,
    15200.0,
    15240.0,
    15280.0,
    15320.0,
    15360.0,
    15400.0,
]