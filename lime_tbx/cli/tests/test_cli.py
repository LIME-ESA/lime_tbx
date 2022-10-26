"""Test for cli module"""

"""___Built-In Modules___"""
import getopt
import sys
import filecmp
import io

"""___Third-Party Modules___"""
import unittest

"""___LIME_TBX Modules___"""
from lime_tbx.datatypes.datatypes import KernelsPath
from ..cli import (
    CLI,
    OPTIONS,
    LONG_OPTIONS,
)

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "26/10/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


KERNELS_PATH = KernelsPath("./kernels", "./kernels")
EOCFI_PATH = "./eocfi_data"


def get_cli():
    cli = CLI(KERNELS_PATH, EOCFI_PATH)
    return cli


def get_opts(args_str: str):
    args = args_str.split()
    opts, args = getopt.getopt(args, OPTIONS, LONG_OPTIONS)
    return opts


class TestCLI_CaptureSTDOUTERR(unittest.TestCase):
    def setUp(self):
        self.capturedOutput = io.StringIO()
        self.capturedErr = io.StringIO()
        sys.stdout = self.capturedOutput
        sys.stderr = self.capturedErr

    def tearDown(self):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    def test_get_help(self):
        cli = get_cli()
        args = "-h"
        cli.handle_input(get_opts(args))
        f = open("./test_files/cli/help.txt")
        self.assertEqual(self.capturedOutput.getvalue(), f.read())
        f.close()

    def test_sat_err_date(self):
        cli = get_cli()
        cli.handle_input(
            get_opts(
                "-s PROBA-V,2023-01-20T02:00:00 -o graph,png,ignore_folder/refl,ignore_folder/irr,ignore_folder/polar"
            )
        )
        f = open("./test_files/cli/sat_err_date.txt")
        self.assertEqual(self.capturedErr.getvalue(), f.read())
        f.close()

    def test_sat_err_graphd(self):
        cli = get_cli()
        cli.handle_input(
            get_opts(
                "-s PROBA-V,2020-01-20T02:00:00 -o graphd,png,ignore_folder/refl,ignore_folder/irr,ignore_folder/polar"
            )
        )
        f = open("./test_files/cli/sat_err_graphd.txt")
        self.assertEqual(self.capturedErr.getvalue(), f.read())
        f.close()


class TestCLI(unittest.TestCase):

    """
    # This method doesnt work because interpolation gives different results each time
    def test_sat_probav_csv(self):
        path_refl = "./test_files/cli/proba_refl.test.csv"
        path_irr = "./test_files/cli/proba_irr.test.csv"
        path_polar = "./test_files/cli/proba_polar.test.csv"
        cli = get_cli()
        cli.handle_input(get_opts("-s PROBA-V,2020-01-20T02:00:00 -o csv,{},{},{}".format(path_refl, path_irr, path_polar)))
        self.assertTrue(filecmp.cmp(path_refl, "./test_files/cli/proba_refl.csv"))
        self.assertTrue(filecmp.cmp(path_irr, "./test_files/cli/proba_irr.csv"))
        self.assertTrue(filecmp.cmp(path_polar, "./test_files/cli/proba_polar.csv"))
    """

    def test_sat_probav_graph(self):
        try:
            from sewar.full_ref import uqi
        except:
            print("Missing 'sewar' library (needed for some image testing).")
            return
        import matplotlib.image as mpimg

        path_refl = "./test_files/cli/proba_refl.test.png"
        path_irr = "./test_files/cli/proba_irr.test.png"
        path_polar = "./test_files/cli/proba_polar.test.png"
        cli = get_cli()
        cli.handle_input(
            get_opts(
                "-s PROBA-V,2020-01-20T02:00:00 -o graph,png,test_files/cli/proba_refl.test,test_files/cli/proba_irr.test,test_files/cli/proba_polar.test"
            )
        )
        self.assertGreater(
            uqi(
                mpimg.imread(path_refl), mpimg.imread("./test_files/cli/proba_refl.png")
            ),
            0.99,
        )
        self.assertGreater(
            uqi(mpimg.imread(path_irr), mpimg.imread("./test_files/cli/proba_irr.png")),
            0.99,
        )
        self.assertGreater(
            uqi(
                mpimg.imread(path_polar),
                mpimg.imread("./test_files/cli/proba_polar.png"),
            ),
            0.99,
        )
