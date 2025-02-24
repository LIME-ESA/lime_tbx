"""Test for cli module"""

"""___Built-In Modules___"""
import getopt
import os
import sys
import io
import shlex
import locale
import warnings
import threading

"""___Third-Party Modules___"""
import unittest
import requests

"""___LIME_TBX Modules___"""
from lime_tbx.common.datatypes import KernelsPath, EocfiPath
from lime_tbx.business.interpolation.interp_data import interp_data
from lime_tbx.application.coefficients.tests.test_update import HTTPServer, get_updater
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
EOCFI_PATH = EocfiPath("./eocfi_data", "./eocfi_data2")

GITLAB_CI = "GITLAB_CI"
GITLAB_CI_VALUE = "GITLAB_CI"

NOT_FORBIDDEN_ROOTROOT_WARN = "Not in a unix env with a protected /root path"


def get_cli():
    cli = CLI(KERNELS_PATH, EOCFI_PATH)
    return cli


def get_opts(args_str: str):
    args = shlex.split(args_str)
    opts, args = getopt.gnu_getopt(args, OPTIONS, LONG_OPTIONS)
    return opts, args


def forbidden_rootroot():
    if os.path.exists("/root") and not os.access("/root", os.W_OK):
        return True
    return False


class TestCLI_CaptureSTDOUTERR(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists("ignore_folder"):
            os.mkdir("ignore_folder")
        cls._prev_lang = ""
        if "LC_ALL" in os.environ:
            cls._prev_lang = os.environ["LC_ALL"]
        locale.setlocale(locale.LC_ALL, "C")

    @classmethod
    def tearDownClass(cls):
        locale.setlocale(locale.LC_ALL, cls._prev_lang)

    def setUp(self):
        warnings.filterwarnings("ignore")
        self.capturedOutput = io.StringIO()
        self.capturedErr = io.StringIO()
        sys.stdout = self.capturedOutput
        sys.stderr = self.capturedErr
        self._prev_skip = interp_data.is_skip_uncertainties()
        interp_data.set_skip_uncertainties(True)

    def tearDown(self):
        warnings.resetwarnings()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        interp_data.set_skip_uncertainties(self._prev_skip)

    def test_get_help(self):
        cli = get_cli()
        args = "-h"
        errcode = cli.handle_input(*get_opts(args))
        self.assertEqual(errcode, 0)
        f = open("./test_files/cli/help.txt")
        self.assertEqual(self.capturedOutput.getvalue(), f.read())
        f.close()

    def test_sat_err_date(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                "-s PROBA-V,2250-01-20T02:00:00 -o graph,png,ignore_folder/refl,ignore_folder/irr,ignore_folder/polar"
            )
        )
        self.assertEqual(errcode, 1)
        f = open("./test_files/cli/sat_err_date.txt")
        self.assertEqual(self.capturedErr.getvalue(), f.read())
        f.close()

    def test_sat_err_date_timeseries(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                "-s ENVISAT -t ./test_files/csv/timeseries.csv -o graph,png,ignore_folder/refl,ignore_folder/irr,ignore_folder/polar"
            )
        )
        self.assertEqual(errcode, 1)
        f = open("./test_files/cli/sat_err_date.txt")
        self.assertEqual(self.capturedErr.getvalue(), f.read())
        f.close()

    @unittest.skipIf(not forbidden_rootroot(), NOT_FORBIDDEN_ROOTROOT_WARN)
    def test_sat_err_forbidden_path_refl(self):
        cli = get_cli()
        with self.assertRaises(SystemExit) as cm:
            cli.handle_input(
                *get_opts(
                    "-s PROBA-V,2020-01-20T02:00:00 -o graph,png,/root,ignore_folder/irr,ignore_folder/polar"
                )
            )
        self.assertEqual(cm.exception.code, 1)
        f = open("./test_files/cli/sat_err_forb_path_refl.txt")
        self.assertEqual(self.capturedErr.getvalue(), f.read())
        f.close()

    @unittest.skipIf(not forbidden_rootroot(), NOT_FORBIDDEN_ROOTROOT_WARN)
    def test_sat_err_forbidden_path_irr(self):
        cli = get_cli()
        with self.assertRaises(SystemExit) as cm:
            cli.handle_input(
                *get_opts(
                    "-s PROBA-V,2020-01-20T02:00:00 -o graph,png,ignore_folder/refl,/root,ignore_folder/polar"
                )
            )
        self.assertEqual(cm.exception.code, 1)
        f = open("./test_files/cli/sat_err_forb_path_irr.txt")
        self.assertEqual(self.capturedErr.getvalue(), f.read())
        f.close()

    @unittest.skipIf(not forbidden_rootroot(), NOT_FORBIDDEN_ROOTROOT_WARN)
    def test_sat_err_forbidden_path_polar(self):
        cli = get_cli()
        with self.assertRaises(SystemExit) as cm:
            cli.handle_input(
                *get_opts(
                    "-s PROBA-V,2020-01-20T02:00:00 -o graph,png,ignore_folder/refl,ignore_folder/irr,/root"
                )
            )
        self.assertEqual(cm.exception.code, 1)
        f = open("./test_files/cli/sat_err_forb_path_polar.txt")
        self.assertEqual(self.capturedErr.getvalue(), f.read())
        f.close()

    def test_sat_err_graphd(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                "-s PROBA-V,2020-01-20T02:00:00 -o graphd,png,ignore_folder/refl,ignore_folder/irr,ignore_folder/polar"
            )
        )
        self.assertEqual(errcode, 1)
        f = open("./test_files/cli/sat_err_graphd.txt")
        self.assertEqual(self.capturedErr.getvalue(), f.read())
        f.close()

    def test_sat_probav_csv_missing_arg(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts("-s PROBA-V,2020-01-20T02:00:00 -o csv,p1,p2,p3")
        )
        self.assertEqual(errcode, 1)
        f = open("./test_files/cli/err_num_args_o_csv.txt")
        self.assertEqual(self.capturedErr.getvalue(), f.read())
        f.close()

    def test_sat_probav_csv_extra_arg(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts("-s PROBA-V,2020-01-20T02:00:00 -o csv,p1,p2,p3,p4,p5")
        )
        self.assertEqual(errcode, 1)
        f = open("./test_files/cli/err_num_args_o_csv.txt")
        self.assertEqual(self.capturedErr.getvalue(), f.read())
        f.close()

    def test_sat_probav_missing_datetime(self):
        cli = get_cli()
        errcode = cli.handle_input(*get_opts("-s PROBA-V -o csv,p1,p2,p3,p4"))
        self.assertEqual(errcode, 1)
        f = open("./test_files/cli/err_miss_datetime_sat.txt")
        self.assertEqual(self.capturedErr.getvalue(), f.read())
        f.close()

    def test_earth_glod_missing_path(self):
        cli = get_cli()
        errcode = cli.handle_input(*get_opts("-e 80,80,2000,2010-10-1T02:02:02 -o nc"))
        self.assertEqual(errcode, 1)
        f = open("./test_files/cli/err_miss_o_nc_path.txt")
        self.assertEqual(self.capturedErr.getvalue(), f.read())
        f.close()

    def test_earth_glod_missing_datetime(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts("-e 80,80,2000 -o nc,./test_files/cli/cliglod.test.nc")
        )
        self.assertEqual(errcode, 1)
        f = open("./test_files/cli/err_miss_e_datetime.txt")
        self.assertEqual(self.capturedErr.getvalue(), f.read())
        f.close()

    def test_lunar_missing_arg(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                "-l 0.98,420000,20.5,-30.2,0.69 -o nc,test_files/cli/cliglod.test.nc"
            )
        )
        self.assertEqual(errcode, 1)
        f = open("./test_files/cli/err_wrong_l_args.txt")
        self.assertEqual(self.capturedErr.getvalue(), f.read())
        f.close()

    def test_lunar_extra_arg(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                "-l 0.98,420000,20.5,-30.2,0.69,15,2 -o nc,test_files/cli/cliglod.test.nc"
            )
        )
        self.assertEqual(errcode, 1)
        f = open("./test_files/cli/err_wrong_l_args.txt")
        self.assertEqual(self.capturedErr.getvalue(), f.read())
        f.close()

    @unittest.skipIf(not forbidden_rootroot(), NOT_FORBIDDEN_ROOTROOT_WARN)
    def test_lunar_forbidden_path(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts("-l 0.98,420000,20.5,-30.2,0.69,15 -o nc,/root")
        )
        self.assertEqual(errcode, 1)
        f = open("./test_files/cli/err_l_forbidden_path.txt")
        self.assertEqual(self.capturedErr.getvalue(), f.read())
        f.close()

    def test_comparison_no_observations(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                '-c "" -f lime_tbx/business/filedata/sample_data/W_XX-EUMETSAT-Darmstadt_VIS+IR+SRF_MSG3+SEVIRI_C_EUMG.nc -o csvd,BOTH,rel,test_files/cli/comp_out.test.dir/'
            )
        )
        self.assertEqual(errcode, 1)
        f = open("./test_files/cli/err_c_no_obs.txt")
        self.assertEqual(self.capturedErr.getvalue(), f.read())
        f.close()

    def test_comparison_glob_csvd_wrong_mpa_dt(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                '-c "lime_tbx/business/filedata/sample_moon_data/W_XX-EUMETSAT*" -f lime_tbx/business/filedata/sample_data/W_XX-EUMETSAT-Darmstadt_VIS+IR+SRF_MSG3+SEVIRI_C_EUMG.nc -o csvd,BATH,rel,test_files/cli/comp_out.test.dir/'
            )
        )
        self.assertEqual(errcode, 1)
        f = open("./test_files/cli/err_c_o_mpa_dt_both_wrong.txt")
        self.assertEqual(self.capturedErr.getvalue(), f.read())
        f.close()

    @unittest.skipIf(not forbidden_rootroot(), NOT_FORBIDDEN_ROOTROOT_WARN)
    def test_comparison_glob_graphd_forbidden_path(self):
        cli = get_cli()
        with self.assertRaises(SystemExit) as cm:
            cli.handle_input(
                *get_opts(
                    '-c "lime_tbx/business/filedata/sample_moon_data/W_XX-EUMETSAT*" -f lime_tbx/business/filedata/sample_data/W_XX-EUMETSAT-Darmstadt_VIS+IR+SRF_MSG3+SEVIRI_C_EUMG.nc -o graphd,png,BOTH,rel,/root'
                )
            )
        self.assertEqual(cm.exception.code, 1)
        f = open("./test_files/cli/err_g_forbidden_path.txt")
        self.assertEqual(self.capturedErr.getvalue(), f.read())
        f.close()

    def test_get_version_ok(self):
        cli = get_cli()
        errcode = cli.handle_input(*get_opts("-v"))
        self.assertEqual(errcode, 0)

    @unittest.skip("Now doesn't make sense to check against real server.")
    def test_update_err_connection(self):
        cli = get_cli()
        errcode = cli.handle_input(*get_opts("-u"))
        self.assertEqual(errcode, 1)
        f = open("./test_files/cli/err_update_connection.txt")
        self.assertEqual(self.capturedOutput.getvalue(), f.read())
        f.close()

    def test_select_coeff_err_earth_glod(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                "-e 80,80,2000,2010-10-1T02:02:02 -o nc,./test_files/cli/cliglod.test.nc -C inventedcoeffs"
            )
        )
        self.assertEqual(errcode, 1)
        f = open("./test_files/cli/err_select_coeffs.txt")
        self.assertEqual(self.capturedErr.getvalue(), f.read())
        f.close()

    def test_select_spectrum_err_earth_glod(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                '-e 80,80,2000,2010-10-1T02:02:02 -o nc,./test_files/cli/cliglod.test.nc -i \'{"interp_spectrum": "inventedspectrum"}\''
            )
        )
        self.assertEqual(errcode, 1)
        f = open("./test_files/cli/err_select_spectrum.txt")
        self.assertEqual(self.capturedErr.getvalue(), f.read())
        f.close()

    def test_timeseries_fake(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                "-e 80,80,2000 -t fakefilefakefile.csv -o nc,./test_files/cli/cliglod.test.nc"
            )
        )
        self.assertEqual(errcode, 1)
        f = open("./test_files/cli/err_no_timeseries.txt")
        self.assertEqual(self.capturedErr.getvalue(), f.read())
        f.close()

    def test_timeseries_wrong(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                "-e 80,80,2000 -t test_files/csv/export_srf_1.csv -o nc,./test_files/cli/cliglod.test.nc"
            )
        )
        self.assertEqual(errcode, 1)
        f = open("./test_files/cli/err_timeseries_wrong.txt")
        self.assertEqual(self.capturedErr.getvalue(), f.read())
        f.close()

    def test_multiple_simops(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                '-e 80,80,2000,2010-10-1T02:02:02  -o nc,./test_files/cli/cliglod.test.nc -c "lime_tbx/business/filedata/sample_moon_data/W_XX-EUMETSAT*" -f lime_tbx/business/filedata/sample_data/W_XX-EUMETSAT-Darmstadt_VIS+IR+SRF_MSG3+SEVIRI_C_EUMG.nc -o graphd,png,BOTH,rel,/root'
            )
        )
        self.assertEqual(errcode, 1)
        with open("./test_files/cli/err_multiple_simops.txt", "r") as f:
            self.assertEqual(self.capturedErr.getvalue(), f.read())
        with open("./test_files/cli/out_multiple_simops.txt", "r") as f:
            self.assertEqual(self.capturedOutput.getvalue(), f.read())

    def test_no_simop(self):
        cli = get_cli()
        errcode = cli.handle_input(*get_opts("-o nc,./test_files/cli/cliglod.test.nc"))
        self.assertEqual(errcode, 1)
        with open("./test_files/cli/err_no_simop.txt", "r") as f:
            self.assertEqual(self.capturedErr.getvalue(), f.read())
        with open("./test_files/cli/out_no_simop.txt", "r") as f:
            self.assertEqual(self.capturedOutput.getvalue(), f.read())

    def test_srf_fake(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                '-c "lime_tbx/business/filedata/sample_moon_data/W_XX-EUMETSAT*" -f fakefile.nc -o csvd,BOTH,rel,test_files/cli/comp_out.test.dir/'
            )
        )
        self.assertEqual(errcode, 1)
        f = open("./test_files/cli/err_srf_fake.txt")
        self.assertEqual(self.capturedErr.getvalue(), f.read())
        f.close()

    def test_srf_wrong(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                '-c "lime_tbx/business/filedata/sample_moon_data/W_XX-EUMETSAT*" -f test_files/moon/comparison.nc -o csvd,BOTH,rel,test_files/cli/comp_out.test.dir/'
            )
        )
        self.assertEqual(errcode, 1)
        f = open("./test_files/cli/err_srf_wrong.txt")
        self.assertEqual(self.capturedErr.getvalue(), f.read())
        f.close()

    def test_graph_few_args(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                "-s PROBA-V,2020-01-20T02:00:00 -o graph,test_files/cli/proba_refl.test,test_files/cli/proba_irr.test,test_files/cli/proba_polar.test"
            )
        )
        self.assertEqual(errcode, 1)
        f = open("./test_files/cli/err_graph_few.txt")
        self.assertEqual(self.capturedErr.getvalue(), f.read())
        f.close()

    def test_graph_wrong(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                "-s PROBA-V,2020-01-20T02:00:00 -o graph,ascii,test_files/cli/proba_refl.test,test_files/cli/proba_irr.test,test_files/cli/proba_polar.test"
            )
        )
        self.assertEqual(errcode, 1)
        f = open("./test_files/cli/err_graph_wrong.txt")
        self.assertEqual(self.capturedErr.getvalue(), f.read())
        f.close()

    def test_graph_comp_few_args(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                '-c "lime_tbx/business/filedata/sample_moon_data/W_XX-EUMETSAT*" -o graph,pdf,test_files/cli/chan1.test'
            )
        )
        self.assertEqual(errcode, 1)
        f = open("./test_files/cli/err_graph_comp_few.txt")
        self.assertEqual(self.capturedErr.getvalue(), f.read())
        f.close()

    def test_graph_comp_img_wrong(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                '-c "lime_tbx/business/filedata/sample_moon_data/W_XX-EUMETSAT*" -o graph,ascii,MPA,rel,test_files/cli/chan1.test'
            )
        )
        self.assertEqual(errcode, 1)
        f = open("./test_files/cli/err_graph_comp_wrong.txt")
        self.assertEqual(self.capturedErr.getvalue(), f.read())
        f.close()

    def test_graph_comp_type_wrong(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                '-c "lime_tbx/business/filedata/sample_moon_data/W_XX-EUMETSAT*" -o graph,pdf,LUCK,rel,test_files/cli/chan1.test'
            )
        )
        self.assertEqual(errcode, 1)
        f = open("./test_files/cli/err_graph_comp_type_wrong.txt")
        self.assertEqual(self.capturedErr.getvalue(), f.read())
        f.close()


class TestCLI(unittest.TestCase):
    # Cant compare output as interpolation gives different results each time, and its too slow to perform always

    def setUp(self):
        self._prev_skip = interp_data.is_skip_uncertainties()
        interp_data.set_skip_uncertainties(True)

    def tearDown(self):
        interp_data.set_skip_uncertainties(self._prev_skip)

    def test_earth_glod_ok(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                "-e 80,80,2000,2010-10-1T02:02:02 -o nc,./test_files/cli/cliglod.test.nc"
            )
        )
        self.assertEqual(errcode, 0)

    def test_earth_glod_ok_select_coeff(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                "-e 80,80,2000,2010-10-1T02:02:02 -o nc,./test_files/cli/cliglod.test.nc -C 20231120_v1"
            )
        )
        self.assertEqual(errcode, 0)
        os.remove(os.path.join(".", "coeff_data", "selected.txt"))

    def test_earth_glod_ok_select_spectrum(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                '-e 80,80,2000,2010-10-1T02:02:02 -o nc,./test_files/cli/cliglod.test.nc -i \'{"interp_spectrum": "ASD"}\''
            )
        )
        self.assertEqual(errcode, 0)

    def test_earth_glod_ok_select_mult_inter_setts(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                '-e 80,80,2000,2010-10-1T02:02:02 -o nc,./test_files/cli/cliglod.test.nc -i \
\'{"interp_spectrum": "ASD", "interp_srf": "asd", "show_inter_spectrum": "False", "skip_uncertainties": \
"True", "show_cimel_points": "True"}\''
            )
        )
        interp_data.set_interpolation_spectrum_name("ASD")
        interp_data.set_interpolation_SRF(
            "Gaussian SRF with 1nm spectral sampling and 3nm resolution"
        )
        interp_data.set_skip_uncertainties(False)
        self.assertEqual(errcode, 0)

    def test_earth_glod_ok_select_spectrum_apollo(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                '-e 80,80,2000,2010-10-1T02:02:02 -o nc,./test_files/cli/cliglod.test.nc -i \'{"interp_spectrum": "Apollo 16 + Breccia"}\''
            )
        )
        self.assertEqual(errcode, 0)
        interp_data.set_interpolation_spectrum_name("ASD")

    def test_earth_timeseries_glod_ok(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                "-e 80,80,2000 -t ./test_files/csv/timeseries.csv -o nc,./test_files/cli/cliglod.test.nc"
            )
        )
        self.assertEqual(errcode, 0)

    def test_earth_timeseries_with_datetime_glod_ok(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                "-e 80,80,2000,2010-10-1T02:02:02 -t ./test_files/csv/timeseries.csv -o nc,./test_files/cli/cliglod.test.nc"
            )
        )
        self.assertEqual(errcode, 0)

    def test_lunar_ok(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                "-l 0.98,420000,20.5,-30.2,0.69,15 -o nc,test_files/cli/cliglod.test.nc"
            )
        )
        self.assertEqual(errcode, 0)

    def test_sat_probav_csv(self):
        path_refl = "./test_files/cli/proba_refl.test.csv"
        path_irr = "./test_files/cli/proba_irr.test.csv"
        path_polar = "./test_files/cli/proba_polar.test.csv"
        path_integrated = "./test_files/cli/proba_integrated.test.csv"
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                "-s PROBA-V,2020-01-20T02:00:00 -o csv,{},{},{},{}".format(
                    path_refl, path_irr, path_polar, path_integrated
                )
            )
        )
        self.assertEqual(errcode, 0)
        # This method doesnt work because interpolation gives different results each time
        """
        self.assertTrue(filecmp.cmp(path_refl, "./test_files/cli/proba_refl.csv"))
        self.assertTrue(filecmp.cmp(path_irr, "./test_files/cli/proba_irr.csv"))
        self.assertTrue(filecmp.cmp(path_polar, "./test_files/cli/proba_polar.csv"))
        self.assertTrue(filecmp.cmp(path_integrated, "./test_files/cli/proba_integrated.csv"))
        """

    def test_sat_probav_graph(self):
        if GITLAB_CI in os.environ and os.environ[GITLAB_CI] == GITLAB_CI_VALUE:
            self.skipTest("Graph output fails in python docker of gitlab ci")
        path_refl = "./test_files/cli/proba_refl.test.png"
        path_irr = "./test_files/cli/proba_irr.test.png"
        path_polar = "./test_files/cli/proba_polar.test.png"
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                "-s PROBA-V,2020-01-20T02:00:00 -o graph,png,test_files/cli/proba_refl.test,test_files/cli/proba_irr.test,test_files/cli/proba_polar.test"
            )
        )
        self.assertEqual(errcode, 0)
        self.assertTrue(os.path.exists(path_refl))
        self.assertTrue(os.path.exists("./test_files/cli/proba_refl.png"))
        self.assertTrue(os.path.exists(path_irr))
        self.assertTrue(os.path.exists("./test_files/cli/proba_irr.png"))
        self.assertTrue(os.path.exists(path_polar))
        self.assertTrue(os.path.exists("./test_files/cli/proba_polar.png"))

    def test_comparison_glob_csvd_both_ok(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                '-c "lime_tbx/business/filedata/sample_moon_data/W_XX-EUMETSAT*" -f lime_tbx/business/filedata/sample_data/W_XX-EUMETSAT-Darmstadt_VIS+IR+SRF_MSG3+SEVIRI_C_EUMG.nc -o csvd,BOTH,rel,test_files/cli/comp_out.test.dir/'
            )
        )
        self.assertEqual(errcode, 0)

    def test_comparison_glob_csvd_channel_ok(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                '-c "lime_tbx/business/filedata/sample_moon_data/W_XX-EUMETSAT*" -f lime_tbx/business/filedata/sample_data/W_XX-EUMETSAT-Darmstadt_VIS+IR+SRF_MSG3+SEVIRI_C_EUMG.nc -o csvd,CHANNEL,rel,test_files/cli/comp_out.test.dir/'
            )
        )
        self.assertEqual(errcode, 0)

    def test_comparison_glob_csvd_channel_mean_ok(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                '-c "lime_tbx/business/filedata/sample_moon_data/W_XX-EUMETSAT*" -f lime_tbx/business/filedata/sample_data/W_XX-EUMETSAT-Darmstadt_VIS+IR+SRF_MSG3+SEVIRI_C_EUMG.nc -o csvd,CHANNEL_MEAN,rel,test_files/cli/comp_out.test.dir/'
            )
        )
        self.assertEqual(errcode, 0)

    def test_comparison_glob_csvd_both_perc_ok(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                '-c "lime_tbx/business/filedata/sample_moon_data/W_XX-EUMETSAT*" -f lime_tbx/business/filedata/sample_data/W_XX-EUMETSAT-Darmstadt_VIS+IR+SRF_MSG3+SEVIRI_C_EUMG.nc -o csvd,BOTH,perc,test_files/cli/comp_out.test.dir/'
            )
        )
        self.assertEqual(errcode, 0)

    def test_comparison_glob_csvd_channel_perc_ok(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                '-c "lime_tbx/business/filedata/sample_moon_data/W_XX-EUMETSAT*" -f lime_tbx/business/filedata/sample_data/W_XX-EUMETSAT-Darmstadt_VIS+IR+SRF_MSG3+SEVIRI_C_EUMG.nc -o csvd,CHANNEL,perc,test_files/cli/comp_out.test.dir/'
            )
        )
        self.assertEqual(errcode, 0)

    def test_comparison_glob_csvd_channel_mean_perc_ok(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                '-c "lime_tbx/business/filedata/sample_moon_data/W_XX-EUMETSAT*" -f lime_tbx/business/filedata/sample_data/W_XX-EUMETSAT-Darmstadt_VIS+IR+SRF_MSG3+SEVIRI_C_EUMG.nc -o csvd,CHANNEL_MEAN,perc,test_files/cli/comp_out.test.dir/'
            )
        )
        self.assertEqual(errcode, 0)

    def test_comparison_glob_graph_both_ok(self):
        if GITLAB_CI in os.environ and os.environ[GITLAB_CI] == GITLAB_CI_VALUE:
            self.skipTest("Graph output fails in python docker of gitlab ci")
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                '-c "lime_tbx/business/filedata/sample_moon_data/W_XX-EUMETSAT*" -f lime_tbx/business/filedata/sample_data/W_XX-EUMETSAT-Darmstadt_VIS+IR+SRF_MSG3+SEVIRI_C_EUMG.nc -o graph,jpg,DT,rel,test_files/cli/out_comp_chann.test.jpg,test_files/cli/out_comp_chann2.test.jpg,test_files/cli/out_comp_chann3.test.jpg'
            )
        )
        self.assertEqual(errcode, 0)

    def test_comparison_glob_graph_channel_ok(self):
        if GITLAB_CI in os.environ and os.environ[GITLAB_CI] == GITLAB_CI_VALUE:
            self.skipTest("Graph output fails in python docker of gitlab ci")
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                '-c "lime_tbx/business/filedata/sample_moon_data/W_XX-EUMETSAT*" -f lime_tbx/business/filedata/sample_data/W_XX-EUMETSAT-Darmstadt_VIS+IR+SRF_MSG3+SEVIRI_C_EUMG.nc -o graph,jpg,CHANNEL,rel,test_files/cli/out_comp_allchannels.test.jpg'
            )
        )
        self.assertEqual(errcode, 0)

    def test_comparison_glob_graph_channel_mean_ok(self):
        if GITLAB_CI in os.environ and os.environ[GITLAB_CI] == GITLAB_CI_VALUE:
            self.skipTest("Graph output fails in python docker of gitlab ci")
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                '-c "lime_tbx/business/filedata/sample_moon_data/W_XX-EUMETSAT*" -f lime_tbx/business/filedata/sample_data/W_XX-EUMETSAT-Darmstadt_VIS+IR+SRF_MSG3+SEVIRI_C_EUMG.nc -o graph,jpg,CHANNEL_MEAN,rel,test_files/cli/out_comp_allchannels.test.jpg'
            )
        )
        self.assertEqual(errcode, 0)

    def test_comparison_glob_nc_both_ok(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                '-c "lime_tbx/business/filedata/sample_moon_data/W_XX-EUMETSAT*" -f lime_tbx/business/filedata/sample_data/W_XX-EUMETSAT-Darmstadt_VIS+IR+SRF_MSG3+SEVIRI_C_EUMG.nc -o nc,test_files/cli/compcli.test.nc'
            )
        )
        self.assertEqual(errcode, 0)

    def test_comparison_glob_graphd_png_both_ok(self):
        if GITLAB_CI in os.environ and os.environ[GITLAB_CI] == GITLAB_CI_VALUE:
            self.skipTest("Graph output fails in python docker of gitlab ci")
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                '-c "lime_tbx/business/filedata/sample_moon_data/W_XX-EUMETSAT*" -f lime_tbx/business/filedata/sample_data/W_XX-EUMETSAT-Darmstadt_VIS+IR+SRF_MSG3+SEVIRI_C_EUMG.nc -o graphd,png,BOTH,rel,test_files/cli/comp_out.test.dir/'
            )
        )
        self.assertEqual(errcode, 0)
        graphs = [
            "NIR016.dt.png",
            "NIR016.mpa.png",
            "VIS006.dt.png",
            "VIS006.mpa.png",
            "VIS008.dt.png",
            "VIS008.mpa.png",
        ]
        src_path = "test_files/cli/comp_out.dir"
        test_path = "test_files/cli/comp_out.test.dir"
        for gname in graphs:
            # Dont know how to compare
            self.assertTrue(os.path.exists(os.path.join(test_path, gname)))

    def test_comparison_glob_graphd_png_channel_ok(self):
        if GITLAB_CI in os.environ and os.environ[GITLAB_CI] == GITLAB_CI_VALUE:
            self.skipTest("Graph output fails in python docker of gitlab ci")
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                '-c "lime_tbx/business/filedata/sample_moon_data/W_XX-EUMETSAT*" -f lime_tbx/business/filedata/sample_data/W_XX-EUMETSAT-Darmstadt_VIS+IR+SRF_MSG3+SEVIRI_C_EUMG.nc -o graphd,png,CHANNEL,rel,test_files/cli/comp_out.test.dir/'
            )
        )
        self.assertEqual(errcode, 0)
        graphs = [
            "allchannels.png",
        ]
        test_path = "test_files/cli/comp_out.test.dir"
        for gname in graphs:
            self.assertTrue(os.path.exists(os.path.join(test_path, gname)))

    def test_comparison_glob_graphd_png_channel_mean_ok(self):
        if GITLAB_CI in os.environ and os.environ[GITLAB_CI] == GITLAB_CI_VALUE:
            self.skipTest("Graph output fails in python docker of gitlab ci")
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                '-c "lime_tbx/business/filedata/sample_moon_data/W_XX-EUMETSAT*" -f lime_tbx/business/filedata/sample_data/W_XX-EUMETSAT-Darmstadt_VIS+IR+SRF_MSG3+SEVIRI_C_EUMG.nc -o graphd,png,CHANNEL_MEAN,rel,test_files/cli/comp_out.test.dir/'
            )
        )
        self.assertEqual(errcode, 0)
        graphs = [
            "allchannels.png",
        ]
        test_path = "test_files/cli/comp_out.test.dir"
        for gname in graphs:
            self.assertTrue(os.path.exists(os.path.join(test_path, gname)))

    def test_comparison_glob_csvd_dt_ok(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                '-c "lime_tbx/business/filedata/sample_moon_data/W_XX-EUMETSAT*" -f lime_tbx/business/filedata/sample_data/W_XX-EUMETSAT-Darmstadt_VIS+IR+SRF_MSG3+SEVIRI_C_EUMG.nc -o csvd,DT,rel,test_files/cli/comp_out.test.dir/'
            )
        )
        self.assertEqual(errcode, 0)

    def test_comparison_glob_csvd_mpa_ok(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                '-c "lime_tbx/business/filedata/sample_moon_data/W_XX-EUMETSAT*" -f lime_tbx/business/filedata/sample_data/W_XX-EUMETSAT-Darmstadt_VIS+IR+SRF_MSG3+SEVIRI_C_EUMG.nc -o csvd,MPA,rel,test_files/cli/comp_out.test.dir/'
            )
        )
        self.assertEqual(errcode, 0)

    def test_comparison_csvd_both_ok(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                '-c "lime_tbx/business/filedata/sample_moon_data/W_XX-EUMETSAT-Darmstadt,VISNIR+SUBSET+MOON,MSG3+SEVIRI_C_EUMG_20130101145644_01.nc lime_tbx/business/filedata/sample_moon_data/W_XX-EUMETSAT-Darmstadt,VISNIR+SUBSET+MOON,MSG3+SEVIRI_C_EUMG_20140318140112_01.nc, lime_tbx/business/filedata/sample_moon_data/W_XX-EUMETSAT-Darmstadt,VISNIR+SUBSET+MOON,MSG3+SEVIRI_C_EUMG_20140715153303_01.nc" -f lime_tbx/business/filedata/sample_data/W_XX-EUMETSAT-Darmstadt_VIS+IR+SRF_MSG3+SEVIRI_C_EUMG.nc -o csvd,BOTH,rel,test_files/cli/comp_out.test.dir/'
            )
        )
        self.assertEqual(errcode, 0)

    def test_comparison_csvd_channel_ok(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                '-c "lime_tbx/business/filedata/sample_moon_data/W_XX-EUMETSAT-Darmstadt,VISNIR+SUBSET+MOON,MSG3+SEVIRI_C_EUMG_20130101145644_01.nc lime_tbx/business/filedata/sample_moon_data/W_XX-EUMETSAT-Darmstadt,VISNIR+SUBSET+MOON,MSG3+SEVIRI_C_EUMG_20140318140112_01.nc, lime_tbx/business/filedata/sample_moon_data/W_XX-EUMETSAT-Darmstadt,VISNIR+SUBSET+MOON,MSG3+SEVIRI_C_EUMG_20140715153303_01.nc" -f lime_tbx/business/filedata/sample_data/W_XX-EUMETSAT-Darmstadt_VIS+IR+SRF_MSG3+SEVIRI_C_EUMG.nc -o csvd,CHANNEL,rel,test_files/cli/comp_out.test.dir/'
            )
        )
        self.assertEqual(errcode, 0)

    def test_comparison_csvd_channel_mean_ok(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                '-c "lime_tbx/business/filedata/sample_moon_data/W_XX-EUMETSAT-Darmstadt,VISNIR+SUBSET+MOON,MSG3+SEVIRI_C_EUMG_20130101145644_01.nc lime_tbx/business/filedata/sample_moon_data/W_XX-EUMETSAT-Darmstadt,VISNIR+SUBSET+MOON,MSG3+SEVIRI_C_EUMG_20140318140112_01.nc, lime_tbx/business/filedata/sample_moon_data/W_XX-EUMETSAT-Darmstadt,VISNIR+SUBSET+MOON,MSG3+SEVIRI_C_EUMG_20140715153303_01.nc" -f lime_tbx/business/filedata/sample_data/W_XX-EUMETSAT-Darmstadt_VIS+IR+SRF_MSG3+SEVIRI_C_EUMG.nc -o csvd,CHANNEL_MEAN,rel,test_files/cli/comp_out.test.dir/'
            )
        )
        self.assertEqual(errcode, 0)

    def test_comparison_csvd_both_perc_ok(self):
        cli = get_cli()
        errcode = cli.handle_input(
            *get_opts(
                '-c "lime_tbx/business/filedata/sample_moon_data/W_XX-EUMETSAT-Darmstadt,VISNIR+SUBSET+MOON,MSG3+SEVIRI_C_EUMG_20130101145644_01.nc lime_tbx/business/filedata/sample_moon_data/W_XX-EUMETSAT-Darmstadt,VISNIR+SUBSET+MOON,MSG3+SEVIRI_C_EUMG_20140318140112_01.nc, lime_tbx/business/filedata/sample_moon_data/W_XX-EUMETSAT-Darmstadt,VISNIR+SUBSET+MOON,MSG3+SEVIRI_C_EUMG_20140715153303_01.nc" -f lime_tbx/business/filedata/sample_data/W_XX-EUMETSAT-Darmstadt_VIS+IR+SRF_MSG3+SEVIRI_C_EUMG.nc -o csvd,BOTH,perc,test_files/cli/comp_out.test.dir/'
            )
        )
        self.assertEqual(errcode, 0)


class TestCLIUpdateNoServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists("ignore_folder"):
            os.mkdir("ignore_folder")
        cls._prev_lang = ""
        if "LC_ALL" in os.environ:
            cls._prev_lang = os.environ["LC_ALL"]
        locale.setlocale(locale.LC_ALL, "C")

    @classmethod
    def tearDownClass(cls):
        locale.setlocale(locale.LC_ALL, cls._prev_lang)

    def setUp(self):
        warnings.filterwarnings("ignore")
        self.capturedOutput = io.StringIO()
        self.capturedErr = io.StringIO()
        sys.stdout = self.capturedOutput
        sys.stderr = self.capturedErr
        self._prev_skip = interp_data.is_skip_uncertainties()
        interp_data.set_skip_uncertainties(True)

    def tearDown(self):
        warnings.resetwarnings()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        interp_data.set_skip_uncertainties(self._prev_skip)

    def test_connection_error(self):
        cli = get_cli()
        cli.updater = get_updater()
        cli.updater.url = "http://localhost:6969/listv.txt"  # Which is not the same
        errcode = cli.handle_input(*get_opts("-u"))
        self.assertEqual(errcode, 1)
        f = open("./test_files/cli/err_update_connection.txt")
        self.assertEqual(self.capturedOutput.getvalue(), f.read())
        f.close()


class TestCLIUpdate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists("ignore_folder"):
            os.mkdir("ignore_folder")
        cls._prev_lang = ""
        if "LC_ALL" in os.environ:
            cls._prev_lang = os.environ["LC_ALL"]
        locale.setlocale(locale.LC_ALL, "C")

        dirname = os.path.join(os.path.dirname(__file__), "../../../coeff_data")
        cls.httpd = HTTPServer(dirname, ("localhost", 8000))
        cls.t = threading.Thread(
            name="test server proc", target=cls.httpd.serve_forever
        )
        cls.t.start()

    @classmethod
    def tearDownClass(cls):
        locale.setlocale(locale.LC_ALL, cls._prev_lang)
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.t.join()

    def setUp(self):
        warnings.filterwarnings("ignore")
        self.capturedOutput = io.StringIO()
        self.capturedErr = io.StringIO()
        sys.stdout = self.capturedOutput
        sys.stderr = self.capturedErr
        self._prev_skip = interp_data.is_skip_uncertainties()
        interp_data.set_skip_uncertainties(True)

    def tearDown(self):
        warnings.resetwarnings()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        interp_data.set_skip_uncertainties(self._prev_skip)

    def test_download(self):
        cli = get_cli()
        cli.updater = get_updater()
        errcode = cli.handle_input(*get_opts("-u"))
        self.assertEqual(errcode, 0)
        msg = "Download finished.\nThere were no updates.\n"
        self.assertEqual(self.capturedOutput.getvalue(), msg)


class TestCLITrueUpdate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists("ignore_folder"):
            os.mkdir("ignore_folder")
        cls._prev_lang = ""
        if "LC_ALL" in os.environ:
            cls._prev_lang = os.environ["LC_ALL"]
        locale.setlocale(locale.LC_ALL, "C")

        dirname = os.path.join(
            os.path.dirname(__file__), "../../../test_files/update/coeff_data"
        )
        cls.httpd = HTTPServer(dirname, ("localhost", 8000))
        cls.t = threading.Thread(
            name="test server proc", target=cls.httpd.serve_forever
        )
        cls.t.start()

    @classmethod
    def tearDownClass(cls):
        locale.setlocale(locale.LC_ALL, cls._prev_lang)
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.t.join()
        os.remove("coeff_data/versions/LIME_MODEL_COEFS_20230123_V02.nc")

    def setUp(self):
        warnings.filterwarnings("ignore")
        self.capturedOutput = io.StringIO()
        self.capturedErr = io.StringIO()
        sys.stdout = self.capturedOutput
        sys.stderr = self.capturedErr
        self._prev_skip = interp_data.is_skip_uncertainties()
        interp_data.set_skip_uncertainties(True)

    def tearDown(self):
        warnings.resetwarnings()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        interp_data.set_skip_uncertainties(self._prev_skip)

    def test_download_working(self):
        cli = get_cli()
        cli.updater = get_updater()
        errcode = cli.handle_input(*get_opts("-u"))
        self.assertEqual(errcode, 0)
        msg = (
            f"Download finished with errors.\nThere were 2 updates, 1 of them failed.\n"
        )
        self.assertEqual(self.capturedOutput.getvalue(), msg)
