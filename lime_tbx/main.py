"""___Built-In Modules___"""
import os
import getopt
import sys
from datetime import datetime

from lime_tbx.filedata import csv

"""___Third-Party Modules___"""
# import here

"""___LIME_TBX Modules___"""
from lime_tbx.cli.cli import (
    CLI,
    ExportCSV,
    ExportComparisonCSV,
    ExportData,
    ExportNetCDF,
)
from lime_tbx.coefficients.access_data.appdata import (
    get_appdata_folder,
    get_programfiles_folder,
)
from lime_tbx.datatypes.datatypes import KernelsPath

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"

_DT_FORMAT = "%Y-%m-%dT%H:%M:%S"


def main():
    programfiles = get_programfiles_folder()
    appdata = get_appdata_folder()
    kernels_path = KernelsPath(
        os.path.join(programfiles, "kernels"), os.path.join(appdata, "kernels")
    )
    eocfi_path = os.path.join(programfiles, "eocfi_data")
    args = sys.argv[1:]
    options = "he:l:s:c:o:f:t:"
    long_options = []
    try:
        opts, args = getopt.getopt(args, options, long_options)
    except getopt.GetoptError:
        print("lime {}".format(["-" + o for o in options if o != ":"]))
        sys.exit(2)
    if len(opts) == 0:
        from lime_tbx.gui.gui import GUI

        gui = GUI(kernels_path, eocfi_path)
    else:
        cli = CLI(kernels_path, eocfi_path)
        srf_file = ""
        export_data: ExportData = None
        timeseries_file: str = None
        # find settings data
        is_comparison = False
        for opt, arg in opts:
            if opt == "-c":
                is_comparison = True
                break
        for opt, arg in opts:
            if opt == "-h":
                print(
                    'lime [-h | -t timeseries.csv (-e lat_deg,lon_deg,height_m,{} | -l <distance_sun_moon,distance_observer_moon,\
selen_obs_lat,selen_obs_lon,selen_sun_lon,moon_phase_angle> | -s <sat_name,{}> | -c "input_glod1.nc input_lglod2.nc ...")\
-o (csv,refl.csv,irr.csv,polar.csv | csv,comparisons.csv| nc,output_file.nc) -f srf.nc]'.format(
                        _DT_FORMAT, _DT_FORMAT
                    )
                )
                return
            if opt == "-o":
                splitted = arg.split(",")
                o_type = splitted[0]
                if o_type == "csv":
                    if not is_comparison:
                        export_data = ExportCSV(splitted[1], splitted[2], splitted[3])
                    else:
                        export_data = ExportComparisonCSV(splitted[1:])
                elif o_type == "nc":
                    export_data = ExportNetCDF(splitted[1])
            elif opt == "-f":
                srf_file = arg
            elif opt == "-t":
                timeseries_file = arg
        cli.load_srf(srf_file)
        for opt, arg in opts:
            if opt == "-e":  # Earth
                params_str = arg.split(",")
                params = list(map(float, params_str[:3]))
                lat = params[0]
                lon = params[1]
                height = params[2]
                if timeseries_file != None:
                    dt = csv.read_datetimes(timeseries_file)
                else:
                    dt = datetime.strptime(params_str[3] + "+00:00", _DT_FORMAT + "%z")
                cli.calculate_geographic(lat, lon, height, dt, export_data)
            elif opt == "-s":  # Satellite
                params_str = arg.split(",")
                sat_name = params_str[0]
                if timeseries_file != None:
                    dt = csv.read_datetimes(timeseries_file)
                else:
                    dt = datetime.strptime(params_str[1] + "+00:00", _DT_FORMAT + "%z")
                cli.calculate_satellital(sat_name, dt, export_data)
            elif opt == "-l":  # Lunar
                params = list(map(float, arg.split(",")))
                cli.calculate_selenographic(*params, export_data)
            elif opt == "-c":  # Comparison
                input_files = arg.split(" ")
                cli.calculate_comparisons(input_files, export_data)


if __name__ == "__main__":
    main()
