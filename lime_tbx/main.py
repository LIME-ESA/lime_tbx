"""___Built-In Modules___"""
import os
import getopt
import sys
from datetime import datetime

"""___Third-Party Modules___"""
# import here

"""___LIME_TBX Modules___"""
from lime_tbx.cli.cli import CLI
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
    options = "hg:l:s:o:f:"
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
        output_file = "output.csv"
        srf_file = ""
        # find settings data
        for opt, arg in opts:
            if opt == "-o":
                output_file = arg
            elif opt == "-f":
                srf_file = arg
        cli.load_srf(srf_file)
        for opt, arg in opts:
            if opt == "-h":
                print(
                    "lime [-h | -g lat_deg,lon_deg,height_m,{} | -l <distance_sun_moon,distance_observer_moon,\
selen_obs_lat,selen_obs_lon,selen_sun_lon,moon_phase_angle> | -s <sat_name,{}> | -o output_file.csv | -f srf.nc]".format(
                        _DT_FORMAT, _DT_FORMAT
                    )
                )
            elif opt == "-g":  # Geographic
                params_str = arg.split(",")
                params = list(map(float, params_str[:3]))
                lat = params[0]
                lon = params[1]
                height = params[2]
                dt = datetime.strptime(params_str[3] + "+00:00", _DT_FORMAT + "%z")
                cli.calculate_geographic(lat, lon, height, dt, output_file)
            elif opt == "-s":
                params_str = arg.split(",")
                sat_name = params_str[0]
                dt = datetime.strptime(params_str[1] + "+00:00", _DT_FORMAT + "%z")
                cli.calculate_satellital(sat_name, dt, output_file)
            elif opt == "-l":
                params = list(map(float, arg.split(",")))
                cli.calculate_selenographic(*params, output_file)


if __name__ == "__main__":
    main()
