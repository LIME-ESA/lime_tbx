"""
LIME Toolbox CLI Package.

This package provides the command-line interface for the LIME Toolbox, enabling 
users to perform lunar simulations, compare observational data, and export results 
in various formats.

Modules
-------
- `cli.py` : Handles command-line argument parsing and execution.
- `export.py` : Handles the exporting of the obtained results.

Main Features
-------------
- Simulations of lunar irradiance, reflectance, and polarization.
- Comparisons with GLOD observational data.
- Support for CSV, Graph, and NetCDF output formats.
- Coefficient updates for improved accuracy.
- Customizable interpolation and spectral response function settings.

Usage
-----
The CLI can be accessed via the main module:
    $ python3 -m lime_tbx -e lat,lon,height,datetime -o csv,output.csv

For a full list of options, run:
    $ python3 -m lime_tbx -h
"""
