# Simulations

LIME Toolbox allows users to simulate lunar reflectance, irradiance, the degree of linear polarisation (DoLP),
and the angle of linear polarisation (AoLP).
Simulations are performed using the Simulation page ([Figure 6](#fig-6)), which is the default view when
launching LIME Toolbox.

Users can select one of three simulation methods:
- **Geographic Coordinates** – Based on an Earth-based location.
- **Selenographic Coordinates** – Based on a location on the Moon.
- **Satellite Position** – Based on an orbiting satellite’s position.

```{eval-rst}
.. figure:: ../../images/user_guide/simulation_view.png
   :name: fig-6
   :align: center
   :alt: Simulation page

   Simulation page.
```

## Simulation Using Geographic Coordinates

Selecting the "Geographic" tab allows users to perform simulations based on Earth coordinates ([Figure 7](#fig-7)).
Users must provide:
- **Latitude & Longitude** (in decimal degrees)
- **Altitude** (in kilometers)
- **UTC Date & Time**

```{eval-rst}
.. figure:: ../../images/user_guide/geographic_input.png
   :name: fig-7
   :align: center
   :alt: Geographic coordinates input

   Geographic coordinates input.
```

### Time-Series Input
For multiple timestamps, click "LOAD TIME-SERIES", which enables time-series file input ([Figure 8](#fig-8)).

```{eval-rst}
.. figure:: ../../images/user_guide/geographic_input_multidates.png
   :name: fig-8
   :align: center
   :alt: Geographic coordinates input for multiple timestamps

   Geographic coordinates input for multiple timestamps.
```

#### Loading Time-Series Data
1. Click "LOAD FILE" to open a file selection dialog.
2. Select a file containing timestamps in one of the following formats:
   - **ISO 8601 format**
   - **Comma-separated values (CSV)**:
     - `yyyy,mm,dd,HH,MM,SS`
     - `yyyy,mm,dd,HH,MM,SS,microseconds`

#### Example Valid File
```
2022,01,17,02,02,04
2023,01,17,02,02,04,123545
2020-01-12 00:22:43.124
```

**Additional options:**
- **"SEE TIMES"** – Opens a window displaying imported timestamps.
- **"INPUT SINGLE TIME"** – Switches back to single timestamp mode.


### Command-Line Interface (CLI)
Use the `-e` or `--earth` option:
```sh
lime -e latitude_deg,longitude_deg,height_km,datetime_isoformat
```
#### Example
For a simulation of 35° of latitude, -25.2° of longitude, 400 meters of altitude,
and at the date and time of 2022-01-20 02:00:00 UTC:
```sh
lime -e 35,-25.2,0.4,2022-01-20T02:00:00
```
For time-series input:
```sh
lime -e 35,-25.2,0.4 -t timeseries.txt
```

## Simulation Using Selenographic Coordinates

Selecting the "Selenographic" tab enables simulations using Moon-based coordinates ([Figure 9](#fig-9)).
Users must provide:
- **Sun-Moon Distance** (AU)
- **Observer-Moon Distance** (km)
- **Selenographic Latitude & Longitude of Observer** (decimal degrees)
- **Selenographic Longitude of the Sun** (decimal degrees)
- **Moon Phase Angle** (decimal degrees)

```{eval-rst}
.. figure:: ../../images/user_guide/selenographic_input.png
   :name: fig-9
   :align: center
   :alt: Selenographic coordinates input

   Selenographic coordinates input.
```

### Multiple Points Input  
You can load several selenographic coordinates at once instead of entering them individually.

```{eval-rst}
.. figure:: ../../images/user_guide/dropdown_multiseleno.png
   :name: fig-10
   :align: center
   :alt: Selenographic tab's dropdown menu, with Multiple Points option highlighted

   Selenographic tab's dropdown menu, with Multiple Points option highlighted.
```

1. Open the Selenographic tab's dropdown menu and select Multiple Points ([Figure 10](#fig-10)).
2. The multiple selenographic input page will appear ([Figure 11](#fig-11)). Click "LOAD FILE" and choose
   a `.csv` file containing the coordinates.

```{eval-rst}
.. figure:: ../../images/user_guide/initial_multiseleno_page.png
   :name: fig-11
   :align: center
   :alt: Multiple selenographic points input page

   Multiple selenographic points input page.
```

The file should have one point per line, with six comma-separated values in the following order:

```csv
sun_moon_distance_au,observer_moon_distance_km,observer_selenographic_latitude_deg,observer_selenographic_longitude_deg,solar_selenographic_longitude_deg,moon_phase_angle_deg
```

### Command-Line Interface (CLI)
Use the `-l` or `--lunar` option:
```sh
lime -l distance_sun_moon,distance_observer_moon,selen_obs_lat,selen_obs_lon,selen_sun_lon,moon_phase_angle
```
#### Example
For a simulation for 0.98 AU of Sun-Moon Distance, 420000 km of Observer-Moon Distance, 20.5° and -30.2° as the
selenographic latitude and longitude of the observer,
69° as the selenographic longitude of the Sun, and 15° as the moon phase angle:
```sh
lime -l 0.98,420000,20.5,-30.2,69,15
```

## Simulation Using Satellite Positions


Selecting the **"Satellite"** tab enables simulation based on **orbiting satellites** ([Figure 12](#fig-12)).  
Users must:
- **Select a satellite** from the available list.
- **Provide a UTC Date & Time**.

> *Note:* Time-series input works the same as in [Geographic Coordinates](#simulation-using-geographic-coordinates).


Selecting the "Satellite" tab enables simulations based on orbiting satellite ([Figure 12](#fig-12)).
Users must:
- **Select a satellite** from the available list.
- **Provide a UTC Date & Time**.
  >  *Note*: Time-series input works the same as in [Geographic Coordinates](#simulation-using-geographic-coordinates).

```{eval-rst}
.. figure:: ../../images/user_guide/satellite_input.png
   :name: fig-12
   :align: center
   :alt: Satellite position input

   Satellite position input.
```

### Adding a New Satellite

It's possible to add user-defined satellites to the list of available satellites.
1. Click the button with a plus (+) sign on the right of the dropdown.
2. The user will be prompted with the *Add Satellite Data* window ([Figure 13](#fig-13)).
3. Click "LOAD FILE" which opens a file selection dialog where users can load Orbit Scenario Files
   (OSF) or Three-Line Element Set files (TLE/3LE).
4. The *Add Satellite Data* window will now display information of the loaded file, and the user
   must fill the last details.
    - **Satellite Name**: Only editable for OSF files. Name that will be displayed in the satellite list.
    - **Start time**: Date when the file validity starts.
    - **End time**: Date when the file validity stops.
5. Click "SAVE SATELLITE DATA" to store the loaded data in the local LIME Toolbox system.

```{eval-rst}
.. figure:: ../../images/user_guide/add_satellite_data_window.png
   :name: fig-13
   :align: center
   :alt: Add Satellite Data window

   Add Satellite Data window.
```


### Command-Line Interface (CLI)
Use the `-s` or `--satellite` option:
```sh
lime -s sat_name,datetime_isoformat
```
#### Example
To perform a simulation for the satellite PROVA-B at the date and time of
2020-01-20 02:00:00h:
```sh
lime -s PROBA-V,2020-01-20T02:00:00
```
As in [Geographic Coordinates](#simulation-using-geographic-coordinates), one can use time-series input:
```sh
lime -s PROBA-V -t timeseries.txt
```

It's not possible to add a user-defined satellite throught the CLI at the moment.

## Simulation output 

The simulation results are displayed below the input section.
Users can select one of the following button to run simulations and view results:
- **"IRRADIANCE"** – Displays irradiance results.
- **"REFLECTANCE"** – Shows reflectance results.
- **"DOLP"** – Visualises the Degree of Linear Polarisation (DoLP).
- **"AOLP"** – Visualises the Angle of Linear Polarisation (AoLP).

These buttons are positioned between the input and the output, as shown on top of [Figure 14](#fig-14).

```{eval-rst}
.. figure:: ../../images/user_guide/simulation_output.png
   :name: fig-14
   :align: center
   :alt: Simulation output

   Simulation output.
```

Results are plotted with:
- **X-axis** → Wavelengths.
- **Y-axis** → Computed values.

### Exporting Simulation Results

Users can export the results in multiple formats:

#### Graphical Export (Image/PDF)
- Click "Export Graph" (top-center) to save the graph as an image or PDF.
- A file system dialog will prompt the user to select the location, filename, and format.
- Supported formats include JPG, PNG, PDF, and other common image formats.

#### Data Export (CSV File)
- Click "Export CSV" (top-center) to save the simulation data as a comma-separated values (CSV) file.
- The user will be prompted to choose the location and filename.

#### NetCDF Export
- Go to the action menu bar and navigat to "File → Save as a netCDF file" to save results as a netCDF file.
> **Why use netCDF?**
> - Unlike CSV, netCDF allows LIME Toolbox to reload previous simulations for visualization.
> - This format enables efficient data storage and retrieval.

<!-- TODO: Add "How to load simulation/comparison netCDF file into Toolbox" in format section -->

### Command-Line Interface (CLI) Export Options
In the CLI, results are only accessible by exporting them to data files.
To do this, users must utilize the `-o` or `--output` option, with the argument varying
based on the desired output type.

#### Graphical Export (Image/PDF)
Specify the image type, paths and filenames of each graph:
```sh
-o graph,(pdf|jpg|png|svg),reflectance_path,irradiance_path,dolp_path,aolp_path
```
For example:
```sh
-o graph,jpg,reflectance.jpg,irradiance.jpg,dolp.jpg,aolp.jpg
```

#### Data Export (CSV File)
Specify paths and names of each file:
```sh
-o csv,reflectance_path,irradiance_path,dolp_path,aolp_path,integrated_irradiance_path
```
For example:
```sh
-o csv,reflectance.csv,irradiance.csv,dolp.csv,aolp.csv,integrated_irradiance.csv
```

#### NetCDF Export
Specify the netCDF file path:
```sh
-o nc,output_path
```
For example:
```sh
-o nc,output.nc
```

## Integrated Irradiance

LIME Toolbox integrates the calculated irradiance over the selected spectral response function (SRF).
These values appear in the "Signal" tab ([Figure 15](#fig-15)), showing one value per SRF channel,
and can be exported as CSV.

```{eval-rst}
.. figure:: ../../images/user_guide/signal_tab.png
   :name: fig-15
   :align: center
   :alt: Signal tab showing some results for the CIMEL SRF

   Signal tab showing some results for the CIMEL SRF.
```

### Spectral Response Functions (SRF)

Users can visualise, load, and switch between different SRFs in the "SRF" tab
([Figure 16](#fig-16)). This tab presents:
- A graph of the currently selected SRF, where:
  - X-axis represents wavelengths.
  - Y-axis represents the spectral response.
- Two vertical black lines marking the lower and upper simulation limits of LIME Toolbox.

```{eval-rst}
.. figure:: ../../images/user_guide/srf_tab.png
   :name: fig-16
   :align: center
   :alt: SRF tab showing CIMEL SRF

   SRF tab showing CIMEL SRF.
```

By default, LIME Toolbox includes a default SRF that encompasses the entire LIME spectrum, named "Default".

#### Loading a New SRF
1. Click "LOAD" (top-right corner in [Figure 16](#fig-16)).  
2. Select a netCDF SRF file in GLOD format (explained in the [File Formats](./formats.md) section).
3. Once loaded, use the dropdown menu next to the "LOAD" button to switch between SRFs.

This can also be done in the CLI using the `-f` or `--srf` option followed by the SRF path:
```sh
-f srf_path
```

#### Exporting SRF Data
- Graphs can be exported as images or PDFs.
- SRF data can be exported as CSV files.
