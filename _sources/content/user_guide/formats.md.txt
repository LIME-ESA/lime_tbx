# File Formats

LIME Toolbox reads various types of data from netCDF files, each following a specific format.
These formats adhere as closely as possible to the
[GSICS Lunar Observation Dataset (GLOD) format][glod-format-pdf] standards.

[glod-format-pdf]: https://gsics.atmos.umd.edu/pub/Development/LunarWorkArea/GSICS_ROLO_HighLevDescript_IODefinition.pdf

## Spectral Response Function (SRF)

Spectral Response Function files are structured with two dimensions:
- `channel`: A coordinate and dimension, where each value represents a different spectral channel.
- `sample`: A dimension without coordinates, which allows multiple wavelengths to be associated
  with each channel.  
  > *(This dimension may have different names depending on the dataset.)*

These dimensions organise the mandatory data variables described in [Table 2](#tab-2).

<center id='tab-2'>

| **Variable** | **Dimensions** | **dtype** | **Description** |
|:--------:|:-----------|:------|:------------|
| `channel_id` | `channel` | str | Name of the channel |
| `wavelength` | `sample`, `channel` | float | Wavelengths present in each channel |
| `srf` | `sample`, `channel` | float | Spectral response of each wavelength per channel |

<i>Table 2</i>: Mandatory variables of the Spectral Response Function netCDF file format.

</center>

*Note*: The variable `wavelength` must contain the attribute `units`, which value must be the
variable unit symbol by the IS. For example, "nm".

## Lunar Observation

Lunar Observation files format is a subset of the GLOD format for lunar observations.
It's structured using three dimensions:
- `date`: A coordinate and dimension representing the time of the lunar observation.
  - Stored as a double precision float. Units: seconds since EPOCH.
- `chan`: A dimension without coordinates, where each value represents a different spectral channel.
- `sat_xyz`: A dimension without coordinates, representing the three spatial coordinates (x, y, z).
  - This dimension must have a fixed length of three.

These dimensions structure the required data variables outlined in [Table 3](#tab-3).

<center id='tab-3'>

| **Variable** | **Dimensions** | **dtype** | **Description** |
|:--------:|:-----------|:------|:------------|
| `channel_name` | `chan` | str | Name of the spectral channel |
| `irr_obs` | `chan` | float | Observed lunar irradiance per spectral channel |
| `sat_pos` | `sat_xyz` | float | Coordinates of the observer in `sat_pos_ref` reference frame |
| `sat_pos_ref` | None | str | Reference frame of the satellite position |

<i>Table 3</i>: Mandatory variables of the Lunar Observation netCDF file format.

</center>

*Note*: The variable `sat_pos` must contain the attribute `units`, which value must be the
variable unit symbol by the IS. For example, "km".

In addition to these variables, the file must include the attribute:
- `data_source`: Specifies the origin of the observation data.

### Lunar Observation Format Extension

Lunar Observation files should follow the subset of the GLOD format previously described.
However, if the `sat_pos` variable is not available, LIME Toolbox will instead look for selenographic coordinates.
In this case, the selenographic variables must follow the schema in [Table 4](#tab-4), where all variables must
have `date` as their dimension.

<center id='tab-4'>

| **Variable Name** | **Mandatory** | **Description** |
|-------------------|---------------|-----------------|
| `distance_sun_moon` | Yes | Distance between the Sun and the Moon (AU) |
| `sun_sel_lon` | Yes | Selenographic longitude of the Sun (radians) |
| `distance_sat_moon` | Yes | Distance between the satellite and the Moon (km) |
| `sat_sel_lon` | Yes | Selenographic longitude of the satellite (degrees) |
| `sat_sel_lat` | Yes | Selenographic latitude of the satellite (degrees) |
| `phase_angle` | Yes | Moon phase angle (degrees) |
| `sat_name` | No | Satellite name (used for [handling missing data](#handling-missing-data)) |
| `geom_factor` or `geom_const` | No | Geometric constant used to normalize observed irradiance |

<i>Table 4</i>: Data variables in the Lunar Observation netCDF file format representing selenographic coordinates.

</center>

#### Understanding `geom_factor`

The `geom_factor` (or `geom_const`) represents the geometric constant used in irradiance normalization.  
- If a given, the toolbox will use this value to normalize its simulated irradiance.
  ```python
  normalised_irr = irr/geom_factor
  ```

#### Handling Missing Data

If any mandatory variable is missing, LIME Toolbox will automatically compute them using:  
- **SPICE library** → Computes `distance_sun_moon` and `sun_sel_lon`.  
- **SPICE + EO-CFI libraries** → Compute `distance_sat_moon`, `sat_sel_lon`, `sat_sel_lat`, and `phase_angle`.  
  > For these four latter variables, LIME Toolbox requires the satellite name (`sat_name`),  
  > - It must be a string and must match a satellite in the LIME Toolbox satellite list.

#### Optional Attribute: `to_correct_distance`
- If present with a value of `1`, LIME Toolbox will normalise the observation's `irr_obs` value
  using the observation's distances.  
- This is useful when the irradiance is not pre-normalized.

## LIME Toolbox Simulations and Comparisons


LIME Toolbox allows exporting simulations and comparisons to netCDF format files, which can be reloaded
within the toolbox for future visualization and analysis.

These files follow a structure similar to [Lunar Observation](#lunar-observation) files but contain
multiple observation timestamps instead of just one. Additionally, they include more detailed data such
as simulated irradiance, reflectance, and spectral values.

Both simulation and comparison files share a similar format but with distinct differences.
This format is referred to as the LIME Toolbox netCDF format, previously known in documentation
as LIME-GLOD (LGLOD) format.

### File Attributes

LIME Toolbox netCDF format files include multiple attributes based on the GLOD format. The key attributes are:
- **data_source**: Indicates the origin of the data (LIME Toolbox).
- **reference_model**: Specifies the LIME coefficient version used.
- **not_default_srf**:  
  - `0`: Default LIME Toolbox SRF was used.  
  - `1`: A user-defined custom SRF was used.
- **spectrum_name**: Name of the spectrum used for reflectance interpolation.
- **is_comparison**:  
  - `0`: Indicates a **simulation file**.  
  - `1`: Indicates a **comparison file**.
- **skipped_uncertainties**:  
  - `0`: Uncertainty computations were **performed**.  
  - `1`: Uncertainty computations were **skipped**.
- **polarisation_spectrum_name**: Name of the spectrum used in polarization interpolation (*only present in simulation files*).

### File Dimensions

All LIME Toolbox netCDF files are structured using four core dimensions:
- **chan**: Number of spectral channels in the data.
- **date**: Number of timestamps present.
- **number_obs**: Number of observation positions where simulations exist for at least one channel.
- **sat_xyz**: Fixed length of `3`, representing the `(x, y, z)` coordinates of the satellite.

Simulations files contain two additional dimensions:
- **wlens**: Fixed length of `2151`, representing the number of wavelengths in full-spectrum simulations.
- **wlens_cimel**: Fixed length of `6`, representing the number of CIMEL wavelengths.

### File Variables

#### Common Data Variables

LIME Toolbox netCDF files share several core variables based on the GLOD format, described in [Table 5](#tab-5).

<center id='tab-5'>

```{eval-rst}
.. tabularcolumns:: |p{3.1cm}|p{2cm}|p{1cm}|p{7.5cm}|
```
| **Variable** | **Dimensions** | **dtype** | **Description** |
|------------|------------|--------|-------------|
| `date` | `date` | float64 | Time of lunar observation, seconds since epoch. |
| `outside_mpa_range` | `number_obs` | int8 | 1 if the observation is outside the Moon phase angle valid range, 0 if inside. |
| `mpa` | `number_obs` | float64 | Moon phase angle in degrees. |
| `channel_name` | `chan` | str | Channel/Sensor band identifier. |
| `sat_pos` | `number_obs`, `sat_xyz` | float64 | Satellite position in (x, y, z) coordinates for `sat_pos_ref` frame. |
| `sat_pos_ref` | `number_obs` | str | Reference frame of the satellite position. |
| `sat_name` | None | str | Name of the satellite (or empty if it wasn’t a satellite measure). |
| `irr_obs` | `number_obs`, `chan` | float64 | Simulated integrated lunar irradiance for each channel. |
| `irr_obs_unc` | `number_obs`, `chan` | float64 | Uncertainties of the simulated integrated lunar irradiance for each channel. |

<i>Table 5</i>: Common data variables in all LIME Toolbox netCDF files.

</center>

#### Comparison Specific Variables

Comparison files contain additional variables that store observed and computed differences, detailed in [Table 6](#tab-6).

<center id='tab-6'>

```{eval-rst}
.. tabularcolumns:: |p{2.5cm}|p{2cm}|p{1cm}|p{8cm}|
```
| **Variable** | **Dimensions** | **dtype** | **Description** |
|------------|------------|--------|-------------|
| `irr_comp` | `number_obs`, `chan` | float64 | Integrated lunar irradiance for each channel observed with the instrument, obtained from the user defined observation files. |
| `irr_comp_unc` | `number_obs`, `chan` | float64 | Uncertainties of the integrated lunar irradiance for each channel observed with the instrument, obtained from the user defined observation files. |
| `irr_diff` | `number_obs`, `chan` | float64 | Lunar irradiance comparison difference for each channel. |
| `irr_diff_unc` | `number_obs`, `chan` | float64 | Uncertainties of the lunar irradiance comparison difference for each channel. |
| `perc_diff` | `number_obs`, `chan` | float64 | Percentage difference in the lunar irradiance comparison for each channel. |
| `perc_diff_unc` | `number_obs`, `chan` | float64 | Uncertainties of the percentage difference in the lunar irradiance comparison for each channel. |
| `mrd` | `chan` | float64 | Mean relative difference. |
| `mard` | `chan` | float64 | Mean absolute relative difference. |
| `mpd` | `chan` | float64 | Mean percentage difference. |
| `std_mrd` | `chan` | float64 | Standard deviation of the mean relative difference. |
| `number_samples` | `chan` | float64 | Number of comparisons for each channel. |


<i>Table 6</i>: Data variables in LIME Toolbox Comparison netCDF files.

</center>

#### Simulation Specific Variables 

Simulation files contain additional variables that store spectral data, described in [Table 7](#tab-7).

<center id='tab-7'>

```{eval-rst}
.. tabularcolumns:: |p{3.3cm}|p{2cm}|p{1cm}|p{8cm}|
```
| **Variable** | **Dimensions** | **dtype** | **Description** |
|------------|------------|--------|-------------|
| `wlens` | `wlens` | float64 | Wavelengths for `irr_spectrum`, `refl_spectrum`, and `polar_spectrum`. |
| `irr_spectrum` | `number_obs`, `wlens` | float64 | Simulated lunar irradiance per wavelength. |
| `irr_spectrum_unc` | `number_obs`, `wlens` | float64 | Uncertainties for the simulated lunar irradiance per wavelength. |
| `refl_spectrum` | `number_obs`, `wlens` | float64 | Simulated lunar reflectance per wavelength. |
| `refl_spectrum_unc` | `number_obs`, `wlens` | float64 | Uncertainties for the simulated lunar reflectance per wavelength. |
| `polar_spectrum` | `number_obs`, `wlens` | float64 | Simulated lunar degree of polarization per wavelength. |
| `polar_spectrum_unc` | `number_obs`, `wlens` | float64 | Uncertainties for the simulated lunar degree of polarization per wavelength. |
| `cimel_wlens` | `wlens_cimel` | float64 | CIMEL wavelengths. |
| `irr_cimel` | `number_obs`, `wlens_cimel` | float64 | Simulated lunar irradiance for the CIMEL wavelengths. |
| `irr_cimel_unc` | `number_obs`, `wlens_cimel` | float64 | Uncertainties for the simulated lunar irradiance for the CIMEL wavelengths. |
| `refl_cimel` | `number_obs`, `wlens_cimel` | float64 | Simulated lunar reflectance for the CIMEL wavelengths. |
| `refl_cimel_unc` | `number_obs`, `wlens_cimel` | float64 | Uncertainties for the simulated lunar reflectance for the CIMEL wavelengths. |
| `polar_cimel` | `number_obs`, `wlens_cimel` | float64 | Simulated lunar degree of polarization for the CIMEL wavelengths. |
| `polar_cimel_unc` | `number_obs`, `wlens_cimel` | float64 | Uncertainties for the simulated lunar degree of polarization for the CIMEL wavelengths. |


<i>Table 7</i>: Data variables in LIME Toolbox Simulation netCDF files.

</center>
