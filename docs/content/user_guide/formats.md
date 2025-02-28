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

These dimensions organise the mandatory data variables described in [Table 1](#tab-1).

<center id='tab-1'>

| **Variable** | **Dimensions** | **dtype** | **Description** |
|:--------:|:-----------|:------|:------------|
| `channel_id` | `channel` | str | Name of the channel |
| `wavelength` | `sample`, `channel` | float | Wavelengths present in each channel |
| `srf` | `sample`, `channel` | float | Spectral response of each wavelength per channel |

<i>Table 1</i>: Mandatory variables of the Spectral Response Function netCDF file format.

</center>

*Note*: The variable `wavelength` must contain the attribute `units`, which value must be the
variable unit symbol by the IS. For example "nm".

## Lunar Observation

Lunar Observation files format is a subset of the GLOD format for lunar observations.
It's structured using three dimensions:
- `date`: A coordinate and dimension representing the time of the lunar observation.
  - Stored as a double precission float. Units: seconds since EPOCH.
- `chan`: A dimension without coordinates, where each value represents a different spectral channel.
- `sat_xyz`: A dimension without coordinates, representing the three spatial coordinates (x, y, z).
  - This dimension must have a fixed length of three.

<center id='tab-2'>

| **Variable** | **Dimensions** | **dtype** | **Description** |
|:--------:|:-----------|:------|:------------|
| `channel_name` | `chan` | str | Name of the spectral channel |
| `irr_obs` | `chan` | float | Observed lunar irradiance per spectral channel |
| `sat_pos` | `sat_xyz` | float | Coordinates of the observer in `sat_pos_ref` reference frame |
| `sat_pos_ref` | None | str | Reference frame of the satellite position |

<i>Table 2</i>: Mandatory variables of the Lunar Observation netCDF file format.

</center>

*Note*: The variable `sat_pos` must contain the attribute `units`, which value must be the
variable unit symbol by the IS. For example km".

In addition to these variables, the file must include the attribute:
- `data_source`: Specifies the origin of the observation data.

### Lunar Observation Format Extension

Lunar Observation files must follow the subset of the GLOD format previously described.  

However, if the `sat_pos` variable is not available, LIME Toolbox will instead look for selenographic coordinates.
In this case, the selenographic variables must follow the schema in [Table 3](#tab-3), where all variables must
have `date` as their dimension.

<center id='tab-3'>

| **Variable Name** | **Mandatory** | **Description** |
|-------------------|---------------|-----------------|
| `distance_sun_moon` | Yes | Distance between the Sun and the Moon (AU) |
| `sun_sel_lon` | Yes | Selenographic longitude of the Sun (radians) |
| `distance_sat_moon` | Yes | Distance between the satellite and the Moon (km) |
| `sat_sel_lon` | Yes | Selenographic longitude of the satellite (degrees) |
| `sat_sel_lat` | Yes | Selenographic latitude of the satellite (degrees) |
| `phase_angle` | Yes | Moon phase angle (degrees) |
| `sat_name` | No | Satellite name (used if any mandatory variable is missing) |
| `geom_factor` or `geom_const` | No | Geometric constant used to normalize observed irradiance |

<i>Table 3</i>: Data variables in the Lunar Observation netCDF file format representing selenographic coordinates.

</center>

#### Understanding `geom_factor`

The `geom_factor` (or `geom_const`) represents the geometric constant used in irradiance normalization.  
- If a given, the toolbox will use this value to normalize its simulated irradiance.
  - `normalised_irr` = `irr`/`geom_factor`

#### Handling Missing Data

If any mandatory variable is missing, LIME Toolbox will automatically compute them using:  
- **SPICE library** → Computes `distance_sun_moon` and `sun_sel_lon`.  
- **SPICE + EOCFI libraries** → Compute `distance_sat_moon`, `sat_sel_lon`, `sat_sel_lat`, and `phase_angle`.  
  > For these four latter variables, LIME Toolbox requires the satellite name (`sat_name`),  
  > - It must be a string and must match a satellite in the LIME Toolbox satellite list.

#### Optional Attribute: `to_correct_distance`
- If present with a value of `1`, LIME Toolbox will normalise the observation's `irr_obs` value
  using the observation's distances.  
- This is useful when the irradiance is not pre-normalized.

## LIME Toolbox Simulations and Comparisons

### LIME Toolbox Simulation

### LIME Toolbox Comparison
