# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

[//]: # "## [unreleased] - yyyy-mm-dd"

## [1.1.0] - 2025-04-21

### Added

- GUI more user friendly: Simulation input takes less window space.
- The ESA satellites available for selection now include METOP first generation: METOP-A, METOP-B and METOP-C. (**NFR109**)
- Allow adding satellites through 3LE/TLE or OSF files through the GUI. (To fully comply with **NFR109**)
- Allow to visualize the comparison for all spectral channels simultaneously. (**FR203**) This is done through two new comparison options:
  - By wavelength (mean): Show the temporal average of all the channel/wavelengths in a plot, having the wavelengths on the x-axis and the irradiance and differences on the y-axis. CLI name: CHANNEL_MEAN
  - By wavelength (boxplot): Show the statistics of all the channel/wavelengths in a plot, showing one boxplot per channel. The boxplot are shown having the wavelengths on the x-axis and the irradiance and differences (the box plot) on the y-axis. CLI name: CHANNEL
- Improve speed of switching between comparison graphs. (**NFR208**)
- Geometry information (coordinates, angles, etc.) and timestamps (if available) are now included in all NetCDF and CSV output files. (**NFR305**)
- The TBX accepts coefficients that also include data for the 1088 CIMEL photometer's 2130 nanometer band. (**NFR108-A**)
- Automated Build & Packaging (**NFR409**):
  - Linux: Fully automated build through Docker.
  - Windows: Partially automated build through Docker for Windows. All automated except first step:
    compiling EO-CFI C Code, which isn't mandatory for each build.
- Timeseries input file now also accepts regular format timestamps, not only CSV.
- Added METOP-A, METOP-B and METOP-C data (TLE/3LE). METOP-B and METOP-C will need to be periodically updated.
- Updated TLE/3LE data for PLEIADES 1A, PLEIADES 1B and PROBA-V, now covering the period from their launch until 2025-04-14.
- Updated EO-CFI library dependencies to version 4.28.

### Changed

- Drop support for Python 3.8.
- Provide support for Python 3.11 and 3.12.
- Updated library dependencies and versions.
- Refactored main Python package grouping subpackages in layer architecture based packages.

### Fixed

- Removed forced conversion of error correlation matrices to `float32`. ASD error correlation matrices in `ds_ASD.nc`
are `float64`, now they are kept as `float64`, which ensures they are positive-definite as the conversion to `float32` was
slightly modifying the values making them not be positive-definite. This eliminates the
overhead of computing the closest positive-definite matrix, achieving a 2.5× speedup in uncertainty
calculations compared to v1.0.3. (**NFR306**)
- Optimized the ASD wavelengths error correlation matrix by converting it from `float64` to `float32` while preserving
  its positive semi-definite (PSD) property. This enhancement achieves a mean relative difference of less than 0.001%,
  ensuring numerical stability. Additionally, this change provides a 2.5× speedup, contributing to an overall 6.25× improvement
  in uncertainty calculations compared to v1.0.3. (**NFR306**)
- Updated LIME reflectance equation to align with the LIME ATBD. Observer's selenographic longitude and latitude were
  previously swapped in the implementation. Corresponding coefficients were also swapped, resulting in correct outputs
  despite the mismatch. Both the equation and the coefficients are now fully consistent with the ATBD.
  - Replaced old coefficients file `20231120_v01` with corrected version `20231120_v02`. Outputs of the current toolbox version
    using the new coefficients remain unchanged compared to the previous toolbox version (1.0.3) using the old coefficients.
- Fixed minor bugs

## [1.0.3] - 2024-01-25

Initial version that serves as the baseline for tracking changes in the change log.


[unreleased]: https://github.com/LIME-ESA/lime_tbx/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/LIME-ESA/lime_tbx/compare/v1.0.3...v1.1.0
[1.0.3]: https://github.com/LIME-ESA/lime_tbx/releases/tag/v1.0.3
