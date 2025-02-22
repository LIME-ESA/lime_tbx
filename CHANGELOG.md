# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

[//]: # "## [1.1.0] - yyyy-mm-dd"

## [unreleased] - yyyy-mm-dd

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
    compiling EOCFI C Code, which isn't mandatory for each build.
- Timeseries input file now also accepts regular format timestamps, not only CSV.
- Added METOP-A, METOP-B and METOP-C data (TLE/3LE). METOP-B and METOP-C will need to be periodically updated.
- Updated TLE/3LE data for PLEIADES 1A, PLEIADES 1B and PROBA-V, now covering the period from their launch until 2024-12-06.

### Changed

- Drop support for Python 3.8.
- Provide support for Python 3.11 and 3.12.
- Updated library dependencies and versions.

### Fixed

- Removed conversion of error correlation matrices to `np.float32`, keeping them as `np.float64`.
This ensures they are positive-definite, eliminating the overhead of computing the closest
positive-definite matrix, achieving a 2.5× speedup in uncertainty calculations compared to v1.0.3. (**NFR306**)
- Fixed minor bugs

## [1.0.3] - 2024-01-25

Initial version that serves as the baseline for tracking changes in the change log.


[unreleased]: https://github.com/LIME-ESA/lime_tbx/compare/v1.0.3...HEAD
[//]: # "[1.1.0]: https://github.com/LIME-ESA/lime_tbx/compare/v1.0.3...v1.1.0"
[1.0.3]: https://github.com/LIME-ESA/lime_tbx/releases/tag/v1.0.3
