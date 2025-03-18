# Design

## Software architecture

All the toolbox features and functionalities are encompassed in the `lime_tbx` Python package.
This package is structured following a layered architecture with four main layers: `presentation`,
`application`, `business` and `persistence`, and a helper one: `common`.


<figure align="center" id="fig-1">
  <img src="../uml/layer_architecture.png" alt="lime_tbx package architecture"/>
  <figcaption><i>Figure 1</i>: lime_tbx package architecture.</figcaption>
</figure>


The architecture is described in UML in [Figure 1](#fig-1). We can observe the following subpackages:
- **lime_tbx/presentation**: Manages user interaction through graphical and command-line interfaces.
  - **lime_tbx/presentation/gui**: Provides graphical user interface components for visual interaction,
  including windows, forms, and data visualizations.
  - **lime_tbx/presentation/cli**: Handles command-line interactions, including parsing
  user commands, displaying outputs, and handling user input.
- **lime_tbx/application**: Defines application-level workflows and orchestrates operations
  across different components. It encapsulates high-level logic of user stories.
  - **lime_tbx/application/simulation**: Manages simulation and comparison workflows. 
  - **lime_tbx/application/coefficients**: Provides logic for managing and downloading LIME coefficients.
  - **lime_tbx/application/filedata**: Handles use cases related to exporting and loading data files. 
- **lime_tbx/business**: Implements the core domain logic and numerical algorithms used by the toolbox.
  - **lime_tbx/business/lime_algorithms**: Implements the main LIME algorithms, enabling the calculation
  of irradiance, reflectance and polarisation.
  - **lime_tbx/business/interpolation**: Provides numerical interpolation methods.
  - **lime_tbx/business/spectral_integration**: Handles spectral integration calculations.
  - **lime_tbx/business/eocfi_adapter**: Interfaces with the EO-CFI library to compute satellite positions.
  - **lime_tbx/business/spice_adapter**: Provides a simplified interface with NASA's SPICE toolkit, for
  precise spacecraft geometry and ephemerides calculations.
- **lime_tbx/common**: Contains shared utilities, data structures, and configurations used across the application.
  - **lime_tbx/common/datatypes**: Defines common data structures, custom classes, and typed objects for
  consistent data representation.
  - **lime_tbx/common/constants**: Stores global constants, physical values, and predefined
  configurations.
  - **lime_tbx/common/logger**: Implements a centralized logging system for tracking application events,
  errors, and debugging information.
  - **lime_tbx/common/templates**: Stores datatype templates needed for compatibility with the `obsarray` library.
- **lime_tbx/persistence**: Handles data storage and retrieval mechanisms.
  - **lime_tbx/persistence/local_storage**: Manages the retrieval and storage of application data on the
  local filesystem, including logs and configurations.
