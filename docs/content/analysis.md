# Analysis

The LIME Toolbox is being developed as part of ESA's project: "Improving the Lunar Irradiance Model of ESA".
Throughout its planning, development, and implementation, ESA and the LIME Team have collaboratively proposed
and refined various objectives, features and constraints. This analysis breaks down these elements to ensure
a clear, structured approach, aligning design and implementation with the project's objectives.

## Main Features

The main objectives and expected features are described in a similar fashion to user stories, which are short
descriptions written from a user point of view that focus on what the user needs.


|  #  |  User Story  |
|:---:|:-------------|
| 1.1 | Simulate Single Observation from Earth |
| 1.2 | Simulate Time Series of Observations from Earth |
| 2 | Simulate Single Observation with Custom Selenographic Coordinates |
| 3.1 | Simulate Single Observation for a Satellite |
| 3.2 | Simulate Time Series of Observations for a Satellite |
| 3.3 | Simulate Satellite Observation for a User-Defined Satellite |
| 4.1 | Input a User-Defined Spectral Response Function |
| 4.2 | Visualise User-Defined Spectral Response Function |
| 5 | Compare LIME Against Data from a Remote-Sensing Instrument |
| 6 | Export Simulation/Comparison Data to a File |
| 7.1 | Download Updated Coefficients |
| 7.2 | Change Coefficients Version in Use |


## Requirements Elicitation

A set of requirements has been obtained from the accepted LIME functionalities and restrictions. This
set is organised into four categories based on the user story they appeal to. These categories are:
1. **Simulations**: Requirements related to User Stories 1, 2, 3 and 4.
2. **Comparisons**: Requirements related to mainly User Story 5, but also 4.
3. **Output**: Requirements related to User Story 6.
4. **System**: Requirements related to User Story 7 and also the requirements related to the general
   functioning of the toolbox, not related to any User Story in particular.

Each set is subsequently split into two subsets: functional requirements (FR) and non-functional requirements (NFR).
Functional requirements specify the essential tasks a system must perform to fulfill user and business
objectives, akin to a feature checklist. In contrast, non-functional requirements outline the system's operational standards,
addressing elements such as performance, security, and user experience. Essentially, functional requirements
answer "what must be done" whereas non-functional requirements define "how it should be done".

Take into account that the requirements listed in this documentation may not fully align with
those in any previous official LIME documentation, as they are subject to changes and additions.

### 1. Simulations

#### Functional Requirements (FR)

- **FR101**: Allow users to simulate lunar observations for any observer's position around the Earth and at any time.
- **FR102**: Allow users to simulate lunar observation for any observer/solar selenographic latitude and longitude
  (thus bypassing the need for their computation from the position/time of the observer).
- **FR103**: Allow users to load user-defined spectral response functions (SRF).
- **FR104**: Allow users to simulate lunar observation for a single observation or for a time series of observations,
  for **FR101** and **FR105**.
- **FR105**: The user must be able to simulate lunar observations for an ESA satellite.
- **FR106**: Allow users to choose which SRF are simulations running with (**FR103**).
- **FR107**: Allow users to simulate a series of lunar observations, where multiple paremeters can vary simultaneosuly, not only the time.
- **FR108**: Allow users to simulate lunar observation using a user-defined satellite.

#### Non Functional Requirements (NFR)

- **NFR101**: The simulation input of a single lunar observation (**FR104**) must be introduced via the GUI.
- **NFR102**: The simulation input for a time series of lunar observations (**FR104**) must be done via an input file.
- **NFR103**: In order to simulate lunar observations from a satellite position (**FR105**) the user must provide an
  orbital scenario file in EO-CFI compatible format.
- **NFR104**: The ESA satellites available for selection must include ENVISAT, Proba-V, S2, S3, FLEX.
- **NFR105**: The user defined SRF (**FR103**) must be defined via a user generated SRF file.
- **NFR106**: The user should be able to perform the simulation via command line using parameters or input files.
- **NFR107**: Simulations must run with a SRF, either a default one or a user-defined SRF.
- **NFR108**: The TBX must be flexible in terms of wavelengths. It should accept different amounts of coefficients and process them differently.
  - **NFR108-A** : The TBX must accept coefficients that also include data for the 1088 CIMEL photometer's 2130 nanometer band.
  - **NFR108-B** : The TBX must accept coefficients made for a different response function, for the wavelengths specified in the coefficients.
- **NFR109**: The ESA satellites available for selection must include METOP first generation: METOP-A, METOP-B and METOP-C.
- **NFR110**: The simulation input for multi-parameter series of lunar observations (**FR107**) must be provided via an input file.
- **NFR111**: The satellites for **FR108** must be defined using Orbit Scenario Files (OSF) or Three-Line
  Element Set (TLE/3LE) files.
- **NFR112**: The simulation input for multi-parameter series of lunar observations (**FR107**) must allow multiple entries of selenographic coordinates, where all selenographic coordinates can vary independently.

### 2. Comparisons

#### Functional Requirements (FR)
- **FR201**: Allow performing comparisons of lunar observations from a remote sensing instrument to the LIME model output.
- **FR202**: Allow exporting plots.
- **FR203**: The TBX must allow the user to visualize the comparison for all spectral channels simultaneously. 

#### Non Functional Requirements (NFR)
- **NFR201**: The remote sensing instrument observations (**FR201**) must be pre-stored in a GLOD format file.
- **NFR202**: The comparison plots shall provide: relative differences between measured and modeled lunar
  irradiance/reflectance vs. time and vs. lunar phase angle.
- **NFR203**: The comparison plots shall display statistical indicators: mean relative difference, standard
  deviation of the mean relative difference, temporal trend if applicable, number of comparison samples, etc.
- **NFR204**: The exported plots must be in .jpg or .pdf format.
- **NFR205**: The user should be able to perform the simulations via command line using parameters or input files.
- **NFR206**: Users must be informed that they have to load a user-defined SRF to run the comparison's simulations with.
- **NFR207**: The comparison of **FR203** must be plotted in the same graph, with x-axis being the wavelength, and y-axis being comparisons & irradiance.
  The plots should show the temporal average of the relative difference between the model and the observation.
- **NFR208**: Switching between comparison graphs should be fast, there shouldn’t be a loading time of longer than 2
  seconds when switching and drawing them after the comparison shouldn’t take more than 20 seconds.

### 3. Output

#### Functional Requirements (FR)
- **FR301**: The LIME TBX shall output simulated lunar disk irradiance or reflectance.
- **FR302**: The LIME TBX shall output the simulated lunar disk degree of polarization.
- **FR303**: The LIME TBX shall output simulated lunar irradiance or reflectance associated uncertainty.
- **FR304**: The LIME TBX shall output the simulated lunar degree of polarization’s associated uncertainty.
- **FR305**: Allow the visualization of the user defined spectral response used for the spectral integration
  of the LIME output into a sensor spectral band.
- **FR306**: The LIME TBX shall output the simulated lunar disk angle of linear polarization (AoLP).
- **FR307**: The LIME TBX shall output the simulated lunar angle of linear polarization’s associated uncertainty.

#### Non Functional Requirements (NFR)
- **NFR301**: The simulated lunar disk irradiance or reflectance (**FR301**) must be in the spectral range of 400 nm to 2500 nm.
- **NFR302**: The simulated lunar disk degree of polarization (**FR302**) must be in the spectral range of 400 nm to 2500 nm.
- **NFR303**: The LIME simulated output shall be available to be exported to GLOD format files.
- **NFR304**: The LIME version number shall be visible on all outputs (plots/files) of the TBX.
- **NFR305**: Geometry information (coordinates, angles, etc.) and timestamps must be included in NetCDF and CSV output files.
- **NFR306**: Calculation of uncertainties (**FR303**, **FR304**, **FR307**) must be performed as fast as possible.
- **NFR307**: The simulated lunar disk angle of linear polarization (**FR306**) must be in the spectral range of 400 nm to 2500 nm.

### 4. System

#### Functional Requirements (FR)
- **FR401**: The LIME TBX must be able to perform automatic updates of the LIME coefficients.
- **FR402**: The user shall be able to select past LIME coefficients with whom perform the calculations.

#### Non Functional Requirements (NFR)
- **NFR401**: The LIME coefficients of the automatic updates (**FR401**) must be stored on a dedicated repository.
- **NFR402**: The LIME TBX shall read the database of lunar observations formatted in GLOD format.
- **NFR403**: The LIME TBX shall use the EO-CFI as orbit propagator and to derive satellite orbital positions.
- **NFR404**: Be to the largest extent platform/operating system independent.
- **NFR405**: Run at least under Windows / mac OS / Linux operating systems.
- **NFR406**: Be to the largest extent a self-installing SW package.
- **NFR407**: The LIME TBX shall be to the largest extent developed in Python.
- **NFR408**: The LIME TBX code shall be available on a password protected web repository allowing versioning of
  the software (e.g., GitHub).
- **NFR409**: The TBX must be compiled and packaged through Docker or similar.
  - **NFR409-A**: Automated compilation and packaging for Windows.
  - **NFR409-B**: Automated compilation and packaging for Linux.
  - **NFR409-C**: Automated compilation and packaging for Mac.
- **NFR410**: The TBX compilation must be automated as much as possible, ideally using GitLab CI.
- **NFR411**: Migrate the project to GitHub
  - **NFR411-A**: During development, mirror the repository to GitHub.
  - **NFR411-B**: Migrate the issues to GitHub. GitLab issues must be used during development by the LIME team to
    add issues, and GitHub issues when finished.
  - **NFR411-C**: Migrate the CI pipeline to GitHub.
  - **NFR411-D**: Migrate the code documentation to GitHub. 
- **NFR412**: Code modules should be as independent as possible so it could be possible to add a “choose model” option
  in the future without requiring too much development.
- **NFR413**: Keep EO-CFI version updated to the latest release.
- **NFR414**: The TBX should be able to simulate satellite positions for all satellites with OSF files in the EOP-CFI server.
- **NFR415**: The TBX code will be under LGPL license, in a repository with public access.
