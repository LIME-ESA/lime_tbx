# Analysis

The LIME Toolbox is being developed as part of ESA's project: "Improving the Lunar Irradiance Model of ESA".
Throughout its planning, development, and implementation, ESA and the LIME Team have collaboratively proposed
and refined various features and constraints. This analysis breaks down these elements to ensure a clear,
structured approach, aligning design and implementation with the project's objectives.

## Use cases

| ID | Name | Description |
|:--:|:----:|:-----------:|
| 1.1 | Simulate  From Earth |


## Requirements Elicitation

A set of requirements has been obtained from the accepted LIME functionalities and restrictions. This
set is organised into four categories based on the use case they are related to. These categories are:
1. Simulations
2. Comparisons
3. Output
4. System

Each set is subsequently split into two subsets: functional requirements (FR) and non-functional requirements (NFR).
Functional requirements specify the essential tasks a system must perform to fulfill user and business
objectives, akin to a feature checklist. In contrast, non-functional requirements outline the system's operational standards,
addressing elements such as performance, security, and user experience. Essentially, functional requirements
answer "what must be done" whereas non-functional requirements define "how it should be done".

### Simulations

#### Functional Requirements (FR)

- **FR101**: Allow users to simulate lunar observations for any observer's position around the Earth and at any time.
- **FR102**: Allow users to simulate lunar observation for any observer/solar selenographic latitude and longitude
  (thus bypassing the need for their computation from the position/time of the observer).
- **FR103**: Allow users to simulate the lunar observations for any user defined instrument spectral response function (SRF).
- **FR104**: Allow users to simulate lunar observation for a single observation or for a time series of observations.
- **FR105**: The user must be able to simulate lunar observations for an ESA satellite.
- **FR106**: Allow users to either choose between default satellite spectral band SRF or select a user defined SRF (**FR103**).

#### Non Functional Requirements (NFR)

- **NFR101**: The simulation input of a single lunar observation (**FR104**) must be introduced via the GUI.
- **NFR102**: The simulation input for a time series of lunar observations (**FR104**) must be done via an input file.
- **NFR103**: In order to simulate lunar observations from a satellite position (**FR105**) the user must provide an
  orbital scenario file in EOCFI compatible format.
- **NFR104**: The ESA satellites available for selection must include ENVISAT, Proba-V, S2, S3, FLEX.
- **NFR105**: The user defined SRF (**FR103**) must be defined via a user generated SRF file.
- **NFR106**: The user should be able to perform the simulation via command line using parameters or input files.


### Comparisons

#### Functional Requirements (FR)
- **FR201**: Allow performing comparisons of lunar observations from a remote sensing instrument to the LIME model output.
- **FR202**: Allow exporting plots.

#### Non Functional Requirements (NFR)
- **NFR201**: The remote sensing instrument observations (**FR201**) must be pre-stored in a GLOD format file.
- **NFR202**: The comparison plots shall provide: relative differences between measured and modeled lunar
  irradiance/reflectance vs. time and vs. lunar phase angle.
- **NFR203**: The comparison plots shall display statistical indicators: mean relative difference, standard
  deviation of the mean relative difference, temporal trend if applicable, number of comparison samples, etc.
- **NFR204**: The exported plots must be in .jpg or .pdf format.
- **NFR205**: The user should be able to perform the simulations via command line using parameters or input files.

### Output

#### Functional Requirements (FR)
- **FR301**: The LIME TBX shall output simulated lunar disk irradiance or reflectance.
- **FR302**: The LIME TBX shall output the simulated lunar disk degree of polarization.
- **FR303**: The LIME TBX shall output simulated lunar irradiance or reflectance associated uncertainty.
- **FR304**: The LIME TBX shall output the simulated lunar degree of polarization’s associated uncertainty.
- **FR305**: Allow the visualization of the user defined spectral response used for the spectral integration
  of the LIME output into a sensor spectral band.

#### Non Functional Requirements (NFR)
- **NFR301**: The simulated lunar disk irradiance or reflectance (**FR301**) must be in the spectral range of 400 nm to 2500 nm.
- **NFR302**: The simulated lunar disk degree of polarization (**FR302**) must be in the spectral range of 400 nm to 2500 nm.
- **NFR303**: The LIME simulated output shall be available to be exported to GLOD format files.
- **NFR304**: The LIME version number shall be visible on all outputs (plots/files) of the TBX.

### System

#### Functional Requirements (FR)
- **FR401**: The LIME TBX must be able to perform automatic updates of the LIME coefficients.
- **FR402**: The user shall be able to select past LIME coefficients with whom perform the calculations.

#### Non Functional Requirements (NFR)
- **NFR401**: The LIME coefficients of the automatic updates (**FR401**) must be stored on a dedicated repository.
- **NFR402**: The LIME TBX shall read the database of lunar observations formatted in GLOD format.
- **NFR403**: The LIME TBX shall use the EOCFI as orbit propagator and to derive satellite orbital positions.
- **NFR404**: Be to the largest extent platform/operating system independent.
- **NFR405**: Run at least under Windows / mac OS / Linux operating systems.
- **NFR406**: Be to the largest extent a self-installing SW package.
- **NFR407**: The LIME TBX shall be to the largest extent developed in Python.
- **NFR408**: The LIME TBX code shall be available on a password protected web repository allowing versioning of
  the software (e.g., GitHub).
 