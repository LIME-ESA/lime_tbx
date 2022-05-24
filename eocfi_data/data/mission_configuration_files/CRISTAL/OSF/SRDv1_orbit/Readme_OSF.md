# ORBSCT for CRISTAL (SRDv1 example orbit)
The Orbit Scenario File (filetype ORBSCT)
CRT_TEST_MPL_ORBSCT_20280101T180001_99999999T999999_0001.EOF
is intended to be used with tools built using EOCFI SW v4.16 onwards.


Starting from EOCFI SW v4.20, a dedicated satellite identifier (XO_SAT_CRISTAL) is available to initialise the satellite ID.

## Orbit Parameters
The orbit parameters are given according to CRISTAL SRD issue 1.0
- Repeat Cycle (days): 367

- Cycle Length (orbits): 5292

- Mean Local Solar Time drift (s/day): -179.786851 (corresponding to 92.0 deg inclination)

- Date: 01/Jan/2028 (--> launch date still unknown)

- Absolute Orbit Number: 1

- Longitude at ANX of Absolute Orbit Number (deg): 0.0 (--> default, actual value still unknown)

- Mean Local Solar Time at Ascending Node Crossing (ANX) of orbit 1 (h): 18:00 (--> default , actual value still unknown)

- Note the following:
    - The longitude at ANX of the reference orbit will be set later on, e.g. based on certain overpass criteria or desired phasing with another mission
    - Since the orbit is non Sun-Synchronous, with an MLST drift of about 180 seconds per day, the MLST at ANX will be changing and it will be only coming back to the intial value after approx. 480 days (=24 h * 86400 s / 180 s)
    - The selection of the initial MLST at ANX will play a role regarding Sun illumination conditions after launch and commissiong
    - Cross-observations with other missions will also be impacted by the drifting MLST at ANX

## Assumptions
- Time Correlations
    - TAI-UTC correlation (s): 37 s
    - UT1-UTC correlation (s): 0 s

- The file is compliant with GS File Format Standard v3.0.

## ORBSCT Generation
This ORBSCT has been generated with the following commands (using executable tools available as part of the EOCFI distribution)

- From Terminal window (Linux, macOS). For LInux, replace MACIN64 folder by LINUX64 folder
```$bash
./EOCFI-4.20-CLIB-MACIN64/bin/MACIN64/time_conv -date 2028-01-01T00:00:00 -fmt_in ASCII_CCSDSA -ref_in UTC -fmt_out PROC -ref_out UTC -tai 1.00042824074074 -utc 1.0 -ut1 1.0 -gps 1.000208333333 -show
```
It returns the date in MJD2000: 10227.0 days
```$bash
./EOCFI-4.20-CLIB-MACIN64/bin/MACIN64/gen_osf_create -sat CRISTAL -orbit 1 -cyc 1 -pha 1 -repcyc 367 -cyclen 5292 -mlst 18.0 -date 10227.0 -tai 1.00042824074074 -utc 1.0 -ut1 1.0 -gps 1.000208333333 -anx 0.0 -mlstdr -179.786851 -flcl TEST -show
```
- From Command Prompt  window (Windows)
```$bash
EOCFI-4.20-CLIB-MACIN64\bin\WINDOWS64\time_conv.exe -date 2028-01-01T00:00:00 -fmt_in ASCII_CCSDSA -ref_in UTC -fmt_out PROC -ref_out UTC -tai 1.00042824074074 -utc 1.0 -ut1 1.0 -gps 1.000208333333 -show
```
It returns the date in MJD2000: 10227.0 days
```$bash
EOCFI-4.20-CLIB-MACIN64\bin\WINDOWS64\gen_osf_create.exe -sat CRISTAL -orbit 1 -cyc 1 -pha 1 -repcyc 367 -cyclen 5292 -mlst 18.0 -date 10227.0 -tai 1.00042824074074 -utc 1.0 -ut1 1.0 -gps 1.000208333333 -anx 0.0 -mlstdr -179.786851 -flcl TEST -show
```
Further details can be found in the EO CFI SW Orbit User Manual.

## Additional Considerations
In case your application is built using EOCFI SW v4.15 or below, the MLST drift value needs to be adjusted.
Please consider upgrading your application latest EOCFI version in order to use the given ORBSCT file..

## Contact
For further questions, please contact us at the Mission Software CFI Support Team account:

cfi@eopp.esa.int
