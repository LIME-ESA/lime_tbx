
/******************************************************************************
 *
 * Project     : EOCFI
 * Copyright   : DEIMOS Space S.L.U.
 *
 * Module      : explorer_orbit.h
 *
 * Purpose     : It is the header file of the EXPLORER_ORBIT CFI to be used by
 *               the final user
 *
 * Reference   : Based on Envisat Mission CFI Software
 *
 * History:      +----------------------------------------------------------------------------------+
 *               | Version |   Date   |       Name        | Change                                  |
 *               |----------------------------------------------------------------------------------|
 *               |    1.1  | 23/05/02 | DEIMOS Space S.L. | First version from                      |
 *               |         |          |                   | Envisat CFI                             |
 *               |----------------------------------------------------------------------------------|
 *               |    1.2  | 19/07/02 | DEIMOS Space S.L. | New functions from                      |
 *               |         |          |                   | Envisat Visibility                      |
 *               +----------------------------------------------------------------------------------+
 *               |    2.0  | 29/11/02 | DEIMOS Space S.L. | Maintenance release                     |
 *               +----------------------------------------------------------------------------------+
 *               |    2.1  | 13/05/03 | DEIMOS Space S.L. | Maintenance release                     |
 *               |         |          |                   | MAC OS Version                          |
 *               |         |          |                   | Support for EE DORIS                    |
 *               |         |          |                   | Nav. files                              |
 *               |         |          |                   | Incorporate upgrades                    |
 *               |         |          |                   | from ENVISAT to                         |
 *               |         |          |                   | interpol                                |
 *               +----------------------------------------------------------------------------------+
 *               |    2.2  |30/09/03  | DEIMOS Space S.L. | Maintenance release                     |
 *               |         |          |                   | New options for                         |
 *               |         |          |                   | xo_propag_init_file                     |
 *               |         |          |                   | xo_interpol_init_file                   |
 *               +----------------------------------------------------------------------------------+
 *               |   2.2.2 | 26/04/04 | DEIMOS Space S.L. | Maintenance release                     |
 *               |         |          |                   |* Abs orbit returned by                  |
 *               |         |          |                   |  xo_propag_extra                        |
 *               |         |          |                   |  xo_interpol_extra                      |
 *               |         |          |                   |* New flag to use                        |
 *               |         |          |                   |  simplified model in                    |
 *               |         |          |                   |  xo_propag_init_def                     |
 *               |         |          |                   |* Use of enumerations                    |
 *               |         |          |                   |  for array sizing                       |
 *               +----------------------------------------------------------------------------------+
 *               |    3.0  | 21/07/04 | DEIMOS Space S.L. | New initialisation                      |
 *               |         |          |                   | strategy                                |
 *               |         |          |                   | New interfaces                          |
 *               +----------------------------------------------------------------------------------+
 *               |    3.1  | 13/10/04 | DEIMOS Space S.L. | Maintenance release                     |
 *               +----------------------------------------------------------------------------------+
 *               |    3.2  | 15/11/04 | DEIMOS Space S.L. | Maintenance release                     |
 *               +----------------------------------------------------------------------------------+
 *               |    3.3  | 11/07/05 | DEIMOS Space S.L. | Maintenance release                     |
 *               |         |          |                   | New features:                           |
 *               |         |          |                   | + explorer_data_handling                |
 *               |         |          |                   |   support                               |
 *               |         |          |                   | + Id. Accessors                         |
 *               |         |          |                   | + ENVISAT support removed               |
 *               +----------------------------------------------------------------------------------+
 *               |    3.4  | 18/11/05 | DEIMOS Space S.L. | Maintenance release                     |
 *               |         |          |                   | New features:                           |
 *               |         |          |                   | + Routines for orbit file               |
 *               |         |          |                   |   generation                            |
 *               +----------------------------------------------------------------------------------+
 *               |    3.5  | 26/05/06 | DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |          |                   | New features:                           |
 *               |         |          |                   |  + Thread safe library                  |
 *               |         |          |                   |  + C99 compliant                        |
 *               |         |          |                   |  + Optimized libray                     |
 *               |         |          |                   |  + 64-Bits library for MACOS,           |
 *               |         |          |                   |    LINUX and SOLARIS                    |
 *               |         |          |                   |  + New satellites: SWARM & EARTHCARE    |
 *               |         |          |                   |  + New time<->orbit executables         |
 *               +----------------------------------------------------------------------------------+
 *               |    3.6  | 24/11/06 | DEIMOS Space S.L. | Maintenance release                     |
 *               |         |          |                   | New features:                           |
 *               |         |          |                   |  + xo_gen_oef                           |
 *               |         |          |                   |  + xo_check_osf and xo_check_oef        |
 *               |         |          |                   |  + New DORIS TM format                  |
 *               |         |          |                   |  + New satellite: SENTINEL-1            |
 *               +----------------------------------------------------------------------------------+
 *               |    3.7  | 13/07/07 | DEIMOS Space S.L. | Maintenance release                     |
 *               |         |          |                   | New features:                           |
 *               |         |          |                   |  + expcfi_check_libs                    |
 *               |         |          |                   |  + library version for Mac OS X         |
 *               |         |          |                   |    on Intel (32 and 64 bits)            |
 *               +----------------------------------------------------------------------------------+
 *               |  3.7.2  | 31/07/08 | DEIMOS Space S.L. | Maintenance release                     |
 *               |         |          |                   | New features:                           |
 *               |         |          |                   |  + Orbit operations/Propagation with    |
 *               |         |          |                   |    TLE.                                 |
 *               |         |          |                   |  + New satellites: Sentinel 2 & 3,      |
 *               |         |          |                   |    Seosat and Generic.                  |
 *               |         |          |                   |  + xo_gen_oef function                  |
 *               +----------------------------------------------------------------------------------+
 *               |    4.0  | 19/01/09 | DEIMOS Space S.L. | Maintenance release                     |
 *               |         |          |                   | New features:                           |
 *               |         |          |                   |  + Numerical propagator.                |
 *               |         |          |                   |  + xo_osv_compute replaces xo_propag and|
 *               |         |          |                   |    xo_interpol.                         |
 *               |         |          |                   |  + xo_osv_compute_extra replaces        |
 *               |         |          |                   |    xo_propag_extra and xo_interpol_extra|
 *               |         |          |                   |  + xo_gen_oef function                  |
 *               +----------------------------------------------------------------------------------+
 *               |    4.1  | 07/05/10 | DEIMOS Space S.L. | Maintenance release                     |
 *               |         |          |                   | New features:                           |
 *               |         |          |                   |  + Time correlation compatibility check |
 *               |         |          |                   |    between time_id and orbit file data  |
 *               +----------------------------------------------------------------------------------+
 *               |    4.2  | 31/01/11 | DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |          |                   | New features:                           |
 *               |         |          |                   |  + Support for curved MLST              |
 *               +----------------------------------------------------------------------------------+
 *               |    4.3  | 06/02/12 | DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |          |                   | New features:                           |
 *               |         |          |                   |  + New init. funciton xo_orbit_id_init  |
 *               +----------------------------------------------------------------------------------+
 *               |    4.4  | 05/07/12 | DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |          |                   | New features:                           |
 *               |         |          |                   |  + Support for GEO orbits. New functions|
 *               |         |          |                   |    - xo_orbit_init_geo                  |
 *               |         |          |                   |    - xo_orbit_(get/set)_geo_orbit_info  |
 *               |         |          |                   |  + xo_position_on_orbit_to_time         |
 *               +----------------------------------------------------------------------------------+
 *               |    4.5  | 01/03/13 | DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |          |                   | New features:                           |
 *               |         |          |                   |  + New function xo_orbit_data_filter    |
 *               |         |          |                   |  + Acceleration vector is computed for  |
 *               |         |          |                   |    TLE and numeric propagators.         |
 *               +----------------------------------------------------------------------------------+
 *               |    4.6  | 03/10/13 | DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |          |                   | New features:                           |
 *               |         |          |                   |  + SDP4 TLE propagator        .         |
 *               |         |          |                   |  + Executable to generate TLE files     |
 *               |         |          |                   |  + Fitting method to comptue TLE in     |
 *               |         |          |                   |    xo_osv_to_tle function               |
 *               +----------------------------------------------------------------------------------+
 *               |    4.7  | 28/03/14 | DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |          |                   | New features:                           |
 *               |         |          |                   |  + Spacecraft Midnight computed by      |
 *               |         |          |                   |    xo_orbit_info function               |
 *               |         |          |                   |  + New function xo_orbit_id_change      |
 *               +----------------------------------------------------------------------------------+
 *               |    4.8  | 29/10/14 | DEIMOS Space S.L.U| Maintenance release                     |
 *               +----------------------------------------------------------------------------------+
 *               |    4.9  | 23/04/15 | DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |          |                   | New features:                           |
 *               |         |          |                   |  + Support for Orbit Ephemeris          |
 *               |         |          |                   |    Message files                        |
 *               +----------------------------------------------------------------------------------+
 *               |  4.9.1  | 05/06/15 | DEIMOS Space S.L.U| Maintenance release                     |
 *               +----------------------------------------------------------------------------------+
 *               |   4.10  | 29/10/15 | DEIMOS Space S.L.U| Maintenance release                     |
 *               +----------------------------------------------------------------------------------+
 *               |   4.11  | 15/04/16 | DEIMOS Space S.L.U| Maintenance release                     |
 *               +----------------------------------------------------------------------------------+
 *               |   4.12  | 03/11/16 | DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |          |                   | New features:                           |
 *               |         |          |                   |  + New function xo_osv_check            |
 *               +----------------------------------------------------------------------------------+
 *               |   4.13  | 05/04/17 | DEIMOS Space S.L.U| Maintenance release                     |
 *               +----------------------------------------------------------------------------------+
 *               |   4.14  | 16/11/17 | DEIMOS Space S.L.U| Maintenance release                     |
 *               +----------------------------------------------------------------------------------+
 *               |   4.15  | 20/04/18 | DEIMOS Space S.L.U| Maintenance release                     |
 *               +----------------------------------------------------------------------------------+
 *               |   4.16  | 09/11/18 | DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |          |                   | New features:                           |
 *               |         |          |                   |  + TLE support for Sentinel-5P,         |
 *               |         |          |                   |    Sentinel-3B, Aeolus                  |
 *               |         |          |                   |  + New function: xo_orbit_id_check      |
 *               +----------------------------------------------------------------------------------+
 *               |   4.17  | 10/05/19 | DEIMOS Space S.L.U| Maintenance release                     |
 *               +----------------------------------------------------------------------------------+
 *               |   4.18  | 08/11/19 | DEIMOS Space S.L.U| Maintenance release                     |
 *               +----------------------------------------------------------------------------------+
 *               |   4.19  | 29/05/20 | DEIMOS Space S.L.U| Maintenance release                     |
 *               +----------------------------------------------------------------------------------+
 *               |   4.20  | 30/11/20 | DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |          |                   | New features:                           |
 *               |         |          |                   |  + Support for LEO satellites with SP3  |
 *               |         |          |                   |    files                                |
 *               |         |          |                   |  + Support for new missions: Sentinel 6,|
 *               |         |          |                   |    CIMR, ROSEL, CHIME, CRISTAL, CO2M,   |
 *               |         |          |                   |    LSTM, FORUM                          |
 *               +----------------------------------------------------------------------------------+
 *               |   4.21  | 10/06/21 | DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |          |                   | New features:                           |
 *               |         |          |                   |  + Optimisation of EOCFI library size   |
 *               +----------------------------------------------------------------------------------+
 *               |   4.22  | 22/12/21 | DEIMOS Space S.L.U| Maintenance release                     |
 *               +----------------------------------------------------------------------------------+
 *               |   4.23  | 23/06/22 | DEIMOS Space S.L.U| Maintenance release                     |
 *               +----------------------------------------------------------------------------------+
 *
 *****************************************************************************/

#ifndef EXPLORER_ORBIT_H
#define EXPLORER_ORBIT_H

#include <explorer_data_handling.h>
#include <explorer_lib.h>

/*******************************************************************************
*
* Defines
*
*******************************************************************************/

/* Length of ierr vectors */
/* ---------------------- */
#define XO_NUM_ERR_ORBIT_INIT_DEF              1
#define XO_NUM_ERR_ORBIT_CART_INIT             1
#define XO_NUM_ERR_ORBIT_CART_INIT_PRECISE     1
#define XO_NUM_ERR_ORBIT_ID_INIT               2 /* AN-577; updated from 1 to 2 since the function has many errors and jump to XL_FIRST_E is used */
#define XO_NUM_ERR_ORBIT_INIT_FILE             2 /* AN-577; updated from 1 to 2 since the function has many errors and jump to XL_FIRST_E is used */
#define XO_NUM_ERR_ORBIT_INIT_FILE_PRECISE     1
#define XO_NUM_ERR_ORBIT_INIT_GEO              1
#define XO_NUM_ERR_ORBIT_CLOSE                 1
#define XO_NUM_ERR_GENSTATE                    1
#define XO_NUM_ERR_OSV_COMPUTE                 1
#define XO_NUM_ERR_OSV_COMPUTE_EXTRA           1
#define XO_NUM_ERR_PROPAG_CONFIG               1
#define XO_NUM_ERR_ORBIT_TO_TIME               1
#define XO_NUM_ERR_TIME_TO_ORBIT               1
#define XO_NUM_ERR_ORBIT_REL_FROM_ABS          1
#define XO_NUM_ERR_ORBIT_ABS_FROM_REL          1
#define XO_NUM_ERR_ORBIT_ABS_FROM_PHASE        1
#define XO_NUM_ERR_ORBIT_INFO                  1
#define XO_NUM_ERR_OSV_TO_TLE                  1
#define XO_NUM_ERR_RUN_INIT                    1
#define XO_NUM_ERR_GEN_OEF                     1
#define XO_NUM_ERR_GEN_OSF_CREATE              1
#define XO_NUM_ERR_GEN_OSF_APPEND_ORBIT_CHANGE 1
#define XO_NUM_ERR_GEN_OSF_CHANGE_REPEAT_CYCLE 1
#define XO_NUM_ERR_GEN_OSF_ADD_DRIFT_CYCLE     1
#define XO_NUM_ERR_GEN_POF                     1
#define XO_NUM_ERR_GEN_ROF                     1
#define XO_NUM_ERR_GEN_ROF_PROTOTYPE           1
#define XO_NUM_ERR_GEN_DNF                     1
#define XO_NUM_ERR_GEN_TLE                     1
#define XO_NUM_ERR_CHECK_OSF                   1
#define XO_NUM_ERR_CHECK_OEF                   1
#define XO_NUM_ERR_POSITION_ON_ORBIT_TO_TIME   1
#define XO_NUM_ERR_ORBIT_DATA_FILTER           1
#define XO_NUM_ERR_ORBIT_ID_CHANGE             1
#define XO_NUM_ERR_ORBIT_ID_INIT_DATA_CLOSE    1
#define XO_NUM_ERR_ORBIT_INFO_CONFIGURE        1 // AN-523
#define XO_NUM_ERR_OSV_CHECK                   1 // AN-632
#define XO_NUM_ERR_ORBIT_ID_CHECK              1 // ANR-692
#define XO_NUM_ERR_PROPAG_CLOSE                1 // AN-738

/* Max length of ierr vectors */
/* -------------------------- */
#define XO_ERR_VECTOR_MAX_LENGTH 2 //AN-577; updated from 1 to 2 since XO_NUM_ERR_ORBIT_ID_INIT and XO_NUM_ERR_ORBIT_INIT_FILE returns 2 longs
/* Max of the above lengths.
   This value can be safely used in every function declaration. */

/* Maximum length of a string (function name or error message) */
#define XO_MAX_COD XL_MAX_COD

/* Maximum number of error codes that can be defined for a single function */
#define XO_MAX_STR XL_MAX_STR

/* Number of elements to compare in xo_check_osf and xo_check_oef */
#define XO_NUM_CHECK_PARAMS 6

/* AN-353: Geostationary default values */
#define XO_GEO_DEFAULT_LATITUDE 0.
#define XO_GEO_DEFAULT_ALTITUDE 35786000.

/* Generic maximum lengths */
#define XO_MAX_LENGTH_1 1
#define XO_MAX_LENGTH_3 3
#define XO_MAX_LENGTH_6 6

#ifdef __cplusplus
extern "C"
{
#endif

  /*******************************************************************************
*
* ENUMERATIONS: Modes and switches
*
*******************************************************************************/

  /* Returned status code */
  /* -------------------- */
  typedef enum
  {

    XO_ERR = XL_ERR, /* Error status */
    XO_OK = XL_OK, /* Nominal status */
    XO_WARN = XL_WARN /* Warning status */

  } XO_PErr_type_enum; /* CAREFUL: variables holding these values MUST be declared as long !!! */

  /* Satellite ID */
  /* ------------ */

  typedef enum
  {

    XO_SAT_DEFAULT = XL_SAT_DEFAULT,
    XO_SAT_DEFAULT1 = XL_SAT_DEFAULT1,
    XO_SAT_DEFAULT2 = XL_SAT_DEFAULT2,
    XO_SAT_DEFAULT3 = XL_SAT_DEFAULT3,
    XO_SAT_DEFAULT4 = XL_SAT_DEFAULT4,
    XO_SAT_DEFAULT5 = XL_SAT_DEFAULT5,
    XO_SAT_DEFAULT6 = XL_SAT_DEFAULT6,
    XO_SAT_DEFAULT7 = XL_SAT_DEFAULT7,
    XO_SAT_DEFAULT8 = XL_SAT_DEFAULT8,
    XO_SAT_DEFAULT9 = XL_SAT_DEFAULT9,
    XO_SAT_ERS1 = XL_SAT_ERS1,
    XO_SAT_ERS2 = XL_SAT_ERS2,
    XO_SAT_ENVISAT = XL_SAT_ENVISAT,
    XO_SAT_METOP1 = XL_SAT_METOP1,
    XO_SAT_METOP2 = XL_SAT_METOP2,
    XO_SAT_METOP3 = XL_SAT_METOP3,
    XO_SAT_CRYOSAT = XL_SAT_CRYOSAT,
    XO_SAT_ADM = XL_SAT_ADM,
    XO_SAT_GOCE = XL_SAT_GOCE,
    XO_SAT_SMOS = XL_SAT_SMOS,
    XO_SAT_TERRASAR = XL_SAT_TERRASAR,
    XO_SAT_EARTHCARE = XL_SAT_EARTHCARE,
    XO_SAT_SWARM_A = XL_SAT_SWARM_A,
    XO_SAT_SWARM_B = XL_SAT_SWARM_B,
    XO_SAT_SWARM_C = XL_SAT_SWARM_C,
    XO_SAT_SENTINEL_1A = XL_SAT_SENTINEL_1A,
    XO_SAT_SENTINEL_1B = XL_SAT_SENTINEL_1B,
    XO_SAT_SENTINEL_2 = XL_SAT_SENTINEL_2,
    XO_SAT_SENTINEL_3 = XL_SAT_SENTINEL_3,
    XO_SAT_SEOSAT = XL_SAT_SEOSAT,
    XO_SAT_SENTINEL_1C = XL_SAT_SENTINEL_1C,
    XO_SAT_SENTINEL_2A = XL_SAT_SENTINEL_2A,
    XO_SAT_SENTINEL_2B = XL_SAT_SENTINEL_2B,
    XO_SAT_SENTINEL_2C = XL_SAT_SENTINEL_2C,
    XO_SAT_SENTINEL_3A = XL_SAT_SENTINEL_3A,
    XO_SAT_SENTINEL_3B = XL_SAT_SENTINEL_3B,
    XO_SAT_SENTINEL_3C = XL_SAT_SENTINEL_3C,
    XO_SAT_JASON_CSA = XL_SAT_JASON_CSA, // AN-531
    XO_SAT_JASON_CSB = XL_SAT_JASON_CSB, // AN-531
    XO_SAT_METOP_SG_A1 = XL_SAT_METOP_SG_A1, // AN-530
    XO_SAT_METOP_SG_A2 = XL_SAT_METOP_SG_A2, // AN-530
    XO_SAT_METOP_SG_A3 = XL_SAT_METOP_SG_A3, // AN-530
    XO_SAT_METOP_SG_B1 = XL_SAT_METOP_SG_B1, // AN-530
    XO_SAT_METOP_SG_B2 = XL_SAT_METOP_SG_B2, // AN-530
    XO_SAT_METOP_SG_B3 = XL_SAT_METOP_SG_B3, // AN-530
    XO_SAT_SENTINEL_5P = XL_SAT_SENTINEL_5P, // AN-502
    XO_SAT_BIOMASS = XL_SAT_BIOMASS, // AN-601
    XO_SAT_SENTINEL_5 = XL_SAT_SENTINEL_5, // AN-639
    XO_SAT_SAOCOM_CS = XL_SAT_SAOCOM_CS, // AN-640
    XO_SAT_FLEX = XL_SAT_FLEX, /* AN-683 */
    XO_SAT_SENTINEL_6A = XL_SAT_SENTINEL_6A, /* AN-809 */
    XO_SAT_SENTINEL_6B = XL_SAT_SENTINEL_6B, /* AN-809 */
    XO_SAT_CIMR = XL_SAT_CIMR,
    XO_SAT_ROSEL = XL_SAT_ROSEL,
    XO_SAT_CHIME = XL_SAT_CHIME,
    XO_SAT_CRISTAL = XL_SAT_CRISTAL,
    XO_SAT_CO2M = XL_SAT_CO2M,
    XO_SAT_LSTM = XL_SAT_LSTM,
    XO_SAT_FORUM = XL_SAT_FORUM,
    XO_SAT_TRUTHS = XL_SAT_TRUTHS,
    XO_SAT_GENERIC = XL_SAT_GENERIC,
    XO_SAT_GENERIC_GEO = XL_SAT_GENERIC_GEO, /* ANR-353 */
    XO_SAT_MTG = XL_SAT_MTG, /* ANR-353 */
    XO_SAT_GENERIC_MEO = XL_SAT_GENERIC_MEO /* ANR-550 */

  } XO_Sat_id_enum; /* CAREFUL: if you modify this enum, you must modify XO_Sat_array_enum enum
 			       and XL_Sat_id_to_array routine */

  /* Time reference ID */
  /* ----------------- */

  typedef enum
  {

    XO_TIME_UNDEF = XL_TIME_UNDEF,
    XO_TIME_TAI = XL_TIME_TAI,
    XO_TIME_UTC = XL_TIME_UTC,
    XO_TIME_UT1 = XL_TIME_UT1,
    XO_TIME_GPS = XL_TIME_GPS

  } XO_Time_ref_enum;

  /* File reference time */
  typedef enum
  {
    XO_TIME_REF_OF_TAI = XD_TIME_REF_OF_TAI,
    XO_TIME_REF_OF_UTC = XD_TIME_REF_OF_UTC,
    XO_TIME_REF_OF_UT1 = XD_TIME_REF_OF_UT1
  } XO_Time_ref_of_enum;

  /* orbit initialisation modes */
  /* -------------------------- */

  typedef enum
  {
    XO_SEL_FILE = XL_SEL_FILE,
    XO_SEL_TIME = XL_SEL_TIME,
    XO_SEL_ORBIT = XL_SEL_ORBIT,
    XO_SEL_DEFAULT = XL_SEL_DEFAULT

  } XO_Time_init_mode;

  /* orbit_init modes */
  /* ---------------- */

  typedef enum
  {
    XO_ORBIT_INIT_UNKNOWN_MODE = -1,
    XO_ORBIT_INIT_AUTO = XD_AUTO,
    XO_ORBIT_INIT_ORBIT_CHANGE_MODE = XD_ORBIT_CHANGE,
    XO_ORBIT_INIT_STATE_VECTOR_MODE = XD_STATE_VECTOR,
    XO_ORBIT_INIT_OSF_MODE = XD_OSF_TYPE,
    XO_ORBIT_INIT_POF_MODE = XD_POF_TYPE,
    XO_ORBIT_INIT_ROF_MODE = XD_ROF_TYPE,
    XO_ORBIT_INIT_DORIS_MODE = XD_DORIS_TYPE,
    XO_ORBIT_INIT_POF_N_DORIS_MODE = XD_POF_N_DORIS_TYPE,
    XO_ORBIT_INIT_OEF_OSF_MODE = XD_OEF_OSF_TYPE,
    XO_ORBIT_INIT_OEF_POF_MODE = XD_OEF_POF_TYPE,
    XO_ORBIT_INIT_TLE_MODE = XD_TLE_TYPE,
    XO_ORBIT_INIT_SP3_MODE = XD_SP3_TYPE, /* AN-550 */
    XO_ORBIT_INIT_OEM_MODE = XD_OEM_TYPE, /* AN-578 */
    XO_ORBIT_INIT_STATE_VECTOR_PRECISE_MODE,
    XO_ORBIT_INIT_POF_PRECISE_MODE,
    XO_ORBIT_INIT_ROF_PRECISE_MODE,
    XO_ORBIT_INIT_DORIS_PRECISE_MODE,
    XO_ORBIT_INIT_OEF_POF_PRECISE_MODE,
    XO_ORBIT_INIT_POF_N_DORIS_PRECISE_MODE,
    XO_ORBIT_INIT_GEO_LON_MODE, /* ANR-353 */
    XO_ORBIT_INIT_TLE_SGP4_MODE, /* ANR-494 */
    XO_ORBIT_INIT_TLE_SDP4_MODE, /* ANR-494 */
    XO_ORBIT_INIT_USER_OSV_LIST_MODE, /* AN-497 */
    XO_ORBIT_INIT_POF_ORBNUM_ADJ_MODE, /* ANR-527 */
    XO_ORBIT_INIT_ROF_ORBNUM_ADJ_MODE, /* ANR-527 */
    XO_ORBIT_INIT_DORIS_ORBNUM_ADJ_MODE, /* ANR-527 */
    XO_ORBIT_INIT_OEM_ORBNUM_ADJ_MODE, /* ANR-578 */
    XO_ORBIT_INIT_MAX_VALUE

  } XO_Orbit_init_mode;

  /* xo_orbit_init_def MLST modes */
  /* ---------------------------- */

  typedef enum
  {
    XO_NOSUNSYNC_DRIFT = 0,
    XO_NOSUNSYNC_INCLINATION,
    XO_NOSUNSYNC_DRIFT_NONLINEAR,
    XO_NOSUNSYNC_USE_SIM_MODEL = 10,

  } XO_No_sun_sychronous_mode;

  /* xo_orbit_info results vector */
  /* ---------------------------- */
  typedef enum
  {

    XO_ORBIT_INFO_EXTRA_REPEAT_CYCLE = 0,
    XO_ORBIT_INFO_EXTRA_CYCLE_LENGTH,
    XO_ORBIT_INFO_EXTRA_MLST_DRIFT,
    XO_ORBIT_INFO_EXTRA_MLST,
    XO_ORBIT_INFO_EXTRA_ANX_LONG,
    XO_ORBIT_INFO_EXTRA_UTC_ANX,
    XO_ORBIT_INFO_EXTRA_POS_X,
    XO_ORBIT_INFO_EXTRA_POS_Y,
    XO_ORBIT_INFO_EXTRA_POS_Z,
    XO_ORBIT_INFO_EXTRA_VEL_X,
    XO_ORBIT_INFO_EXTRA_VEL_Y,
    XO_ORBIT_INFO_EXTRA_VEL_Z,
    XO_ORBIT_INFO_EXTRA_MEAN_KEPL_A,
    XO_ORBIT_INFO_EXTRA_MEAN_KEPL_E,
    XO_ORBIT_INFO_EXTRA_MEAN_KEPL_I,
    XO_ORBIT_INFO_EXTRA_MEAN_KEPL_RA,
    XO_ORBIT_INFO_EXTRA_MEAN_KEPL_W,
    XO_ORBIT_INFO_EXTRA_MEAN_KEPL_M,
    XO_ORBIT_INFO_EXTRA_OSC_KEPL_A,
    XO_ORBIT_INFO_EXTRA_OSC_KEPL_E,
    XO_ORBIT_INFO_EXTRA_OSC_KEPL_I,
    XO_ORBIT_INFO_EXTRA_OSC_KEPL_RA,
    XO_ORBIT_INFO_EXTRA_OSC_KEPL_W,
    XO_ORBIT_INFO_EXTRA_OSC_KEPL_M,
    XO_ORBIT_INFO_EXTRA_NODAL_PERIOD,
    XO_ORBIT_INFO_EXTRA_UTC_SMX, // AN-523
    XO_ORBIT_INFO_EXTRA_NUM_ELEMENTS

  } XO_ORBIT_INFO_EXTRA_enum;

  /* Propagation models enum */
  /* ----------------------- */

  typedef enum
  {
    XO_PROPAG_MODEL_NOT_INITIALIZED = -1, /* Propagator not initialized (AN-497)*/
    XO_PROPAG_MODEL_MEAN_KEPL = 0, /* Propagation using Mean Keplerian OSV model */
    XO_PROPAG_MODEL_SPOT = 1, /* Propagation using Spot model */
    XO_PROPAG_MODEL_TLE = 2, /* Propagation using TLE model */
    XO_PROPAG_MODEL_PRECISE = 3, /* Propagation using COWELL propagator */
    XO_PROPAG_MODEL_GEO = 4, /* Propagation for geostationary satellites (ANR-353) */
    XO_PROPAG_MODEL_AUTO = 10, /* Propagation using Auto initialization model */
    XO_PROPAG_MODEL_DOUBLE = 100 /* Propagation using double initialization model */

  } XO_Propag_model_enum;

  /* xo_osv_compute_extra extra_choice modes */
  /* --------------------------------------- */

  typedef enum
  {
    /* model independant*/
    XO_ORBIT_EXTRA_NO_RESULTS = 0, /* To calculate no extra results */
    XO_ORBIT_EXTRA_GEOLOCATION = 1, /* To calculate the geolocation data set */
    XO_ORBIT_EXTRA_GEOLOCATION_D = 2, /* To calculate the geolocation rate data set */
    XO_ORBIT_EXTRA_GEOLOCATION_2D = 4, /* To calculate the geolocation rate rate data  set */
    XO_ORBIT_EXTRA_GEOLOCATION_EXTRA = 8, /* To calculate the geolocation extra data set */
    XO_ORBIT_EXTRA_EARTH_FIXED_D = 16, /* To calculate the earth-fixed velocity data set */
    XO_ORBIT_EXTRA_EARTH_FIXED_2D = 32, /* To calculate the earth-fixed acceleration data set */
    XO_ORBIT_EXTRA_SUN = 64, /* To calculate the sun data set */
    XO_ORBIT_EXTRA_MOON = 128, /* To calculate the moon data set */
    XO_ORBIT_EXTRA_OSCULATING_KEPLER = 256, /* To calculate the osculating Kepler data set */
    XO_ORBIT_EXTRA_INERTIAL_AUX = 512, /* To calculate the inertial auxiliary data set */
    /* model dependant */
    XO_ORBIT_EXTRA_DEP_ANX_TIMING = 1024, /* To calculate ANX timing data set */
    XO_ORBIT_EXTRA_DEP_MEAN_KEPLER = 2048, /* To calculate the mean keplerian elements */
    XO_ORBIT_EXTRA_ALL_RESULTS = 4095 /* To calculate all extra results */

  } XO_ORBIT_EXTRA_choice_enum; /* CAREFUL: look up thoroughly the SUM to understand how to use the results vector mode */

  /* xo_orbit_extra MKO model dependant results vector */
  /* ------------------------------------------------- */
  typedef enum
  {
    XO_ORBIT_EXTRA_DEP_NODAL_PERIOD = 0,
    XO_ORBIT_EXTRA_DEP_UTC_CURRENT_ANX,
    XO_ORBIT_EXTRA_DEP_ORBIT_NUMBER,
    XO_ORBIT_EXTRA_DEP_SEC_SINCE_ANX,
    XO_ORBIT_EXTRA_DEP_MEAN_KEPL_A,
    XO_ORBIT_EXTRA_DEP_MEAN_KEPL_E,
    XO_ORBIT_EXTRA_DEP_MEAN_KEPL_I,
    XO_ORBIT_EXTRA_DEP_MEAN_KEPL_RA,
    XO_ORBIT_EXTRA_DEP_MEAN_KEPL_W,
    XO_ORBIT_EXTRA_DEP_MEAN_KEPL_M,
    XO_ORBIT_EXTRA_NUM_DEP_ELEMENTS

  } XO_ORBIT_EXTRA_Model_MKO_dependant_enum;

  /* xo_orbit_extra model independant results vector */
  /* ----------------------------------------------- */
  typedef enum
  {
    XO_ORBIT_EXTRA_GEOC_LONG = 0,
    XO_ORBIT_EXTRA_GEOD_LAT,
    XO_ORBIT_EXTRA_GEOD_ALT,
    XO_ORBIT_EXTRA_GEOC_LONG_D,
    XO_ORBIT_EXTRA_GEOD_LAT_D,
    XO_ORBIT_EXTRA_GEOD_ALT_D,
    XO_ORBIT_EXTRA_GEOC_LONG_2D,
    XO_ORBIT_EXTRA_GEOD_LAT_2D,
    XO_ORBIT_EXTRA_GEOD_ALT_2D,
    XO_ORBIT_EXTRA_RAD_CUR_PARALLEL_MERIDIAN,
    XO_ORBIT_EXTRA_RAD_CUR_ORTHO_MERIDIAN,
    XO_ORBIT_EXTRA_RAD_CUR_ALONG_GROUNDTRACK,
    XO_ORBIT_EXTRA_NORTH_VEL,
    XO_ORBIT_EXTRA_EAST_VEL,
    XO_ORBIT_EXTRA_MAG_VEL,
    XO_ORBIT_EXTRA_AZ_VEL,
    XO_ORBIT_EXTRA_NORTH_ACC,
    XO_ORBIT_EXTRA_EAST_ACC,
    XO_ORBIT_EXTRA_GROUNDTRACK_ACC,
    XO_ORBIT_EXTRA_AZ_ACC,
    XO_ORBIT_EXTRA_SAT_ECLIPSE_FLAG,
    XO_ORBIT_EXTRA_SZA,
    XO_ORBIT_EXTRA_MLST,
    XO_ORBIT_EXTRA_TLST,
    XO_ORBIT_EXTRA_TRUE_SUN_RA,
    XO_ORBIT_EXTRA_TRUE_SUN_DEC,
    XO_ORBIT_EXTRA_TRUE_SUN_SEMIDIAM,
    XO_ORBIT_EXTRA_MOON_RA,
    XO_ORBIT_EXTRA_MOON_DEC,
    XO_ORBIT_EXTRA_MOON_SEMI_DIAM,
    XO_ORBIT_EXTRA_MOON_AREA_LIT,
    XO_ORBIT_EXTRA_OSC_KEPL_A,
    XO_ORBIT_EXTRA_OSC_KEPL_E,
    XO_ORBIT_EXTRA_OSC_KEPL_I,
    XO_ORBIT_EXTRA_OSC_KEPL_RA,
    XO_ORBIT_EXTRA_OSC_KEPL_W,
    XO_ORBIT_EXTRA_OSC_KEPL_M,
    XO_ORBIT_EXTRA_ORBIT_RAD,
    XO_ORBIT_EXTRA_RADIAL_ORB_VEL,
    XO_ORBIT_EXTRA_TRANS_ORB_VEL,
    XO_ORBIT_EXTRA_ORB_VEL_MAG,
    XO_ORBIT_EXTRA_RA_SAT,
    XO_ORBIT_EXTRA_DEC_SAT,
    XO_ORBIT_EXTRA_EARTH_ROTATION_ANGLE,
    XO_ORBIT_EXTRA_RA_SAT_D,
    XO_ORBIT_EXTRA_RA_SAT_2D,
    XO_ORBIT_EXTRA_OSC_TRUE_LAT,
    XO_ORBIT_EXTRA_OSC_TRUE_LAT_D,
    XO_ORBIT_EXTRA_OSC_TRUE_LAT_2D,
    XO_ORBIT_EXTRA_NUM_INDEP_ELEMENTS

  } XO_ORBIT_EXTRA_Model_independant_enum;

  /* Interpolation model */
  /* ------------------- */

  typedef enum
  {

    XO_INTERPOL_MODEL_DEFAULT = 0

  } XO_INTERPOL_model_enum;

  /* Phase increment */
  typedef enum
  {

    XO_NO_PHASE_INCREMENT = 0,
    XO_PHASE_INCREMENT = 1

  } XO_Gen_osf_phase_increment_enum;

  /* Search direction */
  typedef enum
  {

    XO_SEARCH_FORWARD = 1,
    XO_SEARCH_BACKWARD = -1

  } XO_Gen_osf_search_direction_enum;

  typedef enum
  {
    XO_REF_FILETYPE_OSF = 1, /* Orbit Scenario File */
    XO_REF_FILETYPE_OEF_OSF, /* Orbit Event file used as an OSF */
    XO_REF_FILETYPE_POF, /* FOS Predicted Orbit File */
    XO_REF_FILETYPE_OEF_POF, /* Orbit Event file used as a POF */
    XO_REF_FILETYPE_DORIS_NAV, /* DORIS Navigator File */
    XO_REF_FILETYPE_ROF, /* FOS Restituted Orbit File */
    XO_REF_FILETYPE_TLE, /* TLE Orbit File */
    XO_REF_FILETYPE_DORIS_PREM, /* DORIS Preliminary Orbit File */
    XO_REF_FILETYPE_DORIS_PREC /* DORIS Precise Orbit File */

  } XO_Ref_filetype_enum;

  typedef enum
  {
    XO_OSV_PRECISE_NO = 1, /* Default value, non-precise */
    XO_OSV_PRECISE_MINUTE, /* Precise location every integer minute */
    XO_OSV_PRECISE_TEN_SECONDS /* Precise location every integer ten secods */

  } XO_Osv_precise_enum;

  typedef enum
  {
    XO_FIT_TLE = XD_FIT_TLE, /*Used for POF, not valid for OSV LIST*/
    XO_FIT_TLE_LIST = XD_FIT_TLE_LIST,
    XO_ONE_TLE_PER_OSV = XD_ONE_TLE_PER_OSV

  } XO_TLE_fitmode_enum;

  /* CFI functions IDs (for error handling) */
  /* -------------------------------------- */

  typedef enum
  {
    XO_NO_FUNC_ID = -1, /* No function code */
    XO_ORBIT_INIT_DEF_ID,
    XO_ORBIT_CART_INIT_ID,
    XO_ORBIT_CART_INIT_PRECISE_ID,
    XO_ORBIT_ID_INIT_ID,
    XO_ORBIT_INIT_FILE_ID,
    XO_ORBIT_INIT_FILE_PRECISE_ID,
    XO_ORBIT_INIT_GEO_ID,
    XO_ORBIT_CLOSE_ID,
    XO_OSV_COMPUTE_ID,
    XO_OSV_COMPUTE_EXTRA_ID,
    XO_PROPAG_CONFIG_ID,
    XO_ORBIT_TO_TIME_ID,
    XO_TIME_TO_ORBIT_ID,
    XO_ORBIT_ABS_FROM_REL_ID,
    XO_ORBIT_REL_FROM_ABS_ID,
    XO_ORBIT_ABS_FROM_PHASE_ID,
    XO_ORBIT_INFO_ID,
    XO_OSV_TO_TLE_ID,
    XO_RUN_INIT_ID,
    XO_GEN_OEF_ID,
    XO_GEN_OSF_CREATE_ID,
    XO_GEN_OSF_APPEND_ORBIT_CHANGE_ID,
    XO_GEN_OSF_CHANGE_REPEAT_CYCLE_ID,
    XO_GEN_OSF_ADD_DRIFT_CYCLE_ID,
    XO_GEN_POF_ID,
    XO_GEN_ROF_ID,
    XO_GEN_ROF_PROTOTYPE_ID,
    XO_GEN_DNF_ID,
    XO_GEN_TLE_ID,
    XO_CHECK_OSF_ID,
    XO_CHECK_OEF_ID,
    XO_POSITION_ON_ORBIT_TO_TIME_ID,
    XO_ORBIT_DATA_FILTER_ID,
    XO_ORBIT_ID_INIT_DATA_CLOSE_ID,
    XO_ORBIT_ID_CHANGE_ID,
    XO_ORBIT_INFO_CONFIGURE_ID, // AN-523
    XO_OSV_CHECK_ID, // AN-632
    XO_ORBIT_ID_CHECK_ID, //  ANR-692

    XO_MAX_FUNC_ID /* Number of functions defined in this list */

  } XO_Func_enum;

  /* User flag for precise propagator */
  typedef enum
  {
    XO_DEFAULT_VALUES = XD_DEFAULT_VALUES, /* Use predefined default values for some parameters  */
    XO_USER_VALUES = XD_USER_VALUES /* Use values introduced by user */
  } XO_Precise_propag_user_flag_enum;

  /* Flag to select or not contributions to precise propagator  */
  typedef enum
  {
    XO_NOT_SELECT = XD_NOT_SELECT, /* Do not select contribution */
    XO_SELECT = XD_SELECT /* Select contribution */
  } XO_Precise_propag_selection_flag_enum;

  /* Flag to select if SGA values are taken for input parameters or
   read from file */
  typedef enum
  {
    XO_SGA_USE_PARAMETERS = XD_SGA_USE_PARAMETERS, /* Use input parameters */
    XO_SGA_READ_VALUES_FROM_FILE = XD_SGA_READ_VALUES_FROM_FILE /* Read values from files */
  } XO_Precise_propag_sga_flag_enum;

  /* AN-353 */
  typedef enum
  {
    XO_GC_LONGITUDE_ONLY
  } XO_Geo_coord_enum;

  // AN-523: orbit info items
  typedef enum
  {
    XO_ORBIT_INFO_ITEM_SMX // Spacecraft midnight
  } XO_Orbit_info_items_enum;

  // AN-523: orbit info activation/deactivation of item computations
  typedef enum
  {
    XO_DEACTIVATE_ITEM, // Item not computed
    XO_ACTIVATE_ITEM // Item computed
  } XO_Orbit_info_flag_enum;

  /* AN-596 */
  typedef enum
  {
    XO_TIME_TYPE, /* Only time info is provided*/
    XO_ORBIT_TYPE, /* Only orbit info is provided */
    XO_BOTH_TYPE /* UTC time and orbit info is provided */
  } XO_Time_type_enum;

  /* AN-596 */
  typedef enum
  {
    XO_ORBIT_ABS = 0, /* Absolute orbit chosen as input */
    XO_ORBIT_REL = 1 /* Relative orbit chosen as input */
  } XO_Orbit_type_enum;

  /*******************************************************************************
*
* List of errors of the main CFI functions (Look into the EXPLORER_ORBIT SUM)
*
*******************************************************************************/

  /* Error codes list of xo_orbit_init_def */
  /* ------------------------------------- */
  typedef enum
  {
    XO_CFI_ORBIT_INIT_DEF_SAT_ERR = XL_ZERO_E,
    XO_CFI_ORBIT_INIT_DEF_GEO_SAT_ERR,
    XO_CFI_ORBIT_INIT_DEF_FLAG_ERR,
    XO_CFI_ORBIT_INIT_DEF_TIME_CHANGE_ERR,
    XO_CFI_ORBIT_INIT_DEF_INPUTS_ERR,
    XO_CFI_ORBIT_INIT_DEF_GENSTATE_ERR,
    XO_CFI_ORBIT_INIT_DEF_MEMORY_ERR,
    XO_CFI_ORBIT_INIT_DEF_PROPAG_INIT_ERR,
    XO_CFI_ORBIT_INIT_DEF_DEPRECATED_WARN

  } XO_CFI_Orbit_init_def_err_enum;

  /* Error codes list of xo_orbit_cart_init */
  /* -------------------------------------- */
  typedef enum
  {
    XO_CFI_ORBIT_CART_INIT_SAT_ERR = XL_ZERO_E,
    XO_CFI_ORBIT_CART_INIT_GEO_SAT_ERR,
    XO_CFI_ORBIT_CART_INIT_FLAG_ERR,
    XO_CFI_ORBIT_CART_INIT_TIME_STATUS_ERR,
    XO_CFI_ORBIT_CART_INIT_STATUS_ERR,
    XO_CFI_ORBIT_CART_INIT_TIME_TRANSFORMING_ERR,
    XO_CFI_ORBIT_CART_INIT_TIME_RANGE_ERR,
    XO_CFI_ORBIT_CART_INIT_MEMORY_ERR,
    XO_CFI_ORBIT_CART_INIT_PROPAG_INIT_ERR

  } XO_CFI_Orbit_cart_init_err_enum;

  /* Error codes list of xo_orbit_cart_init_precise */
  /* ---------------------------------------------- */
  typedef enum
  {
    XO_CFI_ORBIT_CART_INIT_PRECISE_CART_INIT_ERR = XL_ZERO_E,
    XO_CFI_ORBIT_CART_INIT_PRECISE_PROPAG_INIT_ERR,
    XO_CFI_ORBIT_CART_INIT_PRECISE_TIME_CONVERSION_ERR,
    XO_CFI_ORBIT_CART_INIT_PRECISE_PRECISE_PARAMS_ERR,
    XO_CFI_ORBIT_CART_INIT_PRECISE_SAT_ERR,
    XO_CFI_ORBIT_CART_INIT_PRECISE_FLAG_ERR,
    XO_CFI_ORBIT_CART_INIT_PRECISE_TIME_STATUS_ERR,
    XO_CFI_ORBIT_CART_INIT_PRECISE_STATUS_ERR,
    XO_CFI_ORBIT_CART_INIT_PRECISE_TIME_RANGE_ERR,
    XO_CFI_ORBIT_CART_INIT_PRECISE_MEMORY_ERR

  } XO_CFI_Orbit_cart_init_precise_err_enum;

  /* Error codes list of xo_orbit_id_init */
  /* ------------------------------------ */
  typedef enum
  {
    XO_CFI_ORBIT_ID_INIT_SAT_ERR = XL_ZERO_E,
    XO_CFI_ORBIT_ID_INIT_GEO_SAT_ERR,
    XO_CFI_ORBIT_ID_INIT_FLAG_ERR,
    XO_CFI_ORBIT_ID_INIT_TIME_STATUS_ERR,
    XO_CFI_ORBIT_ID_INIT_ORBIT_STATUS_ERR,
    XO_CFI_ORBIT_ID_INIT_MEMORY_ERR,
    XO_CFI_ORBIT_ID_INIT_NO_DATA_ERR,
    XO_CFI_ORBIT_ID_INIT_WRONG_ORBIT_MODEL_ERR,
    XO_CFI_ORBIT_ID_INIT_INCONSISTENT_FILES_ERR,
    XO_CFI_ORBIT_ID_INIT_LOAD_FILES_ERR,
    XO_CFI_ORBIT_ID_INIT_TIME_INPUT_INCORR_ERR,
    XO_CFI_ORBIT_ID_INIT_TIME_CONVERSION_WITH_REF_ERR,
    XO_CFI_ORBIT_ID_INIT_NO_ENOUGH_DATA_ERR,
    XO_CFI_ORBIT_ID_INIT_INTERPOL_INIT_ANX_ERR,
    XO_CFI_ORBIT_ID_INIT_CALC_ORBIT_ERR,
    XO_CFI_ORBIT_ID_INIT_TIME_CONVERSION_ERR,
    XO_CFI_ORBIT_ID_INIT_INTERPOL_INIT_ERR,
    XO_CFI_ORBIT_ID_INIT_PROPAG_INIT_ERR,
    XO_CFI_ORBIT_ID_INIT_DISCARDED_OSV_WARN,
    XO_CFI_ORBIT_ID_INIT_ONLY_FIRST_OSF_WARN,
    XO_CFI_ORBIT_ID_INIT_WRONG_TIME_CORRELATIONS_WARN,
    XO_CFI_ORBIT_ID_INIT_FILE_TIME_CORR_OVERRIDE_WARN,
    XO_CFI_ORBIT_ID_INIT_INTERPOL_INIT_ANX_WARN,
    XO_CFI_ORBIT_ID_INIT_ORBIT_ID_CHANGE_ERR,
    XO_CFI_ORBIT_ID_INIT_DEFAULT_ORBIT_WARN,
    XO_CFI_ORBIT_ID_INIT_DORIS_TYPE_WARN,
    XO_CFI_ORBIT_ID_INIT_INVALID_FILE_TYPE_ERR, // AN-583

    XO_CFI_ORBIT_ID_INIT_ONLY_FIRST_TLE_WARN = XL_FIRST_E, // AN-583; AN-577 jump to XL_FIRST_E since the function has many errors and the limit set by WARN_BIT to 30 is exceeded
    XO_CFI_ORBIT_ID_INIT_TLE_AUTO_ERR, //AN-583
    XO_CFI_ORBIT_ID_INIT_ANALYZE_ORBIT_DIAGNOSTICS_ERR, //AN-577
    XO_CFI_ORBIT_ID_INIT_GAPS_WARN, //AN-577
    XO_CFI_ORBIT_ID_INIT_DUPLICATED_GOING_BACK_OSV_WARN, //AN-577
    XO_CFI_ORBIT_ID_INIT_INCONSISTENT_ORBIT_NUM_WARN, //AN-577
    XO_CFI_ORBIT_ID_INIT_MEO_GAP_WARN // AN-867

  } XO_CFI_Orbit_id_init_err_enum;

  /* Error codes list of xo_orbit_init_file */
  /* -------------------------------------- */
  typedef enum
  {
    XO_CFI_ORBIT_INIT_FILE_SAT_ERR = XL_ZERO_E,
    XO_CFI_ORBIT_INIT_FILE_GEO_SAT_ERR,
    XO_CFI_ORBIT_INIT_FILE_FLAG_ERR,
    XO_CFI_ORBIT_INIT_FILE_TIME_STATUS_ERR,
    XO_CFI_ORBIT_INIT_FILE_ORBIT_STATUS_ERR,
    XO_CFI_ORBIT_INIT_FILE_MEMORY_ERR,
    XO_CFI_ORBIT_INIT_FILE_INPUT_FILES_ERR,
    XO_CFI_ORBIT_INIT_FILE_WRONG_OSF_FILE_FORMAT_ERR,
    XO_CFI_ORBIT_INIT_FILE_TIME_INPUT_INCORR_ERR,
    XO_CFI_ORBIT_INIT_FILE_DORIS_INIT_ERR,

    XO_CFI_ORBIT_INIT_FILE_TIME_CONVERSION_ERR,
    XO_CFI_ORBIT_INIT_FILE_READ_FILES_ERR,
    XO_CFI_ORBIT_INIT_FILE_TLE_AUTO_ERR,
    XO_CFI_ORBIT_INIT_FILE_NO_ENOUGH_DATA_ERR,
    XO_CFI_ORBIT_INIT_FILE_INTERPOL_INIT_ANX_ERR,
    XO_CFI_ORBIT_INIT_FILE_CALC_ORBIT_ERR,
    XO_CFI_ORBIT_INIT_FILE_PROPAG_INIT_ERR,
    XO_CFI_ORBIT_INIT_FILE_INTERPOL_INIT_ERR,
    XO_CFI_ORBIT_INIT_FILE_INTERPOL_INIT_ANX_WARN,
    XO_CFI_ORBIT_INIT_FILE_DORIS_INIT_WARN,
    XO_CFI_ORBIT_INIT_FILE_READ_FILES_WARN,

    XO_CFI_ORBIT_INIT_FILE_WRONG_TIME_CORRELATIONS_WARN,
    XO_CFI_ORBIT_INIT_FILE_FILE_TIME_CORR_OVERRIDE_WARN,
    XO_CFI_ORBIT_INIT_FILE_TIME_CONVERSION_WITH_REF_ERR,
    XO_CFI_ORBIT_INIT_FILE_DISCARDED_OSV_WARN,
    XO_CFI_ORBIT_INIT_FILE_GEO_AND_TLE_WARN,
    XO_CFI_ORBIT_INIT_FILE_READ_OSF_ERR,
    XO_CFI_ORBIT_INIT_FILE_ORBIT_ID_CHANGE_ERR,

    XO_CFI_ORBIT_INIT_FILE_DORIS_TYPE_WARN = XL_FIRST_E, //AN-577 jump to XL_FIRST_E since the function has many errors and the limit set by WARN_BIT to 30 is exceeded
    XO_CFI_ORBIT_INIT_FILE_DEFAULT_ORBIT_WARN,
    XO_CFI_ORBIT_INIT_FILE_ANALYZE_ORBIT_DIAGNOSTICS_ERR, //AN-577
    XO_CFI_ORBIT_INIT_FILE_GAPS_WARN, //AN-577
    XO_CFI_ORBIT_INIT_FILE_DUPLICATED_GOING_BACK_OSV_WARN, //AN-577
    XO_CFI_ORBIT_INIT_FILE_INCONSISTENT_ORBIT_NUM_WARN, //AN-577
    XO_CFI_ORBIT_INIT_FILE_MEO_GAP_WARN // AN-867

  } XO_CFI_Orbit_Init_File_err_enum;

  /* Error codes list of xo_orbit_init_file_precise */
  /* ---------------------------------------------- */
  typedef enum
  {
    XO_CFI_ORBIT_INIT_FILE_PRECISE_NOT_ALLOWED_FILE_TYPE_ERR = XL_ZERO_E,
    XO_CFI_ORBIT_INIT_FILE_PRECISE_INIT_FILE_ERR,
    XO_CFI_ORBIT_INIT_FILE_PRECISE_PROPAG_INIT_ERR,
    XO_CFI_ORBIT_INIT_FILE_PRECISE_INPUT_FILES_ERR,
    XO_CFI_ORBIT_INIT_FILE_PRECISE_PRECISE_PARAMS_ERR
  } XO_CFI_Orbit_Init_File__Precise_err_enum;

  /* Error codes list of xo_orbit_init_geo (ANR-353) */
  /* ----------------------------------------------- */
  typedef enum
  {
    XO_CFI_ORBIT_INIT_GEO_SAT_ERR = XL_ZERO_E,
    XO_CFI_ORBIT_INIT_GEO_LON_ERR,
    XO_CFI_ORBIT_INIT_GEO_TYPE_ERR,
    XO_CFI_ORBIT_INIT_GEO_TIME_STATUS_ERR,
    XO_CFI_ORBIT_INIT_GEO_STATUS_ERR,
    XO_CFI_ORBIT_INIT_GEO_MEMORY_ERR,
    XO_CFI_ORBIT_INIT_GEO_GEOD_TO_CART_ERR,
    XO_CFI_ORBIT_INIT_GEO_PROPAG_INIT_ERR

  } XO_CFI_Orbit_init_geo_err_enum;

  /* Error codes list of xo_orbit_close */
  /* ---------------------------------- */
  typedef enum
  {
    XO_CFI_ORBIT_CLOSE_WRONG_ID_ERR = XL_ZERO_E

  } XO_CFI_Orbit_close_err_enum;

  /* Error codes list of  xo_osv_compute */
  /* ---------------------------------- */
  typedef enum
  {
    XO_CFI_OSV_COMPUTE_PROPAG_INIT_ERR = XL_ZERO_E,
    XO_CFI_OSV_COMPUTE_NOT_INTERNAL_DATA_ERR,
    XO_CFI_OSV_COMPUTE_PROPAG_ERR,
    XO_CFI_OSV_COMPUTE_INTERPOL_ERR,
    XO_CFI_OSV_COMPUTE_PROPAG_WARN,
    XO_CFI_OSV_COMPUTE_INTERPOL_WARN,
    XO_CFI_OSV_COMPUTE_USER_OSV_LIST_ERR, /* AN-497 */
    XO_CFI_OSV_COMPUTE_TLE_OSV_GAP_WARN, /* AN-660 */
    XO_CFI_OSV_COMPUTE_INTERPOL_ACCELERATION_WARN /* AN-696 */
  } XO_CFI_Osv_compute_err_enum;

  /* Error codes list of xo_osv_compute_extra */
  /* ---------------------------------------- */

  typedef enum
  {
    XO_CFI_OSV_COMPUTE_EXTRA_PROPAG_INIT_ERR = XL_ZERO_E,
    XO_CFI_OSV_COMPUTE_EXTRA_NOT_INTERNAL_DATA_ERR,
    XO_CFI_OSV_COMPUTE_EXTRA_PROPAG_ERR,
    XO_CFI_OSV_COMPUTE_EXTRA_INTERPOL_ERR
  } XO_CFI_osv_compute_extra_err_enum;

  /* Error codes list of xo_propag_config */
  /* ------------------------------------ */
  typedef enum
  {
    XO_CFI_PROPAG_CONFIG_INIT_ERR = XL_ZERO_E
  } XO_CFI_Propag_config_err_enum;

  /* Error codes list of xo_orbit_to_time */
  /* ------------------------------------ */
  typedef enum
  {
    XO_CFI_ORBIT_TO_TIME_FLAG_ERR = XL_ZERO_E,
    XO_CFI_ORBIT_TO_TIME_ORB_NUM_1ST_ERR,
    XO_CFI_ORBIT_TO_TIME_ORBIT_STATUS_ERR,
    XO_CFI_ORBIT_TO_TIME_SEC_MICROSEC_ERR,
    XO_CFI_ORBIT_TO_TIME_ORB_ERR,
    XO_CFI_ORBIT_TO_TIME_SEC_ERR,
    XO_CFI_ORBIT_TO_TIME_MICROSEC_ERR,
    XO_CFI_ORBIT_TO_TIME_COMPUTE_ERR,
    XO_CFI_ORBIT_TO_TIME_TIME_CHANGE_ERR,
    XO_CFI_ORBIT_TO_TIME_GEO_SAT_ERR,
    XO_CFI_ORBIT_TO_TIME_SP3_ERR
  } XO_CFI_Orbit_to_time_err_enum;

  /* Error codes list of xo_time_to_orbit */
  /* ------------------------------------ */
  typedef enum
  {
    XO_CFI_TIME_TO_ORBIT_FLAG_ERR = XL_ZERO_E,
    XO_CFI_TIME_TO_ORBIT_ORBIT_STATUS_ERR,
    XO_CFI_TIME_TO_ORBIT_TIME_ERR,
    XO_CFI_TIME_TO_ORBIT_BEFORE_RANGE_ERR,
    XO_CFI_TIME_TO_ORBIT_COMPUTE_ERR,
    XO_CFI_TIME_TO_ORBIT_WRONG_ORBIT_MODE_ERR,
    XO_CFI_TIME_TO_ORBIT_TIME_BEFORE_RANGE_WARN,
    XO_CFI_TIME_TO_ORBIT_TIME_AFTER_RANGE_WARN,
    XO_CFI_TIME_TO_ORBIT_COMPUTE_WARN,
    XO_CFI_TIME_TO_ORBIT_GEO_SAT_ERR,
    XO_CFI_TIME_TO_ORBIT_SP3_ERR // AN-550
  } XO_CFI_Time_to_orbit_err_enum;

  /* Error codes list of xo_orbit_rel_from_abs */
  /* ----------------------------------------- */
  typedef enum
  {
    XO_CFI_ORBIT_REL_FROM_ABS_ORBIT_INIT_ERR = XL_ZERO_E,
    XO_CFI_ORBIT_REL_FROM_ABS_ORBIT_WRONG_MODE_ERR,
    XO_CFI_ORBIT_REL_FROM_ABS_WRONG_ORBIT

  } XO_CFI_Orbit_rel_from_abs_err_enum;

  /* Error codes list of xo_orbit_abs_from_rel */
  /* ----------------------------------------- */
  typedef enum
  {
    XO_CFI_ORBIT_ABS_FROM_REL_ORBIT_INIT_ERR = XL_ZERO_E,
    XO_CFI_ORBIT_ABS_FROM_REL_ORBIT_WRONG_MODE_ERR,
    XO_CFI_ORBIT_ABS_FROM_REL_INPUT_PARAMETER_ERR

  } XO_CFI_Orbit_abs_from_rel_err_enum;

  /* Error codes list of xo_orbit_abs_from_phase */
  /* ------------------------------------------- */
  typedef enum
  {
    XO_CFI_ORBIT_ABS_FROM_PHASE_ORBIT_INIT_ERR = XL_ZERO_E,
    XO_CFI_ORBIT_ABS_FROM_PHASE_ORBIT_WRONG_MODE_ERR,
    XO_CFI_ORBIT_ABS_FROM_PHASE_INPUT_PARAMETER_ERR

  } XO_CFI_Orbit_abs_from_phase_err_enum;

  /* Error codes list of xo_orbit_info */
  /* --------------------------------- */
  typedef enum
  {
    XO_CFI_ORBIT_INFO_ORBIT_INIT_ERR = XL_ZERO_E,
    XO_CFI_ORBIT_INFO_OUT_OF_LIMITS_ERR,
    XO_CFI_ORBIT_INFO_RESULTS_ERR,
    XO_CFI_ORBIT_INFO_GEO_SAT_ERR,
    XO_CFI_ORBIT_INFO_ORBIT_MODE_NOT_ALLOWED_ERR, // AN-550
    XO_CFI_ORBIT_INFO_SMX_ERR, // AN-523
    XO_CFI_ORBIT_INFO_NO_SMX_WARN // AN-523

  } XO_CFI_Orbit_info_err_enum;

  /* Error codes list of xo_orbit_info_configure (AN-523) */
  /* ---------------------------------------------------- */
  typedef enum
  {
    XO_CFI_ORBIT_INFO_CONFIGURE_ORBIT_INIT_STATUS_ERR = XL_ZERO_E,
    XO_CFI_ORBIT_INFO_CONFIGURE_WRONG_OPTION_ERR,
    XO_CFI_ORBIT_INFO_CONFIGURE_WRONG_ITEM_ERR
  } XO_CFI_Orbit_info_configure_err_enum;

  /* Error codes list of xo_osv_to_tle */
  /* --------------------------------- */
  typedef enum
  {
    XO_CFI_OSV_TO_TLE_WRONG_FILE_MODEL_ERR = XL_ZERO_E,
    XO_CFI_OSV_TO_TLE_WRONG_INPUT_INTERVAL_ERR,
    XO_CFI_OSV_TO_TLE_TIME_TRANS_ERR,
    XO_CFI_OSV_TO_TLE_WRONG_TIME_MODEL_ERR,
    XO_CFI_OSV_TO_TLE_CHANGE_CS_ERR,
    XO_CFI_OSV_TO_TLE_CART_TO_KEPLER_ERR,
    XO_CFI_OSV_TO_TLE_FIT_ERR /*AN-497*/

  } XO_CFI_osv_to_tle_err_enum;

  /* Error codes list of xo_run_init */
  /* ------------------------------- */
  typedef enum
  {
    XO_CFI_RUN_INIT_STATUS_ERR = XL_ZERO_E,
    XO_CFI_RUN_INIT_MEMORY_ERR,
    XO_CFI_RUN_INIT_INCONSISTENCY_ERR
  } XO_CFI_run_init_err_enum;

  /* Error codes list of xo_gen_oef */
  /* ------------------------------ */
  typedef enum
  {
    XO_CFI_GEN_OEF_OPEN_FILE_ERR = XL_ZERO_E,
    XO_CFI_GEN_OEF_COPY_FILE_ERR,
    XO_CFI_GEN_OEF_COPY_NODE_ERR,
    XO_CFI_GEN_OEF_CLOSE_ERR,
    XO_CFI_GEN_OEF_READ_OSF_ERR,
    XO_CFI_GEN_OEF_READ_POF_ERR,
    XO_CFI_GEN_OEF_WRITE_ERR,
    XO_CFI_GEN_OEF_CURRENT_TIME_ERR,
    XO_CFI_GEN_OEF_SET_SCHEMA_WARN
  } XO_CFI_Gen_oef_err_enum;

  /* Error codes list of xo_gen_osf_create */
  /* ------------------------------------- */
  typedef enum
  {
    XO_CFI_GEN_OSF_CREATE_INPUTS_ERR = XL_ZERO_E,
    XO_CFI_GEN_OSF_CREATE_TIME_INIT_ERR,
    XO_CFI_GEN_OSF_CREATE_ALLOC_ERR,
    XO_CFI_GEN_OSF_CREATE_DRIFT_MODE_ERR,
    XO_CFI_GEN_OSF_CREATE_DRIFT_CALC_ERR,
    XO_CFI_GEN_OSF_CREATE_UTC_CALC_ERR,
    XO_CFI_GEN_OSF_CREATE_TAI_CALC_ERR,
    XO_CFI_GEN_OSF_CREATE_UT1_CALC_ERR,
    XO_CFI_GEN_OSF_CREATE_GET_FH_ERR,
    XO_CFI_GEN_OSF_CREATE_GET_SCHEMA_ERR,
    XO_CFI_GEN_OSF_CREATE_WRITE_ERR,
    XO_CFI_GEN_OSF_CREATE_DEPRECATED_WARN

  } XO_CFI_Gen_osf_create_err_enum;

  /* Error codes list of xo_gen_osf_append_orbit_change */
  /* -------------------------------------------------- */
  typedef enum
  {
    XO_CFI_GEN_OSF_APPEND_INPUTS_ERR = XL_ZERO_E,
    XO_CFI_GEN_OSF_APPEND_TIME_INIT_ERR,
    XO_CFI_GEN_OSF_APPEND_READ_IN_OSF_ERR,
    XO_CFI_GEN_OSF_APPEND_ANX_LONG_WARN, /* AN-501 */
    XO_CFI_GEN_OSF_APPEND_MLST_WARN, /* AN-501 */
    XO_CFI_GEN_OSF_APPEND_DRIFT_MODE_ERR,
    XO_CFI_GEN_OSF_APPEND_DRIFT_CALC_ERR,
    XO_CFI_GEN_OSF_APPEND_UTC_CALC_ERR,
    XO_CFI_GEN_OSF_APPEND_TAI_CALC_ERR,
    XO_CFI_GEN_OSF_APPEND_UT1_CALC_ERR,
    XO_CFI_GEN_OSF_APPEND_ALLOC_ERR,
    XO_CFI_GEN_OSF_APPEND_GET_FH_ERR,
    XO_CFI_GEN_OSF_APPEND_WRITE_ERR,
    XO_CFI_GEN_OSF_APPEND_DEPRECATED_WARN

  } XO_CFI_Gen_osf_append_err_enum;

  /* Error codes list of xo_gen_osf_change_repeat_cycle */
  /* -------------------------------------------------- */
  typedef enum
  {
    XO_CFI_GEN_OSF_CHANGE_INPUTS_ERR = XL_ZERO_E,
    XO_CFI_GEN_OSF_CHANGE_TIME_INIT_ERR,
    XO_CFI_GEN_OSF_CHANGE_READ_IN_OSF_ERR,
    XO_CFI_GEN_OSF_CHANGE_DRIFT_MODE_ERR,
    XO_CFI_GEN_OSF_CHANGE_DRIFT_CALC_ERR,
    XO_CFI_GEN_OSF_CHANGE_NO_TRANSITION_ERR,
    XO_CFI_GEN_OSF_CHANGE_UTC_CALC_ERR,
    XO_CFI_GEN_OSF_CHANGE_TAI_CALC_ERR,
    XO_CFI_GEN_OSF_CHANGE_UT1_CALC_ERR,
    XO_CFI_GEN_OSF_CHANGE_ALLOC_ERR,
    XO_CFI_GEN_OSF_CHANGE_GET_FH_ERR,
    XO_CFI_GEN_OSF_CHANGE_WRITE_ERR,
    XO_CFI_GEN_OSF_CHANGE_DEPRECATED_WARN,
    XO_CFI_GEN_OSF_CHANGE_SELECT_SCHEMA_ERR // AN-721

  } XO_CFI_Gen_osf_change_repeat_cycle_err_enum;

  /* Error codes list of xo_gen_osf_add_drift_cycle */
  /* ---------------------------------------------- */
  typedef enum
  {
    XO_CFI_GEN_OSF_DRIFT_INPUTS_ERR = XL_ZERO_E,
    XO_CFI_GEN_OSF_DRIFT_TIME_INIT_ERR,
    XO_CFI_GEN_OSF_DRIFT_READ_IN_OSF_ERR,
    XO_CFI_GEN_OSF_DRIFT_NO_ADD_ERR,
    XO_CFI_GEN_OSF_DRIFT_INCL_CALC_ERR,
    XO_CFI_GEN_OSF_DRIFT_NOT_FOUND_ERR,
    XO_CFI_GEN_OSF_DRIFT_UTC_CALC_ERR,
    XO_CFI_GEN_OSF_DRIFT_TAI_CALC_ERR,
    XO_CFI_GEN_OSF_DRIFT_UT1_CALC_ERR,
    XO_CFI_GEN_OSF_DRIFT_ALLOC_ERR,
    XO_CFI_GEN_OSF_DRIFT_GET_FH_ERR,
    XO_CFI_GEN_OSF_DRIFT_WRITE_ERR,
    XO_CFI_GEN_OSF_DRIFT_COMPUTE_DRIFT_ERR // AN-739

  } XO_CFI_Gen_osf_add_drift_cycle_err_enum;

  /* Error codes list of xo_gen_pof */
  /* ------------------------------ */
  typedef enum
  {
    XO_CFI_GEN_POF_WRONG_SAT_ID_ERR = XL_ZERO_E,
    XO_CFI_GEN_POF_WRONG_FLAG_ERR,
    XO_CFI_GEN_POF_TIME_INIT_ERR,
    XO_CFI_GEN_POF_TIME_INITIALIZATION_ERR,
    XO_CFI_GEN_POF_ORBIT_INIT_FILE_ERR,
    XO_CFI_GEN_POF_PROPAG_INIT_ERR,
    XO_CFI_GEN_POF_INTERPOL_INIT_ERR,
    XO_CFI_GEN_POF_INTERNAL1_ERR,
    XO_CFI_GEN_POF_CALCULATING_STATE_VECTOR_ERR,
    XO_CFI_GEN_POF_CLOSE_ERR,
    XO_CFI_GEN_POF_TIME_TRANS_ERR,
    XO_CFI_GEN_POF_MEMORY_ERR,
    XO_CFI_GEN_POF_GET_FH_ERR,
    XO_CFI_GEN_POF_WRITE_ERR,
    XO_CFI_GEN_POF_READ_PRECISE_FILE_ERR,
    XO_CFI_GEN_POF_ORBIT_TIME_ERR,
    XO_CFI_GEN_POF_TIME_ID_NOT_ALLOWED_ERR,
    XO_CFI_GEN_POF_TIME_TLE_INITIALIZATION_ERR //AN-649
  } XO_CFI_Gen_pof_err_enum;

  /* Error codes list of xo_gen_rof */
  /* ------------------------------ */
  typedef enum
  {
    XO_CFI_GEN_ROF_WRONG_SAT_ID_ERR = XL_ZERO_E,
    XO_CFI_GEN_ROF_WRONG_FLAG_ERR,
    XO_CFI_GEN_ROF_TIME_INIT_ERR,
    XO_CFI_GEN_ROF_TIME_INITIALIZATION_ERR,
    XO_CFI_GEN_ROF_ORBIT_INIT_FILE_ERR,
    XO_CFI_GEN_ROF_PROPAG_INIT_ERR,
    XO_CFI_GEN_ROF_TIME_ORBIT_ERR,
    XO_CFI_GEN_ROF_INTERPOL_INIT_ERR,
    XO_CFI_GEN_ROF_CALCULATING_STATE_VECTOR_ERR,
    XO_CFI_GEN_ROF_TIME_ERR,
    XO_CFI_GEN_ROF_TIME_TO_EXTERNAL_ERR,
    XO_CFI_GEN_ROF_WRITE_ERR,
    XO_CFI_GEN_ROF_CLOSE_ERR,
    XO_CFI_GEN_ROF_MEMORY_ERR,
    XO_CFI_GEN_ROF_GET_FH_ERR,
    XO_CFI_GEN_ROF_WRONG_INTERVAL_WARN,
    XO_CFI_GEN_ROF_WRONG_OSV_INTERVAL_ERR, // AN-688
    XO_CFI_GEN_ROF_READ_PRECISE_FILE_ERR,
    XO_CFI_GEN_ROF_TIME_ID_NOT_ALLOWED_ERR,
    XO_CFI_GEN_ROF_TIME_TLE_INITIALIZATION_ERR, // AN-649
    XO_CFI_GEN_ROF_SELECT_SCHEMA_ERR // AN-721
  } XO_CFI_Gen_rof_err_enum;

  /* Error codes list of xo_gen_rof_prototype */
  /* ---------------------------------------- */
  typedef enum
  {
    XO_CFI_GEN_ROF_PROTOTYPE_WRONG_SAT_ID_ERR = XL_ZERO_E,
    XO_CFI_GEN_ROF_PROTOTYPE_TIME_ID_ERR,
    XO_CFI_GEN_ROF_PROTOTYPE_WRONG_FLAG_ERR,
    XO_CFI_GEN_ROF_PROTOTYPE_ORBIT_INIT_DEF_ERR,
    XO_CFI_GEN_ROF_PROTOTYPE_PROPAG_INIT_ERR,
    XO_CFI_GEN_ROF_PROTOTYPE_CALCULATING_STATE_VECTOR_ERR,
    XO_CFI_GEN_ROF_PROTOTYPE_TIME_ERR,
    XO_CFI_GEN_ROF_PROTOTYPE_CLOSE_ERR,
    XO_CFI_GEN_ROF_PROTOTYPE_GET_FH_ERR,
    XO_CFI_GEN_ROF_PROTOTYPE_MEMORY_ERR,
    XO_CFI_GEN_ROF_PROTOTYPE_WRITE_ERR,
    XO_CFI_GEN_ROF_PROTOTYPE_SELECT_SCHEMA_ERR // AN-721
  } XO_CFI_Gen_rof_prototype_err_enum;

  /* Error codes list of xo_gen_dnf */
  /* ------------------------------ */
  typedef enum
  {
    XO_CFI_GEN_DNF_WRONG_SAT_ID_ERR = XL_ZERO_E,
    XO_CFI_GEN_DNF_WRONG_FLAG_ERR,
    XO_CFI_GEN_DNF_TIME_INIT_ERR,
    XO_CFI_GEN_DNF_TIME_INITIALIZATION_ERR,
    XO_CFI_GEN_DNF_ORBIT_INIT_FILE_ERR,
    XO_CFI_GEN_DNF_PROPAG_INIT_ERR,
    XO_CFI_GEN_DNF_INTERPOL_INIT_ERR,
    XO_CFI_GEN_DNF_TIME_ORBIT_ERR,
    XO_CFI_GEN_DNF_TIME_ERR,
    XO_CFI_GEN_DNF_MEMORY_ERR,
    XO_CFI_GEN_DNF_CALCULATING_STATE_VECTOR_ERR,
    XO_CFI_GEN_DNF_READ_CONTROL_FILE_ERR,
    XO_CFI_GEN_DNF_CORRECT_OSV_ERR,
    XO_CFI_GEN_DNF_CHANGE_COORD_ERR,
    XO_CFI_GEN_DNF_COMPUTE_HEADER_ERR,
    XO_CFI_GEN_DNF_CLOSE_ERR,
    XO_CFI_GEN_DNF_WRITE_FILE_ERR,
    XO_CFI_GEN_DNF_WRONG_INTERVAL_WARN,
    XO_CFI_GEN_DNF_READ_PRECISE_FILE_ERR
  } XO_CFI_Gen_dnf_err_enum;

  /* Error codes list of xo_gen_tle */
  /* ------------------------------ */
  typedef enum
  {
    XO_CFI_GEN_TLE_WRONG_SAT_ID_ERR = XL_ZERO_E,
    XO_CFI_GEN_TLE_WRONG_FLAG_ERR,
    XO_CFI_GEN_TLE_WRONG_FIT_MODE_ERR,
    XO_CFI_GEN_TLE_TIME_INITIALIZATION_ERR,
    XO_CFI_GEN_TLE_ORBIT_INIT_FILE_ERR,
    XO_CFI_GEN_TLE_MEMORY_ERR,
    XO_CFI_GEN_TLE_OSV_TO_TLE_ERR,
    XO_CFI_GEN_TLE_CLOSE_ERR,
    XO_CFI_GEN_TLE_WRITE_FILE_ERR,
    XO_CFI_GEN_TLE_OSV_COMPUTE_ERR,
    XO_CFI_GEN_TLE_TIME_TO_ORBIT_ERR,
    XO_CFI_GEN_TLE_INVALID_INIT_MODE_ERR,
    XO_CFI_GEN_TLE_FILENAME_WARN,
    XO_CFI_GEN_TLE_INPUT_FILES_ERR // AN-595
  } XO_CFI_Gen_tle_err_enum;

  /* Error codes list of xo_check_osf */
  /* -------------------------------- */
  typedef enum
  {
    XO_CFI_CHECK_OSF_TIME_INIT_ERR = XL_ZERO_E,
    XO_CFI_CHECK_OSF_OSF_READ_ERR,
    XO_CFI_CHECK_OSF_WRONG_TRANSITION_ERR,
    XO_CFI_CHECK_OSF_ORBIT_INIT_ERR,
    XO_CFI_CHECK_OSF_ORBIT_INFO_ERR,
    XO_CFI_CHECK_OSF_UTC_WARN,
    XO_CFI_CHECK_OSF_ANX_LONG_WARN,
    XO_CFI_CHECK_OSF_MLST_WARN,
    XO_CFI_CHECK_OSF_OSC_A_WARN,
    XO_CFI_CHECK_OSF_OSC_I_WARN,
    XO_CFI_CHECK_OSF_TNOD_WARN
  } XO_CFI_Gen_osf_err_enum;

  /* Error codes list of xo_check_oef */
  /* -------------------------------- */
  typedef enum
  {
    XO_CFI_CHECK_OEF_ORBIT_INIT_ERR = XL_ZERO_E,
    XO_CFI_CHECK_OEF_ORBIT_INFO_ERR,
    XO_CFI_CHECK_OEF_MEM_ERR,
    XO_CFI_CHECK_OEF_UTC_WARN,
    XO_CFI_CHECK_OEF_ANX_LONG_WARN,
    XO_CFI_CHECK_OEF_MLST_WARN,
    XO_CFI_CHECK_OEF_OSC_A_WARN,
    XO_CFI_CHECK_OEF_OSC_I_WARN,
    XO_CFI_CHECK_OEF_TNOD_WARN
  } XO_CFI_Check_oef_err_enum;

  /* Error code list of xo_position_on_orbit_to_time */
  /* ----------------------------------------------- */
  typedef enum
  {
    XO_CFI_POSITION_ON_ORBIT_ORBIT_INIT_STATUS_ERR = XL_ZERO_E,
    XO_CFI_POSITION_ON_ORBIT_ORBIT_INFO_ERR,
    XO_CFI_POSITION_ON_ORBIT_ORBIT_TO_TIME_ERR,
    XO_CFI_POSITION_ON_ORBIT_OSV_COMPUTE_ERR,
    XO_CFI_POSITION_ON_ORBIT_POSITION_ON_ORBIT_ERR,
    XO_CFI_POSITION_ON_ORBIT_MAX_ITERATIONS_ERR,
    XO_CFI_POSITION_ON_ORBIT_PROCESSING_TIME_ERR
  } XO_CFI_Position_on_orbit_to_time_err_enum;

  /* Error code list of xo_orbit_data_filter */
  /* ----------------------------------------*/
  typedef enum
  {
    XO_CFI_ORBIT_DATA_FILTER_UNKNOWN_FILTER_ERR = XL_ZERO_E,
    XO_CFI_ORBIT_DATA_FILTER_NO_OSV_ERR,
    XO_CFI_ORBIT_DATA_FILTER_FILE_ERR,
    XO_CFI_ORBIT_DATA_FILTER_INTERPOL_ERR,
    XO_CFI_ORBIT_DATA_FILTER_ANALYSE_OSV_SAMPLE_ERR,
    XO_CFI_ORBIT_DATA_FILTER_SAMPLES_ERR,
    XO_CFI_ORBIT_DATA_FILTER_FILTER_OUTLIERS_ERR // AN-551
  } XO_CFI_Orbit_data_filter_err_enum;

  /* Error code list of xo_orbit_id_init_data_close */
  /* ----------------------------------------------*/
  typedef enum
  {
    XO_CFI_ORBIT_ID_INIT_DATA_CLOSE_WRONG_TYPE_ERR = XL_ZERO_E

  } XO_CFI_Orbit_id_init_data_close_err_enum;

  /* Error code list of xo_orbit_id_change ANR-527  */
  /* -----------------------------------------------*/
  typedef enum
  {
    XO_CFI_ORBIT_ID_CHANGE_MODE_ERR = XL_ZERO_E,
    XO_CFI_ORBIT_ID_CHANGE_CHANGE_DATA_ERR,
    XO_CFI_ORBIT_ID_CHANGE_ID_NULL_ERR,
    XO_CFI_ORBIT_ID_CHANGE_TIME_ID_ERR,
    XO_CFI_ORBIT_ID_CHANGE_ORBIT_ID_ERR,
    XO_CFI_ORBIT_ID_CHANGE_PROPAG_ERR,
    XO_CFI_ORBIT_ID_CHANGE_PROP_TIME_ERR,
    XO_CFI_ORBIT_ID_CHANGE_CLOSE_ORBIT_ID_ERR,
    XO_CFI_ORBIT_ID_CHANGE_CLOSE_TIME_ID_ERR
  } XO_CFI_Orbit_id_change_err_enum;

  // AN-632: Error codes list of xo_osv_check
  typedef enum
  {
    XO_CFI_OSV_CHECK_CHANGE_CART_CS_ERR = XL_ZERO_E,
    XO_CFI_OSV_CHECK_ORBIT_RADIUS_ZERO_ERR,
    XO_CFI_OSV_CHECK_ORBIT_VEL_ZERO_ERR,
    XO_CFI_OSV_CHECK_SEMI_MAJOR_AXIS_ERR,
    XO_CFI_OSV_CHECK_INCLINATION_WARN,
    XO_CFI_OSV_CHECK_SAT_ARRAY_ERR,
    XO_CFI_OSV_CHECK_LOOSE_TOLERANCE_ERR,
    XO_CFI_OSV_CHECK_TIGHT_TOLERANCE_WARN
  } XO_CFI_osv_check_err_enum;

  //  ANR-692: Error codes list of xo_orbit_id_check
  typedef enum
  {
    XO_CFI_ORBIT_ID_CHECK_ORBIT_ID_UNINITIALIZED_ERR = XL_ZERO_E,
    XO_CFI_ORBIT_ID_CHECK_NULL_OUTPUT_ERR,
    XO_CFI_ORBIT_ID_CHECK_NO_OSVS_ERR,
    XO_CFI_ORBIT_ID_CHECK_NO_TIME_ID_ERR,
    XO_CFI_ORBIT_ID_CHECK_FILE_DIAGNOSTICS_ERR,
    XO_CFI_ORBIT_ID_CHECK_OSV_CHECK_ERR,
    XO_CFI_ORBIT_ID_CHECK_MEM_ALLOC_ERR
  } XO_CFI_orbit_id_check_err_enum;

  /*******************************************************************************
*
* Data structures
*
*******************************************************************************/

  /* Orbit Id. Structure */
  typedef struct
  {
    void* ee_id;
  } xo_orbit_id;

  /* structures for orbit_id accessors */
  typedef xd_osv_rec xo_osv_rec;

  typedef struct
  {
    long abs_orbit;
    double tanx;
    double tnod;
  } xo_anx_extra_info;

  typedef struct
  {
    long abs_orbit;
    long rel_orbit;
    long cycle_num;
    long phase_num;
  } xo_mission_info;

  typedef xd_mlst_nonlinear_drift xo_mlst_nonlinear_drift;

  typedef struct
  {
    long drift_mode;
    double inclination;
    long rep_cycle;
    long cycle_len;
    double ANX_long;
    double mlst; /* hours */
    double mlst_drift; /* s/day */
    xo_mlst_nonlinear_drift mlst_nonlinear_drift;
  } xo_ref_orbit_info;

  typedef struct
  {
    double anx_tai;
    double anx_utc;
    double anx_ut1;
    double time_ref_of;
    double anx_pos[XO_MAX_LENGTH_3];
    double anx_vel[XO_MAX_LENGTH_3];
    double kepl[XO_MAX_LENGTH_6];
    double tnod;
  } xo_anx_info;

  typedef struct
  {
    xo_mission_info mission_info;
    xo_ref_orbit_info ref_orbit_info;
    xo_anx_info anx_info;
  } xo_osf_records;

  typedef struct
  {
    long time_ref;
    double start;
    double stop;
  } xo_validity_time;

  /* structures for propag_id accessors */

  typedef struct
  {
    XO_Time_ref_enum time_ref; /* Time reference in use */
    xo_validity_time val_time; /* validity propagation time range in UT1 time */
    long abs_orbit; /* Predicted Absolute orbit (only for xo_propag_init_def and init_file) */
    double time_since_anx; /* Time since ANX           (only for xo_propag_init_def and init_file) */
    double time; /* Predicted time (UT1) */
    double pos[XO_MAX_LENGTH_3]; /* Osculating position vector at pred. time (EF) */
    double vel[XO_MAX_LENGTH_3]; /* Osculating velocity vector at pred. time (EF) */
    double acc[XO_MAX_LENGTH_3]; /* Osculating acceleration vector at pred. time (EF) */
    double x[XO_MAX_LENGTH_6]; /* Osculating keplerian elements at pred. time (TOD) */

  } xo_uni_propag; /* Universal model internal propagation structure */

  typedef struct
  {
    long double_propag_flag;
    long accu_mode;
    xo_uni_propag propag_osv;
  } xo_propag_id_data;

  /* structures for interpol_id accessors */

  typedef struct
  {
    long time_ref;
    double time;
    long abs_orbit;
    double time_since_anx;
    double pos[XO_MAX_LENGTH_3];
    double vel[XO_MAX_LENGTH_3];
    double acc[XO_MAX_LENGTH_3];
    double kep[XO_MAX_LENGTH_6];
    xo_validity_time val_time; /* Interpolation Validity time range */
  } xo_interpol_id_data;

  /* Precise propag struct */
  typedef xd_propag_precise_config xo_propag_precise_config; // AN-780

  /* ANR-527 */
  typedef struct
  {
    long change_mode; // See XO_Orbit_Id_change_enum.
    xd_eocfi_file eocfi_file; // File to use as reference for orbit update
    long change_time_ref; // Time reference
    double change_time; // Time corresponding to change_orbit
    long change_orbit; // Orbit corresponding to change_time
  } xo_orbit_id_change_data;

  /* data structures for orbit_id initialization (AN-404) */
  typedef union
  {
    xd_eocfi_file_set file_set;
  } xo_orbit_id_init_data_union;

  typedef struct
  {
    long data_type; /* Enumeration: only XO_FILE_DATA */
    xo_orbit_id_init_data_union orbit_id_init_data;
    xo_orbit_id_change_data change_data;
  } xo_orbit_id_init_data;

  /* AN-353 */
  typedef struct
  {
    double gc_longitude; /* GEO geocentric longitude [deg] */
    double gd_latitude; /* GEO geodetic latitude [deg] */
    double gd_altitude; /* GEO geodetic altitude [m] */
  } xo_geo_geod_coord;

  typedef struct
  {
    xo_geo_geod_coord geod_coord; /* Geodetic coordinates of GEO satellite */
  } xo_geo_orbit_info;

  typedef struct
  {
    long init_type; /* XO_Geo_coord_enum */
    xo_geo_orbit_info geo_orbit_info;
  } xo_geo_orbit_init_data;
  /* END AN-353 */

  /* ANR-596 */
  typedef struct
  {
    long type; /* XO_Time_type_enum */
    long time_ref; // XO_Time_ref_enum
    double time;
    long orbit_type; /* abs or rel (XO_Orbit_type_enum) */
    long orbit_num;
    long cycle;
    long sec; /* after ANX */
    long msec; /* after ANX */
  } xo_time;

  typedef struct
  {
    xo_time tstart;
    xo_time tstop;
  } xo_time_interval;
  /* end ANR-596 */

  /* ANR-382 */
  typedef enum
  {
    XO_REMOVE = 0
  } XO_DATA_FILTER_ACTION;

  typedef struct
  {
    double threshold_pos;
    double threshold_vel;
    long action; /* XO_DATA_FILTER_ACTION */
  } xo_orbit_filter_outliers_cfg;

  typedef struct
  {
    long nof_OSV_in;
    long nof_OSV_filtered;
    double min_time_gap;
    double max_time_gap;
    double min_RMS_pos;
    double max_RMS_pos;
    double min_RMS_vel;
    double max_RMS_vel;
  } xo_orbit_filter_outliers_report;
  /* END ANR-382 */

  /* ANR-455 */
  typedef enum
  {
    XO_FILTER_OUTLIERS

  } XO_Orbit_data_filter_enum;

  typedef union
  {
    xo_orbit_filter_outliers_cfg outliers_cfg;
  } xo_orbit_filter_cfg_union;

  typedef struct
  {
    long type;
    xo_orbit_filter_cfg_union filter_cfg;
  } xo_orbit_filter_settings;

  typedef union
  {
    xo_orbit_filter_outliers_report outliers_report;
  } xo_orbit_filter_report_union;

  typedef struct
  {
    long type;
    xo_orbit_filter_report_union filter_report;
  } xo_orbit_filter_report;
  /* END ANR-455 */

  /* ANR-692 */
  typedef struct
  {
    long num_osvs_outside_loose_tolerance;
    long* index_osvs_outside_loose_tolerance;
    long num_osvs_outside_tight_tolerance;
    long* index_osvs_outside_tight_tolerance;
  } xo_orbit_id_check_osv_report;

  typedef struct
  {
    xd_orbit_file_diagnostics_report file_diag_report;
    xo_orbit_id_check_osv_report osv_report;
  } xo_orbit_id_check_report;
  /* END ANR-692 */

  typedef enum /* ANR-527 */
  {
    XO_ORBIT_ID_CHANGE_OSF,
    XO_ORBIT_ID_CHANGE_TIME_ORBIT
  } XO_Orbit_Id_change_enum;

  /*******************************************************************************
*
* Main CFI Functions - prototype declaration
*
*******************************************************************************/

  /* Orbit initialisation functions */

  long xo_orbit_init_def(long* sat_id, xl_model_id* model_id, xl_time_id* time_id, long* time_ref, double* time0, long* orbit0, long* drift_mode, double* ascmlst_drift, double* inclination, long* irep, long* icyc, double* rlong, double* ascmlst,
                         /* output */
                         double* val_time0,
                         double* val_time1,
                         xo_orbit_id* orbit_id,
                         long ierr[XO_NUM_ERR_ORBIT_INIT_DEF]);

  long xo_orbit_init_def_2(long* sat_id, xl_model_id* model_id, xl_time_id* time_id, long* time_ref, double* time0, long* orbit0, xo_ref_orbit_info* ref_orbit_info,
                           /* output */
                           double* val_time0,
                           double* val_time1,
                           xo_orbit_id* orbit_id,
                           long ierr[XO_NUM_ERR_ORBIT_INIT_DEF]);

  long xo_orbit_cart_init(long* sat_id, xl_model_id* model_id, xl_time_id* time_id, long* time_ref, double* time, double pos[XO_MAX_LENGTH_3], double vel[XO_MAX_LENGTH_3], long* abs_orbit,
                          /* output */
                          double* val_time0,
                          double* val_time1,
                          xo_orbit_id* orbit_id,
                          long ierr[XO_NUM_ERR_ORBIT_CART_INIT]);

  long xo_orbit_cart_init_precise(long* sat_id, xl_model_id* model_id, xl_time_id* time_id, long* time_ref, double* time, double pos[XO_MAX_LENGTH_3], double vel[XO_MAX_LENGTH_3], long* abs_orbit, xo_propag_precise_config* propag_precise_conf,
                                  /* output */
                                  double* val_time0,
                                  double* val_time1,
                                  xo_orbit_id* orbit_id,
                                  long ierr[XO_NUM_ERR_ORBIT_CART_INIT_PRECISE]);

  long xo_orbit_id_init(long* sat_id, xl_model_id* model_id, xl_time_id* time_id, long* orbit_file_mode, xo_orbit_id_init_data* orbit_data, long* time_mode, long* time_ref, double* time0, double* time1, long* orbit0, long* orbit1,
                        /* output */
                        double* val_time0,
                        double* val_time1,
                        xo_orbit_id* orbit_id,
                        long ierr[XO_NUM_ERR_ORBIT_ID_INIT]);

  long xo_orbit_init_file(long* sat_id, xl_model_id* model_id, xl_time_id* time_id, long* orbit_file_mode, long* n_files, char** orbit_file, long* time_mode, long* time_ref, double* time0, double* time1, long* orbit0, long* orbit1,
                          /* output */
                          double* val_time0,
                          double* val_time1,
                          xo_orbit_id* orbit_id,
                          long ierr[XO_NUM_ERR_ORBIT_INIT_FILE]);

  long xo_orbit_init_file_precise(long* sat_id, xl_model_id* model_id, xl_time_id* time_id, long* orbit_file_mode, long* n_files, char** orbit_file, long* time_mode, long* time_ref, double* time0, double* time1, long* orbit0, long* orbit1, xo_propag_precise_config* propag_precise_conf,
                                  /* output */
                                  double* val_time0,
                                  double* val_time1,
                                  xo_orbit_id* orbit_id,
                                  long ierr[XO_NUM_ERR_ORBIT_INIT_FILE_PRECISE]);

  /* AN-353 */
  long xo_orbit_init_geo(long* sat_id, xl_model_id* model_id, xl_time_id* time_id, xo_geo_orbit_init_data* geo_orbit_init_data,
                         /* output */
                         double* val_time0,
                         double* val_time1,
                         xo_orbit_id* orbit_id,
                         long ierr[XO_NUM_ERR_ORBIT_INIT_GEO]);

  long xo_orbit_close(xo_orbit_id* orbit_id, long ierr[XO_NUM_ERR_ORBIT_CLOSE]);

  long xo_orbit_init_status(xo_orbit_id* orbit_id);

  long xo_orbit_get_sat_id(xo_orbit_id* orbit_id);

  long xo_orbit_get_mode(xo_orbit_id* orbit_id);

  long xo_orbit_get_osv(xo_orbit_id* orbit_id,
                        long* num_rec,
                        xo_osv_rec** osv);

  long xo_orbit_set_osv(xo_orbit_id* orbit_id,
                        long* num_rec,
                        xo_osv_rec* osv);

  long xo_orbit_get_anx(xo_orbit_id* orbit_id,
                        long* num_rec,
                        xo_anx_extra_info** extra_info);

  long xo_orbit_set_anx(xo_orbit_id* orbit_id,
                        long* num_rec,
                        xo_anx_extra_info* extra_info);

  long xo_orbit_get_osf_rec(xo_orbit_id* orbit_id,
                            long* num_rec,
                            xo_osf_records** osf);

  long xo_orbit_set_osf_rec(xo_orbit_id* orbit_id,
                            long* num_rec,
                            xo_osf_records* osf);

  long xo_orbit_get_val_time(xo_orbit_id* orbit_id,
                             xo_validity_time* val_time);

  long xo_orbit_set_val_time(xo_orbit_id* orbit_id,
                             xo_validity_time* val_time);

  xl_time_id xo_orbit_get_time_id(xo_orbit_id* orbit_id);

  xl_model_id xo_orbit_get_model_id(xo_orbit_id* orbit_id);

  long xo_orbit_get_osv_compute_validity(xo_orbit_id* orbit_id,
                                         xo_validity_time* val_time);

  long xo_orbit_get_propag_mode(xo_orbit_id* orbit_id);

  long xo_orbit_get_propag_config(xo_orbit_id* orbit_id,
                                  xo_propag_id_data* data);

  long xo_orbit_get_interpol_mode(xo_orbit_id* orbit_id);

  long xo_orbit_get_interpol_config(xo_orbit_id* orbit_id,
                                    xo_interpol_id_data* data);

  long xo_orbit_id_clone(xo_orbit_id* in_id, xo_orbit_id* out_id);

  long xo_orbit_get_precise_propag_config(xo_orbit_id* orbit_id,
                                          xo_propag_precise_config* precise_conf);

  long xo_orbit_set_precise_propag_config(xo_orbit_id* orbit_id,
                                          xo_propag_precise_config* precise_conf);

  /* ANR-353 */
  long xo_orbit_get_geo_orbit_info(xo_orbit_id* orbit_id,
                                   xo_geo_orbit_info* geo_orbit_info);

  long xo_orbit_set_geo_orbit_info(xo_orbit_id* orbit_id,
                                   xo_geo_orbit_info* geo_orbit_info);
  /* END ANR-353*/

  long xo_osv_compute(xo_orbit_id* orbit_id,
                      long* mode,
                      long* time_ref,
                      double* time,
                      double pos_out[XO_MAX_LENGTH_3],
                      double vel_out[XO_MAX_LENGTH_3],
                      double acc_out[XO_MAX_LENGTH_3],
                      long ierr[XO_NUM_ERR_OSV_COMPUTE]);

  long xo_osv_compute_run(long* run_id,
                          long* mode,
                          long* time_ref,
                          double* time,
                          double pos_out[XO_MAX_LENGTH_3],
                          double vel_out[XO_MAX_LENGTH_3],
                          double acc_out[XO_MAX_LENGTH_3],
                          long ierr[XO_NUM_ERR_OSV_COMPUTE]);

  long xo_osv_compute_extra(xo_orbit_id* orbit_id,
                            long* extra_choice,
                            double model_out[XO_ORBIT_EXTRA_NUM_DEP_ELEMENTS],
                            double extra_out[XO_ORBIT_EXTRA_NUM_INDEP_ELEMENTS],
                            long ierr[XO_NUM_ERR_OSV_COMPUTE_EXTRA]);

  long xo_osv_compute_extra_run(long* run_id,
                                long* extra_choice,
                                double model_out[XO_ORBIT_EXTRA_NUM_DEP_ELEMENTS],
                                double extra_out[XO_ORBIT_EXTRA_NUM_INDEP_ELEMENTS],
                                long ierr[XO_NUM_ERR_OSV_COMPUTE_EXTRA]);

  long xo_propag_config(xo_orbit_id* orbit_id, long* propag_model, long* time_mode, long* time_ref, double* time, long* orbit, xo_propag_precise_config* propag_precise_conf,
                        /* output*/
                        double* val_time0,
                        double* val_time1,
                        long ierr[XO_NUM_ERR_PROPAG_CONFIG]);

  long xo_osv_check(xl_model_id* model_id, long sat_id, xl_time_id* time_id, long time_ref, double time, double* pos, double* vel,
                    long ierr[XO_NUM_ERR_OSV_CHECK]); // AN-632

  /* Orbit/time transformation functions */

  long xo_orbit_to_time(xo_orbit_id* orbit_id,
                        long* orbit,
                        long* second,
                        long* microsec,
                        long* time_ref,
                        /* output */
                        double* time,
                        long ierr[XO_NUM_ERR_ORBIT_TO_TIME]);

  long xo_orbit_to_time_run(long* run_id,
                            long* orbit,
                            long* second,
                            long* microsec,
                            long* time_ref,
                            /* output */
                            double* time,
                            long ierr[XO_NUM_ERR_ORBIT_TO_TIME]);

  long xo_time_to_orbit(xo_orbit_id* orbit_id,
                        long* time_ref,
                        double* time,
                        /* output */
                        long* orbit,
                        long* second,
                        long* microsec,
                        long ierr[XO_NUM_ERR_TIME_TO_ORBIT]);

  long xo_time_to_orbit_run(long* run_id,
                            long* time_ref,
                            double* time,
                            /* output */
                            long* orbit,
                            long* second,
                            long* microsec,
                            long ierr[XO_NUM_ERR_TIME_TO_ORBIT]);

  long xo_orbit_info(xo_orbit_id* orbit_id,
                     long* abs_orbit,
                     /* outputs */
                     double result_vector[XO_ORBIT_INFO_EXTRA_NUM_ELEMENTS],
                     long ierr[XO_NUM_ERR_ORBIT_INFO]);

  long xo_orbit_info_configure(xo_orbit_id* orbit_id, long* item, long* option,
                               long ierr[XO_NUM_ERR_ORBIT_INFO_CONFIGURE]); // AN-523

  long xo_orbit_info_run(long* run_id,
                         long* abs_orbit,
                         /* outputs */
                         double result_vector[XO_ORBIT_INFO_EXTRA_NUM_ELEMENTS],
                         long ierr[XO_NUM_ERR_ORBIT_INFO]);

  long xo_osv_to_tle(xo_orbit_id* orbit_id,
                     long* time_mode,
                     long* time_ref,
                     double* time0,
                     double* time1,
                     long* orbit0,
                     long* orbit1,
                     /* outputs */
                     xd_tle_rec* tle_rec,
                     long ierr[]);

  long xo_orbit_abs_from_rel(xo_orbit_id* orbit_id,
                             long* rel_orbit,
                             long* cycle,
                             /* outputs */
                             long* abs_orbit,
                             long* phase,
                             long ierr[XO_NUM_ERR_ORBIT_ABS_FROM_REL]);

  long xo_orbit_abs_from_rel_run(long* run_id,
                                 long* rel_orbit,
                                 long* cycle,
                                 /* outputs */
                                 long* abs_orbit,
                                 long* phase,
                                 long ierr[XO_NUM_ERR_ORBIT_ABS_FROM_REL]);

  long xo_orbit_abs_from_phase(xo_orbit_id* orbit_id,
                               long* phase,
                               /* outputs */
                               long* abs_orbit,
                               long* rel_orbit,
                               long* cycle,
                               long ierr[XO_NUM_ERR_ORBIT_ABS_FROM_PHASE]);

  long xo_orbit_abs_from_phase_run(long* run_id,
                                   long* phase,
                                   /* outputs */
                                   long* abs_orbit,
                                   long* rel_orbit,
                                   long* cycle,
                                   long ierr[XO_NUM_ERR_ORBIT_ABS_FROM_PHASE]);

  long xo_orbit_rel_from_abs(xo_orbit_id* orbit_id,
                             long* abs_orbit,
                             /* outputs */
                             long* rel_orbit,
                             long* cycle,
                             long* phase,
                             long ierr[XO_NUM_ERR_ORBIT_REL_FROM_ABS]);

  long xo_orbit_rel_from_abs_run(long* run_id,
                                 long* abs_orbit,
                                 /* outputs */
                                 long* rel_orbit,
                                 long* cycle,
                                 long* phase,
                                 long ierr[XO_NUM_ERR_ORBIT_REL_FROM_ABS]);

  /* run_id functions */
  long xo_run_init(long* run_id, xo_orbit_id* orbit_id,
                   /* output */
                   long ierr[XO_NUM_ERR_RUN_INIT]);

  void xo_run_get_ids(long* run_id, xo_orbit_id* orbit_id);

  void xo_run_close(long* run_id);

  /* file generation functions */
  long xo_gen_oef(char* oef,
                  char* osf,
                  char* pof,
                  char* file_class,
                  long* version_number,
                  char* system,
                  /* output */
                  long ierr[XO_NUM_ERR_GEN_OEF]);

  long xo_gen_osf_create(long* sat_id, xl_model_id* model_id, xl_time_id* time_id, long* abs_orbit_number, long* cycle_number, long* phase_number, long* repeat_cycle, long* cycle_length, double* anx_long, long* drift_mode, double* inclination, double* mlst_drift, double* mlst, double* date, char* output_dir, char* output_filename, char* file_class, long* version_number, char* system,
                         /*output*/
                         long ierr[XO_NUM_ERR_GEN_OSF_CREATE]);

  long xo_gen_osf_create_run(long* run_id,
                             long* abs_orbit_number,
                             long* cycle_number,
                             long* phase_number,
                             long* repeat_cycle,
                             long* cycle_length,
                             double* anx_long,
                             long* drift_mode,
                             double* inclination,
                             double* mlst_drift,
                             double* mlst,
                             double* date,
                             char* output_dir,
                             char* output_filename,
                             char* file_class,
                             long* version_number,
                             char* system,
                             /* output */
                             long ierr[XO_NUM_ERR_GEN_OSF_CREATE]);

  long xo_gen_osf_create_2(long* sat_id, xl_model_id* model_id, xl_time_id* time_id, double* date, xo_mission_info* mission_info, xo_ref_orbit_info* ref_orbit_info, char* output_dir, char* output_filename, char* file_class, long* version_number, char* system,
                           /*output*/
                           long ierr[XO_NUM_ERR_GEN_OSF_CREATE]);

  long xo_gen_osf_create_run_2(long* run_id,
                               double* date,
                               xo_mission_info* mission_info,
                               xo_ref_orbit_info* ref_orbit_info,
                               char* output_dir,
                               char* output_filename,
                               char* file_class,
                               long* version_number,
                               char* system,
                               /* output */
                               long ierr[XO_NUM_ERR_GEN_OSF_CREATE]);

  long xo_gen_osf_append_orbit_change(long* sat_id,
                                      xl_model_id* model_id,
                                      xl_time_id* time_id,
                                      char* input_filename,
                                      long* abs_orbit_number,
                                      long* repeat_cycle,
                                      long* cycle_length,
                                      double* anx_long,
                                      long* drift_mode,
                                      double* inclination,
                                      double* mlst_drift,
                                      double* mlst,
                                      long* phase_increment,
                                      char* output_dir,
                                      char* output_filename,
                                      char* file_class,
                                      long* version_number,
                                      char* system,
                                      /* output */
                                      long ierr[]);

  long xo_gen_osf_append_orbit_change_run(long* run_id,
                                          char* input_filename,
                                          long* abs_orbit_number,
                                          long* repeat_cycle,
                                          long* cycle_length,
                                          double* anx_long,
                                          long* drift_mode,
                                          double* inclination,
                                          double* mlst_drift,
                                          double* mlst,
                                          long* phase_increment,
                                          char* output_dir,
                                          char* output_filename,
                                          char* file_class,
                                          long* version_number,
                                          char* system,
                                          /* output */
                                          long ierr[]);

  long xo_gen_osf_append_orbit_change_2(long* sat_id,
                                        xl_model_id* model_id,
                                        xl_time_id* time_id,
                                        char* input_filename,
                                        long* abs_orbit_number,
                                        xo_ref_orbit_info* ref_orbit_info,
                                        long* phase_increment,
                                        char* output_dir,
                                        char* output_filename,
                                        char* file_class,
                                        long* version_number,
                                        char* system,
                                        /* output */
                                        long ierr[]);

  long xo_gen_osf_append_orbit_change_run_2(long* run_id,
                                            char* input_filename,
                                            long* abs_orbit_number,
                                            xo_ref_orbit_info* ref_orbit_info,
                                            long* phase_increment,
                                            char* output_dir,
                                            char* output_filename,
                                            char* file_class,
                                            long* version_number,
                                            char* system,
                                            /* output */
                                            long ierr[]);

  long xo_gen_osf_change_repeat_cycle(long* sat_id,
                                      xl_model_id* model_id,
                                      xl_time_id* time_id,
                                      char* input_filename,
                                      long* abs_orbit_number,
                                      long* search_direction,
                                      long* repeat_cycle,
                                      long* cycle_length,
                                      double* anx_long,
                                      long* drift_mode,
                                      double* inclination,
                                      double* mlst_drift,
                                      long* phase_increment,
                                      char* output_dir,
                                      char* output_filename,
                                      char* file_class,
                                      long* version_number,
                                      char* system,
                                      /* output */
                                      long ierr[]);

  long xo_gen_osf_change_repeat_cycle_run(long* run_id,
                                          char* input_filename,
                                          long* abs_orbit_number,
                                          long* search_direction,
                                          long* repeat_cycle,
                                          long* cycle_length,
                                          double* anx_long,
                                          long* drift_mode,
                                          double* inclination,
                                          double* mlst_drift,
                                          long* phase_increment,
                                          char* output_dir,
                                          char* output_filename,
                                          char* file_class,
                                          long* version_number,
                                          char* system,
                                          /* output */
                                          long ierr[]);

  long xo_gen_osf_change_repeat_cycle_2(long* sat_id,
                                        xl_model_id* model_id,
                                        xl_time_id* time_id,
                                        char* input_filename,
                                        long* abs_orbit_number,
                                        long* search_direction,
                                        xo_ref_orbit_info* ref_orbit_info,
                                        long* phase_increment,
                                        char* output_dir,
                                        char* output_filename,
                                        char* file_class,
                                        long* version_number,
                                        char* system,
                                        /* output */
                                        long ierr[]);

  long xo_gen_osf_change_repeat_cycle_run_2(long* run_id,
                                            char* input_filename,
                                            long* abs_orbit_number,
                                            long* search_direction,
                                            xo_ref_orbit_info* ref_orbit_info,
                                            long* phase_increment,
                                            char* output_dir,
                                            char* output_filename,
                                            char* file_class,
                                            long* version_number,
                                            char* system,
                                            /* output */
                                            long ierr[]);

  long xo_gen_osf_add_drift_cycle(long* sat_id,
                                  xl_model_id* model_id,
                                  xl_time_id* time_id,
                                  char* input_filename,
                                  long* drift_start_orbit,
                                  long* drift_stop_orbit,
                                  double* drift_stop_anx_long,
                                  double* max_altitude_change,
                                  long* phase_inc_start,
                                  long* phase_inc_stop,
                                  char* output_dir,
                                  char* output_filename,
                                  char* file_class,
                                  long* version_number,
                                  char* system,
                                  /* output */
                                  long ierr[]);

  long xo_gen_osf_add_drift_cycle_run(long* run_id,
                                      char* input_filename,
                                      long* drift_start_orbit,
                                      long* drift_stop_orbit,
                                      double* drift_stop_anx_long,
                                      double* max_altitude_change,
                                      long* phase_inc_start,
                                      long* phase_inc_stop,
                                      char* output_dir,
                                      char* output_filename,
                                      char* file_class,
                                      long* version_number,
                                      char* system,
                                      /* output */
                                      long ierr[]);

  long xo_gen_pof(long* sat_id, xl_model_id* model_id, xl_time_id* time_id, long* time_mode, long* time_ref, double* start_time, double* stop_time, long* start_orbit, long* stop_orbit, double* osv_location, long* ref_filetype, char* reference_file, char* precise_conf_file, long* pof_filetype, char* output_dir, char* pof_filename, char* file_class, long* version_number, char* fh_system,
                  /* output */
                  long ierr[]);

  long xo_gen_pof_run(long* run_id,
                      long* time_init,
                      long* time_ref,
                      double* start_time,
                      double* stop_time,
                      long* start_orbit,
                      long* stop_orbit,
                      double* osv_location,
                      long* ref_filetype,
                      char* reference_file,
                      char* precise_conf_file,
                      long* pof_filetype,
                      char* output_directory,
                      char* pof_filename,
                      char* file_class,
                      long* version_number,
                      char* system,
                      /* output */
                      long ierr[]);

  long xo_gen_rof(long* sat_id,
                  xl_model_id* model_id,
                  xl_time_id* time_id,
                  long* time_mode,
                  long* time_ref,
                  double* start_time,
                  double* stop_time,
                  long* start_orbit,
                  long* stop_orbit,
                  double* osv_interval,
                  long* osv_precise,
                  long* ref_filetype,
                  char* reference_file,
                  char* precise_conf_file,
                  long* rof_filetype,
                  char* output_dir,
                  char* rof_filename,
                  char* file_class,
                  long* version_number,
                  char* fh_system,
                  /* output */
                  long ierr[]);

  long xo_gen_rof_run(long* run_id,
                      long* time_mode,
                      long* time_ref,
                      double* start_time,
                      double* stop_time,
                      long* start_orbit,
                      long* stop_orbit,
                      double* osv_interval,
                      long* osv_precise,
                      long* ref_filetype,
                      char* reference_file,
                      char* precise_conf_file,
                      long* rof_filetype,
                      char* output_directory,
                      char* rof_filename,
                      char* file_class,
                      long* version_number,
                      char* fh_system,
                      /* output */
                      long ierr[]);

  long xo_gen_rof_prototype(long* sat_id,
                            xl_model_id* model_id,
                            xl_time_id* time_id,
                            long* propag_model,
                            long* time_ref,
                            double* time0,
                            long* orbit0,
                            long* time_mode,
                            double* start_time,
                            long* start_orbit,
                            double* stop_time,
                            long* stop_orbit,
                            long* drift_mode,
                            double* ascmlst_drift,
                            double* inclination,
                            long* irep,
                            long* icyc,
                            double* rlong,
                            double* ascmlst,
                            double* osv_interval,
                            long* rof_filetype,
                            char* output_dir,
                            char* rof_filename,
                            char* file_class,
                            long* version_number,
                            char* fh_system,
                            /* output */
                            long ierr[XO_MAX_LENGTH_1]);

  long xo_gen_rof_prototype_run(long* run_id,
                                long* propag_model,
                                long* time_ref,
                                double* time0,
                                long* orbit0,
                                long* time_mode,
                                double* start_time,
                                long* start_orbit,
                                double* stop_time,
                                long* stop_orbit,
                                long* drift_mode,
                                double* ascmlst_drift,
                                double* inclination,
                                long* irep,
                                long* icyc,
                                double* rlong,
                                double* ascmlst,
                                double* osv_interval,
                                long* rof_filetype,
                                char* output_directory,
                                char* rof_filename,
                                char* file_class,
                                long* version_number,
                                char* system,
                                /* output */
                                long ierr[]);

  long xo_gen_dnf(long* sat_id,
                  xl_model_id* model_id,
                  xl_time_id* time_id,
                  long* time_mode,
                  long* time_ref,
                  double* start_time,
                  double* stop_time,
                  long* start_orbit,
                  long* stop_orbit,
                  double* osv_interval,
                  long* osv_precise,
                  long* ref_filetype,
                  char* reference_file,
                  char* control_file,
                  char* precise_conf_file,
                  long* dnf_filetype,
                  char* output_dir,
                  char* dnf_filename,
                  char* file_class,
                  long* version_number,
                  char* fh_system,
                  /* output */
                  long ierr[]);

  long xo_gen_dnf_run(long* run_id,
                      long* time_mode,
                      long* time_ref,
                      double* start_time,
                      double* stop_time,
                      long* start_orbit,
                      long* stop_orbit,
                      double* osv_interval,
                      long* osv_precise,
                      long* ref_filetype,
                      char* reference_file,
                      char* control_file,
                      char* precise_conf_file,
                      long* dnf_filetype,
                      char* output_directory,
                      char* dnf_filename,
                      char* file_class,
                      long* version_number,
                      char* fh_system,
                      /* output */
                      long ierr[]);

  long xo_gen_tle(long* sat_id,
                  long* fit_mode,
                  long* time_mode,
                  long* time_ref,
                  double* start_time,
                  double* stop_time,
                  long* start_orbit,
                  long* stop_orbit,
                  char* reference_file,
                  char* tle_filename,
                  long ierr[]);

  long xo_check_osf(long* sat_id,
                    xl_model_id* model_id,
                    xl_time_id* time_id,
                    char* osf_file,
                    long* transition_number,
                    double thr[XO_NUM_CHECK_PARAMS],
                    /*output*/
                    double diffs[XO_NUM_CHECK_PARAMS],
                    long ierr[XO_NUM_ERR_CHECK_OSF]);

  long xo_check_osf_run(long* run_id,
                        char* osf_file,
                        long* transition_number,
                        double thr[XO_NUM_CHECK_PARAMS],
                        /*output*/
                        double diffs[XO_NUM_CHECK_PARAMS],
                        long ierr[XO_NUM_ERR_CHECK_OSF]);

  long xo_check_oef(long* sat_id,
                    xl_model_id* model_id,
                    xl_time_id* time_id,
                    long* time_mode,
                    long* time_ref,
                    double* start_time,
                    double* stop_time,
                    long* start_orbit,
                    long* stop_orbit,
                    char* oef_file,
                    double thr[XO_NUM_CHECK_PARAMS],
                    /*output*/
                    double max_diffs[XO_NUM_CHECK_PARAMS],
                    double rms[XO_NUM_CHECK_PARAMS],
                    long ierr[XO_NUM_ERR_CHECK_OEF]);

  long xo_check_oef_run(long* run_id,
                        long* time_mode,
                        long* time_ref,
                        double* start_time,
                        double* stop_time,
                        long* start_orbit,
                        long* stop_orbit,
                        char* oef_file,
                        double thr[XO_NUM_CHECK_PARAMS],
                        /*output*/
                        double max_diffs[XO_NUM_CHECK_PARAMS],
                        double rms[XO_NUM_CHECK_PARAMS],
                        long ierr[XO_NUM_ERR_CHECK_OEF]);

  long xo_position_on_orbit_to_time(xo_orbit_id* orbit_id, long* abs_orbit_number, long* angle_type, double* angle, double* angle_rate, double* angle_rate_rate, long* deriv, long* time_ref,
                                    /*output*/
                                    double* time,
                                    double pos_out[XO_MAX_LENGTH_3],
                                    double vel_out[XO_MAX_LENGTH_3],
                                    double acc_out[XO_MAX_LENGTH_3],
                                    long ierr[XO_NUM_ERR_OSV_COMPUTE]);

  long xo_orbit_data_filter(xo_orbit_id_init_data* orbit_data_in,
                            xo_orbit_filter_settings* filter_settings,
                            xo_orbit_id_init_data* orbit_data_out,
                            xo_orbit_filter_report* report,
                            long ierr[XO_NUM_ERR_ORBIT_DATA_FILTER]);

  long xo_orbit_id_init_data_close(xo_orbit_id_init_data* orbit_data_in,
                                   long ierr[XO_NUM_ERR_ORBIT_ID_INIT_DATA_CLOSE]);

  long xo_orbit_id_change(xo_orbit_id* orbit_id,
                          xo_orbit_id_change_data* change_data,
                          long ierr[XO_NUM_ERR_ORBIT_ID_CHANGE]); // AN-527

  /* ANR-692 */
  long xo_orbit_id_check(xo_orbit_id* orbit_id,
                         xd_orbit_file_diagnostics_settings* diagnostics_settings,
                         xo_orbit_id_check_report* report,
                         long ierr[XO_NUM_ERR_ORBIT_ID_CHECK]); //  ANR-692

  void xo_free_orbit_id_check_report(xo_orbit_id_check_report* report);
  /* END ANR-692 */

  /* Check libraries compatibility */

  long xo_check_library_version();

/* Alias for xx_check_library_version */
#if defined(expcfi_check_libs)

  #if !defined(xp_check_library_version) && !defined(xv_check_library_version)

    #undef expcfi_check_libs
    #define expcfi_check_libs xo_check_library_version

  #endif

#else

  #define expcfi_check_libs xo_check_library_version

#endif

  /*******************************************************************************
*
* Error Handling Functions - prototype declaration
*
*******************************************************************************/

  long xo_silent(void);

  long xo_verbose(void);

  long xo_print_msg(long* n, char msg[XO_MAX_COD][XO_MAX_STR]);

  long xo_get_msg(long* func_id, long* ierr,
                  /* output */
                  long* n,
                  char msg[XO_MAX_COD][XO_MAX_STR]);

  long xo_get_code(long* func_id, long* ierr,
                   /* output */
                   long* n,
                   long vec[XO_MAX_COD]);

#ifdef __cplusplus
}
#endif

/****************************************************************************/
#endif
