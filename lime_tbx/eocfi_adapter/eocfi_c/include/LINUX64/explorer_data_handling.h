/******************************************************************************
 *
 * Project     : EOCFI
 * Copyright   : DEIMOS Space S.L.U.
 *
 * Module      : explorer_data_handling.h
 *
 * Purpose     : It is the header file of the EXPLORER_DATA_HANDLING CFI to be used by the
 *               final user
 *
 * Reference   : -
 *
 * History:     +-----------------------------------------------------------------------------+
 *              | Version |   Date   |       Name        | Change                             |
 *              |-----------------------------------------------------------------------------|
 *              |    3.3  | 11/07/05 | DEIMOS Space S.L. | First release                      |
 *              +-----------------------------------------------------------------------------+
 *              |    3.4  | 18/11/05 | DEIMOS Space S.L. | Library exposed to users           |
 *              +-----------------------------------------------------------------------------+
 *              |    3.5  | 26/05/06 | DEIMOS Space S.L. | New features:                      |
 *              |         |          |                   |  + Thread safe library             |
 *              |         |          |                   |  + C99 compliant                   |
 *              |         |          |                   |  + Optimized libray                |
 *              |         |          |                   |  + 64-Bits library for MACOS,      |
 *              |         |          |                   |    LINUX and SOLARIS               |
 *              +-----------------------------------------------------------------------------+
 *              |    3.6  | 24/11/06 | DEIMOS Space S.L. | New routines:                      |
 *              |         |          |                   |  + xd_xml_validate                 |
 *              |         |          |                   | New executable function:           |
 *              |         |          |                   |  + xml_validator                   |
 *              +-----------------------------------------------------------------------------+
 *              |    3.7  | 13/07/07 | DEIMOS Space S.L. | Maintenance release                |
 *              |         |          |                   | New features:                      |
 *              |         |          |                   |  + expcfi_check_libs               |
 *              |         |          |                   |  + library version for Mac OS X    |
 *              |         |          |                   |    on Intel (32 and 64 bits)       |
 *              +-----------------------------------------------------------------------------+
 *              |  3.7.2  | 31/07/08 | DEIMOS Space S.L. | Maintenance release                |
 *              +-----------------------------------------------------------------------------+
 *              |    4.0  | 19/01/09 | DEIMOS Space S.L. | Maintenance release                |
 *              +-----------------------------------------------------------------------------+
 *              |    4.1  | 07/05/10 | DEIMOS Space S.L. | Maintenance release                |
 *              |         |          |                   | New features:                      |
 *              |         |          |                   |  + DEM GETASSEv2 supported         |
 *              |         |          |                   |  + IERS B Bulletin format 2010     |
 *              |         |          |                   |  + Ground Station DB 1.4 supported |
 *              +-----------------------------------------------------------------------------+
 *              |    4.2  | 31/01/11 | DEIMOS Space S.L.U| Maintenance release                |
 *              |         |          |                   | New features:                      |
 *              |         |          |                   |  + New Dem configuration file      |
 *              |         |          |                   |    (schema version 1.3)            |
 *              |         |          |                   |  + NORAD TLE designators for       |
 *              |         |          |                   |    SMOS, GOCE and CRYOSAT          |
 *              |         |          |                   |  + New OSF format for curved MLST  |
 *              +-----------------------------------------------------------------------------+
 *              |    4.3  | 06/02/12 | DEIMOS Space S.L.U| Maintenance release                |
 *              |         |          |                   | New features:                      |
 *              |         |          |                   |  + Support for reading new IERS    |
 *              |         |          |                   |    bulletins A and B               |
 *              |         |          |                   |  + New functions to decimate orbit |
 *              |         |          |                   |    and attitude data               |
 *              |         |          |                   |    (xd_orbit_file_decimate and     |
 *              |         |          |                   |     xd_attitude_file_decimate)     |
 *              +-----------------------------------------------------------------------------+
 *              |    4.4  | 05/07/12 | DEIMOS Space S.L.U| Maintenance release                |
 *              |         |          |                   | New features:                      |
 *              |         |          |                   |  + New tags in Dem configuration   |
 *              |         |          |                   |    for DEM cache                   |
 *              +-----------------------------------------------------------------------------+
 *              |    4.5  | 01/03/13 | DEIMOS Space S.L.U| Maintenance release                |
 *              |         |          |                   | New features:                      |
 *              |         |          |                   |  + New tags in Dem configuration   |
 *              |         |          |                   |    for mini-tiles and geoid        |
 *              |         |          |                   |    computation.                    |
 *              |         |          |                   |  + EarthCase filenames compliant   |
 *              |         |          |                   |    with FFS 2.0.                   |
 *              +-----------------------------------------------------------------------------+
 *              |    4.6  | 03/10/13 | DEIMOS Space S.L.U| Maintenance release                |
 *              |         |          |                   | New features:                      |
 *              |         |          |                   |  + Support for new Attitude        |
 *              |         |          |                   |    Definition File                 |
 *              +-----------------------------------------------------------------------------+
 *              |    4.7  | 28/03/14 | DEIMOS Space S.L.U| Maintenance release                |
 *              |         |          |                   | New features:                      |
 *              |         |          |                   |  + Support for Doris files for     |
 *              |         |          |                   |    Sentinel 3.                     |
 *              |         |          |                   |  + Support for files with SP3      |
 *              |         |          |                   |    format (GPS orbit format)       |
 *              +-----------------------------------------------------------------------------+
 *              |    4.8  | 29/10/14 | DEIMOS Space S.L.U| Maintenance release                |
 *              |         |          |                   | New features:                      |
 *              |         |          |                   |  + Support for DEM GETASSE v3.0    |
 *              |         |          |                   |  + Support for GDEM v2             |
 *              |         |          |                   |  + New function to add style sheet |
 *              |         |          |                   |    to files: xd_xslt_add           |
 *              +-----------------------------------------------------------------------------+
 *              |    4.9  | 23/04/15 | DEIMOS Space S.L.U| Maintenance release                |
 *              |         |          |                   | New features:                      |
 *              |         |          |                   |  + Support for Orbit Ephemeris     |
 *              |         |          |                   |    Message files                   |
 *              +-----------------------------------------------------------------------------+
 *              |  4.9.1  | 05/06/15 | DEIMOS Space S.L.U| Maintenance release                |
 *              +-----------------------------------------------------------------------------+
 *              |   4.10  | 29/10/15 | DEIMOS Space S.L.U| Maintenance release                |
 *              |         |          |                   | New features:                      |
 *              |         |          |                   |  + Support for DEM ACE2 30 secs    |
 *              |         |          |                   |  + New diagnostic function for     |
 *              |         |          |                   |    orbit files with state vectors: |
 *              |         |          |                   |    xd_orbit_file_diagnostics       |
 *              |         |          |                   |  + Change of interface in functions|
 *              |         |          |                   |    xd_read_oem and xd_read_sp3     |
 *              +-----------------------------------------------------------------------------+
 *              |   4.11  | 15/04/16 | DEIMOS Space S.L.U| Maintenance release                |
 *              |         |          |                   | New features:                      |
 *              |         |          |                   |  + Support for DEM ACE2 3 secs     |
 *              +-----------------------------------------------------------------------------+
 *              |   4.12  | 03/11/16 | DEIMOS Space S.L.U| Maintenance release                |
 *              |         |          |                   | New features:                      |
 *              |         |          |                   |  + Support for FFS 3               |
 *              +-----------------------------------------------------------------------------+
 *              |   4.13  | 05/04/17 | DEIMOS Space S.L.U| Maintenance release                |
 *              +-----------------------------------------------------------------------------+
 *              |   4.14  | 16/11/17 | DEIMOS Space S.L.U| Maintenance release                |
 *              |         |          |                   | New features:                      |
 *              |         |          |                   |  + Added support for Jason-CS Doris|
 *              +-----------------------------------------------------------------------------+
 *              |   4.15  | 20/04/18 | DEIMOS Space S.L.U| Maintenance release                |
 *              |         |          |                   | New features:                      |
 *              |         |          |                   |  + Refactored code                 |
 *              +-----------------------------------------------------------------------------+
 *              |   4.16  | 09/11/18 | DEIMOS Space S.L.U| Maintenance release                |
 *              +-----------------------------------------------------------------------------+
 *              |   4.17  | 10/05/19 | DEIMOS Space S.L.U| Maintenance release                |
 *              |         |          |                   | New features:                      |
 *              |         |          |                   |  + DEM configurable raster         |
 *              +-----------------------------------------------------------------------------+
 *              |   4.18  | 08/11/19 | DEIMOS Space S.L.U| Maintenance release                |
 *              |         |          |                   | New features:                      |
 *              |         |          |                   |  + Support for TanDEM-X 90m DEM    |
 *              |         |          |                   |  + Support new IERS bulleting B    |
 *              |         |          |                   |    format                          |
 *              +-----------------------------------------------------------------------------+
 *              |   4.19  | 29/05/20 | DEIMOS Space S.L.U| Maintenance release                |
 *              |         |          |                   | New features:                      |
 *              |         |          |                   |  + Support for ASTER GDEM v3       |
 *              |         |          |                   |  + Support for Reference_Frame tag |
 *              |         |          |                   |    in Attitude Angles files        |
 *              |         |          |                   |  + Support for XML Orbit Ephemeris |
 *              |         |          |                   |    Message files                   |
 *              +-----------------------------------------------------------------------------+
 *              |   4.20  | 30/11/20 | DEIMOS Space S.L.U| Maintenance release                |
 *              |         |          |                   | New features:                      |
 *              |         |          |                   |  + Support for CCSDS AEM files     |
 *              +-----------------------------------------------------------------------------+
 *              |   4.21  | 10/06/21 | DEIMOS Space S.L.U| Maintenance release                |
 *              |         |          |                   | New features:                      |
 *              |         |          |                   |  + Optimisation of EOCFI library   |
 *              |         |          |                   |    size                            |
 *              +-----------------------------------------------------------------------------+
 *              |   4.22  | 22/12/21 | DEIMOS Space S.L.U| Maintenance release                |
 *              |         |          |                   | New features:                      |
 *              |         |          |                   |  + Enable support for AEM with     |
 *              |         |          |                   |    custom REF_FRAME used by EUM    |
 *              +-----------------------------------------------------------------------------+
 *              |   4.23  | 23/06/22 | DEIMOS Space S.L.U| Maintenance release                |
 *              +-----------------------------------------------------------------------------+
 *
 *****************************************************************************/

#ifndef EXPLORER_DATA_HANDLING_H
#define EXPLORER_DATA_HANDLING_H

#include <explorer_file_handling.h>

#ifdef __cplusplus
extern "C"
{
#endif

/* -------------------- */
/* CONSTANTS DEFINITION */
/* -------------------- */

/* Length of ierr vectors */
#define XD_NUM_ERR_READ_FHR                            1
#define XD_NUM_ERR_READ_BULLETIN                       1
#define XD_NUM_ERR_READ_ORBIT_FILE                     1
#define XD_NUM_ERR_ORBIT_FILE_DECIMATE                 1
#define XD_NUM_ERR_READ_DORIS                          1
#define XD_NUM_ERR_READ_DORIS_HEADER                   1
#define XD_NUM_ERR_READ_OSF                            1
#define XD_NUM_ERR_READ_SDF                            1
#define XD_NUM_ERR_READ_STF                            2
#define XD_NUM_ERR_READ_STF_VHR                        1
#define XD_NUM_ERR_READ_ATT                            1
#define XD_NUM_ERR_ATTITUDE_FILE_DECIMATE              1
#define XD_NUM_ERR_READ_STAR_TRACKER                   1
#define XD_NUM_ERR_READ_STR_CONF_FILE                  1
#define XD_NUM_ERR_READ_DEM_CONFIG_FILE                1
#define XD_NUM_ERR_READ_DEM                            1
#define XD_NUM_ERR_READ_STAR                           1
#define XD_NUM_ERR_READ_STAR_FILE                      1
#define XD_NUM_ERR_READ_STAR_ID                        1
#define XD_NUM_ERR_READ_STATION                        1
#define XD_NUM_ERR_READ_STATION_FILE                   1
#define XD_NUM_ERR_READ_STATION_ID                     1
#define XD_NUM_ERR_READ_ZONE                           1
#define XD_NUM_ERR_READ_ZONE_FILE                      1
#define XD_NUM_ERR_READ_ZONE_ID                        1
#define XD_NUM_ERR_READ_FOV                            1 /* ANR-628*/
#define XD_NUM_ERR_WRITE_ORBIT_FILE                    1
#define XD_NUM_ERR_WRITE_DORIS                         1
#define XD_NUM_ERR_WRITE_OSF                           1
#define XD_NUM_ERR_WRITE_STF                           1
#define XD_NUM_ERR_WRITE_ATT                           1
#define XD_NUM_ERR_XML_VALIDATE                        1
#define XD_NUM_ERR_READ_TLE                            1
#define XD_NUM_ERR_WRITE_TLE                           1
#define XD_NUM_ERR_READ_PRECISE_PROPAG                 1
#define XD_NUM_ERR_WRITE_PRECISE_PROPAG                1
#define XD_NUM_ERR_SELECT_SCHEMA                       1
#define XD_NUM_ERR_READ_ATT_DEF                        1 /* ANR-469*/
#define XD_NUM_ERR_WRITE_ATT_DEF                       1 /* ANR-469*/
#define XD_NUM_ERR_READ_SP3                            1 /* ANR-550*/
#define XD_NUM_ERR_GDEM_READ                           2 /* ANR-504*/
#define XD_NUM_ERR_GDEM_WRITE                          1 /* ANR-504*/
#define XD_NUM_ERR_XSLT_ADD                            1 /* ANR-546*/
#define XD_NUM_ERR_READ_OEM                            1 /* ANR-578*/
#define XD_NUM_ERR_READ_AEM                            1 // ANR-856
#define XD_NUM_ERR_ORBIT_FILE_DIAGNOSTICS              1 /* ANR-576*/
#define XD_NUM_ERR_SET_FILE_FORMAT_STANDARD_VERSION    1 /* ANR-638*/
#define XD_NUM_ERR_DETECT_FILE_FORMAT_STANDARD_VERSION 1 /* ANR-638*/
#define XD_ERR_VECTOR_MAX_LENGTH                       2 /* Max length of ierr vectors \
                                                            Max of the above lengths.  \
                                                            This value can be safely used in every function declaration. */

/* Maximum length of a string (function name or error message) */
#define XD_MAX_STR 512

/* Maximum length of path (AN-821) */
#define XD_MAX_PATH 4096

/* Maximum number of error codes that can be defined for a single function */
#define XD_MAX_COD 256

/* Generic maximum lengths */
#define XD_MAX_LENGTH_2   2
#define XD_MAX_LENGTH_3   3
#define XD_MAX_LENGTH_4   4
#define XD_MAX_LENGTH_5   5
#define XD_MAX_LENGTH_6   6
#define XD_MAX_LENGTH_8   8
#define XD_MAX_LENGTH_9   9
#define XD_MAX_LENGTH_10  10
#define XD_MAX_LENGTH_15  15
#define XD_MAX_LENGTH_20  20
#define XD_MAX_LENGTH_24  24
#define XD_MAX_LENGTH_25  25
#define XD_MAX_LENGTH_29  29
#define XD_MAX_LENGTH_30  30
#define XD_MAX_LENGTH_31  31
#define XD_MAX_LENGTH_32  32
#define XD_MAX_LENGTH_100 100
#define XD_MAX_LENGTH_250 250

/* Error handling constants */
#define XD_ZERO_E    0
#define XD_FIRST_E   32
#define XD_SECOND_E  64
#define XD_THIRD_E   96
#define XD_FORTH_E   128
#define XD_FIFTH_E   160
#define XD_SIXTH_E   192
#define XD_SEVENTH_E 224
#define XD_EIGHTH_E  256
#define XD_NINETH_E  288
#define XD_TENTH_E   320

/* Number of the parameters in the parameter att. model */
#define XD_NUM_MODEL_PARAM 10
/* Number of the parameters in the harmonic att. model */
#define XD_MAX_NUM_HARMONIC 10

/* Maximum number of points defining a station */
#define XD_VERTICES 999
/* Maximum number of missions for a station */
#define XD_MISSIONS 20

/* Maximum Star Name size */
#define XD_MAX_STAR_ID_SIZE 30

/* EOM orbit */
#define XD_ORBIT_EOM 99999

/* No schema */
#define NO_SCHEMA_STRING "_NOSCHEMA_\0"

/* Default DEM cache size (AN-487) */
#define XD_DEM_CACHE_MAX_SIZE_DEFAULT 2048 /* In MB */

/* Default number of harmonics (AN-466) */
#define XD_DEM_GEOID_COMPUTATION_FLAG_DEFAULT 1 /* Same value as XD_DEM_GEOID_COMPUTATION_ENABLED enum */
#define XD_DEM_NUMBER_OF_HARMONICS_DEFAULT    30

// AN-550 Number of chars of satellites ids
#define SIZE_SP3_ID 3

  /* ------------ */
  /* ENUMERATIONS */
  /* ------------ */

  /* Boolean enumeration */
  typedef enum
  {
    XD_FALSE = 0, /* False variable */
    XD_TRUE = 1 /* True variable */
  } XD_Boolean;

  /* Returned status code */
  typedef enum
  {
    XD_ERR = -1, /* Error status   */
    XD_OK = 0, /* Nominal status */
    XD_WARN = 1 /* Warning status */
  } XD_PErr_type_enum;
  /* CAREFUL: variables holding these values MUST be declared as long !!! */

  /* Satellite ID */
  typedef enum
  {
    XD_SAT_DEFAULT = 0,
    XD_SAT_DEFAULT1 = 1,
    XD_SAT_DEFAULT2 = 2,
    XD_SAT_DEFAULT3 = 3,
    XD_SAT_DEFAULT4 = 4,
    XD_SAT_DEFAULT5 = 5,
    XD_SAT_DEFAULT6 = 6,
    XD_SAT_DEFAULT7 = 7,
    XD_SAT_DEFAULT8 = 8,
    XD_SAT_DEFAULT9 = 9,
    XD_SAT_ERS1 = 11,
    XD_SAT_ERS2 = 12,
    XD_SAT_ENVISAT = 21,
    XD_SAT_METOP1 = 31,
    XD_SAT_METOP2 = 32,
    XD_SAT_METOP3 = 33,
    XD_SAT_CRYOSAT = 41,
    XD_SAT_ADM = 51,
    XD_SAT_GOCE = 61,
    XD_SAT_SMOS = 71,
    XD_SAT_TERRASAR = 81,
    XD_SAT_EARTHCARE = 91,
    XD_SAT_SWARM_A = 101,
    XD_SAT_SWARM_B = 102,
    XD_SAT_SWARM_C = 103,
    XD_SAT_SENTINEL_1A = 110,
    XD_SAT_SENTINEL_1B = 111,
    XD_SAT_SENTINEL_2 = 112,
    XD_SAT_SENTINEL_3 = 113,
    XD_SAT_SEOSAT = 120,
    XD_SAT_SENTINEL_1C = 125,
    XD_SAT_SENTINEL_2A = 126,
    XD_SAT_SENTINEL_2B = 127,
    XD_SAT_SENTINEL_2C = 128,
    XD_SAT_SENTINEL_3A = 129,
    XD_SAT_SENTINEL_3B = 130,
    XD_SAT_SENTINEL_3C = 131,
    XD_SAT_JASON_CSA = 132, // AN-531
    XD_SAT_JASON_CSB = 133, // AN-531
    XD_SAT_METOP_SG_A1 = 134, // AN-530
    XD_SAT_METOP_SG_A2 = 135, // AN-530
    XD_SAT_METOP_SG_A3 = 136, // AN-530
    XD_SAT_METOP_SG_B1 = 137, // AN-530
    XD_SAT_METOP_SG_B2 = 138, // AN-530
    XD_SAT_METOP_SG_B3 = 139, // AN-530
    XD_SAT_SENTINEL_5P = 140, // AN-502
    XD_SAT_BIOMASS = 141, // AN-601
    XD_SAT_SENTINEL_5 = 142, // AN-639
    XD_SAT_SAOCOM_CS = 143, // AN-640
    XD_SAT_FLEX = 144, /* AN-683 */
    XD_SAT_SENTINEL_6A = 145, /* AN-809 */
    XD_SAT_SENTINEL_6B = 146, /* AN-809 */
    XD_SAT_CIMR = 147,
    XD_SAT_ROSEL = 148,
    XD_SAT_CHIME = 149,
    XD_SAT_CRISTAL = 150,
    XD_SAT_CO2M = 151,
    XD_SAT_LSTM = 152,
    XD_SAT_FORUM = 153,
    XD_SAT_TRUTHS = 154,
    XD_SAT_GENERIC = 200,
    XD_SAT_GENERIC_GEO = 300, /* ANR-353 */
    XD_SAT_MTG = 301, /* ANR-353 */
    XD_SAT_GENERIC_MEO = 400 /* ANR-550 */
  } XD_Sat_id_enum;

  /* File types */
  typedef enum
  {
    XD_UNKNOWN_TYPE = -1,
    XD_AUTO,
    XD_ORBIT_CHANGE,
    XD_STATE_VECTOR,
    XD_OSF_TYPE,
    XD_POF_TYPE,
    XD_ROF_TYPE,
    XD_DORIS_TYPE,
    XD_POF_N_DORIS_TYPE,
    XD_OEF_OSF_TYPE,
    XD_OEF_POF_TYPE,
    XD_IERS_B_TYPE,
    XD_TLE_TYPE,
    XD_STF_TYPE,
    XD_DORISPREC_TYPE,
    XD_DORISPREM_TYPE,
    XD_ATT_TYPE,
    XD_SCF_TYPE,
    XD_PRECISE_PROPAG_TYPE,
    XD_DEMCFG_TYPE,
    XD_SATCFG_TYPE,
    XD_GND_DB_TYPE,
    XD_SW_DEF_TYPE,
    XD_ZON_DB_TYPE,
    XD_STR1ATT_TYPE,
    XD_IERS_A_TYPE,
    XD_IERS_B_AND_A_TYPE,
    XD_ATT_DEF_TYPE, /* AN-469  */
    XD_USER_OSV_LIST_TYPE, /* AN-497  */
    XD_SP3_TYPE, /* AN-550  */
    XD_OSF_POF_MODE, /* ANR-527 */
    XD_OSF_ROF_MODE, /* ANR-527 */
    XD_OSF_DORIS_MODE, /* ANR-527 */
    XD_OEM_TYPE, /* AN-578 */
    XD_OSF_OEM_MODE, /* AN-578 */
    XD_FOV_TYPE, /* AN-628 */
    XD_AEM_TYPE, // AN-856
    XD_FILE_TYPE_MAX_VALUE
  } XD_File_types;

  /* Time Initialization mode */
  typedef enum
  {
    XD_SEL_FILE = 0,
    XD_SEL_TIME,
    XD_SEL_ORBIT,
    XD_SEL_DEFAULT,
    XD_SEL_NONE // AN-634
  } XD_Time_init_mode_enum;

  /* Time reference ID */
  typedef enum
  {
    XD_TIME_UNDEF = -1,
    XD_TIME_TAI,
    XD_TIME_UTC,
    XD_TIME_UT1,
    XD_TIME_GPS
  } XD_Time_ref_enum;

  /* File reference time */
  typedef enum
  {
    XD_TIME_REF_OF_TAI = XD_TIME_TAI,
    XD_TIME_REF_OF_UTC = XD_TIME_UTC,
    XD_TIME_REF_OF_UT1 = XD_TIME_UT1
  } XD_Time_ref_of_enum;

  /* Attitude data type */
  typedef enum
  {
    XD_ATT_QUATERNIONS = 0,
    XD_ATT_ANGLES

  } XD_Attitude_data_type_enum;

  /* Ray tracing model */
  typedef enum
  {
    XD_NO_REF = 0,
    XD_STD_REF,
    XD_USER_REF,
    XD_PRED_REF,
    XD_STD_REF_N = 10,
    XD_USER_REF_N = 20,
    XD_PRED_REF_N = 30,
    XD_US76_REF = 300,
    XD_TROPIC_REF,
    XD_MID_SUM_REF,
    XD_MID_WIN_REF,
    XD_SUBAR_SUM_REF,
    XD_SUBAR_WIN_REF,
    XD_LUT_REF = 400,
    XD_US76_REF_N = 3000,
    XD_TROPIC_REF_N,
    XD_MID_SUM_REF_N,
    XD_MID_WIN_REF_N,
    XD_SUBAR_SUM_REF_N,
    XD_SUBAR_WIN_REF_N,
    XD_LUT_REF_N = 4000

  } XD_Target_ray_enum;

  /* Swath types*/
  typedef enum
  {
    XD_OPEN_SWATH,
    XD_CLOSED_SWATH

  } XD_Swath_type_enum;

  /* Swath point types*/
  typedef enum
  {
    XD_GEODETIC_SWATH_TYPE,
    XD_INERTIAL_SWATH_TYPE

  } XD_Swath_point_type_enum;

  /* Swath geometry definition = algoritm */
  typedef enum
  {
    XD_SWATH_POINTING_GEOM = 0, /* Algorithms of calculation of Swaths */
    XD_SWATH_DISTANCE_GEOM,
    XD_SWATH_LIMB_GEOM,
    XD_SWATH_INERTIAL_GEOM,
    XD_SWATH_SUBSATELLITE_GEOM,
    XD_SWATH_ASAR_GEOM,
    XD_SWATH_ASAR_GEOM_RANGERATE_ALGO, /* Old algo for ASAR, for testing purposes */
    XD_SWATH_INCIDENCE_ANGLE_GEOM /* AN-665 */

  } XD_Swath_geom_enum;

  /*  ASAR Swath types */
  typedef enum
  {
    XD_NO_ASAR = 0,
    XD_NARROW_ASAR,
    XD_WIDE_ASAR

  } XD_Asar_type_enum;

  /* File types */
  typedef enum
  {
    XD_REF_FILETYPE_OSF = 1, /* Orbit Scenario File */
    XD_REF_FILETYPE_OEF_OSF, /* Orbit Event file used as an OSF */
    XD_REF_FILETYPE_POF, /* FOS Predicted Orbit File */
    XD_REF_FILETYPE_OEF_POF, /* Orbit Event file used as a POF */
    XD_REF_FILETYPE_DORIS_NAV, /* DORIS Navigator File */
    XD_REF_FILETYPE_ROF, /* FOS Restituted Orbit File */
    XD_REF_FILETYPE_TLE, /* TLE Restituted Orbit File */
    XD_REF_FILETYPE_DORIS_PREM, /* DORIS Preliminary Orbit File */
    XD_REF_FILETYPE_DORIS_PREC /* DORIS Precise Orbit File */

  } XD_Ref_filetype_enum;

  /* Coordinate systems */
  typedef enum
  {
    XD_BAR_MEAN_2000 = 1, /* Barycentric Mean of 2000.0 coordinate system */
    XD_HEL_MEAN_2000, /* Heliocentric Mean of 2000.0 coordinate system */
    XD_GEO_MEAN_2000, /* Geocentric Mean of 2000.0 coordinate system */
    XD_MEAN_DATE, /* Mean of date coordinate system */
    XD_TRUE_DATE, /* True of date coordinate system */
    XD_PSEUDO_EARTH_FIXED, /* Pseudo Earth Fixed coordinate system */
    XD_EARTH_FIXED, /* Earth Fixed coordinate system */
    XD_LIF, /* Launch Inertial Frame (AN-341) */
    XD_BAR_MEAN_1950, /* Barycentric Mean of 2000.0 coordinate system */
    XD_GALACTIC, /* Galactic coordinate system*/
    XD_SAT_ACT_REF, /* Satellite relative actual reference cs */
    XD_QUASI_MEAN_DATE, /* Quasi-Mean of Date coordinate system */
    XD_PSE_TRUE_DATE, /* Pseudo-True of Date coordinate system */
    XD_TOPOCENTRIC, /* Topocentric coordinate system */
    XD_SAT_REF, /* Satellite reference coordinate system */
    XD_SAT_REL_REF /* Satellite relative reference coordinate system */
  } XD_Cs_enum;

  /* Attitude reference frames */
  typedef enum
  {
    XD_SAT_ORBITAL_REF = 0,
    XD_SAT_NOMINAL_ATT,
    XD_SAT_ATT,
    XD_INSTR_ATT
  } XD_Attitude_frame_enum;

  /* DEM Models */
  typedef enum
  {
    XD_DEM_ACE_MODEL = 0, /* DEPRECATED */
    XD_DEM_GETASSE30_V1,
    XD_DEM_GETASSE30_V2,
    XD_DEM_ACE2_9SEC,
    XD_DEM_GETASSE30_V3, /* AN-503 */
    XD_DEM_GDEM_V2, /* AN-504 */
    XD_DEM_ACE2_30SEC, /* AN-615 */
    XD_DEM_ACE2_3SEC, /* AN-619 */
    XD_DEM_ACE2_5MIN, /* AN-726 */
    XD_DEM_GENERIC_RASTER, /* ANR-740 */
    XD_DEM_TANDEMX_90,
    XD_DEM_GDEM_V3, /* ANR-827 */
    XD_DEM_NUM_MODELS /* -> number of models defined in the EO CFI */
  } XD_Dem_model_enum;

  /* DEM Data Source Types for GETASSE30 V1, V2, V3 */
  typedef enum
  {
    XD_DEM_GETASSE30_SOURCE_ACE, /* land-ice/snow */
    XD_DEM_GETASSE30_SOURCE_MSS, /* sea */
    XD_DEM_GETASSE30_SOURCE_EGM96, /* sea-ice */
    XD_DEM_GETASSE30_SOURCE_SRTM30 /* land */
  } XD_Dem_getasse_data_source_enum;

  /* DEM Data Source Types for ACE2 9secs */
  typedef enum
  {
    XD_DEM_ACE2_SOURCE_SRTM0, /* Pure SRTM (above 60N pure GLOBE data, below 60S pure ACE [original] data) */
    XD_DEM_ACE2_SOURCE_SRTM1, /* SRTM voids filled by interpolation and/or altimeter data */
    XD_DEM_ACE2_SOURCE_SRTM2, /* SRTM data warped using the ERS-1 Geodetic Mission */
    XD_DEM_ACE2_SOURCE_SRTM3, /* SRTM data warped using EnviSat & ERS-2 data */
    XD_DEM_ACE2_SOURCE_SRTM_LAKE, /* Mean lake level data derived from Altimetry */
    XD_DEM_ACE2_SOURCE_SRTM_GLOBE, /* GLOBE/ACE data warped using combined altimetry (only above 60N) */
    XD_DEM_ACE2_SOURCE_SRTM_ALT /* Pure altimetry data (derived from ERS-1 Geodetic Mission, 
                                    ERS-2 and EnviSat data using Delaunay Triangulation and Bilinear interpolation) */
  } XD_Dem_ace2_data_source_enum;

  /* DEM Data Source Types for GDEM v2 (AN-504) */
  typedef enum
  {
    XD_DEM_GDEM_SOURCE_NO_SOURCE_FILE = -1, /*QA value contain the number of scene-based DEMs contributing to the final GDEM value for each 30m pixel (stack number)*/
    XD_DEM_GDEM_SOURCE_SRTM3_V3 = 0, /*SRTM3 V3: Posting: 3 arc seconds Coverage: north 60° to south 56°  Only about 90 % tiles of SRTM V3 are void filled*/
    XD_DEM_GDEM_SOURCE_SRTM3_V2, /*SRTM3 V2: Posting: 3 arc seconds Coverage: north 60° to south 56*/
    XD_DEM_GDEM_SOURCE_NED, /*NED: Posting: 1 arc second Coverage: Conterminous U.S.*/
    XD_DEM_GDEM_SOURCE_CDED, /*CDED: Posting: 3 arc seconds for latitude; 3, 6 and 12 arc seconds for longitude, depending on latitude Coverage: all Canada territory */
    XD_DEM_GDEM_SOURCE_ALASKA_DEM /*ALASKA_DEM: Posting: 2 arc seconds Coverage: all Alaska territory*/
  } XD_Dem_gdem_data_source_enum;

  /* DEM cell position*/
  typedef enum
  {
    XD_DEM_CELL_CENTER,
    XD_DEM_CELL_NORTHWEST,
    XD_DEM_CELL_NORTHEAST,
    XD_DEM_CELL_SOUTHEAST,
    XD_DEM_CELL_SOUTHWEST,
  } XD_Dem_cell_location_enum;

  /* Type of zone */
  typedef enum
  {
    XD_NOT_DEFINED = -1, /* The zone is not defined as an input and must be read from a file */
    XD_POINT = 0, /* The zone is a point (resulting segments will be of zero duration) */
    XD_CIRCLE = 1, /* The zone is a circle defined by its centre point and its radius */
    XD_SEGMENT = 2, /* The zone is a segment defined by its two extremes */
    XD_POLYGON = 3 /* The zone is a polygon */
  } XD_Zone_type_enum;

  /* Type of projections  */
  typedef enum
  {
    XD_READ_DB = 0, /* Read projection from DB file */
    XD_GNOMONIC = 1, /* Use gnomonic projection */
    XD_RECTANGULAR = 2 /* Use rectangular projection */
  } XD_Proj_enum;

  /* Modes for the input schema in the xd_xml_validate */
  typedef enum
  {
    XD_DEFAULT_SCHEMA, /* Use the default schema in the root element of the XML file */
    XD_USER_SCHEMA /* Use the input schema provided via interface */
  } XD_Schema_mode_enum;

  /* validation status returned by xl_xml_validate */
  typedef enum
  {
    XD_XML_INVALID = -1, /* Invalid XML file */
    XD_XML_VALID /* Valid XML file */
  } XD_XML_validate_result_enum;

  /* Quality index for Orbit files */
  typedef enum
  {
    XD_3_ADJUST_NOMI = 1, /* Adjusted out of orbit manoeuvre period */
    XD_4_ADJUST_DMAN, /* Adjusted during an orbit manoeuvre */
    XD_5_INTERP_DGAP, /* Interpolated during a data gap */
    XD_6_EXTRAP_LT1D, /* Extrapolated from less than 1 day */
    XD_7_EXTRAP_1D2D, /* Extrapolated from more than 1 day, but less than 2 days */
    XD_8_EXTRAP_GT2D, /* Extrapolated from more than 2 days */
    XD_8_EXTRAP_AMAN, /* Extrapolated after an orbit manoeuvre */
    XD_Q_INDEX_MAX_ENUM /* Number of defined values (+1) */
  } XD_Quality_index_enum;

  /* Draw modes for the SCF */
  typedef enum
  {
    XD_SCF_DRAW_SOLID,
    XD_SCF_DRAW_DASHED,
    XD_SCF_DRAW_DOTTED,
    XD_SCF_DRAW_TIMELINE
  } XD_SCF_Draw_modes_enum;

  /* Fill modes for the SCF */
  typedef enum
  {
    XD_SCF_FILL_SOLID,
    XD_SCF_FILL_HOLLOW
  } XD_SCF_Fill_modes_enum;

  /* Event types for SCF*/
  typedef enum
  {
    XD_EVENT_TYPE_NONE,
    XD_EVENT_TYPE_ZONE,
    XD_EVENT_TYPE_STATION,
    XD_EVENT_TYPE_GEO
  } XD_Event_type_enum;

  /* CFI functions IDs (for error handling) */
  typedef enum
  {
    XD_NO_FUNC_ID = -1, /* No function code */
    XD_READ_FHR_ID,
    XD_READ_BULLETIN_ID,
    XD_READ_ORBIT_FILE_ID,
    XD_READ_DORIS_ID,
    XD_READ_DORIS_HEADER_ID,
    XD_READ_OSF_ID,
    XD_READ_SDF_ID,
    XD_READ_STF_ID,
    XD_READ_STF_VHR_ID,
    XD_READ_ATT_ID,
    XD_READ_STAR_TRACKER_ID,
    XD_READ_STR_CONF_FILE_ID,
    XD_READ_DEM_CONFIG_FILE_ID,
    XD_READ_DEM_ID,
    XD_READ_STAR_ID,
    XD_READ_STAR_FILE_ID,
    XD_READ_STAR_ID_ID,
    XD_READ_STATION_ID,
    XD_READ_STATION_FILE_ID,
    XD_READ_STATION_ID_ID,
    XD_READ_ZONE_ID,
    XD_READ_ZONE_FILE_ID,
    XD_READ_ZONE_ID_ID,
    XD_WRITE_ORBIT_FILE_ID,
    XD_WRITE_DORIS_ID,
    XD_WRITE_OSF_ID,
    XD_WRITE_STF_ID,
    XD_WRITE_ATT_ID,
    XD_XML_VALIDATE_ID,
    XD_READ_TLE_ID,
    XD_WRITE_TLE_ID,
    XD_READ_PRECISE_PROPAG_FILE_ID,
    XD_WRITE_PRECISE_PROPAG_FILE_ID,
    XD_ORBIT_FILE_DECIMATE_ID,
    XD_ATTITUDE_FILE_DECIMATE_ID,
    XD_READ_ATT_DEF_ID, /*ANR-469*/
    XD_WRITE_ATT_DEF_ID, /*ANR-469*/
    XD_READ_SP3_ID, // AN-550
    XD_XSLT_ADD_ID, /* AN-546 */
    XD_READ_OEM_ID, /* AN-578 */
    XD_ORBIT_FILE_DIAGNOSTICS_ID, /* AN-576 */
    XD_READ_FOV_CONSTRAINTS_FILE_ID, /* AN-628 */
    XD_SET_FILE_FORMAT_STANDARD_VERSION_ID, /* AN-638 */
    XD_DETECT_FILE_FORMAT_STANDARD_VERSION_ID, /* AN-638 */
    XD_READ_AEM_ID, // AN-586
    XD_MAX_FUNC_ID /* Number of functions defined in this list */
  } XD_Func_enum;

  /* User flag for precise propagator */
  typedef enum
  {
    XD_DEFAULT_VALUES = 0, /* Use predefined default values for some parameters  */
    XD_USER_VALUES = 1 /* Use values introduced by user */
  } XD_Precise_propag_user_flag_enum;

  /* Flag to select or not contributions to precise propagator  */
  typedef enum
  {
    XD_NOT_SELECT = 0, /* Do not select contribution */
    XD_SELECT = 1 /* Select contribution */
  } XD_Precise_propag_selection_flag_enum;

  /* Flag to select if SGA values are taken for input parameters or
   read from file */
  typedef enum
  {
    XD_SGA_USE_PARAMETERS = 0, /* Use input parameters */
    XD_SGA_READ_VALUES_FROM_FILE = 1 /* Read values from files */
  } XD_Precise_propag_sga_flag_enum;

  /* Enumeration to distinguish between bulletin A and B */
  typedef enum
  {
    XD_BULLETIN_A = 0,
    XD_BULLETIN_B,
    XD_BULLETIN_B_OLD
  } XD_bulletin_type_enum;

  /* Enumeration to data file types */
  typedef enum
  {
    XD_ORBIT_FILE,
    XD_OSF_FILE,
    XD_DORIS_FILE,
    XD_BULLETIN_FILE,
    XD_USER_OSV_LIST, /* AN-497 */
    XD_SP3_FILE, // AN-550
    XD_OEM_FILE, // AN-578
    XD_TLE_FILE // AN-583
  } XD_data_file_type_enum;

  /* DEM cache type (AN-487) */
  typedef enum
  {
    XD_NO_CACHE = 0,
    XD_PRELOAD_CACHE,
    XD_FIFO_CACHE
  } XD_Dem_cache_type_enum;

  /* DEM geoid computation flag (AN-466) */
  typedef enum
  {
    XD_DEM_GEOID_COMPUTATION_DISABLED = 0,
    XD_DEM_GEOID_COMPUTATION_ENABLED = 1
  } XD_Dem_geoid_flag_enum;

  typedef enum
  {
    XD_FIT_TLE = 0, /*Used for POF, not valid for OSV LIST*/
    XD_FIT_TLE_LIST,
    XD_ONE_TLE_PER_OSV

  } XD_TLE_fitmode_enum;

  /* AN-469: attitude model type */
  typedef enum
  {
    XD_ATT_NONE_MODEL = 0,
    XD_ATT_AOCS_MODEL,
    XD_ATT_PARAMETER_MODEL,
    XD_ATT_HARMONIC_MODEL,
    XD_ATT_FILE_MODEL,
    XD_ATT_ANGLE_MODEL,
    XD_ATT_MATRIX_MODEL,
    XD_ATT_QUATERNION_ANGLE_MODEL,
    XD_ATT_QUATERNION_MATRIX_MODEL
  } XD_Att_def_model_enum;

  /* AN-469: Attitude reference frames */
  typedef enum
  {
    XD_SAT_NOMINAL_ATT_DEF = 0,
    XD_SAT_ATT_DEF,
    XD_INSTR_ATT_DEF
  } att_def_frame;

  /* AN-469: AOCS models */
  typedef enum
  {
    XD_AOCS_GPM = 0,
    XD_AOCS_LNP,
    XD_AOCS_YSM,
    XD_AOCS_ZDOPPLER
  } XD_Sat_nominal_att_aocs_mode_enum;

  /* AN-469: Sat Nom param model */
  /* ---------------------------- */
  typedef enum
  {
    XD_MODEL_GENERIC = 0,
    XD_MODEL_ENVISAT,
    XD_MODEL_CRYOSAT,
    XD_MODEL_ADM,
    XD_MODEL_SENTINEL1,
    XD_MODEL_SENTINEL2,
    XD_MODEL_GEO,
    XD_MODEL_METOPSG /* AN-666 */
  } XD_Sat_nominal_att_model_enum;

  /* AN-469: Angle Type enum*/
  /* ----------------------- */
  typedef enum
  {
    XD_ANGLE_TYPE_TRUE_LAT_TOD = 0,
    XD_ANGLE_TYPE_TRUE_LAT_EF /* AN-508 */
  } XD_Angle_type_enum;

  /* AN-550: SP3 file type */
  /* --------------------- */
  typedef enum
  {
    XD_SP3_POSITION_TYPE,
    XD_SP3_POSITION_PLUS_VELOCITY_TYPE
  } XD_SP3_file_type_enum;

  // AN-550: File descriptor. Note that positions must correspond to sp3_file_type_descriptors in xd_sp3.h
  typedef enum
  {
    XD_SP3_GPS = 0,
    XD_SP3_MIXED,
    XD_SP3_GLONASS,
    XD_SP3_LEO,
    XD_SP3_GALILEO,
    XD_SP3_COMPASS,
    XD_SP3_QZSS
  } XD_SP3_file_descriptor_enum;

  // AN-550: Time system. Note that positions must correspond to sp3_time_systems in xd_sp3.h
  typedef enum
  {
    XD_SP3_TIME_GPS = 0,
    XD_SP3_TIME_GLONASS,
    XD_SP3_TIME_GALILEO,
    XD_SP3_TIME_TAI,
    XD_SP3_TIME_UTC,
    XD_SP3_TIME_QZSS
  } XD_SP3_time_system_enum;

  // AN-550: Sat descriptor. Note that positions must correspond to sp3_sat_descriptors in xd_sp3.h
  typedef enum
  {
    XD_SAT_GPS = 0,
    XD_SAT_GLONASS,
    XD_SAT_LEO,
    XD_SAT_GALILEO,
    XD_SAT_COMPASS,
    XD_SAT_QZSS
  } XD_SAT_SP3_sat_descriptor_enum;

  /* AN-524: DORIS file type */
  /* ----------------------- */
  typedef enum
  {
    XD_DORIS_CRYOSAT_TYPE,
    XD_DORIS_SENTINEL3_TYPE,
    XD_DORIS_JASON_TYPE, // AN-701
    XD_DORIS_UNKNOWN_TYPE // AN-701
  } XD_DORIS_file_type_enum;

  // AN-634: Extension type
  typedef enum
  {
    XD_EXTEND_NUM_OSV = 0,
    XD_EXTEND_TIME
  } XD_Extend_type_enum;

  // AN-628
  typedef enum
  {
    XD_FOV_CONSTRAINTS_SC_LINK,
    XD_FOV_CONSTRAINTS_CELESTIAL_BODY_LINK, //ANR-677
    XD_FOV_CONTRAINTS_MAX
  } XD_Fov_constraints_enum;

  // AN-740
  typedef enum
  {
    XD_DEM_DATA_TYPE_INT16 = 0,
    XD_DEM_DATA_TYPE_INT32,
    XD_DEM_DATA_TYPE_INT64,
    XD_DEM_DATA_TYPE_UINT16,
    XD_DEM_DATA_TYPE_UINT32,
    XD_DEM_DATA_TYPE_UINT64,
    XD_DEM_DATA_TYPE_FLOAT32,
    XD_DEM_DATA_TYPE_FLOAT64,
    XD_DEM_DATA_TYPE_INT8,
  } XD_Dem_data_types_enum;

  // AN-740
  typedef enum
  {
    XD_DEM_DATA_UNIT_METER = 0,
    XD_DEM_DATA_UNIT_KILOMETER
  } XD_Dem_data_units_enum;

  // AN-740
  typedef enum
  {
    XD_DEM_DATA_REFERENCE_WGS84 = 0,
    XD_DEM_DATA_REFERENCE_EGM96
  } XD_Dem_data_references_enum;

  /* --------------------------------------------- */
  /* Error codes list for all high level functions */
  /* --------------------------------------------- */

  /* Error codes list of xd_read_fhr */
  typedef enum
  {
    XD_CFI_READ_FHR_OPEN_FILE_ERR = XD_ZERO_E,
    XD_CFI_READ_FHR_GET_FIXED_HEADER_ERR,
    XD_CFI_READ_FHR_GET_EOFFS_ERR,
    XD_CFI_READ_FHR_CLOSE_FILE_ERR
  } XD_CFI_read_fhr_err_enum;

  /* Error codes list of xd_read_bulletin */
  typedef enum
  {
    XD_CFI_READ_BULLETIN_FILE_ERR = XD_ZERO_E,
    XD_CFI_READ_BULLETIN_TABLE_ERR,
    XD_CFI_READ_BULLETIN_FILE_RECOG_ERR,
    XD_CFI_READ_BULLETIN_DEPRECATED_WARN
  } XD_CFI_read_Bulletin_err_enum;

  /* Error codes list of xd_read_orbit_file */
  typedef enum
  {
    XD_CFI_READ_ORBIT_FILE_READ_ERR = XD_ZERO_E,
    XD_CFI_READ_ORBIT_FILE_VHR_NOT_FOUND_ERR,
    XD_CFI_READ_ORBIT_FILE_INPUT_RANGE_ERR,
    XD_CFI_READ_ORBIT_FILE_MEMORY_ERR,
    XD_CFI_READ_ORBIT_FILE_INTERNAL_1_ERR,
    XD_CFI_READ_ORBIT_FILE_DATA_READ_ERR,
    XD_CFI_READ_ORBIT_FILE_GAP_ERR,
    XD_CFI_READ_ORBIT_FILE_REF_CS_WARN,
    XD_CFI_READ_ORBIT_FILE_DEFAULT_TIME_REF_OF_WARN,
    XD_CFI_READ_ORBIT_FILE_REPEATED_OSV_WARN, //AN-577
    XD_CFI_READ_ORBIT_FILE_GAP_WARN, //AN-577
    XD_CFI_READ_ORBIT_FILE_TIME_GOING_BACK_WARN, //AN-577
    XD_CFI_READ_ORBIT_FILE_ORBIT_NUMBER_WARN, //AN-577
    XD_CFI_READ_ORBIT_FILE_MEO_GAP_WARN // AN-867
  } XD_CFI_read_orbit_file_err_enum;

  /* Error codes list of xd_orbit_file_decimate */
  typedef enum
  {
    XD_CFI_ORBIT_FILE_DECIMATE_MEM_ERR = XD_ZERO_E,
    XD_CFI_ORBIT_FILE_DECIMATE_DELTA_WARN,
    XD_CFI_ORBIT_FILE_DECIMATE_VAL_TIME_ERR
  } XD_CFI_Orbit_file_decimate_err_enum;

  /* Error codes list of xd_read_doris */
  typedef enum
  {
    XD_CFI_READ_DORIS_ERROR_IN_MEMORY_ASIG_ERR = XD_ZERO_E,
    XD_CFI_READ_DORIS_WRONG_TIME_MODE_ERR,
    XD_CFI_READ_DORIS_WRONG_TIME_1_ERR,
    XD_CFI_READ_DORIS_WRONG_TIME_2_ERR,
    XD_CFI_READ_DORIS_NO_FILENAME_ERR,
    XD_CFI_READ_DORIS_CANNOT_OPEN_ERR,
    XD_CFI_READ_DORIS_FINDKW_ERROR_ERR,
    XD_CFI_READ_DORIS_READ_ERR,
    XD_CFI_READ_DORIS_READ_BIN_ERR,
    XD_CFI_READ_DORIS_ASCII_TO_PROCESSING_ERR,
    XD_CFI_READ_DORIS_GAP_IN_FILE_ERR,
    XD_CFI_READ_DORIS_DOES_NOT_COVER_TIME_INTERVAL_ERR,
    XD_CFI_READ_DORIS_NO_SYNC_WORD_ERR,
    XD_CFI_READ_DORIS_DEFAULT_TIME_REF_OF_WARN,
    XD_CFI_READ_DORIS_DEFAULT_ORBIT_WARN,
    XD_CFI_READ_DORIS_WRONG_CRC_WARN,
    XD_CFI_READ_DORIS_BAD_QUALITY_PACKAGE_WARN,
    XD_CFI_READ_DORIS_OSV_TOO_CLOSE_WARN,
    XD_CFI_READ_DORIS_GAP_IN_FILE_WARN,
    XD_CFI_READ_DORIS_KW_EXISTS_ERR, // AN-582
    XD_CFI_READ_DORIS_TYPE_NOT_RECOGNIZED_ERR, // AN-582
    XD_CFI_READ_DORIS_MAX_NUM_CRC_WARN, // AN-582
    XD_CFI_READ_DORIS_MAX_NUM_BAD_QUALITY_WARN, // AN-582
    XD_CFI_READ_DORIS_SEQ_COUNTER_WARN, // AN-565
    XD_CFI_READ_DORIS_OSV_REPEATED_WARN,
    XD_CFI_READ_DORIS_NOMINAL_RATE_GAP_WARN, // AN-565
    XD_CFI_READ_DORIS_UNKNOWN_DORIS_TYPE_ERR,
    XD_CFI_READ_DORIS_SEQ_MISMATCH_WARN
  } XD_CFI_read_doris_err_enum;

  /* Error codes list of xd_read_doris_header */
  typedef enum
  {
    XD_CFI_READ_DORIS_HEADER_NO_FILENAME_ERR = XD_ZERO_E,
    XD_CFI_READ_DORIS_HEADER_CANNOT_OPEN_ERR,
    XD_CFI_READ_DORIS_HEADER_FINDKW_ERROR_ERR,
    XD_CFI_READ_DORIS_HEADER_READ_ERR
  } XD_CFI_read_doris_header_err_enum;

  /* Error codes list of xd_read_osf */
  typedef enum
  {
    XD_CFI_READ_XML_OSF_INIT_PARSER_ERR = XD_ZERO_E,
    XD_CFI_READ_XML_OSF_XML_DATA_BLOCK_ERR,
    XD_CFI_READ_XML_OSF_XML_ATTRIBUTE_ERR,
    XD_CFI_READ_XML_OSF_XML_TYPE_ERR,
    XD_CFI_READ_XML_OSF_READ_PARAM_ERR,
    XD_CFI_READ_XML_OSF_XML_DATA_BLOCK_SIZE_ERR,
    XD_CFI_READ_XML_OSF_MEMORY_ERR,
    XD_CFI_READ_XML_OSF_VHR_NOT_FOUND_ERR,
    XD_CFI_READ_XML_OSF_TIME_REF_OF_ERR,
    XD_CFI_READ_XML_OSF_TIME_REF_OF_WARN
  } XD_CFI_read_osf_err_enum;

  /* Error codes list of xd_read_sdf */
  typedef enum
  {
    XD_CFI_READ_SDF_OPEN_FILE_ERR = XD_ZERO_E,
    XD_CFI_READ_SDF_MEMORY_ERR,
    XD_CFI_READ_SDF_REC_READ_ERR,
    XD_CFI_READ_SDF_VERSION_ERR
  } XD_CFI_read_sdf_err_enum;

  /* Error codes list of xd_read_stf */
  typedef enum
  {
    XD_CFI_READ_STF_INIT_PARSER_ERR = XD_ZERO_E,
    XD_CFI_READ_STF_VHR_ERR,
    XD_CFI_READ_STF_PARAM_READ_ERR,
    XD_CFI_READ_STF_DATA_BLOCK_ERR,
    XD_CFI_READ_STF_ATTRIBUTE_ERR,
    XD_CFI_READ_STF_XML_TYPE_ERR,
    XD_CFI_READ_STF_DATA_BLOCK_SIZE_ERR,
    XD_CFI_READ_STF_MEMORY_ERR,
    XD_CFI_READ_STF_REC_READ_ERR,
    XD_CFI_READ_STF_WRONG_LAT_ERR,
    XD_CFI_READ_STF_WRONG_LONG_ERR,
    XD_CFI_READ_STF_SWATH_ERR,
    XD_CFI_READ_STF_CHECK_LATLON_ERR
  } XD_CFI_read_stf_err_enum;

  /* Error codes list of xd_read_stf_vhr */
  typedef enum
  {
    XD_CFI_READ_STF_VHR_INIT_PARSER_ERR = XD_ZERO_E,
    XD_CFI_READ_STF_VHR_VARIABLE_HEADER_ERR,
    XD_CFI_READ_STF_VHR_INTERNAL_1_ERR,
    XD_CFI_READ_STF_VHR_PARAM_READ_ERR,
    XD_CFI_READ_STF_VHR_SWATH_TYPE_ERR,
    XD_CFI_READ_STF_VHR_SWATH_POINT_TYPE_ERR,
    XD_CFI_READ_STF_VHR_ORBIT_PARAMS_ERR,
    XD_CFI_READ_STF_VHR_GEOM_PARAMS_ERR,
    XD_CFI_READ_STF_VHR_ALTITUDE_READ_ERR,
    XD_CFI_READ_STF_VHR_MEMORY_ERR,
    XD_CFI_READ_STF_VHR_CHECK_FORMAT_ERR,
    XD_CFI_READ_STF_VHR_FORMAT_NEW_ERR,
    XD_CFI_READ_STF_VHR_FORMAT_OLD_ERR,
    XD_CFI_READ_STF_VHR_SWATH_TYPE_FUNC_ERR,
    XD_CFI_READ_STF_VHR_SWATH_POINT_TYPE_FUNC_ERR,
    XD_CFI_READ_STF_VHR_ORBGEOMSV_ERR,
    XD_CFI_READ_STF_VHR_TSTEP_ALT_REFR_FUNCT_ERR
  } XD_CFI_read_stf_vhr_err_enum;

  /* Error codes list of xd_read_att */
  typedef enum
  {
    XD_CFI_READ_ATT_INIT_PARSER_ERR = XD_ZERO_E,
    XD_CFI_READ_ATT_READ_PARAM_ERR,
    XD_CFI_READ_ATT_WRONG_FILE_TYPE_ERR,
    XD_CFI_READ_XML_ATT_NAVIGATION_ERR,
    XD_CFI_READ_ATT_WRONG_DATA_TYPE_ERR,
    XD_CFI_READ_ATT_INCONSISTENT_DATA_TYPE_ERR,
    XD_CFI_READ_ATT_XML_DATA_BLOCK_SIZE_ERR,
    XD_CFI_READ_ATT_WRONG_REF_FRAME_ERR,
    XD_CFI_READ_ATT_READ_LIST_ERR,
    XD_CFI_READ_ATT_TIME_CONV_ERR,
    XD_CFI_READ_ATT_MEMORY_ERR,
    XD_CFI_READ_ATT_CLEANUP_PARSER_ERR,
    XD_CFI_READ_ATT_WRONG_TIME_REF_ERR,
    XD_CFI_READ_ATT_WRONG_QUATERNION_ERR,
    XD_CFI_READ_ATT_WRONG_ANGLE_ERR,
    XD_CFI_READ_ATT_MAX_GAP_ERR,
    XD_CFI_READ_ATT_OBSOLETE_TAG_WARN, //AN-516
  } XD_CFI_read_att_err_enum;

  /* Error codes list of xd_attitude_file_decimate */
  typedef enum
  {
    XD_CFI_ATTITUDE_FILE_DECIMATE_MEM_ERR = XD_ZERO_E,
    XD_CFI_ATTITUDE_FILE_DECIMATE_DELTA_WARN,
    XD_CFI_ATTITUDE_FILE_DECIMATE_VAL_TIME_ERR,
    XD_CFI_ATTITUDE_FILE_DECIMATE_ATT_NOT_UTC_WARN
  } XD_CFI_Attitude_file_decimate_err_enum;

  /* Error code list of "xd_read_star_tracker" */
  typedef enum
  {
    XD_CFI_READ_STR_TRACKER_OPEN_FILE_ERR = XD_ZERO_E,
    XD_CFI_READ_STR_TRACKER_READ_FILE_ERR,
    XD_CFI_READ_STR_TRACKER_MEMORY_FILE_ERR,
    XD_CFI_READ_STR_TRACKER_GAP_ERR,
    XD_CFI_READ_STR_TRACKER_NO_ENOUGH_DATA_ERR,
    XD_CFI_READ_STR_TRACKER_READ_FILE_LIST_ERR
  } XD_CFI_read_star_tracker_err_enum;

  /* error code list of "xd_read_star_tracker_conf_file" */
  typedef enum
  {
    XD_CFI_READ_STR_CONF_FILE_READ_FILE_ERR = XD_ZERO_E,
    XD_CFI_READ_STR_CONF_FILE_READ_ABERR_ERR,
    XD_CFI_READ_STR_CONF_FILE_READ_LIMITS_ERR,
    XD_CFI_READ_STR_CONF_FILE_READ_ANGLES_ERR
  } XD_CFI_read_star_tracker_conf_file_err_enum;

  /* error codes list of "xd_read_dem_config_file" */
  typedef enum
  {
    XD_CFI_READ_DEM_CONFIG_FILE_OPEN_ERR = XD_ZERO_E,
    XD_CFI_READ_DEM_CONFIG_FILE_READ_ERR,
    XD_CFI_READ_DEM_CONFIG_FILE_READ_MODEL_ERR,
    XD_CFI_READ_DEM_CONFIG_FILE_MEMORY_ERR,
    XD_CFI_READ_DEM_CONFIG_FILE_OPEN_DEM_FILE_WARN,
    XD_CFI_READ_DEM_CONFIG_FILE_READ_DEM_FILE_ERR,
    XD_CFI_READ_DEM_CONFIG_FILE_DEPRECATED_WARN,
    XD_CFI_READ_DEM_CONFIG_FILE_CACHE_TYPE_WARN,
    XD_CFI_READ_DEM_CONFIG_FILE_CACHE_TYPE_NO_CACHE_WARN
  } XD_CFI_read_dem_config_file_err_enum;

  /* error codes list of "xd_read_dem" */
  typedef enum
  {
    XD_CFI_READ_DEM_MEMORY_ERR = XD_ZERO_E,
    XD_CFI_READ_DEM_NO_CONFIG_FILE_ERR,
    XD_CFI_READ_DEM_WRONG_FILENAME_ERR,
    XD_CFI_READ_DEM_OPEN_FILE_ERR,
    XD_CFI_READ_DEM_READ_FILE_ERR,
    XD_CFI_READ_DEM_UNKNOWN_MODEL_ERR,
    XD_CFI_READ_DEM_MODEL_NOT_SUPPORTED_ERR //AN-619
  } XD_CFI_read_dem_err_enum;

  /* Error code list of xd_read_star */
  typedef enum
  {
    XD_CFI_READ_STAR_FILE_NOT_FOUND_ERR = XD_ZERO_E,
    XD_CFI_READ_STAR_STAR_NOT_FOUND_ERR
  } XD_CFI_read_star_err_enum;

  /* Error code list of xd_read_star_file */
  typedef enum
  {
    XD_CFI_READ_STAR_FILE_FILE_NOT_FOUND_ERR = XD_ZERO_E,
    XD_CFI_READ_STAR_FILE_MEMORY_ERR,
    XD_CFI_READ_STAR_FILE_NO_STARS_ERR
  } XD_CFI_read_star_file_err_enum;

  /* Error code list of xd_read_star_id */
  typedef enum
  {
    XD_CFI_READ_STAR_ID_FILE_NOT_FOUND_ERR = XD_ZERO_E,
    XD_CFI_READ_STAR_ID_MEMORY_ERR,
    XD_CFI_READ_STAR_ID_NO_STARS_ERR
  } XD_CFI_read_star_id_err_enum;

  /* Error codes list of xd_read_station */
  typedef enum
  {
    XD_CFI_READ_STATION_INIT_PARSER_ERR = XD_ZERO_E,
    XD_CFI_READ_STATION_DATA_BLOCK_ERR,
    XD_CFI_READ_STATION_DATA_BLOCK_ATTRIBUTE_ERR,
    XD_CFI_READ_STATION_XML_TYPE_ERR,
    XD_CFI_READ_STATION_LIST_GS_READ_ERR,
    XD_CFI_READ_STATION_LIST_GS_SIZE_ERR,
    XD_CFI_READ_STATION_INTERNAL_1_ERR,
    XD_CFI_READ_STATION_STATION_ID_READ_ERR,
    XD_CFI_READ_STATION_STATION_ID_NOT_FOUND_ERR,
    XD_CFI_READ_STATION_REC_READ_ERR,
    XD_CFI_READ_STATION_INIT_ERR
  } XD_CFI_read_station_err_enum;

  /* Error codes list of xd_read_station_file */
  typedef enum
  {
    XD_CFI_READ_STATION_FILE_INIT_PARSER_ERR = XD_ZERO_E,
    XD_CFI_READ_STATION_FILE_DATA_BLOCK_ERR,
    XD_CFI_READ_STATION_FILE_DATA_BLOCK_ATTRIBUTE_ERR,
    XD_CFI_READ_STATION_FILE_XML_TYPE_ERR,
    XD_CFI_READ_STATION_FILE_LIST_GS_READ_ERR,
    XD_CFI_READ_STATION_FILE_LIST_GS_SIZE_ERR,
    XD_CFI_READ_STATION_FILE_MEM_ERR,
    XD_CFI_READ_STATION_FILE_REC_READ_ERR,
    XD_CFI_READ_STATION_FILE_INIT_ERR
  } XD_CFI_read_station_file_err_enum;

  /* Error codes list of xd_read_station_id */
  typedef enum
  {
    XD_CFI_READ_STATION_ID_INIT_PARSER_ERR = XD_ZERO_E,
    XD_CFI_READ_STATION_ID_DATA_BLOCK_ERR,
    XD_CFI_READ_STATION_ID_LIST_GS_READ_ERR,
    XD_CFI_READ_STATION_ID_LIST_GS_SIZE_ERR,
    XD_CFI_READ_STATION_ID_MEM_ERR,
    XD_CFI_READ_STATION_ID_READ_ID_ERR,
    XD_CFI_READ_STATION_ID_INIT_ERR
  } XD_CFI_read_station_id_err_enum;

  /* Error codes list of xd_read_zone  */
  typedef enum
  {
    XD_CFI_READ_ZONE_INTERNAL_1_ERR = XD_ZERO_E,
    XD_CFI_READ_ZONE_ZONE_ID_READ_ERR,
    XD_CFI_READ_ZONE_ZONE_ID_NOT_FOUND_ERR,
    XD_CFI_READ_ZONE_RECORD_READ_ERR,
    XD_CFI_READ_ZONE_DATA_INIT_ERR /* AN-713 */
  } XD_CFI_read_zone_err_enum;

  /* Error codes list of xd_read_zone_file  */
  typedef enum
  {
    XD_CFI_READ_ZONE_FILE_MEM_ERR = XD_ZERO_E,
    XD_CFI_READ_ZONE_FILE_RECORD_READ_ERR,
    XD_CFI_READ_ZONE_FILE_DATA_INIT_ERR
  } XD_CFI_read_zone_file_err_enum;

  /* Error codes list of xd_read_zone_id  */
  typedef enum
  {
    XD_CFI_READ_ZONE_ID_MEMORY_ERR = XD_ZERO_E,
    XD_CFI_READ_ZONE_ID_READ_ZONE_ERR,
    XD_CFI_READ_ZONE_ID_DATA_INIT_ERR /* AN-713 */
  } XD_CFI_read_zone_id_err_enum;

  /* Error codes list of xd_read_att_def */ /* ANR-469*/
  typedef enum
  {
    XD_CFI_READ_ATT_DEF_OPEN_FILE_ERR = XD_ZERO_E,
    XD_CFI_READ_ATT_DEF_MEMORY_ERR,
    XD_CFI_READ_ATT_DEF_REC_READ_ERR,
    XD_CFI_READ_ATT_DEF_OBSOLETE_TAG_WARN /*AN-516*/
  } XD_CFI_read_ATT_DEF_err_enum;

  typedef enum /* Error codes list of xd_write_att_def AN-469 */
  {
    XD_CFI_WRITE_ATT_DEF_CREATE_TREE_ERR = XD_ZERO_E,
    XD_CFI_WRITE_ATT_DEF_CREATE_ROOT_ERR,
    XD_CFI_WRITE_ATT_DEF_WRITE_FHR_ERR,
    XD_CFI_WRITE_ATT_DEF_WRITE_ERR,
    XD_CFI_WRITE_ATT_DEF_SET_SCHEMA_WARN,
    XD_CFI_WRITE_ATT_DEF_ADD_CHILD_ERR,
    XD_CFI_WRITE_ATT_DEF_ADD_ATTRIBUTE_ERR,
    XD_CFI_WRITE_ATT_DEF_SET_NODE_VALUE_ERR,
    XD_CFI_WRITE_ATT_DEF_GET_NODE_VALUE_ERR,
    XD_CFI_WRITE_ATT_DEF_DATA_ERR
  } XD_CFI_Write_att_def_err_enum;

  /* Error codes list of xd_write_orbit_file */
  typedef enum
  {
    XD_CFI_WRITE_ORBIT_FILE_CREATE_TREE_ERR = XD_ZERO_E,
    XD_CFI_WRITE_ORBIT_FILE_CREATE_ROOT_ERR,
    XD_CFI_WRITE_ORBIT_FILE_WRITE_FHR_ERR,
    XD_CFI_WRITE_ORBIT_FILE_WRITE_VHR_ERR,
    XD_CFI_WRITE_ORBIT_FILE_WRITE_DBL_ERR,
    XD_CFI_WRITE_ORBIT_FILE_WRITE_ERR,
    XD_CFI_WRITE_ORBIT_FILE_SET_SCHEMA_WARN
  } XD_CFI_write_orbit_file_err_enum;

  /* Error codes list of xd_write_doris */
  typedef enum
  {
    XD_CFI_WRITE_DORIS_OPEN_ERR = XD_ZERO_E,
    XD_CFI_WRITE_DORIS_WRITE_FHR_ERR,
    XD_CFI_WRITE_DORIS_WRITE_BINARY_ERR,
    XD_CFI_CREATE_HEADER_FILE_MPH_WRITE_ERR,
    XD_CFI_CREATE_HEADER_FILE_SPH_WRITE_ERR,
    XD_CFI_CREATE_HEADER_FILE_ORBIT_INFORMATION_WRITE_ERR,
    XD_CFI_CREATE_HEADER_FILE_PRODUCT_LOCATION_WRITE_ERR,
    XD_CFI_CREATE_HEADER_FILE_PRODUCT_CONFIDENCE_DATA_WRITE_ERR
  } XD_CFI_write_doris_err_enum;

  /* Error codes list of xd_write_osf */
  typedef enum
  {
    XD_CFI_WRITE_OSF_CREATE_TREE_ERR = XD_ZERO_E,
    XD_CFI_WRITE_OSF_CREATE_ROOT_ERR,
    XD_CFI_WRITE_OSF_WRITE_FHR_ERR,
    XD_CFI_WRITE_OSF_WRITE_VHR_ERR,
    XD_CFI_WRITE_OSF_WRITE_DBL_ERR,
    XD_CFI_WRITE_OSF_WRITE_ERR,
    XD_CFI_WRITE_OSF_SET_SCHEMA_WARN
  } XD_CFI_write_osf_err_enum;

  /* Error codes list of xd_write_stf */
  typedef enum
  {
    XD_CFI_WRITE_STF_CREATE_TREE_ERR = XD_ZERO_E,
    XD_CFI_WRITE_STF_CREATE_ROOT_ERR,
    XD_CFI_WRITE_STF_WRITE_FHR_ERR,
    XD_CFI_WRITE_STF_WRITE_VHR_ERR,
    XD_CFI_WRITE_STF_WRITE_DBL_ERR,
    XD_CFI_WRITE_STF_WRITE_ERR,
    XD_CFI_WRITE_STF_SET_SCHEMA_WARN
  } XD_CFI_write_stf_err_enum;

  /* Error codes list of xd_write_att */
  typedef enum
  {
    XD_CFI_WRITE_ATT_CREATE_TREE_ERR = XD_ZERO_E,
    XD_CFI_WRITE_ATT_CREATE_ROOT_ERR,
    XD_CFI_WRITE_ATT_WRITE_FHR_ERR,
    XD_CFI_WRITE_ATT_CREATE_NODE_ERR,
    XD_CFI_WRITE_ATT_GET_ASCII_TIME_ERR,
    XD_CFI_WRITE_ATT_GOTO_NODE_ERR,
    XD_CFI_WRITE_ATT_WRITE_ERR,
    XD_CFI_WRITE_ATT_SET_SCHEMA_WARN,
    XD_CFI_WRITE_ATT_CREATE_DATA_BLOCK_ERR,
    XD_CFI_WRITE_ATT_CREATE_LIST_ERR
  } XD_CFI_write_att_err_enum;

  /* Error codes list of xd_xml_validate */
  typedef enum
  {
    XD_CFI_XML_VALIDATE_SET_SCHEMA_ERR = XD_ZERO_E,
    XD_CFI_XML_VALIDATE_INIT_PARSER_ERR,
    XD_CFI_XML_VALIDATE_MEMORY_ERR,
    XD_CFI_XML_VALIDATE_NO_SCHEMA_ERR,
    XD_CFI_XML_VALIDATE_WRONG_MODE_ERR,
    XD_CFI_XML_VALIDATE_OPEN_FILE_ERR,
    XD_CFI_XML_VALIDATE_COPY_FILE_ERR,
    XD_CFI_XML_VALIDATE_NO_SCHEMA_IN_FILE_ERR,
    XD_CFI_XML_VALIDATE_INCONSITENT_SCHEMA_VERS_ERR,
    XD_CFI_XML_VALIDATE_READ_NS_ERR,
    XD_CFI_XML_VALIDATE_NO_SCH_VERS_IN_FILE_WARN,
    XD_CFI_XML_VALIDATE_NO_VERS_IN_SCHEMA_WARN,
    XD_CFI_XML_VALIDATE_LESS_SCHEMA_VERS_WARN,
    XD_CFI_XML_VALIDATE_GREATER_SCHEMA_VERS_WARN,
    XD_CFI_XML_VALIDATE_REMOVE_TMP_FILE_ERR
  } XD_CFI_xml_validate_err_enum;

  /* Error codes list of xd_read_tle */
  typedef enum
  {
    XD_CFI_READ_TLE_FILE_NOT_FOUND_ERR = XD_ZERO_E,
    XD_CFI_READ_TLE_WRONG_LINE0_ERR,
    XD_CFI_READ_TLE_WRONG_LINE1_ERR,
    XD_CFI_READ_TLE_WRONG_LINE2_ERR,
    XD_CFI_READ_TLE_MEM_ERR,
    XD_CFI_READ_TLE_WRONG_SAT_ERR,
    XD_CFI_READ_TLE_NO_LINES_ERR,
    XD_CFI_READ_TLE_WRONG_CHECKSUM1_WARN,
    XD_CFI_READ_TLE_WRONG_CHECKSUM2_WARN
  } XD_CFI_read_tle_err_enum;

  /* Error codes list of xd_write_tle */
  typedef enum
  {
    XD_CFI_WRITE_TLE_FILE_OPEN_ERR = XD_ZERO_E,
    XD_CFI_WRITE_TLE_WRITE_ERR
  } XD_CFI_write_tle_err_enum;

  /* Error codes list of xd_read_precise_propag_file */
  typedef enum
  {
    XD_CFI_READ_PRECISE_PROPAG_INIT_PARSER_ERR = XD_ZERO_E,
    XD_CFI_READ_PRECISE_PROPAG_READ_PARAM_ERR,
    XD_CFI_READ_PRECISE_PROPAG_WRONG_FLAG_ERR,
    XD_CFI_READ_PRECISE_PROPAG_CLEANUP_PARSER_ERR
  } XD_CFI_read_precise_propag_enum;

  /* Error codes list of xd_write_precise_propag_file */
  typedef enum
  {
    XD_CFI_WRITE_PRECISE_PROPAG_CREATE_TREE_ERR = XD_ZERO_E,
    XD_CFI_WRITE_PRECISE_PROPAG_CREATE_ROOT_ERR,
    XD_CFI_WRITE_PRECISE_PROPAG_WRITE_ERR,
    XD_CFI_WRITE_PRECISE_PROPAG_WRITE_FHR_ERR,
    XD_CFI_WRITE_PRECISE_PROPAG_WRITE_DBL_ERR,
    XD_CFI_WRITE_PRECISE_PROPAG_SET_SCHEMA_WARN
  } XD_CFI_write_precise_propag_enum;

  /* Error codes list of xd_read_sp3 (AN-550) */
  typedef enum
  {
    XD_CFI_READ_SP3_OPEN_FILE_ERR = XD_ZERO_E,
    XD_CFI_READ_SP3_READ_LINE_ERR,
    XD_CFI_READ_SP3_WRONG_FILE_VERSION_ERR,
    XD_CFI_READ_SP3_WRONG_FILE_TYPE_ERR,
    XD_CFI_READ_SP3_GET_PROC_TIME_ERR,
    XD_CFI_READ_SP3_SAT_ID_ERR,
    XD_CFI_READ_SP3_MEMORY_ERR,
    XD_CFI_READ_SP3_NUM_SAT_ID_ERR,
    XD_CFI_READ_SP3_SAT_ACCURACY_ERR,
    XD_CFI_READ_SP3_TIME_SYSTEM_ERR,
    XD_CFI_READ_SP3_TYPE_DESCRIPTOR_ERR,
    XD_CFI_READ_SP3_READ_CONFIG_ERR, // AN-634
    XD_CFI_READ_SP3_TIME_GOING_BACK_WARN, //AN-577
    XD_CFI_READ_SP3_REPEATED_OSV_WARN, //AN-577
    XD_CFI_READ_SP3_FITING_OSV_ARRAY_TO_REQUESTED_TIME_ERR,
    XD_CFI_READ_SP3_CONFIG_TIME_REF_WARN, // AN-634
    XD_CFI_READ_SP3_READ_METADATA_ERR, /* AN-713 */
    XD_CFI_READ_SP3_READ_EPOCHS_ERR /* AN-713 */
  } XD_CFI_read_sp3_err_enum;

  /* Error codes list of xd_xslt_add (AN-546) */
  typedef enum
  {
    XD_CFI_XSLT_ADD_WRONG_FILE_TYPE_ERR = XD_ZERO_E,
    XD_CFI_XSLT_ADD_INIT_PARSER_ERR,
    XD_CFI_XSLT_ADD_SAVE_DOC_ERR,
    XD_CFI_XSLT_ADD_READ_ATT_ERR,
    XD_CFI_XSLT_ADD_WRONG_ATT_FILE_TYPE_ERR
  } XD_CFI_xslt_add_err_enum;

  /* Error codes list of xd_read_oem (AN-578) */
  typedef enum
  {
    XD_CFI_READ_OEM_READ_CONFIG_ERR = XD_ZERO_E, // AN-634
    XD_CFI_READ_OEM_OPEN_FILE_ERR,
    XD_CFI_READ_OEM_MEMORY_ERR,
    XD_CFI_READ_OEM_WRONG_REF_FRAME_ERR,
    XD_CFI_READ_OEM_WRONG_TIME_SYSTEM_ERR,
    XD_CFI_READ_OEM_READ_LINE_ERR,
    XD_CFI_READ_OEM_GET_PROC_TIME_ERR,
    XD_CFI_READ_OEM_TIME_GOING_BACK_WARN, //AN-577
    XD_CFI_READ_OEM_REPEATED_OSV_WARN, //AN-577
    XD_CFI_READ_OEM_FITING_OSV_ARRAY_TO_REQUESTED_TIME_ERR, //AN-577
    XD_CFI_READ_OEM_CONFIG_TIME_REF_WARN, // AN-634
    XD_CFI_READ_OEM_READ_XML_TAG_ERR // AN-850
  } XD_CFI_read_oem_err_enum;

  /* Error codes list of xd_orbit_file_diagnostics (AN-576) */
  typedef enum
  {
    XD_CFI_ORBIT_FILE_DIAGNOSTICS_DETECT_INPUT_ERR = XD_ZERO_E,
    XD_CFI_ORBIT_FILE_DIAGNOSTICS_COMPUTE_DIAGNOSTICS_ERR,
    XD_CFI_ORBIT_FILE_DIAGNOSTICS_MEMORY_ERR,
    XD_CFI_ORBIT_FILE_DIAGNOSTICS_AUTOMATIC_DETECTION_WARN,
    XD_CFI_ORBIT_FILE_DIAGNOSTICS_GET_OSV_ERR
  } XD_CFI_orbit_file_diagnostics_err_enum;

  /* Error codes list of xd_orbit_file_diagnostics (AN-628) */
  typedef enum
  {
    XD_CFI_READ_FOV_READ_CONFIG_ERR = XD_ZERO_E,
    XD_CFI_READ_FOV_INIT_PARSER_ERR,
    XD_CFI_READ_FOV_PARAM_READ_ERR,
    XD_CFI_READ_FOV_ATTRIBUTE_ERR,
    XD_CFI_READ_FOV_WRONG_VALUE_ERR
  } XD_CFI_fov_file_err_enum;

  /* Error codes list of xd_eoffs_version */
  typedef enum
  {
    XD_CFI_SET_FILE_FORMAT_STANDARD_VERSION_WRONG_INPUT_ERR = XD_ZERO_E,
    XD_CFI_DETECT_FILE_FORMAT_STANDARD_VERSION_WRONG_INPUT_ERR
  } XD_CFI_eoffs_version; // AN-638

  /* Error codes list of xd_read_aem (AN-856) */
  typedef enum
  {
    XD_CFI_READ_AEM_OPEN_FILE_ERR = XD_ZERO_E,
    XD_CFI_READ_AEM_MEMORY_ERR,
    XD_CFI_READ_AEM_HEADER_ERR,
    XD_CFI_READ_AEM_WRONG_METADATA_ERR,
    XD_CFI_READ_AEM_WRONG_EPHEMERIS_ERR,
    XD_CFI_READ_AEM_READ_XML_TAG_ERR
  } XD_CFI_read_aem_err_enum;

  /* ------------------------------------------------ */
  /* DATA STRUCTURES FOR EARTH EXPLORER FIXED HEADERS */
  /* ------------------------------------------------ */

  typedef struct
  {
    char file_name[XD_MAX_STR];
    char schema[XD_MAX_STR];
    char file_description[XD_MAX_STR];
    char mission[XD_MAX_STR];
    char file_class[XD_MAX_STR];
    char file_type[XD_MAX_STR];
    long version;
    char eoffs_version[XD_MAX_STR];
    char val_start_date[XD_MAX_LENGTH_32];
    char val_stop_date[XD_MAX_LENGTH_32];
    char system[XD_MAX_STR];
    char creator[XD_MAX_STR];
    char creator_version[XD_MAX_STR];
    char creation_date[XD_MAX_LENGTH_32];
  } xd_fhr;

  typedef struct
  {
    long sat_id;
    XD_File_types filetype;
  } xd_fileinfo;

  /* ------------------------------------------------ */
  /* DATA STRUCTURES FOR IERS BULLETIN B (DEPRECATED)*/
  /* ------------------------------------------------ */
  typedef struct
  {
    double day;
    double ut1_utc;
    double ut1_tai;
  } xd_bulb_table;

  typedef struct
  {
    xd_bulb_table table1[XD_MAX_LENGTH_100];
    xd_bulb_table table2[XD_MAX_LENGTH_100];
  } xd_iers_bulletin_b;

  /* --------------------------------------------------- */
  /* DATA STRUCTURES FOR IERS BULLETIN A and B (ANR-438) */
  /* --------------------------------------------------- */
  typedef struct
  {
    double x;
    double y;
  } xd_polar_motion_params;

  typedef struct
  {
    double day;
    double ut1_utc;
    double ut1_tai;
    xd_polar_motion_params polar_motion_params;
  } xd_iers_bulletin_b_rec;

  typedef struct
  {
    double day;
    double ut1_utc;
    double ut1_tai;
    xd_polar_motion_params polar_motion_params;
  } xd_iers_bulletin_a_rec;

  /* Parameters for polar motion formula */
  typedef struct
  {
    /* x parameter formula */
    double ax; /* Constant term */
    double bx; /* cos A coefficient */
    double cx; /* sin A coefficient */
    double dx; /* cos C coefficient */
    double ex; /* sin C coefficient */

    /* y parameter formula */
    double ay; /* Constant term */
    double by; /* cos A coefficient */
    double cy; /* sin A coefficient */
    double dy; /* cos C coefficient */
    double ey; /* sin C coefficient */

    /* A and C parameters */
    double A_ref; /* Reference day for A parameter formula */
    double A_div; /* Divisor for A parameter formula */
    double C_ref; /* Reference day for C parameter formula */
    double C_div; /* Divisor for C parameter formula */

  } xd_polar_motion_formula;

  /* Parameters for time correlation formula (UT1-UTC) */
  typedef struct
  {
    double a; /* Constant term */
    double b; /* Linear coefficient */
    double b_ref; /* Reference day (MJD) */
  } xd_time_correlation_formula;

  typedef struct
  {
    char bulletin_id[XD_MAX_STR]; /* �Example: 28 July 2011 Vol. XXIV No. 030� */
    long num_final_table;
    long num_preliminary_table;
    xd_iers_bulletin_b_rec* final_table;
    xd_iers_bulletin_b_rec* preliminary_table;
  } xd_iers_bulletin_b_file;

  typedef struct
  {
    char bulletin_id[XD_MAX_STR]; /* �Example: 28 July 2011 Vol. XXIV No. 030� */
    long num_rec_pred_table;
    xd_iers_bulletin_a_rec* prediction_table;
    xd_polar_motion_formula polar_motion_formula;
    xd_time_correlation_formula time_correlation_formula;
  } xd_iers_bulletin_a_file;

  typedef union
  {
    xd_iers_bulletin_a_file iers_bulletin_a_file;
    xd_iers_bulletin_b_file iers_bulletin_b_file;
  } xd_iers_bulletin_file_union;

  typedef struct
  {
    long bulletin_type; /* BULLETIN_A or BULLETIN_B (XD_bulletin_type_enum) */
    xd_iers_bulletin_file_union iers_bulletin_file;
  } xd_iers_bulletin_file;

  /* ------------------------------- */
  /* DATA STRUCTURES FOR ORBIT FILES */
  /* ------------------------------- */

  typedef struct
  {
    double tai_time; /* TAI time */
    double ut1_time; /* UT1 time */
    double utc_time; /* UTC time AN-525/549 */
    double tai_utc; /* difference between TAI and UTC */
    double tai_ut1; /* difference between TAI and UT1 */
    double tai_gps; /* difference between TAI and GPS */
  } xd_time_rec;

  typedef struct
  {
    double tai_time; /* TAI time */
    double utc_time; /* UTC time */
    double ut1_time; /* UT1 time */
    long abs_orbit; /* absolute orbit */
    long ref_frame; /* reference frame */
    long time_ref_of; /* Reference time to be considered as original */
    double pos[XD_MAX_LENGTH_3]; /* position of the OSV (x, y, z) components*/
    double vel[XD_MAX_LENGTH_3]; /* velocity of the OSV (x, y, z) components*/
    double quality; /* quality index = uncertainty of the location in EF (LSB = 1cm)*/
  } xd_osv_rec;

  typedef struct
  {
    long num_rec; /* size of the following arrays */
    xd_osv_rec* osv_rec; /* array of state vectors */
  } xd_orbit_file;

  /* ------------------------------- */
  /* DATA STRUCTURES FOR DORIS FILES */
  /* ------------------------------- */

  typedef struct
  {
    long file_type; /* file type (XD_DORIS_file_type_enum) */
    long num_rec; /* Number of elements in osv_rec */
    xd_osv_rec* osv_rec; /* array of state vectors (EF) */
    long num_rec_j2; /* Number of elements in osv_rec_j2 */
    xd_osv_rec* osv_rec_j2; /* array of state vectors (J2000) */
    double leap_time; /* leap time */
    int leap_sign; /* leap sign */
    long abs_orbit; /* first absolute orbit number */
    long rel_orbit; /* first relative orbit number */
  } xd_doris_file;

  typedef struct
  {
    char filename[XD_MAX_STR];
    char sensing_start[XD_MAX_LENGTH_30]; /* UTC */
    char sensing_stop[XD_MAX_LENGTH_30]; /* UTC */
    long rel_orbit;
    long abs_orbit;
    double delta_ut1;
    char leap_utc[XD_MAX_STR];
    int leap_sign;
    int leap_err;
    long num_dsd;
    long ds_offset;
    long num_dsr;

    /* extra data: the following values are not read from the file.
     but it must be provided for the writing function */
    /*MPH*/
    char proc_stage_code[XD_MAX_LENGTH_10];
    char ref_doc[XD_MAX_LENGTH_24];
    char proc_time[XD_MAX_LENGTH_31];
    char software_version[XD_MAX_LENGTH_20];
    char phase[XD_MAX_LENGTH_2];
    long cycle;
    char state_vector_time[XD_MAX_LENGTH_31];
    double x_position;
    double y_position;
    double z_position;
    double x_velocity;
    double y_velocity;
    double z_velocity;
    char state_vector_source[XD_MAX_LENGTH_3];
    char product_err[XD_MAX_LENGTH_2];
    long tot_size;
    long num_data_sets;

    /*SPH*/
    char sph_descriptor[XD_MAX_LENGTH_29];
    char sensing_start_tai[XD_MAX_LENGTH_31]; /* TAI */
    long abs_orbit_start;
    double rel_time_asc_node_start;
    char sensing_stop_tai[XD_MAX_LENGTH_31]; /* TAI */
    long abs_orbit_stop;
    double rel_time_asc_node_stop;
    char equator_cross_time[XD_MAX_LENGTH_31];
    long equator_cross_long;
    char ascending_flag[XD_MAX_LENGTH_2];
    long start_lat;
    long start_long;
    long stop_lat;
    long stop_long;
    long num_isps;
    long num_missing_isps;
    long num_error_isps;
    long num_discarded_isps;
    long num_rs_isps;
    long num_rs_corrections;
    long dsr_size;
  } xd_doris_mph_sph;

  /* ---------------------------------------- */
  /* DATA STRUCTURES FOR ORBIT SCENARIO FILES */
  /* ---------------------------------------- */
  typedef struct
  {
    double ref_time;
    double period;
    double sin_amplitude;
    double cos_amplitude;
  } xd_mlst_harmonics;

  typedef struct
  {
    long linear_approx_validity;
    double quadratic_term; /*sec/day^2*/
    long nof_harmonics;
    xd_mlst_harmonics* mlst_harmonics;
  } xd_mlst_nonlinear_drift;

  typedef struct
  {
    long abs_orb; /* Absolute orbit number */
    long rel_orb; /* Relative orbit number */
    long cycle_days;
    long cycle_orbits;

    double mlst; /* Mean local solar time (hours and fractions of hour). */
    double mlst_drift; /* Mean local solar time drift (secs per day). */
    double inclination; /* Inclination. */
    long drift_mode; /* inclination or mlst_drift*/

    double anx_tai; /* ANX time in TAI format (MJD2000) */
    double anx_ut1; /* ANX time in UT1 format (MJD2000) */
    double anx_utc; /* ANX time in UTC format (MJD2000) */

    double anx_long; /* ANX longitude (degrees) */
    long cycle;
    long phase; /* These first variables contain the information of an OSF record. */

    long time_ref_of; /* Reference time to be considered as original */

    xd_mlst_nonlinear_drift mlst_nonlinear_drift;

  } xd_osf_rec;

  typedef struct
  {
    long num_rec; /* Number of elements in osf_rec */
    xd_osf_rec* osf_rec; /* array of state vectors */
  } xd_osf_file;

  /* ----------------------------------------- */
  /* DATA STRUCTURES FOR SWATH DEFINITION FILE */
  /* ----------------------------------------- */

  typedef struct
  {
    long num_terms[XD_MAX_LENGTH_3];
    long harmonic_type_pitch[XD_MAX_NUM_HARMONIC];
    long harmonic_type_roll[XD_MAX_NUM_HARMONIC];
    long harmonic_type_yaw[XD_MAX_NUM_HARMONIC];
    double harmonic_coef_pitch[XD_MAX_NUM_HARMONIC];
    double harmonic_coef_roll[XD_MAX_NUM_HARMONIC];
    double harmonic_coef_yaw[XD_MAX_NUM_HARMONIC];
  } xd_harmonic_data;

  /* Structure containing Instrument View Direction Definition */
  typedef struct
  {
    long geom_type; /* type of geometry (XD_Swath_geom_enum) */
    double az;
    double el;
    double alt;
    double distance;
    double angle; /* AN-665 */
  } xd_swath_geometry; /*  Some of the components of a "Swath_geometry" structure
                        * can remain unfilled, depending on the value of "geom_type" */

  /* Parameter model structure */
  typedef struct
  {
    long model;
    long param_num;
    double model_param[XD_NUM_MODEL_PARAM];
  } xd_param_model_str;

  /* Harmonic model structure */
  typedef struct
  {
    long angle_type;
    xd_harmonic_data harmonics;
    double offsets[XD_MAX_LENGTH_3];
  } xd_harmonic_model_str;

  /* File model structure */
  typedef struct
  {
    long num_files;
    char** files;
    char* aux_file;
    char* time_selection;
    long time_ref;
    double time0; /* time0=time1=0 for selecting the whole file*/
    double time1;
  } xd_file_model_str;

  /* Angle model structure */
  typedef struct
  {
    double angles[XD_MAX_LENGTH_3];
    double offsets[XD_MAX_LENGTH_3];
  } xd_angle_model_str;

  /* Matrix model structure */
  typedef struct
  {
    double att_matrix[XD_MAX_LENGTH_3][XD_MAX_LENGTH_3];
    double offsets[XD_MAX_LENGTH_3];
  } xd_matrix_model_str;

  /* Attitude model union */
  typedef struct
  {
    long attitude_model; /* Attitude model type */
    union
    {
      long AOCS; /* AOCS model */
      xd_param_model_str param_mode; /* Parameters model */
      xd_harmonic_model_str harmonic_mode; /* Harmonic model */
      xd_file_model_str file_mode; /* File model */
      xd_angle_model_str angle_mode; /* Angle Model */
      xd_matrix_model_str matrix_mode; /* Matrix Model */
    } data;
  } xd_attitude_model_str;

  /* ASAR geometry */
  typedef struct
  {
    long asar_type; /* XD_Asar_type_enum */
    double slant_range_left; /* for narrow and wide ASAR */
    double slant_range_right; /* only for wide ASAR */
  } xd_asar_geometry;

  /* Instrument Swath Definition structure */
  typedef struct
  {
    char swath_descr[XD_MAX_STR]; /* Swath description */
    char swath_id[XD_MAX_STR]; /* Swath_id or File_Type */
    long swath_type; /* Swath type: Open/Closed */
    long num_swath_rec; /* Number of swath records to write in a single STF */

    long refr_mode; /* Refraction mode (XD_Target_ray_enum) */
    double freq; /* Frequency (Hz) */

    long num_points; /* Number of points in the instantaneous swath */
    xd_swath_geometry* swath_geom; /* Swath geometry: list of points that define the instantaneous swath */
    xd_asar_geometry asar_geom; /* ASAR geometry: used when the geom_type in swath_geom is ASAR */

    xd_attitude_model_str* sat_nom_att; /* Attitude data for sat. nominal att. */
    xd_attitude_model_str* sat_att; /* Attitude data for sat. attribute */
    xd_attitude_model_str* instr_att; /* Attitude data for instrument att. */
  } xd_sdf_rec;

  /* Swath Definition file structure */
  typedef struct
  {
    long num_rec; /* Number of swath records in a SDF */
    xd_sdf_rec* sdf_rec; /* Swath record data array */
  } xd_sdf_file;

  /* --------------------------------------- */
  /* DATA STRUCTURES FOR SWATH TEMPLATE FILE */
  /* --------------------------------------- */

  /* Swath point definition structure */
  typedef struct
  {
    double lon; /* longitude or RA */
    double lat; /* latitude or Dec */
  } xd_stf_pt;

  /* Swath record definition structure */
  typedef struct
  {
    long num_points; /* number of points in the instantaneous swath */
    xd_stf_pt* stf_pt; /* list of points for the instantaneous swath */
  } xd_stf_rec;

  /* Swath variable header structure */
  typedef struct
  {
    char* stf_name; /* swath template file name */
    char* Reference_OSF; /* Reference OSF file name used to generate the STF */
    char* Reference_SDF; /* Reference swath definition file used to generate the STF*/
    long swath_type; /* Swath type: open/close */
    long swath_point_type; /* Swath point type: geodetic/inertial */

    double time_step;
    long refr_mode; /* Refraction model (XD_Target_ray_enum) */
    double freq; /* Frequency (Hz) */

    long num_points; /* number of points in the instantaneous swath */
    double* altitude; /* list of altitudes */

    long geom_flag; /* true if the geometry of the orbit is defined.
                                     false if the OSV */
    long rep_cycl;
    long cycle_length;
    double mlst_drift;

    long abs_orbit;
    long orbit_start; /* Start orbit of validity range */
    long orbit_stop; /* Stop orbit of validity range */

    double pos[XD_MAX_LENGTH_3];
    double vel[XD_MAX_LENGTH_3];
  } xd_stf_vhr;

  /* xd_stf_data */
  typedef struct
  {
    long num_rec; /* number of points in the swath */
    xd_stf_vhr vhr; /* variable header */
    xd_stf_rec* stf_rec; /* array with the points in the swath */
  } xd_stf_file;

  /* --------------------------------- */
  /* DATA STRUCTURES FOR ATTITUDE FILE */
  /* --------------------------------- */

  /* xd_quat_rec */
  typedef struct
  {
    long time_ref; /* time ref */
    double time; /* time */
    double data[XD_MAX_LENGTH_4]; /* Quaternions or angles:
                            if angles, the fourth value is dummy */
  } xd_att_rec;

  /* xd_att_data */
  typedef struct
  {
    long sat_ref; /* target reference frame */
    long source_ref; /* initial reference frame: Inertial reference frame */
    long data_type; /* data type: quaternions or angles acording to XD_Attitude_data_type_enum */
    long num_rec; /* number of records in the attitude lists */
    double max_gap;
    xd_att_rec* att_rec; /* array with the angle/quaternion records */
  } xd_att_file;

  /* ----------------------------------------------------------- */
  /* DATA STRUCTURES FOR CRYOSAT STAR TRACKER CONFIGURATION FILE */
  /* ----------------------------------------------------------- */

  /* star trackers limits data*/
  typedef struct
  {
    double max_penalty;
    double norm_thr;
    double max_gap;
  } xd_tracker_limits;

  /* star trackers cofiguration file data*/
  typedef struct
  {
    long aberr_correction;
    char satellite[XD_MAX_STR];
    xd_tracker_limits str_limit;
    double str_att_rot[XD_MAX_LENGTH_3][XD_MAX_LENGTH_3];
  } xd_tracker_conf_file;

  /* --------------------------------------------- */
  /* DATA STRUCTURES FOR CRYOSAT STAR TRACKER FILE */
  /* --------------------------------------------- */

  /*  Star tracker quaternion */
  typedef struct
  {
    float quaternion[XD_MAX_LENGTH_4];
    double time; /* in STR_TIME_REF time reference */
    unsigned char status;
  } xd_star_tracker;

  /* xd_stf_data */
  typedef struct
  {
    long str_id; /* Star tracker Id (1,2 or 3) */
    long num_rec; /* number of lines */
    xd_star_tracker* str_rec; /* array with the telemetry data */
  } xd_star_tracker_file;

  /* ------------------------------------------ */
  /* DATA_STRUCTURES FOR DEM CONFIGURATION FILE */
  /* ------------------------------------------ */

  /* xd_dem_ace */
  typedef struct
  {
    char dir[XD_MAX_PATH]; /* Directory for the the DEM files */

    double res_X; /* Interval between points along X-axis */
    double res_Y; /* Interval between points along Y-axis */
    double res_unit; /* Conversion factor from x,y units to the res_X, res_Y units.
                                  i.e. if res_X is given in seconds and X in degrees => res_unit=3600 */

    long X_num_points; /* Number of points along X-axis (columns) */
    long Y_num_points; /* Number of points along Y-axis (lines))*/

    double x_range; /* longitude of the x-axis for one file (grid).*/
    double y_range; /* longitude of the y-axis for one file (grid).*/

    long data_size; /* Size in bytes of the data stored in the files */
    long data_type; /* data type (int, long, float, double) */

    double north_alt[XD_MAX_LENGTH_4]; /* Altitude at the North pole cell */
    double south_alt[XD_MAX_LENGTH_4]; /* Altitude at the South pole cell */

    double offset_x;
    double offset_y; /* For ANR - 187: The points will be given for the center
                                  of a cell instead of the corners. The offset is the difference 
                                  betweeen the center of the cell an the side of the cell (in degrees) */
  } xd_dem_ace; /* DEPRECATED */

  /* Mini tiles configuration (AN-492) */
  typedef struct
  {
    char file_name[XD_MAX_PATH]; /* File name of maximum altitude file */
    double lon_size; /* Longitude size [deg] */
    double lat_size; /* Latitude  size [deg] */
  } xd_dem_mini_tiles;

  /* xd_dem_user_params */
  typedef struct
  {
    char directory[XD_MAX_PATH];
    long cache_type; /* Enum XD_Dem_cache_type_enum (AN-487) */
    long cache_max_size; /* In MB (AN-487)*/
    xd_dem_mini_tiles mini_tiles; /* AN-492 */
    long geoid_computation; /* AN-466 (XD_Dem_geoid_flag_enum) */
    long geoid_num_harmonics; /* AN-466 */
  } xd_dem_user_params;

  /* xd_dem_metadata */
  typedef struct
  {
    long model;
    long n_rows; /*ANR-778: number of rows */
    long n_cols; /*ANR-778: number of columns */
    XD_Dem_cell_location_enum cell_location; /*ANR-778: cell location */
  } xd_dem_metadata;

  /* xd_dem_raster */
  typedef struct
  {
    long model; /* DEM model */
    long data_type; /* Data type (int, long, float, double)        */
    long data_unit; /* Data unit (meter, kilometer)                */
    long rows; /* Number of points along Y-axis (lines)       */
    long columns; /* Number of points along X-axis (columns)     */
    long data_resolution; /* Resolution                                  */
    long data_res_unit; /* Resolution type (deg, min, sec)             */
    long data_reference; /* Data reference type (WGS84, EGM96)          */
    long void_value; /* Value that identifies a void value (-32768) */
    double maximum_height_value; /* Maximum height value in the DEM (optional, by default 9000. m) */
    long flag_type; /* Flag type (int8, int16, int32, int64)       */

  } xd_dem_raster; /* ANR-740 */

  /* xd_dem_config_file */
  typedef struct
  {
    long model;
    xd_dem_ace* dem_data; /* DEPRECATED */
    xd_dem_user_params dem_user_params;
    xd_dem_metadata dem_metadata;
    xd_dem_raster dem_raster; /* ANR-740 */
  } xd_dem_config_file;

  /* ----------------------------- */
  /* DATA_STRUCTURES FOR DEM FILES */
  /* ----------------------------- */

  /* xd_dem_point */
  typedef struct
  {
    double lon;
    double lat;
    double alt;
  } xd_dem_point;

  /* xd_dem_file */
  typedef struct
  {
    long num_points_X;
    long num_points_Y;
    xd_dem_point** point;
  } xd_dem_file;

  /* --------------------------------------- */
  /* DATA STRUCTURES FOR SWATH CONTROL FILES */
  /* --------------------------------------- */

  /* SCF appearance data structure */
  typedef struct
  {
    long colour;
    long draw;
    long fill;
    long opacity;
  } xd_scf_appear;

  /* --------------------------------------- */
  /* DATA STRUCTURES FOR STAR DATABASE FILES */
  /* --------------------------------------- */

  /* xd_star_rec */
  typedef struct
  {
    long flag;
    char star_id[XD_MAX_STAR_ID_SIZE];
    double par; /* Parallax of the star at JD2000 (rads) */
    double mu_ra; /* RA's proper motion at JD2000 (rad/century) */
    double mu_dec; /* DEC's proper motion at JD2000 (rad/century) */
    double rad_vel; /* Radial velocity of the star (au/century) */
    double star_ra; /* RA of the star at JD2000 (rads) */
    double star_dec; /* DEC of the star at JD2000 (rads) */
  } xd_star_rec; /* structure containing all relevant information contained in the star's database file*/

  /* xd_star_file */
  typedef struct
  {
    long num_rec;
    xd_star_rec* star_rec;
  } xd_star_file; /* star database file info */

  /* ------------------------------------------ */
  /* DATA STRUCTURES FOR STATION DATABASE FILES */
  /* ------------------------------------------ */

  /* xd_station_rec */
  typedef struct
  {
    char station_id[XD_MAX_STR];
    char descriptor[XD_MAX_STR];
    char antenna[XD_MAX_STR];
    char purpose[XD_MAX_STR];
    char type[XD_MAX_STR];
    long num_mask_pt;
    double azimuth[XD_VERTICES];
    double elevation[XD_VERTICES];
    double station_long;
    double station_lat;
    double station_alt;
    double proj_long[XD_VERTICES];
    double proj_lat[XD_VERTICES];
    long points;
    double long_max; /* Maximum longitude of the station zone */
    double lat_max; /* Maximum latitude of the station zone */
    double long_min; /* Minimum longitude of the station zone */
    double lat_min; /* Minimum latitude of the station zone */
    long mission_list;
    char mission_name[XD_MISSIONS][XD_MAX_STR];
    double mis_aos_el[XD_MISSIONS];
    double mis_los_el[XD_MISSIONS];
    char mask_type[XD_MISSIONS][XD_MAX_STR];
  } xd_station_rec;

  typedef struct
  {
    long num_rec;
    xd_station_rec* station_rec;
  } xd_station_file;

  /* --------------------------------------- */
  /* DATA STRUCTURES FOR ZONE DATABASE FILES */
  /* --------------------------------------- */

  /* Information for each zone */
  typedef struct
  {
    double pt_long;
    double pt_lat;
  } xd_zone_point;

  /* Zone information structure: General information of the ZONE */
  typedef struct
  {
    char zone_id[XD_MAX_STR];
    char description[XD_MAX_STR];
    char surface[XD_MAX_STR];
    char creator[XD_MAX_STR];

    XD_Zone_type_enum zone_type;
    long projection;
    double zone_diam; /* Zone diameter in meters. This value is only used when
                        the ZONE is a POINT zone (with the POINT swath) or a
                        CIRCULAR zone. */

    /* list of points in the zone */
    long num_points; /* Number of ZONE points (last one, equal to the first one, included) */
    xd_zone_point* zone_point; /* Information related with each POINT record */
  } xd_zone_rec;

  /* xd_zone_file */
  typedef struct
  {
    long num_rec;
    xd_zone_rec* zone_rec;

  } xd_zone_file; /* zone database file info */

  /* ----------------------------- */
  /* DATA STRUCTURES FOR TLE FILES */
  /* ----------------------------- */

  /* TLE data (data from the 2 lines) */
  typedef struct
  {
    /* line 0 */
    char norad_satcat[XD_MAX_LENGTH_25]; /* Satellite name  consistent with the NORAD SATCAT */

    /* line 1 */
    long sat_number; /* NORAD Catalogue number */
    char classification; /* Classification: U=unclassified, S=secret data */
    char int_des[XD_MAX_LENGTH_9]; /* International Designator: (Last two digits of launch year)
                                                        (Launch number of the year)
                                                        (Piece of the launch) */
    double time; /* reference time for the element set (UTC processing days MJ2000)*/
    double n_1st; /* First Time Derivative of the Mean Motion */
    double n_2nd; /* Second Time Derivative of Mean Motion */
    double bstar; /* BSTAR drag term  */
    int ephemeris_type; /* Ephemeris type */
    int index; /* Element number */
    int checksum1; /* Checksum (Modulo 10)
                              (Letters, blanks, periods, plus signs = 0
                              minus signs = 1) */
    /* Line 2 */
    double i; /* Inclination [Degrees] */
    double ra; /* Right Ascension of the Ascending Node [Degrees] */
    double e; /* Eccentricity */
    double w; /* Argument of Perigee [Degrees] */
    double m; /* Mean Anomaly [Degrees] */
    double n; /* Mean Motion [Revs per day] */
    long abs_orbit; /* Revolution number at epoch [Revs] */
    int checksum2; /* Checksum (Modulo 10)
                              (Letters, blanks, periods, plus signs = 0
                              minus signs = 1) */
  } xd_tle_rec;

  /* TLE file data */
  typedef struct
  {
    long num_rec;
    xd_tle_rec* tle_rec;
  } xd_tle_file;

  /* ---------------------------------------- */
  /* DATA STRUCTURES FOR PRECISE PROPAG FILES */
  /* ---------------------------------------- */
  typedef struct
  {
    long user_flag; /* Use default or user values (XO_Precise_propag_user_flag_enum) */
    char models_path[XD_MAX_PATH]; /* Path where files necessary for models are looked for. */

    /* Flags */
    long gravity_flag; /* Gravity perturbation used (XD_SELECT) or not (XD_NOT_SELECT) */
    long thirdbody_flag; /* Third bodies perturbation used (XD_SELECT) or not (XD_NOT_SELECT) */
    long atmos_flag; /* Atmosphere perturbation used (XD_SELECT) or not (XD_NOT_SELECT) */
    long srp_flag; /* Solar radiation pressure perturbation used (XD_SELECT) or not (XD_NOT_SELECT) */

    /* Simulation step [s] */
    double step;

    /* Gravity model */
    char grav_file[XD_MAX_PATH]; /* File with data of gravitational model */
    long grav_degree; /* Degree used gravity model */
    long grav_order; /* Order used in gravity model */

    /* Solar Geomagnetic Activity */
    long sga_flag; /* Parameters used (XD_SGA_USE_PARAMETERS) or data read from file (XD_SGA_READ_VALUES_FROM_FILE) */
    char sga_ap_file[XD_MAX_PATH]; /* File with Geomagnetic Activity index values */
    char sga_f107_file[XD_MAX_PATH]; /* File with F10.7 Solar Activity index values */
    double ap; /* Geomagnetic Activity Index (daily value) */
    double f107; /* F10.7 Index Solar Activity Index (daily value) */
    double f107a; /* F10.7 Index Solar Activity Index (value averaged over 3 months) */

    /* S/C characteristics */
    double sc_mass; /* S/C mass [kg] */
    double sc_drag_area; /* S/C effective drag area [m^2] */
    double sc_drag_coeff; /* S/C drag coefficient [adim] */
    double sc_srp_area; /* S/C effective SRP area [m^2] */
    double sc_srp_coeff; /* S/C SRP coefficient [adim] */

  } xd_propag_precise_config;

  /* ------------------------------------- */
  /* DATA STRUCTURES FOR SP3 FILE (AN-550) */
  /* ------------------------------------- */
  typedef struct
  {
    long type; // XD_SAT_SP3_sat_descriptor_enum
    long identifier;
    char id_string[SIZE_SP3_ID + 1];
    long sat_accuracy;
    long num_rec;
    xd_osv_rec* osv_rec;
  } xd_osv_rec_sp3;

  typedef struct
  {
    // Line 1
    long type; // XD_SP3_file_type_enum (position of position+velocity)
    double global_time_start; // Gegorian initial time for all the file (MJD2000).
    char data_used[XD_MAX_LENGTH_6];
    char coordinate_system[XD_MAX_LENGTH_6];
    char orbit_type[XD_MAX_LENGTH_4];
    char agency[XD_MAX_LENGTH_5];

    // Line 2
    long gps_week;
    double seconds_of_week;
    double epoch_interval_seconds;
    double julian_date_start;
    double fractional_day;

    // Lines 3-7
    long num_sat; // Number of satellites

    // Line 13
    long file_type_descriptor; // XD_SP3_file_descriptor_enum
    long time_system_indicator; // XD_SP3_time_system_enum

    // Line 15
    double pos_vel_std_dev; // base number used for computing the standard deviations for the components of the satellite position and velocity (units: mm and 10**-4 mm/sec)
    double clock_std_dev; // base number for computing the standard deviations for the clock correction and the rate-of-change of the clock correction (units: picosec and 10**-4 picosec/sec)

    // Lines 19-22
    char* comments[XD_MAX_LENGTH_4]; // Comments

    double delta_tai_gps; // Delta TAI-GPS [seconds]

    xd_osv_rec_sp3* osv_rec_sp3; // Every array position corresponds to a different satellite

  } xd_sp3_file;

  /* AN-578 */
  typedef struct
  {
    // OEM Header
    char ccsds_oem_vers[XD_MAX_LENGTH_5];
    char comment_header[XD_MAX_STR];
    char creation_date[XD_MAX_LENGTH_25];
    char originator[XD_MAX_STR];

    // OEM METADATA
    char comment_metadata[XD_MAX_STR];
    char object_name[XD_MAX_STR];
    char object_id[XD_MAX_STR];
    char center_name[XD_MAX_STR];
    char ref_frame[XD_MAX_STR];
    char ref_frame_epoch[XD_MAX_LENGTH_25];
    char time_system[XD_MAX_LENGTH_4];
    char start_time[XD_MAX_LENGTH_25];
    char useable_start_time[XD_MAX_LENGTH_25];
    char useable_stop_time[XD_MAX_LENGTH_25];
    char stop_time[XD_MAX_LENGTH_25];
    char interpolation[XD_MAX_STR];
    char interpolation_degree[XD_MAX_LENGTH_2];

    //EPHEMERIS DATA LINES
    xd_osv_rec* osv_rec; /* array of state vectors */
    long num_rec;
  } xd_oem_file;

  /* AN-856 */
  typedef struct
  {
    // AEM METADATA
    char comment[XD_MAX_STR];
    char object_name[XD_MAX_STR];
    char object_id[XD_MAX_STR];
    char center_name[XD_MAX_STR];
    char ref_frame_A[XD_MAX_STR];
    char ref_frame_B[XD_MAX_STR];
    char attitude_dir[XD_MAX_LENGTH_4];
    char time_system[XD_MAX_LENGTH_4];
    char start_time[XD_MAX_LENGTH_25];
    char useable_start_time[XD_MAX_LENGTH_25];
    char useable_stop_time[XD_MAX_LENGTH_25];
    char stop_time[XD_MAX_LENGTH_25];
    char attitude_type[XD_MAX_STR];
    char quaternion_type[XD_MAX_STR];
    char euler_rot_seq[XD_MAX_LENGTH_4];
    char rate_frame[XD_MAX_LENGTH_15];
    char interpolation[XD_MAX_STR];
    char interpolation_degree[XD_MAX_LENGTH_2];
  } xd_aem_metadata;

  typedef struct
  {
    long data_type; // data type XD_Attitude_data_type_enum
    long time_ref;
    double time;
    double data[XD_MAX_LENGTH_8];
  } xd_aem_att_rec;

  typedef struct
  {
    //EPHEMERIS DATA LINES
    xd_aem_att_rec* att_rec; /* array of state vectors */
    long num_rec;
  } xd_aem_attitude;

  typedef struct
  {
    xd_aem_metadata metadata;
    xd_aem_attitude attitude;
  } xd_aem_segment;

  typedef struct
  {
    // AEM Header
    char ccsds_aem_vers[XD_MAX_LENGTH_5];
    char comment_header[XD_MAX_STR];
    char creation_date[XD_MAX_LENGTH_25];
    char originator[XD_MAX_STR];

    xd_aem_segment* segment;
    long num_segment;
  } xd_aem_file;

  /* -------------------------------------------------------------------- */
  /* DATA STRUCTURES FOR TIME AND ORBIT INITIALIZATION (added for AN-404) */
  /* -------------------------------------------------------------------- */

  typedef union
  {
    xd_orbit_file orbit_file;
    xd_osf_file osf_file;
    xd_doris_file doris_file;
    xd_iers_bulletin_file bulletin_file;
    xd_sp3_file sp3_file; // AN-550
    xd_oem_file oem_file; // AN-578
    xd_tle_file tle_file; // AN-583
  } xd_eocfi_file_union;

  typedef struct
  {
    long file_type; /* One from XD_data_file_type_enum */
    xd_eocfi_file_union eocfi_file;
  } xd_eocfi_file;

  typedef struct
  {
    long num_files;
    xd_eocfi_file* eocfi_file_array;
  } xd_eocfi_file_set;

  /*ANR-469*/
  /* -------------------------------------------- */
  /* DATA STRUCTURES FOR ATTITUDE DEFINITION FILE */
  /* -------------------------------------------- */

  typedef struct
  {
    char* quat_def_file; /* ANR-651: Quaternion File parameters */
    xd_angle_model_str angle_model; /* Angles value */
  } xd_quaternion_plus_angle;

  typedef struct
  {
    char* quat_def_file; /* ANR-651: Quaternion File parameters */
    xd_matrix_model_str matrix_model; /* Rotation matrix */
  } xd_quaternion_plus_matrix;

  /* Attitude model union */
  typedef struct
  {
    long attitude_model; /* Attitude model type */
    union
    {
      long AOCS; /* AOCS model */
      xd_param_model_str param_mode; /* Parameters model */
      xd_harmonic_model_str harmonic_mode; /* Harmonic model */
      xd_file_model_str file_mode; /* File model */
      xd_angle_model_str angle_mode; /* Angle Model */
      xd_matrix_model_str matrix_mode; /* Matrix Model */
      xd_quaternion_plus_angle quaternion_angle_mode; /*Quaternion plus angel mode*/
      xd_quaternion_plus_matrix quaternion_matrix_mode; /*Quaternion plus matrix mode*/
    } data;
  } xd_attitude_definition_model_str;

  /* Instrument Swath Definition structure */
  typedef struct
  {
    char* att_def_file_dir_path; /* Directory where the Atittude DEF read is placed */
    xd_attitude_definition_model_str* sat_nom_att; /* Attitude data for sat. nominal att. */
    xd_attitude_definition_model_str* sat_att; /* Attitude data for sat. attribute */
    xd_attitude_definition_model_str* instr_att; /* Attitude data for instrument att. */
  } xd_attitude_definition_data;

  /* Diagnostics configuration structure */
  typedef struct
  {
    double gap_threshold; // time to identify a gap [s]
    double duplicated_osv_threshold; // time to identify a duplicated OSV [s]
    double time_step; // expected time step [s]
    double time_step_threshold; // time step threshold, to identify non-equally spaced OSVs [s]
    long time_ref; // time system that will be used to fill time related fields in the report structure
  } xd_orbit_file_diagnostics_settings; //AN-576

  /* Diagnostics report structure for orbit files, OEM files, doris files*/
  typedef struct
  {
    long osv_list_id; //identifier associated to the OSV list (applicable for file storing multiple OSV lists, e.g. SP3)
    long num_osv; //number of OSVs which were checked
    double total_time; //total time covered by the file (i.e. from first to last OSV)
    double time_first_osv; //time of first OSV
    double time_last_osv; //time of last OSV
    long time_ref; //time system of time related fields in this structure
    double* time_start_gap; //list containing start time of GAPs
    double* time_stop_gap; //list containing stop time of GAPs
    long* index_gap; //list containing index of GAPs (the index represents the ID of OSV which is preceded by a GAP)
    long num_gaps; //number of identified GAPs
    double* time_going_back_osv; //list containing time of going back OSVs
    long* index_going_back_osv; //list containing index of going back OSVs
    long num_going_back_osv; //number of identified going back OSVs
    double* time_duplicated_osv; //list containing time of duplicated OSVs
    long* index_duplicated_osv; //list containing index of duplicated OSVs
    long num_duplicated_osv; //number of identified duplicated OSVs
    double* time_inconsistent_orbit_number; //list containing time of OSVs with inconsistent orbit number
    long* index_inconsistent_orbit_number; //list containing index of OSVs with inconsistent orbit number
    long num_inconsistent_orbit_number; //number of OSVs with inconsistent orbit number
    double* time_non_equally_spaced_osv; //list containing time of non equally spaced OSVs
    long* index_non_equally_spaced_osv; //list containing index of non equally spaced OSVs
    long num_non_equally_spaced_osv; //number of OSVs with time step different from expected (absolute value of difference from step and expected > threshold)
  } xd_orbit_file_diagnostics_report_single; //AN-576

  /* Diagnostics report structure */
  typedef struct
  {
    long file_type; //orbit file type: XD_POF_TYPE, XD_ROF_TYPE, XD_DORIS_TYPE, XD_OEM_TYPE, XD_SP3_TYPE
    long osv_list_num; //number of OSV lists processed
    xd_orbit_file_diagnostics_report_single* diagnostics_reports;
  } xd_orbit_file_diagnostics_report; //AN-576

  // AN-634: configuration for reading OSV state vectors
  typedef struct
  {
    long time_mode; // XD_Time_init_mode_enum
    long time_ref; // XD_Time_ref_enum
    long extend_type; // XD_Extend_type_enum
    double time_start, time_stop; // Initialization time interval (only applicable if time_mode == XD_SEL_TIME)
    long orbit_start, orbit_stop; // Initialization ORBIT interval (only applicable if time_mode == XD_SEL_ORBIT)
    long extend_num_osv; // Number of OSVs to be added to initialization interval (only applicable if extend_type == XD_EXTEND_NUM_OSV)
    double extend_osv_sec; // Size of interval whose OSVs must be added before/after input interval (only applicable if extend_type == XD_EXTEND_TIME)
  } xd_osv_list_read_configuration;

  /* AN-628 */

  typedef struct
  {
    long num_mask_pt;
    long status; /*AN-533 ENABLED, DISABLED */
    double azimuth[XD_VERTICES];
    double elevation[XD_VERTICES];
  } xd_az_el_mask;

  typedef struct
  {
    xd_az_el_mask incl_mask; /* Inclusive mask */
    xd_az_el_mask excl_mask; /* Exclusive mask */
  } xd_link_mask;

  typedef struct
  {
    xd_link_mask mask_data; /* Attitude frame mask */
    double min_tg_height; /* Minimum tangent height */
  } xd_link_data;

  typedef union
  {
    xd_link_data sc_link_data;
    xd_link_data celestial_body_link_data; //ANR-677
  } xd_fov_constraints_union;

  typedef struct
  {
    long type; // XD_Fov_constraints_enum : sun, moon, sc
    xd_fov_constraints_union constraints;
  } xd_fov_constraints_file;

  /* END AN-628 */

  /* ----------------------------------------- */
  /* PROTOTYPE DECLARATION (Main CFI Functions)*/
  /* ----------------------------------------- */

  /***** Error Handling Functions *****/

  long xd_silent(void);

  long xd_verbose(void);

  long xd_print_msg(long* n, char msg[XD_MAX_COD][XD_MAX_STR]);

  long xd_get_msg(long* func_id, long* ierr,
                  /* output */
                  long* n,
                  char msg[XD_MAX_COD][XD_MAX_STR]);

  long xd_get_code(long* func_id, long* ierr,
                   /* output */
                   long* n,
                   long vec[XD_MAX_COD]);

  /***** Set parameters *****/

  /***** Main functions *****/

  long xd_read_fhr(char* file,
                   /* output */
                   xd_fhr* fhr,
                   long ierr[XD_NUM_ERR_READ_FHR]);

  long xd_read_bulletin(char* bulb_file,
                        /* output */
                        xd_iers_bulletin_b* iers_data,
                        long ierr[XD_NUM_ERR_READ_BULLETIN]);

  long xd_read_bulletin_2(char* bulletin_file_name,
                          /* output */
                          xd_iers_bulletin_file* iers_data,
                          long ierr[XD_NUM_ERR_READ_BULLETIN]);

  void xd_free_bulletin(xd_iers_bulletin_file* iers_data);

  long xd_read_orbit_file(char* file_name, long* read_fro_flag, long* time_orbit_flag, long* time_ref, double* start_range, double* stop_range, long* reading_osv_flag,
                          /* output */
                          xd_orbit_file* orbit_data,
                          long ierr[XD_NUM_ERR_READ_ORBIT_FILE]);

  void xd_free_orbit_file(xd_orbit_file* orbit_data);

  long xd_orbit_file_decimate(xd_fhr* fhr_in,
                              xd_orbit_file* osv_in,
                              double decimation_delta_time,
                              /* output */
                              xd_fhr* fhr_out,
                              xd_orbit_file* osv_out,
                              long ierr[XD_NUM_ERR_ORBIT_FILE_DECIMATE]);

  long xd_read_doris(char* doris_file, long* time_mode, double* time0, double* time1, long* interpol_flag,
                     /* output */
                     xd_doris_file* doris_data,
                     long ierr[XD_NUM_ERR_READ_DORIS]);

  void xd_free_doris(xd_doris_file* doris_data);

  long xd_read_doris_header(char* doris_file,
                            /* output */
                            xd_doris_mph_sph* header,
                            long ierr[XD_NUM_ERR_READ_DORIS_HEADER]);

  long xd_read_osf(char* file_name,
                   /* output */
                   xd_osf_file* osf_data,
                   long ierr[XD_NUM_ERR_READ_OSF]);

  void xd_free_osf(xd_osf_file* osf_data);

  long xd_read_sdf(char* file_name,
                   /* output */
                   xd_sdf_file* sdf,
                   long ierr[XD_NUM_ERR_READ_SDF]);

  void xd_free_sdf(xd_sdf_file* sdf);

  long xd_read_stf(char* file_name,
                   /* output */
                   xd_stf_file* stf_data,
                   long ierr[]);

  long xd_read_stf_vhr(char* file_name,
                       /* output */
                       xd_stf_vhr* stf_vhr,
                       long ierr[]);

  void xd_free_stf(xd_stf_file* stf_data);
  void xd_free_stf_vhr(xd_stf_vhr* stf_vhr);

  long xd_read_att(char* file_name,
                   /* output */
                   xd_att_file* att_data,
                   long ierr[XD_NUM_ERR_READ_ATT]);

  void xd_free_att(xd_att_file* att_data);

  long xd_attitude_file_decimate(xd_fhr* fhr_in,
                                 xd_att_file* att_in,
                                 double decimation_delta_time,
                                 /* output */
                                 xd_fhr* fhr_out,
                                 xd_att_file* att_out,
                                 long ierr[XD_NUM_ERR_ATTITUDE_FILE_DECIMATE]);

  long xd_read_star_tracker(long* n_files, char** file_list, long* time_init_mode, double* time0, double* time1, xd_tracker_limits* str_limit,
                            /* output */
                            xd_star_tracker_file* str_data,
                            long ierr[XD_NUM_ERR_READ_STAR_TRACKER]);

  void xd_free_star_tracker(xd_star_tracker_file* str_data);

  long xd_read_star_tracker_conf_file(char* file, long* star_tracker_id,
                                      /* output */
                                      xd_tracker_conf_file* conf_data,
                                      long ierr[XD_NUM_ERR_READ_STR_CONF_FILE]);

  long xd_read_dem_config_file(char* dem_file,
                               /* output */
                               xd_dem_config_file* dem_config_data,
                               long ierr[XD_NUM_ERR_READ_DEM_CONFIG_FILE]);

  void xd_free_dem_config_file(xd_dem_config_file* dem_config_data);

  long xd_read_dem(char* dem_file, xd_dem_config_file* config_file,
                   /* output */
                   xd_dem_file* dem_data,
                   long ierr[XD_NUM_ERR_READ_DEM]);

  void xd_free_dem(xd_dem_file* dem_data);

  long xd_read_star(char* star_db_file,
                    char* star_id,
                    /* output */
                    xd_star_rec* star_data,
                    long ierr[XD_NUM_ERR_READ_STAR]);

  long xd_read_star_file(char* star_db_file,
                         /* output */
                         xd_star_file* star_file,
                         long ierr[XD_NUM_ERR_READ_STAR_FILE]);

  void xd_free_star_file(xd_star_file* star_file);

  long xd_read_star_id(char* star_db_file,
                       /* output */
                       long* num_stars,
                       char*** star_list,
                       long ierr[XD_NUM_ERR_READ_STAR_ID]);

  void xd_free_star_id(long* num_stars, char*** star_list);

  long xd_read_station(char* station_file,
                       char* station_id,
                       /* output */
                       xd_station_rec* sta_rec,
                       long ierr[XD_NUM_ERR_READ_STATION]);

  long xd_read_station_file(char* station_file,
                            /* output */
                            xd_station_file* sta_data,
                            long ierr[XD_NUM_ERR_READ_STATION_FILE]);

  void xd_free_station_file(xd_station_file* sta_data);

  long xd_read_station_id(char* station_file,
                          /* output */
                          long* num_stations,
                          char*** sta_list,
                          long ierr[XD_NUM_ERR_READ_STATION_ID]);

  void xd_free_station_id(char*** sta_ids);

  long xd_read_zone(char* zone_file, char* zone_id,
                    /* output */
                    xd_zone_rec* zone_data,
                    long ierr[XD_NUM_ERR_READ_ZONE]);

  void xd_free_zone(xd_zone_rec* zone_data);

  long xd_read_zone_file(char* zone_file,
                         /* output */
                         xd_zone_file* zone_data,
                         long ierr[XD_NUM_ERR_READ_ZONE]);

  void xd_free_zone_file(xd_zone_file* zone_data);

  long xd_read_zone_id(char* zone_file,
                       /* output */
                       long* num_zones,
                       char*** zone_ids,
                       long ierr[XD_NUM_ERR_READ_ZONE_ID]);

  void xd_free_zone_id(char*** zone_ids);

  long xd_read_att_def(char* file_name, /* ANR-469*/
                       /* output */
                       xd_attitude_definition_data* att,
                       long ierr[XD_NUM_ERR_READ_ATT_DEF]);
  void xd_free_att_def(xd_attitude_definition_data* att); /* ANR-469*/

  long xd_write_orbit_file(char* filename,
                           xd_fhr* fhr,
                           xd_orbit_file* osv,
                           /* output */
                           long ierr[XD_NUM_ERR_WRITE_ORBIT_FILE]);

  long xd_write_doris(char* filename,
                      xd_doris_mph_sph* fhr,
                      xd_doris_file* doris_data,
                      /* output */
                      long ierr[XD_NUM_ERR_WRITE_DORIS]);

  long xd_write_osf(char* filename,
                    xd_fhr* fhr,
                    xd_osf_file* osf_data,
                    /* output */
                    long ierr[XD_NUM_ERR_WRITE_OSF]);

  long xd_write_stf(char* swath_filename, xd_fhr* fhr, xd_stf_file* stf,
                    /* output */
                    long ierr[]);

  long xd_write_att(char* filename,
                    xd_fhr* fhr,
                    xd_att_file* att_data,
                    /* output */
                    long ierr[XD_NUM_ERR_WRITE_ATT]);

  long xd_write_att_def(char* filename, /* ANR-469*/
                        xd_fhr* fhr,
                        xd_attitude_definition_data* att_data,
                        long ierr[XD_NUM_ERR_WRITE_ATT_DEF]); /* AN-469 */

  long xd_xml_validate(char* filename, long* mode, char* schema, char* logfile,
                       /* output */
                       long* valid_status,
                       long ierr[XD_NUM_ERR_XML_VALIDATE]);

  long xd_read_tle(char* tle_file, char* satellite, xd_tle_file* tle_data, long ierr[XD_NUM_ERR_READ_TLE]);

  long xd_write_tle(char* filename,
                    xd_tle_file* tle_data,
                    long ierr[XD_NUM_ERR_WRITE_TLE]);

  void xd_free_tle(xd_tle_file* tle_data);

  /*ANR-628*/
  long xd_read_fov_constraints_file(char* filename,
                                    /*output*/
                                    xd_fov_constraints_file* fov_contraints,
                                    long ierr[XD_NUM_ERR_READ_FOV]);

  long xd_read_precise_propag_file(char* file_name,
                                   xd_propag_precise_config* precise_conf,
                                   long ierr[XD_NUM_ERR_READ_PRECISE_PROPAG]);

  long xd_select_schema(xd_fileinfo* info,
                        char* schema,
                        long ierr[XD_NUM_ERR_SELECT_SCHEMA]);

  long XD_select_schema_by_satellite(xd_fileinfo* info,
                                     char* schema,
                                     long ierr[XD_NUM_ERR_SELECT_SCHEMA]);
  long XD_select_schema_by_eoffs(xd_fileinfo* info,
                                 char* schema,
                                 long ierr[XD_NUM_ERR_SELECT_SCHEMA]);

  long xd_check_library_version();

  long xd_read_sp3(char* file_name,
                   xd_osv_list_read_configuration* osv_list_read_configuration, // AN-634
                   /*output*/
                   xd_sp3_file* sp3_file,
                   long ierr[XD_NUM_ERR_READ_SP3]); // AN-550

  void xd_free_sp3(xd_sp3_file* sp3_file); // AN-550

  long xd_xslt_add(char* fname_in,
                   long ierr[XD_NUM_ERR_XSLT_ADD]); /* AN-546 */

  long xd_read_oem(char* file_name,
                   xd_osv_list_read_configuration* osv_list_read_configuration, // AN-634
                   /*output*/
                   xd_oem_file* oem_file,
                   long ierr[XD_NUM_ERR_READ_OEM]); /* AN-578 */
  void xd_free_oem(xd_oem_file* oef_file); /* AN-578 */

  long xd_orbit_file_diagnostics(char* orbit_file,
                                 xd_eocfi_file* eocfi_file,
                                 xd_orbit_file_diagnostics_settings* diagnostics_settings,
                                 // output
                                 xd_orbit_file_diagnostics_report* diagnostics_report,
                                 long ierr[XD_NUM_ERR_ORBIT_FILE_DIAGNOSTICS]); //AN-576
  void xd_free_orbit_file_diagnostics_report(xd_orbit_file_diagnostics_report* diagnostics_report); //AN-576

  /* ANR-638*/
  long xd_set_file_format_standard_version(long eoffs, long ierr[XD_NUM_ERR_SET_FILE_FORMAT_STANDARD_VERSION]);

  void xd_free_eocfi_file(xd_eocfi_file* eocfi_file); // AN-713

  long xd_read_aem(char* file_name,
                   /*output*/
                   xd_aem_file* aem_file,
                   long ierr[XD_NUM_ERR_READ_AEM]);

  void xd_free_aem(xd_aem_file* aem_file);

  /*ANR-638*/
  typedef enum
  {
    XD_FFS_DEFAULT = 0,
    XD_FFS_V1,
    XD_FFS_V2,
    XD_FFS_V3,
    XD_FFS_MAX,
    XD_FFS_ERROR
  } XD_EOFFS_version;

  static const char file_format_standard_version[sizeof(XD_EOFFS_version) + 2][XD_MAX_LENGTH_250] = {"DEFAULT", "1.0", "2.0", "3.0", "MAX", "ERROR"};
  extern XD_EOFFS_version eoffs_version;

/* Alias for xx_check_library_version */
#if defined(expcfi_check_libs)

  #if !defined(xl_check_library_version) && !defined(xo_check_library_version) && !defined(xp_check_library_version) && !defined(xv_check_library_version)

    #undef expcfi_check_libs
    #define expcfi_check_libs xd_check_library_version

  #endif

#else

  #define expcfi_check_libs xd_check_library_version

#endif

#ifdef __cplusplus
}
#endif

#endif
