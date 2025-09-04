/******************************************************************************
 *
 * Project     : EOCFI
 * Copyright   : DEIMOS Space S.L.U.
 *
 * Module      : explorer_lib.h
 *
 * Purpose     : It is the header file of the EXPLORER_LIB CFI to be used by the
 *               final user
 *
 * Reference   : Based on Envisat Mission CFI Software
 *
 * History:      +----------------------------------------------------------------------------------+
 *               | Version |   Date   |       Name        | Change                                  |
 *               |----------------------------------------------------------------------------------|
 *               |    1.2  | 15/04/02 | DEIMOS Space S.L. | First version from                      |
 *               |         |          |                   | Envisat CFI                             |
 *               |----------------------------------------------------------------------------------|
 *               |    1.3  | 19/07/02 | DEIMOS Space S.L. | Maintenance release                     |
 *               +----------------------------------------------------------------------------------+
 *               |    2.0  | 29/11/02 | DEIMOS Space S.L. | Maintenance release                     |
 *               |         |          |                   | Leap Second Function                    |
 *               |         |          |                   | BOM, EOM ASCII times                    |
 *               |         |          |                   | CCSDS compact ASCII                     |
 *               +----------------------------------------------------------------------------------+
 *               |    2.0  | 10/12/02 | DEIMOS Space S.L. | Patch to solve SPRs:                    |
 *               | patch A |          |                   | - EXPCFI-SPR-007                        |
 *               |         |          |                   | - EXPCFI-SPR-008                        |
 *               |         |          |                   | - EXPCFI-SPR-009                        |
 *               |         |          |                   | - EXPCFI-SPR-010                        |
 *               +----------------------------------------------------------------------------------+
 *               |    2.0  | 16/12/02 | DEIMOS Space S.L. | Patch to solve SPRs:                    |
 *               | patch B |          |                   | - EXPCFI-SPR-014                        |
 *               |         |          |                   | - EXPCFI-SPR-015                        |
 *               |         |          |                   | - EXPCFI-SPR-016                        |
 *               |         |          |                   | - EXPCFI-SPR-017                        |
 *               +----------------------------------------------------------------------------------+
 *               |    2.1  | 13/05/03 | DEIMOS Space S.L. | Maintenance release                     |
 *               |         |          |                   | MAC OS Version                          |
 *               +----------------------------------------------------------------------------------+
 *               |    2.2  | 13/08/03 | DEIMOS Space S.L. | Pre-release                             |
 *               +----------------------------------------------------------------------------------+
 *               |    2.2  | 30/09/03 | DEIMOS Space S.L. | Maintenance release                     |
 *               |         |          |                   | New functions added:                    |
 *               |         |          |                   |  + xl_default_sat_init                  |
 *               +----------------------------------------------------------------------------------+
 *               |   2.2.2 | 26/04/04 | DEIMOS Space S.L. | Maintenance release                     |
 *               |         |          |                   | Added TERRASAR S/C                      |
 *               +----------------------------------------------------------------------------------+
 *               |   3.0   | 21/07/04 | DEIMOS Space S.L. | New initialization                      |
 *               |         |          |                   | strategy  (New IF)                      |
 *               +----------------------------------------------------------------------------------+
 *               |   3.1   | 13/10/04 | DEIMOS Space S.L. | Maintenance release                     |
 *               |         |          |                   | New functions:                          |
 *               |         |          |                   | + xl_get_rotation_angles                |
 *               |         |          |                   | + xl_get_rotated_vectors                |
 *               |         |          |                   | + xl_position_on_orbit                  |
 *               +----------------------------------------------------------------------------------+
 *               |   3.2   | 15/11/04 | DEIMOS Space S.L. | Maintenance release                     |
 *               +----------------------------------------------------------------------------------+
 *               |    3.3  | 11/07/05 | DEIMOS Space S.L. | Maintenance release                     |
 *               |         |          |                   | New features:                           |
 *               |         |          |                   | + explorer_data_handling support        |
 *               |         |          |                   | + Id. Accessors                         |
 *               |         |          |                   | + ADM/SMOS OBT<->UTC conversion         |
 *               |         |          |                   | + ENVISAT support removed               |
 *               |         |          |                   | + New tolerances and orbital            |
 *               |         |          |                   |   parameters (METOP, ERS1, ERS2         |
 *               |         |          |                   |   Cryosat Eccentricity)                 |
 *               +----------------------------------------------------------------------------------+
 *               |    3.4  | 18/11/05 | DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |          |                   | New features:                           |
 *               |         |          |                   | + changes in xl_default_sat_init        |
 *               |         |          |                   | + New function xl_defaul_sat_close      |
 *               |         |          |                   | + EXPCFI-SPR-070 Closed                 |
 *               +----------------------------------------------------------------------------------+
 *               |    3.5  | 26/05/06 | DEIMOS Space S.L. | New features:                           |
 *               |         |          |                   |  + Thread safe library                  |
 *               |         |          |                   |  + C99 compliant                        |
 *               |         |          |                   |  + Optimized libray                     |
 *               |         |          |                   |  + 64-Bits library for MACOS,           |
 *               |         |          |                   |    LINUX and SOLARIS                    |
 *               |         |          |                   |  + New time_conv executable             |
 *               |         |          |                   |  + New satellites: SWARM & EARTHCARE    |
 *               |         |          |                   |  + xl_time_ref_init_file improved       |
 *               |         |          |                   |  + New routines:                        |
 *               |         |          |                   |     xl_star_catalog                     |
 *               |         |          |                   |     xl_radec_to_cart & xl_cart_to_radec |
 *               |         |          |                   |     xl_topocentric_to_ef                |
 *               |         |          |                   |     xl_ef_to_topocentric                |
 *               |         |          |                   |  + SPR- 075 corrected                   |
 *               +----------------------------------------------------------------------------------+
 *               |    3.6  | 24/11/06 | DEIMOS Space S.L. | Maintenance release                     |
 *               |         |          |                   | New features:                           |
 *               |         |          |                   |  + New Coordinate Reference Systems for |
 *               |         |          |                   |    tranformation routines: Barycentric  |
 *               |         |          |                   |    1950 and Galactic                    |
 *               |         |          |                   |  + New time format: SMOS UTC Proteus    |
 *               |         |          |                   |  + New routines:                        |
 *               |         |          |                   |     xl_euler_to_matrix                  |
 *               |         |          |                   |     xl_matrix_to_euler                  |
 *               |         |          |                   |     xl_topocentric_to_ef                |
 *               |         |          |                   |     xl_ef_to_topocentric                |
 *               |         |          |                   |     xl_radec_to_cart                    |
 *               |         |          |                   |     xl_cart_to_radec                    |
 *               |         |          |                   |     xl_star_catalog                     |
 *               +----------------------------------------------------------------------------------+
 *               |    3.7  | 13/07/07 | DEIMOS Space S.L. | Maintenance release                     |
 *               |         |          |                   | New features:                           |
 *               |         |          |                   |  + expcfi_check_libs                    |
 *               |         |          |                   |  + library version for Mac OS X         |
 *               |         |          |                   |    on Intel (32 and 64 bits)            |
 *               +----------------------------------------------------------------------------------+
 *               |  3.7.2  | 31/07/08 | DEIMOS Space S.L. | Maintenance release                     |
 *               +----------------------------------------------------------------------------------+
 *               |    4.0  | 19/01/09 | DEIMOS Space S.L. | New features:                           |
 *               |         |          |                   |  + Funtion interfaces changed for model |
 *               |         |          |                   |    support                              |
 *               +----------------------------------------------------------------------------------+
 *               |    4.1  | 07/05/10 | DEIMOS Space S.L. | Maintenance release                     |
 *               |         |          |                   | New features:                           |
 *               |         |          |                   |  + Time init with list of files         |
 *               |         |          |                   |  + Time initialization with OSF         |
 *               +----------------------------------------------------------------------------------+
 *               |    4.2  | 31/01/11 | DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |          |                   | New features:                           |
 *               |         |          |                   |  + Check and normalization of input     |
 *               |         |          |                   |    vectors and matrices                 |
 *               |         |          |                   |  + Support for curved MLST              |
 *               +----------------------------------------------------------------------------------+
 *               |    4.3  | 06/02/12 | DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |          |                   | New features:                           |
 *               |         |          |                   |  + Pseudo-EF CS added.                  |
 *               |         |          |                   |    Polar motion included in EF CS.      |
 *               |         |          |                   |  + New init function xl_time_id_init    |
 *               |         |          |                   |  + Time init. support IERS Bulletin A   |
 *               |         |          |                   |    and Bulletins A+B                    |
 *               |         |          |                   |  + New time transport formats:          |
 *               |         |          |                   |      - XL_TRANS_GENERIC_GPS             |
 *               |         |          |                   |      - XL_TRANS_GENERIC_GPS_WEEK        |
 *               +----------------------------------------------------------------------------------+
 *               |    4.4  | 05/07/12 | DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |          |                   | New features:                           |
 *               |         |          |                   |  + New non-iterative method to compute  |
 *               |         |          |                   |    transformation from cartesian to     |
 *               |         |          |                   |    geodetic coordinates                 |
 *               |         |          |                   |  + New reference frames for on-board    |
 *               |         |          |                   |    position scheduling: EF and GM2000   |
 *               +----------------------------------------------------------------------------------+
 *               |    4.5  | 01/03/13 | DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |          |                   | New features:                           |
 *               |         |          |                   |  + New function xl_geoid_calc to        |
 *               |         |          |                   |    transform between heights relative to|
 *               |         |          |                   |    the ellipsoid and the geoid.         |
 *               +----------------------------------------------------------------------------------+
 *               |    4.6  | 03/10/13 | DEIMOS Space S.L.U| Maintenance release                     |
 *               +----------------------------------------------------------------------------------+
 *               |    4.7  | 28/03/14 | DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |          |                   | New features:                           |
 *               |         |          |                   |  + Support for SENTINEL-5P, MetOp-SG    |
 *               |         |          |                   |    and Jason-CS satellites              |
 *               +----------------------------------------------------------------------------------+
 *               |    4.8  | 29/10/14 | DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |          |                   | New features:                           |
 *               |         |          |                   |  + New reference frame: Earth Fixed non |
 *               |         |          |                   |    rotating.                            |
 *               |         |          |                   |  + New Sun model to take into account   |
 *               |         |          |                   |    Sun light travel time.               |
 *               |         |          |                   |  + New function for quaternion          |
 *               |         |          |                   |    interpolation:xl_quaternions_interpol|
 *               +----------------------------------------------------------------------------------+
 *               |    4.9  | 23/04/15 | DEIMOS Space S.L.U| Maintenance release                     |
 *               +----------------------------------------------------------------------------------+
 *               |  4.9.1  | 05/06/15 | DEIMOS Space S.L.U| Maintenance release                     |
 *               +----------------------------------------------------------------------------------+
 *               |   4.10  | 29/10/15 | DEIMOS Space S.L.U| Maintenance release                     |
 *               +----------------------------------------------------------------------------------+
 *               |  4.11   | 15/04/16 | DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |          |                   | New features:                           |
 *               |         |          |                   |  + Support for BIOMASS, SENTINEL-5,     |
 *               |         |          |                   |    and SAOCOM-CS satellites             |
 *               +----------------------------------------------------------------------------------+
 *               |  4.12   | 03/11/16 | DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |          |                   | New features:                           |
 *               |         |          |                   |  + Added extrapolation algorithm for    |
 *               |         |          |                   |    quaternions (xl_quaternions_interpol)|
 *               +----------------------------------------------------------------------------------+
 *               |  4.13   | 05/04/17 | DEIMOS Space S.L.U| Maintenance release                     |
 *               +----------------------------------------------------------------------------------+
 *               |  4.14   | 16/11/17 | DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |          |                   | New features:                           |
 *               |         |          |                   |  + New functions for CUC time managing  |
 *               |         |          |                   |    xl_time_cuc_to_processing            |
 *               |         |          |                   |    xl_time_processing_to_cuc            |
 *               |         |          |                   |  + Support for FLEX satellite           |
 *               +----------------------------------------------------------------------------------+
 *               |   4.15  | 20/04/18 | DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |          |                   | New features:                           |
 *               |         |          |                   |  + Refactored code                      |
 *               +----------------------------------------------------------------------------------+
 *               |   4.16  | 09/11/18 | DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |          |                   | New features:                           |
 *               |         |          |                   |  + Improved runtime in time conversions |
 *               +----------------------------------------------------------------------------------+
 *               |   4.17  | 10/05/19 | DEIMOS Space S.L.U| Maintenance release                     |
 *               +----------------------------------------------------------------------------------+
 *               |   4.18  | 08/11/19 | DEIMOS Space S.L.U| Maintenance release                     |
 *               +----------------------------------------------------------------------------------+
 *               |   4.19  | 29/05/20 | DEIMOS Space S.L.U| Maintenance release                     |
 *               +----------------------------------------------------------------------------------+
 *               |   4.20  | 30/11/20 | DEIMOS Space S.L.U| Maintenance release                     |
 *               +----------------------------------------------------------------------------------+
 *               |   4.21  | 10/06/21 | DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |          |                   | New features:                           |
 *               |         |          |                   |  + Optimisation of EOCFI library size   |
 *               |         |          |                   |  + Enable use and customization of      |
 *               |         |          |                   |    satellite SP3 identifiers            |
 *               +----------------------------------------------------------------------------------+
 *               |   4.22  | 22/12/21 | DEIMOS Space S.L.U| Maintenance release                     |
 *               +----------------------------------------------------------------------------------+
 *               |   4.23  | 23/06/22 | DEIMOS Space S.L.U| Maintenance release                     |
 *               +----------------------------------------------------------------------------------+
 *               |   4.24  | 29/11/22 | DEIMOS Space S.L.U| Maintenance release                     |
 *               +----------------------------------------------------------------------------------+
 *               |   4.25  | 10/05/23 | DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |          |                   | New features:                           |
 *               |         |          |                   |  + Support for ANX drift in OSF         |
 *               +----------------------------------------------------------------------------------+
 *               |   4.26  | 31/10/23 | DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |          |                   | New features:                           |
 *               |         |          |                   |  + Support for longitude drift          |
 *               +----------------------------------------------------------------------------------+
 *               |   4.27  | 07/06/24 | DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |          |                   | New features:                           |
 *               |         |          |                   |  + Use the NORAD catalogue number to    |
 *               |         |          |                   |    identify TLE record instead of Name  |
 *               +----------------------------------------------------------------------------------+
 *               |   4.28  |13/12/24  | DEIMOS Space S.L.U| Maintenance release                     |
 *               +----------------------------------------------------------------------------------+
 *               |   4.29  |09/04/25  | DEIMOS Space S.L.U| Maintenance release                     |
 *               +----------------------------------------------------------------------------------+
 *
 ****************************************************************************************************/

#ifndef EXPLORER_LIB_H
#define EXPLORER_LIB_H

#include <explorer_data_handling.h>
#include <explorer_file_handling.h>
/*
  Global define in order to include Earth Explorer Header functions
  from EXPLORER_FILE_HANDLING
 */
#ifndef XF_EARTH_EXPLORER_HEADER
  #define XF_EARTH_EXPLORER_HEADER
#endif

/*******************************************************************************
*
* Defines
*
*******************************************************************************/

/* Length of ierr vectors */
/* ---------------------- */
#define XL_NUM_ERR_TIME_REF_INIT_FILE   1
#define XL_NUM_ERR_TIME_REF_INIT        1
#define XL_NUM_ERR_TIME_ID_INIT         1
#define XL_NUM_ERR_TIME_CLOSE           1
#define XL_NUM_ERR_ASCII_ASCII          2 // AN-679
#define XL_NUM_ERR_ASCII_PROC           1
#define XL_NUM_ERR_ASCII_TRANS          1
#define XL_NUM_ERR_PROC_ASCII           1
#define XL_NUM_ERR_PROC_PROC            1
#define XL_NUM_ERR_PROC_TRANS           1
#define XL_NUM_ERR_TRANS_ASCII          1
#define XL_NUM_ERR_TRANS_TRANS          1
#define XL_NUM_ERR_TRANS_PROC           1
#define XL_NUM_ERR_TIME_ADD             1
#define XL_NUM_ERR_TIME_DIFF            1
#define XL_NUM_ERR_OBT_TIME             1
#define XL_NUM_ERR_TIME_OBT             1
#define XL_NUM_ERR_KEPL_CART            1
#define XL_NUM_ERR_CART_KEPL            1
#define XL_NUM_ERR_MOON                 1
#define XL_NUM_ERR_SUN                  1
#define XL_NUM_ERR_PLANET               1
#define XL_NUM_ERR_STAR                 1
#define XL_NUM_ERR_LEAP_INFO            1
#define XL_NUM_ERR_DEFAULT_SAT_INIT     1
#define XL_NUM_ERR_RUN_INIT             1
#define XL_NUM_ERR_POSITION_ON_ORBIT    1
#define XL_NUM_ERR_GET_ROTATION_ANGLES  1
#define XL_NUM_ERR_GET_ROTATED_VECTORS  1
#define XL_NUM_ERR_QUATERNION_TO_VEC    1
#define XL_NUM_ERR_VEC_TO_QUATERNION    1
#define XL_NUM_ERR_STAR_CATALOG         1
#define XL_NUM_ERR_CART_RADEC           1
#define XL_NUM_ERR_RADEC_CART           1
#define XL_NUM_ERR_TOP_TO_EF            1
#define XL_NUM_ERR_EF_TO_TOP            1
#define XL_NUM_ERR_EULER_TO_MATRIX      1
#define XL_NUM_ERR_MATRIX_TO_EULER      1
#define XL_NUM_ERR_MODEL_INIT           1
#define XL_NUM_ERR_MODEL_CLOSE          1
#define XL_NUM_ERR_GEOID_CALC           1 /* AN-425 */
#define XL_NUM_ERR_QUATERNIONS_INTERPOL 1 /* AN-518 */
#define XL_NUM_ERR_CUC_PROC             1 // AN-673
#define XL_NUM_ERR_PROC_CUC             1 // AN-673
#define XL_NUM_ERR_CHANGE_CART_CS       1 /* AN-713-LIB */

/* Max length of ierr vectors */
/* -------------------------- */
#define XL_ERR_VECTOR_MAX_LENGTH 2 // AN-679
/* Max of the above lengths.
   This value can be safely used in every function declaration. */

/* Maximum size of time variables */
/* ------------------------------ */
/* Max length of time vector in Transport Format */
#define XL_TIME_TRANS_DIM_MAX 4

/* Max length of time string in Ascii Format */
#define XL_TIME_ASCII_DIM_MAX 32

/* AN-673: maximum positions for CUC array
   Note that taken into account P-field description:
   - T-field unit bytes:  
     P0 byte: 3+1 (11 in binary plus one)
     P1 byte: 3 (11 in binary)
     TOTAL: 7 bytes
   - T-field subunit bytes
     P0 byte: 3 (11 in binary)
     P1 byte: 7 (111 in binary)
     TOTAL: 10 bytes
*/
#define XL_MAX_P_FIELD_CUC_UNIT_OCTETS    2
#define XL_MAX_T_FIELD_CUC_UNIT_OCTETS    7
#define XL_MAX_T_FIELD_CUC_SUBUNIT_OCTETS 10
#define XL_MAX_CUC_ARRAY_LENGTH           (XL_MAX_P_FIELD_CUC_UNIT_OCTETS + XL_MAX_T_FIELD_CUC_UNIT_OCTETS + XL_MAX_T_FIELD_CUC_SUBUNIT_OCTETS)

/* Generic maximum lengths */
#define XL_MAX_LENGTH_2  2
#define XL_MAX_LENGTH_3  3
#define XL_MAX_LENGTH_4  4
#define XL_MAX_LENGTH_6  6
#define XL_MAX_LENGTH_9  9
#define XL_MAX_LENGTH_25 25

#ifdef __cplusplus
extern "C"
{
#endif

  /*******************************************************************************
*
* Modes and switches
*
*******************************************************************************/

  /* Returned status code */
  /* -------------------- */

  typedef enum
  {
    XL_ERR = -1, /* Error status   */
    XL_OK = 0, /* Nominal status */
    XL_WARN = 1 /* Warning status */
  } XL_PErr_type_enum;
  /* CAREFUL: variables holding these values MUST be declared as long !!! */

  /* XL_Boolean variable */
  /* ------------------- */

  typedef enum
  {
    XL_FALSE = XD_FALSE, /* False variable */
    XL_TRUE = XD_TRUE /* True variable */
  } XL_Boolean;

  /* Satellite ID */
  /* ------------ */

  typedef enum
  {
    XL_SAT_DEFAULT = 0,
    XL_SAT_DEFAULT1 = 1,
    XL_SAT_DEFAULT2 = 2,
    XL_SAT_DEFAULT3 = 3,
    XL_SAT_DEFAULT4 = 4,
    XL_SAT_DEFAULT5 = 5,
    XL_SAT_DEFAULT6 = 6,
    XL_SAT_DEFAULT7 = 7,
    XL_SAT_DEFAULT8 = 8,
    XL_SAT_DEFAULT9 = 9,
    XL_SAT_ERS1 = 11,
    XL_SAT_ERS2 = 12,
    XL_SAT_ENVISAT = 21,
    XL_SAT_METOP1 = 31,
    XL_SAT_METOP2 = 32,
    XL_SAT_METOP3 = 33,
    XL_SAT_CRYOSAT = 41,
    XL_SAT_ADM = 51,
    XL_SAT_GOCE = 61,
    XL_SAT_SMOS = 71,
    XL_SAT_TERRASAR = 81,
    XL_SAT_EARTHCARE = 91,
    XL_SAT_SWARM_A = 101,
    XL_SAT_SWARM_B = 102,
    XL_SAT_SWARM_C = 103,
    XL_SAT_SENTINEL_1A = 110,
    XL_SAT_SENTINEL_1B = 111,
    XL_SAT_SENTINEL_2 = 112,
    XL_SAT_SENTINEL_3 = 113,
    XL_SAT_SEOSAT = 120,
    XL_SAT_SENTINEL_1C = 125,
    XL_SAT_SENTINEL_2A = 126,
    XL_SAT_SENTINEL_2B = 127,
    XL_SAT_SENTINEL_2C = 128,
    XL_SAT_SENTINEL_3A = 129,
    XL_SAT_SENTINEL_3B = 130,
    XL_SAT_SENTINEL_3C = 131,
    XL_SAT_JASON_CSA = 132, // AN-531
    XL_SAT_JASON_CSB = 133, // AN-531
    XL_SAT_METOP_SG_A1 = 134, // AN-530
    XL_SAT_METOP_SG_A2 = 135, // AN-530
    XL_SAT_METOP_SG_A3 = 136, // AN-530
    XL_SAT_METOP_SG_B1 = 137, // AN-530
    XL_SAT_METOP_SG_B2 = 138, // AN-530
    XL_SAT_METOP_SG_B3 = 139, // AN-530
    XL_SAT_SENTINEL_5P = 140, // AN-502
    XL_SAT_BIOMASS = 141, // AN-601
    XL_SAT_SENTINEL_5 = 142, // AN-639
    XL_SAT_SAOCOM_CS = 143, // AN-640
    XL_SAT_FLEX = 144, /* AN-683 */
    XL_SAT_SENTINEL_6A = 145, /* AN-809 */
    XL_SAT_SENTINEL_6B = 146, /* AN-809 */
    XL_SAT_CIMR = 147,
    XL_SAT_ROSEL = 148,
    XL_SAT_CHIME = 149,
    XL_SAT_CRISTAL = 150,
    XL_SAT_CO2M = 151,
    XL_SAT_LSTM = 152,
    XL_SAT_FORUM = 153,
    XL_SAT_TRUTHS = 154,
    XL_SAT_GENERIC = 200,
    XL_SAT_GENERIC_GEO = 300, /* ANR-353 */
    XL_SAT_MTG = 301, /* ANR-353 */
    XL_SAT_GENERIC_MEO = 400 /* ANR-550 */
  } XL_Sat_id_enum;

  /* Processing time format ID */
  /* ------------------------- */

  typedef enum
  {
    XL_PROC = 0
  } XL_Proc_enum;

  /* Transport time format ID */
  /* ------------------------ */

  typedef enum
  {
    XL_TRANS_STD = 0,
    XL_TRANS_ENVI_GS = 11,
    XL_TRANS_CRYO_GS = 21,
    XL_TRANS_CRYO_TM = 22,
    XL_TRANS_CRYO_TM_SIRAL = 23,
    XL_TRANS_SMOS_TM = 31,
    XL_TRANS_GENERIC_GPS_SEC = 41,
    XL_TRANS_GENERIC_GPS_WEEK = 42
  } XL_Trans_enum;

  /* Time reference ID */
  /* ----------------- */

  typedef enum
  {
    XL_TIME_UNDEF = -1,
    XL_TIME_TAI,
    XL_TIME_UTC,
    XL_TIME_UT1,
    XL_TIME_GPS
  } XL_Time_ref_enum;

  /* ASCII time format ID */
  /* -------------------- */

  typedef enum
  {
    XL_ASCII_UNDEF = -1,
    XL_ASCII_STD = 11,
    XL_ASCII_STD_REF = 12,
    XL_ASCII_STD_MICROSEC = 13,
    XL_ASCII_STD_REF_MICROSEC = 14,
    XL_ASCII_COMPACT = 21,
    XL_ASCII_COMPACT_REF = 22,
    XL_ASCII_COMPACT_MICROSEC = 23,
    XL_ASCII_COMPACT_REF_MICROSEC = 24,
    XL_ASCII_ENVI = 31,
    XL_ASCII_ENVI_REF = 32,
    XL_ASCII_ENVI_MICROSEC = 33,
    XL_ASCII_ENVI_REF_MICROSEC = 34,
    XL_ASCII_CCSDSA = 41,
    XL_ASCII_CCSDSA_REF = 42,
    XL_ASCII_CCSDSA_MICROSEC = 43,
    XL_ASCII_CCSDSA_REF_MICROSEC = 44,
    XL_ASCII_CCSDSA_COMPACT = 51,
    XL_ASCII_CCSDSA_COMPACT_REF = 52,
    XL_ASCII_CCSDSA_COMPACT_MICROSEC = 53,
    XL_ASCII_CCSDSA_COMPACT_REF_MICROSEC = 54
  } XL_Ascii_enum;

  /* Kepler state vector mode */
  /* ------------------------ */

  typedef enum
  {
    XL_KEPLER_MEAN = 1,
    XL_KEPLER_OSC
  } XL_Kepler_mode_enum;

  /* AOCS mode */
  /* --------- */

  typedef enum
  {
    XL_AOCS_DEFAULT = 0,
    XL_AOCS_USER,
    XL_AOCS_GPM,
    XL_AOCS_LNP,
    XL_AOCS_YSM,
    XL_AOCS_FILE
  } XL_Aocs_mode_enum;

  /* Calculation mode */
  /* ---------------- */

  typedef enum
  {
    XL_CALC_POS = 1,
    XL_CALC_POS_VEL,
    XL_CALC_POS_VEL_ACC,
    XL_CALC_ITER_POS,
    XL_CALC_ITER_POS_VEL,
    XL_CALC_NO_ITER_POS,
    XL_CALC_NO_ITER_POS_VEL,

  } XL_Calc_mode_enum;

  /* Time Initialization mode */
  /* ------------------------ */

  typedef enum
  {
    XL_SEL_FILE = 0,
    XL_SEL_TIME,
    XL_SEL_ORBIT,
    XL_SEL_DEFAULT
  } XL_Time_init_mode_enum;

  /* Time Model */
  /* ---------- */

  typedef enum
  {
    XL_TIMEMOD_AUTO = -2,
    XL_TIMEMOD_USER = -1,
    XL_TIMEMOD_NONE = 0,
    XL_TIMEMOD_IERS_B_PREDICTED = 1,
    XL_TIMEMOD_IERS_B_RESTITUTED,
    XL_TIMEMOD_FOS_PREDICTED,
    XL_TIMEMOD_FOS_RESTITUTED,
    XL_TIMEMOD_DORIS_PRELIMINARY,
    XL_TIMEMOD_DORIS_PRECISE,
    XL_TIMEMOD_DORIS_NAVIGATOR,
    XL_TIMEMOD_OSF,
    XL_TIMEMOD_IERS_A_ONLY_PREDICTION,
    XL_TIMEMOD_IERS_A_PREDICTION_AND_FORMULA,
    XL_TIMEMOD_IERS_B_AND_A_ONLY_PREDICTION
  } XL_Time_model_enum;

  /* Reference coordinate frames */
  /* --------------------------- */

  typedef enum
  {
    XL_BM2000 = 1, /* Barycentric Mean of 2000.0 coordinate system */
    XL_HM2000, /* Heliocentric Mean of 2000.0 coordinate system */
    XL_GM2000, /* Geocentric Mean of 2000.0 coordinate system */
    XL_MOD, /* Mean of Date coordinate system */
    XL_TOD, /* True of Date coordinate system */
    XL_PEF, /* Pseudo Earth Fixed coordinate system (AN-385) */
    XL_EF, /* Earth Fixed coordinate system */
    XL_LIF, /* Launch Inertial Frame (AN-341) */
    XL_BM1950, /* Barycentric Mean of 2000.0 coordinate system */
    XL_GALACTIC /* Galactic coordinate system*/
  } XL_CS_rl_enum;

  typedef enum /* It defines a coordinate system */
  {
    BAR_MEAN_2000 = XL_BM2000, /* Barycentric Mean of 2000.0 coordinate system */
    HEL_MEAN_2000 = XL_HM2000, /* Heliocentric Mean of 2000.0 coordinate system */
    GEO_MEAN_2000 = XL_GM2000, /* Geocentric Mean of 2000.0 coordinate system */
    MEAN_DATE = XL_MOD, /* Mean of date coordinate system */
    TRUE_DATE = XL_TOD, /* True of date coordinate system */
    PSEUDO_EARTH_FIXED = XL_PEF, /* Pseudo Earth Fixed coordinate system (AN-385) */
    EARTH_FIXED = XL_EF, /* Earth Fixed coordinate system */
    LIF = XL_LIF, /* Launch Inertial Frame (AN-341) */
    BAR_MEAN_1950 = XL_BM1950, /* Barycentric Mean of 2000.0 coordinate system */
    GALACTIC = XL_GALACTIC, /* Galactic coordinate system*/
    SAT_ACT_REF, /* Satellite relative actual reference cs */
    QUASI_MEAN_DATE, /* Quasi-Mean of Date coordinate system */
    PSE_TRUE_DATE, /* Pseudo-True of Date coordinate system */
    TOPOCENTRIC, /* Topocentric coordinate system */
    SAT_REF, /* Satellite reference coordinate system */
    SAT_REL_REF /* Satellite relative reference coordinate system */
  } xl_cs_enum;

  /* Coordinate System structure */
  typedef enum
  {
    XL_SAT_ORBITAL_REF = 0,
    XL_SAT_NOMINAL_ATT,
    XL_SAT_ATT,
    XL_INSTR_ATT
  } XL_Attitude_fr_enum;

  /* Planets */
  /* ------- */

  typedef enum
  {
    XL_MERCURY = 1, /* Mercury */
    XL_VENUS, /* Venus */
    XL_EM_BAR, /* Earth-Moon barycenter */
    XL_MARS, /* Mars */
    XL_JUPITER, /* Jupiter */
    XL_SATURN, /* Saturn */
    XL_URANUS, /* Uranus */
    XL_NEPTUNE /* Neptune */
  } XL_Planet_enum;

  /* Derivatives */
  /* ----------- */

  typedef enum /* Indicate whether the 1st and 2nd derivatives of data
                           structure are defined or control whether a function
                           calculates the 1st and 2nd derivatives */
  {
    XL_NO_DER = 0, /* No derivative is defined or calculated */
    XL_DER_1ST, /* Only first derivative is defined or calculated */
    XL_DER_2ND /* Both 1st and 2nd derivatives are defined or calculated */
  } XL_Deriv_enum;

  /* Id types */
  /* -------- */
  typedef enum
  {
    XL_INIT_UNKNOWN,
    XL_INIT_RUN,
    XL_INIT_TIME,
    XL_INIT_MODEL,
    XO_INIT_ORBIT,
    XO_INIT_PROPAG,
    XO_INIT_INTERPOL,
    XP_INIT_SAT_NOM_ATT,
    XP_INIT_SAT_ATT,
    XP_INIT_INSTR_ATT,
    XP_INIT_ATTITUDE,
    XP_INIT_ATMOS,
    XP_INIT_DEM,
    XP_INIT_TARGET,
    XV_INIT_SWATH /* AN-468 */
  } XL_Init_types;

  /* Angle Type */
  /* ---------- */
  // AN-736: enumerations aligned with the ones in DH, POINTING
  typedef enum
  {
    XL_ANGLE_TYPE_TRUE_LAT_TOD = 0,
    XL_ANGLE_TYPE_TRUE_LAT_EF,
    XL_ANGLE_TYPE_TRUE_LAT_GM2000
  } XL_Angle_type_enum;

  /* Star catalogues Type */
  /* -------------------- */
  typedef enum
  {
    XL_FK4 = 0,
    XL_FK5
  } XL_Star_catalog_enum;

  /* Vector types */
  /* ------------ */
  typedef enum
  {
    XL_MODE_FLAG_LOCATION = 0,
    XL_MODE_FLAG_DIRECTION
  } XL_Mode_flag_enum;

  /* CFI functions IDs (for error handling) */
  /* -------------------------------------- */
  typedef enum
  {
    XL_NO_FUNC_ID = -1, /* No function code */

    XL_TIME_TRANSPORT_TO_ASCII_ID,
    XL_TIME_TRANSPORT_TO_TRANSPORT_ID,
    XL_TIME_TRANSPORT_TO_PROCESSING_ID,
    XL_TIME_PROCESSING_TO_ASCII_ID,
    XL_TIME_PROCESSING_TO_TRANSPORT_ID,
    XL_TIME_PROCESSING_TO_PROCESSING_ID,
    XL_TIME_ASCII_TO_ASCII_ID,
    XL_TIME_ASCII_TO_TRANSPORT_ID,
    XL_TIME_ASCII_TO_PROCESSING_ID,
    XL_TIME_ADD_ID,
    XL_TIME_DIFF_ID,
    XL_TIME_OBT_TO_TIME_ID,
    XL_TIME_TIME_TO_OBT_ID,
    XL_TIME_REF_INIT_FILE_ID,
    XL_TIME_REF_INIT_ID,
    XL_TIME_ID_INIT_ID,
    XL_TIME_CLOSE_ID,
    XL_CHANGE_CART_CS_ID,
    XL_GEOD_TO_CART_ID,
    XL_CART_TO_GEOD_ID,
    XL_KEPL_TO_CART_ID,
    XL_CART_TO_KEPL_ID,
    XL_SUN_ID,
    XL_MOON_ID,
    XL_PLANET_ID,
    XL_STAR_RADEC_ID,
    XL_GEOD_DISTANCE_ID,
    XL_TIME_GET_LEAP_SECOND_INFO_ID,
    XL_DEFAULT_SAT_INIT_ID,
    XL_RUN_INIT_ID,
    XL_GET_ROTATION_ANGLES_ID,
    XL_GET_ROTATED_VECTORS_ID,
    XL_POSITION_ON_ORBIT_ID,
    XL_QUATERNIONS_TO_VEC_ID,
    XL_VEC_TO_QUATERNIONS_ID,
    XL_STAR_CATALOG_ID,
    XL_CART_TO_RADEC_ID,
    XL_RADEC_TO_CART_ID,
    XL_TOPOCENTRIC_TO_EF_ID,
    XL_EF_TO_TOPOCENTRIC_ID,
    XL_EULER_TO_MATRIX_ID,
    XL_MATRIX_TO_EULER_ID,
    XL_MODEL_INIT_ID,
    XL_MODEL_CLOSE_ID,
    XL_GEOID_CALC_ID, /* AN-425 */
    XL_QUATERNIONS_INTERPOL_ID, /* AN-518 */
    XL_TIME_CUC_TO_PROCESSING_ID, // AN-673
    XL_TIME_PROCESSING_TO_CUC_ID, // AN-673

    XL_MAX_FUNC_ID /* Number of functions defined in this list */
  } XL_Func_enum;

  /* Axis enumeration */
  /* ---------------- */
  typedef enum
  {
    XL_X_AXIS = 0,
    XL_NEG_X_AXIS,
    XL_Y_AXIS,
    XL_NEG_Y_AXIS,
    XL_Z_AXIS,
    XL_NEG_Z_AXIS
  } XL_Axis_enum;

  /* Model sets enumeration */
  /* ---------------------- */
  typedef enum
  {
    XL_MODEL_DEFAULT,
    XL_MODEL_CONFIG
  } xl_model_sets_enum;

  typedef enum
  {
    XL_MODEL_TYPE_EARTH,
    XL_MODEL_TYPE_SUN,
    XL_MODEL_TYPE_MOON,
    XL_MODEL_TYPE_PLANET,
    XL_MODEL_TYPE_STAR,
    XL_MODEL_TYPE_NUTATION,
    XL_MODEL_TYPE_PRECESSION,
    XL_MODEL_TYPE_CONSTANTS,
    XL_MODEL_TYPE_LIGHT_PROPAGATION, // AN-553
    XL_NUM_MODEL_TYPES_ENUM
  } xl_model_types_enum;

  /* Earth Model enumeration */
  /* --------------------- */
  typedef enum
  {
    XL_MODEL_EARTH_DEFAULT
  } xl_model_earth_enum;

  /* Sun Model enumeration */
  /* ----------------------- */
  typedef enum
  {
    XL_MODEL_SUN_DEFAULT,
    XL_MODEL_SUN_TRAVEL_TIME /* AN-517 */
  } xl_model_sun_enum;

  /* Moon Model enumeration */
  /* ---------------------- */
  typedef enum
  {
    XL_MODEL_MOON_DEFAULT
  } xl_model_moon_enum;

  /* Planet Model enumeration */
  /* ------------------------ */
  typedef enum
  {
    XL_MODEL_PLANETS_DEFAULT
  } xl_model_planets_enum;

  /* Star Model enumeration */
  /* ---------------------- */
  typedef enum
  {
    XL_MODEL_STAR_DEFAULT
  } xl_model_star_enum;

  /* Nutation Model enumeration */
  /* -------------------------- */
  typedef enum
  {
    XL_MODEL_NUTATION_DEFAULT
  } xl_model_nutation_enum;

  /* Precession Model enumeration */
  /* ---------------------------- */
  typedef enum
  {
    XL_MODEL_PRECESSION_DEFAULT
  } xl_model_precession_enum;

  /* Constants Model enumeration */
  /* --------------------------- */
  typedef enum
  {
    XL_MODEL_CONSTANTS_DEFAULT
  } xl_model_constants_enum;

  /* AN-553: Light propagation Model enumeration */
  /* ------------------------------------------- */
  typedef enum
  {
    XL_MODEL_LIGHT_PROPAGATION_DISABLED,
    XL_MODEL_LIGHT_PROPAGATION_RECEIVER,
    XL_MODEL_LIGHT_PROPAGATION_TRANSMITTER
  } xl_model_light_propagation_enum;

  /* AN-438 */
  /* BULLETIN TYPE */
  /* ------------- */
  typedef enum
  {
    XL_NO_BULLETIN = -1,
    XL_BULLETIN_B,
    XL_BULLETIN_A,
    XL_BULLETIN_B_AND_A
  } xl_bulletin_type_enum;

  /* IERS prediction formula enabled/disabled */
  /* ---------------------------------------- */
  typedef enum
  {
    XL_FORMULA_ENABLED,
    XL_FORMULA_DISABLED
  } xl_iers_formula_flag_enum;
  /* END AN-438 */

  typedef enum
  {
    XL_FILE_DATA,
    XL_TIME_CORRELATIONS_DATA
  } xl_time_data_origin_enum;

  /* AN-518 */
  typedef enum
  {
    XL_INTERPOL_SLERP
  } xl_quaternions_interpol_algo_enum;

  // AN-673
  typedef enum
  {
    XL_CUC_T_FIELD, /* only T-field */
    XL_CUC_T_AND_P_FIELDS /* T- and P-field */
  } xl_cuc_time_type_enum;

  typedef enum
  {
    XL_EPOCH_CCSDS, // 1/1/1958
    XL_EPOCH_GPS, // 6-Jan-1980, 00h00
    XL_EPOCH_USER_DEFINED // Taken from user inputs
  } xl_cuc_epoch_type_enum;

  /*******************************************************************************
*
* Data structures
*
*******************************************************************************/

  /* Model Id. Structure */
  typedef struct
  {
    void* ee_id;
  } xl_model_id;

  /* Time Id. Structure */
  typedef struct
  {
    void* ee_id;
  } xl_time_id;

  /* Time Id. Accesors */
  typedef struct
  {
    long flag;
    double utc_time;
  } xl_leap_second;

  typedef xd_time_rec xl_time_correlations;

  /* AN-438 */
  typedef xd_polar_motion_params xl_polar_motion_params;
  typedef xd_polar_motion_formula xl_polar_motion_formula;
  typedef xd_time_correlation_formula xl_time_correlation_formula;
  /* END AN-438 */

  typedef struct
  {
    long enabled_flag; /* Flag indicating the time is initialized with eogs (AN-341)*/
    double longitude; /* End of Giro Setting (longitude) (AN-341)*/
    double utc_time; /* End of Giro Setting (UTC time) (AN-341)*/
  } xl_launch_inertial_frame_config; /* AN-341 */

  typedef struct
  {
    long iers_bulletin_type; /* xl_bulletin_type_enum */
    long iers_formula_flag; /* xl_iers_formula_flag_enum (enabled/disabled) */
    long prediction_first_record; /* Indicates the 1st record belonging to bulletin A if applicable */
    xl_polar_motion_formula polar_motion_formula; /* AN-438 */
    xl_time_correlation_formula time_correlation_formula; /* AN-438 */
    long num_lines;
    xl_time_correlations* time_str;
    xl_polar_motion_params* polar_motion_params;
    xl_leap_second leap_sec;
    xl_launch_inertial_frame_config launch_inertial_frame_config; /* AN-341 */
  } xl_time_id_data;

  /* Envisat OBT Structure */
  typedef struct
  {
    long sat_id;
    double time0;
    unsigned long obt0[XL_MAX_LENGTH_2];
    unsigned long period0;
  } xl_envisat_obt_param;

  typedef struct
  {
    long sat_id;
    unsigned long obt[XL_MAX_LENGTH_2];
  } xl_envisat_obt_value;

  /* GOCE OBT Structure */
  typedef struct
  {
    long sat_id;
    unsigned long utc0_c;
    unsigned int utc0_f;
    unsigned long obt0_c;
    unsigned int obt0_f;
    double gradient;
    double offset;
  } xl_goce_obt_param;

  typedef struct
  {
    long sat_id;
    double obt;
  } xl_goce_obt_value;

  /* SMOS OBT Structure */
  typedef struct
  {
    long sat_id;
    long delta_seconds; /* number of seconds to be applied to UTC to give UTC Proteus (just in case UTC Proteus reference is actually GPS Time)*/
    unsigned long obet0_c; /* OBET Coarse Time (in seconds) */
    unsigned long obet0_f; /* OBET Fine Time */
    unsigned long utc0_week; /* UTC (Proteus format) week number */
    unsigned long utc0_seconds; /* UTC (Proteus format) seconds of week */
    unsigned long utc0_fraction; /* UTC (Proteus format) fraction of seconds */
  } xl_smos_obt_param;

  typedef struct
  {
    long sat_id;
    unsigned long obet_c; /* OBET Coarse Time (in seconds) */
    unsigned long obet_f; /* OBET Fine Time */
  } xl_smos_obt_value;

  /* AEOLUS OBT Structure */
  typedef struct
  {
    long sat_id;
    long delta_seconds; /* it refers to the number of seconds to be applied to UTC to give GPS (GPST - UTC) */
  } xl_adm_obt_param;

  typedef struct
  {
    long sat_id;
    unsigned long cuc_sec; /* CCSDS Unsegmented Time Code (seconds) */
    unsigned long cuc_subsec; /* CCSDS Unsegmented Time Code (subseconds) */
  } xl_adm_obt_value;

  /* -- Parameter -- */
  typedef struct
  {
    XL_Deriv_enum deriv; /* Indicate if the 1st and 2nd derivatives are defined */
    double p; /* The parameter, expressed in the appropiate units */
    double pd; /* 1st time derivative of the parameter */
    double p2d; /* 2nd time derivative of the parameter */
  } xl_par_der;

  /* Coordinates data */
  typedef struct
  {
    XL_CS_rl_enum cs;
    XL_Deriv_enum deriv;
    double v[XL_MAX_LENGTH_3];
    double vd[XL_MAX_LENGTH_3];
    double v2d[XL_MAX_LENGTH_3];
  } xl_cord;

  /* Azimuth/ elevation definition */
  typedef struct
  {
    long az_0_axis;
    long az_90_axis;
    long el_90_axis;
  } xl_az_el_definition;

  /* rotation+translation structure */
  typedef struct
  {
    XL_Boolean azel_flag;
    xl_az_el_definition azel_def;
    XL_CS_rl_enum ref_i;
    XL_Attitude_fr_enum ref_f;
    XL_Boolean amb_flag;
    XL_Deriv_enum deriv;
    double v[XL_MAX_LENGTH_3];
    double vd[XL_MAX_LENGTH_3];
    double v2d[XL_MAX_LENGTH_3];
    double m[XL_MAX_LENGTH_3][XL_MAX_LENGTH_3];
    double md[XL_MAX_LENGTH_3][XL_MAX_LENGTH_3];
    double m2d[XL_MAX_LENGTH_3][XL_MAX_LENGTH_3];
  } xl_cs_tra;

  /* data from model */
  typedef struct
  {
    long earth_model; /* Earth model */
    long sun_model; /* Sun model */
    long moon_model; /* Moon model */
    long planet_model; /* Planets model */
    long star_model; /* Stars model */
    long nutation_model; /* Nutation model */
    long precession_model; /* Precession model */
    long constants_model; /* Constants model */
    long light_propagation; /* Light propagation model (AN-553) */

    /* Earth shape and gravity */
    double re; /* Earth equatorial radius [m] */
    double mu; /* Earth's gravitational constant [m3/s2] */
    double j2; /* Second zonal harmonic */
    double j3; /* Third zonal harmonic */
    double j4; /* Fourth zonal harmonic */

    double major_axis; /* Semi-major axis  [m] */
    double minor_axis; /* Semi-minor axis  [m] */
    double ecc; /* First eccentricity [-] */
    double flat; /* Flattening [-] */

    /* Earth rotation */
    double gcoef_0;
    double gcoef_1;
    double gcoef_2;

    double gcoef_sim_0;
    double gcoef_sim_1;
    double gcoef_sim_2;

    /* constants */
    double au; /* Astronomical unit in kms */
  } xl_model_data;

  /* Data for time initialization (added for AN-404) */
  typedef union
  {
    xd_eocfi_file_set file_set;
    xl_time_id_data time_id_data;
  } xl_time_id_init_data_union;

  typedef struct
  {
    long data_type; /* Enumeration value from xl_time_data_origin_enum */
    xl_time_id_init_data_union time_id_init_data;
  } xl_time_id_init_data;

  /* AN-425 */
  typedef struct
  {
    xl_model_id* model_id;
    double latitude;
    double longitude;
    double utc_time;
    long nof_harmonics;
  } xl_geoid_calc_inputs;

  typedef struct
  {
    double undulation;
  } xl_geoid_calc_outputs;
  /* END AN-425 */

  /* AN-518 */
  typedef struct
  {
    xl_quaternions_interpol_algo_enum algo;
  } xl_quaternions_interpol_cfg;

  // AN-673
  typedef struct
  {
    long cuc_type; // xl_cuc_time_type_enum
    long epoch_type; // xl_cuc_epoch_type_enum
    long time_ref; // XL_Time_ref_enum
    double epoch;
    long basic_time_unit_num_octets;
    long fractional_time_unit_num_octets;
  } xl_cuc_time_config;

  /*******************************************************************************
*
* Main CFI Functions - prototype declaration
*
*******************************************************************************/

  /* Model Id. */
  /* --------- */
  long xl_model_init(long* mode, long models[XL_NUM_MODEL_TYPES_ENUM],
                     /* output */
                     xl_model_id* model_id,
                     long ierr[XL_NUM_ERR_MODEL_INIT]);

  long xl_model_close(xl_model_id* model_id, long ierr[]);

  long xl_model_init_status(xl_model_id* model_id);

  long xl_model_get_data(xl_model_id* model_id,
                         xl_model_data* model_data);
  /*
long xl_model_set_data(xl_model_id* model_id,
                       xl_model_data* model_data);
*/

  /* Time Id. Accessors */
  /* ------------------ */

  long xl_time_init_status(xl_time_id* time_id);

  long xl_time_get_mode(xl_time_id* time_id);

  long xl_time_get_sat_id(xl_time_id* time_id);

  long xl_time_get_id_data(xl_time_id* time_id, xl_time_id_data* data);

  long xl_time_set_id_data(xl_time_id* time_id, xl_time_id_data* data);

  void xl_time_free_id_data(xl_time_id_data* data);

  /* Time initialization functions */
  /* ----------------------------- */
  long xl_time_ref_init_file(long* time_model, long* n_files, char** time_file, long* time_init_mode, long* time_ref, double* time0, double* time1, long* orbit0, long* orbit1,
                             /* output */
                             double* val_time0,
                             double* val_time1,
                             xl_time_id* time_id,
                             long ierr[XL_NUM_ERR_TIME_REF_INIT_FILE]);

  long xl_time_ref_init(double time[XL_MAX_LENGTH_4], long* orbit_num, double* anx_time, double* orbit_duration,
                        /* output */
                        xl_time_id* time_id,
                        long ierr[XL_NUM_ERR_TIME_REF_INIT]);

  long xl_time_id_init(long* time_model, xl_time_id_init_data* init_data, long* time_init_mode, long* time_ref, double* time0, double* time1, long* orbit0, long* orbit1,
                       /* output */
                       double* val_time0,
                       double* val_time1,
                       xl_time_id* time_id,
                       long ierr[XL_NUM_ERR_TIME_ID_INIT]);

  long xl_time_close(xl_time_id* time_id, long ierr[XL_NUM_ERR_TIME_CLOSE]);

  /* Time transformation functions */
  /* ----------------------------- */
  long xl_time_transport_to_ascii(xl_time_id* time_id,
                                  long* trans_id_in,
                                  long* time_ref_in,
                                  long transport_in[],
                                  long* ascii_id_out,
                                  long* time_ref_out,
                                  /* output */
                                  char ascii_out[],
                                  long ierr[XL_NUM_ERR_TRANS_ASCII]);

  long xl_time_transport_to_ascii_run(long* run_id,
                                      long* trans_id_in,
                                      long* time_ref_in,
                                      long transport_in[],
                                      long* ascii_id_out,
                                      long* time_ref_out,
                                      /* output */
                                      char ascii_out[],
                                      long ierr[XL_NUM_ERR_TRANS_ASCII]);

  long xl_time_transport_to_transport(xl_time_id* time_id,
                                      long* trans_id_in,
                                      long* time_ref_in,
                                      long transport_in[],
                                      long* trans_id_out,
                                      long* time_ref_out,
                                      /* output */
                                      long transport_out[],
                                      long ierr[XL_NUM_ERR_TRANS_TRANS]);

  long xl_time_transport_to_transport_run(long* run_id,
                                          long* trans_id_in,
                                          long* time_ref_in,
                                          long transport_in[],
                                          long* trans_id_out,
                                          long* time_ref_out,
                                          /* output */
                                          long transport_out[],
                                          long ierr[XL_NUM_ERR_TRANS_TRANS]);

  long xl_time_transport_to_processing(xl_time_id* time_id,
                                       long* trans_id_in,
                                       long* time_ref_in,
                                       long transport_in[],
                                       long* proc_id_out,
                                       long* time_ref_out,
                                       /* output */
                                       double* processing_out,
                                       long ierr[XL_NUM_ERR_TRANS_PROC]);

  long xl_time_transport_to_processing_run(long* run_id,
                                           long* trans_id_in,
                                           long* time_ref_in,
                                           long transport_in[],
                                           long* proc_id_out,
                                           long* time_ref_out,
                                           /* output */
                                           double* processing_out,
                                           long ierr[XL_NUM_ERR_TRANS_PROC]);

  long xl_time_processing_to_ascii(xl_time_id* time_id,
                                   long* proc_id_in,
                                   long* time_ref_in,
                                   double* processing_in,
                                   long* ascii_id_out,
                                   long* time_ref_out,
                                   /* output */
                                   char ascii_out[],
                                   long ierr[XL_NUM_ERR_PROC_ASCII]);

  long xl_time_processing_to_ascii_run(long* run_id,
                                       long* proc_id_in,
                                       long* time_ref_in,
                                       double* processing_in,
                                       long* ascii_id_out,
                                       long* time_ref_out,
                                       /* output */
                                       char ascii_out[],
                                       long ierr[XL_NUM_ERR_PROC_ASCII]);

  long xl_time_processing_to_transport(xl_time_id* time_id,
                                       long* proc_id_in,
                                       long* time_ref_in,
                                       double* processing_in,
                                       long* trans_id_out,
                                       long* time_ref_out,
                                       /* output */
                                       long transport_out[],
                                       long ierr[XL_NUM_ERR_PROC_TRANS]);

  long xl_time_processing_to_transport_run(long* run_id,
                                           long* proc_id_in,
                                           long* time_ref_in,
                                           double* processing_in,
                                           long* trans_id_out,
                                           long* time_ref_out,
                                           /* output */
                                           long transport_out[],
                                           long ierr[XL_NUM_ERR_PROC_TRANS]);

  long xl_time_processing_to_processing(xl_time_id* time_id,
                                        long* proc_id_in,
                                        long* time_ref_in,
                                        double* processing_in,
                                        long* proc_id_out,
                                        long* time_ref_out,
                                        /* output */
                                        double* processing_out,
                                        long ierr[XL_NUM_ERR_PROC_PROC]);

  long xl_time_processing_to_processing_run(long* run_id,
                                            long* proc_id_in,
                                            long* time_ref_in,
                                            double* processing_in,
                                            long* proc_id_out,
                                            long* time_ref_out,
                                            /* output */
                                            double* processing_out,
                                            long ierr[XL_NUM_ERR_PROC_PROC]);

  long xl_time_ascii_to_ascii(xl_time_id* time_id,
                              long* ascii_id_in,
                              long* time_ref_in,
                              char ascii_in[],
                              long* ascii_id_out,
                              long* time_ref_out,
                              /* output */
                              char ascii_out[],
                              long ierr[XL_NUM_ERR_ASCII_ASCII]);

  long xl_time_ascii_to_ascii_run(long* run_id,
                                  long* ascii_id_in,
                                  long* time_ref_in,
                                  char ascii_in[],
                                  long* ascii_id_out,
                                  long* time_ref_out,
                                  /* output */
                                  char ascii_out[],
                                  long ierr[XL_NUM_ERR_ASCII_ASCII]);

  long xl_time_ascii_to_processing(xl_time_id* time_id,
                                   long* ascii_id_in,
                                   long* time_ref_in,
                                   char ascii_in[],
                                   long* proc_id_out,
                                   long* time_ref_out,
                                   /* output */
                                   double* processing_out,
                                   long ierr[XL_NUM_ERR_ASCII_PROC]);

  long xl_time_ascii_to_processing_run(long* run_id,
                                       long* ascii_id_in,
                                       long* time_ref_in,
                                       char ascii_in[],
                                       long* proc_id_out,
                                       long* time_ref_out,
                                       /* output */
                                       double* processing_out,
                                       long ierr[XL_NUM_ERR_ASCII_PROC]);

  long xl_time_ascii_to_transport(xl_time_id* time_id,
                                  long* ascii_id_in,
                                  long* time_ref_in,
                                  char ascii_in[],
                                  long* trans_id_out,
                                  long* time_ref_out,
                                  /* output */
                                  long transport_out[],
                                  long ierr[XL_NUM_ERR_ASCII_TRANS]);

  long xl_time_ascii_to_transport_run(long* run_id,
                                      long* ascii_id_in,
                                      long* time_ref_in,
                                      char ascii_in[],
                                      long* trans_id_out,
                                      long* time_ref_out,
                                      /* output */
                                      long transport_out[],
                                      long ierr[XL_NUM_ERR_ASCII_TRANS]);

  long xl_time_cuc_to_processing(xl_time_id* time_id,
                                 xl_cuc_time_config config,
                                 unsigned char cuc_time[XL_MAX_CUC_ARRAY_LENGTH],
                                 long time_ref,
                                 /*output*/
                                 double* processing_out,
                                 long ierr[XL_NUM_ERR_CUC_PROC]); // AN-673

  long xl_time_processing_to_cuc(xl_time_id* time_id,
                                 xl_cuc_time_config config,
                                 long time_ref,
                                 double processing_in,
                                 /*output*/
                                 unsigned char cuc_time[XL_MAX_CUC_ARRAY_LENGTH],
                                 long ierr[XL_NUM_ERR_PROC_CUC]); // AN-673

  /* Other time functions */
  long xl_time_add(long* proc_id, long* time_ref, double* processing_in, double* added_duration,
                   /* output */
                   double* processing_out,
                   long ierr[XL_NUM_ERR_TIME_ADD]);

  long xl_time_diff(long* proc_id, long* time_ref, double* processing_in_1, double* processing_in_2,
                    /* output */
                    double* processing_out,
                    long ierr[XL_NUM_ERR_TIME_DIFF]);

  long xl_time_obt_to_time(long* sat_id,
                           long* proc_id,
                           void* obt_param, /* Place here a pointer to xl_<satellite>_obt_param */
                           void* obt_value_in, /* Place here a pointer to xl_<satellite>_obt_value */
                           /* output */
                           double* time_out,
                           long ierr[XL_NUM_ERR_OBT_TIME]);

  long xl_time_obt_to_time_run(long* run_id,
                               long* proc_id,
                               void* obt_param, /* Place here a pointer to xl_<satellite>_obt_param */
                               void* obt_value_in, /* Place here a pointer to xl_<satellite>_obt_value */
                               /* output */
                               double* time_out,
                               long ierr[XL_NUM_ERR_OBT_TIME]);

  long xl_time_time_to_obt(long* sat_id,
                           long* proc_id,
                           void* obt_param, /* Place here a pointer to xl_<satellite>_obt_param */
                           double* time_in,
                           /* output */
                           void* obt_value_out, /* Place here a pointer to xl_<satellite>_obt_value */
                           long ierr[XL_NUM_ERR_TIME_OBT]);

  long xl_time_time_to_obt_run(long* run_id,
                               long* proc_id,
                               void* obt_param, /* Place here a pointer to xl_<satellite>_obt_param */
                               double* time_in,
                               /* output */
                               void* obt_value_out, /* Place here a pointer to xl_<satellite>_obt_value */
                               long ierr[XL_NUM_ERR_TIME_OBT]);

  long xl_time_get_leap_second_info(xl_time_id* time_id, long* ascii_id_out,
                                    /* output */
                                    long* leap_flag,
                                    char ascii_utc_time_before_leap[],
                                    char ascii_utc_time_after_leap[],
                                    long ierr[XL_NUM_ERR_LEAP_INFO]);

  long xl_time_get_leap_second_info_run(long* run_id, long* ascii_id_out,
                                        /* output */
                                        long* leap_flag,
                                        char ascii_utc_time_before_leap[],
                                        char ascii_utc_time_after_leap[],
                                        long ierr[XL_NUM_ERR_LEAP_INFO]);
  /* Ephemerides functions */
  long xl_sun(xl_model_id* model_id, xl_time_id* time_id, long* time_ref, double* time,
              /* output */
              double sun_pos[XL_MAX_LENGTH_3],
              double sun_vel[XL_MAX_LENGTH_3],
              long ierr[XL_NUM_ERR_SUN]);

  long xl_sun_run(long* run_id, long* time_ref, double* time,
                  /* output */
                  double sun_pos[XL_MAX_LENGTH_3],
                  double sun_vel[XL_MAX_LENGTH_3],
                  long ierr[XL_NUM_ERR_SUN]);

  long xl_moon(xl_model_id* model_id, xl_time_id* time_id, long* time_ref, double* time,
               /* output */
               double moon_pos[XL_MAX_LENGTH_3],
               double moon_vel[XL_MAX_LENGTH_3],
               long ierr[XL_NUM_ERR_MOON]);

  long xl_moon_run(long* run_id, long* time_ref, double* time,
                   /* output */
                   double moon_pos[XL_MAX_LENGTH_3],
                   double moon_vel[XL_MAX_LENGTH_3],
                   long ierr[XL_NUM_ERR_MOON]);

  long xl_planet(xl_model_id* model_id, xl_time_id* time_id, long* planet, long* time_ref, double* time,
                 /* output */
                 double planet_pos[XL_MAX_LENGTH_3],
                 double planet_vel[XL_MAX_LENGTH_3],
                 long ierr[XL_NUM_ERR_PLANET]);

  long xl_planet_run(long* run_id, long* planet, long* time_ref, double* time,
                     /* output */
                     double planet_pos[XL_MAX_LENGTH_3],
                     double planet_vel[XL_MAX_LENGTH_3],
                     long ierr[XL_NUM_ERR_PLANET]);

  long xl_star_radec(xl_model_id* model_id, xl_time_id* time_id, long* time_ref, double* time, double* ra0, double* dec0, double* mu_ra, double* mu_dec, double* rad_vel, double* par,
                     /* output */
                     double* ra,
                     double* dec,
                     long ierr[XL_NUM_ERR_STAR]);

  long xl_star_radec_run(long* run_id, long* time_ref, double* time, double* ra0, double* dec0, double* mu_ra, double* mu_dec, double* rad_vel, double* par,
                         /* output */
                         double* ra,
                         double* dec,
                         long ierr[XL_NUM_ERR_STAR]);

  long xl_star_catalog(xl_model_id* model_id, xl_time_id* time_id, long* time_ref, double* time, long* mode, long* catalog_in, long* catalog_out, double* ra0, double* dec0, double* mu_ra0, double* mu_dec0, double* rad_vel0, double* par0,
                       /* output */
                       double* ra,
                       double* dec,
                       long ierr[XL_NUM_ERR_STAR_CATALOG]);

  long xl_star_catalog_run(long* run_id, long* time_ref, double* time, long* mode, long* catalog_in, long* catalog_out, double* ra0, double* dec0, double* mu_ra0, double* mu_dec0, double* rad_vel0, double* par0,
                           /* output */
                           double* ra,
                           double* dec,
                           long ierr[XL_NUM_ERR_STAR_CATALOG]);

  /* Reference system transformations */
  long xl_change_cart_cs(xl_model_id* model_id, xl_time_id* time_id, long* mode, long* cs_in, long* cs_out, long* time_ref, double* time, double pos[XL_MAX_LENGTH_3], double vel[XL_MAX_LENGTH_3], double acc[XL_MAX_LENGTH_3],
                         /* output */
                         double pos_out[XL_MAX_LENGTH_3],
                         double vel_out[XL_MAX_LENGTH_3],
                         double acc_out[XL_MAX_LENGTH_3]);

  long xl_change_cart_cs_run(long* run_id, long* mode, long* cs_in, long* cs_out, long* time_ref, double* time, double pos[XL_MAX_LENGTH_3], double vel[XL_MAX_LENGTH_3], double acc[XL_MAX_LENGTH_3],
                             /* output */
                             double pos_out[XL_MAX_LENGTH_3],
                             double vel_out[XL_MAX_LENGTH_3],
                             double acc_out[XL_MAX_LENGTH_3]);

  long xl_cart_to_geod(xl_model_id* model_id, long* mode, double* pos, double* vel,
                       /* output */
                       double* lon,
                       double* lat,
                       double* h,
                       double* lon_rate,
                       double* lat_rate,
                       double* h_rate);

  long xl_cart_to_geod_run(long* run_id, long* mode, double* pos, double* vel,
                           /* output */
                           double* lon,
                           double* lat,
                           double* h,
                           double* lon_rate,
                           double* lat_rate,
                           double* h_rate);

  long xl_geod_to_cart(xl_model_id* model_id, long* mode, double* lon, double* lat, double* h, double* lon_rate, double* lat_rate, double* h_rate,
                       /* output */
                       double* pos,
                       double* vel);

  long xl_geod_to_cart_run(long* run_id, long* mode, double* lon, double* lat, double* h, double* lon_rate, double* lat_rate, double* h_rate,
                           /* output */
                           double* pos,
                           double* vel);

  long xl_kepl_to_cart(xl_model_id* model_id,
                       long* kepl_mode,
                       double kepl_in[XL_MAX_LENGTH_6],
                       /* output */
                       double pos_out[XL_MAX_LENGTH_3],
                       double vel_out[XL_MAX_LENGTH_3],
                       long ierr[XL_NUM_ERR_KEPL_CART]);

  long xl_kepl_to_cart_run(long* run_id,
                           long* kepl_mode,
                           double kepl_in[XL_MAX_LENGTH_6],
                           /* output */
                           double pos_out[XL_MAX_LENGTH_3],
                           double vel_out[XL_MAX_LENGTH_3],
                           long ierr[XL_NUM_ERR_KEPL_CART]);

  long xl_cart_to_kepl(xl_model_id* model_id,
                       double pos_in[XL_MAX_LENGTH_3],
                       double vel_in[XL_MAX_LENGTH_3],
                       long* kepl_mode,
                       /* output */
                       double kepl_out[XL_MAX_LENGTH_6],
                       long ierr[XL_NUM_ERR_CART_KEPL]);

  long xl_cart_to_kepl_run(long* run_id,
                           double pos_in[XL_MAX_LENGTH_3],
                           double vel_in[XL_MAX_LENGTH_3],
                           long* kepl_mode,
                           /* output */
                           double kepl_out[XL_MAX_LENGTH_6],
                           long ierr[XL_NUM_ERR_CART_KEPL]);

  long xl_cart_to_radec(xl_model_id* model_id,
                        long* mode,
                        long* cs_in,
                        double pos[XL_MAX_LENGTH_3],
                        double vel[XL_MAX_LENGTH_3],
                        /* output */
                        double* ra,
                        double* dec,
                        double* mu_ra,
                        double* mu_dec,
                        double* rad_vel,
                        double* par,
                        long ierr[XL_NUM_ERR_CART_RADEC]);

  long xl_cart_to_radec_run(long* run_id,
                            long* mode,
                            long* cs_in,
                            double pos[XL_MAX_LENGTH_3],
                            double vel[XL_MAX_LENGTH_3],
                            /* output */
                            double* ra,
                            double* dec,
                            double* mu_ra,
                            double* mu_dec,
                            double* rad_vel,
                            double* par,
                            long ierr[XL_NUM_ERR_CART_RADEC]);

  long xl_radec_to_cart(xl_model_id* model_id,
                        long* mode,
                        long* cs_in,
                        double* ra,
                        double* dec,
                        double* mu_ra,
                        double* mu_dec,
                        double* rad_vel,
                        double* par,
                        /* output */
                        double pos[XL_MAX_LENGTH_3],
                        double vel[XL_MAX_LENGTH_3],
                        long ierr[XL_NUM_ERR_RADEC_CART]);

  long xl_radec_to_cart_run(long* run_id,
                            long* mode,
                            long* cs_in,
                            double* ra,
                            double* dec,
                            double* mu_ra,
                            double* mu_dec,
                            double* rad_vel,
                            double* par,
                            /* output */
                            double pos[XL_MAX_LENGTH_3],
                            double vel[XL_MAX_LENGTH_3],
                            long ierr[XL_NUM_ERR_RADEC_CART]);

  long xl_topocentric_to_ef(xl_model_id* model_id, long* mode, long* deriv, double pos[XL_MAX_LENGTH_3], double vel[XL_MAX_LENGTH_3], double* azim, double* elev, double* range, double* azim_d, double* elev_d, double* range_d,
                            /* output */
                            double ef_dir[XL_MAX_LENGTH_3],
                            double ef_dir_d[XL_MAX_LENGTH_3],
                            long ierr[XL_NUM_ERR_TOP_TO_EF]);

  long xl_topocentric_to_ef_run(long* run_id, long* mode, long* deriv, double pos[XL_MAX_LENGTH_3], double vel[XL_MAX_LENGTH_3], double* azim, double* elev, double* range, double* azim_d, double* elev_d, double* range_d,
                                /* output */
                                double ef_dir[XL_MAX_LENGTH_3],
                                double ef_dir_d[XL_MAX_LENGTH_3],
                                long ierr[XL_NUM_ERR_TOP_TO_EF]);

  long xl_ef_to_topocentric(xl_model_id* model_id, long* mode, long* deriv, double pos[XL_MAX_LENGTH_3], double vel[XL_MAX_LENGTH_3], double ef_dif[XL_MAX_LENGTH_3], double ef_dir_d[XL_MAX_LENGTH_3],
                            /* output */
                            double* azim,
                            double* elev,
                            double* range,
                            double* azim_d,
                            double* elev_d,
                            double* range_d,
                            long ierr[XL_NUM_ERR_EF_TO_TOP]);

  long xl_ef_to_topocentric_run(long* run_id, long* mode, long* deriv, double pos[XL_MAX_LENGTH_3], double vel[XL_MAX_LENGTH_3], double ef_dif[XL_MAX_LENGTH_3], double ef_dir_d[XL_MAX_LENGTH_3],
                                /* output */
                                double* azim,
                                double* elev,
                                double* range,
                                double* azim_d,
                                double* elev_d,
                                double* range_d,
                                long ierr[XL_NUM_ERR_EF_TO_TOP]);

  long xl_euler_to_matrix(double ang[XL_MAX_LENGTH_3],
                          /* output */
                          double matrix[XL_MAX_LENGTH_3][XL_MAX_LENGTH_3],
                          long ierr[XL_NUM_ERR_EULER_TO_MATRIX]);

  long xl_matrix_to_euler(double matrix[XL_MAX_LENGTH_3][XL_MAX_LENGTH_3],
                          /* output */
                          double ang[XL_MAX_LENGTH_3],
                          long ierr[XL_NUM_ERR_MATRIX_TO_EULER]);

  /* Geodesic functions */
  long xl_geod_distance(xl_model_id* model_id,
                        double* lon1,
                        double* lat1,
                        double* lon2,
                        double* lat2,
                        double* h,
                        /* output */
                        double* distance,
                        double* az_1_to_2,
                        double* az_2_to_1);

  long xl_geod_distance_run(long* run_id,
                            double* lon1,
                            double* lat1,
                            double* lon2,
                            double* lat2,
                            double* h,
                            /* output */
                            double* distance,
                            double* az_1_to_2,
                            double* az_2_to_1);

  /* default satellites init*/
  long xl_default_sat_init(long* sat_id, char* file, long ierr[XL_NUM_ERR_DEFAULT_SAT_INIT]);

  void xl_default_sat_close(long* sat_id);

  long xl_set_tle_sat_data(long* sat_id,
                           long* norad_sat_number,
                           char norad_satcat[XL_MAX_LENGTH_25],
                           char int_des[XL_MAX_LENGTH_9]);

  long xl_set_sp3_sat_data(long sat_id,
                           const char* sp3_id);
  /* run_id functions */
  long xl_run_init(long* sat_id,
                   xl_time_id* time_id,
                   xl_model_id* model_id,
                   /* output */
                   long* run_id,
                   long ierr[XL_NUM_ERR_RUN_INIT]);

  long xl_run_init_status(long* run_id);

  long xl_run_id_status(long* run_id, long* type);

  void xl_run_get_ids(long* run_id, long* sat_id, xl_time_id* time_id, xl_model_id* model_id);

  void xl_run_close(long* run_id);

  /* functions for attitude calculations */

  long xl_get_rotation_angles(double* xs_initial, double* ys_initial, double* zs_initial, double* xs_final, double* ys_final, double* zs_final,
                              /* output */
                              double* ang,
                              long* ierr);

  long xl_get_rotated_vectors(double* xs_initial, double* ys_initial, double* zs_initial, double* ang, double* xs_final, double* ys_final,
                              /* output */
                              double* zs_final,
                              long* ierr);

  long xl_quaternions_to_vectors(double quaternions[],
                                 /* output */
                                 double ux_vec[],
                                 double uy_vec[],
                                 double uz_vec[],
                                 long ierr[XL_NUM_ERR_QUATERNION_TO_VEC]);

  long xl_vectors_to_quaternions(double ux_vec[], double uy_vec[], double uz_vec[],
                                 /* output */
                                 double quaternions[],
                                 long ierr[XL_NUM_ERR_VEC_TO_QUATERNION]);

  /* xl_position_on_orbit */
  long xl_position_on_orbit(xl_model_id* model_id, xl_time_id* time_id, long* angle_type, long* time_ref, double* time, double pos[XL_MAX_LENGTH_3], double vel[XL_MAX_LENGTH_3], double acc[XL_MAX_LENGTH_3], long* deriv,
                            /* Output */
                            double* angle,
                            double* angle_rate,
                            double* angle_rate_rate,
                            long ierr[XL_NUM_ERR_POSITION_ON_ORBIT]);

  long xl_position_on_orbit_run(long* run_id,
                                long* angle_type,
                                long* time_ref,
                                double* time,
                                double pos[XL_MAX_LENGTH_3],
                                double vel[XL_MAX_LENGTH_3],
                                double acc[XL_MAX_LENGTH_3],
                                long* deriv,
                                /* Output */
                                double* angle,
                                double* angle_rate,
                                double* angle_rate_rate,
                                long ierr[XL_NUM_ERR_POSITION_ON_ORBIT]);

  /* xl_geoid_undu */
  long xl_geoid_calc(xl_geoid_calc_inputs* geoid_calc_inputs,
                     xl_geoid_calc_outputs* geoid_calc_outputs,
                     long ierr[XL_NUM_ERR_GEOID_CALC]);

  /* AN-518 */
  long xl_quaternions_interpol(xl_quaternions_interpol_cfg* quaternions_interpol_cfg, double* time_1, double* q1, double* time_2, double* q2, double* time_out, double* q_out, long ierr[XL_NUM_ERR_QUATERNIONS_INTERPOL]);

  /* Check libraries compatibility */

  long xl_check_library_version();

/* Alias for xx_check_library_version */
#if defined(expcfi_check_libs)

  #if !defined(xo_check_library_version) && !defined(xp_check_library_version) && !defined(xv_check_library_version)

    #undef expcfi_check_libs
    #define expcfi_check_libs xl_check_library_version

  #endif

#else

  #define expcfi_check_libs xl_check_library_version

#endif

/*******************************************************************************
*
* Error Handling Functions - prototype declaration
*
*******************************************************************************/

/* Maximum length of a string (function name or error message) */
#define XL_MAX_STR 512

/* Maximum number of error codes that can be defined for a single function */
#define XL_MAX_COD 256

  long xl_silent(void);

  long xl_verbose(void);

  long xl_print_msg(long* n, char msg[XL_MAX_COD][XL_MAX_STR]);

  long xl_get_msg(long* func_id, long* ierr,
                  /* output */
                  long* n,
                  char msg[XL_MAX_COD][XL_MAX_STR]);

  long xl_get_code(long* func_id, long* ierr,
                   /* output */
                   long* n,
                   long vec[XL_MAX_COD]);

  /*******************************************************************************
*
* List of errors of the main CFI functions (Look into the EXPLORER_LIB SUM)
*
*******************************************************************************/

  /* Error handling constants */
  /* ------------------------ */

#define XL_ZERO_E    0
#define XL_FIRST_E   32
#define XL_SECOND_E  64
#define XL_THIRD_E   96
#define XL_FORTH_E   128
#define XL_FIFTH_E   160
#define XL_SIXTH_E   192
#define XL_SEVENTH_E 224
#define XL_EIGHTH_E  256
#define XL_NINETH_E  288
#define XL_TENTH_E   320

  /* Error codes list of xl_time_transport_to_transport */
  /* -------------------------------------------------- */
  typedef enum
  {
    XL_CFI_TIME_TRANS_TRANS_TRANS_IN_ERR = XL_ZERO_E,
    XL_CFI_TIME_TRANS_TRANS_TIME_IN_ERR,
    XL_CFI_TIME_TRANS_TRANS_TRANS_OUT_ERR,
    XL_CFI_TIME_TRANS_TRANS_TIME_OUT_ERR,
    XL_CFI_TIME_TRANS_TRANS_DAY_ERR,
    XL_CFI_TIME_TRANS_TRANS_SEC_ERR,
    XL_CFI_TIME_TRANS_TRANS_MILLISEC_ERR,
    XL_CFI_TIME_TRANS_TRANS_MICROSEC_ERR,
    XL_CFI_TIME_TRANS_TRANS_TICK_ERR,
    XL_CFI_TIME_TRANS_TRANS_REF_INIT_ERR,
    XL_CFI_TIME_TRANS_TRANS_REF_INIT_WARN,
    XL_CFI_TIME_TRANS_TRANS_BUL_A_FORMULA_WARN,
    XL_CFI_TIME_TRANS_TRANS_BUL_B_A_GAP_WARN,
    XL_CFI_TIME_TRANS_TRANS_VALIDITY_WARN,
    XL_CFI_TIME_TRANS_TRANS_GPS_REF_ERR,
    XL_CFI_TIME_TRANS_TRANS_WEEK_ERR,
    XL_CFI_TIME_TRANS_TRANS_GPS_EPOCH_ERR,
    XL_CFI_TIME_TRANS_TRANS_STD_ERR

  } XL_CFI_Time_transport_to_transport_err_enum;

  /* Error codes list of xl_time_transport_to_processing */
  /* --------------------------------------------------- */
  typedef enum
  {
    XL_CFI_TIME_TRANS_PROC_TRANS_IN_ERR = XL_ZERO_E,
    XL_CFI_TIME_TRANS_PROC_TIME_IN_ERR,
    XL_CFI_TIME_TRANS_PROC_PROC_OUT_ERR,
    XL_CFI_TIME_TRANS_PROC_TIME_OUT_ERR,
    XL_CFI_TIME_TRANS_PROC_DAY_ERR,
    XL_CFI_TIME_TRANS_PROC_SEC_ERR,
    XL_CFI_TIME_TRANS_PROC_MILLISEC_ERR,
    XL_CFI_TIME_TRANS_PROC_MICROSEC_ERR,
    XL_CFI_TIME_TRANS_PROC_TICK_ERR,
    XL_CFI_TIME_TRANS_PROC_REF_INIT_ERR,
    XL_CFI_TIME_TRANS_PROC_REF_INIT_WARN,
    XL_CFI_TIME_TRANS_PROC_BUL_A_FORMULA_WARN,
    XL_CFI_TIME_TRANS_PROC_BUL_B_A_GAP_WARN,
    XL_CFI_TIME_TRANS_PROC_VALIDITY_WARN,
    XL_CFI_TIME_TRANS_PROC_GPS_REF_ERR,
    XL_CFI_TIME_TRANS_PROC_WEEK_ERR,
    XL_CFI_TIME_TRANS_PROC_STD_ERR

  } XL_CFI_Time_transport_to_processing_err_enum;

  /* Error codes list of xl_time_processing_to_transport */
  /* --------------------------------------------------- */
  typedef enum
  {
    XL_CFI_TIME_PROC_TRANS_PROC_IN_ERR = XL_ZERO_E,
    XL_CFI_TIME_PROC_TRANS_TIME_IN_ERR,
    XL_CFI_TIME_PROC_TRANS_TRANS_OUT_ERR,
    XL_CFI_TIME_PROC_TRANS_TIME_OUT_ERR,
    XL_CFI_TIME_PROC_TRANS_DAY_ERR,
    XL_CFI_TIME_PROC_TRANS_REF_INIT_ERR,
    XL_CFI_TIME_PROC_TRANS_REF_INIT_WARN,
    XL_CFI_TIME_PROC_TRANS_BUL_A_FORMULA_WARN,
    XL_CFI_TIME_PROC_TRANS_BUL_B_A_GAP_WARN,
    XL_CFI_TIME_PROC_TRANS_VALIDITY_WARN,
    XL_CFI_TIME_PROC_TRANS_GPS_REF_ERR,
    XL_CFI_TIME_PROC_TRANS_GPS_EPOCH_ERR,
    XL_CFI_TIME_PROC_TRANS_STD_ERR

  } XL_CFI_Time_processing_to_transport_err_enum;

  /* Error codes list of xl_time_processing_to_processing */
  /* ---------------------------------------------------- */
  typedef enum
  {
    XL_CFI_TIME_PROC_PROC_PROC_IN_ERR = XL_ZERO_E,
    XL_CFI_TIME_PROC_PROC_TIME_IN_ERR,
    XL_CFI_TIME_PROC_PROC_PROC_OUT_ERR,
    XL_CFI_TIME_PROC_PROC_TIME_OUT_ERR,
    XL_CFI_TIME_PROC_PROC_DAY_ERR,
    XL_CFI_TIME_PROC_PROC_REF_INIT_ERR,
    XL_CFI_TIME_PROC_PROC_REF_INIT_WARN,
    XL_CFI_TIME_PROC_PROC_BUL_A_FORMULA_WARN,
    XL_CFI_TIME_PROC_PROC_BUL_B_A_GAP_WARN,
    XL_CFI_TIME_PROC_PROC_VALIDITY_WARN,
    XL_CFI_TIME_PROC_PROC_STD_ERR

  } XL_CFI_Time_processing_to_processing_err_enum;

  /* Error codes list of xl_time_ascii_to_transport */
  /* ---------------------------------------------- */
  typedef enum
  {
    XL_CFI_TIME_ASCII_TRANS_ASCII_IN_ERR = XL_ZERO_E,
    XL_CFI_TIME_ASCII_TRANS_TIME_IN_ERR,
    XL_CFI_TIME_ASCII_TRANS_FORMAT_IN_ERR,
    XL_CFI_TIME_ASCII_TRANS_REF_INC_IN_ERR,
    XL_CFI_TIME_ASCII_TRANS_TRANS_OUT_ERR,
    XL_CFI_TIME_ASCII_TRANS_TIME_OUT_ERR,
    XL_CFI_TIME_ASCII_TRANS_YEAR_ERR,
    XL_CFI_TIME_ASCII_TRANS_MONTH_ERR,
    XL_CFI_TIME_ASCII_TRANS_DAY_ERR,
    XL_CFI_TIME_ASCII_TRANS_HOUR_ERR,
    XL_CFI_TIME_ASCII_TRANS_MIN_ERR,
    XL_CFI_TIME_ASCII_TRANS_SEC_ERR,
    XL_CFI_TIME_ASCII_TRANS_MICROSEC_ERR,
    XL_CFI_TIME_ASCII_TRANS_MJD_ERR,
    XL_CFI_TIME_ASCII_TRANS_REF_INIT_ERR,
    XL_CFI_TIME_ASCII_TRANS_REF_INIT_WARN,
    XL_CFI_TIME_ASCII_TRANS_BUL_A_FORMULA_WARN,
    XL_CFI_TIME_ASCII_TRANS_BUL_B_A_GAP_WARN,
    XL_CFI_TIME_ASCII_TRANS_VALIDITY_WARN,
    XL_CFI_TIME_ASCII_TRANS_GPS_REF_ERR,
    XL_CFI_TIME_ASCII_TRANS_GPS_EPOCH_ERR

  } XL_CFI_Time_ascii_to_transport_err_enum;

  /* Error codes list of xl_time_ascii_to_processing */
  /* ----------------------------------------------- */
  typedef enum
  {
    XL_CFI_TIME_ASCII_PROC_ASCII_IN_ERR = XL_ZERO_E,
    XL_CFI_TIME_ASCII_PROC_TIME_IN_ERR,
    XL_CFI_TIME_ASCII_PROC_FORMAT_IN_ERR,
    XL_CFI_TIME_ASCII_PROC_REF_INC_IN_ERR,
    XL_CFI_TIME_ASCII_PROC_PROC_OUT_ERR,
    XL_CFI_TIME_ASCII_PROC_TIME_OUT_ERR,
    XL_CFI_TIME_ASCII_PROC_YEAR_ERR,
    XL_CFI_TIME_ASCII_PROC_MONTH_ERR,
    XL_CFI_TIME_ASCII_PROC_DAY_ERR,
    XL_CFI_TIME_ASCII_PROC_HOUR_ERR,
    XL_CFI_TIME_ASCII_PROC_MIN_ERR,
    XL_CFI_TIME_ASCII_PROC_SEC_ERR,
    XL_CFI_TIME_ASCII_PROC_MICROSEC_ERR,
    XL_CFI_TIME_ASCII_PROC_MJD_ERR,
    XL_CFI_TIME_ASCII_PROC_REF_INIT_ERR,
    XL_CFI_TIME_ASCII_PROC_REF_INIT_WARN,
    XL_CFI_TIME_ASCII_PROC_BUL_A_FORMULA_WARN,
    XL_CFI_TIME_ASCII_PROC_BUL_B_A_GAP_WARN,
    XL_CFI_TIME_ASCII_PROC_VALIDITY_WARN

  } XL_CFI_Time_ascii_to_processing_err_enum;

  /* Error codes list of xl_time_processing_to_ascii */
  /* ----------------------------------------------- */
  typedef enum
  {
    XL_CFI_TIME_PROC_ASCII_PROC_IN_ERR = XL_ZERO_E,
    XL_CFI_TIME_PROC_ASCII_TIME_IN_ERR,
    XL_CFI_TIME_PROC_ASCII_DAY_ERR,
    XL_CFI_TIME_PROC_ASCII_ASCII_OUT_ERR,
    XL_CFI_TIME_PROC_ASCII_TIME_OUT_ERR,
    XL_CFI_TIME_PROC_ASCII_ASCII_MJD_ERR,
    XL_CFI_TIME_PROC_ASCII_ASCII_YEAR_ERR,
    XL_CFI_TIME_PROC_ASCII_ASCII_MONTH_ERR,
    XL_CFI_TIME_PROC_ASCII_ASCII_DAY_ERR,
    XL_CFI_TIME_PROC_ASCII_ASCII_HOUR_ERR,
    XL_CFI_TIME_PROC_ASCII_ASCII_MIN_ERR,
    XL_CFI_TIME_PROC_ASCII_ASCII_SEC_ERR,
    XL_CFI_TIME_PROC_ASCII_ASCII_MICROSEC_ERR,
    XL_CFI_TIME_PROC_ASCII_FORMAT_OUT_ERR,
    XL_CFI_TIME_PROC_ASCII_REF_INIT_ERR,
    XL_CFI_TIME_PROC_ASCII_REF_INIT_WARN,
    XL_CFI_TIME_PROC_ASCII_BUL_A_FORMULA_WARN,
    XL_CFI_TIME_PROC_ASCII_BUL_B_A_GAP_WARN,
    XL_CFI_TIME_PROC_ASCII_VALIDITY_WARN

  } XL_CFI_Time_processing_to_ascii_err_enum;

  /* Error codes list of xl_time_ascii_to_ascii */
  /* ------------------------------------------ */
  typedef enum
  {
    XL_CFI_TIME_ASCII_ASCII_ASCII_IN_ERR = XL_ZERO_E,
    XL_CFI_TIME_ASCII_ASCII_TIME_IN_ERR,
    XL_CFI_TIME_ASCII_ASCII_FORMAT_IN_ERR,
    XL_CFI_TIME_ASCII_ASCII_REF_INC_IN_ERR,
    XL_CFI_TIME_ASCII_ASCII_ASCII_OUT_ERR,
    XL_CFI_TIME_ASCII_ASCII_TIME_OUT_ERR,
    XL_CFI_TIME_ASCII_ASCII_YEAR_IN_ERR,
    XL_CFI_TIME_ASCII_ASCII_MONTH_IN_ERR,
    XL_CFI_TIME_ASCII_ASCII_DAY_IN_ERR,
    XL_CFI_TIME_ASCII_ASCII_HOUR_IN_ERR,
    XL_CFI_TIME_ASCII_ASCII_MIN_IN_ERR,
    XL_CFI_TIME_ASCII_ASCII_SEC_IN_ERR,
    XL_CFI_TIME_ASCII_ASCII_MICROSEC_IN_ERR,
    XL_CFI_TIME_ASCII_ASCII_MJD_IN_ERR,
    XL_CFI_TIME_ASCII_ASCII_MJD_OUT_ERR,
    XL_CFI_TIME_ASCII_ASCII_YEAR_OUT_ERR,
    XL_CFI_TIME_ASCII_ASCII_MONTH_OUT_ERR,
    XL_CFI_TIME_ASCII_ASCII_DAY_OUT_ERR,
    XL_CFI_TIME_ASCII_ASCII_HOUR_OUT_ERR,
    XL_CFI_TIME_ASCII_ASCII_MIN_OUT_ERR,
    XL_CFI_TIME_ASCII_ASCII_SEC_OUT_ERR,
    XL_CFI_TIME_ASCII_ASCII_MICROSEC_OUT_ERR,
    XL_CFI_TIME_ASCII_ASCII_FORMAT_OUT_ERR,
    XL_CFI_TIME_ASCII_ASCII_REF_INIT_ERR,
    XL_CFI_TIME_ASCII_ASCII_REF_INIT_WARN,
    XL_CFI_TIME_ASCII_ASCII_BUL_A_FORMULA_WARN,
    XL_CFI_TIME_ASCII_ASCII_BUL_B_A_GAP_WARN,

    XL_CFI_TIME_ASCII_ASCII_VALIDITY_WARN = XL_FIRST_E,
    XL_CFI_TIME_ASCII_ASCII_EOM_FFS_COMPLIANCE_WARN // AN-679

  } XL_CFI_Time_ascii_to_ascii_err_enum;

  /* Error codes list of xl_time_transport_to_ascii */
  /* ---------------------------------------------- */
  typedef enum
  {
    XL_CFI_TIME_TRANS_ASCII_TRANS_IN_ERR = XL_ZERO_E,
    XL_CFI_TIME_TRANS_ASCII_TIME_IN_ERR,
    XL_CFI_TIME_TRANS_ASCII_DAY_ERR,
    XL_CFI_TIME_TRANS_ASCII_SEC_ERR,
    XL_CFI_TIME_TRANS_ASCII_MILLISEC_ERR,
    XL_CFI_TIME_TRANS_ASCII_MICROSEC_ERR,
    XL_CFI_TIME_TRANS_ASCII_TICK_ERR,
    XL_CFI_TIME_TRANS_ASCII_ASCII_OUT_ERR,
    XL_CFI_TIME_TRANS_ASCII_TIME_OUT_ERR,
    XL_CFI_TIME_TRANS_ASCII_ASCII_MJD_ERR,
    XL_CFI_TIME_TRANS_ASCII_ASCII_YEAR_ERR,
    XL_CFI_TIME_TRANS_ASCII_ASCII_MONTH_ERR,
    XL_CFI_TIME_TRANS_ASCII_ASCII_DAY_ERR,
    XL_CFI_TIME_TRANS_ASCII_ASCII_HOUR_ERR,
    XL_CFI_TIME_TRANS_ASCII_ASCII_MIN_ERR,
    XL_CFI_TIME_TRANS_ASCII_ASCII_SEC_ERR,
    XL_CFI_TIME_TRANS_ASCII_ASCII_MICROSEC_ERR,
    XL_CFI_TIME_TRANS_ASCII_FORMAT_OUT_ERR,
    XL_CFI_TIME_TRANS_ASCII_REF_INIT_ERR,
    XL_CFI_TIME_TRANS_ASCII_REF_INIT_WARN,
    XL_CFI_TIME_TRANS_ASCII_BUL_A_FORMULA_WARN,
    XL_CFI_TIME_TRANS_ASCII_BUL_B_A_GAP_WARN,
    XL_CFI_TIME_TRANS_ASCII_VALIDITY_WARN,
    XL_CFI_TIME_TRANS_ASCII_GPS_REF_ERR,
    XL_CFI_TIME_TRANS_ASCII_WEEK_ERR,
    XL_CFI_TIME_TRANS_ASCII_TRANSPORT_ERR

  } XL_CFI_Time_transport_to_ascii_err_enum;

  /* Error codes list of xl_time_add */
  /* ------------------------------- */
  typedef enum
  {
    XL_CFI_TIME_ADD_PROC_ERR = XL_ZERO_E,
    XL_CFI_TIME_ADD_TIME_ERR,
    XL_CFI_TIME_ADD_DAY_IN_ERR,
    XL_CFI_TIME_ADD_DAY_OUT_ERR

  } XL_CFI_Time_add_err_enum;

  /* Error codes list of xl_time_diff */
  /* -------------------------------- */
  typedef enum
  {
    XL_CFI_TIME_DIFF_PROC_ERR = XL_ZERO_E,
    XL_CFI_TIME_DIFF_TIME_ERR,
    XL_CFI_TIME_DIFF_DAY_IN_1_ERR,
    XL_CFI_TIME_DIFF_DAY_IN_2_ERR

  } XL_CFI_Time_diff_err_enum;

  /* Error codes list of xl_time_obt_to_time */
  /* --------------------------------------- */
  typedef enum
  {
    XL_CFI_TIME_OBT_TIME_SAT_ERR = XL_ZERO_E,
    XL_CFI_TIME_OBT_TIME_PROC_ERR,
    XL_CFI_TIME_OBT_TIME_INCONSISTENT_STRUCT_ERR,
    XL_CFI_TIME_OBT_TIME_DAY_REF_ERR,
    XL_CFI_TIME_OBT_TIME_OBT_SAT_ERR,
    XL_CFI_TIME_OBT_TIME_OBT_ERR,
    XL_CFI_TIME_OBT_TIME_CLOCK_ERR,
    XL_CFI_TIME_OBT_TIME_DAY_OUT_ERR

  } XL_CFI_Time_obt_to_time_err_enum;

  /* Error codes list of xl_time_time_to_obt */
  /* --------------------------------------- */
  typedef enum
  {
    XL_CFI_TIME_TIME_OBT_SAT_ERR = XL_ZERO_E,
    XL_CFI_TIME_TIME_OBT_PROC_ERR,
    XL_CFI_TIME_TIME_OBT_INCONSISTENT_STRUCT_ERR,
    XL_CFI_TIME_TIME_OBT_DAY_IN_ERR,
    XL_CFI_TIME_TIME_OBT_DAY_REF_ERR,
    XL_CFI_TIME_TIME_OBT_OBT_SAT_ERR,
    XL_CFI_TIME_TIME_OBT_OBT_ERR,
    XL_CFI_TIME_TIME_OBT_CLOCK_ERR

  } XL_CFI_Time_time_to_obt_err_enum;

  /* Error codes list of xl_kepl_to_cart */
  /* ----------------------------------- */
  typedef enum
  {
    XL_CFI_K2C_A_ZERO_ERR = XL_ZERO_E,
    XL_CFI_K2C_E_ZERO_ERR,
    XL_CFI_K2C_E_ONE_ERR,
    XL_CFI_K2C_INTERNAL_M2O_ERR,
    XL_CFI_K2C_COMPUTATION_ERR,
    XL_CFI_K2C_INTERNAL_M2O_WARN,
    XL_CFI_K2C_NO_CONVERGED_WARN

  } XL_CFI_Kepl_to_cart_err_enum;

  /* Error codes list of xl_cart_to_kepl */
  /* ----------------------------------- */
  typedef enum
  {
    XL_CFI_C2K_MU_ZERO_ERR = XL_ZERO_E,
    XL_CFI_C2K_OR_ZERO_ERR,
    XL_CFI_C2K_OV_ZERO_ERR,
    XL_CFI_C2K_OA_UNDEFINED_ERR,
    XL_CFI_C2K_OA_ZERO_ERR,
    XL_CFI_C2K_COMPUTATION_ERR,
    XL_CFI_C2K_INTERNAL_O2M_ERR,
    XL_CFI_C2K_OI_ZERO_WARN,
    XL_CFI_C2K_OE_ZERO_WARN,
    XL_CFI_C2K_INTERNAL_O2M_WARN

  } XL_CFI_Cart_to_kepl_err_enum;

  /* Error codes list of xl_sun */
  /* -------------------------- */
  typedef enum
  {
    XL_CFI_SUN_REF_ERR = XL_ZERO_E,
    XL_CFI_SUN_DAY_ERR,
    XL_CFI_SUN_REF_INIT_ERR,
    XL_CFI_SUN_SUN_ERR,
    XL_CFI_SUN_TIME_P2P_ERR,
    XL_CFI_SUN_PROC_TO_PROC_ERR,
    XL_CFI_SUN_CHANGE_COORD_ERR

  } XL_CFI_Sun_err_enum;

  /* Error codes list of xl_moon */
  /* --------------------------- */
  typedef enum
  {
    XL_CFI_MOON_REF_ERR = XL_ZERO_E,
    XL_CFI_MOON_DAY_ERR,
    XL_CFI_MOON_REF_INIT_ERR,
    XL_CFI_MOON_MOON_ERR,
    XL_CFI_MOON_TIME_P2P_ERR,
    XL_CFI_MOON_PROC_TO_PROC_ERR,
    XL_CFI_MOON_CHANGE_COORD_ERR

  } XL_CFI_Moon_err_enum;

  /* Error codes list of xl_planet */
  /* ----------------------------- */
  typedef enum
  {
    XL_CFI_PLANET_REF_ERR = XL_ZERO_E,
    XL_CFI_PLANET_DAY_ERR,
    XL_CFI_PLANET_REF_INIT_ERR,
    XL_CFI_PLANET_PLANET_ERR,
    XL_CFI_PLANET_TIME_P2P_ERR,
    XL_CFI_PLANET_PROC_TO_PROC_ERR,
    XL_CFI_PLANET_CHANGE_COORD_ERR,
    XL_CFI_PLANET_CONV_WARN

  } XL_CFI_Planet_err_enum;

  /* Error codes list of xl_star_radec */
  /* --------------------------------- */
  typedef enum
  {
    XL_CFI_STAR_RADEC_REF_ERR = XL_ZERO_E,
    XL_CFI_STAR_RADEC_DAY_ERR,
    XL_CFI_STAR_RADEC_REF_INIT_ERR,
    XL_CFI_STAR_RADEC_STAR_ERR,
    XL_CFI_STAR_RADEC_DIRPOINT_ERR,
    XL_CFI_STAR_RADEC_TIME_P2P_ERR,
    XL_CFI_STAR_RADEC_DIRPOINT_WARN

  } XL_CFI_Star_radec_err_enum;

  /* Error codes list of xl_cart_to_geod */
  /* ----------------------------------- */
  typedef enum
  {
    XL_CFI_CART_GEOD_FRAME_ERR = XL_ZERO_E,
    XL_CFI_CART_GEOD_VECTOR_ERR,
    XL_CFI_CART_GEOD_MODE_WARN,
    XL_CFI_CART_GEOD_AMBIGUITY_WARN,
    XL_CFI_CART_GEOD_ACCURACY_WARN,
    XL_CFI_CART_GEOD_ITERATIONS_WARN,
    XL_CFI_CART_GEOD_DEFVAL_WARN,
    XL_CFI_CART_GEOD_COMPUTE_ERR

  } XL_CFI_Cart_to_geod_err_enum;

  /* Error codes list of xl_geod_to_cart */
  /* ----------------------------------- */
  typedef enum
  {
    XL_CFI_GEOD_CART_EL_GT_90_ERR = XL_ZERO_E,
    XL_CFI_GEOD_CART_MODE_WARN

  } XL_CFI_Geod_to_cart_err_enum;

  /* Error codes list of xl_geod_distance */
  /* ------------------------------------ */
  typedef enum
  {
    XL_CFI_GEOD_DIST_ALTITUDE_ERR = XL_ZERO_E,
    XL_CFI_GEOD_DIST_GEO_CAR_ERR,
    XL_CFI_GEOD_DIST_DIR_RANGE_ERR,
    XL_CFI_GEOD_DIST_DIR_POINTING_ERR,
    XL_CFI_GEOD_DIST_ANTIPODAL_POINTS_WARN,
    XL_CFI_GEOD_DIST_DIR_POINTING_WARN

  } XL_CFI_Geod_distance_err_enum;

  /* Error codes list of xl_change_cart_cs */
  /* ------------------------------------- */
  typedef enum
  {
    XL_CFI_CHANGE_CART_CS_REF_ERR = XL_ZERO_E,
    XL_CFI_CHANGE_CART_CS_DAY_ERR,
    XL_CFI_CHANGE_CART_CS_MODE_ERR,
    XL_CFI_CHANGE_CART_CS_INPUT_CS_ERR,
    XL_CFI_CHANGE_CART_CS_OUTPUT_CS_ERR,
    XL_CFI_CHANGE_CART_CS_REF_INIT_ERR,
    XL_CFI_CHANGE_CART_CS_BUL_A_FORMULA_WARN,
    XL_CFI_CHANGE_CART_CS_BUL_B_A_GAP_WARN,
    XL_CFI_CHANGE_CART_CS_VALIDITY_WARN,
    XL_CFI_CHANGE_CART_CS_TIME_COMPUTATION_ERR,
    XL_CFI_CHANGE_CART_CS_CHANGE_CS_ERR, /* AN-341 */
    XL_CFI_CHANGE_CART_CS_EOGS_NOT_INITIALISED_ERR /* AN-341 */

  } XL_CFI_Change_cart_cs_err_enum;

  /* Error codes list of xl_time_ref_init */
  /* ------------------------------------ */
  typedef enum
  {
    XL_CFI_TIME_REF_INIT_TAI_ERR = XL_ZERO_E,
    XL_CFI_TIME_REF_INIT_UTC_ERR,
    XL_CFI_TIME_REF_INIT_UT1_ERR,
    XL_CFI_TIME_REF_INIT_GPS_ERR,
    XL_CFI_TIME_REF_INIT_ORBNUM_ERR,
    XL_CFI_TIME_REF_INIT_ANXTIME_ERR,
    XL_CFI_TIME_REF_INIT_ORBDUR_ERR,
    XL_CFI_TIME_REF_INIT_COMP_ERR,
    XL_CFI_TIME_REF_INIT_MEMORY_ERR,
    XL_CFI_TIME_REF_INIT_STATUS_ERR

  } XL_CFI_Time_ref_init_err_enum;

  /* Error codes list of xl_time_ref_init_file */
  /* ----------------------------------------- */
  typedef enum
  {
    XL_CFI_TIME_REF_INIT_FILE_TIME_MODEL_ERR = XL_ZERO_E,
    XL_CFI_TIME_REF_INIT_FILE_N_FILES_ERR,
    XL_CFI_TIME_REF_INIT_FILE_FILE_NAMES_ERR,
    XL_CFI_TIME_REF_INIT_FILE_INIT_MODE_ERR,
    XL_CFI_TIME_REF_INIT_FILE_TIME_ERR,
    XL_CFI_TIME_REF_INIT_FILE_DAY_0_ERR,
    XL_CFI_TIME_REF_INIT_FILE_DAY_1_ERR,
    XL_CFI_TIME_REF_INIT_FILE_DAY_RANGE_ERR,
    XL_CFI_TIME_REF_INIT_FILE_ORB_0_ERR,
    XL_CFI_TIME_REF_INIT_FILE_ORB_1_ERR,
    XL_CFI_TIME_REF_INIT_FILE_ORB_RANGE_ERR,
    XL_CFI_TIME_REF_INIT_FILE_FILE_ERR,
    XL_CFI_TIME_REF_INIT_FILE_TABLE_ERR,
    XL_CFI_TIME_REF_INIT_FILE_TIME_OUTSIDE_RANGE_ERR,
    XL_CFI_TIME_REF_INIT_FILE_ORB_OUTSIDE_RANGE_ERR,
    XL_CFI_TIME_REF_INIT_FILE_MEMORY_ERR,
    XL_CFI_TIME_REF_INIT_FILE_READ_FILE_ERR,
    XL_CFI_TIME_REF_INIT_FILE_STATUS_ERR,
    XL_CFI_TIME_REF_INIT_FILE_DETECT_FILE_ERR,
    XL_CFI_TIME_REF_INIT_FILE_WRONG_FILE_TYPE_ERR,
    XL_CFI_TIME_REF_INIT_FILE_TIME_REF_FOR_DORIS_ERR,
    XL_CFI_TIME_REF_INIT_FILE_LOAD_FILES_ERR,
    XL_CFI_TIME_REF_INIT_FILE_ONLY_FIRST_OSF_WARN,
    XL_CFI_TIME_REF_INIT_FILE_INIT_MODE_WARN,
    XL_CFI_TIME_REF_INIT_FILE_IERS_INIT_ERR,
    XL_CFI_TIME_REF_INIT_FILE_LEAP_SEC_ERR,
    XL_CFI_TIME_REF_INIT_FILE_LOAD_IERS_LIST_ERR

  } XL_CFI_Time_ref_init_file_err_enum;

  /* Error codes list of xl_time_id_init */
  /* ----------------------------------- */
  typedef enum
  {
    XL_CFI_TIME_ID_INIT_STATUS_ERR = XL_ZERO_E,
    XL_CFI_TIME_ID_INIT_TIME_MODEL_ERR,
    XL_CFI_TIME_ID_INIT_INIT_MODE_ERR,
    XL_CFI_TIME_ID_INIT_TIME_ERR,
    XL_CFI_TIME_ID_INIT_DAY_0_ERR,
    XL_CFI_TIME_ID_INIT_DAY_1_ERR,
    XL_CFI_TIME_ID_INIT_DAY_RANGE_ERR,
    XL_CFI_TIME_ID_INIT_ORB_0_ERR,
    XL_CFI_TIME_ID_INIT_ORB_1_ERR,
    XL_CFI_TIME_ID_INIT_ORB_RANGE_ERR,
    XL_CFI_TIME_ID_INIT_NO_DATA_ERR,
    XL_CFI_TIME_ID_INIT_INCONSISTENT_FILES_ERR,
    XL_CFI_TIME_ID_INIT_MEMORY_ERR,
    XL_CFI_TIME_ID_INIT_NO_DATA_IN_RANGE_ERR,
    XL_CFI_TIME_ID_INIT_TIME_CORR_INIT_ERR,
    XL_CFI_TIME_ID_INIT_LOAD_TIME_INIT_LIST_ERR,
    XL_CFI_TIME_ID_INIT_WRONG_TIME_MODEL_ERR,
    XL_CFI_TIME_ID_INIT_WRONG_DATA_TYPE_ERR,
    XL_CFI_TIME_ID_INIT_INIT_MODE_WARN,
    XL_CFI_TIME_ID_INIT_ONLY_FIRST_OSF_WARN,
    XL_CFI_TIME_ID_INIT_INVALID_FILE_TYPE_ERR, //AN-583
    XL_CFI_TIME_ID_INIT_LOAD_IERS_LIST_ERR

  } XL_CFI_Time_id_init_err_enum;

  /* Error codes list of xl_time_close */
  /* ------------------------------------- */
  typedef enum
  {
    XL_CFI_TIME_CLOSE_WRONG_ID_ERR = XL_ZERO_E

  } XL_CFI_Time_close_err_enum;

  /* Error codes list of xl_time_get_leap_second_info */
  /* ------------------------------------------------ */
  typedef enum
  {
    XL_CFI_TIME_LEAP_SECOND_ASCII_OUT_ERR = XL_ZERO_E,
    XL_CFI_TIME_LEAP_SECOND_ADD_ERR,
    XL_CFI_TIME_LEAP_SECOND_P2A_ERR,
    XL_CFI_TIME_LEAP_SECOND_TIME_REF_INIT_WARN

  } XL_CFI_Time_get_leap_second_info_err_enum;

  /* Error codes list of xl_default_sat_init */
  /* --------------------------------------- */
  typedef enum
  {
    XL_CFI_DEFAULT_SAT_INIT_NO_FREE_SAT_ERR = XL_ZERO_E,
    XL_CFI_DEFAULT_SAT_INIT_READ_FILE_ERR,
    XL_CFI_DEFAULT_SAT_INIT_LOCK_ERR,
    XL_CFI_DEFAULT_SAT_INIT_UNLOCK_ERR
  } XL_CFI_Default_sat_init_err_enum;

  /* Error codes list of xl_run_init */
  /* ------------------------------- */
  typedef enum
  {
    XL_CFI_RUN_INIT_MAX_INIT_ERR = XL_ZERO_E,
    XL_CFI_RUN_INIT_SAT_ERR,
    XL_CFI_RUN_INIT_TIME_INIT_ERR,
    XL_CFI_RUN_INIT_MEMORY_ERR,
    XL_CFI_RUN_INIT_INCONSISTENCY_ERR,
    XL_CFI_RUN_INIT_LOCK_ERR,
    XL_CFI_RUN_INIT_UNLOCK_ERR
  } XL_CFI_Run_init_err_enum;

  /* Error codes list of xl_get_rotation_angles */
  /* ------------------------------------------ */
  typedef enum
  {
    XL_CFI_GET_ROTATION_ANGLES_NO_ORTHOGONAL_ERR = XL_ZERO_E,
    XL_CFI_GET_ROTATION_ANGLES_CS_ROTATION_ERR

  } XL_CFI_Get_rotation_angles_err_enum;

  /* Error codes list of xl_get_rotated_vectors */
  /* ------------------------------------------ */
  typedef enum
  {
    XL_CFI_GET_ROTATED_VECTORS_NO_ORTHOGONAL_ERR = XL_ZERO_E,
    XL_CFI_GET_ROTATED_VECTORS_ROTATE_CS_ERR

  } XL_CFI_Get_rotated_vectors_err_enum;

  /* Error codes list of xl_position_on_orbit */
  /* ---------------------------------------- */
  typedef enum
  {
    XL_CFI_POSITION_ON_ORBIT_ANGLE_TYPE_ERR = XL_ZERO_E,
    XL_CFI_POSITION_ON_ORBIT_CHANGE_CART_CS_ERR,
    XL_CFI_POSITION_ON_ORBIT_TRUE_LAT_ERR,
    XL_CFI_POSITION_ON_ORBIT_PARALLEL_POS_VEL_ERR,
    XL_CFI_POSITION_ON_ORBIT_EQUATORIAL_ORBIT_ERR

  } XL_CFI_Position_on_orbit_err_enum;

  /* Error codes list of xl_quaternion_to_vector */
  /* ------------------------------------------- */
  typedef enum
  {
    XL_CFI_QUATERNIONS_TO_VEC_WRONG_INPUT_ERR = XL_ZERO_E

  } XL_CFI_Quaternions_to_vectors_enum;

  /* Error codes list of xl_vector_to_quaternion */
  /* ------------------------------------------- */
  typedef enum
  {
    XL_CFI_VEC_TO_QUATERNIONS_WRONG_INPUT_ERR = XL_ZERO_E,
    XL_CFI_VEC_TO_QUATERNIONS_ORTHONORMAL_ERR

  } XL_CFI_Vectors_to_quaternions_enum;

  /* Error codes list of xl_star_catalog */
  /* ----------------------------------- */
  typedef enum
  {
    XL_CFI_STAR_CATALOG_WRONG_INPUT_CATALOG_ERR = XL_ZERO_E,
    XL_CFI_STAR_CATALOG_WRONG_OUTPUT_CATALOG_ERR,
    XL_CFI_STAR_CATALOG_STAR_RADEC_ERR,
    XL_CFI_STAR_CATALOG_FK4_TO_FK5_ERR,
    XL_CFI_STAR_CATALOG_RADEC_TO_CART_ERR,
    XL_CFI_STAR_CATALOG_CHANGE_CART_CS_ERR,
    XL_CFI_STAR_CATALOG_CART_TO_RADEC_ERR
  } XL_CFI_Star_catalog_enum;

  /* Error codes list of xl_cart_to_radec */
  /* ------------------------------------ */
  typedef enum
  {
    XL_CFI_CART_TO_RADEC_WRONG_INPUT_ERR = XL_ZERO_E,
    XL_CFI_CART_TO_RADEC_WRONG_MODE_ERR,
    XL_CFI_CART_TO_RADEC_WRONG_CS_IN_ERR,
    XL_CFI_CART_TO_RADEC_WRONG_POSITION_ERR
  } XL_CFI_Cart_to_radec_enum;

  /* Error codes list of xl_radec_to_cart */
  /* ------------------------------------ */
  typedef enum
  {
    XL_CFI_RADEC_TO_CART_WRONG_INPUT_ERR = XL_ZERO_E,
    XL_CFI_RADEC_TO_CART_WRONG_MODE_ERR,
    XL_CFI_RADEC_TO_CART_WRONG_CS_IN_ERR,
    XL_CFI_RADEC_TO_CART_PAR_ERR
  } XL_CFI_Radec_to_cart_enum;

  /* Error codes list of xl_topocentric_to_ef */
  /* ---------------------------------------- */
  typedef enum
  {
    XL_CFI_TOP_TO_EF_WRONG_MODE_FLAG_ERR = XL_ZERO_E,
    XL_CFI_TOP_TO_EF_WRONG_DERIV_FLAG_ERR,
    XL_CFI_TOP_TO_EF_CART_TO_GEOD_ERR,
    XL_CFI_TOP_TO_EF_POINTING_DIR_ERR
  } XL_CFI_Topocentric_to_ef_enum;

  /* Error codes list of xl_to_ef_topocentric */
  /* ---------------------------------------- */
  typedef enum
  {
    XL_CFI_EF_TO_TOP_WRONG_MODE_FLAG_ERR = XL_ZERO_E,
    XL_CFI_EF_TO_TOP_CART_TO_GEOD_ERR,
    XL_CFI_EF_TO_TOP_WRONG_DERIV_FLAG_ERR,
    XL_CFI_EF_TO_TOP_DIR_POINTING_ERR
  } XL_CFI_Ef_to_topocentric_enum;

  /* Error codes list of xl_euler_to_matrix */
  /* -------------------------------------- */
  /* No errors */

  /* Error codes list of xl_matrix_to_euler */
  /* -------------------------------------- */
  typedef enum
  {
    XL_CFI_MATRIX_TO_EULER_WRONG_MATRIX_ERR = XL_ZERO_E,
    XL_CFI_MATRIX_TO_EULER_ANGLES_UNDEFINED_WARN,
    XL_CFI_MATRIX_TO_EULER_ORTHONORMAL_ERR
  } XL_CFI_Matrix_to_euler_enum;

  /* Error codes list of xl_model_init */
  /* --------------------------------- */
  typedef enum
  {
    XL_CFI_MODEL_INIT_STATUS_ERR = XL_ZERO_E,
    XL_CFI_MODEL_INIT_MEMORY_ERR,
    XL_CFI_MODEL_INIT_WRONG_MODEL_ERR
  } XL_CFI_Model_init_err_enum;

  /* Error codes list of xl_model_close */
  /* ---------------------------------- */
  typedef enum
  {
    XL_CFI_MODEL_CLOSE_WRONG_ID_ERR = XL_ZERO_E
  } XL_CFI_Model_close_err_enum;

  /* Error codes list of xl_geoid_calc */
  /* --------------------------------- */
  typedef enum
  {
    XL_CFI_GEOID_CALC_WRONG_NUM_HARMONICS_ERR = XL_ZERO_E,
    XL_CFI_GEOID_CALC_WRONG_LAT_ERR,
    XL_CFI_GEOID_CALC_WRONG_LON_ERR,
    XL_CFI_GEOID_CALC_UNDU_ERR
  } XL_CFI_Geoid_calc_err_enum;

  /* AN-518 */
  /* Error codes list of xl_quaternions_interpol */
  /* --------------------------------- */
  typedef enum
  {
    XL_CFI_QUATERNIONS_INTERPOL_INPUT_TIME_OUT_WARN = XL_ZERO_E, // AN-641
    XL_CFI_QUATERNIONS_INTERPOL_WRONG_INPUT_ALGORITHM_ERR
  } XL_CFI_Quaternions_interpol_err_enum;

  // AN-673
  /* Error codes list of xl_time_cuc_to_processing */
  typedef enum
  {
    XL_CFI_TIME_CUC_TO_PROCESSING_WRONG_CONFIG_ERR = XL_ZERO_E,
    XL_CFI_TIME_CUC_TO_PROCESSING_GET_P_FIELD_DATA_ERR,
    XL_CFI_TIME_CUC_TO_PROCESSING_GET_CUC_EPOCH_ERR
  } XL_CFI_Time_cuc_to_processing_err_enum;

  // AN-673
  /* Error codes list of xl_time_processing_to_cuc */
  typedef enum
  {
    XL_CFI_TIME_PROCESSING_TO_CUC_WRONG_CONFIG_ERR = XL_ZERO_E,
    XL_CFI_TIME_PROCESSING_TO_CUC_GET_CUC_EPOCH_ERR,
    XL_CFI_TIME_PROCESSING_TO_CUC_WRONG_EPOCH_ERR,
    XL_CFI_TIME_PROCESSING_TO_CUC_COMPUTE_P_FIELD_ERR
  } XL_CFI_Time_processing_to_cuc_err_enum;

#ifdef __cplusplus
}
#endif

#endif
