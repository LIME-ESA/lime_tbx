/******************************************************************************
 *
 * Project     : EOCFI
 * Copyright   : DEIMOS Space S.L.U.
 *
 * Module      : explorer_pointing.h
 *
 * Purpose     : It is the header file of the EXPLORER_POINTING CFI to be
 *               used by the final user
 *
 * Reference   : Based on Envisat Mission CFI Software
 *
 * History:      +--------------------------------------------------------------------------------+
 *               | Version |  Date  |       Name        | Change                                  |
 *               |--------------------------------------------------------------------------------|
 *               |    1.0  |19/07/02| DEIMOS Space S.L. | First version from                      |
 *               |         |        |                   | Envisat CFI                             |
 *               +--------------------------------------------------------------------------------+
 *               |    2.0  |29/11/02| DEIMOS Space S.L. | Maintenance release                     |
 *               +--------------------------------------------------------------------------------+
 *               |    2.1  |13/05/03| DEIMOS Space S.L. | Maintenance release                     |
 *               |         |        |                   | MAC OS Version                          |
 *               +--------------------------------------------------------------------------------+
 *               |    2.2  |13/08/03| DEIMOS Space S.L. | Pre-release                             |
 *               +--------------------------------------------------------------------------------+
 *               |    2.2  |30/09/03| DEIMOS Space S.L. | Maintenance release                     |
 *               |         |        |                   | New Functions:                          |
 *               |         |        |                   | xp_target_extra_target_to_sun           |
 *               |         |        |                   | xp_target_extra_ef_target               |
 *               |         |        |                   | xp_target_extra_aux                     |
 *               |         |        |                   | xp_converter                            |
 *               +--------------------------------------------------------------------------------+
 *               |   2.2.2 |26/04/04| DEIMOS Space S.L. | Maintenance release                     |
 *               |         |        |                   | Added TERRASAR S/C                      |
 *               +--------------------------------------------------------------------------------+
 *               |    3.0  |21/07/04| DEIMOS Space S.L. | New Interfaces                          |
 *               |         |        |                   |                                         |
 *               +--------------------------------------------------------------------------------+
 *               |    3.1  |13/10/04| DEIMOS Space S.L. | Maintenance release                     |
 *               |         |        |                   | New features:                           |
 *               |         |        |                   |  + Attitude models                      |
 *               |         |        |                   |  + Multitarget functions                |
 *               +--------------------------------------------------------------------------------+
 *               |    3.2  |15/11/04| DEIMOS Space S.L. | Maintenance release                     |
 *               |         |        |                   | New features:                           |
 *               |         |        |                   |  + DEM model                            |
 *               +--------------------------------------------------------------------------------+
 *               |    3.3  |11/07/05| DEIMOS Space S.L. | Maintenance release                     |
 *               |         |        |                   | New features:                           |
 *               |         |        |                   | + xp_target_travel_time                 |
 *               |         |        |                   | + Attitude models (Cryosat YSM,         |
 *               |         |        |                   |   ADM model)                            |
 *               |         |        |                   | + New attitude files for                |
 *               |         |        |                   |   initialisation                        |
 *               |         |        |                   | + Id. Accessors                         |
 *               |         |        |                   | + ENVISAT support removed               |
 *               +--------------------------------------------------------------------------------+
 *               |    3.4  |18/11/05| DEIMOS Space S.L. | Maintenance release                     |
 *               |         |        |                   | New features:                           |
 *               |         |        |                   | + New Attitude model for ADM            |
 *               |         |        |                   | + Routine xp_dem_compute                |
 *               |         |        |                   | + Interfaces for specular               |
 *               |         |        |                   |   reflection target routines            |
 *               |         |        |                   | + New axis attitude computation         |
 *               +--------------------------------------------------------------------------------+
 *               |    3.5  |26/05/06| DEIMOS Space S.L. | Maintenance release                     |
 *               |         |        |                   | New features:                           |
 *               |         |        |                   |  + Thread safe library                  |
 *               |         |        |                   |  + C99 compliant                        |
 *               |         |        |                   |  + Optimized libray                     |
 *               |         |        |                   |  + 64-Bits library for MACOS,           |
 *               |         |        |                   |    LINUX and SOLARIS                    |
 *               |         |        |                   |  + Aberration correction for CryoSat    |
 *               |         |        |                   |    attitude determination               |
 *               +--------------------------------------------------------------------------------+
 *               |    3.6  |24/11/06| DEIMOS Space S.L. | Maintenance release                     |
 *               |         |        |                   | New features:                           |
 *               |         |        |                   |  + xp_target_extra_specular_reflection  |
 *               |         |        |                   |  + xp_target_reflected                  |
 *               +--------------------------------------------------------------------------------+
 *               |    3.7  |13/07/07| DEIMOS Space S.L. | Maintenance release                     |
 *               |         |        |                   | New features:                           |
 *               |         |        |                   |  + expcfi_check_libs                    |
 *               |         |        |                   |  + library version for Mac OS X         |
 *               |         |        |                   |    on Intel (32 and 64 bits)            |
 *               +--------------------------------------------------------------------------------+
 *               |  3.7.2  |31/07/08| DEIMOS Space S.L. | Maintenance release                     |
 *               |         |        |                   | New features:                           |
 *               |         |        |                   |  + Support for misaligment or           |
 *               |         |        |                   |    configuration matrices at attitude   |
 *               |         |        |                   |    frame level                          |
 *               |         |        |                   |  + Azimuth/elevation definitions        |
 *               +--------------------------------------------------------------------------------+
 *               |    4.0  |19/01/09| DEIMOS Space S.L. | Maintenance release                     |
 *               +--------------------------------------------------------------------------------+
 *               |    4.1  |07/05/10| DEIMOS Space S.L. | Maintenance release                     |
 *               |         |        |                   | New features:                           |
 *               |         |        |                   |  + DEM GETASSEv2 supported              |
 *               |         |        |                   |  + Sentinel-1 att model (roll steering) |
 *               +--------------------------------------------------------------------------------+
 *               |    4.2  |31/01/11| DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |        |                   | New features:                           |
 *               |         |        |                   |  + New DEM model: ACE2 9secs            |
 *               +--------------------------------------------------------------------------------+
 *               |    4.3  |06/02/12| DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |        |                   | New features:                           |
 *               |         |        |                   |  + Raytracing model in target fucntions |
 *               |         |        |                   |    determined by input atmos_id         |
 *               |         |        |                   |  + New attitude model for SENTINEL2     |
 *               |         |        |                   |    (XP_MODEL_SENTINEL2)                 |
 *               +--------------------------------------------------------------------------------+
 *               |    4.4  |05/07/12| DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |        |                   | New features:                           |
 *               |         |        |                   |  + GEO satellites:                      |
 *               |         |        |                   |    - New Yaw flip attitude              |
 *               |         |        |                   |    - xp_target_sc                       |
 *               |         |        |                   |  + xp_target_list_inter                 |
 *               |         |        |                   |  + DEM computation using memory cache   |
 *               +--------------------------------------------------------------------------------+
 *               |    4.5  |01/03/13| DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |        |                   | New features:                           |
 *               |         |        |                   |  + New functions xp_target_list_extra   |
 *               |         |        |                   |    to compute extra results for targets |
 *               |         |        |                   |    computed with xp_target_list_inter   |
 *               |         |        |                   |  + New DEM algorithm of maximum heights |
 *               +--------------------------------------------------------------------------------+
 *               |    4.6  |03/10/13| DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |        |                   | New features:                           |
 *               |         |        |                   |  + New function xp_attitude_define      |
 *               |         |        |                   |  + Internal improvements to improve     |
 *               |         |        |                   |    runtime performance in DEM           |
 *               |         |        |                   |    computations                         |
 *               +--------------------------------------------------------------------------------+
 *               |    4.7  |28/03/14| DEIMOS Space S.L.U| Maintenance release                     |
 *               +--------------------------------------------------------------------------------+
 *               |    4.8  |29/10/14| DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |        |                   | New features:                           |
 *               |         |        |                   |  + Support for Earth Fixed input in     |
 *               |         |        |                   |    satellite nominal attitude           |
 *               |         |        |                   |    initialization with harmonics.       |
 *               |         |        |                   |  + Support for DEM GETASSE v3.0         |
 *               |         |        |                   |  + Support for GDEM v2                  |
 *               +--------------------------------------------------------------------------------+
 *               |    4.9  |23/04/15| DEIMOS Space S.L.U| Maintenance release                     |
 *               +--------------------------------------------------------------------------------+
 *               |  4.9.1  |05/06/15| DEIMOS Space S.L.U| Maintenance release                     |
 *               +--------------------------------------------------------------------------------+
 *               |   4.10  |29/10/15| DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |        |                   | New features:                           |
 *               |         |        |                   |  + Support for DEM ACE2 30 secs         |
 *               |         |        |                   |  + Target functions: possibility of     |
 *               |         |        |                   |    considering light travel time in     |
 *               |         |        |                   |    target computation                   |
 *               |         |        |                   |  + Run-time improvements in target      |
 *               |         |        |                   |    functions                            |
 *               +--------------------------------------------------------------------------------+
 *               |   4.11  |15/04/16| DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |        |                   | New features:                           |
 *               |         |        |                   |  + Support for DEM ACE2 3 secs          |
 *               |         |        |                   |  + New functions xp_gen_attiude_data    |
 *               |         |        |                   |    and xp_gen_attitude_file             |
 *               +--------------------------------------------------------------------------------+
 *               |   4.12  |03/11/16| DEIMOS Space S.L.U| Maintenance release                     |
 *               +--------------------------------------------------------------------------------+
 *               |   4.13  |05/04/17| DEIMOS Space S.L.U| Maintenance release                     |
 *               +--------------------------------------------------------------------------------+
 *               |   4.14  |16/11/17| DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |        |                   | New features:                           |
 *               |         |        |                   |  + Support for MetOp-SG attitude law    |
 *               |         |        |                   |  + New function xp_free_target_id_data  |
 *               +--------------------------------------------------------------------------------+
 *               |   4.15  |20/04/18| DEIMOS Space S.L.U| Maintenance release                     |
 *               +--------------------------------------------------------------------------------+
 *               |   4.16  |09/11/18| DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |        |                   | New features:                           |
 *               |         |        |                   |  + Support for DEM ACE2 5 minutes       |
 *               +--------------------------------------------------------------------------------+
 *               |   4.17  |10/05/19| DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |        |                   | New features:                           |
 *               |         |        |                   |  + DEM configurable raster              |
 *               |         |        |                   |  + functions xp_dem_get_cell_value and  |
 *               |         |        |                   |    xp_dem_get_cell_geod                 |
 *               +--------------------------------------------------------------------------------+
 *               |   4.18  |08/11/19| DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |        |                   | New features:                           |
 *               |         |        |                   |  + Support for TanDEM-X 90m DEM         |
 *               +--------------------------------------------------------------------------------+
 *               |   4.19  |29/05/20| DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |        |                   | New features:                           |
 *               |         |        |                   |  + Support for ASTER GDEM v3            |
 *               +--------------------------------------------------------------------------------+
 *               |   4.20  |30/11/20| DEIMOS Space S.L.U| Maintenance release                     |
 *               +--------------------------------------------------------------------------------+
 *               |   4.21  |10/06/21| DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |        |                   | New features:                           |
 *               |         |        |                   |  + Optimisation of EOCFI library size   |
 *               +--------------------------------------------------------------------------------+
 *               |   4.22  |22/12/21| DEIMOS Space S.L.U| Maintenance release                     |
 *               +--------------------------------------------------------------------------------+
 *               |   4.23  |23/06/22| DEIMOS Space S.L.U| Maintenance release                     |
 *               +--------------------------------------------------------------------------------+
 *               |   4.24  |29/11/22| DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |        |                   | New features:                           |
 *               |         |        |                   | + Support storing pre-computed maximum  |
 *               |         |        |                   |   altitude per tile in Generic DEM      |
 *               |         |        |                   | + Allow the use of multiple AEM files   |
 *               |         |        |                   |   to initialize Attitude                |
 *               +--------------------------------------------------------------------------------+
 *               |   4.25  |10/05/23| DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |        |                   | New features:                           |
 *               |         |        |                   |  + Support for ANX drift in OSF         |
 *               +--------------------------------------------------------------------------------+
 *               |   4.26  |31/10/23| DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |        |                   | New features:                           |
 *               |         |        |                   |  + Support for longitude drift          |
 *               +--------------------------------------------------------------------------------+
 *               |   4.27  |07/06/24| DEIMOS Space S.L.U| Maintenance release                     |
 *               |         |        |                   | New features:                           |
 *               |         |        |                   |  + Use the NORAD catalogue number to    |
 *               |         |        |                   |    identify TLE record instead of Name  |
 *               +--------------------------------------------------------------------------------+
 *               |   4.28  |13/12/24| DEIMOS Space S.L.U| Maintenance release                     |
 *               +--------------------------------------------------------------------------------+
 *               |   4.29  |09/04/25| DEIMOS Space S.L.U| Maintenance release                     |
 *               +--------------------------------------------------------------------------------+
 *
 **************************************************************************************************/
#ifndef EXPLORER_POINTING_H
#define EXPLORER_POINTING_H

#include <explorer_lib.h>
#include <explorer_orbit.h>

/*******************************************************************************
*
* Defines
*
*******************************************************************************/

/* Length of ierr vectors */
/* ---------------------- */
#define XP_NUM_ERR_TARGET_INTER                     1
#define XP_NUM_ERR_TARGET_TRAVEL_TIME               1
#define XP_NUM_ERR_TARGET_GROUND_RANGE              1
#define XP_NUM_ERR_TARGET_INCIDENCE_ANGLE           1
#define XP_NUM_ERR_TARGET_RANGE                     1
#define XP_NUM_ERR_TARGET_RANGE_RATE                1
#define XP_NUM_ERR_TARGET_TANGENT                   1
#define XP_NUM_ERR_TARGET_ALTITUDE                  1
#define XP_NUM_ERR_TARGET_STAR                      1
#define XP_NUM_ERR_TARGET_TANGENT_SUN               1
#define XP_NUM_ERR_TARGET_TANGENT_MOON              1
#define XP_NUM_ERR_TARGET_STATION                   1
#define XP_NUM_ERR_TARGET_GENERIC                   1
#define XP_NUM_ERR_TARGET_REFLECTED                 2
#define XP_NUM_ERR_TARGET_SC                        1
#define XP_NUM_ERR_MULTI_TARGET_INTER               1
#define XP_NUM_ERR_MULTI_TARGET_TRAVEL_TIME         1
#define XP_NUM_ERR_TARGET_LIST_INTER                1 /* AN-470 */
#define XP_NUM_ERR_TARGET_EXTRA_VECTOR              1
#define XP_NUM_ERR_TARGET_EXTRA_MAIN                1
#define XP_NUM_ERR_TARGET_EXTRA_AUX                 1
#define XP_NUM_ERR_TARGET_EXTRA_TARGET_TO_SUN       1
#define XP_NUM_ERR_TARGET_EXTRA_TARGET_TO_MOON      1
#define XP_NUM_ERR_TARGET_EXTRA_EF_TARGET           1
#define XP_NUM_ERR_TARGET_EXTRA_SPEC_REFL           1
#define XP_NUM_ERR_TARGET_LIST_EXTRA_VECTOR         1 /* AN-471 */
#define XP_NUM_ERR_TARGET_LIST_EXTRA_MAIN           1 /* AN-471 */
#define XP_NUM_ERR_TARGET_LIST_EXTRA_AUX            1 /* AN-471 */
#define XP_NUM_ERR_TARGET_LIST_EXTRA_EF_TARGET      1 /* AN-471 */
#define XP_NUM_ERR_TARGET_LIST_EXTRA_SPEC_REFL      1 /* AN-471 */
#define XP_NUM_ERR_TARGET_LIST_EXTRA_TARGET_TO_SUN  1 /* AN-471 */
#define XP_NUM_ERR_TARGET_LIST_EXTRA_TARGET_TO_MOON 1 /* AN-471 */
#define XP_NUM_ERR_TARGET_CLOSE                     1

#define XP_NUM_ERR_SAT_NOM_ATT_INIT              1
#define XP_NUM_ERR_SAT_NOM_ATT_INIT_MODEL        1
#define XP_NUM_ERR_SAT_NOM_ATT_INIT_HARMONIC     1
#define XP_NUM_ERR_SAT_NOM_ATT_INIT_FILE         1
#define XP_NUM_ERR_SAT_NOM_ATT_CLOSE             1
#define XP_NUM_ERR_SAT_ATT_ANGLE_INIT            1
#define XP_NUM_ERR_SAT_ATT_MATRIX_INIT           1
#define XP_NUM_ERR_SAT_ATT_INIT_HARMONIC         1
#define XP_NUM_ERR_SAT_ATT_INIT_FILE             1
#define XP_NUM_ERR_SAT_ATT_QUAT_PLUS_MATRIX_INIT 1
#define XP_NUM_ERR_SAT_ATT_QUAT_PLUS_ANGLE_INIT  1
#define XP_NUM_ERR_SAT_ATT_CLOSE                 1
#define XP_NUM_ERR_INSTR_ATT_ANGLE_INIT          1
#define XP_NUM_ERR_INSTR_ATT_MATRIX_INIT         1
#define XP_NUM_ERR_INSTR_ATT_INIT_HARMONIC       1
#define XP_NUM_ERR_INSTR_ATT_INIT_FILE           1
#define XP_NUM_ERR_INSTR_ATT_CLOSE               1
#define XP_NUM_ERR_CHANGE_FRAME                  1
#define XP_NUM_ERR_ATTITUDE_INIT                 1
#define XP_NUM_ERR_ATTITUDE_COMPUTE              1
#define XP_NUM_ERR_ATTITUDE_USER_SET             1
#define XP_NUM_ERR_ATTITUDE_CLOSE                1

#define XP_NUM_ERR_ATMOS_INIT             1
#define XP_NUM_ERR_ATMOS_CLOSE            1
#define XP_NUM_ERR_DEM_INIT               1
#define XP_NUM_ERR_DEM_COMPUTE            1
#define XP_NUM_ERR_DEM_CLOSE              1
#define XP_NUM_ERR_DEM_GET_INFO           1
#define XP_NUM_ERR_DEM_ID_CONFIGURE       1 /* AN-487 */
#define XP_NUM_ERR_DEM_ID_CONFIGURE       1 /* AN-487 */
#define XP_NUM_ERR_DEM_GET_CELL_VALUE     1 /* AN-778 */
#define XP_NUM_ERR_DEM_GET_CELL_GEOD      1 /* AN-778 */
#define XP_NUM_ERR_RUN_INIT               1
#define XP_NUM_ERR_DEM_MAX_ALT_FILE       1 /* AN-492*/
#define XP_NUM_ERR_SET_AZ_EL_DEFINITION   1
#define XP_NUM_ERR_DEM_ALT_FROM_ELLIPSOID 1 /* AN-466 */
#define XP_NUM_ERR_ATTITUDE_DEFINE        1 /* AN-469 */
#define XP_NUM_ERR_GET_ATTITUDE_DATA      1 /* AN-597 */
#define XP_NUM_ERR_GEN_ATTITUDE_DATA      1 /* AN-596 */
#define XP_NUM_ERR_GEN_ATTITUDE_FILE      1 /* AN-596 */
#define XP_NUM_ERR_ATTITUDE_TRANSFORM     1 /* AN-526 */
/* Max length of ierr vectors */
/* -------------------------- */
#define XP_ERR_VECTOR_MAX_LENGTH 2
/* Max of the above lengths.
   This value can be safely used in every function declaration. */

#define XP_NUM_MODEL_PARAM 14

#define XP_MAX_NUM_HARMONIC 10 /* TBD */

#define XP_MAX_TARGETS 30 /* TBD */

#define XP_MAX_NUM_MULTI_TARGET 30 /* TBD */

#define XP_NUM_SOURCE_PARAM 6

/* Length of the output vectors */
#define XP_SIZE_TARGET_RESULT_VECTOR    10 // AN-553
#define XP_SIZE_TARGET_RESULT_MAIN      12
#define XP_SIZE_TARGET_RESULT_AUX       12
#define XP_SIZE_SUN_RESULT              4
#define XP_SIZE_MOON_RESULT             4
#define XP_SIZE_EF_TARGET_RESULT        8
#define XP_SIZE_TARGET_RESULT_SPEC_REFL 7

/* Synonyms DEM models */
#define XP_DEM_ACE_MODEL XD_DEM_ACE_MODEL

/*******************************************************************************
*
* CFI Functions - prototype declaration
*
*******************************************************************************/
#ifdef __cplusplus
extern "C"
{
#endif

  /*******************************************************************************
*
* Data structures
*
*******************************************************************************/

  /* Sat_nom_trans Id. Structure */
  typedef struct
  {
    void* ee_id;
  } xp_sat_nom_trans_id;

  /* Sat_trans Id. Structure */
  typedef struct
  {
    void* ee_id;
  } xp_sat_trans_id;

  /* Instr_trans Id. Structure */
  typedef struct
  {
    void* ee_id;
  } xp_instr_trans_id;

  /* Attitude Id. Structure */
  typedef struct
  {
    void* ee_id;
  } xp_attitude_id;

  /* Atmos Id. Structure */
  typedef struct
  {
    void* ee_id;
  } xp_atmos_id;

  /* DEM Id. Structure */
  typedef struct
  {
    void* ee_id;
  } xp_dem_id;

  /* Target Id. Structure */
  typedef struct
  {
    void* ee_id;
  } xp_target_id;

  /* Sat_nom_trans Id. Accessors */

  typedef struct
  {
    long model_enum;
    double model_param[XP_NUM_MODEL_PARAM];
  } xp_param_model_str;

  typedef struct
  {
    long num_terms[3];
    long harmonic_type_pitch[XP_MAX_NUM_HARMONIC];
    long harmonic_type_roll[XP_MAX_NUM_HARMONIC];
    long harmonic_type_yaw[XP_MAX_NUM_HARMONIC];
    double harmonic_coef_pitch[XP_MAX_NUM_HARMONIC];
    double harmonic_coef_roll[XP_MAX_NUM_HARMONIC];
    double harmonic_coef_yaw[XP_MAX_NUM_HARMONIC];

  } xp_harmonic_data;

  typedef struct
  {
    long angle_type;
    xp_harmonic_data harmonic;
  } xp_harmonic_model_str;

  typedef struct
  {
    long time_ref; /* time ref */
    double time; /* time */
    double quaternion[4]; /* Quaternions */
    double angles[3]; /* Angles */
  } xp_att_data_rec;

  typedef struct
  {
    long file_model; /* target reference frame */
    double val_time0;
    double val_time1;
    long data_type; /* data type: quaternions or angles acording to XD_Attitude_data_type_enum */
    long inertial_frame; /* initial reference frame: Inertial reference frame */
    long lines; /* number of records in the attitude lists */
    double max_gap;
    xp_att_data_rec* att_data; /* array with the angle/quaternion records */
  } xp_sat_nom_att_file_model_str;

  /* Sat_trans Id. Accessors */
  typedef struct
  {
    double pitch;
    double roll;
    double yaw;
  } xp_angle_model_str;

  typedef struct
  {
    double att_matrix[3][3];
  } xp_matrix_model_str;

  typedef struct
  {
    long inertial_frame; /* Initial reference frame: Inertial reference frame */
    long num_quat; /* Number of quaternions */
    xd_att_rec* quat; /* List of quaternions. */
    double rot_matrix[3][3]; /* Rotation matrix */
  } xp_quat_plus_matrix_model_str;

  typedef struct
  {
    long inertial_frame; /* initial reference frame: Inertial reference frame */
    long num_quat; /* Number of quaternions */
    xd_att_rec* quat; /* List of quaternions. */
    double angles[3]; /* Rotation angles */
  } xp_quat_plus_angle_model_str;

  typedef xd_star_tracker xp_star_tracker;

  typedef struct
  {
    long star_tr_id; /* Star tracker Id (1,2 or 3) */
    long aberr_correction;
    double str_att_rot[3][3];
  } xp_star_tracker_aux;

  typedef struct
  {
    long file_model;
    double val_time0;
    double val_time1;
    long data_type; /* data type: quaternions or angles acording to XD_Attitude_data_type_enum */
    long inertial_frame; /* initial reference frame: Inertial reference frame */
    long lines; /* number of records in the attitude lists */
    double max_gap;
    xp_att_data_rec* att_data; /* array with the angle/quaternion records */
    xp_star_tracker_aux aux_data;
    xp_star_tracker* tm_data;

  } xp_sat_att_file_model_str;

  /* Instr_trans Id. Accessors */

  typedef struct
  {
    long file_model; /* target reference frame */
    double val_time0;
    double val_time1;
    long data_type; /* data type: quaternions or angles acording to XD_Attitude_data_type_enum */
    long inertial_frame; /* initial reference frame: Inertial reference frame */
    long lines; /* number of records in the attitude lists */
    double max_gap;
    xp_att_data_rec* att_data; /* array with the angle/quaternion records */
  } xp_instr_att_file_model_str;

  /* Attitude Id. Accessors */
  typedef struct
  {
    long model;
    long time_ref;
    double time;
    xl_cord sat_vector;
    long source_frame; /* according to xl_cs_enum */
    long target_frame; /* according to XP_Attitude_frame_enum */
    xl_cs_tra sat_mat;
    double offset[3];
    xl_cs_tra attitude_EF; /* position and orientation of instrument frame in EF */
  } xp_attitude_id_data;

  /* Atmospheric Id. Accessors */
  typedef struct
  {
    double atm_max_alt_std;
    double atm_max_alt_user;
  } xp_atmos_id_data;

  /* DEM Id. Accessors */
  typedef xd_dem_config_file xp_dem_id_data;

  /* DEM data */
  typedef struct
  {
    long dem_model; /* -> according to XD_Dem_model_enum */
    long data_source; /* -> according to XD_Dem_data_source_enum */
  } xp_dem_info;

  /* Target Accessors */
  typedef struct
  {
    long time_ref;
    double time;
    xl_cord sat_vector;
    long iray;
    double freq;
    long deriv;
  } xp_generic_data;

  typedef struct
  {
    long target_number; /* AN-470 */
    xl_cord tar_vector;
    xl_par_der z_tan;
    xl_par_der range;
    xl_par_der time;
    xl_cord tar_sat_vector;
    xl_cord sat_tar_vector;
  } xp_target_data;

  typedef struct
  {
    long num_target;
    xp_target_data* target;
  } xp_target_str;

  typedef struct
  {
    xp_generic_data generic_data;
    long earth_crossed;
    long atm_crossed;
    xp_target_str user;
    xp_target_str los;
    xp_target_str earth;
    xl_cord exit_atm_vector;
  } xp_target_id_data;

  /* Start AN-470 */
  typedef struct
  {
    double azimuth;
    double elevation;
    double azimuth_rate;
    double elevation_rate;
  } xp_azimuth_elevation;

  typedef struct
  {
    long num_rec;
    xp_azimuth_elevation* az_el_list;
  } xp_azimuth_elevation_list;

  typedef struct
  {
    double min_azimuth;
    double max_azimuth;
    double step_azimuth;
    double min_elevation;
    double max_elevation;
    double step_elevation;
  } xp_azimuth_elevation_grid;

  typedef struct
  {
    double azimuth; /* One azimuth needs to be verified for the whole strip */
    double min_elevation;
    double max_elevation;
    double step_elevation;
  } xp_azimuth_elevation_strip;

  typedef union
  {
    xp_azimuth_elevation_list azimuth_elevation_list;
    xp_azimuth_elevation_strip azimuth_elevation_strip;
    xp_azimuth_elevation_grid azimuth_elevation_grid;
  } xp_azimuth_elevation_input_union;

  typedef struct
  {
    long type; /* XP_Az_el_type_enum */
    xp_azimuth_elevation_input_union azimuth_elevation_input_union;
    double signal_frequency;
  } xp_instrument_data;

  typedef struct
  {
    long num_user_target;
    long* num_los_target;
  } xp_target_output;
  /* END AN-470 */

  /* AN-487 */
  typedef struct
  {
    double lon_min;
    double lon_max;
    double lat_min;
    double lat_max;
  } xp_latlon_area;

  typedef struct
  {
    long command; /* XP_Dem_config_enum */
    long max_cache_size;
    xp_latlon_area area;
  } xp_dem_id_config;
  /* END AN-487 */

  /* AN-471 */
  typedef struct
  {
    long input_type; /* list � grid � stripe */
    double azimuth, elevation; /* azimuth and elevation used as input to
                               compute the target */
    long azimuth_idx, elevation_idx; /* indexexes in azimuth/elevation grid */
  } xp_target_list_input_info;

  typedef union
  {
    xp_target_list_input_info target_list_input_info;
  } xp_target_input_info_union;

  typedef struct
  {
    long target_function; /* it defines the function used to compute the
                           target, only one so far, target_list_inter */
    xp_target_input_info_union target_input_info_union;
  } xp_target_input_info;

  typedef struct
  {
    long status; /* computation status */
    xp_target_input_info target_input_info;
    double vector_results[XP_SIZE_TARGET_RESULT_VECTOR];
    double vector_results_rate[XP_SIZE_TARGET_RESULT_VECTOR];
    double vector_results_rate_rate[XP_SIZE_TARGET_RESULT_VECTOR];
  } xp_target_extra_vector_results;

  typedef struct
  {
    long num_rec;
    xp_target_extra_vector_results* extra_vector_results;
  } xp_target_extra_vector_results_list;

  typedef struct
  {
    long status; /* computation status */
    xp_target_input_info target_input_info;
    double main_results[XP_SIZE_TARGET_RESULT_MAIN];
    double main_results_rate[XP_SIZE_TARGET_RESULT_MAIN];
    double main_results_rate_rate[XP_SIZE_TARGET_RESULT_MAIN];
  } xp_target_extra_main_results;

  typedef struct
  {
    long num_rec;
    xp_target_extra_main_results* extra_main_results;
  } xp_target_extra_main_results_list;

  typedef struct
  {
    long status; /* computation status */
    xp_target_input_info target_input_info;
    double aux_results[XP_SIZE_TARGET_RESULT_AUX];
    double aux_results_rate[XP_SIZE_TARGET_RESULT_AUX];
    double aux_results_rate_rate[XP_SIZE_TARGET_RESULT_AUX];
  } xp_target_extra_aux_results;

  typedef struct
  {
    long num_rec;
    xp_target_extra_aux_results* extra_aux_results;
  } xp_target_extra_aux_results_list;

  typedef struct
  {
    long status; /* computation status */
    xp_target_input_info target_input_info;
    double ef_target_results_rate[XP_SIZE_EF_TARGET_RESULT];
    double ef_target_results_rate_rate[XP_SIZE_EF_TARGET_RESULT];
  } xp_target_extra_ef_target_results;

  typedef struct
  {
    long num_rec;
    xp_target_extra_ef_target_results* extra_ef_target_results;
  } xp_target_extra_ef_target_results_list;

  typedef struct
  {
    long status; /* computation status */
    xp_target_input_info target_input_info;
    double spec_reflec_results[XP_SIZE_TARGET_RESULT_SPEC_REFL];
    double spec_reflec_results_rate[XP_SIZE_TARGET_RESULT_SPEC_REFL];
    double spec_reflec_results_rate_rate[XP_SIZE_TARGET_RESULT_SPEC_REFL];
  } xp_target_extra_spec_reflec_target_results;

  typedef struct
  {
    long num_rec;
    xp_target_extra_spec_reflec_target_results* extra_spec_reflec_target_results;
  } xp_target_extra_spec_reflec_target_results_list;

  typedef struct
  {
    long status; /* computation status */
    xp_target_input_info target_input_info;
    double moon_results[XP_SIZE_MOON_RESULT];
    double moon_results_rate[XP_SIZE_MOON_RESULT];
    double moon_results_rate_rate[XP_SIZE_MOON_RESULT];
  } xp_target_extra_moon_target_results;

  typedef struct
  {
    long num_rec;
    xp_target_extra_moon_target_results* extra_moon_target_results;
  } xp_target_extra_moon_target_results_list;

  typedef struct
  {
    long status; /* computation status */
    xp_target_input_info target_input_info;
    double sun_results[XP_SIZE_SUN_RESULT];
    double sun_results_rate[XP_SIZE_SUN_RESULT];
    double sun_results_rate_rate[XP_SIZE_SUN_RESULT];
  } xp_target_extra_sun_target_results;

  typedef struct
  {
    long num_rec;
    xp_target_extra_sun_target_results* extra_sun_target_results;
  } xp_target_extra_sun_target_results_list;
  /* END AN-471 */

  /* AN-466 */
  typedef struct
  {
    long set_type; /* XP_Dem_set_enum */
    double lon_min; /* Longitude/latitude intervals are only used for XP_DEM_SET */
    double lon_max;
    double lat_min;
    double lat_max;
    long verbose; /* If 0, no message will be printed. If > 0, a message will be printed every verbose points */
  } xp_gen_dem_alt_from_ellipsoid_inputs;

  /* AN-468 */
  typedef struct
  {
    long type; /* XP_NONE_ATTITUDE, XP_SAT_NOMINAL_ATT, XP_SAT_ATT, XP_INSTR_ATT */
    xp_sat_nom_trans_id sat_nom_trans_id;
    xp_sat_trans_id sat_trans_id;
    xp_instr_trans_id instr_trans_id;
  } xp_attitude_def;
  /* END AN-468 */

  /* AN-526 */
  typedef struct
  {
    long ref_frame;
    xl_time_id time_id;
    xl_model_id model_id;
  } xp_transform_cfg;
  /* END AN-526 */

  /*******************************************************************************
*
* Main CFI Functions - prototype declaration
*
*******************************************************************************/

  /* Satellite Nominal Attitude Transformation initialisation functions */
  /* ------------------------------------------------------------------ */

  long xp_sat_nominal_att_init(long* aocs_mode,
                               /* output */
                               xp_sat_nom_trans_id* sat_nom_trans_id,
                               long ierr[XP_NUM_ERR_SAT_NOM_ATT_INIT]);

  long xp_sat_nominal_att_init_model(long* model_enum,
                                     double model_param[XP_NUM_MODEL_PARAM],
                                     /* output */
                                     xp_sat_nom_trans_id* sat_nom_trans_id,
                                     long ierr[XP_NUM_ERR_SAT_NOM_ATT_INIT_MODEL]);

  long xp_sat_nominal_att_init_harmonic(long* angle_type, long* num_terms, long harmonic_type_pitch[XP_MAX_NUM_HARMONIC], long harmonic_type_roll[XP_MAX_NUM_HARMONIC], long harmonic_type_yaw[XP_MAX_NUM_HARMONIC], double harmonic_coef_pitch[XP_MAX_NUM_HARMONIC], double harmonic_coef_roll[XP_MAX_NUM_HARMONIC], double harmonic_coef_yaw[XP_MAX_NUM_HARMONIC],
                                        /* output */
                                        xp_sat_nom_trans_id* sat_nom_trans_id,
                                        long ierr[XP_NUM_ERR_SAT_NOM_ATT_INIT_HARMONIC]);

  long xp_sat_nominal_att_init_file(xl_time_id* time_id,
                                    long* n_files,
                                    char** attitude_file,
                                    long* time_init_mode,
                                    long* time_ref,
                                    double* time0,
                                    double* time1,
                                    /* output */
                                    double* val_time0,
                                    double* val_time1,
                                    xp_sat_nom_trans_id* sat_nom_trans_id,
                                    long ierr[XP_NUM_ERR_SAT_NOM_ATT_INIT_FILE]);

  long xp_sat_nominal_att_close(xp_sat_nom_trans_id* sat_nom_trans_id,
                                /* output */
                                long ierr[XP_NUM_ERR_SAT_NOM_ATT_CLOSE]);

  long xp_sat_nominal_att_init_status(xp_sat_nom_trans_id* sat_nom_trans_id);

  long xp_sat_nominal_att_get_sat_id(xp_sat_nom_trans_id* sat_nom_trans_id);

  long xp_sat_nominal_att_get_mode(xp_sat_nom_trans_id* sat_nom_trans_id);

  long xp_sat_nominal_att_get_aocs(xp_sat_nom_trans_id* sat_nom_trans_id,
                                   long* aocs);

  long xp_sat_nominal_att_set_aocs(xp_sat_nom_trans_id* sat_nom_trans_id,
                                   long* aocs);

  long xp_sat_nominal_att_get_param(xp_sat_nom_trans_id* sat_nom_trans_id,
                                    xp_param_model_str* data);

  long xp_sat_nominal_att_set_param(xp_sat_nom_trans_id* sat_nom_trans_id,
                                    xp_param_model_str* data);

  long xp_sat_nominal_att_get_harmonic(xp_sat_nom_trans_id* sat_nom_trans_id,
                                       xp_harmonic_model_str* data);

  long xp_sat_nominal_att_set_harmonic(xp_sat_nom_trans_id* sat_nom_trans_id,
                                       xp_harmonic_model_str* data);

  long xp_sat_nominal_att_get_file(xp_sat_nom_trans_id* sat_nom_trans_id,
                                   xp_sat_nom_att_file_model_str* data);

  long xp_sat_nominal_att_set_file(xp_sat_nom_trans_id* sat_nom_trans_id,
                                   xp_sat_nom_att_file_model_str* data);

  /* Satellite Attitude Transformation initialisation functions */
  /* ---------------------------------------------------------- */

  long xp_sat_att_angle_init(double ang[3],
                             /* output */
                             xp_sat_trans_id* sat_trans_id,
                             long ierr[XP_NUM_ERR_SAT_ATT_ANGLE_INIT]);

  long xp_sat_att_matrix_init(double att_matrix[3][3],
                              /* output */
                              xp_sat_trans_id* sat_trans_id,
                              long ierr[XP_NUM_ERR_SAT_ATT_MATRIX_INIT]);

  long xp_sat_att_init_harmonic(long* angle_type, long* num_terms, long harmonic_type_pitch[XP_MAX_NUM_HARMONIC], long harmonic_type_roll[XP_MAX_NUM_HARMONIC], long harmonic_type_yaw[XP_MAX_NUM_HARMONIC], double harmonic_coef_pitch[XP_MAX_NUM_HARMONIC], double harmonic_coef_roll[XP_MAX_NUM_HARMONIC], double harmonic_coef_yaw[XP_MAX_NUM_HARMONIC],
                                /* output */
                                xp_sat_trans_id* sat_trans_id,
                                long ierr[XP_NUM_ERR_SAT_ATT_INIT_HARMONIC]);

  long xp_sat_att_init_file(xl_time_id* time_id,
                            long* n_files,
                            char** attitude_file,
                            char* auxiliary_file,
                            long* time_init_mode,
                            long* time_ref,
                            double* time0,
                            double* time1,
                            /* output */
                            double* val_time0,
                            double* val_time1,
                            xp_sat_trans_id* sat_trans_id,
                            long ierr[XP_NUM_ERR_SAT_ATT_INIT_FILE]);

  long xp_sat_att_quat_plus_matrix_init(long* inertial_frame,
                                        long* num_rec,
                                        xd_att_rec* quaternions,
                                        double** matrix,
                                        /* outputs  */
                                        xp_sat_trans_id* sat_trans_id,
                                        long ierr[XP_NUM_ERR_SAT_ATT_QUAT_PLUS_MATRIX_INIT]);

  long xp_sat_att_quat_plus_angle_init(long* inertial_frame,
                                       long* num_rec,
                                       xd_att_rec* quaternions,
                                       double* angles,
                                       /* outputs  */
                                       xp_sat_trans_id* sat_trans_id,
                                       long ierr[XP_NUM_ERR_SAT_ATT_QUAT_PLUS_ANGLE_INIT]);

  long xp_sat_att_close(xp_sat_trans_id* sat_trans_id,
                        /* output */
                        long ierr[XP_NUM_ERR_SAT_NOM_ATT_CLOSE]);

  long xp_sat_att_init_status(xp_sat_trans_id* sat_trans_id);

  long xp_sat_att_get_sat_id(xp_sat_trans_id* sat_trans_id);

  long xp_sat_att_get_mode(xp_sat_trans_id* sat_trans_id);

  long xp_sat_att_get_angles(xp_sat_trans_id* sat_trans_id,
                             xp_angle_model_str* data);

  long xp_sat_att_set_angles(xp_sat_trans_id* sat_trans_id,
                             xp_angle_model_str* data);

  long xp_sat_att_get_matrix(xp_sat_trans_id* sat_trans_id,
                             xp_matrix_model_str* data);

  long xp_sat_att_set_matrix(xp_sat_trans_id* sat_trans_id,
                             xp_matrix_model_str* data);

  long xp_sat_att_get_harmonic(xp_sat_trans_id* sat_trans_id,
                               xp_harmonic_model_str* data);

  long xp_sat_att_set_harmonic(xp_sat_trans_id* sat_trans_id,
                               xp_harmonic_model_str* data);

  long xp_sat_att_get_file(xp_sat_trans_id* sat_trans_id,
                           xp_sat_att_file_model_str* data);

  long xp_sat_att_set_file(xp_sat_trans_id* sat_trans_id,
                           xp_sat_att_file_model_str* data);

  long xp_sat_att_get_quat_plus_matrix(xp_sat_trans_id* sat_trans_id,
                                       xp_quat_plus_matrix_model_str* data);

  long xp_sat_att_set_quat_plus_matrix(xp_sat_trans_id* sat_trans_id,
                                       xp_quat_plus_matrix_model_str* data);

  long xp_sat_att_get_quat_plus_angle(xp_sat_trans_id* sat_trans_id,
                                      xp_quat_plus_angle_model_str* data);

  long xp_sat_att_set_quat_plus_angle(xp_sat_trans_id* sat_trans_id,
                                      xp_quat_plus_angle_model_str* data);

  /* Satellite Instrument Transformation initialisation functions */
  /* ------------------------------------------------------------ */

  long xp_instr_att_angle_init(double ang[3], double offset[3],
                               /* output */
                               xp_instr_trans_id* instr_trans_id,
                               long ierr[XP_NUM_ERR_INSTR_ATT_ANGLE_INIT]);

  long xp_instr_att_matrix_init(double att_matrix[3][3], double offset[3],
                                /* output */
                                xp_instr_trans_id* instr_trans_id,
                                long ierr[XP_NUM_ERR_INSTR_ATT_MATRIX_INIT]);

  long xp_instr_att_init_harmonic(long* angle_type, long* num_terms, long harmonic_type_pitch[XP_MAX_NUM_HARMONIC], long harmonic_type_roll[XP_MAX_NUM_HARMONIC], long harmonic_type_yaw[XP_MAX_NUM_HARMONIC], double harmonic_coef_pitch[XP_MAX_NUM_HARMONIC], double harmonic_coef_roll[XP_MAX_NUM_HARMONIC], double harmonic_coef_yaw[XP_MAX_NUM_HARMONIC], double offset[3],
                                  /* output */
                                  xp_instr_trans_id* instr_trans_id,
                                  long ierr[XP_NUM_ERR_INSTR_ATT_INIT_HARMONIC]);

  long xp_instr_att_init_file(xl_time_id* time_id,
                              long* n_files,
                              char** instrument_file,
                              long* time_init_mode,
                              long* time_ref,
                              double* time0,
                              double* time1,
                              /* output */
                              double* val_time0,
                              double* val_time1,
                              xp_instr_trans_id* instr_trans_id,
                              long ierr[XP_NUM_ERR_INSTR_ATT_INIT_FILE]);

  long xp_instr_att_close(xp_instr_trans_id* instr_trans_id,
                          /* output */
                          long ierr[XP_NUM_ERR_INSTR_ATT_CLOSE]);

  long xp_instr_att_init_status(xp_instr_trans_id* instr_trans_id);

  long xp_instr_att_get_sat_id(xp_instr_trans_id* instr_trans_id);

  long xp_instr_att_get_mode(xp_instr_trans_id* instr_trans_id);

  long xp_instr_att_get_angles(xp_instr_trans_id* instr_trans_id,
                               xp_angle_model_str* data);

  long xp_instr_att_set_angles(xp_instr_trans_id* instr_trans_id,
                               xp_angle_model_str* data);

  long xp_instr_att_get_matrix(xp_instr_trans_id* instr_trans_id,
                               xp_matrix_model_str* data);

  long xp_instr_att_set_matrix(xp_instr_trans_id* instr_trans_id,
                               xp_matrix_model_str* data);

  long xp_instr_att_get_harmonic(xp_instr_trans_id* instr_trans_id,
                                 xp_harmonic_model_str* data);

  long xp_instr_att_set_harmonic(xp_instr_trans_id* instr_trans_id,
                                 xp_harmonic_model_str* data);

  long xp_instr_att_get_file(xp_instr_trans_id* instr_trans_id,
                             xp_instr_att_file_model_str* data);

  long xp_instr_att_set_file(xp_instr_trans_id* instr_trans_id,
                             xp_instr_att_file_model_str* data);

  long xp_instr_att_get_offset(xp_instr_trans_id* instr_trans_id,
                               double offset[3]);

  long xp_instr_att_set_offset(xp_instr_trans_id* instr_trans_id,
                               double offset[3]);

  /* Function to set the azimuth-elevation definition */
  /* ------------------------------------------------ */
  long xp_set_az_el_definition(void* att_frame_id,
                               xl_az_el_definition* azel_def,
                               /* output */
                               long ierr[XP_NUM_ERR_SET_AZ_EL_DEFINITION]);

  /* Frame calculation functions */
  /* --------------------------- */

  long xp_change_frame(long* sat_id,
                       xl_model_id* model_id,
                       xl_time_id* time_id,
                       xp_sat_nom_trans_id* sat_nom_trans_id,
                       xp_sat_trans_id* sat_trans_id,
                       xp_instr_trans_id* instr_trans_id,
                       long* mode_flag,
                       long* frame_flag_in,
                       long* frame_id_in,
                       long* frame_flag_out,
                       long* frame_id_out,
                       long* time_ref,
                       double* time,
                       double* pos,
                       double* vel,
                       double* acc,
                       long* deriv,
                       double* vec_in,
                       double* vec_rate_in,
                       double* vec_rate_rate_in,
                       /* output */
                       double* vec_out,
                       double* vec_rate_out,
                       double* vec_rate_rate_out,
                       long ierr[XP_NUM_ERR_CHANGE_FRAME]);

  long xp_change_frame_run(long* run_id,
                           long* mode_flag,
                           long* frame_flag_in,
                           long* frame_id_in,
                           long* frame_flag_out,
                           long* frame_id_out,
                           long* time_ref,
                           double* time,
                           double* pos,
                           double* vel,
                           double* acc,
                           long* deriv,
                           double* vec_in,
                           double* vec_rate_in,
                           double* vec_rate_rate_in,
                           /* output */
                           double* vec_out,
                           double* vec_rate_out,
                           double* vec_rate_rate_out,
                           long* ierr);

  /* Attitude Calculation functions */
  /* ------------------------------ */

  long xp_attitude_init(/* output */
                        xp_attitude_id* attitude_id,
                        long ierr[XP_NUM_ERR_ATTITUDE_INIT]);

  long xp_attitude_compute(xl_model_id* model_id,
                           xl_time_id* time_id,
                           xp_sat_nom_trans_id* sat_nom_trans_id,
                           xp_sat_trans_id* sat_trans_id,
                           xp_instr_trans_id* instr_trans_id,
                           xp_attitude_id* attitude_id, /* input / output */
                           long* time_ref,
                           double* time,
                           double* pos,
                           double* vel,
                           double* acc,
                           long* target_frame,
                           /* output */
                           long ierr[XP_NUM_ERR_ATTITUDE_COMPUTE]);

  long xp_attitude_compute_run(long* run_id,
                               xp_attitude_id* attitude_id, /* input / output */
                               long* time_ref,
                               double* time,
                               double* pos,
                               double* vel,
                               double* acc,
                               long* target_frame,
                               /* output */
                               long ierr[XP_NUM_ERR_ATTITUDE_COMPUTE]);

  long xp_attitude_user_set(xl_model_id* model_id,
                            xl_time_id* time_id,
                            xp_attitude_id* attitude_id, /* input / output */
                            long* time_ref,
                            double* time,
                            double* pos,
                            double* vel,
                            double* acc,
                            long* target_frame,
                            double matrix[3][3],
                            double matrix_rate[3][3],
                            double matrix_rate_rate[3][3],
                            double offset[3],
                            /* output */
                            long ierr[XP_NUM_ERR_ATTITUDE_USER_SET]);

  long xp_attitude_user_set_run(long* run_id,
                                xp_attitude_id* attitude_id, /* input / output */
                                long* time_ref,
                                double* time,
                                double* pos,
                                double* vel,
                                double* acc,
                                long* target_frame,
                                double matrix[3][3],
                                double matrix_rate[3][3],
                                double matrix_rate_rate[3][3],
                                double offset[3],
                                /* output */
                                long ierr[XP_NUM_ERR_ATTITUDE_USER_SET]);

  long xp_attitude_close(xp_attitude_id* attitude_id, long ierr[XP_NUM_ERR_ATTITUDE_CLOSE]);

  long xp_attitude_init_status(xp_attitude_id* attitude_id);

  long xp_attitude_get_sat_id(xp_attitude_id* attitude_id);

  long xp_attitude_get_mode(xp_attitude_id* attitude_id);

  long xp_attitude_get_id_data(xp_attitude_id* attitude_id,
                               xp_attitude_id_data* data);

  long xp_attitude_set_id_data(xp_attitude_id* attitude_id,
                               xp_attitude_id_data* data);

  xl_model_id xp_attitude_get_model_id(xp_attitude_id* attitude_id);

  xl_time_id xp_attitude_get_time_id(xp_attitude_id* attitude_id);

  xp_sat_nom_trans_id xp_attitude_get_sat_nom_trans_id(xp_attitude_id* attitude_id);

  xp_sat_trans_id xp_attitude_get_sat_trans_id(xp_attitude_id* attitude_id);

  xp_instr_trans_id xp_attitude_get_instr_id(xp_attitude_id* attitude_id);

  /*ANR-469*/
  long xp_attitude_define(xd_attitude_definition_data* data,
                          /* outputs */
                          xp_sat_nom_trans_id* sat_nom_trans_id,
                          xp_sat_trans_id* sat_trans_id,
                          xp_instr_trans_id* instr_trans_id,
                          long ierr[XP_NUM_ERR_ATTITUDE_DEFINE]);

  /* ANR-597 */
  long xp_get_attitude_data(xp_attitude_id* attitude_id,
                            long* data_type,
                            long* source_ref_type,
                            long* source_ref,
                            /* output */
                            xd_att_rec* att_rec,
                            long ierr[XP_NUM_ERR_GET_ATTITUDE_DATA]);

  /* ANR-596 */
  long xp_gen_attitude_data(xo_orbit_id* orbit_id,
                            xp_attitude_def* att_def,
                            xo_time_interval* time_interval,
                            double* time_step,
                            long* data_type,
                            long* source_ref_type,
                            long* source_ref,
                            /* output */
                            xd_att_file* att_file,
                            long ierr[XP_NUM_ERR_GEN_ATTITUDE_DATA]);

  long xp_gen_attitude_data_run(long* run_id,
                                long* type,
                                xo_time_interval* time_interval,
                                double* time_step,
                                long* data_type,
                                long* source_ref_type,
                                long* source_ref,
                                /* output */
                                xd_att_file* att_file,
                                long ierr[XP_NUM_ERR_GEN_ATTITUDE_DATA]);

  long xp_gen_attitude_file(xo_orbit_id* orbit_id,
                            xp_attitude_def* att_def,
                            xo_time_interval* time_interval,
                            double* time_step,
                            long* data_type,
                            long* source_ref_type,
                            long* source_ref,
                            char* output_dir,
                            char* file_class,
                            long* version_number,
                            char* fh_system,
                            /* input/output */
                            char filename[XD_MAX_STR],
                            /* output */
                            long ierr[XP_NUM_ERR_GEN_ATTITUDE_FILE]);

  long xp_gen_attitude_file_run(long* run_id,
                                long* type,
                                xo_time_interval* time_interval,
                                double* time_step,
                                long* data_type,
                                long* source_ref_type,
                                long* source_ref,
                                char* output_dir,
                                char* file_class,
                                long* version_number,
                                char* fh_system,
                                /* input/output */
                                char filename[XD_MAX_STR],
                                /* output */
                                long ierr[XP_NUM_ERR_GEN_ATTITUDE_FILE]);

  /* The following functions initialise the atmospheric models and DEM */
  /* ----------------------------------------------------------------- */

  long xp_atmos_init(long* mode, long* model, char* atmos_file,
                     /* output */
                     xp_atmos_id* atmos_id,
                     long ierr[XP_NUM_ERR_ATMOS_INIT]);

  long xp_atmos_close(xp_atmos_id* atmos_id,
                      /* output */
                      long ierr[XP_NUM_ERR_ATMOS_CLOSE]);

  long xp_atmos_init_status(xp_atmos_id* atmos_id);

  long xp_atmos_get_sat_id(xp_atmos_id* atmos_id);

  long xp_atmos_get_mode(xp_atmos_id* atmos_id);

  long xp_atmos_get_id_data(xp_atmos_id* atmos_id,
                            xp_atmos_id_data* data);

  long xp_dem_init(long* mode, long* model, char* dem_file, /* TBD */
                   /* output */
                   xp_dem_id* dem_id,
                   long ierr[XP_NUM_ERR_DEM_INIT]);

  long xp_dem_compute(xl_model_id* model_id,
                      xp_dem_id* dem_id,
                      double* lon,
                      double* lat,
                      /* output */
                      double* alt,
                      long ierr[XP_NUM_ERR_DEM_COMPUTE]);

  long xp_dem_close(xp_dem_id* dem_id,
                    /* output */
                    long ierr[XP_NUM_ERR_DEM_CLOSE]);

  long xp_dem_init_status(xp_dem_id* dem_id);

  long xp_dem_get_sat_id(xp_dem_id* dem_id);

  long xp_dem_get_mode(xp_dem_id* dem_id);

  long xp_dem_get_id_data(xp_dem_id* dem_id,
                          xp_dem_id_data* data);

  long xp_dem_get_info(xl_model_id* model_id,
                       xp_dem_id* dem_id,
                       double* lon,
                       double* lat,
                       xp_dem_info* dem_info,
                       long ierr[XP_NUM_ERR_DEM_GET_INFO]);

  /* AN-487 */
  long xp_dem_id_configure(xp_dem_id* dem_id, xp_dem_id_config* config, long ierr[XP_NUM_ERR_DEM_ID_CONFIGURE]);

  long xp_dem_get_cell_value(const xp_dem_id* dem_id,
                             long row,
                             long column,
                             double* value,
                             long ierr[XP_NUM_ERR_DEM_GET_CELL_VALUE]);

  long xp_dem_get_cell_geod(const xp_dem_id* dem_id,
                            long row,
                            long column,
                            double* lat,
                            double* lon,
                            long ierr[XP_NUM_ERR_DEM_GET_CELL_GEOD]);

  /* Target pointing functions */

  long xp_target_inter(long* sat_id,
                       xp_attitude_id* attitude_id,
                       xp_atmos_id* atmos_id,
                       xp_dem_id* dem_id,
                       long* deriv,
                       long* inter_flag,
                       double* los_az,
                       double* los_el,
                       double* geod_alt,
                       double* los_az_rate,
                       double* los_el_rate,
                       long* iray,
                       double* freq,
                       /* output */
                       long* num_user_target,
                       long* num_los_target,
                       xp_target_id* target_id,
                       long ierr[XP_NUM_ERR_TARGET_INTER]);

  long xp_target_inter_run(long* run_id,
                           xp_attitude_id* attitude_id,
                           long* deriv,
                           long* inter_flag,
                           double* los_az,
                           double* los_el,
                           double* geod_alt,
                           double* los_az_rate,
                           double* los_el_rate,
                           long* iray,
                           double* freq,
                           /* output */
                           long* num_user_target,
                           long* num_los_target,
                           xp_target_id* target_id,
                           long ierr[XP_NUM_ERR_TARGET_INTER]);

  long xp_target_travel_time(long* sat_id,
                             xp_attitude_id* attitude_id,
                             xp_atmos_id* atmos_id,
                             xp_dem_id* dem_id,
                             long* deriv,
                             double* los_az,
                             double* los_el,
                             double* travel_time,
                             double* los_az_rate,
                             double* los_el_rate,
                             double* travel_time_rate,
                             long* iray,
                             double* freq,
                             /* output */
                             long* num_user_target,
                             long* num_los_target,
                             xp_target_id* target_id,
                             long ierr[XP_NUM_ERR_TARGET_TRAVEL_TIME]);

  long xp_target_travel_time_run(long* run_id,
                                 xp_attitude_id* attitude_id,
                                 long* deriv,
                                 double* los_az,
                                 double* los_el,
                                 double* travel_time,
                                 double* los_az_rate,
                                 double* los_el_rate,
                                 double* travel_time_rate,
                                 long* iray,
                                 double* freq,
                                 /* output */
                                 long* num_user_target,
                                 long* num_los_target,
                                 xp_target_id* target_id,
                                 long ierr[XP_NUM_ERR_TARGET_TRAVEL_TIME]);

  long xp_target_ground_range(long* sat_id,
                              xp_attitude_id* attitude_id,
                              xp_dem_id* dem_id,
                              long* deriv,
                              double* los_az,
                              double* los_el,
                              double* geod_alt,
                              double* distance,
                              double* los_az_rate,
                              double* los_el_rate,
                              /* output */
                              long* num_user_target,
                              long* num_los_target,
                              xp_target_id* target_id,
                              long ierr[XP_NUM_ERR_TARGET_GROUND_RANGE]);

  long xp_target_ground_range_run(long* run_id,
                                  xp_attitude_id* attitude_id,
                                  long* deriv,
                                  double* los_az,
                                  double* los_el,
                                  double* geod_alt,
                                  double* distance,
                                  double* los_az_rate,
                                  double* los_el_rate,
                                  /* output */
                                  long* num_user_target,
                                  long* num_los_target,
                                  xp_target_id* target_id,
                                  long ierr[XP_NUM_ERR_TARGET_GROUND_RANGE]);

  long xp_target_incidence_angle(long* sat_id,
                                 xp_attitude_id* attitude_id,
                                 xp_dem_id* dem_id,
                                 long* deriv,
                                 double* los_az,
                                 double* inc_angle,
                                 double* geod_alt,
                                 double* los_az_rate,
                                 /* output */
                                 long* num_user_target,
                                 long* num_los_target,
                                 xp_target_id* target_id,
                                 long ierr[XP_NUM_ERR_TARGET_INCIDENCE_ANGLE]);

  long xp_target_incidence_angle_run(long* run_id,
                                     xp_attitude_id* attitude_id,
                                     long* deriv,
                                     double* los_az,
                                     double* inc_angle,
                                     double* geod_alt,
                                     double* los_az_rate,
                                     /* output */
                                     long* num_user_target,
                                     long* num_los_target,
                                     xp_target_id* target_id,
                                     long ierr[XP_NUM_ERR_TARGET_INCIDENCE_ANGLE]);

  long xp_target_range(long* sat_id,
                       xp_attitude_id* attitude_id,
                       xp_dem_id* dem_id,
                       long* deriv,
                       double* los_az,
                       double* range,
                       double* geod_alt,
                       double* los_az_rate,
                       double* range_rate,
                       /* output */
                       long* num_user_target,
                       long* num_los_target,
                       xp_target_id* target_id,
                       long ierr[XP_NUM_ERR_TARGET_RANGE]);

  long xp_target_range_run(long* run_id,
                           xp_attitude_id* attitude_id,
                           long* deriv,
                           double* los_az,
                           double* range,
                           double* geod_alt,
                           double* los_az_rate,
                           double* range_rate,
                           /* output */
                           long* num_user_target,
                           long* num_los_target,
                           xp_target_id* target_id,
                           long ierr[XP_NUM_ERR_TARGET_RANGE]);

  long xp_target_range_rate(long* sat_id,
                            xp_attitude_id* attitude_id,
                            xp_dem_id* dem_id,
                            long* deriv,
                            double* ef_range_rate,
                            double* range,
                            double* geod_alt,
                            double* ef_range_rate_rate,
                            double* range_rate,
                            /* output */
                            long* num_user_target,
                            long* num_los_target,
                            xp_target_id* target_id,
                            long ierr[XP_NUM_ERR_TARGET_RANGE_RATE]);

  long xp_target_range_rate_run(long* run_id,
                                xp_attitude_id* attitude_id,
                                long* deriv,
                                double* ef_range_rate,
                                double* range,
                                double* geod_alt,
                                double* ef_range_rate_rate,
                                double* range_rate,
                                /* output */
                                long* num_user_target,
                                long* num_los_target,
                                xp_target_id* target_id,
                                long ierr[XP_NUM_ERR_TARGET_RANGE_RATE]);

  long xp_target_tangent(long* sat_id,
                         xp_attitude_id* attitude_id,
                         xp_atmos_id* atmos_id,
                         xp_dem_id* dem_id,
                         long* deriv,
                         double* los_az,
                         double* los_el,
                         double* los_az_rate,
                         double* los_el_rate,
                         long* iray,
                         double* freq,
                         /* output */
                         long* num_user_target,
                         long* num_los_target,
                         xp_target_id* target_id,
                         long ierr[XP_NUM_ERR_TARGET_TANGENT]);

  long xp_target_tangent_run(long* run_id,
                             xp_attitude_id* attitude_id,
                             long* deriv,
                             double* los_az,
                             double* los_el,
                             double* los_az_rate,
                             double* los_el_rate,
                             long* iray,
                             double* freq,
                             /* output */
                             long* num_user_target,
                             long* num_los_target,
                             xp_target_id* target_id,
                             long ierr[XP_NUM_ERR_TARGET_TANGENT]);

  long xp_target_altitude(long* sat_id,
                          xp_attitude_id* attitude_id,
                          xp_atmos_id* atmos_id,
                          xp_dem_id* dem_id,
                          long* deriv,
                          double* los_az,
                          double* geod_alt,
                          double* los_az_rate,
                          long* iray,
                          double* freq,
                          /* output */
                          long* num_user_target,
                          long* num_los_target,
                          xp_target_id* target_id,
                          long ierr[XP_NUM_ERR_TARGET_ALTITUDE]);

  long xp_target_altitude_run(long* run_id,
                              xp_attitude_id* attitude_id,
                              long* deriv,
                              double* los_az,
                              double* geod_alt,
                              double* los_az_rate,
                              long* iray,
                              double* freq,
                              /* output */
                              long* num_user_target,
                              long* num_los_target,
                              xp_target_id* target_id,
                              long ierr[XP_NUM_ERR_TARGET_ALTITUDE]);

  long xp_target_star(long* sat_id,
                      xp_attitude_id* attitude_id,
                      xp_atmos_id* atmos_id,
                      xp_dem_id* dem_id,
                      long* deriv,
                      double* star_ra,
                      double* star_dec,
                      double* star_ra_rate,
                      double* star_dec_rate,
                      long* iray,
                      double* freq,
                      /* output */
                      long* num_user_target,
                      long* num_los_target,
                      xp_target_id* target_id,
                      long ierr[XP_NUM_ERR_TARGET_STAR]);

  long xp_target_star_run(long* run_id,
                          xp_attitude_id* attitude_id,
                          long* deriv,
                          double* star_ra,
                          double* star_dec,
                          double* star_ra_rate,
                          double* star_dec_rate,
                          long* iray,
                          double* freq,
                          /* output */
                          long* num_user_target,
                          long* num_los_target,
                          xp_target_id* target_id,
                          long ierr[XP_NUM_ERR_TARGET_STAR]);

  long xp_target_tangent_sun(long* sat_id,
                             xp_attitude_id* attitude_id,
                             xp_atmos_id* atmos_id,
                             xp_dem_id* dem_id,
                             long* deriv,
                             long* iray,
                             double* freq,
                             /* output */
                             long* num_user_target,
                             long* num_los_target,
                             xp_target_id* target_id,
                             long ierr[XP_NUM_ERR_TARGET_TANGENT_SUN]);

  long xp_target_tangent_sun_run(long* run_id,
                                 xp_attitude_id* attitude_id,
                                 long* deriv,
                                 long* iray,
                                 double* freq,
                                 /* output */
                                 long* num_user_target,
                                 long* num_los_target,
                                 xp_target_id* target_id,
                                 long ierr[XP_NUM_ERR_TARGET_TANGENT_SUN]);

  long xp_target_tangent_moon(long* sat_id,
                              xp_attitude_id* attitude_id,
                              xp_atmos_id* atmos_id,
                              xp_dem_id* dem_id,
                              long* deriv,
                              long* iray,
                              double* freq,
                              /* output */
                              long* num_user_target,
                              long* num_los_target,
                              xp_target_id* target_id,
                              long ierr[XP_NUM_ERR_TARGET_TANGENT_MOON]);

  long xp_target_tangent_moon_run(long* run_id,
                                  xp_attitude_id* attitude_id,
                                  long* deriv,
                                  long* iray,
                                  double* freq,
                                  /* output */
                                  long* num_user_target,
                                  long* num_los_target,
                                  xp_target_id* target_id,
                                  long ierr[XP_NUM_ERR_TARGET_TANGENT_MOON]);

  long xp_target_station(long* sat_id,
                         xp_attitude_id* attitude_id,
                         xp_dem_id* dem_id,
                         long* deriv,
                         double* geoc_long,
                         double* geod_lat,
                         double* geod_alt,
                         double* min_link_el,
                         /* output */
                         long* num_user_target,
                         long* num_los_target,
                         xp_target_id* target_id,
                         long ierr[XP_NUM_ERR_TARGET_STATION]);

  long xp_target_station_run(long* run_id,
                             xp_attitude_id* attitude_id,
                             long* deriv,
                             double* geoc_long,
                             double* geod_lat,
                             double* geod_alt,
                             double* min_link_el,
                             /* output */
                             long* num_user_target,
                             long* num_los_target,
                             xp_target_id* target_id,
                             long ierr[XP_NUM_ERR_TARGET_STATION]);

  long xp_target_generic(long* sat_id,
                         xp_attitude_id* attitude_id,
                         xp_dem_id* dem_id,
                         long* deriv,
                         double targ_pos[3],
                         double targ_vel[3],
                         double targ_acc[3],
                         /* output */
                         long* num_user_target,
                         long* num_los_target,
                         xp_target_id* target_id,
                         long ierr[XP_NUM_ERR_TARGET_GENERIC]);

  long xp_target_generic_run(long* run_id,
                             xp_attitude_id* attitude_id,
                             long* deriv,
                             double targ_pos[3],
                             double targ_vel[3],
                             double targ_acc[3],
                             /* output */
                             long* num_user_target,
                             long* num_los_target,
                             xp_target_id* target_id,
                             long ierr[XP_NUM_ERR_TARGET_GENERIC]);

  long xp_target_reflected(/* inputs */
                           long* sat_id,
                           xp_attitude_id* attitude_id,
                           long* deriv,
                           double* geod_alt,
                           double* deflection_north,
                           double* deflection_east,
                           long* source_type,
                           double source_param[XP_NUM_SOURCE_PARAM],
                           /* outputs */
                           long* num_user_target,
                           long* num_los_target,
                           xp_target_id* target_id,
                           long ierr[XP_NUM_ERR_TARGET_REFLECTED]);

  /* AN-353 */
  long xp_target_sc(long* sat_id1, xp_attitude_id* attitude_id1, long* sat_id2, xp_attitude_id* attitude_id2, long* deriv,
                    /* output */
                    long* num_user_target,
                    long* num_los_target,
                    xp_target_id* target_id,
                    long ierr[XP_NUM_ERR_TARGET_SC]);

  long xp_target_reflected_run(/*    inputs */
                               long* run_id,
                               xp_attitude_id* attitude_id,
                               long* deriv,
                               double* geod_alt,
                               double* deflection_north,
                               double* deflection_east,
                               long* source_type,
                               double source_param[XP_NUM_SOURCE_PARAM],
                               /* outputs */
                               long* num_user_target,
                               long* num_los_target,
                               xp_target_id* target_id,
                               long ierr[XP_NUM_ERR_TARGET_REFLECTED]);

  long xp_multi_target_inter(long* sat_id,
                             xp_attitude_id* attitude_id,
                             xp_atmos_id* atmos_id,
                             xp_dem_id* dem_id,
                             long* deriv,
                             long* inter_flag,
                             double* los_az,
                             double* los_el,
                             long* num_target,
                             double* geod_alt,
                             double* los_az_rate,
                             double* los_el_rate,
                             long* iray,
                             double* freq,
                             /* output */
                             long* num_user_target,
                             long* num_los_target,
                             xp_target_id* target_id,
                             long ierr[XP_NUM_ERR_MULTI_TARGET_INTER]);

  long xp_multi_target_inter_run(long* run_id,
                                 xp_attitude_id* attitude_id,
                                 long* deriv,
                                 long* inter_flag,
                                 double* los_az,
                                 double* los_el,
                                 long* num_target,
                                 double* geod_alt,
                                 double* los_az_rate,
                                 double* los_el_rate,
                                 long* iray,
                                 double* freq,
                                 /* output */
                                 long* num_user_target,
                                 long* num_los_target,
                                 xp_target_id* target_id,
                                 long ierr[XP_NUM_ERR_MULTI_TARGET_INTER]);

  long xp_multi_target_travel_time(long* sat_id,
                                   xp_attitude_id* attitude_id,
                                   xp_atmos_id* atmos_id,
                                   xp_dem_id* dem_id,
                                   long* deriv,
                                   double* los_az,
                                   double* los_el,
                                   long* num_target,
                                   double* travel_time,
                                   double* los_az_rate,
                                   double* los_el_rate,
                                   double* travel_time_rate,
                                   long* iray,
                                   double* freq,
                                   /* output */
                                   long* num_user_target,
                                   long* num_los_target,
                                   xp_target_id* target_id,
                                   long ierr[XP_NUM_ERR_MULTI_TARGET_TRAVEL_TIME]);

  long xp_multi_target_travel_time_run(long* run_id,
                                       xp_attitude_id* attitude_id,
                                       long* deriv,
                                       double* los_az,
                                       double* los_el,
                                       long* num_target,
                                       double* travel_time,
                                       double* los_az_rate,
                                       double* los_el_rate,
                                       double* travel_time_rate,
                                       long* iray,
                                       double* freq,
                                       /* output */
                                       long* num_user_target,
                                       long* num_los_target,
                                       xp_target_id* target_id,
                                       long ierr[XP_NUM_ERR_MULTI_TARGET_TRAVEL_TIME]);

  long xp_target_list_inter(long* sat_id,
                            xp_attitude_id* attitude_id,
                            xp_atmos_id* atmos_id,
                            xp_dem_id* dem_id,
                            long* deriv,
                            long* inter_flag,
                            xp_instrument_data* azimuth_elevation_data,
                            double* geod_alt,
                            /* output */
                            xp_target_output* target_out,
                            xp_target_id* target_id,
                            long ierr[XP_NUM_ERR_TARGET_LIST_INTER]); /* AN-470 */

  /* Target extra calculation functions */

  long xp_target_extra_vector(xp_target_id* target_id,
                              long* choice,
                              long* target_type,
                              long* target_number,
                              /* output */
                              double vector_results[XP_SIZE_TARGET_RESULT_VECTOR],
                              double vector_results_rate[XP_SIZE_TARGET_RESULT_VECTOR],
                              double vector_results_rate_rate[XP_SIZE_TARGET_RESULT_VECTOR],
                              long ierr[XP_NUM_ERR_TARGET_EXTRA_VECTOR]);

  long xp_target_extra_main(xp_target_id* target_id,
                            long* choice,
                            long* target_type,
                            long* target_number,
                            /* output */
                            double main_results[XP_SIZE_TARGET_RESULT_MAIN],
                            double main_results_rate[XP_SIZE_TARGET_RESULT_MAIN],
                            double main_results_rate_rate[XP_SIZE_TARGET_RESULT_MAIN],
                            long ierr[XP_NUM_ERR_TARGET_EXTRA_MAIN]);

  long xp_target_extra_aux(xp_target_id* target_id,
                           long* choice,
                           long* target_type,
                           long* target_number,
                           /* output */
                           double aux_results[XP_SIZE_TARGET_RESULT_AUX],
                           double aux_results_rate[XP_SIZE_TARGET_RESULT_AUX],
                           double aux_results_rate_rate[XP_SIZE_TARGET_RESULT_AUX],
                           long ierr[XP_NUM_ERR_TARGET_EXTRA_AUX]);

  long xp_target_extra_target_to_sun(xp_target_id* target_id,
                                     long* choice,
                                     long* target_type,
                                     long* target_number,
                                     long* iray,
                                     double* freq,
                                     /* output */
                                     double sun_results[XP_SIZE_SUN_RESULT],
                                     double sun_results_rate[XP_SIZE_SUN_RESULT],
                                     double sun_results_rate_rate[XP_SIZE_SUN_RESULT],
                                     long ierr[XP_NUM_ERR_TARGET_EXTRA_TARGET_TO_SUN]);

  long xp_target_extra_target_to_moon(xp_target_id* target_id,
                                      long* choice,
                                      long* target_type,
                                      long* target_number,
                                      long* iray,
                                      double* freq,
                                      /* output */
                                      double moon_results[XP_SIZE_MOON_RESULT],
                                      double moon_results_rate[XP_SIZE_MOON_RESULT],
                                      double moon_results_rate_rate[XP_SIZE_MOON_RESULT],
                                      long ierr[XP_NUM_ERR_TARGET_EXTRA_TARGET_TO_MOON]);

  long xp_target_extra_ef_target(xp_target_id* target_id,
                                 long* choice,
                                 long* target_type,
                                 long* target_number,
                                 double* freq,
                                 /* output */
                                 double ef_target_results_rate[XP_SIZE_EF_TARGET_RESULT],
                                 double ef_target_results_rate_rate[XP_SIZE_EF_TARGET_RESULT],
                                 long ierr[XP_NUM_ERR_TARGET_EXTRA_EF_TARGET]);

  long xp_target_extra_specular_reflection(/*inputs*/
                                           xp_target_id* target_id,
                                           long* choice,
                                           long* target_type,
                                           long* target_number,
                                           double* deflection_north,
                                           double* deflection_east,
                                           /* outputs */
                                           double spec_reflec_results[XP_SIZE_TARGET_RESULT_SPEC_REFL],
                                           double spec_reflec_results_rate[XP_SIZE_TARGET_RESULT_SPEC_REFL],
                                           double spec_reflec_results_rate_rate[XP_SIZE_TARGET_RESULT_SPEC_REFL],
                                           long ierr[XP_NUM_ERR_TARGET_EXTRA_SPEC_REFL]);

  long xp_target_list_extra_vector(xp_target_id* target_id,
                                   long* choice,
                                   long* target_type,
                                   /* output */
                                   xp_target_extra_vector_results_list* list,
                                   long ierr[XP_NUM_ERR_TARGET_LIST_EXTRA_VECTOR]); /*AN-471*/

  long xp_target_list_extra_main(xp_target_id* target_id,
                                 long* choice,
                                 long* target_type,
                                 /* output */
                                 xp_target_extra_main_results_list* list,
                                 long ierr[XP_NUM_ERR_TARGET_LIST_EXTRA_MAIN]); /*AN-471*/

  long xp_target_list_extra_aux(xp_target_id* target_id,
                                long* choice,
                                long* target_type,
                                /* output */
                                xp_target_extra_aux_results_list* list,
                                long ierr[XP_NUM_ERR_TARGET_LIST_EXTRA_AUX]); /*AN-471*/

  long xp_target_list_extra_ef_target(xp_target_id* target_id,
                                      long* choice,
                                      long* target_type,
                                      double* freq,
                                      /* output */
                                      xp_target_extra_ef_target_results_list* list,
                                      long ierr[XP_NUM_ERR_TARGET_LIST_EXTRA_EF_TARGET]); /*AN-471*/

  long xp_target_list_extra_specular_reflection(xp_target_id* target_id,
                                                long* choice,
                                                long* target_type,
                                                double* deflection_north,
                                                double* deflection_east,
                                                /* outputs */
                                                xp_target_extra_spec_reflec_target_results_list* list,
                                                long ierr[XP_NUM_ERR_TARGET_LIST_EXTRA_SPEC_REFL]); /* AN-471 */

  long xp_target_list_extra_target_to_moon(xp_target_id* target_id,
                                           long* choice,
                                           long* target_type,
                                           long* iray,
                                           double* freq,
                                           /* output */
                                           xp_target_extra_moon_target_results_list* list,
                                           long ierr[XP_NUM_ERR_TARGET_LIST_EXTRA_TARGET_TO_MOON]); /* AN-471 */

  long xp_target_list_extra_target_to_sun(xp_target_id* target_id,
                                          long* choice,
                                          long* target_type,
                                          long* iray,
                                          double* freq,
                                          /* output */
                                          xp_target_extra_sun_target_results_list* list,
                                          long ierr[XP_NUM_ERR_TARGET_LIST_EXTRA_TARGET_TO_SUN]); /* AN-471 */

  long xp_target_close(xp_target_id* target_id,
                       /* output */
                       long ierr[XP_NUM_ERR_TARGET_CLOSE]);

  long xp_target_status(xp_target_id* target_id);

  long xp_target_get_sat_id(xp_target_id* target_id);

  long xp_target_get_mode(xp_target_id* target_id);

  long xp_target_get_id_data(xp_target_id* target_id,
                             xp_target_id_data* data);

  /* AN-492: Generate maximum altitude files */
  long xp_gen_dem_max_altitude_file(const char* dem_cfg_file,
                                    /* output */
                                    long ierr[XP_NUM_ERR_DEM_MAX_ALT_FILE]);

  /* AN-466 */
  long xp_gen_dem_altitudes_from_ellipsoid(char* dem_file, long* num_harmonics, char* output_dir, xp_gen_dem_alt_from_ellipsoid_inputs* inputs,
                                           /* output */
                                           long ierr[XP_NUM_ERR_DEM_ALT_FROM_ELLIPSOID]);

  /* AN-526 */
  long xp_attitude_transform(xp_transform_cfg* transform_cfg, xp_attitude_def* attitude_def, long ierr[XP_NUM_ERR_ATTITUDE_TRANSFORM]);

  /*AN-687 */
  long xp_free_target_id_data(xp_target_id_data* data, long num_user_target);

  /* run_id functions */
  /* ---------------- */

  long xp_run_init(long* run_id,
                   xp_sat_nom_trans_id* sat_nom_trans_id,
                   xp_sat_trans_id* sat_trans_id,
                   xp_instr_trans_id* instr_trans_id,
                   xp_atmos_id* atmos_id,
                   xp_dem_id* dem_id,
                   /* output */
                   long ierr[XP_NUM_ERR_RUN_INIT]);

  void xp_run_get_ids(long* run_id,
                      /* output */
                      xp_sat_nom_trans_id* sat_nom_trans_id,
                      xp_sat_trans_id* sat_trans_id,
                      xp_instr_trans_id* instr_trans_id,
                      xp_atmos_id* atmos_id,
                      xp_dem_id* dem_id);

  void xp_run_close(long* run_id);

  /* Check libraries compatibility */
  /* ----------------------------- */

  long xp_check_library_version();

/* Alias for xx_check_library_version */
#if defined(expcfi_check_libs)

  #if !defined(xv_check_library_version)

    #undef expcfi_check_libs
    #define expcfi_check_libs xp_check_library_version

  #endif

#else

  #define expcfi_check_libs xp_check_library_version

#endif

  /*******************************************************************************
*
* Error Handling Functions - prototype declaration
*
*******************************************************************************/

#define XP_MAX_COD XL_MAX_COD
#define XP_MAX_STR XL_MAX_STR

  long xp_silent(void);

  long xp_verbose(void);

  long xp_print_msg(long* n, char msg[XP_MAX_COD][XP_MAX_STR]);

  long xp_get_msg(long* func_id, long* ierr,
                  /* output */
                  long* n,
                  char msg[XP_MAX_COD][XP_MAX_STR]);

  long xp_get_code(long* func_id, long* ierr,
                   /* output */
                   long* n,
                   long vec[XP_MAX_COD]);

#ifdef __cplusplus
}
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

  XP_ERR = XL_ERR, /* Error status */
  XP_OK = XL_OK, /* Nominal status */
  XP_WARN = XL_WARN /* Warning status */

} XP_PErr_type_enum; /* CAREFUL: variables holding these values MUST be declared as long !!! */

/* Satellite ID */
/* ------------ */

typedef enum
{

  XP_SAT_DEFAULT = XL_SAT_DEFAULT,
  XP_SAT_DEFAULT1 = XL_SAT_DEFAULT1,
  XP_SAT_DEFAULT2 = XL_SAT_DEFAULT2,
  XP_SAT_DEFAULT3 = XL_SAT_DEFAULT3,
  XP_SAT_DEFAULT4 = XL_SAT_DEFAULT4,
  XP_SAT_DEFAULT5 = XL_SAT_DEFAULT5,
  XP_SAT_DEFAULT6 = XL_SAT_DEFAULT6,
  XP_SAT_DEFAULT7 = XL_SAT_DEFAULT7,
  XP_SAT_DEFAULT8 = XL_SAT_DEFAULT8,
  XP_SAT_DEFAULT9 = XL_SAT_DEFAULT9,
  XP_SAT_ERS1 = XL_SAT_ERS1,
  XP_SAT_ERS2 = XL_SAT_ERS2,
  XP_SAT_ENVISAT = XL_SAT_ENVISAT,
  XP_SAT_METOP1 = XL_SAT_METOP1,
  XP_SAT_METOP2 = XL_SAT_METOP2,
  XP_SAT_METOP3 = XL_SAT_METOP3,
  XP_SAT_CRYOSAT = XL_SAT_CRYOSAT,
  XP_SAT_ADM = XL_SAT_ADM,
  XP_SAT_GOCE = XL_SAT_GOCE,
  XP_SAT_SMOS = XL_SAT_SMOS,
  XP_SAT_TERRASAR = XL_SAT_TERRASAR,
  XP_SAT_EARTHCARE = XL_SAT_EARTHCARE,
  XP_SAT_SWARM_A = XL_SAT_SWARM_A,
  XP_SAT_SWARM_B = XL_SAT_SWARM_B,
  XP_SAT_SWARM_C = XL_SAT_SWARM_C,
  XP_SAT_SENTINEL_1A = XL_SAT_SENTINEL_1A,
  XP_SAT_SENTINEL_1B = XL_SAT_SENTINEL_1B,
  XP_SAT_SENTINEL_2 = XL_SAT_SENTINEL_2,
  XP_SAT_SENTINEL_3 = XL_SAT_SENTINEL_3,
  XP_SAT_SEOSAT = XL_SAT_SEOSAT,
  XP_SAT_SENTINEL_1C = XL_SAT_SENTINEL_1C,
  XP_SAT_SENTINEL_2A = XL_SAT_SENTINEL_2A,
  XP_SAT_SENTINEL_2B = XL_SAT_SENTINEL_2B,
  XP_SAT_SENTINEL_2C = XL_SAT_SENTINEL_2C,
  XP_SAT_SENTINEL_3A = XL_SAT_SENTINEL_3A,
  XP_SAT_SENTINEL_3B = XL_SAT_SENTINEL_3B,
  XP_SAT_SENTINEL_3C = XL_SAT_SENTINEL_3C,
  XP_SAT_JASON_CSA = XL_SAT_JASON_CSA, // AN-531
  XP_SAT_JASON_CSB = XL_SAT_JASON_CSB, // AN-531
  XP_SAT_METOP_SG_A1 = XL_SAT_METOP_SG_A1, // AN-530
  XP_SAT_METOP_SG_A2 = XL_SAT_METOP_SG_A2, // AN-530
  XP_SAT_METOP_SG_A3 = XL_SAT_METOP_SG_A3, // AN-530
  XP_SAT_METOP_SG_B1 = XL_SAT_METOP_SG_B1, // AN-530
  XP_SAT_METOP_SG_B2 = XL_SAT_METOP_SG_B2, // AN-530
  XP_SAT_METOP_SG_B3 = XL_SAT_METOP_SG_B3, // AN-530
  XP_SAT_SENTINEL_5P = XL_SAT_SENTINEL_5P, // AN-502
  XP_SAT_BIOMASS = XL_SAT_BIOMASS, // AN-601
  XP_SAT_SENTINEL_5 = XL_SAT_SENTINEL_5, // AN-639
  XP_SAT_SAOCOM_CS = XL_SAT_SAOCOM_CS, // AN-640
  XP_SAT_FLEX = XL_SAT_FLEX, /* AN-683 */
  XP_SAT_SENTINEL_6A = XL_SAT_SENTINEL_6A, /* AN-809 */
  XP_SAT_SENTINEL_6B = XL_SAT_SENTINEL_6B, /* AN-809 */
  XP_SAT_CIMR = XL_SAT_CIMR,
  XP_SAT_ROSEL = XL_SAT_ROSEL,
  XP_SAT_CHIME = XL_SAT_CHIME,
  XP_SAT_CRISTAL = XL_SAT_CRISTAL,
  XP_SAT_CO2M = XL_SAT_CO2M,
  XP_SAT_LSTM = XL_SAT_LSTM,
  XP_SAT_FORUM = XL_SAT_FORUM,
  XP_SAT_TRUTHS = XL_SAT_TRUTHS,
  XP_SAT_GENERIC = XL_SAT_GENERIC,
  XP_SAT_GENERIC_GEO = XL_SAT_GENERIC_GEO, /* ANR-353 */
  XP_SAT_MTG = XL_SAT_MTG, /* ANR-353 */
  XP_SAT_GENERIC_MEO = XL_SAT_GENERIC_MEO /* ANR-550 */

} XP_Sat_id_enum; /* CAREFUL: if you modify this enum, you must modify XP_Sat_array_enum enum
                    and XL_Sat_id_to_array routine */

/* Instrument ID */
/* ------------- */

typedef enum
{

  XP_INSTR_AOCS_ONLY = 0,
  XP_INSTR_DEFAULT_GENERIC = 1,
  XP_INSTR_ERS_GENERIC = 10,
  XP_INSTR_ERS_AMI = 11,
  XP_INSTR_ERS_ATSR = 12,
  XP_INSTR_ERS_MWS = 13,
  XP_INSTR_ERS_LR = 14,
  XP_INSTR_ERS_PRARE = 15,
  XP_INSTR_ERS_RA = 16,
  XP_INSTR_ENVI_GENERIC = 200,
  XP_INSTR_ENVI_AATSR = 201,
  XP_INSTR_ENVI_ASAR = 202,
  XP_INSTR_ENVI_GOMOS = 203,
  XP_INSTR_ENVI_LRR = 204,
  XP_INSTR_ENVI_MERIS = 205,
  XP_INSTR_ENVI_MIPAS = 206,
  XP_INSTR_ENVI_MWR = 207,
  XP_INSTR_ENVI_RA2 = 208,
  XP_INSTR_ENVI_SCIAMACHY = 209,
  XP_INSTR_ENVI_DORIS = 210,
  XP_INSTR_METOP_GENERIC = 30,
  XP_INSTR_METOP_AMSUA = 31,
  XP_INSTR_METOP_ASCAT = 32,
  XP_INSTR_METOP_AVHRR = 33,
  XP_INSTR_METOP_GOME2 = 34,
  XP_INSTR_METOP_GRAS = 35,
  XP_INSTR_METOP_HIRS = 36,
  XP_INSTR_METOP_IASI = 37,
  XP_INSTR_METOP_MHS = 38,
  XP_INSTR_CRYO_GENERIC = 40,
  XP_INSTR_CRYO_SIRAL = 41,
  XP_INSTR_CRYO_DORIS = 42,
  XP_INSTR_ADM_GENERIC = 50,
  XP_INSTR_ADM_ALADIN = 51,
  XP_INSTR_ADM_DWL = 52,
  XP_INSTR_GOCE_GENERIC = 60,
  XP_INSTR_GOCE_GRAD = 61,
  XP_INSTR_GOCE_GPSR = 62,
  XP_INSTR_SMOS_GENERIC = 70,
  XP_INSTR_SMOS_MIRAS_FOV = 71,
  XP_INSTR_SMOS_MIRAS_NOMINAL = 72,
  XP_INSTR_SMOS_MIRAS_NARROW = 73,
  XP_INSTR_TERRASAR_GENERIC = 80

  /*
  Important: The instruments for the default satellites have IDs with values between 
  XP_DEFAULT_INSTRUMENT_MIN and XP_DEFAULT_INSTRUMENT_MAX (Present values are 301 to 500).
  See defines in xl_sat_default.h
*/
} XP_Instr_id_enum;

/* Time reference ID */
/* ----------------- */

typedef enum
{

  XP_TIME_UNDEF = XL_TIME_UNDEF,
  XP_TIME_TAI = XL_TIME_TAI,
  XP_TIME_UTC = XL_TIME_UTC,
  XP_TIME_UT1 = XL_TIME_UT1,
  XP_TIME_GPS = XL_TIME_GPS

} XP_Time_ref_enum;

/* Input Time modes */
/* ---------------- */

typedef enum
{
  XP_SEL_FILE = XL_SEL_FILE,
  XP_SEL_TIME = XL_SEL_TIME,
  XP_SEL_ORBIT = XL_SEL_ORBIT,
  XP_SEL_DEFAULT = XL_SEL_DEFAULT
} XP_Time_init_mode;

/* Sat. Nominal Attitude mode */
/* -------------------------- */

typedef enum
{

  XP_SAT_NOMINAL_ATT_INIT_MODE = 0,
  XP_SAT_NOMINAL_ATT_INIT_MODEL_MODE,
  XP_SAT_NOMINAL_ATT_INIT_HARMONIC_MODE,
  XP_SAT_NOMINAL_ATT_INIT_FILE_MODE

} XP_Sat_nominal_att_mode_enum;

/* Sat. Attitude mode */
/* ------------------ */

typedef enum
{

  XP_SAT_ATT_ANGLE_INIT_MODE = 0,
  XP_SAT_ATT_MATRIX_INIT_MODE,
  XP_SAT_ATT_INIT_HARMONIC_MODE,
  XP_SAT_ATT_INIT_FILE_MODE

} XP_Sat_att_mode_enum;

/* Instrument Attitude mode */
/* ------------------------ */

typedef enum
{

  XP_INSTR_ATT_ANGLE_INIT_MODE = 0,
  XP_INSTR_ATT_MATRIX_INIT_MODE,
  XP_INSTR_ATT_INIT_HARMONIC_MODE,
  XP_INSTR_ATT_INIT_FILE_MODE

} XP_Instr_att_mode_enum;

/* Attitude mode */
/* ------------- */

typedef enum
{

  XP_ATTITUDE_INIT_NO_DATA_MODE = 0,
  XP_ATTITUDE_COMPUTE_MODE,
  XP_ATTITUDE_USER_SET_MODE

} XP_Attitude_mode_enum;

/* Attitude File models */
/* -------------------- */

typedef enum
{

  XP_ATTITUDE_GENERIC_FILE_MODEL = 0,
  XP_ATTITUDE_STAR_TRACKER_FILE_MODEL,
  XP_ATTITUDE_QUATERNION_NO_FILE_MODEL

} XP_Attitude_file_model_enum;

/* Target mode */
/* ----------- */

typedef enum
{
  XP_TARGET_INTER_1ST_MODE = 0,
  XP_TARGET_INTER_2ND_MODE,
  XP_TARGET_TRAVEL_TIME_1ST_MODE,
  XP_TARGET_TRAVEL_TIME_2ND_MODE,
  XP_TARGET_GROUND_RANGE_MODE,
  XP_TARGET_INCIDENCE_ANGLE_MODE,
  XP_TARGET_RANGE_MODE,
  XP_TARGET_RANGE_RATE_MODE,
  XP_TARGET_TANGENT_MODE,
  XP_TARGET_ALTITUDE_MODE,
  XP_TARGET_STAR_MODE,
  XP_TARGET_TANGENT_SUN_MODE,
  XP_TARGET_TANGENT_MOON_MODE,
  XP_TARGET_STATION_MODE,
  XP_TARGET_GENERIC_MODE,
  XP_TARGET_SC_MODE,
  XP_MULTI_TARGET_INTER_1ST_MODE,
  XP_MULTI_TARGET_INTER_2ND_MODE,
  XP_MULTI_TARGET_TRAVEL_TIME_1ST_MODE,
  XP_MULTI_TARGET_TRAVEL_TIME_2ND_MODE,
  XP_TARGET_REFLECTED_MODE,
  XP_TARGET_LIST_INTER_1ST_MODE, /* AN-470 */
  XP_TARGET_LIST_INTER_2ND_MODE /* AN-470 */

} XP_Target_mode_enum;

/* AOCS mode. AN-469: must be equal to DH enumeration */
/* -------------------------------------------------- */

typedef enum
{

  XP_AOCS_GPM = XD_AOCS_GPM,
  XP_AOCS_LNP = XD_AOCS_LNP,
  XP_AOCS_YSM = XD_AOCS_YSM,
  XP_AOCS_ZDOPPLER = XD_AOCS_ZDOPPLER

} XP_Sat_nominal_att_aocs_mode_enum;

/* Sat Nom model. AN-469: must be equal to DH enum */
/* ----------------------------------------------- */

typedef enum
{

  XP_MODEL_GENERIC = XD_MODEL_GENERIC,
  XP_MODEL_ENVISAT = XD_MODEL_ENVISAT,
  XP_MODEL_CRYOSAT = XD_MODEL_CRYOSAT,
  XP_MODEL_ADM = XD_MODEL_ADM,
  XP_MODEL_SENTINEL1 = XD_MODEL_SENTINEL1,
  XP_MODEL_SENTINEL2 = XD_MODEL_SENTINEL2,
  XP_MODEL_GEO = XD_MODEL_GEO, /* ANR-353 */
  XP_MODEL_METOPSG = XD_MODEL_METOPSG /* AN-666 */

} XP_Sat_nominal_att_model_enum;

/* Axis enumeration */
/* ---------------- */

typedef enum
{

  XP_X_AXIS = XL_X_AXIS,
  XP_NEG_X_AXIS = XL_NEG_X_AXIS,
  XP_Y_AXIS = XL_Y_AXIS,
  XP_NEG_Y_AXIS = XL_NEG_Y_AXIS,
  XP_Z_AXIS = XL_Z_AXIS,
  XP_NEG_Z_AXIS = XL_NEG_Z_AXIS

} XP_Axis_enum;

/* Axis target enumeration */
/* ----------------------- */

typedef enum
{

  XP_SUN_VEC = 0,
  XP_MOON_VEC,
  XP_EARTH_VEC,
  XP_NADIR_VEC,
  XP_INERTIAL_VEL_VEC,
  XP_EF_VEL_VEC,
  XP_INERTIAL_TARGET_VEC,
  XP_EF_TARGET_VEC,
  XP_SC_EF_VEL_VEC,
  XP_ORBIT_POLE,
  XP_INERTIAL_POS_VEC_CORRECTED,
  XP_INERTIAL_VEL_VEC_ROTATED,
  XP_EF_NORTH, /* AN-353 */
  XP_EF_SOUTH /* AN-353 */

} XP_Axis_target_enum;

/* Angle Type. AN-469: must be equal to DH enum */
/* -------------------------------------------- */

typedef enum
{

  XP_ANGLE_TYPE_TRUE_LAT_TOD = XD_ANGLE_TYPE_TRUE_LAT_TOD,
  XP_ANGLE_TYPE_TRUE_LAT_EF = XD_ANGLE_TYPE_TRUE_LAT_EF /* AN-508 */

} XP_Angle_type_enum;

/* Attitude Frame */
/* -------------- */

typedef enum
{
  XP_NONE_ATTITUDE = -1, /* AN-468 */
  XP_SAT_ORBITAL_REF = XL_SAT_ORBITAL_REF,
  XP_SAT_NOMINAL_ATT = XL_SAT_NOMINAL_ATT,
  XP_SAT_ATT = XL_SAT_ATT,
  XP_INSTR_ATT = XL_INSTR_ATT

} XP_Attitude_frame_enum;

/* Target Type */
/* ----------- */

typedef enum
{

  XP_USER_TARGET_TYPE = 0,
  XP_LOS_TARGET_TYPE,
  XP_DEM_TARGET_TYPE

} XP_Target_type_enum;

/* Intersection mode */
/* ----------------- */

typedef enum
{

  XP_NO_INTER = 0,
  XP_INTER_1ST,
  XP_INTER_2ND

} XP_Inter_mode_enum;

/* Atmosphere initialization mode */
/* ------------------------------ */
/* Changed for AN-407. The current values are similar to 
those of the old enumeration XP_Target_ray_enum, now removed */
typedef enum
{
  XP_NO_REF_INIT = 0,
  XP_STD_INIT,
  XP_USER_INIT,
  XP_PRED_INIT,
  XP_STD_INIT_N = 10,
  XP_USER_INIT_N = 20,
  XP_PRED_INIT_N = 30,
  XP_US76_INIT = 300,
  XP_TROPIC_INIT,
  XP_MID_SUM_INIT,
  XP_MID_WIN_INIT,
  XP_SUBAR_SUM_INIT,
  XP_SUBAR_WIN_INIT,
  XP_LUT_INIT = 400,
  XP_US76_INIT_N = 3000,
  XP_TROPIC_INIT_N,
  XP_MID_SUM_INIT_N,
  XP_MID_WIN_INIT_N,
  XP_SUBAR_SUM_INIT_N,
  XP_SUBAR_WIN_INIT_N,
  XP_LUT_INIT_N = 4000

} XP_Atmos_mode_enum;

typedef enum
{
  XP_NO_REF = XP_NO_REF_INIT,
  XP_STD_REF = XP_STD_INIT,
  XP_USER_REF = XP_USER_INIT,
  XP_PRED_REF = XP_PRED_INIT,
  XP_STD_REF_N = XP_STD_INIT_N,
  XP_USER_REF_N = XP_USER_INIT_N,
  XP_PRED_REF_N = XP_PRED_INIT_N,
  XP_US76_REF = XP_US76_INIT,
  XP_TROPIC_REF = XP_TROPIC_INIT,
  XP_MID_SUM_REF = XP_MID_SUM_INIT,
  XP_MID_WIN_REF = XP_MID_WIN_INIT,
  XP_SUBAR_SUM_REF = XP_SUBAR_SUM_INIT,
  XP_SUBAR_WIN_REF = XP_SUBAR_WIN_INIT,
  XP_LUT_REF = XP_LUT_INIT,
  XP_US76_REF_N = XP_US76_INIT_N,
  XP_TROPIC_REF_N = XP_TROPIC_INIT_N,
  XP_MID_SUM_REF_N = XP_MID_SUM_INIT_N,
  XP_MID_WIN_REF_N = XP_MID_WIN_INIT_N,
  XP_SUBAR_SUM_REF_N = XP_SUBAR_SUM_INIT_N,
  XP_SUBAR_WIN_REF_N = XP_SUBAR_WIN_INIT_N,
  XP_LUT_REF_N = XP_LUT_INIT_N

} XP_Target_ray_enum; /* CAREFUL: variables holding these values MUST be declared as long !!! */

/* xp_target main results vector names */
/* ----------------------------------- */
typedef enum
{
  XP_TARG_EXTRA_MAIN_GEO = 1,
  XP_TARG_EXTRA_MAIN_GEO_D = 2,
  XP_TARG_EXTRA_MAIN_GEO_2D = 4,
  XP_TARG_EXTRA_MAIN_TARG2SAT_TOP = 8,
  XP_TARG_EXTRA_MAIN_TARG2SAT_TOP_D = 16,
  XP_TARG_EXTRA_MAIN_TARG2SAT_TOP_2D = 32,
  XP_TARG_EXTRA_MAIN_SAT2TARG_TOP = 64,
  XP_TARG_EXTRA_MAIN_SAT2TARG_TOP_D = 128,
  XP_TARG_EXTRA_MAIN_SAT2TARG_TOP_2D = 256,
  XP_TARG_EXTRA_MAIN_SAT2TARG_ATTITUDE = 512,
  XP_TARG_EXTRA_MAIN_SAT2TARG_ATTITUDE_D = 1024,
  XP_TARG_EXTRA_MAIN_SAT2TARG_ATTITUDE_2D = 2048,
  XP_TARG_EXTRA_MAIN_TARG2SAT_ATTITUDE = 4096,
  XP_TARG_EXTRA_MAIN_TARG2SAT_ATTITUDE_D = 8192,
  XP_TARG_EXTRA_MAIN_TARG2SAT_ATTITUDE_2D = 16384,
  XP_TARG_EXTRA_MAIN_ALL = XP_TARG_EXTRA_MAIN_GEO + XP_TARG_EXTRA_MAIN_GEO_D + XP_TARG_EXTRA_MAIN_GEO_2D + XP_TARG_EXTRA_MAIN_TARG2SAT_TOP + XP_TARG_EXTRA_MAIN_TARG2SAT_TOP_D + XP_TARG_EXTRA_MAIN_TARG2SAT_TOP_2D + XP_TARG_EXTRA_MAIN_SAT2TARG_TOP + XP_TARG_EXTRA_MAIN_SAT2TARG_TOP_D + XP_TARG_EXTRA_MAIN_SAT2TARG_TOP_2D + XP_TARG_EXTRA_MAIN_SAT2TARG_ATTITUDE + XP_TARG_EXTRA_MAIN_SAT2TARG_ATTITUDE_D + XP_TARG_EXTRA_MAIN_SAT2TARG_ATTITUDE_2D + XP_TARG_EXTRA_MAIN_TARG2SAT_ATTITUDE + XP_TARG_EXTRA_MAIN_TARG2SAT_ATTITUDE_D + XP_TARG_EXTRA_MAIN_TARG2SAT_ATTITUDE_2D

} XP_Target_res_main_name_enum; /* CAREFUL: variables holding these values MUST be declared as long !!! */

/* xp_target auxiliary results vector names */
/* ---------------------------------------- */
typedef enum
{
  XP_TARG_EXTRA_AUX_DIST_NAD_TARG_GT = 1,
  XP_TARG_EXTRA_AUX_RAD_CUR = 2,
  XP_TARG_EXTRA_AUX_DIST_NAD_TARG_GT_D = 4,
  XP_TARG_EXTRA_AUX_DIST_NAD_TARG_GT_2D = 8,
  XP_TARG_EXTRA_AUX_RAD_CUR_D = 16,
  XP_TARG_EXTRA_AUX_RAD_CUR_2D = 32,
  XP_TARG_EXTRA_AUX_TARGET_NADIR_VEL = 64,
  XP_TARG_EXTRA_AUX_MLST = 128,
  XP_TARG_EXTRA_AUX_TLST = 256,
  XP_TARG_EXTRA_AUX_DIST_NAD_TARG_SAT_NAD = 512,
  XP_TARG_EXTRA_AUX_DIST_NAD_TARG_SAT_NAD_D = 1024,
  XP_TARG_EXTRA_AUX_DIST_NAD_TARG_SAT_NAD_2D = 2048,
  XP_TARG_EXTRA_AUX_LOOK_DIR = 4096,
  XP_TARG_EXTRA_AUX_DIST_SSP_MIN_DIST_GT = 8192,
  XP_TARG_EXTRA_AUX_DIST_SSP_MIN_DIST_GT_D = 16384,
  XP_TARG_EXTRA_AUX_DIST_SSP_MIN_DIST_GT_2D = 32768,
  XP_TARG_EXTRA_AUX_ALL = XP_TARG_EXTRA_AUX_DIST_NAD_TARG_GT + XP_TARG_EXTRA_AUX_RAD_CUR + XP_TARG_EXTRA_AUX_DIST_NAD_TARG_GT_D + XP_TARG_EXTRA_AUX_DIST_NAD_TARG_GT_2D + XP_TARG_EXTRA_AUX_RAD_CUR_D + XP_TARG_EXTRA_AUX_RAD_CUR_2D + XP_TARG_EXTRA_AUX_TARGET_NADIR_VEL + XP_TARG_EXTRA_AUX_MLST + XP_TARG_EXTRA_AUX_TLST + XP_TARG_EXTRA_AUX_DIST_NAD_TARG_SAT_NAD + XP_TARG_EXTRA_AUX_DIST_NAD_TARG_SAT_NAD_D + XP_TARG_EXTRA_AUX_DIST_NAD_TARG_SAT_NAD_2D + XP_TARG_EXTRA_AUX_LOOK_DIR + XP_TARG_EXTRA_AUX_DIST_SSP_MIN_DIST_GT + XP_TARG_EXTRA_AUX_DIST_SSP_MIN_DIST_GT_D + XP_TARG_EXTRA_AUX_DIST_SSP_MIN_DIST_GT_2D

} XP_Target_res_aux_name_enum; /* CAREFUL: variables holding these values MUST be declared as long !!! */

/* xp_change_frame mode flag */
/* ------------------------- */
typedef enum
{
  XP_MODE_FLAG_LOCATION = 0,
  XP_MODE_FLAG_DIRECTION

} XP_Mode_flag_enum;

/* xp_change_frame frame flag */
/* -------------------------- */
typedef enum
{
  XP_FRAME_FLAG_EXT = 0,
  XP_FRAME_FLAG_SAT

} XP_Frame_flag_enum;

/* Derivatives */
/* ----------- */
typedef enum
{
  XP_NO_DER = XL_NO_DER,
  XP_DER_1ST = XL_DER_1ST,
  XP_DER_2ND = XL_DER_2ND
} XP_Deriv_enum;

/* Source Types */
/* ------------ */
typedef enum
{
  XP_SOURCE_STAR,
  XP_SOURCE_SUN,
  XP_SOURCE_MOON,
  XP_SOURCE_GENERIC
} XP_Source_type_enum;

/* DEM configuration commands */
/* -------------------------- */
typedef enum
{
  XP_LOAD_TILE_SET,
  XP_CLEAR_CACHE,
  XP_FREE_CACHE,
  XP_SET_MAX_SIZE
} XP_Dem_config_enum;

/* Yaw-flip (AN-353) */
typedef enum
{
  XP_AUTOMATIC_FLIP_MODE = 0,
  XP_WINTER_MODE = 1,
  XP_SUMMER_MODE = 2
} XP_Yaw_flip_enum;

/* AN-466 */
typedef enum
{
  XP_ALL_DEM,
  XP_DEM_SET
} XP_Dem_set_enum;

/* CFI functions IDs (for error handling) */
/* ====================================== */

typedef enum
{

  XP_NO_FUNC_ID = -1, /* No function code */

  XP_SAT_NOMINAL_ATT_INIT_ID,
  XP_SAT_NOMINAL_ATT_INIT_MODEL_ID,
  XP_SAT_NOMINAL_ATT_INIT_HARMONIC_ID,
  XP_SAT_NOMINAL_ATT_INIT_FILE_ID,
  XP_SAT_NOMINAL_ATT_CLOSE_ID,
  XP_SAT_ATT_ANGLE_INIT_ID,
  XP_SAT_ATT_MATRIX_INIT_ID,
  XP_SAT_ATT_INIT_HARMONIC_ID,
  XP_SAT_ATT_INIT_FILE_ID,
  XP_SAT_ATT_QUAT_PLUS_MATRIX_INIT_ID,
  XP_SAT_ATT_QUAT_PLUS_ANGLE_INIT_ID,
  XP_SAT_ATT_CLOSE_ID,
  XP_INSTR_ATT_ANGLE_INIT_ID,
  XP_INSTR_ATT_MATRIX_INIT_ID,
  XP_INSTR_ATT_INIT_HARMONIC_ID,
  XP_INSTR_ATT_INIT_FILE_ID,
  XP_INSTR_ATT_CLOSE_ID,

  XP_CHANGE_FRAME_ID,
  XP_ATTITUDE_INIT_ID,
  XP_ATTITUDE_COMPUTE_ID,
  XP_ATTITUDE_USER_SET_ID,
  XP_ATTITUDE_CLOSE_ID,
  XP_SET_AZ_EL_DEFINITION_ID,

  XP_ATMOS_INIT_ID,
  XP_ATMOS_CLOSE_ID,
  XP_DEM_INIT_ID,
  XP_DEM_COMPUTE_ID,
  XP_DEM_CLOSE_ID,
  XP_DEM_GET_INFO_ID,
  XP_DEM_ID_CONFIGURE_ID, /* AN-487 */
  XP_DEM_GET_CELL_VALUE_ID, /* AN-778 */
  XP_DEM_GET_CELL_GEOD_ID, /* AN-778 */

  XP_TARGET_INTER_ID,
  XP_TARGET_TRAVEL_TIME_ID,
  XP_TARGET_GROUND_RANGE_ID,
  XP_TARGET_INCIDENCE_ANGLE_ID,
  XP_TARGET_RANGE_ID,
  XP_TARGET_RANGE_RATE_ID,
  XP_TARGET_TANGENT_ID,
  XP_TARGET_ALTITUDE_ID,
  XP_TARGET_STAR_ID,
  XP_TARGET_STATION_ID,
  XP_TARGET_GENERIC_ID,
  XP_TARGET_REFLECTED_ID,
  XP_TARGET_SC_ID,
  XP_MULTI_TARGET_INTER_ID,
  XP_MULTI_TARGET_TRAVEL_TIME_ID,
  XP_TARGET_LIST_INTER_ID, /* AN-470 */
  XP_TARGET_EXTRA_VECTOR_ID,
  XP_TARGET_EXTRA_MAIN_ID,
  XP_TARGET_EXTRA_AUX_ID,
  XP_TARGET_EXTRA_EF_TARGET_ID,
  XP_TARGET_EXTRA_TARGET_TO_SUN_ID,
  XP_TARGET_EXTRA_TARGET_TO_MOON_ID,
  XP_TARGET_EXTRA_SPECULAR_REFLECTION_ID,
  XP_TARGET_LIST_EXTRA_VECTOR_ID, /* AN-471 */
  XP_TARGET_LIST_EXTRA_MAIN_ID, /* AN-471 */
  XP_TARGET_LIST_EXTRA_AUX_ID, /* AN-471 */
  XP_TARGET_LIST_EXTRA_EF_TARGET_ID, /* AN-471 */
  XP_TARGET_LIST_EXTRA_SPECULAR_REFLECT_ID, /* AN-471 */
  XP_TARGET_LIST_EXTRA_TARGET_TO_SUN_ID, /* AN-471 */
  XP_TARGET_LIST_EXTRA_TARGET_TO_MOON_ID, /* AN-471 */
  XP_TARGET_TANGENT_SUN_ID,
  XP_TARGET_TANGENT_MOON_ID,
  XP_TARGET_CLOSE_ID,

  XP_GEN_DEM_MAX_ALTITUDE_ID, /* AN-492 */

  XP_GEN_DEM_ALTITUDE_FROM_ELLIPSOID_ID, /* AN-466 */

  XP_ATTITUDE_DEFINE_ID, /* AN-469 */

  XP_ATTITUDE_TRANSFORM_ID, /* AN-526 */
  XP_GET_ATTITUDE_DATA_ID, /* AN-597 */
  XP_GEN_ATTITUDE_DATA_ID, /* AN-596 */
  XP_GEN_ATTITUDE_FILE_ID, /* AN-596 */

  XP_RUN_INIT_ID,

  XP_MAX_FUNC_ID /* Number of functions defined in this list */

} XP_Func_enum;

/* AN-470 */
typedef enum
{
  XP_AZ_EL_LIST,
  XP_AZ_EL_STRIP,
  XP_AZ_EL_GRID
} XP_Az_el_type_enum;
/* END AN-470 */

/*******************************************************************************
*
* List of errors of the main CFI functions (Refer to the EXPLORER_POINTING SUM)
*
*******************************************************************************/

typedef enum /* error codes list of "xp_sat_nominal_att_init"  */
{
  XP_CFI_SAT_NOMINAL_ATT_INIT_MEMORY_ERR = XL_ZERO_E
} XP_CFI_Sat_nom_att_init_err_enum;

typedef enum /* error codes list of "xp_sat_nominal_att_init_model"  */
{
  XP_CFI_SAT_NOMINAL_ATT_INIT_MODEL_MEMORY_ERR = XL_ZERO_E
} XP_CFI_Sat_nom_att_init_model_err_enum;

typedef enum /* error codes list of "xp_sat_nominal_att_init_harmonic"  */
{
  XP_CFI_SAT_NOMINAL_ATT_INIT_HARMONIC_MEMORY_ERR = XL_ZERO_E,
  XP_CFI_SAT_NOMINAL_ATT_INIT_HARMONIC_MAX_TERMS_ERR
} XP_CFI_Sat_nom_att_init_harmonic_err_enum;

typedef enum /* error codes list of "xp_sat_nominal_att_init_file"  */
{
  XP_CFI_SAT_NOMINAL_ATT_INIT_FILE_MEMORY_ERR = XL_ZERO_E,
  XP_CFI_SAT_NOMINAL_ATT_INIT_FILE_WRONG_TIME_REF_ERR,
  XP_CFI_SAT_NOMINAL_ATT_INIT_FILE_OPEN_FILES_ERR,
  XP_CFI_SAT_NOMINAL_ATT_INIT_FILE_READ_ATT_FILES_ERR,
  XP_CFI_SAT_NOMINAL_ATT_INIT_FILE_TIME_CONV_ERR,
  XP_CFI_SAT_NOMINAL_ATT_INIT_FILE_WRONG_FILE_FORMAT_ERR, // AN-835
  XP_CFI_SAT_NOMINAL_ATT_INIT_FILE_REPEATED_SAMPLES_WARN
} XP_CFI_Sat_nom_att_init_file_err_enum;

typedef enum /* error codes list of "xp_sat_nominal_att_close"  */
{
  XP_CFI_SAT_NOMINAL_ATT_CLOSE_WRONG_ID_ERR = XL_ZERO_E
} XP_CFI_Sat_nom_att_close_err_enum;

typedef enum /* error codes list of "xp_sat_att_angle_init"  */
{
  XP_CFI_SAT_ATT_ANGLE_INIT_MEMORY_ERR = XL_ZERO_E
} XP_CFI_Sat_att_angle_init_err_enum;

typedef enum /* error codes list of "xp_sat_att_matrix_init"  */
{
  XP_CFI_SAT_ATT_MATRIX_INIT_MEMORY_ERR = XL_ZERO_E,
  XP_CFI_SAT_ATT_MATRIX_INIT_ORTHONORMAL_ERR
} XP_CFI_Sat_att_matrix_init_err_enum;

typedef enum /* error codes list of "xp_sat_att_init_harmonic"  */
{
  XP_CFI_SAT_ATT_INIT_HARMONIC_MEMORY_ERR = XL_ZERO_E,
  XP_CFI_SAT_ATT_INIT_HARMONIC_MAX_TERMS_ERR
} XP_CFI_Sat_att_init_harmonic_err_enum;

typedef enum /* error codes list of "xp_sat_att_init_file"  */
{
  XP_CFI_SAT_ATT_INIT_FILE_MEMORY_ERR = XL_ZERO_E,
  XP_CFI_SAT_ATT_INIT_FILE_OPEN_FILES_ERR,
  XP_CFI_SAT_ATT_INIT_FILE_READ_FILES_ERR,
  XP_CFI_SAT_ATT_INIT_FILE_READ_ATT_FILES_ERR,
  XP_CFI_SAT_ATT_INIT_FILE_NO_READ_DATA_ERR,
  XP_CFI_SAT_ATT_INIT_FILE_READ_AUX_FILE_ERR,
  XP_CFI_SAT_ATT_INIT_FILE_WRONG_TIME_REF_ERR,
  XP_CFI_SAT_ATT_INIT_FILE_TIME_REF_ERR,
  XP_CFI_SAT_ATT_INIT_FILE_READ_STR_ID_ERR,
  XP_CFI_SAT_ATT_INIT_FILE_QUAT_UNITARY_ERR,
  XP_CFI_SAT_ATT_INIT_FILE_REPEATED_SAMPLES_WARN
} XP_CFI_Sat_att_init_file_err_enum;

typedef enum /* error codes list of "xp_sat_att_quat_plus_matrix_init"  */
{
  XP_CFI_SAT_ATT_QUAT_PLUS_MATRIX_INIT_MEMORY_ERR = XL_ZERO_E,
  XP_CFI_SAT_ATT_QUAT_PLUS_MATRIX_INIT_QUAT_UNITARY_ERR,
  XP_CFI_SAT_ATT_QUAT_PLUS_MATRIX_INIT_MATRIX_ORTHONORMAL_ERR
} XP_CFI_Sat_att_quat_plus_matrix_init_err_enum;

typedef enum /* error codes list of "xp_sat_att_quat_plus_angle_init"  */
{
  XP_CFI_SAT_ATT_QUAT_PLUS_ANGLE_INIT_MEMORY_ERR = XL_ZERO_E,
  XP_CFI_SAT_ATT_QUAT_PLUS_ANGLE_EULER_TO_MATRIX_ERR,
  XP_CFI_SAT_ATT_QUAT_PLUS_ANGLE_INIT_QUAT_UNITARY_ERR
} XP_CFI_Sat_att_quat_plus_angle_init_err_enum;

typedef enum /* error codes list of "xp_sat_att_close"  */
{
  XP_CFI_SAT_ATT_CLOSE_WRONG_ID_ERR = XL_ZERO_E
} XP_CFI_Sat_att_close_err_enum;

typedef enum /* error codes list of "xp_instr_att_angle_init"  */
{
  XP_CFI_INSTR_ATT_ANGLE_INIT_MEMORY_ERR = XL_ZERO_E
} XP_CFI_Instr_att_angle_init_err_enum;

typedef enum /* error codes list of "xp_instr_att_matrix_init"  */
{
  XP_CFI_INSTR_ATT_MATRIX_INIT_MEMORY_ERR = XL_ZERO_E,
  XP_CFI_INSTR_ATT_MATRIX_INIT_MATRIX_ORTHONORMAL_ERR
} XP_CFI_Instr_att_matrix_init_err_enum;

typedef enum /* error codes list of "xp_instr_att_init_harmonic"  */
{
  XP_CFI_INSTR_ATT_INIT_HARMONIC_MEMORY_ERR = XL_ZERO_E,
  XP_CFI_INSTR_ATT_INIT_HARMONIC_MAX_TERMS_ERR
} XP_CFI_Instr_att_init_harmonic_err_enum;

typedef enum /* error codes list of "xp_instr_att_init_file"  */
{
  XP_CFI_INSTR_ATT_INIT_FILE_MEMORY_ERR = XL_ZERO_E,
  XP_CFI_INSTR_ATT_INIT_FILE_WRONG_TIME_REF_ERR,
  XP_CFI_INSTR_ATT_INIT_FILE_OPEN_FILES_ERR,
  XP_CFI_INSTR_ATT_INIT_FILE_READ_ATT_FILES_ERR,
  XP_CFI_INSTR_ATT_INIT_FILE_TIME_CONV_ERR,
  XP_CFI_INSTR_ATT_INIT_FILE_WRONG_FILE_FORMAT_ERR, // AN-836
  XP_CFI_INSTR_ATT_INIT_FILE_REPEATED_SAMPLES_WARN
} XP_CFI_Instr_att_init_file_err_enum;

typedef enum /* error codes list of "xp_instr_att_close"  */
{
  XP_CFI_INSTR_ATT_CLOSE_WRONG_ID_ERR = XL_ZERO_E
} XP_CFI_Instr_att_close_err_enum;

typedef enum /* error codes list of "xp_change_frame"  */
{
  XP_CFI_CHANGE_FRAME_ATTITUDE_INIT_ERR = XL_ZERO_E,
  XP_CFI_CHANGE_FRAME_INPUT_FRAME_ERR,
  XP_CFI_CHANGE_FRAME_OUTPUT_FRAME_ERR,
  XP_CFI_CHANGE_FRAME_CHANGE_CART_CS_ERR,
  XP_CFI_CHANGE_FRAME_ATTITUDE_COMP_ERR,
  XP_CFI_CHANGE_FRAME_ATTITUDE_CLOSE_ERR

} XP_CFI_Change_frame_err_enum;

typedef enum /* error codes list of "xp_attitude_init"  */
{
  XP_CFI_ATTITUDE_INIT_MEMORY_ERR = XL_ZERO_E
} XP_CFI_Attitude_init_err_enum;

typedef enum /* error codes list of "xp_attitude_compute"  */
{
  XP_CFI_ATTITUDE_COMPUTE_TIME_STATUS_ERR = XL_ZERO_E,
  XP_CFI_ATTITUDE_COMPUTE_INSTR_TRANS_STATUS_ERR,
  XP_CFI_ATTITUDE_COMPUTE_SAT_TRANS_STATUS_ERR,
  XP_CFI_ATTITUDE_COMPUTE_SAT_NOM_TRANS_STATUS_ERR,
  XP_CFI_ATTITUDE_COMPUTE_ATTITUDE_STATUS_ERR,
  XP_CFI_ATTITUDE_COMPUTE_WRONG_TIME_REF_ERR,
  XP_CFI_ATTITUDE_COMPUTE_BEING_USED_ERR,
  XP_CFI_ATTITUDE_COMPUTE_ORB_REF_ERR,
  XP_CFI_ATTITUDE_COMPUTE_AOCS_CALC_ERR,
  XP_CFI_ATTITUDE_COMPUTE_SAT_NOM_TRANS_ERR,
  XP_CFI_ATTITUDE_COMPUTE_TRUE_LAT_ERR,
  XP_CFI_ATTITUDE_COMPUTE_HARMONIC_CALC_ERR,
  XP_CFI_ATTITUDE_COMPUTE_SAT_TRANS_ERR,
  XP_CFI_ATTITUDE_COMPUTE_ATT_TO_J2000_ERR,
  XP_CFI_ATTITUDE_COMPUTE_INSTR_TRANS_ERR,
  XP_CFI_ATTITUDE_COMPUTE_MEMORY_ERR,
  XP_CFI_ATTITUDE_COMPUTE_SAME_TARGETS_ERR,
  XP_CFI_ATTITUDE_COMPUTE_VEC_FIND_ERR,
  XP_CFI_ATTITUDE_COMPUTE_CREATE_BASE_ERR,
  XP_CFI_ATTITUDE_COMPUTE_ATT_FILE_INTER_ERR,
  XP_CFI_ATTITUDE_COMPUTE_CHANGE_CS_ERR,
  XP_CFI_ATTITUDE_COMPUTE_VEC_FIND_WARN,
  XP_CFI_ATTITUDE_COMPUTE_SUN_ERR,
  XP_CFI_ATTITUDE_COMPUTE_CHANGE_TIME_ERR, //AN-604
  XP_CFI_ATTITUDE_COMPUTE_QUATERNION_INTERPOL_INPUT_TIME_OUT_WARN // AN-686
} XP_CFI_Attitude_compute_err_enum;

typedef enum /* error codes list of "xp_attitude_user_set"  */
{
  XP_CFI_ATTITUDE_USER_SET_TIME_STATUS_ERR = XL_ZERO_E,
  XP_CFI_ATTITUDE_USER_SET_WRONG_TARGET_FRAME_ERR,
  XP_CFI_ATTITUDE_USER_SET_ATTITUDE_STATUS_ERR,
  XP_CFI_ATTITUDE_USER_SET_BEING_USED_ERR,
  XP_CFI_ATTITUDE_USER_SET_ORB_REF_ERR,
  XP_CFI_ATTITUDE_USER_SET_MEMORY_ERR,
  XP_CFI_ATTITUDE_USER_SET_MATRIX_ORTHONORMAL_ERR
} XP_CFI_Attitude_user_set_err_enum;

typedef enum /* error codes list of "xp_attitude_close"  */
{
  XP_CFI_ATTITUDE_CLOSE_WRONG_ID_ERR = XL_ZERO_E
} XP_CFI_Attitude_close_err_enum;

typedef enum /* error codes list of "xp_atmos_init"  */
{
  XP_CFI_ATMOS_INIT_MODE_ID_ERR = XL_ZERO_E,
  XP_CFI_ATMOS_INIT_MODEL_ID_ERR,
  XP_CFI_ATMOS_INIT_FILE_NOT_OPEN_ERR,
  XP_CFI_ATMOS_INIT_MEMORY_ERR,
  XP_CFI_ATMOS_INIT_FILE_READING_ERR
} XP_CFI_Atmos_init_err_enum;

typedef enum /* error codes list of "xp_atmos_close"  */
{
  XP_CFI_ATMOS_CLOSE_WRONG_ID_ERR = XL_ZERO_E,
} XP_CFI_Atmos_close_err_enum;

typedef enum /* error codes list of "xp_dem_init"  */
{
  XP_CFI_DEM_INIT_MODE_ID_ERR = XL_ZERO_E,
  XP_CFI_DEM_INIT_MODEL_ID_ERR,
  XP_CFI_DEM_INIT_FILE_NOT_OPEN_ERR,
  XP_CFI_DEM_INIT_MEMORY_ERR,
  XP_CFI_DEM_INIT_FILE_READING_ERR,
  XP_CFI_DEM_INIT_FILE_READING_WARN,
  XP_CFI_DEM_INIT_WRONG_MODEL_WARN, // AN-594
  XP_CFI_DEM_INIT_DEPRECATED_WARN,
  XP_CFI_DEM_INIT_CACHE_WARN,
  XP_CFI_DEM_INIT_TILE_DB_ERR, /* AN-487 */
  XP_CFI_DEM_INIT_READ_POLES_ERR,
  XP_CFI_DEM_INIT_READ_POLES_WARN,
  XP_CFI_DEM_INIT_MINI_TILE_LON_SIZE_ERR, /* AN-492 */
  XP_CFI_DEM_INIT_MINI_TILE_LAT_SIZE_ERR, /* AN-492 */
  XP_CFI_DEM_INIT_OPEN_MAX_ALT_FILE_ERR, /* AN-492 */
  XP_CFI_DEM_INIT_READ_MAX_ALT_FILE_ERR /* AN-492 */
} XP_CFI_DEM_init_err_enum;

typedef enum /* error codes list of "xp_dem_compute"  */
{
  XP_CFI_DEM_COMPUTE_GET_CELL_ERR = XL_ZERO_E,
  XP_CFI_DEM_COMPUTE_MEMORY_ERR,
  XP_CFI_DEM_COMPUTE_VOID_VALUE_DETECTED_WARN /* ANR-746 */
} XP_CFI_DEM_compute_err_enum;

typedef enum /* error codes list of "xp_dem_close"  */
{
  XP_CFI_DEM_CLOSE_WRONG_ID_ERR = XL_ZERO_E,
} XP_CFI_DEM_close_err_enum;

typedef enum /* error codes list of "xp_dem_get_info"  */
{
  XP_CFI_DEM_GET_INFO_STATUS_ERR = XL_ZERO_E,
  XP_CFI_DEM_GET_INFO_WRONG_LONGITUDE_ERR,
  XP_CFI_DEM_GET_INFO_WRONG_LATITUDE_ERR,
  XP_CFI_DEM_GET_INFO_OPEN_FILE_ERR,
  XP_CFI_DEM_GET_INFO_READ_FILE_ERR,
  XP_CFI_DEM_GET_INFO_READ_RASTER_ERR
} XP_CFI_DEM_get_info_err_enum;

typedef enum /* error codes list of "xp_dem_get_cell_value"  */
{
  XP_CFI_DEM_GET_CELL_VALUE_READ_POINT_ERR = XL_ZERO_E,
  XP_CFI_DEM_GET_CELL_VALUE_WRONG_ABS_ROW_ERR,
  XP_CFI_DEM_GET_CELL_VALUE_WRONG_ABS_COL_ERR,
  XP_CFI_DEM_GET_CELL_VALUE_WRONG_ROW_ERR,
  XP_CFI_DEM_GET_CELL_VALUE_WRONG_COLUMN_ERR,
  XP_CFI_DEM_GET_CELL_VALUE_GET_GEOID_UNDU_ERR,
  XP_CFI_DEM_GET_CELL_VALUE_READ_CACHE_POINT_ERR
} XP_CFI_DEM_get_cell_value_err_enum;

typedef enum /* error codes list of "xp_dem_get_cell_geod"  */
{
  XP_CFI_DEM_GET_CELL_GEOD_DEM_STATUS_ERR = XL_ZERO_E
} XP_CFI_DEM_get_cell_geod_err_enum;

typedef enum /* error codes list of "xp_target_inter"  */
{
  XP_CFI_TARGET_INTER_ATTITUDE_STATUS_ERR = XL_ZERO_E,
  XP_CFI_TARGET_INTER_INTER_FLAG_ERR,
  XP_CFI_TARGET_INTER_FREQ_ERR,
  XP_CFI_TARGET_INTER_ATM_NOT_INIT_ERR,
  XP_CFI_TARGET_INTER_TIME_REF_ERR,
  XP_CFI_TARGET_INTER_DERIV_FLAG_ERR,
  XP_CFI_TARGET_INTER_IRAY_ID_ERR,
  XP_CFI_TARGET_INTER_LOS_AZIMUTH_ERR,
  XP_CFI_TARGET_INTER_LOS_ELEVATION_ERR,
  XP_CFI_TARGET_INTER_GEODETIC_ALT_ERR,
  XP_CFI_TARGET_INTER_MEMORY_ERR,
  XP_CFI_TARGET_INTER_INITIAL_LOOK_DIR_OR_PLANE_ERR,
  XP_CFI_TARGET_INTER_TIME_REF_INIT_ERR,
  XP_CFI_TARGET_INTER_TARGET_NOT_FOUND_ERR,
  XP_CFI_TARGET_INTER_RANGE_OR_POINTING_CALC_ERR,
  XP_CFI_TARGET_INTER_NEGATIVE_ALTITUDE_WARN,
  XP_CFI_TARGET_INTER_IRAY_ID_WARN,
  XP_CFI_TARGET_INTER_EF_TO_EF_ERR, // AN-553
  XP_CFI_TARGET_INTER_LINK_ID_ERR, // AN-553
  XP_CFI_TARGET_INTER_LIGHT_MODE_NOT_SUPPORTED_WARN // AN-553
} XP_CFI_Target_inter_err_enum;

typedef enum /* error codes list of "xp_target_travel_time"  */
{
  XP_CFI_TARGET_TRAVEL_TIME_ATTITUDE_STATUS_ERR = XL_ZERO_E,
  XP_CFI_TARGET_TRAVEL_TIME_INTER_FLAG_ERR,
  XP_CFI_TARGET_TRAVEL_TIME_FREQ_ERR,
  XP_CFI_TARGET_TRAVEL_TIME_TIME_REF_ERR,
  XP_CFI_TARGET_TRAVEL_TIME_DERIV_FLAG_ERR,
  XP_CFI_TARGET_TRAVEL_TIME_IRAY_ID_ERR,
  XP_CFI_TARGET_TRAVEL_TIME_LOS_AZIMUTH_ERR,
  XP_CFI_TARGET_TRAVEL_TIME_LOS_ELEVATION_ERR,
  XP_CFI_TARGET_TRAVEL_TIME_GEODETIC_ALT_ERR,
  XP_CFI_TARGET_TRAVEL_TIME_MEMORY_ERR,
  XP_CFI_TARGET_TRAVEL_TIME_INITIAL_LOOK_DIR_OR_PLANE_ERR,
  XP_CFI_TARGET_TRAVEL_TIME_TIME_REF_INIT_ERR,
  XP_CFI_TARGET_TRAVEL_TIME_TARGET_NOT_FOUND_ERR,
  XP_CFI_TARGET_TRAVEL_TIME_RANGE_OR_POINTING_CALC_ERR,
  XP_CFI_TARGET_TRAVEL_TIME_NEGATIVE_ALTITUDE_WARN,
  XP_CFI_TARGET_TRAVEL_TIME_IRAY_ID_WARN
} XP_CFI_Target_travel_time_err_enum;

typedef enum /* error codes list of "xp_target_ground_range"  */
{
  XP_CFI_TARGET_GR_RANGE_ATTITUDE_STATUS_ERR = XL_ZERO_E,
  XP_CFI_TARGET_GR_RANGE_TIME_REF_ERR,
  XP_CFI_TARGET_GR_RANGE_DERIV_FLAG_ERR,
  XP_CFI_TARGET_GR_RANGE_LOS_AZIMUTH_ERR,
  XP_CFI_TARGET_GR_RANGE_LOS_ELEVATION_ERR,
  XP_CFI_TARGET_GR_RANGE_GEODETIC_ALT_ERR,
  XP_CFI_TARGET_GR_RANGE_MEMORY_ERR,
  XP_CFI_TARGET_GR_RANGE_INITIAL_LOOK_DIR_OR_PLANE_ERR,
  XP_CFI_TARGET_GR_RANGE_TIME_REF_INIT_ERR,
  XP_CFI_TARGET_GR_RANGE_TARGET_NOT_FOUND_ERR,
  XP_CFI_TARGET_GR_RANGE_RANGE_OR_POINTING_CALC_ERR,
  XP_CFI_TARGET_GR_RANGE_EF_TO_EF_ERR //AN-553
} XP_CFI_Target_ground_range_err_enum;

typedef enum /* error codes list of "xp_target_incidence_angle"  */
{
  XP_CFI_TARGET_INC_ANGLE_ATTITUDE_STATUS_ERR = XL_ZERO_E,
  XP_CFI_TARGET_INC_ANGLE_DERIV_FLAG_ERR,
  XP_CFI_TARGET_INC_ANGLE_LOS_AZIMUTH_ERR,
  XP_CFI_TARGET_INC_ANGLE_INC_ANGLE_ERR,
  XP_CFI_TARGET_INC_ANGLE_GEODETIC_ALT_ERR,
  XP_CFI_TARGET_INC_ANGLE_MEMORY_ERR,
  XP_CFI_TARGET_INC_ANGLE_INITIAL_LOOK_DIR_OR_PLANE_ERR,
  XP_CFI_TARGET_INC_ANGLE_TIME_REF_INIT_ERR,
  XP_CFI_TARGET_INC_ANGLE_TARGET_NOT_FOUND_ERR,
  XP_CFI_TARGET_INC_ANGLE_EF_TO_EF_ERR //AN-553
} XP_CFI_Target_incidence_angle_err_enum;

typedef enum /* error codes list of "xp_target_range"  */
{
  XP_CFI_TARGET_RANGE_ATTITUDE_STATUS_ERR = XL_ZERO_E,
  XP_CFI_TARGET_RANGE_DERIV_FLAG_ERR,
  XP_CFI_TARGET_RANGE_LOS_AZIMUTH_ERR,
  XP_CFI_TARGET_RANGE_RANGE_ERR,
  XP_CFI_TARGET_RANGE_GEODETIC_ALT_ERR,
  XP_CFI_TARGET_RANGE_MEMORY_ERR,
  XP_CFI_TARGET_RANGE_INITIAL_LOOK_DIR_OR_PLANE_ERR,
  XP_CFI_TARGET_RANGE_TIME_TRANSFORMATION_ERR,
  XP_CFI_TARGET_RANGE_TARGET_NOT_FOUND_ERR,
  XP_CFI_TARGET_RANGE_CHANGE_COORDINATES_ERR, // AN-553
  XP_CFI_TARGET_RANGE_EF_TO_EF_ERR // AN-553
} XP_CFI_Target_range_err_enum;

typedef enum /* error codes list of "xp_target_range_rate"  */
{
  XP_CFI_TARGET_RANGE_RATE_ATTITUDE_STATUS_ERR = XL_ZERO_E,
  XP_CFI_TARGET_RANGE_RATE_DERIV_FLAG_ERR,
  XP_CFI_TARGET_RANGE_RATE_RANGE_ERR,
  XP_CFI_TARGET_RANGE_RATE_GEODETIC_ALT_ERR,
  XP_CFI_TARGET_RANGE_RATE_MEMORY_ERR,
  XP_CFI_TARGET_RANGE_RATE_TIME_REF_INIT_ERR,
  XP_CFI_TARGET_RANGE_RATE_INITIAL_LOOK_DIR_OR_PLANE_ERR,
  XP_CFI_TARGET_RANGE_RATE_TARGET_NOT_FOUND_ERR,
  XP_CFI_TARGET_RANGE_RATE_DERIV_FLAG_WARN,
  XP_CFI_TARGET_RANGE_RATE_EF_TO_EF_ERR, //AN-553
  XP_CFI_TARGET_RANGE_RATE_ECLIPSE_ERR
} XP_CFI_Target_range_rate_err_enum;

typedef enum /* error codes list of "xp_target_tangent"  */
{
  XP_CFI_TARGET_TANGENT_ATTITUDE_STATUS_ERR = XL_ZERO_E,
  XP_CFI_TARGET_TANGENT_DERIV_FLAG_ERR,
  XP_CFI_TARGET_TANGENT_LOS_AZIMUTH_ERR,
  XP_CFI_TARGET_TANGENT_LOS_ELEVATION_ERR,
  XP_CFI_TARGET_TANGENT_IRAY_ID_ERR,
  XP_CFI_TARGET_TANGENT_FREQ_ERR,
  XP_CFI_TARGET_TANGENT_MEMORY_ERR,
  XP_CFI_TARGET_TANGENT_INITIAL_LOOK_DIR_OR_PLANE_ERR,
  XP_CFI_TARGET_TANGENT_TIME_REF_INIT_ERR,
  XP_CFI_TARGET_TANGENT_TARGET_NOT_FOUND_ERR,
  XP_CFI_TARGET_TANGENT_TG_PT_BEHIND_LOOK_DIR_ERR,
  XP_CFI_TARGET_TANGENT_RANGE_OR_POINTING_CALC_ERR,
  XP_CFI_TARGET_TANGENT_NEGATIVE_ALTITUDE_WARN,
  XP_CFI_TARGET_TANGENT_PRED_WRONG_LAT_WARN,
  XP_CFI_TARGET_TANGENT_TG_PT_BEHIND_LOOK_DIR_WARN,
  XP_CFI_TARGET_TANGENT_IRAY_ID_WARN,
  XP_CFI_TARGET_TANGENT_EF_TO_EF_ERR, //AN-553
  XP_CFI_TARGET_TANGENT_LIGHT_MODE_NOT_SUPPORTED_WARN //AN-553
} XP_CFI_Target_tangent_err_enum;

typedef enum /* error codes list of "xp_target_altitude"  */
{
  XP_CFI_TARGET_ALT_ATTITUDE_STATUS_ERR = XL_ZERO_E,
  XP_CFI_TARGET_ALT_DERIV_FLAG_ERR,
  XP_CFI_TARGET_ALT_LOS_AZIMUTH_ERR,
  XP_CFI_TARGET_ALT_GEODETIC_ALT_ERR,
  XP_CFI_TARGET_ALT_IRAY_ID_ERR,
  XP_CFI_TARGET_ALT_FREQ_ERR,
  XP_CFI_TARGET_ALT_MEMORY_ERR,
  XP_CFI_TARGET_ALT_TIME_REF_INIT_ERR,
  XP_CFI_TARGET_ALT_INITIAL_LOOK_DIR_OR_PLANE_ERR,
  XP_CFI_TARGET_ALT_TARGET_NOT_FOUND_ERR,
  XP_CFI_TARGET_ALT_RANGE_OR_POINTING_CALC_ERR,
  XP_CFI_TARGET_ALT_NEGATIVE_ALTITUDE_WARN,
  XP_CFI_TARGET_ALT_PRED_WRONG_LAT_WARN,
  XP_CFI_TARGET_ALT_IRAY_ID_WARN,
  XP_CFI_TARGET_ALT_INACCURATE_RESULT_WARN, /* AN-420 */
  XP_CFI_TARGET_ALT_RANGE_EF_TO_EF_ERR, /* AN-553 */
  XP_CFI_TARGET_ALT_LIGHT_MODE_NOT_SUPPORTED_WARN /* AN-553 */
} XP_CFI_Target_altitude_err_enum;

typedef enum /* error codes list of "xp_target_star"  */
{
  XP_CFI_TARGET_STAR_ATTITUDE_STATUS_ERR = XL_ZERO_E,
  XP_CFI_TARGET_STAR_DERIV_FLAG_ERR,
  XP_CFI_TARGET_STAR_RA_ERR,
  XP_CFI_TARGET_STAR_DEC_ERR,
  XP_CFI_TARGET_STAR_IRAY_ID_ERR,
  XP_CFI_TARGET_STAR_FREQ_ERR,
  XP_CFI_TARGET_STAR_MEMORY_ERR,
  XP_CFI_TARGET_STAR_INITIAL_LOOK_DIR_OR_PLANE_ERR,
  XP_CFI_TARGET_STAR_TIME_REF_INIT_ERR,
  XP_CFI_TARGET_STAR_TARGET_NOT_FOUND_ERR,
  XP_CFI_TARGET_STAR_TG_PT_BEHIND_LOOK_DIR_ERR,
  XP_CFI_TARGET_STAR_RANGE_OR_POINTING_CALC_ERR,
  XP_CFI_TARGET_STAR_NEGATIVE_ALTITUDE_WARN,
  XP_CFI_TARGET_STAR_PRED_WRONG_LAT_WARN,
  XP_CFI_TARGET_STAR_TG_PT_BEHIND_LOOK_DIR_WARN,
  XP_CFI_TARGET_STAR_IRAY_ID_WARN,
  XP_CFI_TARGET_STAR_EF_TO_EF_ERR, //AN-553
  XP_CFI_TARGET_STAR_LIGHT_MODE_NOT_SUPPORTED_WARN //AN-553
} XP_CFI_Target_star_err_enum;

typedef enum /* error codes list of "xp_target_station"  */
{
  XP_CFI_TARGET_STATION_ATTITUDE_STATUS_ERR = XL_ZERO_E,
  XP_CFI_TARGET_STATION_DERIV_FLAG_ERR,
  XP_CFI_TARGET_STATION_GEOC_LONG_ERR,
  XP_CFI_TARGET_STATION_GEOD_LAT_ERR,
  XP_CFI_TARGET_STATION_GEODETIC_ALT_ERR,
  XP_CFI_TARGET_STATION_MIN_LINK_EL_ERR,
  XP_CFI_TARGET_STATION_MEMORY_ERR,
  XP_CFI_TARGET_STATION_STAVIS_COMP_FAILED_ERR
} XP_CFI_Target_station_err_enum;

typedef enum /* error codes list of "xp_target_generic"  */
{
  XP_CFI_TARGET_GENERIC_ATTITUDE_STATUS_ERR = XL_ZERO_E,
  XP_CFI_TARGET_GENERIC_DERIV_FLAG_ERR,
  XP_CFI_TARGET_GENERIC_MEMORY_ERR,
  XP_CFI_TARGET_GENERIC_INITIAL_LOOK_DIR_OR_PLANE_ERR,
  XP_CFI_TARGET_GENERIC_EF_TO_EF_ERR // AN-553
} XP_CFI_Target_generic_err_enum;

typedef enum /* error codes list of "xp_target_reflected"  */
{
  XP_CFI_TARGET_REFLECTED_ATTITUDE_STATUS_ERR = XL_ZERO_E,
  XP_CFI_TARGET_REFLECTED_TARGET_STAR_ERR,
  XP_CFI_TARGET_REFLECTED_TARGET_TG_SUN_ERR,
  XP_CFI_TARGET_REFLECTED_TARGET_TG_MOON_ERR,
  XP_CFI_TARGET_REFLECTED_CHANGE_CS_ERR,
  XP_CFI_TARGET_REFLECTED_DIR_POINT_ERR,
  XP_CFI_TARGET_REFLECTED_TARGET_TG_ERR,
  XP_CFI_TARGET_REFLECTED_WRONG_SRC_TYPE_ERR,
  XP_CFI_TARGET_REFLECTED_TGT_EXTRA_MAIN_ERR,
  XP_CFI_TARGET_REFLECTED_WRONG_ALTITUDE_ERR,
  XP_CFI_TARGET_REFLECTED_RADEC_TO_CART_ERR,
  XP_CFI_TARGET_REFLECTED_SUN_ERR,
  XP_CFI_TARGET_REFLECTED_MOON_ERR,
  XP_CFI_TARGET_REFLECTED_ITER_ERR,
  XP_CFI_TARGET_REFLECTED_SAT_NOM_ATT_INIT_ERR,
  XP_CFI_TARGET_REFLECTED_ATT_INIT_ERR,
  XP_CFI_TARGET_REFLECTED_ATT_COMPUTE_ERR,
  XP_CFI_TARGET_REFLECTED_TGT_EXTRA_REF_ERR,
  XP_CFI_TARGET_REFLECTED_TARGET_INTER_ERR,
  XP_CFI_TARGET_REFLECTED_TARGET_EXTRA_ERR,
  XP_CFI_TARGET_REFLECTED_TARGET_GENERIC_ERR,
  XP_CFI_TARGET_REFLECTED_EF_TO_TOP_ERR,
  XP_CFI_TARGET_REFLECTED_TOP_TO_EF_ERR,
  XP_CFI_TARGET_REFLECTED_TGT_CLOSE_ERR,
  XP_CFI_TARGET_REFLECTED_ATT_CLOSE_ERR,
  XP_CFI_TARGET_REFLECTED_MEMORY_ERR,
  XP_CFI_TARGET_REFLECTED_RANGE_CALC_ERR,
  XP_CFI_TARGET_REFLECTED_DEGRADED_SOL_WARN // AN-647
} XP_CFI_Target_reflected_err_enum;

typedef enum /* error codes list of "xp_target_sc" (AN-353) */
{
  XP_CFI_TARGET_SC_ATTITUDE_STATUS_ERR = XL_ZERO_E,
  XP_CFI_TARGET_SC_DERIV_FLAG_ERR,
  XP_CFI_TARGET_SC_TARGET_GENERIC_ERR,
  XP_CFI_TARGET_SC_CHANGE_FRAME_ERR,
  XP_CFI_TARGET_SC_POINT_PARAM_COMPUTE_ERR
} XP_CFI_Target_sc_err_enum;

typedef enum /* error codes list of "xp_multi_target_inter"  */
{
  XP_CFI_MULTI_TARGET_INTER_ATTITUDE_STATUS_ERR = XL_ZERO_E,
  XP_CFI_MULTI_TARGET_INTER_INTER_FLAG_ERR,
  XP_CFI_MULTI_TARGET_INTER_FREQ_ERR,
  XP_CFI_MULTI_TARGET_INTER_TIME_REF_ERR,
  XP_CFI_MULTI_TARGET_INTER_DERIV_FLAG_ERR,
  XP_CFI_MULTI_TARGET_INTER_IRAY_ID_ERR,
  XP_CFI_MULTI_TARGET_INTER_LOS_AZIMUTH_ERR,
  XP_CFI_MULTI_TARGET_INTER_LOS_ELEVATION_ERR,
  XP_CFI_MULTI_TARGET_INTER_GEODETIC_ALT_ERR,
  XP_CFI_MULTI_TARGET_INTER_MEMORY_ERR,
  XP_CFI_MULTI_TARGET_INTER_INITIAL_LOOK_DIR_OR_PLANE_ERR,
  XP_CFI_MULTI_TARGET_INTER_TIME_REF_INIT_ERR,
  XP_CFI_MULTI_TARGET_INTER_TARGET_NOT_FOUND_ERR,
  XP_CFI_MULTI_TARGET_INTER_RANGE_OR_POINTING_CALC_ERR,
  XP_CFI_MULTI_TARGET_INTER_NEGATIVE_ALTITUDE_WARN,
  XP_CFI_MULTI_TARGET_INTER_IRAY_ID_WARN,
  XP_CFI_MULTI_TARGET_INTER_EF_TO_EF_ERR, //AN-553
  XP_CFI_MULTI_TARGET_INTER_LIGHT_MODE_NOT_SUPPORTED_WARN, //AN-553
  XP_CFI_MULTI_TARGET_INTER_ERROR_LINKING_IDS_ERR //ANR-790
} XP_CFI_Multi_Target_inter_err_enum;

typedef enum /* error codes list of "xp_multi_target_travel_time"  */
{
  XP_CFI_MULTI_TARGET_TRAVEL_TIME_ATTITUDE_STATUS_ERR = XL_ZERO_E,
  XP_CFI_MULTI_TARGET_TRAVEL_TIME_INTER_FLAG_ERR,
  XP_CFI_MULTI_TARGET_TRAVEL_TIME_FREQ_ERR,
  XP_CFI_MULTI_TARGET_TRAVEL_TIME_TIME_REF_ERR,
  XP_CFI_MULTI_TARGET_TRAVEL_TIME_DERIV_FLAG_ERR,
  XP_CFI_MULTI_TARGET_TRAVEL_TIME_IRAY_ID_ERR,
  XP_CFI_MULTI_TARGET_TRAVEL_TIME_LOS_AZIMUTH_ERR,
  XP_CFI_MULTI_TARGET_TRAVEL_TIME_LOS_ELEVATION_ERR,
  XP_CFI_MULTI_TARGET_TRAVEL_TIME_GEODETIC_ALT_ERR,
  XP_CFI_MULTI_TARGET_TRAVEL_TIME_MEMORY_ERR,
  XP_CFI_MULTI_TARGET_TRAVEL_TIME_INITIAL_LOOK_DIR_OR_PLANE_ERR,
  XP_CFI_MULTI_TARGET_TRAVEL_TIME_TIME_REF_INIT_ERR,
  XP_CFI_MULTI_TARGET_TRAVEL_TIME_TARGET_NOT_FOUND_ERR,
  XP_CFI_MULTI_TARGET_TRAVEL_TIME_RANGE_OR_POINTING_CALC_ERR,
  XP_CFI_MULTI_TARGET_TRAVEL_TIME_NEGATIVE_ALTITUDE_WARN,
  XP_CFI_MULTI_TARGET_TRAVEL_TIME_IRAY_ID_WARN
} XP_CFI_Multi_Target_travel_time_err_enum;

typedef enum /* error codes list of "xp_target_list_inter"  (AN-470) */
{
  XP_CFI_TARGET_LIST_INTER_ATTITUDE_STATUS_ERR = XL_ZERO_E,
  XP_CFI_TARGET_LIST_INTER_INTER_FLAG_ERR,
  XP_CFI_TARGET_LIST_INTER_FREQ_ERR,
  XP_CFI_TARGET_LIST_INTER_TIME_REF_ERR,
  XP_CFI_TARGET_LIST_INTER_DERIV_FLAG_ERR,
  XP_CFI_TARGET_LIST_INTER_IRAY_ID_ERR,
  XP_CFI_TARGET_LIST_INTER_LOS_AZIMUTH_ERR,
  XP_CFI_TARGET_LIST_INTER_LOS_ELEVATION_ERR,
  XP_CFI_TARGET_LIST_INTER_GEODETIC_ALT_ERR,
  XP_CFI_TARGET_LIST_INTER_MEMORY_ERR,
  XP_CFI_TARGET_LIST_INTER_INITIAL_LOOK_DIR_OR_PLANE_ERR,
  XP_CFI_TARGET_LIST_INTER_TIME_REF_INIT_ERR,
  XP_CFI_TARGET_LIST_INTER_TARGET_NOT_FOUND_ERR,
  XP_CFI_TARGET_LIST_INTER_RANGE_OR_POINTING_CALC_ERR,
  XP_CFI_TARGET_LIST_INTER_NEGATIVE_ALTITUDE_WARN,
  XP_CFI_TARGET_LIST_INTER_INSTRUMENT_TYPE_ERR,
  XP_CFI_TARGET_LIST_INTER_LINK_IDS_ERR,
  XP_CFI_TARGET_LIST_INTER_EF_TO_EF_ERR, //AN-553
  XP_CFI_TARGET_LIST_INTER_LIGHT_MODE_NOT_SUPPORTED_WARN //AN-553
} XP_CFI_Target_list_inter_err_enum;

typedef enum /* error codes list of "xp_target_extra_vector"  */
{
  XP_CFI_TARGET_EXTRA_VECTOR_NO_DATA_ERR = XL_ZERO_E,
  XP_CFI_TARGET_EXTRA_VECTOR_NO_SUCH_USER_TARGET_ERR,
  XP_CFI_TARGET_EXTRA_VECTOR_NO_SUCH_LOS_TARGET_ERR,
  XP_CFI_TARGET_EXTRA_VECTOR_NO_SUCH_EARTH_TARGET_ERR,
  XP_CFI_TARGET_EXTRA_VECTOR_EARTH_TARGET_COMPUT_ERR,
  XP_CFI_TARGET_EXTRA_VECTOR_WRONG_TARGET_TYPE_ERR,
  XP_CFI_TARGET_EXTRA_VECTOR_DERIV_FLAG_ERR,
  XP_CFI_TARGET_EXTRA_VECTOR_DER_1ST_NOT_AVAIL_WARN,
  XP_CFI_TARGET_EXTRA_VECTOR_DER_2ND_NOT_AVAIL_WARN,
  XP_CFI_TARGET_EXTRA_VECTOR_ELLIPSOID_WARN,
  XP_CFI_TARGET_EXTRA_VECTOR_DEM_INTER_WARN,
  XP_CFI_TARGET_EXTRA_VECTOR_CONVERT_TO_UTC_ERR, // AN-553
  XP_CFI_TARGET_EXTRA_VECTOR_DEM_DEGRADED_SOLUTION_WARN, // AN-646
  XP_CFI_TARGET_EXTRA_VECTOR_DEM_VOID_VALUE_DETECTED_ERR, /* ANR-746 */
} XP_CFI_Target_extra_vector_err_enum;

typedef enum /* error codes list of "xp_target_extra_main"  */
{
  XP_CFI_TARGET_EXTRA_MAIN_NO_DATA_ERR = XL_ZERO_E,
  XP_CFI_TARGET_EXTRA_MAIN_NO_SUCH_USER_TARGET_ERR,
  XP_CFI_TARGET_EXTRA_MAIN_NO_SUCH_LOS_TARGET_ERR,
  XP_CFI_TARGET_EXTRA_MAIN_NO_SUCH_EARTH_TARGET_ERR,
  XP_CFI_TARGET_EXTRA_MAIN_EARTH_TARGET_COMPUT_ERR,
  XP_CFI_TARGET_EXTRA_MAIN_WRONG_TARGET_TYPE_ERR,
  XP_CFI_TARGET_EXTRA_MAIN_INVALID_TIME_REF_ERR,
  XP_CFI_TARGET_EXTRA_MAIN_CAR_TO_GEO_ERR,
  XP_CFI_TARGET_EXTRA_MAIN_TOPO_ERR,
  XP_CFI_TARGET_EXTRA_MAIN_DIR_POINTING_ERR,
  XP_CFI_TARGET_EXTRA_MAIN_CS_CHANGE_ERR,
  XP_CFI_TARGET_EXTRA_MAIN_DERIV_WARN,
  XP_CFI_TARGET_EXTRA_MAIN_AMBIGUOUS_SINGULAR_WARN,
  XP_CFI_TARGET_EXTRA_MAIN_ELLIPSOID_WARN,
  XP_CFI_TARGET_EXTRA_MAIN_DEM_INTER_WARN,
  XP_CFI_TARGET_EXTRA_MAIN_DEM_DEGRADED_SOLUTION_WARN, // AN-646
  XP_CFI_TARGET_EXTRA_MAIN_DEM_VOID_VALUE_DETECTED_ERR /* ANR-746 */
} XP_CFI_Target_extra_main_err_enum;

typedef enum /* error codes list of "xp_target_extra_aux"  */
{
  XP_CFI_TARGET_EXTRA_AUX_NO_DATA_ERR = XL_ZERO_E,
  XP_CFI_TARGET_EXTRA_AUX_NO_SUCH_USER_TARGET_ERR,
  XP_CFI_TARGET_EXTRA_AUX_NO_SUCH_LOS_TARGET_ERR,
  XP_CFI_TARGET_EXTRA_AUX_NO_SUCH_EARTH_TARGET_ERR,
  XP_CFI_TARGET_EXTRA_AUX_EARTH_TARGET_COMPUT_ERR,
  XP_CFI_TARGET_EXTRA_AUX_WRONG_TARGET_TYPE_ERR,
  XP_CFI_TARGET_EXTRA_AUX_INVALID_TIME_REF_ERR,
  XP_CFI_TARGET_EXTRA_AUX_CAR_TO_GEO_ERR,
  XP_CFI_TARGET_EXTRA_AUX_TOPO_ERR,
  XP_CFI_TARGET_EXTRA_AUX_DIR_POINTING_ERR,
  XP_CFI_TARGET_EXTRA_AUX_RADII_CURVATURE_CALC_ERR,
  XP_CFI_TARGET_EXTRA_AUX_POINTING_AFTER_ATM_CALC_ERR,
  XP_CFI_TARGET_EXTRA_AUX_DISTANCE_CALC_ERR,
  XP_CFI_TARGET_EXTRA_AUX_TOP_VEL_CALC_ERR,
  XP_CFI_TARGET_EXTRA_AUX_MLST_OR_TLST_CALC_ERR,
  XP_CFI_TARGET_EXTRA_AUX_NEGATIVE_ALTITUDE_WARN,
  XP_CFI_TARGET_EXTRA_AUX_DERIV_WARN,
  XP_CFI_TARGET_EXTRA_AUX_AMBIGUOUS_SINGULAR_WARN,
  XP_CFI_TARGET_EXTRA_AUX_ELLIPSOID_WARN,
  XP_CFI_TARGET_EXTRA_AUX_DEM_INTER_WARN,
  XP_CFI_TARGET_EXTRA_AUX_SC_ERR,
  XP_CFI_TARGET_EXTRA_AUX_DEM_DEGRADED_SOLUTION_WARN, // AN-646
  XP_CFI_TARGET_EXTRA_AUX_DEM_VOID_VALUE_DETECTED_ERR /* ANR-746 */
} XP_CFI_Target_extra_aux_err_enum;

typedef enum /* error codes list of "xp_target_extra_ef_target" */
{
  XP_CFI_TARGET_EXTRA_EF_TARGET_NO_DATA_ERR = XL_ZERO_E,
  XP_CFI_TARGET_EXTRA_EF_TARGET_NO_SUCH_USER_TARGET_ERR,
  XP_CFI_TARGET_EXTRA_EF_TARGET_NO_SUCH_LOS_TARGET_ERR,
  XP_CFI_TARGET_EXTRA_EF_TARGET_NO_SUCH_EARTH_TARGET_ERR,
  XP_CFI_TARGET_EXTRA_EF_TARGET_EARTH_TARGET_COMPUT_ERR,
  XP_CFI_TARGET_EXTRA_EF_TARGET_WRONG_TARGET_TYPE_ERR,
  XP_CFI_TARGET_EXTRA_EF_TARGET_DERIV_FLAG_ERR,
  XP_CFI_TARGET_EXTRA_EF_TARGET_GEO_COORD_ERR,
  XP_CFI_TARGET_EXTRA_EF_TARGET_INVALID_TIME_REF_ERR,
  XP_CFI_TARGET_EXTRA_EF_TARGET_RANGE_OR_POINTING_CALC_ERR,
  XP_CFI_TARGET_EXTRA_EF_TARGET_MODE_COMBINATION_SWITCHES_ERR,
  XP_CFI_TARGET_EXTRA_EF_TARGET_CAR_GEO_ERR,
  XP_CFI_TARGET_EXTRA_EF_TARGET_DER_2ND_NOT_AVAIL_WARN,
  XP_CFI_TARGET_EXTRA_EF_TARGET_AMBIGUOUS_SINGULAR_WARN,
  XP_CFI_TARGET_EXTRA_EF_TARGET_DERIV_FLAG_WARN,
  XP_CFI_TARGET_EXTRA_EF_TARGET_ELLIPSOID_WARN,
  XP_CFI_TARGET_EXTRA_EF_TARGET_DEM_INTER_WARN,
  XP_CFI_TARGET_EXTRA_EF_TARGET_SC_ERR,
  XP_CFI_TARGET_EXTRA_EF_TARGET_DEM_DEGRADED_SOLUTION_WARN, // AN-646
  XP_CFI_TARGET_EXTRA_EF_TARGET_DEM_VOID_VALUE_DETECTED_ERR /* ANR-746 */
} XP_CFI_Target_extra_ef_target_err_enum;

typedef enum /* error codes list of "xp_target_list_extra_vector" (AN-471) */
{
  XP_CFI_TARGET_LIST_EXTRA_VECTOR_NO_DATA_ERR = XL_ZERO_E,
  XP_CFI_TARGET_LIST_EXTRA_VECTOR_NO_SUCH_EARTH_TARGET_ERR,
  XP_CFI_TARGET_LIST_EXTRA_VECTOR_EARTH_TARGET_COMPUT_ERR,
  XP_CFI_TARGET_LIST_EXTRA_VECTOR_WRONG_TARGET_TYPE_ERR,
  XP_CFI_TARGET_LIST_EXTRA_VECTOR_DERIV_FLAG_ERR,
  XP_CFI_TARGET_LIST_EXTRA_VECTOR_DER_1ST_NOT_AVAIL_WARN,
  XP_CFI_TARGET_LIST_EXTRA_VECTOR_DER_2ND_NOT_AVAIL_WARN,
  XP_CFI_TARGET_LIST_EXTRA_VECTOR_ELLIPSOID_WARN,
  XP_CFI_TARGET_LIST_EXTRA_VECTOR_DEM_INTER_WARN,
  XP_CFI_TARGET_LIST_EXTRA_VECTOR_MEMORY_ERR,
  XP_CFI_TARGET_LIST_EXTRA_VECTOR_CONVERT_TO_UTC_ERR, //AN-553
  XP_CFI_TARGET_LIST_EXTRA_VECTOR_DEM_DEGRADED_SOLUTION_WARN, // AN-646
  XP_CFI_TARGET_LIST_EXTRA_VECTOR_DEM_VOID_VALUE_DETECTED_ERR /* ANR-746 */
} XP_CFI_Target_list_extra_vector_err_enum;

typedef enum /* error codes list of "xp_target_list_extra_main" (AN-471) */
{
  XP_CFI_TARGET_LIST_EXTRA_MAIN_NO_DATA_ERR = XL_ZERO_E,
  XP_CFI_TARGET_LIST_EXTRA_MAIN_INVALID_TIME_REF_ERR,
  XP_CFI_TARGET_LIST_EXTRA_MAIN_CAR_TO_GEO_ERR,
  XP_CFI_TARGET_LIST_EXTRA_MAIN_TOPO_ERR,
  XP_CFI_TARGET_LIST_EXTRA_MAIN_DIR_POINTING_ERR,
  XP_CFI_TARGET_LIST_EXTRA_MAIN_CS_CHANGE_ERR,
  XP_CFI_TARGET_LIST_EXTRA_MAIN_DERIV_WARN,
  XP_CFI_TARGET_LIST_EXTRA_MAIN_AMBIGUOUS_SINGULAR_WARN,
  XP_CFI_TARGET_LIST_EXTRA_MAIN_MEMORY_ERR,
  XP_CFI_TARGET_LIST_EXTRA_MAIN_EARTH_TARGET_COMPUT_ERR,
  XP_CFI_TARGET_LIST_EXTRA_MAIN_ELLIPSOID_WARN,
  XP_CFI_TARGET_LIST_EXTRA_MAIN_DEM_INTER_WARN,
  XP_CFI_TARGET_LIST_EXTRA_MAIN_NO_SUCH_EARTH_TARGET_ERR,
  XP_CFI_TARGET_LIST_EXTRA_MAIN_WRONG_TARGET_TYPE_ERR,
  XP_CFI_TARGET_LIST_EXTRA_MAIN_DEM_DEGRADED_SOLUTION_WARN, // AN-646
  XP_CFI_TARGET_LIST_EXTRA_MAIN_DEM_VOID_VALUE_DETECTED_ERR /* ANR-746 */
} XP_CFI_Target_LIST_extra_main_err_enum;

typedef enum /* error codes list of "xp_target_list_extra_aux"  */
{
  XP_CFI_TARGET_LIST_EXTRA_AUX_NO_DATA_ERR = XL_ZERO_E,
  XP_CFI_TARGET_LIST_EXTRA_AUX_NO_SUCH_EARTH_TARGET_ERR,
  XP_CFI_TARGET_LIST_EXTRA_AUX_EARTH_TARGET_COMPUT_ERR,
  XP_CFI_TARGET_LIST_EXTRA_AUX_WRONG_TARGET_TYPE_ERR,
  XP_CFI_TARGET_LIST_EXTRA_AUX_INVALID_TIME_REF_ERR,
  XP_CFI_TARGET_LIST_EXTRA_AUX_CAR_TO_GEO_ERR,
  XP_CFI_TARGET_LIST_EXTRA_AUX_TOPO_ERR,
  XP_CFI_TARGET_LIST_EXTRA_AUX_DIR_POINTING_ERR,
  XP_CFI_TARGET_LIST_EXTRA_AUX_RADII_CURVATURE_CALC_ERR,
  XP_CFI_TARGET_LIST_EXTRA_AUX_POINTING_AFTER_ATM_CALC_ERR,
  XP_CFI_TARGET_LIST_EXTRA_AUX_DISTANCE_CALC_ERR,
  XP_CFI_TARGET_LIST_EXTRA_AUX_TOP_VEL_CALC_ERR,
  XP_CFI_TARGET_LIST_EXTRA_AUX_MLST_OR_TLST_CALC_ERR,
  XP_CFI_TARGET_LIST_EXTRA_AUX_NEGATIVE_ALTITUDE_WARN,
  XP_CFI_TARGET_LIST_EXTRA_AUX_DERIV_WARN,
  XP_CFI_TARGET_LIST_EXTRA_AUX_AMBIGUOUS_SINGULAR_WARN,
  XP_CFI_TARGET_LIST_EXTRA_AUX_ELLIPSOID_WARN,
  XP_CFI_TARGET_LIST_EXTRA_AUX_DEM_INTER_WARN,
  XP_CFI_TARGET_LIST_EXTRA_AUX_SC_ERR,
  XP_CFI_TARGET_LIST_EXTRA_AUX_MEMORY_ERR,
  XP_CFI_TARGET_LIST_EXTRA_AUX_DEM_DEGRADED_SOLUTION_WARN, // AN-646
  XP_CFI_TARGET_LIST_EXTRA_AUX_DEM_VOID_VALUE_DETECTED_ERR /* ANR-746 */
} XP_CFI_Target_list_extra_aux_err_enum;

typedef enum /* error codes list of "xp_target_list_extra_ef_target" (AN-471)  */
{
  XP_CFI_TARGET_LIST_EXTRA_EF_TARGET_NO_DATA_ERR = XL_ZERO_E,
  XP_CFI_TARGET_LIST_EXTRA_EF_TARGET_WRONG_TARGET_TYPE_ERR,
  XP_CFI_TARGET_LIST_EXTRA_EF_TARGET_DERIV_FLAG_ERR,
  XP_CFI_TARGET_LIST_EXTRA_EF_TARGET_GEO_COORD_ERR,
  XP_CFI_TARGET_LIST_EXTRA_EF_TARGET_INVALID_TIME_REF_ERR,
  XP_CFI_TARGET_LIST_EXTRA_EF_TARGET_RANGE_OR_POINTING_CALC_ERR,
  XP_CFI_TARGET_LIST_EXTRA_EF_TARGET_MODE_COMBINATION_SWITCHES_ERR,
  XP_CFI_TARGET_LIST_EXTRA_EF_TARGET_CAR_GEO_ERR,
  XP_CFI_TARGET_LIST_EXTRA_EF_TARGET_DER_2ND_NOT_AVAIL_WARN,
  XP_CFI_TARGET_LIST_EXTRA_EF_TARGET_AMBIGUOUS_SINGULAR_WARN,
  XP_CFI_TARGET_LIST_EXTRA_EF_TARGET_DERIV_FLAG_WARN,
  XP_CFI_TARGET_LIST_EXTRA_EF_TARGET_ELLIPSOID_WARN,
  XP_CFI_TARGET_LIST_EXTRA_EF_TARGET_DEM_INTER_WARN,
  XP_CFI_TARGET_LIST_EXTRA_EF_TARGET_SC_ERR,
  XP_CFI_TARGET_LIST_EXTRA_EF_TARGET_MEMORY_ERR,
  XP_CFI_TARGET_LIST_EXTRA_EF_TARGET_DEM_DEGRADED_SOLUTION_WARN, // AN-646
  XP_CFI_TARGET_LIST_EXTRA_EF_TARGET_NO_SUCH_EARTH_TARGET_ERR, // AN-646
  XP_CFI_TARGET_LIST_EXTRA_EF_TARGET_EARTH_TARGET_COMPUT_ERR, // AN-646
  XP_CFI_TARGET_LIST_EXTRA_EF_TARGET_DEM_VOID_VALUE_DETECTED_ERR /* ANR-746 */
} XP_CFI_Target_list_extra_ef_target_err_enum;

typedef enum /* error codes list of "xp_target_extra_target_to_sun"  */
{
  XP_CFI_TARGET_TO_SUN_NO_DATA_ERR = XL_ZERO_E,
  XP_CFI_TARGET_TO_SUN_NO_SUCH_USER_TARGET_ERR,
  XP_CFI_TARGET_TO_SUN_NO_SUCH_LOS_TARGET_ERR,
  XP_CFI_TARGET_TO_SUN_NO_SUCH_EARTH_TARGET_ERR,
  XP_CFI_TARGET_TO_SUN_EARTH_TARGET_COMPUT_ERR,
  XP_CFI_TARGET_TO_SUN_WRONG_TARGET_TYPE_ERR,
  XP_CFI_TARGET_TO_SUN_DERIV_FLAG_ERR,
  XP_CFI_TARGET_TO_SUN_SUN_POS_ERR,
  XP_CFI_TARGET_TO_SUN_INVALID_TIME_REF_ERR,
  XP_CFI_TARGET_TO_SUN_TOD_TO_EF_ERR,
  XP_CFI_TARGET_TO_SUN_DIR_VECTOR_ERR,
  XP_CFI_TARGET_TO_SUN_CAR_GEO_ERR,
  XP_CFI_TARGET_TO_SUN_TARGET_NOT_FOUND_ERR,
  XP_CFI_TARGET_TO_SUN_MODE_COMBINATION_SWITCHES_ERR,
  XP_CFI_TARGET_TO_SUN_TOPO_ERR,
  XP_CFI_TARGET_TO_SUN_DIR_POINTING_ERR,
  XP_CFI_TARGET_TO_SUN_DERIV_FLAG_WARN,
  XP_CFI_TARGET_TO_SUN_MAX_ALLOWED_ITERATIONS_WARN,
  XP_CFI_TARGET_TO_SUN_ELLIPSOID_WARN,
  XP_CFI_TARGET_TO_SUN_DEM_INTER_WARN,
  XP_CFI_TARGET_TO_SUN_IRAY_ID_WARN,
  XP_CFI_TARGET_TO_SUN_DEM_DEGRADED_SOLUTION_WARN, // AN-646
  XP_CFI_TARGET_TO_SUN_DEM_VOID_VALUE_DETECTED_ERR /* ANR-746 */
} XP_CFI_Target_extra_target_to_sun_err_enum;

typedef enum /* error codes list of "xp_target_list_extra_target_to_sun"  */
{
  XP_CFI_TARGET_LIST_TO_SUN_NO_DATA_ERR = XL_ZERO_E,
  XP_CFI_TARGET_LIST_TO_SUN_NO_SUCH_EARTH_TARGET_ERR,
  XP_CFI_TARGET_LIST_TO_SUN_EARTH_TARGET_COMPUT_ERR,
  XP_CFI_TARGET_LIST_TO_SUN_WRONG_TARGET_TYPE_ERR,
  XP_CFI_TARGET_LIST_TO_SUN_DERIV_FLAG_ERR,
  XP_CFI_TARGET_LIST_TO_SUN_SUN_POS_ERR,
  XP_CFI_TARGET_LIST_TO_SUN_INVALID_TIME_REF_ERR,
  XP_CFI_TARGET_LIST_TO_SUN_TOD_TO_EF_ERR,
  XP_CFI_TARGET_LIST_TO_SUN_DIR_VECTOR_ERR,
  XP_CFI_TARGET_LIST_TO_SUN_CAR_GEO_ERR,
  XP_CFI_TARGET_LIST_TO_SUN_TARGET_NOT_FOUND_ERR,
  XP_CFI_TARGET_LIST_TO_SUN_MODE_COMBINATION_SWITCHES_ERR,
  XP_CFI_TARGET_LIST_TO_SUN_TOPO_ERR,
  XP_CFI_TARGET_LIST_TO_SUN_DIR_POINTING_ERR,
  XP_CFI_TARGET_LIST_TO_SUN_DERIV_FLAG_WARN,
  XP_CFI_TARGET_LIST_TO_SUN_MAX_ALLOWED_ITERATIONS_WARN,
  XP_CFI_TARGET_LIST_TO_SUN_ELLIPSOID_WARN,
  XP_CFI_TARGET_LIST_TO_SUN_DEM_INTER_WARN,
  XP_CFI_TARGET_LIST_TO_SUN_IRAY_ID_WARN,
  XP_CFI_TARGET_LIST_TO_SUN_MEMORY_ERR,
  XP_CFI_TARGET_LIST_TO_SUN_DEM_DEGRADED_SOLUTION_WARN, // AN-646
  XP_CFI_TARGET_LIST_TO_SUN_DEM_VOID_VALUE_DETECTED_ERR /* ANR-746 */
} XP_CFI_Target_list_extra_target_to_sun_err_enum;

typedef enum /* error codes list of "xp_target_extra_target_to_moon"  */
{
  XP_CFI_TARGET_TO_MOON_NO_DATA_ERR = XL_ZERO_E,
  XP_CFI_TARGET_TO_MOON_NO_SUCH_USER_TARGET_ERR,
  XP_CFI_TARGET_TO_MOON_NO_SUCH_LOS_TARGET_ERR,
  XP_CFI_TARGET_TO_MOON_NO_SUCH_EARTH_TARGET_ERR,
  XP_CFI_TARGET_TO_MOON_EARTH_TARGET_COMPUT_ERR,
  XP_CFI_TARGET_TO_MOON_WRONG_TARGET_TYPE_ERR,
  XP_CFI_TARGET_TO_MOON_DERIV_FLAG_ERR,
  XP_CFI_TARGET_TO_MOON_MOON_POS_ERR,
  XP_CFI_TARGET_TO_MOON_INVALID_TIME_REF_ERR,
  XP_CFI_TARGET_TO_MOON_TOD_TO_EF_ERR,
  XP_CFI_TARGET_TO_MOON_DIR_VECTOR_ERR,
  XP_CFI_TARGET_TO_MOON_CAR_GEO_ERR,
  XP_CFI_TARGET_TO_MOON_TARGET_NOT_FOUND_ERR,
  XP_CFI_TARGET_TO_MOON_MODE_COMBINATION_SWITCHES_ERR,
  XP_CFI_TARGET_TO_MOON_TOPO_ERR,
  XP_CFI_TARGET_TO_MOON_DIR_POINTING_ERR,
  XP_CFI_TARGET_TO_MOON_DERIV_FLAG_WARN,
  XP_CFI_TARGET_TO_MOON_MAX_ALLOWED_ITERATIONS_WARN,
  XP_CFI_TARGET_TO_MOON_ELLIPSOID_WARN,
  XP_CFI_TARGET_TO_MOON_DEM_INTER_WARN,
  XP_CFI_TARGET_TO_MOON_IRAY_ID_WARN,
  XP_CFI_TARGET_TO_MOON_DEM_DEGRADED_SOLUTION_WARN, // AN-646
  XP_CFI_TARGET_TO_MOON_DEM_VOID_VALUE_DETECTED_ERR /* ANR-746 */
} XP_CFI_Target_extra_target_to_moon_err_enum;

typedef enum /* error codes list of "xp_target_list_extra_target_to_moon" (AN-471) */
{
  XP_CFI_TARGET_LIST_TO_MOON_NO_DATA_ERR = XL_ZERO_E,
  XP_CFI_TARGET_LIST_TO_MOON_NO_SUCH_EARTH_TARGET_ERR,
  XP_CFI_TARGET_LIST_TO_MOON_EARTH_TARGET_COMPUT_ERR,
  XP_CFI_TARGET_LIST_TO_MOON_WRONG_TARGET_TYPE_ERR,
  XP_CFI_TARGET_LIST_TO_MOON_DERIV_FLAG_ERR,
  XP_CFI_TARGET_LIST_TO_MOON_MOON_POS_ERR,
  XP_CFI_TARGET_LIST_TO_MOON_INVALID_TIME_REF_ERR,
  XP_CFI_TARGET_LIST_TO_MOON_TOD_TO_EF_ERR,
  XP_CFI_TARGET_LIST_TO_MOON_DIR_VECTOR_ERR,
  XP_CFI_TARGET_LIST_TO_MOON_CAR_GEO_ERR,
  XP_CFI_TARGET_LIST_TO_MOON_TARGET_NOT_FOUND_ERR,
  XP_CFI_TARGET_LIST_TO_MOON_MODE_COMBINATION_SWITCHES_ERR,
  XP_CFI_TARGET_LIST_TO_MOON_TOPO_ERR,
  XP_CFI_TARGET_LIST_TO_MOON_DIR_POINTING_ERR,
  XP_CFI_TARGET_LIST_TO_MOON_DERIV_FLAG_WARN,
  XP_CFI_TARGET_LIST_TO_MOON_MAX_ALLOWED_ITERATIONS_WARN,
  XP_CFI_TARGET_LIST_TO_MOON_ELLIPSOID_WARN,
  XP_CFI_TARGET_LIST_TO_MOON_DEM_INTER_WARN,
  XP_CFI_TARGET_LIST_TO_MOON_IRAY_ID_WARN,
  XP_CFI_TARGET_LIST_TO_MOON_MEMORY_ERR,
  XP_CFI_TARGET_LIST_TO_MOON_DEM_DEGRADED_SOLUTION_WARN, // AN-646
  XP_CFI_TARGET_LIST_TO_MOON_DEM_VOID_VALUE_DETECTED_ERR /* ANR-746 */
} XP_CFI_Target_list_extra_target_to_moon_err_enum;

typedef enum /* error codes list of "xp_target_extra_specular_reflection"  */
{
  XP_CFI_TARGET_EXTRA_SPECULAR_REFLECT_NO_DATA_ERR = XL_ZERO_E,
  XP_CFI_TARGET_EXTRA_SPECULAR_REFLECT_WRONG_DEF_ANGLE_ERR,
  XP_CFI_TARGET_EXTRA_SPECULAR_REFLECT_NO_SUCH_USER_TARGET_ERR,
  XP_CFI_TARGET_EXTRA_SPECULAR_REFLECT_NO_SUCH_LOS_TARGET_ERR,
  XP_CFI_TARGET_EXTRA_SPECULAR_REFLECT_EARTH_TARGET_COMPUT_ERR,
  XP_CFI_TARGET_EXTRA_SPECULAR_REFLECT_NO_SUCH_EARTH_TARGET_ERR,
  XP_CFI_TARGET_EXTRA_SPECULAR_REFLECT_WRONG_TARGET_TYPE_ERR,
  XP_CFI_TARGET_EXTRA_SPECULAR_REFLECT_CAR_TO_GEO_ERR,
  XP_CFI_TARGET_EXTRA_SPECULAR_REFLECT_TOPO_CS_ERR,
  XP_CFI_TARGET_EXTRA_SPECULAR_REFLECT_DIR_POINTING_ERR,
  XP_CFI_TARGET_EXTRA_SPECULAR_REFLECT_CHANGE_CS_ERR,
  XP_CFI_TARGET_EXTRA_SPECULAR_REFLECT_DERIV_WARN,
  XP_CFI_TARGET_EXTRA_SPECULAR_REFLECT_ELLIPSOID_WARN,
  XP_CFI_TARGET_EXTRA_SPECULAR_REFLECT_DEM_INTER_WARN,
  XP_CFI_TARGET_EXTRA_SPECULAR_REFLECT_SC_ERR,
  XP_CFI_TARGET_EXTRA_SPECULAR_REFLECT_DEM_DEGRADED_SOLUTION_WARN, // AN-646
  XP_CFI_TARGET_EXTRA_SPECULAR_REFLECT_DEM_VOID_VALUE_DETECTED_ERR /* ANR-746 */
} XP_CFI_Target_extra_specular_reflection_err_enum;

typedef enum /* error codes list of "xp_target_list_extra_specular_reflection" (AN-471) */
{
  XP_CFI_TARGET_LIST_EXTRA_SPECULAR_REFLECT_NO_DATA_ERR = XL_ZERO_E,
  XP_CFI_TARGET_LIST_EXTRA_SPECULAR_REFLECT_WRONG_DEF_ANGLE_ERR,
  XP_CFI_TARGET_LIST_EXTRA_SPECULAR_REFLECT_EARTH_TARGET_COMPUT_ERR,
  XP_CFI_TARGET_LIST_EXTRA_SPECULAR_REFLECT_WRONG_TARGET_TYPE_ERR,
  XP_CFI_TARGET_LIST_EXTRA_SPECULAR_REFLECT_NO_SUCH_EARTH_TARGET_ERR,
  XP_CFI_TARGET_LIST_EXTRA_SPECULAR_REFLECT_CAR_TO_GEO_ERR,
  XP_CFI_TARGET_LIST_EXTRA_SPECULAR_REFLECT_TOPO_CS_ERR,
  XP_CFI_TARGET_LIST_EXTRA_SPECULAR_REFLECT_DIR_POINTING_ERR,
  XP_CFI_TARGET_LIST_EXTRA_SPECULAR_REFLECT_CHANGE_CS_ERR,
  XP_CFI_TARGET_LIST_EXTRA_SPECULAR_REFLECT_DERIV_WARN,
  XP_CFI_TARGET_LIST_EXTRA_SPECULAR_REFLECT_ELLIPSOID_WARN,
  XP_CFI_TARGET_LIST_EXTRA_SPECULAR_REFLECT_DEM_INTER_WARN,
  XP_CFI_TARGET_LIST_EXTRA_SPECULAR_REFLECT_SC_ERR,
  XP_CFI_TARGET_LIST_EXTRA_SPECULAR_REFLECT_MEMORY_ERR,
  XP_CFI_TARGET_LIST_EXTRA_SPECULAR_REFLECT_DEM_DEGRADED_SOLUTION_WARN, // AN-646
  XP_CFI_TARGET_LIST_EXTRA_SPECULAR_REFLECT_DEM_VOID_VALUE_DETECTED_ERR /* ANR-746 */
} XP_CFI_Target_list_extra_specular_reflection_err_enum;

typedef enum /* error codes list of "xp_sun"  */
{
  XP_CFI_SUN_ATTITUDE_STATUS_ERR = XL_ZERO_E,
  XP_CFI_SUN_DERIV_FLAG_ERR,
  XP_CFI_SUN_IRAY_ID_ERR,
  XP_CFI_SUN_FREQ_ERR,
  XP_CFI_SUN_TIME_REF_INIT_ERR,
  XP_CFI_SUN_SUN_POSITION_CALC_ERR,
  XP_CFI_SUN_SUN_CS_CALC_ERR,
  XP_CFI_SUN_SUN_POINTING_CALC_ERR,
  XP_CFI_SUN_TARGET_STAR_ERR,
  XP_CFI_SUN_TG_PT_BEHIND_LOOK_DIR_ERR,
  XP_CFI_SUN_TG_PT_BEHIND_LOOK_DIR_WARN,
  XP_CFI_SUN_IRAY_ID_WARN,
  XP_CFI_SUN_LIGHT_MODE_NOT_SUPPORTED_WARN //AN-553
} XP_CFI_Sun_err_enum;

typedef enum /* error codes list of "xp_moon"  */
{
  XP_CFI_MOON_ATTITUDE_STATUS_ERR = XL_ZERO_E,
  XP_CFI_MOON_DERIV_FLAG_ERR,
  XP_CFI_MOON_IRAY_ID_ERR,
  XP_CFI_MOON_FREQ_ERR,
  XP_CFI_MOON_TIME_REF_INIT_ERR,
  XP_CFI_MOON_MOON_POSITION_CALC_ERR,
  XP_CFI_MOON_MOON_CS_CALC_ERR,
  XP_CFI_MOON_MOON_POINTING_CALC_ERR,
  XP_CFI_MOON_TARGET_STAR_ERR,
  XP_CFI_MOON_TG_PT_BEHIND_LOOK_DIR_ERR,
  XP_CFI_MOON_TG_PT_BEHIND_LOOK_DIR_WARN,
  XP_CFI_MOON_IRAY_ID_WARN
} XP_CFI_Moon_err_enum;

typedef enum /* error codes list of "xp_target_close"  */
{
  XP_CFI_TARGET_CLOSE_WRONG_ID_ERR = XL_ZERO_E,
} XP_CFI_Target_close_err_enum;

typedef enum /* error codes list of "xp_run_init" */
{
  XP_CFI_RUN_INIT_STATUS_ERR = XL_ZERO_E,
  XP_CFI_RUN_INIT_MEMORY_ERR,
  XP_CFI_RUN_INIT_INCONSISTENCY_ERR

} XP_CFI_Run_init_err_enum;

typedef enum /* error codes list of "xp_set_az_el_definition" */
{
  XP_CFI_SET_AZ_EL_DEF_ID_NOT_INITIALIZED_ERR = XL_ZERO_E,
  XP_CFI_SET_AZ_EL_DEF_NOT_ATTITUDE_ID_ERR,
  XP_CFI_SET_AZ_EL_DEF_NOT_PERPENDICULAR_AZIMUTH_AXIS_ERR,
  XP_CFI_SET_AZ_EL_DEF_NOT_PERPENDICULAR_ELEVATION_AXIS_ERR
} XP_CFI_Set_az_el_definition_err_enum;

typedef enum /* error codes list of "xp_dem_id_configure" (AN-487) */
{
  XP_CFI_DEM_ID_CONFIGURE_NO_CACHE_ERR,
  XP_CFI_DEM_ID_CONFIGURE_WRONG_COMMAND_ERR,
  XP_CFI_DEM_ID_CONFIGURE_MEMORY_ERR,
  XP_CFI_DEM_ID_CONFIGURE_LON_ERR,
  XP_CFI_DEM_ID_CONFIGURE_LAT_ERR,
  XP_CFI_DEM_ID_CONFIGURE_MAX_CACHE_ERR,
  XP_CFI_DEM_ID_CONFIGURE_LOAD_TILE_ERR,
  XP_CFI_DEM_ID_CONFIGURE_LOCK_THREAD_ERR

} XP_CFI_dem_id_config_err_enum;

typedef enum /* error codes list of "xp_gen_dem_max_altitude_file" (AN-492) */
{
  XP_CFI_GEN_DEM_MAX_ALTITUDE_READ_DEM_CFG_ERR = XL_ZERO_E,
  XP_CFI_GEN_DEM_MAX_ALTITUDE_MEMORY_ERR,
  XP_CFI_GEN_DEM_MAX_ALTITUDE_OPEN_TILE_WARN,
  XP_CFI_GEN_DEM_MAX_ALTITUDE_READ_TILE_ERR,
  XP_CFI_GEN_DEM_MAX_ALTITUDE_DEM_INIT_ERR,
  XP_CFI_GEN_DEM_MAX_ALTITUDE_DEM_CLOSE_ERR,
  XP_CFI_GEN_DEM_MAX_ALTITUDE_OPEN_OUTPUT_FILE_ERR,
  XP_CFI_GEN_DEM_MAX_ALTITUDE_WRITE_OUTPUT_ERR,
  XP_CFI_GEN_DEM_MAX_ALTITUDE_CONFIG_ERR
} XP_CFI_gen_dem_max_altititude_file_enum;

typedef enum /* error codes list of "xp_gen_dem_altitudes_from_ellipsoid" (AN-466) */
{
  XP_CFI_GEN_DEM_ALTITUDE_FROM_ELLIPSOID_READ_DEM_CFG_ERR = XL_ZERO_E,
  XP_CFI_GEN_DEM_ALTITUDE_FROM_ELLIPSOID_WRONG_DEM_TYPE_ERR,
  XP_CFI_GEN_DEM_ALTITUDE_FROM_ELLIPSOID_MEMORY_ERR,
  XP_CFI_GEN_DEM_ALTITUDE_FROM_ELLIPSOID_READ_TILE_ERR,
  XP_CFI_GEN_DEM_ALTITUDE_FROM_ELLIPSOID_GEOID_UNDU_ERR,
  XP_CFI_GEN_DEM_ALTITUDE_FROM_ELLIPSOID_OPEN_OUTPUT_ERR,
  XP_CFI_GEN_DEM_ALTITUDE_FROM_ELLIPSOID_WRITE_OUTPUT_ERR,
  XP_CFI_GEN_DEM_ALTITUDE_FROM_ELLIPSOID_OPEN_TILE_WARN,
  XP_CFI_GEN_DEM_ALTITUDE_FROM_ELLIPSOID_WRONG_DEM_SET_TYPE_ERR,
  XP_CFI_GEN_DEM_ALTITUDE_FROM_ELLIPSOID_WRONG_SET_LONGITUDE_ERR,
  XP_CFI_GEN_DEM_ALTITUDE_FROM_ELLIPSOID_WRONG_SET_LATITUDE_ERR
} XP_CFI_gen_dem_altitudes_from_ellipsoid_err_enum;

typedef enum /* xp_attitude_define AN-469 */
{
  XP_CFI_ATTITUDE_DEFINE_INIT_SAT_NOM_TRANS_ERR = XL_ZERO_E,
  XP_CFI_ATTITUDE_DEFINE_INIT_SAT_TRANS_ERR,
  XP_CFI_ATTITUDE_DEFINE_INIT_INSTR_TRANS_ERR
} XP_CFI_attitude_define_enum_err;

typedef enum /* xp_attitude_transform AN-526 */
{
  XP_CFI_ATTITUDE_TRANSFORM_QUAT_2_VEC_ERR = XL_ZERO_E,
  XP_CFI_ATTITUDE_TRANSFORM_GET_ROTATION_ERR,
  XP_CFI_ATTITUDE_TRANSFORM_VEC_2_QUAT_ERR,
  XP_CFI_ATTITUDE_TRANSFORM_UNINIT_ATTITUDE_ERR,
  XP_CFI_ATTITUDE_TRANSFORM_UNALLOWED_ATT_TYPE_ERR,
  XP_CFI_ATTITUDE_TRANSFORM_STAR_TRACKER_NOT_SUPPORTED_ERR // AN-600
} XP_CFI_attitude_transform_err_enum;

typedef enum /* xp_get_attitude_data AN-597 */
{
  XP_CFI_GET_ATT_DATA_WRONG_SAT_CS_ERR = XL_ZERO_E,
  XP_CFI_GET_ATT_DATA_WRONG_EXT_CS_ERR,
  XP_CFI_GET_ATT_DATA_WRONG_FRAME_ERR,
  XP_CFI_GET_ATT_DATA_GET_ATT_DATA_ERR,
  XP_CFI_GET_ATT_DATA_GET_ROTATION_MATRIX_ERR,
  XP_CFI_GET_ATT_DATA_GET_ATT_INIT_ERR,
  XP_CFI_GET_ATT_DATA_GET_ATT_COMPUTE_ERR,
  XP_CFI_GET_ATT_DATA_MATRIX_INV_ERR,
  XP_CFI_GET_ATT_DATA_CLOSE_ATT_ERR,
  XP_CFI_GET_ATT_DATA_ANGLE_COMP_ERR,
  XP_CFI_GET_ATT_DATA_QUAT_COMP_ERR,
  XP_CFI_GET_ATT_DATA_SAT_NOM_INIT_WARN /* AN-712 */
} XP_CFI_get_attitude_data_err_enum;

typedef enum /* xp_gen_attitude_data AN-596 */
{
  XP_CFI_GEN_ATT_DATA_ORBIT_INIT_ERR = XL_ZERO_E,
  XP_CFI_GEN_ATT_DATA_ATT_INIT_ERR,
  XP_CFI_GEN_ATT_DATA_ATT_TARGET_TYPE_ERR,
  XP_CFI_GEN_ATT_DATA_INSTR_ATT_ERR,
  XP_CFI_GEN_ATT_DATA_SAT_ATT_ERR,
  XP_CFI_GEN_ATT_DATA_SAT_NOM_ATT_ERR,
  XP_CFI_GEN_ATT_DATA_START_INTERVAL_ERR,
  XP_CFI_GEN_ATT_DATA_STOP_INTERVAL_ERR,
  XP_CFI_GEN_ATT_DATA_TIME_STEP_ERR,
  XP_CFI_GEN_ATT_DATA_DATA_TYPE_ERR,
  XP_CFI_GEN_ATT_DATA_SAT_CS_ERR,
  XP_CFI_GEN_ATT_DATA_EXTERNAL_CS_ERR,
  XP_CFI_GEN_ATT_DATA_CS_TYPE_ERR,
  XP_CFI_GEN_ATT_DATA_MEM_ALLOC_ERR,
  XP_CFI_GEN_ATT_DATA_OSV_COMP_ERR,
  XP_CFI_GEN_ATT_DATA_ATT_COMP_ERR,
  XP_CFI_GEN_ATT_DATA_GET_ATT_DATA_ERR,
  XP_CFI_GEN_ATT_DATA_TIME_REF_ERR,
  XP_CFI_GEN_ATT_DATA_TIME_CONVERSION_ERR
} XP_CFI_gen_attitude_data_err_enum;

typedef enum /* xp_gen_attitude_file AN-596 */
{
  XP_CFI_GEN_ATT_FILE_GENDATA_ERR = XL_ZERO_E,
  XP_CFI_GEN_ATT_FILE_GENFHR_ERR,
  XP_CFI_GEN_ATT_FILE_NO_DATA_ERR,
  XP_CFI_GEN_ATT_FILE_MEM_ERR,
  XP_CFI_GEN_ATT_FILE_WRITE_ERR

} XP_CFI_gen_attitude_file_err_enum;

/****************************************************************************************/

#endif
