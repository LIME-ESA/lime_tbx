/*****************************************************************************
 * 
 * Copyright: DEIMOS Space S.L.U.
 * Project  : EOCFI
 * 
 * Module   : explorer_visibility.h
 * Purpose  : It is the header file of the EXPLORER_VISIBILITY CFI to be used 
 *                by the final user
 *
 * History:    +----------------------------------------------------------------------------------+
 *             | Version | Date     | Name                  | Change                              |
 *             |---------|----------|-----------------------|-------------------------------------|
 *             |    1.0  | 19/06/02 |   DEIMOS Space, S.L.  | First release from                  |
 *             |         |          |                       | Envisat                             |
 *             +----------------------------------------------------------------------------------+
 *             |    2.0  | 29/11/02 |   DEIMOS Space S.L.   | New functions added                 |
 *             |         |          |                       |  + xv_station_vis_time              |
 *             |         |          |                       |  + xv_star_vis_time                 |
 *             |         |          |                       |  + xv_drs_vis_time                  |
 *             |         |          |                       |  + xv_time_segment_not              |
 *             |         |          |                       |  + xv_time_segment_add              |
 *             |         |          |                       |  + xv_time_segment_or               |
 *             +----------------------------------------------------------------------------------+
 *             |    2.1  | 13/05/03 |   DEIMOS Space S.L.   | Maintenance release                 |
 *             |         |          |                       | MAC OS Version                      |
 *             |         |          |                       | New functions added                 |
 *             |         |          |                       |  + xv_multizones_vis_time           |
 *             |         |          |                       |  + xv_multistations_vis_time        |
 *             |         |          |                       |  + xv_time_segment_mapping          |
 *             |         |          |                       |  + xv_orbit_extra                   |
 *             +----------------------------------------------------------------------------------+
 *             |    2.2  | 13/08/03 |   DEIMOS Space S.L.   | Pre-release                         |
 *             +----------------------------------------------------------------------------------+
 *             |    2.2  | 30/09/03 |   DEIMOS Space S.L.   | Maintenance release                 |
 *             +----------------------------------------------------------------------------------+
 *             |   2.2.2 | 26/04/04 |   DEIMOS Space S.L.   | Maintenance release                 |
 *             |         |          |                       |* Interface changed for              |
 *             |         |          |                       |   xv_time_segment_delta             |
 *             |         |          |                       |* Added Terrasar S/C                 |
 *             +----------------------------------------------------------------------------------+
 *             |    3.0  | 21/07/04 |   DEIMOS Space S.L.   | New interfaces for the              |
 *             |         |          |                       | new initialisation                  |
 *             |         |          |                       | strategy                            |
 *             +----------------------------------------------------------------------------------+
 *             |    3.1  | 13/10/04 |   DEIMOS Space S.L.   | Maintenance release                 |
 *             |         |          |                       | New features:                       |
 *             |         |          |                       |  + Use of the OEF/POF files         |
 *             |         |          |                       |    for xv_zone_vis_time             |
 *             |         |          |                       |  + Use of Swath Def. files in       |
 *             |         |          |                       |    xv_zone_vis_time                 |
 *             +----------------------------------------------------------------------------------+
 *             |    3.2  | 15/11/04 |   DEIMOS Space S.L.   | Maintenance release                 |
 *             +----------------------------------------------------------------------------------+
 *             |    3.3  | 11/07/05 |   DEIMOS Space S.L.   | Maintenance release                 |
 *             |         |          |                       | New features:                       |
 *             |         |          |                       | + Use of the OEF/POF files in       |
 *             |         |          |                       |   all visibility functions          |
 *             |         |          |                       | + Use of Swath Def. Files in        |
 *             |         |          |                       |   all visibility functions          |
 *             |         |          |                       | + ENVISAT support removed           |
 *             +----------------------------------------------------------------------------------+
 *             |    3.4  | 18/11/05 |   DEIMOS Space S.L.   | Maintenance release:                |
 *             |         |          |                       | + Changes in the interface          |
 *             |         |          |                       |   of xv_swath_pos                   |
 *             +----------------------------------------------------------------------------------+
 *             |    3.5  | 26/05/06 |   DEIMOS Space S.L.   | Maintenance release                 |
 *             |         |          |                       | New features:                       |
 *             |         |          |                       |  + Thread safe library              |
 *             |         |          |                       |  + C99 compliant                    |
 *             |         |          |                       |  + Optimized libray                 |
 *             |         |          |                       |  + 64-Bits library for MACOS,       |
 *             |         |          |                       |    LINUX and SOLARIS                |
 *             +----------------------------------------------------------------------------------+
 *             |    3.6  | 24/11/06 |   DEIMOS Space S.L.   | Maintenance release                 |
 *             +----------------------------------------------------------------------------------+
 *             |    3.7  | 13/07/07 |   DEIMOS Space S.L  . | Maintenance release                 |
 *             |         |          |                       | New features:                       |
 *             |         |          |                       |  + xv_gen_scf                       |
 *             |         |          |                       |  + expcfi_check_libs                |
 *             |         |          |                       |  + xv_zone_vis_time_no_file         |
 *             |         |          |                       |  + xv_station_vis_time_no_file      |
 *             |         |          |                       |  + xv_gen_swath_no_file             |
 *             |         |          |                       |  + library version for Mac OS X     |
 *             |         |          |                       |    on Intel (32 and 64 bits)        |
 *             +----------------------------------------------------------------------------------+
 *             |  3.7.2  | 31/07/08 |   DEIMOS Space S.L  . | Maintenance release                 |
 *             |         |          |                       | New features:                       |
 *             |         |          |                       |  + Support for area/curved swath    |
 *             +----------------------------------------------------------------------------------+
 *             |    4.0  | 19/01/09 | DEIMOS Space S.L.     | Maintenance release                 |
 *             +----------------------------------------------------------------------------------+
 *             |    4.1  | 07/05/10 | DEIMOS Space S.L.     | Maintenance release                 |
 *             |         |          |                       | New features:                       |
 *             |         |          |                       |  + Visib.across orbital changes     |
 *             |         |          |                       |  + AOS/LOS mask mode from GND_DB    |
 *             +----------------------------------------------------------------------------------+
 *             |    4.2  | 31/01/11 | DEIMOS Space S.L.U    | Maintenance release                 |
 *             |         |          |                       | New features:                       |
 *             |         |          |                       |  + Support for curved MLST          |
 *             |         |          |                       |  + Visibility functions adapted     |
 *             |         |          |                       |    for TLE and Precise propagator   |
 *             +----------------------------------------------------------------------------------+
 *             |    4.3  | 06/02/12 | DEIMOS Space S.L.U    | Maintenance release                 |
 *             +----------------------------------------------------------------------------------+
 *             |    4.4  | 05/07/12 | DEIMOS Space S.L.U    | Maintenance release                 |
 *             |         |          |                       | New features:                       |
 *             |         |          |                       |  + xv_sc_vis_time                   |
 *             +----------------------------------------------------------------------------------+
 *             |    4.5  | 01/03/13 | DEIMOS Space S.L.U    | Maintenance release                 |
 *             +----------------------------------------------------------------------------------+
 *             |    4.6  | 03/10/13 | DEIMOS Space S.L.U    | Maintenance release                 |
 *             |         |          |                       | New features:                       |
 *             |         |          |                       |  + Support for new swath ID         |
 *             |         |          |                       |  + New functions:                   |
 *             |         |          |                       |    xv_zonevistime_compute           |
 *             |         |          |                       |    xv_stationvistime_compute        |
 *             |         |          |                       |    xv_swathpos_compute              |
 *             |         |          |                       |    xv_timesegments_xxx_compute      |
 *             +----------------------------------------------------------------------------------+
 *             |    4.7  | 28/03/14 | DEIMOS Space S.L.U    | Maintenance release                 |
 *             |         |          |                       | New features:                       |
 *             |         |          |                       |  + New function:                    |
 *             |         |          |                       |    xv_zonevistime_coverage          |
 *             +----------------------------------------------------------------------------------+
 *             |    4.8  | 29/10/14 | DEIMOS Space S.L.U    | Maintenance release                 |
 *             +----------------------------------------------------------------------------------+
 *             |    4.9  | 23/04/15 | DEIMOS Space S.L.U    | Maintenance release                 |
 *             +----------------------------------------------------------------------------------+
 *             |  4.9.1  | 05/06/15 | DEIMOS Space S.L.U    | Maintenance release                 |
 *             +----------------------------------------------------------------------------------+
 *             |   4.10  | 29/10/15 | DEIMOS Space S.L.U    | Maintenance release                 |
 *             +----------------------------------------------------------------------------------+
 *             |   4.11  | 15/04/16 | DEIMOS Space S.L.U    | Maintenance release                 |
 *             +----------------------------------------------------------------------------------+
 *             |   4.12  | 03/11/16 | DEIMOS Space S.L.U    | Maintenance release                 |
 *             |         |          |                       | New features:                       |
 *             |         |          |                       |  + New functions:                   |
 *             |         |          |                       |    xv_set_sc_visibility_step        |
 *             +----------------------------------------------------------------------------------+
 *             |   4.13  | 05/04/17 | DEIMOS Space S.L.U    | Maintenance release                 |
 *             +----------------------------------------------------------------------------------+
 *             |   4.14  | 16/11/17 | DEIMOS Space S.L.U    | Maintenance release                 |
 *             |         |          |                       | New features:                       |
 *             |         |          |                       |  + Support for swath defined by     |
 *             |         |          |                       |    incidence angles                 |
 *             |         |          |                       |  + Updated generation of SCF        |
 *             +----------------------------------------------------------------------------------+
 *             |   4.15  | 20/04/18 | DEIMOS Space S.L.U    | Maintenance release                 |
 *             +----------------------------------------------------------------------------------+
 *             |   4.16  | 09/11/18 | DEIMOS Space S.L.U    | Maintenance release                 |
 *             |         |          |                       | New features:                       |
 *             |         |          |                       |  + New functions:                   |
 *             |         |          |                       |     xv_celestial_body_vistime,      |
 *             |         |          |                       |     xv_compute_aoi                  |
 *             +----------------------------------------------------------------------------------+
 *             |   4.17  | 10/05/19 | DEIMOS Space S.L.U    | Maintenance release                 |
 *             +----------------------------------------------------------------------------------+
 *             |   4.18  | 08/11/19 | DEIMOS Space S.L.U    | Maintenance release                 |
 *             +----------------------------------------------------------------------------------+
 *             |   4.19  | 29/05/20 | DEIMOS Space S.L.U    | Maintenance release                 |
 *             +----------------------------------------------------------------------------------+
 *             |   4.20  | 30/11/20 | DEIMOS Space S.L.U    | Maintenance release                 |
 *             |         |          |                       | New features:                       |
 *             |         |          |                       |  + Modernized generation of Swath   |
 *             |         |          |                       |    Control Files                    |
 *             +----------------------------------------------------------------------------------+
 *             |   4.21  | 10/06/21 | DEIMOS Space S.L.U    | Maintenance release                 |
 *             |         |          |                       | New features:                       |
 *             |         |          |                       |  + Optimisation of EOCFI library    |
 *             |         |          |                       |    size                             |
 *             +----------------------------------------------------------------------------------+
 *             |   4.22  | 22/12/21 | DEIMOS Space S.L.U    | Maintenance release                 |
 *             +----------------------------------------------------------------------------------+
 *             |   4.23  | 23/06/22 | DEIMOS Space S.L.U    | Maintenance release                 |
 *             +----------------------------------------------------------------------------------+
 *             |   4.24  | 29/11/22 | DEIMOS Space S.L.U    | Maintenance release                 |
 *             +----------------------------------------------------------------------------------+
 *             |   4.25  | 10/05/23 | DEIMOS Space S.L.U    | Maintenance release                 |
 *             |         |          |                       | New features:                       |
 *             |         |          |                       |  + Support for ANX drift in OSF     |
 *             +----------------------------------------------------------------------------------+
 *
 *****************************************************************************/
#ifndef EXPLORER_VISIBILITY_H
#define EXPLORER_VISIBILITY_H

#include <explorer_lib.h>
#include <explorer_orbit.h>
#include <explorer_pointing.h>

/*******************************************************************************
 *
 * DefinesXV_ZONE_VIS_TIME_TWO_SEGMENTS_INTERSECT_ERR
 *
 *******************************************************************************/

/* Length of ierr vectors */
/* ---------------------- */
#define XV_NUM_ERR_SWATH_POS               2
#define XV_NUM_ERR_ZONE_VIS_TIME           3
#define XV_NUM_ERR_STATION_VIS_TIME        2
#define XV_NUM_ERR_STAR_VIS_TIME           2
#define XV_NUM_ERR_SC_VIS_TIME             1
#define XV_NUM_ERR_CELESTIAL_BODY_VIS_TIME 1 /* AN-677*/
#define XV_NUM_ERR_MULTIZONES_VIS_TIME     1
#define XV_NUM_ERR_MULTISTATIONS_VIS_TIME  1
#define XV_NUM_ERR_NOT                     1
#define XV_NUM_ERR_OR                      1
#define XV_NUM_ERR_AND                     1
#define XV_NUM_ERR_SORT                    1
#define XV_NUM_ERR_MERGE                   1
#define XV_NUM_ERR_DELTA                   1
#define XV_NUM_ERR_MAPPING                 1
#define XV_NUM_ERR_ORBIT_EXTRA             1
#define XV_NUM_ERR_GEN_SWATH               1
#define XV_NUM_ERR_GEN_SCF                 1
#define XV_NUM_ERR_SWATH_ID_INIT           1 /* AN-468 */
#define XV_NUM_ERR_SWATH_ID_CLOSE          1 /* AN-468 */
#define XV_NUM_ERR_ZONEVISTIME_COMPUTE     2 /* AN-468 */
#define XV_NUM_ERR_STATIONVISTIME_COMPUTE  2 /* AN-468 */
#define XV_NUM_ERR_SWATHPOS_COMPUTE        1 /* AN-468 */
#define XV_NUM_ERR_COMPUTE_NOT             1 /* AN-468 */
#define XV_NUM_ERR_COMPUTE_OR              1 /* AN-468 */
#define XV_NUM_ERR_COMPUTE_AND             1 /* AN-468 */
#define XV_NUM_ERR_COMPUTE_SORT            1 /* AN-468 */
#define XV_NUM_ERR_COMPUTE_MERGE           1 /* AN-468 */
#define XV_NUM_ERR_COMPUTE_DELTA           1 /* AN-468 */
#define XV_NUM_ERR_COMPUTE_MAPPING         1 /* AN-468 */
#define XV_NUM_ERR_ZONEVISTIME_COVERAGE    1 /* AN-505 */
#define XV_NUM_ERR_SET_SC_VIS_TIME_STEP    1 // AN-628
#define XV_NUM_ERR_COMPUTE_AOI             1 /* AN-672 */

/* Max length of ierr vectors */
/* -------------------------- */
#define XV_ERR_VECTOR_MAX_LENGTH 3
/* Max of the above lengths.
   This value can be safely used in every function declaration. */

/* AN-628 */
#define xv_link_data  xd_link_data
#define xv_link_mask  xd_link_mask
#define xv_az_el_mask xd_az_el_mask
/* End AN-628 */

/*******************************************************************************
 *
 * Main CFI Functions - prototype declaration
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
  /* AN-430 */

  /* END AN-430 */

  /* AN-468 */
  typedef struct
  {
    long type; /* XV_Swath_init_enum */
    char* filename;
    xd_sdf_file* sdf_file;
    xd_stf_file* stf_file;
    long nof_regen_orbits;
  } xv_swath_info;

  typedef struct
  {
    void* ee_id;
  } xv_swath_id;

  /* Swath id accesor */
  typedef struct
  {
    long type; /* XV_Swath_init_enum */
    char* filename;
    xd_sdf_file* sdf_file;
    xd_stf_file* stf_file;
    long nof_regen_orbits;

    // The following arguments are only used in getter function
    long orbit_ref; // Orbit for which the STF has been generated
    long val_orbit_start; // Start validity orbit for STF
    long val_orbit_stop; // Stop validity orbit for STF
  } xv_swath_id_data;

  /* I/F definition: */
  typedef struct
  {
    long type; /* XV_Time_type_enum */
    double utc_time;
    long orbit_type; /* abs or rel (XO_Orbit_type_enum) */
    long orbit_num;
    long cycle;
    long sec; /* after ANX */
    long msec; /* after ANX */
  } xv_time;

  typedef struct
  {
    xv_time tstart;
    xv_time tstop;
  } xv_time_interval;
  /* end ANR-596 */

  typedef struct
  {
    char* zone_id;
    long coverage;
  } xv_zone_coverage_info;

  typedef struct
  {
    long num_rec;
    xv_zone_coverage_info* zone_coverage_info;
  } xv_zone_coverage_info_list;

  typedef struct
  {
    xv_time_interval time_interval;
    xv_zone_coverage_info_list zone_coverage_info_list;
  } xv_zonevisibility_interval;

  typedef struct
  {
    long num_rec;
    xv_zonevisibility_interval* visibility_interval;
  } xv_zonevisibility_interval_list;

  /*
if type = XV_USE_ZONE_DB_FILE: zone information is taken from zone_db file
if type = XV_USE_ZONE_DATA: zone information is taken from zone_data
*/
  typedef struct
  {
    char* zone_id; /* to be used if type = USE XV_USE_ZONE_DB_FILE */
    long type; /* XV_Zone_type_enum */
    char* zone_db_filename; /* to be used if type = XV_USE_ZONE_DB_FILE */
    xd_zone_rec* zone_data; /* to be used if type = XV_USE_ZONE_DATA */
    long projection; /* XD_Proj_enum */
    double min_duration;
  } xv_zone_info;

  typedef struct
  {
    long calc_flag; /* indicates if zone_coverage_info_list has to be computed (XV_Compute_flag_enum) */
    long num_rec;
    xv_zone_info* zone_info;
  } xv_zone_info_list;

  /*
if type = XV_USE_STATION_FILE, XV_USE_STATION_FILE_AND_MASK_OVERRIDE: 
   station information is taken from station_db file
if type = XV_USE_STATION_DATA, XV_USE_STATION_DATA_AND_MASK_OVERRIDE: 
   station information is taken from station_data
*/
  typedef struct
  {
    long type; /* XV_Station_type_enum */
    char* station_id;
    char* station_db_filename;
    xd_station_rec* station_data;
    double default_aos, default_los;
    long default_mask;
    double min_duration;
  } xv_station_info;

  /* default_aos/los/mask shall be used either:
- if type = XV_USE_STATION_FILE_AND_MASK_OVERRIDE or
- if type = XV_USE_STATION_DATA_AND_MASK_OVERRIDE or
- if type = XV_USE_STATION_FILE and aos/los/mask are not defined in the file
- if type = XV_USE_STATION_DATA and aos/los/mask are not defined in the data
*/
  typedef struct
  {
    long calc_flag; /* flag to indicate if additional results have to be computed (XV_Compute_flag_enum) */
    long num_rec;
    xv_station_info* station_info;
  } xv_station_info_list;

  typedef struct
  {
    char* station_id;
    xv_time zdop_time;
  } xv_station_coverage_info;

  typedef struct
  {
    long num_rec;
    xv_station_coverage_info* station_coverage_info;
  } xv_station_coverage_info_list;

  typedef struct
  {
    xv_time_interval time_interval;
    xv_station_coverage_info_list station_coverage_info_list;
  } xv_stationvisibility_interval;

  typedef struct
  {
    long num_rec;
    xv_stationvisibility_interval* visibility_interval;
  } xv_stationvisibility_interval_list;

  typedef struct
  {
    double lon;
    double lat;
    double alt;
  } xv_swath_point;

  typedef struct
  {
    long num_rec;
    xv_swath_point* swath_point;
  } xv_swath_point_list;

  typedef struct
  {
    xv_time_interval time_interval;
  } xv_visibility_interval;

  typedef struct
  {
    long num_rec;
    xv_visibility_interval* visibility_interval;
  } xv_visibility_interval_list;
  /* END AN-468 */

  /* AN-505 */
  typedef struct
  {
    long type_coverage; /* (XV_Type_Coverage_enum) COVERAGE_FIXED_DISTANCE, COVERAGE_PERCENTAGE_PRECISION */
    double point_geod_distance; /* distance between contigous points (Km) */
    double percent_precision; /* expected precision in percentage */
    xo_orbit_id* orbit_id;
    xp_attitude_def* attitude_def;
    xv_swath_id* swath_id;
    xv_zone_info* zone_info;
    xv_zonevisibility_interval_list* visibility_interval_list;
  } xv_zonevisibility_coverage_in;

  typedef struct
  {
    double zone_area; /* zone area in km2 */
    double total_coverage; /* percentage of zone covered by all intervals, i.e. 100% - percentage of zone not covered by any interval */
    double* coverage_per_interval; /* array, item with index i (0,1,2,...N-1) is the percentage of zone covered by interval i+1 (1,2,...N) only */
    double* coverage_by_N_intervals; /* array, item with index i (0,1,2,...N-1) is the percentage of the zone covered by exactly i+1 (1,2,...N) intervals */
    double* cumulative_coverage; /* array, with index i (0,1,2,...N-1)  is the percentage of zone covered by intervals 1,2...i+1 considered together */

  } xv_zonevisibility_coverage_out;

  /* END AN-505 */

  /* AN-843 */
  typedef struct
  {
    xv_time_interval time_interval;
    long event_type; // XD_Event_type_enum
    char* event_id; // NULL, in case event_type == XD_TIME_INTERVAL_NONE
    xd_scf_appear appearance;
  } xv_timeline_interval;

  typedef struct
  {
    long num_rec;
    xv_timeline_interval* timeline_interval;
  } xv_timeline_interval_list;

  /*******************************************************************************
*
* Main CFI Functions - prototype declaration
*
*******************************************************************************/
  long xv_zone_vis_time(xo_orbit_id* orbit_id,
                        long* orbit_type,
                        long* start_orbit,
                        long* start_cycle,
                        long* stop_orbit,
                        long* stop_cycle,
                        long* swath_flag,
                        char* swath_file,
                        char* zone_id,
                        char* zone_db_file,
                        long* projection,
                        long* zone_num,
                        double* zone_long,
                        double* zone_lat,
                        double* zone_diam,
                        double* min_duration,
                        /* outputs */
                        long* number_segments,
                        long** bgn_orbit,
                        long** bgn_second,
                        long** bgn_microsec,
                        long** bgn_cycle,
                        long** end_orbit,
                        long** end_second,
                        long** end_microsec,
                        long** end_cycle,
                        long** coverage,
                        long ierr[XV_NUM_ERR_ZONE_VIS_TIME]);

  long xv_zone_vis_time_run(long* run_id,
                            long* orbit_type,
                            long* start_orbit,
                            long* start_cycle,
                            long* stop_orbit,
                            long* stop_cycle,
                            long* swath_flag,
                            char* swath_file,
                            char* zone_id,
                            char* zone_db_file,
                            long* projection,
                            long* zone_num,
                            double* zone_long,
                            double* zone_lat,
                            double* zone_diam,
                            double* min_duration,
                            /* outputs */
                            long* number_segments,
                            long** bgn_orbit,
                            long** bgn_second,
                            long** bgn_microsec,
                            long** bgn_cycle,
                            long** end_orbit,
                            long** end_second,
                            long** end_microsec,
                            long** end_cycle,
                            long** coverage,
                            long ierr[XV_NUM_ERR_ZONE_VIS_TIME]);

  long xv_zone_vis_time_no_file(xo_orbit_id* orbit_id,
                                long* orbit_type,
                                long* start_orbit,
                                long* start_cycle,
                                long* stop_orbit,
                                long* stop_cycle,
                                xd_stf_file* stf_data,
                                xd_zone_rec* zone_data,
                                long* projection,
                                long* zone_num,
                                double* zone_long,
                                double* zone_lat,
                                double* zone_diam,
                                double* min_duration,
                                /* outputs */
                                long* number_segments,
                                long** bgn_orbit,
                                long** bgn_second,
                                long** bgn_microsec,
                                long** bgn_cycle,
                                long** end_orbit,
                                long** end_second,
                                long** end_microsec,
                                long** end_cycle,
                                long** coverage,
                                long ierr[XV_NUM_ERR_ZONE_VIS_TIME]);

  long xv_zone_vis_time_no_file_run(long* run_id,
                                    long* orbit_type,
                                    long* start_orbit,
                                    long* start_cycle,
                                    long* stop_orbit,
                                    long* stop_cycle,
                                    xd_stf_file* stf_data,
                                    xd_zone_rec* zone_data,
                                    long* projection,
                                    long* zone_num,
                                    double* zone_long,
                                    double* zone_lat,
                                    double* zone_diam,
                                    double* min_duration,
                                    /* outputs */
                                    long* number_segments,
                                    long** bgn_orbit,
                                    long** bgn_second,
                                    long** bgn_microsec,
                                    long** bgn_cycle,
                                    long** end_orbit,
                                    long** end_second,
                                    long** end_microsec,
                                    long** end_cycle,
                                    long** coverage,
                                    long ierr[XV_NUM_ERR_ZONE_VIS_TIME]);

  long xv_station_vis_time(xo_orbit_id* orbit_id,
                           long* orbit_type,
                           long* start_orbit,
                           long* start_cycle,
                           long* stop_orbit,
                           long* stop_cycle,
                           long* swath_flag,
                           char* orbit_swath_file,
                           char* sta_id,
                           char* sta_db_file,
                           long* mask,
                           double* aos_elevation,
                           double* los_elevation,
                           double* min_duration,
                           /* output */
                           long* number_segments,
                           long** bgn_orbit,
                           long** bgn_second,
                           long** bgn_microsec,
                           long** bgn_cycle,
                           long** end_orbit,
                           long** end_second,
                           long** end_microsec,
                           long** end_cycle,
                           long** zdop_orbit,
                           long** zdop_second,
                           long** zdop_microsec,
                           long** zdop_cycle,
                           long ierr[XV_NUM_ERR_STATION_VIS_TIME]);

  long xv_station_vis_time_run(long* run_id,
                               long* orbit_type,
                               long* start_orbit,
                               long* start_cycle,
                               long* stop_orbit,
                               long* stop_cycle,
                               long* swath_flag,
                               char* orbit_swath_file,
                               char* sta_id,
                               char* sta_db_file,
                               long* mask,
                               double* aos_elevation,
                               double* los_elevation,
                               double* min_duration,
                               /* output */
                               long* number_segments,
                               long** bgn_orbit,
                               long** bgn_second,
                               long** bgn_microsec,
                               long** bgn_cycle,
                               long** end_orbit,
                               long** end_second,
                               long** end_microsec,
                               long** end_cycle,
                               long** zdop_orbit,
                               long** zdop_second,
                               long** zdop_microsec,
                               long** zdop_cycle,
                               long ierr[XV_NUM_ERR_STATION_VIS_TIME]);

  long xv_station_vis_time_no_file(xo_orbit_id* orbit_id,
                                   long* orbit_type,
                                   long* start_orbit,
                                   long* start_cycle,
                                   long* stop_orbit,
                                   long* stop_cycle,
                                   xd_stf_file* stf_data,
                                   xd_station_rec* station_data,
                                   long* mask,
                                   double* aos_elevation,
                                   double* los_elevation,
                                   double* min_duration,
                                   /* output */
                                   long* number_segments,
                                   long** bgn_orbit,
                                   long** bgn_second,
                                   long** bgn_microsec,
                                   long** bgn_cycle,
                                   long** end_orbit,
                                   long** end_second,
                                   long** end_microsec,
                                   long** end_cycle,
                                   long** zdop_orbit,
                                   long** zdop_second,
                                   long** zdop_microsec,
                                   long** zdop_cycle,
                                   long ierr[XV_NUM_ERR_STATION_VIS_TIME]);

  long xv_station_vis_time_no_file_run(long* run_id,
                                       long* orbit_type,
                                       long* start_orbit,
                                       long* start_cycle,
                                       long* stop_orbit,
                                       long* stop_cycle,
                                       xd_stf_file* stf_data,
                                       xd_station_rec* station_data,
                                       long* mask,
                                       double* aos_elevation,
                                       double* los_elevation,
                                       double* min_duration,
                                       /* output */
                                       long* number_segments,
                                       long** bgn_orbit,
                                       long** bgn_second,
                                       long** bgn_microsec,
                                       long** bgn_cycle,
                                       long** end_orbit,
                                       long** end_second,
                                       long** end_microsec,
                                       long** end_cycle,
                                       long** zdop_orbit,
                                       long** zdop_second,
                                       long** zdop_microsec,
                                       long** zdop_cycle,
                                       long ierr[XV_NUM_ERR_STATION_VIS_TIME]);

  /* AN-468 */
  long xv_zonevistime_compute(xo_orbit_id* orbit_id,
                              xp_attitude_def* attitude_def,
                              xv_swath_id* swath_id,
                              xv_zone_info_list* zone_info_list,
                              xv_time_interval* search_interval,
                              /*output*/
                              xv_zonevisibility_interval_list* visibility_interval_list,
                              long ierr[XV_NUM_ERR_ZONEVISTIME_COMPUTE]);

  long xv_stationvistime_compute(xo_orbit_id* orbit_id,
                                 xp_attitude_def* attitude_def,
                                 xv_swath_id* swath_id,
                                 xv_station_info_list* station_info_list,
                                 xv_time_interval* search_interval,
                                 /*output*/
                                 xv_stationvisibility_interval_list* stationvisibility_interval_list,
                                 long ierr[XV_NUM_ERR_STATIONVISTIME_COMPUTE]);

  long xv_swathpos_compute(xo_orbit_id* orbit_id,
                           xv_swath_id* swath_id,
                           xv_time* swathpos_time,
                           /*output*/
                           xv_swath_point_list* swath_point_list,
                           long ierr[XV_NUM_ERR_SWATHPOS_COMPUTE]);

  long xv_compute_aoi(xo_orbit_id* orbit_id, xv_swath_id* swath_id, xv_time_interval time_interval, double min_lon, double max_lon, double min_lat, double max_lat,
                      /* output */
                      double lon[4],
                      double lat[4],
                      long ierr[XV_NUM_ERR_COMPUTE_AOI]);

  xp_atmos_id xv_swath_get_atmos_id(xv_swath_id* swath_id);

  /* END AN-468 */

  long xv_sc_vis_time(xo_orbit_id* orbit_id1,
                      xp_sat_nom_trans_id* sat_nom_trans_id1,
                      xp_sat_trans_id* sat_trans_id1,
                      xp_instr_trans_id* instr_trans_id1,
                      long* orbit_type,
                      long* start_orbit,
                      long* start_cycle,
                      long* stop_orbit,
                      long* stop_cycle,
                      xo_orbit_id* orbit_id2,
                      xp_sat_nom_trans_id* sat_nom_trans_id2,
                      xp_sat_trans_id* sat_trans_id2,
                      xp_instr_trans_id* instr_trans_id2,
                      xv_link_data* link_data,
                      double* min_duration,
                      /* Output */
                      long* number_segments,
                      long** bgn_orbit,
                      long** bgn_second,
                      long** bgn_microsec,
                      long** bgn_cycle,
                      long** end_orbit,
                      long** end_second,
                      long** end_microsec,
                      long** end_cycle,
                      long ierr[XV_NUM_ERR_SC_VIS_TIME]);

  long xv_sc_vis_time_run(long* run_id1, long* run_id2, long* orbit_type, long* start_orbit, long* start_cycle, long* stop_orbit, long* stop_cycle, xv_link_data* link_data, double* min_duration,
                          /* Output */
                          long* number_segments,
                          long** bgn_orbit,
                          long** bgn_second,
                          long** bgn_microsec,
                          long** bgn_cycle,
                          long** end_orbit,
                          long** end_second,
                          long** end_microsec,
                          long** end_cycle,
                          long ierr[XV_NUM_ERR_SC_VIS_TIME]);

  // AN-628
  long xv_set_sc_vis_time_step(xo_orbit_id* orbit_id, double step, long ierr[XV_NUM_ERR_SET_SC_VIS_TIME_STEP]);

  /* AN-677 */
  long xv_celestial_body_vis_time(xo_orbit_id* orbit_id1,
                                  xp_sat_nom_trans_id* sat_nom_trans_id1,
                                  xp_sat_trans_id* sat_trans_id1,
                                  xp_instr_trans_id* instr_trans_id1,
                                  long* orbit_type,
                                  long* start_orbit,
                                  long* start_cycle,
                                  long* stop_orbit,
                                  long* stop_cycle,
                                  long* celestial_type,
                                  xv_link_data* link_data,
                                  double* min_duration,
                                  /* Output */
                                  long* number_segments,
                                  long** bgn_orbit,
                                  long** bgn_second,
                                  long** bgn_microsec,
                                  long** bgn_cycle,
                                  long** end_orbit,
                                  long** end_second,
                                  long** end_microsec,
                                  long** end_cycle,
                                  long ierr[XV_NUM_ERR_SC_VIS_TIME]);

  long xv_swath_pos(xo_orbit_id* orbit_id,
                    xd_stf_file* stf_data,
                    long* orbit_type,
                    long* orb_num,
                    long* second,
                    long* microsec,
                    long* cycle,
                    double* longi,
                    double* lati,
                    double* alti,
                    long ierr[XV_NUM_ERR_SWATH_POS]);

  long xv_swath_pos_run(long* run_id,
                        xd_stf_file* stf_data,
                        long* orbit_type,
                        long* orb_num,
                        long* second,
                        long* microsec,
                        long* cycle,
                        double* longi,
                        double* lati,
                        double* alti,
                        long ierr[XV_NUM_ERR_SWATH_POS]);

  long xv_star_vis_time(xo_orbit_id* orbit_id,
                        long* orbit_type,
                        long* start_orbit,
                        long* start_cycle,
                        long* stop_orbit,
                        long* stop_cycle,
                        long* swath_flag,
                        char* swath_file_upper,
                        char* swath_file_lower,
                        char* star_id,
                        char* star_db_file,
                        double* star_ra,
                        double* star_dec,
                        double* min_duration,
                        /* output */
                        double* star_ra_deg,
                        double* star_dec_deg,
                        long* number_segments,
                        long** bgn_orbit,
                        long** bgn_second,
                        long** bgn_microsec,
                        long** bgn_cycle,
                        long** bgn_coverage,
                        long** end_orbit,
                        long** end_second,
                        long** end_microsec,
                        long** end_cycle,
                        long** end_coverage,
                        long ierr[XV_NUM_ERR_STAR_VIS_TIME]);

  long xv_star_vis_time_run(long* run_id,
                            long* orbit_type,
                            long* start_orbit,
                            long* start_cycle,
                            long* stop_orbit,
                            long* stop_cycle,
                            long* swath_flag,
                            char* swath_file_upper,
                            char* swath_file_lower,
                            char* star_id,
                            char* star_db_file,
                            double* star_ra,
                            double* star_dec,
                            double* min_duration,
                            /* output */
                            double* star_ra_deg,
                            double* star_dec_deg,
                            long* number_segments,
                            long** bgn_orbit,
                            long** bgn_second,
                            long** bgn_microsec,
                            long** bgn_cycle,
                            long** bgn_coverage,
                            long** end_orbit,
                            long** end_second,
                            long** end_microsec,
                            long** end_cycle,
                            long** end_coverage,
                            long ierr[XV_NUM_ERR_STAR_VIS_TIME]);

  long xv_multizones_vis_time(xo_orbit_id* orbit_id,
                              long* orbit_type,
                              long* start_orbit,
                              long* start_cycle,
                              long* stop_orbit,
                              long* stop_cycle,
                              long* swath_flag,
                              char* swath_file,
                              long* num_zones,
                              char** zone_id,
                              char* zone_db_file,
                              long* projection,
                              long* zone_num,
                              double* zone_long,
                              double* zone_lat,
                              double* zone_diam,
                              double* min_duration,
                              long* extra_info_flag,
                              /* outputs */
                              long* number_segments,
                              long** bgn_orbit,
                              long** bgn_second,
                              long** bgn_microsec,
                              long** bgn_cycle,
                              long** end_orbit,
                              long** end_second,
                              long** end_microsec,
                              long** end_cycle,
                              long** nb_zon_in_segment,
                              long*** zones_in_segment,
                              long*** coverage,
                              long ierr[XV_NUM_ERR_MULTIZONES_VIS_TIME]);

  long xv_multizones_vis_time_run(long* run_id,
                                  long* orbit_type,
                                  long* start_orbit,
                                  long* start_cycle,
                                  long* stop_orbit,
                                  long* stop_cycle,
                                  long* swath_flag,
                                  char* swath_file,
                                  long* num_zones,
                                  char** zone_id,
                                  char* zone_db_file,
                                  long* projection,
                                  long* zone_num,
                                  double* zone_long,
                                  double* zone_lat,
                                  double* zone_diam,
                                  double* min_duration,
                                  long* extra_info_flag,
                                  /* outputs */
                                  long* number_segments,
                                  long** bgn_orbit,
                                  long** bgn_second,
                                  long** bgn_microsec,
                                  long** bgn_cycle,
                                  long** end_orbit,
                                  long** end_second,
                                  long** end_microsec,
                                  long** end_cycle,
                                  long** nb_zon_in_segment,
                                  long*** zones_in_segment,
                                  long*** coverage,
                                  long ierr[XV_NUM_ERR_MULTIZONES_VIS_TIME]);

  long xv_multistations_vis_time(xo_orbit_id* orbit_id,
                                 long* orbit_type,
                                 long* start_orbit,
                                 long* start_cycle,
                                 long* stop_orbit,
                                 long* stop_cycle,
                                 long* swath_flag,
                                 char* swath_file,
                                 long* nb_stations,
                                 char* stat_db_file,
                                 char** station_id,
                                 double* aos_elevation,
                                 double* los_elevation,
                                 long* mask,
                                 double* min_duration,
                                 long* extra_info_flag,
                                 /* outputs */
                                 long* number_segments,
                                 long** bgn_orbit,
                                 long** bgn_second,
                                 long** bgn_microsec,
                                 long** bgn_cycle,
                                 long** end_orbit,
                                 long** end_second,
                                 long** end_microsec,
                                 long** end_cycle,
                                 long*** zdop_orbit,
                                 long*** zdop_second,
                                 long*** zdop_microsec,
                                 long*** zdop_cycle,
                                 long** nb_stat_in_segment,
                                 long*** stat_in_segment,
                                 long ierr[XV_NUM_ERR_MULTISTATIONS_VIS_TIME]);

  long xv_multistations_vis_time_run(long* run_id,
                                     long* orbit_type,
                                     long* start_orbit,
                                     long* start_cycle,
                                     long* stop_orbit,
                                     long* stop_cycle,
                                     long* swath_flag,
                                     char* swath_file,
                                     long* nb_stations,
                                     char* stat_db_file,
                                     char** station_id,
                                     double* aos_elevation,
                                     double* los_elevation,
                                     long* mask,
                                     double* min_duration,
                                     long* extra_info_flag,
                                     /* outputs */
                                     long* number_segments,
                                     long** bgn_orbit,
                                     long** bgn_second,
                                     long** bgn_microsec,
                                     long** bgn_cycle,
                                     long** end_orbit,
                                     long** end_second,
                                     long** end_microsec,
                                     long** end_cycle,
                                     long*** zdop_orbit,
                                     long*** zdop_second,
                                     long*** zdop_microsec,
                                     long*** zdop_cycle,
                                     long** nb_stat_in_segment,
                                     long*** stat_in_segment,
                                     long ierr[XV_NUM_ERR_MULTISTATIONS_VIS_TIME]);

  long xv_orbit_extra(xo_orbit_id* orbit_id,
                      long* orbit,
                      double* orbit_info_vector,
                      long* num_sza,
                      double* sza,
                      /*outputs*/
                      double** sza_up,
                      double** sza_down,
                      double* eclipse_entry,
                      double* eclipse_exit,
                      double* sun_moon_entry,
                      double* sun_moon_exit,
                      long ierr[XV_NUM_ERR_ORBIT_EXTRA]);

  long xv_orbit_extra_run(long* run_id,
                          long* orbit,
                          double* orbit_info_vector,
                          long* num_sza,
                          double* sza,
                          /*outputs*/
                          double** sza_up,
                          double** sza_down,
                          double* eclipse_entry,
                          double* eclipse_exit,
                          double* sun_moon_entry,
                          double* sun_moon_exit,
                          long ierr[XV_NUM_ERR_ORBIT_EXTRA]);

  long xv_gen_scf(xo_orbit_id* orbit_id,
                  char* instrument,
                  xv_timeline_interval_list* segments,
                  char* dir_name,
                  char scf_filename[XD_MAX_STR],
                  char* file_class,
                  long* version_number,
                  char* fh_system,
                  long ierr[XV_NUM_ERR_GEN_SCF]);

  long xv_gen_scf_run(long* run_id,
                      char* instrument,
                      xv_timeline_interval_list* segments,
                      char* dir_name,
                      char scf_filename[XD_MAX_STR],
                      char* file_class,
                      long* version_number,
                      char* fh_system,
                      long ierr[XV_NUM_ERR_GEN_SCF]);

  long xv_gen_swath(xo_orbit_id* orbit_id, xp_atmos_id* atmos_id, long* requested_orbit, char* swath_def_file, char* dir_name, char* swath_filename, char* file_class, long* version_number, char* fh_system, long ierr[]);

  long xv_gen_swath_run(long* run_id,
                        long* requested_orbit,
                        char* swath_definition_file,
                        char* dir_name,
                        char* swath_filename,
                        char* file_class,
                        long* version_number,
                        char* fh_system,
                        long ierr[]);

  long xv_gen_swath_no_file(xo_orbit_id* orbit_id, xp_atmos_id* atmos_id, long* requested_orbit, xd_sdf_file* sdf, xd_stf_file* stf, long ierr[]);

  long xv_gen_swath_no_file_run(long* run_id,
                                long* requested_orbit,
                                xd_sdf_file* sdf,
                                xd_stf_file* stf,
                                long ierr[]);

  /*  Time Segments Manipulation  */
  /*  --------------------------  */
  long xv_time_segments_not(xo_orbit_id* orbit_id,
                            long* orbit_type,
                            long* order_switch,
                            long* num_segments_in, /* input segment list */
                            long* bgn_orbit_in,
                            long* bgn_secs_in,
                            long* bgn_microsecs_in,
                            long* bgn_cycle_in,
                            long* end_orbit_in,
                            long* end_secs_in,
                            long* end_microsecs_in,
                            long* end_cycle_in,
                            long* num_segments_out, /* output segment list */
                            long** bgn_orbit_out,
                            long** bgn_secs_out,
                            long** bgn_microsecs_out,
                            long** bgn_cycle_out,
                            long** end_orbit_out,
                            long** end_secs_out,
                            long** end_microsecs_out,
                            long** end_cycle_out,
                            long ierr[XV_NUM_ERR_NOT]);

  long xv_time_segments_not_run(long* run_id,
                                long* orbit_type,
                                long* order_switch,
                                long* num_segments_in, /* input segment list */
                                long* bgn_orbit_in,
                                long* bgn_secs_in,
                                long* bgn_microsecs_in,
                                long* bgn_cycle_in,
                                long* end_orbit_in,
                                long* end_secs_in,
                                long* end_microsecs_in,
                                long* end_cycle_in,
                                long* num_segments_out, /* output segment list */
                                long** bgn_orbit_out,
                                long** bgn_secs_out,
                                long** bgn_microsecs_out,
                                long** bgn_cycle_out,
                                long** end_orbit_out,
                                long** end_secs_out,
                                long** end_microsecs_out,
                                long** end_cycle_out,
                                long ierr[XV_NUM_ERR_NOT]);

  long xv_time_segments_or(xo_orbit_id* orbit_id,
                           long* orbit_type,
                           long* order_switch,
                           long* num_segments_1, /* input segment list 1*/
                           long* bgn_orbit_1,
                           long* bgn_secs_1,
                           long* bgn_microsecs_1,
                           long* bgn_cycle_1,
                           long* end_orbit_1,
                           long* end_secs_1,
                           long* end_microsecs_1,
                           long* end_cycle_1,
                           long* num_segments_2, /* input segment list 2*/
                           long* bgn_orbit_2,
                           long* bgn_secs_2,
                           long* bgn_microsecs_2,
                           long* bgn_cycle_2,
                           long* end_orbit_2,
                           long* end_secs_2,
                           long* end_microsecs_2,
                           long* end_cycle_2,
                           long* num_segments_out, /* output segment list */
                           long** bgn_orbit_out,
                           long** bgn_secs_out,
                           long** bgn_microsecs_out,
                           long** bgn_cycle_out,
                           long** end_orbit_out,
                           long** end_secs_out,
                           long** end_microsecs_out,
                           long** end_cycle_out,
                           long ierr[XV_NUM_ERR_OR]);

  long xv_time_segments_or_run(long* run_id,
                               long* orbit_type,
                               long* order_switch,
                               long* num_segments_1, /* input segment list 1*/
                               long* bgn_orbit_1,
                               long* bgn_secs_1,
                               long* bgn_microsecs_1,
                               long* bgn_cycle_1,
                               long* end_orbit_1,
                               long* end_secs_1,
                               long* end_microsecs_1,
                               long* end_cycle_1,
                               long* num_segments_2, /* input segment list 2*/
                               long* bgn_orbit_2,
                               long* bgn_secs_2,
                               long* bgn_microsecs_2,
                               long* bgn_cycle_2,
                               long* end_orbit_2,
                               long* end_secs_2,
                               long* end_microsecs_2,
                               long* end_cycle_2,
                               long* num_segments_out, /* output segment list */
                               long** bgn_orbit_out,
                               long** bgn_secs_out,
                               long** bgn_microsecs_out,
                               long** bgn_cycle_out,
                               long** end_orbit_out,
                               long** end_secs_out,
                               long** end_microsecs_out,
                               long** end_cycle_out,
                               long ierr[XV_NUM_ERR_OR]);

  long xv_time_segments_and(xo_orbit_id* orbit_id,
                            long* orbit_type,
                            long* order_switch,
                            long* num_segments_1, /* input segment list 1*/
                            long* bgn_orbit_1,
                            long* bgn_secs_1,
                            long* bgn_microsecs_1,
                            long* bgn_cycle_1,
                            long* end_orbit_1,
                            long* end_secs_1,
                            long* end_microsecs_1,
                            long* end_cycle_1,
                            long* num_segments_2, /* input segment list 2*/
                            long* bgn_orbit_2,
                            long* bgn_secs_2,
                            long* bgn_microsecs_2,
                            long* bgn_cycle_2,
                            long* end_orbit_2,
                            long* end_secs_2,
                            long* end_microsecs_2,
                            long* end_cycle_2,
                            long* num_segments_out, /* output segment list */
                            long** bgn_orbit_out,
                            long** bgn_secs_out,
                            long** bgn_microsecs_out,
                            long** bgn_cycle_out,
                            long** end_orbit_out,
                            long** end_secs_out,
                            long** end_microsecs_out,
                            long** end_cycle_out,
                            long ierr[XV_NUM_ERR_AND]);

  long xv_time_segments_and_run(long* run_id,
                                long* orbit_type,
                                long* order_switch,
                                long* num_segments_1, /* input segment list 1*/
                                long* bgn_orbit_1,
                                long* bgn_secs_1,
                                long* bgn_microsecs_1,
                                long* bgn_cycle_1,
                                long* end_orbit_1,
                                long* end_secs_1,
                                long* end_microsecs_1,
                                long* end_cycle_1,
                                long* num_segments_2, /* input segment list 2*/
                                long* bgn_orbit_2,
                                long* bgn_secs_2,
                                long* bgn_microsecs_2,
                                long* bgn_cycle_2,
                                long* end_orbit_2,
                                long* end_secs_2,
                                long* end_microsecs_2,
                                long* end_cycle_2,
                                long* num_segments_out, /* output segment list */
                                long** bgn_orbit_out,
                                long** bgn_secs_out,
                                long** bgn_microsecs_out,
                                long** bgn_cycle_out,
                                long** end_orbit_out,
                                long** end_secs_out,
                                long** end_microsecs_out,
                                long** end_cycle_out,
                                long ierr[XV_NUM_ERR_AND]);

  long xv_time_segments_sort(xo_orbit_id* orbit_id,
                             long* orbit_type,
                             long* sort_criteria,
                             long* num_segments, /* input segment list */
                             long* bgn_orbit,
                             long* bgn_secs,
                             long* bgn_microsecs,
                             long* bgn_cycle,
                             long* end_orbit,
                             long* end_secs,
                             long* end_microsecs,
                             long* end_cycle,
                             long ierr[XV_NUM_ERR_SORT]);

  long xv_time_segments_sort_run(long* run_id,
                                 long* orbit_type,
                                 long* sort_criteria,
                                 long* num_segments, /* input segment list */
                                 long* bgn_orbit,
                                 long* bgn_secs,
                                 long* bgn_microsecs,
                                 long* bgn_cycle,
                                 long* end_orbit,
                                 long* end_secs,
                                 long* end_microsecs,
                                 long* end_cycle,
                                 long ierr[XV_NUM_ERR_SORT]);

  long xv_time_segments_merge(xo_orbit_id* orbit_id,
                              long* orbit_type,
                              long* order_switch,
                              long* num_segments, /* input segment list 1*/
                              long* bgn_orbit,
                              long* bgn_secs,
                              long* bgn_microsecs,
                              long* bgn_cycle,
                              long* end_orbit,
                              long* end_secs,
                              long* end_microsecs,
                              long* end_cycle,
                              long* num_segments_out, /* output segment list */
                              long** bgn_orbit_out,
                              long** bgn_secs_out,
                              long** bgn_microsecs_out,
                              long** bgn_cycle_out,
                              long** end_orbit_out,
                              long** end_secs_out,
                              long** end_microsecs_out,
                              long** end_cycle_out,
                              long ierr[XV_NUM_ERR_MERGE]);

  long xv_time_segments_merge_run(long* run_id,
                                  long* orbit_type,
                                  long* order_switch,
                                  long* num_segments, /* input segment list 1*/
                                  long* bgn_orbit,
                                  long* bgn_secs,
                                  long* bgn_microsecs,
                                  long* bgn_cycle,
                                  long* end_orbit,
                                  long* end_secs,
                                  long* end_microsecs,
                                  long* end_cycle,
                                  long* num_segments_out, /* output segment list */
                                  long** bgn_orbit_out,
                                  long** bgn_secs_out,
                                  long** bgn_microsecs_out,
                                  long** bgn_cycle_out,
                                  long** end_orbit_out,
                                  long** end_secs_out,
                                  long** end_microsecs_out,
                                  long** end_cycle_out,
                                  long ierr[XV_NUM_ERR_MERGE]);

  long xv_time_segments_delta(xo_orbit_id* orbit_id,
                              long* orbit_type,
                              double* entry_offset,
                              double* exit_offset,
                              long* num_segments, /* input segment list 1*/
                              long* bgn_orbit,
                              long* bgn_secs,
                              long* bgn_microsecs,
                              long* bgn_cycle,
                              long* end_orbit,
                              long* end_secs,
                              long* end_microsecs,
                              long* end_cycle,
                              long* num_segments_out, /* output segment list */
                              long** bgn_orbit_out,
                              long** bgn_secs_out,
                              long** bgn_microsecs_out,
                              long** bgn_cycle_out,
                              long** end_orbit_out,
                              long** end_secs_out,
                              long** end_microsecs_out,
                              long** end_cycle_out,
                              long ierr[XV_NUM_ERR_DELTA]);

  long xv_time_segments_delta_run(long* run_id,
                                  long* orbit_type,
                                  double* entry_offset,
                                  double* exit_offset,
                                  long* num_segments, /* input segment list 1*/
                                  long* bgn_orbit,
                                  long* bgn_secs,
                                  long* bgn_microsecs,
                                  long* bgn_cycle,
                                  long* end_orbit,
                                  long* end_secs,
                                  long* end_microsecs,
                                  long* end_cycle,
                                  long* num_segments_out, /* output segment list */
                                  long** bgn_orbit_out,
                                  long** bgn_secs_out,
                                  long** bgn_microsecs_out,
                                  long** bgn_cycle_out,
                                  long** end_orbit_out,
                                  long** end_secs_out,
                                  long** end_microsecs_out,
                                  long** end_cycle_out,
                                  long ierr[XV_NUM_ERR_DELTA]);

  long xv_time_segments_mapping(xo_orbit_id* orbit_id,
                                long* orbit_type,
                                long* start_orbit,
                                long* start_cycle,
                                long* stop_orbit,
                                long* stop_cycle,
                                long* swath_flag,
                                char* swath_file,
                                long* zone_num,
                                char* zone_id,
                                char* zone_db_file,
                                long* projection,
                                double* zone_diam,
                                double* zone_long,
                                double* zone_lat,
                                /* outputs */
                                long* num_mappings,
                                long** num_segments,
                                long** orbit_direction,
                                long*** bgn_orbit,
                                long*** bgn_secs,
                                long*** bgn_microsecs,
                                long*** bgn_cycle,
                                long*** end_orbit,
                                long*** end_secs,
                                long*** end_microsecs,
                                long*** end_cycle,
                                long*** segment_coverage,
                                long ierr[XV_NUM_ERR_MAPPING]);

  long xv_time_segments_mapping_run(long* run_id,
                                    long* orbit_type,
                                    long* start_orbit,
                                    long* start_cycle,
                                    long* stop_orbit,
                                    long* stop_cycle,
                                    long* swath_flag,
                                    char* swath_file,
                                    long* zone_num,
                                    char* zone_id,
                                    char* zone_db_file,
                                    long* projection,
                                    double* zone_diam,
                                    double* zone_long,
                                    double* zone_lat,
                                    /* outputs */
                                    long* num_mappings,
                                    long** num_segments,
                                    long** orbit_direction,
                                    long*** bgn_orbit,
                                    long*** bgn_secs,
                                    long*** bgn_microsecs,
                                    long*** bgn_cycle,
                                    long*** end_orbit,
                                    long*** end_secs,
                                    long*** end_microsecs,
                                    long*** end_cycle,
                                    long*** segment_coverage,
                                    long ierr[XV_NUM_ERR_MAPPING]);

  /*AN-468*/
  /*  Time Segments Manipulation  */
  /*  --------------------------  */
  long xv_timesegments_compute_not(xo_orbit_id* orbit_id,
                                   long* order_switch,
                                   xv_visibility_interval_list* seg_in,
                                   /* output */
                                   xv_visibility_interval_list* seg_out,
                                   long ierr[XV_NUM_ERR_COMPUTE_NOT]);

  long xv_timesegments_compute_not_run(long* run_id,
                                       long* order_switch,
                                       xv_visibility_interval_list* seg_in,
                                       /* output */
                                       xv_visibility_interval_list* seg_out,
                                       long ierr[XV_NUM_ERR_COMPUTE_NOT]);

  long xv_timesegments_compute_or(xo_orbit_id* orbit_id,
                                  long* order_switch,
                                  xv_visibility_interval_list* seg_in1,
                                  xv_visibility_interval_list* seg_in2,
                                  /* output */
                                  xv_visibility_interval_list* seg_out,
                                  long ierr[XV_NUM_ERR_COMPUTE_OR]);

  long xv_timesegments_compute_or_run(long* run_id,
                                      long* order_switch,
                                      xv_visibility_interval_list* seg_in1,
                                      xv_visibility_interval_list* seg_in2,
                                      /* output */
                                      xv_visibility_interval_list* seg_out,
                                      long ierr[XV_NUM_ERR_COMPUTE_OR]);

  long xv_timesegments_compute_and(xo_orbit_id* orbit_id,
                                   long* order_switch,
                                   xv_visibility_interval_list* seg_in1,
                                   xv_visibility_interval_list* seg_in2,
                                   /* output */
                                   xv_visibility_interval_list* seg_out,
                                   long ierr[XV_NUM_ERR_COMPUTE_AND]);

  long xv_timesegments_compute_and_run(long* run_id,
                                       long* order_switch,
                                       xv_visibility_interval_list* seg_in1,
                                       xv_visibility_interval_list* seg_in2,
                                       /* output */
                                       xv_visibility_interval_list* seg_out,
                                       long ierr[XV_NUM_ERR_COMPUTE_AND]);

  long xv_timesegments_compute_sort(xo_orbit_id* orbit_id,
                                    long* sort_criteria,
                                    xv_visibility_interval_list* seg_in,
                                    long ierr[XV_NUM_ERR_COMPUTE_SORT]);

  long xv_timesegments_compute_sort_run(long* run_id,
                                        long* sort_criteria,
                                        xv_visibility_interval_list* seg_in,
                                        long ierr[XV_NUM_ERR_COMPUTE_SORT]);

  long xv_timesegments_compute_merge(xo_orbit_id* orbit_id,
                                     long* order_switch,
                                     xv_visibility_interval_list* seg_in,
                                     /* output */
                                     xv_visibility_interval_list* seg_out,
                                     long ierr[XV_NUM_ERR_COMPUTE_MERGE]);

  long xv_timesegments_compute_merge_run(long* run_id,
                                         long* order_switch,
                                         xv_visibility_interval_list* seg_in,
                                         /* output */
                                         xv_visibility_interval_list* seg_out,
                                         long ierr[XV_NUM_ERR_COMPUTE_MERGE]);

  long xv_timesegments_compute_delta(xo_orbit_id* orbit_id,
                                     double* entry_offset,
                                     double* exit_offset,
                                     xv_visibility_interval_list* seg_in,
                                     /* output */
                                     xv_visibility_interval_list* seg_out,
                                     long ierr[XV_NUM_ERR_COMPUTE_DELTA]);

  long xv_timesegments_compute_delta_run(long* run_id,
                                         double* entry_offset,
                                         double* exit_offset,
                                         xv_visibility_interval_list* seg_in,
                                         /* output */
                                         xv_visibility_interval_list* seg_out,
                                         long ierr[XV_NUM_ERR_COMPUTE_DELTA]);

  long xv_timesegments_compute_mapping(xo_orbit_id* orbit_id,
                                       xv_swath_id* swath_id,
                                       xv_zone_info_list* zone_info_list,
                                       xv_time_interval* search_interval,
                                       /*outpus*/
                                       long* num_mappings,
                                       long** orbit_direction,
                                       xv_zonevisibility_interval_list** seg_out,
                                       long ierr[XV_NUM_ERR_COMPUTE_MAPPING]);

  long xv_timesegments_compute_mapping_run(long* run_id,
                                           xv_swath_id* swath_id,
                                           xv_zone_info_list* zone_info_list,
                                           xv_time_interval* search_interval,
                                           /*outpus*/
                                           long* num_mappings,
                                           long** orbit_direction,
                                           xv_zonevisibility_interval_list** seg_out,
                                           long ierr[XV_NUM_ERR_COMPUTE_MAPPING]);
  /* END AN-468 */

  /* Check libraries compatibility */
  long xv_check_library_version();

/* Alias for xx_check_library_version */
#ifdef expcfi_check_libs

  #undef expcfi_check_libs

#endif

#define expcfi_check_libs xv_check_library_version

  /* AN-468 */
  long xv_swath_id_init(xv_swath_info* swath_info,
                        xp_atmos_id* atmos_id,
                        xv_swath_id* swath_id,
                        long ierr[XV_NUM_ERR_SWATH_ID_INIT]);

  long xv_swath_init_status(xv_swath_id* swath_id);

  long xv_swath_id_close(xv_swath_id* swath_id,
                         long ierr[XV_NUM_ERR_SWATH_ID_CLOSE]);

  long xv_swath_get_id_data(xv_swath_id* swath_id, xv_swath_id_data* swath_data);
  long xv_swath_set_id_data(xv_swath_id* swath_id, xv_swath_id_data* swath_data);
  /* END AN-468 */

  long xv_zonevistime_coverage(xv_zonevisibility_coverage_in* zonevisibility_coverage_in,
                               /* outputs */
                               xv_zonevisibility_coverage_out* zonevisibility_coverage_out,
                               long ierr[XV_NUM_ERR_ZONEVISTIME_COVERAGE]); // AN-505

/*******************************************************************************
*
* Error Handling Functions - prototype declaration
*
*******************************************************************************/

/* Maximum number of error codes that can be defined for a single function */
#define XV_MAX_COD XL_MAX_COD
/* Maximum length of a string (function name or error message) */
#define XV_MAX_STR XL_MAX_STR

  long xv_silent(void);

  long xv_verbose(void);

  long xv_print_msg(long* n, char msg[XV_MAX_COD][XV_MAX_STR]);

  long xv_get_msg(long* func_id, long* ierr, long* n, char msg[XV_MAX_COD][XV_MAX_STR]);

  long xv_get_code(long* func_id, long* ierr, long* n, long vec[XV_MAX_COD]);

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

  XV_ERR = XL_ERR, /* Error status */
  XV_OK = XL_OK, /* Nominal status */
  XV_WARN = XL_WARN /* Warning status */

} XV_PErr_type_enum; /* CAREFUL: variables holding these values MUST be declared as long !!! */

/* Satellite ID */
/* ------------ */

typedef enum
{

  XV_SAT_DEFAULT = XL_SAT_DEFAULT,
  XV_SAT_DEFAULT1 = XL_SAT_DEFAULT1,
  XV_SAT_DEFAULT2 = XL_SAT_DEFAULT2,
  XV_SAT_DEFAULT3 = XL_SAT_DEFAULT3,
  XV_SAT_DEFAULT4 = XL_SAT_DEFAULT4,
  XV_SAT_DEFAULT5 = XL_SAT_DEFAULT5,
  XV_SAT_DEFAULT6 = XL_SAT_DEFAULT6,
  XV_SAT_DEFAULT7 = XL_SAT_DEFAULT7,
  XV_SAT_DEFAULT8 = XL_SAT_DEFAULT8,
  XV_SAT_DEFAULT9 = XL_SAT_DEFAULT9,
  XV_SAT_ERS1 = XL_SAT_ERS1,
  XV_SAT_ERS2 = XL_SAT_ERS2,
  XV_SAT_ENVISAT = XL_SAT_ENVISAT,
  XV_SAT_METOP1 = XL_SAT_METOP1,
  XV_SAT_METOP2 = XL_SAT_METOP2,
  XV_SAT_METOP3 = XL_SAT_METOP3,
  XV_SAT_CRYOSAT = XL_SAT_CRYOSAT,
  XV_SAT_ADM = XL_SAT_ADM,
  XV_SAT_GOCE = XL_SAT_GOCE,
  XV_SAT_SMOS = XL_SAT_SMOS,
  XV_SAT_TERRASAR = XL_SAT_TERRASAR,
  XV_SAT_EARTHCARE = XL_SAT_EARTHCARE,
  XV_SAT_SWARM_A = XL_SAT_SWARM_A,
  XV_SAT_SWARM_B = XL_SAT_SWARM_B,
  XV_SAT_SWARM_C = XL_SAT_SWARM_C,
  XV_SAT_SENTINEL_1A = XL_SAT_SENTINEL_1A,
  XV_SAT_SENTINEL_1B = XL_SAT_SENTINEL_1B,
  XV_SAT_SENTINEL_2 = XL_SAT_SENTINEL_2,
  XV_SAT_SENTINEL_3 = XL_SAT_SENTINEL_3,
  XV_SAT_SEOSAT = XL_SAT_SEOSAT,
  XV_SAT_SENTINEL_1C = XL_SAT_SENTINEL_1C,
  XV_SAT_SENTINEL_2A = XL_SAT_SENTINEL_2A,
  XV_SAT_SENTINEL_2B = XL_SAT_SENTINEL_2B,
  XV_SAT_SENTINEL_2C = XL_SAT_SENTINEL_2C,
  XV_SAT_SENTINEL_3A = XL_SAT_SENTINEL_3A,
  XV_SAT_SENTINEL_3B = XL_SAT_SENTINEL_3B,
  XV_SAT_SENTINEL_3C = XL_SAT_SENTINEL_3C,
  XV_SAT_JASON_CSA = XL_SAT_JASON_CSA, // AN-531
  XV_SAT_JASON_CSB = XL_SAT_JASON_CSB, // AN-531
  XV_SAT_METOP_SG_A1 = XL_SAT_METOP_SG_A1, // AN-530
  XV_SAT_METOP_SG_A2 = XL_SAT_METOP_SG_A2, // AN-530
  XV_SAT_METOP_SG_A3 = XL_SAT_METOP_SG_A3, // AN-530
  XV_SAT_METOP_SG_B1 = XL_SAT_METOP_SG_B1, // AN-530
  XV_SAT_METOP_SG_B2 = XL_SAT_METOP_SG_B2, // AN-530
  XV_SAT_METOP_SG_B3 = XL_SAT_METOP_SG_B3, // AN-530
  XV_SAT_SENTINEL_5P = XL_SAT_SENTINEL_5P, // AN-502
  XV_SAT_BIOMASS = XL_SAT_BIOMASS, // AN-601
  XV_SAT_SENTINEL_5 = XL_SAT_SENTINEL_5, // AN-639
  XV_SAT_SAOCOM_CS = XL_SAT_SAOCOM_CS, // AN-640
  XV_SAT_FLEX = XL_SAT_FLEX, /* AN-683 */
  XV_SAT_SENTINEL_6A = XL_SAT_SENTINEL_6A, /* AN-809 */
  XV_SAT_SENTINEL_6B = XL_SAT_SENTINEL_6B, /* AN-809 */
  XV_SAT_CIMR = XL_SAT_CIMR,
  XV_SAT_ROSEL = XL_SAT_ROSEL,
  XV_SAT_CHIME = XL_SAT_CHIME,
  XV_SAT_CRISTAL = XL_SAT_CRISTAL,
  XV_SAT_CO2M = XL_SAT_CO2M,
  XV_SAT_LSTM = XL_SAT_LSTM,
  XV_SAT_FORUM = XL_SAT_FORUM,
  XV_SAT_TRUTHS = XL_SAT_TRUTHS,
  XV_SAT_GENERIC = XL_SAT_GENERIC,
  XV_SAT_GENERIC_GEO = XL_SAT_GENERIC_GEO, /* ANR-353 */
  XV_SAT_MTG = XL_SAT_MTG, /* ANR-353 */
  XV_SAT_GENERIC_MEO = XL_SAT_GENERIC_MEO /* ANR-550 */

} XV_Sat_id_enum; /* CAREFUL: if you modify this enum, you must modify XO_Sat_array_enum enum
                and XL_Sat_id_to_array routine */

/* Types of orbit inputs */
/* --------------------- */
typedef enum
{

  XV_ORBIT_ABS = 0, /* Absolute orbit chosen as input */
  XV_ORBIT_REL = 1 /* Relative orbit chosen as input */

} XV_Orbit_type_enum;

/* xv_zone_vis_time coverage outputs */
/* --------------------------------- */

typedef enum
{ /* Indicate the type of coverage as result of a visibility segment from ZONEVISTIME */

  XV_COMPLETE = 0, /* Zone completely covered by swath */
  XV_LEFT = 1, /* LEFT extreme transition found by ZONEVISTIME */
  XV_RIGHT = 2, /* RIGHT extreme transition found by ZONEVISTIME */
  XV_BOTH = 3 /* BOTH transitions found by ZONEVISTIME*/

} XV_Coverage_enum; /* CAREFUL: variables holding these values MUST be declared as long !!! */

/* xv_stavistime mask inputs */
/* ------------------------- */

typedef enum
{ /* It describes the ground station mask used to define visibility in STAVISTIME*/

  XV_COMBINE = 0, /* Combination of AOS, LOS and physical mask */
  XV_AOS_LOS = 1, /* Combination of AOS/LOS only */
  XV_PHYSICAL = 2, /* Consider physical mask only */
  XV_FROM_FILE = 3 /* Consider mask from GS DB */

} XV_Mask_enum; /* CAREFUL: variables holding these values MUST be declared as long !!! */

/* xv_starvistime coverage outputs */
/* ------------------------------- */

typedef enum
{ /* Indicate the type of coverage as result of a visibility segment from STARVISTIME */

  XV_STAR_UNDEFINED = 0, /* Visibility starts/ends at the first/last FOV in STARVISTIME */
  XV_STAR_UPPER = 1, /* Visibility starts/ends at the upper FOV in STARVISTIME */
  XV_STAR_LOWER = 2, /* Visibility starts/ends at the lower FOV in STARVISTIME */
  XV_STAR_LEFT = 3, /* Visibility starts/ends at the left FOV in STARVISTIME */
  XV_STAR_RIGHT = 4 /* Visibility starts/ends at the right FOV in STARVISTIME */

} XV_Coverage_star_enum;

/* Swath types */
/* ----------- */
/* Swath types for the swath_flag in all visibility 
   functions using a swath file in the interface */
typedef enum
{
  XV_STF, /* Swath Template file */
  XV_SDF /* Swath Definition file */

} XV_Swath_type;

/* CFI functions IDs (for error handling) */
/* -------------------------------------- */

typedef enum
{

  XV_NO_FUNC_ID = -1, /* No function code */

  XV_ZONE_VIS_TIME_ID,
  XV_STATION_VIS_TIME_ID,
  XV_SC_VIS_TIME_ID,
  XV_SWATH_POS_ID,
  XV_STAR_VIS_TIME_ID,
  XV_MULTIZONES_VIS_TIME_ID,
  XV_MULTISTATIONS_VIS_TIME_ID,
  XV_TIME_SEGMENTS_NOT_ID,
  XV_TIME_SEGMENTS_OR_ID,
  XV_TIME_SEGMENTS_AND_ID,
  XV_TIME_SEGMENTS_SORT_ID,
  XV_TIME_SEGMENTS_MERGE_ID,
  XV_TIME_SEGMENTS_DELTA_ID,
  XV_TIME_SEGMENTS_MAPPING_ID,
  XV_ORBIT_EXTRA_ID,
  XV_GEN_SCF_ID,
  XV_GEN_SWATH_ID,
  XV_SWATH_ID_INIT_ID, /* AN-468 */
  XV_SWATH_ID_CLOSE_ID, /* AN-468 */
  XV_ZONEVISTIME_COMPUTE_ID, /* AN-468 */
  XV_STATIONVISTIME_COMPUTE_ID, /* AN-468 */
  XV_SWATHPOS_COMPUTE_ID, /* AN-468 */
  XV_TIMESEGMENTS_COMPUTE_NOT_ID, /* AN-468 */
  XV_TIMESEGMENTS_COMPUTE_OR_ID, /* AN-468 */
  XV_TIMESEGMENTS_COMPUTE_AND_ID, /* AN-468 */
  XV_TIMESEGMENTS_COMPUTE_SORT_ID, /* AN-468 */
  XV_TIMESEGMENTS_COMPUTE_MERGE_ID, /* AN-468 */
  XV_TIMESEGMENTS_COMPUTE_DELTA_ID, /* AN-468 */
  XV_TIMESEGMENTS_COMPUTE_MAPPING_ID, /* AN-468 */
  XV_ZONEVISTIME_COVERAGE_ID, // AN-505
  XV_SET_SC_VIS_TIME_STEP_ID, // AN-628
  XV_CELESTIAL_BODY_VIS_TIME_ID, // AN-677
  XV_COMPUTE_AOI_ID, // AN-672
  XV_MAX_FUNC_ID /* Number of functions defined in this list */

} XV_Func_enum;

/* Indicate that a segment list is ordered by increasing start time*/
typedef enum
{
  XV_TIME_ORDER = 0,
  XV_NO_TIME_ORDER = 1

} XV_time_segments_order_enum;

typedef enum
{
  XV_ASCENDING = 0,
  XV_DESCENDING = 1

} XV_time_segments_orbit_direction_enum;

/* AN-468 */
typedef enum
{
  XV_FILE_AUTO,
  XV_FILE_SDF,
  XV_FILE_STF,
  XV_SDF_DATA,
  XV_STF_DATA
} XV_Swath_init_enum;

typedef enum
{
  XV_USE_ZONE_DB_FILE,
  XV_USE_ZONE_DATA
} XV_Zone_type_enum;

typedef enum
{
  XV_DO_NOT_COMPUTE = XL_FALSE,
  XV_COMPUTE = XL_TRUE
} XV_Compute_flag_enum;

typedef enum
{
  XV_USE_STATION_FILE,
  XV_USE_STATION_DATA,
  XV_USE_STATION_FILE_AND_MASK_OVERRIDE,
  XV_USE_STATION_DATA_AND_MASK_OVERRIDE
} XV_Station_type_enum;

typedef enum
{
  XV_UTC_TYPE, /* Only UTC time info is provided*/
  XV_ORBIT_TYPE, /* Only orbit info is provided */
  XV_BOTH_TYPE /* UTC time and orbit info is provided */
} XV_Time_type_enum;
/* END AN-468 */

typedef enum
{
  XV_COVERAGE_FIXED_DISTANCE,
  XV_COVERAGE_PERCENTAGE_PRECISION
} XV_Type_Coverage_enum; // AN-505

typedef enum
{
  XV_SUN,
  XV_MOON
} XV_Type_Celestial_enum; // AN-677

/*********************************************************************************
*
* List of errors of the main CFI functions (Look into the EXPLORER_VISIBILITY SUM)
*
*********************************************************************************/

/* Error codes list of xv_zone_vis_time */
/* ------------------------------------ */
typedef enum
{

  XV_CFI_ZONE_VIS_TIME_NEGATIVE_NUM_ZONE_ERR = XL_ZERO_E,
  XV_CFI_ZONE_VIS_TIME_ORBIT_STATUS_ERR,
  XV_CFI_ZONE_VIS_TIME_ORBIT_TYPE_ERR,
  XV_CFI_ZONE_VIS_TIME_NEGATIVE_MIN_DURATION_ERR,
  XV_CFI_ZONE_VIS_TIME_PROJECTION_OUT_OF_RANGE_ERR,
  XV_CFI_ZONE_VIS_TIME_SWATH_FLAG_ERR,
  XV_CFI_ZONE_VIS_TIME_WRONG_SWATH_ERR,
  XV_CFI_ZONE_VIS_TIME_GENSWATH_ERR,
  XV_CFI_ZONE_VIS_TIME_IN_ORBIT_ERR,
  XV_CFI_ZONE_VIS_TIME_READ_SWATH_FILE_ERR,
  XV_CFI_ZONE_VIS_TIME_INCORRECT_SWATH_TYPE_ERR,
  XV_CFI_ZONE_VIS_TIME_ALLOCATE_SWATH_MEMORY_ERR,
  XV_CFI_ZONE_VIS_TIME_NEGATIVE_START_ORBIT_ERR,
  XV_CFI_ZONE_VIS_TIME_READ_OSF_ERR,
  XV_CFI_ZONE_VIS_TIME_EARLIER_START_ORBIT_WARN,
  XV_CFI_ZONE_VIS_TIME_LATER_STOP_ORBIT_WARN,
  XV_CFI_ZONE_VIS_TIME_WRONG_ORBIT_RANGE_ERR,
  XV_CFI_ZONE_VIS_TIME_ORBITINFO_CALL_ERR,
  XV_CFI_ZONE_VIS_TIME_INCONSISTENT_SWATH_ERR,
  XV_CFI_ZONE_VIS_TIME_ORBITAL_CHANGE_WARN,
  XV_CFI_ZONE_VIS_TIME_ZONE_ID_EMPTY_ERR,
  XV_CFI_ZONE_VIS_TIME_READ_ZONE_DB_FILE_ERR,
  XV_CFI_ZONE_VIS_TIME_DEFAULT_PROJECTION_WARN,
  XV_CFI_ZONE_VIS_TIME_ALLOCATE_ZONE_MEMORY_ERR,
  XV_CFI_ZONE_VIS_TIME_WRONG_LATITUDE_RANGE_ERR,
  XV_CFI_ZONE_VIS_TIME_TWO_EQUAL_POINTS_WARN,
  XV_CFI_ZONE_VIS_TIME_DIFF_LONG_180_ERR,
  XV_CFI_ZONE_VIS_TIME_ANTIPODAL_POINTS_ERR,

  XV_CFI_ZONE_VIS_TIME_SEGMENT_INTERSECT_PREC_ERR = XL_FIRST_E,
  XV_CFI_ZONE_VIS_TIME_SEGMENT_INTERSECT_COMP_ERR,
  XV_CFI_ZONE_VIS_TIME_GNOMONIC_COORD_ERR,
  XV_CFI_ZONE_VIS_TIME_TWO_SEGMENTS_INTERSECT_ERR,
  XV_CFI_ZONE_VIS_TIME_ALLIGNED_SEGMENTS_ERR,
  XV_CFI_ZONE_VIS_TIME_ZONE_DIAM_NEGATIVE_ERR,
  XV_CFI_ZONE_VIS_TIME_POLE_IN_SWATH_ERR,
  XV_CFI_ZONE_VIS_TIME_CUADRILATERAL_NOT_CONVEX_ERR,
  XV_CFI_ZONE_VIS_TIME_POINT_IN_CUADRILATERAL_ERR,
  XV_CFI_ZONE_VIS_TIME_SORT_INTERSECTIONS_ERR,
  XV_CFI_ZONE_VIS_TIME_SEGMENTS_MEMORY_ERR,
  XV_CFI_ZONE_VIS_TIME_MAX_ORBITS_ERR,
  XV_CFI_ZONE_VIS_TIME_COVERAGE_MEMORY_ERR,
  XV_CFI_ZONE_VIS_TIME_CHECK_SEGMENTS_WARN,
  XV_CFI_ZONE_VIS_TIME_CHECK_SEGMENTS_ERR,
  XV_CFI_ZONE_VIS_TIME_ORBIT_TO_TIME_CALL_ERR,
  XV_CFI_ZONE_VIS_TIME_ORBIT_MODEL_ERR,
  XV_CFI_ZONE_VIS_TIME_STOP_ORBIT_WARN,
  XV_CFI_ZONE_VIS_TIME_COMPUTE_ANX_ERR,
  XV_CFI_ZONE_VIS_TIME_ORBIT_INFO_ERR,
  XV_CFI_ZONE_VIS_TIME_MULTI_POINT_SWATH_INTERS_ERR,
  XV_CFI_ZONE_VIS_TIME_POINT_SWATH_INTERS_ERR,
  XV_CFI_ZONE_VIS_TIME_ON_OFF_CHECKING_ERR,
  XV_CFI_ZONE_VIS_TIME_MERGE_SWATH_SEGMENTS_VISIBILITIES_ERR,
  XV_CFI_ZONE_VIS_TIME_MEMORY_ALLOCATION_ERR,
  XV_CFI_ZONE_VIS_TIME_SWATH_POS_ERR,
  XV_CFI_ZONE_VIS_TIME_POLYGON_INNER_POINT_ERR,
  XV_CFI_ZONE_VIS_TIME_CHECK_ORBITAL_CHANGE_ERR,

  XV_CFI_ZONE_VIS_TIME_CONVERT_ZONE_ERR = XL_SECOND_E,
  XV_CFI_ZONE_VIS_TIME_ZONE_ORBIT_ERR,
  XV_CFI_ZONE_VIS_TIME_CLONE_ZONE_ERR,
  XV_CFI_ZONE_VIS_TIME_ORBIT_INTERVAL_STF_ERR,
  XV_CFI_ZONE_VIS_TIME_ORBIT_INTERVAL_STF_WARN,
  XV_CFI_ZONE_VIS_TIME_OSF_NON_LIN_STF_OLD_WARN,
  XV_CFI_ZONE_VIS_TIME_SWATH_FLAG_LARGER_THAN_LIN_APPROX_VAL_WARN,
  XV_CFI_ZONE_VIS_TIME_GEO_SAT_ERR,
  XV_CFI_ZONE_VIS_TIME_DEPRECATED_WARN,
  XV_CFI_ZONE_VIS_TIME_PRECISE_PROPAG_WARN,
  XV_CFI_ZONE_VIS_TIME_CHECK_ORBIT_VS_STF_WARN
} XV_CFI_Zone_vis_time_err_enum;

/* Error codes list of xv_zonevistime_compute (AN-468) */
typedef enum
{
  XV_CFI_ZONEVISTIME_COMPUTE_ORBIT_STATUS_ERR = XL_ZERO_E,
  XV_CFI_ZONEVISTIME_COMPUTE_GEO_SAT_ERR,
  XV_CFI_ZONEVISTIME_COMPUTE_SWATH_STATUS_ERR,
  XV_CFI_ZONEVISTIME_COMPUTE_ORBIT_TYPE_ERR,
  XV_CFI_ZONEVISTIME_COMPUTE_ORBIT_MODEL_ERR,
  XV_CFI_ZONEVISTIME_COMPUTE_NEGATIVE_NUM_ZONE_ERR,
  XV_CFI_ZONEVISTIME_COMPUTE_NEGATIVE_MIN_DURATION_ERR,
  XV_CFI_ZONEVISTIME_COMPUTE_PROJECTION_OUT_OF_RANGE_ERR,
  XV_CFI_ZONEVISTIME_COMPUTE_NEGATIVE_START_ORBIT_ERR,
  XV_CFI_ZONEVISTIME_COMPUTE_EARLIER_START_ORBIT_WARN,
  XV_CFI_ZONEVISTIME_COMPUTE_STOP_ORBIT_WARN,
  XV_CFI_ZONEVISTIME_COMPUTE_WRONG_ORBIT_RANGE_ERR,
  XV_CFI_ZONEVISTIME_COMPUTE_READ_SWATH_FILE_ERR,
  XV_CFI_ZONEVISTIME_COMPUTE_WRONG_SWATH_ERR,
  XV_CFI_ZONEVISTIME_COMPUTE_ORBIT_INTERVAL_STF_ERR,
  XV_CFI_ZONEVISTIME_COMPUTE_ORBIT_INTERVAL_STF_WARN,
  XV_CFI_ZONEVISTIME_COMPUTE_OSF_NON_LIN_STF_OLD_WARN,
  XV_CFI_ZONEVISTIME_COMPUTE_ZONE_ID_EMPTY_ERR,
  XV_CFI_ZONEVISTIME_COMPUTE_READ_ZONE_DB_FILE_ERR,
  XV_CFI_ZONEVISTIME_COMPUTE_CLONE_ZONE_ERR,
  XV_CFI_ZONEVISTIME_COMPUTE_SWATH_FLAG_LARGER_THAN_LIN_APPROX_VAL_WARN,
  XV_CFI_ZONEVISTIME_COMPUTE_ORBIT_INFO_ERR,
  XV_CFI_ZONEVISTIME_COMPUTE_SEGMENTS_MEMORY_ERR,
  XV_CFI_ZONEVISTIME_COMPUTE_OVERLAP_ERR,
  XV_CFI_ZONEVISTIME_COMPUTE_ORBITINFO_CALL_ERR,
  XV_CFI_ZONEVISTIME_COMPUTE_ZONEVISTIME_LOOP_ERR,
  XV_CFI_ZONEVISTIME_COMPUTE_TWO_SEGMENTS_INTERSECT_ERR,
  XV_CFI_ZONEVISTIME_COMPUTE_ORBITAL_CHANGE_WARN,

  XV_CFI_ZONEVISTIME_COMPUTE_INCONSISTENT_SWATH_ERR = XL_FIRST_E,
  XV_CFI_ZONEVISTIME_COMPUTE_TIME_TYPE_ERR,
  XV_CFI_ZONEVISTIME_COMPUTE_CHECK_SEGMENTS_ERR,
  XV_CFI_ZONEVISTIME_COMPUTE_TIME_TO_ORBIT_ERR,
  XV_CFI_ZONEVISTIME_COMPUTE_GET_UTC_TIME_ERR,
  XV_CFI_ZONEVISTIME_COMPUTE_DETECT_SWATH_TYPE_ERR,
  XV_CFI_ZONEVISTIME_COMPUTE_PRECISE_PROPAG_WARN,
  XV_CFI_ZONEVISTIME_COMPUTE_ATTITUDE_TYPE_ERR,
  XV_CFI_ZONEVISTIME_COMPUTE_LINK_ID_ERR, // AN-573
  XV_CFI_ZONEVISTIME_COMPUTE_ORBIT_VALIDITY_ERR, // AN-663
  XV_CFI_ZONEVISTIME_COMPUTE_ORBIT_VAL_START_WARN, // AN-663
  XV_CFI_ZONEVISTIME_COMPUTE_ORBIT_VAL_STOP_WARN, // AN-663
  XV_CFI_ZONEVISTIME_COMPUTE_ADJUST_TO_ORBIT_VALIDITY_ERR, // AN-663
  XV_CFI_ZONEVISTIME_COMPUTE_CHECK_ORBIT_VS_STF_WARN // AN-961
} XV_CFI_Zonevistime_compute_err_enum;

/* Error codes list of xv_stationvistime_compute (AN-468) */
typedef enum
{
  XV_CFI_STATIONVISTIME_COMPUTE_INPUTS_CHECK_ERR = XL_ZERO_E,
  XV_CFI_STATIONVISTIME_COMPUTE_FIRST_ORBIT_WARN,
  XV_CFI_STATIONVISTIME_COMPUTE_SWATH_ERR, // AN-713
  XV_CFI_STATIONVISTIME_COMPUTE_SWATH_WARN, // AN-713
  XV_CFI_STATIONVISTIME_COMPUTE_ORBIT_INFO_ERR,
  XV_CFI_STATIONVISTIME_COMPUTE_INCONSISTENT_SWATH_ERR,
  XV_CFI_STATIONVISTIME_COMPUTE_STA_COMPUTE_ERR,
  XV_CFI_STATIONVISTIME_COMPUTE_SWATH_FLAG_LARGER_THAN_LIN_APPROX_VAL_WARN,
  XV_CFI_STATIONVISTIME_COMPUTE_CHECK_SEGMENTS_ERR,
  XV_CFI_STATIONVISTIME_COMPUTE_PRECISE_PROPAG_WARN,
  XV_CFI_STATIONVISTIME_COMPUTE_ORBIT_RANGE_ERR, // AN-713
  XV_CFI_STATIONVISTIME_COMPUTE_PREPARE_OUTPUT_ERR, // AN-713
  XV_CFI_STATIONVISTIME_COMPUTE_MEM_ERR, // AN-713
  XV_CFI_STATIONVISTIME_COMPUTE_OSF_OPT_ERR, // AN-713
  XV_CFI_STATIONVISTIME_COMPUTE_SEGMENT_MERGE_ERR, // AN-713
  XV_CFI_STATIONVISTIME_COMPUTE_SEGMENT_LIST_COPY_ERR // AN-713

} XV_CFI_Stationvistime_compute_err_enum;

/* Error codes list of xv_station_vis_time */
/* --------------------------------------- */
typedef enum
{
  /* Errors and warnings related to calls to other functions */
  XV_CFI_STATION_VIS_TIME_ORBIT_STATUS_ERR = XL_ZERO_E,
  XV_CFI_STATION_VIS_TIME_INPUTS_CHECK_ERR,
  XV_CFI_STATION_VIS_TIME_ORBIT_TYPE_ERR,
  XV_CFI_STATION_VIS_TIME_ORBIT_MODEL_ERR,
  XV_CFI_STATION_VIS_TIME_REL_TO_ABS_START_ERR,
  XV_CFI_STATION_VIS_TIME_REL_TO_ABS_STOP_ERR,
  XV_CFI_STATION_VIS_TIME_OSF_READ_ERR,
  XV_CFI_STATION_VIS_TIME_SWATH_FLAG_ERR,
  XV_CFI_STATION_VIS_TIME_SWATH_READ_ERR,
  XV_CFI_STATION_VIS_TIME_SWATH_TYPE_ERR,
  XV_CFI_STATION_VIS_TIME_WRONG_SWATH_ERR,
  XV_CFI_STATION_VIS_TIME_FIRST_ORBIT_WARN,
  XV_CFI_STATION_VIS_TIME_LAST_ORBIT_WARN,
  XV_CFI_STATION_VIS_TIME_WRONG_INTERVAL_ERR,
  XV_CFI_STATION_VIS_TIME_ORBIT_INFO_ERR,
  XV_CFI_STATION_VIS_TIME_ORBIT_CHANGE_WARN,
  XV_CFI_STATION_VIS_TIME_INTERNAL_MEMORY_ERR,
  XV_CFI_STATION_VIS_TIME_POTENTIAL_MEMORY_ERR,
  XV_CFI_STATION_VIS_TIME_INCONSISTENT_SWATH_ERR,
  XV_CFI_STATION_VIS_TIME_READ_STA_ERR,
  XV_CFI_STATION_VIS_TIME_AZEL2LONLAT_ERR,
  XV_CFI_STATION_VIS_TIME_ZONE_VIS_TIME_CALL_ERR,
  XV_CFI_STATION_VIS_TIME_CALL_STAVIS_ERR,
  XV_CFI_STATION_VIS_TIME_CALL_STAVIS_WARN,
  XV_CFI_STATION_VIS_TIME_SEGMENTS_MEMORY_ERR,
  XV_CFI_STATION_VIS_TIME_ZERO_DOPPLER_ERR,
  XV_CFI_STATION_VIS_TIME_LONG_SEGM_SKIPPED_WARN,
  XV_CFI_STATION_VIS_TIME_ABS_TO_REL_ERR,

  XV_CFI_STATION_VIS_TIME_MASK_FROM_FILE_MASK_TYPE_ERR = XL_FIRST_E,
  XV_CFI_STATION_VIS_TIME_MASK_FROM_FILE_NO_SC_ERR,
  XV_CFI_STATION_VIS_TIME_CONVERT_ZONE_ERR,
  XV_CFI_STATION_VIS_TIME_ORBIT_INTERVAL_STF_ERR,
  XV_CFI_STATION_VIS_TIME_ORBIT_INTERVAL_STF_WARN,
  XV_CFI_STATION_VIS_TIME_OSF_NON_LIN_STF_OLD_WARN,
  XV_CFI_STATION_VIS_TIME_SWATH_FLAG_LARGER_THAN_LIN_APPROX_VAL_WARN,
  XV_CFI_STATION_VIS_TIME_GEO_SAT_ERR,
  XV_CFI_STATION_VIS_TIME_DEPRECATED_WARN,
  XV_CFI_STATION_VIS_TIME_PRECISE_PROPAG_WARN

} XV_CFI_Station_vis_time_err_enum;

/* Error codes list of xv_sc_vistime */
/* --------------------------------- */
typedef enum
{
  XV_CFI_SC_VIS_TIME_INTERNAL_MEMORY_ERR = XL_ZERO_E,
  XV_CFI_SC_VIS_TIME_ORBIT_STATUS_ERR,
  XV_CFI_SC_VIS_TIME_XV_SC_INPUTS_CHECK_ERR,
  XV_CFI_SC_VIS_TIME_ORBIT_TYPE_ERR,
  XV_CFI_SC_VIS_TIME_START_ORBIT_WARN,
  XV_CFI_SC_VIS_TIME_REL_TO_ABS_START_ERR,
  XV_CFI_SC_VIS_TIME_REL_TO_ABS_STOP_ERR,
  XV_CFI_SC_VIS_TIME_WRONG_ORBIT_RANGE_ERR,
  XV_CFI_SC_VIS_TIME_XV_ORBIT_INFO_ERR,
  XV_CFI_SC_VIS_TIME_TIME_CHANGE_ERR,
  XV_CFI_SC_VIS_TIME_VIS_CHECK_ERR,
  XV_CFI_SC_VIS_TIME_FIRST_ORBIT_VIS_WARN,
  XV_CFI_SC_VIS_TIME_MAX_NUMBER_ITER_ERR,
  XV_CFI_SC_VIS_TIME_XV_TIME_SEC_ERR,
  XV_CFI_SC_VIS_TIME_ABS_TO_REL_BGN_ERR,
  XV_CFI_SC_VIS_TIME_ABS_TO_REL_END_ERR,
  XV_CFI_SC_VIS_TIME_LAST_ORBIT_VIS_WARN,
  XV_CFI_SC_VIS_TIME_SAT_NOM_ATT_STATUS_ERR,
  XV_CFI_SC_VIS_TIME_SAT_ATT_STATUS_ERR,
  XV_CFI_SC_VIS_TIME_INSTR_ATT_STATUS_ERR,
  XV_CFI_SC_VIS_TIME_PRECISE_PROPAG_WARN

} XV_CFI_Sc_vis_time_err_enum;

/* Error codes list of xv_set_sc_vis_time_step (AN-628) */
typedef enum
{
  XV_CFI_SET_SC_VIS_TIME_STEP_ORBIT_INIT_STATUS_ERR = XL_ZERO_E

} XV_CFI_Set_sc_vis_time_step_err_enum;

/* Error codes list of xv_sc_vistime */
/* --------------------------------- */
typedef enum
{
  XV_CFI_CELESTIAL_BODY_VIS_TIME_INTERNAL_MEMORY_ERR = XL_ZERO_E,
  XV_CFI_CELESTIAL_BODY_VIS_TIME_ORBIT_STATUS_ERR,
  XV_CFI_CELESTIAL_BODY_VIS_TIME_XV_SC_INPUTS_CHECK_ERR,
  XV_CFI_CELESTIAL_BODY_VIS_TIME_ORBIT_TYPE_ERR,
  XV_CFI_CELESTIAL_BODY_VIS_TIME_START_ORBIT_WARN,
  XV_CFI_CELESTIAL_BODY_VIS_TIME_REL_TO_ABS_START_ERR,
  XV_CFI_CELESTIAL_BODY_VIS_TIME_REL_TO_ABS_STOP_ERR,
  XV_CFI_CELESTIAL_BODY_VIS_TIME_WRONG_ORBIT_RANGE_ERR,
  XV_CFI_CELESTIAL_BODY_VIS_TIME_XV_ORBIT_INFO_ERR,
  XV_CFI_CELESTIAL_BODY_VIS_TIME_TIME_CHANGE_ERR,
  XV_CFI_CELESTIAL_BODY_VIS_TIME_VIS_CHECK_ERR,
  XV_CFI_CELESTIAL_BODY_VIS_TIME_FIRST_ORBIT_VIS_WARN,
  XV_CFI_CELESTIAL_BODY_VIS_TIME_MAX_NUMBER_ITER_ERR,
  XV_CFI_CELESTIAL_BODY_VIS_TIME_XV_TIME_SEC_ERR,
  XV_CFI_CELESTIAL_BODY_VIS_TIME_ABS_TO_REL_BGN_ERR,
  XV_CFI_CELESTIAL_BODY_VIS_TIME_ABS_TO_REL_END_ERR,
  XV_CFI_CELESTIAL_BODY_VIS_TIME_LAST_ORBIT_VIS_WARN,
  XV_CFI_CELESTIAL_BODY_VIS_TIME_SAT_NOM_ATT_STATUS_ERR,
  XV_CFI_CELESTIAL_BODY_VIS_TIME_SAT_ATT_STATUS_ERR,
  XV_CFI_CELESTIAL_BODY_VIS_TIME_INSTR_ATT_STATUS_ERR,
  XV_CFI_CELESTIAL_BODY_VIS_TIME_PRECISE_PROPAG_WARN

} XV_CFI_Celestial_body_vis_time_err_enum;

/* Error codes list of xv_swath_id_init (AN-468) */
typedef enum
{
  XV_CFI_SWATH_ID_INIT_TYPE_ERR = XL_ZERO_E,
  XV_CFI_SWATH_ID_INIT_WRONG_INPUT_COMBINATION_ERR,
  XV_CFI_SWATH_ID_INIT_SWATH_INIT_STATUS_ERR,
  XV_CFI_SWATH_ID_INIT_MEMORY_ERR,
  XV_CFI_SWATH_ID_INIT_SDF_CLONE_ERR,
  XV_CFI_SWATH_ID_INIT_STF_CLONE_ERR,
  XV_CFI_SWATH_ID_INIT_LINK_ID_ERR

} XV_CFI_Swath_id_init_err_enum;

/* Error codes list of xv_swath_id_close (AN-468) */
typedef enum
{
  XV_CFI_SWATH_ID_CLOSE_WRONG_ID_ERR = XL_ZERO_E

} XV_CFI_Swath_id_close_err_enum;

/* Error codes list of xv_swath_pos */
/* -------------------------------- */
typedef enum
{
  /* Errors and warnings related to internal processing */
  XV_CFI_SWATH_POS_ORBIT_STATUS_ERR = XL_ZERO_E,
  XV_CFI_SWATH_POS_ORBIT_MODEL_ERR,
  XV_CFI_SWATH_POS_INCONSISTENT_SWATH_ERR,
  XV_CFI_SWATH_POS_ORB_NUM_LIM_ERR,
  XV_CFI_SWATH_POS_SEC_LIM_ERR,
  XV_CFI_SWATH_POS_MICROSEC_1ST_ERR,
  XV_CFI_SWATH_POS_MICROSEC_2ND_ERR,
  XV_CFI_SWATH_POS_ORBIT_TYPE_ERR,
  XV_CFI_SWATH_POS_CYCLE_ERR,
  XV_CFI_SWATH_POS_ORB_NUM_OEF_ERR,
  XV_CFI_SWATH_POS_TIME_ERR,
  XV_CFI_SWATH_POS_REP_CYCLE_ERR,
  XV_CFI_SWATH_POS_CYCLE_LENGTH_ERR,
  XV_CFI_SWATH_POS_MLST_DRIFT_ERR,
  XV_CFI_SWATH_POS_SPHER_TRIANG_ERR,
  XV_CFI_SWATH_POS_REL_TO_ABS_ERR = XL_FIRST_E,
  XV_CFI_SWATH_POS_XV_OSF_RECORDS_READ_ERR,
  XV_CFI_SWATH_POS_XV_ORBIT_INFO_ERR,
  XV_CFI_SWATH_POS_SWATH_INIT_ERR,
  XV_CFI_SWATH_POS_MEMORY_ERR,
  XV_CFI_SWATH_POS_GEO_SAT_ERR,
  XV_CFI_SWATH_POS_DEPRECATED_WARN
} XV_CFI_Swath_pos_err_enum;

/* Error codes list of xv_swathpos_compute (AN-468) */
typedef enum
{
  XV_CFI_SWATHPOS_COMPUTE_ORBIT_STATUS_ERR = XL_ZERO_E,
  XV_CFI_SWATHPOS_COMPUTE_ORBIT_MODEL_ERR,
  XV_CFI_SWATHPOS_COMPUTE_INCONSISTENT_SWATH_ERR,
  XV_CFI_SWATHPOS_COMPUTE_ORB_NUM_LIM_ERR,
  XV_CFI_SWATHPOS_COMPUTE_SEC_LIM_ERR,
  XV_CFI_SWATHPOS_COMPUTE_MICROSEC_1ST_ERR,
  XV_CFI_SWATHPOS_COMPUTE_MICROSEC_2ND_ERR,
  XV_CFI_SWATHPOS_COMPUTE_ORBIT_TYPE_ERR,
  XV_CFI_SWATHPOS_COMPUTE_CYCLE_ERR,
  XV_CFI_SWATHPOS_COMPUTE_ORB_NUM_OEF_ERR,
  XV_CFI_SWATHPOS_COMPUTE_TIME_ERR,
  XV_CFI_SWATHPOS_COMPUTE_REP_CYCLE_ERR,
  XV_CFI_SWATHPOS_COMPUTE_CYCLE_LENGTH_ERR,
  XV_CFI_SWATHPOS_COMPUTE_MLST_DRIFT_ERR,
  XV_CFI_SWATHPOS_COMPUTE_SPHER_TRIANG_ERR,
  XV_CFI_SWATHPOS_COMPUTE_REL_TO_ABS_ERR,
  XV_CFI_SWATHPOS_COMPUTE_XV_ORBIT_INFO_ERR,
  XV_CFI_SWATHPOS_COMPUTE_SWATH_INIT_ERR,
  XV_CFI_SWATHPOS_COMPUTE_MEMORY_ERR,
  XV_CFI_SWATHPOS_COMPUTE_GEO_SAT_ERR,
  XV_CFI_SWATHPOS_COMPUTE_TIME_TYPE_ERR,
  XV_CFI_SWATHPOS_COMPUTE_GEN_SWATH_ERR,
  XV_CFI_SWATHPOS_COMPUTE_READ_SWATH_FILE_ERR,
  XV_CFI_SWATHPOS_COMPUTE_SWATH_VALIDITY_WARN,
  XV_CFI_SWATHPOS_COMPUTE_TIME_TO_ORBIT_ERR,
  XV_CFI_SWATHPOS_COMPUTE_DETECT_SWATH_TYPE_ERR
} XV_CFI_Swathpos_compute_err_enum;

/* Error codes list of xv_starvistime */
/* ---------------------------------- */
typedef enum
{

  /* Errors and warnings related to internal processing */
  XV_CFI_STAR_VIS_TIME_ORBIT_STATUS_ERR = XL_ZERO_E,
  XV_CFI_STAR_VIS_TIME_FIRST_ORBIT_WARN,
  XV_CFI_STAR_VIS_TIME_LAST_ORBIT_WARN,
  XV_CFI_STAR_VIS_TIME_ORBIT_CHANGE_WARN,
  XV_CFI_STAR_VIS_TIME_INERTIAL_SWATH_ERR,
  XV_CFI_STAR_VIS_TIME_INCONSISTENT_SWATH_ERR,
  XV_CFI_STAR_VIS_TIME_SWATH_FLAG_ERR,
  XV_CFI_STAR_VIS_TIME_GENSWATH_ERR,
  XV_CFI_STAR_VIS_TIME_ALT_ERR,
  XV_CFI_STAR_VIS_TIME_INTERNAL_MEMORY_ERR,
  XV_CFI_STAR_VIS_TIME_SEGMENTS_MEMORY_ERR,
  XV_CFI_STAR_VIS_TIME_COVERAGE_MEMORY_ERR,
  XV_CFI_STAR_VIS_TIME_ORBIT_TYPE_ERR,
  XV_CFI_STAR_VIS_TIME_ORBIT_MODEL_ERR,
  /* Errors and warnings related to calls to other functions */
  XV_CFI_STAR_VIS_TIME_INPUTS_CHECK_ERR,
  XV_CFI_STAR_VIS_TIME_REL_TO_ABS_START_ERR,
  XV_CFI_STAR_VIS_TIME_REL_TO_ABS_STOP_ERR,
  XV_CFI_STAR_VIS_TIME_STAR_RADEC_ERR,
  XV_CFI_STAR_VIS_TIME_ORBIT_INFO_ERR,
  XV_CFI_STAR_VIS_TIME_SWATH_UPPER_READ_ERR,
  XV_CFI_STAR_VIS_TIME_SWATH_LOWER_READ_ERR,
  XV_CFI_STAR_VIS_TIME_READ_STAR_ERR,
  XV_CFI_STAR_VIS_TIME_STAR_MAIN_ERR,
  XV_CFI_STAR_VIS_TIME_ABS_TO_REL_ERR,
  XV_CFI_STAR_VIS_TIME_ORBIT_TO_TIME_ERR,
  XV_CFI_STAR_VIS_TIME_READ_SDF_ERR,
  XV_CFI_STAR_VIS_TIME_CHECK_ORBITAL_CHANGE_ERR,
  XV_CFI_STAR_VIS_TIME_ORBIT_INTERVAL_STF_ERR,

  XV_CFI_STAR_VIS_TIME_ORBIT_INTERVAL_STF_WARN = XL_FIRST_E,
  XV_CFI_STAR_VIS_TIME_OSF_NON_LIN_STF_OLD_WARN,
  XV_CFI_STAR_VIS_TIME_SWATH_FLAG_LARGER_THAN_LIN_APPROX_VAL_WARN,
  XV_CFI_STAR_VIS_TIME_PRECISE_PROPAG_WARN

} XV_CFI_Star_vis_time_err_enum;

/* Error codes list of xv_multizones_vis_time */
/* ------------------------------------------ */
typedef enum
{
  /* Errors and warnings related to internal processing */
  XV_CFI_MULTIZONES_VIS_TIME_MEMORY_ERR = XL_ZERO_E,
  /* Errors and warnings related to calls to other functions */
  XV_CFI_MULTIZONES_VIS_TIME_COMPUTE_SEGMENTS_ERR,
  XV_CFI_MULTIZONES_VIS_TIME_ABS_TO_REL_ORBIT_ERR,
  XV_CFI_MULTIZONES_VIS_TIME_ABS_TO_REL_VECTOR_ERR,
  XV_CFI_MULTIZONES_VIS_TIME_OVERLAP_ERR,
  XV_CFI_MULTIZONES_VIS_TIME_ORBIT_STATUS_ERR,
  XV_CFI_MULTIZONES_VIS_TIME_GEO_SAT_ERR,
  XV_CFI_MULTIZONES_VIS_TIME_DEPRECATED_WARN
} XV_CFI_Multizones_vis_time_err_enum;

/* Error codes list of xv_multistations_vis_time */
/* ------------------------------------------ */
typedef enum
{
  /* Errors and warnings related to internal processing */
  XV_CFI_MULTISTATIONS_VIS_TIME_MEMORY_ERR = XL_ZERO_E,
  /* Errors and warnings related to calls to other functions */
  XV_CFI_MULTISTATIONS_VIS_TIME_COMPUTE_SEGMENTS_ERR,
  XV_CFI_MULTISTATIONS_VIS_TIME_ABS_TO_REL_ORBIT_ERR,
  XV_CFI_MULTISTATIONS_VIS_TIME_ABS_TO_REL_VECTOR_ERR,
  XV_CFI_MULTISTATIONS_VIS_TIME_OVERLAP_ERR,
  XV_CFI_MULTISTATIONS_VIS_TIME_ORBIT_STATUS_ERR,
  XV_CFI_MULTISTATIONS_VIS_TIME_GEO_SAT_ERR,
  XV_CFI_MULTISTATIONS_VIS_TIME_DEPRECATED_WARN

} XV_CFI_Multistations_vis_time_err_enum;

/* Error codes list of xv_time_segments_not */
/* ---------------------------------------- */
typedef enum
{
  /* Errors and warnings related to internal processing */
  /* Errors and warnings related to calls to other functions */
  XV_CFI_TIME_SEGMENTS_NOT_MEMORY_ERR = XL_ZERO_E,
  XV_CFI_TIME_SEGMENTS_NOT_REL_TO_ABS_ORBIT_ERR,
  XV_CFI_TIME_SEGMENTS_NOT_ABS_TO_REL_ORBIT_ERR,
  XV_CFI_TIME_SEGMENTS_NOT_SORTING_ERR,
  XV_CFI_TIME_SEGMENTS_NOT_GEO_SAT_ERR,
  XV_CFI_TIME_SEGMENTS_NOT_DEPRECATED_WARN

} XV_CFI_time_segments_not_err_enum;

/* Error codes list of xv_time_segments_or */
/* ---------------------------------------- */
typedef enum
{
  /* Errors and warnings related to internal processing */
  /* Errors and warnings related to calls to other functions */
  XV_CFI_TIME_SEGMENTS_OR_MEMORY_ERR = XL_ZERO_E,
  XV_CFI_TIME_SEGMENTS_OR_REL_TO_ABS_ORBIT_ERR,
  XV_CFI_TIME_SEGMENTS_OR_ABS_TO_REL_ORBIT_ERR,
  XV_CFI_TIME_SEGMENTS_OR_SORTING_ERR,
  XV_CFI_TIME_SEGMENTS_OR_GEO_SAT_ERR,
  XV_CFI_TIME_SEGMENTS_OR_DEPRECATED_WARN

} XV_CFI_time_segments_or_err_enum;

/* Error codes list of xv_time_segments_and */
/* ---------------------------------------- */
typedef enum
{
  /* Errors and warnings related to internal processing */
  /* Errors and warnings related to calls to other functions */
  XV_CFI_TIME_SEGMENTS_AND_MEMORY_ERR = XL_ZERO_E,
  XV_CFI_TIME_SEGMENTS_AND_REL_TO_ABS_ORBIT_ERR,
  XV_CFI_TIME_SEGMENTS_AND_ABS_TO_REL_ORBIT_ERR,
  XV_CFI_TIME_SEGMENTS_AND_SORTING_ERR,
  XV_CFI_TIME_SEGMENTS_AND_GEO_SAT_ERR,
  XV_CFI_TIME_SEGMENTS_AND_DEPRECATED_WARN

} XV_CFI_time_segments_and_err_enum;

/* Error codes list of xv_time_segments_sort */
/* ----------------------------------------- */
typedef enum
{
  /* Errors and warnings related to internal processing */
  /* Errors and warnings related to calls to other functions */
  XV_CFI_TIME_SEGMENTS_SORT_MEMORY_ERR = XL_ZERO_E,
  XV_CFI_TIME_SEGMENTS_SORT_CHANGING_ORBIT_ERR,
  XV_CFI_TIME_SEGMENTS_SORT_GEO_SAT_ERR,
  XV_CFI_TIME_SEGMENTS_SORT_DEPRECATED_WARN

} XV_CFI_time_segments_sort_err_enum;

/* Error codes list of xv_time_segments_merge */
/* ----------------------------------------- */
typedef enum
{
  /* Errors and warnings related to internal processing */
  /* Errors and warnings related to calls to other functions */
  XV_CFI_TIME_SEGMENTS_MERGE_MEMORY_ERR = XL_ZERO_E,
  XV_CFI_TIME_SEGMENTS_MERGE_REL_TO_ABS_ORBIT_ERR,
  XV_CFI_TIME_SEGMENTS_MERGE_ABS_TO_REL_ORBIT_ERR,
  XV_CFI_TIME_SEGMENTS_MERGE_SORTING_ERR,
  XV_CFI_TIME_SEGMENTS_MERGE_GEO_SAT_ERR,
  XV_CFI_TIME_SEGMENTS_MERGE_DEPRECATED_WARN

} XV_CFI_time_segments_merge_err_enum;

/* Error codes list of xv_time_segments_delta */
/* ----------------------------------------- */
typedef enum
{
  /* Errors and warnings related to internal processing */
  /* Errors and warnings related to calls to other functions */
  XV_CFI_TIME_SEGMENTS_DELTA_MEMORY_ERR = XL_ZERO_E,
  XV_CFI_TIME_SEGMENTS_DELTA_REL_TO_ABS_ORBIT_ERR,
  XV_CFI_TIME_SEGMENTS_DELTA_ABS_TO_REL_ORBIT_ERR,
  XV_CFI_TIME_SEGMENTS_DELTA_ORBIT_TO_TIME_ERR,
  XV_CFI_TIME_SEGMENTS_DELTA_TIME_TO_ORBIT_ERR,
  XV_CFI_TIME_SEGMENTS_DELTA_TIME_ADD_ERR,
  XV_CFI_TIME_SEGMENTS_DELTA_SORT_ERR,
  XV_CFI_TIME_SEGMENTS_DELTA_GEO_SAT_ERR,
  XV_CFI_TIME_SEGMENTS_DELTA_DEPRECATED_WARN

} XV_CFI_time_segments_delta_err_enum;

/* Error codes list of xv_time_segments_mapping */
/* -------------------------------------------- */
typedef enum
{
  /* Errors and warnings related to internal processing */
  /* Errors and warnings related to calls to other functions */
  XV_CFI_TIME_SEGMENTS_MAPPING_ORBIT_STATUS_ERR = XL_ZERO_E,
  XV_CFI_TIME_SEGMENTS_MAPPING_REL_TO_ABS_ERR,
  XV_CFI_TIME_SEGMENTS_MAPPING_REF_LATITUDE_ERR,
  XV_CFI_TIME_SEGMENTS_MAPPING_SWATH_WIDTH_ERR,
  XV_CFI_TIME_SEGMENTS_MAPPING_ZONEVISTIME_ERR,
  XV_CFI_TIME_SEGMENTS_MAPPING_LOAD_OSF_ERR,
  XV_CFI_TIME_SEGMENTS_MAPPING_WRONG_START_ORB_ERR,
  XV_CFI_TIME_SEGMENTS_MAPPING_WRONG_STOP_ORB_ERR,
  XV_CFI_TIME_SEGMENTS_MAPPING_MEM_ERR,
  XV_CFI_TIME_SEGMENTS_MAPPING_SORT_ERR,
  XV_CFI_TIME_SEGMENTS_MAPPING_ABS_TO_REL_ERR,
  XV_CFI_TIME_SEGMENTS_MAPPING_CHECK_EXTREMES_ERR,
  XV_CFI_TIME_SEGMENTS_MAPPING_SWATH_POS_ERR,
  XV_CFI_TIME_SEGMENTS_MAPPING_SWATH_READ_ERR,
  XV_CFI_TIME_SEGMENTS_MAPPING_INCORRECT_SWATH_ERR,
  XV_CFI_TIME_SEGMENTS_MAPPING_NO_CHECK_PERFORMED_WARN,
  XV_CFI_TIME_SEGMENTS_MAPPING_ASC_INCOMPLETE_MAPPING_WARN,
  XV_CFI_TIME_SEGMENTS_MAPPING_DESC_INCOMPLETE_MAPPING_WARN,
  XV_CFI_TIME_SEGMENTS_MAPPING_GEO_SAT_ERR,
  XV_CFI_TIME_SEGMENTS_MAPPING_DEPRECATED_WARN
} XV_CFI_time_segments_mapping_err_enum;

/* Error codes list of XV_timesegments_compute_not (AN-468) */
typedef enum
{
  XV_CFI_TIMESEGMENTS_COMPUTE_NOT_MEMORY_ERR = XL_ZERO_E,
  XV_CFI_TIMESEGMENTS_COMPUTE_NOT_REL_TO_ABS_ORBIT_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_NOT_ABS_TO_REL_ORBIT_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_NOT_SORTING_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_NOT_GEO_SAT_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_NOT_TIME_TYPE_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_NOT_TIME_TO_ORBIT_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_NOT_GET_UTC_TIME_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_NOT_ORBIT_INIT_ERR
} XV_CFI_Timesegments_compute_not_err_enum;

/* Error codes list of XV_timesegments_compute_or (AN-468) */
typedef enum
{
  XV_CFI_TIMESEGMENTS_COMPUTE_OR_MEMORY_ERR = XL_ZERO_E,
  XV_CFI_TIMESEGMENTS_COMPUTE_OR_REL_TO_ABS_ORBIT_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_OR_ABS_TO_REL_ORBIT_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_OR_SORTING_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_OR_GEO_SAT_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_OR_TIME_TYPE_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_OR_TIME_TO_ORBIT_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_OR_GET_UTC_TIME_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_OR_ORBIT_INIT_ERR
} XV_CFI_Timesegments_compute_or_err_enum;

/* Error codes list of XV_timesegments_compute_and (AN-468) */
typedef enum
{
  XV_CFI_TIMESEGMENTS_COMPUTE_AND_MEMORY_ERR = XL_ZERO_E,
  XV_CFI_TIMESEGMENTS_COMPUTE_AND_REL_TO_ABS_ORBIT_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_AND_ABS_TO_REL_ORBIT_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_AND_SORTING_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_AND_GEO_SAT_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_AND_TIME_TYPE_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_AND_TIME_TO_ORBIT_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_AND_GET_UTC_TIME_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_AND_ORBIT_INIT_ERR
} XV_CFI_Timesegments_compute_and_err_enum;

/* Error codes list of XV_timesegments_compute_sort (AN-468) */
typedef enum
{
  XV_CFI_TIMESEGMENTS_COMPUTE_SORT_MEMORY_ERR = XL_ZERO_E,
  XV_CFI_TIMESEGMENTS_COMPUTE_SORT_CHANGING_ORBIT_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_SORT_GEO_SAT_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_SORT_TIME_TYPE_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_SORT_TIME_TO_ORBIT_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_SORT_GET_UTC_TIME_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_SORT_ORBIT_INIT_ERR
} XV_CFI_Timesegments_compute_sort_err_enum;

/* Error codes list of XV_timesegments_compute_merge (AN-468) */
typedef enum
{
  XV_CFI_TIMESEGMENTS_COMPUTE_MERGE_MEMORY_ERR = XL_ZERO_E,
  XV_CFI_TIMESEGMENTS_COMPUTE_MERGE_REL_TO_ABS_ORBIT_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_MERGE_ABS_TO_REL_ORBIT_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_MERGE_SORTING_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_MERGE_GEO_SAT_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_MERGE_TIME_TYPE_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_MERGE_TIME_TO_ORBIT_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_MERGE_GET_UTC_TIME_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_MERGE_ORBIT_INIT_ERR
} XV_CFI_Timesegments_compute_merge_err_enum;

/* Error codes list of XV_timesegments_compute_delta (AN-468) */
typedef enum
{
  XV_CFI_TIMESEGMENTS_COMPUTE_DELTA_MEMORY_ERR = XL_ZERO_E,
  XV_CFI_TIMESEGMENTS_COMPUTE_DELTA_REL_TO_ABS_ORBIT_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_DELTA_ABS_TO_REL_ORBIT_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_DELTA_ORBIT_TO_TIME_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_DELTA_TIME_TO_ORBIT_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_DELTA_TIME_ADD_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_DELTA_SORT_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_DELTA_GEO_SAT_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_DELTA_TIME_TYPE_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_DELTA_GET_UTC_TIME_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_DELTA_ORBIT_INIT_ERR

} XV_CFI_Timesegments_compute_delta_err_enum;

/* Error codes list of XV_timesegments_compute_mapping (AN-468) */
typedef enum
{
  XV_CFI_TIMESEGMENTS_COMPUTE_MAPPING_ORBIT_STATUS_ERR = XL_ZERO_E,
  XV_CFI_TIMESEGMENTS_COMPUTE_MAPPING_REL_TO_ABS_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_MAPPING_REF_LATITUDE_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_MAPPING_SWATH_WIDTH_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_MAPPING_ZONEVISTIME_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_MAPPING_LOAD_OSF_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_MAPPING_WRONG_START_ORB_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_MAPPING_WRONG_STOP_ORB_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_MAPPING_MEM_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_MAPPING_SORT_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_MAPPING_ABS_TO_REL_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_MAPPING_CHECK_EXTREMES_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_MAPPING_SWATH_POS_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_MAPPING_SWATH_READ_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_MAPPING_INCORRECT_SWATH_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_MAPPING_NO_CHECK_PERFORMED_WARN,
  XV_CFI_TIMESEGMENTS_COMPUTE_MAPPING_ASC_INCOMPLETE_MAPPING_WARN,
  XV_CFI_TIMESEGMENTS_COMPUTE_MAPPING_DESC_INCOMPLETE_MAPPING_WARN,
  XV_CFI_TIMESEGMENTS_COMPUTE_MAPPING_GEO_SAT_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_MAPPING_SWATH_INIT_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_MAPPING_GEN_SWATH_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_MAPPING_TIME_TYPE_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_MAPPING_TIME_TO_ORBIT_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_MAPPING_GET_UTC_TIME_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_MAPPING_DETECT_SWATH_TYPE_ERR,
  XV_CFI_TIMESEGMENTS_COMPUTE_MAPPING_ORBIT_INIT_ERR
} XV_CFI_Timesegments_compute_mapping_err_enum;

/* Error codes list of xv_orbit_extra */
/* ---------------------------------- */
typedef enum
{
  /* Errors and warnings related to internal processing */
  /* Errors and warnings related to calls to other functions */
  XV_CFI_ORBIT_EXTRA_ORBIT_STATUS_ERR = XL_ZERO_E,
  XV_CFI_ORBIT_EXTRA_MEM_ERR,
  XV_CFI_ORBIT_EXTRA_SZA_ERR,
  XV_CFI_ORBIT_EXTRA_ECLIPSE_ERR,
  XV_CFI_ORBIT_EXTRA_SUN_OCC_BY_MOON_ERR,
  XV_CFI_ORBIT_EXTRA_GEO_SAT_ERR
} XV_CFI_orbit_extra_err_enum;

/* Error codes list of xv_gen_scf */
/* ------------------------------ */
typedef enum
{
  XV_CFI_GENSCF_NO_SEGMENTS_ERR = XL_ZERO_E,
  XV_CFI_GENSCF_ORBIT_INIT_ERR,
  XV_CFI_GENSCF_ORBIT_INIT_MODE_ERR,
  XV_CFI_GENSCF_CURRENT_TIME_ERR,
  XV_CFI_GENSCF_ORBIT_TO_TIME_CONVERSION_ERR,
  XV_CFI_GENSCF_TIME_CONVERSION_ERR,
  XV_CFI_GENSCF_CREATE_FILENAME_ERR,
  XV_CFI_GENSCF_WRONG_FILENAME_ERR,
  XV_CFI_GENSCF_WRITE_ERR,
  XV_CFI_GENSCF_MEM_ERR,
  XV_CFI_GENSCF_WRONG_DRAW_ERR,
  XV_CFI_GENSCF_WRONG_FILL_ERR,
  XV_CFI_GENSCF_NO_DIR_WARN

} XV_CFI_gen_scf_err_enum;

/* Error codes list of xv_gen_swath */
/* -------------------------------- */
typedef enum
{
  XV_CFI_GEN_SWATH_ORBIT_INIT_ERR = XL_ZERO_E,
  XV_CFI_GEN_SWATH_REQUESTED_ORBIT_ERR,
  XV_CFI_GEN_SWATH_CURRENT_TIME_ERR,
  XV_CFI_GEN_SWATH_TIME_CONVERSION_ERR,
  XV_CFI_GEN_SWATH_CREATE_FILENAME_ERR,
  XV_CFI_GEN_SWATH_SDF_READ_ERR,
  XV_CFI_GEN_SWATH_XV_ALGOR_ERR,
  XV_CFI_GEN_SWATH_WRITE_ERR,
  XV_CFI_GEN_SWATH_WRONG_FILENAME_ERR,
  XV_CFI_GEN_SWATH_NO_DIR_WARN,
  XV_CFI_GEN_SWATH_MEMORY_ERR,
  XV_CFI_GEN_SWATH_GEO_SAT_ERR
} XV_CFI_gen_swath_err_enum;

/* Error codes list of xv_zonevistime_coverage (AN-505) */
typedef enum
{
  XV_CFI_ZONEVISTIME_COVERAGE_ZONE_ID_EMPTY_ERR = XL_ZERO_E,
  XV_CFI_ZONEVISTIME_COVERAGE_READ_ZONE_DB_FILE_ERR,
  XV_CFI_ZONEVISTIME_COVERAGE_CLONE_ZONE_ERR,
  XV_CFI_ZONEVISTIME_COVERAGE_CREATE_DB_ERR,
  XV_CFI_ZONEVISTIME_COVERAGE_MEMORY_ERR,
  XV_CFI_ZONEVISTIME_COVERAGE_ZVT_COMPUTE_ERR,
  XV_CFI_ZONEVISTIME_COVERAGE_COMPUTE_AREA_ERR
} XV_CFI_Zonevistime_coverage_err_enum;

/* Error codes list of xv_compute_aoi (AN-672) */
typedef enum
{
  XV_CFI_COMPUTE_AOI_INIT_ERR = XL_ZERO_E,
  XV_CFI_COMPUTE_AOI_ORBIT_TO_TIME_ERR,
  XV_CFI_COMPUTE_AOI_SWATH_COMPUTE_ERR
} XV_CFI_compute_aoi_err_enum;

/* Functions names aliases (historical names LEFT - official names RIGHT) */
/* ====================================================================== */

#define XV_Silent      xv_silent
#define XV_Verbose     xv_verbose
#define XV_Print_Msg   xv_print_msg
#define XV_Vector_Msg  xv_get_msg
#define XV_Vector_Code xv_get_code

/* old names for xv_get_msg and  xv_get_code: */
#define xv_vector_msg  xv_get_msg
#define xv_vector_code xv_get_code

/*************************************************************************/

#endif
