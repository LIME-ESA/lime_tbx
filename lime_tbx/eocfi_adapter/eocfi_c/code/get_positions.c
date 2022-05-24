#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <explorer_orbit.h>

#define XO_MAX_STR_LENGTH        256
#define DEFAULT_TERMINATE_CHECK_STATUS 1

#define DEFINE_CHECK_STATUS(xx, XX)                                                                 \
  static void check_##xx##_status(long func, long status, const long *err, int terminate) {         \
    if (status != XX##_OK)                                                                          \
    {                                                                                               \
      long size;                                                                                    \
      char msgs[XX##_MAX_COD][XX##_MAX_STR];                                                        \
      long codes[XX##_MAX_COD];                                                                     \
      xx##_get_msg(&func, (long *)(err), &size, msgs);                                              \
      xx##_get_code(&func, (long *)(err), &size, codes);                                            \
      for (size_t i = 0; i < (size_t)(size); ++i)                                                   \
      { printf("[***] %s [%ld][%ld]\n", msgs[i], codes[i], status); }                               \
      if (terminate && (status == XX##_ERR))                                                        \
      { exit(1); }                                                                                  \
    }                                                                                               \
  }
#define DEFAULT_TERMINATE_CHECK_STATUS 1

#if defined(EXAMPLE_DATA_BASED_INIT)
DEFINE_CHECK_STATUS(xd, XD)
#endif
DEFINE_CHECK_STATUS(xl, XL)

struct Date{
    int y, m, d, hh, mm, ss;
};

const int monthDays[12]
        = { 31, 28, 31, 30, 31, 30,
            31, 31, 30, 31, 30, 31
        };

int countLeapYears(struct Date d)
{
    int years = d.y;

    // Check if the current year needs to be
    //  considered for the count of leap years
    // or not
    if (d.m <= 2)
        years--;

    // An year is a leap year if it
    // is a multiple of 4,
    // multiple of 400 and not a
    // multiple of 100.
    return years / 4
           - years / 100
           + years / 400;
}

// This function returns number of
// days between two given dates
int getDifference(struct Date dt)
{
    struct Date dt2 = dt;
    struct Date dt1 = {2000, 1, 1, 00, 00, 00};
    // COUNT TOTAL NUMBER OF DAYS
    // BEFORE FIRST DATE 'dt1'

    // initialize count using years and day
    long int n1 = dt1.y * 365 + dt1.d;

    // Add days for months in given date
    for (int i = 0; i < dt1.m - 1; i++)
        n1 += monthDays[i];

    // Since every leap year is of 366 days,
    // Add a day for every leap year
    n1 += countLeapYears(dt1);

    // SIMILARLY, COUNT TOTAL NUMBER OF
    // DAYS BEFORE 'dt2'

    long int n2 = dt2.y * 365 + dt2.d;
    for (int i = 0; i < dt2.m - 1; i++)
        n2 += monthDays[i];
    n2 += countLeapYears(dt2);

    // return difference between two counts
    return (n2 - n1);
}

char* to_timestamp(const xl_time_id *time, double instant) {

    long time_ref = XL_TIME_UTC;
    long fmt_in   = XL_PROC;
    long fmt_out  = XL_ASCII_CCSDSA_MICROSEC;
    char* buffer = (char*)malloc(100*sizeof(char));
    long errors[XL_NUM_ERR_PROC_ASCII];

    long status = xl_time_processing_to_ascii((xl_time_id *)(time), &fmt_in, &time_ref, &instant, &fmt_out,
                                              &time_ref, buffer, errors);
    //check_xl_status(XL_TIME_PROCESSING_TO_ASCII_ID, status, errors, DEFAULT_TERMINATE_CHECK_STATUS);

    return buffer;
}

double * get_sun_position(
        int y,
        int m,
        int d,
        int hh,
        int mm,
        int ss,
        char *init_time_file
){

    double static position_returns[3];
    struct Date dt = {y, m, d, hh, mm, ss};

    xl_model_id model = {NULL};
    {
        long mode = XL_MODEL_DEFAULT;
        long errors[XL_NUM_ERR_MODEL_INIT];
        long status = xl_model_init(&mode, NULL, &model, errors);
        check_xl_status(XL_MODEL_INIT_ID, status, errors, DEFAULT_TERMINATE_CHECK_STATUS);
    }

    // Initialize time, based on the correlation provided with the orbit
    xl_time_id time = {NULL};
    double vstart, vstop;  // Validity start/stop, decimal days from 2000-01-01T00:00:00.000000 UTC
    {
        long mode           = XL_TIMEMOD_AUTO;
        const char *files[] = {init_time_file};
        long n_files        = 1;
        long selection      = XL_SEL_FILE;
        long time_ref       = XL_TIME_UTC;
        long errors[XL_NUM_ERR_TIME_REF_INIT_FILE];

        long status = xl_time_ref_init_file(&mode, &n_files, (char **)(files), &selection, &time_ref, NULL,
                                            NULL, NULL, NULL, &vstart, &vstop, &time, errors);

        check_xl_status(XL_TIME_REF_INIT_FILE_ID, status, errors, DEFAULT_TERMINATE_CHECK_STATUS);
    }

    // Convert the datetime to a correct format
    double seconds = (dt.hh * 3600) + (dt.mm * 60) + dt.ss;  // [seconds]
    double start     = getDifference(dt);

    double instant =
            start + seconds / 86400.;  // calculate instant with 300. seconds step from start

#if defined DEBUG
    printf("At time instant: %lf (%s)\n", instant, to_timestamp(&time, instant));
#endif

    // Calculate Sun ephemeris
    double sun_position[3];  // (px, py, pz), in Earth Fixed
    double sun_velocity[3];  // (vx, vy, vz), in Earth Fixed
    {
        long time_ref = XL_TIME_UTC;
        long errors[XL_NUM_ERR_SUN];

        long status = xl_sun(&model, &time, &time_ref, &instant, sun_position, sun_velocity, errors);
        check_xl_status(XL_SUN_ID, status, errors, DEFAULT_TERMINATE_CHECK_STATUS);
    }
#if defined DEBUG
    printf("Moon position: (px, py, pz) = (%lf, %lf, %lf)\n", moon_position[0], moon_position[1], moon_position[2]);
#endif
    position_returns[0] = sun_position[0];
    position_returns[1] = sun_position[1];
    position_returns[2] = sun_position[2];

    return position_returns;

}

double * get_moon_position(
        int y,
        int m,
        int d,
        int hh,
        int mm,
        int ss,
        char *init_time_file
        ){

    double static position_returns[3];
    struct Date dt = {y, m, d, hh, mm, ss};

    xl_model_id model = {NULL};
    {
        long mode = XL_MODEL_DEFAULT;
        long errors[XL_NUM_ERR_MODEL_INIT];
        long status = xl_model_init(&mode, NULL, &model, errors);
        check_xl_status(XL_MODEL_INIT_ID, status, errors, DEFAULT_TERMINATE_CHECK_STATUS);
    }

    // Initialize time, based on the correlation provided with the orbit
    xl_time_id time = {NULL};
    double vstart, vstop;  // Validity start/stop, decimal days from 2000-01-01T00:00:00.000000 UTC
    {
        long mode           = XL_TIMEMOD_AUTO;
        const char *files[] = {init_time_file};
        long n_files        = 1;
        long selection      = XL_SEL_FILE;
        long time_ref       = XL_TIME_UTC;
        long errors[XL_NUM_ERR_TIME_REF_INIT_FILE];

        long status = xl_time_ref_init_file(&mode, &n_files, (char **)(files), &selection, &time_ref, NULL,
                                            NULL, NULL, NULL, &vstart, &vstop, &time, errors);

        check_xl_status(XL_TIME_REF_INIT_FILE_ID, status, errors, DEFAULT_TERMINATE_CHECK_STATUS);
    }

    // Convert the datetime to a correct format
    double seconds = (dt.hh * 3600) + (dt.mm * 60) + dt.ss;  // [seconds]
    double start     = getDifference(dt);

    double instant =
            start + seconds / 86400.;  // calculate instant with 300. seconds step from start

#if defined DEBUG
    printf("At time instant: %lf (%s)\n", instant, to_timestamp(&time, instant));
#endif

    // Calculate Moon ephemeris
    double moon_position[3];  // (px, py, pz), in Earth Fixed
    double moon_velocity[3];  // (vx, vy, vz), in Earth Fixed
    {
        long time_ref = XL_TIME_UTC;
        long errors[XL_NUM_ERR_MOON];

        long status = xl_moon(&model, &time, &time_ref, &instant, moon_position, moon_velocity, errors);
        check_xl_status(XL_MOON_ID, status, errors, DEFAULT_TERMINATE_CHECK_STATUS);
    }
#if defined DEBUG
    printf("Moon position: (px, py, pz) = (%lf, %lf, %lf)\n", moon_position[0], moon_position[1], moon_position[2]);
#endif
    position_returns[0] = moon_position[0];
    position_returns[1] = moon_position[1];
    position_returns[2] = moon_position[2];

    return position_returns;

}


double *  get_satellite_position(
        long sat_id,
        int y,
        int m,
        int d,
        int hh,
        int mm,
        int ss,
        char *init_time_file,
        char **orbit_files, //char orbit_file[],
        long n_orbit_files
        ){
    
    double static position_returns[3];

    char msg[XO_MAX_COD][XO_MAX_STR];
    long n = 0;
    long func_id;

    // Initialize time, based on the correlation provided with the orbit
    xl_time_id time_id = {NULL};
    double vstart, vstop;  // Validity start/stop, decimal days from 2000-01-01T00:00:00.000000 UTC
    double vstart2, vstop2;  // Validity start/stop, decimal days from 2000-01-01T00:00:00.000000 UTC

    /* xl_time_ref_init_file */
    long   trif_time_model, trif_n_files, trif_time_init_mode, trif_time_ref ;
    long   errors[XO_ERR_VECTOR_MAX_LENGTH], status;

    /* xo_orbit_init_file */
    xo_orbit_id    orbit_id    = {NULL};
    xl_model_id    model_id    = {NULL};
    long time_init_mode, orbit_mode;
    char   *input_orbit_files[3];
    long orbit0, orbit1;
    double val_time0, val_time1, time;
    xo_validity_time val_time;

    /* variables for xo_osv_compute_extra */

    long extra_choice;
    double orbit_model_out[XO_ORBIT_EXTRA_NUM_DEP_ELEMENTS],
            orbit_extra_out[XO_ORBIT_EXTRA_NUM_INDEP_ELEMENTS];

    /* common variables */
    long propag_model = XO_PROPAG_MODEL_MEAN_KEPL;
    long time_ref_utc = XO_TIME_UTC;

    double pos[3];
    double vel[3];
    double acc[3];
    struct Date dt = {y, m, d, hh, mm, ss};

    /* Time Initialization */
    /* ------------------- */

    /* Important note: In order to have consistent set of data, the time correlations should be
       the same in all files. For that reason, in the following examples the time reference
       and the orbit will be initialized using the same file.*/

    double seconds = (dt.hh * 3600) + (dt.mm * 60) + dt.ss;
    double start_time     = getDifference(dt);
    vstart2 = start_time + seconds / 86400.;
    vstop2 = vstart2 + 0.5;

    trif_time_model     = XL_TIMEMOD_AUTO;
    trif_n_files        = 1;
    trif_time_init_mode = XL_SEL_FILE;
    trif_time_ref       = XL_TIME_UTC;

    //input_orbit_files[0] = orbit_file;

    for (int i=0;i<n_orbit_files; i++){
        input_orbit_files[i] = orbit_files[i];
    }

    status = xl_time_ref_init_file(&trif_time_model, &trif_n_files,
                                   input_orbit_files,
                                   &trif_time_init_mode, &trif_time_ref, NULL,
                                   NULL, NULL, NULL, &vstart, &vstop, &time_id, errors);

    if (status != XO_OK)
    {
        func_id = XL_TIME_REF_INIT_FILE_ID;
        xl_get_msg(&func_id, errors, &n, msg);
        xl_print_msg(&n, msg);
    }

    /* Orbit initialization */
    /* -------------------- */

    time_init_mode = XO_SEL_FILE;


//    xl_verbose();
//    xo_verbose();
    // input_orbit_files[0] = orbit_file;
    for (int i=0;i<n_orbit_files; i++){
        input_orbit_files[i] = orbit_files[i];
    }

    orbit_mode = XO_ORBIT_INIT_AUTO;

    status =  xo_orbit_init_file(&sat_id, &model_id, &time_id,
                                 &orbit_mode, &n_orbit_files, input_orbit_files,
                                 &time_init_mode, &time_ref_utc,
                                 &vstart, &vstop, &orbit0, &orbit1,
                                 &val_time0, &val_time1, &orbit_id,
                                 errors);

    if (status != XO_OK)
    {
        func_id = XO_ORBIT_INIT_FILE_ID;
        xo_get_msg(&func_id, errors, &n, msg);
        xo_print_msg(&n, msg);
    }

    status = xo_orbit_get_osv_compute_validity(&orbit_id, &val_time);

    val_time0 = vstart2;
    val_time1 = vstop2;
#if defined DEBUG
    printf("\n\t-  validity times = ( %f , %f )", val_time0, val_time1 );
#endif
    time = val_time0;

    status = xo_osv_compute(&orbit_id, &propag_model, &time_ref_utc, &time,
            /* outputs */
                            pos, vel, acc, errors);

    if (status != XO_OK)
    {
        func_id = XO_OSV_COMPUTE_ID;
        xo_get_msg(&func_id, errors, &n, msg);
        xo_print_msg(&n, msg);
    }
#if defined DEBUG
    printf( "\n\t-  time = %lf (%s)", time,  to_timestamp(&time_id, time));
    printf( "\n\t-  pos[0] = %lf", pos[0] );
    printf( "\n\t-  pos[1] = %lf", pos[1] );
    printf( "\n\t-  pos[2] = %lf", pos[2] );
    printf( "\n\t-  vel[0] = %lf", vel[0] );
    printf( "\n\t-  vel[1] = %lf", vel[1] );
    printf( "\n\t-  vel[2] = %lf", vel[2] );
    printf( "\n\t-  acc[0] = %lf", acc[0] );
    printf( "\n\t-  acc[1] = %lf", acc[1] );
    printf( "\n\t-  acc[2] = %lf", acc[2] );
#endif
    extra_choice = XO_ORBIT_EXTRA_NO_RESULTS;

    status = xo_osv_compute_extra(&orbit_id, &extra_choice,
                                  orbit_model_out, orbit_extra_out, errors);
    if (status != XO_OK)
    {
        func_id = XO_OSV_COMPUTE_EXTRA_ID;
        xo_get_msg(&func_id, errors, &n, msg);
        xo_print_msg(&n, msg);
    }



    /* Calling to xo_orbit_close */
    /* ------------------------- */

    status = xo_orbit_close(&orbit_id, errors);
    if (status != XO_OK)
    {
        func_id = XO_ORBIT_CLOSE_ID;
        xo_get_msg(&func_id, errors, &n, msg);
        xo_print_msg(&n, msg);
    }

    xl_time_close(&time_id, errors);

    position_returns[0] = pos[0];
    position_returns[1] = pos[1];
    position_returns[2] = pos[2];

    return position_returns;
}

int main (int argc, char *argv[]){
    double *positions;
    char   *orbit_file[5] = {'\0'};
    orbit_file[0] = "data/mission_configuration_files/SENTINEL5P/OSF/S5P_OPER_MPL_ORBSCT_20171013T104928_99999999T999999_0008.EOF";
    positions = get_satellite_position(
            XO_SAT_SENTINEL_5P,
            2022, 1, 2, 00, 12, 00,
            "data/207_BULLETIN_B207.txt",
            orbit_file,
            1
            );
    printf("****************************************************\n");
    printf("*    position_x  = %lf\n", positions[0]);
    printf("*    position_y  = %lf\n", positions[1]);
    printf("*    position_z  = %lf\n", positions[2]);
    printf("****************************************************\n");
}
