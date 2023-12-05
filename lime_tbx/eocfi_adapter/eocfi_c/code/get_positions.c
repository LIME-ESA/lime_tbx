#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

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
    int y, m, d, hh, mm, ss, microsecs;
};

static const int monthDays[12]
        = { 31, 28, 31, 30, 31, 30,
            31, 31, 30, 31, 30, 31
        };

static int countLeapYears(struct Date d)
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
static int getDifference(struct Date dt)
{
    struct Date dt2 = dt;
    struct Date dt1 = {2000, 1, 1, 00, 00, 00, 00};
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

static char* to_timestamp(const xl_time_id *time, double instant) {

    long time_ref = XL_TIME_UTC;
    long fmt_in   = XL_PROC;
    long fmt_out  = XL_ASCII_CCSDSA_MICROSEC;
    char* buffer = (char*)malloc(100*sizeof(char));
    long errors[XL_NUM_ERR_PROC_ASCII];

    long status = xl_time_processing_to_ascii((xl_time_id *)(time), &fmt_in, &time_ref, &instant, &fmt_out,
                                              &time_ref, buffer, errors);
    check_xl_status(XL_TIME_PROCESSING_TO_ASCII_ID, status, errors, DEFAULT_TERMINATE_CHECK_STATUS);

    return buffer;
}

double * get_sun_position(
        int y,
        int m,
        int d,
        int hh,
        int mm,
        int ss,
        int microsecs,
        char *init_time_file
){

    double static position_returns[3];
    struct Date dt = {y, m, d, hh, mm, ss, microsecs};

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
    double seconds = (dt.hh * 3600) + (dt.mm * 60) + dt.ss + dt.microsecs/1000000.0;  // [seconds]
    double start     = getDifference(dt);

    double instant =
            start + seconds / 86400.;  // calculate instant with 300. seconds step from start

#ifdef DEBUG
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
#ifdef DEBUG
    printf("Moon position: (px, py, pz) = (%lf, %lf, %lf)\n", sun_position[0], sun_position[1], sun_position[2]);
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
        int microsecs,
        char *init_time_file
        ){

    double static position_returns[3];
    struct Date dt = {y, m, d, hh, mm, ss, microsecs};

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
    double seconds = (dt.hh * 3600) + (dt.mm * 60) + dt.ss + dt.microsecs/1000000.0;  // [seconds]
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



#ifdef _WIN32
__declspec(dllexport) int position_to_j2000(
#else
    int  position_to_j2000(
#endif
        double* position,
        double* vel,
        double* acc,
        xl_time_id* time_id,
        long* time_ref,
        double* time
    ){
    long status, func_id, n;
    long xl_ierr[XL_ERR_VECTOR_MAX_LENGTH];
    char msg[XL_MAX_COD][XL_MAX_STR];
    xl_model_id model_id = {NULL};
    long model_mode,
    models[XL_NUM_MODEL_TYPES_ENUM];
    long cs_in, cs_out;
    long calc_mode = XL_CALC_POS_VEL_ACC;
    double pos_out[3] = {0.0, 0.0, 0.0};
    double vel_out[3] = {0.0, 0.0, 0.0};
    double acc_out[3] = {0.0, 0.0, 0.0};

    model_mode = XL_MODEL_DEFAULT;
    status = xl_model_init(&model_mode, models, &model_id, xl_ierr);
    if (status != XL_OK)
    {
        func_id = XL_MODEL_INIT_ID;
        xl_get_msg(&func_id, xl_ierr, &n, msg);
        xl_print_msg(&n, msg);
        if (status <= XL_ERR) return(XL_ERR);
    }


    cs_in = XL_EF; /* Initial coordinate system = True of Date */
    cs_out = XL_BM2000; /* Final coordinate system = Earth fixed */
    status = xl_change_cart_cs(&model_id, time_id, &calc_mode, &cs_in, &cs_out,
                                time_ref, time, position, vel, acc, pos_out, vel_out, acc_out);
    if (status != XL_OK)
    {
        func_id = XL_CHANGE_CART_CS_ID;
        xl_get_msg(&func_id, &status, &n, msg);
        xl_print_msg(&n, msg);
        if (status <= XL_ERR) return(XL_ERR);
    }
    position[0] = pos_out[0];
    position[1] = pos_out[1];
    position[2] = pos_out[2];

    return XO_OK;
}


#ifdef _WIN32
__declspec(dllexport) int  get_satellite_position_osf(
#else
    int  get_satellite_position_osf(
#endif
        long sat_id,
        int quant_dates,
        int** dates,
        char* orbit_file,
        double** position_returns
    ){
    char msg[XO_MAX_COD][XO_MAX_STR];
    long n = 0;
    long one = 1;
    long func_id;

    // Initialize time, based on the correlation provided with the orbit
    xl_time_id time_id = {NULL};
    double vstart, vstop;  // Validity start/stop, decimal days from 2000-01-01T00:00:00.000000 UTC
    double vstart2, vstop2;  // Validity start/stop, decimal days from 2000-01-01T00:00:00.000000 UTC

    /* xl_time_ref_init_file */
    long   trif_time_model, trif_time_init_mode, trif_time_ref ;
    long   errors[XO_ERR_VECTOR_MAX_LENGTH], status;

    /* xo_orbit_init_file */
    xo_orbit_id    orbit_id    = {NULL};
    xl_model_id    model_id    = {NULL};
    long time_init_mode, orbit_mode;
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

#ifdef DEBUG
    xl_verbose();
    xo_verbose();
    xd_verbose();
    xf_verbose();
#endif

    /* Time Initialization */
    /* ------------------- */

    trif_time_model     = XL_TIMEMOD_AUTO;
    trif_time_init_mode = XL_SEL_FILE;
    trif_time_ref       = XL_TIME_UTC;

    status = xl_time_ref_init_file(&trif_time_model, &one,
                                   &orbit_file,
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

    orbit_mode = XO_ORBIT_INIT_AUTO;

    status =  xo_orbit_init_file(&sat_id, &model_id, &time_id,
                                 &orbit_mode, &one, &orbit_file,
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

    for (int i = 0; i < quant_dates; i++){
        struct Date dt = {dates[i][0], dates[i][1], dates[i][2], dates[i][3], dates[i][4], dates[i][5], dates[i][6]};
        double seconds = (dt.hh * 3600) + (dt.mm * 60) + dt.ss + dt.microsecs/1000000.0;
        double start_time     = getDifference(dt);
        vstart2 = start_time + seconds / 86400.;
        vstop2 = vstart2 + 0.5;
        val_time0 = vstart2;
        val_time1 = vstop2;
#ifdef DEBUG
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
#ifdef DEBUG
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

        status = position_to_j2000(pos, vel, acc, &time_id, &time_ref_utc, &time);
#ifdef DEBUG
        printf( "\n\t-  time = %lf (%s)", time,  to_timestamp(&time_id, time));
        printf( "\n\t-  pos[0] = %lf", pos[0] );
        printf( "\n\t-  pos[1] = %lf", pos[1] );
        printf( "\n\t-  pos[2] = %lf", pos[2] );
#endif
        if (status != XO_OK)
        {
            return status;
        }

        position_returns[i][0] = pos[0];
        position_returns[i][1] = pos[1];
        position_returns[i][2] = pos[2];
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
    return XL_OK;
}

#ifdef _WIN32
__declspec(dllexport) int  get_satellite_position_tle(
#else
    int  get_satellite_position_tle(
#endif
        long sat_id,
        int quant_dates,
        int** dates,
        char *time_file,
        long norad,
        char *sat_name,
        char *intdes,
        char *tle_file,
        double** position_returns
        ){

#ifdef DEBUG
    printf("%ld, %d\n", sat_id, quant_dates);
    for(int i = 0; i < quant_dates; i++){
        for(int j = 0; j < 6; j++){
            printf("%d ", dates[i][j]);
        }
        printf("\n");
    }
    printf("%s\n%ld, %s, %s\n%s\n", time_file, norad, sat_name, intdes, tle_file);
    fflush(stdout);
#endif

    char msg[XO_MAX_COD][XO_MAX_STR];
    long n = 0;
    long func_id;
    long one = 1;

    // Initialize time, based on the correlation provided with the orbit
    xl_time_id time_id = {NULL};
    double vstart, vstop;  // Validity start/stop, decimal days from 2000-01-01T00:00:00.000000 UTC
    double vstart2, vstop2;  // Validity start/stop, decimal days from 2000-01-01T00:00:00.000000 UTC

    /* xl_time_ref_init_file */
    long   trif_time_model, trif_time_init_mode, trif_time_ref ;
    long   errors[XO_ERR_VECTOR_MAX_LENGTH], status;

    /* xo_orbit_init_file */
    xo_orbit_id    orbit_id    = {NULL};
    xl_model_id    model_id    = {NULL};
    long time_init_mode, orbit_mode;
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

#ifdef DEBUG
    xl_verbose();
    xo_verbose();
    xd_verbose();
    xf_verbose();
#endif

    /* Time Initialization */
    /* ------------------- */


    trif_time_model     = XL_TIMEMOD_AUTO;
    trif_time_init_mode = XL_SEL_FILE;
    trif_time_ref       = XL_TIME_UTC;

#ifdef DEBUG
    printf("init\n");fflush(stdout);
#endif

    status = xl_time_ref_init_file(&trif_time_model, &one,
                                   &time_file,
                                   &trif_time_init_mode, &trif_time_ref, NULL,
                                   NULL, NULL, NULL, &vstart, &vstop, &time_id, errors);

#ifdef DEBUG
    printf("check\n");fflush(stdout);
#endif

    if (status != XO_OK)
    {
        func_id = XL_TIME_REF_INIT_FILE_ID;
        xl_get_msg(&func_id, errors, &n, msg);
        xl_print_msg(&n, msg);
    }

    /* TLE Initialization */
    char norad_satcat[50];
    char int_des[10];
    strcpy(norad_satcat, sat_name);
    strcpy(int_des, intdes);
    status = xl_set_tle_sat_data(&sat_id, &norad, norad_satcat, int_des);
    if (status != XL_OK){
        if (status <= XL_ERR) return(XL_ERR);
    }

    /* Orbit initialization */
    /* -------------------- */

    time_init_mode = XO_SEL_FILE;

    orbit_mode = XO_ORBIT_INIT_TLE_MODE;
#ifdef DEBUG
    printf("almost\n");fflush(stdout);
#endif
    status =  xo_orbit_init_file(&sat_id, &model_id, &time_id,
                                 &orbit_mode, &one, &tle_file,
                                 &time_init_mode, &time_ref_utc,
                                 &vstart, &vstop, &orbit0, &orbit1,
                                 &val_time0, &val_time1, &orbit_id,
                                 errors);
#ifdef DEBUG
    printf("fin\n");fflush(stdout);
#endif
    if (status != XO_OK)
    {
        func_id = XO_ORBIT_INIT_FILE_ID;
        xo_get_msg(&func_id, errors, &n, msg);
        xo_print_msg(&n, msg);
    }

    status = xo_orbit_get_osv_compute_validity(&orbit_id, &val_time);

    for (int i = 0; i < quant_dates; i++){
        struct Date dt = {dates[i][0], dates[i][1], dates[i][2], dates[i][3], dates[i][4], dates[i][5], dates[i][6]};
        double seconds = (dt.hh * 3600) + (dt.mm * 60) + dt.ss + dt.microsecs/1000000.0;
        double start_time     = getDifference(dt);
        vstart2 = start_time + seconds / 86400.;
        vstop2 = vstart2 + 0.5;
        val_time0 = vstart2;
        val_time1 = vstop2;
#ifdef DEBUG
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
#ifdef DEBUG
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

        position_to_j2000(pos, vel, acc, &time_id, &time_ref_utc, &time);

        position_returns[i][0] = pos[0];
        position_returns[i][1] = pos[1];
        position_returns[i][2] = pos[2];
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
    return XL_OK;
}



#define STR_SIZE_INPUT 1000

int main (int argc, char *argv[]){
    // Main code can be called to perform tle calculations, as the shared library doesn't seem to work.
    int n_dates, is_tle;
    long sat_id;
    long norad;
    char *tle_file = (char*)malloc(sizeof(char) * STR_SIZE_INPUT);
    char *orbit_file = (char*)malloc(sizeof(char) * STR_SIZE_INPUT);
    char *sat_name = (char*)malloc(sizeof(char) * STR_SIZE_INPUT);
    char *intdes = (char*)malloc(sizeof(char) * STR_SIZE_INPUT);
    if(argc < 6){
        is_tle = 1;
        n_dates = 1;
    }else{
        is_tle = atoi(argv[1]);
        n_dates = atoi(argv[2]);
    }
    int **dates = (int**)malloc(sizeof(int*)*n_dates);
    double **positions = (double**)malloc(sizeof(double*)*n_dates);
    for(int i = 0; i < n_dates; i++){
        positions[i] = (double*)malloc(sizeof(double)*3);
        dates[i] = (int*)malloc(sizeof(int)*7);
    }
    if(argc < 6){
        sat_id = 200;
        dates[0][0] = 2022; dates[0][1] = 1; dates[0][2] = 1; dates[0][3] = 0; dates[0][4] = 12; dates[0][5] = 0; dates[0][6] = 0;
        strcpy(orbit_file, "../../../eocfi_data/data/mission_configuration_files/PROBAV/OSF/PROBA-V_TLE2ORBPRE_20130507T052939_20221002T205653_0001.EOF");
        strcpy(tle_file, "../../../eocfi_data/data/mission_configuration_files/PROBAV/TLE/PROBA-V_20130507T000000_20221012T000000_0001.TLE");
        norad = 39159;
        strcpy(sat_name, "PROBA-V");
        strcpy(intdes, "13021A");
    }else if(is_tle){
        sat_id = atol(argv[3]);
        norad = atol(argv[4]);
        strcpy(tle_file, argv[5]);
        strcpy(orbit_file, argv[6]);
        strcpy(sat_name, argv[7]);
        strcpy(intdes, argv[8]);
        for(int i = 0; i < n_dates; i++){
            sscanf(argv[9+i], "%d-%d-%dT%d:%d:%d.%d", &dates[i][0], &dates[i][1], &dates[i][2], &dates[i][3], &dates[i][4], &dates[i][5], &dates[i][6]);
        }
        get_satellite_position_tle(
            sat_id,
            n_dates,
            dates,
            orbit_file,
            norad,
            sat_name,
            intdes,
            tle_file,
            positions
        );
    }else{
        sat_id = atol(argv[3]);
        strcpy(orbit_file, argv[4]);
        for(int i = 0; i < n_dates; i++){
            sscanf(argv[5+i], "%d-%d-%dT%d:%d:%d.%d", &dates[i][0], &dates[i][1], &dates[i][2], &dates[i][3], &dates[i][4], &dates[i][5], &dates[i][6]);
        }
        get_satellite_position_osf(
            sat_id,
            n_dates,
            dates,
            orbit_file,
            positions
        );
    }
    free(tle_file);
    free(orbit_file);
    free(sat_name);
    free(intdes);
    int digs = DECIMAL_DIG;
    for(int i = 0; i < n_dates; i++){
        printf("%.*e\n", digs, positions[i][0]);
        printf("%.*e\n", digs, positions[i][1]);
        printf("%.*e\n", digs, positions[i][2]);
        fflush(stdout);
        free(dates[i]);
        free(positions[i]);
    }
    free(dates);
    free(positions);
    return 0;
}
