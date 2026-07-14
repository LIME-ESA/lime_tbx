#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include <explorer_orbit.h>
#include <explorer_pointing.h>
#include <explorer_visibility.h>

#define XO_MAX_STR_LENGTH        256
#define DEFAULT_TERMINATE_CHECK_STATUS 1

typedef enum {
    INPUT_TYPE_OSF = 0,
    INPUT_TYPE_TLE = 1
} InputType;

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

#if defined(EXAMPLE_DATA_BASED_INIT)
DEFINE_CHECK_STATUS(xd, XD)
#endif

#ifdef DEBUG
DEFINE_CHECK_STATUS(xl, XL)
#endif

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
    if (d.m <= 2)
        years--;
    return years / 4
           - years / 100
           + years / 400;
}

static int getDifference(struct Date dt)
{
    struct Date dt2 = dt;
    struct Date dt1 = {2000, 1, 1, 00, 00, 00, 00};

    if (dt.y < 2000) {
        fprintf(stderr, "Error: Date before year 2000 not supported\n");
        return -1;
    }

    long int n1 = dt1.y * 365 + dt1.d;
    for (int i = 0; i < dt1.m - 1; i++)
        n1 += monthDays[i];
    n1 += countLeapYears(dt1);

    long int n2 = dt2.y * 365 + dt2.d;
    for (int i = 0; i < dt2.m - 1; i++)
        n2 += monthDays[i];
    n2 += countLeapYears(dt2);

    return (n2 - n1);
}

#ifdef DEBUG
static char* to_timestamp(const xl_time_id *time, double instant) {
    long time_ref = XL_TIME_UTC;
    long fmt_in   = XL_PROC;
    long fmt_out  = XL_ASCII_CCSDSA_MICROSEC;
    static char buffer[100];
    long errors[XL_NUM_ERR_PROC_ASCII];
    long status = xl_time_processing_to_ascii((xl_time_id *)(time), &fmt_in, &time_ref, &instant, &fmt_out,
                                              &time_ref, buffer, errors);
    check_xl_status(XL_TIME_PROCESSING_TO_ASCII_ID, status, errors, DEFAULT_TERMINATE_CHECK_STATUS);
    return buffer;
}
#endif

static int compute_position_for_date(
        xo_orbit_id *orbit_id,
        const xl_time_id *time_id,
        const struct Date *dt,
        long propag_model,
        double *pos,
        double *vel,
        double *acc,
        long *errors)
{
    char msg[XO_MAX_COD][XO_MAX_STR];
    long n = 0;
    double seconds = (dt->hh * 3600) + (dt->mm * 60) + dt->ss + dt->microsecs / 1000000.0;
    long diff_days = getDifference(*dt);
    if (diff_days < 0) return XL_ERR;  // invalid date

    double vstart2 = diff_days + seconds / 86400.0;
    double val_time0 = vstart2;

#ifdef DEBUG
    printf("\n\t-  validity time = ( %f )", val_time0);
#endif

    double time = val_time0;

    long time_ref_utc = XO_TIME_UTC;

    long status = xo_osv_compute(orbit_id, &propag_model, &time_ref_utc, &time,
                            pos, vel, acc, errors);

    if (status != XO_OK) {
        long func_id = XO_OSV_COMPUTE_ID;
        xo_get_msg(&func_id, errors, &n, msg);
        xo_print_msg(&n, msg);
        if (status == XO_ERR) {
            return status;
        }
    }

#ifdef DEBUG
    printf( "\n\t-  time = %lf (%s)", time,  to_timestamp(time_id, time));
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

    // Extra computation not used - removed to avoid unused vars
    return XL_OK;
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

    xl_time_id time_id = {NULL};
    double vstart, vstop;

    long   trif_time_model, trif_time_init_mode, trif_time_ref ;
    long   errors[XO_ERR_VECTOR_MAX_LENGTH], status;

    xo_orbit_id    orbit_id    = {NULL};
    xl_model_id    model_id    = {NULL};
    long time_init_mode, orbit_mode;
    long orbit0, orbit1;
    double val_time0, val_time1;

    long time_ref_utc = XO_TIME_UTC;

    double pos[3], vel[3], acc[3];

#ifdef DEBUG
    xl_verbose();
    xo_verbose();
    xd_verbose();
    xf_verbose();
#endif

    // Input validation
    if (dates == NULL || position_returns == NULL || orbit_file == NULL) {
        return XL_ERR;
    }

    /* Time Initialization */
    trif_time_model     = XL_TIMEMOD_AUTO;
    trif_time_init_mode = XL_SEL_FILE;
    trif_time_ref       = XL_TIME_UTC;

    char *file_list[1] = { orbit_file };
    status = xl_time_ref_init_file(&trif_time_model, &one, file_list,
                                   &trif_time_init_mode, &trif_time_ref, NULL,
                                   NULL, NULL, NULL, &vstart, &vstop, &time_id, errors);

    if (status != XL_OK)
    {
        func_id = XL_TIME_REF_INIT_FILE_ID;
        xl_get_msg(&func_id, errors, &n, msg);
        xl_print_msg(&n, msg);
        return XL_ERR;
    }

    /* Orbit initialization */
    time_init_mode = XO_SEL_FILE;
    orbit_mode = XO_ORBIT_INIT_AUTO;

    status =  xo_orbit_init_file(&sat_id, &model_id, &time_id,
                                 &orbit_mode, &one, file_list,
                                 &time_init_mode, &time_ref_utc,
                                 &vstart, &vstop, &orbit0, &orbit1,
                                 &val_time0, &val_time1, &orbit_id,
                                 errors);

    if (status != XO_OK)
    {
        func_id = XO_ORBIT_INIT_FILE_ID;
        xo_get_msg(&func_id, errors, &n, msg);
        xo_print_msg(&n, msg);
        return XL_ERR;
    }

    // Loop over dates
    for (int i = 0; i < quant_dates; i++){
        struct Date dt = {dates[i][0], dates[i][1], dates[i][2],
                          dates[i][3], dates[i][4], dates[i][5], dates[i][6]};

        status = compute_position_for_date(&orbit_id, &time_id, &dt, XO_PROPAG_MODEL_MEAN_KEPL, pos, vel, acc, errors);
        if (status != XL_OK) {
            return XL_ERR;
        }

        position_returns[i][0] = pos[0];
        position_returns[i][1] = pos[1];
        position_returns[i][2] = pos[2];
    }

    /* Calling to xo_orbit_close */
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
    printf("%ld, %s, %s\n%s\n", norad, sat_name, intdes, tle_file);
    fflush(stdout);
#endif

    char msg[XO_MAX_COD][XO_MAX_STR];
    long n = 0;
    long func_id;
    long one = 1;

    xl_time_id time_id = {NULL};
    double vstart, vstop;

    long   errors[XO_ERR_VECTOR_MAX_LENGTH], status;

    xo_orbit_id    orbit_id    = {NULL};
    xl_model_id    model_id    = {NULL};
    long time_init_mode, orbit_mode;
    long orbit0, orbit1;
    double val_time0, val_time1;

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

    // Input validation
    if (dates == NULL || position_returns == NULL || tle_file == NULL) {
        return XL_ERR;
    }

#ifdef DEBUG
    printf("init\n");fflush(stdout);
#endif

    double time[4] = {0.0, 0.0, 0.0, 0.0};  // TAI, UTC, UT1, GPS (all zero)
    long orbit_num = 0;
    double anx_time = 0.0;
    double orbit_duration = 0.0;

    status = xl_time_ref_init(time, &orbit_num, &anx_time, &orbit_duration,
                              &time_id, errors);

    if (status != XL_OK) {
        fprintf(stderr, "xl_time_ref_init failed\n");
        return XL_ERR;
    }

    // Los tiempos de validez los calculas tú con getDifference
    vstart = -1.0;  // No se usan porque el archivo no tiene validez fija
    vstop = -1.0;

#ifdef DEBUG
    printf("check\n");fflush(stdout);
#endif

    /* TLE Initialization */
    char norad_satcat[100];
    char int_des[50];
    snprintf(norad_satcat, 100, "%s", sat_name);
    snprintf(int_des, 50, "%s", intdes);
    status = xl_set_tle_sat_data(&sat_id, &norad, norad_satcat, int_des);
    if (status != XL_OK){
        if (status <= XL_ERR) return(XL_ERR);
    }

    /* Orbit initialization */
    time_init_mode = XO_SEL_FILE;
    orbit_mode = XO_ORBIT_INIT_TLE_MODE;
#ifdef DEBUG
    printf("almost\n");fflush(stdout);
#endif
    char *file_list[1] = { tle_file };
    status =  xo_orbit_init_file(&sat_id, &model_id, &time_id,
                                 &orbit_mode, &one, file_list,
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
        return XL_ERR;
    }

    // Loop over dates
    for (int i = 0; i < quant_dates; i++){
        struct Date dt = {dates[i][0], dates[i][1], dates[i][2],
                          dates[i][3], dates[i][4], dates[i][5], dates[i][6]};

        status = compute_position_for_date(&orbit_id, &time_id, &dt, XO_PROPAG_MODEL_TLE, pos, vel, acc, errors);
        if (status != XL_OK) {
            return XL_ERR;
        }

        position_returns[i][0] = pos[0];
        position_returns[i][1] = pos[1];
        position_returns[i][2] = pos[2];
    }

    /* Calling to xo_orbit_close */
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


#define STR_SIZE_INPUT 2000

int main (int argc, char *argv[]){
    int n_dates, file_input_type;
    long sat_id;
    long norad;
    char *tle_file = (char*)malloc(sizeof(char) * STR_SIZE_INPUT);
    char *orbit_file = (char*)malloc(sizeof(char) * STR_SIZE_INPUT);
    char *sat_name = (char*)malloc(sizeof(char) * STR_SIZE_INPUT);
    char *intdes = (char*)malloc(sizeof(char) * STR_SIZE_INPUT);
    long status = XL_OK;
    if (tle_file == NULL || orbit_file == NULL || sat_name == NULL || intdes == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    if(argc < 6){
        file_input_type = INPUT_TYPE_TLE;
        n_dates = 1;
    }else{
        file_input_type = atoi(argv[1]);
        n_dates = atoi(argv[2]);
    }
    int **dates = (int**)malloc(sizeof(int*)*n_dates);
    double **positions = (double**)malloc(sizeof(double*)*n_dates);
    if (dates == NULL || positions == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        free(tle_file); free(orbit_file); free(sat_name); free(intdes);
        return 1;
    }
    for(int i = 0; i < n_dates; i++){
        positions[i] = (double*)malloc(sizeof(double)*3);
        dates[i] = (int*)malloc(sizeof(int)*7);
        if (positions[i] == NULL || dates[i] == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            // cleanup
            for (int j=0; j<i; j++) { free(positions[j]); free(dates[j]); }
            free(positions); free(dates);
            free(tle_file); free(orbit_file); free(sat_name); free(intdes);
            return 1;
        }
    }

    if(argc < 6){
        sat_id = 200;
        dates[0][0] = 2022; dates[0][1] = 1; dates[0][2] = 1; dates[0][3] = 0; dates[0][4] = 12; dates[0][5] = 0; dates[0][6] = 0;
        snprintf(tle_file, STR_SIZE_INPUT, "%s", "../../../eocfi_data/data/mission_configuration_files/PROBAV/TLE/PROBA-V_20130507T000000_20221012T000000_0001.TLE");
        norad = 39159;
        snprintf(sat_name, STR_SIZE_INPUT, "%s", "PROBA-V");
        snprintf(intdes, STR_SIZE_INPUT, "%s", "13021A");
    }else if(file_input_type == INPUT_TYPE_TLE){
        sat_id = atol(argv[3]);
        norad = atol(argv[4]);
        snprintf(tle_file, STR_SIZE_INPUT, "%s", argv[5]);
        snprintf(sat_name, STR_SIZE_INPUT, "%s", argv[6]);
        snprintf(intdes, STR_SIZE_INPUT, "%s", argv[7]);
        for(int i = 0; i < n_dates; i++){
            sscanf(argv[8+i], "%d-%d-%dT%d:%d:%d.%d", &dates[i][0], &dates[i][1], &dates[i][2], &dates[i][3], &dates[i][4], &dates[i][5], &dates[i][6]);
        }
        status = get_satellite_position_tle(
            sat_id,
            n_dates,
            dates,
            norad,
            sat_name,
            intdes,
            tle_file,
            positions
        );
    }else if (file_input_type == INPUT_TYPE_OSF){
        sat_id = atol(argv[3]);
        snprintf(orbit_file, STR_SIZE_INPUT, "%s", argv[4]);
        for(int i = 0; i < n_dates; i++){
            sscanf(argv[5+i], "%d-%d-%dT%d:%d:%d.%d", &dates[i][0], &dates[i][1], &dates[i][2], &dates[i][3], &dates[i][4], &dates[i][5], &dates[i][6]);
        }
        status = get_satellite_position_osf(
            sat_id,
            n_dates,
            dates,
            orbit_file,
            positions
        );
    }else{
        fprintf(stderr, "Input type not understood.");
    }

    // Free resources
    free(tle_file);
    free(orbit_file);
    free(sat_name);
    free(intdes);
    if (status != XL_OK) {
        fprintf(stderr, "function failed with code %ld\n", status);
        fflush(stderr);
        for(int i = 0; i < n_dates; i++){
            free(dates[i]);
            free(positions[i]);
        }
        free(dates);
        free(positions);
        return status;
    }
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
