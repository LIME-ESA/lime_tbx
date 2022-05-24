#define UTCLEN 34

//#define DEBUG
//#define COMPARISION

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <SpiceUsr.h>


//definimos las constantes necesarias
#define ABCORR "NONE"
#define TARGET "Moon"
#define FRAME "MOON_ME"
#define M_PI 3.14159265358979323846
#define SOLID_ANGLE 6.4177E-5  //angulo solido de la luna
#define AU 149597871           //una unidad astronomica
#define AVG_ME_DISTANCE 384400 //distancia media tierra luna

#define M_PI 3.14159265358979323846

//variables astronomicas
double earthsun_distance=0;
double earthmoon_distance=0;
double sunmoon_distance=0;
double cenital=0,acimutal=0;
double angulo_fase=0;
double tita_obs=0, phi_obs=0,phi_sun=0;
double barreto_correction=0;
int signo_fase=1;

/**
 * Return day of year from calendar
 * @param date Calendar
 * @return Day of year
 */
int dayofyear (struct tm *date)
{
    struct tm *x;
    int result;

    x=(struct tm*)malloc(sizeof(struct tm));
    x->tm_year=date->tm_year;
    x->tm_mon=date->tm_mon;
    x->tm_mday=date->tm_mday;
    x->tm_hour=0;
    x->tm_min=0;
    x->tm_sec=0;

    timegm(x);
    result=x->tm_yday+1;
    free (x);
    return result;
}

/**
 * Calculation day angle
 * @param date Calendar
 * @return day angle in degrees
 */
double day_angle(struct tm *date){
    return (dayofyear(date)-1)*360./365.*M_PI/180.;
}

/**
 * Calculations of distance between earth and sun
 * From Nakajima
 * @param date Calendar
 * @return distance
 */
double earth_sun_distance(struct tm *date){
    double d;

    d=day_angle(date);
    return 1/(1.00011+0.034221*cos(d)+0.00128*sin(d)+0.000719*cos(2*d)+0.000077*sin(2*d));
}

void angulos_astronomicos(char* fecha, char* fecha2, char* site, char *obs_local_level,double *earthmoon_distance, double *sunmoon_distance, double *cenital, double *acimutal, double *angulo_fase, double *phi_obs, double *tita_obs,double *phi_sun, int *signofase){

    SpiceDouble radius;
    SpiceDouble longi, lati, alti;
    SpiceDouble phiSun, phiObs, titaObs;
    SpiceDouble et;
    SpiceDouble et2;
    SpiceDouble lt;
    SpiceDouble state[6];
    SpiceDouble rectan[3];
    SpiceDouble fase_lunar, fase_lunar2;
    SpiceInt dimension;
    SpiceDouble radios_luna[3];
    SpiceDouble relacion_radios;
    SpiceDouble spoint[3];
    SpiceDouble trgepc;
    SpiceDouble srfvec[3];

    double rad2deg;

    //pasamos la fecha al formato adecuado
    utc2et_c ( fecha, &et );
    utc2et_c ( fecha2, &et2 );
    //printf("%s - %s\n", fecha, fecha2);
    //printf("%f\n", et);

    SpiceChar utcstr[UTCLEN];
    et2utc_c (  et , "C", 0, UTCLEN, utcstr  );
    //printf( "ET converts to %s\n\n", utcstr);

    //calcula el vector posicion y velocidad entre la luna y el observador (Estacion de medida)
    spkezr_c ( TARGET, et, FRAME,  ABCORR, site,  state,  &lt);
    *earthmoon_distance = sqrt(pow(state[0],2) + pow(state[1],2) + pow(state[2],2));

    //obtenemos el angulo cenital de la luna
    spkezr_c(TARGET, et, obs_local_level, ABCORR, site, state, &lt); //cambiamos site por 'EARTH'
    rectan[0] = state[0];
    rectan[1] = state[1];
    rectan[2] = state[2];

    reclat_c(rectan, &radius, &longi, &lati);
    rad2deg = M_PI / 180.0;

    *cenital = 90.0 - lati / rad2deg;
    *acimutal = 180 - longi / rad2deg;


    //calculamos los datos sol-luna
    spkezr_c( TARGET, et, FRAME,  ABCORR, "Sun",  state,  &lt);
    *sunmoon_distance = sqrt(pow(state[0],2) + pow(state[1],2) + pow(state[2],2));

    //calculamos el angulo de la fase lunar
    //corregimos el angulo
    fase_lunar = phaseq_c(et, TARGET, "Sun", site, ABCORR);
    fase_lunar2 = phaseq_c(et2, TARGET, "Sun", site, ABCORR);
    if(fase_lunar2 < fase_lunar)
        *signofase = -1;

    *angulo_fase = fase_lunar;

    //calculamos las coordenadas selenograficas
    bodvrd_c("MOON", "RADII", 3, &dimension, radios_luna );
    relacion_radios = (radios_luna[0]-radios_luna[2])/radios_luna[0];

    subpnt_c("Intercept: ellipsoid", TARGET, et, FRAME, ABCORR, site, spoint, &trgepc, srfvec);

    //cambiamos el sistema de coordenadas de rectangular to planetograficas
    recpgr_c(TARGET, spoint, radios_luna[0], relacion_radios,&longi, &lati, &alti);

    titaObs = lati * dpr_c();
    phiObs = longi * dpr_c();

    //calculamos las coordenadas del punto solar
    subslr_c("Intercept: ellipsoid", TARGET, et, FRAME, ABCORR, "Sun", spoint, &trgepc,    srfvec);
    recpgr_c("Sun", spoint, radios_luna[0], relacion_radios,&longi, &lati, &alti);

    phiSun = longi * dpr_c();
    if (phiSun > 180) phiSun = phiSun - 360;
    if (phiSun <= -180) phiSun = phiSun + 360;
    phiSun = phiSun * rad2deg;
    *phi_sun = phiSun;

    if (phiObs > 180) phiObs -= 360;
    if (phiObs <= -180) phiObs += 360;
    *phi_obs = phiObs;

    if (titaObs > 180) titaObs -= 360;
    if (titaObs <= -180) titaObs += 360;
    *tita_obs = titaObs;

    kclear_c();
}

int moon_zenith_angle(int fecha, char* site){
    //variables temporales
    char stringtemp[150];
    char obs_local_level[150]="";
    char fecha_str[300];
    char fecha_str2[300];
    struct tm *f;
    time_t t;

    t = (int) fecha;
    f = localtime(&t);
    strftime(fecha_str, sizeof(fecha_str), "%Y-%m-%d %H:%M:%S (UTC)", f);
    //Esto lo hacemos para ver el signo del angulo de fase
    t = (int) fecha+30;
    f = localtime(&t);
    strftime(fecha_str2, sizeof(fecha_str2), "%Y-%m-%d %H:%M:%S (UTC)", f);
    //variables para guardar las opciones

    //cargamos las efemerides para obtener todos los angulos
    kclear_c();
    furnsh_c("/opt/efemerides/EarthStations.tf");
    furnsh_c("/opt/efemerides/EarthStations.bsp");
    furnsh_c("/opt/efemerides/earth_070425_370426_predict.bpc");
    furnsh_c("/opt/efemerides/naif0011.tls");
    furnsh_c("/opt/efemerides/spk_de405.bsp");
    furnsh_c("/opt/efemerides/de421.bsp");
    furnsh_c("/opt/efemerides/pck00010.tpc");
    furnsh_c("/opt/efemerides/earth_latest_high_prec.bpc");
    furnsh_c("/opt/efemerides/moon_080317.tf");
    furnsh_c("/opt/efemerides/moon_pa_de421_1900-2050.bpc");

    //sprintf(obs_local_level,"");
    strcat(obs_local_level,site);
    sprintf(stringtemp,"_LOCAL_LEVEL");
    strcat(obs_local_level,stringtemp);

    //llamamos a las funciones necesarias
    angulos_astronomicos(fecha_str, fecha_str2,site,obs_local_level, &earthmoon_distance, &sunmoon_distance, &cenital, &acimutal, &angulo_fase, &phi_obs, &tita_obs,&phi_sun, &signo_fase);

    earthsun_distance = earth_sun_distance(f);

#if defined COMPARISION
    printf("%s\t%lf\t%lf\t%lf\t%lf\n", fecha_str, earthsun_distance,earthmoon_distance,sunmoon_distance,angulo_fase * (180 / 3.14159265358979323846));
#else
    printf("****************************************************\n");
    printf("*    Date observation    = %s\n", fecha_str);
    printf("*    Earth-Sun distance  = %lf\n", earthsun_distance);
    printf("*    Earth-Moon distance = %lf\n", earthmoon_distance);
    printf("*    Moon-Sun distance   = %lf\n", sunmoon_distance);
    printf("*    Moon phase angle    = %lf\n", angulo_fase * (180 / 3.14159265358979323846));
    printf("****************************************************\n");
#endif

    return 1;
}

long date_to_epoch(char timeString[80]){
    time_t epoch;
    struct tm my_tm;
    char buffer[80];

    memset(&my_tm, 0, sizeof(my_tm));
    if (sscanf(timeString, "%d %d %d %d %d %d", &my_tm.tm_year, &my_tm.tm_mon, &my_tm.tm_mday, &my_tm.tm_hour, &my_tm.tm_min, &my_tm.tm_sec) != 6)
    {
        /* ... error parsing ... */
        printf(" sscanf failed");
        return 1;
    }
    my_tm.tm_isdst = -1;
    my_tm.tm_year -= 1900;
    my_tm.tm_mon -= 1;

    epoch = mktime(&my_tm);

    strftime(buffer, sizeof(buffer), "%c", &my_tm);
    //printf("%s  (epoch=%ld)\n", buffer, (long)epoch);

    return((long)epoch);
}


int main (int argc, char *argv[]){

    long date_epoch = date_to_epoch("2022 01 27 0 8 22");

#if defined COMPARISION
    for (size_t i=0; i<17520;i++) {
        moon_zenith_angle(date_epoch+i*30*60, "Valladolid");
    }
#else
    moon_zenith_angle(date_epoch, "Valladolid");
#endif
}