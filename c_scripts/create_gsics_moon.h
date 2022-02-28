#ifndef create_gsics_moon_h
#define create_gsics_moon_h

#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <netcdf>
#include <vector>
using namespace std;
using namespace netCDF;
using namespace netCDF::exceptions;


class create_gsics_moon
{

private:
    
    struct MoonDetails {

    string band_id;


    std::vector<double> irradiance;
    double julianday;
    double pos_x;
    double pos_y;
    double pos_z;

    void add_irradiance(double irradiance) {
        this->irradiance.push_back(irradiance);
    }


	void set_dummy_irradiance(size_t bands) {
		irradiance.clear();
		for (size_t b = 0; b < bands; b++) {
			irradiance.push_back(0.001);
		}
	}


    /*   double  disc_reflectance ;
    double  lunarmodel_reflectance ;
    double  total_radiance;
    int     start_col;
    int     stop_col;
    int     start_row;
    int     stop_row;
    double  pixel_solid_angle ;
    double  sub_earth_lat ;
    double  sub_earth_lon ;
    double  sel_pos_sun_lat ;
    double  sel_pos_sun_lon ;*/
};



public:

    

    void set_output_folder(string output_folder) {
        this->output_folder = output_folder;
    }
    
    string get_date_string(double julianday)
    {
        string date;

        double  theDay, theMonth, theYear, theDummy;

        int Z = (int)(floor(julianday - 1721118.5));
        double   R = julianday - 1721118.5 - Z;
        double   G = Z - 0.25;
        int A = (int)(floor(G / 36524.25));
        int B = A - ((int)(floor(A / 4.0)));
        theYear = (floor((B + G) / 365.25));
        int C = B + Z - ((int)(floor(365.25 * theYear)));
        theDummy = ((5.0 * C + 456.0) / 153.0);
        modf(theDummy, &theMonth);
        theDummy = ((153.0 * theMonth - 457.0) / 5.0);
        modf(theDummy, &theDay);
        theDay = C - theDay + R;
        if (theMonth > 12)
        {
            theYear = theYear + 1;
            theMonth = theMonth - 12;
        }

        int outDay = (int)(theDay);
        int outMonth = (int)(theMonth);
        int outYear = (int)(theYear);

        double  theHours = (theDay - outDay)*24.0;
        int outHour = (int)(floor(theHours));

        double  theMinutes = (theHours - outHour)*60.0;
        int outMinute = (int)(floor(theMinutes));

        double  theSeconds = (theMinutes - outMinute)*60.0;
        int outSec = (int)(floor(theSeconds));

        double outMilliSec = (theSeconds - outSec)*1000.0;


        //20140317083551

        date.resize(14);

        sprintf(&date[0], "%4d%02d%02d%02d%02d%02d", outYear, outMonth, outDay, outHour, outMinute, outSec);




        return date;
    }


    int run(string filename)
    {

        try
        {
            this->n_channels = this->ch_names.size();
            vector<MoonDetails> moondetails;

            string file_location = filename;

            cout << "Opening inputfile : " << file_location << endl;

            ifstream in(file_location);


            string line;
            string cell;
            while (getline(in, line))
            {
                MoonDetails moon;

                stringstream linestr(line);


                getline(linestr, cell, ',');
                moon.julianday = atof(cell.c_str());
                getline(linestr, cell, ',');
                moon.pos_x = atof(cell.c_str());
                getline(linestr, cell, ',');
                moon.pos_y = atof(cell.c_str());
                getline(linestr, cell, ',');
                moon.pos_z = atof(cell.c_str());
                
                if (!dummy_irradiance) {
                    while (getline(linestr, cell, ',')) {
                        moon.add_irradiance(atof(cell.c_str()));
                    }
                }
                else {
                    moon.set_dummy_irradiance(n_channels);
                }

				

                moondetails.push_back(moon);
            }
            in.close();


            for (size_t obs = 0; obs < moondetails.size(); obs++)
            {


                const double seconds_per_day = 24.0*60.0*60.0;
                
                size_t channel_str_len = 4;
                size_t date_num = 1;
                size_t sat_ref_strlen = 6;
                size_t sat_xyz = 3;
                size_t ncols = 1;
                size_t nrows = 1;

                double fillval = -999.0;

                



                //int pos = find(ch_names.begin(), ch_names.end(), moondetails[obs].band_id) - ch_names.begin();


                
                string file_name = this->filename_prefix;
                file_name.append(get_date_string(moondetails[obs].julianday));
                file_name.append("_01.nc");

                // first create ncdf file
                NcFile ncFile(this->output_folder + "/" +file_name, NcFile::replace, NcFile::nc4);

                // create dimensions;

                vector<NcDim> dims;
                dims.push_back(ncFile.addDim("chan", n_channels));
                dims.push_back(ncFile.addDim("chan_strlen", channel_str_len));
                dims.push_back(ncFile.addDim("date", date_num));
                dims.push_back(ncFile.addDim("sat_ref_strlen", sat_ref_strlen));
                dims.push_back(ncFile.addDim("sat_xyz", sat_xyz));
                dims.push_back(ncFile.addDim("col", ncols));
                dims.push_back(ncFile.addDim("row", nrows));


                string name = "Conventions";
                string value = "CF-1.6";
                ncFile.putAtt(name, NcType::nc_CHAR, value.length(), &value[0]);

                name = "Metadata_Conventions"; value = "Unidata Dataset Discovery v1.0";
                ncFile.putAtt(name, NcType::nc_CHAR, value.length(), &value[0]);
                name = "standard_name_vocabulary"; value = "CF Standard Name Table (Version 21, 12 January 2013)";
                ncFile.putAtt(name, NcType::nc_CHAR, value.length(), &value[0]);
                name = "project"; value = "Global Space-based Inter-Calibration System <http://gsics.wmo.int>";
                ncFile.putAtt(name, NcType::nc_CHAR, value.length(), &value[0]);
                name = "title"; value = "ESA-CIMEL lunar observation file";
                ncFile.putAtt(name, NcType::nc_CHAR, value.length(), &value[0]);
                name = "summary"; value = "Lunar observation file";
                ncFile.putAtt(name, NcType::nc_CHAR, value.length(), &value[0]);
                name = "keywords"; value = "GSICS, ESA,CIMEL, lunar, moon, observation, visible";
                ncFile.putAtt(name, NcType::nc_CHAR, value.length(), &value[0]);
                name = "references"; value = "TBD";
                ncFile.putAtt(name, NcType::nc_CHAR, value.length(), &value[0]);
                name = "institution"; value = "ESA";
                ncFile.putAtt(name, NcType::nc_CHAR, value.length(), &value[0]);
                name = "license"; value = "This file was produced in support of GSICS activities and thus is not meant for public use although the data is in the public domain. Any publication using this file should acknowledge both GSICS and the relevant organization. Neither the data creator, nor the data publisher, nor any of their employees or contractors, makes any warranty, express or implied, including warranties of merchantability and fitness for a particular purpose, or assumes any legal liability for the accuracy, completeness, or usefulness, of this information. The user of the data do so at their own risk. That there is no support from EUMETSAT related to problems in the course of the data evaluation.";
                ncFile.putAtt(name, NcType::nc_CHAR, value.length(), &value[0]);
                name = "creator_name"; value = "Stefan Adriaensen";
                ncFile.putAtt(name, NcType::nc_CHAR, value.length(), &value[0]);
                name = "creator_email"; value = "stefan.adriaensen@vito.be";
                ncFile.putAtt(name, NcType::nc_CHAR, value.length(), &value[0]);
                name = "creator_url"; value = "http://www.esa.int";
                ncFile.putAtt(name, NcType::nc_CHAR, value.length(), &value[0]);
                name = "naming_authority"; value = "int.esa";
                ncFile.putAtt(name, NcType::nc_CHAR, value.length(), &value[0]);
                name = "instrument'"; value = instrument_name;
                ncFile.putAtt(name, NcType::nc_CHAR, value.length(), &value[0]);
                name = "instrument_wmo_code"; value = "TBD";
                ncFile.putAtt(name, NcType::nc_CHAR, value.length(), &value[0]);
                name = "data_source"; value = data_source;
                ncFile.putAtt(name, NcType::nc_CHAR, value.length(), &value[0]);
                name = "date_created"; value = creation_date;
                ncFile.putAtt(name, NcType::nc_CHAR, value.length(), &value[0]);
                name = "date_modified";
                ncFile.putAtt(name, NcType::nc_CHAR, value.length(), &value[0]);
                name = "history"; value = "TBD";
                ncFile.putAtt(name, NcType::nc_CHAR, value.length(), &value[0]);
                name = "id"; name = "fileName";
                ncFile.putAtt(name, NcType::nc_CHAR, value.length(), &value[0]);
                name = "wmo_data_category"; value = "101";
                ncFile.putAtt(name, NcType::nc_CHAR, value.length(), &value[0]);
                name = "wmo_international_data_subcategory";
                ncFile.putAtt(name, NcType::nc_CHAR, value.length(), &value[0]);
                name = "processing_level"; value = processing_level;
                ncFile.putAtt(name, NcType::nc_CHAR, value.length(), &value[0]);
                name = "doc_url"; value = "N/A";
                ncFile.putAtt(name, NcType::nc_CHAR, value.length(), &value[0]);
                name = "doc_doi";
                ncFile.putAtt(name, NcType::nc_CHAR, value.length(), &value[0]);


                //put compulsory vars




                vector<NcDim> chandims(2);
                chandims[0] = dims[0];
                chandims[1] = dims[1];

                name = "channel_name";
                NcVar ch_name = ncFile.addVar(name, NcType::nc_CHAR, chandims);
                name = "sensor_band_identifier";
                ch_name.putAtt("standard_name", NcType::nc_CHAR, name.size(), &name[0]);
                name = "channel identifier";
                ch_name.putAtt("long_name", NcType::nc_CHAR, name.size(), &name[0]);

                char* mempos = (char*)malloc(n_channels*channel_str_len*sizeof(char));
                char* startpos_pntr = mempos;
                for (size_t ch = 0; ch < n_channels; ch++)
                {
                    for (size_t chstrl = 0; chstrl < ch_names[ch].size(); chstrl++)
                    {
                        *startpos_pntr = ch_names[ch][chstrl];
                        startpos_pntr++;

                    }
                    if (ch_names[ch].size() == 3)
                    {
                        *startpos_pntr = ' ';
                        startpos_pntr++;
                    }
                }


                ch_name.putVar(mempos);



                name = "date";
                NcVar date = ncFile.addVar(name, NcType::nc_DOUBLE, dims[2]);
                name = "standard_name";
                value = "time";
                date.putAtt(name, NcType::nc_CHAR, name.size(), &value[0]);


                name = "sat_pos";

                NcVar satpos = ncFile.addVar(name, NcType::nc_DOUBLE, dims[4]);

                name = "satellite position x y z in sat_pos_ref";
                satpos.putAtt("long_name", NcType::nc_CHAR, name.size(), &name[0]);
                name = "km";
                satpos.putAtt("units", NcType::nc_CHAR, name.size(), &name[0]);
                satpos.putAtt("valid_min", NcType::nc_DOUBLE, 0.0);
                satpos.putAtt("valid_max", NcType::nc_DOUBLE, 1000000000000.00);

                name = "sat_pos_ref";

                NcVar satposref = ncFile.addVar(name, NcType::nc_CHAR, dims[3]);
                name = "reference frame of satellite position";
                satposref.putAtt("long_name", NcType::nc_CHAR, name.size(), &name[0]);

                satposref.putVar("J2000");


                name = "irr_obs";
                NcVar irr_obs = ncFile.addVar(name, NcType::nc_DOUBLE, dims[0]);

                name = "observed lunar irradiance";
                irr_obs.putAtt("long_name", NcType::nc_CHAR, name.size(), &name[0]);
                name = "W m-2 um-1";
                irr_obs.putAtt("units", NcType::nc_CHAR, name.size(), &name[0]);
                irr_obs.putAtt("valid_min", NcType::nc_DOUBLE, 0.0);
                irr_obs.putAtt("valid_max", NcType::nc_DOUBLE, 1000000000000.00);


                vector<double> obs_irradiance(n_channels);

                for (size_t ch = 0; ch < n_channels; ch++) {
                    obs_irradiance[ch] = moondetails[obs].irradiance[ch];
                }

                

                irr_obs.putVar(&obs_irradiance[0]);


                double moon_center_date = moondetails[obs].julianday;
                moon_center_date = (-2440587.500000 + moon_center_date) * seconds_per_day;
                date.putVar(&moon_center_date);

                vector<double> sat_pos_km;
                sat_pos_km.push_back(moondetails[obs].pos_x);
                sat_pos_km.push_back(moondetails[obs].pos_y);
                sat_pos_km.push_back(moondetails[obs].pos_z);

                satpos.putVar(&sat_pos_km[0]);




                name = "pix_solid_ang";
                NcVar pix_solid_ang = ncFile.addVar(name, NcType::nc_DOUBLE, dims[0]);

                name = "pixel solid angle";
                pix_solid_ang.putAtt("long_name", NcType::nc_CHAR, name.size(), &name[0]);
                name = "sr";
                pix_solid_ang.putAtt("units", NcType::nc_CHAR, name.size(), &name[0]);
                double dmin = 0.0;
                double dmax = 1000000.0;
                pix_solid_ang.putAtt("valid_min", NcType::nc_DOUBLE, dmin);
                pix_solid_ang.putAtt("valid_max", NcType::nc_DOUBLE, dmax);
                pix_solid_ang.putAtt("_FillValue", NcType::nc_DOUBLE, fillval);

                vector<double> pxsa(n_channels);
                pxsa[0] = fillval;
                pxsa[1] = fillval;
                pxsa[2] = fillval;
                pxsa[3] = fillval;

                pix_solid_ang.putVar(&pxsa[0]);

                name = "ovrsamp_fa";
                NcVar ovrsamp_fa = ncFile.addVar(name, NcType::nc_DOUBLE, dims[0]);

                name = "oversampling factor";
                ovrsamp_fa.putAtt("long_name", NcType::nc_CHAR, name.size(), &name[0]);
                name = "1";
                ovrsamp_fa.putAtt("units", NcType::nc_CHAR, name.size(), &name[0]);
                ovrsamp_fa.putAtt("valid_min", NcType::nc_DOUBLE, dmin);
                ovrsamp_fa.putAtt("valid_max", NcType::nc_DOUBLE, dmax);
                ovrsamp_fa.putAtt("_FillValue", NcType::nc_DOUBLE, fillval);

                vector<double> osf(n_channels);
                osf[0] = fillval;
                osf[1] = fillval;
                osf[2] = fillval;
                osf[3] = fillval;

                ovrsamp_fa.putVar(&osf[0]);

                name = "dc_obs";
                NcVar dc_obs = ncFile.addVar(name, NcType::nc_INT64, dims[0]);

                name = "integrated digital counts of lunar observation";
                dc_obs.putAtt("long_name", NcType::nc_CHAR, name.size(), &name[0]);
                name = "1";
                dc_obs.putAtt("units", NcType::nc_CHAR, name.size(), &name[0]);
                long min = 0;
                long max = 2147483647;
                dc_obs.putAtt("valid_min", NcType::nc_INT64, min);
                dc_obs.putAtt("valid_max", NcType::nc_INT64, max);
                dc_obs.putAtt("_FillValue", NcType::nc_INT64, -999);

                vector<long> dc(n_channels);
                dc[0] = -999;
                dc[1] = -999;
                dc[2] = -999;
                dc[3] = -999;


                dc_obs.putVar(&dc[0]);



                name = "dc_obs_offset";
                NcVar dc_obs_offset = ncFile.addVar(name, NcType::nc_DOUBLE, dims[0]);

                name = "averaged digital counts offset of deep space";
                dc_obs_offset.putAtt("long_name", NcType::nc_CHAR, name.size(), &name[0]);
                name = "1";
                dc_obs_offset.putAtt("units", NcType::nc_CHAR, name.size(), &name[0]);
                dc_obs_offset.putAtt("valid_min", NcType::nc_DOUBLE, dmin);
                dc_obs_offset.putAtt("valid_max", NcType::nc_DOUBLE, dmax);
                dc_obs_offset.putAtt("_FillValue", NcType::nc_DOUBLE, fillval);


                vector<double> dc_count(n_channels);

                dc_count[0] = fillval;
                dc_count[1] = fillval;
                dc_count[2] = fillval;
                dc_count[3] = fillval;


                dc_obs_offset.putVar(&dc_count[0]);


                name = "moon_pix_num";
                NcVar moon_pix_num = ncFile.addVar(name, NcType::nc_INT64, dims[0]);

                name = "integrated digital counts of lunar obserevation";
                moon_pix_num.putAtt("long_name", NcType::nc_CHAR, name.size(), &name[0]);
                name = "1";
                moon_pix_num.putAtt("units", NcType::nc_CHAR, name.size(), &name[0]);
                moon_pix_num.putAtt("valid_min", NcType::nc_INT64, min);
                moon_pix_num.putAtt("valid_max", NcType::nc_INT64, max);
                moon_pix_num.putAtt("_FillValue", NcType::nc_INT64, -999);

                vector<long> pxn(n_channels);
                pxn[0] = -999;
                pxn[1] = -999;
                pxn[2] = -999;
                pxn[3] = -999;


                moon_pix_num.putVar(&pxn[0]);

                name = "moon_pix_thld";
                NcVar moon_pix_thld = ncFile.addVar(name, NcType::nc_INT64, dims[0]);

                name = "digital counts threshold for moon masking";
                moon_pix_thld.putAtt("long_name", NcType::nc_CHAR, name.size(), &name[0]);
                name = "1";
                moon_pix_thld.putAtt("units", NcType::nc_CHAR, name.size(), &name[0]);
                moon_pix_thld.putAtt("valid_min", NcType::nc_INT64, min);
                moon_pix_thld.putAtt("valid_max", NcType::nc_INT64, max);
                moon_pix_thld.putAtt("_FillValue", NcType::nc_INT64, -999);

                vector<long> pxthd(n_channels);
                pxthd[0] = -999;
                pxthd[1] = -999;
                pxthd[2] = -999;
                pxthd[3] = -999;

                moon_pix_thld.putVar(&pxthd[0]);


                // real imagette 
                vector<NcDim> dims_imgt(3);
                // !! row col chan
                dims_imgt[0] = dims[6];
                dims_imgt[1] = dims[5];
                dims_imgt[2] = dims[0];

                name = "rad_obs_imgt";
                NcVar rad_obs_imgt = ncFile.addVar(name, NcType::nc_DOUBLE, dims_imgt);

                name = "observed lunar radiance imagette";
                rad_obs_imgt.putAtt("long_name", NcType::nc_CHAR, name.size(), &name[0]);
                name = "W sr-1 m-2 um-1";
                rad_obs_imgt.putAtt("units", NcType::nc_CHAR, name.size(), &name[0]);
                rad_obs_imgt.putAtt("valid_min", NcType::nc_DOUBLE, dmin);
                rad_obs_imgt.putAtt("valid_max", NcType::nc_DOUBLE, dmax);
                rad_obs_imgt.putAtt("_FillValue", NcType::nc_DOUBLE, fillval);


                size_t alloc_size_double = nrows*ncols*n_channels*sizeof(double);
                double* radiance_imgt = (double*)malloc(alloc_size_double);
                double* start_p = radiance_imgt;
                for (size_t i = 0; i < alloc_size_double / sizeof(double); i++)
                {
                    *start_p = fillval;
                    start_p++;
                }

                rad_obs_imgt.putVar(radiance_imgt);

                name = "dc_obs_imgt";
                NcVar dc_obs_imgt = ncFile.addVar(name, NcType::nc_INT64, dims_imgt);

                name = "observed moon digital counts imagette";
                dc_obs_imgt.putAtt("long_name", NcType::nc_CHAR, name.size(), &name[0]);
                name = "W sr-1 m-2 um-1";
                dc_obs_imgt.putAtt("units", NcType::nc_CHAR, name.size(), &name[0]);
                dc_obs_imgt.putAtt("valid_min", NcType::nc_INT64, min);
                dc_obs_imgt.putAtt("valid_max", NcType::nc_INT64, max);
                dc_obs_imgt.putAtt("_FillValue", NcType::nc_INT64, -999);

                size_t alloc_size_long = nrows*ncols*n_channels*sizeof(long);
                long* dc_imgt = (long*)malloc(alloc_size_long);
                long* start_p_long = dc_imgt;


                for (size_t i = 0; i < alloc_size_long / sizeof(long); i++)
                {
                    *start_p_long = -999;
                    start_p_long++;
                }

                dc_obs_imgt.putVar(dc_imgt);
                

            }

        }
        catch (NcException& e)
        {
            cout << "unknown error" << endl;
            e.what();
            return -1;
        }
        return 0;
    }


    void set_channel_names(vector<string> channel_names) {
        this->ch_names = channel_names;
    }

    void set_dummy_irradiance(bool d_irradiance) {
        this->dummy_irradiance = d_irradiance;
    }

    void set_filename_prefix(string filename_prefix) {
        this->filename_prefix = filename_prefix;
    }


    void set_instrument_name(string instrument_name) {
        this->instrument_name = instrument_name;
    }

    void set_data_source(string data_source) {
        this->data_source = data_source;
    }

    void set_creation_date(string creation_date) {
        this->creation_date = creation_date;
    }

    void set_processing_level(string processing_level) {
        this->processing_level = processing_level;
    }

    private:

        bool            dummy_irradiance;
        size_t          n_channels;
        string          filename_prefix;
        vector<string>  ch_names;
        
        string          output_folder;

        string instrument_name;
        string data_source;
        string creation_date;
        string processing_level;
};



#endif //create_gsics_moon_h