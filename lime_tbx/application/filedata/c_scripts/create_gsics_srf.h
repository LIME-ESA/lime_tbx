#ifndef create_gsics_srf_h
#define create_gsics_srf_h

#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <netcdf>
#include <vector>

#include "rsut/delimited_file.hpp"

using namespace std;
using namespace netCDF;
using namespace netCDF::exceptions;

bool ignore_value_compare(double a, double b) {


    if (a != -9999.0 && b != -9999.0 && a < b) {
        return true;
    }

    return false;

}


class create_gsics_srf
{


private:




public:

    void set_input_file_location(string input_file_location) {
        this->input_file_location= input_file_location;
    }

    void set_sensor_name(string sensor_name) {
        this->sensor_name = sensor_name;
    }

    void set_output_filename() {
        this->output_file_name = get_filename();
    }


    void set_folder_name(std::string folder) {
        this->output_folder = folder;
    }

    void add_channel(double channel_micro) {

        this->channel_microm.push_back(channel_micro);
        this->channel_id.push_back("band_" + std::to_string(this->channel_microm.size()));
        this->channel_origin.push_back(1);

    }


    void set_bands() {
        /*move to class*/
        this->add_channel(0.440);
        this->add_channel(0.500);
        this->add_channel(0.675);
        this->add_channel(0.870);
        this->add_channel(1.020);
        this->add_channel(1.640);
    }


    void read_srf() {


        rsut::delimited_file file(this->input_file_location,",",false);

        size_t cols = file.cols();
        this->nr_of_channels = cols / 2;
        this->nr_of_rows = file.rows();


        
        vector<double> srf(nr_of_channels*nr_of_rows,this->fill_value);
        vector<double> wavelength_micro(nr_of_channels * nr_of_rows, this->fill_value);
        vector<double> wavenumber(nr_of_channels * nr_of_rows, this->fill_value);

        

        for (size_t r = 0; r < nr_of_rows; r++) {
            for (size_t c = 0; c < nr_of_channels; c++) {

                double wl = file.get_as<double>(r, 2 * c) / 1000.0;
                double resp = file.get_as<double>(r, 2 * c + 1);
                if (resp < 0.0) resp = 0.0;

                if (wl > 0.0) {

                    wavelength_micro[c + r * nr_of_channels] = wl;
                    srf[c + r * nr_of_channels] = resp;
                    wavenumber[c + r * nr_of_channels] = 10000.0 / wl;
                }
            }
        }
        

        this->srf = srf;
        this->wavelength_micro = wavelength_micro;
        this->wavenumber_cm = wavenumber;
 
        set_bands();

    }


    void nc_file() {


        this->fill_value = -9999.0;

        read_srf();


        set_output_filename();
        string name = this->output_folder + "/" + this->output_file_name;


        NcFile nc_file(name, NcFile::replace, NcFile::nc4);
        
        fill_att_header(nc_file);


        

        NcDim dim_channels = nc_file.addDim("channel", this->nr_of_channels);
        NcDim dim_samples = nc_file.addDim("sample", this->nr_of_rows);

        NcVar channel = nc_file.addVar("channel", NcType::nc_DOUBLE, dim_channels);

        channel.putVar(&this->channel_microm[0]);
        // att for channel
        string txt = "nominal channel central wavelength";
        channel.putAtt("long_name", NcType::nc_CHAR, txt.size(),&txt[0]);
        txt = "micrometer";
        channel.putAtt("units", NcType::nc_CHAR, txt.size(), &txt[0]);
        double min = *min_element(this->channel_microm.begin(),this->channel_microm.end());
        double max = *max_element(this->channel_microm.begin(), this->channel_microm.end());

        channel.putAtt("valid_min", NcType::nc_DOUBLE, 1, &min);
        channel.putAtt("valid_max", NcType::nc_DOUBLE, 1, &max);
        
        
        NcDim dim = nc_file.addDim("channel_id", this->channel_id.size());


        NcVar ch_id = nc_file.addVar("channel_id", NcType::nc_STRING, dim);


        /*
        */ 
        vector<size_t> id(1,0);
        ch_id.putVar(id, this->channel_id[0]);
        id[0] = 1;
        ch_id.putVar(id, this->channel_id[1]);
        id[0] = 2;
        ch_id.putVar(id, this->channel_id[2]);
        id[0] = 3;
        ch_id.putVar(id, this->channel_id[3]);
        id[0] = 4;
        ch_id.putVar(id, this->channel_id[4]);
        id[0] = 5;
        ch_id.putVar(id, this->channel_id[5]);

        
        
        txt = "channel identifier";
        ch_id.putAtt("long_name", txt);
        txt = "sensor_band_identifier";
        ch_id.putAtt("standard_name", txt);

        NcVar origin = nc_file.addVar("origin", NcType::nc_UINT, dim);

        origin.putVar(&channel_origin[0]);
        origin.putAtt("flag_meanings", "wavelength wavenumber");
        vector<uint8_t> values;
        values.push_back(1);
        values.push_back(2);

        origin.putAtt("flag_values", NcType::nc_UBYTE,2, &values[0]);
        origin.putAtt("long name", "original sample domain");
        origin.putAtt("valid max", NcType::nc_INT64, 1);
        origin.putAtt("valid min", NcType::nc_INT64, 1);


        double min_v = 0.0;
        double max_v = 0.0;

        vector<NcDim> srf_dims(2);

        srf_dims[0] = dim_samples;
        srf_dims[1] = dim_channels;

        NcVar srf = nc_file.addVar("srf", NcType::nc_DOUBLE, srf_dims);

        srf.putVar(&this->srf[0]);


        srf.putAtt("long_name", "normalized spectral response");
        
        min_v = *std::min_element(this->srf.begin(), this->srf.end(), ignore_value_compare);
        max_v = *std::max_element(this->srf.begin(), this->srf.end(), ignore_value_compare);
        
        srf.putAtt("valid_max", NcType::nc_DOUBLE,max_v);
        srf.putAtt("valid_min", NcType::nc_DOUBLE, min_v);
        
        NcVar wavelength = nc_file.addVar("wavelength", NcType::nc_DOUBLE, srf_dims);
        wavelength.putAtt("long_name", "wavelength");
        wavelength.putAtt("ancillary_variables", "origin");
        wavelength.putAtt("units", "um");
        wavelength.putAtt("_FillValue", NcType::nc_DOUBLE, this->fill_value);

        min_v = *std::min_element(this->wavelength_micro.begin(), this->wavelength_micro.end(), ignore_value_compare);
        max_v = *std::max_element(this->wavelength_micro.begin(), this->wavelength_micro.end(), ignore_value_compare);

        wavelength.putAtt("valid_max", NcType::nc_DOUBLE, max_v);
        wavelength.putAtt("valid_min", NcType::nc_DOUBLE, min_v);
        wavelength.putVar(&this->wavelength_micro[0]);

        NcVar wavenumber = nc_file.addVar("wavenumber", NcType::nc_DOUBLE, srf_dims);
        wavenumber.putAtt("long_name", "wavenumber");
        wavenumber.putAtt("ancillary_variables", "origin");
        wavenumber.putAtt("comment", "values are in descending order");
        wavenumber.putAtt("units", "cm-1");
        wavenumber.putAtt("_FillValue", NcType::nc_DOUBLE, this->fill_value);


        min_v = *std::min_element(this->wavenumber_cm.begin(), this->wavenumber_cm.end(), ignore_value_compare);
        max_v = *std::max_element(this->wavenumber_cm.begin(), this->wavenumber_cm.end(), ignore_value_compare);

        wavenumber.putAtt("valid_max", NcType::nc_DOUBLE, max_v);
        wavenumber.putAtt("valid_min", NcType::nc_DOUBLE, min_v);

        wavenumber.putVar(&this->wavenumber_cm[0]);


    }


    void fill_att_header(NcFile& nc_file) {


        set_att_str(nc_file, "Conventions", "CF-1.6");
        set_att_str(nc_file, "Metadata_Conventions", "Unidata Dataset Discovery v1.0");
        set_att_str(nc_file, "creator_email", "stefan.adriaensen@vito.be");
        set_att_str(nc_file, "creator_url", "http://www.vito.be");
        set_att_str(nc_file, "date_modified", "2020-11-26T19:00:00Z");
        set_att_str(nc_file, "hirstory", "2020-11-26T19:02:00Z create_gsics_srf.hpp v1.0");
        set_att_str(nc_file, "id", get_filename());

        set_att_str(nc_file, "institution", "European Space Agency");
        set_att_str(nc_file, "instrument", "CE318-TP9");
        set_att_str(nc_file, "license", "This file was produced in support of GSICS activities and thus is not meant for public use although the data is in the public domain. Any publication using this file should acknowledge both GSICS and the data's relevant organization. Neither the data creator, nor the data publisher, nor any of their employees or contractors, makes any warranty, express or implied, including warranties of merchantability and fitness for a particular purpose, or assumes any legal liability for the accuracy, completeness, or usefulness, of this information.");
        set_att_str(nc_file, "naming_authority", "gsics");
        set_att_str(nc_file, "platform", "Izana METEO institute Tenerife Spain");
        set_att_str(nc_file, "project", "Lunar Irradiance Measurements and Modelling for EO");
        set_att_str(nc_file, "publisher_email", "marc.bouvet@esa.int");
        set_att_str(nc_file, "publisher_name", "ESA");
        set_att_str(nc_file, "publisher_url", "http://www.esa.int");
        set_att_str(nc_file, "source", "repos/lunar_model/data/responses/responses_1088_13112020.txt");
        set_att_str(nc_file, "standard_name_vocabulary", "CF Standard Name Table (v24, 27 June 2013)");
        set_att_str(nc_file, "summary", "Normalized spectral response functions (SRF) for all channels of the ESA Lunar Model instrument CIMEL TP 286 - deployed at Izana with nr 1088, are stored in this file.");
        set_att_str(nc_file, "title", "ESA Lunar Irradiance Measurements - instrument 1088 ");
        set_att_int(nc_file, "wmo_satellite_code", -9999);
        set_att_int(nc_file, "wmo_satellite_instrument_code", -9999);
    }



    void set_att_str(NcFile& ncfile, string name, string value) {

        ncfile.putAtt(name, NcType::nc_CHAR, value.length(), &value[0]);

    }
    void set_att_int(NcFile& ncfile, string name, int64_t value) {

        ncfile.putAtt(name, NcType::nc_INT64,  value);

    }
    string get_filename() {

        string filename = "" ;

        // example filename from EUM "W_XX - EUMETSAT - Darmstadt, VIS + IR + SRF, MSG4 + SEVIRI_C_EUMG.nc"
        
        filename += "W_XX-ESA-Noordwijk,VIS+NIR+SWIR,CIMEL_1088.nc";


        return filename;


    }

    


private:

    string input_file_location;
    string sensor_name;
    string output_file_name;
    string output_folder;

    size_t nr_of_channels;
    size_t nr_of_rows;

    vector<double> channel_microm;
    vector<size_t> channel_origin;
    vector<string> channel_id;

    vector<double> srf;
    vector<double> wavelength_micro;
    vector<double> wavenumber_cm;
    double fill_value;


};



#endif //create_gsics_srf_h