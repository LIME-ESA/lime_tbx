#ifndef read_gsics_moon_h
#define read_gsics_moon_h

#include <iostream>
#include <fstream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <netcdf>
#include <vector>

#include "rsut/delimited_file.hpp"

using namespace netCDF;
using namespace netCDF::exceptions;


class read_gsics_moon
{

   

public:

    void set_input_location(std::string input_location) {

        this->input_location = input_location;
    }
	
    void set_output_location(std::string output_location) {
        this->output_location = output_location;
    }

	void read(){
		
		
        NcFile netcdfFile(input_location,NcFile::read);

        this->vars =  netcdfFile.getVars();
        
        
        vector<double> date;
        vector<double> phase;
        vector<double> obs_irradiance;
        vector<double> rolo_irradiance;
        vector<double> sun_selen_lat;
        vector<double> sun_selen_lon;
        vector<double> sat_selen_lat;
        vector<double> sat_selen_lon;
        vector<vector<double> > moon_pos(3);
        vector<vector<double> > sun_pos(3);
        vector<vector<double> > sat_pos(3);

        int obs_id = 3;

        get_var("date", date);

        get_var("phase_ang", phase);

        get_var("irr_obs", obs_irradiance, obs_id, 4);
        get_var("irr_rolo", rolo_irradiance, obs_id, 4);

        get_var("sun_selen_lat",sun_selen_lat);
        get_var("sun_selen_lon",sun_selen_lon);

        get_var("sat_selen_lon", sat_selen_lon);
        get_var("sat_selen_lat", sat_selen_lat);

        for (size_t i = 0; i < moon_pos.size(); i++) {
            get_var("moon_pos", moon_pos[i],(int)i,3);
        }


        for (size_t i = 0; i < sun_pos.size(); i++) {
            get_var("sun_pos", sun_pos[i], (int)i, 3);
        }

        for (size_t i = 0; i < sat_pos.size(); i++) {
            get_var("sat_pos", sat_pos[i], (int)i, 3);
        }




        double fillvalue = -999.0;


        ofstream output(this->output_location);

        

        for (size_t i = 0; i < obs_irradiance.size(); i++) {

            if ( obs_irradiance[i] != fillvalue ) {

                
                output << date[i] << "," << obs_irradiance[i] << "," << rolo_irradiance[i] << "," << phase[i] << "," << sun_selen_lon[i] << "," 
                    << sun_selen_lat[i] << "," << sat_selen_lon[i] << "," << sat_selen_lat[i] << endl;


            }

        }




		
	}


private:

    
    void get_var(std::string name, vector<double>& var,int dim = 0, int dims=1) {

        multimap<string, netCDF::NcVar>::iterator it = vars.find(name);

        int dim_count = dims;
        int count = it->second.getDim(0).getSize();
        vector<double> vec(count*dim_count);
        it->second.getVar(&vec[0]);
        vector<double> out(count);

        size_t dim_select = 0;
        size_t tot_select = 0;

        for (size_t i = 0; i < vec.size(); i++) {

            if (dim_select == dim) {
                out[tot_select] = vec[i];
                tot_select++;
            }

            if (dim_select == (dim_count - 1)) {
                dim_select = 0;
            }
            else {
                dim_select++;
            }

        }



        var = out;

    }

    multimap<string, netCDF::NcVar> vars;

    std::string input_location;
    std::string output_location;
};



#endif //read_gsics_moon_h