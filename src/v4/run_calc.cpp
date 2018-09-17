#include <iostream>
#include <fstream>
#include <streambuf>
#include <string>
#include <sstream>
#include <tuple>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <argp.h>
#include <experimental/filesystem>
#include <ratio>
#include <array>
#include "logging.h"
#include "io.h"
#include "calc_distances.h"
#include "read_config.h"
#include "parse_args.h"
using namespace std;

severity_level LOG_LEVEL;
/*
 * Positional arguments:
 * 	PARAMETER_FILE: The configuration paramter file, with required parameters
 * 	"ifname1", "ifname2", "db_file", "table_name", "rp_min", "rp_max", "rl_min",
 * 	"rl_max", "has_true1", "has_true2", "has_obs1", "has_obs2", "use_true",
 * 	"use_obs", "is_auto", "SIGMA_R_EFF1", "Z_EFF1", "SIGMA_Z1", "SIGMA_R_EFF2",
 * 	"Z_EFF2", "SIGMA_Z2", "meta_name1", "meta_name2"
 *
 * Optional arguments:
 * 	-t, --test: Run the code in test mode
*/

int main(int argc, char* argv[]) {
	const auto str_keys = array_of<string>("ifname1", "ifname2", "db_file", "table_name", "meta_name1", "meta_name2");
	const auto dbl_keys = array_of<string>("rp_min", "rp_max", "rl_min", "rl_max", "Z_EFF1", "SIGMA_Z1", "SIGMA_R_EFF1", "Z_EFF2", "SIGMA_Z2", "SIGMA_R_EFF2");
	const auto bool_keys = array_of<string>("has_true1", "has_obs1", "has_true2", "has_obs2", "use_true", "use_obs", "is_auto");
	chrono::steady_clock::time_point start, stop;
	chrono::microseconds exec_time;

	// Create a holder for command line arguments
	arguments args;
	// Set default values
	args.test = false;
	args.level = severity_level::fatal;
	// Parse the arguments
	argp_parse(&argp, argc, argv, 0, 0, &args);
	LOG_LEVEL = args.level;

	// Read the parameter file and check for the keys we need
	configuration::data params = reader(args.args[0]);
	check_req_keys(params, str_keys, dbl_keys, bool_keys);

	LOG(severity_level::info) << "Reading catalog 1 into 'std::vector<Pos>'...";
	//cout << "Reading catalog 1 into 'std::vector<Pos>'...";
	start = chrono::steady_clock::now();
	vector<Pos> cat1(readCatalog(params["ifname1"], params.as_bool("use_true"), params.as_bool("use_obs"), params.as_bool("has_true1"), params.as_bool("has_obs1")));
	stop = chrono::steady_clock::now();
	//cout << "[done: " << (chrono::duration<double, ratio<1>>(stop - start)).count() << " sec]" << endl;
	LOG(severity_level::info) << "Reading catalog 1 into 'std::vector<Pos>'...[done: " << (chrono::duration<double, ratio<1>>(stop - start)).count() << " sec]";

	LOG(severity_level::info) << "Reading catalog 2 into 'std::vector<Pso>'...";
	//cout << "Reading catalog 2 into 'std::vector<Pos>'...";
	start = chrono::steady_clock::now();
	vector<Pos> cat2 = readCatalog(params["ifname2"], params.as_bool("use_true"), params.as_bool("use_obs"), params.as_bool("has_true2"), params.as_bool("has_obs2"));
	stop = chrono::steady_clock::now();
	//cout << "[done: " << (chrono::duration<double, ratio<1>>(stop - start)).count() << " sec]" << endl;
	LOG(severity_level::info) << "Reading catalog 2 into 'std::vector<Pso>'...[done: " << (chrono::duration<double, ratio<1>>(stop - start)).count() << " sec]";

	LOG(severity_level::info) << "Getting separations...";
	//cout << "Getting separations...";
	start = chrono::steady_clock::now();
	tuple<vector<vector<double> >, vector<vector<size_t> > > seps_output = get_separations(cat1, cat2, params.as_double("rp_min"), params.as_double("rp_max"), params.as_double("rl_min"), params.as_double("rl_max"), params.as_bool("use_true"), params.as_bool("use_obs"), params.as_bool("is_auto"));
	stop = chrono::steady_clock::now();
	//cout << "[done: " << (chrono::duration<double, ratio<1>>(stop - start)).count() << " sec]" << endl;
	LOG(severity_level::info) << "Getting separations...[done: " << (chrono::duration<double, ratio<1>>(stop - start)).count() << " sec]";

	LOG(severity_level::info) << "Writing separations to database...";
	start = chrono::steady_clock::now();
	write_separations(get<0>(seps_output), params["db_file"], params["table_name"]);
	stop = chrono::steady_clock::now();
	LOG(severity_level::info) << "Writing separations to database...[done: " << (chrono::duration<double, ratio<1>>(stop - start)).count() << " sec]";

	LOG(severity_level::info) << "Adding meta-data for catalog 1...";
	//cout << "Adding meta-data for catalog 1...";
	start = chrono::steady_clock::now();
	write_meta_data(params["db_file"], params["meta_name1"], params.as_double("Z_EFF1"), params.as_double("SIGMA_R_EFF1"), params.as_double("SIGMA_Z1"));
	stop = chrono::steady_clock::now();
	//cout << "[done: " << (chrono::duration<double, ratio<1>>(stop - start)).count() << " sec]" << endl;
	LOG(severity_level::info) << "Adding meta-data for catalog 1...[done: " << (chrono::duration<double, ratio<1>>(stop - start)).count() << " sec]";

	if (!params.as_bool("is_auto")) {
	    LOG(severity_level::info) << "Adding meta-data for catalog 2...";
	    //cout << "Adding meta-data for catalog 2...";
	    start = chrono::steady_clock::now();
	    write_meta_data(params["db_file"], params["meta_name2"], params.as_double("Z_EFF2"), params.as_double("SIGMA_R_EFF2"), params.as_double("SIGMA_Z2"));
	    stop = chrono::steady_clock::now();
	    //cout << "[done: " << (chrono::duration<double, ratio<1>>(stop - start)).count() << " sec]" << endl;
	    LOG(severity_level::info) << "Adding meta-data for catalog 2...[done: " << (chrono::duration<double, ratio<1>>(stop - start)).count() << " sec]";
	}

	return 0;
}
