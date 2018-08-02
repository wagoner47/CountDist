#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <tuple>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <argp.h>
#include <experimental/filesystem>
#include <ratio>
#include "io.h"
#include "calc_distances.h"
#include "read_config.h"
#include "parse_args.h"
namespace fs = std::experimental::filesystem;
using namespace std;

/*
 * Positional arguments:
 * 	PARAMETER_FILE: The configuration paramter file, with required parameters "ifname1", "ifname2",
 * 	"ofdir", "rp_min", "rp_max", "rl_min", "rl_max", "has_true1", "has_true2", "has_obs1", "has_obs2",
 * 	"use_true", "use_obs", and "is_auto"
 *
 * Optional arguments:
 * 	-t, --test: Run the code in test mode
*/

const double FROM_MICRO = 0.000001;

int main(int argc, char* argv[]) {
	size_t nReq = 15;
	const string req_keys[] = {"ifname1", "ifname2", "db_file", "table_name", "rp_min", "rp_max", "rl_min", "rl_max", "has_true1", "has_true2", "has_obs1", "has_obs2", "use_true", "use_obs", "is_auto"};
	chrono::steady_clock::time_point start, stop;
	chrono::microseconds exec_time;

	// Create a holder for command line arguments
	arguments arguments;
	// Set default values
	arguments.test = false;
	// Parse the arguments
	argp_parse(&argp, argc, argv, 0, 0, &arguments);

	// Read the parameter file and check for the keys we need
	configuration::data params = reader(arguments.args[0]);
	for (size_t i = 0; i < nReq; i++) {
		if (!params.iskey(req_keys[i])) {
			cerr << "Missing required parameter '" << req_keys[i] << "'" << endl;
			return 1;
		}
	}

	cout << "Reading catalog 1 into 'std::vector<Pos>'...";
	start = chrono::steady_clock::now();
	vector<Pos> cat1 = readCatalog(params["ifname1"], params.as_bool("use_true"), params.as_bool("use_obs"), params.as_bool("has_true1"), params.as_bool("has_obs1"));
	stop = chrono::steady_clock::now();
	cout << "[done: " << (chrono::duration<double, ratio<1>>(stop - start)).count() << " sec]" << endl;

	cout << "Reading catalog 2 into 'std::vector<Pos>'...";
	start = chrono::steady_clock::now();
	vector<Pos> cat2 = readCatalog(params["ifname2"], params.as_bool("use_true"), params.as_bool("use_obs"), params.as_bool("has_true2"), params.as_bool("has_obs2"));
	stop = chrono::steady_clock::now();
	cout << "[done: " << (chrono::duration<double, ratio<1>>(stop - start)).count() << " sec]" << endl;

	cout << "Getting separations...";
	start = chrono::steady_clock::now();
	get_dist(cat1, cat2, params.as_double("rp_min"), params.as_double("rp_max"), params.as_double("rl_min"), params.as_double("rl_max"), params["db_file"], params["table_name"], params.as_bool("use_true"), params.as_bool("use_obs"), params.as_bool("is_auto"));
	stop = chrono::steady_clock::now();
	cout << "[done: " << (chrono::duration<double, ratio<1>>(stop - start)).count() << " sec]" << endl;

	return 0;
}
