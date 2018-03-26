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
#include "io.h"
#include "calc_distances.h"
#include "read_config.h"
#include "parse_args.h"
namespace fs = std::experimental::filesystem;
using namespace std;

/*
 * Positional arguments:
 * 	PARAMETER_FILE: The configuration paramter file, with required parameters "ifname",
 * 	"ofdir", "rp_min", "rp_max", "rl_min", and "rl_max"
 *
 * Optional arguments:
 * 	-t, --test: Run the code in test mode
*/

const double FROM_MICRO = 0.000001;

int main(int argc, char* argv[]) {
	size_t nReq = 6;
	const string req_keys[] = {"ifname", "db_file", "rp_min", "rp_max", "rl_min", "rl_max"};
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

	cout << "Reading catalog" << endl;
	start = chrono::steady_clock::now();
	vector<Pos> cat = readCatalog(params["ifname"]);
	stop = chrono::steady_clock::now();
	cout << "Catalog read into 'std::vector<toPos>'" << endl;
	cout << "Time to read catalog from file (sec) = " << (chrono::duration_cast<chrono::microseconds>(stop - start).count()) * FROM_MICRO << endl;
	// Check if we have a second catalog to cross-correlate
	if (params.find("ifname2") != params.end()) {
		// Do cross-correlation
		cout << "Reading catalog 2" << endl;
		start = chrono::steady_clock::now();
		vector<Pos> cat2 = readCatalog(params["ifname"]);
		stop = chrono::steady_clock::now();
		cout << "Catalog 2 read into 'std::vector<toPos>'" << endl;
		cout << "Time to read catalog from file (sec) = " << (chrono::duration_cast<chrono::microseconds>(stop - start).count()) * FROM_MICRO << endl;
		cout << "Getting cross-correlation separations" << endl;
		start = chrono::steady_clock::now();
		get_dist_cross(cat, cat2, params.as_double("rp_min"), params.as_double("rp_max"), params.as_double("rl_min"), params.as_double("rl_max"), params["db_file"]);
		stop = chrono::steady_clock::now();
		cout << "Cross-correlation separations calculated" << endl;
		cout << "Time to calculate cross-correlation separations (sec) = " << (chrono::duration_cast<chrono::microseconds>(stop - start).count()) * FROM_MICRO << endl;
	}
	else {
		// Do auto-correlation
		cout << "Getting auto-correlation separations" << endl;
		start = chrono::steady_clock::now();
		get_dist_auto(cat, params.as_double("rp_min"), params.as_double("rp_max"), params.as_double("rl_min"), params.as_double("rl_max"), params["db_file"]);
		stop = chrono::steady_clock::now();
		cout << "Auto-correlation separations calculated" << endl;
		cout << "Time to calculate auto-correlation separations (sec) = " << (chrono::duration_cast<chrono::microseconds>(stop - start).count()) * FROM_MICRO << endl;
	}
	return 0;
}
