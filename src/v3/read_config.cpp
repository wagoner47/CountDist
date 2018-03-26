#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <unordered_map>
#include <list>
#include "read_config.h"
using namespace std;

configuration::data reader(const string& fname) {
	ifstream pfile(fname, ifstream::in);
	configuration::data params;
	pfile >> params;
	pfile.close();
	return params;
}
