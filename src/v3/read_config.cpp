#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <unordered_map>
#include "read_config.h"
using namespace std;

unordered_map<string, string> reader(string pname) {
	ifstream pfile(pname, ifstream::in);
	unorderd_map<string, string> params;
	string line;
	size_t loc;

	while (getline(pfile, line)) {
		if (*line.begin() != "#") {
			loc = line.find("=");
			if (loc != string::npos) {
				params[line.substr(0,loc-1)] = line.substr(loc+1);
			}
			else {
				continue;
			}
		}
		else {
			continue;
		}
	}
	return params;
}
