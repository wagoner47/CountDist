#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <experimental/filesystem>
#include "io.h"
#include "calc_distances.h"
namespace fs = std::experimental::filesystem;
using namespace std;

const double DELTAP = ioconstants::DELTAP;
const double DELTAL = ioconstants::DELTAL;

vector<Pos> readCatalog(string fname) {
	double ra, dec, rt, ro;
	size_t idx = 0;
	vector<Pos> pos;
	ifstream fin(fname, ifstream::in);
	string line;
	while (getline(fin, line)) {
		if (line.substr(0, 1).compare("#") != 0) {
			istringstream iss(line);
			iss >> ra >> dec >> rt >> ro;
			Pos posi(ra, dec, rt, ro, idx);
			pos.push_back(posi);
			idx++;
		}
	}
	fin.close();
	return pos;
}

void writeFnames(string fname, vector<string> temp_names) {
	ofstream fout(fname, ofstream::out);
	fout << temp_names[0];
	for (vector<string>::iterator it = next(temp_names.begin(), 1); it != temp_names.end(); it++) {
		fout << endl << *it;
	}
	fout.close();
}
