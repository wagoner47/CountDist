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

namespace configuration {
	std::istream& operator>> (std::istream& ins, data& d) {
		std::string s, key, value;

		while (std::getline(ins, s)) {
			std::string::size_type begin = s.find_first_not_of(" \f\t\v");
			if (begin == std::string::npos) continue;

			if (std::string("#").find(s[begin]) != std::string::npos) continue;

			std::string::size_type end = s.find('=', begin);
			key = s.substr(begin, end - begin);

			key.erase(key.find_last_not_of(" \f\t\v") + 1);

			if (key.empty()) continue;

			begin = s.find_first_not_of(" \f\n\r\t\v", end + 1);
			end = s.find_last_not_of(" \f\n\t\r\v") + 1;

			value = s.substr(begin, end - begin);

			d[key] = value;
		}
		return ins;
	}

	std::ostream& operator<< (std::ostream& outs, const data& d) {
		for (auto iter : d) {
			outs << iter.first << " = " << iter.second << std::endl;
		}
		return outs;
	}
}
