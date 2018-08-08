#ifndef READ_CONFIG_H
#define READ_CONFIG_H

#include <string>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <list>

namespace configuration {
	struct data: std::unordered_map<std::string, std::string> {
		bool iskey(const std::string& s) const {
			return count(s) != 0;
		}
		double as_double(const std::string& key) {
            if (!iskey(key)) throw 0;
            std::istringstream ss(this->operator[](key));
            double result;
            ss >> result;
            if (!ss.eof()) throw 1;
            return result;
        }
        bool as_bool(const std::string& key) {
			if (!iskey(key)) throw 0;
			std::istringstream ss(this->operator[](key));
			bool result;
			ss >> result;
			if (!ss.eof()) throw 1;
			return result;
		}
	};

	std::istream& operator>> (std::istream& ins, data& d);

	std::ostream& operator<< (std::ostream& outs, const data& d);
}

configuration::data reader(const std::string& fname);

#endif
