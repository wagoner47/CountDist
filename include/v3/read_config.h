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
	};

	std::istream& operator >> (std::istream& ins, data& d) {
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
	        end = s.find_last_not_of(" \f\n\r\t\v") + 1;

	        value = s.substr(begin, end - begin);

	        d[key] = value;
	    }
	    return ins;
	}

	std::ostream& operator << (std::ostream& outs, const data& d) {
	    for(const auto iter : data) {
	        outs << iter->first << " = " << iter->last << std::endl;
	    }
	    return outs;
	}
}

configuration::data reader(const std::string& fname);

#endif
