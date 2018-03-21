#ifndef READ_CONFIG_H
#define READ_CONFIG_H

#include <string>
#include <sstream>
#include <iostream>
#include <map>

namespace configuration {
	struct data: std::map<std::string, std::string> {
		bool iskey(const std::string& s) const {
			return count(s) != 0;
		}

		bool check_keys(size_t nReq, const std::string keys[]) {
			bool key;
			for (size_t i = 0; i < nReq; i++) {
				key = iskey(keys[i]);
				if (!key) {
					return false;
				}
			}
			return true;
		}
		double as_double(const std::string& key) {
			if (!iskey(key)) throw 0;
			std::istringstream ss(this->operator [] (key));
			double result;
			ss >> result;
			if (!ss.eof()) throw 1;
			return result;
		}
		unsigned long long as_ull(const std::string& key) {
			if (!iskey(key)) throw 0;
			std::istringstream ss(this->operator [] (key));
			unsigned long long result;
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

			if (std::string("#;").find(s[begin]) != std::string::npos) continue;

			std::string::size_type end = s.find("=", begin);
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
		data::const_iterator iter;
		for (iter = d.begin(); iter != d.end(); iter++) {
			outs << iter->first << " = " << iter->second << std::endl;
		}
		return outs;
	}
}

#endif
