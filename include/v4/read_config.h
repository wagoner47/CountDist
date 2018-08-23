#ifndef READ_CONFIG_H
#define READ_CONFIG_H

#include <string>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <list>
#include <locale>
#include <array>
#include <utility>
#include "logging.h"

std::string lowercase(std::string input);

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
			std::string str_result;
			ss >> str_result;
			str_result = lowercase(str_result);
			bool result = (str_result.substr(0, 1).compare("t") == 0);
			if (!ss.eof()) throw 1;
			return result;
		}
	};

	std::istream& operator>> (std::istream& ins, data& d);

	std::ostream& operator<< (std::ostream& outs, const data& d);
}

configuration::data reader(const std::string& fname);

void check_key(configuration::data params, std::string key);

template<std::size_t SIZE>
void check_str_keys(configuration::data params, std::array<std::string, SIZE> const& keys) {
  for (const auto& key : keys) {
    check_key(params, key);
    //std::cout << "Value for parameter '" << key << "': '" << params[key] << "'" << std::endl;
    LOG(severity_level::debug) << "Value for parameter '" << key << "': '" << params[key] << "'"; 
  }
}

template<std::size_t SIZE>
void check_dbl_keys(configuration::data params, std::array<std::string, SIZE> const& keys) {
  for (const auto& key : keys) {
    check_key(params, key);
    //std::cout << "Value for parameter '" << key << "': " << params.as_double(key) << std::endl;
    LOG(severity_level::debug) << "Value for parameter '" << key << "': " << params.as_double(key); 
  }
}

template<std::size_t SIZE>
void check_bool_keys(configuration::data params, std::array<std::string, SIZE> const& keys) {
  for (const auto& key : keys) {
    check_key(params, key);
    //std::cout << std::boolalpha << "Value for parameter '" << key << "': " << params.as_bool(key) << std::noboolalpha << std::endl;
    LOG(severity_level::debug) << std::boolalpha << "Value for parameter '" << key << "': " << params.as_bool(key) << std::noboolalpha; 
  }
}

template<std::size_t SIZE1, std::size_t SIZE2, std::size_t SIZE3>
  void check_req_keys(configuration::data params, std::array<std::string, SIZE1> const& str_keys, std::array<std::string, SIZE2> const& dbl_keys, std::array<std::string, SIZE3> const& bool_keys) {
  check_str_keys(params, str_keys);
  check_dbl_keys(params, dbl_keys);
  check_bool_keys(params, bool_keys);
}

// Not sure where else to put this
template <typename V, typename... T>
constexpr auto array_of(T&&... t) -> std::array<V, sizeof...(T)> {
	return {{ std::forward<T>(t)... }};
}

#endif
