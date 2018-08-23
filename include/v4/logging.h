#ifndef LOGGING_H
#define LOGGING_H

#include <iostream>
#include <string>
#include <sstream>
#include <ctime>

enum class severity_level : int {
	fatal=10,
	error=20,
	warning=30,
	info=40,
	debug=50
};

inline std::string severity_level_string(severity_level level) {
	switch(level) {
		case severity_level::debug:
			return "DEBUG";
		case severity_level::info:
			return "INFO";
		case severity_level::warning:
			return "WARNING";
		case severity_level::error:
			return "ERROR";
		case severity_level::fatal:
			return "FATAL";
		default:
			std::cerr << "Unknown severity level: " << static_cast<int>(level) << std::endl;
			exit(30);
	}
}

inline std::string current_date_time() {
	char buf[100];
	auto now = std::time(0);
	std::strftime(buf, sizeof(buf), "%F %T", std::localtime(&now));
	return std::string(buf);
}

extern severity_level LOG_LEVEL;

class Log {
	public:
		Log(severity_level L, const char* file, int line) : level(L) {
			if (L <= LOG_LEVEL) {
				std::cout << current_date_time() << " " << severity_level_string(L) << " - " << file << "[" << line << "]: ";
				opened = true;
			}
		}

		template <typename T> Log& operator<<(const T& msg) {
			if (level <= LOG_LEVEL) {
				std::cout << msg;
				opened = true;
			}
			return *this;
		}

		~Log() {
			if (opened) {
				std::cout << std::endl;
			}
			opened = false;
		}
	private:
		bool opened = false;
		severity_level level = severity_level::fatal;
};

#define LOG(level) Log(level, __FILE__, __LINE__)

#endif
