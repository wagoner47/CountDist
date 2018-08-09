#ifndef IO_H
#define IO_H

#include <vector>
#include <string>
#include <sqlite3.h>
#include "calc_distances.h"

std::vector<Pos> readCatalog(std::string, bool, bool, bool, bool);

void setup_sqlite(bool);

void open_db(sqlite3*, std::string);

void drop_table(sqlite3*, std::string);

void create_table(sqlite3*, std::string, int, std::string[], std::string[]);

void setup_db(sqlite3*, std::string, bool, bool);

void setup_stmt(sqlite3*, sqlite3_stmt*, std::string, bool, bool);

void begin_transaction(sqlite3*);

void end_transaction(sqlite3*);

void start_sqlite(sqlite3*, sqlite3_stmt*, std::string, std::string, bool, bool, bool);

void write_and_restart(sqlite3*);

void step_stmt(sqlite3*, sqlite3_stmt*, std::tuple<double, double>, std::tuple<double, double>, double, bool, bool);

void write_meta_data(std::string, std::string, double, double, double);

namespace ioconstants {
	const double DELTAP = 50.0;
	const double DELTAL = 60.0;
}

#endif
