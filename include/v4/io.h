#ifndef IO_H
#define IO_H

#include <vector>
#include <string>
#include <sqlite3.h>
#include <array>
#include <cstddef>
#include "calc_distances.h"

//! Read a catalog from a file
/*!
 * This function will read the data from a catalog and set up a vector of the positions, skipping a separation if it is not to be used
 * \param fname a string containing the path to the catalog file
 * \param use_true a boolean flag that says to use the true separations from the file if true
 * \param use_obs a boolean flag that says to use the observed separations from the file if true
 * \param has_true a boolean flag that says that the file has (true) or doesn't have (false) true separations
 * \param has_obs a boolean flag that says that the file has (true) or doesn't have (false) observed separations
 */
std::vector<Pos> readCatalog(std::string fname, bool use_true, bool use_obs, bool has_true, bool has_obs);

//! Setup sqlite options
/*!
 * Setup sqlite using OpenMP settings or not
 * \param use_omp a boolean flag that says to set up serialized mode in sqlite if true
 */
void setup_sqlite(bool use_omp);

//! Open a database connection
/*!
 * Opens the connection to the requested database file
 * \param db reference to a pointer to a sqlite3 database connection
 * \param db_file a string containing the path to the database file to open
 */
void open_db(sqlite3*& db, std::string db_file);

void drop_table(sqlite3* db, std::string table_name);

template <std::size_t ncols>
extern void create_table(sqlite3 *db, std::string table_name, std::array<std::string, ncols> col_names);

void setup_db(sqlite3 *db, std::string table_name, bool use_true, bool use_obs);

void setup_stmt(sqlite3 *db, sqlite3_stmt *&stmt, std::string table_name, bool use_true, bool use_obs);

void begin_transaction(sqlite3 *db);

void end_transaction(sqlite3 *db);

void start_sqlite(sqlite3 *&db, sqlite3_stmt *&stmt, std::string db_file, std::string table_name, bool use_true, bool use_obs, bool use_omp);

int count_callback(void *count, int argc, char **argv, char **colname);

void check_rows_written(sqlite3 *db, std::string table_name, std::size_t num_rows_expected);

void write_and_restart(sqlite3 *db);

void write_and_restart_check(sqlite3 *db, std::string table_name, std::size_t num_rows_expected);

void step_stmt(sqlite3 *db, sqlite3_stmt *&stmt, std::tuple<double, double> rp, std::tuple<double, double> rl, double ave_dist, bool use_true, bool use_obs);

void end_all(sqlite3 *db);

void end_and_check(sqlite3 *db, std::string table_name, std::size_t num_rows_expected);

void write_meta_data(std::string db_file, std::string meta_name, double Z_EFF, double SIGMA_R_EFF, double SIGMA_Z);

#endif
