#include <cmath>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>
#include <functional>
#include <string>
#include <sstream>
#include <cstring>
#include <tuple>
#include <iterator>
#include <fstream>
#include <typeinfo>
#include <sqlite3.h>
#include "calc_distances.h"
using namespace std;

// Conditionally include OpenMP, and set variable to tell if included
#if USE_OMP
#include <omp.h>
omp_set_num_threads(OMP_NUM_THREADS);
#else
#define omp_get_thread_num() 0
#endif

const double one_over_root2 = 0.707106781186548;
const size_t widx = 100000;

int exec_callback(void*, int, char*[], char*[]);

int exec_callback(void *ptr, int argc, char *argv[], char *names[]) {
	vector<vector<size_t>> *list = reinterpret_cast<vector<vector<size_t>> *>(ptr);
	vector<size_t> r;
	r.push_back((size_t) atoi(argv[0]));
	r.push_back((size_t) atoi(argv[1]));
	list->push_back(r);
	return 0;
}

inline void check_vals(sqlite3 *db, size_t nrows, size_t id1[], size_t id2[]) {
	vector<vector<size_t>> idx;
	char *errmsg = NULL;
	string query = "SELECT ID1, ID2 FROM SEPARATIONS LIMIT " + to_string(nrows) + ";";
	sqlite3_exec(db, query.c_str(), exec_callback, &idx, &errmsg);
	if (errmsg) {
		cerr << errmsg << endl;
		exit(20);
	}
	for (size_t i = 0; i < nrows; i++) {
		cout << "Row " << i << ": " << id1[i] << ", " << id2[i] << endl;
		if (idx[i][0] != id1[i]) {
			cerr << "ID1 does not match for row " << i << endl;
			cerr << "Expected: " << id1[i] << ", Observed: " << idx[i][0] << endl;
			exit(21);
		}
		if (idx[i][1] != id2[i]) {
			cerr << "ID2 does not match for row " << i << endl;
			cerr << "Expected: " << id2[i] << ", Observed: " << idx[i][1] << endl;
			exit(22);
		}
	}
}

double unit_dot(Pos pos1, Pos pos2) {
	vector<double> n1(pos1.nvec());
	vector<double> n2(pos2.nvec());
	return inner_product(n1.begin(), n1.end(), n2.begin(), 0.0);
}

tuple<double, double> dot(Pos pos1, Pos pos2) {
	vector<double> rt1(pos1.rtvec()), ro1(pos1.rovec()), rt2(pos2.rtvec()), ro2(pos2.rovec());
	return make_tuple(inner_product(rt1.begin(), rt1.end(), rt2.begin(), 0.0), inner_product(ro1.begin(), ro1.end(), ro2.begin(), 0.0));
}

double dot(Pos pos1, Pos pos2, bool use_true) {
	vector<double> r1(3), r2(3);
	if (use_true) {
		r1 = pos1.rtvec();
		r2 = pos2.rtvec();
	}
	else {
		r1 = pos1.rovec();
		r2 = pos2.rovec();
	}
	return inner_product(r1.begin(), r1.end(), r2.begin(), 0.0);
}

tuple<double, double> r_par(Pos pos1, Pos pos2) {
	double mult_fac = one_over_root2 * sqrt(1.0 + unit_dot(pos1, pos2));
	return make_tuple(mult_fac * fabs(pos1.rt - pos2.rt), mult_fac * fabs(pos1.ro - pos2.ro));
}

double r_par(Pos pos1, Pos pos2, bool use_true) {
	if (use_true) {
		return one_over_root2 * sqrt(1.0 + unit_dot(pos1, pos2)) * fabs(pos1.rt - pos2.rt);
	}
	return one_over_root2 * sqrt(1.0 + unit_dot(pos1, pos2)) * fabs(pos1.ro - pos2.ro);
}

tuple<double, double> r_perp(Pos pos1, Pos pos2) {
	double mult_fac = one_over_root2 * sqrt(1.0 - unit_dot(pos1, pos2));
	return make_tuple(mult_fac * (pos1.rt + pos2.rt), mult_fac * (pos1.ro + pos2.ro));
}

double r_perp(Pos pos1, Pos pos2, bool use_true) {
	if (use_true) {
		return one_over_root2 * sqrt(1.0 - unit_dot(pos1, pos2)) * (pos1.rt + pos2.rt);
	}
	return one_over_root2 * sqrt(1.0 - unit_dot(pos1, pos2)) * (pos1.ro + pos2.ro);
}

bool check_box(Pos pos1, Pos pos2, double max, bool use_true) {
	if (use_true) {
		return ((fabs(pos1.xt - pos2.xt) <= max) && (fabs(pos1.yt - pos2.yt) <= max) && (fabs(pos1.zt - pos2.zt) <= max));
	}
	return ((fabs(pos1.xo - pos2.xo) <= max) && (fabs(pos1.yo - pos2.yo) <= max) && (fabs(pos1.zo - pos2.zo) <= max));
}

bool check_lims(double val, double min, double max) {
	return (isfinite(val) && (val >= min) && (val <= max));
}

bool check_2lims(Pos pos1, Pos pos2, double rp_min, double rp_max, double rl_min, double rl_max, bool use_true) {
	if (!check_lims(r_perp(pos1, pos2, use_true), rp_min, rp_max)) return false;
	if (!check_lims(r_par(pos1, pos2, use_true), rl_min, rl_max)) return false;
	return true;
}

void get_dist_auto(vector<Pos> pos, double rp_min, double rp_max, double rl_min, double rl_max, string db_file, bool use_true) {
	if (USE_OMP) {
		int config_status = sqlite3_config(SQLITE_CONFIG_SERIALIZED);
		if (config_status != SQLITE_OK) {
			cerr << "Cannot configure SQLite in serialized mode for OpenMP" << endl;
			exit(4);
		}
	}
	sqlite3 *db;  // Database connection object pointer
	string sql;
	int out = sqlite3_open(db_file.c_str(), &db);
	if (out != SQLITE_OK) {
		cerr << "Cannot open database: " << db_file << endl;
		exit(1);
	}
	// Some time saving options for SQLite
	sqlite3_exec(db, "PRAGMA synchronous = OFF", 0, 0, 0);
	sqlite3_exec(db, "PRAGMA journal_mode = MEMORY", 0, 0, 0);
	// Drop the table if it already exits
	sql = "DROP TABLE IF EXISTS SEPARATIONS";
	out = sqlite3_exec(db, sql.c_str(), 0, 0, 0);
	if (out != SQLITE_OK) {
		cerr << "Cannot drop table SEPARATIONS from database: " << db_file << endl;
		sqlite3_close(db);
		exit(2);
	}
	// Create the table in the database
	sql = "CREATE TABLE SEPARATIONS(" \
	       "R_PERP NUM NOT NULL,"\
	       "R_PAR NUM NOT NULL,"\
	       "ID1 TEXT NOT NULL,"\
	       "ID2 TEXT NOT NULL);";
	out = sqlite3_exec(db, sql.c_str(), 0, 0, 0);
	if (out != SQLITE_OK) {
		cerr << "Cannot create table SEPARATIONS in database: " << db_file << endl;
		sqlite3_close(db);
		exit(3);
	}
	sql = "INSERT INTO SEPARATIONS VALUES (?1, ?2, ?3, ?4)";
	sqlite3_stmt *stmt;
	out = sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, 0);
	if (out != SQLITE_OK) {
		cerr << "Cannot prepare SQL insert statement for transaction in database: " << db_file << endl;
		sqlite3_close(db);
		exit(5);
	}
	if (isinf(rp_max)) {
		rp_max = HUGE_VAL;
	}
	if (isinf(rl_max)) {
		rl_max = HUGE_VAL;
	}
	double r_max = sqrt((rp_max * rp_max) + (rl_max * rl_max));
	SeparationsSingle output;

	size_t n = pos.size();
	// Keep track of number of lines in transaction: commit after reaching MAX_ROWS
	size_t num_rows = 0;
	// Begin a transaction
	out = sqlite3_exec(db, "BEGIN TRANSACTION", 0, 0, 0);
	if (out != SQLITE_OK) {
		cerr << "Cannot begin transaction in database: " << db_file << endl;
		sqlite3_close(db);
		exit(6);
	}
	string id1, id2;
	#pragma omp parallel for if(USE_OMP) private(output, id1, id2)
	for (size_t k = 0; k < (n * (n - 1)) / 2; k++) {
		size_t i = k / n;
		size_t j = k % n;
		if (i <= j) {
			continue;
		}
		if (check_box(pos[i], pos[j], r_max, use_true)) {
			if (check_2lims(pos[i], pos[j], rp_min, rp_max, rl_min, rl_max, use_true)) {
				output.set_all(r_perp(pos[i], pos[j], use_true), r_par(pos[i], pos[j], use_true), pos[i].idx, pos[j].idx);
				id1 = to_string(output.id1);
				id2 = to_string(output.id2);
				#pragma omp critical
				{
					num_rows++;
					sqlite3_bind_double(stmt, 1, output.rp);
					sqlite3_bind_double(stmt, 2, output.rl);
					sqlite3_bind_text(stmt, 3, id1.c_str(), -1, SQLITE_STATIC);
					sqlite3_bind_text(stmt, 4, id2.c_str(), -1, SQLITE_STATIC);
					sqlite3_step(stmt);
					sqlite3_reset(stmt);
					if (num_rows == sepconstants::MAX_ROWS) {
						// End current transaction and begin a new one
						out = sqlite3_exec(db, "END TRANSACTION", 0, 0, 0);
						if (out != SQLITE_OK) {
							cerr << "Cannot complete transaction in database: " << db_file << endl;
							sqlite3_close(db);
							exit(8);
						}
						num_rows = 0;
						out = sqlite3_exec(db, "BEGIN TRANSACTION", 0, 0, 0);
						if (out != SQLITE_OK) {
							cerr << "Cannot begin transaction in database: " << db_file << endl;
							sqlite3_close(db);
							exit(9);
						}
					}
				}
			}
		}
	}
	out = sqlite3_exec(db, "END TRANSACTION", 0, 0, 0);
	if (out != SQLITE_OK) {
		cerr << "Cannot complete transaction in database: " << db_file << endl;
		sqlite3_close(db);
		exit(7);
	}
	sqlite3_close(db);
}

void get_dist_auto(vector<Pos> pos, double rp_min, double rp_max, double rl_min, double rl_max, string db_file) {
	if (USE_OMP) {
		int config_status = sqlite3_config(SQLITE_CONFIG_SERIALIZED);
		if (config_status != SQLITE_OK) {
			cerr << "Cannot configure SQLite in serialized mode for OpenMP" << endl;
			exit(4);
		}
	}
	sqlite3 *db;  // Database connection object pointer
	string sql;
	int out = sqlite3_open(db_file.c_str(), &db);
	if (out != SQLITE_OK) {
		cerr << "Cannot open database: " << db_file << endl;
		exit(1);
	}
	// Some time saving options for SQLite
	sqlite3_exec(db, "PRAGMA synchronous = OFF", 0, 0, 0);
	sqlite3_exec(db, "PRAGMA journal_mode = MEMORY", 0, 0, 0);
	// Drop the table if it already exits
	sql = "DROP TABLE IF EXISTS SEPARATIONS";
	out = sqlite3_exec(db, sql.c_str(), 0, 0, 0);
	if (out != SQLITE_OK) {
		cerr << "Cannot drop table SEPARATIONS from database: " << db_file << endl;
		sqlite3_close(db);
		exit(2);
	}
	// Create the table in the database
	sql = "CREATE TABLE SEPARATIONS(" \
	       "R_PERP_T NUM,"\
	       "R_PAR_T NUM,"\
	       "R_PERP_O NUM NOT NULL,"\
	       "R_PAR_O NUM NOT NULL,"\
	       "ID1 INTEGER NOT NULL,"\
	       "ID2 INTEGER NOT NULL);";
	out = sqlite3_exec(db, sql.c_str(), 0, 0, 0);
	if (out != SQLITE_OK) {
		cerr << "Cannot create table SEPARATIONS in database: " << db_file << endl;
		sqlite3_close(db);
		exit(3);
	}
	sql = "INSERT INTO SEPARATIONS VALUES (?1, ?2, ?3, ?4, ?5, ?6)";
	sqlite3_stmt *stmt;
	out = sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, 0);
	if (out != SQLITE_OK) {
		cerr << "Cannot prepare SQL insert statement for transaction in database: " << db_file << endl;
		sqlite3_close(db);
		exit(5);
	}
	if (isinf(rp_max)) {
		rp_max = HUGE_VAL;
	}
	if (isinf(rl_max)) {
		rl_max = HUGE_VAL;
	}
	double r_max = sqrt((rp_max * rp_max) + (rl_max * rl_max));
	Separations output;

	size_t n = pos.size();
	// Keep track of number of lines in transaction: commit after reaching MAX_ROWS
	size_t num_rows = 0;
	// Begin a transaction
	out = sqlite3_exec(db, "BEGIN TRANSACTION", 0, 0, 0);
	if (out != SQLITE_OK) {
		cerr << "Cannot begin transaction in database: " << db_file << endl;
		sqlite3_close(db);
		exit(6);
	}
	size_t id1[10], id2[10];
	#pragma omp parallel for if(USE_OMP) private(output)
	for (size_t k = 0; k < (n * (n - 1)) / 2; k++) {
		size_t i = k / n;
		size_t j = k % n;
		if (i <= j) {
			continue;
		}
		if (check_box(pos[i], pos[j], r_max, false)) {
			if (check_2lims(pos[i], pos[j], rp_min, rp_max, rl_min, rl_max, false)) {
				output.set_all(r_perp(pos[i], pos[j]), r_par(pos[i], pos[j]), pos[i].idx, pos[j].idx);
				#pragma omp critical
				{
					if (num_rows < 10) {
						cout << "i = " << i << ", ID1 = " << output.id1 << endl;
						cout << "j = " << j << ", ID2 = " << output.id2 << endl;
						id1[num_rows] = output.id1;
						id2[num_rows] = output.id2;
					}
					num_rows++;
					out = sqlite3_bind_double(stmt, 1, output.rpt);
					out = sqlite3_bind_double(stmt, 2, output.rlt);
					out = sqlite3_bind_double(stmt, 3, output.rpo);
					out = sqlite3_bind_double(stmt, 4, output.rlo);
					out = sqlite3_bind_int64(stmt, 5, i);
					out = sqlite3_bind_int64(stmt, 6, j);
					out = sqlite3_step(stmt);
					if (out != SQLITE_DONE) {
						cerr << out << endl;
						cerr << sqlite3_errmsg(db) << endl;
						sqlite3_close(db);
						exit(10);
					}
					sqlite3_reset(stmt);
					if (num_rows % sepconstants::MAX_ROWS == 0) {
						// End current transaction and begin a new one
						out = sqlite3_exec(db, "END TRANSACTION", 0, 0, 0);
						if (out != SQLITE_OK) {
							cerr << "Cannot complete transaction in database: " << db_file << endl;
							sqlite3_close(db);
							exit(8);
						}
						out = sqlite3_exec(db, "BEGIN TRANSACTION", 0, 0, 0);
						if (out != SQLITE_OK) {
							cerr << "Cannot begin transaction in database: " << db_file << endl;
							sqlite3_close(db);
							exit(9);
						}
					}
				}
			}
		}
	}
	out = sqlite3_exec(db, "END TRANSACTION", 0, 0, 0);
	if (out != SQLITE_OK) {
		cerr << "Cannot complete transaction in database: " << db_file << endl;
		sqlite3_close(db);
		exit(7);
	}
	check_vals(db, 10, id1, id2);
	sqlite3_close(db);
}

void get_dist_cross(vector<Pos> pos1, vector<Pos> pos2, double rp_min, double rp_max, double rl_min, double rl_max, string db_file, bool use_true) {
	if (USE_OMP) {
		int config_status = sqlite3_config(SQLITE_CONFIG_SERIALIZED);
		if (config_status != SQLITE_OK) {
			cerr << "Cannot configure SQLite in serialized mode for OpenMP" << endl;
			exit(4);
		}
	}
	sqlite3 *db;  // Database connection object pointer
	string sql;
	int out = sqlite3_open(db_file.c_str(), &db);
	if (out != SQLITE_OK) {
		cerr << "Cannot open database: " << db_file << endl;
		exit(1);
	}
	// Some time saving options for SQLite
	sqlite3_exec(db, "PRAGMA synchronous = OFF", 0, 0, 0);
	sqlite3_exec(db, "PRAGMA journal_mode = MEMORY", 0, 0, 0);
	// Drop the table if it already exits
	sql = "DROP TABLE IF EXISTS SEPARATIONS";
	out = sqlite3_exec(db, sql.c_str(), 0, 0, 0);
	if (out != SQLITE_OK) {
		cerr << "Cannot drop table SEPARATIONS from database: " << db_file << endl;
		sqlite3_close(db);
		exit(2);
	}
	// Create the table in the database
	sql = "CREATE TABLE SEPARATIONS(" \
	       "R_PERP NUM NOT NULL,"\
	       "R_PAR NUM NOT NULL,"\
	       "ID1 TEXT NOT NULL,"\
	       "ID2 TEXT NOT NULL);";
	out = sqlite3_exec(db, sql.c_str(), 0, 0, 0);
	if (out != SQLITE_OK) {
		cerr << "Cannot create table SEPARATIONS in database: " << db_file << endl;
		sqlite3_close(db);
		exit(3);
	}
	sql = "INSERT INTO SEPARATIONS VALUES (?1, ?2, ?3, ?4)";
	sqlite3_stmt *stmt;
	out = sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, 0);
	if (out != SQLITE_OK) {
		cerr << "Cannot prepare SQL insert statement for transaction in database: " << db_file << endl;
		sqlite3_close(db);
		exit(5);
	}
	if (isinf(rp_max)) {
		rp_max = HUGE_VAL;
	}
	if (isinf(rl_max)) {
		rl_max = HUGE_VAL;
	}
	double r_max = sqrt((rp_max * rp_max) + (rl_max * rl_max));
	SeparationsSingle output;

	size_t n1 = pos1.size();
	size_t n2 = pos2.size();
	// Keep track of number of lines in transaction: commit after reaching MAX_ROWS
	size_t num_rows = 0;
	// Begin a transaction
	out = sqlite3_exec(db, "BEGIN TRANSACTION", 0, 0, 0);
	if (out != SQLITE_OK) {
		cerr << "Cannot begin transaction in database: " << db_file << endl;
		sqlite3_close(db);
		exit(6);
	}
	#pragma omp parallel for if(USE_OMP) private(output, sql) collapse(2)
	for (size_t i = 0; i < n1; i++) {
		for (size_t j = 0; j < n2; j++) {
			if (check_box(pos1[i], pos2[j], r_max, use_true)) {
				if (check_2lims(pos1[i], pos2[j], rp_min, rp_max, rl_min, rl_max, use_true)) {
					output.set_all(r_perp(pos1[i], pos2[j], use_true), r_par(pos1[i], pos2[j], use_true), pos1[i].idx, pos2[j].idx);
					#pragma omp critical
					{
						num_rows++;
						sqlite3_bind_double(stmt, 1, output.rp);
						sqlite3_bind_double(stmt, 2, output.rl);
						sqlite3_bind_text(stmt, 3, to_string(output.id1).c_str(), -1, SQLITE_STATIC);
						sqlite3_bind_text(stmt, 4, to_string(output.id2).c_str(), -1, SQLITE_STATIC);
						sqlite3_step(stmt);
						sqlite3_reset(stmt);
						if (num_rows == sepconstants::MAX_ROWS) {
							// End current transaction and begin a new one
							out = sqlite3_exec(db, "END TRANSACTION", 0, 0, 0);
							if (out != SQLITE_OK) {
								cerr << "Cannot complete transaction in database: " << db_file << endl;
								sqlite3_close(db);
								exit(8);
							}
							num_rows = 0;
							out = sqlite3_exec(db, "BEGIN TRANSACTION", 0, 0, 0);
							if (out != SQLITE_OK) {
								cerr << "Cannot begin transaction in database: " << db_file << endl;
								sqlite3_close(db);
								exit(9);
							}
						}
					}
				}
			}
		}
	}
	out = sqlite3_exec(db, "END TRANSACTION", 0, 0, 0);
	if (out != SQLITE_OK) {
		cerr << "Cannot complete transaction in database: " << db_file << endl;
		sqlite3_close(db);
		exit(7);
	}
	sqlite3_close(db);
}

void get_dist_cross(vector<Pos> pos1, vector<Pos> pos2, double rp_min, double rp_max, double rl_min, double rl_max, string db_file) {
	sqlite3 *db;  // Database connection object pointer
	string sql;
	int out = sqlite3_open(db_file.c_str(), &db);
	if (out != SQLITE_OK) {
		cerr << "Cannot open database: " << db_file << endl;
		exit(1);
	}
	// Some time saving options for SQLite
	sqlite3_exec(db, "PRAGMA synchronous = OFF", 0, 0, 0);
	sqlite3_exec(db, "PRAGMA journal_mode = MEMORY", 0, 0, 0);
	// Drop the table if it already exits
	sql = "DROP TABLE IF EXISTS SEPARATIONS";
	out = sqlite3_exec(db, sql.c_str(), 0, 0, 0);
	if (out != SQLITE_OK) {
		cerr << "Cannot drop table SEPARATIONS from database: " << db_file << endl;
		cerr << "Returned status: " << out << endl;
		sqlite3_close(db);
		exit(2);
	}
	// Create the table in the database
	sql = "CREATE TABLE SEPARATIONS(" \
	       "R_PERP_T NUM,"\
	       "R_PAR_T NUM,"\
	       "R_PERP_O NUM NOT NULL,"\
	       "R_PAR_O NUM NOT NULL,"\
	       "ID1 TEXT NOT NULL,"\
	       "ID2 TEXT NOT NULL);";
	out = sqlite3_exec(db, sql.c_str(), 0, 0, 0);
	if (out != SQLITE_OK) {
		cerr << "Cannot create table SEPARATIONS in database: " << db_file << endl;
		sqlite3_close(db);
		exit(3);
	}
	sql = "INSERT INTO SEPARATIONS VALUES (?1, ?2, ?3, ?4, ?5, ?6)";
	sqlite3_stmt *stmt;
	out = sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, 0);
	if (out != SQLITE_OK) {
		cerr << "Cannot prepare SQL insert statement for transaction in database: " << db_file << endl;
		sqlite3_close(db);
		exit(5);
	}
	if (isinf(rp_max)) {
		rp_max = HUGE_VAL;
	}
	if (isinf(rl_max)) {
		rl_max = HUGE_VAL;
	}
	double r_max = sqrt((rp_max * rp_max) + (rl_max * rl_max));
	Separations output;

	size_t n1 = pos1.size();
	size_t n2 = pos2.size();
	// Keep track of number of lines in transaction: commit after reaching MAX_ROWS
	size_t num_rows = 0;
	// Begin a transaction
	out = sqlite3_exec(db, "BEGIN TRANSACTION", 0, 0, 0);
	if (out != SQLITE_OK) {
		cerr << "Cannot begin transaction in database: " << db_file << endl;
		sqlite3_close(db);
		exit(6);
	}
	#pragma omp parallel for if(USE_OMP) private(output, sql) collapse(2)
	for (size_t i = 0; i < n1; i++) {
		for (size_t j = 0; j < n2; j++) {
			if (check_box(pos1[i], pos2[j], r_max, false)) {
				if (check_2lims(pos1[i], pos2[j], rp_min, rp_max, rl_min, rl_max, false)) {
					output.set_all(r_perp(pos1[i], pos2[j]), r_par(pos1[i], pos2[j]), pos1[i].idx, pos2[j].idx);
					#pragma omp critical
					{
						num_rows++;
						sqlite3_bind_double(stmt, 1, output.rpt);
						sqlite3_bind_double(stmt, 2, output.rlt);
						sqlite3_bind_double(stmt, 3, output.rpo);
						sqlite3_bind_double(stmt, 4, output.rlo);
						sqlite3_bind_text(stmt, 5, to_string(output.id1).c_str(), -1, SQLITE_STATIC);
						sqlite3_bind_text(stmt, 6, to_string(output.id2).c_str(), -1, SQLITE_STATIC);
						sqlite3_step(stmt);
						sqlite3_reset(stmt);
						if (num_rows == sepconstants::MAX_ROWS) {
							// End current transaction and begin a new one
							out = sqlite3_exec(db, "END TRANSACTION", 0, 0, 0);
							if (out != SQLITE_OK) {
								cerr << "Cannot complete transaction in database: " << db_file << endl;
								sqlite3_close(db);
								exit(8);
							}
							num_rows = 0;
							out = sqlite3_exec(db, "BEGIN TRANSACTION", 0, 0, 0);
							if (out != SQLITE_OK) {
								cerr << "Cannot begin transaction in database: " << db_file << endl;
								sqlite3_close(db);
								exit(9);
							}
						}
					}
				}
			}
		}
	}
	out = sqlite3_exec(db, "END TRANSACTION", 0, 0, 0);
	if (out != SQLITE_OK) {
		cerr << "Cannot complete transaction in database: " << db_file << endl;
		sqlite3_close(db);
		exit(7);
	}
	sqlite3_close(db);
}
