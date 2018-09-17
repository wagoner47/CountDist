#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <sqlite3.h>
#include <typeinfo>
#include "io.h"
#include "calc_distances.h"
#include "read_config.h"
#include "type.h"
using namespace std;

vector<Pos> readCatalog(string fname, bool use_true, bool use_obs, bool has_true, bool has_obs) {
	double ra, dec, rt, ro;
	if (!(has_true || has_obs)) {
	    cerr << "Must have at least true or observed distances in catalog, or both" << endl;
	    exit(1);
	}
	if (!(use_true || use_obs)) {
	    cerr << "Must use at least true or observed distances, or both" << endl;
	    exit(2);
	}
	if (use_true && !has_true) {
	    cerr << "Cannot use true distances when not provided in catalog" << endl;
	    exit(3);
	}
	if (use_obs && !has_obs) {
	    cerr << "Cannot use observed distances when not provided in catalog" << endl;
	    exit(4);
	}
	vector<Pos> pos;
	ifstream fin(fname, ifstream::in);
	string line;
	while (getline(fin, line)) {
		if (line.substr(0, 1).compare("#") != 0) {
			istringstream iss(line);
			if (has_true && has_obs) {
			    iss >> ra >> dec >> rt >> ro;
			    if (!use_true) {
			        rt = NAN;
			    }
			    if (!use_obs) {
			        ro = NAN;
			    }
			}
			else if (has_true) {
			    iss >> ra >> dec >> rt;
			    ro = NAN;
			}
			else {
			    iss >> ra >> dec >> ro;
			    rt = NAN;
			}
			Pos posi(ra, dec, rt, ro);
			pos.push_back(posi);
		}
	}
	fin.close();
	return pos;
}

void setup_sqlite(bool use_omp) {
    if (use_omp) {
        int config_status = sqlite3_config(SQLITE_CONFIG_SERIALIZED);
        if (config_status != SQLITE_OK) {
            cerr << "Cannot configure SQLite in serialized mode for OpenMP" << endl;
            exit(5);
        }
    }
}

void open_db(sqlite3*& db, string db_file) {
    int open_status = sqlite3_open(db_file.c_str(), &db);
    if (open_status != SQLITE_OK) {
        cerr << "Cannot open database: " << db_file << endl;
        sqlite3_close(db);
        exit(6);
    }
    // Extra options to set up on the handler
    sqlite3_exec(db, "PRAGMA synchronous = OFF", 0, 0, 0);
    sqlite3_exec(db, "PRAGMA journal_mode = MEMORY", 0, 0, 0);
}

void drop_table(sqlite3 *db, string table_name) {
    string drop_stmt = "DROP TABLE IF EXISTS " + table_name;
    int drop_status = sqlite3_exec(db, drop_stmt.c_str(), 0, 0, 0);
    if (drop_status != SQLITE_OK) {
        cerr << "Cannot drop table " << table_name << " from database" << endl;
	int errcode = sqlite3_extended_errcode(db);
        sqlite3_close(db);
        exit(errcode);
    }
}

template <size_t ncols>
void create_table(sqlite3 *db, string table_name, array<string, ncols> col_names) {
    string create_stmt = "CREATE TABLE " + table_name + "(";
    for (const auto& col_name : col_names) {
	create_stmt += col_name + " NUM NOT NULL" + ((&col_name == &col_names.back()) ? ");" : ",");
    }
    if (sqlite3_exec(db, create_stmt.c_str(), 0, 0, 0) != SQLITE_OK) {
	cerr << "Cannot create table " << table_name << "in database" << endl;
	int errcode = sqlite3_extended_errcode(db);
	sqlite3_close(db);
	exit(errcode);
    }
}

void setup_db(sqlite3 *db, string table_name, bool use_true_and_obs) {
  drop_table(db, table_name);
  if (use_true_and_obs) {
    array<string, 5> col_names = {"R_PERP_T", "R_PAR_T", "R_PERP_O", "R_PAR_O", "AVE_OBS_LOS"};
    create_table(db, table_name, col_names);
  }
  else {
    array<string, 2> col_names = {"R_PERP", "R_PAR"};
    create_table(db, table_name, col_names);
  }
}

void setup_stmt(sqlite3 *db, sqlite3_stmt *&stmt, string table_name, bool use_true_and_obs) {
  string sql_stmt = "INSERT INTO " + table_name + " VALUES (?1, ?2";
  if (use_true_and_obs) {
    sql_stmt += ", ?3, ?4, ?5";
  }
  sql_stmt += ")";
  if (sqlite3_prepare_v2(db, sql_stmt.c_str(), -1, &stmt, 0) != SQLITE_OK) {
    cerr << "Cannot prepare SQL insert statement for transaction in database" << endl;
    int errcode = sqlite3_extended_errcode(db);
    sqlite3_close(db);
    exit(errcode);
  }
    // Debug: check statement
    // cout << "SQL statement in setup_stmt: " << sqlite3_sql(stmt) << endl;
}

void begin_transaction(sqlite3 *db) {
  if (sqlite3_exec(db, "BEGIN TRANSACTION", 0, 0, 0) != SQLITE_OK) {
    cerr << "Cannot begin transaction in database" << endl;
    int errcode = sqlite3_extended_errcode(db);
    sqlite3_close(db);
    exit(errcode);
  }
}

void end_transaction(sqlite3 *db) {
  if (sqlite3_exec(db, "END_TRANSACTION", 0, 0, 0) != SQLITE_OK) {
    cerr << "Cannot complete transaction in database" << endl;
    int errcode = sqlite3_extended_errcode(db);
    sqlite3_close(db);
    exit(errcode);
  }
}

void start_sqlite(sqlite3*& db, sqlite3_stmt *&stmt, string db_file, string table_name, bool use_true, bool use_obs, bool use_omp) {
    setup_sqlite(use_omp);
    open_db(db, db_file);
    setup_db(db, table_name, (use_true && use_obs));
    setup_stmt(db, stmt, table_name, (use_true && use_obs));
    // Debug: check statement
    // cout << "SQL statement in start_sqlite: " << sqlite3_sql(stmt) << endl;
    begin_transaction(db);
}

int count_callback(void *row_count, int argc, char **argv, char **colname) {
    size_t *c = (size_t *)row_count;
    *c = atoi(argv[0]);
    return 0;
}

void check_rows_written(sqlite3 *db, string table_name, size_t num_rows_expected) {
    string query = "SELECT COUNT(*) FROM " + table_name;
    size_t num_rows_retrieved = 0;
    int count_status = sqlite3_exec(db, query.c_str(), count_callback, &num_rows_retrieved, 0);
    if (count_status != SQLITE_OK) {
	cerr << "Cannot get count from table " << table_name << endl;
	sqlite3_close(db);
	exit(13);
    }
    if (num_rows_retrieved != num_rows_expected) {
	cerr << "Wrong number of rows in table: expected " << num_rows_expected << ", found " << num_rows_retrieved << endl;
	sqlite3_close(db);
	exit(14);
    }
}

void write_and_restart(sqlite3 *db) {
    end_transaction(db);
    begin_transaction(db);
}

void write_and_restart_check(sqlite3 *db, string table_name, size_t num_rows_expected) {
	end_transaction(db);
	check_rows_written(db, table_name, num_rows_expected);
	begin_transaction(db);
}

void bind_and_check(sqlite3 *db, sqlite3_stmt *&stmt, int position, double value) {
  if (sqlite3_bind_double(stmt, position, value) != SQLITE_OK) {
    cerr << "Could not bind value " << value << " to position " << position << " to insert statment" << endl;
    int errcode = sqlite3_extended_errcode(db);
    sqlite3_close(db);
    exit(errcode);
  }
}

void take_step(sqlite3 *db, sqlite3_stmt *&stmt, vector<double> row_separations) {
  for (int i = 0; i < row_separations.size(); i++) {
    bind_and_check(db, stmt, i, row_separations[i]);
  }
  int step_status = sqlite3_step(stmt);
  if (step_status != SQLITE_OK && step_status != SQLITE_DONE && step_status != SQLITE_ROW) {
    cerr << "Error stepping sqlite statment" << endl;
    int errcode = sqlite3_extended_errcode(db);
    sqlite3_close(db);
    exit(errcode);
  }
  if (sqlite3_reset(stmt) != SQLITE_OK) {
    cerr << "Error resetting sqlite statement" << endl;
    int errcode = sqlite3_extended_errcode(db);
    sqlite3_close(db);
    exit(errcode);
  }
}

void step_stmt(sqlite3 *db, sqlite3_stmt *&stmt, tuple<double, double> rp, tuple<double, double> rl, double ave_dist, bool use_true, bool use_obs) {
    if (use_true && use_obs) {
	bind_and_check(db, stmt, 1, get<0>(rp));
	bind_and_check(db, stmt, 3, get<1>(rp));
	bind_and_check(db, stmt, 2, get<0>(rl));
	bind_and_check(db, stmt, 4, get<1>(rl));
	bind_and_check(db, stmt, 5, ave_dist);
    }
    else if (use_true) {
	bind_and_check(db, stmt, 1, get<0>(rp));
	bind_and_check(db, stmt, 2, get<0>(rl));
    }
    else if (use_obs) {
	bind_and_check(db, stmt, 1, get<1>(rp));
	bind_and_check(db, stmt, 2, get<1>(rl));
    }
    else {
        cerr << "Must use at least true or observed separations, or both" << endl;
        sqlite3_close(db);
        exit(2);
    }
    int step_status = sqlite3_step(stmt);
    if (step_status != SQLITE_OK && step_status != SQLITE_DONE && step_status != SQLITE_ROW) {
	cerr << "Error stepping sqlite statement" << endl;
	cerr << "SQLite error code: " << sqlite3_errcode(db) << endl;
	cerr << "SQLite extended error code: " << sqlite3_extended_errcode(db) << endl;
	cerr << "SQLite error message: " << sqlite3_errmsg(db) << endl;
	sqlite3_close(db);
	exit(16);
    }
    if (sqlite3_reset(stmt) != SQLITE_OK) {
	cerr << "Error resetting sqlite statement" << endl;
	sqlite3_close(db);
	exit(17);
    }
}

void end_all(sqlite3 *db) {
    end_transaction(db);
    sqlite3_close(db);
}

void end_and_check(sqlite3 *db, string table_name, size_t num_rows_expected) {
    end_transaction(db);
    check_rows_written(db, table_name, num_rows_expected);
    sqlite3_close(db);
}

void write_meta_data(string db_file, string meta_name, double Z_EFF, double SIGMA_R_EFF, double SIGMA_Z) {
    sqlite3 *db;
    array<string, 3> col_names = {"Z_EFF", "SIGMA_R_EFF", "SIGMA_Z"};
    open_db(db, db_file);
    drop_table(db, meta_name);
    create_table(db, meta_name, col_names);
    string insert_stmt = "INSERT INTO " + meta_name + " VALUES (" + to_string(Z_EFF) + ", " + to_string(SIGMA_R_EFF) + ", " + to_string(SIGMA_Z) + ");";
    sqlite3_exec(db, insert_stmt.c_str(), 0, 0, 0);
    sqlite3_close(db);
}
