#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <experimental/filesystem>
#include <sqlite3.h>
#include "io.h"
#include "calc_distances.h"
namespace fs = std::experimental::filesystem;
using namespace std;

const double DELTAP = ioconstants::DELTAP;
const double DELTAL = ioconstants::DELTAL;

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

void open_db(sqlite3 *db, string db_file) {
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
        sqlite3_close(db);
        exit(7);
    }
}

void create_table(sqlite3 *db, string table_name, size_t ncols, string col_names[]) {
    string create_stmt = "CREATE TABLE " + table_name + "(";
    string col_stmt;
    for (size_t i = 0; i < ncols - 1; i++) {
        col_stmt = col_names[i] + " NUM NOT NULL,";
        create_stmt += col_stmt;
    }
    col_stmt = col_names[ncols - 1] + " NUM NOT NULL);";
    create_stmt += col_stmt;
    int create_status = sqlite3_exec(db, create_stmt.c_str(), 0, 0, 0);
    if (create_status != SQLITE_OK) {
        cerr << "Cannot create table " << table_name << " in database" << endl;
        sqlite3_close(db);
        exit(8);
    }
}

void setup_db(sqlite3 *db, string table_name, bool use_true, bool use_obs) {
    size_t ncols;
    string *col_names;
    vector<string> col_vec;
    if (use_true && use_obs) {
        ncols = 5;
        col_names = new string[ncols]{"R_PERP_T", "R_PAR_T", "R_PERP_O", "R_PAR_O", "AVE_OBS_LOS"};
    }
    else if (use_true || use_obs) {
        ncols = 2;
        col_names = new string[ncols]{"R_PERP", "R_PAR"};
    }
    else {
        cerr << "Must use at least true or observed separations, or both" << endl;
        sqlite3_close(db);
        exit(2);
    }
    drop_table(db, table_name);
    create_table(db, table_name, ncols, col_names);
}

void setup_stmt(sqlite3 *db, sqlite3_stmt *stmt, string table_name, bool use_true, bool use_obs) {
    string sql_stmt = "INSERT INTO " + table_name + " VALUES (";
    if (use_true && use_obs) {
        sql_stmt += "?1, ?2, ?3, ?4, ?5)";
    }
    else if (use_true || use_obs) {
        sql_stmt += "?1, ?2)";
    }
    else {
        cerr << "Must use at least true or observed separations, or both" << endl;
        sqlite3_close(db);
        exit(2);
    }
    int prep_status = sqlite3_prepare_v2(db, sql_stmt.c_str(), -1, &stmt, 0);
    if (prep_status != SQLITE_OK) {
        cerr << "Cannot prepare SQL insert statement for transaction in database" << endl;
        sqlite3_close(db);
        exit(9);
    }
}

void begin_transaction(sqlite3 *db) {
    int begin_status = sqlite3_exec(db, "BEGIN TRANSACTION", 0, 0, 0);
    if (begin_status != SQLITE_OK) {
        cerr << "Cannot begin transaction in database" << endl;
        sqlite3_close(db);
        exit(10);
    }
}

void end_transaction(sqlite3 *db) {
    int end_status = sqlite3_exec(db, "END TRANSACTION", 0, 0, 0);
    if (end_status != SQLITE_OK) {
        cerr << "Cannot complete transaction in database" << endl;
        sqlite3_close(db);
        exit(12);
    }
}

void start_sqlite(sqlite3 *db, sqlite3_stmt *stmt, string db_file, string table_name, bool use_true, bool use_obs, bool use_omp) {
    setup_sqlite(use_omp);
    open_db(db, db_file);
    setup_db(db, table_name, use_true, use_obs);
    setup_stmt(db, stmt, table_name, use_true, use_obs);
    begin_transaction(db);
}

void write_and_restart(sqlite3 *db) {
    end_transaction(db);
    begin_transaction(db);
}

void step_stmt(sqlite3 *db, sqlite3_stmt *stmt, tuple<double, double> rp, tuple<double, double> rl, double ave_dist, bool use_true, bool use_obs) {
    if (use_true && use_obs) {
        sqlite3_bind_double(stmt, 1, get<0>(rp));
        sqlite3_bind_double(stmt, 3, get<1>(rp));
        sqlite3_bind_double(stmt, 2, get<0>(rl));
        sqlite3_bind_double(stmt, 4, get<1>(rl));
        sqlite3_bind_double(stmt, 5, ave_dist);
    }
    else if (use_true) {
        sqlite3_bind_double(stmt, 1, get<0>(rp));
        sqlite3_bind_double(stmt, 2, get<0>(rl));
    }
    else if (use_obs) {
        sqlite3_bind_double(stmt, 1, get<1>(rp));
        sqlite3_bind_double(stmt, 2, get<1>(rl));
    }
    else {
        cerr << "Must use at least true or observed separations, or both" << endl;
        sqlite3_close(db);
        exit(2);
    }
    sqlite3_step(stmt);
    sqlite3_reset(stmt);
}


void write_meta_data(string db_file, string meta_name, double Z_EFF, double SIGMA_R_EFF, double SIGMA_Z) {
    sqlite3 *db;
    sqlite3_stmt *stmt;
    size_t ncols = 3;
    string col_names[] = {"Z_EFF", "SIGMA_R_EFF", "SIGMA_Z"};
    open_db(db, db_file);
    drop_table(db, meta_name);
    create_table(db, meta_name, ncols, col_names);
    string insert_stmt = "INSERT INTO " + meta_name + " VALUES (" + to_string(Z_EFF) + ", " + to_string(SIGMA_R_EFF) + ", " + to_string(SIGMA_Z) + ");";
    sqlite3_prepare(db, insert_stmt.c_str(), -1, &stmt, 0);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    sqlite3_close(db);
}
