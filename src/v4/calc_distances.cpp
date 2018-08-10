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
#include <type_traits>
#include <sqlite3.h>
#include "calc_distances.h"
#include "io.h"
using namespace std;

// Conditionally include OpenMP, and set variable to tell if included
#if USE_OMP
#include <omp.h>
omp_set_num_threads(OMP_NUM_THREADS);
#else
#define omp_get_thread_num() 0
#endif

template <typename T> inline constexpr int signum(T x, false_type is_signed) {
    return T(0) < x;
}

template <typename T> inline constexpr int signum(T x, true_type is_signed) {
    return (T(0) < x) - (x < T(0));
}

template <typename T> inline constexpr int signum(T x) {
    return isnan(x) ? 1 : signum(x, is_signed<T>());
}

const double one_over_root2 = 0.707106781186548;
const size_t widx = 100000;

double unit_dot(Pos pos1, Pos pos2) {
	vector<double> n1(pos1.nvec());
	vector<double> n2(pos2.nvec());
	return inner_product(n1.begin(), n1.end(), n2.begin(), 0.0);
}

tuple<double, double> dot(Pos pos1, Pos pos2) {
	vector<double> rt1(pos1.rtvec()), ro1(pos1.rovec()), rt2(pos2.rtvec()), ro2(pos2.rovec());
	return make_tuple(inner_product(rt1.begin(), rt1.end(), rt2.begin(), 0.0), inner_product(ro1.begin(), ro1.end(), ro2.begin(), 0.0));
}

tuple<double, double> r_par(Pos pos1, Pos pos2) {
	double mult_fac = one_over_root2 * sqrt(1.0 + unit_dot(pos1, pos2));
	int sign = 1;
	if (pos1.has_obs && pos2.has_obs) {
	    sign = signum(pos1.ro - pos2.ro) * signum(pos1.rt - pos2.rt);
	}
	return make_tuple(mult_fac * fabs(pos1.rt - pos2.rt) * sign, mult_fac * fabs(pos1.ro - pos2.ro));
}

tuple<double, double> r_perp(Pos pos1, Pos pos2) {
	double mult_fac = one_over_root2 * sqrt(1.0 - unit_dot(pos1, pos2));
	return make_tuple(mult_fac * (pos1.rt + pos2.rt), mult_fac * (pos1.ro + pos2.ro));
}

double ave_los_distance(Pos pos1, Pos pos2) {
    return 0.5 * (pos1.ro + pos2.ro);
}

bool check_box(Pos pos1, Pos pos2, double max) {
	if (!(pos1.has_obs && pos2.has_obs)) {
		return ((fabs(pos1.xt - pos2.xt) <= max) && (fabs(pos1.yt - pos2.yt) <= max) && (fabs(pos1.zt - pos2.zt) <= max));
	}
	else if ((pos1.has_obs && !pos2.has_obs) || (pos2.has_obs && !pos1.has_obs)) {
	    cerr << "Cannot mix true and observed distances" << endl;
	    exit(13);
	}
	return ((fabs(pos1.xo - pos2.xo) <= max) && (fabs(pos1.yo - pos2.yo) <= max) && (fabs(pos1.zo - pos2.zo) <= max));
}

bool check_lims(double val, double min, double max) {
	return (isfinite(val) && (val >= min) && (val <= max));
}

bool check_2lims(Pos pos1, Pos pos2, double rp_min, double rp_max, double rl_min, double rl_max, bool use_true) {
    double rp, rl;
    if (use_true) {
        rp = get<0>(r_perp(pos1, pos2));
        rl = get<0>(r_par(pos1, pos2));
    }
    else {
        rp = get<1>(r_perp(pos1, pos2));
        rl = get<1>(r_par(pos1, pos2));
    }
	if (!check_lims(rp, rp_min, rp_max)) return false;
	if (!check_lims(rl, rl_min, rl_max)) return false;
	return true;
}

void get_dist(vector<Pos> pos1, vector<Pos> pos2, double rp_min, double rp_max, double rl_min, double rl_max, string db_file, string table_name, bool use_true, bool use_obs, bool is_auto) {
    if (isinf(rp_max)) {
        rp_max = HUGE_VAL;
    }
    if (isinf(rl_max)) {
        rl_max = HUGE_VAL;
    }
    double r_max = sqrt((rp_max * rp_max) + (rl_max * rl_max));

    size_t n1 = pos1.size();
    size_t n2 = pos2.size();
    size_t num_rows = 0;

    sqlite3 *db = 0;
    sqlite3_stmt *stmt = 0;
    start_sqlite(db, stmt, db_file, table_name, use_true, use_obs, USE_OMP);

    #pragma omp parallel for if(USE_OMP) collapse(2)
	for (size_t i = 0; i < n1; i++) {
	    for (size_t j = 0; j < n2; j++) {
	        if (is_auto && i >= j) {
	            continue;
	        }
	        if (check_box(pos1[i], pos2[j], r_max)) {
	            if (check_2lims(pos1[i], pos2[j], rp_min, rp_max, rl_min, rl_max, (use_true && !use_obs))) {
                    #pragma omp critical
                    {
                        num_rows++;
                        step_stmt(db, stmt, r_perp(pos1[i], pos2[j]), r_par(pos1[i], pos2[j]), ave_los_distance(pos1[i], pos2[j]), use_true, use_obs);
                        if (num_rows == sepconstants::MAX_ROWS) {
                            write_and_restart(db);
                        }
	                }
	            }
	        }
	    }
	}
	end_transaction(db);
	sqlite3_close(db);
}
