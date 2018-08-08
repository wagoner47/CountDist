#ifndef CALC_DISTANCES_H
#define CALC_DISTANCES_H

#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <algorithm>
#include <cstdarg>
#include <cstring>
#include <string>
#include <tuple>

#ifdef omp_num_threads
constexpr int OMP_NUM_THREADS = omp_num_threads;
#else
constexpr int OMP_NUM_THREADS = 0;
#endif
constexpr auto USE_OMP = (OMP_NUM_THREADS != 0);
#if USE_OMP
#include <omp.h>
#endif

namespace sepconstants {
	const size_t MAX_ROWS = 1000;
}

constexpr auto RAD2DEG = 180.0 / M_PI;
constexpr auto DEG2RAD = M_PI / 180.0;

inline std::tuple<double,double,double> get_nxyz(double ra, double dec) {
	return std::make_tuple(std::cos(DEG2RAD * ra) * std::cos(DEG2RAD * dec), std::sin(DEG2RAD * ra) * std::cos(DEG2RAD * dec), std::sin(DEG2RAD * dec));
}

inline std::tuple<double,double> get_radec(double nx, double ny, double nz) {
	return std::make_tuple(atan2(ny, nx) * RAD2DEG, atan2(nz, sqrt((nx * nx) + (ny * ny))) * RAD2DEG);
}

struct Pos{
	double ra, dec, rt, ro, nx, ny, nz;
	double xt, yt, zt, xo, yo, zo;
	bool has_true, has_obs;
	Pos(double ra, double dec, double rt, double ro) : ra(ra), dec(dec), rt(rt), ro(ro) {
		auto nxyz = get_nxyz(ra, dec);
		nx = std::get<0>(nxyz);
		ny = std::get<1>(nxyz);
		nz = std::get<2>(nxyz);
		has_true = std::isnan(rt);
		has_obs = std::isnan(ro);
        xt = rt * nx;
        yt = rt * ny;
        zt = rt * nz;
        xo = ro * nx;
        yo = ro * ny;
        zo = ro * nz;
	}
	Pos(double nx, double ny, double nz, double rt, double ro) : rt(rt), ro(ro), nx(nx), ny(ny), nz(nz) {
		auto radec = get_radec(nx, ny, nz);
		ra = std::get<0>(radec);
		dec = std::get<1>(radec);
		has_true = std::isnan(rt);
		has_obs = std::isnan(ro);
		xt = rt * nx;
		yt = rt * ny;
		zt = rt * nz;
		xo = ro * nx;
		yo = ro * ny;
		zo = ro * nz;
	}
	std::vector<double> nvec() {
		std::vector<double> n(3);
		n[0] = nx;
		n[1] = ny;
		n[2] = nz;
		return n;
	}
	std::vector<double> rtvec() {
		std::vector<double> r{xt, yt, zt};
		return r;
	}
	std::vector<double> rovec() {
		std::vector<double> r{xo, yo, zo};
		return r;
	}
};

double unit_dot(Pos, Pos);

std::tuple<double, double> dot(Pos, Pos);

std::tuple<double, double> r_par(Pos, Pos);

std::tuple<double, double> r_perp(Pos, Pos);

double ave_los_distance(Pos, Pos);

bool check_box(Pos, Pos, double);

bool check_lims(double, double, double);

bool check_2lims(Pos, Pos, double, double, double, double);

void get_dist(std::vector<Pos>, std::vector<Pos>, double, double, double, double, std::string, std::string, bool, bool, bool);

#endif
