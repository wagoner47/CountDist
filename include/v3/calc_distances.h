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
	size_t idx;
	Pos(double ra, double dec, double rt, double ro, size_t idx) : ra(ra), dec(dec), rt(rt), ro(ro), idx(idx) {
		auto nxyz = get_nxyz(ra, dec);
		nx = std::get<0>(nxyz);
		ny = std::get<1>(nxyz);
		nz = std::get<2>(nxyz);
		xt = rt * nx;
		yt = rt * ny;
		zt = rt * nz;
		xo = ro * nx;
		yo = ro * ny;
		zo = ro * nz;
	}
	Pos(double nx, double ny, double nz, double rt, double ro, size_t idx) : rt(rt), ro(ro), nx(nx), ny(ny), nz(nz), idx(idx) {
		auto radec = get_radec(nx, ny, nz);
		ra = std::get<0>(radec);
		dec = std::get<1>(radec);
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
		std::vector<double> r(nvec());
		std::transform(r.begin(), r.end(), r.begin(), std::bind1st(std::multiplies<double>(), rt));
		return r;
	}
	std::vector<double> rovec() {
		std::vector<double> r(nvec());
		std::transform(r.begin(), r.end(), r.begin(), std::bind1st(std::multiplies<double>(), ro));
		return r;
	}
};

struct SeparationsSingle{
	double rp, rl;
	size_t id1, id2;
	SeparationsSingle() {}
	SeparationsSingle(double rp, double rl, size_t id1, size_t id2) : rp(rp), rl(rl), id1(id1), id2(id2) {}
	void set_all(double rpi, double rli, size_t id1i, size_t id2i) {
		rp = rpi;
		rl = rli;
		id1 = id1i;
		id2 = id2i;
	}
	std::string get_header() {
		return "# r_perp r_par id1 id2";
	}
};

inline std::ostream& operator<<(std::ostream& os, SeparationsSingle& ss) {
	os << std::endl << ss.rp << " " << ss.rl << " " << ss.id1 << " " << ss.id2;
	return os;
}

struct Separations{
	double rpt, rlt, rpo, rlo;
	size_t id1, id2;
	Separations() {}
	Separations(double rpt, double rlt, double rpo, double rlo, size_t id1, size_t id2) : rpt(rpt), rlt(rlt), rpo(rpo), rlo(rlo), id1(id1), id2(id2) {}
	Separations(std::tuple<double,double> rp, std::tuple<double,double> rl, size_t id1, size_t id2) : id1(id1), id2(id2) {
		rpt = std::get<0>(rp);
		rlt = std::get<0>(rl);
		rpo = std::get<1>(rp);
		rlo = std::get<1>(rl);
	}
	void set_all(double rpti, double rlti, double rpoi, double rloi, size_t id1i, size_t id2i) {
		rpt = rpti;
		rlt = rlti;
		rpo = rpoi;
		rlo = rloi;
		id1 = id1i;
		id2 = id2i;
	}
	void set_all(std::tuple<double,double> rp, std::tuple<double,double> rl, size_t id1i, size_t id2i) {
		rpt = std::get<0>(rp);
		rlt = std::get<0>(rl);
		rpo = std::get<1>(rp);
		rlo = std::get<1>(rl);
		id1 = id1i;
		id2 = id2i;
	}
	std::string get_header() {
		return "# r_perp_t r_par_t r_perp_o r_par_o id1 id2";
	}
};

inline std::ostream& operator<<(std::ostream& os, Separations& s) {
	os << std::endl << s.rpt << " " << s.rlt << " " << s.rpo << " " << s.rlo << " " << s.id1 << " " << s.id2;
	return os;
}

inline std::string format(const char* fmt, ...) {
	int size = 512;
	char* buffer = new char[size];
	va_list vl;
	va_start(vl, fmt);
	int nsize = vsnprintf(buffer, size, fmt, vl);
	if (size <= nsize) {
		size = nsize + 1;
		delete[] buffer;
		buffer = new char[size];
		nsize = vsnprintf(buffer, size, fmt, vl);
	}
	std::string ret(buffer);
	va_end(vl);
	delete[] buffer;
	return ret;
}

double unit_dot(Pos, Pos);

std::tuple<double, double> dot(Pos, Pos);

double dot(Pos, Pos, bool);

std::tuple<double, double> r_par(Pos, Pos);

double r_par(Pos, Pos, bool);

std::tuple<double, double> r_perp(Pos, Pos);

double r_perp(Pos, Pos, bool);

bool check_box(Pos, Pos, double, bool);

bool check_lims(double, double, double);

bool check_2lims(Pos, Pos, double, double, double, double, bool);

void get_dist_auto(std::vector<Pos>, double, double, double, double, std::string, bool);

void get_dist_auto(std::vector<Pos>, double, double, double, double, std::string);

void get_dist_cross(std::vector<Pos>, std::vector<Pos>, double, double, double, double, std::string, bool);

void get_dist_cross(std::vector<Pos>, std::vector<Pos>, double, double, double, double, std::string);

#endif
