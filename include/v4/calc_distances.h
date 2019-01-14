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
#include <stdexcept>
#include <unordered_map>
#include <typeindex>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#if defined(_OPENMP) && defined(omp_num_threads)
constexpr int OMP_NUM_THREADS = omp_num_threads;
#else
constexpr int OMP_NUM_THREADS = 1;
#endif

namespace sepconstants {
	const size_t MAX_ROWS = 10000;
}

typedef std::unordered_map<std::string, std::type_index> cdtype;

typedef std::unordered_map<std::string, std::size_t> indexer;

constexpr auto RAD2DEG = 180.0 / M_PI;
constexpr auto DEG2RAD = M_PI / 180.0;

inline std::tuple<double,double,double> get_nxyz(double ra, double dec) {
	return std::make_tuple(std::cos(DEG2RAD * ra) * std::cos(DEG2RAD * dec), std::sin(DEG2RAD * ra) * std::cos(DEG2RAD * dec), std::sin(DEG2RAD * dec));
}

inline std::tuple<double,double> get_radec(double nx, double ny, double nz) {
    double ra = std::atan2(ny, nx) * RAD2DEG;
    ra += (ra < 0) ? 360.0 : 0.0;
    return std::make_tuple(ra, std::atan2(nz, sqrt((nx * nx) + (ny * ny))) * RAD2DEG);
}

struct Pos{
    Pos(double ra, double dec, double rt, double ro, double tz, double oz)  {
	if ((ra < 0.0) || (ra > 360.0) || std::isnan(ra)) {
	    throw std::invalid_argument("RA outside of valid range [0.0, 360.0]");
	}
	if ((dec < -90.0) || (dec > 90.0) || std::isnan(dec)) {
	    throw std::invalid_argument("DEC outside of valid range [-90.0, 90.0]");
	}
	ra_ = ra;
	dec_ = dec;
	rt_ = rt;
	ro_ = ro;
	zrt_ = tz;
	zro_ = oz;
	auto nxyz = get_nxyz(ra, dec);
	nx_ = std::get<0>(nxyz);
	ny_ = std::get<1>(nxyz);
	nz_ = std::get<2>(nxyz);
	has_true_ = !std::isnan(rt);
	has_obs_ = !std::isnan(ro);
	xt_ = rt * nx_;
	yt_ = rt * ny_;
	zt_ = rt * nz_;
	xo_ = ro * nx_;
	yo_ = ro * ny_;
	zo_ = ro * nz_;
    }
    Pos(double nx, double ny, double nz, double rt, double ro, double tz, double oz) {
	if (((nx * nx) + (ny * ny) + (nz * nz)) != 1.0) {
	    throw std::invalid_argument("Magnitude of unit vector is not 1");
	}
	nx_ = nx;
	ny_ = ny;
	nz_ = nz;
	rt_ = rt;
	ro_ = ro;
	zrt_ = tz;
	zro_ = oz;
	auto radec = get_radec(nx, ny, nz);
	ra_ = std::get<0>(radec);
	dec_ = std::get<1>(radec);
	has_true_ = !std::isnan(rt);
	has_obs_ = !std::isnan(ro);
	xt_ = rt * nx;
	yt_ = rt * ny;
	zt_ = rt * nz;
	xo_ = ro * nx;
	yo_ = ro * ny;
	zo_ = ro * nz;
    }
    std::vector<double> nvec() {
	std::vector<double> n{nx_, ny_, nz_};
	return n;
    }
    std::vector<double> rtvec() {
	std::vector<double> r{xt_, yt_, zt_};
	return r;
    }
    std::vector<double> rovec() {
	std::vector<double> r{xo_, yo_, zo_};
	return r;
    }
    double ra() const { return ra_; }
    double dec() const { return dec_; }
    double rt() const { return rt_; }
    double ro() const { return ro_; }
    double true_redshift() const { return zrt_; }
    double obs_redshift() const { return zro_; }
    double nx() const { return nx_; }
    double ny() const { return ny_; }
    double nz() const { return nz_; }
    double xt() const { return xt_; }
    double yt() const { return yt_; }
    double zt() const { return zt_; }
    double xo() const { return xo_; }
    double yo() const { return yo_; }
    double zo() const { return zo_; }
    bool has_true() const { return has_true_; }
    bool has_obs() const { return has_obs_; }
private:
    double ra_, dec_, rt_, ro_, nx_, ny_, nz_;
    double xt_, yt_, zt_, xo_, yo_, zo_;
    double zrt_, zro_;
    bool has_true_, has_obs_;
};

std::vector<Pos> fill_catalog_vector(std::vector<double> ra_vec, std::vector<double> dec_vec, std::vector<double> rt_vec, std::vector<double> ro_vec, std::vector<double> tz_vec, std::vector<double> oz_vec);

std::vector<Pos> fill_catalog_vector(std::vector<double> nx_vec, std::vector<double> ny_vec, std::vector<double> nz_vec, std::vector<double> rt_vec, std::vector<double> ro_vec, std::vector<double> tz_vec, std::vector<double> oz_vec);

struct Separation {
    double r_perp_t, r_par_t, r_perp_o, r_par_o, ave_zo;
    std::size_t id1, id2;
    Separation() 
        : r_perp_t(), r_par_t(), r_perp_o(), r_par_o(), ave_zo(), id1(), 
	  id2() {}
    Separation(double rpt, double rlt, double rpo, double rlo, double ave_z, std::size_t i1, std::size_t i2) 
        : r_perp_t(rpt), r_par_t(rlt), r_perp_o(rpo), r_par_o(rlo), 
          ave_zo(ave_z), id1(i1), id2(i2) {}
    Separation(std::tuple<double, double> r_perp, std::tuple<double, double> r_par, double zbar, std::size_t i1, std::size_t i2) 
      : r_perp_t(std::get<0>(r_perp)), r_par_t(std::get<0>(r_par)), 
	r_perp_o(std::get<1>(r_perp)), r_par_o(std::get<1>(r_par)), 
	ave_zo(zbar), id1(i1), id2(i2) {}
};

std::ostream& operator<<(std::ostream &os, const Separation &s);

struct VectorSeparation {
    std::vector<Separation> seps_vec;
VectorSeparation() : seps_vec(), size_(0) {}
VectorSeparation(std::vector<Separation> s_vec) : seps_vec(s_vec), size_(s_vec.size()) {}
    void push_back(std::tuple<double, double> r_perp, std::tuple<double, double> r_par, double zbar, std::size_t i1, std::size_t i2) {
	seps_vec.push_back(Separation(r_perp, r_par, zbar, i1, i2));
	size_++;
    }
    void push_back(Separation s) {
	seps_vec.push_back(s);
	size_++;
    }
    const Separation& operator[](int i) const {
	if (i < 0) {
	    i += (int)size_;
	}
	if (i >= (int)size_) {
	    throw std::out_of_range("Index " + std::to_string(i) + " out of range for vector of length " + std::to_string(size_));
	}
	return seps_vec[i];
    }
    std::vector<double> r_perp_t() {
	std::vector<double> out(size_);
	for (std::size_t i = 0; i < size_; i++) {
	    out[i] = seps_vec[i].r_perp_t;
	}
	return out;
    }
    std::vector<double> r_par_t() {
	std::vector<double> out(size_);
	for (std::size_t i = 0; i < size_; i++) {
	    out[i] = seps_vec[i].r_par_t;
	}
	return out;
    }
    std::vector<double> r_perp_o() {
	std::vector<double> out(size_);
	for (std::size_t i = 0; i < size_; i++) {
	    out[i] = seps_vec[i].r_perp_o;
	}
	return out;
    }
    std::vector<double> r_par_o() {
	std::vector<double> out(size_);
	for (std::size_t i = 0; i < size_; i++) {
	    out[i] = seps_vec[i].r_par_o;
	}
	return out;
    }
    std::vector<double> ave_zo() {
	std::vector<double> out(size_);
	for (std::size_t i = 0; i < size_; i++) {
	    out[i] = seps_vec[i].ave_zo;
	}
	return out;
    }
    std::vector<std::size_t> id1() {
	std::vector<std::size_t> out(size_);
	for (std::size_t i = 0; i < size_; i++) {
	    out[i] = seps_vec[i].id1;
	}
	return out;
    }
    std::vector<std::size_t> id2() {
	std::vector<std::size_t> out(size_);
	for (std::size_t i = 0; i < size_; i++) {
	    out[i] = seps_vec[i].id2;
	}
	return out;
    }
    const std::size_t size() const {
	return size_;
    }
    void reserve(std::size_t new_size) {
	seps_vec.reserve(new_size);
    }
    const std::size_t max_size() const {
	return seps_vec.max_size();
    }
    void insert(VectorSeparation other) {
	seps_vec.insert(seps_vec.end(), other.seps_vec.begin(), other.seps_vec.end());
	size_ += other.size();
    }
private:
    std::size_t size_;
};

std::ostream& operator<<(std::ostream &os, const VectorSeparation &v);

double unit_dot(Pos pos1, Pos pos2);

std::tuple<double, double> dot(Pos pos1, Pos pos2);

std::tuple<double, double> r_par(Pos pos1, Pos pos2);

std::tuple<double, double> r_perp(Pos pos1, Pos pos2);

double ave_los_distance(Pos pos1, Pos pos2);

bool check_box(Pos pos1, Pos pos2, double max);

bool check_lims(double val, double min, double max);

bool check_2lims(Pos pos1, Pos pos2, double rp_min, double rp_max, double rl_min, double rl_max, bool use_true);

VectorSeparation get_separations(std::vector<Pos> pos1, std::vector<Pos> pos2, double rp_min, double rp_max, double rl_min, double rl_max, bool use_true, bool use_obs, bool is_auto);

struct BinSpecifier {
    double bin_min, bin_max, bin_size;
    std::size_t nbins;
    bool log_binning;

    BinSpecifier(double min, double max, double width, bool log_bins) {
	bin_min = min;
	bin_max = max;
	log_binning = log_bins;
	if (log_bins) {
	    nbins = (std::size_t) ceil((log(max) - log(min)) / width);
	    bin_size = (log(max) - log(min)) / (double) nbins;
	}
	else {
	    nbins = (std::size_t) ceil((max - min) / width);
	    bin_size = (max - min) / (double) nbins;
	}
    }

    BinSpecifier(double min, double max, std::size_t num_bins, bool log_bins) {
	bin_min = min;
	bin_max = max;
	log_binning = log_bins;
	nbins = num_bins;
	if (log_bins) {
	    bin_size = (log(max) - log(min)) / (double) num_bins;
	}
	else {
	    bin_size = (max - min) / (double) num_bins;
	}
    }
};

class NNCounts3D {
    BinSpecifier rpo_bins, rlo_bins, zo_bins;
    py::array_t<int> counts_;
    std::size_t n_tot_;

    int assign_1d_bin(double value, BinSpecifier binning) {
	if ((value < binning.bin_min) || (value > binning.bin_max)) {
	    return -1;
	}
	else {
	    double diff;
	    if (binning.log_binning) {
		diff = (log(value) - log(binning.bin_min));
	    }
	    else {
		diff = value - binning.bin_min;
	    }
	    return (int) floor(diff / binning.bin_size);
	}
    }

 public:
 NNCounts3D(BinSpecifier rpo_binning, BinSpecifier rlo_binning, BinSpecifier zo_binning) : rpo_bins(rpo_binning), rlo_bins(rlo_binning) zo_bins(zo_binning) {
	n_tot_ = 0;
	std::size_t size = rpo_binning.nbins * rlo_binning.nbins * zo_binning.nbins;
	int *array_ptr = new int[size];
	for (std::size_t i = 0; i < size; i++) {
	    array_ptr[i] = 0;
	}
	py::capsule free_when_done(array_ptr, [](void *f) {
		int *array_ptr = reinterpret_cast<int *>(f);
		delete[] array_ptr;
	    });
	counts_ = py::array_t<int>(
	    {rpo_binning.nbins, rlo_binning.nbins, zo_binning.nbins},
	    {rlo_binning.nbins * zo_binning.nbins * 8, zo_binning.nbins * 8, 8},
	    array_ptr,
	    free_when_done);
    }

    void assign_bin(double r_perp, double r_par, double zbar) {
	n_tot_++;
	int rpo_bin = assign_1d_bin(r_perp, rpo_bins);
	if (rpo_bin > -1) {
	    int rlo_bin = assign_1d_bin(r_par, rlo_bins);
	    if (rlo_bin > -1) {
		int zo_bin = assign_1d_bin(zbar, zo_bins);
		if (zo_bin > -1) {
		    counts_[rpo_bin, rlo_bin, zo_bin]++;
		}
	    }
	}
    }

    std::size_t n_tot() const { return n_tot_; }

    py::array_t<int> counts() const { return counts_; }

    BinSpecifier rpo_bin_info() const { return rpo_bins; }

    BinSpecifier rlo_bin_info() const { return rlo_bins; }

    BinSpecifier zo_bin_info() const { return zo_bins; }

    NNCounts3D& operator+=(const NNCounts3D& other) const {
	for (std::size_t i = 0; i < rpo_bins.nbins; i++) {
	    for (std::size_t j = 0; j < rlo_bins.nbins; j++) {
		for (std::size_t k = 0; k < zo_bins.nbins; k++) {
		    counts_[i,j,k] += other.counts_[i,j,k];
		}
	    }
	}
	n_tot_ += other.n_tot_;
	return *this;
    }
};

class NNCounts1D {
    BinSpecifier binner;
    std::size_t n_tot_;
    py::array_t<int> counts_;

 public:
 NNCounts1D(BinSpecifier binning) : binner(binning) {
	n_tot_ = 0;
	int *array_ptr = new int[binning.nbins];
	for (std::size_t i = 0; i < binning.nbins; i++) {
	    array_ptr[i] = 0;
	}
	py::capsule free_when_done(array_ptr, [](void *f) {
		int *array_ptr = reinterpret_cast<int *>(f);
		delete[] array_ptr;
	    });
	counts_ = py::array_t<int>(
	    {binning.nbins,},
	    {8,},
	    array_ptr,
	    free_when_done);
    }

    void assign_bin(double value) {
	n_tot_++;
	if (value >= binner.bin_min && value <= binner.bin_max) {
	    double diff;
	    if (binner.log_binning) {
		diff = (log(value) - log(binner.bin_min));
	    }
	    else {
		diff = value - binner.bin_min;
	    }
	    counts_[(std::size_t) floor(diff / binner.bin_size)]++;
	}
    }

    std::size_t n_tot() const { return n_tot_; }

    py::array_t<int> counts() const { return counts_; }

    BinSpecifer bin_info() const { return binner; }

    NNCounts1D& operator+=(const NNCounts1D& other) const {
	for (std::size_t i = 0; i < binner.nbins; i++) {
	    counts_[i] += other.counts_[i];
	}
	n_tot_ += other.n_tot_;
	return *this;
    }
};

#endif
