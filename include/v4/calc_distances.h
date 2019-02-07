// -*-c++-*-
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
#include <sstream>
#include <tuple>
#include <stdexcept>
#include <unordered_map>
#include <typeindex>
#include <utility>
#include <iterator>
#include <atomic>
#include "fast_math.h"

#if defined(_OPENMP) && defined(omp_num_threads)
constexpr int OMP_NUM_THREADS = omp_num_threads;
#else
constexpr int OMP_NUM_THREADS = 1;
#endif

struct SPos {
    SPos()
	: ra_(0.0), dec_(0.0), r_(0.0), z_(0.0), uvec_({0.0, 0.0, 0.0}), is_initialized_(false) {}

    SPos(const double ra, const double dec, const double r, const double z)
	: r_(r), z_(z), is_initialized_(true) {
        set_ra(ra);
        set_dec(dec);
        uvec_ = get_nxyz(ra, dec);
    }

    SPos(const SPos& other)
	: ra_(other.ra_), dec_(other.dec_), r_(other.r_), z_(other.z_), uvec_(other.uvec_), is_initialized_(other.is_initialized_) {}

    bool operator==(const SPos& other) const {
        bool same_init = is_initialized_ == other.is_initialized_;
        if (!is_initialized_) return same_init;
        bool rclose = std::isnan(r_) ? std::isnan(other.r_) : math::isclose(r_, other.r_);
        bool zclose = std::isnan(z_) ? std::isnan(other.z_) : math::isclose(z_, other.z_);
        return same_init && math::isclose(ra_, other.ra_) && math::isclose(dec_, other.dec_) && rclose && zclose;
    }

    bool operator!=(const SPos& other) const { return !(*this == other); }

    SPos& operator=(const SPos& other) {
        if (*this != other) {
            uvec_.clear();
            uvec_.resize(other.uvec_.size());
            ra_ = other.ra_;
            dec_ = other.dec_;
            r_ = other.r_;
            z_ = other.z_;
            std::copy(other.uvec_.begin(), other.uvec_.end(), uvec_.begin());
            is_initialized_ = other.is_initialized_;
        }
        return *this;
    }

    SPos& operator=(SPos&& other) noexcept {
        if (*this != other) {
            uvec_.clear();
            uvec_.shrink_to_fit();
            uvec_ = std::exchange(other.uvec_, std::vector<double>());
            ra_ = std::exchange(other.ra_, 0.0);
            dec_ = std::exchange(other.dec_, 0.0);
            r_ = std::exchange(other.r_, 0.0);
            z_ = std::exchange(other.z_, 0.0);
            is_initialized_ = std::exchange(other.is_initialized_, false);
        }
	    return *this;
    }

    double ra() const { return ra_; }

    double dec() const { return dec_; }

    double r() const { return r_; }

    double z() const { return z_; }

    std::vector<double> uvec() const { return uvec_; }

    std::vector<double> rvec() const { return std::vector<double>({r_ * uvec_[0], r_ * uvec_[1], r_ * uvec_[2]}); }

private:
    double ra_, dec_, r_, z_;
    std::vector<double> uvec_;
    bool is_initialized_;

    void set_ra(const double value) {
        if ((value < 0.0) || (value > 360.0) || std::isnan(value)) {
            throw std::invalid_argument("RA outside of valid range [0.0, 360.0]");
        }
        ra_ = value;
    }

    void set_dec(const double value) {
        if ((value < -90.0) || (value > 90.0) || std::isnan(value)) {
            throw std::invalid_argument("DEC outside of valid range [-90.0, 90.0]");
        }
        dec_ = value;
    }

    friend struct Pos;
};

struct Pos{
    Pos() {}

    Pos(double ra, double dec, double rt, double ro, double zt, double zo)
    : tpos_(SPos(ra, dec, rt, zt)), opos_(SPos(ra, dec, ro, zo)) {}

    Pos(const Pos& other)
    : tpos_(other.tpos_), opos_(other.opos_) {}

    std::vector<double> nvec() { return tpos_.uvec_; }

    std::vector<double> rtvec() { return tpos_.rvec(); }

    std::vector<double> rovec() { return opos_.rvec(); }

    double ra() const { return tpos_.ra_; }

    double dec() const { return tpos_.dec_; }

    double rt() const { return tpos_.r_; }

    double ro() const { return opos_.r_; }

    double zt() const { return tpos_.z_; }

    double zo() const { return opos_.z_; }

    SPos tpos() const { return tpos_; }

    SPos opos() const { return opos_; }

    bool has_true() const { return !std::isnan(tpos_.r_); }

    bool has_obs() const { return !std::isnan(opos_.r_); }

    bool operator==(const Pos& other) const {
        return std::isnan(tpos_.r_) == std::isnan(other.opos_.r_) && std::isnan(opos_.r_) == std::isnan(other.opos_.r_) && tpos_ == other.tpos_ && opos_ == other.opos_;
    }

    bool operator!=(const Pos& other) const { return !(*this == other); }
    
private:
    SPos tpos_, opos_;
};

std::vector<Pos> fill_catalog_vector(std::vector<double> ra_vec, std::vector<double> dec_vec, std::vector<double> rt_vec, std::vector<double> ro_vec, std::vector<double> tz_vec, std::vector<double> oz_vec);

struct Separation {
    double r_perp_t, r_par_t, r_perp_o, r_par_o, ave_zo;
    std::size_t id1, id2;
    Separation() {}
    Separation(double rpt, double rlt, double rpo, double rlo, double ave_z, std::size_t i1, std::size_t i2)
	: r_perp_t(rpt), r_par_t(rlt), r_perp_o(rpo), r_par_o(rlo), ave_zo(ave_z), id1(i1), id2(i2) {}
    Separation(std::tuple<double, double> r_perp, std::tuple<double, double> r_par, double zbar, std::size_t i1, std::size_t i2)
	: r_perp_t(std::get<0>(r_perp)), r_par_t(std::get<0>(r_par)), r_perp_o(std::get<1>(r_perp)), r_par_o(std::get<1>(r_par)), ave_zo(zbar), id1(i1), id2(i2) {}

    bool operator==(const Separation& other) const {
        bool rpt = std::isnan(r_perp_t) ? std::isnan(other.r_perp_t) : math::isclose(r_perp_t, other.r_perp_t);
        bool rlt = std::isnan(r_par_t) ? std::isnan(other.r_par_t) : math::isclose(r_par_t, other.r_par_t);
        bool rpo = std::isnan(r_perp_o) ? std::isnan(other.r_perp_o) : math::isclose(r_perp_o, other.r_perp_o);
        bool rlo = std::isnan(r_par_o) ? std::isnan(other.r_par_o) : math::isclose(r_par_o, other.r_par_o);
        bool zo = std::isnan(ave_zo) ? std::isnan(other.ave_zo) : math::isclose(ave_zo, other.ave_zo);
        return rpt && rlt && rpo && rlo && zo && id1 == other.id1 && id2 == other.id2;
    }

    bool operator!=(const Separation& other) const { return !(*this == other); }

    friend std::ostream& operator<<(std::ostream& os, const Separation& s) {
        os << s.r_perp_t << " " << s.r_par_t << " "
           << s.r_perp_o << " " << s.r_par_o << " "
           << s.ave_zo << " " << s.id1 << " " << s.id2;
        return os;
    }
};

struct VectorSeparation {
    std::vector<Separation> seps_vec;
    VectorSeparation()
	: seps_vec(), size_(0) {}
    VectorSeparation(std::vector<Separation> s_vec)
	: seps_vec(s_vec), size_(s_vec.size()) {}
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

    friend std::ostream& operator<<(std::ostream& os, const VectorSeparation& v) {
	auto vec = v.seps_vec;
	for (const auto& this_v : vec) {
	    os << this_v;
	    if (vec.back() != this_v) { os << std::endl; }
	}
	return os;
    }

    bool operator==(const VectorSeparation& other) const {
	for (auto sep : other.seps_vec) {
	    if (std::find(seps_vec.begin(), seps_vec.end(), sep) == seps_vec.end()) return false;
	}
	return true;
    }

    bool operator!=(const VectorSeparation& other) const { return !(*this == other); }

private:
    std::size_t size_;
};

std::ostream& operator<<(std::ostream &os, const VectorSeparation &v);

double unit_dot(SPos pos1, SPos pos2);

double unit_dot(Pos pos1, Pos pos2);

double dot(SPos pos1, SPos pos2);

std::tuple<double, double> dot(Pos pos1, Pos pos2);

double r_par(SPos pos1, SPos pos2);

std::tuple<double, double> r_par(Pos pos1, Pos pos2);

double r_perp(SPos pos1, SPos pos2);

std::tuple<double, double> r_perp(Pos pos1, Pos pos2);

double ave_z(SPos pos1, SPos pos2);

std::tuple<double, double> ave_z(Pos pos1, Pos pos2);

double ave_los_distance(Pos pos1, Pos pos2);

bool check_sphere(SPos pos1, SPos pos2, double max);

bool check_sphere(Pos pos1, Pos pos2, double max);

bool check_shell(SPos pos1, SPos pos2, double min, double max);

bool check_shell(Pos pos1, Pos pos2, double min, double max);

bool check_lims(double val, double min, double max);

bool check_2lims(SPos pos1, SPos pos2, double rp_min, double rp_max, double rl_min, double rl_max);

bool check_2lims(Pos pos1, Pos pos2, double rp_min, double rp_max, double rl_min, double rl_max, bool use_true);

VectorSeparation get_separations(std::vector<Pos> pos1, std::vector<Pos> pos2, double rp_min, double rp_max, double rl_min, double rl_max, bool use_true, bool use_obs, bool is_auto);

struct BinSpecifier {
private:
    double bin_min, bin_max, bin_size;
    std::size_t nbins;
    bool log_binning, _is_set;
    // These are only internal variables so I know what else I need to update
    // when a single parameter is changed
    bool _min_set, _max_set, _size_set, _nbins_set, _log_set;

    void get_nbins_from_size(const double size) {
	if (log_binning) {
	    // Assume the size is already given in log-space
	    nbins = (std::size_t) ceil((log(bin_max) - log(bin_min)) / size);
	}
	else {
	    nbins = (std::size_t) ceil((bin_max - bin_min) / size);
	}
	_nbins_set = true;
    }

    void get_size_from_nbins() {
	if (log_binning) {
	    // Store the size in log-space
	    bin_size = (log(bin_max) - log(bin_min)) / (double) nbins;
	}
	else {
	    bin_size = (bin_max - bin_min) / (double) nbins;
	}
	_size_set = true;
    }

    void on_update() {
	if (_min_set && bin_min <= 0.0) {
	    log_binning = false;
	    _log_set = true;
	}
	if (_log_set && _min_set && _max_set && (_size_set || _nbins_set)) {
	    if (_nbins_set) {
		get_size_from_nbins();
	    }
	    else {
		double old_size = bin_size;
		get_nbins_from_size(old_size);
	    }
	}

	_is_set = (_min_set && _max_set && _size_set && _nbins_set && _log_set);
    }

public:
    BinSpecifier(double min, double max, double width, bool log_bins) {
	bin_min = min;
	_min_set = true;
	bin_max = max;
	_max_set = true;
	log_binning = log_bins;
	_log_set = true;
	bin_size = width;
	_size_set = true;
	on_update();
    }

    BinSpecifier(double min, double max, std::size_t num_bins, bool log_bins) {
	bin_min = min;
	_min_set = true;
	bin_max = max;
	_max_set = true;
	log_binning = log_bins;
	_log_set = true;
	nbins = num_bins;
	_nbins_set = true;
	on_update();
    }

    // copy constructor
    BinSpecifier(const BinSpecifier& other) {
	_min_set = other._min_set;
	_max_set = other._max_set;
	_size_set = other._size_set;
	_nbins_set = other._nbins_set;
	_log_set = other._log_set;
	bin_min = other.bin_min;
	bin_max = other.bin_max;
	log_binning = other.log_binning;
	nbins = other.nbins;
	bin_size = other.bin_size;
	on_update();
    }

    // default constructor
    BinSpecifier() {
	_min_set = false;
	_max_set = false;
	_size_set = false;
	_nbins_set = false;
	_log_set = false;
	bin_min = 0.0;
	bin_max = 0.0;
	log_binning = false;
	nbins = 0;
	bin_size = 0.0;
	on_update();
    }

    // Instead of copying everything with this one, just update the parameters
    // in this that are set in other. With this function, we prefer values in
    // other. See 'fill' for a version that prefers valuse in this
    void update(const BinSpecifier& other) {
	if (other._min_set) {
	    bin_min = other.bin_min;
	    _min_set = true;
	}
	if (other._max_set) {
	    bin_max = other.bin_max;
	    _max_set = true;
	}
	if (other._log_set) {
	    log_binning = other.log_binning;
	    _log_set = true;
	}
	if (other._nbins_set) {
	    nbins = other.nbins;
	    _nbins_set = true;
	}
	if (other._size_set) {
	    bin_size = other.bin_size;
	    _size_set = true;
	}
	on_update();
    }

    // This function updates values of this from values of other, but prefers
    // values of this when both are set. See 'update' for a version that prefers
    // values of other instead
    void fill(const BinSpecifier& other) {
	if (other._min_set && !_min_set) {
	    bin_min = other.bin_min;
	    _min_set = true;
	}
	if (other._max_set && !_max_set) {
	    bin_max = other.bin_max;
	    _max_set = true;
	}
	if (other._log_set && !_log_set) {
	    log_binning = other.log_binning;
	    _log_set = true;
	}
	if (other._nbins_set && !_nbins_set) {
	    nbins = other.nbins;
	    _nbins_set = true;
	}
	if (other._size_set && !_size_set) {
	    bin_size = other.bin_size;
	    _size_set = true;
	}
	on_update();
    }

    std::string toString() const {
	std::ostringstream oss;
	oss << std::boolalpha << "BinSpecifier(is_set=" << _is_set << ", bin_min=" << bin_min << ", bin_max=" << bin_max << ", bin_size=" << bin_size << ", nbins="  << nbins << ", log_binning=" << log_binning << ")" << std::noboolalpha;
	return oss.str();
    }

    const bool is_set() const {
	return _is_set;
    }

    const double get_bin_min() const {
	return bin_min;
    }

    void set_bin_min(const double min) {
	bin_min = min;
	_min_set = true;
	on_update();
    }

    const double get_bin_max() const {
	return bin_max;
    }

    void set_bin_max(const double max) {
	bin_max = max;
	_max_set = true;
	on_update();
    }

    const double get_bin_size() const {
	return bin_size;
    }

    void set_bin_size(const double size) {
	bin_size = size;
	_size_set = true;
	on_update();
    }

    const std::size_t get_nbins() const {
	return nbins;
    }

    void set_nbins(const std::size_t num_bins) {
	nbins = num_bins;
	_nbins_set = true;
	on_update();
    }

    const bool get_log_binning() const {
	return log_binning;
    }

    void set_log_binning(const bool log_bins) {
	log_binning = log_bins;
	_log_set = true;
	on_update();
    }

    bool operator==(const BinSpecifier &other) const {
	return ((_is_set == other._is_set) && (log_binning == other.log_binning) && (nbins == other.nbins) && math::isclose(bin_size, other.bin_size) && math::isclose(bin_min, other.bin_min) && math::isclose(bin_max, other.bin_max));
    }

    bool operator!=(const BinSpecifier &other) const { return !(*this == other); }

    int assign_bin(double value) {
	if ((value < bin_min) || (value > bin_max)) return -1;
	double diff;
	if (log_binning) diff = log(value) - log(bin_min);
	else diff = value - bin_min;
	return (int) floor(diff / bin_size);
    }

    std::vector<double> lower_bin_edges() const {
	std::vector<double> edges(nbins, bin_min);
	for (std::size_t i = 0; i < nbins; i++) {
	    edges[i] += (i * bin_size);
	}
	return edges;
    }

    std::vector<double> upper_bin_edges() const {
	std::vector<double> edges(nbins, bin_max);
	for (std::size_t i = 0; i < nbins; i++) {
	    edges[i] -= (i * bin_size);
	}
	std::reverse(edges.begin(), edges.end());
	return edges;
    }
};

std::size_t get_1d_indexer_from_3d(std::size_t x_idx, std::size_t y_idx, std::size_t z_idx, BinSpecifier x_bins, BinSpecifier y_bins, BinSpecifier z_bins);

class NNCounts3D {
    BinSpecifier rpo_bins, rlo_bins, zo_bins;
    std::vector<int> counts_;
    std::size_t n_tot_, max_index_;

    void on_bin_update() {
	n_tot_ = 0;
	max_index_ = rpo_bins.get_nbins() * rlo_bins.get_nbins() * zo_bins.get_nbins();
	std::vector<int>empty(max_index_, 0);
	counts_.swap(empty);
    }

 public:
    // (default) empty constructor
    NNCounts3D() {}

    // copy constructor
    NNCounts3D(const NNCounts3D& other) {
	rpo_bins = other.rpo_bins;
	rlo_bins = other.rlo_bins;
	zo_bins = other.zo_bins;
	on_bin_update();
	counts_ = other.counts_;
	n_tot_ = other.n_tot_;
    }

    // Like a copy constructor, but from pickled objects (for python)
    NNCounts3D(BinSpecifier rpo_binning, BinSpecifier rlo_binning, BinSpecifier zo_binning, std::vector<int> counts, std::size_t n_tot) {
	rpo_bins = rpo_binning;
	rlo_bins = rlo_binning;
	zo_bins = zo_binning;
	on_bin_update();
	counts_ = counts;
	n_tot_ = n_tot;
    }

    NNCounts3D(BinSpecifier rpo_binning, BinSpecifier rlo_binning, BinSpecifier zo_binning) {
	rpo_bins = rpo_binning;
	rlo_bins = rlo_binning;
	zo_bins = zo_binning;
	on_bin_update();
    }

    void update_rpo_binning(BinSpecifier new_binning, bool prefer_old=true) {
	if (prefer_old) {
	    rpo_bins.fill(new_binning);
	}
	else {
	    rpo_bins.update(new_binning);
	}
	on_bin_update();
    }

    void update_rlo_binning(BinSpecifier new_binning, bool prefer_old=true) {
	if (prefer_old) {
	    rlo_bins.fill(new_binning);
	}
	else {
	    rlo_bins.update(new_binning);
	}
	on_bin_update();
    }

    void update_zo_binning(BinSpecifier new_binning, bool prefer_old=true) {
	if (prefer_old) {
	    zo_bins.fill(new_binning);
	}
	else {
	    zo_bins.update(new_binning);
	}
	on_bin_update();
    }

    std::size_t get_1d_indexer(std::size_t x_idx, std::size_t y_idx, std::size_t z_idx) {
	return get_1d_indexer_from_3d(x_idx, y_idx, z_idx, rpo_bins, rlo_bins, zo_bins);
    }

    int get_bin(double r_perp, double r_par, double zbar) {
	int rpo_bin = rpo_bins.assign_bin(r_perp);
	if (rpo_bin == -1) return -1;
	int rlo_bin = rlo_bins.assign_bin(r_par);
	if (rlo_bin == -1) return -1;
	int zo_bin = zo_bins.assign_bin(zbar);
	if (zo_bin == -1) return -1;
	return get_1d_indexer(rpo_bin, rlo_bin, zo_bin);
    }

    void assign_bin(double r_perp, double r_par, double zbar) {
	n_tot_++;
	int bin_index = get_bin(r_perp, r_par, zbar);
	if (bin_index > -1) counts_[bin_index]++;
    }

    const int operator[](std::size_t idx) const { return counts_[idx]; }

    std::size_t n_tot() const { return n_tot_; }

    std::vector<int> counts() const { return counts_; }

    BinSpecifier rpo_bin_info() const { return rpo_bins; }

    BinSpecifier rlo_bin_info() const { return rlo_bins; }

    BinSpecifier zo_bin_info() const { return zo_bins; }

    NNCounts3D& operator+=(const NNCounts3D& other) {
	if (rpo_bins != other.rpo_bins) {
	    std::cerr << "Attempted to combine NNCounts3D instances with different rpo_bins" << std::endl;
	    std::cerr << "this.rpo_bins: " << rpo_bins.toString() << std::endl;
	    std::cerr << "other.rpo_bins: " << other.rpo_bins.toString() << std::endl;
	    throw std::runtime_error("Cannot combine NNCounts3D instances with different perpendicular binning schemes");
	}
	if (rlo_bins != other.rlo_bins) {
	    std::cerr << "Attempted to combine NNCounts3D instances with different rlo_bins" << std::endl;
	    std::cerr << "this.rlo_bins: " << rlo_bins.toString() << std::endl;
	    std::cerr << "other.rlo_bins: " << other.rlo_bins.toString() << std::endl;
	    throw std::runtime_error("Cannot combine NNCounts3D instances with different parallel binning schemes");
	}
	if (zo_bins != other.zo_bins) {
	    std::cerr << "Attempted to combine NNCounts3D instances with different zo_bins" << std::endl;
	    std::cerr << "this.zo_bins: " << zo_bins.toString() << std::endl;
	    std::cerr << "other.zo_bins: " << other.zo_bins.toString() << std::endl;
	    throw std::runtime_error("Cannot combine NNCounts3D instances with different redshift binning schemes");
	}
	n_tot_ += other.n_tot_;
	std::transform(counts_.begin(), counts_.end(), other.counts_.begin(), counts_.begin(), std::plus<int>());
	return *this;
    }

    template<typename T>
    typename std::enable_if_t<std::is_arithmetic<T>::value, NNCounts3D&>
    operator+=(const T& x) {
	if (!math::isclose(x, (T)0)) {
	    throw std::invalid_argument("Only 0 valid for scalar addition with NNCounts3D");
	}
	return *this;
    }

    friend NNCounts3D operator+(const NNCounts3D& lhs, const NNCounts3D& rhs) {
	return NNCounts3D(lhs) += rhs;
    }

    template<typename T>
    friend typename std::enable_if_t<std::is_arithmetic<T>::value, NNCounts3D>
    operator+(const NNCounts3D& lhs, const T& rhs) {
	return NNCounts3D(lhs) += rhs;
    }

    template<typename T>
    friend typename std::enable_if_t<std::is_arithmetic<T>::value, NNCounts3D>
    operator+(const T& lhs, const NNCounts3D& rhs) {
	return NNCounts3D(rhs) += lhs;
    }

    std::string toString() {
	std::ostringstream oss;
	std::string init_string = "NNCounts3D(";
	std::string pad_string;
	for (std::size_t i = 0; i < init_string.size(); i++) {
	    pad_string += " ";
	}
	oss << init_string << std::endl << pad_string << "rpo_bins=" << rpo_bins.toString() << "," << std::endl << pad_string << "rlo_bins=" << rlo_bins.toString() << "," << std::endl << pad_string << "zo_bins=" << zo_bins.toString() << std::endl << pad_string << ")";
	return oss.str();
    }
};

class NNCounts1D {
    BinSpecifier binner;
    std::size_t n_tot_, max_index_;
    std::vector<int> counts_;

    void on_bin_update() {
	n_tot_ = 0;
	max_index_ = binner.get_nbins();
	std::vector<int> temp(max_index_, 0);
	counts_.swap(temp);
    }

 public:
    // (default) empty constructor
    NNCounts1D() {}

    // copy constructor
    NNCounts1D(const NNCounts1D& other) {
	binner = other.binner;
	n_tot_ = other.n_tot_;
	max_index_ = other.max_index_;
	counts_ = other.counts_;
    }

    // Like the copy constructor, but from pickled objects (for python)
    NNCounts1D(BinSpecifier binning, std::vector<int> counts, std::size_t n_tot) {
	binner = binning;
	on_bin_update();
	counts_ = counts;
	n_tot_ = n_tot;
    }

    NNCounts1D(BinSpecifier binning) {
	binner = binning;
	on_bin_update();
    }

    void update_binning(BinSpecifier new_binning, bool prefer_old=true) {
	if (prefer_old) {
	    binner.fill(new_binning);
	}
	else {
	    binner.update(new_binning);
	}
	on_bin_update();
    }

    int get_bin(double value) { return binner.assign_bin(value); }

    void assign_bin(double value) {
	n_tot_++;
	int bin_index = binner.assign_bin(value);
	if (bin_index > -1) counts_[bin_index]++;
    }

    const int operator[](std::size_t idx) const { return counts_[idx]; }

    std::size_t n_tot() const { return n_tot_; }

    std::vector<int> counts() const { return counts_; }

    BinSpecifier bin_info() const { return binner; }

    NNCounts1D& operator+=(const NNCounts1D& other) {
	if (binner != other.binner) {
	    std::cerr << "Attempted to combine NNCounts1D instances with different binning" << std::endl;
	    std::cerr << "this.binner: " << binner.toString() << std::endl;
	    std::cerr << "other.binner: " << other.binner.toString() << std::endl;
	    throw std::runtime_error("Cannot combine NNCounts1D instances with different binning schemes");
	}
	n_tot_ += other.n_tot_;
	std::transform(counts_.begin(), counts_.end(), other.counts_.begin(), counts_.begin(), std::plus<int>());
	return *this;
    }

    template<typename T>
    typename std::enable_if_t<std::is_arithmetic<T>::value, NNCounts1D&>
    operator+=(const T& x) {
	if (!math::isclose(x, (T)0)) {
	    throw std::invalid_argument("Only 0 valid for scalar addition with NNCounts1D");
	}
	return *this;
    }

    friend NNCounts1D operator+(const NNCounts1D& lhs, const NNCounts1D& rhs) {
	return NNCounts1D(lhs) += rhs;
    }

    template<typename T>
    friend typename std::enable_if_t<std::is_arithmetic<T>::value, NNCounts1D>
    operator+(const NNCounts1D& lhs, const T& rhs) {
	return NNCounts1D(lhs) += rhs;
    }

    template<typename T>
    friend typename std::enable_if_t<std::is_arithmetic<T>::value, NNCounts1D>
    operator+(const T& lhs, const NNCounts1D& rhs) {
	return NNCounts1D(rhs) += lhs;
    }

    std::string toString() {
	return "NNCounts1D(bins=" + binner.toString() + ")";
    }
};

NNCounts3D get_obs_pair_counts(std::vector<SPos> pos1, std::vector<SPos> pos2, BinSpecifier rpo_binning, BinSpecifier rlo_binning, BinSpecifier zo_binning, bool is_auto);

NNCounts3D get_obs_pair_counts(std::vector<Pos> pos1, std::vector<Pos> pos2, BinSpecifier rpo_binning, BinSpecifier rlo_binning, BinSpecifier zo_binning, bool is_auto);

NNCounts1D get_true_pair_counts(std::vector<Pos> pos1, std::vector<Pos> pos2, BinSpecifier r_binning, bool is_auto, bool use_true=true);

#endif
