// -*-c++-*-
#ifndef CALC_DISTANCES_H
#define CALC_DISTANCES_H

#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <algorithm>
#include <numeric>
#include <cstdarg>
#include <cstring>
#include <string>
#include <sstream>
#include <tuple>
#include <array>
#include <stdexcept>
#include <unordered_map>
#include <typeindex>
#include <utility>
#include <iterator>
#include <atomic>
#include "fast_math.h"

#if defined(_OPENMP) && defined(omp_num_threads)
constexpr int OMP_NUM_THREADS = omp_num_threads;
#include <omp.h>
#else
constexpr bool _OPENMP = false;
constexpr int OMP_NUM_THREADS = 1;
typedef int omp_int_t;
inline void omp_set_num_threads(int) {}
inline omp_int_t omp_get_num_threads() { return 1; }
#endif

template<typename T>
inline std::enable_if_t<std::is_arithmetic<T>::value, bool>
check_val_in_limits(const T& val, const T& min, const T& max) {
    return std::isfinite(val) && (val >= min) && (val <= max);
}

template<typename T, std::size_t N>
std::array<T,N> make_filled_array(const T& value) {
    std::array<T,N> arr;
    arr.fill(value);
    return arr;
}

template<typename T, std::size_t N, typename std::enable_if_t<std::is_arithmetic<T>::value, int> = 0>
std::array<T,N> make_filled_array() {
    return make_filled_array<T,N>((T)0);
}

template<typename T, std::size_t N, typename std::enable_if_t<!std::is_arithmetic<T>::value, int> = 0>
std::array<T,N> make_filled_array() {
    return make_filled_array<T,N>(T());
}

// Forward declare some classes to be friends with BinSpecifier
template<std::size_t N>
class NNCountsNDBase;
template<std::size_t N>
class NNCountsND;
template<std::size_t N>
class ExpectedNNCountsNDBase;
template<std::size_t N>
class ExpectedNNCountsND;

struct BinSpecifier {
private:
    bool _min_set = false;
    bool _max_set = false;
    bool _size_set = false;
    bool _nbins_set = false;
    bool _log_set = false;
    double bin_min = 0.0;
    double bin_max = 0.0;
    bool log_binning = false;
    double bin_size = 0.0;
    std::size_t nbins = 0;
    bool _is_set = false;

    void get_nbins_from_size(double size) {
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
    // empty constructor
    BinSpecifier() = default;

    // copy constructor
    BinSpecifier(const BinSpecifier&) = default;

    // move constructor
    BinSpecifier(BinSpecifier&&) = default;

    BinSpecifier(double min, double max, double width, bool log_bins)
	: _min_set(true), _max_set(true), _size_set(true), _nbins_set(false), _log_set(true), bin_min(min), bin_max(max), log_binning(log_bins), bin_size(width) {
	on_update();
    }

    BinSpecifier(double min, double max, std::size_t num_bins, bool log_bins)
	: _min_set(true), _max_set(true), _size_set(false), _nbins_set(true), _log_set(true), bin_min(min), bin_max(max), log_binning(log_bins), nbins(num_bins) {
	on_update();
    }

    // copy assignment operator
    BinSpecifier& operator=(const BinSpecifier&) = default;

    // move assignment operator
    BinSpecifier& operator=(BinSpecifier&&) = default;

    bool operator==(const BinSpecifier &other) const {
	return ((_is_set == other._is_set) && (log_binning == other.log_binning) && (nbins == other.nbins) && math::isclose(bin_size, other.bin_size) && math::isclose(bin_min, other.bin_min) && math::isclose(bin_max, other.bin_max));
    }

    bool operator!=(const BinSpecifier &other) const { return !(*this == other); }

    // Instead of copying everything with this one, just update the parameters
    // in this that are set in other. With this function, we prefer values in
    // other. See 'fill' for a version that prefers valuse in this
    void update(const BinSpecifier& other) {
	if (operator==(other)) return;  // other is the same as this already
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
	if (operator==(other)) return;  // other is the same as this already
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
	oss << std::boolalpha;
	oss << "BinSpecifier(";
	oss << "is_set=" << _is_set;
	oss << ", min=" << bin_min;
	oss << ", max=" << bin_max;
	oss << ", size=" << bin_size;
	oss << ", nbins=" << nbins;
	oss << ", log=" << log_binning;
	oss << ")" << std::noboolalpha;
	return oss.str();
    }

    friend std::ostream& operator<<(std::ostream& os, const BinSpecifier& b) {
	os << b.toString();
	return os;
    }

    bool is_set() const {
	return _is_set;
    }

    double get_bin_min() const {
	return bin_min;
    }

    void set_bin_min(double min) {
	bin_min = min;
	_min_set = true;
	on_update();
    }

    double get_bin_max() const {
	return bin_max;
    }

    void set_bin_max(double max) {
	bin_max = max;
	_max_set = true;
	on_update();
    }

    double get_bin_size() const {
	return bin_size;
    }

    void set_bin_size(double size) {
	bin_size = size;
	_size_set = true;
	on_update();
    }

    std::size_t get_nbins() const {
	return nbins;
    }

    void set_nbins(std::size_t num_bins) {
	nbins = num_bins;
	_nbins_set = true;
	on_update();
    }

    bool get_log_binning() const {
	return log_binning;
    }

    void set_log_binning(bool log_bins) {
	log_binning = log_bins;
	_log_set = true;
	on_update();
    }

    int assign_bin(double value) const {
	if (!_is_set) throw std::runtime_error("Cannot assign bin if values are not set");
	if (operator==(BinSpecifier())) return -1;
	if ((value < bin_min) || (value > bin_max)) return -1;
	double diff = log_binning ? log(value / bin_min) : value - bin_min;
	return (int) floor(diff / bin_size);
    }

    std::vector<double> lower_bin_edges() const {
	if (!_is_set) return {};
	std::vector<double> edges(nbins, bin_min);
	for (std::size_t i = 0; i < nbins; i++) {
	    if (log_binning) edges[i] *= std::exp(i * bin_size);
	    else edges[i] += (i * bin_size);
	}
	return edges;
    }

    std::vector<double> upper_bin_edges() const {
	if (!_is_set) return {};
	std::vector<double> edges(nbins, bin_min);
	for (std::size_t i = 0; i < nbins; i++) {
	    if (log_binning) edges[i] *= std::exp((i + 1) * bin_size);
	    else edges[i] += ((i + 1) * bin_size);
	}
	return edges;
    }

    std::vector<double> bin_edges() const {
	if (!_is_set) return {};
	std::vector<double> edges(nbins+1, bin_min);
	for (std::size_t i = 0; i <= nbins; i++) {
	    if (log_binning) edges[i] *= std::exp(i * bin_size);
	    else edges[i] += (i * bin_size);
	}
	return edges;
    }

    std::vector<double> bin_centers() const {
	if (!_is_set) return {};
	std::vector<double> bins(nbins, bin_min);
	for (std::size_t i = 0; i < nbins; i++) {
	    if (log_binning) bins[i] *= std::exp((i + 0.5) * bin_size);
	    else bins[i] += ((i + 0.5) * bin_size);
	}
	return bins;
    }

    std::vector<double> bin_widths() const {
	if (!_is_set) return {};
	std::vector<double> widths;
	if (!log_binning) widths.assign(nbins, bin_size);
	else {
	    widths.assign(nbins, bin_min * (std::exp(bin_size) - 1));
	    for (std::size_t i = 0; i < nbins; i++) {
		widths[i] *= std::exp(i * bin_size);
	    }
	}
	return widths;
    }

private:
    template<std::size_t> friend class NNCountsNDBase;
    template<std::size_t> friend class NNCountsND;
    template<std::size_t> friend class ExpectedNNCountsNDBase;
    template<std::size_t> friend class ExpectedNNCountsND;
};

inline double get_r_min(const BinSpecifier& rp_binner, const BinSpecifier& rl_binner) {
    return math::isclose(rp_binner.get_bin_min(), 0.0) && math::isclose(rl_binner.get_bin_min(), 0.0) ? 0.0 : std::sqrt(math::square(rp_binner.get_bin_min()) + math::square(rl_binner.get_bin_min()));
}

inline double get_r_max(const BinSpecifier& rp_binner, const BinSpecifier& rl_binner) {
    return std::isinf(rp_binner.get_bin_max()) || std::isinf(rl_binner.get_bin_max()) ? std::numeric_limits<double>::max() : std::sqrt(math::square(rp_binner.get_bin_max()) + math::square(rl_binner.get_bin_max()));
}

template<typename T, std::size_t N>
std::array<T,N> multiply_array_by_constant(std::array<T,N> arr, T value) {
    std::array<T,N> ret;
    std::transform(arr.begin(), arr.end(), ret.begin(), std::bind1st(std::multiplies<T>(), value));
    return ret;
}

struct SPos {
private:
    double ra_ = 0.0;
    double dec_ = 0.0;
    double r_ = math::dnan;
    double z_ = math::dnan;
    std::array<double,3> uvec_ = make_filled_array<double,3>(0.0);
    std::array<double,3> rvec_ = make_filled_array<double,3>(math::dnan);
    bool is_initialized_ = false;
public:
    // empty constructor
    SPos() = default;

    // copy constructor
    SPos(const SPos&) = default;

    // move constructor
    SPos(SPos&&) = default;

    SPos(double ra, double dec, double r, double z)
	: ra_(check_ra(ra)), dec_(check_dec(dec)), r_(r), z_(z), uvec_(get_nxyz_array(ra_, dec_)), rvec_(multiply_array_by_constant(uvec_, r_)), is_initialized_(true) {}

    bool operator==(const SPos& other) const {
        bool same_init = is_initialized_ == other.is_initialized_;
        if (!is_initialized_) return same_init;
        bool rclose = std::isnan(r_) ? std::isnan(other.r_) : math::isclose(r_, other.r_);
        bool zclose = std::isnan(z_) ? std::isnan(other.z_) : math::isclose(z_, other.z_);
        return same_init && math::isclose(ra_, other.ra_) && math::isclose(dec_, other.dec_) && rclose && zclose;
    }

    bool operator!=(const SPos& other) const { return !operator==(other); }

    // copy assignment
    SPos& operator=(const SPos&) = default;

    // move assignment
    SPos& operator=(SPos&&) = default;

    double ra() const { return ra_; }

    double dec() const { return dec_; }

    double r() const { return r_; }

    double z() const { return z_; }

    std::array<double,3> uvec() const { return uvec_; }

    std::array<double,3> rvec() const { return rvec_; }

    double dot_norm(const SPos& other) const { return operator==(other) ? 1.0 : math::isclose(other.uvec_, uvec_) ? 1.0 : (uvec_[0] * other.uvec_[0]) + (uvec_[1] * other.uvec_[1]) + (uvec_[2] * other.uvec_[2]); }

    double dot_mag(const SPos& other) const { return r_ * other.r_ * dot_norm(other); }

    double distance_zbar(const SPos& other) const { return std::isnan(z_) || std::isnan(other.z_) ? math::dnan : 0.5 * (z_ + other.z_); }

    double distance_par(const SPos& other) const { return std::isnan(r_) || std::isnan(other.r_) ? math::dnan : math::dsqrt_2 * std::sqrt(1.0 + dot_norm(other)) * std::fabs(r_ - other.r_); }

    double distance_perp(const SPos& other) const { return std::isnan(r_) || std::isnan(other.r_) ? math::dnan : math::dsqrt_2 * std::sqrt(1.0 - dot_norm(other)) * (r_ + other.r_); }

    std::array<double,3> distance_vector(const SPos& other) const { return operator==(other) ? (std::array<double,3>){{0.0, 0.0, 0.0}} : (std::array<double,3>){{rvec_[0] - other.rvec_[0], rvec_[1] - other.rvec_[1], rvec_[2] - other.rvec_[2]}}; }

    double distance_magnitude(const SPos& other) const {
	if (operator==(other)) return 0.0;
	auto dist_vec = distance_vector(other);
	return std::sqrt(math::square(dist_vec[0]) + math::square(dist_vec[1]) + math::square(dist_vec[2]));
    }


    bool check_shell(const SPos& other, double min, double max) const {
	for (auto d : distance_vector(other)) {
	    if (std::fabs(d) < min || std::fabs(d) > max) return false;
	}
	return true;
    }

    bool check_shell(const SPos& other, const BinSpecifier& binner) const { return check_shell(other, binner.get_bin_min(), binner.get_bin_max()); }

    bool check_shell(const SPos& other, double max) const { return check_shell(other, 0.0, max); }


    bool check_limits(const SPos& other, double rp_min, double rp_max, double rl_min, double rl_max) const {
	if (!check_val_in_limits(distance_perp(other), rp_min, rp_max)) return false;
	return check_val_in_limits(distance_par(other), rl_min, rl_max);
    }

    bool check_limits(const SPos& other, const BinSpecifier& rp_binner, const BinSpecifier& rl_binner) const { return check_limits(other, rp_binner.get_bin_min(), rp_binner.get_bin_max(), rl_binner.get_bin_min(), rl_binner.get_bin_max()); }

    std::string toString() const {
	std::ostringstream oss;
	oss << "SPos(";
	if (is_initialized_) {
	    oss << "ra = " << ra_;
	    oss << ", dec = " << dec_;
	    oss << ", r = ";
	    if (math::isnan(r_)) oss << math::snan;
	    else oss << r_;
	    oss << ", z = ";
	    if (math::isnan(z_)) oss << math::snan;
	    else oss << z_;
	}
	oss << ")";
	return oss.str();
    }

    friend std::ostream& operator<<(std::ostream& os, const SPos& s) {
	os << s.toString();
	return os;
    }

private:
    void check(double value, double min, double max, std::string&& arg_name) {
	if (value < min || value > max || std::isnan(value)) throw std::invalid_argument(arg_name + " value " + std::to_string(value) + " outside of allowed range [" + std::to_string(min) + "," + std::to_string(max) + "]");
    }

    double check_ra(double value) {
	check(value, 0.0, 360.0, "RA");
	return value;
    }

    double check_dec(double value) {
	check(value, -90.0, 90.0, "DEC");
	return value;
    }

    friend struct Pos;
};

struct Pos{
private:
    SPos tpos_, opos_;
public:
    // empty constructor
    Pos() = default;

    // copy constructor
    Pos(const Pos&) = default;

    // move constructor
    Pos(Pos&&) = default;

    Pos(double ra, double dec, double rt, double ro, double zt, double zo)
	: tpos_(SPos(ra, dec, rt, zt)), opos_(SPos(ra, dec, ro, zo)) {}

    // copy assignment
    Pos& operator=(const Pos&) = default;

    // move assignment
    Pos& operator=(Pos&&) = default;

    std::array<double,3> uvec() const { return tpos_.uvec_; }

    std::array<double,3> rtvec() const { return tpos_.rvec_; }

    std::array<double,3> rovec() const { return opos_.rvec_; }

    double ra() const { return tpos_.ra_; }

    double dec() const { return tpos_.dec_; }

    double rt() const { return tpos_.r_; }

    double ro() const { return opos_.r_; }

    double zt() const { return tpos_.z_; }

    double zo() const { return opos_.z_; }

    SPos tpos() const { return tpos_; }

    SPos opos() const { return opos_; }

    bool has_true() const { return tpos_.is_initialized_ && !std::isnan(tpos_.r_); }

    bool has_obs() const { return opos_.is_initialized_ && !std::isnan(opos_.r_); }

    bool operator==(const Pos& other) const {
        return std::isnan(tpos_.r_) == std::isnan(other.opos_.r_) && std::isnan(opos_.r_) == std::isnan(other.opos_.r_) && tpos_ == other.tpos_ && opos_ == other.opos_;
    }

    bool operator!=(const Pos& other) const { return !(*this == other); }

    double dot_norm(const SPos& other) const {
	if (!other.is_initialized_) throw std::runtime_error("Cannot take dot product with unset position");
	if (!(tpos_.is_initialized_ || opos_.is_initialized_)) throw std::runtime_error("Cannot take dot product when no positions are set in self");
	return tpos_.is_initialized_ ? tpos_.dot_norm(other) : opos_.dot_norm(other);
    }

    double dot_norm(const Pos& other) const {
	if (!(other.tpos_.is_initialized_ || other.opos_.is_initialized_)) throw std::runtime_error("Cannot take dot product with unset true and observed positions");
	return other.tpos_.is_initialized_ ? dot_norm(other.tpos_) : dot_norm(other.opos_);
    }

    double dot_mag(const SPos& other) const {
	if (!other.is_initialized_ || math::isnan(other.r_)) throw std::runtime_error("Cannot take dot product with unset position or NaN distance");
	if (!((tpos_.is_initialized_ && !math::isnan(tpos_.r_)) || (opos_.is_initialized_ && !std::isnan(opos_.r_)))) throw std::runtime_error("Cannot take dot product when no positions without NaN distances are set in self");
	return tpos_.is_initialized_ && !math::isnan(tpos_.r_) ? tpos_.dot_mag(other) : opos_.dot_mag(other);
    }

    double dot_mag(const Pos& other) const {
	if (!((other.tpos_.is_initialized_ && !math::isnan(other.tpos_.r_)) || (other.opos_.is_initialized_ && !math::isnan(other.opos_.r_)))) throw std::runtime_error("Cannot take dot product with invalid true and observed positions");
	return other.tpos_.is_initialized_ && !math::isnan(other.tpos_.r_) ? dot_mag(other.tpos_) : dot_mag(other.opos_);
    }

    double zbar_t(const SPos& other) const { return tpos_.distance_zbar(other); }

    double zbar_t(const Pos& other) const { return zbar_t(other.tpos_); }

    double zbar_o(const SPos& other) const { return opos_.distance_zbar(other); }

    double zbar_o(const Pos& other) const { return zbar_o(other.opos_); }

    std::tuple<double, double> distance_zbar(const Pos& other) const { return std::make_tuple(zbar_t(other), zbar_o(other)); }

    double r_par_t(const SPos& other) const { return tpos_.distance_par(other); }

    double r_par_t(const Pos& other) const { return r_par_t(other.tpos_); }

    double r_par_t_signed(const Pos& other) const { return (has_obs() && other.has_obs() ? math::signof(opos_.r_ - other.opos_.r_) * math::signof(tpos_.r_ - other.tpos_.r_) : 1) * this->r_par_t(other); }

    double r_par_o(const SPos& other) const { return opos_.distance_par(other); }

    double r_par_o(const Pos& other) const { return r_par_o(other.opos_); }

    std::tuple<double, double> distance_par(const Pos& other) const { return std::make_tuple(r_par_t_signed(other), r_par_o(other)); }

    double r_perp_t(const SPos& other) const { return tpos_.distance_perp(other); }

    double r_perp_t(const Pos& other) const { return r_perp_t(other.tpos_); }

    double r_perp_o(const SPos& other) const { return opos_.distance_perp(other); }

    double r_perp_o(const Pos& other) const { return r_perp_o(other.opos_); }

    std::tuple<double, double> distance_perp(const Pos& other) const { return std::make_tuple(r_perp_t(other), r_perp_o(other)); }

    std::array<double,3> distance_vector_t(const SPos& other) const { return tpos_.distance_vector(other); }

    std::array<double,3> distance_vector_t(const Pos& other) const { return tpos_.distance_vector(other.tpos_); }

    std::array<double,3> distance_vector_o(const SPos& other) const { return opos_.distance_vector(other); }

    std::array<double,3> distance_vector_o(const Pos& other) const { return opos_.distance_vector(other.opos_); }

    std::array<double,3> distance_vector(const SPos& other, bool use_true) const { return use_true ? tpos_.distance_vector(other) : opos_.distance_vector(other); }

    std::array<double,3> distance_vector(const Pos& other, bool use_true) const { return use_true ? tpos_.distance_vector(other.tpos_) : opos_.distance_vector(other.opos_); }

    double distance_magnitude_t(const SPos& other) const { return tpos_.distance_magnitude(other); }

    double distance_magnitude_t(const Pos& other) const { return tpos_.distance_magnitude(other.tpos_); }

    double distance_magnitude_o(const SPos& other) const { return opos_.distance_magnitude(other); }

    double distance_magnitude_o(const Pos& other) const { return opos_.distance_magnitude(other.opos_); }

    double distance_magnitude(const SPos& other, const bool use_true) const { return use_true ? tpos_.distance_magnitude(other) : opos_.distance_magnitude(other); }

    double distance_magnitude(const Pos& other, const bool use_true) const { return use_true ? tpos_.distance_magnitude(other.tpos_) : opos_.distance_magnitude(other.opos_); }

    bool check_shell(const SPos& other, double min, double max) const {
	if (!other.is_initialized_ || std::isnan(other.r_)) throw std::runtime_error("No distance set in other");
	if (!has_obs()) {
	    if (!has_true()) throw std::runtime_error("Neither true nor observed distances set in self");
	    return tpos_.check_shell(other, min, max);
	}
	return opos_.check_shell(other, min, max);
    }

    bool check_shell(const SPos& other, const BinSpecifier& binner) const { return check_shell(other, binner.get_bin_min(), binner.get_bin_max()); }

    bool check_shell(const SPos& other, double max) const { return check_shell(other, 0.0, max); }

    bool check_shell(const Pos& other, double min, double max) const {
	if (!(has_obs() && other.has_obs())) {
	    if (!(has_true() && other.has_true())) throw std::runtime_error("Cannot mix true and observed distances");
	    return tpos_.check_shell(other.tpos_, min, max);
	}
	return opos_.check_shell(other.opos_, min, max);
    }

    bool check_shell(const Pos& other, const BinSpecifier& binner) const { return check_shell(other, binner.get_bin_min(), binner.get_bin_max()); }

    bool check_shell(const Pos& other, double max) const { return check_shell(other, 0.0, max); }

    bool check_limits(const SPos& other, double rp_min, double rp_max, double rl_min, double rl_max) const {
	if (!other.is_initialized_ || std::isnan(other.r_)) throw std::runtime_error("No distance set in other");
	if (!has_obs()) {
	    if (!has_true()) throw std::runtime_error("Neither true nor observed distances set in self");
	    return tpos_.check_limits(other, rp_min, rp_max, rl_min, rl_max);
	}
	return opos_.check_limits(other, rp_min, rp_max, rl_min, rl_max);
    }

    bool check_limits(const SPos& other, const BinSpecifier& rp_binner, const BinSpecifier& rl_binner) const { return check_limits(other, rp_binner.get_bin_min(), rp_binner.get_bin_max(), rl_binner.get_bin_min(), rl_binner.get_bin_max()); }

    bool check_limits(const Pos& other, double rp_min, double rp_max, double rl_min, double rl_max) const {
	if (!(has_obs() && other.has_obs())) {
	    if (!(has_true() && other.has_true())) throw std::runtime_error("Cannot mix true and observed distances");
	    return tpos_.check_limits(other.tpos_, rp_min, rp_max, rl_min, rl_max);
	}
	return opos_.check_limits(other.opos_, rp_min, rp_max, rl_min, rl_max);
    }

    bool check_limits(const Pos& other, const BinSpecifier& rp_binner, const BinSpecifier& rl_binner) const { return check_limits(other, rp_binner.get_bin_min(), rp_binner.get_bin_max(), rl_binner.get_bin_min(), rl_binner.get_bin_max()); }

    std::string toString() const {
	std::ostringstream oss;
	oss << "Pos(" << std::endl;
	oss << "    true = " << tpos_ << "," << std::endl;
	oss << "     obs = " << opos_ << std::endl;
	oss << "   )";
	return oss.str();
    }

    friend std::ostream& operator<<(std::ostream& os, const Pos& p) {
	os << p.toString();
	return os;
    }
};

inline std::vector<SPos> tpos(const std::vector<Pos>& pos) {
    std::vector<SPos> out;
    for (auto p : pos) { out.push_back(p.tpos()); }
    return out;
}

inline std::vector<SPos> opos(const std::vector<Pos>& pos) {
    std::vector<SPos> out;
    for (auto p : pos) { out.push_back(p.opos()); }
    return out;
}

struct Separation {
    double r_perp = math::dnan;
    double r_par = math::dnan;
    double zbar = math::dnan;
    std::size_t id1 = math::nan<std::size_t>;
    std::size_t id2 = math::nan<std::size_t>;

private:
    double get_r_perp(const SPos& pos1, const SPos& pos2, bool=true) {
	return pos1.distance_perp(pos2);
    }

    double get_r_perp(const Pos& pos1, const SPos& pos2, bool use_true) {
	return use_true ? pos1.r_perp_t(pos2) : pos1.r_perp_o(pos2);
    }

    double get_r_perp(const SPos& pos1, const Pos& pos2, bool use_true) {
	return pos1.distance_perp(use_true ? pos2.tpos() : pos2.opos());
    }

    double get_r_perp(const Pos& pos1, const Pos& pos2, bool use_true) {
	return use_true ? pos1.r_perp_t(pos2) : pos1.r_perp_o(pos2);
    }

    double get_r_par(const SPos& pos1, const SPos& pos2, bool=true, bool=false) {
	return pos1.distance_par(pos2);
    }

    double get_r_par(const Pos& pos1, const SPos& pos2, bool use_true, bool=false) {
	return use_true ? pos1.r_par_t(pos2) : pos1.r_par_o(pos2);
    }

    double get_r_par(const SPos& pos1, const Pos& pos2, bool use_true, bool=false) {
	return pos1.distance_par(use_true ? pos2.tpos() : pos2.opos());
    }

    double get_r_par(const Pos& pos1, const Pos& pos2, bool use_true, bool use_signed=false) {
	return use_true ? use_signed ? pos1.r_par_t_signed(pos2) : pos1.r_par_t(pos2) : pos1.r_par_o(pos2);
    }

    double get_zbar(const SPos& pos1, const SPos& pos2, bool=true) {
	return pos1.distance_zbar(pos2);
    }

    double get_zbar(const Pos& pos1, const SPos& pos2, bool use_true) {
	return use_true ? pos1.zbar_t(pos2) : pos1.zbar_o(pos2);
    }

    double get_zbar(const SPos& pos1, const Pos& pos2, bool use_true) {
	return pos1.distance_zbar(use_true ? pos2.tpos() : pos2.opos());
    }

    double get_zbar(const Pos& pos1, const Pos& pos2, bool use_true) {
	return use_true ? pos1.zbar_t(pos2) : pos1.zbar_o(pos2);
    }

public:
    // empty constructor
    Separation() = default;

    // copy constructor
    Separation(const Separation&) = default;

    // move constructor
    Separation(Separation&&) = default;

    Separation(double rp, double rl, double zb, std::size_t i1, std::size_t i2)
	: r_perp(rp), r_par(rl), zbar(zb), id1(i1), id2(i2) {}

    Separation(const SPos& pos1, const SPos& pos2, std::size_t i1, std::size_t i2)
	:r_perp(get_r_perp(pos1, pos2)), r_par(get_r_par(pos1, pos2)), zbar(get_zbar(pos1, pos2)), id1(i1), id2(i2) {}

    Separation(const Pos& pos1, const SPos& pos2, std::size_t i1, std::size_t i2, bool use_true)
	: r_perp(get_r_perp(pos1, pos2, use_true)), r_par(get_r_par(pos1, pos2, use_true)), zbar(get_zbar(pos1, pos2, use_true)), id1(i1), id2(i2) {}

    Separation(const SPos& pos1, const Pos& pos2, std::size_t i1, std::size_t i2, bool use_true)
	: r_perp(get_r_perp(pos1, pos2, use_true)), r_par(get_r_par(pos1, pos2, use_true)), zbar(get_zbar(pos1, pos2, use_true)), id1(i1), id2(i2) {}

    Separation(const Pos& pos1, const Pos& pos2, std::size_t i1, std::size_t i2, bool use_true, bool use_signed=false)
	: r_perp(get_r_perp(pos1, pos2, use_true)), r_par(get_r_par(pos1, pos2, use_true, use_signed)), zbar(get_zbar(pos1, pos2, use_true)), id1(i1), id2(i2) {}

    // copy assignment
    Separation& operator=(const Separation&) = default;

    // move assignment
    Separation& operator=(Separation&&) = default;

    bool operator==(const Separation& other) const {
	if (!(std::isnan(r_perp) ? std::isnan(other.r_perp) : math::isclose(r_perp, other.r_perp))) return false;
	if (!(std::isnan(r_par) ? std::isnan(other.r_par) : math::isclose(r_par, other.r_par))) return false;
	if (!(std::isnan(zbar) ? std::isnan(other.zbar) : math::isclose(zbar, other.zbar))) return false;
	return id1 == other.id1 && id2 == other.id2;
    }

    bool operator!=(const Separation& other) const { return !operator==(other); }

    std::string toString() const {
	std::ostringstream oss;
	oss << "Separation(";
	oss << "r_perp = ";
	if (math::isnan(r_perp)) oss << math::snan;
	else oss << r_perp;
	oss << ", r_par = ";
	if (math::isnan(r_par)) oss << math::snan;
	else oss << r_par;
	oss << ", zbar = ";
	if (math::isnan(zbar)) oss << math::snan;
	else oss << zbar;
	oss << ", id1 = ";
	if (math::isnan(id1)) oss << math::snan;
	else oss << id1;
	oss << ", id2 = ";
	if (math::isnan(id2)) oss << math::snan;
	else oss << id2;
	oss << ")";
	return oss.str();
    }

    friend std::ostream& operator<<(std::ostream& os, const Separation& s) {
	os << s.toString();
	return os;
    }

private:
    friend struct TOSeparation;
};

struct TOSeparation {
    Separation tsep, osep;

    // empty constructor
    TOSeparation() = default;

    // copy constructor
    TOSeparation(const TOSeparation&) = default;

    // move constructor
    TOSeparation(TOSeparation&&) = default;

    TOSeparation(double rpt, double rlt, double zbt, double rpo, double rlo, double zbo, std::size_t i1, std::size_t i2)
	: tsep(Separation(rpt, rlt, zbt, i1, i2)), osep(Separation(rpo, rlo, zbo, i1, i2)) {}

    TOSeparation(const Pos& pos1, const Pos& pos2, std::size_t i1, std::size_t i2)
	: tsep(Separation(pos1, pos2, i1, i2, true, true)), osep(Separation(pos1, pos2, i1, i2, false)) {}

    TOSeparation(const Separation& true_sep, const Separation& obs_sep)
	: tsep(true_sep), osep(obs_sep) {}

    bool operator==(const TOSeparation& other) const { return tsep == other.tsep && osep == other.osep; }

    bool operator!=(const TOSeparation& other) const { return !(*this == other); }

    // copy assignment
    TOSeparation& operator=(const TOSeparation&) = default;

    // move assignment
    TOSeparation& operator=(TOSeparation&&) = default;

    double r_perp_t() const { return tsep.r_perp; }

    double r_par_t() const { return tsep.r_par; }

    double zbar_t() const { return tsep.zbar; }

    double r_perp_o() const { return osep.r_perp; }

    double r_par_o() const { return osep.r_par; }

    double zbar_o() const { return osep.zbar; }

    std::size_t id1() const { return tsep.id1; }

    std::size_t id2() const { return tsep.id2; }

    std::string toString() const {
	std::ostringstream oss;
	oss << "TOSeparation(";
	std::string pad;
	for (std::size_t i = 0; i < oss.str().size() - 1; i++) {
	    pad += " ";
	}
	oss << std::endl;
	oss << pad << " true = " << tsep << std::endl;
	oss << pad << " obs = " << osep << std::endl;
	oss << pad << ")";
	return oss.str();
    }

    friend std::ostream& operator<<(std::ostream& os, const TOSeparation& s) {
	os << s.toString();
	return os;
    }
};

std::vector<Separation> get_auto_separations(const std::vector<SPos>& pos, const BinSpecifier& rp_bins, const BinSpecifier& rl_bins, int num_threads = OMP_NUM_THREADS);

std::vector<Separation> get_auto_separations(const std::vector<Pos>& pos, const BinSpecifier& rp_bins, const BinSpecifier& rl_bins, bool use_true, int num_threads = OMP_NUM_THREADS);

std::vector<Separation> get_cross_separations(const std::vector<SPos>& pos1, const std::vector<SPos>& pos2, const BinSpecifier& rp_bins, const BinSpecifier& rl_bins, int num_threads = OMP_NUM_THREADS);

std::vector<Separation> get_cross_separations(const std::vector<Pos>& pos1, const std::vector<Pos>& pos2, const BinSpecifier& rp_bins, const BinSpecifier& rl_bins, bool use_true, int num_threads = OMP_NUM_THREADS);

std::vector<Separation> get_separations(const std::vector<SPos>& pos1, const std::vector<SPos>& pos2, const BinSpecifier& rp_bins, const BinSpecifier& rl_bins, bool is_auto, int num_threads = OMP_NUM_THREADS);

std::vector<Separation> get_separations(const std::vector<Pos>& pos1, const std::vector<Pos>& pos2, const BinSpecifier& rp_bins, const BinSpecifier& rl_bins, bool use_true, bool is_auto, int num_threads = OMP_NUM_THREADS);

std::vector<TOSeparation> get_auto_separations(const std::vector<Pos>& pos, const BinSpecifier& rp_bins, const BinSpecifier& rl_bins, int num_threads = OMP_NUM_THREADS);

std::vector<TOSeparation> get_cross_separations(const std::vector<Pos>& pos1, const std::vector<Pos>& pos2, const BinSpecifier& rp_bins, const BinSpecifier& rl_bins, int num_threads = OMP_NUM_THREADS);

std::vector<TOSeparation> get_separations(const std::vector<Pos>& pos1, const std::vector<Pos>& pos2, const BinSpecifier& rp_bins, const BinSpecifier& rl_bins, bool is_auto, int num_threads = OMP_NUM_THREADS);


template<std::size_t N>
std::size_t get_1d_indexer_from_nd(const std::array<int, N>& indices, const std::array<BinSpecifier, N>& bins) {
    std::size_t index = 0;
    for (std::size_t i = 0; i < N; i++) {
	if (indices[i] < 0 || indices[i] >= (int)bins[i].get_nbins()) throw std::out_of_range("Invalid index " + std::to_string(indices[i]) + " for dimension " + std::to_string(i) + " with size " + std::to_string(bins[i].get_nbins()));
	index *= bins[i].get_nbins();
	index += indices[i];
    }
    return index;
}

template<std::size_t N>
std::array<std::size_t, N> get_nd_indexer_from_1d(std::size_t index, const std::array<BinSpecifier, N>& bins) {
    std::array<std::size_t, N> indices;
    auto iit = indices.begin();
    auto bit = bins.rbegin();
    for (; iit != indices.end() && bit != bins.rend(); ++iit, ++bit) {
	auto divmod = std::div(index, bit->get_nbins());
	*iit = divmod.quot;
	index = divmod.rem;
    }
    return indices;
}

template<typename TupleT>
constexpr auto get_array_from_tuple(TupleT&& tuple) {
    constexpr auto get_array = [](auto&&... x) { return std::array{std::forward<decltype(x)>(x)...}; };
    return std::apply(get_array, std::forward<TupleT>(tuple));
}

template<class C>
class CorrFuncBase;

template<class C>
class CorrFunc;

template<std::size_t N>
class CorrFuncNDBase;

template<std::size_t N>
class NNCountsNDBase {
    using BSType = std::array<BinSpecifier, N>;
    using NNType = NNCountsND<N>;
private:
    static std::size_t bin_size_multiply(std::size_t a, const BinSpecifier& b) { return !b._is_set? 0 : a * b.nbins; }

protected:
    BSType binners_ = make_filled_array<BinSpecifier,N>(BinSpecifier());
    std::size_t n_tot_ = 0, max_index_ = 0;
    std::vector<int> counts_ = {};
    double r_min = 0.0, r_max = 0.0;

    static double bin_min_combine(double a, const BinSpecifier& b) { return !b._is_set ? 0.0 : math::isclose(a, 0.0) && math::isclose(b.bin_min, 0.0) ? 0.0 : std::sqrt(math::square(a) + math::square(b.bin_min)); }

    static double bin_max_combine(double a, const BinSpecifier& b) { return !b._is_set ? 0.0 : std::isinf(a) || std::isinf(b.bin_max) ? std::numeric_limits<double>::max() : std::sqrt(math::square(a) + math::square(b.bin_max)); }

    virtual std::string class_name() const { return "NNCounts" + std::to_string(N) + "D"; }

    virtual std::string binners_string() const {
	std::ostringstream oss;
	std::string name = this->class_name();
	std::string pad = " ";
	for (std::size_t i = 0; i < name.size(); i++) {
	    pad += " ";
	}
	for (std::size_t i = 0; i < N; i++) {
	    oss << pad << "binner[" << i << "]=" << binners_[i] << "," << std::endl;
	}
	return oss.str();
    }


    std::size_t get_max_index() const { return std::accumulate(std::next(binners_.begin()), binners_.end(), binners_[0].nbins, bin_size_multiply); }

    virtual double get_r_min() const {
	return max_index_ == 0 ? 0.0 : std::accumulate(std::next(binners_.begin()), binners_.end(), binners_[0].bin_min, bin_min_combine);
    }

    virtual double get_r_max() const {
	return max_index_ == 0 ? 0.0 : std::accumulate(std::next(binners_.begin()), binners_.end(), binners_[0].bin_max, bin_max_combine);
    }

    void on_bin_update() {
	n_tot_ = 0;
	max_index_ = get_max_index();
	counts_ = std::vector<int>(max_index_, 0);
	r_min = this->get_r_min();
	r_max = this->get_r_max();
    }

    void update_binning(std::size_t binner_index, const BinSpecifier& new_binner, bool prefer_old=true) {
	if (binner_index >= N) throw std::out_of_range("Invalid index " + std::to_string(binner_index) + " for " + std::to_string(N) + "D binning");
	if (prefer_old) binners_[binner_index].fill(new_binner);
	else binners_[binner_index].update(new_binner);
	on_bin_update();
    }

    void process_separation(const std::array<double, N>& values) {
	n_tot_++;
	int index = get_bin(values);
	if (index < 0) return;
	counts_[index]++;
    }

    // empty constructor
    NNCountsNDBase() = default;

    // copy constructor
    NNCountsNDBase(const NNCountsNDBase&) = default;

    // constructor for pickling support
    NNCountsNDBase(const BSType& bins, const std::vector<int>& counts, std::size_t n_tot)
	: binners_(bins), n_tot_(n_tot), max_index_(get_max_index()), counts_(counts), r_min(0.0), r_max(0.0) {}

    // from binning
    NNCountsNDBase(const BSType& bins)
	: binners_(bins), n_tot_(0), max_index_(get_max_index()), counts_(max_index_, 0), r_min(0.0), r_max(0.0) {}

    void init() {
	r_min = this->get_r_min();
	r_max = this->get_r_max();
    }

public:
    BSType bin_info() const { return binners_; }

    std::size_t get_1d_indexer(const std::array<int, N>& indices) const { return get_1d_indexer_from_nd(indices, binners_); }

    int get_bin(const std::array<double, N>& values) const {
	std::array<int, N> indices;
	for (std::size_t i = 0; i < N; i++) {
	    indices[i] = binners_[i].assign_bin(values[i]);
	    if (indices[i] == -1) return -1;
	}
	return get_1d_indexer(indices);
    }

    virtual void process_pair(const SPos& pos1, const SPos& pos2) = 0;

    void process_pair(const Pos& pos1, const Pos& pos2, bool use_true) {
	if (use_true) this->process_pair(pos1.tpos(), pos2.tpos());
	else this->process_pair(pos1.opos(), pos2.opos());
    }

    int operator[](std::size_t idx) const { return counts_[idx]; }

    std::size_t n_tot() const { return n_tot_; }

    std::vector<int> counts() const { return counts_; }

    std::vector<double> normed_counts() const {
	std::vector<double> normed(counts_.begin(), counts_.end());
	std::transform(normed.begin(), normed.end(), normed.begin(), std::bind2nd(std::divides<double>(), n_tot_));
	return normed;
    }

    NNType operator+=(const NNType& other) {
	for (std::size_t i = 0; i < N; i++) {
	    if (binners_[i] != other.binners_[i]) {
		std::cerr << "Attempted to combine " << this->class_name() << " instances with different binning in dimension " << i << std::endl;
		std::cerr << "this.binners[" << i << "]: " << binners_[i] << std::endl;
		std::cerr << "other.binners[" << i << "]: " << other.binners_[i] << std::endl;
		throw std::runtime_error("Cannot combine " + this->class_name() + " instances with different binning schemes");
	    }
	}
	n_tot_ += other.n_tot_;
	std::transform(counts_.begin(), counts_.end(), other.counts_.begin(), counts_.begin(), std::plus<int>());
	return NNType(*this);
    }

    template<typename T>
    typename std::enable_if_t<std::is_arithmetic<T>::value, NNType>
    operator+=(const T& x) {
	if (!math::isclose(x, (T)0)) {
	    throw std::invalid_argument("Only 0 valid for scalar addition with " + class_name());
	}
	return NNType(*this);
    }

    template<typename T>
    typename std::enable_if_t<std::is_arithmetic<T>::value, NNType>
    operator+(const T& x) const { return NNType(*this).operator+=(x); }

    template<typename T>
    friend typename std::enable_if_t<std::is_arithmetic<T>::value, NNType>
    operator+(const T& x, const NNType& rhs) { return rhs.operator+(x); }

    bool operator==(const NNType& other) const { return n_tot_ == other.n_tot_ && std::equal(binners_.begin(), binners_.end(), other.binners_.begin()) && std::equal(counts_.begin(), counts_.end(), other.counts_.begin()); }

    bool operator!=(const NNType& other) const { return !(this->operator==(other)); }

    std::vector<std::size_t> shape_vec() const {
	std::vector<std::size_t> sv;
	for (auto b : binners_) { sv.push_back(b.nbins); }
	return sv;
    }

    std::size_t size() const { return max_index_; }

    std::size_t nbins_nonzero() const { return max_index_ - std::count(counts_.begin(), counts_.end(), 0); }

    int ncounts() const { return std::accumulate(counts_.begin(), counts_.end(), 0); }

    std::string toString() const {
	std::ostringstream oss;
	std::string name = this->class_name();
	std::string pad = "";
	for (std::size_t i = 0; i < name.size(); i++) {
	    pad += " ";
	}
	oss << name << "(" << std::endl;
	oss << this->binners_string();
	oss << pad << " ntot=" << n_tot_ << std::endl;
	oss << ")";
	return oss.str();
    }

    friend std::ostream& operator<<(std::ostream& os, const NNType& nn) {
	os << nn.toString();
	return os;
    }

private:
    friend class ExpectedNNCountsNDBase<N>;
    friend class CorrFuncNDBase<N>;
};

template<std::size_t N>
class NNCountsND : public NNCountsNDBase<N> {
    using Base = NNCountsNDBase<N>;
    using BSType = std::array<BinSpecifier, N>;

public:
    NNCountsND() = default;

    NNCountsND(const NNCountsND&) = default;

    NNCountsND(const BSType& bins, const std::vector<int> counts, std::size_t n_tot)
	: Base(bins, counts, n_tot) {
	this->init();
    }

    NNCountsND(const BSType& bins)
	: Base(bins) {
	this->init();
    }

    NNCountsND(const Base& b)
	: Base(b) {}

    NNCountsND(Base&& b)
	: Base(std::move(b)) {}

    void process_pair(const SPos&, const SPos&) override { this->n_tot_++; return; }

private:
    friend class ExpectedNNCountsNDBase<N>;
    friend class ExpectedNNCountsND<N>;
};

template<>
class NNCountsND<3> : public NNCountsNDBase<3> {
    using Base = NNCountsNDBase<3>;
    using BSType = std::array<BinSpecifier, 3>;
protected:
    std::string binners_string() const override {
	std::ostringstream oss;
	std::array<std::string, 3> binner_names = {"rperp_bins", "rpar_bins", "zbar_bins"};
	std::string name = this->class_name();
	std::string pad = " ";
	for (std::size_t i = 0; i < name.size(); i++) {
	    pad += " ";
	}
	for (std::size_t i = 0; i < 3; i++) {
	    oss << pad << binner_names[i] << "=" << binners_[i] << "," << std::endl;
	}
	return oss.str();
    }

    double get_r_min() const override { return max_index_ == 0 ? 0.0 : bin_min_combine(binners_[0].bin_min, binners_[1]); }

    double get_r_max() const override { return max_index_ == 0 ? 0.0 : bin_max_combine(binners_[0].bin_max, binners_[1]); }

    using Base::process_separation;

    void process_separation(double r_perp, double r_par, double zbar) {
	process_separation((std::array<double,3>){{r_perp, r_par, zbar}});
    }

public:
    NNCountsND() = default;

    NNCountsND(const NNCountsND&) = default;

    NNCountsND(const BSType& bins, const std::vector<int> counts, std::size_t n_tot)
	: Base(bins, counts, n_tot) {
	this->init();
    }

    NNCountsND(const BSType& bins)
	: Base(bins) {
	this->init();
    }

    NNCountsND(const Base& b)
	: Base(b) {}

    NNCountsND(Base&& b)
	: Base(std::move(b)) {}

    NNCountsND(const BinSpecifier& rp_bins, const BinSpecifier& rl_bins, const BinSpecifier& zb_bins)
	: Base((BSType){{rp_bins, rl_bins, zb_bins}}) {
	this->init();
    }

    BinSpecifier rperp_bins() const { return binners_[0]; }

    void rperp_bins(const BinSpecifier& new_binner, bool prefer_old=true) { update_binning(0, new_binner, prefer_old); }

    BinSpecifier rpar_bins() const { return binners_[1]; }

    void rpar_bins(const BinSpecifier& new_binner, bool prefer_old=true) { update_binning(1, new_binner, prefer_old); }

    BinSpecifier zbar_bins() const { return binners_[2]; }

    void zbar_bins(const BinSpecifier& new_binner, bool prefer_old=true) { update_binning(2, new_binner, prefer_old); }

    using Base::get_1d_indexer;

    std::size_t get_1d_indexer(int rp_bin, int rl_bin, int zb_bin) const {
	return get_1d_indexer((std::array<int,3>){{rp_bin, rl_bin, zb_bin}});
    }

    using Base::process_pair;

    void process_pair(const SPos& pos1, const SPos& pos2) override {
	n_tot_++;
	if (!pos1.check_shell(pos2, r_min, r_max)) return;
	int zo_bin = binners_[2].assign_bin(pos1.distance_zbar(pos2));
	if (zo_bin == -1) return;
	int rp_bin = binners_[0].assign_bin(pos1.distance_perp(pos2));
	if (rp_bin == -1) return;
	int rl_bin = binners_[1].assign_bin(pos1.distance_par(pos2));
	if (rl_bin == -1) return;
	counts_[get_1d_indexer(rp_bin, rl_bin, zo_bin)]++;
    }

    std::tuple<std::size_t, std::size_t, std::size_t> shape() const {
	auto sv = shape_vec();
	return std::make_tuple(sv[0], sv[1], sv[2]);
    }

private:
    friend class ExpectedNNCountsNDBase<3>;
    friend class ExpectedNNCountsND<3>;
};

template<>
class NNCountsND<2> : public NNCountsNDBase<2> {
    using Base = NNCountsNDBase<2>;
    using BSType = std::array<BinSpecifier, 2>;
protected:
    std::string binners_string() const override {
	std::ostringstream oss;
	std::array<std::string, 2> binner_names = {"rperp_bins", "rpar_bins"};
	std::string name = this->class_name();
	std::string pad = " ";
	for (std::size_t i = 0; i < name.size(); i++) {
	    pad += " ";
	}
	for (std::size_t i = 0; i < 2; i++) {
	    oss << pad << binner_names[i] << "=" << binners_[i] << "," << std::endl;
	}
	return oss.str();
    }

    using Base::process_separation;

    void process_separation(double r_perp, double r_par) {
	process_separation((std::array<double,2>){{r_perp, r_par}});
    }

public:
    NNCountsND() = default;

    NNCountsND(const NNCountsND&) = default;

    NNCountsND(const BSType& bins, const std::vector<int> counts, std::size_t n_tot)
	: Base(bins, counts, n_tot) {
	this->init();
    }

    NNCountsND(const BSType& bins)
	: Base(bins) {
	this->init();
    }

    NNCountsND(const BinSpecifier& rp_bins, const BinSpecifier& rl_bins)
	: Base((BSType){{rp_bins, rl_bins}}) {
	this->init();
    }

    NNCountsND(const Base& b)
	: Base(b) {}

    NNCountsND(Base&& b)
	: Base(std::move(b)) {}

    BinSpecifier rperp_bins() const { return binners_[0]; }

    void rperp_bins(const BinSpecifier& new_binner, bool prefer_old=true) { update_binning(0, new_binner, prefer_old); }

    BinSpecifier rpar_bins() const { return binners_[1]; }

    void rpar_bins(const BinSpecifier& new_binner, bool prefer_old=true) { update_binning(1, new_binner, prefer_old); }

    using Base::get_1d_indexer;

    std::size_t get_1d_indexer(int rp_bin, int rl_bin) const {
	return get_1d_indexer((std::array<int,2>){{rp_bin, rl_bin}});
    }

    using Base::process_pair;

    void process_pair(const SPos& pos1, const SPos& pos2) override {
	n_tot_++;
	if (!pos1.check_shell(pos2, r_min, r_max)) return;
	int rp_bin = binners_[0].assign_bin(pos1.distance_perp(pos2));
	if (rp_bin == -1) return;
	int rl_bin = binners_[1].assign_bin(pos1.distance_par(pos2));
	if (rl_bin == -1) return;
	counts_[get_1d_indexer(rp_bin, rl_bin)]++;
    }

    std::tuple<std::size_t, std::size_t> shape() const {
	auto sv = shape_vec();
	return std::make_tuple(sv[0], sv[1]);
    }

private:
    friend class ExpectedNNCountsNDBase<2>;
    friend class ExpectedNNCountsND<2>;
};

template<>
class NNCountsND<1> : public NNCountsNDBase<1> {
    using Base = NNCountsNDBase<1>;
    std::size_t N = 1;
    using BSType = std::array<BinSpecifier, 1>;
protected:
    std::string binners_string() const override {
	std::ostringstream oss;
	std::string name = this->class_name();
	std::string pad = " ";
	for (std::size_t i = 0; i < name.size(); i++) {
	    pad += " ";
	}
	oss << pad << "r_bins=" << binners_[0] << "," << std::endl;
	return oss.str();
    }

    using Base::process_separation;

    void process_separation(double r) {
	process_separation((std::array<double,1>){{r}});
    }

public:
    NNCountsND() = default;

    NNCountsND(const NNCountsND&) = default;

    NNCountsND(const BSType& bins, const std::vector<int> counts, std::size_t n_tot)
	: Base(bins, counts, n_tot) {
	this->init();
    }

    NNCountsND(const BSType& bins)
	: Base(bins) {
	this->init();
    }

    NNCountsND(const BinSpecifier& r_bins)
	: Base((BSType){{r_bins}}) {
	this->init();
    }

    NNCountsND(const Base& b)
	: Base(b) {}
    NNCountsND(Base&& b)
	: Base(std::move(b)) {}

    BinSpecifier r_bins() const { return binners_[0]; }

    void r_bins(const BinSpecifier& new_binner, bool prefer_old=true) { update_binning(0, new_binner, prefer_old); }

    using Base::get_1d_indexer;

    std::size_t get_1d_indexer(int r_bin) const {
	return get_1d_indexer((std::array<int,1>){{r_bin}});
    }

    using Base::process_pair;

    void process_pair(const SPos& pos1, const SPos& pos2) override {
	n_tot_++;
	if (!pos1.check_shell(pos2, r_min, r_max)) return;
	int bin = binners_[0].assign_bin(std::sqrt(math::square(pos1.distance_perp(pos2)) + math::square(pos1.distance_par(pos2))));
	if (bin > -1) counts_[bin]++;
    }

    std::tuple<std::size_t> shape() const {
	return std::make_tuple(max_index_);
    }

private:
    friend class ExpectedNNCountsNDBase<1>;
    friend class ExpectedNNCountsND<1>;
};

template<std::size_t N>
NNCountsND<N> get_pair_counts(std::vector<SPos> pos1, std::vector<SPos> pos2, std::array<BinSpecifier, N> binning, bool is_auto, int num_threads = OMP_NUM_THREADS) {
    using NNType = NNCountsND<N>;
    NNType nn(binning);
    std::size_t n1 = pos1.size();
    std::size_t n2 = pos2.size();
    omp_set_num_threads(num_threads);
#if _OPENMP
#pragma omp declare reduction(add : NNType : omp_out+=omp_in) initializer(omp_priv=omp_orig)
#pragma omp parallel for collapse(2) reduction(add : nn)
#endif
    for (std::size_t i = 0; i < n1; i++) {
	for (std::size_t j = 0; j < n2; j++) {
	    if (is_auto && i >= j) continue;
	    nn.process_pair(pos1[i], pos2[j]);
	}
    }
    return nn;
}

template<typename T>
std::vector<std::vector<T>> transpose(const std::vector<std::vector<T>>& data) {
    std::vector<std::vector<T>> result(data[0].size(), std::vector<T>(data.size()));
    for (std::size_t i = 0; i < result.size(); i++) {
	for (std::size_t j = 0; j < data.size(); j++) {
	    result[i][j] = data[j][i];
	}
    }
    return result;
}

template<typename T>
void transpose_in_place(std::vector<std::vector<T>>& data) {
    std::vector<std::vector<T>> temp = transpose(data);
    data.clear();
    data.swap(temp);
}

template<typename T, std::size_t N, std::size_t R>
std::array<T, R*N> repeat_array(std::array<T,N> input_arr) {
    std::array<T, R*N> output_arr;
    for (std::size_t i = 0; i < N; i++) {
	for (std::size_t j = 0; j < R; j++) {
	    output_arr[i + (j * N)] = input_arr[i];
	}
    }
    return output_arr;
}

template<std::size_t N>
class ExpectedNNCountsNDBase {
    using NNBaseType = NNCountsNDBase<N>;
    using NNType = NNCountsND<N>;
    using ENNType = ExpectedNNCountsND<N>;
    using BSType = std::array<BinSpecifier,N>;
protected:
    BSType binners_ = make_filled_array<BinSpecifier,N>(BinSpecifier());
    std::vector<NNType> nn_list_ = {};
    std::size_t n_real_ = 0, n_tot_ = 0, max_index_ = 0;
    std::vector<double> mean_ = {}, cov_ = {};
    bool mean_updated_ = true, cov_updated_ = true;

private:
    static std::size_t bin_size_multiply(std::size_t a, const BinSpecifier& b) { return !b._is_set? 0 : a * b.nbins; }

    void on_bin_ntot_update() {
	nn_list_ = {};
	mean_ = {};
	cov_ = {};
	n_real_ = 0;
	calculate_max_index();
	calculate_mean();
	calculate_cov();
    }

    void calculate_max_index() {
	max_index_ = std::accumulate(std::next(binners_.begin()), binners_.end(), binners_[0].nbins, bin_size_multiply);
    }

    void calculate_mean() {
	if (n_real_ == 0 || n_tot_ == 0) return;
	if (n_real_ == 1) {
	    mean_ = std::vector<double>(nn_list_[0].counts_.begin(), nn_list_[0].counts_.end());
	    std::transform(mean_.begin(), mean_.end(), mean_.begin(), std::bind2nd(std::divides<double>(), (double)n_tot_));
	}
	else {
	    mean_ = std::vector<double>(max_index_, 0.0);
	    for (auto nn : nn_list_) {
		std::vector<double> nc(nn.counts_.begin(), nn.counts_.end());
		std::transform(nc.begin(), nc.end(), nc.begin(), std::bind2nd(std::divides<double>(), (double)n_tot_));
		std::transform(mean_.begin(), mean_.end(), nc.begin(), mean_.begin(), std::plus<double>());
	    }
	    std::transform(mean_.begin(), mean_.end(), mean_.begin(), std::bind2nd(std::divides<double>(), (double)n_real_));
	}
	mean_updated_ = true;
    }

    void calculate_cov() {
	if (nn_list_.size() < 2 || n_tot_ == 0) return;
	std::size_t nn_size = nn_list_.size();
	std::vector<std::vector<double>> diff;
	for (auto nn : nn_list_) {
	    std::vector<double> temp(nn.counts_.begin(), nn.counts_.end());
	    std::transform(temp.begin(), temp.end(), temp.begin(), std::bind2nd(std::divides<double>(), (double)n_tot_));
	    std::transform(temp.begin(), temp.end(), mean_.begin(), temp.begin(), std::minus<double>());
	    diff.push_back(temp);
	}
	cov_ = std::vector<double>(math::square(max_index_), 0.0);
	for (std::size_t i = 0; i < max_index_; i++) {
	    for (std::size_t j = i; j < max_index_; j++) {
		for (std::size_t k = 0; k < nn_size; k++) {
		    cov_[j + max_index_ * i] += diff[k][i] * diff[k][j];
		}
	    }
	}
	std::transform(cov_.begin(), cov_.end(), cov_.begin(), std::bind2nd(std::divides<double>(), nn_size * (nn_size - 1.)));
	cov_updated_ = true;
    }

    void start_new_realization() {
	calculate_mean();
	calculate_cov();
	n_real_++;
	nn_list_.push_back(NNType(binners_));
    }

protected:
    virtual std::string class_name() const { return "ExpectedNNCounts" + std::to_string(N) + "D"; }

    virtual std::string binners_string() const {
	std::ostringstream oss;
	std::string name = this->class_name();
	std::string pad = " ";
	for (std::size_t i = 0; i < name.size(); i++) {
	    pad += " ";
	}
	for (std::size_t i = 0; i < N; i++) {
	    oss << pad << "binner[" << i << "]=" << binners_[i] << "," << std::endl;
	}
	return oss.str();
    }

    void update_binning(std::size_t binner_index, const BinSpecifier& new_binner, bool prefer_old=true) {
	if (binner_index >= N) throw std::out_of_range("Invalid index " + std::to_string(binner_index) + " for " + std::to_string(N) + "D binning");
	if (prefer_old) binners_[binner_index].fill(new_binner);
	else binners_[binner_index].update(new_binner);
	on_bin_ntot_update();
    }

public:
    // empty constructor
    ExpectedNNCountsNDBase() = default;

    // copy constructor
    ExpectedNNCountsNDBase(const ExpectedNNCountsNDBase&) = default;

    // pickling support
    ExpectedNNCountsNDBase(const BSType& binners, const std::vector<NNType>& nn_list, std::size_t n_real, std::size_t n_tot)
	: binners_(binners), nn_list_(nn_list), n_real_(n_real), n_tot_(n_tot), max_index_(0), mean_({}), cov_({}), mean_updated_(false), cov_updated_(false) {
	calculate_max_index();
	calculate_mean();
	calculate_cov();
    }

    // no data to start, but binners and n_tot specified
    ExpectedNNCountsNDBase(const BSType& binners, std::size_t n_tot)
	: binners_(binners), n_tot_(n_tot) {
	on_bin_ntot_update();
    }

    BSType bin_info() const { return binners_; }

    std::size_t get_1d_mean_indexer(const std::array<int, N>& indices) const { return get_1d_indexer_from_nd(indices, binners_); }

    std::size_t get_1d_cov_indexer(const std::array<int, 2*N>& indices) const {
	return get_1d_indexer_from_nd(indices, repeat_array<BinSpecifier, N, 2>(binners_));
    }

    void process_separation(const std::array<double, N>& values, bool new_real = false) {
	if (new_real || nn_list_.size() == 0) this->start_new_realization();
	nn_list_[n_real_ - 1].process_separation(values);
	mean_updated_ = false;
	cov_updated_ = false;
    }

    NNType operator[](std::size_t idx) const {
	if (idx > n_real_ + 1) throw std::out_of_range("Invalid index " + std::to_string(idx) + " for " + std::to_string(n_real_ + 1) + " realizations");
	// return the NNCountsND object at index idx, but make sure n_tot is
	// set to the n_tot we are using, without changing our nn_list_ object
	NNType temp(nn_list_[idx]);
	temp.n_tot_ = n_tot_;
	return temp;
    }

    std::size_t n_tot() const { return n_tot_; }

    void n_tot(std::size_t new_n_tot) {
	n_tot_ = new_n_tot;
	on_bin_ntot_update();
    }

    std::size_t n_real() const { return n_real_; }

    std::vector<std::size_t> mean_shape_vec() const {
	std::vector<std::size_t> sv;
	for (auto b : binners_) { sv.push_back(b.nbins); }
	return sv;
    }

    std::vector<std::size_t> cov_shape_vec() const {
	std::vector<std::size_t> sv;
	for (auto b : repeat_array<BinSpecifier,N,2>(binners_)) { sv.push_back(b.nbins); }
	return sv;
    }

    std::size_t mean_size() const { return max_index_; }

    std::size_t cov_size() const { return max_index_ * max_index_; }

    std::vector<NNType> nn_list() const { return nn_list_; }

    void update() {
	calculate_mean();
	calculate_cov();
    }

    std::vector<double> mean() const { return mean_updated_ ? mean_ : std::vector<double>(mean_.size(), 0.0); }

    std::vector<double> cov() const { return cov_updated_ ? cov_ : std::vector<double>(cov_.size(), 0.0); }

    ENNType operator+=(const ENNType& other) {
	if (n_tot_ != other.n_tot_) throw std::runtime_error("Cannot combine " + this->class_name() + " instances with different n_tot");
	for (std::size_t i = 0; i < N; i++) {
	    if (binners_[i] != other.binners_[i]) {
		std::cerr << "Attempted to combine " << this->class_name() << " instances with different binning in dimension " << i << std::endl;
		std::cerr << "this.binners[" << i << "]: " << binners_[i] << std::endl;
		std::cerr << "other.binners[" << i << "]: " << other.binners_[i] << std::endl;
		throw std::runtime_error("Cannot combine " + this->class_name() + " instances with different binning schemes");
	    }
	}
	if (other.n_real_ == 0) return ENNType(*this);
	if (n_real_ == 0) {
	    n_real_ = other.n_real_;
	    nn_list_ = other.nn_list_;
	    mean_ = other.mean_;
	    cov_ = other.cov_;
	    mean_updated_ = other.mean_updated_;
	    cov_updated_ = other.cov_updated_;
	}
	else {
	    nn_list_[nn_list_.size() - 1] += other.nn_list_[other.nn_list_.size() - 1];
	    mean_updated_ = false;
	    cov_updated_ = false;
	}
	return ENNType(*this);
    }

    ENNType operator+=(const NNType& other) {
	for (std::size_t i = 0; i < N; i++) {
	    if (binners_[i] != other.binners_[i]) {
		std::cerr << "Attempted to combine " << this->class_name() << " instances with different binning in dimension " << i << std::endl;
		std::cerr << "this.binners[" << i << "]: " << binners_[i] << std::endl;
		std::cerr << "other.binners[" << i << "]: " << other.binners_[i] << std::endl;
		throw std::runtime_error("Cannot combine " + this->class_name() + " instances with different binning schemes");
	    }
	}
	if (nn_list_.size() == 0) {
	    nn_list_.push_back(other);
	    n_real_++;
	    mean_updated_ = false;
	    cov_updated_ = false;
	}
	else {
	    nn_list_[nn_list_.size() - 1] += other;
	    mean_updated_ = false;
	    cov_updated_ = false;
	}
	return ENNType(*this);
    }

    template<typename T>
    typename std::enable_if_t<std::is_arithmetic<T>::value, ENNType>
    operator+=(const T& x) {
	if (!math::isclose(x, (T)0)) {
	    throw std::invalid_argument("Only 0 valid for scalar addition with " + this->class_name());
	}
	return ENNType(*this);
    }

    template<typename T>
    typename std::enable_if_t<std::is_arithmetic<T>::value, ENNType>
    operator+(const T& x) const { return ENNType(*this).operator+=(x); }

    template<typename T>
    friend typename std::enable_if_t<std::is_arithmetic<T>::value, ENNType>
    operator+(const T& x, const ENNType& rhs) { return rhs.operator+(x); }

    bool operator==(const ENNType& other) const { return n_tot_ == other.n_tot_ && n_real_ == other.n_real_ && std::equal(binners_.begin(), binners_.end(), other.binners_.begin()) && math::isclose(mean_, other.mean_) && math::isclose(cov_, other.cov_); }

    bool operator!=(const ENNType& other) const { return !(operator==(other)); }

    void append_real(const NNType& other) {
	calculate_mean();
	calculate_cov();
	n_real_++;
	nn_list_.push_back(other);
	mean_updated_ = false;
	cov_updated_ = false;
    }

    void append_real(const ENNType& other) {
	nn_list_.insert(nn_list_.end(), other.nn_list_.begin(), std::prev(other.nn_list_.end()));
	n_real_ = nn_list_.size();
	calculate_mean();
	calculate_cov();
	n_real_++;
	nn_list_.push_back(other.nn_list_[other.nn_list_.size() - 1]);
	mean_updated_ = false;
	cov_updated_ = false;
    }

    std::string toString() const {
	std::ostringstream oss;
	std::string name = this->class_name();
	std::string pad = "";
	for (std::size_t i = 0; i < name.size(); i++) {
	    pad += " ";
	}
	oss << name << "(" << std::endl;
	oss << this->binners_string();
	oss << pad << " ntot=" << n_tot_ << std::endl;
	oss << pad << " nreal=" << n_real_ << std::endl;
	oss << ")";
	return oss.str();
    }

    friend std::ostream& operator<<(std::ostream& os, const ENNType& nn) {
	os << nn.toString();
	return os;
    }
};

template<std::size_t N>
class ExpectedNNCountsND : public ExpectedNNCountsNDBase<N> {
    using Base = ExpectedNNCountsNDBase<N>;
public:
    using Base::Base;

    ExpectedNNCountsND(const Base& b)
	: Base(b) {}

    ExpectedNNCountsND(Base&& b)
	: Base(std::move(b)) {}
};

template<>
class ExpectedNNCountsND<3> : public ExpectedNNCountsNDBase<3> {
    using Base = ExpectedNNCountsNDBase<3>;
    using BSType = std::array<BinSpecifier, 3>;
protected:
    std::string binners_string() const override {
	std::ostringstream oss;
	std::array<std::string, 3> binner_names = {"rperp_bins", "rpar_bins", "zbar_bins"};
	std::string name = class_name();
	std::string pad = " ";
	for (std::size_t i = 0; i < name.size(); i++) {
	    pad += " ";
	}
	for (std::size_t i = 0; i < 3; i++) {
	    oss << pad << binner_names[i] << "=" << binners_[i] << "," << std::endl;
	}
	return oss.str();
    }

public:
    using Base::Base;

    ExpectedNNCountsND(const Base& b)
	: Base(b) {}

    ExpectedNNCountsND(const BinSpecifier& rperp_bins, const BinSpecifier& rpar_bins, const BinSpecifier& zbar_bins, std::size_t n_tot)
	: Base((BSType){{rperp_bins, rpar_bins, zbar_bins}}, n_tot) {}

    BinSpecifier rperp_bins() const { return binners_[0]; }

    void rperp_bins(const BinSpecifier& new_binner, bool prefer_old=true) { update_binning(0, new_binner, prefer_old); }

    BinSpecifier rpar_bins() const { return binners_[1]; }

    void rpar_bins(const BinSpecifier& new_binner, bool prefer_old=true) { update_binning(1, new_binner, prefer_old); }

    BinSpecifier zbar_bins() const { return binners_[2]; }

    void zbar_bins(const BinSpecifier& new_binner, bool prefer_old=true) { update_binning(2, new_binner, prefer_old); }

    using Base::get_1d_mean_indexer;

    std::size_t get_1d_mean_indexer_from_args(int rp_bin, int rl_bin, int zb_bin) const { return get_1d_mean_indexer((std::array<int,3>){{rp_bin, rl_bin, zb_bin}}); }

    using Base::get_1d_cov_indexer;

    std::size_t get_1d_cov_indexer_from_args(int rpi_bin, int rli_bin, int zbi_bin, int rpj_bin, int rlj_bin, int zbj_bin) const { return get_1d_cov_indexer((std::array<int,6>){{rpi_bin, rli_bin, zbi_bin, rpj_bin, rlj_bin, zbj_bin}}); }

    using Base::process_separation;

    void process_separation(double r_perp, double r_par, double zbar, bool new_real = false) { process_separation((std::array<double,3>){{r_perp, r_par, zbar}}, new_real); }

    std::tuple<int, int, int> mean_shape() const {
	return std::make_tuple(binners_[0].nbins, binners_[1].nbins, binners_[2].nbins);
    }

    std::tuple<int, int, int, int, int, int> cov_shape() const {
	return std::make_tuple(binners_[0].nbins, binners_[1].nbins, binners_[2].nbins, binners_[0].nbins, binners_[1].nbins, binners_[2].nbins);
    }
};

template<>
class ExpectedNNCountsND<2> : public ExpectedNNCountsNDBase<2> {
    using Base = ExpectedNNCountsNDBase<2>;
    using BSType = std::array<BinSpecifier, 2>;
protected:
    std::string binners_string() const override {
	std::ostringstream oss;
	std::array<std::string, 2> binner_names = {"rperp_bins", "rpar_bins"};
	std::string name = class_name();
	std::string pad = " ";
	for (std::size_t i = 0; i < name.size(); i++) {
	    pad += " ";
	}
	for (std::size_t i = 0; i < 2; i++) {
	    oss << pad << binner_names[i] << "=" << binners_[i] << "," << std::endl;
	}
	return oss.str();
    }

public:
    using Base::Base;

    ExpectedNNCountsND(const Base& b)
	: Base(b) {}

    ExpectedNNCountsND(const BinSpecifier& rperp_bins, const BinSpecifier& rpar_bins, std::size_t n_tot)
	: Base((BSType){{rperp_bins, rpar_bins}}, n_tot) {}

    BinSpecifier rperp_bins() const { return binners_[0]; }

    void rperp_bins(const BinSpecifier& new_binner, bool prefer_old=true) { update_binning(0, new_binner, prefer_old); }

    BinSpecifier rpar_bins() const { return binners_[1]; }

    void rpar_bins(const BinSpecifier& new_binner, bool prefer_old=true) { update_binning(1, new_binner, prefer_old); }

    using Base::get_1d_mean_indexer;

    std::size_t get_1d_mean_indexer_from_args(int rp_bin, int rl_bin) const { return get_1d_mean_indexer((std::array<int,2>){{rp_bin, rl_bin}}); }

    using Base::get_1d_cov_indexer;

    std::size_t get_1d_cov_indexer_from_args(int rpi_bin, int rli_bin, int rpj_bin, int rlj_bin) const { return get_1d_cov_indexer((std::array<int,4>){{rpi_bin, rli_bin, rpj_bin, rlj_bin}}); }

    using Base::process_separation;

    void process_separation(double r_perp, double r_par, bool new_real = false) { process_separation((std::array<double,2>){{r_perp, r_par}}, new_real); }

    std::tuple<int, int> mean_shape() const {
	return std::make_tuple(binners_[0].nbins, binners_[1].nbins);
    }

    std::tuple<int, int, int, int> cov_shape() const {
	return std::make_tuple(binners_[0].nbins, binners_[1].nbins, binners_[0].nbins, binners_[1].nbins);
    }
};

template<>
class ExpectedNNCountsND<1> : public ExpectedNNCountsNDBase<1> {
    using Base = ExpectedNNCountsNDBase<1>;
    using BSType = std::array<BinSpecifier, 1>;
protected:
    std::string binners_string() const override {
	std::ostringstream oss;
	std::string name = class_name();
	std::string pad = " ";
	for (std::size_t i = 0; i < name.size(); i++) {
	    pad += " ";
	}
	oss << pad << "r_bins=" << binners_[0] << "," << std::endl;
	return oss.str();
    }

public:
    using Base::Base;

    ExpectedNNCountsND(const Base& b)
	: Base(b) {}

    ExpectedNNCountsND(const BinSpecifier& r_bins, std::size_t n_tot)
	: Base((BSType){{r_bins}}, n_tot) {}

    BinSpecifier r_bins() const { return binners_[0]; }

    void r_bins(const BinSpecifier& new_binner, bool prefer_old=true) { update_binning(0, new_binner, prefer_old); }

    using Base::get_1d_mean_indexer;

    std::size_t get_1d_mean_indexer_from_args(int r_bin) const { return get_1d_mean_indexer((std::array<int,1>){{r_bin}}); }

    using Base::get_1d_cov_indexer;

    std::size_t get_1d_cov_indexer_from_args(int ri_bin, int rj_bin) const { return get_1d_cov_indexer((std::array<int,2>){{ri_bin, ri_bin}}); }

    using Base::process_separation;

    void process_separation(double r, bool new_real = false) { process_separation((std::array<double,1>){{r}}, new_real); }

    std::tuple<int> mean_shape() const {
	return std::make_tuple(max_index_);
    }

    std::tuple<int, int> cov_shape() const {
	return std::make_tuple(max_index_, max_index_);
    }
};

template<class C>
class CorrFuncBase {
protected:
    C dd_, dr_, rd_, rr_;
    std::vector<double> values_;
    std::size_t max_index_;

    virtual void verify_nn(const C& nn) { return; }

public:
    CorrFuncBase() {}

    CorrFuncBase(const CorrFuncBase& other)
	: dd_(other.dd_), dr_(other.dr_), rd_(other.rd_), rr_(other.rr_), values_(other.values_), max_index_(other.max_index_) {}

    CorrFuncBase(const CorrFunc<C>& other)
	: dd_(other.dd_), dr_(other.dr_), rd_(other.rd_), rr_(other.rr_), values_(other.values_), max_index_(other.max_index_) {}

    CorrFuncBase(const C& dd)
	: dd_(dd), values_(std::vector<double>(dd.max_index_, 0.0)), max_index_(dd.max_index_) {}

    void assign_dd(const C& new_dd) {
	dd_ = new_dd;
	if (rr_ != C()) {
	    try { verify_nn(rr_); }
	    catch (const std::runtime_error& e) { rr_ = C(); }
	}
	if (dr_ != C()) {
	    try { verify_nn(dr_); }
	    catch (const std::runtime_error& e) { dr_ = C(); }
	}
	if (rd_ != C()) {
	    try { verify_nn(rd_); }
	    catch (const std::runtime_error& e) { rd_ = C(); }
	}
	max_index_ = new_dd.max_index_;
	values_ = std::vector<double>(max_index_, 0.0);
    }

    void assign_rr(const C& new_rr) {
	verify_nn(new_rr);
	rr_ = new_rr;
    }

    void assign_dr(const C& new_dr) {
	verify_nn(new_dr);
	dr_ = new_dr;
    }

    void assign_rd(const C& new_rd) {
	verify_nn(new_rd);
	rd_ = new_rd;
    }

    void calculate_xi() {
	if (dd_ == C()) throw std::runtime_error("Cannot calculate correlation function without data-data pair counts");
	if (rr_ == C()) throw std::runtime_error("Cannot calculate correlation function without random-random pair counts");
	std::vector<double> rr_norm(rr_.normed_counts());
	values_ = dd_.normed_counts();
	if (dr_ == C() && rd_ == C()) {
	    // (DD / RR) - 1
	    std::transform(values_.begin(), values_.end(), rr_norm.begin(), values_.begin(), std::divides<double>());
	    std::transform(values_.begin(), values_.end(), values_.begin(), std::bind2nd(std::minus<double>(), 1.0));
	}
	else {
	    // All options have DD + RR
	    std::transform(values_.begin(), values_.end(), rr_norm.begin(), values_.begin(), std::plus<double>());
	    std::vector<double> dr_rd_norm(max_index_);
	    if (rd_ == C()) {
		// (DD - 2 DR + RR) / RR
		dr_rd_norm = dr_.normed_counts();
		std::transform(dr_rd_norm.begin(), dr_rd_norm.end(), dr_rd_norm.begin(), std::bind1st(std::multiplies<double>(), 2.0));
	    }
	    else if (dr_ == C()) {
		// (DD - 2 RD + RR) / RR
		dr_rd_norm = rd_.normed_counts();
		std::transform(dr_rd_norm.begin(), dr_rd_norm.end(), dr_rd_norm.begin(), std::bind1st(std::multiplies<double>(), 2.0));
	    }
	    else {
		// (DD - DR - RD + RR) / RR
		std::vector<double> temp = rd_.normed_counts();
		dr_rd_norm = dr_.normed_counts();
		std::transform(dr_rd_norm.begin(), dr_rd_norm.end(), temp.begin(), dr_rd_norm.begin(), std::plus<double>());
	    }
	    std::transform(values_.begin(), values_.end(), dr_rd_norm.begin(), values_.begin(), std::minus<double>());
	    std::transform(values_.begin(), values_.end(), rr_norm.begin(), values_.begin(), std::divides<double>());
	}
    }

    std::vector<double> xi() const { return values_; }

    C dd() const { return dd_; }

    C dr() const { return dr_; }

    C rd() const { return rd_; }

    C rr() const { return rr_; }

    auto shape() const { return dd_.shape(); }

    std::vector<std::size_t> shape_vec() const { return dd_.shape_vec(); }

    std::size_t size() const { return max_index_; }

    bool operator==(const CorrFuncBase& other) const {
	if (dd_ != other.dd_) return false;
	if (rr_ != other.rr_) return false;
	if (dr_ == C() && rd_ == C()) return other.dr_ == C() && other.rd_ == C();
	if (dr_ == C()) return (other.dr_ == C() || other.dr_ == rd_) && (other.rd_ == C() || other.rd_ == rd_) && !(other.dr_ == C() && other.rd_ == C());
	if (rd_ == C()) return (other.dr_ == C() || other.dr_ == dr_) && (other.rd_ == C() || other.rd_ == rd_) && !(other.dr_ == C() && other.rd_ == C());
	if (dr_ == rd_) return (other.dr_ == C() || other.dr_ == dr_) && (other.rd_ == C() || other.rd_ == dr_) && !(other.dr_ == C() && other.rd_ == C());
	return (other.dr_ == dr_ && other.rd_ == rd_) || (other.dr_ == rd_ && other.rd_ == dr_);
    }

    bool operator!=(const CorrFuncBase& other) const { return !operator==(other); }
};

template<class C>
class CorrFunc : public CorrFuncBase<C> {
    using Base = CorrFuncBase<C>;
public:
    CorrFunc() = default;

    CorrFunc(const CorrFunc& other)
	: Base(other) {}

    CorrFunc(const C& dd)
	: Base(dd) {}

    CorrFunc(const Base& cfb)
	: Base(cfb) {}

    std::string toString() const {
	std::ostringstream oss;
	oss << "CorrFunc(";
	std::string pad = "";
	for (std::size_t i = 0; i < oss.str().size() - 1; i++) {
	    pad += " ";
	}
	oss << std::endl << pad << " DD=" << this->dd_ << ",";
	oss << std::endl << pad << " RR=" << this->rr_ << ",";
	oss << std::endl << pad << " DR=" << this->dr_ << ",";
	oss << std::endl << pad << " RD=" << this->rd_ ;
	oss << std::endl << pad << ")";
	return oss.str();
    }
};

template<class C>
std::ostream& operator<<(std::ostream& os, const CorrFunc<C>& cf) {
    os << cf.toString();
    return os;
}

#endif
