#include <utility>

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
#include <omp.h>
constexpr int OMP_NUM_THREADS = omp_num_threads;
#else
constexpr bool _OPENMP = false;
constexpr int OMP_NUM_THREADS = 1;

inline void omp_set_num_threads(int) {}

inline int omp_get_num_threads() { return 1; }

#endif

template<typename T, typename std::enable_if_t<
        std::is_arithmetic_v<T>,
        int> = 0>

inline bool check_val_in_limits(const T& val, const T& min, const T& max) {
    return std::isfinite(val) && (!std::isfinite(min) || val >= min)
           && (!std::isfinite(max) || val <= max);
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
    std::string name = "BinSpecifier";

    double get_diff(double value) const {
        return log_binning ? log(value / bin_min) : value - bin_min;
    }

    double get_diff() const {
        return get_diff(bin_max);
    }

    std::size_t get_num_bins() const {
        return (std::size_t) ceil(get_diff() / bin_size);
    }

    double get_size() const {
        return get_diff() / (double) nbins;
    }

    void on_update() {
        if (_min_set && bin_min <= 0.0) {
            log_binning = false;
            _log_set = true;
        }
        if (_log_set && _min_set && _max_set) {
            if (_nbins_set && !_size_set) {
                bin_size = get_size();
                _size_set = true;
            }
            else if (_size_set && !_nbins_set) {
                nbins = get_num_bins();
                _nbins_set = true;
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
            : _min_set(true),
              _max_set(true),
              _size_set(true),
              _nbins_set(false),
              _log_set(true),
              bin_min(min),
              bin_max(max),
              log_binning(log_bins),
              bin_size(width) {
        on_update();
    }

    BinSpecifier(double min,
                 double max,
                 double width,
                 bool log_bins,
                 std::string name_in)
            : _min_set(true),
              _max_set(true),
              _size_set(true),
              _nbins_set(false),
              _log_set(true),
              bin_min(min),
              bin_max(max),
              log_binning(log_bins),
              bin_size(width),
              name(std::move(name_in)) {
        on_update();
    }

    BinSpecifier(double min, double max, std::size_t num_bins, bool log_bins)
            : _min_set(true),
              _max_set(true),
              _size_set(false),
              _nbins_set(true),
              _log_set(true),
              bin_min(min),
              bin_max(max),
              log_binning(log_bins),
              nbins(num_bins) {
        on_update();
    }

    BinSpecifier(double min,
                 double max,
                 std::size_t num_bins,
                 bool log_bins,
                 std::string name_in)
            : _min_set(true),
              _max_set(true),
              _size_set(false),
              _nbins_set(true),
              _log_set(true),
              bin_min(min),
              bin_max(max),
              log_binning(log_bins),
              nbins(num_bins),
              name(std::move(name_in)) {
        on_update();
    }

    // copy assignment operator
    BinSpecifier& operator=(const BinSpecifier&) = default;

    // move assignment operator
    BinSpecifier& operator=(BinSpecifier&&) = default;

    bool operator==(const BinSpecifier& other) const {
        return ((_is_set == other._is_set) && (log_binning == other.log_binning)
                && (nbins == other.nbins)
                && math::isclose(bin_size, other.bin_size)
                && math::isclose(bin_min, other.bin_min)
                && math::isclose(bin_max, other.bin_max));
    }

    bool operator!=(const BinSpecifier& other) const {
        return !(*this == other);
    }

    // Instead of copying everything with this one, just update the parameters
    // in this that are set in other. With this function, we prefer values in
    // other. See 'fill' for a version that prefers valuse in this
    void update(const BinSpecifier& other) {
        if (operator==(other)) { return; }  // other is the same as this already
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
        if (operator==(other)) { return; }  // other is the same as this already
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
        // assume keep constant bin_size and update nbins
        _nbins_set = false;
        on_update();
    }

    double get_bin_max() const {
        return bin_max;
    }

    void set_bin_max(double max) {
        bin_max = max;
        _max_set = true;
        // assume keep constant bin_size and update nbins
        _nbins_set = false;
        on_update();
    }

    double get_bin_size() const {
        return bin_size;
    }

    void set_bin_size(double size) {
        bin_size = size;
        _size_set = true;
        // assume update nbins
        _nbins_set = false;
        on_update();
    }

    std::size_t get_nbins() const {
        return nbins;
    }

    void set_nbins(std::size_t num_bins) {
        nbins = num_bins;
        _nbins_set = true;
        // assume update bin_size
        _size_set = false;
        on_update();
    }

    bool get_log_binning() const {
        return log_binning;
    }

    void set_log_binning(bool log_bins) {
        log_binning = log_bins;
        _log_set = true;
        // assume keep constant nbins and update bin_size
        _size_set = false;
        on_update();
    }

    const std::string& get_name() const {
        return name;
    }

    void set_name(const std::string& name_in) {
        name = name_in;
    }

    int assign_bin(double value) const {
        if (!_is_set) {
            throw std::runtime_error("Cannot assign bin if values are not set");
        }
        if (operator==(BinSpecifier())) { return -1; }
        if (!check_val_in_limits(value, bin_min, bin_max)) { return -1; }
        return (int) floor(get_diff(value) / bin_size);
    }

    double log_step_func(int index) const {
        return bin_min
               * std::exp(index * bin_size);
    }

    double lin_step_func(int index) const {
        return bin_min
               + (index * bin_size);
    }

    double log_step_func_center(int index) const {
        return bin_min
               * std::exp((index
                           + 0.5)
                          * bin_size);
    }

    double lin_step_func_center(int index) const {
        return bin_min
               + ((index + 0.5)
                  * bin_size);
    }

    std::vector<double> lower_bin_edges() const {
        if (!_is_set) { return {}; }
        std::vector<int> indices(nbins);
        std::iota(indices.begin(), indices.end(), 0);
        std::vector<double> vec(nbins);
        if (log_binning) {
            std::transform(indices.begin(),
                           indices.end(),
                           vec.begin(),
                           [this](int i) { return log_step_func(i); });
        }
        else {
            std::transform(indices.begin(),
                           indices.end(),
                           vec.begin(),
                           [this](int i) { return lin_step_func(i); });
        }
        return vec;
    }

    std::vector<double> upper_bin_edges() const {
        if (!_is_set) { return {}; }
        std::vector<int> indices(nbins);
        std::iota(indices.begin(), indices.end(), 1);
        std::vector<double> vec(nbins);
        if (log_binning) {
            std::transform(indices.begin(),
                           indices.end(),
                           vec.begin(),
                           [this](int i) { return log_step_func(i); });
        }
        else {
            std::transform(indices.begin(),
                           indices.end(),
                           vec.begin(),
                           [this](int i) { return lin_step_func(i); });
        }
        return vec;
    }

    std::vector<double> bin_edges() const {
        if (!_is_set) { return {}; }
        std::vector<int> indices(nbins + 1);
        std::iota(indices.begin(), indices.end(), 0);
        std::vector<double> vec(nbins + 1);
        if (log_binning) {
            std::transform(indices.begin(),
                           indices.end(),
                           vec.begin(),
                           [this](int i) { return log_step_func(i); });
        }
        else {
            std::transform(indices.begin(),
                           indices.end(),
                           vec.begin(),
                           [this](int i) { return lin_step_func(i); });
        }
        return vec;
    }

    std::vector<double> bin_centers() const {
        if (!_is_set) { return {}; }
        std::vector<int> indices(nbins);
        std::iota(indices.begin(), indices.end(), 0);
        std::vector<double> vec(nbins);
        if (log_binning) {
            std::transform(indices.begin(),
                           indices.end(),
                           vec.begin(),
                           [this](int i) { return log_step_func_center(i); });
        }
        else {
            std::transform(indices.begin(),
                           indices.end(),
                           vec.begin(),
                           [this](int i) { return lin_step_func_center(i); });
        }
        return vec;
    }

    std::vector<double> bin_widths() const {
        if (!_is_set) { return {}; }
        std::vector<double> vec(nbins);
        if (log_binning) {
            std::vector<int> indices(nbins);
            std::iota(indices.begin(), indices.end(), 0);
            std::transform(indices.begin(),
                           indices.end(),
                           vec.begin(),
                           [this](int i) {
                             return (std::exp(bin_size) - 1)
                                    * log_step_func(i);
                           });
        }
        else {
            std::fill(vec.begin(), vec.end(), bin_size);
        }
        return vec;
    }

private:
    template<std::size_t> friend
    class NNCountsNDBase;

    template<std::size_t> friend
    class NNCountsND;

    template<std::size_t> friend
    class ExpectedNNCountsNDBase;

    template<std::size_t> friend
    class ExpectedNNCountsND;

    friend std::size_t bin_size_multiply(std::size_t a, const BinSpecifier& b);

    friend double bin_min_combine(double a, const BinSpecifier& b);

    friend double bin_max_combine(double a, const BinSpecifier& b);
};


inline std::size_t
bin_size_multiply(std::size_t a, const BinSpecifier& b) {
    return b._is_set ? a * b.nbins : 0;
}

inline double bin_min_combine(double a, const BinSpecifier& b) {
    return b._is_set ? std::sqrt(math::square(a)
                                 + math::square(b.bin_min)) : 0.0;
}

inline double bin_max_combine(double a, const BinSpecifier& b) {
    return b._is_set ? std::sqrt(math::square(a)
                                 + math::square(b.bin_max)) : 0.0;
}

template<std::size_t N>
std::size_t get_max_index(const std::array<BinSpecifier, N>& binners) {
    return std::accumulate(binners.begin(),
                           binners.end(),
                           (std::size_t) 1,
                           bin_size_multiply);
}

inline double
get_r_min(const BinSpecifier& rp_binner, const BinSpecifier& rl_binner) {
    return bin_min_combine(rp_binner.get_bin_min(), rl_binner);
}

inline double
get_r_max(const BinSpecifier& rp_binner, const BinSpecifier& rl_binner) {
    return bin_max_combine(rp_binner.get_bin_max(), rl_binner);
}


template<std::size_t N>
struct get_r_min_impl {
  static double call(const std::array<BinSpecifier, N>& binners) {
      return std::accumulate(binners.begin(),
                             binners.end(),
                             0.0,
                             bin_min_combine);
  }
};


template<>
struct get_r_min_impl<3> {
  static double call(const std::array<BinSpecifier, 3>& binners) {
      return get_r_min(binners[0], binners[1]);
  }
};


template<std::size_t N>
double get_r_min(const std::array<BinSpecifier, N>& binners) {
    return get_r_min_impl<N>::call(binners);
}


template<std::size_t N>
struct get_r_max_impl {
  static double call(const std::array<BinSpecifier, N>& binners) {
      return std::accumulate(binners.begin(),
                             binners.end(),
                             0.0,
                             bin_max_combine);
  }
};


template<>
struct get_r_max_impl<3> {
  static double call(const std::array<BinSpecifier, 3>& binners) {
      return get_r_max(binners[0], binners[1]);
  }
};


template<std::size_t N>
double get_r_max(const std::array<BinSpecifier, N>& binners) {
    return get_r_max_impl<N>::call(binners);
}


struct SPos {
private:
    double ra_ = 0.0;
    double dec_ = 0.0;
    double r_ = math::dnan;
    double z_ = math::dnan;
    std::array<double, 3> uvec_ = arrays::make_filled_array<double, 3>();
    std::array<double, 3> rvec_ = arrays::make_filled_array<3>(math::dnan);
    bool is_initialized_ = false;
public:
    // empty constructor
    SPos() = default;

    // copy constructor
    SPos(const SPos&) = default;

    // move constructor
    SPos(SPos&&) = default;

    SPos(double ra, double dec, double r, double z)
            : ra_(check_ra(ra)),
              dec_(check_dec(dec)),
              r_(r),
              z_(z),
              uvec_(get_nxyz_array(ra_, dec_)),
              rvec_(arrays::multiply_array_by_constant(uvec_, r_)),
              is_initialized_(true) {}

    bool operator==(const SPos& other) const {
        bool same_init = is_initialized_ == other.is_initialized_;
        if (!is_initialized_) { return same_init; }
        bool
                rclose = std::isnan(r_) ? std::isnan(other.r_) : math::isclose(
                r_,
                other.r_);
        bool
                zclose = std::isnan(z_) ? std::isnan(other.z_) : math::isclose(
                z_,
                other.z_);
        return same_init && math::isclose(ra_, other.ra_)
               && math::isclose(dec_, other.dec_) && rclose && zclose;
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

    std::array<double, 3> uvec() const { return uvec_; }

    std::array<double, 3> rvec() const { return rvec_; }

    double dot_norm(const SPos& other) const {
        return operator==(other)
               ? 1.0
               : math::isclose(other.uvec_,
                               uvec_)
                 ? 1.0
                 : (uvec_[0] * other.uvec_[0]) + (uvec_[1] * other.uvec_[1])
                   + (uvec_[2] * other.uvec_[2]);
    }

    double dot_mag(const SPos& other) const {
        return r_ * other.r_ * dot_norm(other);
    }

    double distance_zbar(const SPos& other) const {
        return std::isnan(z_) || std::isnan(other.z_)
               ? math::dnan
               : 0.5 * (z_ + other.z_);
    }

    double distance_par(const SPos& other) const {
        return std::isnan(r_) || std::isnan(other.r_)
               ? math::dnan
               : math::dsqrt_2 * std::sqrt(1.0 + dot_norm(other)) * std::fabs(r_
                                                                              - other.r_);
    }

    double distance_perp(const SPos& other) const {
        return std::isnan(r_) || std::isnan(other.r_)
               ? math::dnan
               : math::dsqrt_2 * std::sqrt(1.0 - dot_norm(other))
                 * (r_ + other.r_);
    }

    std::array<double, 3> distance_vector(const SPos& other) const {
        return operator==(other)
               ? (std::array<
                        double,
                        3>) {{0.0, 0.0, 0.0}}
               : (std::array<
                        double,
                        3>) {
                        {rvec_[0] - other.rvec_[0], rvec_[1] - other.rvec_[1],
                         rvec_[2] - other.rvec_[2]}};
    }

    double distance_magnitude(const SPos& other) const {
        if (operator==(other)) { return 0.0; }
        auto dist_vec = distance_vector(other);
        return std::sqrt(std::inner_product(dist_vec.begin(),
                                            dist_vec.end(),
                                            dist_vec.begin(),
                                            0.0));
    }

    bool check_box(const SPos& other, double max) const {
        for (const auto& d : distance_vector(other)) {
            if (!check_val_in_limits(std::fabs(d), 0.0, max)) { return false; }
        }
        return true;
    }

    bool check_box(const SPos& other, const BinSpecifier& binner) const {
        return check_box(other, binner.get_bin_max());
    }

    bool check_shell(const SPos& other, double min, double max) const {
        for (auto d : distance_vector(other)) {
            if (std::fabs(d) < min || std::fabs(d) > max) { return false; }
        }
        return true;
    }

    bool check_shell(const SPos& other,
                     const BinSpecifier& binner) const {
        return check_shell(other,
                           binner.get_bin_min(),
                           binner.get_bin_max());
    }

    bool check_shell(const SPos& other, double max) const {
        return check_shell(other,
                           0.0,
                           max);
    }

    bool check_limits(const SPos& other,
                      double rp_min,
                      double rp_max,
                      double rl_min,
                      double rl_max) const {
        if (!check_val_in_limits(distance_perp(other),
                                 rp_min,
                                 rp_max)) {
            return false;
        }
        return check_val_in_limits(distance_par(other), rl_min, rl_max);
    }

    bool check_limits(const SPos& other,
                      const BinSpecifier& rp_binner,
                      const BinSpecifier& rl_binner) const {
        return check_limits(other,
                            rp_binner.get_bin_min(),
                            rp_binner.get_bin_max(),
                            rl_binner.get_bin_min(),
                            rl_binner.get_bin_max());
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "SPos(";
        if (is_initialized_) {
            oss << "ra = " << ra_;
            oss << ", dec = " << dec_;
            oss << ", r = ";
            if (math::isnan(r_)) { oss << math::snan; }
            else { oss << r_; }
            oss << ", z = ";
            if (math::isnan(z_)) { oss << math::snan; }
            else { oss << z_; }
        }
        oss << ")";
        return oss.str();
    }

    friend std::ostream& operator<<(std::ostream& os, const SPos& s) {
        os << s.toString();
        return os;
    }

private:
    static double check_ra(double value) {
        if (!check_val_in_limits(value, 0.0, 360.0)) {
            throw std::invalid_argument("RA value " + std::to_string(value)
                                        + " outside of allowed range [0.0, 360.0]");
        }
        return value;
    }

    static double check_dec(double value) {
        if (!check_val_in_limits(value, -90.0, 90.0)) {
            throw std::invalid_argument("DEC value " + std::to_string(value)
                                        + " outside of allowed range [-90.0, 90.0]");
        }
        return value;
    }

    friend struct Pos;
};


struct Pos {
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

    std::array<double, 3> uvec() const { return tpos_.uvec_; }

    std::array<double, 3> rtvec() const { return tpos_.rvec_; }

    std::array<double, 3> rovec() const { return opos_.rvec_; }

    double ra() const { return tpos_.ra_; }

    double dec() const { return tpos_.dec_; }

    double rt() const { return tpos_.r_; }

    double ro() const { return opos_.r_; }

    double zt() const { return tpos_.z_; }

    double zo() const { return opos_.z_; }

    SPos tpos() const { return tpos_; }

    SPos opos() const { return opos_; }

    bool has_true() const {
        return tpos_.is_initialized_ && !std::isnan(tpos_.r_);
    }

    bool has_obs() const {
        return opos_.is_initialized_ && !std::isnan(opos_.r_);
    }

    bool operator==(const Pos& other) const {
        return std::isnan(tpos_.r_) == std::isnan(other.opos_.r_)
               && std::isnan(opos_.r_) == std::isnan(other.opos_.r_)
               && tpos_ == other.tpos_ && opos_ == other.opos_;
    }

    bool operator!=(const Pos& other) const { return !(*this == other); }

    double dot_norm(const SPos& other) const {
        if (!other.is_initialized_) {
            throw std::runtime_error(
                    "Cannot take dot product with unset position");
        }
        if (!(tpos_.is_initialized_ || opos_.is_initialized_)) {
            throw std::runtime_error(
                    "Cannot take dot product when no positions are set in self");
        }
        return tpos_.is_initialized_ ? tpos_.dot_norm(other) : opos_.dot_norm(
                other);
    }

    double dot_norm(const Pos& other) const {
        if (!(other.tpos_.is_initialized_ || other.opos_.is_initialized_)) {
            throw std::runtime_error(
                    "Cannot take dot product with unset true and observed positions");
        }
        return other.tpos_.is_initialized_ ? dot_norm(other.tpos_) : dot_norm(
                other.opos_);
    }

    double dot_mag(const SPos& other) const {
        if (!other.is_initialized_ || math::isnan(other.r_)) {
            throw std::runtime_error(
                    "Cannot take dot product with unset position or NaN distance");
        }
        if (!((tpos_.is_initialized_ && !math::isnan(tpos_.r_))
              || (opos_.is_initialized_ && !std::isnan(opos_.r_)))) {
            throw std::runtime_error(
                    "Cannot take dot product when no positions without NaN distances are set in self");
        }
        return tpos_.is_initialized_ && !math::isnan(tpos_.r_) ? tpos_.dot_mag(
                other) : opos_.dot_mag(other);
    }

    double dot_mag(const Pos& other) const {
        if (!((other.tpos_.is_initialized_ && !math::isnan(other.tpos_.r_))
              || (other.opos_.is_initialized_
                  && !math::isnan(other.opos_.r_)))) {
            throw std::runtime_error(
                    "Cannot take dot product with invalid true and observed positions");
        }
        return other.tpos_.is_initialized_ && !math::isnan(other.tpos_.r_)
               ? dot_mag(other.tpos_)
               : dot_mag(other.opos_);
    }

    double
    zbar_t(const SPos& other) const { return tpos_.distance_zbar(other); }

    double zbar_t(const Pos& other) const { return zbar_t(other.tpos_); }

    double
    zbar_o(const SPos& other) const { return opos_.distance_zbar(other); }

    double zbar_o(const Pos& other) const { return zbar_o(other.opos_); }

    std::tuple<double, double> distance_zbar(const Pos& other) const {
        return std::make_tuple(zbar_t(other),
                               zbar_o(other));
    }

    double
    r_par_t(const SPos& other) const { return tpos_.distance_par(other); }

    double r_par_t(const Pos& other) const { return r_par_t(other.tpos_); }

    double r_par_t_signed(const Pos& other) const {
        return (has_obs() && other.has_obs() ?
                math::signof(opos_.r_ - other.opos_.r_)
                * math::signof(tpos_.r_ - other.tpos_.r_) : 1)
               * this->r_par_t(other);
    }

    double
    r_par_o(const SPos& other) const { return opos_.distance_par(other); }

    double r_par_o(const Pos& other) const { return r_par_o(other.opos_); }

    std::tuple<double, double> distance_par(const Pos& other) const {
        return std::make_tuple(r_par_t_signed(other), r_par_o(other));
    }

    double
    r_perp_t(const SPos& other) const { return tpos_.distance_perp(other); }

    double r_perp_t(const Pos& other) const { return r_perp_t(other.tpos_); }

    double
    r_perp_o(const SPos& other) const { return opos_.distance_perp(other); }

    double r_perp_o(const Pos& other) const { return r_perp_o(other.opos_); }

    std::tuple<double, double> distance_perp(const Pos& other) const {
        return std::make_tuple(r_perp_t(other), r_perp_o(other));
    }

    std::array<double, 3> distance_vector_t(const SPos& other) const {
        return tpos_.distance_vector(other);
    }

    std::array<double, 3> distance_vector_t(const Pos& other) const {
        return tpos_.distance_vector(other.tpos_);
    }

    std::array<double, 3> distance_vector_o(const SPos& other) const {
        return opos_.distance_vector(other);
    }

    std::array<double, 3> distance_vector_o(const Pos& other) const {
        return opos_.distance_vector(other.opos_);
    }

    std::array<double, 3>
    distance_vector(const SPos& other, bool use_true) const {
        return use_true
               ? tpos_.distance_vector(other)
               : opos_.distance_vector(other);
    }

    std::array<double, 3>
    distance_vector(const Pos& other, bool use_true) const {
        return use_true
               ? tpos_.distance_vector(other.tpos_)
               : opos_.distance_vector(other.opos_);
    }

    double distance_magnitude_t(const SPos& other) const {
        return tpos_.distance_magnitude(other);
    }

    double distance_magnitude_t(const Pos& other) const {
        return tpos_.distance_magnitude(other.tpos_);
    }

    double distance_magnitude_o(const SPos& other) const {
        return opos_.distance_magnitude(other);
    }

    double distance_magnitude_o(const Pos& other) const {
        return opos_.distance_magnitude(other.opos_);
    }

    double distance_magnitude(const SPos& other,
                              const bool use_true) const {
        return use_true
               ? tpos_.distance_magnitude(other)
               : opos_.distance_magnitude(other);
    }

    double distance_magnitude(const Pos& other,
                              const bool use_true) const {
        return use_true
               ? tpos_.distance_magnitude(other.tpos_)
               : opos_.distance_magnitude(other.opos_);
    }

    bool check_box(const SPos& other, double max) const {
        if (!other.is_initialized_ || std::isnan(other.r_)) {
            throw std::runtime_error("No distance set in other");
        }
        if (!has_obs()) {
            if (!has_true()) {
                throw std::runtime_error(
                        "Neither true nor observed distances set in self");
            }
            return tpos_.check_box(other, max);
        }
        return opos_.check_box(other, max);
    }

    bool check_box(const SPos& other, const BinSpecifier& binner) const {
        return check_box(other, binner.get_bin_max());
    }

    bool check_box(const Pos& other, double max) const {
        if (!(has_obs() && other.has_obs())) {
            if (!(has_true() && other.has_true())) {
                throw std::runtime_error(
                        "Cannot mix true and observed distances");
            }
            return tpos_.check_box(other.tpos_, max);
        }
        return opos_.check_box(other.opos_, max);
    }

    bool check_box(const Pos& other, const BinSpecifier& binner) const {
        return check_box(other, binner.get_bin_max());
    }

    bool check_shell(const SPos& other, double min, double max) const {
        if (!other.is_initialized_ || std::isnan(other.r_)) {
            throw std::runtime_error("No distance set in other");
        }
        if (!has_obs()) {
            if (!has_true()) {
                throw std::runtime_error(
                        "Neither true nor observed distances set in self");
            }
            return tpos_.check_shell(other, min, max);
        }
        return opos_.check_shell(other, min, max);
    }

    bool check_shell(const SPos& other,
                     const BinSpecifier& binner) const {
        return check_shell(other,
                           binner.get_bin_min(),
                           binner.get_bin_max());
    }

    bool check_shell(const SPos& other, double max) const {
        return check_shell(other,
                           0.0,
                           max);
    }

    bool check_shell(const Pos& other, double min, double max) const {
        if (!(has_obs() && other.has_obs())) {
            if (!(has_true() && other.has_true())) {
                throw std::runtime_error(
                        "Cannot mix true and observed distances");
            }
            return tpos_.check_shell(other.tpos_, min, max);
        }
        return opos_.check_shell(other.opos_, min, max);
    }

    bool check_shell(const Pos& other,
                     const BinSpecifier& binner) const {
        return check_shell(other,
                           binner.get_bin_min(),
                           binner.get_bin_max());
    }

    bool check_shell(const Pos& other, double max) const {
        return check_shell(other,
                           0.0,
                           max);
    }

    bool check_limits(const SPos& other,
                      double rp_min,
                      double rp_max,
                      double rl_min,
                      double rl_max) const {
        if (!other.is_initialized_ || std::isnan(other.r_)) {
            throw std::runtime_error("No distance set in other");
        }
        if (!has_obs()) {
            if (!has_true()) {
                throw std::runtime_error(
                        "Neither true nor observed distances set in self");
            }
            return tpos_.check_limits(other, rp_min, rp_max, rl_min, rl_max);
        }
        return opos_.check_limits(other, rp_min, rp_max, rl_min, rl_max);
    }

    bool check_limits(const SPos& other,
                      const BinSpecifier& rp_binner,
                      const BinSpecifier& rl_binner) const {
        return check_limits(other,
                            rp_binner.get_bin_min(),
                            rp_binner.get_bin_max(),
                            rl_binner.get_bin_min(),
                            rl_binner.get_bin_max());
    }

    bool check_limits(const Pos& other,
                      double rp_min,
                      double rp_max,
                      double rl_min,
                      double rl_max) const {
        if (!(has_obs() && other.has_obs())) {
            if (!(has_true() && other.has_true())) {
                throw std::runtime_error(
                        "Cannot mix true and observed distances");
            }
            return tpos_.check_limits(other.tpos_,
                                      rp_min,
                                      rp_max,
                                      rl_min,
                                      rl_max);
        }
        return opos_.check_limits(other.opos_, rp_min, rp_max, rl_min, rl_max);
    }

    bool check_limits(const Pos& other,
                      const BinSpecifier& rp_binner,
                      const BinSpecifier& rl_binner) const {
        return check_limits(other,
                            rp_binner.get_bin_min(),
                            rp_binner.get_bin_max(),
                            rl_binner.get_bin_min(),
                            rl_binner.get_bin_max());
    }

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
    static double get_r_perp(const SPos& pos1, const SPos& pos2, bool= true) {
        return pos1.distance_perp(pos2);
    }

    static double get_r_perp(const Pos& pos1, const SPos& pos2, bool use_true) {
        return use_true ? pos1.r_perp_t(pos2) : pos1.r_perp_o(pos2);
    }

    static double get_r_perp(const SPos& pos1, const Pos& pos2, bool use_true) {
        return pos1.distance_perp(use_true ? pos2.tpos() : pos2.opos());
    }

    static double get_r_perp(const Pos& pos1, const Pos& pos2, bool use_true) {
        return use_true ? pos1.r_perp_t(pos2) : pos1.r_perp_o(pos2);
    }

    static double
    get_r_par(const SPos& pos1, const SPos& pos2, bool= true, bool= false) {
        return pos1.distance_par(pos2);
    }

    static double
    get_r_par(const Pos& pos1, const SPos& pos2, bool use_true, bool= false) {
        return use_true ? pos1.r_par_t(pos2) : pos1.r_par_o(pos2);
    }

    static double
    get_r_par(const SPos& pos1, const Pos& pos2, bool use_true, bool= false) {
        return pos1.distance_par(use_true ? pos2.tpos() : pos2.opos());
    }

    static double get_r_par(const Pos& pos1,
                            const Pos& pos2,
                            bool use_true,
                            bool use_signed = false) {
        return use_true ? use_signed ? pos1.r_par_t_signed(pos2) : pos1.r_par_t(
                pos2) : pos1.r_par_o(pos2);
    }

    static double get_zbar(const SPos& pos1, const SPos& pos2, bool= true) {
        return pos1.distance_zbar(pos2);
    }

    static double get_zbar(const Pos& pos1, const SPos& pos2, bool use_true) {
        return use_true ? pos1.zbar_t(pos2) : pos1.zbar_o(pos2);
    }

    static double get_zbar(const SPos& pos1, const Pos& pos2, bool use_true) {
        return pos1.distance_zbar(use_true ? pos2.tpos() : pos2.opos());
    }

    static double get_zbar(const Pos& pos1, const Pos& pos2, bool use_true) {
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

    Separation(const SPos& pos1,
               const SPos& pos2,
               std::size_t i1,
               std::size_t i2)
            : r_perp(get_r_perp(pos1, pos2)),
              r_par(get_r_par(pos1, pos2)),
              zbar(get_zbar(pos1, pos2)),
              id1(i1),
              id2(i2) {}

    Separation(const Pos& pos1,
               const SPos& pos2,
               std::size_t i1,
               std::size_t i2,
               bool use_true)
            : r_perp(get_r_perp(pos1, pos2, use_true)),
              r_par(get_r_par(pos1, pos2, use_true)),
              zbar(get_zbar(pos1, pos2, use_true)),
              id1(i1),
              id2(i2) {}

    Separation(const SPos& pos1,
               const Pos& pos2,
               std::size_t i1,
               std::size_t i2,
               bool use_true)
            : r_perp(get_r_perp(pos1, pos2, use_true)),
              r_par(get_r_par(pos1, pos2, use_true)),
              zbar(get_zbar(pos1, pos2, use_true)),
              id1(i1),
              id2(i2) {}

    Separation(const Pos& pos1,
               const Pos& pos2,
               std::size_t i1,
               std::size_t i2,
               bool use_true,
               bool use_signed = false)
            : r_perp(get_r_perp(pos1, pos2, use_true)),
              r_par(get_r_par(pos1, pos2, use_true, use_signed)),
              zbar(get_zbar(pos1, pos2, use_true)),
              id1(i1),
              id2(i2) {}

    // copy assignment
    Separation& operator=(const Separation&) = default;

    // move assignment
    Separation& operator=(Separation&&) = default;

    bool operator==(const Separation& other) const {
        return id1 == other.id1
               && id2 == other.id2
               && (std::isnan(r_perp)
                   ? std::isnan(other.r_perp)
                   : math::isclose(r_perp, other.r_perp))
               && (std::isnan(r_par) ? std::isnan(other.r_par) : math::isclose(
                       r_par,
                       other.r_par))
               && (std::isnan(zbar) ? std::isnan(other.zbar) : math::isclose(
                       zbar,
                       other.zbar));
    }

    bool
    operator!=(const Separation& other) const { return !operator==(other); }

    std::string toString() const {
        std::ostringstream oss;
        oss << "Separation(";
        oss << "r_perp = ";
        if (math::isnan(r_perp)) { oss << math::snan; }
        else { oss << r_perp; }
        oss << ", r_par = ";
        if (math::isnan(r_par)) { oss << math::snan; }
        else { oss << r_par; }
        oss << ", zbar = ";
        if (math::isnan(zbar)) { oss << math::snan; }
        else { oss << zbar; }
        oss << ", id1 = ";
        if (math::isnan(id1)) { oss << math::snan; }
        else { oss << id1; }
        oss << ", id2 = ";
        if (math::isnan(id2)) { oss << math::snan; }
        else { oss << id2; }
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

  TOSeparation(double rpt,
               double rlt,
               double zbt,
               double rpo,
               double rlo,
               double zbo,
               std::size_t i1,
               std::size_t i2)
          : tsep(Separation(rpt, rlt, zbt, i1, i2)),
            osep(Separation(rpo, rlo, zbo, i1, i2)) {}

  TOSeparation(const Pos& pos1, const Pos& pos2, std::size_t i1, std::size_t i2)
          : tsep(Separation(pos1, pos2, i1, i2, true, true)),
            osep(Separation(pos1, pos2, i1, i2, false)) {}

  TOSeparation(const Separation& true_sep, const Separation& obs_sep)
          : tsep(true_sep), osep(obs_sep) {}

  bool operator==(const TOSeparation& other) const {
      return tsep == other.tsep && osep == other.osep;
  }

  bool
  operator!=(const TOSeparation& other) const { return !operator==(other); }

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


std::vector<Separation> get_auto_separations(const std::vector<SPos>& pos,
                                             const BinSpecifier& rp_bins,
                                             const BinSpecifier& rl_bins,
                                             int num_threads = OMP_NUM_THREADS);

std::vector<Separation> get_auto_separations(const std::vector<Pos>& pos,
                                             const BinSpecifier& rp_bins,
                                             const BinSpecifier& rl_bins,
                                             bool use_true,
                                             int num_threads = OMP_NUM_THREADS);

std::vector<Separation> get_cross_separations(const std::vector<SPos>& pos1,
                                              const std::vector<SPos>& pos2,
                                              const BinSpecifier& rp_bins,
                                              const BinSpecifier& rl_bins,
                                              int num_threads = OMP_NUM_THREADS);

std::vector<Separation> get_cross_separations(const std::vector<Pos>& pos1,
                                              const std::vector<Pos>& pos2,
                                              const BinSpecifier& rp_bins,
                                              const BinSpecifier& rl_bins,
                                              bool use_true,
                                              int num_threads = OMP_NUM_THREADS);

std::vector<Separation> get_separations(const std::vector<SPos>& pos1,
                                        const std::vector<SPos>& pos2,
                                        const BinSpecifier& rp_bins,
                                        const BinSpecifier& rl_bins,
                                        bool is_auto,
                                        int num_threads = OMP_NUM_THREADS);

std::vector<Separation> get_separations(const std::vector<Pos>& pos1,
                                        const std::vector<Pos>& pos2,
                                        const BinSpecifier& rp_bins,
                                        const BinSpecifier& rl_bins,
                                        bool use_true,
                                        bool is_auto,
                                        int num_threads = OMP_NUM_THREADS);

std::vector<TOSeparation> get_auto_separations(const std::vector<Pos>& pos,
                                               const BinSpecifier& rp_bins,
                                               const BinSpecifier& rl_bins,
                                               int num_threads = OMP_NUM_THREADS);

std::vector<TOSeparation> get_cross_separations(const std::vector<Pos>& pos1,
                                                const std::vector<Pos>& pos2,
                                                const BinSpecifier& rp_bins,
                                                const BinSpecifier& rl_bins,
                                                int num_threads = OMP_NUM_THREADS);

std::vector<TOSeparation> get_separations(const std::vector<Pos>& pos1,
                                          const std::vector<Pos>& pos2,
                                          const BinSpecifier& rp_bins,
                                          const BinSpecifier& rl_bins,
                                          bool is_auto,
                                          int num_threads = OMP_NUM_THREADS);

template<std::size_t N>
std::size_t get_1d_indexer_from_nd(const std::array<int, N>& indices,
                                   const std::array<BinSpecifier, N>& bins) {
    std::size_t index = 0;
    for (std::size_t i = 0; i < N; i++) {
        if (indices[i] < 0 || indices[i] >= (int) bins[i].get_nbins()) {
            throw std::out_of_range(
                    "Invalid index " + std::to_string(indices[i])
                    + " for dimension " + std::to_string(i) + " with size "
                    + std::to_string(bins[i].get_nbins()));
        }
        index *= bins[i].get_nbins();
        index += indices[i];
    }
    return index;
}


template<std::size_t N>
class CorrFuncNDBase;


template<std::size_t N>
class NNCountsNDBase {
    using BSType = std::array<BinSpecifier, N>;
    using NNType = NNCountsND<N>;
protected:
    BSType binners_ = arrays::make_filled_array<BinSpecifier, N>();
    std::size_t n_tot_ = 0, max_index_ = 0;
    std::vector<int> counts_ = {};
    double r_min = 0.0, r_max = 0.0;

    void on_bin_update() {
        n_tot_ = 0;
        max_index_ = get_max_index(binners_);
        counts_ = std::vector<int>(max_index_, 0);
        r_min = get_r_min(binners_);
        r_max = get_r_max(binners_);
    }

    void process_separation(const std::array<double, N>& values) {
        n_tot_++;
        int index = get_bin(values);
        if (index < 0) { return; }
        counts_[index]++;
    }

protected:
    // empty constructor
    NNCountsNDBase() {}

    // copy constructor
    NNCountsNDBase(const NNCountsNDBase&) = default;

    // constructor for pickling support
    NNCountsNDBase(BSType bins,
                   std::vector<int> counts,
                   std::size_t n_tot)
            : binners_(std::move(bins)),
              n_tot_(n_tot),
              max_index_(get_max_index(binners_)),
              counts_(std::move(counts)),
              r_min(get_r_min(binners_)),
              r_max(get_r_max(binners_)) {}

    // from binning
    explicit NNCountsNDBase(BSType bins)
            : binners_(std::move(bins)),
              n_tot_(0),
              max_index_(get_max_index(binners_)),
              counts_(max_index_, 0),
              r_min(get_r_min(binners_)),
              r_max(get_r_max(binners_)) {}

public:
    void update_binning(std::size_t binner_index,
                        const BinSpecifier& new_binner,
                        bool prefer_old = true) {
        if (binner_index >= N) {
            throw std::out_of_range(
                    "Invalid index " + std::to_string(binner_index) + " for "
                    + std::to_string(N) + "D binning");
        }
        if (prefer_old) { binners_[binner_index].fill(new_binner); }
        else { binners_[binner_index].update(new_binner); }
        on_bin_update();
    }

    inline static const std::string
            class_name = "NNCounts" + std::to_string(N) + "D";

    BSType bin_info() const { return binners_; }

    std::size_t get_1d_indexer(const std::array<
            int,
            N>& indices) const {
        return get_1d_indexer_from_nd(indices,
                                      binners_);
    }

    int get_bin(const std::array<double, N>& values) const {
        std::array<int, N> indices = arrays::make_filled_array<int, N>();
        for (std::size_t i = 0; i < N; i++) {
            indices[i] = binners_[i].assign_bin(values[i]);
            if (indices[i] == -1) { return -1; }
        }
        return get_1d_indexer(indices);
    }

    virtual void process_pair(const SPos& pos1, const SPos& pos2) = 0;

    void process_pair(const Pos& pos1, const Pos& pos2, bool use_true) {
        if (use_true) { this->process_pair(pos1.tpos(), pos2.tpos()); }
        else { this->process_pair(pos1.opos(), pos2.opos()); }
    }

    int operator[](std::size_t idx) const { return counts_[idx]; }

    std::size_t n_tot() const { return n_tot_; }

    std::vector<int> counts() const { return counts_; }

    std::vector<double> normed_counts() const {
        std::vector<double> normed(counts_.begin(), counts_.end());
        std::transform(normed.begin(),
                       normed.end(),
                       normed.begin(),
                       [this](double el) { return el / n_tot_; });
        return normed;
    }

    NNType operator+=(const NNType& other) {
        for (std::size_t i = 0; i < N; i++) {
            if (binners_[i] != other.binners_[i]) {
                std::cerr << "Attempted to combine" << class_name
                          << " instances with different binning in dimension "
                          << std::to_string(i) << std::endl;
                std::cerr << "this." << binners_[i].get_name() << ": "
                          << binners_[i] << std::endl;
                std::cerr << "ohter." << other.binners_[i].get_name() << ": "
                          << other.binners_[i] << std::endl;
                throw std::runtime_error("Cannot combine "
                                         + class_name
                                         + " instances with different binning schemes");
            }
        }
        n_tot_ += other.n_tot_;
        std::transform(counts_.begin(),
                       counts_.end(),
                       other.counts_.begin(),
                       counts_.begin(),
                       std::plus<>());
        return NNType(*this);
    }

    template<typename T, typename std::enable_if_t<
            std::is_arithmetic_v<T>,
            int> = 0>

    NNType operator+=(const T& x) {
        if (!math::isclose(x, (T) 0)) {
            throw std::invalid_argument(
                    "Only 0 valid for scalar addition with " + class_name);
        }
        return NNType(*this);
    }

    template<typename T, typename std::enable_if_t<
            std::is_arithmetic_v<T>,
            int> = 0>

    NNType operator+(const T& x) const {
        return NNType(*this).operator+=(x);
    }

    template<typename T, typename std::enable_if_t<
            std::is_arithmetic_v<T>,
            int> = 0>

    friend NNType operator+(const T& x, const NNType& rhs) {
        return rhs.operator+(x);
    }

    friend bool
    operator==(const NNCountsNDBase& lhs, const NNCountsNDBase& rhs) {
        return lhs.n_tot_ == rhs.n_tot_
               && std::equal(lhs.binners_.begin(), lhs.binners_.end(),
                             rhs.binners_.begin())
               && std::equal(lhs.counts_.begin(),
                             lhs.counts_.end(),
                             rhs.counts_.begin());
    }

    friend bool
    operator!=(const NNCountsNDBase& lhs, const NNCountsNDBase& rhs) {
        return !(lhs
                 == rhs);
    }

    std::vector<std::size_t> shape_vec() const {
        std::vector<std::size_t> sv;
        for (const auto& b : binners_) { sv.push_back(b.nbins); }
        return sv;
    }

    std::size_t size() const { return max_index_; }

    std::size_t nbins_nonzero() const {
        return max_index_ - std::count(counts_.begin(),
                                       counts_.end(),
                                       0);
    }

    long long ncounts() const {
        return std::accumulate(counts_.begin(),
                               counts_.end(),
                               (long long) 0);
    }

    std::string toString() const {
        std::ostringstream oss;
        std::string pad;
        for (std::size_t i = 0; i <= class_name.size(); i++) {
            pad += " ";
        }
        oss << class_name << "(" << std::endl;
        for (const auto& b : binners_) {
            oss << pad << b.name << "=" << b << "," << std::endl;
        }
        oss << pad << "ntot=" << n_tot_ << std::endl;
        oss << ")";
        return oss.str();
    }

    friend std::ostream& operator<<(std::ostream& os, const NNType& nn) {
        os << nn.toString();
        return os;
    }

    void reset() {
        n_tot_ = 0;
        std::vector<int> temp(max_index_, 0);
        counts_.swap(temp);
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

    NNCountsND(const Base& b)
            : Base(b) {}

    explicit NNCountsND(BSType binners)
            : Base(std::move(binners)) {}

    NNCountsND(BSType bins,
               std::vector<int> counts,
               std::size_t n_tot)
            : Base(std::move(bins), std::move(counts), n_tot) {}

    void process_pair(const SPos&, const SPos&) override {
        this->n_tot_++;
    }

private:
    friend class ExpectedNNCountsNDBase<N>;

    friend class ExpectedNNCountsND<N>;
};


template<>
class NNCountsND<3> : public NNCountsNDBase<3> {
    using Base = NNCountsNDBase<3>;
    using BSType = std::array<BinSpecifier, 3>;
protected:
    using Base::process_separation;

    void process_separation(double r_perp, double r_par, double zbar) {
        process_separation(arrays::make_array(r_perp, r_par, zbar));
    }

public:
    NNCountsND() = default;

    NNCountsND(const NNCountsND&) = default;

    NNCountsND(const Base& b)
            : Base(b) {}

    explicit NNCountsND(BSType binners)
            : Base(std::move(binners)) {}

    NNCountsND(BSType bins,
               std::vector<int> counts,
               std::size_t n_tot)
            : Base(std::move(bins), std::move(counts), n_tot) {}

    NNCountsND(const BinSpecifier& rp_bins,
               const BinSpecifier& rl_bins,
               const BinSpecifier& zb_bins)
            : Base(arrays::make_array(rp_bins, rl_bins, zb_bins)) {}

    BinSpecifier rperp_bins() const { return binners_[0]; }

    void rperp_bins(const BinSpecifier& new_binner,
                    bool prefer_old = true) {
        update_binning(0,
                       new_binner,
                       prefer_old);
    }

    BinSpecifier rpar_bins() const { return binners_[1]; }

    void rpar_bins(const BinSpecifier& new_binner,
                   bool prefer_old = true) {
        update_binning(1,
                       new_binner,
                       prefer_old);
    }

    BinSpecifier zbar_bins() const { return binners_[2]; }

    void zbar_bins(const BinSpecifier& new_binner,
                   bool prefer_old = true) {
        update_binning(2,
                       new_binner,
                       prefer_old);
    }

    using Base::get_1d_indexer;

    std::size_t get_1d_indexer(int rp_bin, int rl_bin, int zb_bin) const {
        return get_1d_indexer(arrays::make_array(rp_bin, rl_bin, zb_bin));
    }

    using Base::process_pair;

    void process_pair(const SPos& pos1, const SPos& pos2) override {
        n_tot_++;
        if (!pos1.check_box(pos2, r_max)) { return; }
        if (!check_val_in_limits(pos1.distance_magnitude(pos2),
                                 r_min,
                                 r_max)) {
            return;
        }
        int zo_bin = binners_[2].assign_bin(pos1.distance_zbar(pos2));
        if (zo_bin == -1) { return; }
        int rp_bin = binners_[0].assign_bin(pos1.distance_perp(pos2));
        if (rp_bin == -1) { return; }
        int rl_bin = binners_[1].assign_bin(pos1.distance_par(pos2));
        if (rl_bin == -1) { return; }
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
    using Base::process_separation;

    void process_separation(double r_perp, double r_par) {
        process_separation(arrays::make_array(r_perp, r_par));
    }

public:
    NNCountsND() = default;

    NNCountsND(const NNCountsND&) = default;

    NNCountsND(const Base& b)
            : Base(b) {}

    explicit NNCountsND(BSType binners)
            : Base(std::move(binners)) {}

    NNCountsND(BSType bins,
               std::vector<int> counts,
               std::size_t n_tot)
            : Base(std::move(bins), std::move(counts), n_tot) {}

    NNCountsND(const BinSpecifier& rp_bins, const BinSpecifier& rl_bins)
            : Base(arrays::make_array(rp_bins, rl_bins)) {}

    BinSpecifier rperp_bins() const { return binners_[0]; }

    void rperp_bins(const BinSpecifier& new_binner,
                    bool prefer_old = true) {
        update_binning(0,
                       new_binner,
                       prefer_old);
    }

    BinSpecifier rpar_bins() const { return binners_[1]; }

    void rpar_bins(const BinSpecifier& new_binner,
                   bool prefer_old = true) {
        update_binning(1,
                       new_binner,
                       prefer_old);
    }

    using Base::get_1d_indexer;

    std::size_t get_1d_indexer(int rp_bin, int rl_bin) const {
        return get_1d_indexer(arrays::make_array(rp_bin, rl_bin));
    }

    using Base::process_pair;

    void process_pair(const SPos& pos1, const SPos& pos2) override {
        n_tot_++;
        if (!pos1.check_box(pos2, r_max)) { return; }
        if (!check_val_in_limits(pos1.distance_magnitude(pos2),
                                 r_min,
                                 r_max)) {
            return;
        }
        int rp_bin = binners_[0].assign_bin(pos1.distance_perp(pos2));
        if (rp_bin == -1) { return; }
        int rl_bin = binners_[1].assign_bin(pos1.distance_par(pos2));
        if (rl_bin == -1) { return; }
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
    using Base::process_separation;

    void process_separation(double r) {
        process_separation(arrays::make_array(r));
    }

public:
    NNCountsND() = default;

    NNCountsND(const NNCountsND&) = default;

    NNCountsND(const Base& b)
            : Base(b) {}

    NNCountsND(Base&& b)
            : Base(std::move(b)) {}

    explicit NNCountsND(BSType binners)
            : Base(std::move(binners)) {}

    NNCountsND(BSType bins,
               std::vector<int> counts,
               std::size_t n_tot)
            : Base(std::move(bins), std::move(counts), n_tot) {}

    explicit NNCountsND(const BinSpecifier& r_bins)
            : Base(arrays::make_array(r_bins)) {}

    BinSpecifier r_bins() const { return binners_[0]; }

    void r_bins(const BinSpecifier& new_binner,
                bool prefer_old = true) {
        update_binning(0,
                       new_binner,
                       prefer_old);
    }

    using Base::get_1d_indexer;

    std::size_t get_1d_indexer(int r_bin) const {
        return get_1d_indexer(arrays::make_array(r_bin));
    }

    using Base::process_pair;

    void process_pair(const SPos& pos1, const SPos& pos2) override {
        n_tot_++;
        if (!pos1.check_box(pos2, r_max)) { return; }
        int bin = binners_[0].assign_bin(pos1.distance_magnitude(pos2));
        if (bin > -1) { counts_[bin]++; }
    }

    std::tuple<std::size_t> shape() const {
        return std::make_tuple(max_index_);
    }

private:
    friend class ExpectedNNCountsNDBase<1>;

    friend class ExpectedNNCountsND<1>;
};


template<std::size_t N>
class ExpectedCorrFuncND;


template<std::size_t N>
class ExpectedNNCountsNDBase {
    using NNBaseType = NNCountsNDBase<N>;
    using NNType = NNCountsND<N>;
    using ENNType = ExpectedNNCountsND<N>;
    using BSType = std::array<BinSpecifier, N>;
    using BBSType = std::array<BinSpecifier, 2 * N>;
protected:
    BSType binners_ = arrays::make_filled_array<BinSpecifier, N>();
    BBSType cov_binners_ = arrays::make_filled_array<BinSpecifier, 2 * N>();
    std::vector<NNType> nn_list_ = {};
    std::size_t n_real_ = 0, n_tot_ = 0, max_index_ = 0, max_cov_index_ = 0;
    std::vector<double> mean_ = {}, cov_ = {};

private:
    void on_bin_ntot_update() {
        cov_binners_ = arrays::repeat_array<2>(binners_);
        max_index_ = get_max_index(binners_);
        max_cov_index_ = get_max_index(cov_binners_);
        nn_list_ = {};
        n_real_ = 0;
        mean_ = calculate_mean();
        cov_ = calculate_cov();
    }

    ENNType& downcast() { return static_cast<ENNType>(*this); }

    ENNType& remove_empty_realizations() {
        std::vector<NNType> non_empty;
        for (const auto& nn : nn_list_) {
            if (nn.n_tot_ > 0) { non_empty.push_back(nn); }
        }
        nn_list_.swap(non_empty);
        n_real_ = nn_list_.size();
        return downcast();
    }

    static ENNType
    remove_empty_realizations(const ExpectedNNCountsNDBase& enn) {
        return ENNType(enn).remove_empty_realizations();
    }

    static std::vector<NNType>
    remove_empty_realizations(const std::vector<NNType> nn_list_in) {
        std::vector<NNType> non_empty;
        for (const auto& nn : nn_list_in) {
            if (nn.n_tot_ > 0) { non_empty.push_back(nn); }
        }
        return non_empty;
    }

    std::vector<double> calculate_mean() const {
        if (n_real_ == 0
            || n_tot_ == 0) { return std::vector<double>(max_index_, 0.0); }
        double div_fac = n_real_ * n_tot_;
        std::vector<double> mean(max_index_, 0.0);
        for (const auto& nn : nn_list_) {
            std::transform(mean.begin(),
                           mean.end(),
                           nn.counts_.begin(),
                           mean.begin(),
                           std::plus<>());
        }
        std::transform(mean.begin(),
                       mean.end(),
                       mean.begin(),
                       [=](double x) { return x / div_fac; });
        return mean;
    }

    std::vector<double> calculate_cov() const {
        if (n_real_ < 2 || n_tot_ == 0) {
            return std::vector<double>(max_cov_index_,
                                       0.0);
        }
        double div_fac = (n_real_ * (n_real_ - 1));
        std::vector<double> mean = calculate_mean();
        std::vector<std::vector<double>> diff;
        for (const auto& nn : nn_list_) {
            std::vector<double> temp(nn.counts_.begin(), nn.counts_.end());
            std::transform(temp.begin(),
                           temp.end(),
                           temp.begin(),
                           [this](double el) { return el / (double) n_tot_; });
            std::transform(temp.begin(),
                           temp.end(),
                           mean.begin(),
                           temp.begin(),
                           std::minus<>());
            diff.push_back(temp);
        }
        std::vector<double> cov(max_cov_index_, 0.0);
        for (std::size_t i = 0; i < max_index_; i++) {
            for (std::size_t j = 0; j < max_index_; j++) {
                for (std::size_t k = 0; k < n_real_; k++) {
                    cov[j + max_index_ * i] += diff[k][i]
                                               * diff[k][j]
                                               / div_fac;
                }
            }
        }
        return cov;
    }

public:
    // empty constructor
    ExpectedNNCountsNDBase() = default;

    // copy constructor
    ExpectedNNCountsNDBase(const ExpectedNNCountsNDBase&) = default;

    // pickling support
    ExpectedNNCountsNDBase(const BSType& binners,
                           const std::vector<NNType>& nn_list,
                           std::size_t n_real,
                           std::size_t n_tot)
            : binners_(binners),
              cov_binners_(arrays::repeat_array<2>(binners_)),
              nn_list_(nn_list),
              n_real_(n_real),
              n_tot_(n_tot),
              max_index_(get_max_index(binners_)),
              max_cov_index_(get_max_index(cov_binners_)),
              mean_(calculate_mean()),
              cov_(calculate_cov()) {}

    // no data to start, but binners and n_tot specified
    explicit ExpectedNNCountsNDBase(const BSType& binners, std::size_t n_tot)
            : binners_(binners),
              cov_binners_(arrays::repeat_array<2>(binners_)),
              nn_list_({}),
              n_real_(0),
              n_tot_(n_tot),
              max_index_(get_max_index(binners_)),
              max_cov_index_(get_max_index(cov_binners_)),
              mean_(calculate_mean()),
              cov_(calculate_cov()) {}

    // using state_t = std::tuple<BSType, std::vector<NNType>, std::size_t, std::size_t>;
    // state_t get_state() const {
    //     return std::make_tuple(binners_, nn_list_, n_real_, n_tot_);
    // }

    inline static const std::string
            class_name = "ExpectedNNCounts" + std::to_string(N) + "D";

    BSType bin_info() const { return binners_; }

    void update_binning(std::size_t binner_index,
                        const BinSpecifier& new_binner,
                        bool prefer_old = true) {
        if (binner_index >= N) {
            throw std::out_of_range(
                    "Invalid index " + std::to_string(binner_index) + " for "
                    + std::to_string(N) + "D binning");
        }
        if (prefer_old) { binners_[binner_index].fill(new_binner); }
        else { binners_[binner_index].update(new_binner); }
        on_bin_ntot_update();
    }

    void start_new_realization() {
        calculate_mean();
        calculate_cov();
        n_real_++;
        nn_list_.push_back(NNType(binners_));
    }

    std::size_t get_1d_mean_indexer(const std::array<
            int,
            N>& indices) const {
        return get_1d_indexer_from_nd(indices,
                                      binners_);
    }

    std::size_t
    get_1d_cov_indexer(const std::array<int, 2 * N>& indices) const {
        return get_1d_indexer_from_nd(indices,
                                      arrays::repeat_array<2>(binners_));
    }

    void process_separation(const std::array<double, N>& values,
                            bool new_real = false) {
        if (new_real || n_real_ == 0) { start_new_realization(); }
        nn_list_[n_real_ - 1].process_separation(values);
    }

    NNType operator[](std::size_t idx) const {
        if (idx >= n_real_) {
            throw std::out_of_range(
                    "Invalid index " + std::to_string(idx) + " for "
                    + std::to_string(n_real_) + " realizations");
        }
        return nn_list_[idx];
    }

    std::size_t n_tot() const { return n_tot_; }

    void n_tot(std::size_t new_n_tot) {
        n_tot_ = new_n_tot;
        on_bin_ntot_update();
    }

    std::size_t n_real() const { return n_real_; }

    std::vector<std::size_t> mean_shape_vec() const {
        std::vector<std::size_t> sv;
        for (const auto& b : binners_) { sv.push_back(b.nbins); }
        return sv;
    }

    std::vector<std::size_t> cov_shape_vec() const {
        std::vector<std::size_t> sv;
        for (const auto& b : cov_binners_) { sv.push_back(b.nbins); }
        return sv;
    }

    std::size_t mean_size() const { return max_index_; }

    std::size_t cov_size() const { return max_cov_index_; }

    std::vector<NNType> nn_list() const { return nn_list_; }

    void update() {
        nn_list_ = remove_empty_realizations(nn_list_);
        mean_ = calculate_mean();
        cov_ = calculate_cov();
    }

    std::vector<double> mean() const { return mean_; }

    std::vector<double> cov() const { return cov_; }

    ENNType& operator+=(const NNType& other) {
        for (std::size_t i = 0; i < N; i++) {
            if (binners_[i] != other.binners_[i]) {
                std::cerr << "Attempted to combine " << class_name
                          << " instance and " << NNType::class_name
                          << " instance with different binning in dimension "
                          << std::to_string(i) << std::endl;
                std::cerr << "this." << binners_[i].name << ": " << binners_[i]
                          << std::endl;
                std::cerr << "other." << other.binners_[i].name << ": "
                          << other.binners_[i]
                          << std::endl;
                throw std::runtime_error("Cannot combine "
                                         + class_name
                                         + " instance and "
                                         + NNType::class_name
                                         + " instance with different binning schemes");
            }
        }
        remove_empty_realizations();
        if (n_real_ == 0) {
            nn_list_.push_back(other);
            n_real_++;
            mean_ = calculate_mean();
            cov_ = calculate_cov();
        }
        else {
            nn_list_[n_real_ - 1] += other;
            mean_ = calculate_mean();
            cov_ = calculate_cov();
        }
        return downcast();
    }

    ENNType& operator+=(const ExpectedNNCountsNDBase& other) {
        if (n_tot_ != other.n_tot_) {
            throw std::runtime_error("Cannot combine " + class_name
                                     + " instances with different n_tot");
        }
        std::vector<NNType>
                onn_list = remove_empty_realizations(other.nn_list_);
        if (onn_list.size() == 0) { return downcast(); }
        return operator+=(onn_list[onn_list.size() - 1]);
    }

    template<typename T, typename std::enable_if_t<
            std::is_arithmetic_v<T>,
            int> = 0>
    ENNType& operator+=(const T& x) {
        if (!math::isclose(x, (T) 0)) {
            throw std::invalid_argument("Only 0 valid for scalar addition with "
                                        + class_name);
        }
        return downcast();
    }

    template<typename T, typename std::enable_if_t<
            std::is_arithmetic_v<T>,
            int> = 0>
    ENNType operator+(const T& x) const { return ENNType(*this).operator+=(x); }

    template<typename T, typename std::enable_if_t<
            std::is_arithmetic_v<T>,
            int> = 0>
    friend ENNType operator+(const T& x,
                             const ExpectedNNCountsNDBase& rhs) {
        return rhs.operator+(x);
    }

    bool operator==(const ExpectedNNCountsNDBase& other) const {
        return n_tot_ == other.n_tot_
               && n_real_ == other.n_real_
               && std::equal(binners_.begin(),
                             binners_.end(),
                             other.binners_.begin())
               && math::isclose(mean_, other.mean_)
               && math::isclose(cov_, other.cov_);
    }

    bool operator!=(const ExpectedNNCountsNDBase& other) const {
        return !operator==(other);
    }

    std::vector<std::vector<double>> normed_counts() const {
        std::vector<std::vector<double>> norm;
        for (const auto& nn : nn_list_) {
            std::vector<double> normi(nn.counts_.begin(), nn.counts_.end());
            std::transform(normi.begin(),
                           normi.end(),
                           normi.begin(),
                           [this](double x) { return x / n_tot_; });
            norm.push_back(normi);
        }
        return norm;
    }

    void append_real(const NNType& other) {
        nn_list_ = remove_empty_realizations(nn_list_);
        n_real_ = nn_list_.size();
        nn_list_.push_back(other);
        n_real_++;
        mean_ = calculate_mean();
        cov_ = calculate_cov();
    }

    void append_real(const ExpectedNNCountsNDBase& other) {
        remove_empty_realizations();
        std::vector<NNType>
                onn_list = remove_empty_realizations(other.nn_list_);
        nn_list_.insert(nn_list_.end(),
                        onn_list.begin(),
                        onn_list.end());
        n_real_ += onn_list.size();
        mean_ = calculate_mean();
        cov_ = calculate_cov();
    }

    void reset() {
        n_real_ = 0;
        std::vector<NNType> temp(n_real_);
        nn_list_.swap(temp);
        mean_ = calculate_mean();
        cov_ = calculate_cov();
    }

    std::string toString() const {
        std::ostringstream oss;
        std::string pad;
        for (std::size_t i = 0; i <= class_name.size(); i++) {
            pad += " ";
        }
        oss << class_name << "(" << std::endl;
        for (const auto& b : binners_) {
            oss << pad << b.name << "=" << b << "," << std::endl;
        }
        oss << pad << " ntot=" << n_tot_ << "," << std::endl;
        oss << pad << " nreal=" << n_real_ << std::endl;
        oss << ")";
        return oss.str();
    }

    friend std::ostream& operator<<(std::ostream& os, const ENNType& nn) {
        os << nn.toString();
        return os;
    }

private:
    friend class ExpectedCorrFuncNDBase<N>;
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

public:
    using Base::Base;

    ExpectedNNCountsND(const Base& b)
            : Base(b) {}

    ExpectedNNCountsND(const BinSpecifier& rperp_bins,
                       const BinSpecifier& rpar_bins,
                       const BinSpecifier& zbar_bins,
                       std::size_t n_tot)
            : Base(arrays::make_array(rperp_bins, rpar_bins, zbar_bins),
                   n_tot) {}

    BinSpecifier rperp_bins() const { return binners_[0]; }

    void rperp_bins(const BinSpecifier& new_binner,
                    bool prefer_old = true) {
        update_binning(0,
                       new_binner,
                       prefer_old);
    }

    BinSpecifier rpar_bins() const { return binners_[1]; }

    void rpar_bins(const BinSpecifier& new_binner,
                   bool prefer_old = true) {
        update_binning(1,
                       new_binner,
                       prefer_old);
    }

    BinSpecifier zbar_bins() const { return binners_[2]; }

    void zbar_bins(const BinSpecifier& new_binner,
                   bool prefer_old = true) {
        update_binning(2,
                       new_binner,
                       prefer_old);
    }

    using Base::get_1d_mean_indexer;

    std::size_t get_1d_mean_indexer_from_args(int rp_bin,
                                              int rl_bin,
                                              int zb_bin) const {
        return get_1d_mean_indexer(arrays::make_array(rp_bin, rl_bin, zb_bin));
    }

    using Base::get_1d_cov_indexer;

    std::size_t get_1d_cov_indexer_from_args(int rpi_bin,
                                             int rli_bin,
                                             int zbi_bin,
                                             int rpj_bin,
                                             int rlj_bin,
                                             int zbj_bin) const {
        return get_1d_cov_indexer(arrays::make_array(rpi_bin,
                                                     rli_bin,
                                                     zbi_bin,
                                                     rpj_bin,
                                                     rlj_bin,
                                                     zbj_bin));
    }

    using Base::process_separation;

    void process_separation(double r_perp,
                            double r_par,
                            double zbar,
                            bool new_real = false) {
        process_separation((std::array<
                double,
                3>) {{r_perp, r_par, zbar}}, new_real);
    }

    std::tuple<int, int, int> mean_shape() const {
        return std::make_tuple(binners_[0].nbins,
                               binners_[1].nbins,
                               binners_[2].nbins);
    }

    std::tuple<int, int, int, int, int, int> cov_shape() const {
        return std::make_tuple(cov_binners_[0].nbins,
                               cov_binners_[1].nbins,
                               cov_binners_[2].nbins,
                               cov_binners_[3].nbins,
                               cov_binners_[4].nbins,
                               cov_binners_[5].nbins);
    }
};


template<>
class ExpectedNNCountsND<2> : public ExpectedNNCountsNDBase<2> {
    using Base = ExpectedNNCountsNDBase<2>;
    using BSType = std::array<BinSpecifier, 2>;

public:
    using Base::Base;

    ExpectedNNCountsND(const Base& b)
            : Base(b) {}

    ExpectedNNCountsND(const BinSpecifier& rperp_bins,
                       const BinSpecifier& rpar_bins,
                       std::size_t n_tot)
            : Base(arrays::make_array(rperp_bins, rpar_bins), n_tot) {}

    BinSpecifier rperp_bins() const { return binners_[0]; }

    void rperp_bins(const BinSpecifier& new_binner,
                    bool prefer_old = true) {
        update_binning(0,
                       new_binner,
                       prefer_old);
    }

    BinSpecifier rpar_bins() const { return binners_[1]; }

    void rpar_bins(const BinSpecifier& new_binner,
                   bool prefer_old = true) {
        update_binning(1,
                       new_binner,
                       prefer_old);
    }

    using Base::get_1d_mean_indexer;

    std::size_t get_1d_mean_indexer_from_args(int rp_bin,
                                              int rl_bin) const {
        return get_1d_mean_indexer(arrays::make_array(rp_bin, rl_bin));
    }

    using Base::get_1d_cov_indexer;

    std::size_t get_1d_cov_indexer_from_args(int rpi_bin,
                                             int rli_bin,
                                             int rpj_bin,
                                             int rlj_bin) const {
        return get_1d_cov_indexer(arrays::make_array(rpi_bin,
                                                     rli_bin,
                                                     rpj_bin,
                                                     rlj_bin));
    }

    using Base::process_separation;

    void process_separation(double r_perp,
                            double r_par,
                            bool new_real = false) {
        process_separation((std::array<
                double,
                2>) {{r_perp, r_par}}, new_real);
    }

    std::tuple<int, int> mean_shape() const {
        return std::make_tuple(binners_[0].nbins, binners_[1].nbins);
    }

    std::tuple<int, int, int, int> cov_shape() const {
        return std::make_tuple(cov_binners_[0].nbins,
                               cov_binners_[1].nbins,
                               cov_binners_[2].nbins,
                               cov_binners_[3].nbins);
    }
};


template<>
class ExpectedNNCountsND<1> : public ExpectedNNCountsNDBase<1> {
    using Base = ExpectedNNCountsNDBase<1>;
    using BSType = std::array<BinSpecifier, 1>;

public:
    using Base::Base;

    ExpectedNNCountsND(const Base& b)
            : Base(b) {}

    ExpectedNNCountsND(const BinSpecifier& r_bins, std::size_t n_tot)
            : Base(arrays::make_array(r_bins), n_tot) {}

    BinSpecifier r_bins() const { return binners_[0]; }

    void r_bins(const BinSpecifier& new_binner,
                bool prefer_old = true) {
        update_binning(0,
                       new_binner,
                       prefer_old);
    }

    using Base::get_1d_mean_indexer;

    std::size_t get_1d_mean_indexer_from_args(int r_bin) const {
        return get_1d_mean_indexer(arrays::make_array(r_bin));
    }

    using Base::get_1d_cov_indexer;

    std::size_t get_1d_cov_indexer_from_args(int ri_bin,
                                             int rj_bin) const {
        return get_1d_cov_indexer(arrays::make_array(ri_bin, ri_bin));
    }

    using Base::process_separation;

    void process_separation(double r,
                            bool new_real = false) {
        process_separation((std::array<
                double,
                1>) {{r}}, new_real);
    }

    std::tuple<int> mean_shape() const {
        return std::make_tuple(max_index_);
    }

    std::tuple<int, int> cov_shape() const {
        return std::make_tuple(max_index_, max_index_);
    }
};


template<std::size_t N>
class CorrFuncNDBase {
    using NNType = NNCountsND<N>;
    using BSType = std::array<BinSpecifier, N>;

    void on_bin_nn_update() {
        max_index_ = get_max_index(binners_);
        for (std::size_t i = 0; i < N; i++) {
            dd_.update_binning(i, binners_[i], false);
            dr_.update_binning(i, binners_[i], false);
            rd_.update_binning(i, binners_[i], false);
            rr_.update_binning(i, binners_[i], false);
        }
    }

protected:
    BSType binners_ = arrays::make_filled_array<BinSpecifier, N>();
    std::size_t max_index_ = 0;
    NNType dd_, rr_, dr_, rd_;

    const NNType& verify_nn(const NNType& nn) const {
        for (std::size_t i = 0; i < N; i++) {
            if (binners_[i].is_set() && nn.bin_info()[i] != binners_[i]) {
                throw std::invalid_argument(NNType::class_name
                                            + " instance given has different binning scheme in dimension "
                                            + std::to_string(i));
            }
        }
        return nn;
    }

public:
    CorrFuncNDBase() = default;

    CorrFuncNDBase(const CorrFuncNDBase&) = default;

    explicit CorrFuncNDBase(const BSType& binners)
            : binners_(binners),
              max_index_(get_max_index(binners_)),
              dd_(binners_),
              rr_(binners_),
              dr_(binners_),
              rd_(binners_) {}

    explicit CorrFuncNDBase(const NNType& dd)
            : binners_(dd.bin_info()),
              max_index_(get_max_index(binners_)),
              dd_(dd),
              rr_(binners_),
              dr_(binners_),
              rd_(binners_) {}

    CorrFuncNDBase(const BSType& binners, const NNType& dd)
            : binners_(binners),
              max_index_(get_max_index(binners_)),
              dd_(verify_nn(dd)),
              rr_(binners_),
              dr_(binners_),
              rd_(binners_) {}

    CorrFuncNDBase(const NNType& dd, const NNType& rr)
            : binners_(dd.bin_info()),
              max_index_(get_max_index(binners_)),
              dd_(dd),
              rr_(verify_nn(rr)),
              dr_(binners_),
              rd_(binners_) {}

    CorrFuncNDBase(const BSType& binners, const NNType& dd, const NNType& rr)
            : binners_(binners),
              max_index_(get_max_index(binners_)),
              dd_(verify_nn(dd)),
              rr_(verify_nn(rr)),
              dr_(binners_),
              rd_(binners_) {}

    CorrFuncNDBase(const NNType& dd, const NNType& rr, const NNType& dr)
            : binners_(dd.bin_info()),
              max_index_(get_max_index(binners_)),
              dd_(dd),
              rr_(verify_nn(rr)),
              dr_(verify_nn(dr)),
              rd_(binners_) {}

    CorrFuncNDBase(const BSType& binners,
                   const NNType& dd,
                   const NNType& rr,
                   const NNType& dr)
            : binners_(binners),
              max_index_(get_max_index(binners_)),
              dd_(verify_nn(dd)),
              rr_(verify_nn(rr)),
              dr_(verify_nn(dr)),
              rd_(binners_) {}

    CorrFuncNDBase(const NNType& dd,
                   const NNType& rr,
                   const NNType& dr,
                   const NNType& rd)
            : binners_(dd.bin_info()),
              max_index_(get_max_index(binners_)),
              dd_(dd),
              rr_(verify_nn(rr)),
              dr_(verify_nn(dr)),
              rd_(verify_nn(rd)) {}

    CorrFuncNDBase(const BSType& binners,
                   const NNType& dd,
                   const NNType& rr,
                   const NNType& dr,
                   const NNType& rd)
            : binners_(binners),
              max_index_(get_max_index(binners_)),
              dd_(verify_nn(dd)),
              rr_(verify_nn(rr)),
              dr_(verify_nn(dr)),
              rd_(verify_nn(rd)) {}

    inline static const std::string
            class_name = "CorrFunc" + std::to_string(N) + "D";

    const NNType& dd() const {
        return dd_;
    }

    void dd(const NNType& dd) {
        dd_ = verify_nn(dd);
        on_bin_nn_update();
    }

    const NNType& rr() const {
        return rr_;
    }

    void rr(const NNType& rr) {
        rr_ = verify_nn(rr);
        on_bin_nn_update();
    }

    const NNType& dr() const {
        return dr_;
    }

    void dr(const NNType& dr) {
        dr_ = verify_nn(dr);
        on_bin_nn_update();
    }

    const NNType& rd() const {
        return rd_;
    }

    void rd(const NNType& rd) {
        rd_ = verify_nn(rd);
        on_bin_nn_update();
    }

    std::size_t size() const {
        return max_index_;
    }

    std::vector<std::size_t> shape_vec() const {
        std::vector<std::size_t> shape;
        for (const auto& b : binners_) { shape.push_back(b.get_nbins()); }
        return shape;
    }

    const BSType& bin_info() const {
        return binners_;
    }

    void update_binning(const BinSpecifier& new_binning,
                        std::size_t dim,
                        bool prefer_old = true) {
        if (dim >= N) {
            throw std::invalid_argument("Index "
                                        + std::to_string(dim)
                                        + " out of bounds for binning in "
                                        + std::to_string(N)
                                        + " dimensions");
        }
        if (prefer_old) { binners_[dim].fill(new_binning); }
        else { binners_[dim].update(new_binning); }
        on_bin_nn_update();
    }

    std::vector<double> calculate_xi() const {
        if (dd_.n_tot() == 0 || rr_.n_tot() == 0) {
            throw std::runtime_error(
                    "Cannot calculate correlation function without at least DD and RR run");
        }
        std::vector<double> xi = dd_.normed_counts(), nrr = rr_.normed_counts();
        if (dr_.n_tot() != 0 || rd_.n_tot() != 0) {
            std::transform(xi.begin(),
                           xi.end(),
                           nrr.begin(),
                           xi.begin(),
                           std::plus<>());
            std::vector<double>
                    ndr = dr_.n_tot() != 0
                          ? dr_.normed_counts()
                          : rd_.normed_counts();
            std::vector<double>
                    nrd = rd_.n_tot() != 0
                          ? rd_.normed_counts()
                          : dr_.normed_counts();
            std::transform(ndr.begin(),
                           ndr.end(),
                           nrd.begin(),
                           ndr.begin(),
                           std::plus<>());
            std::transform(xi.begin(),
                           xi.end(),
                           ndr.begin(),
                           xi.begin(),
                           std::minus<>());
        }
        else {
            std::transform(xi.begin(),
                           xi.end(),
                           nrr.begin(),
                           xi.begin(),
                           std::minus<>());
        }
        std::transform(xi.begin(),
                       xi.end(),
                       nrr.begin(),
                       xi.begin(),
                       std::divides<>());
        return xi;
    }

    std::string toString() const {
        std::ostringstream oss;
        std::string pad;
        for (std::size_t i = 0; i <= class_name.size(); i++) {
            pad += " ";
        }
        oss << class_name << "(" << std::endl;
        if (dd_.n_tot() != 0
            || rr_.n_tot() != 0
            || dr_.n_tot() != 0
            || rd_.n_tot() != 0) {
            if (dd_.n_tot() != 0) {
                oss << pad << "dd=" << dd_ << std::endl;
            }
            if (rr_.n_tot() != 0) {
                oss << pad << "rr=" << rr_ << std::endl;
            }
            if (dr_.n_tot() != 0) {
                oss << pad << "dr=" << dr_ << std::endl;
            }
            if (rd_.n_tot() != 0) {
                oss << pad << "rd=" << rd_ << std::endl;
            }
        }
        else {
            for (const auto& b : binners_) {
                oss << pad << b.get_name() << "=" << b << std::endl;
            }
        }
        oss << ")";
        return oss.str();
    }

    friend std::ostream&
    operator<<(std::ostream& os, const CorrFuncNDBase& cf) {
        os << cf.toString();
        return os;
    }
};


template<std::size_t N>
class CorrFuncND : public CorrFuncNDBase<N> {
    using Base = CorrFuncNDBase<N>;
public:
    using Base::Base;

    CorrFuncND(const Base& b)
            : Base(b) {}
};


template<>
class CorrFuncND<2> : public CorrFuncNDBase<2> {
    using Base = CorrFuncNDBase<2>;
    using NNType = NNCountsND<2>;
public:
    using Base::Base;

    CorrFuncND(const Base& b)
            : Base(b) {}

    CorrFuncND(const BinSpecifier& perp_binner, const BinSpecifier& par_binner)
            : Base(arrays::make_array(perp_binner, par_binner)) {}

    CorrFuncND(const BinSpecifier& perp_binner,
               const BinSpecifier& par_binner,
               const NNType& dd)
            : Base(arrays::make_array(perp_binner, par_binner), dd) {}

    CorrFuncND(const BinSpecifier& perp_binner,
               const BinSpecifier& par_binner,
               const NNType& dd,
               const NNType& rr)
            : Base(arrays::make_array(perp_binner, par_binner), dd, rr) {}

    CorrFuncND(const BinSpecifier& perp_binner,
               const BinSpecifier& par_binner,
               const NNType& dd,
               const NNType& rr,
               const NNType& dr)
            : Base(arrays::make_array(perp_binner, par_binner), dd, rr, dr) {}

    CorrFuncND(const BinSpecifier& perp_binner,
               const BinSpecifier& par_binner,
               const NNType& dd,
               const NNType& rr,
               const NNType& dr,
               const NNType& rd)
            : Base(arrays::make_array(perp_binner, par_binner),
                   dd,
                   rr,
                   dr,
                   rd) {}

    const BinSpecifier& rperp_bins() const {
        return binners_[0];
    }

    void rperp_bins(const BinSpecifier& new_binning, bool prefer_old = true) {
        update_binning(new_binning, 0, prefer_old);
    }

    const BinSpecifier& rpar_bins() const {
        return binners_[1];
    }

    void rpar_bins(const BinSpecifier& new_binning, bool prefer_old = true) {
        update_binning(new_binning, 1, prefer_old);
    }

    std::tuple<std::size_t, std::size_t> shape() const {
        return std::make_tuple(binners_[0].get_nbins(),
                               binners_[1].get_nbins());
    }
};


template<>
class CorrFuncND<1> : public CorrFuncNDBase<1> {
    using Base = CorrFuncNDBase<1>;
    using NNType = NNCountsND<1>;
public:
    using Base::Base;

    CorrFuncND(const Base& b)
            : Base(b) {}

    explicit CorrFuncND(const BinSpecifier& binner)
            : Base(arrays::make_array(binner)) {}

    CorrFuncND(const BinSpecifier& binner,
               const NNType& dd)
            : Base(arrays::make_array(binner), dd) {}

    CorrFuncND(const BinSpecifier& binner,
               const NNType& dd,
               const NNType& rr)
            : Base(arrays::make_array(binner), dd, rr) {}

    CorrFuncND(const BinSpecifier& binner,
               const NNType& dd,
               const NNType& rr,
               const NNType& dr)
            : Base(arrays::make_array(binner), dd, rr, dr) {}

    CorrFuncND(const BinSpecifier& binner,
               const NNType& dd,
               const NNType& rr,
               const NNType& dr,
               const NNType& rd)
            : Base(arrays::make_array(binner),
                   dd,
                   rr,
                   dr,
                   rd) {}

    const BinSpecifier& r_bins() const {
        return binners_[0];
    }

    void r_bins(const BinSpecifier& new_binning, bool prefer_old = true) {
        update_binning(new_binning, 0, prefer_old);
    }

    std::tuple<std::size_t> shape() const {
        return std::make_tuple(max_index_);
    }
};


template<std::size_t N>
class ExpectedCorrFuncND {
    using ENNType = ExpectedNNCountsND<N>;
    using BSType = std::array<BinSpecifier, N>;
    using BBSType = std::array<BinSpecifier, 2 * N>;

    std::vector<std::vector<double>> calculate_xi_i() const {
        if (dd_.n_real_ == 0 || rr_.n_real_ == 0) {
            throw std::runtime_error(
                    "Cannot calculate correlation function without at least DD and RR");
        }
        if (n_real_ == 0) {
            return std::vector<std::vector<double>>(n_real_,
                                                    std::vector<
                                                            double>(max_index_,
                                                                    0.0));
        }
        std::vector<std::vector<double>>
                xi(n_real_, std::vector<double>(max_index_, 1));
        std::vector<std::vector<double>>
                dd_norm = dd_.normed_counts(),
                rr_norm = rr_.normed_counts();
        if (dr_.n_real_ == 0 && rd_.n_real_ == 0) {
            for (std::size_t i = 0; i < n_real_; i++) {
                std::vector<double> xi_i(max_index_);
                std::transform(dd_norm.at(i).begin(),
                               dd_norm.at(i).end(),
                               rr_norm.at(i).begin(),
                               xi_i.begin(),
                               std::divides<>());
                std::transform(xi_i.begin(),
                               xi_i.end(),
                               xi.at(i).begin(),
                               xi.at(i).begin(),
                               std::minus<>());
            }
        }
        else {
            std::vector<std::vector<double>>
                    dr_norm = dr_.n_real_ != 0
                              ? dr_.normed_counts()
                              : rd_.normed_counts();
            std::vector<std::vector<double>>
                    rd_norm = rd_.n_real_ != 0
                              ? rd_.normed_counts()
                              : dr_.normed_counts();
            for (std::size_t i = 0; i < n_real_; i++) {
                std::vector<double> xi_i = dd_norm.at(i);
                std::transform(xi_i.begin(),
                               xi_i.end(),
                               dr_norm.at(i).begin(),
                               xi_i.begin(),
                               std::minus<>());
                std::transform(xi_i.begin(),
                               xi_i.end(),
                               rd_norm.at(i).begin(),
                               xi_i.begin(),
                               std::minus<>());
                std::transform(xi_i.begin(),
                               xi_i.end(),
                               rr_norm.at(i).begin(),
                               xi_i.begin(),
                               std::divides<>());
                std::transform(xi_i.begin(),
                               xi_i.end(),
                               xi.at(i).begin(),
                               xi.at(i).begin(),
                               std::plus<>());
            }
        }
        return xi;
    }

protected:
    BSType binners_ = arrays::make_filled_array<BinSpecifier, N>();
    BBSType cov_binners_ = arrays::make_filled_array<BinSpecifier, 2 * N>();
    std::size_t max_index_ = 0, max_cov_index_ = 0;
    ENNType dd_;
    std::size_t n_real_ = 0;
    ENNType rr_, dr_, rd_;

    ENNType verify_nn(const ENNType& nn) const {
        ENNType out = ENNType::remove_empty_realizations(nn);
        if (n_real_ > 0) {
            if (out.n_real_ != n_real_) {
                throw std::invalid_argument(out.class_name
                                            + " instance given has different number of realizations");
            }
        }
        else { n_real_ = out.n_real_; }
        for (std::size_t i = 0; i < N; i++) {
            if (binners_[i].is_set()) {
                if (out.binners_[i] != binners_[i]) {
                    throw std::invalid_argument(out.class_name
                                                + " instance given has different binning scheme in dimension "
                                                + std::to_string(i));
                }
            }
            else { binners_[i] = out.binners_[i]; }
        }

        return out;
    }

public:
    ExpectedCorrFuncND() = default;

    ExpectedCorrFuncND(const ExpectedCorrFuncND&) = default;

    explicit ExpectedCorrFuncND(const BSType& binners)
            : binners_(binners),
              cov_binners_(arrays::repeat_array<2>(binners_)),
              max_index_(get_max_index(binners_)),
              max_cov_index_(get_max_index(cov_binners_)) {}

    explicit ExpectedCorrFuncND(const ENNType& dd)
            : binners_(dd.binners_),
              cov_binners_(arrays::repeat_array<2>(binners_)),
              max_index_(get_max_index(binners_)),
              max_cov_index_(get_max_index(cov_binners_)),
              dd_(ENNType::remove_empty_realizations(dd)),
              n_real_(dd_.n_real_) {}

    ExpectedCorrFuncND(const ENNType& dd, const ENNType& rr)
            : binners_(dd.binners_),
              cov_binners_(arrays::repeat_array<2>(binners_)),
              max_index_(get_max_index(binners_)),
              max_cov_index_(get_max_index(cov_binners_)),
              dd_(ENNType::remove_empty_realizations(dd)),
              n_real_(dd_.n_real_),
              rr_(verify_nn(rr)) {}

    ExpectedCorrFuncND(const ENNType& dd,
                           const ENNType& rr,
                           const ENNType& dr)
            : binners_(dd.binners_),
              cov_binners_(arrays::repeat_array<2>(binners_)),
              max_index_(get_max_index(binners_)),
              max_cov_index_(get_max_index(cov_binners_)),
              dd_(ENNType::remove_empty_realizations(dd)),
              n_real_(dd_.n_real_),
              rr_(verify_nn(rr)),
              dr_(verify_nn(dr)) {}

    ExpectedCorrFuncND(const ENNType& dd,
                           const ENNType& rr,
                           const ENNType& dr,
                           const ENNType& rd)
            : binners_(dd.binners_),
              cov_binners_(arrays::repeat_array<2>(binners_)),
              max_index_(get_max_index(binners_)),
              max_cov_index_(get_max_index(cov_binners_)),
              dd_(ENNType::remove_empty_realizations(dd)),
              n_real_(dd_.n_real_),
              rr_(verify_nn(rr)),
              dr_(verify_nn(dr)),
              rd_(verify_nn(rd)) {}

    inline static const std::string
            class_name = "ExpectedCorrFunc" + std::to_string(N) + "D";

    std::size_t n_real() const {
        return n_real_;
    }

    std::size_t mean_size() const {
        return max_index_;
    }

    std::vector<std::size_t> mean_shape() const {
        std::vector<std::size_t> shape;
        for (const auto& b : binners_) { shape.push_back(b.get_nbins()); }
        return shape;
    }

    std::size_t cov_size() const {
        return max_cov_index_;
    }

    std::vector<std::size_t> cov_shape() const {
        std::vector<std::size_t> shape;
        for (const auto& b : cov_binners_) { shape.push_back(b.get_nbins()); }
        return shape;
    }

    BSType bin_info() const { return binners_; }

    void update_binning(std::size_t index,
                        const BinSpecifier& new_binner,
                        bool prefer_old = true) {
        dd_.update_binning(index, new_binner, prefer_old);
        dr_.update_binning(index, new_binner, prefer_old);
        rd_.update_binning(index, new_binner, prefer_old);
        rr_.update_binning(index, new_binner, prefer_old);
        binners_ = dd_.binners_;
        cov_binners_ = arrays::repeat_array<2>(binners_);
        max_index_ = get_max_index(binners_);
        max_cov_index_ = get_max_index(cov_binners_);
        n_real_ = 0;
    }

    const ENNType& dd() const {
        return dd_;
    }

    void dd(const ENNType& dd) {
        dd_ = verify_nn(dd);
    }

    const ENNType& rr() const {
        return rr_;
    }

    void rr(const ENNType& rr) {
        rr_ = verify_nn(rr);
    }

    const ENNType& dr() const {
        return dr_;
    }

    void dr(const ENNType& dr) {
        dr_ = verify_nn(dr);
    }

    const ENNType& rd() const {
        return rd_;
    }

    void rd(const ENNType& rd) {
        rd_ = verify_nn(rd);
    }

    std::vector<double> calculate_xi() const {
        if (dd_.n_real_ == 0 || rr_.n_real_ == 0) {
            throw std::runtime_error(
                    "Cannot calculate correlation function without at least DD and RR");
        }
        if (n_real_ == 0) { return std::vector<double>(max_index_, 0.0); }
        std::vector<std::vector<double>> xi_i = calculate_xi_i();
        std::vector<double> xi(max_index_, 0.0);
        for (std::size_t i = 0; i < n_real_; i++) {
            std::transform(xi.begin(),
                           xi.end(),
                           xi_i.at(i).begin(),
                           xi.begin(),
                           std::plus<>());
        }
        std::transform(xi.begin(),
                       xi.end(),
                       xi.begin(),
                       [this](double el) { return el / n_real_; });
        return xi;
    }

    std::vector<double> calculate_xi_cov() const {
        if (dd_.n_real_ == 0 || rr_.n_real_ == 0) {
            throw std::runtime_error(
                    "Cannot calculate correlation function without at least DD and RR");
        }
        if (n_real_ < 2) { return std::vector<double>(max_cov_index_, 0.0); }
        std::vector<std::vector<double>>
                xi_i = transpose_vector(calculate_xi_i());
        std::vector<double> xi_mean = calculate_xi();
        std::vector<double> xi_cov(max_cov_index_, 0.0);
        for (std::size_t i = 0; i < max_index_; i++) {
            for (std::size_t j = 0; j < max_index_; j++) {
                std::vector<double> tempi = xi_i.at(i), tempj = xi_i.at(j);
                std::transform(tempi.begin(),
                               tempi.end(),
                               tempi.begin(),
                               [&](double el) { return el / xi_mean.at(i); });
                std::transform(tempj.begin(),
                               tempj.end(),
                               tempj.begin(),
                               [&](double el) { return el / xi_mean.at(j); });
                std::transform(tempi.begin(),
                               tempi.end(),
                               tempj.begin(),
                               tempi.begin(),
                               std::multiplies<>());
                xi_cov[i + max_index_ * j] = std::accumulate(tempi.begin(),
                                                             tempi.end(),
                                                             0.0);
            }
        }
        return xi_cov;
    }

    std::string toString() const {
        std::ostringstream oss;
        std::string pad;
        for (std::size_t i = 0; i <= class_name.size(); i++) {
            pad += " ";
        }
        oss << class_name << "(" << std::endl;
        if (dd_.n_real_ != 0
            || rr_.n_real_ != 0
            || dr_.n_real_ != 0
            || rd_.n_real_ != 0) {
            if (dd_.n_real_ != 0) {
                oss << pad << "dd=" << dd_ << std::endl;
            }
            if (rr_.n_real_ != 0) {
                oss << pad << "rr=" << rr_ << std::endl;
            }
            if (dr_.n_real_ != 0) {
                oss << pad << "dr=" << dr_ << std::endl;
            }
            if (rd_.n_real_ != 0) {
                oss << pad << "rd=" << rd_ << std::endl;
            }
        }
        else {
            for (const auto& b : binners_) {
                oss << pad << b.get_name() << "=" << b << std::endl;
            }
        }
        oss << ")";
        return oss.str();
    }

    friend std::ostream&
    operator<<(std::ostream& os, const ExpectedCorrFuncND& cf) {
        os << cf.toString();
        return os;
    }
};

// template<class C>
// class CorrFuncBase {
// protected:
//     C dd_, dr_, rd_, rr_;
//     std::vector<double> values_;
//     std::size_t max_index_;
//
//     virtual void verify_nn(const C& nn) { return; }
//
// public:
//     CorrFuncBase() {}
//
//     CorrFuncBase(const CorrFuncBase& other)
//             : dd_(other.dd_),
//               dr_(other.dr_),
//               rd_(other.rd_),
//               rr_(other.rr_),
//               values_(other.values_),
//               max_index_(other.max_index_) {}
//
//     CorrFuncBase(const CorrFunc<C>& other)
//             : dd_(other.dd_),
//               dr_(other.dr_),
//               rd_(other.rd_),
//               rr_(other.rr_),
//               values_(other.values_),
//               max_index_(other.max_index_) {}
//
//     CorrFuncBase(const C& dd)
//             : dd_(dd),
//               values_(std::vector<double>(dd.max_index_, 0.0)),
//               max_index_(dd.max_index_) {}
//
//     void assign_dd(const C& new_dd) {
//         dd_ = new_dd;
//         if (rr_ != C()) {
//             try { verify_nn(rr_); }
//             catch (const std::runtime_error& e) { rr_ = C(); }
//         }
//         if (dr_ != C()) {
//             try { verify_nn(dr_); }
//             catch (const std::runtime_error& e) { dr_ = C(); }
//         }
//         if (rd_ != C()) {
//             try { verify_nn(rd_); }
//             catch (const std::runtime_error& e) { rd_ = C(); }
//         }
//         max_index_ = new_dd.max_index_;
//         values_ = std::vector<double>(max_index_, 0.0);
//     }
//
//     void assign_rr(const C& new_rr) {
//         verify_nn(new_rr);
//         rr_ = new_rr;
//     }
//
//     void assign_dr(const C& new_dr) {
//         verify_nn(new_dr);
//         dr_ = new_dr;
//     }
//
//     void assign_rd(const C& new_rd) {
//         verify_nn(new_rd);
//         rd_ = new_rd;
//     }
//
//     void calculate_xi() {
//         if (dd_ == C()) {
//             throw std::runtime_error(
//                     "Cannot calculate correlation function without data-data pair counts");
//         }
//         if (rr_ == C()) {
//             throw std::runtime_error(
//                     "Cannot calculate correlation function without random-random pair counts");
//         }
//         std::vector<double> rr_norm(rr_.normed_counts());
//         values_ = dd_.normed_counts();
//         if (dr_ == C() && rd_ == C()) {
//             // (DD / RR) - 1
//             std::transform(values_.begin(),
//                            values_.end(),
//                            rr_norm.begin(),
//                            values_.begin(),
//                            std::divides<double>());
//             std::transform(values_.begin(),
//                            values_.end(),
//                            values_.begin(),
//                            std::bind2nd(std::minus<double>(), 1.0));
//         }
//         else {
//             // All options have DD + RR
//             std::transform(values_.begin(),
//                            values_.end(),
//                            rr_norm.begin(),
//                            values_.begin(),
//                            std::plus<double>());
//             std::vector<double> dr_rd_norm(max_index_);
//             if (rd_ == C()) {
//                 // (DD - 2 DR + RR) / RR
//                 dr_rd_norm = dr_.normed_counts();
//                 std::transform(dr_rd_norm.begin(),
//                                dr_rd_norm.end(),
//                                dr_rd_norm.begin(),
//                                std::bind1st(std::multiplies<double>(), 2.0));
//             }
//             else if (dr_ == C()) {
//                 // (DD - 2 RD + RR) / RR
//                 dr_rd_norm = rd_.normed_counts();
//                 std::transform(dr_rd_norm.begin(),
//                                dr_rd_norm.end(),
//                                dr_rd_norm.begin(),
//                                std::bind1st(std::multiplies<double>(), 2.0));
//             }
//             else {
//                 // (DD - DR - RD + RR) / RR
//                 std::vector<double> temp = rd_.normed_counts();
//                 dr_rd_norm = dr_.normed_counts();
//                 std::transform(dr_rd_norm.begin(),
//                                dr_rd_norm.end(),
//                                temp.begin(),
//                                dr_rd_norm.begin(),
//                                std::plus<double>());
//             }
//             std::transform(values_.begin(),
//                            values_.end(),
//                            dr_rd_norm.begin(),
//                            values_.begin(),
//                            std::minus<double>());
//             std::transform(values_.begin(),
//                            values_.end(),
//                            rr_norm.begin(),
//                            values_.begin(),
//                            std::divides<double>());
//         }
//     }
//
//     std::vector<double> xi() const { return values_; }
//
//     C dd() const { return dd_; }
//
//     C dr() const { return dr_; }
//
//     C rd() const { return rd_; }
//
//     C rr() const { return rr_; }
//
//     auto shape() const { return dd_.shape(); }
//
//     std::vector<std::size_t> shape_vec() const { return dd_.shape_vec(); }
//
//     std::size_t size() const { return max_index_; }
//
//     bool operator==(const CorrFuncBase& other) const {
//         if (dd_ != other.dd_) { return false; }
//         if (rr_ != other.rr_) { return false; }
//         if (dr_ == C() && rd_ == C()) {
//             return other.dr_ == C() && other.rd_ == C();
//         }
//         if (dr_ == C()) {
//             return (other.dr_ == C() || other.dr_ == rd_)
//                    && (other.rd_ == C() || other.rd_ == rd_)
//                    && !(other.dr_ == C() && other.rd_ == C());
//         }
//         if (rd_ == C()) {
//             return (other.dr_ == C() || other.dr_ == dr_)
//                    && (other.rd_ == C() || other.rd_ == rd_)
//                    && !(other.dr_ == C() && other.rd_ == C());
//         }
//         if (dr_ == rd_) {
//             return (other.dr_ == C() || other.dr_ == dr_)
//                    && (other.rd_ == C() || other.rd_ == dr_)
//                    && !(other.dr_ == C() && other.rd_ == C());
//         }
//         return (other.dr_ == dr_ && other.rd_ == rd_)
//                || (other.dr_ == rd_ && other.rd_ == dr_);
//     }
//
//     bool
//     operator!=(const CorrFuncBase& other) const { return !operator==(other); }
// };
//
//
// template<class C>
// class CorrFunc : public CorrFuncBase<C> {
//     using Base = CorrFuncBase<C>;
// public:
//     CorrFunc() = default;
//
//     CorrFunc(const CorrFunc& other)
//             : Base(other) {}
//
//     CorrFunc(const C& dd)
//             : Base(dd) {}
//
//     CorrFunc(const Base& cfb)
//             : Base(cfb) {}
//
//     std::string toString() const {
//         std::ostringstream oss;
//         oss << "CorrFunc(";
//         std::string pad = "";
//         for (std::size_t i = 0; i < oss.str().size() - 1; i++) {
//             pad += " ";
//         }
//         oss << std::endl << pad << " DD=" << this->dd_ << ",";
//         oss << std::endl << pad << " RR=" << this->rr_ << ",";
//         oss << std::endl << pad << " DR=" << this->dr_ << ",";
//         oss << std::endl << pad << " RD=" << this->rd_;
//         oss << std::endl << pad << ")";
//         return oss.str();
//     }
// };
//
//
// template<class C>
// std::ostream& operator<<(std::ostream& os, const CorrFunc<C>& cf) {
//     os << cf.toString();
//     return os;
// }

#endif
