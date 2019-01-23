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

#if defined(_OPENMP) && defined(omp_num_threads)
constexpr int OMP_NUM_THREADS = omp_num_threads;
#else
constexpr int OMP_NUM_THREADS = 1;
#endif

namespace sepconstants {
	const size_t MAX_ROWS = 10000;
}

using namespace std::placeholders;

typedef std::unordered_map<std::string, std::type_index> cdtype;

typedef std::unordered_map<std::string, std::size_t> indexer;

constexpr auto RAD2DEG = 180.0 / M_PI;
constexpr auto DEG2RAD = M_PI / 180.0;

template <typename T>
struct default_tol {
    static T get_rtol() { return T(); }
    static T get_atol() { return T(); }
};

template <typename T>
inline bool compare(T a, T b, T rtol = default_tol<T>::get_rtol(), T atol = default_tol<T>::get_atol()) {
    return std::abs(a - b) <= (atol + rtol * std::abs(b));
}

template <>
struct default_tol<double> {
    static double get_rtol() { return 1.e-5; }
    static double get_atol() { return 1.e-8; }
};

template <>
inline bool compare<double>(double a, double b, double rtol, double atol) {
    return std::fabs(a - b) <= (atol + rtol * std::fabs(b));
}

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
    Separation() : r_perp_t(), r_par_t(), r_perp_o(), r_par_o(), ave_zo(), id1(), id2() {}
    Separation(double rpt, double rlt, double rpo, double rlo, double ave_z, std::size_t i1, std::size_t i2) : r_perp_t(rpt), r_par_t(rlt), r_perp_o(rpo), r_par_o(rlo), ave_zo(ave_z), id1(i1), id2(i2) {}
    Separation(std::tuple<double, double> r_perp, std::tuple<double, double> r_par, double zbar, std::size_t i1, std::size_t i2) : r_perp_t(std::get<0>(r_perp)), r_par_t(std::get<0>(r_par)), r_perp_o(std::get<1>(r_perp)), r_par_o(std::get<1>(r_par)), ave_zo(zbar), id1(i1), id2(i2) {}
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
	return ((_is_set == other._is_set) && (log_binning == other.log_binning) && (nbins == other.nbins) && compare(bin_size, other.bin_size) && compare(bin_min, other.bin_min) && compare(bin_max, other.bin_max));
    }

    bool operator!=(const BinSpecifier &other) const {
	return !(*this == other);
    }
};

std::size_t get_1d_indexer_from_3d(std::size_t x_idx, std::size_t y_idx, std::size_t z_idx, BinSpecifier x_bins, BinSpecifier y_bins, BinSpecifier z_bins);

class NNCounts3D {
    BinSpecifier rpo_bins, rlo_bins, zo_bins;
    std::vector<std::pair<std::size_t, std::size_t>> counts_;
    std::size_t n_tot_, max_index_;

    int assign_1d_bin(double value, BinSpecifier binning) {
	if ((value < binning.get_bin_min()) || (value > binning.get_bin_max())) {
	    return -1;
	}
	else {
	    double diff;
	    if (binning.get_log_binning()) {
		diff = (log(value) - log(binning.get_bin_min()));
	    }
	    else {
		diff = value - binning.get_bin_min();
	    }
	    return (int) floor(diff / binning.get_bin_size());
	}
    }

    void on_bin_update() {
	//std::cout << "Updating bins in NNCounts3D" << std::endl;
	n_tot_ = 0;
	max_index_ = rpo_bins.get_nbins() * rlo_bins.get_nbins() * zo_bins.get_nbins();
	if (!counts_.empty()) {
	    //std::cout << "Clearing stored counts" << std::endl;
	    counts_.clear();
	}
    }

    void add_count(std::pair<std::size_t, std::size_t> new_index) {
	auto it = std::find_if(counts_.begin(), counts_.end(), [&](std::pair<std::size_t, std::size_t> el) { return std::get<0>(el) == std::get<0>(new_index); });
	if (it == counts_.end()) {
	    counts_.push_back(new_index);
	    std::sort(counts_.begin(), counts_.end(), [](std::pair<std::size_t, std::size_t> a, std::pair<std::size_t, std::size_t> b) { return std::get<0>(a) < std::get<0>(b); });
	}
	else {
	    auto insert_at = std::distance(counts_.begin(), it);
	    std::pair<std::size_t, std::size_t> temp_pair = counts_[insert_at];
	    counts_[insert_at] = std::make_pair(std::get<0>(temp_pair), std::get<1>(temp_pair)+std::get<1>(new_index));
	}
    }

    void add_count(std::size_t new_index) {
	std::pair<std::size_t, std::size_t> new_pair(new_index, 1);
	add_count(new_pair);
    }

 public:
    // (default) empty constructor
    NNCounts3D() {}

    // copy constructor
    NNCounts3D(const NNCounts3D& other) {
	//std::cout << "In NNCounts3D copy constructor" << std::endl;
	rpo_bins = other.rpo_bins;
	rlo_bins = other.rlo_bins;
	zo_bins = other.zo_bins;
	on_bin_update();
	//std::cout << "Copying counts from other" << std::endl;
	counts_ = other.counts_;
	//std::cout << "Counts copied from other" << std::endl;
	n_tot_ = other.n_tot_;
	//std::cout << "Leaving NNCounts3D copy constructor" << std::endl;
    }

    // Like a copy constructor, but from pickled objects (for python)
    NNCounts3D(BinSpecifier rpo_binning, BinSpecifier rlo_binning, BinSpecifier zo_binning, std::vector<std::pair<std::size_t, std::size_t>> counts, std::size_t n_tot) {
	rpo_bins = rpo_binning;
	rlo_bins = rlo_binning;
	zo_bins = zo_binning;
	on_bin_update();
	counts_ = counts;
	n_tot_ = n_tot;
    }

    NNCounts3D(BinSpecifier rpo_binning, BinSpecifier rlo_binning, BinSpecifier zo_binning) {
	//std::cout << "In NNCounts3D constructor" << std::endl;
	//std::cout << "Setting binning" << std::endl;
	rpo_bins = rpo_binning;
	rlo_bins = rlo_binning;
	zo_bins = zo_binning;
	on_bin_update();
	//std::cout << "Leaving NNCounts3D constructor" << std::endl;
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
	int rpo_bin = assign_1d_bin(r_perp, rpo_bins);
	if (rpo_bin > -1) {
	    int rlo_bin = assign_1d_bin(r_par, rlo_bins);
	    if (rlo_bin > -1) {
		int zo_bin = assign_1d_bin(zbar, zo_bins);
		if (zo_bin > -1) {
		    return get_1d_indexer(rpo_bin, rlo_bin, zo_bin);
		}
	    }
	}
	return -1;
    }

    void assign_bin(double r_perp, double r_par, double zbar) {
	n_tot_++;
	int bin_index = get_bin(r_perp, r_par, zbar);
	if (bin_index > -1) {
	    add_count((std::size_t) bin_index);
	}
    }

    const std::size_t operator[](std::size_t idx) const {
	if (idx >= max_index_) { throw std::out_of_range("Invalid index " + std::to_string(idx) + " for size of " + std::to_string(max_index_)); }
	auto it = std::find_if(counts_.begin(), counts_.end(), [&](std::pair<std::size_t, std::size_t> el) { return std::get<0>(el) == idx; });

	if (it == counts_.end()) {
	    return 0;
	}
	else {
	    return std::get<1>(*it);
	}
    }

    std::size_t n_tot() const { return n_tot_; }

    std::vector<std::size_t> counts() const {
	std::vector<std::size_t> temp(max_index_, 0);
	for (auto p : counts_) {
	    temp[std::get<0>(p)] = std::get<1>(p);
	}
	return temp;
    }

    BinSpecifier rpo_bin_info() const { return rpo_bins; }

    BinSpecifier rlo_bin_info() const { return rlo_bins; }

    BinSpecifier zo_bin_info() const { return zo_bins; }

    std::vector<std::pair<std::size_t, std::size_t>> get_counts_1d() const { return counts_; }

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
	for (auto p : other.counts_) {
	    add_count(p);
	}
	return *this;
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
    std::vector<std::pair<std::size_t, std::size_t>> counts_;

    void on_bin_update() {
	n_tot_ = 0;
	max_index_ = binner.get_nbins();
	if (!counts_.empty()) { counts_.clear(); }
    }

    void add_count(std::pair<std::size_t, std::size_t> new_index) {
	auto it = std::find_if(counts_.begin(), counts_.end(), [&](std::pair<std::size_t, std::size_t> el) { return std::get<0>(el) == std::get<0>(new_index); });
	if (it == counts_.end()) {
	    counts_.push_back(new_index);
	    std::sort(counts_.begin(), counts_.end(), [](std::pair<std::size_t, std::size_t> a, std::pair<std::size_t, std::size_t> b) { return std::get<0>(a) < std::get<0>(b); });
	}
	else {
	    auto insert_at = std::distance(counts_.begin(), it);
	    std::pair<std::size_t, std::size_t> temp_pair = counts_[insert_at];
	    counts_[insert_at] = std::make_pair(std::get<0>(temp_pair), std::get<1>(temp_pair)+std::get<1>(new_index));
	}
    }

    void add_count(std::size_t new_index) {
	std::pair<std::size_t, std::size_t> new_pair(new_index, 1);
	add_count(new_pair);
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
    NNCounts1D(BinSpecifier binning, std::vector<std::pair<std::size_t, std::size_t>> counts, std::size_t n_tot) {
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

    int get_bin(double value) {
	if (value > binner.get_bin_min() && value < binner.get_bin_max()) {
	    double diff;
	    if (binner.get_log_binning()) {
		diff = (log(value) - log(binner.get_bin_min()));
	    }
	    else {
		diff = value - binner.get_bin_min();
	    }
	    return (int) floor(diff / binner.get_bin_size());
	}
	else { return -1; }
    }

    void assign_bin(double value) {
	n_tot_++;
	int bin_index = get_bin(value);
	if (bin_index > -1) {
	    add_count((std::size_t) bin_index);
	}
    }

    const std::size_t operator[](std::size_t idx) const {
	if (idx >= max_index_) { throw std::out_of_range("Index " + std::to_string(idx) + " is out of bounds for size of " + std::to_string(max_index_)); }
	auto it = std::find_if(counts_.begin(), counts_.end(), [&](std::pair<std::size_t, std::size_t> el) { return std::get<0>(el) == idx; });
	if (it == counts_.end()) {
	    return 0;
	}
	else {
	    return std::get<1>(*it);
	}
    }

    std::size_t n_tot() const { return n_tot_; }

    std::vector<std::size_t> counts() const {
	std::vector<std::size_t> temp(max_index_, 0);
	for (auto p : counts_) {
	    temp[std::get<0>(p)] = std::get<1>(p);
	}
	return temp;
    }

    std::vector<std::pair<std::size_t, std::size_t>> get_counts_pairs() const { return counts_; }

    BinSpecifier bin_info() const { return binner; }

    NNCounts1D& operator+=(const NNCounts1D& other) {
	if (binner != other.binner) {
	    std::cerr << "Attempted to combine NNCounts1D instances with different binning" << std::endl;
	    std::cerr << "this.binner: " << binner.toString() << std::endl;
	    std::cerr << "other.binner: " << other.binner.toString() << std::endl;
	    throw std::runtime_error("Cannot combine NNCounts1D instances with different binning schemes");
	}
	n_tot_ += other.n_tot_;
	for (auto p : other.counts_) {
	    add_count(p);
	}
	return *this;
    }

    std::string toString() {
	return "NNCounts1D(bins=" + binner.toString() + ")";
    }
};

NNCounts3D get_obs_pair_counts(std::vector<Pos> pos1, std::vector<Pos> pos2, BinSpecifier rpo_binning, BinSpecifier rlo_binning, BinSpecifier zo_binning, bool is_auto);

NNCounts1D get_true_pair_counts(std::vector<Pos> pos1, std::vector<Pos> pos2, BinSpecifier r_binning, bool is_auto, bool use_true=true);

#endif
