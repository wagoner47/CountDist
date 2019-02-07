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
#include <limits>
#include <stdexcept>
#include "fast_math.h"
#include "calc_distances.h"
using namespace std;

// Conditionally include OpenMP, and set variable to tell if included
#if _OPENMP
#include <omp.h>
#else
//typedef int omp_int_t;
inline void omp_set_num_threads(int n) {}
//inline omp_int_t omp_get_thread_num() { return 0; }
//inline omp_int_t omp_get_max_threads() { return 1; }
#endif

vector<Pos> fill_catalog_vector(vector<double> ra_vec, vector<double> dec_vec, vector<double> rt_vec, vector<double> ro_vec, vector<double> tz_vec, vector<double> oz_vec) {
    vector<Pos> catalog;
    catalog.reserve(ra_vec.size());
    for (size_t i = 0; i < ra_vec.size(); i++) {
        Pos pos(ra_vec[i], dec_vec[i], rt_vec[i], ro_vec[i], tz_vec[i], oz_vec[i]);
        catalog.push_back(pos);
    }
    return catalog;
}

double unit_dot(SPos pos1, SPos pos2) {
    vector<double> n1(pos1.uvec()), n2(pos2.uvec());
    return inner_product(n1.begin(), n1.end(), n2.begin(), 0.0);
}

double unit_dot(Pos pos1, Pos pos2) {
    return unit_dot(pos1.tpos(), pos2.tpos());
}

double dot(SPos pos1, SPos pos2) {
    vector<double> r1(pos1.rvec()), r2(pos2.rvec());
    return inner_product(r1.begin(), r1.end(), r2.begin(), 0.0);
}

tuple<double, double> dot(Pos pos1, Pos pos2) {
    return make_tuple(dot(pos1.tpos(), pos2.tpos()), dot(pos1.opos(), pos2.opos()));
}

double r_par(SPos pos1, SPos pos2) {
    double mult_fac = math::dsqrt_2 * sqrt(1.0 + unit_dot(pos1, pos2));
    return mult_fac * fabs(pos1.r() - pos2.r());
}

tuple<double, double> r_par(Pos pos1, Pos pos2) {
    int sign = (pos1.has_obs() && pos2.has_obs()) ? math::signof(pos1.ro() - pos2.ro()) * math::signof(pos1.rt() - pos2.rt()) : 1;
    return make_tuple(sign * r_par(pos1.tpos(), pos2.tpos()), r_par(pos1.opos(), pos2.opos()));
}

double r_perp(SPos pos1, SPos pos2) {
    double mult_fac = math::dsqrt_2 * sqrt(1.0 - unit_dot(pos1, pos2));
    return mult_fac * (pos1.r() + pos2.r());
}

tuple<double, double> r_perp(Pos pos1, Pos pos2) {
    return make_tuple(r_perp(pos1.tpos(), pos2.tpos()), r_perp(pos1.opos(), pos2.opos()));
}

double ave_z(SPos pos1, SPos pos2) {
    if (isnan(pos1.z()) || isnan(pos2.z())) { return 0.0; }
    else { return 0.5 * (pos1.z() + pos2.z()); }
}

tuple<double, double> ave_z(Pos pos1, Pos pos2) {
    return make_tuple(ave_z(pos1.tpos(), pos2.tpos()), ave_z(pos1.opos(), pos2.opos()));
}

double ave_los_distance(Pos pos1, Pos pos2) {
    return 0.5 * (pos1.zo() + pos2.zo());
}

bool check_sphere(SPos pos1, SPos pos2, double max) {
    vector<double> r1(pos1.rvec()), r2(pos2.rvec()), diff;
    transform(r1.begin(), r1.end(), r2.begin(), diff.begin(), minus<double>());
    for (auto d : diff) {
	    if (fabs(d) > max) return false;
    }
    return true;
}

bool check_sphere(Pos pos1, Pos pos2, double max) {
    if (!(pos1.has_obs() && pos2.has_obs())) {
        if (!(pos1.has_true() && pos2.has_true())) {
            throw runtime_error("Cannot mix true and observed distances");
        }
	    return check_sphere(pos1.tpos(), pos2.tpos(), max);
    }
    return check_sphere(pos1.opos(), pos2.opos(), max);
}

bool check_shell(SPos pos1, SPos pos2, double min, double max) {
    vector<double> r1(pos1.rvec()), r2(pos2.rvec()), diff(3, 0.0);
    transform(r1.begin(), r1.end(), r2.begin(), diff.begin(), minus<>());
    for (auto d : diff) {
	    if (fabs(d) < min || fabs(d) > max) return false;
    }
    return true;
}

bool check_shell(Pos pos1, Pos pos2, double min, double max) {
    if (!(pos1.has_obs() && pos2.has_obs())) {
        if (!(pos1.has_true() && pos2.has_true())) {
            throw runtime_error("Cannot mix true and observed distances");
        }
        return check_shell(pos1.tpos(), pos2.tpos(), min, max);
    }
    return check_shell(pos1.opos(), pos2.opos(), min, max);
}

bool check_lims(double val, double min, double max) {
    return (isfinite(val) && (val >= min) && (val <= max));
}

bool check_2lims(SPos pos1, SPos pos2, double rp_min, double rp_max, double rl_min, double rl_max) {
    auto rp = r_perp(pos1, pos2);
    auto rl = r_par(pos1, pos2);
    return (check_lims(rp, rp_min, rp_max) && check_lims(rl, rl_min, rl_max));
}

bool check_2lims(Pos pos1, Pos pos2, double rp_min, double rp_max, double rl_min, double rl_max, bool use_true) {
    return use_true ? check_2lims(pos1.tpos(), pos2.tpos(), rp_min, rp_max, rl_min, rl_max) : check_2lims(pos1.opos(), pos2.opos(), rp_min, rp_max, rl_min, rl_max);
}

VectorSeparation get_separations(vector<Pos> pos1, vector<Pos> pos2, double rp_min, double rp_max, double rl_min, double rl_max, bool use_true, bool use_obs, bool is_auto) {
  double r_max = (isinf(rp_max) || isinf(rl_max)) ? numeric_limits<double>::max() : sqrt(math::power(rp_max, 2) + math::power(rl_max, 2));
  double r_min = (math::isclose(rp_min, 0.0) && math::isclose(rl_min, 0.0)) ? 0.0 : sqrt(math::power(rp_min, 2) + math::power(rl_min, 2));

  size_t n1 = pos1.size();
  size_t n2 = pos2.size();

  VectorSeparation separations;
  /*
  cout << "Maximum vector size: " << separations.max_size() << endl;
  cout << "Reserving " << max_size << " for separations" << endl;
  cout << boolalpha << "reserve size < max_size? " << (max_size < separations.max_size()) << noboolalpha << endl;
  separations.reserve(max_size);
  cout << "Space reserved" << endl;
  */
  omp_set_num_threads(OMP_NUM_THREADS);

#if _OPENMP
#pragma omp declare reduction (merge_vs : VectorSeparation : omp_out.insert(omp_in)) initializer(omp_priv=omp_orig)
#pragma omp parallel for collapse(2) reduction(merge_vs: separations)
#endif
  for(size_t i = 0; i < n1; i++) {
      for (size_t j = 0; j < n2; j++) {
          if (is_auto && i >= j) continue;
          if (check_shell(pos1[i], pos2[j], r_min, r_max)) {
              if (check_2lims(pos1[i], pos2[j], rp_min, rp_max, rl_min, rl_max, (use_true && !use_obs))) {
                  tuple<double, double> rp = r_perp(pos1[i], pos2[j]);
                  tuple<double, double> rl = r_par(pos1[i], pos2[j]);
                  double rbar = ave_los_distance(pos1[i], pos2[j]);
                  separations.push_back(rp, rl, rbar, i, j);
              }
              else continue;
          }
      }
  }
  return separations;
}

size_t get_1d_indexer_from_3d(size_t x_idx, size_t y_idx, size_t z_idx, BinSpecifier x_bins, BinSpecifier y_bins, BinSpecifier z_bins) {
    if (x_idx < 0 || x_idx >= x_bins.get_nbins()) { throw out_of_range("Invalid index " + to_string(x_idx) + " for dimension with size " + to_string(x_bins.get_nbins())); }
    if (y_idx < 0 || y_idx >= y_bins.get_nbins()) { throw out_of_range("Invalid index " + to_string(y_idx) + " for dimension with size " + to_string(y_bins.get_nbins())); }
    if (z_idx < 0 || z_idx >= z_bins.get_nbins()) { throw out_of_range("Invalid index " + to_string(z_idx) + " for dimension with size " + to_string(z_bins.get_nbins())); }
    return z_bins.get_nbins() * (y_bins.get_nbins() * x_idx + y_idx) + z_idx;
}

NNCounts3D get_obs_pair_counts(vector<SPos> pos1, vector<SPos> pos2, BinSpecifier rpo_binning, BinSpecifier rlo_binning, BinSpecifier zo_binning, bool is_auto) {
    NNCounts3D nn(rpo_binning, rlo_binning, zo_binning);

    double z_min = zo_binning.get_bin_min();
    double z_max = zo_binning.get_bin_max();
    double p_min = rpo_binning.get_bin_min();
    double p_max = rpo_binning.get_bin_max();
    double l_min = rlo_binning.get_bin_min();
    double l_max = rlo_binning.get_bin_max();
    double r_max = (isinf(p_max) || isinf(l_max)) ? numeric_limits<double>::max() : sqrt(math::power(p_max, 2) + math::power(l_max, 2));
    double r_min = (math::isclose(p_min, 0.0) && math::isclose(l_min, 0.0)) ? 0.0 : sqrt(math::power(p_min, 2) + math::power(l_min, 2));

    omp_set_num_threads(OMP_NUM_THREADS);
    cout << "Number of OpenMP threads = " << omp_get_num_threads() << endl;
    size_t n1 = pos1.size(), n2 = pos2.size();

#if _OPENMP
#pragma omp declare reduction (add : NNCounts3D : omp_out+=omp_in) initializer(omp_priv=omp_orig)
#pragma omp parallel for collapse(2) reduction(add : nn)
#endif
    for (size_t i = 0; i < n1; i++) {
	for (size_t j = 0; j < n2; j++) {
	    if (is_auto && i >= j) continue;
	    if (!check_shell(pos1[i], pos2[j], r_min, r_max)) continue;
	    double zbar = ave_z(pos1[i], pos2[j]);
	    if (!check_lims(zbar, z_min, z_max)) continue;
	    double rp = r_perp(pos1[i], pos2[j]);
	    if (!check_lims(rp, p_min, p_max)) continue;
	    double rl = r_par(pos1[i], pos2[j]);
	    if (!check_lims(rl, l_min, l_max)) continue;
	    nn.assign_bin(rp, rl, zbar);
	}
    }
    return nn;
}

NNCounts3D get_obs_pair_counts(vector<Pos> pos1, vector<Pos> pos2, BinSpecifier rpo_binning, BinSpecifier rlo_binning, BinSpecifier zo_binning, bool is_auto) {
    vector<SPos> spos1(pos1.size()), spos2(pos2.size());
    for (size_t i = 0; i < pos1.size(); i++) {
	spos1[i] = pos1[i].opos();
    }
    for (size_t i = 0; i < pos2.size(); i++) {
	spos2[i] = pos2[i].opos();
    }

    return get_obs_pair_counts(spos1, spos2, rpo_binning, rlo_binning, zo_binning, is_auto);
    /*
    //AtomicWriter(debug, cerr) << "Initialize NNCounts3D" << endl;
    NNCounts3D nn(rpo_binning, rlo_binning, zo_binning);

    //AtomicWriter(debug, cerr) << "Set number of threads" << endl;
    omp_set_num_threads(OMP_NUM_THREADS);
    //AtomicWriter(debug, cerr) << "Initialize separation variables" << endl;
    double rp, rl;
    double zbar;
    //AtomicWriter(debug, cerr) << "Get catalog sizes" << endl;
    size_t n1 = pos1.size();
    size_t n2 = pos2.size();
#if _OPENMP
#pragma omp declare reduction (add : NNCounts3D : omp_out+=omp_in) initializer(omp_priv=omp_orig)
#pragma omp parallel for collapse(2) reduction(add: nn) private(rp, rl, zbar)
#endif
    for (size_t i = 0; i < n1; i++) {
	for (size_t j = 0; j < n2; j++) {
	    //AtomicWriter buff(debug, cerr);
	    if (is_auto && i >= j) {
		continue;
	    }
	    if (j - i == 1) {
		buff << "i = " << to_string(i) << endl;
	    }
	    rp = get<1>(r_perp(pos1[i], pos2[j]));
	    rl = get<1>(r_par(pos1[i], pos2[j]));
	    zbar = ave_los_distance(pos1[i], pos2[j]);
	    if (j - i == 1) {
		buff << "assigning bin" << endl;;
	    }
	    nn.assign_bin(rp, rl, zbar);
	    if (j - i == 1) {
		buff << "done" << endl;
	    }
	}
    }
    return nn;
    */
}

NNCounts1D get_true_pair_counts(vector<Pos> pos1, vector<Pos> pos2, BinSpecifier r_binning, bool is_auto, bool use_true) {
    NNCounts1D nn(r_binning);
    size_t n1 = pos1.size();
    size_t n2 = pos2.size();

    omp_set_num_threads(OMP_NUM_THREADS);
    double rp, rl;

#if _OPENMP
#pragma omp declare reduction (add : NNCounts1D : omp_out+=omp_in) initializer(omp_priv=omp_orig)
#pragma omp parallel for collapse(2) reduction(add: nn) private(rp, rl)
#endif
    for (size_t i = 0; i < n1; i++) {
	for (size_t j = 0; j < n2; j++) {
	    if (is_auto && i >= j) {
		continue;
	    }
	    if (use_true) {
		rp = get<0>(r_perp(pos1[i], pos2[j]));
		rl = get<0>(r_par(pos1[i], pos2[j]));
	    }
	    else {
		rp = get<1>(r_perp(pos1[i], pos2[j]));
		rl = get<1>(r_par(pos1[i], pos2[j]));
	    }
	    nn.assign_bin(sqrt((rp * rp) + (rl * rl)));
	}
    }
    return nn;
}
