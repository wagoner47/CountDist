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
#include <stdexcept>
#include "calc_distances.h"
using namespace std;

// Conditionally include OpenMP, and set variable to tell if included
#if _OPENMP
#include <omp.h>
#else
typedef int omp_int_t;
inline void omp_set_num_threads(int n) {}
inline omp_int_t omp_get_thread_num() { return 0; }
inline omp_int_t omp_get_max_threads() { return 1; }
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

vector<Pos> fill_catalog_vector(vector<double> ra_vec, vector<double> dec_vec, vector<double> rt_vec, vector<double> ro_vec, vector<double> tz_vec, vector<double> oz_vec) {
    vector<Pos> catalog;
    catalog.reserve(ra_vec.size());
    for (size_t i = 0; i < ra_vec.size(); i++) {
	Pos pos(ra_vec[i], dec_vec[i], rt_vec[i], ro_vec[i], tz_vec[i], oz_vec[i]);
	catalog.push_back(pos);
    }
    return catalog;
}

vector<Pos> fill_catalog_vector(vector<double> nx_vec, vector<double> ny_vec, vector<double> nz_vec, vector<double> rt_vec, vector<double> ro_vec, vector<double> tz_vec, vector<double> oz_vec) {
    vector<Pos> catalog;
    catalog.reserve(nx_vec.size());
    for (size_t i = 0; i < nx_vec.size(); i++) {
	Pos pos(nx_vec[i], ny_vec[i], nz_vec[i], rt_vec[i], ro_vec[i], tz_vec[i], oz_vec[i]);
	catalog.push_back(pos);
    }
    return catalog;
}

ostream& operator<<(ostream &os, const Separation &s) {
    os << s.r_perp_t << " " << s.r_par_t << " "
       << s.r_perp_o << " " << s.r_par_o << " "
       << s.ave_zo << " " << s.id1 << " " << s.id2;
    return os;
}

ostream& operator<<(ostream &os, const VectorSeparation &v) {
    os << v[0];
    for (size_t i = 1; i < v.size(); i++) {
	os << endl << v[i];
    }
    return os;
}

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
    int sign = (pos1.has_obs() && pos2.has_obs()) ? signum(pos1.ro() - pos2.ro()) * signum(pos1.rt() - pos2.rt()) : 1;
    return make_tuple(mult_fac * fabs(pos1.rt() - pos2.rt()) * sign, mult_fac * fabs(pos1.ro() - pos2.ro()));
}

tuple<double, double> r_perp(Pos pos1, Pos pos2) {
    double mult_fac = one_over_root2 * sqrt(1.0 - unit_dot(pos1, pos2));
    return make_tuple(mult_fac * (pos1.rt() + pos2.rt()), mult_fac * (pos1.ro() + pos2.ro()));
}

double ave_los_distance(Pos pos1, Pos pos2) {
    return 0.5 * (pos1.obs_redshift() + pos2.obs_redshift());
}

bool check_box(Pos pos1, Pos pos2, double max) {
    if (!(pos1.has_obs() && pos2.has_obs())) {
        return ((fabs(pos1.xt() - pos2.xt()) <= max) && (fabs(pos1.yt() - pos2.yt()) <= max) && (fabs(pos1.zt() - pos2.zt()) <= max));
    }
    else if (pos1.has_obs() && pos2.has_obs()) {
	return ((fabs(pos1.xo() - pos2.xo()) <= max) && (fabs(pos1.yo() - pos2.yo()) <= max) && (fabs(pos1.zo() - pos2.zo()) <= max));
    }
    else {
	cerr << "Cannot mix true and observed distances" << endl;
	exit(13);
    }
}

bool check_lims(double val, double min, double max) {
    return (isfinite(val) && (val >= min) && (val <= max));
}

bool check_2lims(Pos pos1, Pos pos2, double rp_min, double rp_max, double rl_min, double rl_max, bool use_true) {
    double rp, rl;
    rp = use_true?get<0>(r_perp(pos1, pos2)):get<1>(r_perp(pos1, pos2));
    rl = use_true?get<0>(r_par(pos1, pos2)):get<1>(r_par(pos1, pos2));
    return (check_lims(rp, rp_min, rp_max) && check_lims(rl, rl_min, rl_max))?true:false;
}

inline size_t npairs_each_est(double rp_min, double rp_max, double rl_min, double rl_max, double n_density) {
    return (size_t)ceil(3 * 2.0 * M_PI * (rp_max * rp_max - rp_min * rp_min) * (rl_max - rl_min) * n_density);
}

VectorSeparation get_separations(vector<Pos> pos1, vector<Pos> pos2, double rp_min, double rp_max, double rl_min, double rl_max, bool use_true, bool use_obs, bool is_auto) {
  double r_max = (isinf(rp_max) || isinf(rl_max)) ? INFINITY : sqrt((rp_max * rp_max) + (rl_max * rl_max));

  size_t n1 = pos1.size();
  size_t n2 = pos2.size();
  // double n2_density = (double) n2 / volume;
  // size_t max_size = n1 * npairs_each_est(rp_min, rp_max, rl_min, rl_max, n2_density);

  VectorSeparation separations;
  /*
  cout << "Maximum vector size: " << separations.max_size() << endl;
  cout << "Reserving " << max_size << " for separations" << endl;
  cout << boolalpha << "reserve size < max_size? " << (max_size < separations.max_size()) << noboolalpha << endl;
  separations.reserve(max_size);
  cout << "Space reserved" << endl;
  */
  omp_set_num_threads(OMP_NUM_THREADS);
  tuple<double, double> rp, rl;
  double rbar;

#if _OPENMP
#pragma omp declare reduction (merge_vs : VectorSeparation : omp_out.insert(omp_in))
#pragma omp parallel for collapse(2) reduction(merge_vs: separations) private(rp, rl, rbar)
#endif
  for(size_t i = 0; i < n1; i++) {
      for (size_t j = 0; j < n2; j++) {
	  if (is_auto && i >= j) {
	      continue;
	  }
	  if (check_box(pos1[i], pos2[j], r_max)) {
	      if (check_2lims(pos1[i], pos2[j], rp_min, rp_max, rl_min, rl_max, (use_true && !use_obs))) {
		  rp = r_perp(pos1[i], pos2[j]);
		  rl = r_par(pos1[i], pos2[j]);
		  rbar = ave_los_distance(pos1[i], pos2[j]);
		  separations.push_back(rp, rl, rbar, i, j);
	      }
	      else {
		  continue;
	      }
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

NNCounts3D get_obs_pair_counts(vector<Pos> pos1, vector<Pos> pos2, BinSpecifier rpo_binning, BinSpecifier rlo_binning, BinSpecifier zo_binning, bool is_auto) {
    NNCounts3D nn(rpo_binning, rlo_binning, zo_binning);

    omp_set_num_threads(OMP_NUM_THREADS);
    double rp, rl;
    double zbar;
    size_t n1 = pos1.size();
    size_t n2 = pos2.size();
#if _OPENMP
#pragma omp declare reduction (add : NNCounts3D : omp_out+=omp_in) initializer(omp_priv=omp_orig)
#pragma omp parallel for collapse(2) reduction(add: nn) private(rp, rl, zbar)
#endif
    for (size_t i = 0; i < n1; i++) {
	for (size_t j = 0; j < n2; j++) {
	    if (is_auto && i >= j) {
		continue;
	    }
	    rp = get<1>(r_perp(pos1[i], pos2[j]));
	    rl = get<1>(r_par(pos1[i], pos2[j]));
	    zbar = ave_los_distance(pos1[i], pos2[j]);
	    nn.assign_bin(rp, rl, zbar);
	}
    }
    return nn;
}

NNCounts1D get_true_pair_counts(vector<Pos> pos1, vector<Pos> pos2, BinSpecifier r_binning, bool is_auto) {
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
	    rp = get<0>(r_perp(pos1[i], pos2[j]));
	    rl = get<0>(r_par(pos1[i], pos2[j]));
	    nn.assign_bin(sqrt((rp * rp) + (rl * rl)));
	}
    }
    return nn;
}
