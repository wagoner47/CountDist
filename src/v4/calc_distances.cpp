#include <cmath>
#include <cstdlib>
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
#endif

vector <Separation>
get_auto_separations(const vector <SPos>& pos, const BinSpecifier& rp_bins,
                     const BinSpecifier& rl_bins, int num_threads) {
    double r_min = get_r_min(rp_bins, rl_bins);
    double r_max = get_r_max(rp_bins, rl_bins);
    size_t n = pos.size();
    vector <Separation> output;
    omp_set_num_threads(num_threads);

#if _OPENMP
#pragma omp declare reduction(merge : vector<Separation> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end())) initializer(omp_priv=omp_orig)
#pragma omp parallel for collapse(2) reduction(merge : output)
#endif
    for (size_t i = 0; i < n - 1; i++) {
        for (size_t j = 0; j < n; j++) {
            if (i < j) {
                if (pos[i].check_box(pos[j], r_max)) {
                    if (check_val_in_limits(pos[i].distance_magnitude(pos[j]),
                                            r_min,
                                            r_max)) {
                        if (pos[i].check_limits(pos[j], rp_bins, rl_bins)) {
                            output.emplace_back(pos[i], pos[j], i, j);
                        }
                    }
                }
            }
        }
    }
    return output;
}

vector <Separation>
get_auto_separations(const vector <Pos>& pos, const BinSpecifier& rp_bins,
                     const BinSpecifier& rl_bins, bool use_true,
                     int num_threads) {
    return use_true ? get_auto_separations(tpos(pos),
                                           rp_bins,
                                           rl_bins,
                                           num_threads) : get_auto_separations(
            opos(pos),
            rp_bins,
            rl_bins,
            num_threads);
}

vector <Separation>
get_cross_separations(const vector <SPos>& pos1, const vector <SPos>& pos2,
                      const BinSpecifier& rp_bins, const BinSpecifier& rl_bins,
                      int num_threads) {
    double r_min = get_r_min(rp_bins, rl_bins);
    double r_max = get_r_max(rp_bins, rl_bins);
    size_t n1 = pos1.size(), n2 = pos2.size();
    vector <Separation> output;
    omp_set_num_threads(num_threads);

#if _OPENMP
#pragma omp declare reduction(merge : vector<Separation> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end())) initializer(omp_priv=omp_orig)
#pragma omp parallel for collapse(2) reduction(merge : output)
#endif
    for (size_t i = 0; i < n1; i++) {
        for (size_t j = 0; j < n2; j++) {
            if (pos1[i].check_box(pos2[j], r_max)) {
                if (check_val_in_limits(pos1[i].distance_magnitude(pos2[j]),
                                        r_min,
                                        r_max)) {
                    if (pos1[i].check_limits(pos2[j], rp_bins, rl_bins)) {
                        output.emplace_back(pos1[i], pos2[j], i, j);
                    }
                }
            }
        }
    }
    return output;
}

vector <Separation>
get_cross_separations(const vector <Pos>& pos1, const vector <Pos>& pos2,
                      const BinSpecifier& rp_bins, const BinSpecifier& rl_bins,
                      bool use_true, int num_threads) {
    return use_true
           ? get_cross_separations(tpos(pos1),
                                   tpos(pos2),
                                   rp_bins,
                                   rl_bins,
                                   num_threads)
           : get_cross_separations(opos(pos1),
                                   opos(pos2),
                                   rp_bins,
                                   rl_bins,
                                   num_threads);
}

vector <Separation>
get_separations(const vector <SPos>& pos1, const vector <SPos>& pos2,
                const BinSpecifier& rp_bins, const BinSpecifier& rl_bins,
                bool is_auto, int num_threads) {
    return is_auto
           ? get_auto_separations(pos1, rp_bins, rl_bins, num_threads)
           : get_cross_separations(pos1, pos2, rp_bins, rl_bins, num_threads);
}

vector <Separation>
get_separations(const vector <Pos>& pos1, const vector <Pos>& pos2,
                const BinSpecifier& rp_bins, const BinSpecifier& rl_bins,
                bool use_true, bool is_auto, int num_threads) {
    return is_auto ? get_auto_separations(pos1,
                                          rp_bins,
                                          rl_bins,
                                          use_true,
                                          num_threads) : get_cross_separations(
            pos1,
            pos2,
            rp_bins,
            rl_bins,
            use_true,
            num_threads);
}

vector <TOSeparation>
get_auto_separations(const vector <Pos>& pos, const BinSpecifier& rp_bins,
                     const BinSpecifier& rl_bins, int num_threads) {
    double r_min = get_r_min(rp_bins, rl_bins);
    double r_max = get_r_max(rp_bins, rl_bins);
    size_t n = pos.size();
    omp_set_num_threads(num_threads);
    vector <TOSeparation> output;

#if _OPENMP
#pragma omp declare reduction(merge : vector<TOSeparation> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end())) initializer(omp_priv=omp_orig)
#pragma omp parallel for collapse(2) reduction(merge : output)
#endif
    for (size_t i = 0; i < n - 1; i++) {
        for (size_t j = 0; j < n; j++) {
            if (i < j) {
                if (pos[i].check_box(pos[j], r_max)) {
                    if (check_val_in_limits(pos[i].distance_magnitude_o(pos[j]),
                                            r_min,
                                            r_max)) {
                        if (pos[i].check_limits(pos[j], rp_bins, rl_bins)) {
                            output.emplace_back(pos[i], pos[j], i, j);
                        }
                    }
                }
            }
        }
    }
    return output;
}

vector <TOSeparation>
get_cross_separations(const vector <Pos>& pos1, const vector <Pos>& pos2,
                      const BinSpecifier& rp_bins, const BinSpecifier& rl_bins,
                      int num_threads) {
    double r_min = get_r_min(rp_bins, rl_bins);
    double r_max = get_r_max(rp_bins, rl_bins);
    size_t n1 = pos1.size(), n2 = pos2.size();
    vector <TOSeparation> output;
    omp_set_num_threads(num_threads);

#if _OPENMP
#pragma omp declare reduction(merge : vector<TOSeparation> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end())) initializer(omp_priv=omp_orig)
#pragma omp parallel for collapse(2) reduction(merge : output)
#endif
    for (size_t i = 0; i < n1; i++) {
        for (size_t j = 0; j < n2; j++) {
            if (pos1[i].check_box(pos2[j], r_max)) {
                if (check_val_in_limits(pos1[i].distance_magnitude_o(pos2[j]),
                                        r_min,
                                        r_max)) {
                    if (pos1[i].check_limits(pos2[j], rp_bins, rl_bins)) {
                        output.emplace_back(pos1[i], pos2[j], i, j);
                    }
                }
            }
        }
    }
    return output;
}

vector <TOSeparation>
get_separations(const vector <Pos>& pos1, const vector <Pos>& pos2,
                const BinSpecifier& rp_bins, const BinSpecifier& rl_bins,
                bool is_auto, int num_threads) {
    return is_auto
           ? get_auto_separations(pos1, rp_bins, rl_bins, num_threads)
           : get_cross_separations(pos1, pos2, rp_bins, rl_bins, num_threads);
}

vec_norm_type calculate_xi_from_vecs(vec_norm_type&& dd, vec_norm_type&& dr,
                                     vec_norm_type&& rd, vec_norm_type&& rr,
                                     CFEstimator estimator) {
    vec_norm_type xi;
    switch (estimator) {
        case CFEstimator::Landy_Szalay: {
            transform(dd.begin(),
                      dd.end(),
                      dr.begin(),
                      back_inserter(xi),
                      minus<>());
            transform(xi.begin(), xi.end(), rd.begin(), xi.begin(), minus<>());
            transform(xi.begin(),
                      xi.end(),
                      rr.begin(),
                      xi.begin(),
                      divides<>());
            transform(xi.begin(),
                      xi.end(),
                      xi.begin(),
                      [](norm_type x) { return x + 1; });
            break;
        }
        case CFEstimator::Dodelson: {
            transform(dr.begin(),
                      dr.end(),
                      dr.begin(),
                      [](norm_type x) { return 2 * x; });
            transform(rr.begin(),
                      rr.end(),
                      dr.begin(),
                      back_inserter(xi),
                      minus<>());
            transform(xi.begin(),
                      xi.end(),
                      dd.begin(),
                      xi.begin(),
                      divides<>());
            transform(xi.begin(),
                      xi.end(),
                      xi.begin(),
                      [](norm_type x) { return x + 1; });
            break;
        }
        case CFEstimator::Hamilton: {
            transform(dr.begin(),
                      dr.end(),
                      dr.begin(),
                      [](norm_type x) { return math::square(x); });
            transform(dd.begin(),
                      dd.end(),
                      rr.begin(),
                      back_inserter(xi),
                      multiplies<>());
            transform(xi.begin(),
                      xi.end(),
                      dr.begin(),
                      xi.begin(),
                      divides<>());
            transform(xi.begin(),
                      xi.end(),
                      xi.begin(),
                      [](norm_type x) { return x - 1; });
            break;
        }
        case CFEstimator::Davis_Peebles: {
            transform(dd.begin(),
                      dd.end(),
                      dr.begin(),
                      back_inserter(xi),
                      divides<>());
            transform(xi.begin(),
                      xi.end(),
                      xi.begin(),
                      [](norm_type x) { return x - 1; });
            break;
        }
        case CFEstimator::Peebles_Hauser: {
            transform(dd.begin(),
                      dd.end(),
                      rr.begin(),
                      back_inserter(xi),
                      divides<>());
            transform(xi.begin(),
                      xi.end(),
                      xi.begin(),
                      [](norm_type x) { return x - 1; });
            break;
        }
        case CFEstimator::Hewett: {
            transform(dd.begin(),
                      dd.end(),
                      dr.begin(),
                      back_inserter(xi),
                      minus<>());
            transform(xi.begin(),
                      xi.end(),
                      rr.begin(),
                      xi.begin(),
                      divides<>());
            break;
        }
    }
    return xi;
}
