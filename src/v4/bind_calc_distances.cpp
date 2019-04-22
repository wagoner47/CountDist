#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/iostream.h>
#include <stdexcept>
#include <cstddef>
#include <iostream>
#include <cmath>
#include <vector>
#include <utility>
#include <string>
#include <type_traits>
#include <tuple>
#include "fast_math.h"
#include "calc_distances.h"

namespace py = pybind11;
using namespace pybind11::literals;

#if _OPENMP
#include <omp.h>
#endif

struct PosCatalog {
    double RA, DEC, D_TRUE, D_OBS, Z_TRUE, Z_OBS;
};

struct SPosCatalog {
    double RA, DEC, D, Z;
};

bool same_catalog(const py::array_t<SPosCatalog>& cat1,
                  const py::array_t<SPosCatalog>& cat2) {
    if (cat1.size() != cat2.size()) { return false; }
    for (ssize_t i = 0; i < cat1.size(); i++) {
        bool dclose = std::isnan(cat1.at(i).D)
                      ? std::isnan(cat2.at(i).D)
                      : math::isclose(cat1.at(i).D, cat2.at(i).D);
        bool zclose = std::isnan(cat1.at(i).Z)
                      ? std::isnan(cat2.at(i).Z)
                      : math::isclose(cat1.at(i).Z, cat2.at(i).Z);
        if (!(math::isclose(cat1.at(i).RA, cat2.at(i).RA)
              && math::isclose(cat1.at(i).DEC, cat2.at(i).DEC) && dclose
              && zclose)) {
            return false;
        }
    }
    return true;
}

bool same_catalog(const py::array_t<PosCatalog>& cat1,
                  const py::array_t<PosCatalog>& cat2) {
    if (cat1.size() != cat2.size()) { return false; }
    for (ssize_t i = 0; i < cat1.size(); i++) {
        bool dtclose = std::isnan(cat1.at(i).D_TRUE)
                       ? std::isnan(cat2.at(i).D_TRUE)
                       : math::isclose(cat1.at(i).D_TRUE, cat2.at(i).D_TRUE);
        bool ztclose = std::isnan(cat1.at(i).Z_TRUE)
                       ? std::isnan(cat2.at(i).Z_TRUE)
                       : math::isclose(cat1.at(i).Z_TRUE, cat2.at(i).Z_TRUE);
        bool doclose = std::isnan(cat1.at(i).D_OBS)
                       ? std::isnan(cat2.at(i).D_OBS)
                       : math::isclose(cat1.at(i).D_OBS, cat2.at(i).D_OBS);
        bool zoclose = std::isnan(cat1.at(i).Z_OBS)
                       ? std::isnan(cat2.at(i).Z_OBS)
                       : math::isclose(cat1.at(i).Z_OBS, cat2.at(i).Z_OBS);
        if (!(math::isclose(cat1.at(i).RA, cat2.at(i).RA)
              && math::isclose(cat1.at(i).DEC, cat2.at(i).DEC) && dtclose
              && ztclose && doclose && zoclose)) {
            return false;
        }
    }
    return true;
}

struct SepDType {
    double r_perp, r_par, zbar;
    std::size_t id1, id2;
};

struct TOSepDType {
    double r_perp_t, r_par_t, zbar_t, r_perp_o, r_par_o, zbar_o;
    std::size_t id1, id2;
};

template<typename T>
py::array_t<T, 0> mkarray_from_vec(const std::vector<T>& vec,
                                   const std::vector<std::size_t>& shape) {
    std::vector<std::size_t> strides;
    for (std::size_t i = 0; i < shape.size(); i++) {
        std::size_t this_size = sizeof(T);
        for (std::size_t j = i + 1; j < shape.size(); j++) {
            this_size *= shape[j];
        }
        strides.push_back(this_size);
    }
    return py::array_t<T>(shape, strides, vec.data());
}

template<typename T>
py::array_t<T, 0> mkarray_from_vec(const std::vector<T>& vec) {
    return py::array_t<T>({vec.size()}, {sizeof(T)}, vec.data());
}

template<typename T, std::size_t N>
py::array_t<T, 0> mkarray_from_array(const std::array<T, N>& arr) {
    return py::array_t<T>({N}, {sizeof(T)}, arr.data());
}

template<std::size_t N>
py::array_t<count_type, 0> convert_counts_vec(const NNCountsND<N>& nn) {
    return mkarray_from_vec(nn.counts(), nn.shape_vec());
}

template<std::size_t N>
py::array_t<double, 0> convert_normed_counts_vec(const NNCountsND<N>& nn) {
    return mkarray_from_vec(nn.normed_counts(), nn.shape_vec());
}

template<typename T, std::size_t N>
auto convert_pyarray_to_vec_stdarray(const py::array_t<T>& input_arr) {
    auto uarr = input_arr.template unchecked<2>();
    if ((std::size_t) uarr.shape(1) != N) {
        throw std::length_error("Invalid input py::array shape ("
                                + std::to_string(uarr.shape(0)) + ","
                                + std::to_string(uarr.shape(1))
                                + "): last dimension must have length "
                                + std::to_string(N));
    }
    std::array<T, N> temp{};
    std::vector<std::array<T, N>> output_arr;
    for (ssize_t i = 0; i < uarr.shape(0); i++) {
        for (ssize_t j = 0; j < uarr.shape(1); j++) {
            temp[j] = uarr(i, j);
        }
        output_arr.push_back(temp);
    }
    return output_arr;
}

template<std::size_t N>
py::array_t<std::size_t> get_1d_index_from_nd(const py::array_t<int>& indices,
                                              const std::array<BinSpecifier,
                                                               N>& bins) {
    auto idx_arr = convert_pyarray_to_vec_stdarray<int, N>(indices);
    auto result = py::array_t<std::size_t>(idx_arr.size());
    for (std::size_t i = 0; i < idx_arr.size(); i++) {
        result.mutable_at(i) = get_1d_indexer_from_nd(idx_arr[i], bins);
    }
    return result;
}

template<std::size_t N>
static void bind_get_1d_indices(py::module& mod) {
    mod.def(("get_1d_index_from_" + std::to_string(N) + "d").c_str(),
            py::overload_cast<const std::array<int, N>&,
                              const std::array<BinSpecifier,
                                               N>&>(& get_1d_indexer_from_nd<
                    N>),
            ("Get the 1-dimensional index corresponding to the given index for "
             + std::to_string(N)
             + "-dimensional data with shape specified by the BinSpecifier objects in 'bins'")
                    .c_str(),
            "index"_a,
            "bins"_a);
    mod.def(("get_1d_index_from_" + std::to_string(N) + "d").c_str(),
            py::overload_cast<const py::array_t<int>&,
                              const std::array<BinSpecifier,
                                               N>&>(& get_1d_index_from_nd<N>),
            ("Get the 1-dimensional indices corresponding to each given ND index for "
             + std::to_string(N)
             + "-dimensional data with shape specified by the BinSpecifier objects in 'bins'")
                    .c_str(),
            "index"_a,
            "bins"_a);
}

void set_separations(SepDType& to, const Separation& from) {
    to.r_perp = from.r_perp;
    to.r_par = from.r_par;
    to.zbar = from.zbar;
    to.id1 = from.id1;
    to.id2 = from.id2;
}

void set_toseparations(TOSepDType& to, const TOSeparation& from) {
    to.r_perp_t = from.tsep.r_perp;
    to.r_par_t = from.tsep.r_par;
    to.zbar_t = from.tsep.zbar;
    to.r_perp_o = from.osep.r_perp;
    to.r_par_o = from.osep.r_par;
    to.zbar_o = from.osep.zbar;
    to.id1 = from.tsep.id1;
    to.id2 = from.tsep.id2;
}

template<typename T>
py::array mkarray_via_buffer(std::size_t n) {
    return py::array(py::buffer_info(nullptr,
                                     sizeof(T),
                                     py::format_descriptor<T>::format(),
                                     1,
                                     {n},
                                     {sizeof(T)}));
}

template<typename S, typename toS>
py::array_t<toS, 0> create_recarray_from_vector(const std::vector<S>& vec,
                                                std::function<void(toS&,
                                                                   const S&)> f) {
    std::size_t n = vec.size();
    auto arr = mkarray_via_buffer<toS>(n);
    auto req = arr.request();
    auto ptr = static_cast<toS*>(req.ptr);
    for (std::size_t i = 0; i < n; i++) {
        f(ptr[i], vec[i]);
    }
    return arr;
}

/*
py::array_t<SepDType> convert_vector_separations(const VectorSeparation& vs) {
    return create_recarray_from_vector<Separation, SepDType>(vs.sep_vec(), set_separations);
}

py::array_t<TOSepDType> convert_vector_separations(const VectorTOSeparation& vs) {
    return create_recarray_from_vector<TOSeparation, TOSepDType>(vs.sep_vec(), set_toseparations);
}
*/

py::array_t<SepDType>
convert_vector_separations(const std::vector<Separation>& vs) {
    return create_recarray_from_vector<Separation, SepDType>(vs,
                                                             set_separations);
}

py::array_t<TOSepDType>
convert_vector_separations(const std::vector<TOSeparation>& vs) {
    return create_recarray_from_vector<TOSeparation, TOSepDType>(vs,
                                                                 set_toseparations);
}

std::vector<Pos> convert_catalog(const py::array_t<PosCatalog>& arr) {
    auto uarr = arr.unchecked<1>();
    auto n = (std::size_t) uarr.size();
    std::vector<Pos> vec;
    for (std::size_t i = 0; i < n; i++) {
        auto row = uarr(i);
        Pos pos(row.RA, row.DEC, row.D_TRUE, row.D_OBS, row.Z_TRUE, row.Z_OBS);
        vec.push_back(pos);
    }
    return vec;
}

std::vector<SPos> convert_catalog(const py::array_t<SPosCatalog>& arr) {
    auto uarr = arr.unchecked<1>();
    auto n = (std::size_t) uarr.size();
    std::vector<SPos> vec(n);
    for (std::size_t i = 0; i < n; i++) {
        auto row = uarr(i);
        vec[i] = SPos(row.RA, row.DEC, row.D, row.Z);
    }
    return vec;
}

py::array_t<SepDType>
get_auto_separations_arr(const py::array_t<SPosCatalog>& cat,
                         const BinSpecifier& rp_bins,
                         const BinSpecifier& rl_bins,
                         int num_threads = OMP_NUM_THREADS) {
    py::gil_scoped_acquire acquire;
    auto pos = convert_catalog(cat);
    auto vs = get_auto_separations(pos, rp_bins, rl_bins, num_threads);
    return convert_vector_separations(vs);
}

py::array_t<SepDType>
get_auto_separations_arr(const py::array_t<PosCatalog>& cat,
                         const BinSpecifier& rp_bins,
                         const BinSpecifier& rl_bins, bool use_true,
                         int num_threads = OMP_NUM_THREADS) {
    py::gil_scoped_acquire acquire;
    auto pos = convert_catalog(cat);
    auto vs =
            get_auto_separations(pos, rp_bins, rl_bins, use_true, num_threads);
    return convert_vector_separations(vs);
}

py::array_t<SepDType>
get_cross_separations_arr(const py::array_t<SPosCatalog>& cat1,
                          const py::array_t<SPosCatalog>& cat2,
                          const BinSpecifier& rp_bins,
                          const BinSpecifier& rl_bins,
                          int num_threads = OMP_NUM_THREADS) {
    py::gil_scoped_acquire acquire;
    auto pos1 = convert_catalog(cat1);
    auto pos2 = convert_catalog(cat2);
    auto vs = get_cross_separations(pos1, pos2, rp_bins, rl_bins, num_threads);
    return convert_vector_separations(vs);
}

py::array_t<SepDType>
get_cross_separations_arr(const py::array_t<PosCatalog>& cat1,
                          const py::array_t<PosCatalog>& cat2,
                          const BinSpecifier& rp_bins,
                          const BinSpecifier& rl_bins, bool use_true,
                          int num_threads = OMP_NUM_THREADS) {
    py::gil_scoped_acquire acquire;
    auto pos1 = convert_catalog(cat1);
    auto pos2 = convert_catalog(cat2);
    auto vs = get_cross_separations(pos1,
                                    pos2,
                                    rp_bins,
                                    rl_bins,
                                    use_true,
                                    num_threads);
    return convert_vector_separations(vs);
}

py::array_t<SepDType> get_separations_arr(const py::array_t<SPosCatalog>& cat1,
                                          const py::array_t<SPosCatalog>& cat2,
                                          const BinSpecifier& rp_bins,
                                          const BinSpecifier& rl_bins,
                                          bool is_auto,
                                          int num_threads = OMP_NUM_THREADS) {
    auto pos1 = convert_catalog(cat1);
    auto pos2 = convert_catalog(cat2);
    auto vs =
            get_separations(pos1, pos2, rp_bins, rl_bins, is_auto, num_threads);
    return convert_vector_separations(vs);
}

py::array_t<SepDType> get_separations_arr(const py::array_t<PosCatalog>& cat1,
                                          const py::array_t<PosCatalog>& cat2,
                                          const BinSpecifier& rp_bins,
                                          const BinSpecifier& rl_bins,
                                          bool use_true, bool is_auto,
                                          int num_threads = OMP_NUM_THREADS) {
    auto pos1 = convert_catalog(cat1);
    auto pos2 = convert_catalog(cat2);
    auto vs = get_separations(pos1,
                              pos2,
                              rp_bins,
                              rl_bins,
                              use_true,
                              is_auto,
                              num_threads);
    return convert_vector_separations(vs);
}

py::array_t<TOSepDType>
get_auto_separations_arr(const py::array_t<PosCatalog>& cat,
                         const BinSpecifier& rp_bins,
                         const BinSpecifier& rl_bins,
                         int num_threads = OMP_NUM_THREADS) {
    py::gil_scoped_acquire acquire;
    auto pos = convert_catalog(cat);
    auto vs = get_auto_separations(pos, rp_bins, rl_bins, num_threads);
    std::cout << vs.size() << std::endl;
    return convert_vector_separations(vs);
}

py::array_t<TOSepDType>
get_cross_separations_arr(const py::array_t<PosCatalog>& cat1,
                          const py::array_t<PosCatalog>& cat2,
                          const BinSpecifier& rp_bins,
                          const BinSpecifier& rl_bins,
                          int num_threads = OMP_NUM_THREADS) {
    py::gil_scoped_acquire acquire;
    auto pos1 = convert_catalog(cat1);
    auto pos2 = convert_catalog(cat2);
    auto vs = get_cross_separations(pos1, pos2, rp_bins, rl_bins, num_threads);
    return convert_vector_separations(vs);
}

py::array_t<TOSepDType> get_separations_arr(const py::array_t<PosCatalog>& cat1,
                                            const py::array_t<PosCatalog>& cat2,
                                            const BinSpecifier& rp_bins,
                                            const BinSpecifier& rl_bins,
                                            bool is_auto,
                                            int num_threads = OMP_NUM_THREADS) {
    auto pos1 = convert_catalog(cat1);
    auto pos2 = convert_catalog(cat2);
    auto vs =
            get_separations(pos1, pos2, rp_bins, rl_bins, is_auto, num_threads);
    return convert_vector_separations(vs);
}

template<std::size_t N>
py::array_t<std::size_t, 0>
get_1d_indexer(const NNCountsND<N>& self, const py::array_t<int>& indices) {
    auto idx_vec = convert_pyarray_to_vec_stdarray<int, N>(indices);
    auto result = py::array_t<std::size_t>(idx_vec.size());
    for (std::size_t i = 0; i < idx_vec.size(); i++) {
        result.mutable_at(i) = self.get_1d_indexer(idx_vec[i]);
    }
    return result;
}

template<std::size_t N>
py::array_t<int, 0>
get_bin(const NNCountsND<N>& self, const py::array_t<double>& values) {
    auto val_vec = convert_pyarray_to_vec_stdarray<double, N>(values);
    auto result = py::array_t<int>(val_vec.size());
    for (std::size_t i = 0; i < val_vec.size(); i++) {
        result.mutable_at(i) = self.get_bin(val_vec[i]);
    }
    return result;
}

template<std::size_t N>
void process_auto(NNCountsND<N>&, const py::array_t<SPosCatalog>&,
                  int= OMP_NUM_THREADS) {}

template<>
void process_auto(NNCountsND<3>& self, const py::array_t<SPosCatalog>& cat,
                  int num_threads) {
    py::gil_scoped_acquire acquire;
    using NNType = NNCountsND<3>;
    NNType temp(self.bin_info());
    auto vcat = convert_catalog(cat);
    std::size_t n = vcat.size();
    omp_set_num_threads(num_threads);
#if _OPENMP
#pragma omp declare reduction(+ : NNType : omp_out+=omp_in) initializer(omp_priv=omp_orig)
#pragma omp parallel for collapse(2) reduction(+ : temp)
#endif
    for (std::size_t i = 0; i < n - 1; i++) {
        for (std::size_t j = 0; j < n; j++) {
            if (i >= j) { continue; }
            temp.process_pair(vcat[i], vcat[j]);
        }
    }
    self += temp;
}

template<>
void process_auto(NNCountsND<2>& self, const py::array_t<SPosCatalog>& cat,
                  int num_threads) {
    py::gil_scoped_acquire acquire;
    using NNType = NNCountsND<2>;
    NNType temp(self.bin_info());
    auto vcat = convert_catalog(cat);
    std::size_t n = vcat.size();
    omp_set_num_threads(num_threads);
#if _OPENMP
#pragma omp declare reduction(+ : NNType : omp_out+=omp_in) initializer(omp_priv=omp_orig)
#pragma omp parallel for collapse(2) reduction(+ : temp)
#endif
    for (std::size_t i = 0; i < n - 1; i++) {
        for (std::size_t j = 0; j < n; j++) {
            if (i >= j) { continue; }
            temp.process_pair(vcat[i], vcat[j]);
        }
    }
    self += temp;
}

template<>
void process_auto(NNCountsND<1>& self, const py::array_t<SPosCatalog>& cat,
                  int num_threads) {
    py::gil_scoped_acquire acquire;
    using NNType = NNCountsND<1>;
    NNType temp(self.bin_info());
    auto vcat = convert_catalog(cat);
    std::size_t n = vcat.size();
    omp_set_num_threads(num_threads);
#if _OPENMP
#pragma omp declare reduction(+ : NNType : omp_out+=omp_in) initializer(omp_priv=omp_orig)
#pragma omp parallel for collapse(2) reduction(+ : temp)
#endif
    for (std::size_t i = 0; i < n; i++) {
        for (std::size_t j = 0; j < n; j++) {
            if (i >= j) { continue; }
            temp.process_pair(vcat[i], vcat[j]);
        }
    }
    self += temp;
}

template<std::size_t N>
void process_auto(NNCountsND<N>&, const py::array_t<PosCatalog>&, bool,
                  int= OMP_NUM_THREADS) {}

template<>
void process_auto(NNCountsND<3>& self, const py::array_t<PosCatalog>& cat,
                  bool use_true, int num_threads) {
    py::gil_scoped_acquire acquire;
    using NNType = NNCountsND<3>;
    NNType temp(self.bin_info());
    auto vcat = convert_catalog(cat);
    std::size_t n = vcat.size();
    omp_set_num_threads(num_threads);
#if _OPENMP
#pragma omp declare reduction(add : NNType : omp_out+=omp_in) initializer(omp_priv=omp_orig)
#pragma omp parallel for collapse(2) reduction(add : temp)
#endif
    for (std::size_t i = 0; i < n - 1; i++) {
        for (std::size_t j = 0; j < n; j++) {
            if (i >= j) { continue; }
            temp.process_pair(vcat[i], vcat[j], use_true);
        }
    }
    self += temp;
}

template<>
void process_auto(NNCountsND<2>& self, const py::array_t<PosCatalog>& cat,
                  bool use_true, int num_threads) {
    py::gil_scoped_acquire acquire;
    using NNType = NNCountsND<2>;
    NNType temp(self.bin_info());
    auto vcat = convert_catalog(cat);
    std::size_t n = vcat.size();
    omp_set_num_threads(num_threads);
#if _OPENMP
#pragma omp declare reduction(add : NNType : omp_out+=omp_in) initializer(omp_priv=omp_orig)
#pragma omp parallel for collapse(2) reduction(add : temp)
#endif
    for (std::size_t i = 0; i < n - 1; i++) {
        for (std::size_t j = 0; j < n; j++) {
            if (i >= j) { continue; }
            temp.process_pair(vcat[i], vcat[j], use_true);
        }
    }
    self += temp;
}

template<>
void process_auto(NNCountsND<1>& self, const py::array_t<PosCatalog>& cat,
                  bool use_true, int num_threads) {
    py::gil_scoped_acquire acquire;
    using NNType = NNCountsND<1>;
    NNType temp(self.bin_info());
    auto vcat = convert_catalog(cat);
    std::size_t n = vcat.size();
    omp_set_num_threads(num_threads);
#if _OPENMP
#pragma omp declare reduction(add : NNType : omp_out+=omp_in) initializer(omp_priv=omp_orig)
#pragma omp parallel for collapse(2) reduction(add : temp)
#endif
    for (std::size_t i = 0; i < n - 1; i++) {
        for (std::size_t j = 0; j < n; j++) {
            if (i >= j) { continue; }
            temp.process_pair(vcat[i], vcat[j], use_true);
        }
    }
    self += temp;
}

template<std::size_t N>
void process_cross(NNCountsND<N>&, const py::array_t<SPosCatalog>&,
                   const py::array_t<SPosCatalog>&, int= OMP_NUM_THREADS) {}

template<>
void process_cross(NNCountsND<3>& self, const py::array_t<SPosCatalog>& cat1,
                   const py::array_t<SPosCatalog>& cat2, int num_threads) {
    py::gil_scoped_acquire acquire;
    using NNType = NNCountsND<3>;
    NNType temp(self.bin_info());
    auto vcat1 = convert_catalog(cat1);
    auto vcat2 = convert_catalog(cat2);
    std::size_t n1 = vcat1.size(), n2 = vcat2.size();
    omp_set_num_threads(num_threads);
#if _OPENMP
#pragma omp declare reduction(add : NNType : omp_out+=omp_in) initializer(omp_priv=omp_orig)
#pragma omp parallel for collapse(2) reduction(add : temp)
#endif
    for (std::size_t i = 0; i < n1; i++) {
        for (std::size_t j = 0; j < n2; j++) {
            temp.process_pair(vcat1[i], vcat2[j]);
        }
    }
    self += temp;
}

template<>
void process_cross(NNCountsND<2>& self, const py::array_t<SPosCatalog>& cat1,
                   const py::array_t<SPosCatalog>& cat2, int num_threads) {
    py::gil_scoped_acquire acquire;
    using NNType = NNCountsND<2>;
    NNType temp(self.bin_info());
    auto vcat1 = convert_catalog(cat1);
    auto vcat2 = convert_catalog(cat2);
    std::size_t n1 = vcat1.size(), n2 = vcat2.size();
    omp_set_num_threads(num_threads);
#if _OPENMP
#pragma omp declare reduction(add : NNType : omp_out+=omp_in) initializer(omp_priv=omp_orig)
#pragma omp parallel for collapse(2) reduction(add : temp)
#endif
    for (std::size_t i = 0; i < n1; i++) {
        for (std::size_t j = 0; j < n2; j++) {
            temp.process_pair(vcat1[i], vcat2[j]);
        }
    }
    self += temp;
}

template<>
void process_cross(NNCountsND<1>& self, const py::array_t<SPosCatalog>& cat1,
                   const py::array_t<SPosCatalog>& cat2, int num_threads) {
    py::gil_scoped_acquire acquire;
    using NNType = NNCountsND<1>;
    NNType temp(self.bin_info());
    auto vcat1 = convert_catalog(cat1);
    auto vcat2 = convert_catalog(cat2);
    std::size_t n1 = vcat1.size(), n2 = vcat2.size();
    omp_set_num_threads(num_threads);
#if _OPENMP
#pragma omp declare reduction(add : NNType : omp_out+=omp_in) initializer(omp_priv=omp_orig)
#pragma omp parallel for collapse(2) reduction(add : temp)
#endif
    for (std::size_t i = 0; i < n1; i++) {
        for (std::size_t j = 0; j < n2; j++) {
            temp.process_pair(vcat1[i], vcat2[j]);
        }
    }
    self += temp;
}

template<std::size_t N>
void process_cross(NNCountsND<N>&, const py::array_t<PosCatalog>&,
                   const py::array_t<PosCatalog>&, bool,
                   int= OMP_NUM_THREADS) {}

template<>
void process_cross(NNCountsND<3>& self, const py::array_t<PosCatalog>& cat1,
                   const py::array_t<PosCatalog>& cat2, bool use_true,
                   int num_threads) {
    py::gil_scoped_acquire acquire;
    using NNType = NNCountsND<3>;
    NNType temp(self.bin_info());
    auto vcat1 = convert_catalog(cat1);
    auto vcat2 = convert_catalog(cat2);
    std::size_t n1 = vcat1.size(), n2 = vcat2.size();
    omp_set_num_threads(num_threads);
#if _OPENMP
#pragma omp declare reduction(add : NNType : omp_out+=omp_in) initializer(omp_priv=omp_orig)
#pragma omp parallel for collapse(2) reduction(add : temp)
#endif
    for (std::size_t i = 0; i < n1; i++) {
        for (std::size_t j = 0; j < n2; j++) {
            temp.process_pair(vcat1[i], vcat2[j], use_true);
        }
    }
    self += temp;
}

template<>
void process_cross(NNCountsND<2>& self, const py::array_t<PosCatalog>& cat1,
                   const py::array_t<PosCatalog>& cat2, bool use_true,
                   int num_threads) {
    py::gil_scoped_acquire acquire;
    using NNType = NNCountsND<2>;
    NNType temp(self.bin_info());
    auto vcat1 = convert_catalog(cat1);
    auto vcat2 = convert_catalog(cat2);
    std::size_t n1 = vcat1.size(), n2 = vcat2.size();
    omp_set_num_threads(num_threads);
#if _OPENMP
#pragma omp declare reduction(add : NNType : omp_out+=omp_in) initializer(omp_priv=omp_orig)
#pragma omp parallel for collapse(2) reduction(add : temp)
#endif
    for (std::size_t i = 0; i < n1; i++) {
        for (std::size_t j = 0; j < n2; j++) {
            temp.process_pair(vcat1[i], vcat2[j], use_true);
        }
    }
    self += temp;
}

template<>
void process_cross(NNCountsND<1>& self, const py::array_t<PosCatalog>& cat1,
                   const py::array_t<PosCatalog>& cat2, bool use_true,
                   int num_threads) {
    py::gil_scoped_acquire acquire;
    using NNType = NNCountsND<1>;
    NNType temp(self.bin_info());
    auto vcat1 = convert_catalog(cat1);
    auto vcat2 = convert_catalog(cat2);
    std::size_t n1 = vcat1.size(), n2 = vcat2.size();
    omp_set_num_threads(num_threads);
#if _OPENMP
#pragma omp declare reduction(add : NNType : omp_out+=omp_in) initializer(omp_priv=omp_orig)
#pragma omp parallel for collapse(2) reduction(add : temp)
#endif
    for (std::size_t i = 0; i < n1; i++) {
        for (std::size_t j = 0; j < n2; j++) {
            temp.process_pair(vcat1[i], vcat2[j], use_true);
        }
    }
    self += temp;
}

template<std::size_t N>
void process(NNCountsND<N>& self, const py::array_t<SPosCatalog>& cat,
             int num_threads = OMP_NUM_THREADS) {
    process_auto(self, cat, num_threads);
}

template<std::size_t N>
void
process(NNCountsND<N>& self, const py::array_t<PosCatalog>& cat, bool use_true,
        int num_threads = OMP_NUM_THREADS) {
    process_auto(self, cat, use_true, num_threads);
}

template<std::size_t N>
void process(NNCountsND<N>& self, const py::array_t<SPosCatalog>& cat1,
             const py::array_t<SPosCatalog>& cat2, bool is_auto,
             int num_threads = OMP_NUM_THREADS) {
    if (is_auto) { process_auto(self, cat1, num_threads); }
    else { process_cross(self, cat1, cat2, num_threads); }
}

template<std::size_t N>
void process(NNCountsND<N>& self, const py::array_t<PosCatalog>& cat1,
             const py::array_t<PosCatalog>& cat2, bool use_true, bool is_auto,
             int num_threads = OMP_NUM_THREADS) {
    if (is_auto) { process_auto(self, cat1, use_true, num_threads); }
    else { process_cross(self, cat1, cat2, use_true, num_threads); }
}

template<std::size_t N>
void process(NNCountsND<N>& self, const py::array_t<SPosCatalog>& cat1,
             const py::array_t<SPosCatalog>& cat2,
             int num_threads = OMP_NUM_THREADS) {
    if (same_catalog(cat1, cat2)) { process_auto(self, cat1, num_threads); }
    else { process_cross(self, cat1, cat2, num_threads); }
}

template<std::size_t N>
void process(NNCountsND<N>& self, const py::array_t<PosCatalog>& cat1,
             const py::array_t<PosCatalog>& cat2, bool use_true,
             int num_threads = OMP_NUM_THREADS) {
    if (same_catalog(cat1, cat2)) {
        process_auto(self, cat1, use_true, num_threads);
    }
    else { process_cross(self, cat1, cat2, use_true, num_threads); }
}

static void declareBinSpecifier(py::module& mod) {
    py::class_<BinSpecifier> cls(mod,
                                 "BinSpecifier",
                                 "Structure for the needed attributes for binning in a variable");
    cls.def(py::init<double, double, double, bool>(),
            "Construct from min, max, and bin size. Set 'use_log_bins' to True for logarithmic binning",
            "bin_min"_a,
            "bin_max"_a,
            "bin_width"_a,
            "use_log_bins"_a);
    cls.def(py::init<double, double, double, bool, std::string>(),
            "Construct from min, max, and bin size, and give a name. Set 'use_log_bins' to True for logarithmic binning",
            "bin_min"_a,
            "bin_max"_a,
            "bin_width"_a,
            "use_log_bins"_a,
            "name"_a);
    cls.def(py::init<double, double, std::size_t, bool>(),
            "Construct from min, max, and number of bins. Set 'use_log_bins' to True for logarithmic binning",
            "bin_min"_a,
            "bin_max"_a,
            "num_bins"_a,
            "use_log_bins"_a);
    cls.def(py::init<double, double, std::size_t, bool, std::string>(),
            "Construct from min, max, and number of bins, and give a name. Set 'use_log_bins' to True for logarithmic binning",
            "bin_min"_a,
            "bin_max"_a,
            "num_bins"_a,
            "use_log_bins"_a,
            "name"_a);
    cls.def(py::init<const BinSpecifier&>(),
            "Copy constructor: make a copy of 'other'",
            "other"_a);
    cls.def(py::init<>(), "Empty initialization");
    cls.def("update",
            & BinSpecifier::update,
            "Update the values of this instance with the values of 'other', preferring 'other'",
            "other"_a);
    cls.def("fill",
            & BinSpecifier::fill,
            "Update the values of this instance with the values of 'other', preferring the current values",
            "other"_a);
    cls.def_property("bin_min",
                     & BinSpecifier::get_bin_min,
                     & BinSpecifier::set_bin_min,
                     "Minimum bin edge");
    cls.def_property("bin_max",
                     & BinSpecifier::get_bin_max,
                     & BinSpecifier::set_bin_max,
                     "Maximum bin edge");
    cls.def_property("bin_size",
                     & BinSpecifier::get_bin_size,
                     & BinSpecifier::set_bin_size,
                     "Size of bins. Note that this may be different than input to make 'nbins' an integer");
    cls.def_property("nbins",
                     & BinSpecifier::get_nbins,
                     & BinSpecifier::set_nbins,
                     "Number of bins");
    cls.def_property("log_binning",
                     & BinSpecifier::get_log_binning,
                     & BinSpecifier::set_log_binning,
                     "Whether to use logarithmic binning (True) or not (False)");
    cls.def_property("name",
                     & BinSpecifier::get_name,
                     & BinSpecifier::set_name,
                     "Name of this instance");
    cls.def("__repr__", & BinSpecifier::toString);
    cls.def(py::pickle([](const BinSpecifier& bs) { // __getstate__
        return py::make_tuple(bs.get_bin_min(),
                              bs.get_bin_max(),
                              bs.get_bin_size(),
                              bs.get_nbins(),
                              bs.get_log_binning(),
                              bs.get_name());
    }, [](py::tuple t) { // __setstate__
        if (t.size() != 5 && t.size() != 6) {
            throw std::runtime_error(
                    "Invalid state with " + std::to_string(t.size())
                    + " elements");
        }

        BinSpecifier bs;
        bs.set_bin_min(t[0].cast<double>());
        bs.set_bin_max(t[1].cast<double>());
        bs.set_bin_size(t[2].cast<double>());
        bs.set_nbins(t[3].cast<std::size_t>());
        bs.set_log_binning(t[4].cast<bool>());
        if (t.size() == 6) {
            bs.set_name(t[5].cast<std::string>());
        }
        return bs;
    }));
    cls.def("__eq__", & BinSpecifier::operator==, py::is_operator());
    cls.def("__neq__", & BinSpecifier::operator!=, py::is_operator());
    cls.def("assign_bin",
            py::vectorize(& BinSpecifier::assign_bin),
            "Get the bin index/indices corresponding to the given value(s)",
            "value"_a);
    cls.def("log_step_func",
            py::vectorize(& BinSpecifier::log_step_func),
            "Finds the bin edge with index 'i' assuming logarithmic steps",
            "i"_a);
    cls.def("lin_step_func",
            py::vectorize(& BinSpecifier::lin_step_func),
            "Finds the bin edge with index 'i' assuming linear steps",
            "i"_a);
    cls.def("log_step_func_center",
            py::vectorize(& BinSpecifier::log_step_func_center),
            "Finds the bin center with index 'i' assuming logarithmic steps",
            "i"_a);
    cls.def("lin_step_func_center",
            py::vectorize(& BinSpecifier::lin_step_func_center),
            "Finds the bin center with index 'i' assuming linear steps",
            "i"_a);
    cls.def_property_readonly("lower_bin_edges", [](const BinSpecifier& self) {
        return mkarray_from_vec(self.lower_bin_edges());
    }, "Get the lower edges of all bins");
    cls.def_property_readonly("upper_bin_edges", [](const BinSpecifier& self) {
        return mkarray_from_vec(self.upper_bin_edges());
    }, "Get the upper edges of all bins");
    cls.def_property_readonly("bin_edges", [](const BinSpecifier& self) {
        return mkarray_from_vec(self.bin_edges());
    }, "Get the nbins+1 bin edges");
    cls.def_property_readonly("bin_centers",
                              [](const BinSpecifier& self) {
                                  return mkarray_from_vec(self.bin_centers());
                              },
                              "Get the centers of all bins. May not be centered in linear space if log binning is used");
    cls.def_property_readonly("bin_widths",
                              [](const BinSpecifier& self) {
                                  return mkarray_from_vec(self.bin_widths());
                              },
                              "Get the difference between linear space bin edges. Note that these will all be the same value for linear binning, but not for log binning");
    cls.def_property_readonly("is_set",
                              & BinSpecifier::is_set,
                              "Flag to specify whether the values have actually been set (True) or not (False) yet");
}

static void declareSPos(py::module& mod) {
    py::class_<SPos> cls(mod,
                         "SPos",
                         "Store only true or only observed position of an object");
    cls.def(py::init<>(), "Empty constructor");
    cls.def(py::init<double, double, double, double>(),
            "Initialize from values",
            "ra"_a,
            "dec"_a,
            "r"_a,
            "z"_a);
    cls.def(py::init<const SPos&>(), "Copy constructor", "other"_a);
    cls.def("__eq__", & SPos::operator==, py::is_operator());
    cls.def("__neq__", & SPos::operator!=, py::is_operator());
    cls.def("__repr__", & SPos::toString);
    cls.def_property_readonly("ra", & SPos::ra, "Right ascension (degrees)");
    cls.def_property_readonly("dec", & SPos::dec, "Declination (degrees)");
    cls.def_property_readonly("r", & SPos::r, "Line of sight distance");
    cls.def_property_readonly("z", & SPos::z, "Redshift");
    cls.def_property_readonly("uvec", [](const SPos& self) {
        return mkarray_from_array(self.uvec());
    }, "Unit vector in cartesian coordinates");
    cls.def_property_readonly("rvec", [](const SPos& self) {
        return mkarray_from_array(self.rvec());
    }, "Cartesian coordinate vector");
    cls.def("dot_norm",
            & SPos::dot_norm,
            "Take the dot product between the unit vectors of this object and the other",
            "other"_a);
    cls.def("dot_mag",
            & SPos::dot_mag,
            "Take the dot product between the vector with magnitude of this object and that of other",
            "other"_a);
    cls.def("distance_zbar",
            & SPos::distance_zbar,
            "Get the average redshift component of the distance between self and other (used in 3D pair count binning). If either redshift is NaN, returns NaN",
            "other"_a);
    cls.def("distance_par",
            & SPos::distance_par,
            "Get the parallel separation between self and other. If r is NaN for either, returns NaN",
            "other"_a);
    cls.def("distance_perp",
            & SPos::distance_perp,
            "Get the perpendicular separation between self and other. If r is NaN for either, returns NaN",
            "other"_a);
    cls.def("distance_vector", [](const SPos& self, const SPos& other) {
        return mkarray_from_array(self.distance_vector(other));
    }, "Get the Cartesian distance vector between self and other", "other"_a);
    cls.def("distance_magnitude",
            & SPos::distance_magnitude,
            "Get the magnitude of the separation between self and other",
            "other"_a);
    cls.def("check_box",
            py::overload_cast<const SPos&, double>(& SPos::check_box,
                                                   py::const_),
            "Check whether all components of the separation are within a box of side length 'max'",
            "other"_a,
            "max"_a);
    cls.def("check_box",
            py::overload_cast<const SPos&,
                              const BinSpecifier&>(& SPos::check_box,
                                                   py::const_),
            "Check whether all components of the separation are within a box of side length 'binner.bin_max'",
            "other"_a,
            "binner"_a);
    cls.def("check_shell",
            py::overload_cast<const SPos&, double, double>(& SPos::check_shell,
                                                           py::const_),
            "Check whether all components of the difference vector between self and other lie within r_min and r_max (each, not magnitude)",
            "other"_a,
            "r_min"_a,
            "r_max"_a);
    cls.def("check_shell",
            py::overload_cast<const SPos&,
                              const BinSpecifier&>(& SPos::check_shell,
                                                   py::const_),
            "Check whether all components of the difference vector between self and other lie within the limits specified in binner",
            "other"_a,
            "binner"_a);
    cls.def("check_shell",
            py::overload_cast<const SPos&, double>(& SPos::check_shell,
                                                   py::const_),
            "Check whether all components of the difference vector between self and other are less than r_max, i.e. that they lie between 0 and r_max",
            "other"_a,
            "r_max"_a);
    cls.def("check_limits",
            py::overload_cast<const SPos&, double, double, double,
                              double>(& SPos::check_limits, py::const_),
            "Check whether the perpendicular and parallel separations between self and other lie within the limits specified in each direction",
            "other"_a,
            "rperp_min"_a,
            "rperp_max"_a,
            "rpar_min"_a,
            "rpar_max"_a);
    cls.def("check_limits",
            py::overload_cast<const SPos&, const BinSpecifier&,
                              const BinSpecifier&>(& SPos::check_limits,
                                                   py::const_),
            "Check whether the perpendicular and parallel separations between self and other lie within the limits specified by the binner in each direction",
            "other"_a,
            "perp_binner"_a,
            "par_binner"_a);
}

static void declarePos(py::module& mod) {
    py::class_<Pos> cls(mod,
                        "Pos",
                        "Store the position of a object. Note that tz/oz refer to true/observed redshift while zt/zo refer to true/observed cartesian coordinates");
    cls.def(py::init<>(), "Empty constructor");
    cls.def(py::init<double, double, double, double, double, double>(),
            "Initialize from values",
            "ra"_a,
            "dec"_a,
            "rt"_a,
            "ro"_a,
            "tz"_a,
            "oz"_a);
    cls.def(py::init<const Pos&>(), "Copy constructor", "other"_a);
    cls.def_property_readonly("uvec", [](const Pos& self) {
        return mkarray_from_array(self.uvec());
    }, "Get the unit vector of the position");
    cls.def_property_readonly("rtvec", [](const Pos& self) {
        return mkarray_from_array(self.rtvec());
    }, "Get the vector for the true position");
    cls.def_property_readonly("rovec", [](const Pos& self) {
        return mkarray_from_array(self.rovec());
    }, "Get the vector for the observed position");
    cls.def_property_readonly("ra", & Pos::ra, "Right ascension (degrees)");
    cls.def_property_readonly("dec", & Pos::dec, "Declination (degrees)");
    cls.def_property_readonly("rt", & Pos::rt, "True line of sight distance");
    cls.def_property_readonly("ro",
                              & Pos::ro,
                              "Observed line of sight distance");
    cls.def_property_readonly("zt", & Pos::zt, "True redshift");
    cls.def_property_readonly("zo", & Pos::zo, "Observed redshift");
    cls.def_property_readonly("tpos",
                              & Pos::tpos,
                              "SPos instance for true position");
    cls.def_property_readonly("opos",
                              & Pos::opos,
                              "SPos instance for observed position");
    cls.def_property_readonly("has_true",
                              & Pos::has_true,
                              "Whether the object has true distance/redshift");
    cls.def_property_readonly("has_obs",
                              & Pos::has_obs,
                              "Whether the object has observed distance/redshift");
    cls.def("__eq__", & Pos::operator==, py::is_operator());
    cls.def("__neq__", & Pos::operator!=, py::is_operator());
    cls.def("__repr__", & Pos::toString);
    cls.def("dot_norm",
            py::overload_cast<const SPos&>(& Pos::dot_norm, py::const_),
            "Take the dot product of the unit vectors of self and an SPos object",
            "other"_a);
    cls.def("dot_norm",
            py::overload_cast<const Pos&>(& Pos::dot_norm, py::const_),
            "Take the dot product of the unit vectors of self and other",
            "other"_a);
    cls.def("dot_mag",
            py::overload_cast<const SPos&>(& Pos::dot_mag, py::const_),
            "Take the dot product with magnitude between self and an SPos object, preferring the true distance if available",
            "other"_a);
    cls.def("dot_mag",
            py::overload_cast<const Pos&>(& Pos::dot_mag, py::const_),
            "Take the dot product with magnitude between self and other, using either the true distance for both or the observed distance for both depending on which is available. If both true and observed distances are available in both cases, prefer the true distance",
            "other"_a);
    cls.def("zbar_t",
            py::overload_cast<const SPos&>(& Pos::zbar_t, py::const_),
            "Get the average of the true redshift of self and the redshift in an SPos object",
            "other"_a);
    cls.def("zbar_t",
            py::overload_cast<const Pos&>(& Pos::zbar_t, py::const_),
            "Get the average true redshift of self and other",
            "other"_a);
    cls.def("zbar_o",
            py::overload_cast<const SPos&>(& Pos::zbar_o, py::const_),
            "Get the average of the observed redshift of self and the redshift in an SPos object",
            "other"_a);
    cls.def("zbar_o",
            py::overload_cast<const Pos&>(& Pos::zbar_o, py::const_),
            "Get the average observed redshift of self and other",
            "other"_a);
    cls.def("distance_zbar",
            & Pos::distance_zbar,
            "Get the average true and observed redshifts of self and other, as a tuple of (zbar_t, zbar_o)",
            "other"_a);
    cls.def("r_par_t",
            py::overload_cast<const SPos&>(& Pos::r_par_t, py::const_),
            "Get the parallel separation between self and an SPos object, using the true distance from self",
            "other"_a);
    cls.def("r_par_t",
            py::overload_cast<const Pos&>(& Pos::r_par_t, py::const_),
            "Get the parallel separation between self and other using the true distances",
            "other"_a);
    cls.def("r_par_t_signed",
            & Pos::r_par_t_signed,
            "Get the true parallel separation with sign based on the orientation of true vs observed positions",
            "other"_a);
    cls.def("r_par_o",
            py::overload_cast<const SPos&>(& Pos::r_par_o, py::const_),
            "Get the parallel separation between self and an SPos object, using the observed distance from self",
            "other"_a);
    cls.def("r_par_o",
            py::overload_cast<const Pos&>(& Pos::r_par_o, py::const_),
            "Get the parallel separation between self and other using the observed distances",
            "other"_a);
    cls.def("distance_par",
            & Pos::distance_par,
            "Get the true and observed parallel separations between self and other, as a tuple of (r_par_t, r_par_o)",
            "other"_a);
    cls.def("r_perp_t",
            py::overload_cast<const SPos&>(& Pos::r_perp_t, py::const_),
            "Get the perpendicular separation between self and an SPos object, using the true distance from self",
            "other"_a);
    cls.def("r_perp_t",
            py::overload_cast<const Pos&>(& Pos::r_perp_t, py::const_),
            "Get the perpendicular separation between self and other using the true distances",
            "other"_a);
    cls.def("r_perp_o",
            py::overload_cast<const SPos&>(& Pos::r_perp_o, py::const_),
            "Get the perpendicular separation between self and an SPos object, using the observed distance from self",
            "other"_a);
    cls.def("r_perp_o",
            py::overload_cast<const Pos&>(& Pos::r_perp_o, py::const_),
            "Get the perpendicular separation between self and other using the observed distances",
            "other"_a);
    cls.def("distance_perp",
            & Pos::distance_perp,
            "Get the true and observed perpendicular separations between self and other, as a tuple of (r_perp_t, r_perp_o)",
            "other"_a);
    cls.def("distance_vector_t",
            [](const Pos& self, const SPos& other) {
                return mkarray_from_array(self.distance_vector_t(other));
            },
            "Get the difference vector between the true position of self and single position other",
            "other"_a);
    cls.def("distance_vector_t",
            [](const Pos& self, const Pos& other) {
                return mkarray_from_array(self.distance_vector_t(other));
            },
            "Get the difference vector between the true position of self and the true position of other",
            "other"_a);
    cls.def("distance_vector_o",
            [](const Pos& self, const SPos& other) {
                return mkarray_from_array(self.distance_vector_o(other));
            },
            "Get the difference vector between the observed position of self and single position other",
            "other"_a);
    cls.def("distance_vector_o",
            [](const Pos& self, const Pos& other) {
                return mkarray_from_array(self.distance_vector_o(other));
            },
            "Get the difference vector between the observed position of self and the observed position of other",
            "other"_a);
    cls.def("distance_vector",
            [](const Pos& self, const SPos& other, bool use_true) {
                return mkarray_from_array(self.distance_vector(other,
                                                               use_true));
            },
            "Get the difference vector between self and single position other, specifying whether to use the true or observed position of self",
            "other"_a,
            "use_true"_a);
    cls.def("distance_vector",
            [](const Pos& self, const Pos& other, bool use_true) {
                return mkarray_from_array(self.distance_vector(other,
                                                               use_true));
            },
            "Get the difference vector between self and other, specifying whether to use the true or observed positions",
            "other"_a,
            "use_true"_a);
    cls.def("distance_magnitude_t",
            py::overload_cast<const SPos&>(& Pos::distance_magnitude_t,
                                           py::const_),
            "Get the separation magnitude between the true position of self and single position other",
            "other"_a);
    cls.def("distance_magnitude_t",
            py::overload_cast<const Pos&>(& Pos::distance_magnitude_t,
                                          py::const_),
            "Get the separation magnitude between the true position of self and the true position of other",
            "other"_a);
    cls.def("distance_magnitude_o",
            py::overload_cast<const SPos&>(& Pos::distance_magnitude_o,
                                           py::const_),
            "Get the separation magnitude between the observed position of self and single position other",
            "other"_a);
    cls.def("distance_magnitude_o",
            py::overload_cast<const Pos&>(& Pos::distance_magnitude_o,
                                          py::const_),
            "Get the separation magnitude between the observed position of self and the observed position of other",
            "other"_a);
    cls.def("distance_magnitude",
            py::overload_cast<const SPos&, bool>(& Pos::distance_magnitude,
                                                 py::const_),
            "Get the separation magnitude between self and single position other, specifying whether to use the true or observed position of self",
            "other"_a,
            "use_true"_a);
    cls.def("distance_magnitude",
            py::overload_cast<const Pos&, bool>(& Pos::distance_magnitude,
                                                py::const_),
            "Get the separation magnitude between self and other, specifying whether to use the true or observed positions",
            "other"_a,
            "use_true"_a);
    cls.def("check_box",
            py::overload_cast<const SPos&, double>(& Pos::check_box,
                                                   py::const_),
            "Check whether all components of the separation are within a box of side length 'max', using the observed position in self if set",
            "other"_a,
            "max"_a);
    cls.def("check_box",
            py::overload_cast<const SPos&,
                              const BinSpecifier&>(& Pos::check_box,
                                                   py::const_),
            "Check whether all components of the separation are within a box of side length 'binner.bin_max', using the observed position in self if set",
            "other"_a,
            "binner"_a);
    cls.def("check_box",
            py::overload_cast<const Pos&, double>(& Pos::check_box, py::const_),
            "Check whether all components of the separation are within a box of side length 'max', using the observed positions if set in self and other",
            "other"_a,
            "max"_a);
    cls.def("check_box",
            py::overload_cast<const Pos&, const BinSpecifier&>(& Pos::check_box,
                                                               py::const_),
            "Check whether all components of the separation are within a box of side length 'binner.bin_max', using the observed positions if set in self and other",
            "other"_a,
            "binner"_a);
    cls.def("check_shell",
            py::overload_cast<const SPos&, double, double>(& Pos::check_shell,
                                                           py::const_),
            "Check whether each component of the difference vector between self and other is within r_min and r_max, defaulting to the observed position of self",
            "other"_a,
            "r_min"_a,
            "r_max"_a);
    cls.def("check_shell",
            py::overload_cast<const SPos&,
                              const BinSpecifier&>(& Pos::check_shell,
                                                   py::const_),
            "Check whether each component of the difference vector between self and other is within the limits specified byt binner, defaulting to the observed position of self",
            "other"_a,
            "binner"_a);
    cls.def("check_shell",
            py::overload_cast<const SPos&, double>(& Pos::check_shell,
                                                   py::const_),
            "Check whether each component of the difference vector between self and other is within (0 and) r_max, defaulting to the observed position of self",
            "other"_a,
            "r_max"_a);
    cls.def("check_shell",
            py::overload_cast<const Pos&, double, double>(& Pos::check_shell,
                                                          py::const_),
            "Check whether each component of the difference vector between self and other is within r_min and r_max, defaulting to the observed positions",
            "other"_a,
            "r_min"_a,
            "r_max"_a);
    cls.def("check_shell",
            py::overload_cast<const Pos&,
                              const BinSpecifier&>(& Pos::check_shell,
                                                   py::const_),
            "Check whether each component of the difference vector between self and other is within the limits specified byt binner, defaulting to the observed positions",
            "other"_a,
            "binner"_a);
    cls.def("check_shell",
            py::overload_cast<const Pos&, double>(& Pos::check_shell,
                                                  py::const_),
            "Check whether each component of the difference vector between self and other is within (0 and) r_max, defaulting to the observed positions",
            "other"_a,
            "r_max"_a);
    cls.def("check_limits",
            py::overload_cast<const SPos&, double, double, double,
                              double>(& Pos::check_limits, py::const_),
            "Check whether the separation components are within the limits in each direction, defaulting to the observed position of self",
            "other"_a,
            "rperp_min"_a,
            "rperp_max"_a,
            "rpar_min"_a,
            "rpar_max"_a);
    cls.def("check_limits",
            py::overload_cast<const SPos&, const BinSpecifier&,
                              const BinSpecifier&>(& Pos::check_limits,
                                                   py::const_),
            "Check whether the separation components are within the limits for the binner in each direction, defaulting to the observed position of self",
            "other"_a,
            "perp_binner"_a,
            "par_binner"_a);
    cls.def("check_limits",
            py::overload_cast<const Pos&, double, double, double,
                              double>(& Pos::check_limits, py::const_),
            "Check whether the separation components are within the limits in each direction, defaulting to the observed positions",
            "other"_a,
            "rperp_min"_a,
            "rperp_max"_a,
            "rpar_min"_a,
            "rpar_max"_a);
    cls.def("check_limits",
            py::overload_cast<const Pos&, const BinSpecifier&,
                              const BinSpecifier&>(& Pos::check_limits,
                                                   py::const_),
            "Check whether the separation components are within the limits for the binner in each direction, defaulting to the observed positions",
            "other"_a,
            "perp_binner"_a,
            "par_binner"_a);
}

template<std::size_t N>
static py::class_<NNCountsND<N>> declareNNCountsND(py::module& mod) {
    using Class = NNCountsND<N>;
    using BSType = std::array<BinSpecifier, N>;
    py::class_<Class> cls(mod,
                          ("NNCounts" + std::to_string(N) + "D").c_str(),
                          ("Pair counts in " + std::to_string(N) + "D")
                                  .c_str());
    cls.def(py::init<>(), "Empty constructor");
    cls.def(py::init<const Class&>(), "Copy constructor", "other"_a);
    cls.def(py::init<BSType>(),
            "Construct from list of BinSpecifier objects",
            "bins"_a);
    cls.def(py::init<BSType, vec_counts_type, std::size_t>(),
            "Constructor from pieces, for pickling support",
            "bins"_a,
            "counts"_a,
            "n_tot"_a);
    cls.def_property_readonly("bin_info",
                              & Class::bin_info,
                              "The binning information in all dimensions");
    cls.def("get_1d_index",
            py::overload_cast<
                    const std::array<int, N>&>(& Class::get_1d_indexer,
                                               py::const_),
            "Get 1D index for the given array assumed to be the ND index",
            "index"_a);
    //cls.def("get_1d_index", (std::size_t (Class::*)(const std::array<int,N>&) const) &Class::get_1d_indexer, "Get  1D index for the given array, assumed to be the ND index of a single bin", "index"_a);
    cls.def("get_1d_index",
            & get_1d_indexer<N>,
            "Get 1D indices for each ND index array",
            "indices"_a);
    cls.def("get_bin",
            & Class::get_bin,
            "Get the 1D bin index corresponding to the (1D, size N) array of ND separation values",
            "values"_a);
    cls.def("get_bin",
            & get_bin<N>,
            "Get the 1D bin indices corresponding to each array of ND separation values",
            "values"_a);
    cls.def("process_pair",
            py::overload_cast<const SPos&, const SPos&>(& Class::process_pair),
            "Process a single pair of single positions",
            "pos1"_a,
            "pos2"_a);
    cls.def("process_pair",
            py::overload_cast<const Pos&, const Pos&,
                              const bool>(& Class::process_pair),
            "Process a single pair of positions, specifying whether to use true or observed separations",
            "pos1"_a,
            "pos2"_a,
            "use_true"_a);
    cls.def("process_auto",
            py::overload_cast<Class&, const py::array_t<SPosCatalog>&,
                              int>(& process_auto<N>),
            py::call_guard<py::gil_scoped_release>(),
            "Process auto-correlation pairs in a catalog of single positions",
            "cat"_a,
            "num_threads"_a = OMP_NUM_THREADS);
    cls.def("process_auto",
            py::overload_cast<Class&, const py::array_t<PosCatalog>&, bool,
                              int>(& process_auto<N>),
            py::call_guard<py::gil_scoped_release>(),
            "Process auto-correlation paris in catalog of positions, specifying whether to use true or observed separations",
            "cat"_a,
            "use_true"_a,
            "num_threads"_a = OMP_NUM_THREADS);
    cls.def("process_cross",
            py::overload_cast<Class&, const py::array_t<SPosCatalog>&,
                              const py::array_t<SPosCatalog>&,
                              int>(& process_cross<N>),
            py::call_guard<py::gil_scoped_release>(),
            "Process cross-correlation pairs in catalogs of single positions",
            "cat1"_a,
            "cat2"_a,
            "num_threads"_a = OMP_NUM_THREADS);
    cls.def("process_cross",
            py::overload_cast<Class&, const py::array_t<PosCatalog>&,
                              const py::array_t<PosCatalog>&, bool,
                              int>(& process_cross<N>),
            py::call_guard<py::gil_scoped_release>(),
            "Process cross-correlation pairs in catalogs of positions, specifying whether to use true or observed separations",
            "cat1"_a,
            "cat2"_a,
            "use_true"_a,
            "num_threads"_a = OMP_NUM_THREADS);
    cls.def("process",
            py::overload_cast<Class&, const py::array_t<SPosCatalog>&,
                              int>(& process<N>),
            py::call_guard<py::gil_scoped_release>(),
            "Process correlation pairs in a catalog of single positions",
            "cat1"_a,
            "num_threads"_a = OMP_NUM_THREADS);
    cls.def("process",
            py::overload_cast<Class&, const py::array_t<PosCatalog>&, bool,
                              int>(& process<N>),
            py::call_guard<py::gil_scoped_release>(),
            "Process correlation pairs in a catalog of positions, specifying whether to use true or observed separations",
            "cat1"_a,
            "use_true"_a,
            "num_threads"_a);
    cls.def("process",
            py::overload_cast<Class&, const py::array_t<SPosCatalog>&,
                              const py::array_t<SPosCatalog>&, bool,
                              int>(& process<N>),
            py::call_guard<py::gil_scoped_release>(),
            "Process pairs between catalogs of single positions, specifying whether this is an auto or cross correlation",
            "cat1"_a,
            "cat2"_a,
            "is_auto"_a,
            "num_threads"_a = OMP_NUM_THREADS);
    cls.def("process",
            py::overload_cast<Class&, const py::array_t<PosCatalog>&,
                              const py::array_t<PosCatalog>&, bool, bool,
                              int>(& process<N>),
            py::call_guard<py::gil_scoped_release>(),
            "Process pairs between catalogs of positions, specifying whether this is an auto or cross correlation and whether to use true or observed separations",
            "cat1"_a,
            "cat2"_a,
            "use_true"_a,
            "is_auto"_a,
            "num_threads"_a = OMP_NUM_THREADS);
    cls.def("process",
            py::overload_cast<Class&, const py::array_t<SPosCatalog>&,
                              const py::array_t<SPosCatalog>&, int>(& process<
                    N>),
            py::call_guard<py::gil_scoped_release>(),
            "Process pairs between catalogs of single positions, allowing for automatic deduction on whether this is an auto or cross correlation based on equivalence of the catalogs",
            "cat1"_a,
            "cat2"_a,
            "num_threads"_a = OMP_NUM_THREADS);
    cls.def("process",
            py::overload_cast<Class&, const py::array_t<PosCatalog>&,
                              const py::array_t<PosCatalog>&, bool,
                              int>(& process<N>),
            py::call_guard<py::gil_scoped_release>(),
            "Process pairs between catalogs of positions, allowing for automatic deduction of auto or cross correlation based on equivalence of the catalogs, but still specifying whether to use true or observed separations",
            "cat1"_a,
            "cat2"_a,
            "use_true"_a,
            "num_threads"_a = OMP_NUM_THREADS);
    cls.def("__getitem__",
            py::vectorize(& Class::operator[]),
            py::is_operator());
    cls.def_property_readonly("ntot",
                              & Class::n_tot,
                              "Total number of processed pairs");
    cls.def_property_readonly("counts", [](const Class& self) {
        return convert_counts_vec(self);
    }, ("Counts of pairs in " + std::to_string(N) + "D bins").c_str());
    cls.def_property_readonly("normed_counts",
                              [](const Class& self) {
                                  return convert_normed_counts_vec(self);
                              },
                              ("Normalized counts of pairs in "
                               + std::to_string(N) + "D bins").c_str());
    cls.def(py::self += py::self);
    cls.def(py::self += float());
    cls.def(py::self += int());
    cls.def(py::self + float());
    cls.def(py::self + int());
    cls.def(float() + py::self);
    cls.def(int() + py::self);
    cls.def("__eq__",
            [](const Class& lhs, const Class& rhs) { return lhs == rhs; },
            py::is_operator());
    cls.def("__neq__",
            [](const Class& lhs, const Class& rhs) { return lhs != rhs; },
            py::is_operator());
    cls.def_property_readonly("size",
                              & Class::size,
                              "Total size of data array");
    cls.def_property_readonly("nbins_nonzero",
                              & Class::nbins_nonzero,
                              "Number of bins with non-zero counts");
    cls.def_property_readonly("ncounts", & Class::ncounts, "Sum of all counts");
    cls.def("__repr__", & Class::toString);
    cls.def(py::pickle([](const Class& c) { // __getstate__
        return py::make_tuple(c.bin_info(), c.counts(), c.n_tot());
    }, [](py::tuple t) { // __setstate__
        // For backwards compatability, define a variable to
        // hold the data
        BSType binners;
        vec_counts_type counts;
        std::size_t n_tot;
        if (t.size() != 3) {
            // For backwards compatability for instances pickled
            // on previous version
            if (t.size() != 2 + N) {
                throw std::length_error(
                        "Invalid state of " + std::to_string(t.size())
                        + " elements");
            }
            for (std::size_t i = 0; i < N; i++) {
                binners[i] = t[i].cast<BinSpecifier>();
            }
            counts = t[N].cast<vec_counts_type>();
            n_tot = t[N + 1].cast<std::size_t>();
        }
        else {
            binners = t[0].cast<BSType>();
            counts = t[1].cast<vec_counts_type>();
            n_tot = t[2].cast<std::size_t>();
        }
        Class c(binners, counts, n_tot);
        return c;
    }));
    cls.def("update_binning",
            & Class::update_binning,
            "Update the binning in direction 'binner_index'",
            "binner_index"_a,
            "new_binner"_a,
            "prefer_old"_a = true);
    cls.def("update_norm",
            & Class::update_norm,
            "Update the normalized pair counts");
    cls.def_readonly_static("class_name", & Class::class_name);
    cls.def("reset", & Class::reset, "Reset the counting");
    return cls;
}

void process_separation(ExpectedNNCountsND<3>& self,
                        const py::array_t<double>& r_perp,
                        const py::array_t<double>& r_par,
                        const py::array_t<double>& zbar, bool new_real = false,
                        int num_threads = OMP_NUM_THREADS) {
    py::gil_scoped_acquire acquire;
    using Type = ExpectedNNCountsND<3>;
    auto uperp = r_perp.unchecked<1>();
    auto upar = r_par.unchecked<1>();
    auto uz = zbar.unchecked<1>();
    if (uperp.size() != upar.size() || uperp.size() != uz.size()) {
        throw std::length_error("r_perp, r_par, and zbar must have same length");
    }
    Type temp(self.bin_info(), self.n_tot());
    auto n = (std::size_t) uperp.size();
    omp_set_num_threads(num_threads);
#if _OPENMP
#pragma omp declare reduction(+ : Type : omp_out+=omp_in) initializer(omp_priv=omp_orig)
#pragma omp parallel for reduction(+ : temp)
#endif
    for (std::size_t i = 0; i < n; i++) {
        temp.process_separation(uperp(i), upar(i), uz(i));
    }
    if (new_real) { self.append_real(temp[0]); }
    else { self += temp; }
}

void process_separation(ExpectedNNCountsND<2>& self,
                        const py::array_t<double>& r_perp,
                        const py::array_t<double>& r_par, bool new_real = false,
                        int num_threads = OMP_NUM_THREADS) {
    py::gil_scoped_acquire acquire;
    using Type = ExpectedNNCountsND<2>;
    auto uperp = r_perp.unchecked<1>();
    auto upar = r_par.unchecked<1>();
    if (uperp.size() != upar.size()) {
        throw std::length_error("r_perp and r_par must have same length");
    }
    Type temp(self.bin_info(), self.n_tot());
    auto n = (std::size_t) uperp.size();
    omp_set_num_threads(num_threads);
#if _OPENMP
#pragma omp declare reduction(+ : Type : omp_out+=omp_in) initializer(omp_priv=omp_orig)
#pragma omp parallel for reduction(+ : temp)
#endif
    for (std::size_t i = 0; i < n; i++) {
        temp.process_separation(uperp(i), upar(i));
    }
    if (new_real) { self.append_real(temp[0]); }
    else { self += temp; }
}

void
process_separation(ExpectedNNCountsND<1>& self, const py::array_t<double>& r,
                   bool new_real = false, int num_threads = OMP_NUM_THREADS) {
    py::gil_scoped_acquire acquire;
    using Type = ExpectedNNCountsND<1>;
    auto ur = r.unchecked<1>();
    Type temp(self.bin_info(), self.n_tot());
    auto n = (std::size_t) ur.size();
    omp_set_num_threads(num_threads);
#if _OPENMP
#pragma omp declare reduction(+ : Type : omp_out+=omp_in) initializer(omp_priv=omp_orig)
#pragma omp parallel for reduction(+ : temp)
#endif
    for (std::size_t i = 0; i < n; i++) {
        temp.process_separation(ur(i));
    }
    if (new_real) { self.append_real(temp[0]); }
    else { self += temp; }
}

template<std::size_t N>
static py::class_<ExpectedNNCountsND<N>>
declareExpectedNNCountsND(py::module& mod) {
    using Base = ExpectedNNCountsNDBase<N>;
    using Class = ExpectedNNCountsND<N>;
    using BSType = std::array<BinSpecifier, N>;
    using NNType = NNCountsND<N>;
    py::class_<ExpectedNNCountsND<N>>
            cls(mod, ("ExpectedNNCounts" + std::to_string(N) + "D").c_str());
    cls.def(py::init<>(), "Empty constructor");
    cls.def(py::init<const Class&>(), "Copy constructor", "other"_a);
    cls.def(py::init<const BSType&, const std::vector<NNType>&, std::size_t,
                     std::size_t>(),
            "Constructor for pickling support",
            "bins"_a,
            "nn_list"_a,
            "n_real"_a,
            "n_tot"_a);
    cls.def(py::init<const BSType&, std::size_t>(),
            "Typical constructor from bin specification and total number of pairs",
            "bins"_a,
            "n_tot"_a);
    cls.def_property_readonly("bin_info",
                              & Class::bin_info,
                              "List of bin specifications");
    cls.def("get_1d_mean_indexer",
            & Class::get_1d_mean_indexer,
            "Get index for flattened mean array from a list providing the ND index",
            "index"_a);
    cls.def("get_1d_cov_indexer",
            & Class::get_1d_cov_indexer,
            "Get index for flattened covariance array from a list providing the (2*N)D index",
            "index"_a);
    cls.def("__getitem__", & Class::operator[], py::is_operator());
    cls.def("__getitem__",
            [](const Class& self, const py::array_t<std::size_t>& idx) {
                std::vector<NNType> output;
                auto uidx = idx.unchecked<1>();
                output.reserve(uidx.size());
                for (ssize_t i = 0; i < uidx.size(); i++) {
                    output.push_back(self.operator[](uidx(i)));
                }
                return output;
            },
            py::is_operator());
    cls.def_property("ntot",
                     py::overload_cast<>(& Class::n_tot, py::const_),
                     py::overload_cast<std::size_t>(& Class::n_tot),
                     "Total number of pairs per realization");
    cls.def_property_readonly("nreal",
                              & Class::n_real,
                              "Number of completed realizations (this will be one less than the number of NNCountsND objects unless a mean or covariance was recently calculated)");
    cls.def_property_readonly("mean_size",
                              & Class::mean_size,
                              "Total number of elements in the mean array");
    cls.def_property_readonly("cov_size",
                              & Class::cov_size,
                              "Total number of elements in the covariance array");
    cls.def_property_readonly("nn_list",
                              & Class::nn_list,
                              "List of the individual NNCountsND objects. The ntot in each of these is updated to be the ntot of this instance");
    cls.def("update",
            & Class::update,
            "Update the calculations of the mean and covariance");
    cls.def_property_readonly("mean", [](const Class& self) {
        return mkarray_from_vec(self.mean(), self.mean_shape_vec());
    }, "The mean counts of the NNCountsND objects");
    cls.def_property_readonly("cov", [](const Class& self) {
        return mkarray_from_vec(self.cov(), self.cov_shape_vec());
    }, "The covariance of counts of the NNCountsND objects");
    cls.def("__repr__", [](const Class& self) { return self.toString(); });
    cls.def(py::self += py::self);
    //cls.def("__iadd__", py::overload_cast<const NNType&>(&Class::operator+=), py::is_operator());
    cls.def(py::self += int());
    cls.def(py::self += float());
    cls.def(py::self + int());
    cls.def(py::self + float());
    cls.def(int() + py::self);
    cls.def(float() + py::self);
    cls.def("__eq__", & Class::operator==, py::is_operator());
    cls.def("__neq__", & Class::operator!=, py::is_operator());
    cls.def("append_real",
            py::overload_cast<const NNType&>(& Class::append_real),
            ("Append a realization as a single NNCounts" + std::to_string(N)
             + "D object").c_str(),
            "other"_a);
    cls.def("append_real",
            py::overload_cast<const Base&>(& Class::append_real),
            ("Append realizations of another ExpectedNNCounts"
             + std::to_string(N) + "D object").c_str(),
            "other"_a);
    cls.def("append_real",
            [](Class& self, const std::vector<NNType>& other) {
                for (const auto& nn : other) { self.append_real(nn); }
            },
            ("Append a list of NNCounts" + std::to_string(N)
             + "D objects as individual realizations").c_str(),
            "realizations"_a);
    cls.def("append_real",
            [](Class& self, const std::vector<Class>& other) {
                for (const auto& nn : other) { self.append_real(nn); }
            },
            ("Append realizations from each ExpectedNNCounts"
             + std::to_string(N) + "D object in a list").c_str(),
            "realizations"_a);
    cls.def(py::pickle([](const Class& c) { // __getstate__
        return py::make_tuple(c.bin_info(), c.nn_list(), c.n_real(), c.n_tot());
    }, [](py::tuple t) { // __setstate__
        // For backwards compatability, define a variable to
        // hold the data
        BSType binners;
        std::vector<NNType> nn_list;
        std::size_t n_real, n_tot;
        if (t.size() != 4) {
            // For backwards compatability for instances pickled
            // on previous version
            if (t.size() != 3 + N) {
                throw std::length_error("Invalid state");
            }
            for (std::size_t i = 0; i < N; i++) {
                binners[i] = t[i].cast<BinSpecifier>();
            }
            nn_list = t[N].cast<std::vector<NNType>>();
            n_real = t[N + 1].cast<std::size_t>();
            n_tot = t[N + 2].cast<std::size_t>();
        }
        else {
            binners = t[0].cast<BSType>();
            nn_list = t[1].cast<std::vector<NNType>>();
            n_real = t[2].cast<std::size_t>();
            n_tot = t[3].cast<std::size_t>();
        }
        Class c(binners, nn_list, n_real, n_tot);
        return c;
    }));
    cls.def("update_binning",
            & Class::update_binning,
            "Update the binning in direction 'binner_index'",
            "new_binner"_a,
            "binner_index"_a,
            "prefer_old"_a = true);
    cls.def_readonly_static("class_name", & Class::class_name);
    cls.def("reset", & Class::reset, "Reset the counting");
    return cls;
}

template<std::size_t N>
py::class_<CorrFuncND<N>> declareCorrFuncND(py::module& mod) {
    using Class = CorrFuncND<N>;
    using BSType = std::array<BinSpecifier, N>;
    using NNType = NNCountsND<N>;
    py::class_<CorrFuncND<N>> cls(mod,
                                  Class::class_name.c_str(),
                                  ("Correlation function in "
                                   + std::to_string(N) + " dimensions")
                                          .c_str());
    cls.def(py::init<>(), "Empty constructor");
    cls.def(py::init<const Class&>(),
            "Copy constructor: create a copy of 'other'",
            "other"_a);
    cls.def(py::init<const BSType&>(), "Initialize from binning", "binners"_a);
    cls.def(py::init<const NNType&>(), "Initialize with only DD", "dd"_a);
    cls.def(py::init<const BSType&, const NNType&>(),
            "Initialize binning and DD",
            "binners"_a,
            "dd"_a);
    cls.def(py::init<const NNType&, const NNType&>(),
            "Initialize with DD and RR",
            "dd"_a,
            "rr"_a);
    cls.def(py::init<const BSType&, const NNType&, const NNType&>(),
            "Initialziae with binning, DD, and RR",
            "binners"_a,
            "dd"_a,
            "rr"_a);
    cls.def(py::init<const NNType&, const NNType&, const NNType&>(),
            "Initialize with DD, RR, and DR",
            "dd"_a,
            "dr"_a,
            "rr"_a);
    cls.def(py::init<const BSType&, const NNType&, const NNType&,
                     const NNType&>(),
            "Initialize with binning, DD, RR, and DR",
            "binners"_a,
            "dd"_a,
            "rr"_a,
            "dr"_a);
    cls.def(py::init<const NNType&, const NNType&, const NNType&,
                     const NNType&>(),
            "Initialize with DD, RR, DR, and RD",
            "dd"_a,
            "rr"_a,
            "dr"_a,
            "rd"_a);
    cls.def(py::init<const BSType&, const NNType&, const NNType&, const NNType&,
                     const NNType&>(),
            "Initialize with binning, DD, RR, DR, and RD",
            "binners"_a,
            "dd"_a,
            "rr"_a,
            "dr"_a,
            "rd"_a);
    cls.def_readonly_static("class_name", & Class::class_name);
    cls.def_property_readonly("bin_info",
                              & Class::bin_info,
                              "Binning specifications");
    cls.def_property_readonly("size",
                              & Class::size,
                              "Flattened size of correlation function");
    cls.def_property_readonly("shape_vec",
                              & Class::shape_vec,
                              "Shape of correlation function as a list");
    cls.def_property("dd",
                     py::overload_cast<>(& Class::dd, py::const_),
                     py::overload_cast<const NNType&>(& Class::dd),
                     "Data-data pair counts holder");
    cls.def_property("rr",
                     py::overload_cast<>(& Class::rr, py::const_),
                     py::overload_cast<const NNType&>(& Class::rr),
                     "Random-random pair counts holder");
    cls.def_property("dr",
                     py::overload_cast<>(& Class::dr, py::const_),
                     py::overload_cast<const NNType&>(& Class::dr),
                     "Data-random pair counts holder");
    cls.def_property("rd",
                     py::overload_cast<>(& Class::rd, py::const_),
                     py::overload_cast<const NNType&>(& Class::rd),
                     "Random-data pair counts holder");
    cls.def("update_binning",
            & Class::update_binning,
            "Update the binning in the specified dimension",
            "dim"_a,
            "new_binning"_a,
            "prefer_old"_a = true);
    cls.def("calculate_xi",
            [](const Class& self,
               CFEstimator estimator = CFEstimator::Landy_Szalay) {
                return mkarray_from_vec(self.calculate_xi(estimator),
                                        self.shape_vec());
            },
            ("Calculate the correlation function in " + std::to_string(N) + "D")
                    .c_str(),
            py::arg_v("estimator", CFEstimator::Landy_Szalay, "LS estimator"));
    cls.def("__repr__", & Class::toString);
    cls.def(py::pickle([](const Class& c) { // __getstate__
        return py::make_tuple(c.bin_info(), c.dd(), c.rr(), c.dr(), c.rd());
    }, [](py::tuple t) { // __setstate__
        if (t.size() != 5) { throw std::length_error("Invalid state"); }
        return Class(t[0].cast<BSType>(),
                     t[1].cast<NNType>(),
                     t[2].cast<NNType>(),
                     t[3].cast<NNType>(),
                     t[4].cast<NNType>());
    }));
    return cls;
}

template<std::size_t N>
py::class_<ExpectedCorrFuncND<N>> declareExpectedCorrFuncND(py::module& mod) {
    using Class = ExpectedCorrFuncND<N>;
    using BSType = std::array<BinSpecifier, N>;
    using ENNType = ExpectedNNCountsND<N>;
    py::class_<ExpectedCorrFuncND<N>> cls(mod,
                                          Class::class_name.c_str(),
                                          ("Expected correlation function in "
                                           + std::to_string(N) + " dimensions")
                                                  .c_str());
    cls.def(py::init<>(), "Empty constructor");
    cls.def(py::init<const Class&>(), "Copy constructor", "other"_a);
    cls.def(py::init<const BSType&>(), "Initialize from binning", "binners"_a);
    cls.def(py::init<const ENNType&>(),
            "Initialize from DD pair counts",
            "dd"_a);
    cls.def(py::init<const ENNType&, const ENNType&>(),
            "Initialize from DD and RR pair counts",
            "dd"_a,
            "rr"_a);
    cls.def(py::init<const ENNType&, const ENNType&, const ENNType&>(),
            "Initialize from DD, RR, and DR pair counts",
            "dd"_a,
            "rr"_a,
            "dr"_a);
    cls.def(py::init<const ENNType&, const ENNType&, const ENNType&,
                     const ENNType&>(),
            "Initialize from DD, RR, DR, and RD pair counts",
            "dd"_a,
            "rr"_a,
            "dr"_a,
            "rd"_a);
    cls.def_readonly_static("class_name", & Class::class_name);
    cls.def_property_readonly("n_real",
                              & Class::n_real,
                              "Number of realizations");
    cls.def_property_readonly("mean_size",
                              & Class::mean_size,
                              "Total number of elements for the mean correlation function");
    cls.def_property_readonly("mean_shape_vec",
                              & Class::mean_shape,
                              "Shape of the mean of the correlation function as a list");
    cls.def_property_readonly("cov_size",
                              & Class::cov_size,
                              "Total number of elements in the covariance matrix");
    cls.def_property_readonly("cov_shape_vec",
                              & Class::cov_shape,
                              "Shape of the covariance matrix as a list");
    cls.def_property_readonly("bin_info",
                              & Class::bin_info,
                              "Binning information");
    cls.def("update_binning",
            & Class::update_binning,
            "Update binning in a given dimension",
            "index"_a,
            "new_binner"_a,
            "prefer_old"_a = true);
    cls.def_property("dd",
                     py::overload_cast<>(& Class::dd, py::const_),
                     py::overload_cast<const ENNType&>(& Class::dd),
                     "DD pair counts");
    cls.def_property("rr",
                     py::overload_cast<>(& Class::rr, py::const_),
                     py::overload_cast<const ENNType&>(& Class::rr),
                     "RR pair counts");
    cls.def_property("dr",
                     py::overload_cast<>(& Class::dr, py::const_),
                     py::overload_cast<const ENNType&>(& Class::dr),
                     "DR pair counts");
    cls.def_property("rd",
                     py::overload_cast<>(& Class::rd, py::const_),
                     py::overload_cast<const ENNType&>(& Class::rd),
                     "RD pair counts");
    cls.def("calculate_xi",
            [](const Class& self,
               CFEstimator estimator = CFEstimator::Landy_Szalay) {
                return mkarray_from_vec(self.calculate_xi(estimator),
                                        self.mean_shape());
            },
            "Get the mean correlation function",
            py::return_value_policy::take_ownership,
            py::arg_v("estimator", CFEstimator::Landy_Szalay, "LS estimator"));
    cls.def("calculate_xi_cov",
            [](const Class& self,
               CFEstimator estimator = CFEstimator::Landy_Szalay) {
                return mkarray_from_vec(self.calculate_xi_cov(estimator),
                                        self.cov_shape());
            },
            "Get the covariance matrix of the correlation function",
            py::return_value_policy::take_ownership,
            py::arg_v("estimator", CFEstimator::Landy_Szalay, "LS estimator"));
    cls.def("__repr__", & Class::toString);
    cls.def(py::pickle([](const Class& self) { // __getstate__
        return py::make_tuple(self.dd(), self.rr(), self.dr(), self.rd());
    }, [](py::tuple t) { // __setstate__
        if (t.size() != 4) { throw std::length_error("Invalid state"); }
        return Class(t[0].cast<ENNType>(),
                     t[1].cast<ENNType>(),
                     t[2].cast<ENNType>(),
                     t[3].cast<ENNType>());
    }));
    return cls;
}

static void declare3D(py::module& mod) {
    using NNClass = NNCountsND<3>;
    auto nncls = declareNNCountsND<3>(mod);
    using ENNClass = ExpectedNNCountsND<3>;
    auto enncls = declareExpectedNNCountsND<3>(mod);
    nncls.def(py::init<const BinSpecifier&, const BinSpecifier&,
                       const BinSpecifier&>(),
              "From three BinSpecifier objects",
              "rperp_bins"_a,
              "rpar_bins"_a,
              "zbar_bins"_a);
    nncls.def_property("rperp_bins",
                       py::overload_cast<>(& NNClass::rperp_bins, py::const_),
                       [](NNClass& self, py::object obj) {
                           try {
                               self.rperp_bins(obj.cast<BinSpecifier>());
                           }
                           catch (const py::cast_error&) {
                               auto t = py::reinterpret_steal<py::tuple>(obj);
                               if (t.size() < 1 || t.size() > 2) {
                                   throw std::length_error("Invalid tuple size "
                                                           + std::to_string(t.size())
                                                           + " for rperp_bins: tuple should have 1 or 2 elements");
                               }
                               if (t.size() == 1) {
                                   self.rperp_bins(t[0].cast<BinSpecifier>());
                               }
                               else {
                                   self.rperp_bins(t[0].cast<BinSpecifier>(),
                                                   t[1].cast<bool>());
                               }
                           }
                       },
                       "Perpendicular separation binning. The setter can take either a BinSpecifier object, or a tuple(BinSpecifier, bool). The boolean parameter specifies that values existing in both the original and new binning should default to the original value (if true, default) or not");
    nncls.def_property("rpar_bins",
                       py::overload_cast<>(& NNClass::rpar_bins, py::const_),
                       [](NNClass& self, py::object obj) {
                           try {
                               self.rpar_bins(obj.cast<BinSpecifier>());
                           }
                           catch (const py::cast_error&) {
                               auto t = py::reinterpret_steal<py::tuple>(obj);
                               if (t.size() < 1 || t.size() > 2) {
                                   throw std::length_error("Invalid tuple size "
                                                           + std::to_string(t.size())
                                                           + " for rpar_bins: tuple should have 1 or 2 elements");
                               }
                               if (t.size() == 1) {
                                   self.rpar_bins(t[0].cast<BinSpecifier>());
                               }
                               else {
                                   self.rpar_bins(t[0].cast<BinSpecifier>(),
                                                  t[1].cast<bool>());
                               }
                           }
                       },
                       "Parallel separation binning. The setter can take either a BinSpecifier object, or a tuple(BinSpecifier, bool). The boolean parameter specifies that values existing in both the original and new binning should default to the original value (if true, default) or not");
    nncls.def_property("zbar_bins",
                       py::overload_cast<>(& NNClass::zbar_bins, py::const_),
                       [](NNClass& self, py::object obj) {
                           try {
                               self.zbar_bins(obj.cast<BinSpecifier>());
                           }
                           catch (const py::cast_error&) {
                               auto t = py::reinterpret_steal<py::tuple>(obj);
                               if (t.size() < 1 || t.size() > 2) {
                                   throw std::length_error("Invalid tuple size "
                                                           + std::to_string(t.size())
                                                           + " for zbar_bins: tuple should have 1 or 2 elements");
                               }
                               if (t.size() == 1) {
                                   self.zbar_bins(t[0].cast<BinSpecifier>());
                               }
                               else {
                                   self.zbar_bins(t[0].cast<BinSpecifier>(),
                                                  t[1].cast<bool>());
                               }
                           }
                       },
                       "Average redshift binning. The setter can take either a BinSpecifier object, or a tuple(BinSpecifier, bool). The boolean parameter specifies that values existing in both the original and new binning should default to the original value (if true, default) or not");
    nncls.def("get_1d_index",
              py::vectorize(py::overload_cast<int, int,
                                              int>(& NNClass::get_1d_indexer,
                                                   py::const_)),
              "Get 1D index/indices from the indices given individually",
              "rp_bin"_a,
              "rl_bin"_a,
              "zb_bin"_a);
    nncls.def_property_readonly("shape",
                                & NNClass::shape,
                                "Shape of data array");

    enncls.def(py::init<const BinSpecifier&, const BinSpecifier&,
                        const BinSpecifier&, std::size_t>(),
               "Initialize with individual BinSpecifier objects",
               "rperp_binning"_a,
               "rpar_binning"_a,
               "zbar_binning"_a,
               "ntot"_a);
    enncls.def_property("rperp_bins",
                        py::overload_cast<>(& ENNClass::rperp_bins, py::const_),
                        py::overload_cast<const BinSpecifier&,
                                          bool>(& ENNClass::rperp_bins),
                        "Perpendicular separation binning");
    enncls.def_property("rpar_bins",
                        py::overload_cast<>(& ENNClass::rpar_bins, py::const_),
                        py::overload_cast<const BinSpecifier&,
                                          bool>(& ENNClass::rpar_bins),
                        "Parallel separation binning");
    enncls.def_property("zbar_bins",
                        py::overload_cast<>(& ENNClass::zbar_bins, py::const_),
                        py::overload_cast<const BinSpecifier&,
                                          bool>(& ENNClass::zbar_bins),
                        "Average redshift binning");
    enncls.def("get_1d_mean_indexer",
               py::vectorize(& ENNClass::get_1d_mean_indexer_from_args),
               "Get the index for the flattened mean array correpsonding to the 3D indices given",
               "rp_bin"_a,
               "rl_bin"_a,
               "zb_bin"_a);
    enncls.def("get_1d_cov_indexer",
               py::vectorize(& ENNClass::get_1d_cov_indexer_from_args),
               "Get the index for the flattened covariance array corresponding to the 6D indices given",
               "rpi_bin"_a,
               "rli_bin"_a,
               "zbi_bin"_a,
               "rpj_bin"_a,
               "rlj_bin"_a,
               "zbj_bin"_a);
    enncls.def("process_separation",
               py::overload_cast<double, double, double,
                                 bool>(& ENNClass::process_separation),
               "Process the given 3D separation",
               "r_perp"_a,
               "r_par"_a,
               "zbar"_a,
               "new_real"_a = false);
    enncls.def("process_separation",
               [](ENNClass& self, const py::array_t<double>& rp,
                  const py::array_t<double> rl, const py::array_t<double> zb,
                  bool new_real = false, int num_threads = OMP_NUM_THREADS) {
                   return process_separation(self,
                                             rp,
                                             rl,
                                             zb,
                                             new_real,
                                             num_threads);
               },
               "Process the given sets of 3D separations",
               "r_perp"_a,
               "r_par"_a,
               "zbar"_a,
               "new_real"_a = false,
               "num_threads"_a = OMP_NUM_THREADS);
    enncls.def_property_readonly("mean_shape",
                                 & ENNClass::mean_shape,
                                 "Shape tuple for the mean counts");
    enncls.def_property_readonly("cov_shape",
                                 & ENNClass::cov_shape,
                                 "Shape tuple for the covariance of the counts");
}

static void declare2D(py::module& mod) {
    using NNClass = NNCountsND<2>;
    auto nncls = declareNNCountsND<2>(mod);
    nncls.def(py::init<const BinSpecifier&, const BinSpecifier&>(),
              "From 2 BinSpecifier objects",
              "rperp_bins"_a,
              "rpar_bins"_a);
    nncls.def_property("rperp_bins",
                       py::overload_cast<>(& NNClass::rperp_bins, py::const_),
                       [](NNClass& self, py::object obj) {
                           try {
                               self.rperp_bins(obj.cast<BinSpecifier>());
                           }
                           catch (const py::cast_error&) {
                               auto t = py::reinterpret_steal<py::tuple>(obj);
                               if (t.size() < 1 || t.size() > 2) {
                                   throw std::length_error("Invalid tuple size "
                                                           + std::to_string(t.size())
                                                           + " for rperp_bins: tuple should have 1 or 2 elements");
                               }
                               if (t.size() == 1) {
                                   self.rperp_bins(t[0].cast<BinSpecifier>());
                               }
                               else {
                                   self.rperp_bins(t[0].cast<BinSpecifier>(),
                                                   t[1].cast<bool>());
                               }
                           }
                       },
                       "Perpendicular separation binning. The setter can take either a BinSpecifier object, or a tuple(BinSpecifier, bool). The boolean parameter specifies that values existing in both the original and new binning should default to the original value (if true, default) or not");
    nncls.def_property("rpar_bins",
                       py::overload_cast<>(& NNClass::rpar_bins, py::const_),
                       [](NNClass& self, py::object obj) {
                           try {
                               self.rpar_bins(obj.cast<BinSpecifier>());
                           }
                           catch (const py::cast_error&) {
                               auto t = py::reinterpret_steal<py::tuple>(obj);
                               if (t.size() < 1 || t.size() > 2) {
                                   throw std::length_error("Invalid tuple size "
                                                           + std::to_string(t.size())
                                                           + " for rpar_bins: tuple should have 1 or 2 elements");
                               }
                               if (t.size() == 1) {
                                   self.rpar_bins(t[0].cast<BinSpecifier>());
                               }
                               else {
                                   self.rpar_bins(t[0].cast<BinSpecifier>(),
                                                  t[1].cast<bool>());
                               }
                           }
                       },
                       "Parallel separation binning. The setter can take either a BinSpecifier object, or a tuple(BinSpecifier, bool). The boolean parameter specifies that values existing in both the original and new binning should default to the original value (if true, default) or not");
    nncls.def("get_1d_index",
              py::vectorize(py::overload_cast<int,
                                              int>(& NNClass::get_1d_indexer,
                                                   py::const_)),
              "Get 1D index/indices from the indices given individually",
              "rp_bin"_a,
              "rl_bin"_a);
    nncls.def_property_readonly("shape",
                                & NNClass::shape,
                                "Shape of data array");

    using ENNClass = ExpectedNNCountsND<2>;
    auto enncls = declareExpectedNNCountsND<2>(mod);
    enncls.def(py::init<const BinSpecifier&, const BinSpecifier&,
                        std::size_t>(),
               "Initialize with individual BinSpecifier objects",
               "rperp_binning"_a,
               "rpar_binning"_a,
               "ntot"_a);
    enncls.def_property("rperp_bins",
                        py::overload_cast<>(& ENNClass::rperp_bins, py::const_),
                        py::overload_cast<const BinSpecifier&,
                                          bool>(& ENNClass::rperp_bins),
                        "Perpendicular separation binning");
    enncls.def_property("rpar_bins",
                        py::overload_cast<>(& ENNClass::rpar_bins, py::const_),
                        py::overload_cast<const BinSpecifier&,
                                          bool>(& ENNClass::rpar_bins),
                        "Parallel separation binning");
    enncls.def("get_1d_mean_indexer",
               py::vectorize(& ENNClass::get_1d_mean_indexer_from_args),
               "Get the index for the flattened mean array correpsonding to the 2D indices given",
               "rp_bin"_a,
               "rl_bin"_a);
    enncls.def("get_1d_cov_indexer",
               py::vectorize(& ENNClass::get_1d_cov_indexer_from_args),
               "Get the index for the flattened covariance array corresponding to the 4D indices given",
               "rpi_bin"_a,
               "rli_bin"_a,
               "rpj_bin"_a,
               "rlj_bin"_a);
    enncls.def("process_separation",
               py::overload_cast<double, double,
                                 bool>(& ENNClass::process_separation),
               "Process the given 2D separation",
               "r_perp"_a,
               "r_par"_a,
               "new_real"_a = false);
    enncls.def("process_separation",
               [](ENNClass& self, const py::array_t<double>& rp,
                  const py::array_t<double> rl, bool new_real = false,
                  int num_threads = OMP_NUM_THREADS) {
                   return process_separation(self,
                                             rp,
                                             rl,
                                             new_real,
                                             num_threads);
               },
               "Process the given sets of 2D separations",
               "r_perp"_a,
               "r_par"_a,
               "new_real"_a = false,
               "num_threads"_a = OMP_NUM_THREADS);
    enncls.def_property_readonly("mean_shape",
                                 & ENNClass::mean_shape,
                                 "Shape tuple for the mean counts");
    enncls.def_property_readonly("cov_shape",
                                 & ENNClass::cov_shape,
                                 "Shape tuple for the covariance of the counts");

    using CFClass = CorrFuncND<2>;
    using NNType = NNCountsND<2>;
    auto cfcls = declareCorrFuncND<2>(mod);
    cfcls.def(py::init<const BinSpecifier&, const BinSpecifier&>(),
              "Initialize from perpendicular and parallel binners",
              "perp_binner"_a,
              "par_binner"_a);
    cfcls.def(py::init<const BinSpecifier&, const BinSpecifier&,
                       const NNType&>(),
              "Initialize with individual binners and DD",
              "perp_binner"_a,
              "par_binner"_a,
              "dd"_a);
    cfcls.def(py::init<const BinSpecifier&, const BinSpecifier&, const NNType&,
                       const NNType&>(),
              "Initialziae from individual binners, DD, and RR",
              "perp_binner"_a,
              "par_binner"_a,
              "dd"_a,
              "rr"_a);
    cfcls.def(py::init<const BinSpecifier&, const BinSpecifier&, const NNType&,
                       const NNType&, const NNType&>(),
              "Initialize from individual binners, DD, RR, and DR",
              "perp_binner"_a,
              "par_binner"_a,
              "dd"_a,
              "rr"_a,
              "dr"_a);
    cfcls.def(py::init<const BinSpecifier&, const BinSpecifier&, const NNType&,
                       const NNType&, const NNType&, const NNType&>(),
              "Initialize from individual binners, DD, RR, DR, and RD",
              "perp_binner"_a,
              "par_binner"_a,
              "dd"_a,
              "rr"_a,
              "dr"_a,
              "rd"_a);
    cfcls.def_property("rperp_bins",
                       py::overload_cast<>(& CFClass::rperp_bins, py::const_),
                       [](CFClass& self, const py::tuple& t) {
                           self.rperp_bins(t[0].cast<BinSpecifier>(),
                                           t[1].cast<bool>());
                       },
                       "Perpendicular binning. Second element in tuple for setter should be boolean for whether original values should be preferred or not");
    cfcls.def_property("rpar_bins",
                       py::overload_cast<>(& CFClass::rpar_bins, py::const_),
                       [](CFClass& self, const py::tuple& t) {
                           self.rpar_bins(t[0].cast<BinSpecifier>(),
                                          t[1].cast<bool>());
                       },
                       "Parallel binning. Second element in tuple for setter should be boolean for whether original values should be preferred or not");
    cfcls.def_property_readonly("shape", & CFClass::shape);

    using ECFClass = ExpectedCorrFuncND<2>;
    auto ecfcls = declareExpectedCorrFuncND<2>(mod);
    ecfcls.def(py::init([](const BinSpecifier& a, const BinSpecifier& b) {
                   return ECFClass(arrays::make_array(a, b));
               }),
               "Initialize from perpendicular and parallel binners",
               "perp_binner"_a,
               "par_binner"_a);
    ecfcls.def_property("rperp_bins",
                        [](const ECFClass& self) { return self.bin_info()[0]; },
                        [](ECFClass& self, const py::tuple& t) {
                            self.update_binning(t[0].cast<BinSpecifier>(),
                                                0,
                                                t[1].cast<bool>());
                        },
                        "Perpendicular binning. Second element in tuple for setter should be boolean for whether original values should be preferred or not");
    ecfcls.def_property("rpar_bins",
                        [](const ECFClass& self) { return self.bin_info()[1]; },
                        [](CFClass& self, const py::tuple& t) {
                            self.update_binning(t[0].cast<BinSpecifier>(),
                                                1,
                                                t[1].cast<bool>());
                        },
                        "Parallel binning. Second element in tuple for setter should be boolean for whether original values should be preferred or not");
    ecfcls.def_property_readonly("mean_shape", [](const ECFClass& self) {
        auto shape_vec = self.mean_shape();
        return py::make_tuple(shape_vec.at(0), shape_vec.at(1));
    });
    ecfcls.def_property_readonly("cov_shape", [](const ECFClass& self) {
        auto shape_vec = self.cov_shape();
        return py::make_tuple(shape_vec.at(0),
                              shape_vec.at(1),
                              shape_vec.at(2),
                              shape_vec.at(3));
    });
}

static void declare1D(py::module& mod) {
    using NNClass = NNCountsND<1>;
    auto nncls = declareNNCountsND<1>(mod);
    nncls.def(py::init<const BinSpecifier&>(),
              "From single BinSpecifier",
              "r_bins"_a);
    nncls.def_property("r_bins",
                       py::overload_cast<>(& NNClass::r_bins, py::const_),
                       [](NNClass& self, py::object obj) {
                           try {
                               self.r_bins(obj.cast<BinSpecifier>());
                           }
                           catch (const py::cast_error&) {
                               auto t = py::reinterpret_steal<py::tuple>(obj);
                               if (t.size() < 1 || t.size() > 2) {
                                   throw std::length_error("Invalid tuple size "
                                                           + std::to_string(t.size())
                                                           + " for r_bins: tuple should have 1 or 2 elements");
                               }
                               if (t.size() == 1) {
                                   self.r_bins(t[0].cast<BinSpecifier>());
                               }
                               else {
                                   self.r_bins(t[0].cast<BinSpecifier>(),
                                               t[1].cast<bool>());
                               }
                           }
                       },
                       "Separation magnitude binning. The setter can take either a BinSpecifier object, or a tuple(BinSpecifier, bool). The boolean parameter specifies that values existing in both the original and new binning should default to the original value (if true, default) or not");
    nncls.def("get_1d_index",
              py::vectorize(py::overload_cast<int>(& NNClass::get_1d_indexer,
                                                   py::const_)),
              "Get 1D index/indices from the indices given individually. Note that for this NNClass, this function is pretty pointless",
              "r_bin"_a);
    nncls.def_property_readonly("shape",
                                & NNClass::shape,
                                "Shape of data array");

    using ENNClass = ExpectedNNCountsND<1>;
    auto enncls = declareExpectedNNCountsND<1>(mod);
    enncls.def(py::init<const BinSpecifier&, std::size_t>(),
               "Initialize with individual BinSpecifier objects",
               "r_binning"_a,
               "ntot"_a);
    enncls.def_property("r_bins",
                        py::overload_cast<>(& ENNClass::r_bins, py::const_),
                        py::overload_cast<const BinSpecifier&,
                                          bool>(& ENNClass::r_bins),
                        "Separation binning");
    enncls.def("get_1d_mean_indexer",
               py::vectorize(& ENNClass::get_1d_mean_indexer_from_args),
               "Get the index for the flattened mean array correpsonding to the 1D indices given",
               "r_bin"_a);
    enncls.def("get_1d_cov_indexer",
               py::vectorize(& ENNClass::get_1d_cov_indexer_from_args),
               "Get the index for the flattened covariance array corresponding to the 2D indices given",
               "ri_bin"_a,
               "rj_bin"_a);
    enncls.def("process_separation",
               py::overload_cast<double, bool>(& ENNClass::process_separation),
               "Process the given 1D separation",
               "r"_a,
               "new_real"_a = false);
    enncls.def("process_separation",
               [](ENNClass& self, const py::array_t<double>& r,
                  bool new_real = false, int num_threads = OMP_NUM_THREADS) {
                   return process_separation(self, r, new_real, num_threads);
               },
               "Process the given sets of 1D separations",
               "r"_a,
               "new_real"_a = false,
               "num_threads"_a = OMP_NUM_THREADS);
    enncls.def_property_readonly("mean_shape",
                                 & ENNClass::mean_shape,
                                 "Shape tuple for the mean counts");
    enncls.def_property_readonly("cov_shape",
                                 & ENNClass::cov_shape,
                                 "Shape tuple for the covariance of the counts");

    using CFClass = CorrFuncND<1>;
    using NNType = NNCountsND<1>;
    auto cfcls = declareCorrFuncND<1>(mod);
    cfcls.def(py::init<const BinSpecifier&>(),
              "Initialize from binner object",
              "binner"_a);
    cfcls.def(py::init<const BinSpecifier&, const NNType&>(),
              "Initialize with individual binner and DD",
              "binner"_a,
              "dd"_a);
    cfcls.def(py::init<const BinSpecifier&, const NNType&, const NNType&>(),
              "Initialziae from individual binner, DD, and RR",
              "binner"_a,
              "dd"_a,
              "rr"_a);
    cfcls.def(py::init<const BinSpecifier&, const NNType&, const NNType&,
                       const NNType&>(),
              "Initialize from individual binner, DD, RR, and DR",
              "binner"_a,
              "dd"_a,
              "rr"_a,
              "dr"_a);
    cfcls.def(py::init<const BinSpecifier&, const NNType&, const NNType&,
                       const NNType&, const NNType&>(),
              "Initialize from individual binner, DD, RR, DR, and RD",
              "binner"_a,
              "dd"_a,
              "rr"_a,
              "dr"_a,
              "rd"_a);
    cfcls.def_property("r_bins",
                       py::overload_cast<>(& CFClass::r_bins, py::const_),
                       [](CFClass& self, const py::tuple& t) {
                           self.r_bins(t[0].cast<BinSpecifier>(),
                                       t[1].cast<bool>());
                       },
                       "Separation binning. Second element in tuple for setter should be boolean for whether original values should be preferred or not");
    cfcls.def_property_readonly("shape", & CFClass::shape);

    using ECFClass = ExpectedCorrFuncND<1>;
    auto ecfcls = declareExpectedCorrFuncND<1>(mod);
    ecfcls.def(py::init([](const BinSpecifier& a) {
        return ECFClass(arrays::make_array(a));
    }), "Initialize from perpendicular and parallel binners", "binner"_a);
    ecfcls.def_property("r_bins",
                        [](const ECFClass& self) { return self.bin_info()[0]; },
                        [](ECFClass& self, const py::tuple& t) {
                            self.update_binning(t[0].cast<BinSpecifier>(),
                                                0,
                                                t[1].cast<bool>());
                        },
                        "Separation magnitude binning. Second element in tuple for setter should be boolean for whether original values should be preferred or not");
    ecfcls.def_property_readonly("mean_shape", [](const ECFClass& self) {
        return py::make_tuple(self.mean_size());
    });
    ecfcls.def_property_readonly("cov_shape", [](const ECFClass& self) {
        auto shape_vec = self.cov_shape();
        return py::make_tuple(shape_vec.at(0), shape_vec.at(1));
    });
}

PYBIND11_MODULE(calculate_distances, m) {
    py::add_ostream_redirect(m);
    PYBIND11_NUMPY_DTYPE_EX(SepDType,
                            r_perp,
                            "R_PERP",
                            r_par,
                            "R_PAR",
                            zbar,
                            "AVE_Z",
                            id1,
                            "ID1",
                            id2,
                            "ID2");
    PYBIND11_NUMPY_DTYPE_EX(TOSepDType,
                            r_perp_t,
                            "R_PERP_T",
                            r_par_t,
                            "R_PAR_T",
                            zbar_t,
                            "AVE_Z_TRUE",
                            r_perp_o,
                            "R_PERP_O",
                            r_par_o,
                            "R_PAR_O",
                            zbar_o,
                            "AVE_Z_OBS",
                            id1,
                            "ID1",
                            id2,
                            "ID2");
    PYBIND11_NUMPY_DTYPE(PosCatalog, RA, DEC, D_TRUE, D_OBS, Z_TRUE, Z_OBS);
    PYBIND11_NUMPY_DTYPE(SPosCatalog, RA, DEC, D, Z);
    py::enum_<CFEstimator> cf_enum(m,
                                   "CFEstimator",
                                   py::arithmetic(),
                                   "Correlation function estimator type");
    cf_enum.value("LS",
                  CFEstimator::Landy_Szalay,
                  "The Landy-Szalay estimator, (DD - 2 DR + RR) / RR");
    cf_enum.value("Hamilton",
                  CFEstimator::Hamilton,
                  "The Hamilton estimator, (DD RR / DR^2) - 1");
    cf_enum.value("Peebles",
                  CFEstimator::Peebles,
                  "The Peebles estimator, DD / DR - 1");
    cf_enum.value("Dodelson",
                  CFEstimator::Dodelson,
                  "The estimator from Dodelson et al. 2018, (DD - DR + RR) / DD");
    declareBinSpecifier(m);
    declareSPos(m);
    declarePos(m);
    m.def("convert_catalog",
          py::overload_cast<const py::array_t<SPosCatalog>&>(& convert_catalog),
          py::return_value_policy::take_ownership,
          "Convert structured array catalog to SPos objects",
          "cat"_a);
    m.def("convert_catalog",
          py::overload_cast<const py::array_t<PosCatalog>&>(& convert_catalog),
          py::return_value_policy::take_ownership,
          "Convert structured array catalog to Pos objects",
          "cat"_a);
    m.def("get_auto_separations",
          py::overload_cast<const py::array_t<SPosCatalog>&,
                            const BinSpecifier&, const BinSpecifier&,
                            int>(& get_auto_separations_arr),
          py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>(),
          "Get auto-correlation separations limited in the perpendicular and parallel directions by the binners",
          "cat"_a,
          "perp_binner"_a,
          "par_binner"_a,
          "num_threads"_a = OMP_NUM_THREADS);
    m.def("get_auto_separations",
          py::overload_cast<const py::array_t<PosCatalog>&, const BinSpecifier&,
                            const BinSpecifier&, bool,
                            int>(& get_auto_separations_arr),
          py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>(),
          "Get auto-correlation separations limited in the perpendicular and parallel directions by the binners, specifying to use true or observed separations",
          "cat"_a,
          "perp_binner"_a,
          "par_binner"_a,
          "use_true"_a,
          "num_threads"_a = OMP_NUM_THREADS);
    m.def("get_cross_separations",
          py::overload_cast<const py::array_t<SPosCatalog>&,
                            const py::array_t<SPosCatalog>&,
                            const BinSpecifier&, const BinSpecifier&,
                            int>(& get_cross_separations_arr),
          py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>(),
          "Get cross-correlation separations limited in the perpendicular and parallel directions by the binners",
          "cat1"_a,
          "cat2"_a,
          "perp_binner"_a,
          "par_binner"_a,
          "num_threads"_a = OMP_NUM_THREADS);
    m.def("get_cross_separations",
          py::overload_cast<const py::array_t<PosCatalog>&,
                            const py::array_t<PosCatalog>&, const BinSpecifier&,
                            const BinSpecifier&, bool,
                            int>(& get_cross_separations_arr),
          py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>(),
          "Get cross-correlation separations limited in the perpendicular and parallel directions by the binners, specifying to use true or observed separations",
          "cat1"_a,
          "cat2"_a,
          "perp_binner"_a,
          "par_binner"_a,
          "use_true"_a,
          "num_threads"_a = OMP_NUM_THREADS);
    m.def("get_separations",
          py::overload_cast<const py::array_t<SPosCatalog>&,
                            const py::array_t<SPosCatalog>&,
                            const BinSpecifier&, const BinSpecifier&, bool,
                            int>(& get_separations_arr),
          py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>(),
          "Get the separations limited in the perpendiculare and parallel directions by the binners, specifying whether this is an auto or cross correlation",
          "cat1"_a,
          "cat2"_a,
          "perp_binner"_a,
          "par_binner"_a,
          "is_auto"_a,
          "num_threads"_a = OMP_NUM_THREADS);
    m.def("get_separations",
          py::overload_cast<const py::array_t<PosCatalog>&,
                            const py::array_t<PosCatalog>&, const BinSpecifier&,
                            const BinSpecifier&, bool, bool,
                            int>(& get_separations_arr),
          py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>(),
          "Get the separations limited in the perpendiculare and parallel directions by the binners, specifying whether this is an auto or cross correlation and whether to use true or observed separations",
          "cat1"_a,
          "cat2"_a,
          "perp_binner"_a,
          "par_binner"_a,
          "use_true"_a,
          "is_auto"_a,
          "num_threads"_a = OMP_NUM_THREADS);
    m.def("get_auto_separations",
          py::overload_cast<const py::array_t<PosCatalog>&, const BinSpecifier&,
                            const BinSpecifier&,
                            int>(& get_auto_separations_arr),
          py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>(),
          "Get auto-correlation separations limited in the observed perpendicular and parallel directions by the binners",
          "cat"_a,
          "perp_binner"_a,
          "par_binner"_a,
          "num_threads"_a = OMP_NUM_THREADS);
    m.def("get_cross_separations",
          py::overload_cast<const py::array_t<PosCatalog>&,
                            const py::array_t<PosCatalog>&, const BinSpecifier&,
                            const BinSpecifier&,
                            int>(& get_cross_separations_arr),
          py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>(),
          "Get cross-correlation separations limited in the observed perpendicular and parallel directions by the binners",
          "cat1"_a,
          "cat2"_a,
          "perp_binner"_a,
          "par_binner"_a,
          "num_threads"_a = OMP_NUM_THREADS);
    m.def("get_separations",
          py::overload_cast<const py::array_t<PosCatalog>&,
                            const py::array_t<PosCatalog>&, const BinSpecifier&,
                            const BinSpecifier&, bool,
                            int>(& get_separations_arr),
          py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>(),
          "Get the separations limited in the observed perpendiculare and parallel directions by the binners, specifying whether this is an auto or cross correlation",
          "cat1"_a,
          "cat2"_a,
          "perp_binner"_a,
          "par_binner"_a,
          "is_auto"_a,
          "num_threads"_a = OMP_NUM_THREADS);
    bind_get_1d_indices<6>(m);
    bind_get_1d_indices<3>(m);
    bind_get_1d_indices<4>(m);
    bind_get_1d_indices<2>(m);
    declare3D(m);
    declare2D(m);
    declare1D(m);
}
