#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/iostream.h>
#include <stdexcept>
#include <cstddef>
#include <iostream>
#include <vector>
#include <utility>
#include <string>
#include <type_traits>
#include "calc_distances.h"
PYBIND11_MAKE_OPAQUE(std::vector<Pos>);
namespace py = pybind11;
using namespace pybind11::literals;

struct PosCatalog {
    double RA;
    double DEC;
    double D_TRUE;
    double D_OBS;
    double Z_TRUE;
    double Z_OBS;
};
struct SPosCatalog {
    double RA, DEC, D, Z;
};

template <typename T>
py::array_t<T, 0> mkarray_from_vec(std::vector<T> vec, std::vector<std::size_t> shape) {
    std::vector<std::size_t> strides = {shape[1] * shape[2] * sizeof(T), shape[2] * sizeof(T), sizeof(T)};
    return py::array_t<T>(shape, strides, vec.data());
}

template <typename T>
py::array_t<T, 0> mkarray_from_vec(std::vector<T> vec) {
    return py::array_t<T>({vec.size()}, {sizeof(T)}, vec.data());
}

std::vector<int> convert_3d_array(py::array_t<int> counts_in, BinSpecifier x_bins, BinSpecifier y_bins, BinSpecifier z_bins) {
    std::vector<int> out;
    auto in = counts_in.unchecked<3>();
    for (ssize_t i = 0; i < in.shape(0); i++) {
	for (ssize_t j = 0; j < in.shape(1); j++) {
	    for (ssize_t k = 0; k < in.shape(2); k++) {
		out.push_back(in(i,j,k));
	    }
	}
    }
    return out;
}

std::vector<int> convert_1d_array(py::array_t<int> counts_in) {
    auto in = counts_in.unchecked<1>();
    std::vector<int> out(in.size());
    for (ssize_t i = 0; i < in.size(); i++) {
	out[i] = in(i);
    }
    return out;
}

/*
py::array_t<std::size_t> get_1d_indices_from_3d(py::array_t<py::array_t<std::size_t>> input_indices, BinSpecifier x_bins, BinSpecifier y_bins, BinSpecifier z_bins) {
    auto f = [x_bins, y_bins, z_bins](std::size_t x, std::size_t y, std::size_t z) { return get_1d_indexer_from_3d(x, y, z, x_bins, y_bins, z_bins); };
    auto g = [x_bins, y_bins, z_bins, f](py::array_t<std::size_t> index) {
	auto arr = index.unchecked<1>();
	return f(arr(0), arr(1), arr(2));
    };
    return py::vectorize(g)(input_indices);
}
*/

///*
py::array_t<std::size_t> get_1d_indices_from_3d(py::array_t<std::size_t> x_indices, py::array_t<std::size_t> y_indices, py::array_t<std::size_t> z_indices, BinSpecifier x_bins, BinSpecifier y_bins, BinSpecifier z_bins) {
    auto f = [x_bins, y_bins, z_bins](std::size_t x, std::size_t y, std::size_t z) { return get_1d_indexer_from_3d(x, y, z, x_bins, y_bins, z_bins); };
    return py::vectorize(f)(x_indices, y_indices, z_indices);
}
//*/

/*
py::array_t<std::size_t> get_1d_indices_from_3d(py::array_t<std::size_t> input_indices, BinSpecifier x_bins, BinSpecifier y_bins, BinSpecifier z_bins) {
    py::buffer_info input_buff = input_indices.request();
    if (!(input_buff.ndim == 1 || input_buff.ndim == 2)) { throw std::runtime_error("Number of dimensions must be 1 or 2 for multiple index conversion, instead have " + std::to_string(input_buff.ndim)); }
    if (input_buff.ndim == 1) {
	input_indices.resize(std::vector<ptrdiff_t>{1, input_buff.shape[0]});
    }
    auto input = input_indices.unchecked<2>();
    if (input.shape(1) != 3) { throw std::runtime_error("Shape must be (N, 3) for converting N indices"); }
    auto output_indices = py::array_t<std::size_t>(input.shape(0));
    auto output = output_indices.mutable_unchecked<1>();
    for (ssize_t i = 0; i < input.shape(0); i++) {
	output(i) = get_1d_indexer_from_3d(input(i,0), input(i,1), input(i,2), x_bins, y_bins, z_bins);
    }
    return output_indices;
}
*/

void assign_bin_vectorized(NNCounts3D &self, py::array_t<double> r_perp, py::array_t<double> r_par, py::array_t<double> zbar) {
    auto rpo = r_perp.unchecked<1>();
    auto rlo = r_par.unchecked<1>();
    if (rpo.size() != rlo.size()) { throw std::runtime_error("Must have same number of perpendicular and parallel separations"); }
    auto zo = zbar.unchecked<1>();
    if (rpo.size() != zo.size()) { throw std::runtime_error("Must have same number of perpendicular separations and average redshifts"); }
    for (std::size_t i = 0; i < (std::size_t) rpo.size(); i++) {
	self.assign_bin(rpo(i), rlo(i), zo(i));
    }
}

void assign_bin_vectorized(NNCounts1D &self, py::array_t<double> r_vals) {
    auto r = r_vals.unchecked<1>();
    for (std::size_t i = 0; i < (std::size_t) r.size(); i++) {
	self.assign_bin(r(i));
    }
}

/*
NNCounts3D gil_pair_counts_wrapper(std::vector<Pos> pos1, std::vector<Pos> pos2, BinSpecifier rpo_binning, BinSpecifier rlo_binning, BinSpecifier zo_binning, bool is_auto) {
    std::cout << "Acquire the GIL" << std::endl;
    py::gil_scoped_acquire acquire;
    std::cout << "Call get_obs_pair_counts" << std::endl;
    return get_obs_pair_counts(pos1, pos2, rpo_binning, rlo_binning, zo_binning, is_auto);
}
*/

void set_separations(Separation &to, Separation &from) {
    to.r_perp_t = from.r_perp_t;
    to.r_par_t = from.r_par_t;
    to.r_perp_o = from.r_perp_o;
    to.r_par_o = from.r_par_o;
    to.ave_zo = from.ave_zo;
    to.id1 = from.id1;
    to.id2 = from.id2;
}

template <typename T>
py::array mkarray_via_buffer(std::size_t n) {
    return py::array(py::buffer_info(nullptr, sizeof(T),
				     py::format_descriptor<T>::format(),
				     1, {n}, {sizeof(T)}));
}

template <typename S>
py::array_t<S, 0> create_recarray_from_vector(std::vector<S> vec, std::function<void(S&, S&)> f) {
    std::size_t n = vec.size();
    auto arr = mkarray_via_buffer<S>(n);
    auto req = arr.request();
    auto ptr = static_cast<S*>(req.ptr);
    for (std::size_t i = 0; i < n; i++) {
	f(ptr[i], vec[i]);
    }
    return arr;
}

py::array_t<Separation> convert_vector_separations(VectorSeparation vs) {
    return create_recarray_from_vector<Separation>(vs.seps_vec, set_separations);
}

std::vector<Pos> convert_catalog(py::array_t<PosCatalog> arr) {
    auto uarr = arr.unchecked<1>();
    std::size_t n = (std::size_t) uarr.size();
    std::vector<Pos> vec;
    for (std::size_t i = 0; i < n; i++) {
	auto row = uarr(i);
	Pos pos(row.RA, row.DEC, row.D_TRUE, row.D_OBS, row.Z_TRUE, row.Z_OBS);
	vec.push_back(pos);
    }
    return vec;
}

std::vector<SPos> convert_catalog(py::array_t<SPosCatalog> arr) {
    auto uarr = arr.unchecked<1>();
    std::size_t n = (std::size_t) uarr.size();
    std::vector<SPos> vec(n);
    for (std::size_t i = 0; i < n; i++) {
	auto row = uarr(i);
	vec[i] = SPos(row.RA, row.DEC, row.D, row.Z);
    }
    return vec;
}

py::array_t<Separation> get_separations_arr(std::vector<Pos> pos1, std::vector<Pos> pos2, BinSpecifier perp_bins, BinSpecifier par_bins, bool use_true, bool use_obs, bool is_auto) {
    VectorSeparation vs = get_separations(pos1, pos2, perp_bins.get_bin_min(), perp_bins.get_bin_max(), par_bins.get_bin_min(), par_bins.get_bin_max(), use_true, use_obs, is_auto);
    return convert_vector_separations(vs);
}

py::array_t<Separation> get_separations_arr(py::array_t<PosCatalog> pos1, py::array_t<PosCatalog> pos2, BinSpecifier perp_bins, BinSpecifier par_bins, bool use_true, bool use_obs, bool is_auto) {
    auto pos1_vec = convert_catalog(pos1);
    auto pos2_vec = convert_catalog(pos2);
    return get_separations_arr(pos1_vec, pos2_vec, perp_bins, par_bins, use_true, use_obs, is_auto);
}

NNCounts3D get_obs_pair_counts(py::array_t<SPosCatalog> pos1, py::array_t<SPosCatalog> pos2, BinSpecifier rpo_bins, BinSpecifier rlo_bins, BinSpecifier zo_bins, bool is_auto) {
    auto pos1_vec = convert_catalog(pos1);
    auto pos2_vec = convert_catalog(pos2);
    return get_obs_pair_counts(pos1_vec, pos2_vec, rpo_bins, rlo_bins, zo_bins, is_auto);
}

NNCounts3D get_obs_pair_counts(py::array_t<PosCatalog> pos1, py::array_t<PosCatalog> pos2, BinSpecifier rpo_bins, BinSpecifier rlo_bins, BinSpecifier zo_bins, bool is_auto) {
    auto pos1_vec = convert_catalog(pos1);
    auto pos2_vec = convert_catalog(pos2);
    return get_obs_pair_counts(pos1_vec, pos2_vec, rpo_bins, rlo_bins, zo_bins, is_auto);
}

NNCounts1D get_true_pair_counts(py::array_t<PosCatalog> pos1, py::array_t<PosCatalog> pos2, BinSpecifier r_bins, bool is_auto, bool use_true=true) {
    auto pos1_vec = convert_catalog(pos1);
    auto pos2_vec = convert_catalog(pos2);
    return get_true_pair_counts(pos1_vec, pos2_vec, r_bins, is_auto, use_true);
}


PYBIND11_MODULE(calculate_distances, m) {
    py::add_ostream_redirect(m);
    py::bind_vector<std::vector<Pos>>(m, "VectorPos");
    PYBIND11_NUMPY_DTYPE_EX(Separation, r_perp_t, "R_PERP_T", r_par_t, "R_PAR_T", r_perp_o, "R_PERP_O", r_par_o, "R_PAR_O", ave_zo, "AVE_Z_OBS", id1, "ID1", id2, "ID2");
    PYBIND11_NUMPY_DTYPE(PosCatalog, RA, DEC, D_TRUE, D_OBS, Z_TRUE, Z_OBS);
    PYBIND11_NUMPY_DTYPE(SPosCatalog, RA, DEC, D, Z);
    py::class_<SPos>(m, "SPos", "Store only true or only observed position of an object")
	.def(py::init<>(), "Empty constructor")
	.def(py::init<const double, const double, const double, const double>(), "Initialize from values", "ra"_a, "dec"_a, "r"_a, "z"_a)
	.def(py::init<const SPos&>(), "Copy constructor", "other"_a)
	.def("__eq__", [](const SPos& self, const SPos& other) { return self == other; })
	.def("__neq__", [](const SPos& self, const SPos& other) { return self != other; })
	.def_property_readonly("ra", &SPos::ra, "Right ascension (degrees)")
	.def_property_readonly("dec", &SPos::dec, "Declination (degrees)")
	.def_property_readonly("r", &SPos::r, "Line of sight distance")
	.def_property_readonly("z", &SPos::z, "Redshift")
	.def_property_readonly("uvec", &SPos::uvec, "Unit vector in cartesian coordinates")
	.def_property_readonly("rvec", &SPos::rvec, "Cartesian coordinate vector");
    py::class_<Pos>(m, "Pos", "Store the position of a object. Note that tz/oz refer to true/observed redshift while zt/zo refer to true/observed cartesian coordinates")
	.def(py::init<>(), "Empty constructor")
	.def(py::init<double, double, double, double, double, double>(), "Initialize from values", "ra"_a, "dec"_a, "rt"_a, "ro"_a, "tz"_a, "oz"_a)
	.def(py::init<const Pos&>(), "Copy constructor", "other"_a)
	.def_property_readonly("nvec", &Pos::nvec, "Get the unit vector of the position")
	.def_property_readonly("rtvec", &Pos::rtvec, "Get the vector for the true position")
	.def_property_readonly("rovec", &Pos::rovec, "Get the vector for the observed position")
	.def_property_readonly("ra", &Pos::ra, "Right ascension (degrees)")
	.def_property_readonly("dec", &Pos::dec, "Declination (degrees)")
	.def_property_readonly("rt", &Pos::rt, "True line of sight distance")
	.def_property_readonly("ro", &Pos::ro, "Observed line of sight distance")
	.def_property_readonly("zt", &Pos::zt, "True redshift")
	.def_property_readonly("zo", &Pos::zo, "Observed redshift")
	.def_property_readonly("tpos", &Pos::tpos, "SPos instance for true position")
	.def_property_readonly("opos", &Pos::opos, "SPos instance for observed position")
	.def_property_readonly("has_true", &Pos::has_true, "Whether the object has true distance/redshift")
	.def_property_readonly("has_obs", &Pos::has_obs, "Whether the object has observed distance/redshift")
	.def("__eq__", [](const Pos& self, const Pos& other) { return self == other; })
	.def("__neq__", [](const Pos& self, const Pos& other) { return self != other; });
    m.def("fill_catalog_vector", &fill_catalog_vector, "Initialize an std::vector<Pos> catalog from vectors of the RA, DEC, distances, and redshifts", "ra_vec"_a, "dec_vec"_a, "rt_vec"_a, "ro_vec"_a, "tz_vec"_a, "oz_vec"_a);
    py::class_<Separation>(m, "Separation", "Container for the separations between two galaxies")
	.def(py::init<>())
	.def(py::init<double, double, double, double, double, std::size_t, std::size_t>(), "r_perp_t"_a, "r_par_t"_a, "r_perp_o"_a, "r_par_o"_a, "ave_r_obs"_a, "id1"_a, "id2"_a)
	.def(py::init<std::tuple<double, double>, std::tuple<double, double>, double, std::size_t, std::size_t>(), "r_perp"_a, "r_par"_a, "ave_r_obs"_a, "id1"_a, "id2"_a)
	.def_readonly("r_perp_t", &Separation::r_perp_t)
	.def_readonly("r_par_t", &Separation::r_par_t)
	.def_readonly("r_perp_o", &Separation::r_perp_o)
	.def_readonly("r_par_o", &Separation::r_par_o)
	.def_readonly("ave_z_obs", &Separation::ave_zo)
	.def_readonly("id1", &Separation::id1)
	.def_readonly("id2", &Separation::id2)
	.def("__eq__", [](const Separation& self, const Separation& other) { return self == other; })
	.def("__neq__", [](const Separation& self, const Separation& other) { return self != other; });
    py::class_<VectorSeparation>(m, "VectorSeparation", "Container for the separations between many galaxies, indexed on individual columns or by rows")
	.def(py::init<>(), "Default empty constructor")
	.def(py::init<std::vector<Separation>>(), "Construct from a vector of Separation instances", "separation_vector"_a)
	.def("push_back", py::overload_cast<std::tuple<double, double>, std::tuple<double, double>, double, std::size_t, std::size_t>(&VectorSeparation::push_back), "Push back a new row from individual items", "r_perpendicular"_a, "r_parallel"_a, "zo_ave"_a, "index1"_a, "index2"_a)
	.def("reserve", &VectorSeparation::reserve, "Reserve space for the internal vectors", "new_size"_a)
	.def("push_back", py::overload_cast<Separation>(&VectorSeparation::push_back), "Push back a new row from a Separation inststance", "new_separation"_a)
	.def("insert", &VectorSeparation::insert, "Insert the contents of another VectorSeparation at the end of this one", "other"_a)
	.def("__getitem__", [](const VectorSeparation &vs, int i) { try { return vs[i]; } catch (std::out_of_range& e) { throw py::index_error(e.what()); } })
	.def_property_readonly("size", &VectorSeparation::size, "Size of internal vectors")
	.def_property_readonly("r_perp_t", &VectorSeparation::r_perp_t, "True perpendicular separations")
	.def_property_readonly("r_par_t", &VectorSeparation::r_par_t, "True parallel separations")
	.def_property_readonly("r_perp_o", &VectorSeparation::r_perp_o, "Observed perpendicular separations")
	.def_property_readonly("r_par_o", &VectorSeparation::r_par_o, "Observed parallel separations")
	.def_property_readonly("ave_z_obs", &VectorSeparation::ave_zo, "Pair-wise average of observed redshifts")
	.def_property_readonly("id1", &VectorSeparation::id1, "Index of the first item of each pair")
	.def_property_readonly("id2", &VectorSeparation::id2, "Index of the second item of each pair")
	.def("__eq__", [](const VectorSeparation& self, const VectorSeparation& other) { return self == other; })
	.def("__neq__", [](const VectorSeparation& self, const VectorSeparation& other) { return self != other; });
    m.def("unit_dot", py::overload_cast<SPos, SPos>(&unit_dot), "Get the dot product between the unit vectors of two single positions", "pos1"_a, "pos2"_a);
    m.def("unit_dot", py::overload_cast<Pos, Pos>(&unit_dot), "Get the dot product between the unit vectors of two full positions", "pos1"_a, "pos2"_a);
    m.def("dot", py::overload_cast<SPos, SPos>(&dot), "Get the dot product between two single positions", "pos1"_a, "pos2"_a);
    m.def("dot", py::overload_cast<Pos, Pos>(&dot), "Get the dot product between two full positions, with order (true, observed)", "pos1"_a, "pos2"_a);
    m.def("r_par", py::overload_cast<SPos, SPos>(&r_par), "Get the parallel separation between two single positions", "pos1"_a, "pos2"_a);
    m.def("r_par", py::overload_cast<Pos, Pos>(&r_par), "Get the parallel separation between two full postions, with order (true, observed)", "pos1"_a, "pos2"_a);
    m.def("r_perp", py::overload_cast<SPos, SPos>(&r_perp), "Get the perpendicular separation between two single positions", "pos1"_a, "pos2"_a);
    m.def("r_perp", py::overload_cast<Pos, Pos>(&r_perp), "Get the perpendicular separation between two full positions, with order (true, observed)", "pos1"_a, "pos2"_a);
    m.def("ave_z", py::overload_cast<SPos, SPos>(&ave_z), "Get the average redshift for two single positions", "pos1"_a, "pos2"_a);
    m.def("ave_z", py::overload_cast<Pos, Pos>(&ave_z), "Get the average redshift for two full positions, as (true, observed)", "pos1"_a, "pos2"_a);
    m.def("ave_lost_distance", &ave_los_distance, "Get the average redshift of the two full positions, using observed positions unless either is missing the observed distance", "pos1"_a, "pos2"_a);
    m.def("get_separations", py::overload_cast<std::vector<Pos>, std::vector<Pos>, BinSpecifier, BinSpecifier, bool, bool, bool>(&get_separations_arr), "Get the separations between two sets of positions in vector format", "pos1"_a, "pos2"_a, "perp_bins"_a, "par_bins"_a, "use_true"_a, "use_obs"_a, "is_auto"_a);
    m.def("get_separations", py::overload_cast<py::array_t<PosCatalog>, py::array_t<PosCatalog>, BinSpecifier, BinSpecifier, bool, bool, bool>(&get_separations_arr), "Get the separations between two sets of positions passed as numpy structured arrays", "pos1"_a, "pos2"_a, "perp_bins"_a, "par_bins"_a, "use_true"_a, "use_obs"_a, "is_auto"_a);
    py::class_<BinSpecifier>(m, "BinSpecifier", "Structure for the needed attributes for binning in a variable")
	.def(py::init<double, double, double, bool>(), "Construct from min, max, and bin size. Set 'use_log_bins' to True for logarithmic binning", "bin_min"_a, "bin_max"_a, "bin_width"_a, "use_log_bins"_a)
	.def(py::init<double, double, std::size_t, bool>(), "Construct from min, max, and number of bins. Set 'use_log_bins' to True for logarithmic binning", "bin_min"_a, "bin_max"_a, "num_bins"_a, "use_log_bins"_a)
	.def(py::init<const BinSpecifier&>(), "Copy constructor: make a copy of 'other'", "other"_a)
	.def(py::init<>(), "Empty initialization")
	.def("update", &BinSpecifier::update, "Update the values of this instance with the values of 'other', preferring 'other'", "other"_a)
	.def("fill", &BinSpecifier::fill, "Update the values of this instance with the values of 'other', preferring the current values", "other"_a)
	.def_property("bin_min", &BinSpecifier::get_bin_min, &BinSpecifier::set_bin_min, "Minimum bin edge")
	.def_property("bin_max", &BinSpecifier::get_bin_max, &BinSpecifier::set_bin_max, "Maximum bin edge")
	.def_property("bin_size", &BinSpecifier::get_bin_size, &BinSpecifier::set_bin_size, "Size of bins. Note that this may be different than input to make 'nbins' an integer")
	.def_property("nbins", &BinSpecifier::get_nbins, &BinSpecifier::set_nbins, "Number of bins")
	.def_property("log_binning", &BinSpecifier::get_log_binning, &BinSpecifier::set_log_binning, "Whether to use logarithmic binning (True) or not (False)")
	.def("__repr__", &BinSpecifier::toString)
	.def(py::pickle(
		 [](const BinSpecifier &bs) { // __getstate__
		     return py::make_tuple(bs.get_bin_min(), bs.get_bin_max(), bs.get_bin_size(), bs.get_nbins(), bs.get_log_binning());
		 },
		 [](py::tuple t) { // __setstate__
		     if (t.size() != 5) { throw std::runtime_error("Invalid state"); }

		     BinSpecifier bs;
		     bs.set_bin_min(t[0].cast<double>());
		     bs.set_bin_max(t[1].cast<double>());
		     bs.set_bin_size(t[2].cast<double>());
		     bs.set_nbins(t[3].cast<std::size_t>());
		     bs.set_log_binning(t[4].cast<bool>());
		     return bs;
		 }
		 ))
	.def("__eq__", [](const BinSpecifier &self, const BinSpecifier &other){ return self == other; })
	.def("__ne__", [](const BinSpecifier &self, const BinSpecifier &other){ return self != other; })
	.def("assign_bin", py::vectorize(&BinSpecifier::assign_bin), "Get the bin index/indices corresponding to the given value(s)", "value"_a)
	.def("lower_bin_edges", [](const BinSpecifier &self) { return mkarray_from_vec<double>(self.lower_bin_edges()); }, "Get the lower edges of all bins")
	.def("upper_bin_edges", [](const BinSpecifier &self) { return mkarray_from_vec<double>(self.upper_bin_edges()); }, "Get the upper edges of all bins")
	.def_property_readonly("is_set", &BinSpecifier::is_set, "Flag to specify whether the values have actually been set (True) or not (False) yet");
    py::class_<NNCounts3D>(m, "NNCounts3D", "Container for getting pair counts in terms of observed perpendicular and parallel separations and average observed redshift")
	.def(py::init<>(), "Empty constructor: use default empty constructor")
	.def(py::init<const NNCounts3D&>(), "Copy constructor: make a copy of 'other'", "other"_a)
	.def(py::init([](BinSpecifier rpo_binning, BinSpecifier rlo_binning, BinSpecifier zo_binning, py::array_t<int> counts, std::size_t n_tot) {
		    auto counts_vec = convert_3d_array(counts, rpo_binning, rlo_binning, zo_binning);
		    return NNCounts3D(rpo_binning, rlo_binning, zo_binning, counts_vec, n_tot);
		}), "Like the copy constructor, but from pickled objects")
	.def(py::init<BinSpecifier, BinSpecifier, BinSpecifier>(), "Initialize a new pair counter with the specified binning", "rperp_bin_specifier"_a, "rpar_bin_specifier"_a, "zbar_bin_specifier"_a)
	.def("update_rpo_binning", &NNCounts3D::update_rpo_binning, "Update the binning in r_perp, optionally preferring the old values over the new (default)", "new_binning"_a, "prefer_old"_a=true)
	.def("update_rlo_binning", &NNCounts3D::update_rlo_binning, "Update the binning in r_parallel, optionally preferring the old values over the new (default)", "new_binning"_a, "prefer_old"_a=true)
	.def("update_zbar_binning", &NNCounts3D::update_zo_binning, "Update the binning in zbar, optionally preferring the old values over the new (default)", "new_binning"_a, "prefer_old"_a=true)
	.def("get_1d_index", &NNCounts3D::get_1d_indexer, "Get the 1D index for the given 3D index", "x_idx"_a, "y_idx"_a, "z_idx"_a)
	.def("get_bin", py::vectorize(&NNCounts3D::get_bin), "Get the 1D bin index/indices corresponding to the/each 3D observed separation, with -1 indicating a separation is outside of the bins", "r_perp"_a, "r_par"_a, "zbar"_a)
	.def("assign_bin", [](NNCounts3D &self, py::array_t<double> r_perp, py::array_t<double> r_par, py::array_t<double> zbar) { return assign_bin_vectorized(self, r_perp, r_par, zbar); }, "Assign a 1D bin number to the/each 3D observed separation, and add to the counts and total")
	.def("__getitem__", [](const NNCounts3D &self, const std::size_t idx){ return self[idx]; })
	.def_property_readonly("n_tot", &NNCounts3D::n_tot, "The total number of pairs considered, for normalization")
	.def_property_readonly("counts", [](const NNCounts3D &self) { return mkarray_from_vec<int>(self.counts(), {self.rpo_bin_info().get_nbins(), self.rlo_bin_info().get_nbins(), self.zo_bin_info().get_nbins()}); }, "The pair counts in 3D bins, as a 3D ndarray")
	.def_property_readonly("r_perp_bin_info", &NNCounts3D::rpo_bin_info, "Get the bin specification for perpendicular bins")
	.def_property_readonly("r_par_bin_info", &NNCounts3D::rlo_bin_info, "Get the bin specification for parallel bins")
	.def_property_readonly("zbar_bin_info", &NNCounts3D::zo_bin_info, "Get the bin specification for average redshift bins")
	.def(py::self += py::self)
	.def(py::self += float())
	.def(py::self += int())
	.def(py::self + py::self)
	.def(py::self + float())
	.def(py::self + int())
	.def(float() + py::self)
	.def(int() + py::self)
	.def(py::pickle(
		 [](const NNCounts3D &self) { // __getstate__
		     return py::make_tuple(self.rpo_bin_info(), self.rlo_bin_info(), self.zo_bin_info(), self.counts(), self.n_tot());
		 },
		 [](py::tuple t) { // __setstate__
		     if (t.size() != 5) { throw std::runtime_error("Invalid state"); }

		     NNCounts3D self(t[0].cast<BinSpecifier>(),
				     t[1].cast<BinSpecifier>(),
				     t[2].cast<BinSpecifier>(),
				     t[3].cast<std::vector<int>>(),
				     t[4].cast<std::size_t>());

		     return self;
		 }
		 ))
	.def_property_readonly("shape", [](const NNCounts3D &self) { return py::make_tuple(self.rpo_bin_info().get_nbins(), self.rlo_bin_info().get_nbins(), self.zo_bin_info().get_nbins()); })
	.def("__repr__", &NNCounts3D::toString);
    py::class_<NNCounts1D>(m, "NNCounts1D", "Container for the pair counts in terms of the magnitude of the separation")
	.def(py::init<>(), "Empty constructor")
	.def(py::init<const NNCounts1D&>(), "Copy constructor", "other"_a)
	.def(py::init([](BinSpecifier binning, py::array_t<std::size_t> counts, std::size_t n_tot) {
		    auto counts_vec = convert_1d_array(counts);
		    return NNCounts1D(binning, counts_vec, n_tot);
		}), "Like the copy constructor, but from pickled objects")
	.def(py::init<BinSpecifier>(), "Initialize an empty NNCounts1D with the specified binning", "binning"_a)
	.def("update_binning", &NNCounts1D::update_binning, "Update the binning specifications, and optionally prefer the original options (default)", "new_binning"_a, "prefer_old"_a=true)
	.def("get_bin", py::vectorize(&NNCounts1D::get_bin), "Get the bin index/indices corresponding to the/each separation, with -1 meaning the separation is outside of the bin range", "r"_a)
	.def("assign_bin", [](NNCounts1D &self, py::array_t<double> r) { return assign_bin_vectorized(self, r); }, "Assign a bin number for the/each separation, and add to counts and total")
	.def("__getitem__", [](const NNCounts1D &self, std::size_t idx) { return self[idx]; })
	.def_property_readonly("n_tot", &NNCounts1D::n_tot)
	.def_property_readonly("counts", [](const NNCounts1D &self) { return mkarray_from_vec<int>(self.counts()); })
	.def_property_readonly("bin_info", &NNCounts1D::bin_info)
	.def(py::self += py::self)
	.def(py::self += float())
	.def(py::self += int())
	.def(py::self + py::self)
	.def(py::self + float())
	.def(py::self + int())
	.def(float() + py::self)
	.def(int() + py::self)
	.def(py::pickle(
		 [](const NNCounts1D &self) { // __getstate__
		     return py::make_tuple(self.bin_info(), self.counts(), self.n_tot());
		 },
		 [](py::tuple t) { // __setstate__
		     if (t.size() != 3) { throw std::runtime_error("Invalid state"); }

		     NNCounts1D self(t[0].cast<BinSpecifier>(),
				     t[1].cast<std::vector<int>>(),
				     t[2].cast<std::size_t>());

		     return self;
		 }
		 ))
	.def("__repr__", &NNCounts1D::toString);
    m.def("get_obs_pair_counts", py::overload_cast<py::array_t<SPosCatalog>, py::array_t<SPosCatalog>, BinSpecifier, BinSpecifier, BinSpecifier, bool>(&get_obs_pair_counts), py::return_value_policy::take_ownership, "Get 3D histogrammed pair counts for separations", "pos1"_a, "pos2"_a, "rpo_binning"_a, "rlo_binning"_a, "zo_binning"_a, "is_auto"_a);
    m.def("get_obs_pair_counts", py::overload_cast<std::vector<Pos>, std::vector<Pos>, BinSpecifier, BinSpecifier, BinSpecifier, bool>(&get_obs_pair_counts), py::return_value_policy::take_ownership, "Get the histogrammed pair counts for observed separations", "pos1"_a, "pos2"_a, "rpo_binning"_a, "rlo_binning"_a, "zo_binning"_a, "is_auto"_a);
    m.def("get_obs_pair_counts", py::overload_cast<py::array_t<PosCatalog>, py::array_t<PosCatalog>, BinSpecifier, BinSpecifier, BinSpecifier, bool>(&get_obs_pair_counts), py::return_value_policy::take_ownership, "Get the histogrammed pair counts for observed separations (accepts numpy arrays)", "pos1"_a, "pos2"_a, "rpo_binning"_a, "rlo_binning"_a, "zo_binning"_a, "is_auto"_a);
    m.def("get_1d_index_from_3d", &get_1d_indices_from_3d, "Get the 1D indices from the given 3D indices and binning specifications", "x_indices"_a, "y_indices"_a, "z_indices"_a, "x_bins"_a, "y_bins"_a, "z_bins"_a);
    m.def("get_true_pair_counts", py::overload_cast<std::vector<Pos>, std::vector<Pos>, BinSpecifier, bool, bool>(&get_true_pair_counts), py::return_value_policy::take_ownership, "Get the histogrammed pair counts for separations in terms of the magnitude of the separation", "pos1"_a, "pos2"_a, "r_binning"_a, "is_auto"_a, "use_true"_a=true);
    m.def("get_true_pair_counts", py::overload_cast<py::array_t<PosCatalog>, py::array_t<PosCatalog>, BinSpecifier, bool, bool>(&get_true_pair_counts), py::return_value_policy::take_ownership, "Get the histogrammed pair counts for separations in terms of the magnitude of the separation (accepts numpy arrays)", "pos1"_a, "pos2"_a, "r_binning"_a, "is_auto"_a, "use_true"_a=true);
}
