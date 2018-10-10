#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include "calc_distances.h"
PYBIND11_MAKE_OPAQUE(std::vector<Pos>);
namespace py = pybind11;
using namespace pybind11::literals;


PYBIND11_MODULE(calculate_distances, m) {
    //PYBIND11_NUMPY_DTYPE(Separation, r_perp_t, r_par_t, r_perp_o, r_par_o, ave_r_obs, id1, id2);
    py::bind_vector<std::vector<Pos>>(m, "VectorPos");
    py::class_<Pos>(m, "Pos", "Store the position of a object")
	.def(py::init<double, double, double, double>(), "ra"_a, "dec"_a, "rt"_a, "ro"_a)
	.def(py::init<double, double, double, double, double>(), "nx"_a, "ny"_a, "nz"_a, "rt"_a, "ro"_a)
	.def("nvec", &Pos::nvec, "Get the unit vector of the position")
	.def("rtvec", &Pos::rtvec, "Get the vector for the true position")
	.def("rovec", &Pos::rovec, "Get the vector for the observed position")
	.def_property_readonly("rt", &Pos::rt)
	.def_property_readonly("ro", &Pos::ro)
	.def_property_readonly("ra", &Pos::ra)
	.def_property_readonly("dec", &Pos::dec)
	.def_property_readonly("nx", &Pos::nx)
	.def_property_readonly("ny", &Pos::ny)
	.def_property_readonly("nz", &Pos::nz)
	.def_property_readonly("xt", &Pos::xt)
	.def_property_readonly("yt", &Pos::yt)
	.def_property_readonly("zt", &Pos::zt)
	.def_property_readonly("xo", &Pos::xo)
	.def_property_readonly("yo", &Pos::yo)
	.def_property_readonly("zo", &Pos::zo)
	.def_property_readonly("has_true", &Pos::has_true)
	.def_property_readonly("has_obs", &Pos::has_obs);
    m.def("fill_catalog_vector", py::overload_cast<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>>(&fill_catalog_vector), "Initialize an std::vector<Pos> catalog from vectors of the RA, DEC, and distances", "ra_vec"_a, "dec_vec"_a, "rt_vec"_a, "ro_vec"_a);
    m.def("fill_catalog_vector", py::overload_cast<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>>(&fill_catalog_vector), "Initialize an std::vector<Pos> catalog from vectors of the unit vector components and the distances", "nx_vec"_a, "ny_vec"_a, "nz_vec"_a, "rt_vec"_a, "ro_vec"_a);
    py::class_<Separation>(m, "Separation", "Container for the separations between two galaxies")
	.def(py::init<>())
	.def(py::init<double, double, double, double, double, std::size_t, std::size_t>(), "r_perp_t"_a, "r_par_t"_a, "r_perp_o"_a, "r_par_o"_a, "ave_r_obs"_a, "id1"_a, "id2"_a)
	.def(py::init<std::tuple<double, double>, std::tuple<double, double>, double, std::size_t, std::size_t>(), "r_perp"_a, "r_par"_a, "ave_r_obs"_a, "id1"_a, "id2"_a)
	.def_readonly("r_perp_t", &Separation::r_perp_t)
	.def_readonly("r_par_t", &Separation::r_par_t)
	.def_readonly("r_perp_o", &Separation::r_perp_o)
	.def_readonly("r_par_o", &Separation::r_par_o)
	.def_readonly("ave_r_obs", &Separation::ave_ro)
	.def_readonly("id1", &Separation::id1)
	.def_readonly("id2", &Separation::id2);
    py::class_<VectorSeparation>(m, "VectorSeparation", "Container for the separations between many galaxies, indexed on individual columns or by rows")
	.def(py::init<>(), "Default empty constructor")
	.def(py::init<std::vector<Separation>>(), "Construct from a vector of Separation instances", "separation_vector"_a)
	.def("push_back", py::overload_cast<std::tuple<double, double>, std::tuple<double, double>, double, std::size_t, std::size_t>(&VectorSeparation::push_back), "Push back a new row from individual items", "r_perpendicular"_a, "r_parallel"_a, "ro_ave"_a, "index1"_a, "index2"_a)
	.def("reserve", &VectorSeparation::reserve, "Reserve space for the internal vectors", "new_size"_a)
	.def("push_back", py::overload_cast<Separation>(&VectorSeparation::push_back), "Push back a new row from a Separation inststance", "new_separation"_a)
	.def("__getitem__", [](const VectorSeparation &vs, int i) { try { return vs[i]; } catch (std::out_of_range& e) { throw py::index_error(e.what()); } })
	.def_property_readonly("size", &VectorSeparation::size, "Size of internal vectors")
	.def_property_readonly("r_perp_t", &VectorSeparation::r_perp_t, "True perpendicular separations")
	.def_property_readonly("r_par_t", &VectorSeparation::r_par_t, "True parallel separations")
	.def_property_readonly("r_perp_o", &VectorSeparation::r_perp_o, "Observed perpendicular separations")
	.def_property_readonly("r_par_o", &VectorSeparation::r_par_o, "Observed parallel separations")
	.def_property_readonly("ave_r_obs", &VectorSeparation::ave_ro, "Pair-wise average line-of-sight distances")
	.def_property_readonly("id1", &VectorSeparation::id1, "Index of the first item of each pair")
	.def_property_readonly("id2", &VectorSeparation::id2, "Index of the second item of each pair");
    m.def("unit_dot", &unit_dot, "Get the dot product between the unit vectors of two positions", "pos1"_a, "pos2"_a);
    m.def("dot", &dot, "Get the dot product between two positions, with order (true, observed)", "pos1"_a, "pos2"_a);
    m.def("r_par", &r_par, "Get the parallel separation between two postions, with order (true, observed)", "pos1"_a, "pos2"_a);
    m.def("r_perp", &r_perp, "Get the perpendicular separation between two positions, with order (true, observed)", "pos1"_a, "pos2"_a);
    m.def("ave_lost_distance", &ave_los_distance, "Get the average LOS distance between two positions, using observed positions unless either is missing the observed distance", "pos1"_a, "pos2"_a);
    m.def("get_separations", &get_separations, py::return_value_policy::take_ownership, "Get the separations between two sets of positions", "pos1"_a, "pos2"_a, "rp_min"_a, "rp_max"_a, "rl_min"_a, "rl_max"_a, "use_true"_a, "use_obs"_a, "is_auto"_a);
}
