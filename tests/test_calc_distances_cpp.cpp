#include <cmath>
#include <stdexcept>
#include <vector>
#include <tuple>
#include <exception>
#include <stdexcept>
#include <string>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <iostream>
#include "external/catch.hpp"
#include "calc_distances.h"

class ExceptionMessage : public Catch::MatcherBase<std::exception> {
    Catch::Matchers::StdString::ContainsMatcher matcher;
public:
    ExceptionMessage(const Catch::Matchers::StdString::CasedString &compare_string) : matcher(compare_string) {}
    bool match(const std::exception& e) const {
	return matcher.match(e.what());
    }
    std::string describe() const {
	std::ostringstream ss;
	ss << "error message contains '" << matcher.m_comparator.m_str << "'";
	return ss.str();
    }
};

inline ExceptionMessage ExMsgContains(const std::string &compare_string) {
    return ExceptionMessage(Catch::Matchers::StdString::CasedString(compare_string, Catch::CaseSensitive::Yes));
}

TEST_CASE("Position vectors", "[pos][vectors]"){
    double one_third = 1.0 / std::sqrt(3.0);
    SECTION("Fail cases (bad angles or unit vectors") {
	SECTION("Angle related exceptions") {
	    SECTION("Only bad RA") {
		REQUIRE_THROWS_MATCHES([&](){ Pos pos(-1.0, 0.0, NAN, NAN); }(), std::invalid_argument, ExMsgContains("RA"));
		REQUIRE_THROWS_MATCHES([&](){ Pos pos(-0.00000005, 0.0, NAN, NAN); }(), std::invalid_argument, ExMsgContains("RA"));
		REQUIRE_THROWS_MATCHES([&](){ Pos pos(360.00000005, 0.0, NAN, NAN); }(), std::invalid_argument, ExMsgContains("RA"));
		REQUIRE_THROWS_MATCHES([&](){ Pos pos(361.0, 0.0, NAN, NAN); }(), std::invalid_argument, ExMsgContains("RA"));
		REQUIRE_THROWS_MATCHES([&](){ Pos pos(NAN, 0.0, NAN, NAN); }(), std::invalid_argument, ExMsgContains("RA"));
	    }
	    SECTION("Only bad DEC") {
		REQUIRE_THROWS_MATCHES([&](){ Pos pos(5.0, -91.0, NAN, NAN); }(), std::invalid_argument, ExMsgContains("DEC"));
		REQUIRE_THROWS_MATCHES([&](){ Pos pos(5.0, -90.00000005, NAN, NAN); }(), std::invalid_argument, ExMsgContains("DEC"));
		REQUIRE_THROWS_MATCHES([&](){ Pos pos(5.0, 90.00000005, NAN, NAN); }(), std::invalid_argument, ExMsgContains("DEC"));
		REQUIRE_THROWS_MATCHES([&](){ Pos pos(5.0, 91.0, NAN, NAN); }(), std::invalid_argument, ExMsgContains("DEC"));
		REQUIRE_THROWS_MATCHES([&](){ Pos pos(5.0, NAN, NAN, NAN); }(), std::invalid_argument, ExMsgContains("DEC"));
	    }
	    SECTION("Bad RA and DEC") {
		REQUIRE_THROWS_MATCHES([&](){ Pos pos(-1.0, NAN, NAN, NAN); }(), std::invalid_argument, ExMsgContains("RA"));
		REQUIRE_THROWS_MATCHES([&](){ Pos pos(NAN, -91.0, NAN, NAN); }(), std::invalid_argument, ExMsgContains("RA"));
		REQUIRE_THROWS_MATCHES([&](){ Pos pos(NAN, NAN, NAN, NAN); }(), std::invalid_argument, ExMsgContains("RA"));
	    }
	}
	SECTION("Unit vector error") {
	    REQUIRE_THROWS_MATCHES([&](){ Pos pos(0.0, 0.0, 0.0, NAN, NAN); }(), std::invalid_argument, ExMsgContains("unit vector"));
	    REQUIRE_THROWS_MATCHES([&](){ Pos pos(1.0, 1.0, 1.0, NAN, NAN); }(), std::invalid_argument, ExMsgContains("unit vector"));
	    REQUIRE_THROWS_MATCHES([&](){ Pos pos(NAN, 0.0, 1.0, NAN, NAN); }(), std::invalid_argument, ExMsgContains("unit vector"));
	    REQUIRE_THROWS_MATCHES([&](){ Pos pos(NAN, 1.0, 0.0, NAN, NAN); }(), std::invalid_argument, ExMsgContains("unit vector"));
	    REQUIRE_THROWS_MATCHES([&](){ Pos pos(0.0, NAN, 1.0, NAN, NAN); }(), std::invalid_argument, ExMsgContains("unit vector"));
	    REQUIRE_THROWS_MATCHES([&](){ Pos pos(1.0, NAN, 0.0, NAN, NAN); }(), std::invalid_argument, ExMsgContains("unit vector"));
	    REQUIRE_THROWS_MATCHES([&](){ Pos pos(0.0, 1.0, NAN, NAN, NAN); }(), std::invalid_argument, ExMsgContains("unit vector"));
	    REQUIRE_THROWS_MATCHES([&](){ Pos pos(1.0, 0.0, NAN, NAN, NAN); }(), std::invalid_argument, ExMsgContains("unit vector"));
	    REQUIRE_THROWS_MATCHES([&](){ Pos pos(one_third - 5.0e-7, one_third, one_third, NAN, NAN); }(), std::invalid_argument, ExMsgContains("unit vector"));
	    REQUIRE_THROWS_MATCHES([&](){ Pos pos(one_third + 5.0e-7, one_third, one_third, NAN, NAN); }(), std::invalid_argument, ExMsgContains("unit vector"));
	    REQUIRE_THROWS_MATCHES([&](){ Pos pos(one_third, one_third - 5.0e-7, one_third, NAN, NAN); }(), std::invalid_argument, ExMsgContains("unit vector"));
	    REQUIRE_THROWS_MATCHES([&](){ Pos pos(one_third, one_third + 5.0e-7, one_third, NAN, NAN); }(), std::invalid_argument, ExMsgContains("unit vector"));
	    REQUIRE_THROWS_MATCHES([&](){ Pos pos(one_third, one_third, one_third - 5.0e-7, NAN, NAN); }(), std::invalid_argument, ExMsgContains("unit vector"));
	    REQUIRE_THROWS_MATCHES([&](){ Pos pos(one_third, one_third, one_third + 5.0e-7, NAN, NAN); }(), std::invalid_argument, ExMsgContains("unit vector"));
	    }
    }
    SECTION("Setting or not setting distances") {
	SECTION("Neither given") {
	    Pos pos_radec(0.0, 0.0, NAN, NAN);
	    Pos pos_nhat(0.0, 0.0, 1.0, NAN, NAN);
	    REQUIRE_FALSE(pos_radec.has_true());
	    REQUIRE_FALSE(pos_radec.has_obs());
	    REQUIRE_FALSE(pos_nhat.has_true());
	    REQUIRE_FALSE(pos_nhat.has_obs());
	}
	SECTION("True distance given") {
	    Pos pos_radec(0.0, 0.0, 1.0, NAN);
	    Pos pos_nhat(0.0, 0.0, 1.0, 1.0, NAN);
	    REQUIRE(pos_radec.has_true());
	    REQUIRE_FALSE(pos_radec.has_obs());
	    REQUIRE(pos_nhat.has_true());
	    REQUIRE_FALSE(pos_nhat.has_obs());
	}
	SECTION("Observed distance given") {
	    Pos pos_radec(0.0, 0.0, NAN, 1.0);
	    Pos pos_nhat(0.0, 0.0, 1.0, NAN, 1.0);
	    REQUIRE_FALSE(pos_radec.has_true());
	    REQUIRE(pos_radec.has_obs());
	    REQUIRE_FALSE(pos_nhat.has_true());
	    REQUIRE(pos_nhat.has_obs());
	}
	SECTION("Both given") {
	    Pos pos_radec(0.0, 0.0, 1.0, 1.0);
	    Pos pos_nhat(0.0, 0.0, 1.0, 1.0, 1.0);
	    REQUIRE(pos_radec.has_true());
	    REQUIRE(pos_radec.has_obs());
	    REQUIRE(pos_nhat.has_true());
	    REQUIRE(pos_nhat.has_obs());
	}
    }
    SECTION("Angle-unit vector conversion") {
	double dec1 = 0.0;
	double ra1 = 0.0;
	double dec2 = 0.0;
	double ra2 = 90.0;
	double dec3 = 90.0;
	double ra3 = 0.0;
	SECTION("Angles given") {
	    Pos pos1(ra1, dec1, NAN, NAN);
	    REQUIRE(pos1.nx() == Approx(1.0));
	    REQUIRE(pos1.ny() == Approx(0.0).margin(1.e-7));
	    REQUIRE(pos1.nz() == Approx(0.0).margin(1.e-7));
	    Pos pos2(ra2, dec2, NAN, NAN);
	    REQUIRE(pos2.nx() == Approx(0.0).margin(1.e-7));
	    REQUIRE(pos2.ny() == Approx(1.0));
	    REQUIRE(pos2.nz() == Approx(0.0).margin(1.e-7));
	    Pos pos3(ra3, dec3, NAN, NAN);
	    REQUIRE(pos3.nx() == Approx(0.0).margin(1.e-7));
	    REQUIRE(pos3.ny() == Approx(0.0).margin(1.e-7));
	    REQUIRE(pos3.nz() == Approx(1.0));
	}
	SECTION("Unit vector given") {
	    Pos pos1(1.0, 0.0, 0.0, NAN, NAN);
	    REQUIRE(pos1.ra() == Approx(ra1));
	    REQUIRE(pos1.dec() == Approx(dec1));
	    Pos pos2(0.0, 1.0, 0.0, NAN, NAN);
	    REQUIRE(pos2.ra() == Approx(ra2));
	    REQUIRE(pos2.dec() == Approx(dec2));
	    Pos pos3(0.0, 0.0, 1.0, NAN, NAN);
	    REQUIRE(pos3.ra() == Approx(ra3));
	    REQUIRE(pos3.dec() == Approx(dec3));
	}
    }
}

TEST_CASE("Dot products of vectors", "[dot][vectors]") {
    // Set up vectors to test
    Pos pos1(32.5, -10.0, 3297.0, NAN);
    Pos pos2(108.2, 2.5, 5360.0, NAN);
    Pos upos1(32.5, -10.0, 1.0, NAN);
    Pos upos2(108.2, 2.5, 1.0, NAN);
    // Get the expected unit dot product
    double unit_dot_exp = (pos1.nx() * pos2.nx()) + (pos1.ny() * pos2.ny()) + (pos1.nz() * pos2.nz());
    // Check that has_obs is always false
    REQUIRE_FALSE(pos1.has_obs());
    REQUIRE_FALSE(pos2.has_obs());
    REQUIRE_FALSE(upos1.has_obs());
    REQUIRE_FALSE(upos2.has_obs());
    
    SECTION("Unit vector dot products for same vectors") {
	REQUIRE(unit_dot(upos1, upos1) == Approx(1.0));
	REQUIRE(unit_dot(upos1, pos1) == Approx(1.0));
	REQUIRE(unit_dot(pos1, upos1) == Approx(1.0));
	REQUIRE(unit_dot(pos1, pos1) == Approx(1.0));
	REQUIRE(unit_dot(upos2, upos2) == Approx(1.0));
	REQUIRE(unit_dot(upos2, pos2) == Approx(1.0));
	REQUIRE(unit_dot(pos2, upos2) == Approx(1.0));
	REQUIRE(unit_dot(pos2, pos2) == Approx(1.0));
    }

    SECTION("Full dot products for same vectors") {
	auto u1u1_dot = dot(upos1, upos1);
	auto u1p1_dot = dot(upos1, pos1);
	auto p1u1_dot = dot(pos1, upos1);
	auto p1p1_dot = dot(pos1, pos1);
	auto u2u2_dot = dot(upos2, upos2);
	auto u2p2_dot = dot(upos2, pos2);
	auto p2u2_dot = dot(pos2, upos2);
	auto p2p2_dot = dot(pos2, pos2);
	REQUIRE(std::get<0>(u1u1_dot) == Approx(1.0));
	REQUIRE(std::isnan(std::get<1>(u1u1_dot)));
	REQUIRE(std::get<0>(u1p1_dot) == Approx(pos1.rt()));
	REQUIRE(std::isnan(std::get<1>(u1p1_dot)));
	REQUIRE(std::get<0>(p1u1_dot) == Approx(pos1.rt()));
	REQUIRE(std::isnan(std::get<1>(p1u1_dot)));
	REQUIRE(std::get<0>(p1p1_dot) == Approx(pos1.rt() * pos1.rt()));
	REQUIRE(std::isnan(std::get<1>(p1p1_dot)));
	REQUIRE(std::get<0>(u2u2_dot) == Approx(1.0));
	REQUIRE(std::isnan(std::get<1>(u2u2_dot)));
	REQUIRE(std::get<0>(u2p2_dot) == Approx(pos2.rt()));
	REQUIRE(std::isnan(std::get<1>(u2p2_dot)));
	REQUIRE(std::get<0>(p2u2_dot) == Approx(pos2.rt()));
	REQUIRE(std::isnan(std::get<1>(p2u2_dot)));
	REQUIRE(std::get<0>(p2p2_dot) == Approx(pos2.rt() * pos2.rt()));
	REQUIRE(std::isnan(std::get<1>(p2p2_dot)));
    }

    SECTION("Different vector unit dot products") {
	REQUIRE(unit_dot(upos1, upos2) == Approx(unit_dot_exp));
	REQUIRE(unit_dot(upos1, pos2) == Approx(unit_dot_exp));
	REQUIRE(unit_dot(pos1, upos2) == Approx(unit_dot_exp));
	REQUIRE(unit_dot(pos1, pos2) == Approx(unit_dot_exp));
    }

    SECTION("Different vector dot products") {
	auto u1u2_dot = dot(upos1, upos2);
	auto u1p2_dot = dot(upos1, pos2);
	auto p1u2_dot = dot(pos1, upos2);
	auto p1p2_dot = dot(pos1, pos2);
	REQUIRE(std::get<0>(u1u2_dot) == Approx(unit_dot_exp));
	REQUIRE(std::isnan(std::get<1>(u1u2_dot)));
	REQUIRE(std::get<0>(u1p2_dot) == Approx(pos2.rt() * unit_dot_exp));
	REQUIRE(std::isnan(std::get<1>(u1p2_dot)));
	REQUIRE(std::get<0>(p1u2_dot) == Approx(pos1.rt() * unit_dot_exp));
	REQUIRE(std::isnan(std::get<1>(p1u2_dot)));
	REQUIRE(std::get<0>(p1p2_dot) == Approx(pos1.rt() * pos2.rt() * unit_dot_exp));
	REQUIRE(std::isnan(std::get<1>(p1p2_dot)));
    }
}

TEST_CASE("Parallel separation", "[rparallel][seps][vectors]") {
    // Set up vectors to test
    Pos pos1(32.5, -10.0, 3297.0, NAN);
    Pos pos2(108.2, 2.5, 5360.0, NAN);
    Pos upos1(32.5, -10.0, 1.0, NAN);
    Pos upos2(108.2, 2.5, 1.0, NAN);
    std::vector<double> r1 = pos1.rtvec();
    std::vector<double> r2 = pos2.rtvec();
    std::vector<double> ur1 = upos1.rtvec();
    std::vector<double> ur2 = upos2.rtvec();
    std::vector<double> lvec = {pos1.nx() + pos2.nx(), pos1.ny() + pos2.ny(), pos1.nz() + pos2.nz()};
    double lmag = std::sqrt((lvec[0] * lvec[0]) + (lvec[1] * lvec[1]) + (lvec[2] * lvec[2]));
    std::vector<double> lhat = {lvec[0] / lmag, lvec[1] / lmag, lvec[2] / lmag};
     SECTION("Unit vectors") {
	std::vector<double> r = {ur1[0] - ur2[0], ur1[1] - ur2[1], ur1[2] - ur2[2]};
	double r_par_exp = std::fabs((r[0] * lhat[0]) + (r[1] * lhat[1]) + (r[2] * lhat[2]));
	std::tuple<double, double> r_parallel = r_par(upos1, upos2);
	REQUIRE(std::get<0>(r_parallel) == Approx(r_par_exp).margin(1.e-7));
	REQUIRE(std::isnan(std::get<1>(r_parallel)));
    }
    SECTION("Unit vector with full vector") {
	std::vector<double> r = {ur1[0] - r2[0], ur1[1] - r2[1], ur1[2] - r2[2]};
	double r_par_exp = std::fabs((r[0] * lhat[0]) + (r[1] * lhat[1]) + (r[2] * lhat[2]));
	std::tuple<double, double> r_parallel = r_par(upos1, pos2);
	REQUIRE(std::get<0>(r_parallel) == Approx(r_par_exp).margin(1.e-7));
	REQUIRE(std::isnan(std::get<1>(r_parallel)));
    }
    SECTION("Full vector with unit vector") {
	std::vector<double> r = {r1[0] - ur2[0], r1[1] - ur2[1], r1[2] - ur2[2]};
	double r_par_exp = std::fabs((r[0] * lhat[0]) + (r[1] * lhat[1]) + (r[2] * lhat[2]));
	std::tuple<double, double> r_parallel = r_par(pos1, upos2);
	REQUIRE(std::get<0>(r_parallel) == Approx(r_par_exp).margin(1.e-7));
	REQUIRE(std::isnan(std::get<1>(r_parallel)));
    }
    SECTION("Full vectors") {
	std::vector<double> r = {r1[0] - r2[0], r1[1] - r2[1], r1[2] - r2[2]};
	double r_par_exp = std::fabs((r[0] * lhat[0]) + (r[1] * lhat[1]) + (r[2] * lhat[2]));
	std::tuple<double, double> r_parallel = r_par(pos1, pos2);
	REQUIRE(std::get<0>(r_parallel) == Approx(r_par_exp).margin(1.e-7));
	REQUIRE(std::isnan(std::get<1>(r_parallel)));
    }
}

TEST_CASE("Perpendicular separation", "[rperpendicular][seps][vectors]") {
    // Set up vectors to test
    Pos pos1(32.5, -10.0, 3297.0, NAN);
    Pos pos2(108.2, 2.5, 5360.0, NAN);
    Pos upos1(32.5, -10.0, 1.0, NAN);
    Pos upos2(108.2, 2.5, 1.0, NAN);
    std::vector<double> r1 = pos1.rtvec();
    std::vector<double> r2 = pos2.rtvec();
    std::vector<double> ur1 = upos1.rtvec();
    std::vector<double> ur2 = upos2.rtvec();
    std::vector<double> lvec = {pos1.nx() + pos2.nx(), pos1.ny() + pos2.ny(), pos1.nz() + pos2.nz()};
    double lmag = std::sqrt((lvec[0] * lvec[0]) + (lvec[1] * lvec[1]) + (lvec[2] * lvec[2]));
    std::vector<double> lhat = {lvec[0] / lmag, lvec[1] / lmag, lvec[2] / lmag};
    SECTION("Unit vectors") {
	double r_par_exp = ((ur1[0] - ur2[0]) * lhat[0]) + ((ur1[1] - ur2[1]) * lhat[1]) + ((ur1[2] - ur2[2]) * lhat[2]);
	std::vector<double> r_perp_vec = {(ur1[0] - ur2[0]) - (r_par_exp * lhat[0]), (ur1[1] - ur2[1]) - (r_par_exp * lhat[1]), (ur1[2] - ur2[2]) - (r_par_exp * lhat[2])};
	double r_perp_exp = std::sqrt((r_perp_vec[0] * r_perp_vec[0]) + (r_perp_vec[1] * r_perp_vec[1]) + (r_perp_vec[2] * r_perp_vec[2]));
	std::tuple<double, double> r_perpendicular = r_perp(upos1, upos2);
	REQUIRE(std::get<0>(r_perpendicular) == Approx(r_perp_exp));
	REQUIRE(std::isnan(std::get<1>(r_perpendicular)));
    }
    SECTION("Unit vector with full vector") {
	double r_par_exp = ((ur1[0] - r2[0]) * lhat[0]) + ((ur1[1] - r2[1]) * lhat[1]) + ((ur1[2] - r2[2]) * lhat[2]);
	std::vector<double> r_perp_vec = {(ur1[0] - r2[0]) - (r_par_exp * lhat[0]), (ur1[1] - r2[1]) - (r_par_exp * lhat[1]), (ur1[2] - r2[2]) - (r_par_exp * lhat[2])};
	double r_perp_exp = std::sqrt((r_perp_vec[0] * r_perp_vec[0]) + (r_perp_vec[1] * r_perp_vec[1]) + (r_perp_vec[2] * r_perp_vec[2]));
	std::tuple<double, double> r_perpendicular = r_perp(upos1, pos2);
	REQUIRE(std::get<0>(r_perpendicular) == Approx(r_perp_exp));
	REQUIRE(std::isnan(std::get<1>(r_perpendicular)));
    }
    SECTION("Full vector with unit vector") {
	double r_par_exp = ((r1[0] - ur2[0]) * lhat[0]) + ((r1[1] - ur2[1]) * lhat[1]) + ((r1[2] - ur2[2]) * lhat[2]);
	std::vector<double> r_perp_vec = {(r1[0] - ur2[0]) - (r_par_exp * lhat[0]), (r1[1] - ur2[1]) - (r_par_exp * lhat[1]), (r1[2] - ur2[2]) - (r_par_exp * lhat[2])};
	double r_perp_exp = std::sqrt((r_perp_vec[0] * r_perp_vec[0]) + (r_perp_vec[1] * r_perp_vec[1]) + (r_perp_vec[2] * r_perp_vec[2]));
	std::tuple<double, double> r_perpendicular = r_perp(pos1, upos2);
	REQUIRE(std::get<0>(r_perpendicular) == Approx(r_perp_exp));
	REQUIRE(std::isnan(std::get<1>(r_perpendicular)));
    }
    SECTION("Full vectors") {
	double r_par_exp = ((r1[0] - r2[0]) * lhat[0]) + ((r1[1] - r2[1]) * lhat[1]) + ((r1[2] - r2[2]) * lhat[2]);
	std::vector<double> r_perp_vec = {(r1[0] - r2[0]) - (r_par_exp * lhat[0]), (r1[1] - r2[1]) - (r_par_exp * lhat[1]), (r1[2] - r2[2]) - (r_par_exp * lhat[2])};
	double r_perp_exp = std::sqrt((r_perp_vec[0] * r_perp_vec[0]) + (r_perp_vec[1] * r_perp_vec[1]) + (r_perp_vec[2] * r_perp_vec[2]));
	std::tuple<double, double> r_perpendicular = r_perp(pos1, pos2);
	REQUIRE(std::get<0>(r_perpendicular) == Approx(r_perp_exp));
	REQUIRE(std::isnan(std::get<1>(r_perpendicular)));
    }
}

TEST_CASE("Catalog of positions", "[VecPos][catalog]") {
    // Read in a test catalog, and then make sure our std::vector<Pos> matches what we would expect
    // Note that this test catalog has NAN for observed distances
    double rai, deci, rti, roi;
    std::vector<double> ra_vec, dec_vec, rt_vec, ro_vec;
    std::string line, nani;
    char *end;
    std::ifstream fin("test_data/catalog_three_objects.txt");
    if (fin.is_open()) {
	while(std::getline(fin, line)) {
	    if (line[0] == '#') continue;
	    CAPTURE(line);
	    std::istringstream iss(line);
	    if (!(iss >> rai >> deci >> rti >> nani)) FAIL("Unable to read test catalog file 'test_data/catalog_three_objects.txt'");
	    roi = std::strtod(nani.c_str(), &end);
	    ra_vec.push_back(rai);
	    dec_vec.push_back(deci);
	    rt_vec.push_back(rti);
	    ro_vec.push_back(roi);
	}
	fin.close();
    }
    else FAIL("Unable to open test catalog file 'test_data/catalog_three_objects.txt'");
    std::vector<Pos> catalog = fill_catalog_vector(ra_vec, dec_vec, rt_vec, ro_vec);
    // First require that the size of the catalog is correct
    REQUIRE(catalog.size() == ra_vec.size());
    SECTION("Check RA is correct for each entry") {
	for (std::size_t i = 0; i < ra_vec.size(); i++) {
	    CAPTURE(i);
	    REQUIRE(catalog[i].ra() == Approx(ra_vec[i]).margin(1.e-7));
	}
    }
    SECTION("Check DEC is correct for each entry") {
	for (std::size_t i = 0; i < dec_vec.size(); i++) {
	    CAPTURE(i);
	    REQUIRE(catalog[i].dec() == Approx(dec_vec[i]).margin(1.e-7));
	}
    }
    SECTION("Check true separation is correct for each entry") {
	for (std::size_t i = 0; i < rt_vec.size(); i++) {
	    CAPTURE(i);
	    // Here I also require that it has true
	    REQUIRE(catalog[i].has_true());
	    REQUIRE(catalog[i].rt() == Approx(rt_vec[i]).margin(1.e-7));
	}
    }
    SECTION("Check observed separation is NAN for each entry") {
	for (std::size_t i = 0; i < ro_vec.size(); i++) {
	    CAPTURE(i);
	    // Here I also require that it doesn't have obs
	    REQUIRE_FALSE(catalog[i].has_obs());
	    REQUIRE(std::isnan(catalog[i].ro()));
	}
    }
}

TEST_CASE("Catalog auto-separations", "[seps][catalog][auto]") {
    // Read in a test catalog with three points
    // Note that this catalog has NAN for observed distances
    double rai, deci, rti, roi;
    std::vector<double> ra_vec, dec_vec, rt_vec, ro_vec;
    std::string line, nani;
    char* end;
    std::ifstream fin("test_data/catalog_three_objects.txt");
    if (fin.is_open()) {
	while (std::getline(fin, line)) {
	    if (line[0] == '#') continue;
	    std::istringstream iss(line);
	    if (!(iss >> rai >> deci >> rti >> nani)) FAIL("Unable to read test catalog file 'test_data/catalog_three_objects.txt'");
	    roi = std::strtod(nani.c_str(), &end);
	    ra_vec.push_back(rai);
	    dec_vec.push_back(deci);
	    rt_vec.push_back(rti);
	    ro_vec.push_back(roi);
	}
	fin.close();
    }
    else FAIL("Unable to open test catalog file 'test_data/catalog_three_objects.txt'");
    CAPTURE(ra_vec);
    CAPTURE(dec_vec);
    CAPTURE(rt_vec);
    CAPTURE(ro_vec);
    // Get the expected separations and IDs (individual calculations already tested)
    // Use a wide range of allowed separations here to keep all of them (only 3)
    std::tuple<double, double> r_perp_i, r_par_i;
    std::vector<std::vector<std::tuple<double, double> > > seps_expected;
    std::vector<std::vector<std::size_t>> ids_expected;
    INFO("Getting separations by hand");
    for (std::size_t i = 0; i < ra_vec.size() - 1; i++) {
	CAPTURE(i);
	CAPTURE(ra_vec[i]);
	CAPTURE(dec_vec[i]);
	CAPTURE(rt_vec[i]);
	CAPTURE(ro_vec[i]);
	Pos posi(ra_vec[i], dec_vec[i], rt_vec[i], ro_vec[i]);
	for (std::size_t j = i + 1; j < ra_vec.size(); j++) {
	    CAPTURE(j);
	    Pos posj(ra_vec[j], dec_vec[j], rt_vec[j], ro_vec[j]);
	    r_perp_i = r_perp(posi, posj);
	    if ((std::get<0>(r_perp_i) >= 0.0) && 
		(std::get<0>(r_perp_i) <= 1.e7)) {
		r_par_i = r_par(posi, posj);
		if ((std::get<0>(r_par_i) >= 0.0) && 
		    (std::get<0>(r_par_i) <= 1.e7)) {
		    seps_expected.push_back(std::vector<std::tuple<double, double>>{r_perp_i, r_par_i});
		    ids_expected.push_back(std::vector<std::size_t>{i, j});
		}
		else continue;
	    }
	    else continue;
	}
    }
    // Fill the catalog (already tested in another test case)
    INFO("Filling std::vector<Pos>");
    std::vector<Pos> catalog = fill_catalog_vector(ra_vec, dec_vec, rt_vec, ro_vec);
    // Get the separations for the catalog
    INFO("Getting catalog separations");
    VectorSeparation seps_result = get_separations(catalog, catalog, 0.0, 1.e7, 0.0, 1.e7, true, false, true);
    REQUIRE(seps_result.size() == seps_expected.size());
    for (std::size_t i = 0; i < seps_expected.size(); i++) {
	CAPTURE(i);
	REQUIRE(seps_result[i].id1 == ids_expected[i][0]);
	REQUIRE(seps_result[i].id2 == ids_expected[i][1]);
	REQUIRE(seps_result[i].r_perp_t == Approx(std::get<0>(seps_expected[i][0])));
	REQUIRE(seps_result[i].r_par_t == Approx(std::get<0>(seps_expected[i][1])));
	REQUIRE(std::isnan(seps_result[i].r_perp_o));
	REQUIRE(std::isnan(seps_result[i].r_par_o));
	REQUIRE(std::isnan(seps_result[i].ave_ro));
    }
    // Write to file for future python test
    std::ofstream fout("test_data/cpp_seps_catalog_three_objects.txt");
    if (fout.is_open()) {
	fout << "#R_PERP R_PAR ID1 ID2" << std::endl;
	fout << seps_result;
    }
    else FAIL("Unable to open output file 'test_data/cpp_seps_catalog_three_objects.txt'");
}
