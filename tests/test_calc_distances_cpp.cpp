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
#include <type_traits>
#include <algorithm>
#include <functional>
#include "external/catch.hpp"
#include "fast_math.h"
#include "calc_distances.h"
using namespace std::placeholders;

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

template<typename T, std::size_t N, typename Compare>
struct CompareMatcher
    : Catch::Matchers::Impl::MatcherBase<std::array<T,N>> {

    CompareMatcher(const std::array<T,N> &comparator, const Compare &compare, const std::array<T,N>& margin)
	: m_comparator(comparator), m_compare(compare), m_margin(margin) {}

    CompareMatcher(const std::array<T,N>& comparator, const Compare& compare, T margin)
	: m_comparator(comparator), m_compare(compare), m_margin(arrays::make_filled_array<N>(margin)) {}

    bool match(const std::array<T,N>& arr) const override {
	for (size_t i = 0; i < N; i++) {
	    if (!m_compare(m_comparator[i], arr[i], m_margin[i])) return false;
	}
	return true;
    }

    template<typename U, std::size_t M>
	typename std::enable_if_t<M != N || !std::is_same<U,T>::value, bool> match(const std::array<U,M>&) const { return false; }

    virtual std::string describe() const override {
        return "Equals: " + Catch::Detail::stringify(m_comparator);
    }

    const std::array<T,N>& m_comparator;
    Compare const &m_compare;
    const std::array<T,N>& m_margin;
};

template<typename T, std::size_t N, typename C>
CompareMatcher<T, N, C>
CompareArray(const std::array<T,N> &comparator, const C &compare, const std::array<T,N>& margin) {
    return CompareMatcher<T,N,C>(comparator, compare, margin);
}

template<typename T, std::size_t N, typename C>
CompareMatcher<T, N, C>
CompareArray(const std::array<T,N>& comparator, const C& compare, T margin) {
    return CompareMatcher<T,N,C>(comparator, compare, margin);
}

template<typename T, std::size_t N, typename std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
auto EqualsApprox(const std::array<T,N>& comparator, const std::array<T,N>& margin) {
    return CompareArray(comparator, [=](T actual, T expected, T slop) {
	    return actual == Approx(expected).margin(slop);
	}, margin);
}

template<typename T, std::size_t N, typename std::enable_if_t<std::is_integral<T>::value, int> = 0>
auto EqualsApprox(const std::array<T,N>& comparator, const std::array<T,N>) {
    return CompareArray(comparator, [=](T actual, T expected, T) {
	    return actual == expected;
	}, (T)0);
}

template<typename T, std::size_t N, typename std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
auto EqualsApprox(const std::array<T,N>& comparator, T margin) {
    return CompareArray(comparator, [=](T actual, T expected, T slop) {
	    return actual == Approx(expected).margin(slop);
	}, margin);
}

template<typename T, std::size_t N, typename std::enable_if_t<std::is_integral<T>::value, int> = 0>
auto EqualsApprox(const std::array<T,N>& comparator, T) {
    return CompareArray(comparator, [=](T actual, T expected, T) {
	    return actual == expected;
	}, (T)0);
}

template<typename T, std::size_t N, typename std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
auto EqualsApprox(const std::array<T,N>& comparator) {
    return CompareArray(comparator, [=](T actual, T expected, T slop) {
	    return actual == Approx(expected).margin(slop);
	}, (T)0);
}

template<typename T, std::size_t N, typename std::enable_if_t<std::is_integral<T>::value, int> = 0>
auto EqualsApprox(const std::array<T,N>& comparator) {
    return CompareArray(comparator, [=](T actual, T expected, T) {
	    return actual == expected;
	}, (T)0);
}

template<typename T, typename Compare>
struct VectorMatcher
    : Catch::Matchers::Impl::MatcherBase<std::vector<T>> {

    VectorMatcher(const std::vector<T> &comparator, const Compare &compare, const std::vector<T>& margin)
	: m_comparator(comparator), m_compare(compare), m_margin(margin) {}

    VectorMatcher(const std::vector<T>& comparator, const Compare& compare, T margin)
	: m_comparator(comparator), m_compare(compare), m_margin(std::vector<T>(m_comparator.size(), margin)) {}

    bool match(const std::vector<T>& vec) const override {
	if (vec.size() != m_comparator.size()) return false;
	auto cit = m_comparator.begin();
	auto mit = m_margin.begin();
	auto vit = vec.begin();
	for (; cit != m_comparator.end() && vit != vec.end(); ++mit, ++cit, ++vit) {
	    if (!m_compare(*cit, *vit, *mit)) return false;
	}
	return true;
    }

    virtual std::string describe() const override {
        return "Equals: " + Catch::Detail::stringify(m_comparator);
    }

    const std::vector<T> &m_comparator;
    Compare const &m_compare;
    const std::vector<T>& m_margin;
};

template<typename T, typename C>
VectorMatcher<T, C>
CompareVector(const std::vector<T> &comparator, const C &compare, const std::vector<T>& margin) {
    return VectorMatcher<T,C>(comparator, compare, margin);
}

template<typename T, typename C>
VectorMatcher<T, C>
CompareVector(const std::vector<T>& comparator, const C& compare, T margin) {
    return VectorMatcher<T,C>(comparator, compare, margin);
}

template<typename T>
auto AllStructsEqual(const std::vector<T> &comparator) {
    return CompareVector(comparator, [=](const T& actual, const T& expected, T()) {
        return actual == expected;
	}, T());
}

template<typename T, typename std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
auto AllClose(const std::vector<T>& comparator, std::vector<T> margin) {
    return CompareVector(comparator, [=](T actual, T expected, T slop) {
	    return actual == Approx(expected).margin(slop);
	}, margin);
}

template<typename T, typename std::enable_if_t<std::is_integral<T>::value, int> = 0>
auto AllClose(const std::vector<T>& comparator, std::vector<T>) {
    return CompareVector(comparator, [=](T actual, T expected, T) {
	    return actual == expected;
	}, (T)0);
}

template<typename T, typename std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
auto AllClose(const std::vector<T>& comparator, T margin) {
    return CompareVector(comparator, [=](T actual, T expected, T slop) {
	    return actual == Approx(expected).margin(slop);
	}, margin);
}

template<typename T, typename std::enable_if_t<std::is_integral<T>::value, int> = 0>
auto AllClose(const std::vector<T>& comparator, T) {
    return CompareVector(comparator, [=](T actual, T expected, T) {
	    return actual == expected;
	}, (T)0);
}

template<typename T, typename std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
auto AllClose(const std::vector<T>& comparator) {
    return CompareVector(comparator, [=](T actual, T expected, T slop) {
	    return actual == Approx(expected).margin(slop);
	}, (T)0);
}

template<typename T, typename std::enable_if_t<std::is_integral<T>::value, int> = 0>
auto AllClose(const std::vector<T>& comparator) {
    return CompareVector(comparator, [=](T actual, T expected, T) {
	    return actual == expected;
	}, (T)0);
}

template<typename T, typename Compare>
struct VectorApproxEquivalent
    : Catch::Matchers::Impl::MatcherBase<std::vector<T>> {

    VectorApproxEquivalent(const std::vector<T> &comparator, const Compare &compare, const std::vector<T>& margin)
	: m_comparator(comparator), m_compare(compare), m_margin(margin) {}

    VectorApproxEquivalent(const std::vector<T>& comparator, const Compare& compare, T margin)
	: m_comparator(comparator), m_compare(compare), m_margin(std::vector<T>(m_comparator.size(), margin)) {}

    bool match(const std::vector<T>& vec) const override {
	if (vec.size() != m_comparator.size()) return false;
	auto mit = m_margin.begin();
	auto cit = m_comparator.begin();
	for (; cit != m_comparator.end(); ++mit, ++cit) {
	    if (std::find_if(vec.begin(), vec.end(), std::bind(m_compare, _1, *cit, *mit)) == vec.end()) return false;
	}
	return true;
    }

    virtual std::string describe() const override {
        return "Equals up to order: " + Catch::Detail::stringify(m_comparator);
    }

    const std::vector<T> &m_comparator;
    Compare const &m_compare;
    const std::vector<T>& m_margin;
};

template<typename T, typename C>
VectorApproxEquivalent<T, C>
CompareOOOVector(const std::vector<T> &comparator, const C &compare, const std::vector<T>& margin) {
    return VectorApproxEquivalent<T,C>(comparator, compare, margin);
}

template<typename T, typename C>
VectorApproxEquivalent<T, C>
CompareOOOVector(const std::vector<T>& comparator, const C& compare, T margin) {
    return VectorApproxEquivalent<T,C>(comparator, compare, margin);
}

template<typename T>
auto AllStructEquivalent(const std::vector<T> &comparator) {
    return CompareOOOVector(comparator, [=](const T& actual, const T& expected, const T&) {
        return actual == expected;
	}, T());
}

template<typename T, typename std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
auto AllEquivalent(const std::vector<T>& comparator, const std::vector<T>& margin) {
    return CompareOOOVector(comparator, [=](T actual, T expected, T slop) {
	    return actual == Approx(expected).margin(slop);
	}, margin);
}

template<typename T, typename std::enable_if_t<std::is_integral<T>::value, int> = 0>
auto AllEquivalent(const std::vector<T>& comparator, const std::vector<T>&) {
    return CompareOOOVector(comparator, [=](T actual, T expected, T) {
	    return actual == expected;
	}, (T)0);
}

template<typename T, typename std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
auto AllEquivalent(const std::vector<T>& comparator, T margin) {
    return CompareOOOVector(comparator, [=](T actual, T expected, T slop) {
	    return actual == Approx(expected).margin(slop);
	}, margin);
}

template<typename T, typename std::enable_if_t<std::is_integral<T>::value, int> = 0>
auto AllEquivalent(const std::vector<T>& comparator, T) {
    return CompareOOOVector(comparator, [=](T actual, T expected, T) {
	    return actual == expected;
	}, (T)0);
}

template<typename T, typename std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
auto AllEquivalent(const std::vector<T>& comparator) {
    return CompareOOOVector(comparator, [=](T actual, T expected, T slop) {
	    return actual == Approx(expected).margin(slop);
	}, (T)0);
}

template<typename T, typename std::enable_if_t<std::is_integral<T>::value, int> = 0>
auto AllEquivalent(const std::vector<T>& comparator) {
    return CompareOOOVector(comparator, [=](T actual, T expected, T) {
	    return actual == expected;
	}, (T)0);
}

TEST_CASE("Single position vectors", "[spos][vectors]"){
    SECTION("Fail cases (bad angles or unit vectors") {
	SECTION("Angle related exceptions") {
	    SECTION("Only bad RA") {
		REQUIRE_THROWS_MATCHES([&](){ SPos pos(-1.0, 0.0, math::dnan, math::dnan); }(), std::invalid_argument, ExMsgContains("RA"));
		REQUIRE_THROWS_MATCHES([&](){ SPos pos(-0.00000005, 0.0, math::dnan, math::dnan); }(), std::invalid_argument, ExMsgContains("RA"));
		REQUIRE_THROWS_MATCHES([&](){ SPos pos(360.00000005, 0.0, math::dnan, math::dnan); }(), std::invalid_argument, ExMsgContains("RA"));
		REQUIRE_THROWS_MATCHES([&](){ SPos pos(361.0, 0.0, math::dnan, math::dnan); }(), std::invalid_argument, ExMsgContains("RA"));
		REQUIRE_THROWS_MATCHES([&](){ SPos pos(math::dnan, 0.0, math::dnan, math::dnan); }(), std::invalid_argument, ExMsgContains("RA"));
	    }
	    SECTION("Only bad DEC") {
		REQUIRE_THROWS_MATCHES([&](){ SPos pos(5.0, -91.0, math::dnan, math::dnan); }(), std::invalid_argument, ExMsgContains("DEC"));
		REQUIRE_THROWS_MATCHES([&](){ SPos pos(5.0, -90.00000005, math::dnan, math::dnan); }(), std::invalid_argument, ExMsgContains("DEC"));
		REQUIRE_THROWS_MATCHES([&](){ SPos pos(5.0, 90.00000005, math::dnan, math::dnan); }(), std::invalid_argument, ExMsgContains("DEC"));
		REQUIRE_THROWS_MATCHES([&](){ SPos pos(5.0, 91.0, math::dnan, math::dnan); }(), std::invalid_argument, ExMsgContains("DEC"));
		REQUIRE_THROWS_MATCHES([&](){ SPos pos(5.0, math::dnan, math::dnan, math::dnan); }(), std::invalid_argument, ExMsgContains("DEC"));
	    }
	    SECTION("Bad RA and DEC") {
		REQUIRE_THROWS_MATCHES([&](){ SPos pos(-1.0, math::dnan, math::dnan, math::dnan); }(), std::invalid_argument, ExMsgContains("RA"));
		REQUIRE_THROWS_MATCHES([&](){ SPos pos(math::dnan, -91.0, math::dnan, math::dnan); }(), std::invalid_argument, ExMsgContains("RA"));
		REQUIRE_THROWS_MATCHES([&](){ SPos pos(math::dnan, math::dnan, math::dnan, math::dnan); }(), std::invalid_argument, ExMsgContains("RA"));
	    }
	}
    }
    SECTION("Angle-unit vector conversion") {
	double dec1 = 0.0;
	double ra1 = 0.0;
	double dec2 = 0.0;
	double ra2 = 90.0;
	double dec3 = 90.0;
	double ra3 = 0.0;
	std::array<double,3> expected = {{1.0, 0.0, 0.0}};
	std::array<double,3> margin = {{0.0, 1.e-7, 1.e-7}};
	SPos pos1(ra1, dec1, math::dnan, math::dnan);
	REQUIRE_THAT(pos1.uvec(), EqualsApprox(expected, margin));
	std::rotate(expected.rbegin(), expected.rbegin() + 1, expected.rend());
	std::rotate(margin.rbegin(), margin.rbegin() + 1, margin.rend());
	SPos pos2(ra2, dec2, math::dnan, math::dnan);
	REQUIRE_THAT(pos2.uvec(), EqualsApprox(expected, margin));
	std::rotate(expected.rbegin(), expected.rbegin() + 1, expected.rend());
	std::rotate(margin.rbegin(), margin.rbegin() + 1, margin.rend());
	SPos pos3(ra3, dec3, math::dnan, math::dnan);
	REQUIRE_THAT(pos3.uvec(), EqualsApprox(expected, margin));
    }
};

TEST_CASE("Double position vectors", "[pos][vectors]"){
    SECTION("Fail cases (bad angles or unit vectors") {
	SECTION("Angle related exceptions") {
	    SECTION("Only bad RA") {
		REQUIRE_THROWS_MATCHES([&](){ Pos pos(-1.0, 0.0, math::dnan, math::dnan, math::dnan, math::dnan); }(), std::invalid_argument, ExMsgContains("RA"));
		REQUIRE_THROWS_MATCHES([&](){ Pos pos(-0.00000005, 0.0, math::dnan, math::dnan, math::dnan, math::dnan); }(), std::invalid_argument, ExMsgContains("RA"));
		REQUIRE_THROWS_MATCHES([&](){ Pos pos(360.00000005, 0.0, math::dnan, math::dnan, math::dnan, math::dnan); }(), std::invalid_argument, ExMsgContains("RA"));
		REQUIRE_THROWS_MATCHES([&](){ Pos pos(361.0, 0.0, math::dnan, math::dnan, math::dnan, math::dnan); }(), std::invalid_argument, ExMsgContains("RA"));
		REQUIRE_THROWS_MATCHES([&](){ Pos pos(math::dnan, 0.0, math::dnan, math::dnan, math::dnan, math::dnan); }(), std::invalid_argument, ExMsgContains("RA"));
	    }
	    SECTION("Only bad DEC") {
		REQUIRE_THROWS_MATCHES([&](){ Pos pos(5.0, -91.0, math::dnan, math::dnan, math::dnan, math::dnan); }(), std::invalid_argument, ExMsgContains("DEC"));
		REQUIRE_THROWS_MATCHES([&](){ Pos pos(5.0, -90.00000005, math::dnan, math::dnan, math::dnan, math::dnan); }(), std::invalid_argument, ExMsgContains("DEC"));
		REQUIRE_THROWS_MATCHES([&](){ Pos pos(5.0, 90.00000005, math::dnan, math::dnan, math::dnan, math::dnan); }(), std::invalid_argument, ExMsgContains("DEC"));
		REQUIRE_THROWS_MATCHES([&](){ Pos pos(5.0, 91.0, math::dnan, math::dnan, math::dnan, math::dnan); }(), std::invalid_argument, ExMsgContains("DEC"));
		REQUIRE_THROWS_MATCHES([&](){ Pos pos(5.0, math::dnan, math::dnan, math::dnan, math::dnan, math::dnan); }(), std::invalid_argument, ExMsgContains("DEC"));
	    }
	    SECTION("Bad RA and DEC") {
		REQUIRE_THROWS_MATCHES([&](){ Pos pos(-1.0, math::dnan, math::dnan, math::dnan, math::dnan, math::dnan); }(), std::invalid_argument, ExMsgContains("RA"));
		REQUIRE_THROWS_MATCHES([&](){ Pos pos(math::dnan, -91.0, math::dnan, math::dnan, math::dnan, math::dnan); }(), std::invalid_argument, ExMsgContains("RA"));
		REQUIRE_THROWS_MATCHES([&](){ Pos pos(math::dnan, math::dnan, math::dnan, math::dnan, math::dnan, math::dnan); }(), std::invalid_argument, ExMsgContains("RA"));
	    }
	}
    }
    SECTION("Setting or not setting distances") {
	SECTION("Neither given") {
	    Pos pos_radec(0.0, 0.0, math::dnan, math::dnan, math::dnan, math::dnan);
	    REQUIRE_FALSE(pos_radec.has_true());
	    REQUIRE_FALSE(pos_radec.has_obs());
	}
	SECTION("True distance given") {
	    Pos pos_radec(0.0, 0.0, 1.0, math::dnan, 1.0, math::dnan);
	    REQUIRE(pos_radec.has_true());
	    REQUIRE_FALSE(pos_radec.has_obs());
	}
	SECTION("Observed distance given") {
	    Pos pos_radec(0.0, 0.0, math::dnan, 1.0, math::dnan, 1.0);
	    REQUIRE_FALSE(pos_radec.has_true());
	    REQUIRE(pos_radec.has_obs());
	}
	SECTION("Both given") {
	    Pos pos_radec(0.0, 0.0, 1.0, 1.0, 1.0, 1.0);
	    REQUIRE(pos_radec.has_true());
	    REQUIRE(pos_radec.has_obs());
	}
    }
    SECTION("Angle-unit vector conversion") {
	double dec1 = 0.0;
	double ra1 = 0.0;
	double dec2 = 0.0;
	double ra2 = 90.0;
	double dec3 = 90.0;
	double ra3 = 0.0;
	std::array<double,3> expected = {{1.0, 0.0, 0.0}};
	std::array<double,3> margin = {{0.0, 1.e-7, 1.e-7}};
	Pos pos1(ra1, dec1, math::dnan, math::dnan, math::dnan, math::dnan);
	REQUIRE_THAT(pos1.uvec(), EqualsApprox(expected, margin));
	std::rotate(expected.rbegin(), expected.rbegin() + 1, expected.rend());
	std::rotate(margin.rbegin(), margin.rbegin() + 1, margin.rend());
	Pos pos2(ra2, dec2, math::dnan, math::dnan, math::dnan, math::dnan);
	REQUIRE_THAT(pos2.uvec(), EqualsApprox(expected, margin));
	std::rotate(expected.rbegin(), expected.rbegin() + 1, expected.rend());
	std::rotate(margin.rbegin(), margin.rbegin() + 1, margin.rend());
	Pos pos3(ra3, dec3, math::dnan, math::dnan, math::dnan, math::dnan);
	REQUIRE_THAT(pos3.uvec(), EqualsApprox(expected, margin));
    }
}

TEST_CASE("Dot products of vectors", "[dot][vectors]") {
    double unit_dot_exp = 0.23544059877913537;
    Pos pos1(32.5, -10.0, 3297.0, math::dnan, math::dnan, math::dnan);
    Pos pos2(108.2, 2.5, 5360.0, math::dnan, math::dnan, math::dnan);
    Pos upos1(32.5, -10.0, 1.0, math::dnan, math::dnan, math::dnan);
    Pos upos2(108.2, 2.5, 1.0, math::dnan, math::dnan, math::dnan);
    SPos spos1 = pos1.tpos();
    SPos spos2 = pos2.tpos();
    SPos supos1 = upos1.tpos();
    SPos supos2 = upos2.tpos();

    SECTION("Verify positions are as expected") {
	REQUIRE(spos1 == SPos(32.5, -10.0, 3297.0, math::dnan));
	REQUIRE_THAT(spos1.uvec(), EqualsApprox(pos1.uvec()));
	REQUIRE(spos2 == SPos(108.2, 2.5, 5360.0, math::dnan));
	REQUIRE(supos1 == SPos(32.5, -10.0, 1.0, math::dnan));
	REQUIRE(supos2 == SPos(108.2, 2.5, 1.0, math::dnan));
	REQUIRE_FALSE(pos1.has_obs());
	REQUIRE_FALSE(pos2.has_obs());
	REQUIRE_FALSE(upos1.has_obs());
	REQUIRE_FALSE(upos2.has_obs());
	REQUIRE(pos1.rt() == Approx(spos1.r()));
	REQUIRE(pos2.rt() == Approx(spos2.r()));
    }

    SECTION("Unit vector dot products for same vectors") {
	REQUIRE(supos1.dot_norm(supos1) == Approx(1.0));
	REQUIRE(supos2.dot_norm(supos2) == Approx(1.0));
	REQUIRE(upos1.dot_norm(upos1) == Approx(1.0));
	REQUIRE(upos2.dot_norm(upos2) == Approx(1.0));
        // The unit vector dot product shouldn't change if there is a magnitude
	REQUIRE(supos1.dot_norm(spos1) == Approx(1.0));
	REQUIRE(supos2.dot_norm(spos2) == Approx(1.0));
	REQUIRE(upos1.dot_norm(pos1) == Approx(1.0));
	REQUIRE(upos2.dot_norm(pos2) == Approx(1.0));
	// It shouldn't matter if we use SPos or Pos (but order matters!)
	REQUIRE(upos1.dot_norm(supos1) == Approx(1.0));
	REQUIRE(upos2.dot_norm(supos2) == Approx(1.0));
    }

    SECTION("Full dot products for same unit vectors") {
	// For these, we should get that the dot product is the same as the
	// dot product of the unit vectors (because they are unit vectors)
	REQUIRE(supos1.dot_mag(supos1) == Approx(1.0));
	REQUIRE(supos2.dot_mag(supos2) == Approx(1.0));
	REQUIRE(upos1.dot_mag(upos1) == Approx(1.0));
	REQUIRE(upos2.dot_mag(upos2) == Approx(1.0));
	// Also make sure it doesn't matter if it's SPos or Pos
	REQUIRE(upos1.dot_mag(supos1) == Approx(1.0));
	REQUIRE(upos2.dot_mag(supos2) == Approx(1.0));
    }

    SECTION("Full dot products for same vectors with magnitude") {
	// Unit vector and full vector: r
	REQUIRE(supos1.dot_mag(spos1) == Approx(spos1.r()));
	REQUIRE(supos2.dot_mag(spos2) == Approx(spos2.r()));
	REQUIRE(upos1.dot_mag(pos1) == Approx(spos1.r()));
	REQUIRE(upos2.dot_mag(pos2) == Approx(spos2.r()));
	REQUIRE(upos1.dot_mag(spos1) == Approx(spos1.r()));
	REQUIRE(upos2.dot_mag(spos2) == Approx(spos2.r()));
	// Full vector and full vector: r**2
	REQUIRE(spos1.dot_mag(spos1) == Approx(math::square(spos1.r())));
	REQUIRE(spos2.dot_mag(spos2) == Approx(math::square(spos2.r())));
	REQUIRE(pos1.dot_mag(pos1) == Approx(math::square(spos1.r())));
	REQUIRE(pos2.dot_mag(pos2) == Approx(math::square(spos2.r())));
	REQUIRE(pos1.dot_mag(spos1) == Approx(math::square(spos1.r())));
	REQUIRE(pos2.dot_mag(spos2) == Approx(math::square(spos2.r())));
    }

    SECTION("Different vector unit dot products") {
	REQUIRE(supos1.dot_norm(supos2) == Approx(unit_dot_exp));
	REQUIRE(supos2.dot_norm(supos1) == Approx(unit_dot_exp));
	REQUIRE(upos1.dot_norm(upos2) == Approx(unit_dot_exp));
	REQUIRE(upos2.dot_norm(upos1) == Approx(unit_dot_exp));
	REQUIRE(upos1.dot_norm(supos2) == Approx(unit_dot_exp));
	REQUIRE(upos2.dot_norm(supos1) == Approx(unit_dot_exp));
	REQUIRE(spos1.dot_norm(spos2) == Approx(unit_dot_exp));
	REQUIRE(spos2.dot_norm(spos1) == Approx(unit_dot_exp));
	REQUIRE(pos1.dot_norm(pos2) == Approx(unit_dot_exp));
	REQUIRE(pos2.dot_norm(pos1) == Approx(unit_dot_exp));
	REQUIRE(pos1.dot_norm(spos2) == Approx(unit_dot_exp));
	REQUIRE(pos2.dot_norm(spos1) == Approx(unit_dot_exp));
	REQUIRE(supos1.dot_norm(spos2) == Approx(unit_dot_exp));
	REQUIRE(spos1.dot_norm(supos2) == Approx(unit_dot_exp));
	REQUIRE(supos2.dot_norm(spos1) == Approx(unit_dot_exp));
	REQUIRE(spos2.dot_norm(supos1) == Approx(unit_dot_exp));
	REQUIRE(upos1.dot_norm(pos2) == Approx(unit_dot_exp));
	REQUIRE(pos1.dot_norm(upos2) == Approx(unit_dot_exp));
	REQUIRE(upos2.dot_norm(pos1) == Approx(unit_dot_exp));
	REQUIRE(pos2.dot_norm(upos1) == Approx(unit_dot_exp));
	REQUIRE(upos1.dot_norm(spos2) == Approx(unit_dot_exp));
	REQUIRE(pos1.dot_norm(supos2) == Approx(unit_dot_exp));
	REQUIRE(upos2.dot_norm(spos1) == Approx(unit_dot_exp));
	REQUIRE(pos2.dot_norm(supos1) == Approx(unit_dot_exp));
    }

    SECTION("Full dot product for different unit vectors") {
	REQUIRE(supos1.dot_mag(supos2) == Approx(unit_dot_exp));
	REQUIRE(supos2.dot_mag(supos1) == Approx(unit_dot_exp));
	REQUIRE(upos1.dot_mag(upos2) == Approx(unit_dot_exp));
	REQUIRE(upos2.dot_mag(upos1) == Approx(unit_dot_exp));
	REQUIRE(upos1.dot_mag(supos2) == Approx(unit_dot_exp));
	REQUIRE(upos2.dot_mag(supos1) == Approx(unit_dot_exp));
    }

    SECTION("Different vector dot products") {
	// Unit vector and full vector: r1 or r2
	REQUIRE(supos1.dot_mag(spos2) == Approx(spos2.r() * unit_dot_exp));
	REQUIRE(spos2.dot_mag(supos1) == Approx(spos2.r() * unit_dot_exp));
	REQUIRE(spos1.dot_mag(supos2) == Approx(spos1.r() * unit_dot_exp));
	REQUIRE(supos2.dot_mag(spos1) == Approx(spos1.r() * unit_dot_exp));
	REQUIRE(upos1.dot_mag(pos2) == Approx(spos2.r() * unit_dot_exp));
	REQUIRE(pos2.dot_mag(upos1) == Approx(spos2.r() * unit_dot_exp));
	REQUIRE(pos1.dot_mag(upos2) == Approx(spos1.r() * unit_dot_exp));
	REQUIRE(upos2.dot_mag(pos1) == Approx(spos1.r() * unit_dot_exp));
	REQUIRE(upos1.dot_mag(spos2) == Approx(spos2.r() * unit_dot_exp));
	REQUIRE(pos2.dot_mag(supos1) == Approx(spos2.r() * unit_dot_exp));
	REQUIRE(pos1.dot_mag(supos2) == Approx(spos1.r() * unit_dot_exp));
	REQUIRE(upos2.dot_mag(spos1) == Approx(spos1.r() * unit_dot_exp));
	// Full vector and full vector: r1 * r2
	REQUIRE(spos1.dot_mag(spos2) == Approx(spos1.r() * spos2.r() * unit_dot_exp));
	REQUIRE(spos2.dot_mag(spos1) == Approx(spos1.r() * spos2.r() * unit_dot_exp));
	REQUIRE(pos1.dot_mag(pos2) == Approx(spos1.r() * spos2.r() * unit_dot_exp));
	REQUIRE(pos2.dot_mag(pos1) == Approx(spos1.r() * spos2.r() * unit_dot_exp));
	REQUIRE(pos1.dot_mag(spos2) == Approx(spos1.r() * spos2.r() * unit_dot_exp));
	REQUIRE(pos2.dot_mag(spos1) == Approx(spos1.r() * spos2.r() * unit_dot_exp));
    }
}

TEST_CASE("True parallel separation", "[true][noobs][rparallel][seps][vectors]") {
    double u1 = 0.7859518429201421;
    double u2 = 0.7859518429201422;
    double r1 = 2591.2832261077083;
    double r2 = 4212.701878051963;
    // Set up vectors to test
    Pos pos1(32.5, -10.0, 3297.0, math::dnan, math::dnan, math::dnan);
    Pos pos2(108.2, 2.5, 5360.0, math::dnan, math::dnan, math::dnan);
    Pos upos1(32.5, -10.0, 1.0, math::dnan, math::dnan, math::dnan);
    Pos upos2(108.2, 2.5, 1.0, math::dnan, math::dnan, math::dnan);
    SPos spos1 = pos1.tpos();
    REQUIRE(spos1 == SPos(32.5, -10.0, 3297.0, math::dnan));
    SPos spos2 = pos2.tpos();
    REQUIRE(spos2 == SPos(108.2, 2.5, 5360.0, math::dnan));
    SPos supos1 = upos1.tpos();
    REQUIRE(supos1 == SPos(32.5, -10.0, 1.0, math::dnan));
    SPos supos2 = upos2.tpos();
    REQUIRE(supos2 == SPos(108.2, 2.5, 1.0, math::dnan));

    SECTION("Observed separations must be NaN") {
	REQUIRE(std::isnan(upos1.r_par_o(upos1)));
	REQUIRE(std::isnan(upos1.r_par_o(upos2)));
	REQUIRE(std::isnan(upos2.r_par_o(upos1)));
	REQUIRE(std::isnan(upos2.r_par_o(upos2)));
	REQUIRE(std::isnan(pos1.r_par_o(pos1)));
	REQUIRE(std::isnan(pos1.r_par_o(pos2)));
	REQUIRE(std::isnan(pos2.r_par_o(pos1)));
	REQUIRE(std::isnan(pos2.r_par_o(pos2)));
	REQUIRE(std::isnan(upos1.r_par_o(supos1)));
	REQUIRE(std::isnan(upos1.r_par_o(supos2)));
	REQUIRE(std::isnan(upos2.r_par_o(supos1)));
	REQUIRE(std::isnan(upos2.r_par_o(supos2)));
	REQUIRE(std::isnan(pos1.r_par_o(spos1)));
	REQUIRE(std::isnan(pos1.r_par_o(spos2)));
	REQUIRE(std::isnan(pos2.r_par_o(spos1)));
	REQUIRE(std::isnan(pos2.r_par_o(spos2)));
	REQUIRE(std::isnan(std::get<1>(pos1.distance_par(pos1))));
	REQUIRE(std::isnan(std::get<1>(pos1.distance_par(pos2))));
	REQUIRE(std::isnan(std::get<1>(upos1.distance_par(upos1))));
	REQUIRE(std::isnan(std::get<1>(upos1.distance_par(upos2))));
	REQUIRE(std::isnan(std::get<1>(pos2.distance_par(pos1))));
	REQUIRE(std::isnan(std::get<1>(pos2.distance_par(pos2))));
	REQUIRE(std::isnan(std::get<1>(upos2.distance_par(upos1))));
	REQUIRE(std::isnan(std::get<1>(upos2.distance_par(upos2))));
    }

    SECTION("Same vectors") {
	REQUIRE(supos1.distance_par(supos1) == Approx(0.0));
	REQUIRE(supos2.distance_par(supos2) == Approx(0.0));
	REQUIRE(spos1.distance_par(spos1) == Approx(0.0));
	REQUIRE(spos2.distance_par(spos2) == Approx(0.0));
	REQUIRE(upos1.r_par_t(upos1) == Approx(0.0));
	REQUIRE(upos2.r_par_t(upos2) == Approx(0.0));
	REQUIRE(pos1.r_par_t(pos1) == Approx(0.0));
	REQUIRE(pos2.r_par_t(pos2) == Approx(0.0));
	REQUIRE(upos1.r_par_t(supos1) == Approx(0.0));
	REQUIRE(upos2.r_par_t(supos2) == Approx(0.0));
	REQUIRE(pos1.r_par_t(spos1) == Approx(0.0));
	REQUIRE(pos2.r_par_t(spos2) == Approx(0.0));
	REQUIRE(std::get<0>(upos1.distance_par(upos1)) == Approx(0.0));
	REQUIRE(std::get<0>(upos2.distance_par(upos2)) == Approx(0.0));
	REQUIRE(std::get<0>(pos1.distance_par(pos1)) == Approx(0.0));
	REQUIRE(std::get<0>(pos2.distance_par(pos2)) == Approx(0.0));
    }

    SECTION("Unit vectors") {
	REQUIRE(supos1.distance_par(supos2) == Approx(std::fabs(u1 - u2)).margin(1.e-7));
	REQUIRE(supos2.distance_par(supos1) == Approx(std::fabs(u2 - u1)).margin(1.e-7));
	REQUIRE(upos1.r_par_t(upos2) == Approx(std::fabs(u1 - u2)).margin(1.e-7));
	REQUIRE(upos2.r_par_t(upos1) == Approx(std::fabs(u2 - u1)).margin(1.e-7));
	REQUIRE(upos1.r_par_t(supos2) == Approx(std::fabs(u1 - u2)).margin(1.e-7));
	REQUIRE(upos2.r_par_t(supos1) == Approx(std::fabs(u2 - u1)).margin(1.e-7));
	REQUIRE(std::get<0>(upos1.distance_par(upos2)) == Approx(std::fabs(u1 - u2)).margin(1.e-7));
	REQUIRE(std::get<0>(upos2.distance_par(upos1)) == Approx(std::fabs(u2 - u1)).margin(1.e-7));
    }

    SECTION("Full vectors") {
	REQUIRE(spos1.distance_par(spos2) == Approx(std::fabs(r1 - r2)).margin(1.e-7));
	REQUIRE(spos2.distance_par(spos1) == Approx(std::fabs(r2 - r1)).margin(1.e-7));
	REQUIRE(pos1.r_par_t(pos2) == Approx(std::fabs(r1 - r2)).margin(1.e-7));
	REQUIRE(pos2.r_par_t(pos1) == Approx(std::fabs(r2 - r1)).margin(1.e-7));
	REQUIRE(pos1.r_par_t(spos2) == Approx(std::fabs(r1 - r2)).margin(1.e-7));
	REQUIRE(pos2.r_par_t(spos1) == Approx(std::fabs(r2 - r1)).margin(1.e-7));
	REQUIRE(std::get<0>(pos1.distance_par(pos2)) == Approx(std::fabs(r1 - r2)).margin(1.e-7));
	REQUIRE(std::get<0>(pos2.distance_par(pos1)) == Approx(std::fabs(r2 - r1)).margin(1.e-7));
    }
}

TEST_CASE("Parallel separation", "[true][obs][rparallel][seps][vectors]") {
    // Set up vectors to test
    Pos pos1(32.5, -10.0, 3297.0, 3297.0, math::dnan, math::dnan);
    Pos pos2(108.2, 2.5, 5360.0, 5360.0, math::dnan, math::dnan);
    Pos pos3(32.5, -10.0, 3297.0, 5360.0, math::dnan, math::dnan);
    Pos pos4(108.2, 2.5, 5360.0, 3297.0, math::dnan, math::dnan);
    Pos upos1(32.5, -10.0, 1.0, 1.0, math::dnan, math::dnan);
    Pos upos2(108.2, 2.5, 1.0, 1.0, math::dnan, math::dnan);
    REQUIRE(pos1.tpos() == pos3.tpos());
    REQUIRE(pos2.tpos() == pos4.tpos());
    REQUIRE(upos1.tpos() == upos1.opos());
    REQUIRE(upos2.tpos() == upos2.opos());

    SECTION("Observed separations with the same positions should all be 0") {
	SECTION("Same unit vectors") {
	    REQUIRE(upos1.r_par_o(upos1) == Approx(0.0));
	    REQUIRE(upos2.r_par_o(upos2) == Approx(0.0));
	}

	SECTION("Same full vectors") {
	    REQUIRE(pos1.r_par_o(pos1) == Approx(0.0));
	    REQUIRE(pos2.r_par_o(pos2) == Approx(0.0));
	    REQUIRE(pos3.r_par_o(pos3) == Approx(0.0));
	    REQUIRE(pos4.r_par_o(pos4) == Approx(0.0));
	}

	SECTION("Same full vectors from tuple") {
	    REQUIRE(std::get<1>(upos1.distance_par(upos1)) == Approx(0.0));
	    REQUIRE(std::get<1>(upos2.distance_par(upos2)) == Approx(0.0));
	    REQUIRE(std::get<1>(pos1.distance_par(pos1)) == Approx(0.0));
	    REQUIRE(std::get<1>(pos2.distance_par(pos2)) == Approx(0.0));
	    REQUIRE(std::get<1>(pos3.distance_par(pos3)) == Approx(0.0));
	    REQUIRE(std::get<1>(pos4.distance_par(pos4)) == Approx(0.0));
	}
    }

    SECTION("Unit vectors") {
	// If the previous test case passed, I can already trust r_par_t
	SECTION("True and observed separations should be the same") {
	    REQUIRE(upos1.r_par_o(upos2) == Approx(upos1.r_par_t(upos2)));
	    REQUIRE(upos2.r_par_o(upos1) == Approx(upos2.r_par_t(upos1)));
	}

	SECTION("True separations should be positive") {
	    REQUIRE(upos1.r_par_t_signed(upos2) == Approx(upos1.r_par_t(upos2)));
	    REQUIRE(upos2.r_par_t_signed(upos1) == Approx(upos2.r_par_t(upos1)));
	}

	SECTION("Observed separations from tuples should work") {
	    // Also check the tuple results
	    auto u1u2 = upos1.distance_par(upos2);
	    auto u2u1 = upos2.distance_par(upos1);
	    REQUIRE(std::get<0>(u1u2) == Approx(upos1.r_par_t(upos2)));
	    REQUIRE(std::get<1>(u1u2) == Approx(upos1.r_par_t(upos2)));
	    REQUIRE(std::get<0>(u2u1) == Approx(upos2.r_par_t(upos1)));
	    REQUIRE(std::get<1>(u2u1) == Approx(upos2.r_par_t(upos1)));
	}
    }

    SECTION("Full vectors") {
	SECTION("Same true and observed distance") {
	    REQUIRE(pos1.r_par_o(pos2) == Approx(pos1.r_par_t(pos2)));
	    REQUIRE(pos2.r_par_o(pos1) == Approx(pos2.r_par_t(pos1)));
	    // Note this is also important: the order shouldn't matter observed
	    REQUIRE(pos2.r_par_o(pos1) == Approx(pos1.r_par_o(pos2)));

	    SECTION("From tuples") {
		REQUIRE(std::get<1>(pos1.distance_par(pos2)) == Approx(pos1.r_par_t(pos2)));
		REQUIRE(std::get<1>(pos2.distance_par(pos1)) == Approx(pos2.r_par_t(pos1)));
	    }
	}

	SECTION("Switched true and observed distances still have same absolute value of dot product") {
	    REQUIRE(pos3.r_par_o(pos4) == Approx(pos1.r_par_t(pos2)));
	    REQUIRE(pos4.r_par_o(pos3) == Approx(pos2.r_par_t(pos1)));
	    REQUIRE(pos4.r_par_o(pos3) == Approx(pos3.r_par_o(pos4)));

	    SECTION("From tuples") {
		REQUIRE(std::get<1>(pos3.distance_par(pos4)) == Approx(pos1.r_par_t(pos2)));
		REQUIRE(std::get<1>(pos4.distance_par(pos3)) == Approx(pos2.r_par_t(pos1)));
	    }
	}
    }

    SECTION("Sign of true separation") {
	SECTION("Same true and observed distances") {
	    REQUIRE(pos1.r_par_t_signed(pos2) == Approx(pos1.r_par_t(pos2)));
	    REQUIRE(pos2.r_par_t_signed(pos1) == Approx(pos2.r_par_t(pos1)));

	    SECTION("From tuples") {
		REQUIRE(std::get<0>(pos1.distance_par(pos2)) == Approx(pos1.r_par_t(pos2)));
		REQUIRE(std::get<0>(pos2.distance_par(pos1)) == Approx(pos2.r_par_t(pos1)));
	    }
	}

	SECTION("Switched true and observed distances") {
	    REQUIRE(pos3.r_par_t_signed(pos4) == Approx(-pos3.r_par_t(pos4)));
	    REQUIRE(pos4.r_par_t_signed(pos3) == Approx(-pos4.r_par_t(pos3)));

	    SECTION("From tuples: same sign as signed") {
		REQUIRE(std::get<0>(pos3.distance_par(pos4)) == Approx(pos3.r_par_t_signed(pos4)));
		REQUIRE(std::get<0>(pos4.distance_par(pos3)) == Approx(pos4.r_par_t_signed(pos3)));
	    }
	}
    }
}

TEST_CASE("Perpendicular separation", "[rperpendicular][seps][vectors]") {
    // Set up vectors to test
    Pos pos1(32.5, -10.0, 3297.0, math::dnan, math::dnan, math::dnan);
    Pos pos2(108.2, 2.5, 5360.0, math::dnan, math::dnan, math::dnan);
    Pos pos3(32.5, -10.0, 3297.0, 3297.0, math::dnan, math::dnan);
    Pos pos4(108.2, 2.5, 5360.0, 5360.0, math::dnan, math::dnan);
    Pos pos5(32.5, -10.0, 3297.0, 5360.0, math::dnan, math::dnan);
    Pos pos6(108.2, 2.5, 5360.0, 3297.0, math::dnan, math::dnan);
    Pos upos1(32.5, -10.0, 1.0, math::dnan, math::dnan, math::dnan);
    Pos upos2(108.2, 2.5, 1.0, math::dnan, math::dnan, math::dnan);
    Pos upos3(32.5, -10.0, 1.0, 1.0, math::dnan, math::dnan);
    Pos upos4(108.2, 2.5, 1.0, 1.0, math::dnan, math::dnan);
    SPos spos1 = pos1.tpos();
    REQUIRE(spos1 == SPos(32.5, -10.0, 3297.0, math::dnan));
    SPos spos2 = pos2.tpos();
    REQUIRE(spos2 == SPos(108.2, 2.5, 5360.0, math::dnan));
    SPos supos1 = upos1.tpos();
    REQUIRE(supos1 == SPos(32.5, -10.0, 1.0, math::dnan));
    SPos supos2 = upos2.tpos();
    REQUIRE(supos2 == SPos(108.2, 2.5, 1.0, math::dnan));
    SPos supos3 = upos1.opos();
    REQUIRE(supos3 == SPos(32.5, -10.0, math::dnan, math::dnan));
    SPos supos4 = upos2.opos();
    REQUIRE(supos4 == SPos(108.2, 2.5, math::dnan, math::dnan));
    double uu = 1.2365754333811299;
    double rr = 5352.516763390221;
    double rr2 = 6628.044322922857;

    SECTION("NaN's for missing observed distances") {
	// I don't need to test all, these should cover all of the valid combos
	// of NaN with NaN, NaN with not NaN, not NaN with NaN, and self
	REQUIRE(std::isnan(upos1.r_perp_o(upos1)));
	REQUIRE(std::isnan(upos1.r_perp_o(upos2)));
	REQUIRE(std::isnan(upos1.r_perp_o(upos3)));
	REQUIRE(std::isnan(upos1.r_perp_o(upos4)));
	REQUIRE(std::isnan(upos3.r_perp_o(upos1)));
	REQUIRE(std::isnan(upos4.r_perp_o(upos1)));
	// Also check tuples
	REQUIRE(std::isnan(std::get<1>(upos1.distance_perp(upos1))));
	REQUIRE(std::isnan(std::get<1>(upos1.distance_perp(upos2))));
	REQUIRE(std::isnan(std::get<1>(upos1.distance_perp(upos3))));
	REQUIRE(std::isnan(std::get<1>(upos1.distance_perp(upos4))));
	REQUIRE(std::isnan(std::get<1>(upos3.distance_perp(upos1))));
	REQUIRE(std::isnan(std::get<1>(upos4.distance_perp(upos1))));
	// Now, check what happens when we use SPos as other
	REQUIRE(std::isnan(upos1.r_perp_o(supos1)));
	REQUIRE(std::isnan(upos1.r_perp_o(supos2)));
	REQUIRE(std::isnan(upos1.r_perp_o(supos3)));
	REQUIRE(std::isnan(upos1.r_perp_o(supos4)));
	REQUIRE(std::isnan(upos3.r_perp_o(supos3)));
	REQUIRE(std::isnan(upos4.r_perp_o(supos3)));
	// Lastly, check NaN combos with SPos function
	REQUIRE(std::isnan(supos1.distance_perp(supos3)));
	REQUIRE(std::isnan(supos1.distance_perp(supos4)));
	REQUIRE(std::isnan(supos3.distance_perp(supos1)));
	REQUIRE(std::isnan(supos3.distance_perp(supos2)));
	REQUIRE(std::isnan(supos3.distance_perp(supos3)));
	REQUIRE(std::isnan(supos3.distance_perp(supos4)));
    }

    SECTION("Same angles") {
	REQUIRE(supos1.distance_perp(supos1) == Approx(0.0));
	REQUIRE(spos1.distance_perp(spos1) == Approx(0.0));
	REQUIRE(spos1.distance_perp(supos1) == Approx(0.0));
	REQUIRE(upos1.r_perp_t(supos1) == Approx(0.0));
	REQUIRE(upos1.r_perp_t(spos1) == Approx(0.0));
	REQUIRE(upos3.r_perp_t(supos1) == Approx(0.0));
	REQUIRE(upos3.r_perp_o(supos1) == Approx(0.0));
	REQUIRE(upos1.r_perp_t(upos1) == Approx(0.0));
	REQUIRE(upos3.r_perp_t(upos3) == Approx(0.0));
	REQUIRE(upos3.r_perp_o(upos3) == Approx(0.0));
	REQUIRE(upos1.r_perp_t(upos3) == Approx(0.0));
	REQUIRE(upos1.r_perp_t(pos1) == Approx(0.0));
	REQUIRE(upos3.r_perp_t(pos3) == Approx(0.0));
	REQUIRE(upos3.r_perp_o(pos3) == Approx(0.0));
	REQUIRE(upos3.r_perp_t(pos5) == Approx(0.0));
	REQUIRE(upos3.r_perp_o(pos5) == Approx(0.0));
	REQUIRE(pos3.r_perp_t(pos5) == Approx(0.0));
	REQUIRE(pos3.r_perp_o(pos5) == Approx(0.0));
	REQUIRE(pos5.r_perp_t(pos3) == Approx(0.0));
	REQUIRE(pos5.r_perp_o(pos3) == Approx(0.0));
	REQUIRE(std::get<0>(upos1.distance_perp(upos1)) == Approx(0.0));
	REQUIRE(std::get<0>(upos3.distance_perp(upos3)) == Approx(0.0));
	REQUIRE(std::get<1>(upos3.distance_perp(upos3)) == Approx(0.0));
	REQUIRE(std::get<0>(upos1.distance_perp(upos3)) == Approx(0.0));
	REQUIRE(std::get<0>(pos3.distance_perp(pos5)) == Approx(0.0));
	REQUIRE(std::get<1>(pos3.distance_perp(pos5)) == Approx(0.0));
	REQUIRE(std::get<0>(pos5.distance_perp(pos3)) == Approx(0.0));
	REQUIRE(std::get<1>(pos5.distance_perp(pos3)) == Approx(0.0));
    }

    SECTION("Unit vectors") {
	REQUIRE(supos1.distance_perp(supos2) == Approx(uu));
	REQUIRE(supos2.distance_perp(supos1) == Approx(supos1.distance_perp(supos2)));
	REQUIRE(upos1.r_perp_t(supos2) == Approx(uu));
	REQUIRE(upos1.r_perp_t(upos2) == Approx(uu));
	REQUIRE(upos2.r_perp_t(upos1) == Approx(upos1.r_perp_t(upos2)));
	REQUIRE(upos3.r_perp_t(supos2) == Approx(uu));
	REQUIRE(upos3.r_perp_o(supos2) == Approx(uu));
	REQUIRE(upos3.r_perp_t(upos4) == Approx(upos1.r_perp_t(upos2)));
	REQUIRE(upos3.r_perp_o(upos4) == Approx(uu));
	REQUIRE(upos4.r_perp_o(upos3) == Approx(upos3.r_perp_o(upos4)));
	REQUIRE(std::get<0>(upos1.distance_perp(upos2)) == Approx(upos1.r_perp_t(upos2)));
	auto tup_dist = upos3.distance_perp(upos4);
	REQUIRE(std::get<0>(tup_dist) == Approx(upos3.r_perp_t(upos4)));
	REQUIRE(std::get<1>(tup_dist) == Approx(upos3.r_perp_o(upos4)));
    }

    SECTION("Full vectors") {
	REQUIRE(spos1.distance_perp(spos2) == Approx(rr));
	REQUIRE(spos2.distance_perp(spos1) == Approx(spos1.distance_perp(spos2)));
	REQUIRE(pos1.r_perp_t(spos2) == Approx(rr));
	REQUIRE(pos2.r_perp_t(spos1) == Approx(rr));
	REQUIRE(pos3.r_perp_t(spos2) == Approx(rr));
	REQUIRE(pos3.r_perp_o(spos2) == Approx(rr));
	REQUIRE(pos5.r_perp_t(spos2) == Approx(rr));
	REQUIRE(pos5.r_perp_o(spos2) == Approx(rr2));
	REQUIRE(pos1.r_perp_t(pos2) == Approx(rr));
	REQUIRE(pos2.r_perp_t(pos1) == Approx(rr));
	REQUIRE(pos3.r_perp_t(pos2) == Approx(rr));
	REQUIRE(pos3.r_perp_t(pos4) == Approx(rr));
	REQUIRE(pos3.r_perp_o(pos4) == Approx(pos3.r_perp_t(pos4)));
	REQUIRE(pos4.r_perp_t(pos3) == Approx(pos3.r_perp_t(pos4)));
	REQUIRE(pos4.r_perp_o(pos3) == Approx(pos3.r_perp_o(pos4)));
	REQUIRE(pos5.r_perp_t(pos6) == Approx(pos3.r_perp_t(pos4)));
	REQUIRE(pos5.r_perp_o(pos6) == Approx(pos3.r_perp_o(pos4)));
	REQUIRE(std::get<0>(pos1.distance_perp(pos2)) == Approx(pos1.r_perp_t(pos2)));
	auto tup_34 = pos3.distance_perp(pos4);
	REQUIRE(std::get<0>(tup_34) == Approx(pos3.r_perp_t(pos4)));
	REQUIRE(std::get<1>(tup_34) == Approx(pos3.r_perp_o(pos4)));
	auto tup_56 = pos5.distance_perp(pos6);
	REQUIRE(std::get<0>(tup_56) == Approx(std::get<0>(tup_34)));
	REQUIRE(std::get<1>(tup_56) == Approx(std::get<1>(tup_34)));
    }
}

TEST_CASE("Single position catalog single auto-separations", "[VecSPos][seps][sseps][catalog][auto]") {
    // Read in a test catalog with three points
    double rai, deci, rti, zti, roi, zoi;
    std::vector<SPos> tcat, ocat;
    std::string line;
    std::ifstream fin("test_data/catalog_three_objects_v2.txt");
    if (fin.is_open()) {
        while (std::getline(fin, line)) {
            if (line[0] == '#') continue;
            std::istringstream iss(line);
            if (!(iss >> rai >> deci >> rti >> roi >> zti >> zoi)) FAIL("Unable to read test catalog file 'test_data/catalog_three_objects_v2.txt'");
	    tcat.push_back(SPos(rai, deci, rti, zti));
	    ocat.push_back(SPos(rai, deci, roi, zoi));
        }
        fin.close();
    }
    else FAIL("Unable to open test catalog file 'test_data/catalog_three_objects_v2.txt'");
    fin.clear();

    // Get the expected separations and IDs for true and observed
    std::vector<double> r_perp_t, r_par_t, zbar_t;
    std::vector<std::size_t> id1, id2;
    double r_perp, r_par, zbar;
    std::size_t i1, i2;
    std::vector<Separation> tvs_expected, ovs_expected;
    //VectorSeparation tvs_expected, ovs_expected;
    fin.open("test_data/expected_single_seps_true_catalog_three_objects_v2.txt");
    if (fin.is_open()) {
	while(std::getline(fin, line)) {
	    if (line[0] == '#') continue;
	    std::istringstream iss(line);
	    if (!(iss >> r_perp >> r_par >> zbar >> i1 >> i2)) FAIL("Unable to read expected separations from file 'test_data/expected_single_seps_true_catalog_three_objects_v2.txt'");
	    r_perp_t.push_back(r_perp);
	    r_par_t.push_back(r_par);
	    zbar_t.push_back(zbar);
	    id1.push_back(i1);
	    id2.push_back(i2);
	    tvs_expected.push_back(Separation(r_perp, r_par, zbar, i1, i2));
	}
	fin.close();
    }
    else FAIL("Unable ro open expected separations file 'test_data/expected_single_seps_true_catalog_three_objects_v2.txt'");
    fin.clear();

    SECTION("Compare vector<Separation> to vectors of separations") {
	REQUIRE(tvs_expected.size() == r_perp_t.size());
	for (std::size_t i = 0; i < r_perp_t.size(); i++) {
	    REQUIRE(tvs_expected[i].r_perp == Approx(r_perp_t[i]));
	    REQUIRE(tvs_expected[i].r_par == Approx(r_par_t[i]));
	    REQUIRE(tvs_expected[i].zbar == Approx(zbar_t[i]));
	    REQUIRE(tvs_expected[i].id1 == id1[i]);
	    REQUIRE(tvs_expected[i].id2 == id2[i]);
	    /*
	    REQUIRE(tvs_expected.r_perp()[i] == Approx(r_perp_t[i]));
	    REQUIRE(tvs_expected.r_par()[i] == Approx(r_par_t[i]));
	    REQUIRE(tvs_expected.zbar()[i] == Approx(zbar_t[i]));
	    REQUIRE(tvs_expected.id1()[i] == id1[i]);
	    REQUIRE(tvs_expected.id2()[i] == id2[i]);
	    */
	    REQUIRE(tvs_expected[i] == Separation(r_perp_t[i], r_par_t[i], zbar_t[i], id1[i], id2[i]));
	}
    }

    /*
    SECTION("Compare filling in loop to constructing from arrays") {
	VectorSeparation from_vecs(r_perp_t, r_par_t, zbar_t, id1, id2);
	REQUIRE(from_vecs == tvs_expected);
    }
    */

    fin.open("test_data/expected_single_seps_obs_catalog_three_objects_v2.txt");
    if (fin.is_open()) {
	while(std::getline(fin, line)) {
	    if (line[0] == '#') continue;
	    std::istringstream iss(line);
	    if (!(iss >> r_perp >> r_par >> zbar >> i1 >> i2)) FAIL("Unable to read expected separations from file 'test_data/expected_single_seps_true_catalog_three_objects_v2.txt'");
	    ovs_expected.push_back(Separation(r_perp, r_par, zbar, i1, i2));
	}
	fin.close();
    }
    else FAIL("Unable ro open expected separations file 'test_data/expected_single_seps_obs_catalog_three_objects_v2.txt'");
    fin.clear();

    // Get the separations for the catalog
    INFO("Getting catalog separations");
    BinSpecifier rp_bins, rl_bins;
    rp_bins.set_bin_min(0.0);
    rp_bins.set_bin_max(1.e7);
    rl_bins.set_bin_min(0.0);
    rl_bins.set_bin_max(1.e7);
    // specifically use auto-correlations
    std::vector<Separation> tvs_auto_result = get_auto_separations(tcat, rp_bins, rl_bins);
    //VectorSeparation tvs_result = get_separations(tcat, tcat, rp_bins, rl_bins, true);
    REQUIRE_THAT(tvs_auto_result, AllStructEquivalent(tvs_expected));
    std::vector<Separation> tvs_result = get_separations(tcat, tcat, rp_bins, rl_bins, true);
    REQUIRE_THAT(tvs_result, AllStructEquivalent(tvs_expected));
    /*
    REQUIRE(tvs_result.size() == tvs_expected.size());
    INFO("Match results by hand");
    bool match_found;
    for (std::size_t i = 0; i < tvs_result.size(); i++) {
	CAPTURE(i);
	CAPTURE("Finding matching pair in expected");
	match_found = false;
	for (std::size_t j = 0; j < tvs_expected.size(); j++) {
	    if (tvs_result[j].id1 == tvs_expected[i].id1 && tvs_result[j].id2 == tvs_expected[i].id2) {
		match_found = true;
		REQUIRE(tvs_result[j].r_perp == Approx(tvs_expected[i].r_perp));
		REQUIRE(tvs_result[j].r_par == Approx(tvs_expected[i].r_par));
		REQUIRE(tvs_result[j].zbar == Approx(tvs_expected[i].zbar));
		// Now test to see if operator== works for Separation objects
		REQUIRE(tvs_result[j] == tvs_expected[i]);
            }
            else continue;
        }
        if (!match_found) {
            CAPTURE(tvs_result[i]);
            FAIL("No match found in output");
        }
    }
    INFO("Test VectorSeparation operator==");
    REQUIRE(tvs_result == tvs_expected);
    */

    std::vector<Separation> ovs_result = get_auto_separations(ocat, rp_bins, rl_bins);
    REQUIRE_THAT(ovs_result, AllStructEquivalent(ovs_expected));
    //VectorSeparation ovs_result = get_separations(ocat, ocat, rp_bins, rl_bins, true);
    //REQUIRE(ovs_result == ovs_expected);
}

TEST_CASE("Both position catalog single auto-separations", "[VecPos][seps][sseps][catalog][auto]") {
    // Read in a test catalog with three points
    double rai, deci, rti, zti, roi, zoi;
    std::vector<Pos> cat;
    std::string line;
    std::ifstream fin("test_data/catalog_three_objects_v2.txt");
    if (fin.is_open()) {
        while (std::getline(fin, line)) {
            if (line[0] == '#') continue;
            std::istringstream iss(line);
            if (!(iss >> rai >> deci >> rti >> roi >> zti >> zoi)) FAIL("Unable to read test catalog file 'test_data/catalog_three_objects_v2.txt'");
	    cat.push_back(Pos(rai, deci, rti, roi, zti, zoi));
        }
        fin.close();
    }
    else FAIL("Unable to open test catalog file 'test_data/catalog_three_objects_v2.txt'");
    fin.clear();

    // Get the expected separations and IDs
    double r_perp, r_par, zbar;
    std::size_t i1, i2;
    //VectorSeparation tvs_expected, ovs_expected;
    std::vector<Separation> tvs_expected, ovs_expected;
    fin.open("test_data/expected_single_seps_true_catalog_three_objects_v2.txt");
    if (fin.is_open()) {
	while (std::getline(fin, line)) {
	    if (line[0] == '#') continue;
	    std::istringstream iss(line);
	    if (!(iss >> r_perp >> r_par >> zbar >> i1 >> i2)) FAIL("Unable to read expected separations from file 'test_data/expected_single_seps_true_catalog_three_objects_v2.txt'");
	    tvs_expected.push_back(Separation(r_perp, r_par, zbar, i1, i2));
	}
	fin.close();
    }
    else FAIL("Unable to open expected separations file 'test_data/expected_single_seps_true_catalog_three_objects_v2.txt'");
    fin.clear();
    fin.open("test_data/expected_single_seps_obs_catalog_three_objects_v2.txt");
    if (fin.is_open()) {
	while (std::getline(fin, line)) {
	    if (line[0] == '#') continue;
	    std::istringstream iss(line);
	    if (!(iss >> r_perp >> r_par >> zbar >> i1 >> i2)) FAIL("Unable to read expected separations from file 'test_data/expected_single_seps_obs_catalog_three_objects_v2.txt'");
	    ovs_expected.push_back(Separation(r_perp, r_par, zbar, i1, i2));
	}
	fin.close();
    }
    else FAIL("Unable to open expected separations file 'test_data/expected_single_seps_obs_catalog_three_objects_v2.txt'");
    fin.clear();

    // Get the separations for the catalog
    INFO("Getting catalog separations");
    BinSpecifier rp_bins, rl_bins;
    rp_bins.set_bin_min(0.0);
    rp_bins.set_bin_max(1.e7);
    rl_bins.set_bin_min(0.0);
    rl_bins.set_bin_max(1.e7);
    std::vector<Separation> tvs_result = get_auto_separations(cat, rp_bins, rl_bins, true);
    REQUIRE_THAT(tvs_result, AllStructEquivalent(tvs_expected));
    std::vector<Separation> ovs_result = get_auto_separations(cat, rp_bins, rl_bins, false);
    REQUIRE_THAT(ovs_result, AllStructEquivalent(ovs_expected));
    /*
    VectorSeparation tvs_result = get_separations(cat, cat, rp_bins, rl_bins, true, true);
    REQUIRE(tvs_result == tvs_expected);
    VectorSeparation ovs_result = get_separations(cat, cat, rp_bins, rl_bins, false, true);
    REQUIRE(ovs_result == ovs_expected);
    */
}

TEST_CASE("Both position catalog both auto-separations", "[VecPos][seps][bseps][catalog][auto]") {
    // Read in a test catalog with three points
    double rai, deci, rti, zti, roi, zoi;
    std::vector<Pos> cat;
    std::string line;
    std::ifstream fin("test_data/catalog_three_objects_v2.txt");
    if (fin.is_open()) {
        while (std::getline(fin, line)) {
            if (line[0] == '#') continue;
            std::istringstream iss(line);
            if (!(iss >> rai >> deci >> rti >> roi >> zti >> zoi)) FAIL("Unable to read test catalog file 'test_data/catalog_three_objects_v2.txt'");
	    cat.push_back(Pos(rai, deci, rti, roi, zti, zoi));
        }
        fin.close();
    }
    else FAIL("Unable to open test catalog file 'test_data/catalog_three_objects_v2.txt'");
    fin.clear();

    // Get the expected separations and IDs
    std::vector<double> r_perp_t, r_par_t, zbar_t, r_perp_o, r_par_o, zbar_o;
    std::vector<std::size_t> id1, id2;
    double rpt, rlt, zbt, rpo, rlo, zbo;
    std::size_t i1, i2;
    std::vector<TOSeparation> vts_expected;
    //VectorTOSeparation vts_expected;
    fin.open("test_data/expected_both_seps_catalog_three_objects_v2.txt");
    if (fin.is_open()) {
	while (std::getline(fin, line)) {
	    if (line[0] == '#') continue;
	    std::istringstream iss(line);
	    if (!(iss >> rpt >> rlt >> zbt >> rpo >> rlo >> zbo >> i1 >> i2)) FAIL("Unable to read expected separations from file 'test_data/expected_both_seps_catalog_three_objects_v2.txt'");
	    r_perp_t.push_back(rpt);
	    r_par_t.push_back(rlt);
	    zbar_t.push_back(zbt);
	    r_perp_o.push_back(rpo);
	    r_par_o.push_back(rlo);
	    zbar_o.push_back(zbo);
	    id1.push_back(i1);
	    id2.push_back(i2);
	    vts_expected.push_back(TOSeparation(rpt, rlt, zbt, rpo, rlo, zbo, i1, i2));
	}
	fin.close();
    }
    else FAIL("Unable to open expected separations file 'test_data/expected_both_seps_catalog_three_objects_v2.txt'");
    fin.clear();

    SECTION("Compare VectorTOSeparation to vectors of separations") {
	REQUIRE(vts_expected.size() == r_perp_t.size());
	for (std::size_t i = 0; i < r_perp_t.size(); i++) {
	    REQUIRE(vts_expected[i].tsep.r_perp == Approx(r_perp_t[i]));
	    REQUIRE(vts_expected[i].tsep.r_par == Approx(r_par_t[i]));
	    REQUIRE(vts_expected[i].tsep.zbar == Approx(zbar_t[i]));
	    REQUIRE(vts_expected[i].tsep.id1 == id1[i]);
	    REQUIRE(vts_expected[i].tsep.id2 == id2[i]);
	    REQUIRE(vts_expected[i].osep.r_perp == Approx(r_perp_o[i]));
	    REQUIRE(vts_expected[i].osep.r_par == Approx(r_par_o[i]));
	    REQUIRE(vts_expected[i].osep.zbar == Approx(zbar_o[i]));
	    REQUIRE(vts_expected[i].osep.id1 == id1[i]);
	    REQUIRE(vts_expected[i].osep.id2 == id2[i]);
	    /*
	    REQUIRE(vts_expected.r_perp_t()[i] == Approx(r_perp_t[i]));
	    REQUIRE(vts_expected.r_par_t()[i] == Approx(r_par_t[i]));
	    REQUIRE(vts_expected.zbar_t()[i] == Approx(zbar_t[i]));
	    REQUIRE(vts_expected.r_perp_o()[i] == Approx(r_perp_o[i]));
	    REQUIRE(vts_expected.r_par_o()[i] == Approx(r_par_o[i]));
	    REQUIRE(vts_expected.zbar_o()[i] == Approx(zbar_o[i]));
	    REQUIRE(vts_expected.id1()[i] == id1[i]);
	    REQUIRE(vts_expected.id2()[i] == id2[i]);
	    */
	    REQUIRE(vts_expected[i] == TOSeparation(r_perp_t[i], r_par_t[i], zbar_t[i], r_perp_o[i], r_par_o[i], zbar_o[i], id1[i], id2[i]));
	}
    }

    /*
    SECTION("Compare filling in loop to constructing from arrays") {
	VectorTOSeparation from_vecs(r_perp_t, r_par_t, zbar_t, r_perp_o, r_par_o, zbar_o, id1, id2);
	REQUIRE(from_vecs == vts_expected);
    }
    */

    // Get the separations for the catalog
    INFO("Getting catalog separations");
    BinSpecifier rp_bins, rl_bins;
    rp_bins.set_bin_min(0.0);
    rp_bins.set_bin_max(1.e7);
    rl_bins.set_bin_min(0.0);
    rl_bins.set_bin_max(1.e7);
    //VectorTOSeparation vts_result = get_separations(cat, cat, rp_bins, rl_bins, true);
    std::vector<TOSeparation> vts_auto_result = get_auto_separations(cat, rp_bins, rl_bins);
    REQUIRE_THAT(vts_auto_result, AllStructEquivalent(vts_expected));
    std::vector<TOSeparation> vts_result = get_separations(cat, cat, rp_bins, rl_bins, true);
    REQUIRE_THAT(vts_result, AllStructEquivalent(vts_expected));
    /*
    REQUIRE(vts_result.size() == vts_expected.size());
    INFO("Match results by hand");
    bool match_found;
    for (std::size_t i = 0; i < vts_result.size(); i++) {
	CAPTURE(i);
	CAPTURE("Finding matching pair in expected");
	match_found = false;
	for (std::size_t j = 0; j < vts_expected.size(); j++) {
	    if (vts_result[j].tsep.id1 == vts_expected[i].tsep.id1 && vts_result[j].tsep.id2 == vts_expected[i].tsep.id2) {
		match_found = true;
		REQUIRE(vts_result[j].tsep.r_perp == Approx(vts_expected[i].tsep.r_perp));
		REQUIRE(vts_result[j].tsep.r_par == Approx(vts_expected[i].tsep.r_par));
		REQUIRE(vts_result[j].tsep.zbar == Approx(vts_expected[i].tsep.zbar));
		REQUIRE(vts_result[j].osep.r_perp == Approx(vts_expected[i].osep.r_perp));
		REQUIRE(vts_result[j].osep.r_par == Approx(vts_expected[i].osep.r_par));
		REQUIRE(vts_result[j].osep.zbar == Approx(vts_expected[i].osep.zbar));
		// Now test to see if operator== works for Separation objects
		REQUIRE(vts_result[j] == vts_expected[i]);
            }
            else continue;
        }
        if (!match_found) {
            CAPTURE(vts_result[i]);
            FAIL("No match found in output");
        }
    }
    */
    //INFO("Test VectorTOSeparation operator==");
    //REQUIRE(vts_result == vts_expected);
}
