// -*-c++-*-
#ifndef FAST_MATH_H
#define FAST_MATH_H

#include <type_traits> // std::decay_t, std::enable_if_t, std::is_floating_point
#include <limits> // std::numeric_limits
#include <cstdint> // std::intmax_t
#include <array> // std::array
#include <cmath> // std::isnan
#include <vector> // std::vector

using size_t = decltype(sizeof(int));

template<size_t... Is> struct seq{};

template<size_t N, size_t... Is>
struct gen_seq : gen_seq<N-1, N, Is...> {};

template<size_t... Is>
struct gen_seq<0, Is...> : seq<Is...> {};

namespace math {
    template<typename T>
    constexpr std::enable_if_t<std::is_arithmetic<T>::value, int>
    signof(T x, std::false_type) {
	return T(0) < x;
    }

    template<typename T>
    constexpr std::enable_if_t<std::is_arithmetic<T>::value, int>
    signof(T x, std::true_type) {
	return (T(0) < x) - (x < T(0));
    }

    template<typename T>
    constexpr std::enable_if_t<std::is_arithmetic<T>::value, int>
    signof(T x) {
	return signof(x, std::is_signed<T>());
    }

    template<typename T>
    constexpr std::enable_if_t<std::is_arithmetic<T>::value, bool>
    isclose(T a, T b, std::false_type, int ulp=2) {
	// isclose for non-integer numbers
	return (std::abs(a-b) <= std::numeric_limits<T>::epsilon() * std::abs(a+b) * ulp || std::abs(a-b) < std::numeric_limits<T>::min());
    }

    template<typename T>
    constexpr std::enable_if_t<std::is_arithmetic<T>::value, bool>
    isclose(T a, T b, std::true_type, int ulp=2) {
	// isclose for integers: exact equality
	return a==b;
    }

    template<typename T>
    constexpr std::enable_if_t<std::is_arithmetic<T>::value, bool>
    isclose(T a, T b, int ulp=2) {
	// isclose wrapper for any numerical type
	return isclose(a, b, std::is_integral<T>{}, ulp);
    }

    template<typename T> constexpr T pi = 3.141592653589793238462643383279502884L;
    template<typename T> constexpr T sqrt2 = 1.414213562373095048801688724209698079L;
    template<typename T> constexpr T sqrt_2 = 0.7071067811865475244008443621048490392L;
    template<typename T> constexpr T rad2deg = 57.29577951308232087679815481410517033L;
    template<typename T> constexpr T deg2rad = 0.01745329251994329576923690768488612713L;
    template<typename T> constexpr T pi2 = 6.283185307179586476925286766559005768L;
    template<typename T> constexpr T pi_2 = 1.570796326794896619231321691639751442L;

    constexpr static double dpi = pi<double>;
    constexpr static double dsqrt2 = sqrt2<double>;
    constexpr static double dsqrt_2 = sqrt_2<double>;
    constexpr static double drad2deg = rad2deg<double>;
    constexpr static double ddeg2rad = deg2rad<double>;
    constexpr static double dpi2 = pi2<double>;
    constexpr static double dpi_2 = pi_2<double>;

    template<class T, class dcy = std::decay_t<T> >
    constexpr std::enable_if_t<std::is_floating_point<T>::value,dcy>
    inverse(T value) {
	return isclose(value, (T)0) ? 0.0 : 1.0 / value;
    }

    constexpr long double factorial(const std::intmax_t& n) {
	return n <= 1 ? 1 : n * factorial(n - 1);
    }

    inline size_t max_factorial() {
	size_t i = 2;
	while (factorial(i) < std::numeric_limits<long double>::max()) { ++i; }
	return i;
    }

    template<typename T, class dcy=std::decay_t<T> >
    constexpr std::enable_if_t<std::is_floating_point<T>::value,dcy>
    power(T x, const std::intmax_t& n) {
	return n==0 ? 1 : n < 0 ? power((T)inverse(x), n) : x * power(x, n - 1);
    }

    template<class base, size_t N>
    class trig_coeffs {
	using T = typename base::value_type;
	using array_type = std::array<T,N>;

	template<size_t... NS>
	constexpr static inline array_type _coeffs(seq<NS...>) {
	    return {{base::coeff(NS)...}};
	}

    public:
	constexpr static array_type coeffs=_coeffs(gen_seq<N>{});
    };

    template<class base, size_t N, class dcy=std::decay_t<typename base::value_type> >
    inline std::enable_if_t<std::is_floating_point<dcy>::value,dcy>
    _sincos(typename base::value_type x) noexcept {
	using c = trig_coeffs<base,N>;
	if (std::isnan(x) && std::numeric_limits<dcy>::has_quiet_NaN) {
	    return static_cast<dcy>(std::numeric_limits<dcy>::quiet_NaN());
	}
	else if (std::isinf(x) && std::numeric_limits<dcy>::has_infinity) {
	    return static_cast<dcy>(std::numeric_limits<dcy>::infinity());
	}
	else {
	    dcy result = 0.0;
	    dcy _x = base::range_reduce(x);
	    {
		const dcy x_2 = _x * _x;
		dcy pow = base::initial_condition(_x);
		for (auto&& cf : c::coeffs) {
		    result += cf * pow;
		    pow *= x_2;
		}
	    }
	    return result;
	}
    }

    template<class base, size_t N, class dcy=std::decay_t<typename base::value_type> >
    inline std::enable_if_t<std::is_floating_point<dcy>::value,dcy>
    _atan(typename base::value_type x) noexcept {
	using c = trig_coeffs<base,N>;
	if (std::isnan(x) && std::numeric_limits<dcy>::has_quiet_NaN) {
	    return static_cast<dcy>(std::numeric_limits<dcy>::quiet_NaN());
	}
	else if (std::isinf(x) && std::numeric_limits<dcy>::has_infinity) {
	    return static_cast<dcy>(std::numeric_limits<dcy>::infinity());
	}
	else {
	    dcy result = 0.0;
	    {
		const dcy increment = base::interval(x);
		dcy pow = base::initial_condition(x);
		for (auto&& cf : c::coeffs) {
		    result += cf * pow;
		    pow *= increment;
		}
	    }
	    return result;
	}
    }

    namespace detail {
	template<class T>
	struct _sin {
	    using value_type = T;
	    constexpr static inline T coeff(size_t n) noexcept {
		return (n % 2 ? 1 : -1) * inverse(factorial((2 * n) - 1));
	    }
	    static inline T range_reduce(T x) noexcept {
		T _x = x;
		_x += math::pi<T>;
		_x -= static_cast<size_t>(_x / math::pi2<T>) * math::pi2<T>;
		_x -= math::pi<T>;
		return _x;
	    }
	    constexpr static inline T initial_condition(T x) noexcept {
		return x;
	    }
	    constexpr static inline size_t default_N() noexcept {
		return 20;
	    }
	};

	template<class T>
	struct _cos {
	    using value_type = T;
	    constexpr static inline T coeff(size_t n) noexcept {
		return (n % 2 ? 1 : -1) * inverse(factorial(2 * n));
	    }
	    static inline T range_reduce(T x) noexcept {
		T _x = x;
		x -= static_cast<size_t>(_x / math::pi2<T>) * math::pi2<T>;
		return _x;
	    }
	    constexpr static inline T initial_condition(T x) noexcept {
		return x * x;
	    }
	    constexpr static inline size_t default_N() noexcept {
		return 20;
	    }
	};

	template<class T>
	struct _atan {
	    using value_type = T;
	    constexpr static inline T coeff(size_t n) noexcept {
		return (n % 2 ? 1 : -1) * inverse((T)((2 * n) + 1));
	    }
	    constexpr static inline T initial_condition(T x) noexcept {
		return std::abs(x) > 1 ? inverse(x) : x;
	    }
	    constexpr static inline T interval(T x) noexcept {
		return std::abs(x) > 1 ? inverse(x * x) : x * x;
	    }
	    constexpr static inline size_t default_N() noexcept {
		return 25;
	    }
	};

    }

    template<class T, size_t N=detail::_sin<T>::default_N()>
    constexpr inline std::decay_t<T> sin(T x) noexcept {
	return _sincos<detail::_sin<T>,N>(x);
    }

    template<class T, size_t N=detail::_cos<T>::default_N()>
    constexpr inline std::decay_t<T> cos(T x) noexcept {
	return 1 - _sincos<detail::_cos<T>,N>(x);
    }

    template<class T, size_t N=detail::_atan<T>::default_N()>
    constexpr inline std::decay_t<T> atan(T x) noexcept {
	return std::abs(x) > 1 ? signof(x) * pi_2<T> - _atan<detail::_atan<T>,N>(x) : _atan<detail::_atan<T>,N>(x);
    }

    template<class T, size_t N=detail::_atan<T>::default_N()>
    inline std::decay_t<T> atan2(T x, T y) noexcept {
	auto at = atan(y / x);
	if (signof(x) < 0) {
	    if (signof(at) >= 0) { at -= pi<T>; }
	    else { at += pi<T>; }
	}
	return at;
    }
}

template<typename T>
constexpr std::vector<T> get_nxyz(T ra, T dec) {
    return std::vector<T>({math::cos(math::deg2rad<T> * ra) * math::cos(math::deg2rad<T> * dec), math::sin(math::deg2rad<T> * ra) * math::cos(math::deg2rad<T> * dec), math::sin(math::deg2rad<T> * dec)});
}

template<typename T>
inline std::vector<T> get_radec(T nx, T ny, T nz) {
    T ra = math::atan2(ny, nx) * math::rad2deg<T>;
    ra += (ra < 0.0) ? 360.0 : 0.0;
    T dec = math::atan2(nz, std::sqrt(math::power(nx, 2) + math::power(ny, 2))) * math::rad2deg<T>;
    return std::vector<T>({ra, dec});
}

#endif
