// -*-c++-*-
#ifndef FAST_MATH_H
#define FAST_MATH_H

#include <type_traits> // std::decay_t, std::enable_if_t, std::is_floating_point
#include <limits> // std::numeric_limits
#include <cstdint> // std::intmax_t
#include <array> // std::array
#include <cmath> // std::isnan
#include <vector> // std::vector
#include <array> // std::array
#include <string> // std::string
#include <cctype> // std::tolower
#include <utility>  // std::index_sequence, std::make_index_sequence
#include <numeric>  // std::iota
#include <functional>  // std::function
#include <iostream>  // temporarily, for std::cout

inline bool lazy_string_equals(std::string_view a, std::string_view b) {
    return a.size() != b.size() ? false : std::equal(a.begin(),
                                                     a.end(),
                                                     b.begin(),
                                                     [](char a, char b) {
                                                       return std::tolower(a)
                                                              == std::tolower(b);
                                                     });
}

namespace arrays {
    namespace details {
        template<class>
        struct is_ref_wrapper : std::false_type {};
        template<class T>
        struct is_ref_wrapper<std::reference_wrapper<T>> : std::true_type {
        };

        template<class T>
        using not_ref_wrapper = std::negation<is_ref_wrapper<std::decay_t<T>>>;

        template<class D, class...>
        struct return_type_helper { using type = D; };
        template<class... Ts>
        struct return_type_helper<void, Ts...> : std::common_type<Ts...> {
            static_assert(std::conjunction_v<not_ref_wrapper<Ts>...>,
                          "Types cannot contain reference_wrappers when D is void");
        };

        template<class D, class... Ts>
        using return_type = std::array<
                typename return_type_helper<
                        D,
                        Ts...>::type,
                sizeof...(Ts)>;

        template<typename T, typename dcy = std::decay_t<T>, std::size_t... Is>
        std::array<dcy, sizeof...(Is)>
        make_filled_array_helper(T&& t, std::index_sequence<Is...>) {
            return {{(static_cast<void>(Is), t)...}};
        }
    } // namespace details

    template<class D = void, class... Ts>
    constexpr details::return_type<D, Ts...> make_array(Ts&& ... t) {
        return {std::forward<Ts>(t)...};
    }

    template<std::size_t N, typename T, typename dcy = std::decay_t<T>>
    std::array<dcy, N> make_filled_array(T&& t) {
        return details::make_filled_array_helper(t,
                                                 std::make_index_sequence<N>{});
    }

    template<typename T, std::size_t N, typename std::enable_if_t<
            std::is_arithmetic_v<
                    T>, int> = 0>
    std::array<T, N> make_filled_array() {
        return make_filled_array<N>((T) 0);
    }

    template<typename T, std::size_t N, typename std::enable_if_t<
            !std::is_arithmetic_v<
                    T>, int> = 0>
    std::array<T, N> make_filled_array() {
        return make_filled_array<N>(T());
    }

    template<typename T,
             typename U,
             typename R = decltype(std::declval
                                           <
                                                   std::decay_t
                                                           <T >>()
                                   * std::declval
                                           <
                                                   std::decay_t
                                                           <U >>()),
            std::size_t N>
    std::array<R, N>
    multiply_array_by_constant(const std::array<T, N>& arr, U&& x) {
        std::array<R, N> ret = make_filled_array<R, N>();
        std::transform(arr.begin(),
                       arr.end(),
                       ret.begin(),
                       [=](T el) { return x * el; });
        return ret;
    }

    template<std::size_t R, typename T, std::size_t N>
    std::array<T, R * N> repeat_array(const std::array<T, N>& arr) {
        std::array<T, R * N> ret = make_filled_array<T, R * N>();
        for (std::size_t i = 0; i < N; i++) {
            for (std::size_t j = 0; j < R; j++) {
                ret[i + (j * N)] = arr[i];
            }
        }
        return ret;
    } // namespace arrays

}

namespace math {
    using namespace std::string_view_literals;

    template<typename T, typename std::enable_if_t<
            std::is_unsigned_v<T>,
            int> = 0>

    constexpr int signof(T x) { return T(0) < x; }

    template<typename T, typename std::enable_if_t<
            std::is_signed_v<T>,
            int> = 0>

    constexpr int signof(T x) { return (T(0) < x) - (x < T(0)); }

    template<typename T, typename U, typename std::enable_if_t<
            std::is_floating_point_v<T> || std::is_floating_point_v<U>,
            int> = 0>

    constexpr bool isclose(T a, U b, int ulp = 10) {
        // isclose for non-integer numbers
        using R = decltype(std::declval<T>() - std::declval<U>());
        return std::abs(a - b)
               <= std::numeric_limits<R>::epsilon() * std::abs(a + b) * ulp
               || std::abs(a - b) < std::numeric_limits<R>::min();
    }

    template<typename T, typename U, typename std::enable_if_t<
            std::is_integral_v<T> && std::is_integral_v<U>, int> = 0>

    constexpr bool isclose(T a, U b, int= 10) {
        // isclose for integers: exact equality
        return signof(a) * a == signof(b) * b;
    }

    template<typename T, typename U, typename std::enable_if_t<
            std::is_arithmetic_v<T> && std::is_arithmetic_v<U>, int> = 0>

    inline bool isclose(std::vector<T> a, std::vector<U> b, int ulp = 10) {
        if (a.size() != b.size()) { return false; }
        return std::equal(a.begin(),
                          a.end(),
                          b.begin(),
                          [=](T x, U y) { return isclose(x, y, ulp); });
    }

    template<std::size_t N, std::size_t M, typename T,
                                           typename U, typename std::enable_if_t<
                    N != M,
                    int> = 0>
    constexpr bool isclose(const std::array<T, N>&,
                           const std::array<U, M>&,
                           int= 10) { return false; }

    template<std::size_t N, std::size_t M, typename T,
                                           typename U, typename std::enable_if_t<
                    N == M && std::is_arithmetic_v<T>
                    && std::is_arithmetic_v<U>,
                    int> = 0>

    inline bool isclose(const std::array<T, N>& a,
                        const std::array<U, M>& b,
                        int ulp = 10) {
        return std::equal(a.begin(),
                          a.end(),
                          b.begin(),
                          [=](T x, U y) { return isclose(x, y, ulp); });
    }

    template<typename T> constexpr T
            pi = 3.141592653589793238462643383279502884L;
    template<typename T> constexpr T
            sqrt2 = 1.414213562373095048801688724209698079L;
    template<typename T> constexpr T
            sqrt_2 = 0.7071067811865475244008443621048490392L;
    template<typename T> constexpr T
            rad2deg = 57.29577951308232087679815481410517033L;
    template<typename T> constexpr T
            deg2rad = 0.01745329251994329576923690768488612713L;
    template<typename T> constexpr T
            pi2 = 6.283185307179586476925286766559005768L;
    template<typename T> constexpr T
            pi_2 = 1.570796326794896619231321691639751442L;
    template<typename T> constexpr T
            nan = std::numeric_limits<T>::has_quiet_NaN
                  ? std::numeric_limits<T>::quiet_NaN()
                  : std::numeric_limits<T>::max();

    constexpr static double dpi = pi<double>;
    constexpr static double dsqrt2 = sqrt2<double>;
    constexpr static double dsqrt_2 = sqrt_2<double>;
    constexpr static double drad2deg = rad2deg<double>;
    constexpr static double ddeg2rad = deg2rad<double>;
    constexpr static double dpi2 = pi2<double>;
    constexpr static double dpi_2 = pi_2<double>;
    constexpr static double dnan = nan<double>;
    constexpr static std::string_view snan = "NaN"sv;

    template<typename T, typename std::enable_if_t<
            std::is_floating_point_v<T>,
            int> = 0>

    constexpr bool isnan(T x) { return !isclose(x, x); }

    template<typename T, typename std::enable_if_t<
            std::is_integral_v<T>,
            int> = 0>

    constexpr bool isnan(T x) { return isclose(x, nan<T>); }

    template<typename T,
             typename dcy = std::decay_t<T>, typename std::enable_if_t<
                    std::is_floating_point_v<
                            T>, int> = 0>

    constexpr dcy inverse(T x) {
        return isclose(x, (T) 0) ? nan<dcy> : (dcy) (1.0 / x);
    }

    template<typename T,
             typename dcy = std::decay_t<T>, typename std::enable_if_t<
                    std::is_integral_v<
                            T>, int> = 0>

    constexpr dcy inverse(T x) {
        return isclose(x, (T) 0) ? (dcy) 0 : (dcy) (1 / x);
    }

    template<std::size_t P, typename T,
                            typename dcy = std::decay_t<T>, std::enable_if_t<
                    std::is_arithmetic_v<
                            T>, int> = 0>
    constexpr dcy power(T x) { return P == 0 ? 1 : x * power<P - 1>(x); }

    template<typename T,
             typename dcy = std::decay_t<T>, typename std::enable_if_t<
                    std::is_arithmetic_v<
                            T>, int> = 0>

    constexpr dcy square(T x) { return x * x; }

    constexpr long double factorial(const std::intmax_t& n) {
        return n <= 1
               ? 1
               : n * factorial(n - 1);
    }


    template<class Base, std::size_t N>
    class trig_coeffs {
        using T = typename Base::value_type;
        using array_type = std::array<T, N>;

        template<std::size_t... NS>
        constexpr static inline array_type _coeffs(std::index_sequence<NS...>) {
            return arrays::make_array(Base::coeff(NS)...);
        }

    public:
        constexpr static array_type
                coeffs = _coeffs(std::make_index_sequence<N>{});
    };


    template<class Base, std::size_t N,
             class dcy = std::decay_t<typename Base::value_type>, typename std::enable_if_t<
                    std::is_floating_point_v<dcy>,
                    int> = 0>

    inline dcy _trig(typename Base::value_type x) noexcept {
        using c = trig_coeffs<Base, N>;
        if (std::isnan(x) && std::numeric_limits<dcy>::has_quiet_NaN) {
            return static_cast<dcy>(std::numeric_limits<
                    dcy>::quiet_NaN());
        }
        else if (std::isinf(x) && std::numeric_limits<dcy>::has_infinity) {
            return static_cast<dcy>(std::numeric_limits<
                    dcy>::infinity());
        }
        else {
            dcy _x = Base::range_reduce(x);
            dcy result = 0.0;
            const dcy step = Base::pow_step(_x);
            dcy pow = Base::initial_condition(_x);
            for (const auto& cf : c::coeffs) {
                result += cf * pow;
                pow *= step;
            }
            return result;
        }
    }

    namespace detail {
        template<typename T, typename std::enable_if_t<
                std::is_arithmetic_v<T>,
                int> = 0>

        struct _sin {
            using value_type = T;

            constexpr static T coeff(std::size_t n) noexcept {
                return (n % 2 == 0 ? 1 : -1) * inverse(factorial((2 * n) + 1));
            }

            static inline T range_reduce(T x) noexcept {
                T _x = x;
                _x += math::pi<T>;
                _x -= static_cast<size_t>(_x / math::pi2<T>) * math::pi2<T>;
                _x -= math::pi<T>;
                return _x;
            }

            constexpr static T initial_condition(T x) noexcept { return x; }

            constexpr static T pow_step(T x) noexcept { return square(x); }

            constexpr static std::size_t default_N = 20;
        };


        template<typename T, typename std::enable_if_t<
                std::is_arithmetic_v<T>,
                int> = 0>

        struct _cos {
            using value_type = T;

            constexpr static T coeff(std::size_t n) noexcept {
                return (n % 2 == 0 ? 1 : -1) * inverse(factorial(2 * n));
            }

            static inline T range_reduce(T x) noexcept {
                T _x = x;
                _x -= static_cast<size_t>(_x / math::pi2<T>) * math::pi2<T>;
                return _x;
            }

            constexpr static T initial_condition(T) noexcept { return (T) 1; }

            constexpr static T pow_step(T x) noexcept { return square(x); }

            constexpr static size_t default_N = 20;
        };

    }

    template<class T, size_t N = detail::_sin<T>::default_N>
    constexpr static std::decay_t<T>
    sin(T x) noexcept { return _trig<detail::_sin<T>, N>(x); }

    template<class T, size_t N = detail::_cos<T>::default_N>
    constexpr static std::decay_t<T> cos(T x) noexcept {
        return 1 - _trig<
                detail::_cos<T>, N>(x);
    }

}

namespace arrays {
    template<typename T>
    std::vector<std::vector<T>>
    transpose_vector(const std::vector<std::vector<T>> vec_in) {
        std::vector<std::vector<T>>
                vec_out(vec_in[0].size(), std::vector<T>(vec_in.size()));
        for (std::size_t i = 0; i < vec_out.size(); i++) {
            for (std::size_t j = 0; j < vec_in.size(); j++) {
                vec_out[i][j] = vec_in[j][i];
            }
        }
        return vec_out;
    }
}

// template<typename T>
// inline std::array<T, 3> get_nxyz_array(T ra, T dec) {
//     return arrays::make_array(math::cos(math::deg2rad<T> * ra)
//                               * math::cos(math::deg2rad<T> * dec),
//                               math::sin(math::deg2rad<T> * ra)
//                               * math::cos(math::deg2rad<T> * dec),
//                               math::sin(math::deg2rad<T> * dec));
// }

template<typename T>
inline std::array<T, 3> get_nxyz_array(T ra, T dec) {
    return arrays::make_array(std::cos(math::deg2rad<T> * ra)
                              * std::cos(math::deg2rad<T> * dec),
                              std::sin(math::deg2rad<T> * ra)
                              * std::cos(math::deg2rad<T> * dec),
                              std::sin(math::deg2rad<T> * dec));
}

#endif
