#ifndef IPPL_INTERPOLATION_KERNELS_H
#define IPPL_INTERPOLATION_KERNELS_H

#include <Kokkos_Core.hpp>

namespace ippl {
    namespace Interpolation {

        /**
         * @brief Nearest Grid Point kernel (NGP / order 0).
         *
         * Natural form: phi(y) = 1 for |y| < 0.5, else 0
         * With inv_hw = 2/w = 2, input x = y * 2, so y = x * 0.5
         * Normalized: phi(x) = 1 for |x| < 1
         */
        template <typename T = double>
        struct NGPKernel {
            using value_type = T;
            static constexpr bool has_width_template = false;
            static constexpr int max_width = 1;

            KOKKOS_INLINE_FUNCTION T operator()(T x) const {
                // Transform: y = x * (w/2) = x * 0.5
                // Condition |y| < 0.5 becomes |x| < 1 (strict, to avoid
                // double-counting particles that sit exactly on a cell boundary).
                return Kokkos::abs(x) < T(1) ? T(1) : T(0);
            }

            KOKKOS_INLINE_FUNCTION static constexpr int width() { return 1; }
        };

        /**
         * @brief Linear kernel (CIC / Cloud-in-Cell / order 1).
         *
         * Natural form: phi(y) = 1 - |y| for |y| < 1, else 0
         * With inv_hw = 2/w = 1, input x = y, no transformation needed
         * Normalized: phi(x) = 1 - |x| for |x| < 1
         */
        template <typename T = double>
        struct LinearKernel {
            using value_type = T;
            static constexpr bool has_width_template = false;
            static constexpr int max_width = 2;

            KOKKOS_INLINE_FUNCTION T operator()(T x) const {
                // w/2 = 1, so y = x (no transformation needed)
                x = Kokkos::abs(x);
                return x >= T(1) ? T(0) : T(1) - x;
            }

            KOKKOS_INLINE_FUNCTION static constexpr int width() { return 2; }
        };

        /**
         * @brief Quadratic B-spline kernel (TSC / Triangular Shaped Cloud / order 2).
         *
         * Natural form (y in [-1.5, 1.5]):
         *   phi(y) = 3/4 - y^2           for |y| < 0.5
         *          = (3/2 - |y|)^2 / 2   for 0.5 <= |y| < 1.5
         *          = 0                    otherwise
         *
         * With inv_hw = 2/w = 2/3, input x relates to natural y by: y = x * 1.5
         * So kernel receives x in [-1, 1], transforms to y in [-1.5, 1.5]
         */
        template <typename T = double>
        struct QuadraticKernel {
            using value_type                         = T;
            static constexpr bool has_width_template = false;
            static constexpr int max_width           = 3;

            KOKKOS_INLINE_FUNCTION T operator()(T x) const {
                // Transform to natural coordinates: y = x * (w/2) = x * 1.5
                T y = Kokkos::abs(x) * T(1.5);

                if (y < T(0.5)) {
                    return T(0.75) - y * y;
                } else if (y < T(1.5)) {
                    const T t = T(1.5) - y;
                    return T(0.5) * t * t;
                }
                return T(0);
            }

            KOKKOS_INLINE_FUNCTION static constexpr int width() { return 3; }
        };

        /**
         * @brief Cubic B-spline kernel (PCS / Piecewise Cubic Spline / order 3).
         *
         * Natural form (y in [-2, 2]):
         *   phi(y) = 2/3 - y^2 + |y|^3/2       for |y| < 1
         *          = (2 - |y|)^3 / 6            for 1 <= |y| < 2
         *          = 0                          otherwise
         *
         * With inv_hw = 2/w = 0.5, input x relates to natural y by: y = x * 2
         * So kernel receives x in [-1, 1], transforms to y in [-2, 2]
         */
        template <typename T = double>
        struct CubicKernel {
            using value_type                         = T;
            static constexpr bool has_width_template = false;
            static constexpr int max_width           = 4;

            KOKKOS_INLINE_FUNCTION T operator()(T x) const {
                // Transform to natural coordinates: y = x * (w/2) = x * 2
                T y = Kokkos::abs(x) * T(2);

                if (y < T(1)) {
                    return T(2.0 / 3.0) - y * y + T(0.5) * y * y * y;
                } else if (y < T(2)) {
                    const T t = T(2) - y;
                    return t * t * t / T(6);
                }
                return T(0);
            }

            KOKKOS_INLINE_FUNCTION static constexpr int width() { return 4; }
        };

        /**
         * @brief Quartic B-spline kernel (order 4).
         *
         * Natural form (y in [-2.5, 2.5]):
         *   phi(y) = (115/192) - (5/8)y^2 + (1/4)y^4                    for |y| < 0.5
         *          = (55/96) + (5/24)|y| - (5/4)y^2 + (5/6)|y|^3 - (1/6)y^4   for 0.5 <= |y| < 1.5
         *          = (5/2 - |y|)^4 / 24                                  for 1.5 <= |y| < 2.5
         *          = 0                                                   otherwise
         *
         * With inv_hw = 2/w = 0.4, input x relates to natural y by: y = x * 2.5
         */
        template <typename T = double>
        struct QuarticKernel {
            using value_type                         = T;
            static constexpr bool has_width_template = false;
            static constexpr int max_width           = 5;

            KOKKOS_INLINE_FUNCTION T operator()(T x) const {
                // Transform to natural coordinates: y = x * (w/2) = x * 2.5
                T y = Kokkos::abs(x) * T(2.5);

                if (y < T(0.5)) {
                    T y2 = y * y;
                    T y4 = y2 * y2;
                    return T(115.0 / 192.0) - T(5.0 / 8.0) * y2 + T(0.25) * y4;
                } else if (y < T(1.5)) {
                    T y2 = y * y;
                    T y3 = y2 * y;
                    T y4 = y2 * y2;
                    return T(55.0 / 96.0) + T(5.0 / 24.0) * y - T(5.0 / 4.0) * y2
                           + T(5.0 / 6.0) * y3 - T(1.0 / 6.0) * y4;
                } else if (y < T(2.5)) {
                    T t  = T(2.5) - y;
                    T t2 = t * t;
                    return t2 * t2 / T(24);
                }
                return T(0);
            }

            KOKKOS_INLINE_FUNCTION static constexpr int width() { return 5; }
        };

    }  // namespace Interpolation
}  // namespace ippl

#endif  // IPPL_INTERPOLATION_KERNELS_H