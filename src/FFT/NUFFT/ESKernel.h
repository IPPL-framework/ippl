//
// NUFFT ES Kernel
//   Exponential-of-Semicircle spreading kernel for NUFFT.
//
#ifndef IPPL_NUFFT_ES_KERNEL_H
#define IPPL_NUFFT_ES_KERNEL_H

#include <Kokkos_Core.hpp>

#include <cmath>

namespace ippl {
    namespace NUFFT {

        /**
         * @brief Exponential-of-Semicircle (ES) spreading kernel.
         *
         * The ES kernel is defined as:
         *   phi(x) = exp(beta * (sqrt(1 - x^2) - 1))  for |x| < 1
         *          = 0                                  otherwise
         *
         * Width (w) and beta are derived from a user-chosen error tolerance.
         *
         * @tparam T Floating point type (float or double)
         */
        template <typename T = double>
        class ESKernel {
        public:
            using value_type = T;

            static constexpr T default_tol = T(1e-10);
            static constexpr T beta_factor = T(2.30);

            /**
             * @brief Construct kernel with given tolerance.
             * @param tol Error tolerance for NUFFT accuracy
             */
            KOKKOS_INLINE_FUNCTION explicit ESKernel(T tol = default_tol)
                : w_(static_cast<int>(std::ceil(std::log10(T(1.0) / tol))) + 1)
                , beta_(beta_factor * w_) {}

            /**
             * @brief Construct kernel with explicit width and beta.
             * @param width Kernel width (number of grid points)
             * @param beta Beta parameter controlling decay
             */
            KOKKOS_INLINE_FUNCTION ESKernel(int width, T beta)
                : w_(width)
                , beta_(beta) {}

            /**
             * @brief Evaluate the ES kernel at position x in [-1, 1].
             * @param x Normalized position
             * @return Normalized kernel value
             */
            KOKKOS_INLINE_FUNCTION T operator()(T x) const {
                x = Kokkos::abs(x);
                // assert(x < 1.0);
                if (x >= 1.0) {
                    printf("");
                }
                return x >= T(1.0) ? T(0.0)
                                   : Kokkos::exp(beta_ * (Kokkos::sqrt(T(1.0) - x * x) - T(1.0)));
            }

            KOKKOS_INLINE_FUNCTION int width() const { return w_; }
            KOKKOS_INLINE_FUNCTION T beta() const { return beta_; }

        private:
            int w_;
            T beta_;
        };

    }  // namespace NUFFT
}  // namespace ippl

#endif  // IPPL_NUFFT_ES_KERNEL_H
