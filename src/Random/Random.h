#ifndef IPPL_RANDOM_H
#define IPPL_RANDOM_H

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_Random.hpp>

#include "Random/Generator.h"

namespace ippl {
    namespace random {

        template <class DeviceType, class RealType = double>
        class uniform_real_distribution {
            static_assert(std::is_floating_point<RealType>::value,
                          "result_type must be a floating point type");

        public:
            typedef Generator<DeviceType> generator_type;
            typedef RealType result_type;

            KOKKOS_FUNCTION
            uniform_real_distribution()
                : uniform_real_distribution(0.0) {}

            KOKKOS_FUNCTION
            uniform_real_distribution(RealType a, RealType b = 1.0)
                : a_m(a)
                , b_m(b) {}

            KOKKOS_FUNCTION
            ~uniform_real_distribution() {}

            KOKKOS_FUNCTION
            RealType a() const { return a_m; }

            KOKKOS_FUNCTION
            RealType b() const { return b_m; }

            KOKKOS_FUNCTION
            result_type operator()(const generator_type& gen) const {
                //             result_type x = gen.template operator()<result_type>();
                result_type x = gen.template next<result_type>();
                return (b_m - a_m) * x + a_m;
            }

        private:
            RealType a_m;
            RealType b_m;
        };

        template <class DeviceType, class RealType = double>
        class normal_distribution {
            static_assert(std::is_floating_point<RealType>::value,
                          "result_type must be a floating point type");

        public:
            typedef Generator<DeviceType> generator_type;
            typedef RealType result_type;

            KOKKOS_FUNCTION
            normal_distribution()
                : normal_distribution(0.0) {}

            KOKKOS_FUNCTION
            normal_distribution(RealType mean, RealType stddev = 1.0)
                : mean_m(mean)
                , stddev_m(stddev)
                , unif_m(0.0, 1.0)
                , twopi_m(2.0 * Kokkos::numbers::pi_v<RealType>) {}

            KOKKOS_FUNCTION
            ~normal_distribution() {}

            KOKKOS_FUNCTION
            RealType mean() const { return mean_m; }

            KOKKOS_FUNCTION
            RealType stddev() const { return stddev_m; }

            KOKKOS_FUNCTION
            result_type operator()(const generator_type& gen) const {
                // Box-Muller transform:
                // 2 August 2023
                // https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
                result_type u1 = unif_m(gen);
                result_type u2 = unif_m(gen);

                return stddev_m * Kokkos::sqrt(-2.0 * Kokkos::log(u1)) * Kokkos::cos(twopi_m * u2)
                       + mean_m;
            }

        private:
            RealType mean_m;
            RealType stddev_m;
            uniform_real_distribution<DeviceType, RealType> unif_m;
            RealType twopi_m;
        };
    }  // namespace random
}  // namespace ippl
#endif
