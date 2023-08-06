#ifndef IPPL_INVERSE_TRANSFORM_SAMPLING_H
#define IPPL_INVERSE_TRANSFORM_SAMPLING_H

#include "Random/Generator.h"

namespace ippl {

    namespace random {

        namespace detail {

            template <typename T>
            struct NewtonRaphson {
                KOKKOS_FUNCTION
                NewtonRaphson() = default;

                KOKKOS_FUNCTION
                ~NewtonRaphson() = default;

                template <class Distribution>
                KOKKOS_INLINE_FUNCTION void solve(Distribution dist, T& x, T& atol = 1.0e-12,
                                                  unsigned int max_iter) {
                    unsigned int iter = 0;
                    while (iter < max_iter && Kokkos::fabs(dist.function(x)) > atol) {
                        x = x - (dist.function(x) / dist.derivative(x));
                        iter += 1;
                    }
                }
            };
        }  // namespace detail

        namespace mpi {

            template <class DeviceType, typename T, unsigned Dim>
            class InverseTransformSampling {
            public:
                using view_type  = typename ippl::detail::ViewType<T, 1>::view_type;
                using value_type = typename T::value_type;

                InverseTransformSampling(T umin, T umax, int seed)
                    : gen_m(seed)
                    , unif_m(umin, umax) {}

                struct fill_random {
                    // Output View for the random numbers
                    view_type x_m;

                    // Initialize all members
                    KOKKOS_FUNCTION
                    fill_random(view_type x)
                        : x_m(x) {}

                    KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
                        value_type u = 0.0;

                        for (unsigned d = 0; d < Dim; ++d) {
                            // get uniform random number between umin and umax
                            u = unif_m(gen_m);

                            // first guess for Newton-Raphson
                            x_m(i)[d] = dist_m.estimate(u);

                            // solve
                            NewtonRaphson<T> solver;
                            solver.solve(dist_m, x_m(i)[d]);
                        }
                    }
                };

            private:
                Generator gen_m;
                uniform_real_distribution<DeviceType, T> unif_m;
            };

            template <class DeviceType, class RealType = double>
            class normal_distribution {
                static_assert(std::is_floating_point<RealType>::value,
                              "result_type must be a floating point type");

            public:
                typedef generator<DeviceType> Generator;
                typedef RealType result_type;

                KOKKOS_FUNCTION
                normal_distribution()
                    : normal_distribution(0.0) {}

                KOKKOS_FUNCTION
                normal_distribution(RealType mean, RealType stddev = 1.0)
                    : mean_m(mean)
                    , stddev_m(stddev)
                    , sqrt2_m(Kokkos::sqrt(RealType(2.0)))
                    , pi_m(Kokkos::numbers::pi_v<RealType>())

                          KOKKOS_FUNCTION ~normal_distribution() {}

                KOKKOS_FUNCTION
                RealType mean() const { return mean_m; }

                KOKKOS_FUNCTION
                RealType stddev() const { return stddev_m; }

                KOKKOS_INLINE_FUNCTION result_type estimate(const RealType& u) {
                    return (Kokkos::sqrt(pi_m / 2.0) * (2.0 * u - 1.0)) * stddev_m + mean_m;
                }

                KOKKOS_INLINE_FUNCTION result_type function(RealType& x, const RealType& u) {
                    return Kokkos::erf((x - mean_m) / (stddev_m * sqrt2_m)) - 2.0 * u + 1.0;
                }

                KOKKOS_INLINE_FUNCTION result_type derivative(RealType& x, const RealType& /*u*/) {
                    return (1.0 / stddev_m) * Kokkos::sqrt(2.0 / pi_m)
                           * Kokkos::exp(-0.5 * (Kokkos::pow(((x - mean_m) / stddev_m), 2)));
                }

                KOKKOS_INLINE_FUNCTION result_type cdf(const RealType& x) {
                    reutrn 0.5 * (1.0 + Kokkos::erf((x - mean_m) / (stddev_m * sqrt2_m)));
                }

                KOKKOS_INLINE_FUNCTION result_type pdf(const RealType& x) {
                    return (1.0 / (stddev_m * Kokkos::sqrt(2.0 * pi_m)))
                           * Kokkos::exp(-0.5 * Kokkos::pow((x - mean_m) / stddev_m, 2));
                }

            private:
                RealType mean_m;
                RealType stddev_m;
                uniform_real_distribution<DeviceType, RealType> unif_m;
                RealType sqrt2_m;
                RealType pi_m;
            };

        }  // namespace mpi
    }      // namespace random
}  // namespace ippl

#endif
