#ifndef IPPL_INVERSE_TRANSFORM_SAMPLING_H
#define IPPL_INVERSE_TRANSFORM_SAMPLING_H

#include "Types/ViewTypes.h"

#include "Random/Generator.h"
#include "Random/Random.h"

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
                KOKKOS_INLINE_FUNCTION void solve(Distribution dist, T& x, T atol = 1.0e-12,
                                                  unsigned int max_iter = 20) {
                    unsigned int iter = 0;
                    while (iter < max_iter && Kokkos::fabs(dist.function(x)) > atol) {
                        x = x - (dist.function(x) / dist.derivative(x));
                        iter += 1;
                    }
                }
            };
        }  // namespace detail

        template <typename T, unsigned Dim, class DeviceType>
        class InverseTransformSampling {
        public:
            using view_type = ippl::detail::ViewType<T, 1>::view_type;

            template <class Distribution, class RegionLayout>
            InverseTransformSampling(const Vector<T, Dim>& rmin, const Vector<T, Dim>& rmax,
                                     const RegionLayout& rlayout, Distribution dist[Dim],
                                     unsigned int ntotal, int seed)
                : gen_m(seed)
                , unif_m(0.0, 1.0) {
                const typename RegionLayout::host_mirror_type regions = rlayout.gethLocalRegions();

                int rank = ippl::Comm->rank();
                for (unsigned d = 0; d < Dim; ++d) {
                    nr_m[d] =
                        dist[d].cdf(regions(rank)[d].max()) - dist[d].cdf(regions(rank)[d].min());
                    dr_m[d]   = dist[d].cdf(rmax[d]) - dist[d].cdf(rmin[d]);
                    umin_m[d] = dist[d].cdf(regions(rank)[d].min());
                    umax_m[d] = dist[d].cdf(regions(rank)[d].max());
                }

                T pnr = std::accumulate(nr_m.begin(), nr_m.end(), 1.0, std::multiplies<T>());
                T pdr = std::accumulate(dr_m.begin(), dr_m.end(), 1.0, std::multiplies<T>());

                double factor = pnr / pdr;
                nlocal_m      = factor * ntotal;

                unsigned int ngobal = 0;
                MPI_Allreduce(&nlocal_m, &ngobal, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                              ippl::Comm->getCommunicator());

                int rest = (int)(ntotal - ngobal);

                if (rank < rest) {
                    ++nlocal_m;
                }
            }

            unsigned int getLocalNum() const { return nlocal_m; }

            //                 // FIXME
            //                 void generate(View : view) { Kokkos::parallel_for(nlocal_m,
            //                 fill_random(view)); }

            template <class Distribution>
            struct fill_random {
                // Output View for the random numbers
                view_type x_m;

                Distribution dist_m[Dim];

                // Initialize all members
                KOKKOS_FUNCTION
                fill_random(Distribution dist[Dim], view_type x)
                    : dist_m(dist)
                    , x_m(x) {}

                KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
                    T u = 0.0;

                    for (unsigned d = 0; d < Dim; ++d) {
                        // get uniform random number between umin and umax
                        u = (umax_m[d] - umin_m[d]) * unif_m(gen_m) + umin_m[d];

                        // first guess for Newton-Raphson
                        x_m(i)[d] = dist_m[d].estimate(u);

                        // solve
                        detail::NewtonRaphson<T> solver;
                        solver.solve(dist_m[d], x_m(i)[d]);
                    }
                }
            };

        private:
            Generator<DeviceType> gen_m;
            uniform_real_distribution<DeviceType, T> unif_m;
            unsigned int nlocal_m;
            Vector<T, Dim> nr_m, dr_m, umin_m, umax_m;
        };

        template <class DeviceType, class RealType = double>
        class mpi_normal_distribution {
            static_assert(std::is_floating_point<RealType>::value,
                          "result_type must be a floating point type");

        public:
            typedef RealType result_type;

            KOKKOS_FUNCTION
            mpi_normal_distribution()
                : mpi_normal_distribution(0.0) {}

            KOKKOS_FUNCTION
            mpi_normal_distribution(RealType mean, RealType stddev = 1.0)
                : mean_m(mean)
                , stddev_m(stddev)
                , sqrt2_m(Kokkos::sqrt(RealType(2.0)))
                , pi_m(Kokkos::numbers::pi_v<RealType>) {}

            KOKKOS_FUNCTION ~mpi_normal_distribution() {}

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
                return 0.5 * (1.0 + Kokkos::erf((x - mean_m) / (stddev_m * sqrt2_m)));
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

    }  // namespace random
}  // namespace ippl

#endif
