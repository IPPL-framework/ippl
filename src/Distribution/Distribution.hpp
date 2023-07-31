#ifndef DISTRIBUTION_H
#define DISTRIBUTION_H

#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_Random.hpp>

#include "PICManager/PICManager.hpp"

/**
 * @namespace Distribution
 * @brief Contains functions and structures for handling distributions.
 */
namespace Distribution {

    // const unsigned Dim = 3;

    template <typename T, unsigned Dim = 3>
    class Distribution {
        std::string distName_m;

    public:
        Distribution(const char* distName)
            : distName_m(distName) {}
        // virtual void repartitionRhs(auto P, const ippl::NDIndex<Dim>& lDom) = 0;
        //  virtual void createParticles(auto P, auto RLayout)                  = 0;
        ~Distribution() {}
    };

    template <typename T, unsigned Dim = 3>
    class LandauDampingDistribution : public Distribution<T, Dim> {
        double alpha_m;
        Vector_t<double, Dim> kw_m;
        Vector_t<double, Dim> rmax_m;
        Vector_t<double, Dim> rmin_m;
        Vector_t<double, Dim> hr_m;
        Vector_t<double, Dim> origin_m;
        const size_type totalP_m;

        /**
         * @brief Compute the cumulative distribution function (CDF) of a distribution.
         * @param x The input value.
         * @param alpha The alpha parameter of the distribution.
         * @param k The k parameter of the distribution.
         * @return The value of the CDF at x.
         */
        double CDF(const double& x, const double& alpha, const double& k) {
            double cdf = x + (alpha / k) * std::sin(k * x);
            return cdf;
        }

        /**
         * @brief Compute the probability density function (PDF) of a distribution in multiple
         * dimensions.
         * @param xvec The input vector containing the coordinates in each dimension.
         * @param alpha The alpha parameter of the distribution.
         * @param kw The k parameter for each dimension of the distribution.
         * @param Dim The number of dimensions.
         * @return The value of the PDF at the given coordinates.
         */
        KOKKOS_FUNCTION
        double PDF(const Vector_t<double, Dim>& xvec, const double& alpha,
                   const Vector_t<double, Dim>& kw) {
            double pdf = 1.0;

            for (unsigned d = 0; d < Dim; ++d) {
                pdf *= (1.0 + alpha * Kokkos::cos(kw[d] * xvec[d]));
            }
            return pdf;
        }

        /**
         * @brief A structure representing the Newton method for finding roots of 1D functions.
         * @tparam T The type of the parameters and variables used in the Newton method.
         */

        struct Newton1D {
            double tol   = 1e-12;
            int max_iter = 20;
            double pi    = Kokkos::numbers::pi_v<double>;

            T k, alpha, u;

            KOKKOS_INLINE_FUNCTION Newton1D() {}

            KOKKOS_INLINE_FUNCTION Newton1D(const T& k_, const T& alpha_, const T& u_)
                : k(k_)
                , alpha(alpha_)
                , u(u_) {}

            KOKKOS_INLINE_FUNCTION ~Newton1D() {}

            KOKKOS_INLINE_FUNCTION T f(T& x) {
                T F;
                F = x + (alpha * (Kokkos::sin(k * x) / k)) - u;
                return F;
            }

            KOKKOS_INLINE_FUNCTION T fprime(T& x) {
                T Fprime;
                Fprime = 1 + (alpha * Kokkos::cos(k * x));
                return Fprime;
            }

            KOKKOS_FUNCTION
            void solve(T& x) {
                int iterations = 0;
                while (iterations < max_iter && Kokkos::fabs(f(x)) > tol) {
                    x = x - (f(x) / fprime(x));
                    iterations += 1;
                }
            }
        };

        template <typename GT, class GeneratorPool>
        struct generate_random {
            using view_type  = typename ippl::detail::ViewType<GT, 1>::view_type;
            using value_type = typename GT::value_type;
            //  Output View for the random numbers
            view_type x, v;

            // The GeneratorPool
            GeneratorPool rand_pool;

            value_type alpha;

            GT k, minU, maxU;

            // Initialize all members
            generate_random(view_type x_, view_type v_, GeneratorPool rand_pool_,
                            value_type& alpha_, GT& k_, GT& minU_, GT& maxU_)
                : x(x_)
                , v(v_)
                , rand_pool(rand_pool_)
                , alpha(alpha_)
                , k(k_)
                , minU(minU_)
                , maxU(maxU_) {}

            KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
                // Get a random number state from the pool for the active thread
                typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

                value_type u;
                for (unsigned d = 0; d < Dim; ++d) {
                    u       = rand_gen.drand(minU[d], maxU[d]);
                    x(i)[d] = u / (1 + alpha);
                    Newton1D solver(k[d], alpha, u);
                    solver.solve(x(i)[d]);
                    v(i)[d] = rand_gen.normal(0.0, 1.0);
                }

                // Give the state back, which will allow another thread to acquire it
                rand_pool.free_state(rand_gen);
            }
        };

    public:
        LandauDampingDistribution(double alpha, Vector_t<double, Dim> kw,
                                  Vector_t<double, Dim> rmax, Vector_t<double, Dim> rmin,
                                  Vector_t<double, Dim> hr, Vector_t<double, Dim> origin,
                                  const size_type totalP)
            : Distribution<T, Dim>("LandauDamping")
            , alpha_m(alpha)
            , kw_m(kw)
            , rmax_m(rmax)
            , rmin_m(rmin)
            , hr_m(hr)
            , origin_m(origin)
            , totalP_m(totalP) {}

        ~LandauDampingDistribution() {}

        void repartitionRhs(auto P, const ippl::NDIndex<Dim>& lDom) {
            using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;

            const int nghost = P->rhs_m.getNghost();
            auto rhoview     = P->rhs_m.getView();
            auto rangePolicy = P->rhs_m.getFieldRangePolicy();

            ippl::parallel_for(
                "Assign initial rho based on PDF", rangePolicy,
                KOKKOS_LAMBDA(const index_array_type& args) {
                    // local to global index conversion
                    Vector_t<double, Dim> xvec =
                        (args + lDom.first() - nghost + 0.5) * hr_m + origin_m;

                    // ippl::apply accesses the view at the given indices and obtains a
                    // reference; see src/Expression/IpplOperations.h
                    ippl::apply(rhoview, args) = PDF(xvec, alpha_m, kw_m);
                });
            Kokkos::fence();
        }

        void createParticles(auto P, auto RLayout) {
            auto Regions = RLayout.gethLocalRegions();
            Vector_t<double, Dim> Nr, Dr, minU, maxU;
            int myRank    = ippl::Comm->rank();
            double factor = 1;

            for (unsigned d = 0; d < Dim; ++d) {
                Nr[d] = CDF(Regions(myRank)[d].max(), alpha_m, kw_m[d])
                        - CDF(Regions(myRank)[d].min(), alpha_m, kw_m[d]);
                Dr[d]   = CDF(rmax_m[d], alpha_m, kw_m[d]) - CDF(rmin_m[d], alpha_m, kw_m[d]);
                minU[d] = CDF(Regions(myRank)[d].min(), alpha_m, kw_m[d]);
                maxU[d] = CDF(Regions(myRank)[d].max(), alpha_m, kw_m[d]);
                factor *= Nr[d] / Dr[d];
            }

            size_type nloc            = (size_type)(factor * totalP_m);
            size_type Total_particles = 0;

            MPI_Allreduce(&nloc, &Total_particles, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                          ippl::Comm->getCommunicator());

            int rest = (int)(totalP_m - Total_particles);

            if (ippl::Comm->rank() < rest) {
                ++nloc;
            }

            P->create(nloc);

            Kokkos::Random_XorShift64_Pool<> rand_pool64(
                (size_type)(42 + 100 * ippl::Comm->rank()));
            Kokkos::parallel_for(
                nloc, generate_random<Vector_t<double, Dim>, Kokkos::Random_XorShift64_Pool<> >(
                          P->R.getView(), P->V.getView(), rand_pool64, alpha_m, kw_m, minU, maxU));

            Kokkos::fence();
        }
    };

    template <typename T, unsigned Dim = 3>
    class PenningTrapDistribution : public Distribution<T, Dim> {
        Vector_t<double, Dim> mu_m;
        Vector_t<double, Dim> sd_m;
        Vector_t<double, Dim> rmax_m;
        Vector_t<double, Dim> rmin_m;
        Vector_t<double, Dim> hr_m;
        Vector_t<double, Dim> origin_m;
        const size_type totalP_m;

        /**
         * @brief Compute the cumulative distribution function (CDF) of a distribution.
         * @param x The input value.
         * @param alpha The alpha parameter of the distribution.
         * @param k The k parameter of the distribution.
         * @return The value of the CDF at x.
         */
        double CDF(const double& x, const double& mu, const double& sigma) {
            double cdf = 0.5 * (1.0 + std::erf((x - mu) / (sigma * std::sqrt(2))));
            return cdf;
        }

        /**
         * @brief Compute the probability density function (PDF) of a distribution in multiple
         * dimensions.
         * @param xvec The input vector containing the coordinates in each dimension.
         * @param alpha The alpha parameter of the distribution.
         * @param kw The k parameter for each dimension of the distribution.
         * @param Dim The number of dimensions.
         * @return The value of the PDF at the given coordinates.
         */
        KOKKOS_FUNCTION
        double PDF(const Vector_t<double, Dim>& xvec, const Vector_t<double, Dim>& mu,
                   const Vector_t<double, Dim>& sigma) {
            double pdf = 1.0;
            double pi  = Kokkos::numbers::pi_v<double>;

            for (unsigned d = 0; d < Dim; ++d) {
                pdf *= (1.0 / (sigma[d] * Kokkos::sqrt(2 * pi)))
                       * Kokkos::exp(-0.5 * Kokkos::pow((xvec[d] - mu[d]) / sigma[d], 2));
            }
            return pdf;
        }

        /**
         * @brief A structure representing the Newton method for finding roots of 1D functions.
         * @tparam T The type of the parameters and variables used in the Newton method.
         */

        struct Newton1D {
            double tol   = 1e-12;
            int max_iter = 20;
            double pi    = Kokkos::numbers::pi_v<double>;

            T mu, sigma, u;

            KOKKOS_INLINE_FUNCTION Newton1D() {}

            KOKKOS_INLINE_FUNCTION Newton1D(const T& mu_, const T& sigma_, const T& u_)
                : mu(mu_)
                , sigma(sigma_)
                , u(u_) {}

            KOKKOS_INLINE_FUNCTION ~Newton1D() {}

            KOKKOS_INLINE_FUNCTION T f(T& x) {
                T F;
                F = Kokkos::erf((x - mu) / (sigma * Kokkos::sqrt(2.0))) - 2 * u + 1;
                return F;
            }

            KOKKOS_INLINE_FUNCTION T fprime(T& x) {
                T Fprime;
                Fprime = (1 / sigma) * Kokkos::sqrt(2 / pi)
                         * Kokkos::exp(-0.5 * (Kokkos::pow(((x - mu) / sigma), 2)));
                return Fprime;
            }

            KOKKOS_FUNCTION
            void solve(T& x) {
                int iterations = 0;
                while (iterations < max_iter && Kokkos::fabs(f(x)) > tol) {
                    x = x - (f(x) / fprime(x));
                    iterations += 1;
                }
            }
        };

        template <typename GT, class GeneratorPool>
        struct generate_random {
            using view_type  = typename ippl::detail::ViewType<GT, 1>::view_type;
            using value_type = typename GT::value_type;
            //  Output View for the random numbers
            view_type x, v;

            // The GeneratorPool
            GeneratorPool rand_pool;

            value_type alpha;

            GT mu, sigma, minU, maxU;
            double pi = Kokkos::numbers::pi_v<double>;
            // Initialize all members
            generate_random(view_type x_, view_type v_, GeneratorPool rand_pool_, GT& mu_,
                            GT& sigma_, GT& minU_, GT& maxU_)
                : x(x_)
                , v(v_)
                , rand_pool(rand_pool_)
                , mu(mu_)
                , sigma(sigma_)
                , minU(minU_)
                , maxU(maxU_) {}

            KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
                // Get a random number state from the pool for the active thread
                typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

                value_type u;
                for (unsigned d = 0; d < Dim; ++d) {
                    u       = rand_gen.drand(minU[d], maxU[d]);
                    x(i)[d] = (Kokkos::sqrt(pi / 2) * (2 * u - 1)) * sigma[d] + mu[d];
                    Newton1D solver(mu[d], sigma[d], u);
                    solver.solve(x(i)[d]);
                    v(i)[d] = rand_gen.normal(0.0, 1.0);
                }

                // Give the state back, which will allow another thread to acquire it
                rand_pool.free_state(rand_gen);
            }
        };

    public:
        PenningTrapDistribution(Vector_t<double, Dim> mu, Vector_t<double, Dim> sd,
                                Vector_t<double, Dim> rmax, Vector_t<double, Dim> rmin,
                                Vector_t<double, Dim> hr, Vector_t<double, Dim> origin,
                                const size_type totalP)
            : Distribution<T, Dim>("PenningTrap")
            , mu_m(mu)
            , sd_m(sd)
            , rmax_m(rmax)
            , rmin_m(rmin)
            , hr_m(hr)
            , origin_m(origin)
            , totalP_m(totalP) {}

        ~PenningTrapDistribution() {}

        void repartitionRhs(auto P, const ippl::NDIndex<Dim>& lDom) {
            using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;

            const int nghost = P->rhs_m.getNghost();
            auto rhoview     = P->rhs_m.getView();
            auto rangePolicy = P->rhs_m.getFieldRangePolicy();

            ippl::parallel_for(
                "Assign initial rho based on PDF", rangePolicy,
                KOKKOS_LAMBDA(const index_array_type& args) {
                    // local to global index conversion
                    Vector_t<double, Dim> xvec =
                        (args + lDom.first() - nghost + 0.5) * hr_m + origin_m;

                    // ippl::apply accesses the view at the given indices and obtains a
                    // reference; see src/Expression/IpplOperations.h
                    ippl::apply(rhoview, args) = PDF(xvec, mu_m, sd_m);
                });
            Kokkos::fence();
        }

        void createParticles(auto P, auto RLayout) {
            auto Regions = RLayout.gethLocalRegions();
            Vector_t<double, Dim> Nr, Dr, minU, maxU;
            int myRank    = ippl::Comm->rank();
            double factor = 1;

            for (unsigned d = 0; d < Dim; ++d) {
                Nr[d] = CDF(Regions(myRank)[d].max(), mu_m[d], sd_m[d])
                        - CDF(Regions(myRank)[d].min(), mu_m[d], sd_m[d]);
                Dr[d]   = CDF(rmax_m[d], mu_m[d], sd_m[d]) - CDF(rmin_m[d], mu_m[d], sd_m[d]);
                minU[d] = CDF(Regions(myRank)[d].min(), mu_m[d], sd_m[d]);
                maxU[d] = CDF(Regions(myRank)[d].max(), mu_m[d], sd_m[d]);
                factor *= Nr[d] / Dr[d];
            }

            size_type nloc            = (size_type)(factor * totalP_m);
            size_type Total_particles = 0;

            MPI_Allreduce(&nloc, &Total_particles, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                          ippl::Comm->getCommunicator());

            int rest = (int)(totalP_m - Total_particles);

            if (ippl::Comm->rank() < rest) {
                ++nloc;
            }

            P->create(nloc);

            Kokkos::Random_XorShift64_Pool<> rand_pool64(
                (size_type)(42 + 100 * ippl::Comm->rank()));
            Kokkos::parallel_for(
                nloc, generate_random<Vector_t<double, Dim>, Kokkos::Random_XorShift64_Pool<> >(
                          P->R.getView(), P->V.getView(), rand_pool64, mu_m, sd_m, minU, maxU));

            Kokkos::fence();
        }
    };
}  // namespace Distribution
#endif
