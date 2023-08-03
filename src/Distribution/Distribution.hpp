#ifndef DISTRIBUTION_H
#define DISTRIBUTION_H

// Copyright (c) 2021, Andreas Adelmann
// Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
//
// This file is part of IPPL.
//
// IPPL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
//

#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_Random.hpp>
#include <functional>

#include "PICManager/PICManager.hpp"

/**
 * @namespace Distribution
 * @brief contains functions and structures for handling distributions.
 * @fixme repartitionRhs and createParticles should be moved to the base class
 * @fixme generate_random should be moved to the base class
 */
namespace Distribution {

    // const unsigned Dim = 3;

    template <typename T, unsigned Dim = 3>
    class Distribution {
        std::string distName_m;

    public:
        /*
          std::function<T(T&)> f_m;
          std::function<T(T&)> fprime_m;

          Distribution(const char* distName, std::function<T(T&)> f1, std::function<T(T&)> f2)
              : distName_m(distName)
              , f_m(f1)
              , fprime_m(f2) {}
        */
        Distribution(const char* distName)
            : distName_m(distName) {}

        // virtual void repartitionRhs(auto P, const ippl::NDIndex<Dim>& lDom) = 0;
        //  virtual void createParticles(auto P, auto RLayout)                  = 0;
        ~Distribution() {}

        /**
         * @brief A structure representing the Newton method for finding roots of 1D functions.
         * @tparam T The type of the parameters and variables used in the Newton method.
         */

        template <typename Func1, typename Func2>
            requires std::invocable<Func1, T> && std::invocable<Func2, T>
        struct Newton1D {
            double tol   = 1e-12;
            int max_iter = 20;
            double pi    = Kokkos::numbers::pi_v<double>;

            Func1 F;
            Func2 Fprime;

            KOKKOS_INLINE_FUNCTION Newton1D() {}

            KOKKOS_INLINE_FUNCTION Newton1D(Func1 f1, Func2 f2)
                : F(f1)
                , Fprime(f2) {}

            KOKKOS_INLINE_FUNCTION ~Newton1D() {}

            KOKKOS_FUNCTION void solve(T& x) {
                int iterations = 0;
                while (iterations < max_iter && Kokkos::fabs(F(x)) > tol) {
                    x = x - (F(x) / Fprime(x));
                    iterations += 1;
                }
            }
        };
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
         * @param alpha The \f$\alpha\f$ parameter of the distribution.
         * @param k The \f$k\f$ parameter of the distribution.
         * @return The value of the CDF at x.
         * The cumulative distribution function (CDF) is calculated using the formula
         * \f[ CDF(x) = x + \frac{\alpha}{k} \cdot \sin(kx) \f]
         */
        double CDF(const double& x, const double& alpha, const double& k) {
            double cdf = x + (alpha / k) * std::sin(k * x);
            return cdf;
        }

        /**
         * @brief Compute the probability density function (PDF) of a distribution in multiple
         * dimensions.
         * @param xvec The input vector containing the coordinates in each dimension.
         * @param alpha The \f$\alpha\f$ parameter of the distribution.
         * @param kw The \f$k_w\f$ parameter for each dimension of the distribution.
         * @param Dim The number of dimensions.
         * @return The value of the PDF at the given coordinates.
         * The probability density function (PDF) is calculated using the formula
         * \f[ PDF(xvec) = \prod_{d=0}^{Dim-1} (1.0 + \alpha \cdot \cos(k_w[d] \cdot xvec[d])) \f]
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

                const auto Alpha = alpha;
                auto K           = k;

                for (unsigned d = 0; d < Dim; ++d) {
                    u       = rand_gen.drand(minU[d], maxU[d]);
                    x(i)[d] = u / (1 + alpha);

                    /**
                     * @brief Calculate the function value at a given point.
                     * @param x The input value.
                     * @return The value of the function at x.
                     * The function f(x) is defined as:
                     * \f[ f(x) = x + \frac{\alpha}{k} \cdot \sin(kx) - u \f]
                     **/
                    auto f = [Alpha, K, d, u](double x) {
                        return x + (Alpha * (Kokkos::sin(K[d] * x) / K[d])) - u;
                    };

                    /**
                     * @brief Calculate the derivative of the function at a given point.
                     * @param x The input value.
                     * @return The value of the derivative of the function at x.
                     * The derivative of the function f(x), denoted as f'(x), is defined as:
                     * \f[ f'(x) = 1 + \alpha \cdot \cos(kx) \f]
                     **/
                    auto fprime = [Alpha, K, d](double x) {
                        return 1 + (Alpha * Kokkos::cos(K[d] * x));
                    };

                    Distribution<double, 3>::Newton1D<decltype(f), decltype(fprime)> solver(f,
                                                                                            fprime);
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
         * @brief Compute the cumulative distribution function (CDF) of a normal distribution.
         * @param x The input value.
         * @param mu The mean (\f$\mu\f$) of the distribution.
         * @param sigma The standard deviation (\f$\sigma\f$) of the distribution.
         * @return The value of the CDF at x.
         * The cumulative distribution function (CDF) is calculated using the formula
         * \f[ CDF(x) = \frac{1}{2} \left[1 +
         * \text{erf}\left(\frac{x-\mu}{\sigma\sqrt{2}}\right)\right] \f]
         */
        double CDF(const double& x, const double& mu, const double& sigma) {
            double cdf = 0.5 * (1.0 + std::erf((x - mu) / (sigma * std::sqrt(2))));
            return cdf;
        }

        /**
         * @brief Compute the probability density function (PDF) of a normal
         * distribution in multiple dimensions.
         * @param xvec The input vector containing the coordinates in each dimension.
         * @param mu The mean vector of the distribution in each dimension.
         * @param sigma The standard deviation vector of the distribution in each
         * dimension.
         * @param Dim The number of dimensions.
         * @return The value of the PDF at the given coordinates.
         * The probability density function (PDF) is calculated using the formula
         * \f[ PDF(xvec) = \prod_{d=0}^{Dim-1} \left(\frac{1}{\sigma_d \sqrt{2\pi}}
         * \exp\left(-\frac{1}{2}\left(\frac{xvec_d -
         * \mu_d}{\sigma_d}\right)^2\right)\right) \f]
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

            /**
             * @brief Calculate the function value at a given point using the error
             * function.
             * @param x The input value.
             * @return The value of the function at x.
             * The function f(x) is defined as:
             * \f[ f(x) = \text{erf}\left(\frac{x - \mu}{\sigma\sqrt{2}}\right) - 2u + 1
             * \f]
             */
            KOKKOS_INLINE_FUNCTION T f(T& x) {
                T F;
                F = Kokkos::erf((x - mu) / (sigma * Kokkos::sqrt(2.0))) - 2 * u + 1;
                return F;
            }

            /**
             * @brief Calculate the derivative of the function at a given point.
             * @param x The input value.
             * @return The value of the derivative of the function at x.
             * The derivative of the function f(x), denoted as f'(x), is defined as:
             * \f[ f'(x) = \frac{1}{\sigma}\sqrt{\frac{2}{\pi}}
             * \exp\left(-\frac{1}{2}\left(\frac{x
             * - \mu}{\sigma}\right)^2\right) \f]
             */
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

    /*
     *
     * BumponTailDistribution
     *
     */

    template <typename T, unsigned Dim = 3>
    class BumponTailDistribution : public Distribution<T, Dim> {
        Vector_t<double, Dim> kw_m;

        T epsilon_m;
        T delta_m;
        Vector_t<double, Dim> sigma_m;
        Vector_t<double, Dim> muBulk_m;
        Vector_t<double, Dim> muBeam_m;

        Vector_t<double, Dim> rmax_m;
        Vector_t<double, Dim> rmin_m;
        Vector_t<double, Dim> hr_m;
        Vector_t<double, Dim> origin_m;

        const size_type totalP_m;

        /**
         * @brief Compute the cumulative distribution function (CDF) of a distribution.
         *
         * This function calculates the cumulative distribution function (CDF) of a
         * distribution with parameters @c delta and @c k.
         *
         * The CDF is computed as follows:
         * \f[
         *     \text{CDF}(x, \alpha, k, \text{dim}) = x + \text{isDimZ} \cdot
         * \left(\frac{\delta}{k} \cdot \sin(k \cdot x)\right) \f]
         *
         * where
         * - @c x is the input value.
         * - @c delta is the delta parameter of the distribution.
         * - @c k is the k parameter of the distribution.
         * - @c dim is the dimension parameter of the distribution.
         * - @c isDimZ is a boolean value indicating if the dimension is Z or not (true
         * or false).
         *
         * @param x The input value.
         * @param delta The delta parameter of the distribution.
         * @param k The k parameter of the distribution.
         * @param dim The dimension parameter of the distribution.
         * @return The value of the CDF at @c x.
         */
        double CDF(const double& x, const double& delta, const double& k, const unsigned& dim) {
            bool isDimZ = (dim == (Dim - 1));
            double cdf  = x + (double)(isDimZ * ((delta / k) * std::sin(k * x)));
            return cdf;
        }

        /**
         * @brief Compute the probability density function (PDF) of a distribution in
         * multiple dimensions.
         *
         * This function computes the probability density function (PDF) of a
         * distribution in multiple dimensions based on the given input vector, alpha
         * parameter, and k parameters. The computation involves the coordinates in each
         * dimension, delta, and trigonometric operations using the cos function.
         *
         * @param xvec The input vector containing the coordinates in each dimension.
         * @param alpha The alpha parameter of the distribution.
         * @param kw The k parameter for each dimension of the distribution.
         * @param Dim The number of dimensions.
         * @return The value of the PDF at the given coordinates.
         *
         * @note This function uses the following formula to compute the PDF:
         * \f[
         * \text{PDF}(\mathbf{x}, \delta, \mathbf{kw}) = 1.0 \times 1.0 \times \left(1.0
         * + \delta \cos(kw[Dim - 1] \cdot xvec[Dim - 1])\right) \f] where:
         * - \f$\mathbf{x}\f$ is the input vector \f$xvec\f$ containing the coordinates
         * in each dimension.
         * - \f$\delta\f$ is the given parameter \f$delta\f$.
         * - \f$\mathbf{kw}\f$ is the vector \f$kw\f$ containing the k parameters for
         * each dimension of the distribution.
         * - \f$Dim\f$ is the given number of dimensions.
         * - \f$\cos(x)\f$ is the cosine function.
         *
         * @see Cosine function: https://en.wikipedia.org/wiki/Cosine
         * @see Probability Density Function (PDF):
         * https://en.wikipedia.org/wiki/Probability_density_function
         */

        KOKKOS_FUNCTION
        double PDF(const Vector_t<double, Dim>& xvec, const double delta,
                   const Vector_t<double, Dim>& kw) {
            double pdf = 1.0 * 1.0 * (1.0 + delta * Kokkos::cos(kw[Dim - 1] * xvec[Dim - 1]));
            return pdf;
        }

        /**
         * @brief A structure representing the Newton method for finding roots of 1D
         * functions.
         * @tparam T The type of the parameters and variables used in the Newton method.
         */

        struct Newton1D {
            double tol   = 1e-12;
            int max_iter = 20;
            double pi    = Kokkos::numbers::pi_v<double>;

            T k, delta, u;

            KOKKOS_INLINE_FUNCTION Newton1D() {}

            KOKKOS_INLINE_FUNCTION Newton1D(const T& k_, const T& delta_, const T& u_)
                : k(k_)
                , delta(delta_)
                , u(u_) {}

            KOKKOS_INLINE_FUNCTION ~Newton1D() {}

            /**
             * @brief Compute the function @f$F(x)@f$.
             *
             * This function calculates the value of the function @f$F(x)@f$ with
             * parameters @c delta and @c k.
             *
             * The function @f$F(x)@f$ is computed as follows:
             * \f[
             *     F(x) = x + (\delta \cdot (\frac{\sin(k \cdot x)}{k})) - u
             * \f]
             *
             * where
             * - @c x is the input value.
             * - @c delta is the delta parameter.
             * - @c k is the k parameter.
             * - @c u is an unspecified parameter that needs to be defined externally.
             *
             * @param x The input value.
             * @return The value of the function @f$F(x)@f$ at @c x.
             */
            KOKKOS_INLINE_FUNCTION T f(T& x) {
                T F;
                F = x + (delta * (Kokkos::sin(k * x) / k)) - u;
                return F;
            }

            /**
             * @brief Compute the derivative of the function @f$F(x)@f$.
             *
             * This function calculates the derivative of the function @f$F(x)@f$ with
             * parameters @c delta and @c k.
             *
             * The derivative of the function @f$F(x)@f$ is computed as follows:
             * \f[
             *     F'(x) = 1 + (\delta \cdot \cos(k \cdot x))
             * \f]
             *
             * where
             * - @c x is the input value.
             * - @c delta is the delta parameter.
             * - @c k is the k parameter.
             *
             * @param x The input value.
             * @return The value of the derivative of the function @f$F(x)@f$ at @c x.
             */
            KOKKOS_INLINE_FUNCTION T fprime(T& x) {
                T Fprime;
                Fprime = 1 + (delta * Kokkos::cos(k * x));
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
            double delta;
            GT sigma, muBulk, muBeam;
            size_type nlocBulk;
            GT k, minU, maxU;
            // Initialize all members
            generate_random(view_type x_, view_type v_, GeneratorPool rand_pool_, double delta_,
                            GT& sigma_, GT& muBulk_, GT& muBeam_, size_type& nlocBulk_, GT& k_,
                            GT& minU_, GT& maxU_)
                : x(x_)
                , v(v_)
                , rand_pool(rand_pool_)
                , delta(delta_)
                , sigma(sigma_)
                , muBulk(muBulk_)
                , muBeam(muBeam_)
                , nlocBulk(nlocBulk_)
                , k(k_)
                , minU(minU_)
                , maxU(maxU_) {}

            KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
                // Get a random number state from the pool for the active thread
                typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

                bool isBeam = (i >= nlocBulk);

                // fixMe this needs to be checked

                value_type muZ =
                    (value_type)((!isBeam) * muBulk[Dim - 1]) + (isBeam * muBeam[Dim - 1]);

                if constexpr (Dim > 1) {
                    for (unsigned d = 0; d < Dim - 1; ++d) {
                        x(i)[d] = rand_gen.drand(minU[d], maxU[d]);
                        v(i)[d] = rand_gen.normal(0.0, sigma[d]);
                    }
                }

                v(i)[Dim - 1] = rand_gen.normal(muZ, sigma[Dim - 1]);

                value_type u  = rand_gen.drand(minU[Dim - 1], maxU[Dim - 1]);
                x(i)[Dim - 1] = u / (1 + delta);
                Newton1D solver(k[Dim - 1], delta, u);
                solver.solve(x(i)[Dim - 1]);

                // Give the state back, which will allow another thread to acquire it
                rand_pool.free_state(rand_gen);
            }
        };

    public:
        BumponTailDistribution(Vector_t<double, Dim> rmax, Vector_t<double, Dim> rmin,
                               Vector_t<double, Dim> hr, Vector_t<double, Dim> origin,
                               const size_type totalP, const char* TestName)
            : Distribution<T, Dim>(TestName)
            , rmax_m(rmax)
            , rmin_m(rmin)
            , hr_m(hr)
            , origin_m(origin)
            , totalP_m(totalP) {
            if (std::strcmp(TestName, "TwoStreamInstability") == 0) {
                // Parameters for two stream instability as in
                //  https://www.frontiersin.org/articles/10.3389/fphy.2018.00105/full
                kw_m      = 0.5;
                sigma_m   = 0.1;
                epsilon_m = 0.5;
                muBulk_m  = -pi / 2.0;
                muBeam_m  = pi / 2.0;
                delta_m   = 0.01;
            } else if (std::strcmp(TestName, "BumponTailInstability") == 0) {
                kw_m      = 0.21;
                sigma_m   = 1.0 / std::sqrt(2.0);
                epsilon_m = 0.1;
                muBulk_m  = 0.0;
                muBeam_m  = 4.0;
                delta_m   = 0.01;
            } else {
                // Default value is two stream instability
                kw_m      = 0.5;
                sigma_m   = 0.1;
                epsilon_m = 0.5;
                muBulk_m  = -pi / 2.0;
                muBeam_m  = pi / 2.0;
                delta_m   = 0.01;
            }
        }
        ~BumponTailDistribution() {}

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
                    ippl::apply(rhoview, args) = PDF(xvec, delta_m, kw_m);
                });
            Kokkos::fence();
        }

        void createParticles(auto P, auto RLayout) {
            auto Regions = RLayout.gethLocalRegions();
            Vector_t<double, Dim> Nr, Dr, minU, maxU;
            int myRank    = ippl::Comm->rank();
            double factor = 1;

            for (unsigned d = 0; d < Dim; ++d) {
                Nr[d] = CDF(Regions(myRank)[d].max(), delta_m, kw_m[d], d)
                        - CDF(Regions(myRank)[d].min(), delta_m, kw_m[d], d);
                Dr[d]   = CDF(rmax_m[d], delta_m, kw_m[d], d) - CDF(rmin_m[d], delta_m, kw_m[d], d);
                minU[d] = CDF(Regions(myRank)[d].min(), delta_m, kw_m[d], d);
                maxU[d] = CDF(Regions(myRank)[d].max(), delta_m, kw_m[d], d);
                factor *= Nr[d] / Dr[d];
            }

            double factorVelBulk      = 1.0 - epsilon_m;
            double factorVelBeam      = 1.0 - factorVelBulk;
            size_type nlocBulk        = (size_type)(factor * factorVelBulk * totalP_m);
            size_type nlocBeam        = (size_type)(factor * factorVelBeam * totalP_m);
            size_type nloc            = nlocBulk + nlocBeam;
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
                          P->R.getView(), P->V.getView(), rand_pool64, delta_m, sigma_m, muBulk_m,
                          muBeam_m, nlocBulk, kw_m, minU, maxU));

            Kokkos::fence();
        }
    };

}  // namespace Distribution
#endif
