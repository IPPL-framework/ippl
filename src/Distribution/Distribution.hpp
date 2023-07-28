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

    const unsigned Dim = 3;

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
               const Vector_t<double, Dim>& kw, const unsigned Dim) {
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
    template <typename T>
    struct Newton1D {
        double tol   = 1e-12;
        int max_iter = 20;
        double pi    = Kokkos::numbers::pi_v<double>;

        T k, alpha, u;

        /**
         * @brief Default constructor for the Newton1D struct.
         */
        KOKKOS_INLINE_FUNCTION Newton1D() {}

        /**
         * @brief Constructor for the Newton1D struct.
         * @param k_ The k parameter of the distribution.
         * @param alpha_ The alpha parameter of the distribution.
         * @param u_ The u parameter of the distribution.
         */
        KOKKOS_INLINE_FUNCTION Newton1D(const T& k_, const T& alpha_, const T& u_)
            : k(k_)
            , alpha(alpha_)
            , u(u_) {}

        /**
         * @brief Destructor for the Newton1D struct.
         */
        KOKKOS_INLINE_FUNCTION ~Newton1D() {}

        /**
         * @brief Compute the function value for the given variable in the Newton method.
         * @param x The variable for which to compute the function value.
         * @return The value of the function at x.
         */
        KOKKOS_INLINE_FUNCTION T f(T& x) {
            T F;
            F = x + (alpha * (Kokkos::sin(k * x) / k)) - u;
            return F;
        }
    };

    /**
     * @brief Function to create particles based on the distribution.
     * @param P Particle object where the generated particles will be stored.
     * @param RLayout Layout for generating the particles.
     */
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

        Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(42 + 100 * ippl::Comm->rank()));
        Kokkos::parallel_for(
            nloc, generate_random<Vector_t<double, Dim>, Kokkos::Random_XorShift64_Pool<>, Dim>(
                      P->R.getView(), P->V.getView(), rand_pool64, alpha_m, kw_m, minU, maxU));

        Kokkos::fence();
    }

}  // namespace Distribution
#endif
