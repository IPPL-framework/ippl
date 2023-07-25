#ifndef DISTRIBUTION_H
#define DISTRIBUTION_H

#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_Random.hpp>

#include "PICManager/PICManager.hpp"

namespace Distribution {

    const unsigned Dim = 3;

    double CDF(const double& x, const double& alpha, const double& k) {
        double cdf = x + (alpha / k) * std::sin(k * x);
        return cdf;
    }

    /*
      FixMe: this needs to go to Distributions
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

    template <typename T>
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

    template <typename T, class GeneratorPool, unsigned Dim>
    struct generate_random {
        using view_type  = typename ippl::detail::ViewType<T, 1>::view_type;
        using value_type = typename T::value_type;
        // Output View for the random numbers
        view_type x, v;

        // The GeneratorPool
        GeneratorPool rand_pool;

        value_type alpha;

        T k, minU, maxU;

        // Initialize all members
        generate_random(view_type x_, view_type v_, GeneratorPool rand_pool_, value_type& alpha_,
                        T& k_, T& minU_, T& maxU_)
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
                Newton1D<value_type> solver(k[d], alpha, u);
                solver.solve(x(i)[d]);
                v(i)[d] = rand_gen.normal(0.0, 1.0);
            }

            // Give the state back, which will allow another thread to acquire it
            rand_pool.free_state(rand_gen);
        }
    };

    template <typename T, unsigned Dim = 3>
    class Distribution {
        double alpha_m;
        Vector_t<double, Dim> kw_m;
        Vector_t<double, Dim> rmax_m;
        Vector_t<double, Dim> rmin_m;
        Vector_t<double, Dim> hr_m;
        Vector_t<double, Dim> origin_m;
        const size_type totalP_m;

    public:
        Distribution(double alpha, Vector_t<double, Dim> kw, Vector_t<double, Dim> rmax,
                     Vector_t<double, Dim> rmin, Vector_t<double, Dim> hr,
                     Vector_t<double, Dim> origin, const size_type totalP)
            : alpha_m(alpha)
            , kw_m(kw)
            , rmax_m(rmax)
            , rmin_m(rmin)
            , hr_m(hr)
            , origin_m(origin)
            , totalP_m(totalP) {}

        ~Distribution() {}

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
                    ippl::apply(rhoview, args) = PDF(xvec, alpha_m, kw_m, Dim);
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
                nloc, generate_random<Vector_t<double, Dim>, Kokkos::Random_XorShift64_Pool<>, Dim>(
                          P->R.getView(), P->V.getView(), rand_pool64, alpha_m, kw_m, minU, maxU));

            Kokkos::fence();
        }
    };

}  // namespace Distribution
#endif
