// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 * This program was prepared by PSI.
 * All rights in the program are reserved by PSI.
 * Neither PSI nor the author(s)
 * makes any warranty, express or implied, or assumes any liability or
 * responsibility for the use of this software
 *
 * Visit www.amas.web.psi for more details
 *
 ***************************************************************************/

// Class InverseTransformSampling
//   This class can be used for generating samples of a given distribution function class using
//   inverse of its cumulative distribution on host or device.
//

#ifndef IPPL_INVERSE_TRANSFORM_SAMPLING_H
#define IPPL_INVERSE_TRANSFORM_SAMPLING_H

#include "Random/Utility.h"

namespace ippl {
  namespace random {
    /*!
     * @file InverseTransformSampling.h
     * @class InverseTransformSampling
     * @brief A class for inverse transform sampling.
     *
     * This class performs inverse transform sampling for a given distribution.
     *
     * @tparam T Data type.
     * @tparam Dim Dimensionality of the sample space.
     * @tparam DeviceType The device type for Kokkos.
     * @tparam Distribution Type of the distribution to sample from.
    */
    template <typename T, unsigned Dim, class DeviceType, class Distribution>
    class InverseTransformSampling{
    public:
        using view_type = typename ippl::detail::ViewType<Vector<T, Dim>, 1>::view_type;
        using size_type = ippl::detail::size_type;
        /*!
         * @param dist_ The distribution to sample from.
         * @param rmax_ Maximum range for sampling.
         * @param rmin_ Minimum range for sampling.
         * @param rlayout The region layout.
         * @param ntotal_ Total number of samples to generate.
        */
        
        Distribution dist;
        const Vector<T, Dim> rmax;
        const Vector<T, Dim> rmin;
        Vector<T, Dim> umin, umax;
        size_type ntotal;
        /*!
         * @brief Constructor for InverseTransformSampling class.
        */
        template <class RegionLayout>
        InverseTransformSampling(Distribution &dist_, Vector<T, Dim> &rmax_, Vector<T, Dim> &rmin_, RegionLayout& rlayout, size_type &ntotal_)
        : dist(dist_),
        rmax(rmax_),
        rmin(rmin_),
        ntotal(ntotal_){
            const typename RegionLayout::host_mirror_type regions = rlayout.gethLocalRegions();
            
            int rank = ippl::Comm->rank();
            Vector<T, Dim> nr_m, dr_m;
            for (unsigned d = 0; d < Dim; ++d) {
                nr_m[d] = dist.cdf(regions(rank)[d].max(), d) - dist.cdf(regions(rank)[d].min(),d);
                dr_m[d]   = dist.cdf(rmax[d],d) - dist.cdf(rmin[d],d);
                umin[d] = dist.cdf(regions(rank)[d].min(),d);
                umax[d] = dist.cdf(regions(rank)[d].max(),d);
            }
            T pnr = std::accumulate(nr_m.begin(), nr_m.end(), 1.0, std::multiplies<T>());
            T pdr = std::accumulate(dr_m.begin(), dr_m.end(), 1.0, std::multiplies<T>());
            
            T factor = pnr / pdr;
            nlocal_m      = (size_type) (factor * ntotal);
            size_type nglobal = 0;

            MPI_Allreduce(&nlocal_m, &nglobal, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                          ippl::Comm->getCommunicator());

            int rest = (int)(ntotal - nglobal);
            
            if (rank < rest) {
                ++nlocal_m;
            }

        }

        /*!
         * @brief Deconstructor for InverseTransformSampling class.
        */
        ~InverseTransformSampling(){}

        /*!
         * @brief Functor that is used for generating samples.
        */
        template <class GeneratorPool>
        struct fill_random {
            using value_type = T;
            Distribution dist;
            view_type x;
            GeneratorPool rand_pool;
            Vector<T, Dim> umin_m;
            Vector<T, Dim> umax_m;
            unsigned int d;
 
            /*!
             * @brief Constructor for the fill_random functor.
             *
             * @param dist_ The distribution to sample from.
             * @param x_ The view to generate samples in.
             * @param rand_pool_ The random number generator pool.
             * @param umin_ Minimum cumulative distribution values.
             * @param umax_ Maximum cumulative distribution values.
            */
            KOKKOS_FUNCTION
            fill_random(Distribution &dist_, view_type &x_, GeneratorPool &rand_pool_, Vector<T, Dim> &umin_, Vector<T, Dim> &umax_, unsigned int &d_)
            : dist(dist_)
            , x(x_)
            , rand_pool(rand_pool_)
            , umin_m(umin_)
            , umax_m(umax_)
            , d(d_){}
            
            /*!
             * @brief Operator to fill random values.
             *
             * @param i Index for the random values.
            */
            KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
                typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();
                
                value_type u = 0.0;

                ippl::random::detail::NewtonRaphson<value_type, Distribution> solver(dist);

                u       = rand_gen.drand(umin_m[d], umax_m[d]);
                    
                // first guess for Newton-Raphson
                x(i)[d] = dist.estimate(u, d);
                    
                // solve
                solver.solve(d, x(i)[d], u);

                rand_pool.free_state(rand_gen);

            }
        };
        
        /*!
         * @brief Get the local number of samples.
         *
         * @returns The local number of samples.
        */
        KOKKOS_INLINE_FUNCTION size_type getLocalNum() const { return nlocal_m; }
        
        /*!
         * @brief Generate random samples using inverse transform sampling.
         *
         * @param view The view to fill with random samples.
         * @param rand_pool64 The random number generator pool.
        */
        void generate(view_type view, Kokkos::Random_XorShift64_Pool<> rand_pool64) {
            Vector<T, Dim> umin_ = umin;
            Vector<T, Dim> umax_ = umax;
            Distribution dist_ = dist;
            size_type nlocal_ = nlocal_m;
            for (unsigned d = 0; d < Dim; ++d) {
              Kokkos::parallel_for(nlocal_, fill_random<Kokkos::Random_XorShift64_Pool<>>(dist_, view, rand_pool64, umin_, umax_, d));
              Kokkos::fence();
              ippl::Comm->barrier();
            }
        }
    private:
        size_type nlocal_m;
    };
  
  }  // namespace random
}  // namespace ippl

#endif
