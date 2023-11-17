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

        Distribution dist_m;
        const Vector<T, Dim> rmax_m;
        const Vector<T, Dim> rmin_m;
        Vector<T, Dim> umin_m, umax_m;
        size_type ntotal_m;

        /*!
         * @brief Constructor for InverseTransformSampling class.
         *
         * @param dist_ The distribution to sample from.
         * @param rmax_ Maximum range for sampling.
         * @param rmin_ Minimum range for sampling.
         * @param rlayout The region layout.
         * @param ntotal_ Total number of samples to generate.
        */
        template <class RegionLayout>
        InverseTransformSampling(Distribution &dist_r, Vector<T, Dim> &rmax_r, Vector<T, Dim> &rmin_r, RegionLayout& rlayout_r, size_type &ntotal_r)
        : dist_m(dist_r),
        rmax_m(rmax_r),
        rmin_m(rmin_r),
        ntotal_m(ntotal_r){
            const typename RegionLayout::host_mirror_type regions = rlayout_r.gethLocalRegions();
            
            int rank = ippl::Comm->rank();
            Vector<T, Dim> nr_m, dr_m;
            for (unsigned d = 0; d < Dim; ++d) {
                nr_m[d] = dist_m.getCdf(regions(rank)[d].max(), d) - dist_m.getCdf(regions(rank)[d].min(),d);
                dr_m[d]   = dist_m.getCdf(rmax_m[d],d) - dist_m.getCdf(rmin_m[d],d);
                umin_m[d] = dist_m.getCdf(regions(rank)[d].min(),d);
                umax_m[d] = dist_m.getCdf(regions(rank)[d].max(),d);
            }
            T pnr = std::accumulate(nr_m.begin(), nr_m.end(), 1.0, std::multiplies<T>());
            T pdr = std::accumulate(dr_m.begin(), dr_m.end(), 1.0, std::multiplies<T>());

            T factor = pnr / pdr;
            nlocal_m      = (size_type) (factor * ntotal_m);
            size_type nglobal = 0;

            MPI_Allreduce(&nlocal_m, &nglobal, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                          ippl::Comm->getCommunicator());

            int rest = (int)(ntotal_m - nglobal);
            
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
            Distribution targetdist_m;
            view_type sample_m;
            GeneratorPool pool_m;
            Vector<T, Dim> minbound_m;
            Vector<T, Dim> maxbound_m;
            unsigned int dim_m;

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
            fill_random(Distribution &dist_r, view_type &x_r, GeneratorPool &rand_pool_r, Vector<T, Dim> &umin_r, Vector<T, Dim> &umax_r, unsigned int &d_r)
            : targetdist_m(dist_r)
            , sample_m(x_r)
            , pool_m(rand_pool_r)
            , minbound_m(umin_r)
            , maxbound_m(umax_r)
            , dim_m(d_r){}
            
            /*!
             * @brief Operator to fill random values.
             *
             * @param i Index for the random values.
            */
            KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
                typename GeneratorPool::generator_type rand_gen = pool_m.get_state();
                
                value_type u = 0.0;

                u       = rand_gen.drand(minbound_m[dim_m], maxbound_m[dim_m]);

                // first guess for Newton-Raphson
                sample_m(i)[dim_m] = targetdist_m.getEstimate(u, dim_m);

                // solve
                ippl::random::detail::NewtonRaphson<value_type, Distribution> solver(targetdist_m);

                solver.solve(dim_m, sample_m(i)[dim_m], u);

                pool_m.free_state(rand_gen);

            }
        };
        
        /*!
         * @brief Get the local number of samples.
         *
         * @returns The local number of samples.
        */
        KOKKOS_INLINE_FUNCTION size_type getLocalSamplesNum() const { return nlocal_m; }
        
        /*!
         * @brief Generate random samples using inverse transform sampling.
         *
         * @param view The view to fill with random samples.
         * @param rand_pool64 The random number generator pool.
        */
        void generate(view_type view, Kokkos::Random_XorShift64_Pool<> rand_pool64) {
            Vector<T, Dim> minbound_m = umin_m;
            Vector<T, Dim> maxbound_m = umax_m;
            Distribution targetdist_m = dist_m;
            size_type numlocal_m = nlocal_m;
            for (unsigned d = 0; d < Dim; ++d) {
              Kokkos::parallel_for(numlocal_m, fill_random<Kokkos::Random_XorShift64_Pool<>>(targetdist_m, view, rand_pool64, minbound_m, maxbound_m, d));
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
