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
        size_type ntotal_m;
        Vector<T, Dim> rmax_m;
        Vector<T, Dim> rmin_m;
        Vector<T, Dim> umin_m, umax_m;

        /*!
         * @brief Constructor for InverseTransformSampling class with domain decomposition.
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
        ntotal_m(ntotal_r){

            updateBounds(rmax_r, rmin_r, rlayout_r);

        }

        /*!
         * @brief Constructor for InverseTransformSampling class without applying domain decomposition..
         *
         * @param dist_ The distribution to sample from.
         * @param rmax_ Maximum range for sampling.
         * @param rmin_ Minimum range for sampling.
         * @param rlayout The region layout.
         * @param ntotal_ Total number of samples to generate.
        */
	template <class RegionLayout>
        InverseTransformSampling(Distribution &dist_r, Vector<T, Dim> &rmax_r, Vector<T, Dim> &rmin_r, Vector<T, Dim> &locrmax_r, Vector<T, Dim> &locrmin_r, size_type &ntotal_r)
        : dist_m(dist_r),
        ntotal_m(ntotal_r){

            updateBounds(rmax_r, rmin_r, locrmax_r, locrmin_r);

        }

        /*!
         * @brief Constructor for InverseTransformSampling class.
         *        In this method, we do not consider any domain decomposition.
         *
         * @param dist_ The distribution to sample from.
         * @param rmax_ Maximum range for sampling.
         * @param rmin_ Minimum range for sampling.
         * @param ntotal_ Total number of samples to generate.
        */
	template <class RegionLayout>
        InverseTransformSampling(Distribution &dist_r, Vector<T, Dim> &rmax_r, Vector<T, Dim> &rmin_r, size_type &ntotal_r)
        : dist_m(dist_r),
        ntotal_m(ntotal_r){

            updateBounds(rmax_r, rmin_r);

            nlocal_m = ntotal_m;

        }

        /*!
         * @brief Updates the sampling bounds and reinitializes internal variables.
         *
         * This method allows the user to update the minimum and maximum bounds 
         * for the sampling process using the region layout. It recalculates
         * the cumulative distribution function (CDF) values for the new bounds and 
         * updates the internal variables to reflect these changes.
         *
         * @param new_rmax The new maximum range for sampling. This vector defines
         *                 the upper bounds for each dimension.
         * @param new_rmin The new minimum range for sampling. This vector defines
         *                 the lower bounds for each dimension.
         * @param new_rlayout The new region layout that defines the subdivisions
         *                    of the sampling space across different ranks.
        */
        template <class RegionLayout>
        void updateBounds(Vector<T, Dim>& rmax, Vector<T, Dim>& rmin, RegionLayout& rlayout) {
            rmax_m = rmax;
            rmin_m = rmin;

            const typename RegionLayout::host_mirror_type regions = rlayout.gethLocalRegions();
            int rank = ippl::Comm->rank();
            Vector<T, Dim> nr_m, dr_m;
            for (unsigned d = 0; d < Dim; ++d) {
               nr_m[d] = dist_m.getCdf(regions(rank)[d].max(), d) - dist_m.getCdf(regions(rank)[d].min(), d);
               dr_m[d] = dist_m.getCdf(rmax_m[d], d) - dist_m.getCdf(rmin_m[d], d);
               umin_m[d] = dist_m.getCdf(regions(rank)[d].min(), d);
               umax_m[d] = dist_m.getCdf(regions(rank)[d].max(), d);
            }
            T pnr = std::accumulate(nr_m.begin(), nr_m.end(), 1.0, std::multiplies<T>());
            T pdr = std::accumulate(dr_m.begin(), dr_m.end(), 1.0, std::multiplies<T>());

            T factor = pnr / pdr;
            nlocal_m = (size_type)(factor * ntotal_m);

            size_type nglobal = 0;
            MPI_Allreduce(&nlocal_m, &nglobal, 1, MPI_UNSIGNED_LONG, MPI_SUM, ippl::Comm->getCommunicator());

            int rest = (int)(ntotal_m - nglobal);
            if (rank < rest) {
                ++nlocal_m;
            }
         }

        /*!
         * @brief Updates the sampling bounds and reinitializes internal variables.
         *
         * This method allows the user to update the minimum and maximum bounds 
         * for the sampling process It recalculates
         * the cumulative distribution function (CDF) values for the new bounds and 
         * updates the internal variables to reflect these changes.
         *
         * @param rmax The new maximum range for sampling. This vector defines
         *                 the upper bounds for each dimension.
         * @param rmin The new minimum range for sampling. This vector defines
         *                 the lower bounds for each dimension.
         * @param locrmax  The new local maximum range for sampling. This vector defines
         *                 the upper bounds for each dimension for a given rank.
         * @param locrmin  The new minimum range for sampling. This vector defines
         *                 the lower bounds for each dimension for a given rank.
        */
        void updateBounds(Vector<T, Dim>& rmax, Vector<T, Dim>& rmin, Vector<T, Dim>& locrmax, Vector<T, Dim>& locrmin) {
            rmax_m = rmax;
            rmin_m = rmin;

	    int rank = ippl::Comm->rank();
            Vector<T, Dim> nr_m, dr_m;
            for (unsigned d = 0; d < Dim; ++d) {
               nr_m[d] = dist_m.getCdf(locrmax[d], d) - dist_m.getCdf(locrmin[d], d);
               dr_m[d] = dist_m.getCdf(rmax_m[d], d) - dist_m.getCdf(rmin_m[d], d);
               umin_m[d] = dist_m.getCdf(locrmin[d], d);
               umax_m[d] = dist_m.getCdf(locrmax[d], d);
            }
            T pnr = std::accumulate(nr_m.begin(), nr_m.end(), 1.0, std::multiplies<T>());
            T pdr = std::accumulate(dr_m.begin(), dr_m.end(), 1.0, std::multiplies<T>());

            T factor = pnr / pdr;
            nlocal_m = (size_type)(factor * ntotal_m);

            size_type nglobal = 0;
            MPI_Allreduce(&nlocal_m, &nglobal, 1, MPI_UNSIGNED_LONG, MPI_SUM, ippl::Comm->getCommunicator());

            int rest = (int)(ntotal_m - nglobal);
            if (rank < rest) {
                ++nlocal_m;
            }
         }

	/*!
         * @brief Updates the sampling bounds using the CDF without any domain decomposition.
         *
         * This method allows the user to update the minimum and maximum bounds 
         * for the inverse transform sampling method. It recalculates
         * the cumulative distribution function (CDF) values for the new bounds and 
         * updates the internal variables to reflect these changes.
         *
         * @param new_rmax The new maximum range for sampling. This vector defines
         *                 the upper bounds for each dimension.
         * @param new_rmin The new minimum range for sampling. This vector defines
         *                 the lower bounds for each dimension.
        */
        void updateBounds(Vector<T, Dim>& new_rmax, Vector<T, Dim>& new_rmin) {
            rmax_m = new_rmax;
            rmin_m = new_rmin;

            Vector<T, Dim> nr_m, dr_m;
            for (unsigned d = 0; d < Dim; ++d) {
               umin_m[d] = dist_m.getCdf(rmin_m[d], d);
               umax_m[d] = dist_m.getCdf(rmax_m[d], d);
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
         * @brief Set the local number of particles.
         *
         * @param nlocal The new number of local particles.
        */
        KOKKOS_INLINE_FUNCTION void setLocalSamplesNum(size_type nlocal) {
            nlocal_m = nlocal;
        }

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
