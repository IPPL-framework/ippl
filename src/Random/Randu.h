// Struct Randu
//   This struct can be used for sampling uniform distribution function
//   on bounded domain.
//
#ifndef IPPL_RANDU_H
#define IPPL_RANDU_H

#include "Random/Utility.h"
#include "Random/Distribution.h"

namespace ippl {
  namespace random {

    /*!
     * @struct randu
     * @brief Functor to generate random numbers from a uniform distribution.
     *
     * This functor generates random numbers from a uniform distribution in the
     * range [rmin, rmax].
     *
     * @tparam T Data type of the random numbers.
     * @tparam GeneratorPool Type of the random number generator pool.
     * @tparam Dim Dimensionality of the random numbers.
    */
    template <typename T, unsigned Dim>
    struct randu {
      using view_type  = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;
      using GeneratorPool = typename Kokkos::Random_XorShift64_Pool<>;

      // Output View for the random numbers
      view_type v;

      // The GeneratorPool
      GeneratorPool rand_pool;

      T rmin[Dim];
      T rmax[Dim];

      /*!
       * @brief Constructor for the randu functor.
       *
       * @param v_ Output view for the random numbers.
       * @param rand_pool_ The random number generator pool.
       * @param rmin The minimum of random number.
       * @param rmax The maximum of random number.
      */
      KOKKOS_INLINE_FUNCTION randu(view_type v_, GeneratorPool rand_pool_, T *rmin_p, T *rmax_p)
        : v(v_)
        , rand_pool(rand_pool_) {
           for(unsigned int i=0; i<Dim; i++){
                rmin[i] = rmin_p[i];
                rmax[i] = rmax_p[i];
             }
	}

      KOKKOS_INLINE_FUNCTION randu(view_type v_, GeneratorPool rand_pool_)
        : v(v_)
        , rand_pool(rand_pool_) {
           for(unsigned int i=0; i<Dim; i++){
                rmin[i] = 0.0;
                rmax[i] = 1.0;
             }
	}

      /*!
       * @brief Operator to generate random numbers.
       *
       * @param i Index for the random numbers.
      */
      KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
        // Get a random number state from the pool for the active thread
        typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

        for (unsigned d = 0; d < Dim; ++d) {
            v(i)[d] = rand_gen.drand(rmin[d], rmax[d]);
        }

	// Give the state back, which will allow another thread to acquire it
        rand_pool.free_state(rand_gen);
      }
    };
  }  // namespace random
}  // namespace ippl

#endif
