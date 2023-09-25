#ifndef IPPL_RANDOM_H
#define IPPL_RANDOM_H

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_Random.hpp>

namespace ippl {
  namespace random {
  template <typename T, class GeneratorPool, unsigned Dim>
  struct randn_functor {
    using view_type  = typename ippl::detail::ViewType<T, 1>::view_type;
    // Output View for the random numbers
    view_type v;

    // The GeneratorPool
    GeneratorPool rand_pool;

    // Initialize all members
    randn_functor(view_type v_, GeneratorPool rand_pool_)
        : v(v_)
        , rand_pool(rand_pool_) {}

    KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
        // Get a random number state from the pool for the active thread
        typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

        for (unsigned d = 0; d < Dim; ++d) {
            v(i)[d] = rand_gen.normal(0.0, 1.0);
        }

        // Give the state back, which will allow another thread to acquire it
        rand_pool.free_state(rand_gen);
    }
  };

  template <typename T, class GeneratorPool, unsigned Dim>
  struct randu_functor {
    using view_type  = typename ippl::detail::ViewType<T, 1>::view_type;
    // Output View for the random numbers
    view_type v;

    // The GeneratorPool
    GeneratorPool rand_pool;

    // Initialize all members
    randu_functor(view_type v_, GeneratorPool rand_pool_)
        : v(v_)
        , rand_pool(rand_pool_) {}

    KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
        // Get a random number state from the pool for the active thread
        typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

        for (unsigned d = 0; d < Dim; ++d) {
            v(i)[d] = rand_gen.drand(0.0, 1.0);
        }

	// Give the state back, which will allow another thread to acquire it
        rand_pool.free_state(rand_gen);
    }
  };
 }  // namespace random
}  // namespace ippl

#endif
