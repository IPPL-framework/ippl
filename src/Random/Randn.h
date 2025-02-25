// Struct Randn
//   This struct can be used for sampling normal distribution function
//   on unbounded domain.
//
#ifndef IPPL_RANDN_H
#define IPPL_RANDN_H

#include "Random/Distribution.h"
#include "Random/Utility.h"

namespace ippl {
    namespace random {

        /*!
         * @struct randn
         * @brief Functor to generate random numbers from a normal distribution.
         *
         * This functor can be used to generates random numbers from a normal distribution with
         * mean 0 and standard deviation 1.
         *
         * @tparam T Data type of the random numbers.
         * @tparam GeneratorPool Type of the random number generator pool.
         * @tparam Dim Dimensionality of the random numbers.
         */
        template <typename T, unsigned Dim>
        struct randn {
            using view_type =
                typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;
            using GeneratorPool = typename Kokkos::Random_XorShift64_Pool<>;

            // Output View for the random numbers
            view_type v;

            // The GeneratorPool
            GeneratorPool rand_pool;

            T mu[Dim];
            T sd[Dim];
            /*!
             * @brief Constructor for the randn functor.
             *
             * @param v_ Output view for the random numbers.
             * @param rand_pool_ The random number generator pool.
             * @param mu The array of means in each dimension
             * @param sd The array of standard deviation in each dimension
             */
            KOKKOS_INLINE_FUNCTION randn(view_type v_, GeneratorPool rand_pool_, T* mu_p, T* sd_p)
                : v(v_)
                , rand_pool(rand_pool_) {
                for (unsigned int i = 0; i < Dim; i++) {
                    mu[i] = mu_p[i];
                    sd[i] = sd_p[i];
                }
            }

            KOKKOS_INLINE_FUNCTION randn(view_type v_, GeneratorPool rand_pool_)
                : v(v_)
                , rand_pool(rand_pool_) {
                for (unsigned int i = 0; i < Dim; i++) {
                    mu[i] = 0.0;
                    sd[i] = 1.0;
                }
            }

            /*!
             * @brief Getter function for mean in idx dimension
             *
             * @param idx The index indicating the dimension.
             */
            KOKKOS_INLINE_FUNCTION const T& getMu(unsigned int idx) const { return mu[idx]; }

            /*!
             * @brief Getter function for the standard deviation in idx dimension
             *
             * @param idx The index indicating the dimension.
             */
            KOKKOS_INLINE_FUNCTION const T& getSd(unsigned int idx) const { return sd[idx]; }

            /*!
             * @brief Operator to generate random numbers.
             *
             * @param i Index for the random numbers.
             */
            KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
                // Get a random number state from the pool for the active thread
                typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

                for (unsigned d = 0; d < Dim; ++d) {
                    v(i)[d] = mu[d] + sd[d] * rand_gen.normal(0.0, 1.0);
                }

                // Give the state back, which will allow another thread to acquire it
                rand_pool.free_state(rand_gen);
            }
        };
    }  // namespace random
}  // namespace ippl

#endif
