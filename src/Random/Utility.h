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

#ifndef IPPL_RANDOM_UTILITY_H
#define IPPL_RANDOM_UTILITY_H

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_Random.hpp>
#include "Types/ViewTypes.h"

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
      KOKKOS_INLINE_FUNCTION randu(view_type v_, GeneratorPool rand_pool_, T *rmin_, T *rmax_)
        : v(v_)
        , rand_pool(rand_pool_) {
           for(unsigned int i=0; i<Dim; i++){
                rmin[i] = rmin_[i];
                rmax[i] = rmax_[i];
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

    template<typename T>
    KOKKOS_FUNCTION T uniform_cdf_func(T x){
      return x;
    }

    template<typename T>
    KOKKOS_FUNCTION T uniform_pdf_func(){
      return 1.;
    }

    template<typename T>
    KOKKOS_FUNCTION T uniform_estimate_func(T u){
      return u;
    }

    namespace detail{
      /*!
       * @struct NewtonRaphson
       * @brief Functor for solving equations using the Newton-Raphson method.
       *
       * In particular, find the root x of the equation dist.obj(x, u)= 0 for a given u using Newton-Raphson method.
       *
       * @tparam T Data type for the equation variables.
      */
      template <typename T>
      struct NewtonRaphson {
          KOKKOS_FUNCTION
          NewtonRaphson() = default;

          KOKKOS_FUNCTION
          ~NewtonRaphson() = default;

          /*!
           * @brief Solve an equation using the Newton-Raphson method.
           *
           * This function iteratively solves an equation of the form "cdf(x) - u = 0"
           * for a given sample `u` using the Newton-Raphson method.
           *
           * @tparam Distribution Type of the distribution providing the objective function and its derivative.
           * @param dist Distribution object providing objective function cdf(x)-u and its derivative.
           * @param d Dimension index.
           * @param x Variable to solve for (initial guess and final solution).
           * @param u Given sample from a uniform distribution [0, 1].
           * @param atol Absolute tolerance for convergence (default: 1.0e-12).
           * @param max_iter Maximum number of iterations (default: 20).
          */
          template <class Distribution>
          KOKKOS_INLINE_FUNCTION void solve(Distribution dist, int d, T& x, T& u, T atol = 1.0e-12,
                                            unsigned int max_iter = 20) {
              unsigned int iter = 0;
              while (iter < max_iter && Kokkos::fabs(dist.obj_func(x, d, u)) > atol) {
                  // Find x, such that "cdf(x) - u = 0" for a given sample of u~uniform(0,1)
                  x = x - (dist.obj_func(x, d, u) / dist.der_obj_func(x, d));
                  iter += 1;
              }
          }
      };
    }  // name space detial
  }  // namespace random
}  // namespace ippl

#endif
