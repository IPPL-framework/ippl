#ifndef IPPL_RANDOM_UTILITY_H
#define IPPL_RANDOM_UTILITY_H

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_Random.hpp>
#include "Types/ViewTypes.h"

namespace ippl {
  namespace random {
    namespace detail{
      /*!
       * @struct NewtonRaphson
       * @brief Functor for solving equations using the Newton-Raphson method.
       *
       * In particular, find the root x of the equation dist.obj(x, u)= 0 for a given u using Newton-Raphson method.
       *
       * @tparam T Data type for the equation variables.
       * @tparam Distribution Class of target distribution to sample from.
       * @param dist Distribution object providing objective function cdf(x)-u and its derivative.
       * @param atol Absolute tolerance for convergence (default: 1.0e-12).
       * @param max_iter Maximum number of iterations (default: 20).
      */
      template <typename T, class Distribution>
      struct NewtonRaphson {
          Distribution dist;
          double atol   = 1e-12;
          unsigned int max_iter = 20;

          KOKKOS_FUNCTION
          NewtonRaphson() = default;

          KOKKOS_FUNCTION
          ~NewtonRaphson() = default;

          KOKKOS_INLINE_FUNCTION NewtonRaphson(const Distribution &dist_)
          : dist(dist_){}

          /*!
           * @brief Solve an equation using the Newton-Raphson method.
           *
           * This function iteratively solves an equation of the form "cdf(x) - u = 0"
           * for a given sample `u` using the Newton-Raphson method.
           *
           * @param d Dimension index.
           * @param x Variable to solve for (initial guess and final solution).
           * @param u Given sample from a uniform distribution [0, 1].
          */
          KOKKOS_INLINE_FUNCTION void solve(unsigned int d, T& x, T& u){
              unsigned int iter = 0;
              while (iter < max_iter && Kokkos::fabs(dist.getObjFunc(x, d, u)) > atol) {
                  // Find x, such that "cdf(x) - u = 0" for a given sample of u~uniform(0,1)
                  x = x - (dist.getObjFunc(x, d, u) / dist.getDerObjFunc(x, d));
                  iter += 1;
              }
          }
      };
    }  // name space detail
  }  // namespace random
}  // namespace ippl

#endif
