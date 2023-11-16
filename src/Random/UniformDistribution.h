// Class NormalDistribution
//   This class can be used for sampling normal distribution function
//   on bounded domain, e.g. using Inverse Transform Sampling.

#ifndef IPPL_UNIFORM_DISTRIBUTION_H
#define IPPL_UNIFORM_DISTRIBUTION_H

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_Random.hpp>
#include "Types/ViewTypes.h"

namespace ippl {
  namespace random {

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

  }  // namespace random
}  // namespace ippl

#endif
