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
//
// Class NormalDistribution
//   This class can be used for sampling normal distribution function
//   on bounded domain, e.g. using Inverse Transform Sampling.
//
#ifndef IPPL_NORMAL_DISTRIBUTION_H
#define IPPL_NORMAL_DISTRIBUTION_H

#include "Random/Utility.h"
#include "Random/Distribution.h"

namespace ippl {
  namespace random {
    /*!
       * @file NormalDistribution.h
       * @class NormalDistribution
    */
    template<typename T, unsigned Dim>
    class NormalDistribution : public ippl::random::Distribution<T, Dim, 2*Dim,
                          ippl::random::normal_pdf<T, Dim>,
                          ippl::random::normal_cdf<T, Dim>,
                          ippl::random::normal_estimate<T, Dim>>{
    public:
      /*!
       * @brief Constructor for the Normal Distribution class.
       * The constructor takes an array of parameters of normal distribution, i.e. mean and standard variation.
      */
      KOKKOS_INLINE_FUNCTION NormalDistribution(const T *par_)
                              : ippl::random::Distribution<T, Dim, 2*Dim,
                              ippl::random::normal_pdf<T, Dim>,
                              ippl::random::normal_cdf<T, Dim>,
                              ippl::random::normal_estimate<T, Dim>>(par_) {}
    };
  }  // namespace random
}  // namespace ippl

#endif
