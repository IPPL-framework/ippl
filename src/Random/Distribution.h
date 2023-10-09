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

// Class Distribution
//   This class can be used for creating a distribution object
//   with custom pdf, cdf, and estimate function that is used for
//   the sampling method.
//

#ifndef IPPL_DISTRIBUTION_H
#define IPPL_DISTRIBUTION_H

#include "Random/Utility.h"
#include "Types/ViewTypes.h"

namespace ippl {
  namespace random {
  
    /*!
     * @file Distribution.h
     * @class Distribution
    */
  
   /*!
    * @ingroup Distribution
    * @brief A class that represents a distribution.
    * @tparam T Datatype.
    * @tparam Dim Dimensionality of sample space.
    * @tparam DimP Dimensionality of the parameter array.
    * @tparam PDF Struct type for the PDF (Probability Density Function).
    * @tparam CDF Struct type for the CDF (Cumulative Distribution Function).
    * @tparam ESTIMATE Struct type for the ESTIMATE function.
    */
    template <typename T, unsigned Dim, unsigned DimP, typename PDF, typename CDF, typename ESTIMATE>
    class Distribution {
    public:
        
       /*!
        * @param par_m An array of distribution parameters.
        * @param pdf_m PDF of the distribution class as a member functor.
        * @param cdf_m CDF of the distribution class as a member functor.
        * @param estimate_m Estimate of the initial guess for the sampling method as a member functor.
       */
       T par_m[DimP];
       PDF pdf_m;
       CDF cdf_m;
       ESTIMATE estimate_m;
        
       /*!
        * @ingroup Distribution
        * @brief Constructor for the Distribution class.
        * @param par_ Pointer to the parameter array.
       */
       KOKKOS_INLINE_FUNCTION Distribution(const T *par_) {
            for(unsigned int i=0; i<DimP; i++){
                par_m[i] = par_[i];
            }
       }
        
       /*!
        * @brief A wrapper to change the signiture arguments of pdf in each dimension d
        * from (x, d, par) to (x, d).
       */
       KOKKOS_INLINE_FUNCTION T pdf(T x, unsigned int d) const{
          return pdf_m(x, d, par_m);
       }

       /*!
        * @brief A wrapper to change the signiture arguments of cdf in each dimension d
        * from (x, d, par) to (x, d).
       */
       KOKKOS_INLINE_FUNCTION T cdf(T x, unsigned int d) const{
          return cdf_m(x, d, par_m);
       }
        
       /*!
        * @brief A wrapper to change the signiture arguments of estimate in each dimension d
        * from (x, d, par) to (x, d).
       */
       KOKKOS_INLINE_FUNCTION T estimate(T x, unsigned int d) const{
          return estimate_m(x, d, par_m);
       }

       /*!
        * @returns Objective function that is used in inverse transform sampling, i.e. obj = cdf(x)-u.
        * Here u is uniformly distribution on [0, 1] and x is the target sample.
       */
       KOKKOS_INLINE_FUNCTION T obj_func(T x, unsigned int d, T u) const{
            return cdf(x, d) - u;
       }
        
       /*!
        * @returns Derivative of the objective function that is used in inverse transform sampling, i.e. d(obj)/dx = pdf(x)
       */
       KOKKOS_INLINE_FUNCTION T der_obj_func(T x, unsigned int d) const{
            return pdf(x, d);
       }
       
       /*!
        * @returns Total pdf given uncorrelated pdf in each dimension.
        * i.e. total_pdf = pdf(x_1) * pdf(x_2) * ...  pdf(x_N).
       */
       KOKKOS_INLINE_FUNCTION T full_pdf(ippl::Vector<T, Dim> x) const{
          T total_pdf = 1.0;
          for(unsigned int d=0; d<Dim; d++){
             total_pdf *= pdf(x[d], d);
          }
          return total_pdf;
       }
    };
  }  // namespace random
}  // namespace ippl

#endif
