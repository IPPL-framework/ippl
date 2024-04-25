// Class Distribution
//   This class can be used for creating a distribution object
//   with custom pdf, cdf, and estimate function that is used for
//   the sampling method.
//

#ifndef IPPL_DISTRIBUTION_H
#define IPPL_DISTRIBUTION_H

#include "Types/ViewTypes.h"

#include "Random/Utility.h"

namespace ippl {
    namespace random {

        /*!
         * @file Distribution.h
         * @class Distribution
         */

        /*!
         * @ingroup Distribution
         * @brief The class that represents a distribution.
         * @tparam T Datatype.
         * @tparam Dim Dimensionality of sample space.
         * @tparam DimP Dimensionality of the parameter array.
         * @tparam PDF Struct type for the PDF (Probability Density Function).
         * @tparam CDF Struct type for the CDF (Cumulative Distribution Function).
         * @tparam ESTIMATE Struct type for the ESTIMATE function.
         */
        template <typename T, unsigned Dim, unsigned DimP, typename DistributionFunctions>
        class Distribution {
        public:
            /*!
             * @param par_m An array of distribution parameters.
             * @param pdf_m PDF of the distribution class as a member functor.
             * @param cdf_m CDF of the distribution class as a member functor.
             * @param estimate_m Estimate of the initial guess for the sampling method as a member
             * functor.
             */
            T par_m[DimP];
            typename DistributionFunctions::PDF pdf_m;
            typename DistributionFunctions::CDF cdf_m;
            typename DistributionFunctions::Estimate estimate_m;
            /*!
             * @ingroup Distribution
             * @brief Constructor for the Distribution class.
             * @param par_ Pointer to the parameter array.
             */
            KOKKOS_INLINE_FUNCTION Distribution(const T* par_p) {
                for (unsigned int i = 0; i < DimP; i++) {
                    par_m[i] = par_p[i];
                }
            }

            /*!
             * @ingroup Distribution
             * @brief Destructor for the Distribution class.
             */
            KOKKOS_INLINE_FUNCTION ~Distribution() {}

            /*!
             * @brief A wrapper to change the signature arguments of pdf in each dimension d
             * from (x, d, par) to (x, d).
             */
            KOKKOS_INLINE_FUNCTION T getPdf(T x, unsigned int d) const {
                return pdf_m(x, d, par_m);
            }

            /*!
             * @brief A wrapper to change the signature arguments of cdf in each dimension d
             * from (x, d, par) to (x, d).
             */
            KOKKOS_INLINE_FUNCTION T getCdf(T x, unsigned int d) const {
                return cdf_m(x, d, par_m);
            }

            /*!
             * @brief A wrapper to change the signature arguments of estimate in each dimension d
             * from (x, d, par) to (x, d).
             */
            KOKKOS_INLINE_FUNCTION T getEstimate(T x, unsigned int d) const {
                return estimate_m(x, d, par_m);
            }

            /*!
             * @returns Objective function that is used in inverse transform sampling, i.e. obj =
             * cdf(x)-u. Here u is uniformly distributed on [0, 1] and x is the sample of target
             * distribution.
             */
            KOKKOS_INLINE_FUNCTION T getObjFunc(T x, unsigned int d, T u) const {
                return getCdf(x, d) - u;
            }

            /*!
             * @returns Derivative of the objective function that is used in inverse transform
             * sampling, i.e. d(obj)/dx = pdf(x)
             */
            KOKKOS_INLINE_FUNCTION T getDerObjFunc(T x, unsigned int d) const {
                return getPdf(x, d);
            }

            /*!
             * @returns Total pdf given uncorrelated pdf in each dimension.
             * i.e. total_pdf = pdf(x_1) * pdf(x_2) * ...  pdf(x_N).
             */
            KOKKOS_INLINE_FUNCTION T getFullPdf(ippl::Vector<T, Dim> x) const {
                T totalPdf = 1.0;
                for (unsigned int d = 0; d < Dim; d++) {
                    totalPdf *= getPdf(x[d], d);
                }
                return totalPdf;
            }
        };
    }  // namespace random
}  // namespace ippl

#endif
