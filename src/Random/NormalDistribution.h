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
     * @brief Calculate the cumulative distribution function (CDF) for a normal distribution.
     *
     * @param x Input value.
     * @param mean Mean of the distribution.
     * @param stddev Standard deviation of the distribution.
     * @return The CDF value.
    */
    template<typename T>
    KOKKOS_FUNCTION T normal_cdf_func(T x, T mean, T stddev) {
      return 0.5 * (1 + Kokkos::erf((x - mean) / (stddev * Kokkos::sqrt(2.0))));
    }

    /*!
     * @brief Calculate the probability density function (PDF) for a normal distribution.
     *
     * @param x Input value.
     * @param mean Mean of the distribution.
     * @param stddev Standard deviation of the distribution.
     * @return The PDF value.
    */
    template<typename T>
    KOKKOS_FUNCTION T normal_pdf_func(T x, T mean, T stddev) {
      const T pi = Kokkos::numbers::pi_v<T>;
      return (1.0 / (stddev * Kokkos::sqrt(2 * pi))) * Kokkos::exp(-(x - mean) * (x - mean) / (2 * stddev * stddev));
    }

    /*!
     * @brief An estimator for the initial guess that is used in Newton-Raphson method of Inverste Transfrom Sampling
     *
     * @param x Input value.
     * @param mean Mean of the distribution.
     * @param stddev Standard deviation of the distribution.
     * @return The estimate value.
    */
    template<typename T>
    KOKKOS_FUNCTION T normal_estimate_func(T u, T mean, T stddev) {
      return mean + 0.*u*stddev;
    }

     /*!
     * @struct normal_cdf
     * @brief Functor to calculate the cumulative distribution function (CDF) for a normal distribution.
     *
     * This functor calculates the CDF for a normal distribution in a specific dimension 'd'.
     *
     * @tparam T Data type for the input value 'x'.
    */
    template <typename T>
    struct normal_cdf{
      KOKKOS_INLINE_FUNCTION double operator()(T x, unsigned int d, const T *params_p) const {
              T mean = params_p[2*d + 0];
              T stddev = params_p[2*d + 1];
              return ippl::random::normal_cdf_func<T>(x, mean, stddev);
      }
    };

    /*!
     * @struct normal_pdf
     * @brief Functor to calculate the probability density function (PDF) for a normal distribution.
     *
     * This functor calculates the PDF for a normal distribution in a specific dimension 'd'.
     *
     * @tparam T Data type for the input value 'x'.
    */
    template <typename T>
    struct normal_pdf{
      KOKKOS_INLINE_FUNCTION double operator()(T x, unsigned int d, T const *params_p) const {
              T mean = params_p[2*d + 0];
              T stddev = params_p[2*d + 1];
              return ippl::random::normal_pdf_func<T>(x, mean, stddev);
      }
    };

     /*!
     * @struct normal_estimate
     * @brief Functor to estimate the initial guess for sampling normal distribution.
     *
     * This functor estimates the value for a normal distribution in a specific dimension 'd'.
     *
     * @tparam T Data type for the input value 'u'.
    */
    template <typename T>
    struct normal_estimate{
      KOKKOS_INLINE_FUNCTION double operator()(T u, unsigned int d,  T const *params_p) const {
              T mean = params_p[2*d + 0];
              T stddev = params_p[2*d + 1];
              return ippl::random::normal_estimate_func<T>(u, mean, stddev);
      }
    };


    template <typename T>
    struct NormalDistributionFunctions {
      // Functor to calculate the probability density function (PDF) for a normal distribution.
      struct PDF {
        KOKKOS_INLINE_FUNCTION double operator()(T x, unsigned int d, const T *params_p) const {
            T mean = params_p[2 * d + 0];
            T stddev = params_p[2 * d + 1];
            return ippl::random::normal_pdf_func<T>(x, mean, stddev);
        }
      };

      // Functor to calculate the cumulative distribution function (CDF) for a normal distribution.
      struct CDF {
        KOKKOS_INLINE_FUNCTION double operator()(T x, unsigned int d, const T *params_p) const {
            T mean = params_p[2 * d + 0];
            T stddev = params_p[2 * d + 1];
            return ippl::random::normal_cdf_func<T>(x, mean, stddev);
        }
      };

      // Functor to estimate the initial guess for sampling a normal distribution.
      struct Estimate {
        KOKKOS_INLINE_FUNCTION double operator()(T u, unsigned int d, T const *params_p) const {
            T mean = params_p[2 * d + 0];
            T stddev = params_p[2 * d + 1];
            return ippl::random::normal_estimate_func<T>(u, mean, stddev);
        }
     };
    };


    /*!
       * @file NormalDistribution.h
       * @class NormalDistribution
    */
    template<typename T, unsigned Dim>
    class NormalDistribution : public ippl::random::Distribution<T, Dim, 2*Dim, NormalDistributionFunctions<T>>{
    public:
      /*!
       * @brief Constructor for the Normal Distribution class.
       * The constructor takes an array of parameters of normal distribution, i.e. mean and standard deviation.
      */
      KOKKOS_INLINE_FUNCTION NormalDistribution(const T *par_p)
                              : ippl::random::Distribution<T, Dim, 2*Dim, NormalDistributionFunctions<T>>(par_p) {}
    };

  }  // namespace random
}  // namespace ippl

#endif
