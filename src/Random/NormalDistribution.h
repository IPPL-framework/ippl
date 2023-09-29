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
      using view_type  = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;
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
      randn(view_type v_, GeneratorPool rand_pool_, T *mu_, T *sd_)
          : v(v_)
          , rand_pool(rand_pool_){
             for(unsigned int i=0; i<Dim; i++){
                mu[i] = mu_[i];
                sd[i] = sd_[i];
             }
           }

      randn(view_type v_, GeneratorPool rand_pool_)
          : v(v_)
          , rand_pool(rand_pool_) {
             for(unsigned int i=0; i<Dim; i++){
                mu[i] = 0.0;
                sd[i] = 0.0;
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
              v(i)[d] = mu[d] + sd[d]*rand_gen.normal(0.0, 1.0);
          }

          // Give the state back, which will allow another thread to acquire it
          rand_pool.free_state(rand_gen);
      }
    };

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
      const T pi = Kokkos::numbers::pi_v<T>;
      return (Kokkos::sqrt(pi / 2.0) * (2.0 * u - 1.0)) * stddev + mean;
    }

     /*!
     * @struct normal_cdf
     * @brief Functor to calculate the cumulative distribution function (CDF) for a normal distribution.
     *
     * This functor calculates the CDF for a normal distribution in a specific dimension 'd'.
     *
     * @tparam T Data type for the input value 'x'.
     * @tparam Dim Dimensionality of the distribution.
    */
    template <typename T, unsigned Dim>
    struct normal_cdf{
      KOKKOS_INLINE_FUNCTION double operator()(T x, unsigned int d, const T *params) const {
              T mean = params[d*Dim + 0];
              T stddev = params[d*Dim + 1];
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
     * @tparam Dim Dimensionality of the distribution.
    */
    template <typename T, unsigned Dim>
    struct normal_pdf{
      KOKKOS_INLINE_FUNCTION double operator()(T x, unsigned int d, T const *params) const {
              T mean = params[d*Dim + 0];
              T stddev = params[d*Dim + 1];
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
     * @tparam Dim Dimensionality of the distribution.
    */
    template <typename T, unsigned Dim>
    struct normal_estimate{
      KOKKOS_INLINE_FUNCTION double operator()(T u, unsigned int d,  T const *params) const {
              T mean = params[d*Dim + 0];
              T stddev = params[d*Dim + 1];
              return ippl::random::normal_estimate_func<T>(u, mean, stddev);
      }
    };

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
       * The constructor takes an array of parameters of normal distribution, i.e. mean and standard deviation.
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
