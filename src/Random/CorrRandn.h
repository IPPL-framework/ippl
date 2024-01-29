// Struct CorrRandn
//   This struct can be used for sampling correlated normal distribution function
//   on unbounded domain.
//
#ifndef IPPL_CORR_RANDN_H
#define IPPL_CORR_RANDN_H

#include "Random/Utility.h"
#include "Random/Distribution.h"

namespace ippl {
  namespace random {

     /*!
     * @struct CorrRandn
     * @brief Functor to generate random numbers from a normal distribution.
     *
     * This functor can be used to generates random numbers from a multivariate normal distribution
     * given mean and covariance matrix
     *
     * @tparam T Data type of the random numbers.
     * @tparam GeneratorPool Type of the random number generator pool.
     * @tparam Dim Dimension of the random numbers.
    */
    template <typename T, unsigned Dim>
    struct CorrRandn {
      using view_type  = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;
      using GeneratorPool = typename Kokkos::Random_XorShift64_Pool<>;

      // Output View for the random numbers
      view_type v;

      // The GeneratorPool
      GeneratorPool rand_pool;

      using Vector_t = ippl::Vector<T, Dim>;
      using Matrix_t = ippl::Vector< ippl::Vector<T, Dim>, Dim>;

      Vector_t mu;
      Matrix_t cov;
      Matrix_t L;
      /*!
       * @brief Constructor for the CorrRandn functor.
       *
       * @param v_ Output view for the random numbers.
       * @param rand_pool_ The random number generator pool.
       * @param mu The array of means in each dimension
       * @param cov The covariance matrix
      */
      KOKKOS_INLINE_FUNCTION CorrRandn(view_type v_, GeneratorPool rand_pool_, Vector_t &mu_, Matrix_t &cov_)
          : v(v_)
          , rand_pool(rand_pool_)
          , mu(mu_)
          , cov(cov_){
            // Perform Cholesky Factorization
            T sum = 0.0;
            for (unsigned int i = 0; i < Dim; i++) {
               for (unsigned int j = 0; j <= i; j++) {
                  sum = 0.0;
                  if (j == i) {
                     for (unsigned int k = 0; k < j; k++) {
                        sum += L[j][k] * L[j][k];
                     }
                     L[j][j] = Kokkos::sqrt(cov[j][j] - sum);
                  } else {
                     for (unsigned int k = 0; k < j; k++) {
                        sum += L[i][k] * L[j][k];
                     }
                     L[i][j] = (cov[i][j] - sum) / L[j][j];
                  }
               }
            }
          }

      /*!
      * @brief Get function for mean
      *
      */
      KOKKOS_INLINE_FUNCTION const Vector_t& getMu() const {
          return mu;
      }

      /*!
      * @brief Get function for the covariance matrix
      *
      */
      KOKKOS_INLINE_FUNCTION const Matrix_t& getCov() const {
          return cov;
      }

      /*!
      * @brief Get function for L
      *
      */
      KOKKOS_INLINE_FUNCTION const Matrix_t& getL() const {
          return L;
      }

      /*!
      * @brief Set function for mean
      *
      */
      KOKKOS_INLINE_FUNCTION void SetMu(Vector_t &mu_) const {
          mu(mu_);
      }

      /*!
      * @brief Set function for the covariance matrix
      *
      */
      KOKKOS_INLINE_FUNCTION void SetCov(Matrix_t &cov_){
          cov(cov_);
      }

      /*!
      * @brief Set function for L
      *
      */
      KOKKOS_INLINE_FUNCTION void SetL(Matrix_t &L_){
          L(L_);
      }


      /*!
       * @brief Operator to generate random numbers.
       *
       * @param i Index for the random number.
      */
      KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
          // Get a random number state from the pool for the active thread
          typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

          Vector_t z = 0;
          for (unsigned d = 0; d < Dim; ++d) {
              z[d] = rand_gen.normal(0.0, 1.0);
              v(i)[d] = 0.;
          }

          for (unsigned int d0 = 0; d0 < Dim; d0++) {
            for (unsigned int d1 = 0; d1 <= d0; d1++) {
               v(i)[d0] += L[d0][d1] * z[d1];
            }
          }

          for (unsigned int d = 0; d < Dim; d++) {
             v(i)[d] += mu[d];
          }

          // Give the state back, which will allow another thread to acquire it
          rand_pool.free_state(rand_gen);
      }

    };
  }  // namespace random
}  // namespace ippl

#endif
