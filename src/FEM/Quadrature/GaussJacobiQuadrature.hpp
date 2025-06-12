
#include <cmath>

namespace ippl {
    template <typename T, unsigned NumNodes1D, typename ElementType>
    GaussJacobiQuadrature<T, NumNodes1D, ElementType>::GaussJacobiQuadrature(
        const ElementType& ref_element, const T& alpha, const T& beta,
        const size_t& max_newton_iterations, const size_t& min_newton_iterations)
        : Quadrature<T, NumNodes1D, ElementType>(ref_element)
        , alpha_m(alpha)
        , beta_m(beta)
        , max_newton_iterations_m(max_newton_iterations)
        , min_newton_iterations_m(min_newton_iterations) {
        assert(alpha > -1.0 && "alpha >= -1.0 is not satisfied");
        assert(beta > -1.0 && "beta >= -1.0 is not satisfied");
        assert(max_newton_iterations >= 1 && "max_newton_iterations >= 1 is not satisfied");
        assert(min_newton_iterations_m >= 1 && "min_newton_iterations_m >= 1 is not satisfied");
        assert(min_newton_iterations_m <= max_newton_iterations_m
               && "min_newton_iterations_m <= max_newton_iterations_m is not satisfied");

        this->degree_m = 2 * NumNodes1D - 1;

        this->a_m = -1.0;  // start of the domain
        this->b_m = 1.0;   // end of the domain

        this->integration_nodes_m = Vector<T, NumNodes1D>();
        this->weights_m           = Vector<T, NumNodes1D>();

        this->computeNodesAndWeights();
    }

    template <typename T, unsigned NumNodes1D, typename ElementType>
    typename GaussJacobiQuadrature<T, NumNodes1D, ElementType>::scalar_t
    GaussJacobiQuadrature<T, NumNodes1D, ElementType>::getChebyshevNodes(const size_t& i) const {
        return -Kokkos::cos((2.0 * static_cast<scalar_t>(i) + 1.0) * Kokkos::numbers::pi_v<scalar_t>
                            / (2.0 * NumNodes1D));
    }

    template <typename T, unsigned NumNodes1D, typename ElementType>
    typename GaussJacobiQuadrature<T, NumNodes1D, ElementType>::scalar_t
    GaussJacobiQuadrature<T, NumNodes1D, ElementType>::getLehrFEMInitialGuess(
        const size_t& i,
        const Vector<GaussJacobiQuadrature<T, NumNodes1D, ElementType>::scalar_t, NumNodes1D>&
            integration_nodes) const {
        const scalar_t alpha = this->alpha_m;
        const scalar_t beta  = this->beta_m;
        scalar_t r1;
        scalar_t r2;
        scalar_t r3;
        scalar_t z = (i > 0) ? integration_nodes[i - 1] : 0.0;

        if (i == 0) {
            // initial guess for the largest root
            const scalar_t an = alpha / NumNodes1D;
            const scalar_t bn = beta / NumNodes1D;
            r1 = (1.0 + alpha) * (2.78 / (4.0 + NumNodes1D * NumNodes1D) + 0.768 * an / NumNodes1D);
            r2 = 1.0 + 1.48 * an + 0.96 * bn + 0.452 * an * an + 0.83 * an * bn;
            z  = 1.0 - r1 / r2;
        } else if (i == 1) {
            // initial guess for the second largest root
            r1 = (4.1 + alpha) / ((1.0 + alpha) * (1.0 + 0.156 * alpha));
            r2 = 1.0 + 0.06 * (NumNodes1D - 8.0) * (1.0 + 0.12 * alpha) / NumNodes1D;
            r3 = 1.0 + 0.012 * beta * (1.0 + 0.25 * Kokkos::abs(alpha)) / NumNodes1D;
            z -= (1.0 - z) * r1 * r2 * r3;
        } else if (i == 2) {
            // initial guess for the third largest root
            r1 = (1.67 + 0.28 * alpha) / (1.0 + 0.37 * alpha);
            r2 = 1.0 + 0.22 * (NumNodes1D - 8.0) / NumNodes1D;
            r3 = 1.0 + 8.0 * beta / ((6.28 + beta) * NumNodes1D * NumNodes1D);
            z -= (integration_nodes[0] - z) * r1 * r2 * r3;
        } else if (i == NumNodes1D - 2) {
            // initial guess for the second smallest root
            r1 = (1.0 + 0.235 * beta) / (0.766 + 0.119 * beta);
            r2 = 1.0 / (1.0 + 0.639 * (NumNodes1D - 4.0) / (1.0 + 0.71 * (NumNodes1D - 4.0)));
            r3 = 1.0 / (1.0 + 20.0 * alpha / ((7.5 + alpha) * NumNodes1D * NumNodes1D));
            z += (z - integration_nodes[NumNodes1D - 4]) * r1 * r2 * r3;
        } else if (i == NumNodes1D - 1) {
            // initial guess for the smallest root
            r1 = (1.0 + 0.37 * beta) / (1.67 + 0.28 * beta);
            r2 = 1.0 / (1.0 + 0.22 * (NumNodes1D - 8.0) / NumNodes1D);
            r3 = 1.0 / (1.0 + 8.0 * alpha / ((6.28 + alpha) * NumNodes1D * NumNodes1D));
            z += (z - integration_nodes[NumNodes1D - 3]) * r1 * r2 * r3;
        } else {
            // initial guess for the other integration_nodes
            z = 3.0 * z - 3.0 * integration_nodes[i - 2] + integration_nodes[i - 3];
        }
        return z;
    }

    template <typename T, unsigned NumNodes1D, typename ElementType>
    void GaussJacobiQuadrature<T, NumNodes1D, ElementType>::computeNodesAndWeights() {
        // set the initial guess type
        const InitialGuessType& initial_guess_type = InitialGuessType::Chebyshev;

        /**
         * the following algorithm for computing the roots and weights for the Gauss-Jacobi
         * quadrature has been taken from LehrFEM++ (MIT License)
         * https://craffael.github.io/lehrfempp/gauss__quadrature_8cc_source.html
         */

        const scalar_t tolerance = 2e-16;

        Vector<scalar_t, NumNodes1D> integration_nodes;
        Vector<scalar_t, NumNodes1D> weights;

        const scalar_t alpha = this->alpha_m;
        const scalar_t beta  = this->beta_m;

        const scalar_t alfbet = alpha + beta;

        scalar_t a;
        scalar_t b;
        scalar_t c;
        scalar_t p1;
        scalar_t p2;
        scalar_t p3;
        scalar_t pp;
        scalar_t temp;
        scalar_t z;
        scalar_t z1;

        // Compute the root of the Jacobi polynomial

        for (size_t i = 0; i < NumNodes1D; ++i) {
            // initial guess depending on which root we are computing
            if (initial_guess_type == InitialGuessType::LehrFEM) {
                z = this->getLehrFEMInitialGuess(i, integration_nodes);
            } else if (initial_guess_type == InitialGuessType::Chebyshev) {
                z = -this->getChebyshevNodes(i);
            } else {
                throw IpplException("GaussJacobiQuadrature::computeNodesAndWeights",
                                    "Unknown initial guess type");
            }

            // std::cout << NumNodes1D - i - 1 << ", initial guess: " << z << " with "
            //           << initial_guess_type << std::endl;

            size_t its = 1;
            do {
                // refinement by Newton's method (from LehrFEM++)
                temp = 2.0 + alfbet;

                // Start the recurrence with P_0 and P1 to avoid a division by zero when
                // alpha * beta = 0 or -1
                p1 = (alpha - beta + temp * z) / 2.0;
                p2 = 1.0;
                for (size_t j = 2; j <= NumNodes1D; ++j) {
                    p3   = p2;
                    p2   = p1;
                    temp = 2 * j + alfbet;
                    a    = 2 * j * (j + alfbet) * (temp - 2.0);
                    b    = (temp - 1.0) * (alpha * alpha - beta * beta + temp * (temp - 2.0) * z);
                    c    = 2.0 * (j - 1 + alpha) * (j - 1 + beta) * temp;
                    p1   = (b * p2 - c * p3) / a;
                }
                pp = (NumNodes1D * (alpha - beta - temp * z) * p1
                      + 2.0 * (NumNodes1D + alpha) * (NumNodes1D + beta) * p2)
                     / (temp * (1.0 - z * z));
                // p1 is now the desired jacobian polynomial. We next compute pp, its
                // derivative, by a standard relation involving p2, the polynomial of one
                // lower order
                z1 = z;
                z  = z1 - p1 / pp;  // Newtons Formula

                // std::cout << "it = " << its << ", i = " << i << ", error: " << Kokkos::abs(z -
                // z1)
                //           << std::endl;
                if (its > this->min_newton_iterations_m && Kokkos::abs(z - z1) <= tolerance) {
                    break;
                }
                ++its;
            } while (its <= this->max_newton_iterations_m);

            if (its > this->max_newton_iterations_m) {
                //  TODO switch to inform
                std::cout << "Root " << NumNodes1D - i - 1
                          << " didn't converge. Tolerance may be too high for data type"
                          << std::endl;
            }

            // std::cout << "i = " << i << ", result after " << its << " iterations: " << z
            //           << std::endl;

            integration_nodes[i] = z;

            // Compute the weight of the Gauss-Jacobi quadrature
            weights[i] =
                Kokkos::exp(Kokkos::lgamma(alpha + NumNodes1D) + Kokkos::lgamma(beta + NumNodes1D)
                            - Kokkos::lgamma(NumNodes1D + 1.)
                            - Kokkos::lgamma(static_cast<double>(NumNodes1D) + alfbet + 1.0))
                * temp * Kokkos::pow(2.0, alfbet) / (pp * p2);

            // store the integration nodes and weights in the correct order
            this->integration_nodes_m[i] = static_cast<T>(-integration_nodes[i]);
            this->weights_m[i]           = static_cast<T>(weights[i]);
        }
    }

}  // namespace ippl
