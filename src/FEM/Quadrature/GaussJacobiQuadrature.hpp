
#include <cmath>

namespace ippl {
    template <typename T, unsigned NumNodes1D, typename ElementType>
    GaussJacobiQuadrature<T, NumNodes1D, ElementType>::GaussJacobiQuadrature(
        const ElementType& ref_element, const T& alpha, const T& beta,
        const std::size_t& max_newton_iterations)
        : Quadrature<T, NumNodes1D, ElementType>(ref_element)
        , alpha_m(alpha)
        , beta_m(beta) {
        assert(alpha > -1.0 && "alpha > -1.0 is not satisfied");
        assert(beta > -1.0 && "beta > -1.0 is not satisfied");
        assert(max_newton_iterations >= 1 && "max_newton_iterations >= 1 is not satisfied");

        this->degree_m = 2 * NumNodes1D - 1;

        this->a_m = -1.0;  // start of the domain
        this->b_m = 1.0;   // end of the domain

        this->integration_nodes_m = Vector<T, NumNodes1D>();
        this->weights_m           = Vector<T, NumNodes1D>();

        /**
         * the following algorithm for computing the roots and weights for the Gauss-Jacobi
         * quadrature has been taken from LehrFEM++ (MIT License)
         * https://craffael.github.io/lehrfempp/gauss__quadrature_8cc_source.html
         */

        T alfbet;
        T an;
        T bn;
        T r1;
        T r2;
        T r3;
        T a;
        T b;
        T c;
        T p1;
        T p2;
        T p3;
        T pp;
        T temp;
        T z = 0.0;  // initialize to prevent "may be uninitialized warning, which it can't be"
        T z1;

        // Compute the root of the Jacobi polynomial

        for (std::size_t i = 0; i < NumNodes1D; ++i) {
            // make an initial guess for the roots
            if (i == 0) {
                // initial guess for the largest root
                an = alpha / NumNodes1D;
                bn = beta / NumNodes1D;
                r1 = (1.0 + alpha)
                     * (2.78 / (4.0 + NumNodes1D * NumNodes1D) + 0.768 * an / NumNodes1D);
                r2 = 1.0 + 1.48 * an + 0.96 * bn + 0.452 * an * an + 0.83 * an * bn;
                z  = 1.0 - r1 / r2;
            } else if (i == 1) {
                // initial guess for the second largest root
                r1 = (4.1 + alpha) / ((1.0 + alpha) * (1.0 + 0.156 * alpha));
                r2 = 1.0 + 0.06 * (NumNodes1D - 8.0) * (1.0 + 0.12 * alpha) / NumNodes1D;
                r3 = 1.0 + 0.012 * beta * (1.0 + 0.25 * std::abs(alpha)) / NumNodes1D;
                z -= (1.0 - z) * r1 * r2 * r3;
            } else if (i == 2) {
                // initial guess for the third largest root
                r1 = (1.67 + 0.28 * alpha) / (1.0 + 0.37 * alpha);
                r2 = 1.0 + 0.22 * (NumNodes1D - 8.0) / NumNodes1D;
                r3 = 1.0 + 8.0 * beta / ((6.28 + beta) * NumNodes1D * NumNodes1D);
                z -= (this->integration_nodes_m[0] - z) * r1 * r2 * r3;
            } else if (i == NumNodes1D - 2) {
                // initial guess for the second smallest root
                r1 = (1.0 + 0.235 * beta) / (0.766 + 0.119 * beta);
                r2 = 1.0 / (1.0 + 0.639 * (NumNodes1D - 4.0) / (1.0 + 0.71 * (NumNodes1D - 4.0)));
                r3 = 1.0 / (1.0 + 20.0 * alpha / ((7.5 + alpha) * NumNodes1D * NumNodes1D));
                z += (z - this->integration_nodes_m[NumNodes1D - 4]) * r1 * r2 * r3;
            } else if (i == NumNodes1D - 1) {
                // initial guess for the smallest root
                r1 = (1.0 + 0.37 * beta) / (1.67 + 0.28 * beta);
                r2 = 1.0 / (1.0 + 0.22 * (NumNodes1D - 8.0) / NumNodes1D);
                r3 = 1.0 / (1.0 + 8.0 * alpha / ((6.28 + alpha) * NumNodes1D * NumNodes1D));
                z += (z - this->integration_nodes_m[NumNodes1D - 3]) * r1 * r2 * r3;
            } else {
                // initial guess for the other integration_nodes_m
                z = 3.0 * this->integration_nodes_m[i - 1] - 3.0 * this->integration_nodes_m[i - 2]
                    + this->integration_nodes_m[i - 3];
            }

            std::cout << NumNodes1D - i - 1 << ", initial guess: " << z << std::endl;

            alfbet          = alpha + beta;
            std::size_t its = 1;
            do {
                // refinement by Newton's method
                temp = 2.0 + alfbet;

                // Start the recurrence with P_0 and P1 to avoid a division by zero when
                // alpha * beta = 0 or -1
                p1 = (alpha - beta + temp * z) / 2.0;
                p2 = 1.0;
                for (std::size_t j = 2; j <= NumNodes1D; ++j) {
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
                // p1 is now the desired jacobia polynomial. We next compute pp, its
                // derivative, by a standard relation involving p2, the polynomial of one
                // lower order
                z1 = z;
                z  = z1 - p1 / pp;  // Newtons Formula

                if (Kokkos::abs(z - z1) <= 1e-17) {
                    // std::cout << "i = " << i << ", error: " << Kokkos::abs(z - z1)
                    //           << ", aborting..." << std::endl;
                    // break;
                }
                ++its;
            } while (its <= max_newton_iterations);

            if (its > max_newton_iterations) {
                // inform "too many iterations."
                // TODO switch to inform
                std::cout << "i = " << i << ", too many iterations" << std::endl;
            }

            std::cout << "i = " << i << ", result after " << its << " iterations: " << z
                      << std::endl;

            this->integration_nodes_m[i] = z;

            // Compute the weight of the Gauss-Jacobi quadrature
            this->weights_m(i) =
                Kokkos::exp(Kokkos::lgamma(alpha + NumNodes1D) + Kokkos::lgamma(beta + NumNodes1D)
                            - Kokkos::lgamma(NumNodes1D + 1.)
                            - Kokkos::lgamma(static_cast<double>(NumNodes1D) + alfbet + 1.0))
                * temp * Kokkos::pow(2.0, alfbet) / (pp * p2);
        }

        this->integration_nodes_m *= -1.0;  // flip it
    }
}  // namespace ippl