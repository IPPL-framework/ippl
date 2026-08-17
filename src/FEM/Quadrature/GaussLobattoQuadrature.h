// Class GaussLobattoQuadrature
//   Gauss-Lobatto-Legendre (GLL) quadrature on [-1, 1]. Node/weight generation delegates to ippl::nodes1d.

#ifndef IPPL_GAUSSLOBATTOQUADRATURE_H
#define IPPL_GAUSSLOBATTOQUADRATURE_H

#include "FEM/Quadrature/Quadrature.h"
#include "Nodes1D/GaussLobatto1D.h"

namespace ippl {

    /**
     * @brief Gauss-Lobatto-Legendre quadrature on a reference element.
     *
     * Endpoints +/-1 are fixed; n-2 interior nodes are roots of P_{n-1}'.
     * An n-point GLL rule is exact for polynomials of degree up to 2n-3.
     *
     * @tparam T floating point number type of the quadrature nodes and weights
     * @tparam NumNodes1D number of quadrature nodes for one dimension (>= 2)
     * @tparam ElementType element type for which the quadrature rule is defined
     */
    template <typename T, unsigned NumNodes1D, typename ElementType>
    class GaussLobattoQuadrature : public Quadrature<T, NumNodes1D, ElementType> {
    public:
        static_assert(NumNodes1D >= 2, "Gauss-Lobatto quadrature requires NumNodes1D >= 2");

        /**
         * @brief Construct a Gauss-Lobatto quadrature rule on [-1, 1].
         *
         * Node/weight generation uses the Nodes1D default backend
         * (RootFinderMethod::GolubWelsch). Callers that need a different root-finding backend
         * should use ippl::nodes1d::computeGaussLobatto directly.
         *
         * @param ref_element reference element to compute the quadrature nodes on
         */
        GaussLobattoQuadrature(const ElementType& ref_element)
            : Quadrature<T, NumNodes1D, ElementType>(ref_element) {
            this->degree_m = 2 * NumNodes1D - 3;

            this->a_m = -1.0;
            this->b_m = 1.0;

            this->integration_nodes_m = Vector<T, NumNodes1D>();
            this->weights_m           = Vector<T, NumNodes1D>();

            computeNodesAndWeights();
        }

        /** @brief Fill integration_nodes_m and weights_m via nodes1d::computeGaussLobatto. */
        void computeNodesAndWeights() override {
            nodes1d::computeGaussLobatto(this->integration_nodes_m, this->weights_m);
        }
    };

}  // namespace ippl

#endif
