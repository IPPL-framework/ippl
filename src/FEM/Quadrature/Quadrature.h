// Class Quadrature
//  This is the base class for all quadrature rules.

#ifndef IPPL_QUADRATURE_H
#define IPPL_QUADRATURE_H

#include <cassert>
#include <cmath>

#include "Types/Vector.h"

#include "Utility/IpplException.h"

#include "FEM/Elements/Element.h"
#include "Nodes1D/AffineMap.h"

// own power function since Kokkos::pow is not constexpr
template <typename T>
constexpr T power(T base, unsigned exponent) {
    return exponent == 0 ? 1 : base * power(base, exponent - 1);
}

inline constexpr unsigned getNumElementNodes(unsigned NumNodes1D, unsigned Dim) {
    return static_cast<unsigned>(power(static_cast<int>(NumNodes1D), static_cast<int>(Dim)));
}

namespace ippl {

    /**
     * @brief This is the base class for all quadrature rules.
     *
     * @tparam T floating point number type of the quadrature nodes and weights
     * @tparam NumNodes1D number of quadrature nodes for one dimension
     * @tparam ElementType element type for which the quadrature rule is defined
     */
    template <typename T, unsigned NumNodes1D, typename ElementType>
    class Quadrature {
    public:
        // the number of quadrature nodes for one dimension
        static constexpr unsigned numNodes1D = NumNodes1D;

        // the dimension of the reference element to compute the quadrature nodes for
        static constexpr unsigned dim = ElementType::dim;

        // the number of quadrature nodes for the reference element
        static constexpr unsigned numElementNodes =
            getNumElementNodes(NumNodes1D, ElementType::dim);

        static_assert(NumNodes1D >= 1, "Quadrature requires NumNodes1D >= 1");
        static_assert(ElementType::dim >= 1, "Quadrature requires ElementType::dim >= 1");

        /**
         * @brief Construct a new Quadrature object
         *
         * @param ref_element reference element to compute the quadrature nodes on
         * @param degree degree of exactness of the quadrature rule
         * @param a source interval start the nodes/weights are computed on
         * @param b source interval end the nodes/weights are computed on
         */
        Quadrature(const ElementType& ref_element, unsigned degree, T a, T b);

        /** @brief Virtual destructor so derived rules can be destroyed through a Quadrature*. */
        virtual ~Quadrature() = default;

        /**
         * @brief Returns the order of the quadrature rule. (order = degree + 1)
         *
         * @return unsigned - order
         */
        size_t getOrder() const;

        /**
         * @brief Returns the degree of exactness of the quadrature rule.
         *
         * @return unsigned - degree
         */
        size_t getDegree() const;

        /**
         * @brief Get the quadrature weights for the reference element.
         *
         * @return Vector<T, numElementNodes>
         */
        Vector<T, numElementNodes> getWeightsForRefElement() const;

        /**
         * @brief Get the integration (quadrature) nodes for the reference element.
         *
         * @return Vector<Vector<T, Dim>, numElementNodes>
         */
        Vector<Vector<T, dim>, numElementNodes> getIntegrationNodesForRefElement() const;

        /**
         * @brief Get the quadrature nodes for one dimension.
         * (With respect to the given domain [a, b])
         *
         * @param a local domain start
         * @param b local domain end
         *
         * @return Vector<T, NumNodes1D>
         */
        Vector<T, NumNodes1D> getIntegrationNodes1D(const T& a, const T& b) const;

        /**
         * @brief Get the quadrature weights for one dimension.
         * (With respect to the given domain [a, b])
         *
         * @param a local domain start
         * @param b local domain end
         *
         * @return Vector<T, NumNodes1D>
         */
        Vector<T, NumNodes1D> getWeights1D(const T& a, const T& b) const;

        /**
         * @brief Pure virtual function that computes the local quadrature nodes and weights.
         * (Needs to be implemented in derived classes)
         */
        virtual void computeNodesAndWeights() = 0;

    protected:
        unsigned degree_m;
        const ElementType& ref_element_m;
        Vector<T, NumNodes1D> integration_nodes_m;
        Vector<T, NumNodes1D> weights_m;

        // local domain start
        T a_m;
        // local domain end
        T b_m;
    };

}  // namespace ippl

#include "FEM/Quadrature/Quadrature.hpp"

#endif
