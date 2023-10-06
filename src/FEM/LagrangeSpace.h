// Class LagrangeSpace
//

#ifndef IPPL_LAGRANGESPACE_H
#define IPPL_LAGRANGESPACE_H

#include "FEM/FiniteElementSpace.h"

namespace ippl {

    template <typename T, unsigned Dim>
    class LagrangeSpace : public FiniteElementSpace<T, Dim> {
    public:
        // This is the number of vertices per element.
        // Since it is assumed that Mesh is a structured grid, the number of vertices per element
        // follows 2^Dim.
        static constexpr std::size_t NumVertices = 1 << Dim;

        template <unsigned Degree>
        static constexpr std::size_t DegreesOfFreedom = ;

        LagrangeSpace(const Mesh<T, Dim>& mesh, const Element<T, Dim>* ref_element,
                      const Quadrature<T>* quadrature, const unsigned& degree);

    private:
        /***/
        Vector<std::size_t, Dim> getElementDimIndices(const std::size_t& element_index) const;

        /***/
        Vector<std::size_t, NumVertices> getVerticesForElement(
            const std::size_t& element_index) const;

        /***/
        Vector<std::size_t, NumVertices> getVerticesForElement(
            const Vector<std::size_t, Dim>& element_indices) const;
    };

}  // namespace ippl

#include "FEM/LagrangeSpace.hpp"

#endif