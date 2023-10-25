// Class EdgeElement

#ifndef IPPL_EDGEELEMENT_H
#define IPPL_EDGEELEMENT_H

#include "FEM/Elements/Element.h"

namespace ippl {

    template <typename T>  // TODO maybe remove the fixed geometric dim at some point
    class EdgeElement : public Element1D<T, 2> {
    public:
        static constexpr unsigned NumVertices = 2;

        typedef typename Element1D<T, NumVertices>::point_t point_t;
        typedef typename Element1D<T, NumVertices>::vertex_vec_t vertex_vec_t;
        typedef typename Element1D<T, NumVertices>::matrix_t matrix_t;

        vertex_vec_t getLocalVertices() const override;

        matrix_t getTransformationJacobian(const vertex_vec_t& global_vertices) const override;

        matrix_t getInverseTransformationJacobian(
            const vertex_vec_t& global_vertices) const override;

        T getDeterminantOfTransformationJacobian(
            const vertex_vec_t& global_vertices) const override;
    };

}  // namespace ippl

#include "FEM/Elements/EdgeElement.hpp"

#endif