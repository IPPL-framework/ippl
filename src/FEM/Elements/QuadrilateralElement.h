// Class QuadriateralElement
//   The QuadrilateralElement class. This is a class representing a quadrilateral element
//   for finite element methods.

#ifndef IPPL_QUADRILATERALELEMENT_H
#define IPPL_QUADRILATERALELEMENT_H

#include "FEM/Elements/Element.h"

namespace ippl {

    template <typename T>
    class QuadrilateralElement : public Element2D<T, 4> {
    public:
        static constexpr unsigned NumVertices = 4;

        typedef typename Element2D<T, NumVertices>::point_t point_t;
        typedef typename Element2D<T, NumVertices>::vertex_vec_t vertex_vec_t;
        typedef typename Element2D<T, NumVertices>::diag_matrix_vec_t diag_matrix_vec_t;

        vertex_vec_t getLocalVertices() const override;

        diag_matrix_vec_t getTransformationJacobian(
            const vertex_vec_t& global_vertices) const override;

        diag_matrix_vec_t getInverseTransformationJacobian(
            const vertex_vec_t& global_vertices) const override;

        T getDeterminantOfTransformationJacobian(
            const vertex_vec_t& global_vertices) const override;
    };
}  // namespace ippl

#include "FEM/Elements/QuadrilateralElement.hpp"

#endif