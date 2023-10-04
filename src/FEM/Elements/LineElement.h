// Class LineElement
//   The LineElement class. This is a class representing a line element
//   for finite element methods.

#ifndef IPPL_LINEELEMENT_H
#define IPPL_LINEELEMENT_H

#include "FEM/Elements/Element.h"

namespace ippl {

    template <typename T>
    class LineElement : public Element<T, 1, 2> {
    public:
        static constexpr unsigned Dim         = 1;
        static constexpr unsigned NumVertices = 2;

        typedef typename Element<T, Dim, NumVertices>::set_of_vertices_type set_of_vertices_type;
        typedef typename Element<T, Dim, NumVertices>::jacobian_type jacobian_type;

        set_of_vertices_type getLocalVertices() const override;

        jacobian_type getTransformationJacobian(
            const set_of_vertices_type& global_vertices) const override;

        set_of_vertices_type getGlobalNodes(
            const jacobian_type& transformation_jacobian) const override;

    private:
        LineElement() {}
    };

}  // namespace ippl

#include "FEM/Elements/LineElement.hpp"

#endif