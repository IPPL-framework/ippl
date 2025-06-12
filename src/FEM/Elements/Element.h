//

#ifndef IPPL_ELEMENT_H
#define IPPL_ELEMENT_H

#include "Types/Vector.h"

namespace ippl {

    /**
     * @brief Base class for all elements.
     * @tparam T The type of the coordinates of the vertices of the element.
     * @tparam Dim The dimension of the element.
     * @tparam NumVertices The number of vertices of the element.
     */
    template <typename T, unsigned Dim, unsigned NumVertices>
    class Element {
    public:
        static constexpr unsigned dim         = Dim;
        static constexpr unsigned numVertices = NumVertices;

        // A point in the local or global coordinate system
        typedef Vector<T, Dim> point_t;

        // A list of all vertices
        typedef Vector<point_t, NumVertices> vertex_points_t;

        // Cannot define common functions to all Element child classes
        // due to problem with accessing the Element functions through
        // the Finite Element Space class from device code 
        // (defaults to base class virtual function which causes errors)
        // 
        // The common functions would be globalToLocal, localToGlobal,
        // getDeterminantOfTransformationJacobian, getInverseTransposeTransformationJacobian,
        // and isPointInRefElement.
        //
        // Virtual functions would be getLocalVertices, getTransformationJacobian, 
        // and getInverseTransformationJacobian.
    };

    /**
     * @brief Base class for all 1D elements.
     * @tparam T The type of the coordinates of the vertices of the element.
     * @tparam NumVertices The number of vertices of the element.
     */
    template <typename T, unsigned NumVertices>
    using Element1D = Element<T, 1, NumVertices>;

    /**
     * @brief Base class for all 2D elements.
     * @tparam T The type of the coordinates of the vertices of the element.
     * @tparam NumVertices The number of vertices of the element.
     */
    template <typename T, unsigned NumVertices>
    using Element2D = Element<T, 2, NumVertices>;

    /**
     * @brief Base class for all 3D elements.
     * @tparam T The type of the coordinates of the vertices of the element.
     * @tparam NumVertices The number of vertices of the element.
     */
    template <typename T, unsigned NumVertices>
    using Element3D = Element<T, 3, NumVertices>;

}  // namespace ippl

#endif
