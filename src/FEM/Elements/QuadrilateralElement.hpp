
namespace ippl {
    template <typename T>
    typename QuadrilateralElement<T>::local_vertex_vec_t QuadrilateralElement<T>::getLocalVertices()
        const {
        QuadrilateralElement::local_vertex_vec_t vertices;
        vertices = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};
        return vertices;
    }

    template <typename T>
    QuadrilateralElement<T>::jacobian_t QuadrilateralElement<T>::getTransformationJacobian(
        const QuadrilateralElement<T>::global_vertex_vec_t& global_vertices) const {
        QuadrilateralElement::jacobian_t jacobian;

        for (unsigned d = 0; d < GeometricDim; ++d) {
            jacobian[d][d] = 1.0 / (global_vertices[d + 1][d] - global_vertices[0][d]);
        }

        return jacobian;
    }

    template <typename T>
    QuadrilateralElement<T>::inverse_jacobian_t
    QuadrilateralElement<T>::getInverseTransformationJacobian(
        const QuadrilateralElement<T>::global_vertex_vec_t& global_vertices) const {
        QuadrilateralElement::inverse_jacobian_t inv_jacobian;

        for (unsigned d = 0; d < GeometricDim; ++d) {
            inv_jacobian[d][d] = global_vertices[d + 1][d] - global_vertices[0][d];
        }

        return inv_jacobian;
    }

    template <typename T>
    T QuadrilateralElement<T>::getDeterminantOfTransformationJacobian(
        const QuadrilateralElement<T>::global_vertex_vec_t& global_vertices) const {
        return 1.0
               / ((global_vertices[1][0] - global_vertices[0][0])
                  * (global_vertices[2][1] - global_vertices[0][1]));
    }

}  // namespace ippl