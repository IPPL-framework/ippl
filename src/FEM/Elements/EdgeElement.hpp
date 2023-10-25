
namespace ippl {

    template <typename T>
    EdgeElement<T>::local_vertex_vec_t EdgeElement<T>::getLocalVertices() const {
        EdgeElement::local_vertex_vec_t vertices;
        vertices[0] = {0.0};
        vertices[1] = {1.0};
        return vertices;
    }

    // global to local
    // template <typename T, unsigned GeometricDim>
    // EdgeElement<T, GeometricDim>::jacobian_t
    // EdgeElement<T, GeometricDim>::getTransformationJacobian(
    //     const EdgeElement<T, GeometricDim>::global_vertex_vec_t& global_vertices) const {
    //     EdgeElement::jacobian_t jacobian;

    //     for (unsigned d = 0; d < GeometricDim; ++d) {
    //         jacobian[0][d] = 1.0 / (global_vertices[1][d] - global_vertices[0][d]);
    //     }

    //     return jacobian;
    // }

    template <typename T>
    EdgeElement<T>::jacobian_t EdgeElement<T>::getTransformationJacobian(
        const EdgeElement<T>::global_vertex_vec_t& global_vertices) const {
        EdgeElement::jacobian_t jacobian;

        jacobian[0][0] = 1.0 / (global_vertices[1][0] - global_vertices[0][0]);

        return jacobian;
    }

    // local to global
    // template <typename T, unsigned GeometricDim>
    // EdgeElement<T, GeometricDim>::inverse_jacobian_t
    // EdgeElement<T, GeometricDim>::getInverseTransformationJacobian(
    //     const EdgeElement<T, GeometricDim>::global_vertex_vec_t& global_vertices) const {
    //     EdgeElement::inverse_jacobian_t
    //         inv_jacobian;  // ippl::Vector<ippl::Vector<T, 1>, GeometricDim>

    //     for (unsigned d = 0; d < GeometricDim; ++d) {
    //         inv_jacobian[d][0] = global_vertices[1][d] - global_vertices[0][d];
    //     }

    //     return inv_jacobian;
    // }

    template <typename T>
    EdgeElement<T>::inverse_jacobian_t EdgeElement<T>::getInverseTransformationJacobian(
        const EdgeElement<T>::global_vertex_vec_t& global_vertices) const {
        EdgeElement::inverse_jacobian_t inv_jacobian;

        inv_jacobian[0][0] = global_vertices[1][0] - global_vertices[0][0];

        return inv_jacobian;
    }

    template <typename T>
    T EdgeElement<T>::getDeterminantOfTransformationJacobian(
        const EdgeElement<T>::global_vertex_vec_t& global_vertices) const {
        return 1.0 / (global_vertices[1][0] - global_vertices[0][0]);
    }

}  // namespace ippl