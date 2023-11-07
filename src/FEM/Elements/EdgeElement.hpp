
namespace ippl {

    template <typename T>
    EdgeElement<T>::mesh_element_vertex_vec_t EdgeElement<T>::getLocalVertices() const {
        EdgeElement::mesh_element_vertex_vec_t vertices;
        vertices[0] = {0.0};
        vertices[1] = {1.0};
        return vertices;
    }

    template <typename T>
    EdgeElement<T>::diag_matrix_vec_t EdgeElement<T>::getTransformationJacobian(
        const EdgeElement<T>::mesh_element_vertex_vec_t& global_vertices) const {
        EdgeElement::diag_matrix_vec_t jacobian;

        jacobian[0] = 1.0 / (global_vertices[1][0] - global_vertices[0][0]);

        return jacobian;
    }

    template <typename T>
    EdgeElement<T>::diag_matrix_vec_t EdgeElement<T>::getInverseTransformationJacobian(
        const EdgeElement<T>::mesh_element_vertex_vec_t& global_vertices) const {
        EdgeElement::diag_matrix_vec_t inv_jacobian;

        inv_jacobian[0] = global_vertices[1][0] - global_vertices[0][0];

        return inv_jacobian;
    }

}  // namespace ippl