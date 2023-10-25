
namespace ippl {

    template <typename T>
    EdgeElement<T>::vertex_vec_t EdgeElement<T>::getLocalVertices() const {
        EdgeElement::vertex_vec_t vertices;
        vertices[0] = {0.0};
        vertices[1] = {1.0};
        return vertices;
    }

    template <typename T>
    EdgeElement<T>::matrix_t EdgeElement<T>::getTransformationJacobian(
        const EdgeElement<T>::vertex_vec_t& global_vertices) const {
        EdgeElement::matrix_t jacobian;

        jacobian[0][0] = 1.0 / (global_vertices[1][0] - global_vertices[0][0]);

        return jacobian;
    }

    template <typename T>
    EdgeElement<T>::matrix_t EdgeElement<T>::getInverseTransformationJacobian(
        const EdgeElement<T>::vertex_vec_t& global_vertices) const {
        EdgeElement::matrix_t inv_jacobian;

        inv_jacobian[0][0] = global_vertices[1][0] - global_vertices[0][0];

        return inv_jacobian;
    }

    template <typename T>
    T EdgeElement<T>::getDeterminantOfTransformationJacobian(
        const EdgeElement<T>::vertex_vec_t& global_vertices) const {
        return 1.0 / (global_vertices[1][0] - global_vertices[0][0]);
    }

}  // namespace ippl