
namespace ippl {
    template <typename T>
    typename QuadrilateralElement<T>::mesh_element_vertex_point_vec_t
    QuadrilateralElement<T>::getLocalVertices() const {
        QuadrilateralElement::mesh_element_vertex_point_vec_t vertices = {
            {0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}};

        return vertices;
    }

    template <typename T>
    QuadrilateralElement<T>::diag_matrix_vec_t QuadrilateralElement<T>::getTransformationJacobian(
        const QuadrilateralElement<T>::mesh_element_vertex_point_vec_t& global_vertices) const {
        QuadrilateralElement::diag_matrix_vec_t jacobian;

        jacobian[0] = 1.0 / (global_vertices[1][0] - global_vertices[0][0]);
        jacobian[1] = 1.0 / (global_vertices[2][1] - global_vertices[0][1]);

        return jacobian;
    }

    template <typename T>
    QuadrilateralElement<T>::diag_matrix_vec_t
    QuadrilateralElement<T>::getInverseTransformationJacobian(
        const QuadrilateralElement<T>::mesh_element_vertex_point_vec_t& global_vertices) const {
        QuadrilateralElement::diag_matrix_vec_t inv_jacobian;

        inv_jacobian[0] = global_vertices[1][0] - global_vertices[0][0];
        inv_jacobian[1] = global_vertices[2][1] - global_vertices[0][1];

        return inv_jacobian;
    }

}  // namespace ippl