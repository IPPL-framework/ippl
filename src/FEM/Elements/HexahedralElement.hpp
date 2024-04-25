namespace ippl {
    template <typename T>
    typename HexahedralElement<T>::mesh_element_vertex_point_vec_t
    HexahedralElement<T>::getLocalVertices() const {
        HexahedralElement::mesh_element_vertex_point_vec_t vertices = {
            {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 0.0}, {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0}, {1.0, 0.0, 1.0}, {1.0, 1.0, 1.0}, {0.0, 1.0, 1.0}};

        return vertices;
    }

    template <typename T>
    typename HexahedralElement<T>::diag_matrix_vec_t
    HexahedralElement<T>::getTransformationJacobian(
        const HexahedralElement<T>::mesh_element_vertex_point_vec_t& global_vertices) const {
        HexahedralElement::diag_matrix_vec_t jacobian;

        jacobian[0] = (global_vertices[1][0] - global_vertices[0][0]);
        jacobian[1] = (global_vertices[2][1] - global_vertices[0][1]);
        jacobian[2] = (global_vertices[4][2] - global_vertices[0][2]);

        return jacobian;
    }

    template <typename T>
    typename HexahedralElement<T>::diag_matrix_vec_t
    HexahedralElement<T>::getInverseTransformationJacobian(
        const HexahedralElement<T>::mesh_element_vertex_point_vec_t& global_vertices) const {
        HexahedralElement::diag_matrix_vec_t inv_jacobian;

        inv_jacobian[0] = 1.0 / (global_vertices[1][0] - global_vertices[0][0]);
        inv_jacobian[1] = 1.0 / (global_vertices[2][1] - global_vertices[0][1]);
        inv_jacobian[2] = 1.0 / (global_vertices[4][2] - global_vertices[0][2]);

        return inv_jacobian;
    }

}  // namespace ippl
