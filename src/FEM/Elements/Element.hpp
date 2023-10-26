namespace ippl {
    template <typename T, unsigned Dim, unsigned NumVertices>
    Element<T, Dim, NumVertices>::point_t Element<T, Dim, NumVertices>::globalToLocal(
        const Element<T, Dim, NumVertices>::vertex_vec_t& global_vertices,
        const Element<T, Dim, NumVertices>::point_t& global_point) const {
        // This is actually not a matrix, but an IPPL vector that represents a diagonal matrix
        const diag_matrix_vec_t glob2loc_matrix = getTransformationJacobian(global_vertices);

        point_t local_point = glob2loc_matrix * (global_point - global_vertices[0]);

        return local_point;
    }

    template <typename T, unsigned Dim, unsigned NumVertices>
    Element<T, Dim, NumVertices>::point_t Element<T, Dim, NumVertices>::localToGlobal(
        const Element<T, Dim, NumVertices>::vertex_vec_t& global_vertices,
        const Element<T, Dim, NumVertices>::point_t& local_point) const {
        // This is actually not a matrix but an IPPL vector that represents a diagonal matrix
        const diag_matrix_vec_t loc2glob_matrix = getInverseTransformationJacobian(global_vertices);

        point_t global_point = (loc2glob_matrix * local_point) + global_vertices[0];

        return global_point;
    }

    template <typename T, unsigned Dim, unsigned NumVertices>
    T Element<T, Dim, NumVertices>::getDeterminantOfTransformationJacobian(
        const Element<T, Dim, NumVertices>::vertex_vec_t& global_vertices) const {
        T determinant = 1.0;

        // Since the jacobian is a diagonal matrix in our case the determinant is the product of the
        // diagonal elements
        for (const T& jacobian_val : getTransformationJacobian(global_vertices)) {
            determinant *= jacobian_val;
        }

        return determinant;
    }
}  // namespace ippl