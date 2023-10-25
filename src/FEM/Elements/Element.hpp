namespace ippl {
    template <typename T, unsigned Dim, unsigned NumVertices>
    Element<T, Dim, NumVertices>::point_t Element<T, Dim, NumVertices>::globalToLocal(
        const Element<T, Dim, NumVertices>::vertex_vec_t& global_vertices,
        const Element<T, Dim, NumVertices>::point_t& global_point) const {
        const matrix_t glob2loc_matrix = getTransformationJacobian(global_vertices);

        point_t local_point;

        const point_t adjusted_point = global_point - global_vertices[0];
        for (unsigned d = 0; d < Dim; d++) {
            local_point[d] = glob2loc_matrix[d].dot(adjusted_point);
        }

        return local_point;
    }

    template <typename T, unsigned Dim, unsigned NumVertices>
    Element<T, Dim, NumVertices>::point_t Element<T, Dim, NumVertices>::localToGlobal(
        const Element<T, Dim, NumVertices>::vertex_vec_t& global_vertices,
        const Element<T, Dim, NumVertices>::point_t& local_point) const {
        const matrix_t loc2glob_matrix = getInverseTransformationJacobian(global_vertices);

        point_t global_point;

        for (unsigned d = 0; d < Dim; d++) {
            global_point[d] = loc2glob_matrix[d].dot(local_point);
        }
        global_point += global_vertices[0];

        return global_point;
    }
}  // namespace ippl