namespace ippl {
    template <typename T, unsigned GeometricDim, unsigned TopologicalDim, unsigned NumVertices>
    Element<T, GeometricDim, TopologicalDim, NumVertices>::local_point_t
    Element<T, GeometricDim, TopologicalDim, NumVertices>::globalToLocal(
        const Element<T, GeometricDim, TopologicalDim, NumVertices>::global_vertex_vec_t&
            global_vertices,
        const Element<T, GeometricDim, TopologicalDim, NumVertices>::global_point_t& global_point)
        const {
        const jacobian_t glob2loc_matrix = getTransformationJacobian(global_vertices);

        local_point_t local_point;

        const global_point_t adjusted_point = global_point - global_vertices[0];
        for (unsigned d = 0; d < TopologicalDim; d++) {
            local_point[d] = glob2loc_matrix[d].dot(adjusted_point);
        }

        return local_point;
    }

    template <typename T, unsigned GeometricDim, unsigned TopologicalDim, unsigned NumVertices>
    Element<T, GeometricDim, TopologicalDim, NumVertices>::global_point_t
    Element<T, GeometricDim, TopologicalDim, NumVertices>::localToGlobal(
        const Element<T, GeometricDim, TopologicalDim, NumVertices>::global_vertex_vec_t&
            global_vertices,
        const Element<T, GeometricDim, TopologicalDim, NumVertices>::local_point_t& local_point)
        const {
        const inverse_jacobian_t loc2glob_matrix =
            getInverseTransformationJacobian(global_vertices);

        global_point_t global_point;

        for (unsigned d = 0; d < GeometricDim; d++) {
            global_point[d] = loc2glob_matrix[d].dot(local_point);
        }
        global_point += global_vertices[0];

        return global_point;
    }
}  // namespace ippl