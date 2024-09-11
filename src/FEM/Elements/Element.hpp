namespace ippl {
    /*
    template <typename T, unsigned Dim, unsigned NumVertices>
    KOKKOS_FUNCTION
    typename Element<T, Dim, NumVertices>::point_t Element<T, Dim, NumVertices>::globalToLocal(
        const Element<T, Dim, NumVertices>::mesh_element_vertex_point_vec_t& global_vertices,
        const Element<T, Dim, NumVertices>::point_t& global_point) const {
        // This is actually not a matrix, but an IPPL vector that represents a diagonal matrix
        const diag_matrix_vec_t glob2loc_matrix = getInverseTransformationJacobian(global_vertices);

        point_t local_point = glob2loc_matrix * (global_point - global_vertices[0]);

        return local_point;
    }

    template <typename T, unsigned Dim, unsigned NumVertices>
    KOKKOS_FUNCTION
    typename Element<T, Dim, NumVertices>::point_t Element<T, Dim, NumVertices>::localToGlobal(
        const Element<T, Dim, NumVertices>::mesh_element_vertex_point_vec_t& global_vertices,
        const Element<T, Dim, NumVertices>::point_t& local_point) const {
        // This is actually not a matrix but an IPPL vector that represents a diagonal matrix
        const diag_matrix_vec_t loc2glob_matrix = getTransformationJacobian(global_vertices);

        point_t global_point = (loc2glob_matrix * local_point) + global_vertices[0];

        return global_point;
    }

    template <typename T, unsigned Dim, unsigned NumVertices>
    KOKKOS_FUNCTION
    T Element<T, Dim, NumVertices>::getDeterminantOfTransformationJacobian(
        const Element<T, Dim, NumVertices>::mesh_element_vertex_point_vec_t& global_vertices)
        const {
        T determinant = 1.0;

        // Since the jacobian is a diagonal matrix in our case the determinant is the product of the
        // diagonal elements
        for (const T& jacobian_val : getTransformationJacobian(global_vertices)) {
            determinant *= jacobian_val;
        }

        return determinant;
    }

    template <typename T, unsigned Dim, unsigned NumVertices>
    KOKKOS_FUNCTION
    typename Element<T, Dim, NumVertices>::diag_matrix_vec_t
    Element<T, Dim, NumVertices>::getInverseTransposeTransformationJacobian(
        const Element<T, Dim, NumVertices>::mesh_element_vertex_point_vec_t& global_vertices)
        const {
        // Simply return the inverse transformation jacobian since it is a diagonal matrix
        return getInverseTransformationJacobian(global_vertices);
    }

    template <typename T, unsigned Dim, unsigned NumVertices>
    KOKKOS_FUNCTION
    bool Element<T, Dim, NumVertices>::isPointInRefElement(const Vector<T, Dim>& point) const {
        // check if the local coordinates are inside the reference element

        for (size_t d = 0; d < Dim; d++) {
            if (point[d] > 1.0 || point[d] < 0.0) {
                // The global coordinates are outside of the support.
                return false;
            }
        }

        return true;
    }
    */
}  // namespace ippl
