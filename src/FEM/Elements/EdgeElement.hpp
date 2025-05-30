
namespace ippl {

    template <typename T>
    KOKKOS_FUNCTION typename EdgeElement<T>::vertex_points_t EdgeElement<T>::getLocalVertices()
        const {
        // For the ordering of local vertices, see section 3.3.1:
        // https://amas.web.psi.ch/people/aadelmann/ETH-Accel-Lecture-1/projectscompleted/phys/bachelor_thesis_buehlluk.pdf
        EdgeElement::vertex_points_t vertices;
        vertices[0] = {0.0};
        vertices[1] = {1.0};
        return vertices;
    }

    template <typename T>
    KOKKOS_FUNCTION typename EdgeElement<T>::point_t EdgeElement<T>::getTransformationJacobian(
        const EdgeElement<T>::vertex_points_t& global_vertices) const {
        EdgeElement::point_t jacobian;

        jacobian[0] = (global_vertices[1][0] - global_vertices[0][0]);

        return jacobian;
    }

    template <typename T>
    KOKKOS_FUNCTION typename EdgeElement<T>::point_t
    EdgeElement<T>::getInverseTransformationJacobian(
        const EdgeElement<T>::vertex_points_t& global_vertices) const {
        EdgeElement::point_t inv_jacobian;

        inv_jacobian[0] = 1.0 / (global_vertices[1][0] - global_vertices[0][0]);

        return inv_jacobian;
    }

    template <typename T>
    KOKKOS_FUNCTION typename EdgeElement<T>::point_t EdgeElement<T>::globalToLocal(
        const EdgeElement<T>::vertex_points_t& global_vertices,
        const EdgeElement<T>::point_t& global_point) const {
        // This is actually not a matrix, but an IPPL vector that represents a diagonal matrix
        const EdgeElement<T>::point_t glob2loc_matrix =
            getInverseTransformationJacobian(global_vertices);

        EdgeElement<T>::point_t local_point = glob2loc_matrix * (global_point - global_vertices[0]);

        return local_point;
    }

    template <typename T>
    KOKKOS_FUNCTION typename EdgeElement<T>::point_t EdgeElement<T>::localToGlobal(
        const EdgeElement<T>::vertex_points_t& global_vertices,
        const EdgeElement<T>::point_t& local_point) const {
        // This is actually not a matrix but an IPPL vector that represents a diagonal matrix
        const EdgeElement<T>::point_t loc2glob_matrix = getTransformationJacobian(global_vertices);

        EdgeElement<T>::point_t global_point = (loc2glob_matrix * local_point) + global_vertices[0];

        return global_point;
    }

    template <typename T>
    KOKKOS_FUNCTION T EdgeElement<T>::getDeterminantOfTransformationJacobian(
        const EdgeElement<T>::vertex_points_t& global_vertices) const {
        T determinant = 1.0;

        // Since the jacobian is a diagonal matrix in our case the determinant is the product of the
        // diagonal elements
        for (const T& jacobian_val : getTransformationJacobian(global_vertices)) {
            determinant *= jacobian_val;
        }

        return determinant;
    }

    template <typename T>
    KOKKOS_FUNCTION typename EdgeElement<T>::point_t
    EdgeElement<T>::getInverseTransposeTransformationJacobian(
        const EdgeElement<T>::vertex_points_t& global_vertices) const {
        // Simply return the inverse transformation jacobian since it is a diagonal matrix
        return getInverseTransformationJacobian(global_vertices);
    }

    template <typename T>
    KOKKOS_FUNCTION bool EdgeElement<T>::isPointInRefElement(const Vector<T, 1>& point) const {
        // check if the local coordinates are inside the reference element
        for (size_t d = 0; d < 1; d++) {
            if (point[d] > 1.0 || point[d] < 0.0) {
                // The global coordinates are outside of the support.
                return false;
            }
        }

        return true;
    }

}  // namespace ippl
