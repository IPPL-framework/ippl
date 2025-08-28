
namespace ippl {
    template <typename T>
    KOKKOS_FUNCTION typename QuadrilateralElement<T>::vertex_points_t
    QuadrilateralElement<T>::getLocalVertices() const {
        // For the ordering of local vertices, see section 3.3.1:
        // https://amas.web.psi.ch/people/aadelmann/ETH-Accel-Lecture-1/projectscompleted/phys/bachelor_thesis_buehlluk.pdf
        QuadrilateralElement::vertex_points_t vertices = {
            {0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}};

        return vertices;
    }

    template <typename T>
    KOKKOS_FUNCTION typename QuadrilateralElement<T>::point_t
    QuadrilateralElement<T>::getTransformationJacobian(
        const QuadrilateralElement<T>::vertex_points_t& global_vertices) const {
        QuadrilateralElement::point_t jacobian;

        jacobian[0] = (global_vertices[1][0] - global_vertices[0][0]);
        jacobian[1] = (global_vertices[2][1] - global_vertices[0][1]);

        return jacobian;
    }

    template <typename T>
    KOKKOS_FUNCTION typename QuadrilateralElement<T>::point_t
    QuadrilateralElement<T>::getInverseTransformationJacobian(
        const QuadrilateralElement<T>::vertex_points_t& global_vertices) const {
        QuadrilateralElement::point_t inv_jacobian;

        inv_jacobian[0] = 1.0 / (global_vertices[1][0] - global_vertices[0][0]);
        inv_jacobian[1] = 1.0 / (global_vertices[2][1] - global_vertices[0][1]);

        return inv_jacobian;
    }

    template <typename T>
    KOKKOS_FUNCTION typename QuadrilateralElement<T>::point_t
    QuadrilateralElement<T>::globalToLocal(
        const QuadrilateralElement<T>::vertex_points_t& global_vertices,
        const QuadrilateralElement<T>::point_t& global_point) const {
        // This is actually not a matrix, but an IPPL vector that represents a diagonal matrix
        const QuadrilateralElement<T>::point_t glob2loc_matrix =
            getInverseTransformationJacobian(global_vertices);

        QuadrilateralElement<T>::point_t local_point =
            glob2loc_matrix * (global_point - global_vertices[0]);

        return local_point;
    }

    template <typename T>
    KOKKOS_FUNCTION typename QuadrilateralElement<T>::point_t
    QuadrilateralElement<T>::localToGlobal(
        const QuadrilateralElement<T>::vertex_points_t& global_vertices,
        const QuadrilateralElement<T>::point_t& local_point) const {
        // This is actually not a matrix but an IPPL vector that represents a diagonal matrix
        const QuadrilateralElement<T>::point_t loc2glob_matrix =
            getTransformationJacobian(global_vertices);

        QuadrilateralElement<T>::point_t global_point =
            (loc2glob_matrix * local_point) + global_vertices[0];

        return global_point;
    }

    template <typename T>
    KOKKOS_FUNCTION T QuadrilateralElement<T>::getDeterminantOfTransformationJacobian(
        const QuadrilateralElement<T>::vertex_points_t& global_vertices) const {
        T determinant = 1.0;

        // Since the jacobian is a diagonal matrix in our case the determinant is the product of the
        // diagonal elements
        for (const T& jacobian_val : getTransformationJacobian(global_vertices)) {
            determinant *= jacobian_val;
        }

        return determinant;
    }

    template <typename T>
    KOKKOS_FUNCTION typename QuadrilateralElement<T>::point_t
    QuadrilateralElement<T>::getInverseTransposeTransformationJacobian(
        const QuadrilateralElement<T>::vertex_points_t& global_vertices) const {
        // Simply return the inverse transformation jacobian since it is a diagonal matrix
        return getInverseTransformationJacobian(global_vertices);
    }

    template <typename T>
    KOKKOS_FUNCTION bool QuadrilateralElement<T>::isPointInRefElement(
        const Vector<T, 2>& point) const {
        // check if the local coordinates are inside the reference element

        for (size_t d = 0; d < 2; d++) {
            if (point[d] > 1.0 || point[d] < 0.0) {
                // The global coordinates are outside of the support.
                return false;
            }
        }

        return true;
    }

}  // namespace ippl
