
namespace ippl {

    template <typename T, unsigned GeometricDim>
    EdgeElement<T, GeometricDim>::local_vertex_vector
    EdgeElement<T, GeometricDim>::getLocalVertices() const {
        EdgeElement::local_vertex_vector vertices;
        vertices[0] = {0.0};
        vertices[1] = {1.0};
        return vertices;
    }

    // global to local
    template <typename T, unsigned GeometricDim>
    EdgeElement<T, GeometricDim>::jacobian_t
    EdgeElement<T, GeometricDim>::getLinearTransformationJacobian(
        const EdgeElement<T, GeometricDim>::global_vertex_vector& global_vertices) const {
        EdgeElement::jacobian_t jacobian;

        for (unsigned d = 0; d < GeometricDim; ++d) {
            jacobian[0][d] = 1.0 / (global_vertices[1][d] - global_vertices[0][d]);
        }

        return jacobian;
    }

    // local to global
    template <typename T, unsigned GeometricDim>
    EdgeElement<T, GeometricDim>::inverse_jacobian_t
    EdgeElement<T, GeometricDim>::getInverseLinearTransformationJacobian(
        const EdgeElement<T, GeometricDim>::global_vertex_vector& global_vertices) const {
        EdgeElement::inverse_jacobian_t
            inv_jacobian;  // ippl::Vector<ippl::Vector<T, 1>, GeometricDim>

        for (unsigned d = 0; d < GeometricDim; ++d) {
            inv_jacobian[d][0] = global_vertices[1][d] - global_vertices[0][d];
        }

        return inv_jacobian;
    }

}  // namespace ippl