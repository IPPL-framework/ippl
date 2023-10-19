
namespace ippl {
    template <typename T, unsigned GeometricDim>
    typename QuadrilateralElement<T, GeometricDim>::local_vertex_vector
    QuadrilateralElement<T, GeometricDim>::getLocalVertices() const {
        QuadrilateralElement::local_vertex_vector vertices;
        vertices = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};
        return vertices;
    }

    template <typename T, unsigned GeometricDim>
    QuadrilateralElement<T, GeometricDim>::jacobian_t
    QuadrilateralElement<T, GeometricDim>::getLinearTransformationJacobian(
        const QuadrilateralElement<T, GeometricDim>::global_vertex_vector& global_vertices) const {
        QuadrilateralElement::jacobian_t jacobian;

        for (unsigned d = 0; d < GeometricDim; ++d) {
            jacobian[0][d] = 1.0 / (global_vertices[1][d] - global_vertices[0][d]);
            jacobian[1][d] = 1.0 / (global_vertices[2][d] - global_vertices[0][d]);
        }

        return jacobian;
    }

    template <typename T, unsigned GeometricDim>
    QuadrilateralElement<T, GeometricDim>::inverse_jacobian_t
    QuadrilateralElement<T, GeometricDim>::getInverseLinearTransformationJacobian(
        const QuadrilateralElement<T, GeometricDim>::global_vertex_vector& global_vertices) const {
        QuadrilateralElement::inverse_jacobian_t inv_jacobian;

        for (unsigned d = 0; d < GeometricDim; ++d) {
            inv_jacobian[0][d] = global_vertices[1][d] - global_vertices[0][d];
            inv_jacobian[1][d] = global_vertices[2][d] - global_vertices[0][d];
        }

        return inv_jacobian;
    }

}  // namespace ippl