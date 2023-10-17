
namespace ippl {
    template <typename T, unsigned GeometricDim>
    typename QuadrilateralElement<T, GeometricDim>::local_vertex_vector
    QuadrilateralElement<T, GeometricDim>::getLocalVertices() const {
        QuadrilateralElement::local_vertex_vector vertices;
        vertices = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}};
        return vertices;
    }
}