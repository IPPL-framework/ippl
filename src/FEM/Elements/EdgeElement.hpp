
namespace ippl {

    template <typename T, unsigned GeometricDim>
    typename EdgeElement<T, GeometricDim>::local_vertex_vector
    EdgeElement<T, GeometricDim>::getLocalVertices() const {
        EdgeElement::local_vertex_vector vertices;
        vertices[0][0] = 0.0;
        vertices[1][0] = 1.0;
        return vertices;
    }

    // template <typename T, unsigned GeometricDim>
    // typename EdgeElement<T, GeometricDim>::jacobian_type
    // EdgeElement<T, GeometricDim>::getTransformationJacobian(
    //     const global_vertex_vector& global_vertices) const {
    //     const jacobian_type jacobian = 0;  // TODO fix

    //     // TODO

    //     return jacobian;
    // }

    // template <typename T, unsigned GeometricDim>
    // typename EdgeElement<T, GeometricDim>::global_vertex_vector
    // EdgeElement<T, GeometricDim>::getGlobalNodes(
    //     const jacobian_type& transformation_jacobian) const {
    //     global_vertex_vector vector;

    //     // TODO

    //     return vector;
    // }

}  // namespace ippl