
namespace ippl {

    template <typename T>
    typename LineElement<T>::set_of_vertices_type LineElement<T>::getLocalVertices() const {
        LineElement::set_of_vertices_type vertices;
        vertices[0][0] = 0.0;
        vertices[1][0] = 1.0;
        return vertices;
    }

    template <typename T>
    typename LineElement<T>::jacobian_type LineElement<T>::getTransformationJacobian(
        const set_of_vertices_type& global_vertices) const {
        // TODO
    }

    template <typename T>
    typename LineElement<T>::set_of_vertices_type LineElement<T>::getGlobalNodes(
        const jacobian_type& transformation_jacobian) const {
        // TODO
    }

}  // namespace ippl