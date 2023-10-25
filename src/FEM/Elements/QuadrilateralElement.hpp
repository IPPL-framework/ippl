
namespace ippl {
    template <typename T>
    typename QuadrilateralElement<T>::vertex_vec_t QuadrilateralElement<T>::getLocalVertices()
        const {
        QuadrilateralElement::vertex_vec_t vertices;
        vertices = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};
        return vertices;
    }

    template <typename T>
    QuadrilateralElement<T>::matrix_t QuadrilateralElement<T>::getTransformationJacobian(
        const QuadrilateralElement<T>::vertex_vec_t& global_vertices) const {
        QuadrilateralElement::matrix_t jacobian;

        jacobian[0][0] = 1.0 / (global_vertices[1][0] - global_vertices[0][0]);
        jacobian[0][1] = 0.0;
        jacobian[1][0] = 0.0;
        jacobian[1][1] = 1.0 / (global_vertices[2][1] - global_vertices[0][1]);

        return jacobian;
    }

    template <typename T>
    QuadrilateralElement<T>::matrix_t QuadrilateralElement<T>::getInverseTransformationJacobian(
        const QuadrilateralElement<T>::vertex_vec_t& global_vertices) const {
        QuadrilateralElement::matrix_t inv_jacobian;

        inv_jacobian[0][0] = global_vertices[1][0] - global_vertices[0][0];
        inv_jacobian[0][1] = 0.0;
        inv_jacobian[1][0] = 0.0;
        inv_jacobian[1][1] = global_vertices[2][1] - global_vertices[0][1];

        return inv_jacobian;
    }

    template <typename T>
    T QuadrilateralElement<T>::getDeterminantOfTransformationJacobian(
        const QuadrilateralElement<T>::vertex_vec_t& global_vertices) const {
        return 1.0
               / ((global_vertices[1][0] - global_vertices[0][0])
                  * (global_vertices[2][1] - global_vertices[0][1]));
    }

}  // namespace ippl