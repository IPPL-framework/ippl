
namespace ippl {
    template <typename T, unsigned Dim, unsigned NumElementVertices, unsigned NumIntegrationPoints,
              unsigned NumDoFs>
    FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints, NumDoFs>::
        FiniteElementSpace(const Mesh<T, Dim>& mesh,
                           const Element<T, Dim, Dim, NumElementVertices>& ref_element,
                           const Quadrature<T, NumIntegrationPoints>& quadrature)
        : mesh_m(mesh)
        , ref_element_m(ref_element)
        , quadrature_m(quadrature) {
        assert(mesh.Dimension == Dim);
    }

    template <typename T, unsigned Dim, unsigned NumElementVertices, unsigned NumIntegrationPoints,
              unsigned NumDoFs>
    FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints, NumDoFs>::vertex_vec_t
    FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints, NumDoFs>::
        getGlobalVerticesForElement(
            const FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints,
                                     NumDoFs>::index_t& element_index) const {
        return getGlobalVerticesForElement(getDimensionIndicesForElement(element_index));
    }

}  // namespace ippl