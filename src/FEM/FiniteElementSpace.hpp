
namespace ippl {
    template <typename T, unsigned Dim, unsigned NumElementVertices, unsigned NumIntegrationPoints>
    FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints>::FiniteElementSpace(
        const Mesh<T, Dim>& mesh, const Element<T, Dim, Dim, NumElementVertices>& ref_element,
        const Quadrature<T, NumIntegrationPoints>& quadrature)
        : mesh_m(mesh)
        , ref_element_m(ref_element)
        , quadrature_m(quadrature) {
        assert(mesh.Dimension == Dim);
    }

}  // namespace ippl