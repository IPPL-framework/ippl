
namespace ippl {
    template <typename T, unsigned Dim>
    template <typename QuadratureType>
    FiniteElementSpace<T, Dim>::FiniteElementSpace(const UniformCartesian<T, Dim>& mesh,
                                                   QuadratureType& quadrature, unsigned degree = 1)
        : degree_m(degree)
        , femesh_m(nullptr)
        , quadrature_m(quadrature) {
        // Then create a FEMesh from the given mesh, in this case a UniformCartesian mesh
        femesh_m = new UniformCartesianFEMesh<T, Dim>(mesh);
    }

    template <typename T, unsigned Dim>
    FiniteElementSpace<T, Dim>::~FiniteElementSpace() {
        delete femesh_m;
    }

}  // namespace ippl