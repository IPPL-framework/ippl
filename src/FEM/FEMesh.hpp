

namespace ippl {
    template <typename T, unsigned Dim>
    FEMesh<T, Dim>::FEMesh(const Mesh<T, Dim>& mesh)
        : Mesh<T, Dim>(mesh) {}
}  // namespace ippl