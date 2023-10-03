
namespace ippl {
    template <typename T, unsigned Dim>
    UniformCartesianFEMesh<T, Dim>::UniformCartesianFEMesh(const UniformCartesian<T, Dim>& mesh) {
        this->initialize(mesh., hx, origin);
    }
}  // namespace ippl