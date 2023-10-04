
namespace ippl {
    template <unsigned Dim, unsigned NumVertices>
    Element<Dim, NumVertices>::Element(const std::size_t& global_index,
                                       Vector<std::size_t, NumVertices>& global_indices_of_vertices)
        : global_index_m(global_index)
        , global_indices_of_vertices_m(std::move(global_indices_of_vertices)) {}

    template <unsigned Dim, unsigned NumVertices>
    Vector<std::size_t, NumVertices> Element<Dim, NumVertices>::getGlobalIndicesOfVertices() const {
        return global_indices_of_vertices_m;
    }
}  // namespace ippl