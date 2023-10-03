
namespace ippl {
    template <unsigned Order>
    HexahedralElement::HexahedralElement(std::size_t global_index,
                                         Vector<std::size_t, 8> global_indices_of_vertices)
        : Element<3, 8>(global_index, global_indices_of_vertices) {}

}  // namespace ippl