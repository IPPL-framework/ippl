
namespace ippl {
    template <typename T, unsigned Dim>
    FiniteElementSpace<T, Dim>::FiniteElementSpace(const Mesh<T, Dim>& mesh,
                                                   const Element& ref_element,
                                                   const Quadrature& quadrature, unsigned degree)
        : mesh_m(mesh)
        , ref_element_m(ref_element)
        , quadrature_m(quadrature)
        , degree_m(degree) {}

    template <typename T, unsigned Dim>
    Vector<std::size_t, Dim> FiniteElementSpace<T, Dim>::getElementDimIndices(
        const std::size_t& element_index) {
        static_assert(Dim >= 1 && Dim <= 3,
                      "Finite Element space only supports 1D, 2D and 3D meshes");

        // Copt the element index to the index variable we can alter in the computation.
        std::size_t index = element_index;

        // Create a vector to store the element indices in each dimension for the corresponding
        // element.
        Vector<std::size_t, Dim> element_indices();

        // This is the number of cells in each dimension. It is one less than the number of
        // vertices in each dimension, which is returned by Mesh::getGridsize().
        Vector<std::size_t, Dim> length_per_dim = getGridsize() - 1u;

        // The number_of_lower_dim_cells is the product of all the number of cells per
        // dimension, it will get divided by the current dimension's size to get the index in
        // that dimension
        std::size_t number_of_lower_dim_cells = std::accumulate(
            length_per_dim.begin(), length_per_dim.end(), 1, [](std::size_t a, std::size_t b) {
                return a * b;
            });

        for (std::size_t d = Dim - 1; d > 0; --i) {
            number_of_lower_dim_cells /= length_per_dim[d];
            const std::size_t new_index = index % number_of_lower_dim_cells;
            element_indices[d]          = new_index;
            index -= new_index * number_of_lower_dim_cells;
        }

        return element_indices;
    }

}  // namespace ippl