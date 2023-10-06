
namespace ippl {
    // LagrangeSpace constructor, which calls the FiniteElementSpace constructor.
    template <typename T, unsigned Dim>
    LagrangeSpace<T, Dim>::LagrangeSpace(const Mesh<T, Dim>& mesh,
                                         const Element<T, Dim>* ref_element,
                                         const Quadrature<T>* quadrature, const unsigned& degree)
        : FiniteElementSpace<T, Dim>(mesh, ref_element, quadrature, degree) {
        // Assert that the dimension is either 1, 2 or 3.
        static_assert(Dim >= 1 && Dim <= 3,
                      "Finite Element space only supports 1D, 2D and 3D meshes");
    }

    // implementation of function to retrieve the index of an element in each dimension
    template <typename T, unsigned Dim>
    Vector<std::size_t, Dim> LagrangeSpace<T, Dim>::getElementDimIndices(
        const std::size_t& element_index) const {
        // Copy the element index to the index variable we can alter during the computation.
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
        std::size_t number_of_lower_dim_cells = 0;
        // TODO Move to KOKKOS reduction or smth
        for (const std::size_t length : length_per_dim) {
            number_of_lower_dim_cells *= length;
        }

        for (std::size_t d = Dim - 1; d > 0; --i) {
            number_of_lower_dim_cells /= length_per_dim[d];
            const std::size_t new_index = index % number_of_lower_dim_cells;
            element_indices[d]          = new_index;
            index -= new_index * number_of_lower_dim_cells;
        }

        return element_indices;
    }

    template <typename T, unsigned Dim>
    Vector<std::size_t, LagrangeSpace<T, Dim>::NumVertices>
    LagrangeSpace<T, Dim>::getVerticesForElement(const std::size_t& element_index) const {
        return getVerticesForElement(getElementDimIndices(element_index));
    }

    template <typename T, unsigned Dim>
    Vector<std::size_t, LagrangeSpace<T, Dim>::NumVertices>
    LagrangeSpace<T, Dim>::getVerticesForElement(
        const Vector<std::size_t, Dim>& element_indices) const {
        // TODO
    }

}  // namespace ippl