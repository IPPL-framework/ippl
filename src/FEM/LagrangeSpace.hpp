
namespace ippl {
    // LagrangeSpace constructor, which calls the FiniteElementSpace constructor.
    template <typename T, unsigned Dim, unsigned NumElementVertices, unsigned NumIntegrationPoints>
    LagrangeSpace<T, Dim, NumElementVertices, NumIntegrationPoints>::LagrangeSpace(
        const Mesh<T, Dim>& mesh, const Element<T, Dim, Dim, NumElementVertices>& ref_element,
        const Quadrature<T, NumIntegrationPoints>& quadrature)
        : FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints>(mesh, ref_element,
                                                                               quadrature) {
        // Assert that the dimension is either 1, 2 or 3.
        static_assert(Dim >= 1 && Dim <= 3,
                      "Finite Element space only supports 1D, 2D and 3D meshes");
    }

    // implementation of function to retrieve the index of an element in each dimension
    template <typename T, unsigned Dim, unsigned NumElementVertices, unsigned NumIntegrationPoints>
    Vector<std::size_t, Dim>
    LagrangeSpace<T, Dim, NumElementVertices, NumIntegrationPoints>::getElementDimIndices(
        const std::size_t& element_index) const {
        // Copy the element index to the index variable we can alter during the computation.
        std::size_t index = element_index;

        // Create a vector to store the element indices in each dimension for the corresponding
        // element.
        Vector<std::size_t, Dim> element_indices;

        // This is the number of cells in each dimension. It is one less than the number of
        // vertices in each dimension, which is returned by Mesh::getGridsize().
        Vector<std::size_t, Dim> length_per_dim = this->mesh_m.getGridsize() - 1u;

        // The number_of_lower_dim_cells is the product of all the number of cells per
        // dimension, it will get divided by the current dimension's size to get the index in
        // that dimension
        std::size_t number_of_lower_dim_cells = 0;
        // TODO Move to KOKKOS reduction or smth
        for (const std::size_t length : length_per_dim) {
            number_of_lower_dim_cells *= length;
        }

        for (std::size_t d = Dim - 1; d > 0; --d) {
            number_of_lower_dim_cells /= length_per_dim[d];
            const std::size_t new_index = index % number_of_lower_dim_cells;
            element_indices[d]          = new_index;
            index -= new_index * number_of_lower_dim_cells;
        }

        return element_indices;
    }

    template <typename T, unsigned Dim, unsigned NumElementVertices, unsigned NumIntegrationPoints>
    Vector<std::size_t,
           LagrangeSpace<T, Dim, NumElementVertices, NumIntegrationPoints>::NumVertices>
    LagrangeSpace<T, Dim, NumElementVertices, NumIntegrationPoints>::getVerticesForElement(
        const std::size_t& element_index) const {
        return getVerticesForElement(getElementDimIndices(element_index));
    }

    template <typename T, unsigned Dim, unsigned NumElementVertices, unsigned NumIntegrationPoints>
    Vector<std::size_t,
           LagrangeSpace<T, Dim, NumElementVertices, NumIntegrationPoints>::NumVertices>
    LagrangeSpace<T, Dim, NumElementVertices, NumIntegrationPoints>::getVerticesForElement(
        const Vector<std::size_t, Dim>& element_indices) const {
        // Vector to store the vertex indices for the element
        Vector<std::size_t, NumVertices> vertex_indices(0);

        // TODO check, this might fail as mesh_m returns a Vector<T, Dim>
        const Vector<std::size_t, Dim> num_vertices = this->mesh_m.getGridsize();

        for (unsigned i = 0; i < NumVertices; ++i) {
            for (unsigned d = 0; d < Dim; ++d) {
                vertex_indices[i] += element_indices[d];

                // We have to add one to the vertex index if it is the second
                // vertex in the current dimension.
                // This is the case if i % 2^(d+1) > 2^d  which is
                // i % (1 << d + 1) > (1 << d) in C++.
                // Or in other words, if the bit at position d is set, which is
                // the case if i & (1 << d) != 0.
                // TODO maybe rewrite this text as it is not very clear
                if ((i & (1 << d)) != 0)
                    vertex_indices[i] += 1;

                if (d > 0)
                    vertex_indices[i] *= num_vertices[d];
            }
        }

        return vertex_indices;
    }

}  // namespace ippl