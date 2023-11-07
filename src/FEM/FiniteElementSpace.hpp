
namespace ippl {
    template <typename T, unsigned Dim, unsigned NumElementVertices, unsigned NumIntegrationPoints,
              unsigned NumElementDOFs>
    FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints, NumElementDOFs>::
        FiniteElementSpace(const Mesh<T, Dim>& mesh,
                           const Element<T, Dim, NumElementVertices>& ref_element,
                           const Quadrature<T, NumIntegrationPoints>& quadrature)
        : mesh_m(mesh)
        , ref_element_m(ref_element)
        , quadrature_m(quadrature) {
        assert(mesh.Dimension == Dim);
    }

    template <typename T, unsigned Dim, unsigned NumElementVertices, unsigned NumIntegrationPoints,
              unsigned NumElementDOFs>
    std::size_t FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints,
                                   NumElementDOFs>::numElements() const {
        Vector<std::size_t, Dim> cells_per_dim = this->mesh_m.getGridsize() - 1u;

        // TODO Use a reduction instead
        std::size_t num_elements = 1;
        for (const std::size_t num_cells : cells_per_dim) {
            num_elements *= num_cells;
        }

        return num_elements;
    }

    template <typename T, unsigned Dim, unsigned NumElementVertices, unsigned NumIntegrationPoints,
              unsigned NumElementDOFs>
    FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints, NumElementDOFs>::nd_index_t
    FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints, NumElementDOFs>::
        getMeshVertexNDIndex(
            const FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints,
                                     NumElementDOFs>::index_t& vertex_index) const {
        // Copy the vertex index to the index variable we can alter during the computation.
        index_t index = global_vertex_index;

        // Create a vector to store the vertex indices in each dimension for the corresponding
        // vertex.
        nd_index_t vertex_indices;

        // This is the number of vertices in each dimension.
        Vector<std::size_t, Dim> vertices_per_dim = this->mesh_m.getGridsize();

        // The number_of_lower_dim_vertices is the product of the number of vertices per
        // dimension, it will get divided by the current dimensions number to get the index in
        // that dimension
        std::size_t remaining_number_of_vertices = 1;
        // TODO Move to KOKKOS reduction or smth
        for (const std::size_t num_vertices : vertices_per_dim) {
            remaining_number_of_vertices *= num_vertices;
        }

        for (int d = Dim - 1; d >= 0; --d) {
            remaining_number_of_vertices /= vertices_per_dim[d];
            vertex_indices[d] = index / remaining_number_of_vertices;
            index -= vertex_indices[d] * remaining_number_of_vertices;
        }

        return vertex_indices;
    };

    // implementation of function to retrieve the index of an element in each dimension
    template <typename T, unsigned Dim, unsigned NumElementVertices, unsigned NumIntegrationPoints,
              unsigned NumElementDOFs>
    FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints, NumElementDOFs>::nd_index_t
    FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints, NumElementDOFs>::
        getElementNDIndex(const FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints,
                                                   NumElementDOFs>::index_t& element_index) const {
        // Copy the element index to the index variable we can alter during the computation.
        index_t index = element_index;

        // Create a vector to store the element indices in each dimension for the corresponding
        // element.
        nd_index_t element_indices;

        // This is the number of cells in each dimension. It is one less than the number of
        // vertices in each dimension, which is returned by Mesh::getGridsize().
        Vector<std::size_t, Dim> cells_per_dim = this->mesh_m.getGridsize() - 1u;

        // The number_of_lower_dim_cells is the product of all the number of cells per
        // dimension, it will get divided by the current dimension's size to get the index in
        // that dimension
        std::size_t remaining_number_of_cells = 1;
        // TODO Move to KOKKOS reduction or smth
        for (const std::size_t num_cells : cells_per_dim) {
            remaining_number_of_cells *= num_cells;
        }

        for (int d = Dim - 1; d >= 0; --d) {
            remaining_number_of_cells /= cells_per_dim[d];
            element_indices[d] = index / remaining_number_of_cells;
            index -= element_indices[d] * remaining_number_of_cells;
        }

        return element_indices;
    }

    template <typename T, unsigned Dim, unsigned NumElementVertices, unsigned NumIntegrationPoints,
              unsigned NumElementDOFs>
    FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints,
                       NumElementDOFs>::mesh_element_vertex_vec_t
    FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints, NumElementDOFs>::
        getElementMeshVertices(
            const FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints,
                                     NumElementDOFs>::index_t& element_index) const {
        return getElementMeshVertices(getElementPositionFromIndex(element_index));
    }

    template <typename T, unsigned Dim, unsigned NumElementVertices, unsigned NumIntegrationPoints,
              unsigned NumElementDOFs>
    FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints,
                       NumElementDOFs>::mesh_element_vertex_vec_t
    FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints, NumElementDOFs>::
        getElementMeshVertices(
            const FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints,
                                     NumElementDOFs>::nd_index_t& element_indices) const {
        // Vector to store the vertex indices for the element
        mesh_vertex_vec_t vertex_indices(0);

        // TODO check, this might fail as mesh_m returns a Vector<T, Dim>
        const Vector<std::size_t, Dim> num_vertices = this->mesh_m.getGridsize();

        for (unsigned i = 0; i < NumElementVertices; ++i) {
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