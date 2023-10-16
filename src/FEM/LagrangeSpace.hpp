
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
    NDIndex<Dim>
    LagrangeSpace<T, Dim, NumElementVertices, NumIntegrationPoints>::getNDIndexForElement(
        const Index& element_index) const {
        // Copy the element index to the index variable we can alter during the computation.
        Index index = element_index;

        // Create a vector to store the element indices in each dimension for the corresponding
        // element.
        NDIndex<Dim> element_indices;

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

    template <typename T, unsigned Dim, unsigned NumElementVertices, unsigned NumIntegrationPoints>
    NDIndex<Dim>
    LagrangeSpace<T, Dim, NumElementVertices, NumIntegrationPoints>::getNDIndexForVertex(
        const Index& vertex_index) const {
        // Copy the vertex index to the index variable we can alter during the computation.
        Index index = vertex_index;

        // Create a vector to store the vertex indices in each dimension for the corresponding
        // vertex.
        NDIndex<Dim> vertex_indices;

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

    template <typename T, unsigned Dim, unsigned NumElementVertices, unsigned NumIntegrationPoints>
    Vector<T, Dim>
    LagrangeSpace<T, Dim, NumElementVertices, NumIntegrationPoints>::getCoordinatesForVertex(
        const NDIndex<Dim>& vertex_indices) const {
        return this->mesh_m.getVertexPosition(vertex_indices);
    }

    template <typename T, unsigned Dim, unsigned NumElementVertices, unsigned NumIntegrationPoints>
    Vector<T, Dim>
    LagrangeSpace<T, Dim, NumElementVertices, NumIntegrationPoints>::getCoordinatesForVertex(
        const Index& vertex_index) const {
        return getCoordinatesForVertex(vertex_index);
    }

    template <typename T, unsigned Dim, unsigned NumElementVertices, unsigned NumIntegrationPoints>
    LagrangeSpace<T, Dim, NumElementVertices, NumIntegrationPoints>::vertex_vector_t
    LagrangeSpace<T, Dim, NumElementVertices, NumIntegrationPoints>::getGlobalVerticesForElement(
        const NDIndex<Dim>& element_indices) const {
        // Vector to store the vertex indices for the element
        Vector<std::size_t, NumElementVertices> vertex_indices(0);

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

    template <typename T, unsigned Dim, unsigned NumElementVertices, unsigned NumIntegrationPoints>
    T LagrangeSpace<T, Dim, NumElementVertices, NumIntegrationPoints>::evaluateLoadVector(
        const Index& j) const {
        assert(j < NumIntegrationPoints);  // TODO change assert to be correct
        // TODO implement
        return 0;
    }

    template <typename T, unsigned Dim, unsigned NumElementVertices, unsigned NumIntegrationPoints>
    T LagrangeSpace<T, Dim, NumElementVertices, NumIntegrationPoints>::evaluateStiffnessMatrix(
        const Index& i, const Index& j) const {
        assert(i < NumIntegrationPoints);  // TODO change assert to be correct
        assert(j < NumIntegrationPoints);  // TODO change assert to be correct
        // TODO implement
        return 0;
    }

    template <typename T, unsigned Dim, unsigned NumElementVertices, unsigned NumIntegrationPoints>
    T LagrangeSpace<T, Dim, NumElementVertices, NumIntegrationPoints>::evaluateBasis(
        const Index& vertex_index, const Vector<T, Dim>& global_coordinates) const {
        const NDIndex<Dim> vertex_indices        = getNDIndexForVertex(vertex_index);
        const Vector<T, Dim> vertex_coodrdinates = getCoordinatesForVertex(vertex_indices);
        const Vector<T, Dim> h                   = mesh_m.getDeltaVertex(vertex_indices);

        // If the global coordinates are outside of the support of the basis function in any
        // dimension return 0.
        for (std::size_t d = 0; d < Dim; d++) {
            if (global_coordinates[d] >= vertex_coodrdinates[d] + h[d]
                || global_coordinates[d] <= vertex_coodrdinates[d] - h[d]) {
                // The global coordinates are outside of the support of the basis function.
                return 0.0;
            }
        }

        // The variable that accumulates the product of the shape functions.
        T product = 1;

        for (std::size_t d = 0; d < Dim; d++) {
            if (global_coordinates[d] < vertex_coodrdinates[d]) {
                product *= (global_coordinates[d] - (vertex_coodrdinates[d] - h[d])) / h[d];
            } else {
                product *= ((vertex_coodrdinates[d] + h[d]) - global_coordinates[d]) / h[d];
            }
        }

        return product;
    }

}  // namespace ippl