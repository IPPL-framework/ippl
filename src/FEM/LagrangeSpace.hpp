
namespace ippl {
    // LagrangeSpace constructor, which calls the FiniteElementSpace constructor.
    template <typename T, unsigned Dim, unsigned Order, unsigned NumElementVertices,
              unsigned NumIntegrationPoints>
    LagrangeSpace<T, Dim, Order, NumElementVertices, NumIntegrationPoints>::LagrangeSpace(
        const Mesh<T, Dim>& mesh, const Element<T, Dim, Dim, NumElementVertices>& ref_element,
        const Quadrature<T, NumIntegrationPoints>& quadrature)
        : FiniteElementSpace<T, Dim, Order, NumElementVertices, NumIntegrationPoints>(
            mesh, ref_element, quadrature) {
        // Assert that the dimension is either 1, 2 or 3.
        static_assert(Dim >= 1 && Dim <= 3,
                      "Finite Element space only supports 1D, 2D and 3D meshes");
    }

    // implementation of function to retrieve the index of an element in each dimension
    template <typename T, unsigned Dim, unsigned Order, unsigned NumElementVertices,
              unsigned NumIntegrationPoints>
    LagrangeSpace<T, Dim, Order, NumElementVertices, NumIntegrationPoints>::index_vec_t
    LagrangeSpace<T, Dim, Order, NumElementVertices, NumIntegrationPoints>::
        getDimensionIndicesForElement(
            const LagrangeSpace<T, Dim, Order, NumElementVertices, NumIntegrationPoints>::index_t&
                element_index) const {
        // Copy the element index to the index variable we can alter during the computation.
        index_t index = element_index;

        // Create a vector to store the element indices in each dimension for the corresponding
        // element.
        index_vec_t element_indices;

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

    template <typename T, unsigned Dim, unsigned Order, unsigned NumElementVertices,
              unsigned NumIntegrationPoints>
    LagrangeSpace<T, Dim, Order, NumElementVertices, NumIntegrationPoints>::index_vec_t
    LagrangeSpace<T, Dim, Order, NumElementVertices, NumIntegrationPoints>::
        getDimensionIndicesForVertex(
            const LagrangeSpace<T, Dim, Order, NumElementVertices, NumIntegrationPoints>::index_t&
                global_vertex_index) const {
        // Copy the vertex index to the index variable we can alter during the computation.
        index_t index = global_vertex_index;

        // Create a vector to store the vertex indices in each dimension for the corresponding
        // vertex.
        index_vec_t vertex_indices;

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

    template <typename T, unsigned Dim, unsigned Order, unsigned NumElementVertices,
              unsigned NumIntegrationPoints>
    NDIndex<Dim>
    LagrangeSpace<T, Dim, Order, NumElementVertices, NumIntegrationPoints>::makeNDIndex(
        const LagrangeSpace<T, Dim, Order, NumElementVertices, NumIntegrationPoints>::index_vec_t&
            vertex_indices) const {
        // Not sure if this is the best way, but the getVertexPosition function expects an NDIndex,
        // with the vertex index used being the first in the NDIndex. No other index is used, so
        // we can just set the first and the last to the index we actually want.
        NDIndex<Dim> nd_index;
        for (unsigned d = 0; d < Dim; ++d) {
            nd_index[d] = Index(vertex_indices[d], vertex_indices[d]);
        }
        return nd_index;
    }

    template <typename T, unsigned Dim, unsigned Order, unsigned NumElementVertices,
              unsigned NumIntegrationPoints>
    LagrangeSpace<T, Dim, Order, NumElementVertices, NumIntegrationPoints>::coord_vec_t
    LagrangeSpace<T, Dim, Order, NumElementVertices, NumIntegrationPoints>::getCoordinatesForVertex(
        const LagrangeSpace<T, Dim, Order, NumElementVertices, NumIntegrationPoints>::index_vec_t&
            vertex_indices) const {
        return this->mesh_m.getVertexPosition(makeNDIndex(vertex_indices));
    }

    template <typename T, unsigned Dim, unsigned Order, unsigned NumElementVertices,
              unsigned NumIntegrationPoints>
    LagrangeSpace<T, Dim, Order, NumElementVertices, NumIntegrationPoints>::coord_vec_t
    LagrangeSpace<T, Dim, Order, NumElementVertices, NumIntegrationPoints>::getCoordinatesForVertex(
        const LagrangeSpace<T, Dim, Order, NumElementVertices, NumIntegrationPoints>::index_t&
            global_vertex_index) const {
        const index_vec_t vertex_indices = getDimensionIndicesForVertex(global_vertex_index);
        return getCoordinatesForVertex(vertex_indices);
    }

    template <typename T, unsigned Dim, unsigned Order, unsigned NumElementVertices,
              unsigned NumIntegrationPoints>
    LagrangeSpace<T, Dim, Order, NumElementVertices, NumIntegrationPoints>::vertex_vec_t
    LagrangeSpace<T, Dim, Order, NumElementVertices, NumIntegrationPoints>::
        getGlobalVerticesForElement(
            const LagrangeSpace<T, Dim, Order, NumElementVertices,
                                NumIntegrationPoints>::index_vec_t& element_indices) const {
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

    // template <typename T, unsigned Dim, unsigned NumElementVertices, unsigned
    // NumIntegrationPoints> void LagrangeSpace<T, Dim, NumElementVertices,
    // NumIntegrationPoints>::evaluateLoadVector(
    //     LagrangeSpace<T, Dim, NumElementVertices, NumIntegrationPoints>::value_vec_t& b) const {
    //     assert(j < NumIntegrationPoints);  // TODO change assert to be correct
    //     // TODO implement
    //     return 0;
    // }

    // template <typename T, unsigned Dim, unsigned NumElementVertices, unsigned
    // NumIntegrationPoints> void LagrangeSpace<T, Dim, NumElementVertices,
    // NumIntegrationPoints>::evaluateAx(
    //     const LagrangeSpace<T, Dim, NumElementVertices, NumIntegrationPoints>::value_vec_t& x,
    //     LagrangeSpace<T, Dim, NumElementVertices, NumIntegrationPoints>::value_vec_t& Ax) const {
    //     assert(i < NumIntegrationPoints);  // TODO change assert to be correct
    //     assert(j < NumIntegrationPoints);  // TODO change assert to be correct
    //     // TODO implement
    //     return 0;
    // }

    template <typename T, unsigned Dim, unsigned Order, unsigned NumElementVertices,
              unsigned NumIntegrationPoints>
    bool LagrangeSpace<T, Dim, Order, NumElementVertices,
                       NumIntegrationPoints>::isLocalPointInRefElement(const Vector<T, Dim>& point)
        const {
        // check if the local coordinates are inside the reference element

        // TODO change from hardcoded for n-cuboid elements to using function of the Element class
        for (std::size_t d = 0; d < Dim; d++) {
            if (point[d] > 1.0 || point[d] < 0.0) {
                // The global coordinates are outside of the support.
                return false;
            }
        }

        return true;
    }

    template <typename T, unsigned Dim, unsigned Order, unsigned NumElementVertices,
              unsigned NumIntegrationPoints>
    T LagrangeSpace<T, Dim, Order, NumElementVertices, NumIntegrationPoints>::evaluateBasis(
        const LagrangeSpace<T, Dim, Order, NumElementVertices, NumIntegrationPoints>::index_t&
            local_vertex_index,
        const Vector<T, Dim>& local_coordinates) const {
        // Assert that the local vertex index is valid.
        assert(local_vertex_index < NumElementVertices
               && "The local vertex index is invalid");  // TODO assumes 1st order Lagrange

        assert(isLocalPointInRefElement(local_coordinates) && "Point is not in reference element");

        // Get the local vertex indices for the local vertex index.
        const index_vec_t local_vertex_indices =
            this->ref_element_m.getLocalVertices()[local_vertex_index];

        // The variable that accumulates the product of the shape functions.
        T product = 1;

        for (std::size_t d = 0; d < Dim; d++) {
            if (local_coordinates[d] < local_vertex_indices[d]) {
                product *= local_coordinates[d];
            } else {
                product *= 1.0 - local_coordinates[d];
            }
        }

        return product;
    }

    template <typename T, unsigned Dim, unsigned Order, unsigned NumElementVertices,
              unsigned NumIntegrationPoints>
    LagrangeSpace<T, Dim, Order, NumElementVertices, NumIntegrationPoints>::gradient_vec_t
    LagrangeSpace<T, Dim, Order, NumElementVertices, NumIntegrationPoints>::evaluateBasisGradient(
        const LagrangeSpace<T, Dim, Order, NumElementVertices, NumIntegrationPoints>::index_t&
            local_vertex_index,
        const Vector<T, Dim>& local_coordinates) const {
        // TODO assumes 1st order Lagrange

        // Assert that the local vertex index is valid.
        assert(local_vertex_index < NumElementVertices && "The local vertex index is invalid");

        assert(isLocalPointInRefElement(local_coordinates) && "Point is not in reference element");

        // Get the local vertex indices for the local vertex index.
        const index_vec_t local_vertex_indices =
            this->ref_element_m.getLocalVertices()[local_vertex_index];

        Vector<T, Dim> gradient(1);

        // To construct the gradient we need to loop over the dimensions and multiply the
        // shape functions in each dimension except the current one. The one of the current
        // dimension is replaced by the derivative of the shape function in that dimension,
        // which is either 1 or -1.
        for (std::size_t d = 0; d < Dim; d++) {
            // The variable that accumulates the product of the shape functions.
            T product = 1;

            for (std::size_t d2 = 0; d2 < Dim; d2++) {
                if (d2 == d) {
                    if (local_coordinates[d] < local_vertex_indices[d]) {
                        product *= 1;
                    } else {
                        product *= -1;
                    }
                } else {
                    if (local_coordinates[d2] < local_vertex_indices[d2]) {
                        product *= local_coordinates[d2];
                    } else {
                        product *= 1.0 - local_coordinates[d2];
                    }
                }
            }

            gradient[d] = product;
        }

        return gradient;
    }

}  // namespace ippl
