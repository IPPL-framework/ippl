
namespace ippl {
    template <typename T, unsigned Dim, unsigned NumElementDOFs, unsigned NumGlobalDOFs,
              typename QuadratureType>
    FiniteElementSpace<T, Dim, NumElementDOFs, NumGlobalDOFs, QuadratureType>::FiniteElementSpace(
        const Mesh<T, Dim>& mesh,
        const FiniteElementSpace<T, Dim, NumElementDOFs, NumGlobalDOFs,
                                 QuadratureType>::ElementType& ref_element,
        const QuadratureType& quadrature)
        : mesh_m(mesh)
        , ref_element_m(ref_element)
        , quadrature_m(quadrature) {
        assert(mesh.Dimension == Dim);
    }

    template <typename T, unsigned Dim, unsigned NumElementDOFs, unsigned NumGlobalDOFs,
              typename QuadratureType>
    std::size_t
    FiniteElementSpace<T, Dim, NumElementDOFs, NumGlobalDOFs, QuadratureType>::numElements() const {
        Vector<std::size_t, Dim> cells_per_dim = this->mesh_m.getGridsize() - 1u;

        // TODO Use a reduction instead
        std::size_t num_elements = 1;
        for (const std::size_t num_cells : cells_per_dim) {
            num_elements *= num_cells;
        }

        return num_elements;
    }

    template <typename T, unsigned Dim, unsigned NumElementDOFs, unsigned NumGlobalDOFs,
              typename QuadratureType>
    std::size_t
    FiniteElementSpace<T, Dim, NumElementDOFs, NumGlobalDOFs, QuadratureType>::numElementsInDim(
        const FiniteElementSpace<T, Dim, NumElementDOFs, NumGlobalDOFs, QuadratureType>::index_t&
            dim) const {
        return this->mesh_m.getGridsize(dim) - 1u;
    }

    template <typename T, unsigned Dim, unsigned NumElementDOFs, unsigned NumGlobalDOFs,
              typename QuadratureType>
    FiniteElementSpace<T, Dim, NumElementDOFs, NumGlobalDOFs, QuadratureType>::nd_index_t
    FiniteElementSpace<T, Dim, NumElementDOFs, NumGlobalDOFs, QuadratureType>::getMeshVertexNDIndex(
        const FiniteElementSpace<T, Dim, NumElementDOFs, NumGlobalDOFs, QuadratureType>::index_t&
            vertex_index) const {
        // Copy the vertex index to the index variable we can alter during the computation.
        index_t index = vertex_index;

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

    template <typename T, unsigned Dim, unsigned NumElementDOFs, unsigned NumGlobalDOFs,
              typename QuadratureType>
    FiniteElementSpace<T, Dim, NumElementDOFs, NumGlobalDOFs, QuadratureType>::index_t
    FiniteElementSpace<T, Dim, NumElementDOFs, NumGlobalDOFs, QuadratureType>::getMeshVertexIndex(
        const FiniteElementSpace<T, Dim, NumElementDOFs, NumGlobalDOFs, QuadratureType>::nd_index_t&
            vertex_nd_index) const {
        std::size_t vertex_index = 0;

        const Vector<std::size_t, Dim> num_vertices = this->mesh_m.getGridsize();

        std::size_t temp_size;
        for (int d = Dim - 1; d >= 0; --d) {
            temp_size = vertex_nd_index[d];
            for (int i = d; i >= 1; --i) {
                temp_size *= num_vertices[i];
            }
            vertex_index += temp_size;
        }

        return vertex_index;
    }

    // implementation of function to retrieve the index of an element in each dimension
    template <typename T, unsigned Dim, unsigned NumElementDOFs, unsigned NumGlobalDOFs,
              typename QuadratureType>
    FiniteElementSpace<T, Dim, NumElementDOFs, NumGlobalDOFs, QuadratureType>::nd_index_t
    FiniteElementSpace<T, Dim, NumElementDOFs, NumGlobalDOFs, QuadratureType>::getElementNDIndex(
        const FiniteElementSpace<T, Dim, NumElementDOFs, NumGlobalDOFs, QuadratureType>::index_t&
            element_index) const {
        // Copy the element index to the index variable we can alter during the computation.
        index_t index = element_index;

        // Create a vector to store the element indices in each dimension for the corresponding
        // element.
        nd_index_t element_nd_index;

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
            element_nd_index[d] = index / remaining_number_of_cells;
            index -= element_nd_index[d] * remaining_number_of_cells;
        }

        return element_nd_index;
    }

    template <typename T, unsigned Dim, unsigned NumElementDOFs, unsigned NumGlobalDOFs,
              typename QuadratureType>
    FiniteElementSpace<T, Dim, NumElementDOFs, NumGlobalDOFs,
                       QuadratureType>::mesh_element_vertex_index_vec_t
    FiniteElementSpace<T, Dim, NumElementDOFs, NumGlobalDOFs, QuadratureType>::
        getElementMeshVertexIndices(
            const FiniteElementSpace<T, Dim, NumElementDOFs, NumGlobalDOFs,
                                     QuadratureType>::nd_index_t& element_nd_index) const {
        const Vector<std::size_t, Dim> num_vertices = this->mesh_m.getGridsize();

        // TODO maybe move into function that gets the vertex index from the vertex nd_index
        std::size_t smallest_vertex_index = 0;
        for (int d = Dim - 1; d >= 0; --d) {
            std::size_t temp_index = element_nd_index[d];
            for (int i = d; i >= 1; --i) {
                temp_index *= num_vertices[i];
            }
            smallest_vertex_index += temp_index;
        }

        // Vector to store the vertex indices for the element
        mesh_element_vertex_index_vec_t vertex_indices;
        vertex_indices[0] = smallest_vertex_index;
        vertex_indices[1] = vertex_indices[0] + 1;

        // vertex_indices[2] = vertex_indices[0] + num_vertices[0];
        // vertex_indices[3] = vertex_indices[1] + num_vertices[0];

        // vertex_indices[4] = vertex_indices[0] + (num_vertices[0] * num_vertices[1]);
        // vertex_indices[5] = vertex_indices[1] + (num_vertices[0] * num_vertices[1]);
        // vertex_indices[6] = vertex_indices[2] + (num_vertices[0] * num_vertices[1]);
        // vertex_indices[7] = vertex_indices[3] + (num_vertices[0] * num_vertices[1]);

        // ...

        for (std::size_t d = 1; d < Dim; ++d) {
            for (std::size_t i = 0; i < static_cast<unsigned>(1 << d); ++i) {
                std::size_t size = 1;
                for (std::size_t j = 0; j < d; ++j) {
                    size *= num_vertices[j];
                }
                vertex_indices[i + (1 << d)] = vertex_indices[i] + size;
                std::cout << "vertex_indices[" << i + (1 << d)
                          << "] = " << vertex_indices[i + (1 << d)] << "\n";
            }
        }

        return vertex_indices;
    }

    template <typename T, unsigned Dim, unsigned NumElementDOFs, unsigned NumGlobalDOFs,
              typename QuadratureType>
    FiniteElementSpace<T, Dim, NumElementDOFs, NumGlobalDOFs,
                       QuadratureType>::mesh_element_vertex_point_vec_t
    FiniteElementSpace<T, Dim, NumElementDOFs, NumGlobalDOFs, QuadratureType>::
        getElementMeshVertexPoints(
            const FiniteElementSpace<T, Dim, NumElementDOFs, NumGlobalDOFs,
                                     QuadratureType>::nd_index_t& elementNDIndex) const {
        assert(elementNDIndex.dim == Dim);

        // Vector to store the vertex points for the element
        mesh_element_vertex_point_vec_t vertex_points(0);

        return vertex_points;
    }

}  // namespace ippl