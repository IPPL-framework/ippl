
namespace ippl {
    template <typename T, unsigned Dim, unsigned NumElementDOFs, typename QuadratureType>
    FiniteElementSpace<T, Dim, NumElementDOFs, QuadratureType>::FiniteElementSpace(
        const Mesh<T, Dim>& mesh,
        const FiniteElementSpace<T, Dim, NumElementDOFs, QuadratureType>::ElementType& ref_element,
        const QuadratureType& quadrature)
        : mesh_m(mesh)
        , ref_element_m(ref_element)
        , quadrature_m(quadrature) {
        assert(mesh.Dimension == Dim && "Mesh dimension does not match the dimension of the space");

        //
        for (std::size_t d = 0; d < Dim; ++d) {
            assert(mesh.getGridsize(d) > 1 && "Mesh has no cells in at least one dimension");
        }
    }
    template <typename T, unsigned Dim, unsigned NumElementDOFs, typename QuadratureType>
    std::size_t FiniteElementSpace<T, Dim, NumElementDOFs, QuadratureType>::numElements() const {
        Vector<std::size_t, Dim> cells_per_dim = this->mesh_m.getGridsize() - 1u;

        std::size_t num_elements = 1;
        for (std::size_t d = 0; d < Dim; ++d) {
            num_elements *= cells_per_dim[d];
        }

        return num_elements;
    }

    template <typename T, unsigned Dim, unsigned NumElementDOFs, typename QuadratureType>
    std::size_t FiniteElementSpace<T, Dim, NumElementDOFs, QuadratureType>::numElementsInDim(
        const FiniteElementSpace<T, Dim, NumElementDOFs, QuadratureType>::index_t& dim) const {
        return this->mesh_m.getGridsize(dim) - 1u;
    }

    template <typename T, unsigned Dim, unsigned NumElementDOFs, typename QuadratureType>
    FiniteElementSpace<T, Dim, NumElementDOFs, QuadratureType>::ndindex_t
    FiniteElementSpace<T, Dim, NumElementDOFs, QuadratureType>::getMeshVertexNDIndex(
        const FiniteElementSpace<T, Dim, NumElementDOFs, QuadratureType>::index_t& vertex_index)
        const {
        // Copy the vertex index to the index variable we can alter during the computation.
        index_t index = vertex_index;

        // Create a vector to store the vertex indices in each dimension for the corresponding
        // vertex.
        ndindex_t vertex_indices;

        // This is the number of vertices in each dimension.
        Vector<std::size_t, Dim> vertices_per_dim = this->mesh_m.getGridsize();

        // The number_of_lower_dim_vertices is the product of the number of vertices per
        // dimension, it will get divided by the current dimensions number to get the index in
        // that dimension
        std::size_t remaining_number_of_vertices = 1;
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

    template <typename T, unsigned Dim, unsigned NumElementDOFs, typename QuadratureType>
    FiniteElementSpace<T, Dim, NumElementDOFs, QuadratureType>::index_t
    FiniteElementSpace<T, Dim, NumElementDOFs, QuadratureType>::getMeshVertexIndex(
        const FiniteElementSpace<T, Dim, NumElementDOFs, QuadratureType>::ndindex_t& vertexNDIndex)
        const {
        const auto meshSizes = this->mesh_m.getGridsize();

        // Compute the vector to multiply the ndindex with
        ippl::Vector<std::size_t, Dim> vec(1);
        for (std::size_t d = 1; d < dim; ++d) {
            for (std::size_t d2 = d; d2 < Dim; ++d2) {
                vec[d2] *= meshSizes[d - 1];
            }
        }

        // return the dot product between the vertex ndindex and vec.
        return vertexNDIndex.dot(vec);
    }

    // implementation of function to retrieve the index of an element in each dimension
    template <typename T, unsigned Dim, unsigned NumElementDOFs, typename QuadratureType>
    FiniteElementSpace<T, Dim, NumElementDOFs, QuadratureType>::ndindex_t
    FiniteElementSpace<T, Dim, NumElementDOFs, QuadratureType>::getElementNDIndex(
        const FiniteElementSpace<T, Dim, NumElementDOFs, QuadratureType>::index_t& element_index)
        const {
        // Copy the element index to the index variable we can alter during the computation.
        index_t index = element_index;

        // Create a vector to store the element indices in each dimension for the corresponding
        // element.
        ndindex_t element_nd_index;

        // This is the number of cells in each dimension. It is one less than the number of
        // vertices in each dimension, which is returned by Mesh::getGridsize().
        Vector<std::size_t, Dim> cells_per_dim = this->mesh_m.getGridsize() - 1;

        // The number_of_lower_dim_cells is the product of all the number of cells per
        // dimension, it will get divided by the current dimension's size to get the index in
        // that dimension
        std::size_t remaining_number_of_cells = 1;
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

    template <typename T, unsigned Dim, unsigned NumElementDOFs, typename QuadratureType>
    FiniteElementSpace<T, Dim, NumElementDOFs, QuadratureType>::mesh_element_vertex_index_vec_t
    FiniteElementSpace<T, Dim, NumElementDOFs, QuadratureType>::getElementMeshVertexIndices(
        const FiniteElementSpace<T, Dim, NumElementDOFs, QuadratureType>::ndindex_t&
            element_nd_index) const {
        const Vector<std::size_t, Dim> num_vertices = this->mesh_m.getGridsize();

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
            }
        }

        return vertex_indices;
    }

    template <typename T, unsigned Dim, unsigned NumElementDOFs, typename QuadratureType>
    FiniteElementSpace<T, Dim, NumElementDOFs, QuadratureType>::mesh_element_vertex_ndindex_vec_t
    FiniteElementSpace<T, Dim, NumElementDOFs, QuadratureType>::getElementMeshVertexNDIndices(
        const FiniteElementSpace<T, Dim, NumElementDOFs, QuadratureType>::ndindex_t& elementNDIndex)
        const {
        mesh_element_vertex_ndindex_vec_t vertex_nd_indices;

        ndindex_t smallest_vertex_nd_index = elementNDIndex;

        // vertex_nd_indices[0] = smallest_vertex_nd_index;
        // vertex_nd_indices[1] = smallest_vertex_nd_index;
        // vertex_nd_indices[1][0] += 1;

        // vertex_nd_indices[2] = vertex_nd_indices[0];
        // vertex_nd_indices[2][1] += 1;
        // vertex_nd_indices[3] = vertex_nd_indices[1];
        // vertex_nd_indices[3][1] += 1;

        // vertex_nd_indices[4] = vertex_nd_indices[0];
        // vertex_nd_indices[4][2] += 1;
        // vertex_nd_indices[5] = vertex_nd_indices[1];
        // vertex_nd_indices[5][2] += 1;
        // vertex_nd_indices[6] = vertex_nd_indices[2];
        // vertex_nd_indices[6][2] += 1;
        // vertex_nd_indices[7] = vertex_nd_indices[3];
        // vertex_nd_indices[7][2] += 1;

        for (std::size_t i = 0; i < (1 << Dim); ++i) {
            vertex_nd_indices[i] = smallest_vertex_nd_index;
            for (std::size_t j = 0; j < Dim; ++j) {
                vertex_nd_indices[i][j] += (i >> j) & 1;
            }
        }

        return vertex_nd_indices;
    }

    template <typename T, unsigned Dim, unsigned NumElementDOFs, typename QuadratureType>
    FiniteElementSpace<T, Dim, NumElementDOFs, QuadratureType>::mesh_element_vertex_point_vec_t
    FiniteElementSpace<T, Dim, NumElementDOFs, QuadratureType>::getElementMeshVertexPoints(
        const FiniteElementSpace<T, Dim, NumElementDOFs, QuadratureType>::ndindex_t& elementNDIndex)
        const {
        mesh_element_vertex_point_vec_t vertex_points;

        // get all the NDIndices for the vertices of this element
        mesh_element_vertex_ndindex_vec_t vertex_nd_indices =
            this->getElementMeshVertexNDIndices(elementNDIndex);

        // get the coordinates of the vertices of this element
        for (std::size_t i = 0; i < vertex_nd_indices.dim; ++i) {
            NDIndex<Dim> temp_ndindex;
            for (std::size_t d = 0; d < Dim; ++d) {
                temp_ndindex[d] = Index(vertex_nd_indices[i][d], vertex_nd_indices[i][d]);
            }
            vertex_points[i] = this->mesh_m.getVertexPosition(temp_ndindex);
        }

        return vertex_points;
    }

}  // namespace ippl