
namespace ippl {
    template <typename T, unsigned Dim, unsigned NumElementDOFs, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    FiniteElementSpace<T, Dim, NumElementDOFs, ElementType, QuadratureType, FieldLHS,
                       FieldRHS>::FiniteElementSpace(UniformCartesian<T, Dim>& mesh,
                                                     ElementType& ref_element,
                                                     const QuadratureType& quadrature)
        : mesh_m(mesh)
        , ref_element_m(ref_element)
        , quadrature_m(quadrature) {
        assert(mesh.Dimension == Dim && "Mesh dimension does not match the dimension of the space");

        nr_m     = mesh_m.getGridsize();
        hr_m     = mesh_m.getMeshSpacing();
        origin_m = mesh_m.getOrigin();

        /*for (size_t d = 0; d < Dim; ++d) {
            assert(nr_m[d] > 1 && "Mesh has no cells in at least one dimension");
        }*/
    }

    template <typename T, unsigned Dim, unsigned NumElementDOFs, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    void FiniteElementSpace<T, Dim, NumElementDOFs, ElementType, QuadratureType, FieldLHS,
                       FieldRHS>::setMesh(UniformCartesian<T, Dim>& mesh)
    {
        assert(mesh.Dimension == Dim && "Mesh dimension does not match the dimension of the space");

        mesh_m = mesh;

        nr_m     = mesh_m.getGridsize();
        hr_m     = mesh_m.getMeshSpacing();
        origin_m = mesh_m.getOrigin();

        for (size_t d = 0; d < Dim; ++d) {
            assert(nr_m[d] > 1 && "Mesh has no cells in at least one dimension");
        }
    }

    template <typename T, unsigned Dim, unsigned NumElementDOFs, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION size_t FiniteElementSpace<T, Dim, NumElementDOFs, ElementType, QuadratureType,
                                              FieldLHS, FieldRHS>::numElements() const {
        Vector<size_t, Dim> cells_per_dim = nr_m - 1u;

        size_t num_elements = 1;
        for (size_t d = 0; d < Dim; ++d) {
            num_elements *= cells_per_dim[d];
        }

        return num_elements;
    }

    template <typename T, unsigned Dim, unsigned NumElementDOFs, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION size_t
    FiniteElementSpace<T, Dim, NumElementDOFs, ElementType, QuadratureType, FieldLHS,
                       FieldRHS>::numElementsInDim(const size_t& dim) const {
        return nr_m[dim] - 1u;
    }

    template <typename T, unsigned Dim, unsigned NumElementDOFs, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION typename FiniteElementSpace<T, Dim, NumElementDOFs, ElementType, QuadratureType,
                                                FieldLHS, FieldRHS>::indices_t
    FiniteElementSpace<T, Dim, NumElementDOFs, ElementType, QuadratureType, FieldLHS,
                       FieldRHS>::getMeshVertexNDIndex(const size_t& vertex_index) const {
        // Copy the vertex index to the index variable we can alter during the computation.
        size_t index = vertex_index;

        // Create a vector to store the vertex indices in each dimension for the corresponding
        // vertex.
        indices_t vertex_indices;

        // This is the number of vertices in each dimension.
        Vector<size_t, Dim> vertices_per_dim = nr_m;

        // The number_of_lower_dim_vertices is the product of the number of vertices per
        // dimension, it will get divided by the current dimensions number to get the index in
        // that dimension
        size_t remaining_number_of_vertices = 1;
        for (const size_t num_vertices : vertices_per_dim) {
            remaining_number_of_vertices *= num_vertices;
        }

        for (int d = Dim - 1; d >= 0; --d) {
            remaining_number_of_vertices /= vertices_per_dim[d];
            vertex_indices[d] = index / remaining_number_of_vertices;
            index -= vertex_indices[d] * remaining_number_of_vertices;
        }

        return vertex_indices;
    };

    template <typename T, unsigned Dim, unsigned NumElementDOFs, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION size_t
    FiniteElementSpace<T, Dim, NumElementDOFs, ElementType, QuadratureType, FieldLHS, FieldRHS>::
        getMeshVertexIndex(
            const FiniteElementSpace<T, Dim, NumElementDOFs, ElementType, QuadratureType, FieldLHS,
                                     FieldRHS>::indices_t& vertexNDIndex) const {
        // Compute the vector to multiply the ndindex with
        ippl::Vector<size_t, Dim> vec(1);
        for (size_t d = 1; d < dim; ++d) {
            for (size_t d2 = d; d2 < Dim; ++d2) {
                vec[d2] *= nr_m[d - 1];
            }
        }

        // return the dot product between the vertex ndindex and vec.
        return vertexNDIndex.dot(vec);
    }

    // implementation of function to retrieve the index of an element in each dimension
    template <typename T, unsigned Dim, unsigned NumElementDOFs, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION typename FiniteElementSpace<T, Dim, NumElementDOFs, ElementType, QuadratureType,
                                                FieldLHS, FieldRHS>::indices_t
    FiniteElementSpace<T, Dim, NumElementDOFs, ElementType, QuadratureType, FieldLHS,
                       FieldRHS>::getElementNDIndex(const size_t& element_index) const {
        // Copy the element index to the index variable we can alter during the computation.
        size_t index = element_index;

        // Create a vector to store the element indices in each dimension for the corresponding
        // element.
        indices_t element_nd_index;

        // This is the number of cells in each dimension. It is one less than the number of
        // vertices in each dimension, which is in nr_m (mesh.getGridsize()).
        Vector<size_t, Dim> cells_per_dim = nr_m - 1;

        // The number_of_lower_dim_cells is the product of all the number of cells per
        // dimension, it will get divided by the current dimension's size to get the index in
        // that dimension
        size_t remaining_number_of_cells = 1;
        for (const size_t num_cells : cells_per_dim) {
            remaining_number_of_cells *= num_cells;
        }

        for (int d = Dim - 1; d >= 0; --d) {
            remaining_number_of_cells /= cells_per_dim[d];
            element_nd_index[d] = (index / remaining_number_of_cells);
            index -= (element_nd_index[d]) * remaining_number_of_cells;
        }

        return element_nd_index;
    }

    // implementation of function to retrieve the global index of an element given the ndindex
    template <typename T, unsigned Dim, unsigned NumElementDOFs, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION size_t
    FiniteElementSpace<T, Dim, NumElementDOFs, ElementType, QuadratureType, FieldLHS, FieldRHS>::
        getElementIndex(
            const FiniteElementSpace<T, Dim, NumElementDOFs, ElementType, QuadratureType, FieldLHS,
                                     FieldRHS>::indices_t& ndindex) const {
        size_t element_index = 0;

        // This is the number of cells in each dimension. It is one less than the number of
        // vertices in each dimension, which is returned by Mesh::getGridsize().
        Vector<size_t, Dim> cells_per_dim = nr_m - 1;

        size_t remaining_number_of_cells = 1;

        for (unsigned int d = 0; d < Dim; ++d) {
            element_index += ndindex[d] * remaining_number_of_cells;
            remaining_number_of_cells *= cells_per_dim[d];
        }

        return element_index;
    }

    template <typename T, unsigned Dim, unsigned NumElementDOFs, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION typename FiniteElementSpace<T, Dim, NumElementDOFs, ElementType, QuadratureType,
                                                FieldLHS, FieldRHS>::vertex_indices_t
    FiniteElementSpace<T, Dim, NumElementDOFs, ElementType, QuadratureType, FieldLHS, FieldRHS>::
        getElementMeshVertexIndices(
            const FiniteElementSpace<T, Dim, NumElementDOFs, ElementType, QuadratureType, FieldLHS,
                                     FieldRHS>::indices_t& element_nd_index) const {
        const Vector<size_t, Dim> num_vertices = nr_m;

        size_t smallest_vertex_index = 0;
        for (int d = Dim - 1; d >= 0; --d) {
            size_t temp_index = element_nd_index[d];
            for (int i = d; i >= 1; --i) {
                temp_index *= num_vertices[i];
            }
            smallest_vertex_index += temp_index;
        }

        // Vector to store the vertex indices for the element
        vertex_indices_t vertex_indices;
        vertex_indices[0] = smallest_vertex_index;
        vertex_indices[1] = vertex_indices[0] + 1;

        /*
        The following for loop computes the following computations:

        2D:
            vertex_indices[2] = vertex_indices[0] + num_vertices[0];
            vertex_indices[3] = vertex_indices[1] + num_vertices[0];
        3D:
            vertex_indices[4] = vertex_indices[0] + (num_vertices[0] * num_vertices[1]);
            vertex_indices[5] = vertex_indices[1] + (num_vertices[0] * num_vertices[1]);
            vertex_indices[6] = vertex_indices[2] + (num_vertices[0] * num_vertices[1]);
            vertex_indices[7] = vertex_indices[3] + (num_vertices[0] * num_vertices[1]);

        ...
        */

        for (size_t d = 1; d < Dim; ++d) {
            for (size_t i = 0; i < static_cast<unsigned>(1 << d); ++i) {
                size_t size = 1;
                for (size_t j = 0; j < d; ++j) {
                    size *= num_vertices[j];
                }
                vertex_indices[i + (1 << d)] = vertex_indices[i] + size;
            }
        }

        return vertex_indices;
    }

    template <typename T, unsigned Dim, unsigned NumElementDOFs, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION typename FiniteElementSpace<T, Dim, NumElementDOFs, ElementType, QuadratureType,
                                                FieldLHS, FieldRHS>::indices_list_t
    FiniteElementSpace<T, Dim, NumElementDOFs, ElementType, QuadratureType, FieldLHS, FieldRHS>::
        getElementMeshVertexNDIndices(
            const FiniteElementSpace<T, Dim, NumElementDOFs, ElementType, QuadratureType, FieldLHS,
                                     FieldRHS>::indices_t& elementNDIndex) const {
        indices_list_t vertex_nd_indices;

        indices_t smallest_vertex_nd_index = elementNDIndex;

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

        for (size_t i = 0; i < (1 << Dim); ++i) {
            vertex_nd_indices[i] = smallest_vertex_nd_index;
            for (size_t j = 0; j < Dim; ++j) {
                vertex_nd_indices[i][j] += (i >> j) & 1;
            }
        }

        return vertex_nd_indices;
    }

    template <typename T, unsigned Dim, unsigned NumElementDOFs, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION typename FiniteElementSpace<T, Dim, NumElementDOFs, ElementType, QuadratureType,
                                                FieldLHS, FieldRHS>::vertex_points_t
    FiniteElementSpace<T, Dim, NumElementDOFs, ElementType, QuadratureType, FieldLHS, FieldRHS>::
        getElementMeshVertexPoints(
            const FiniteElementSpace<T, Dim, NumElementDOFs, ElementType, QuadratureType, FieldLHS,
                                     FieldRHS>::indices_t& elementNDIndex) const {
        vertex_points_t vertex_points;

        // get all the NDIndices for the vertices of this element
        indices_list_t vertex_nd_indices = this->getElementMeshVertexNDIndices(elementNDIndex);

        // get the coordinates of the vertices of this element
        for (size_t i = 0; i < vertex_nd_indices.dim; ++i) {
            NDIndex<Dim> temp_ndindex;
            for (size_t d = 0; d < Dim; ++d) {
                temp_ndindex[d]     = Index(vertex_nd_indices[i][d], vertex_nd_indices[i][d]);
                vertex_points[i][d] = (temp_ndindex[d].first() * this->hr_m[d]) + this->origin_m[d];
            }
        }
        return vertex_points;
    }

}  // namespace ippl
