
namespace ippl {
    // LagrangeSpace constructor, which calls the FiniteElementSpace constructor.
    template <typename T, unsigned Dim, unsigned Order, unsigned NumElementVertices,
              unsigned NumIntegrationPoints>
    LagrangeSpace<T, Dim, Order, NumElementVertices, NumIntegrationPoints>::LagrangeSpace(
        const Mesh<T, Dim>& mesh, const Element<T, Dim, NumElementVertices>& ref_element,
        const Quadrature<T, NumIntegrationPoints>& quadrature)
        : FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints,
                             calculateLagrangeNumDoFs(Dim, Order)>(mesh, ref_element, quadrature) {
        // Assert that the dimension is either 1, 2 or 3.
        static_assert(Dim >= 1 && Dim <= 3,
                      "Finite Element space only supports 1D, 2D and 3D meshes");
    }

    template <typename T, unsigned Dim, unsigned Order, unsigned NumElementVertices,
              unsigned NumIntegrationPoints, unsigned NumGlobalDOFs>
    void LagrangeSpace<T, Dim, Order, NumElementVertices, NumIntegrationPoints>::evaluateAx(
        const Vector<T, NumGlobalDOFs>& x, Vector<T, NumGlobalDOFs>& z) const {
        const std::size_t NumElementDOFs = this->NumElementDOFs;

        const std::size_t NumElementIntegrationPoints =
            this->quadrature_m.numElementIntegrationPoints();

        // List of quadrature weights
        auto w = this->quadrature_m.getWeightsForRefElement();

        // List of quadrature nodes
        auto q = this->quadrature_m.getIntegrationNodesForRefElement();

        // Inverse Transpose Transformation Jacobian
        Vector<T, Dim> DPhiInvT = this->ref_element_m.getInverseTransposeTransformationJacobian(
            this->getElementMeshVertices(0));

        // Absolute value of det Phi_K
        T absDetDPhi = std::abs(this->ref_element_m.getDeterminantOfTransformationJacobian(
            this->getElementMeshVertices(0)));

        // Gradients of the basis functions for the DOF at the quadrature nodes
        Vector<Vector<gradient_vec_t, NumElementDOFs>, NumIntegrationPoints> grad_b_q;
        for (index_t k = 0; k < NumIntegrationPoints; ++k) {
            for (index_t i = 0; i < NumElementDOFs; ++i) {
                grad_b_q[k][i] = this->evaluateRefElementBasisGradient(i, q[k]);
            }
        }

        // Allocate memory for the element matrix
        Vector<Vector<T, NumElementDOFs> NumElementDOFs> A_K;

        for (index_t element_index = 0; element_index < this->numElements; ++element_index) {
            Vector<index_t, NumElementDOFs> global_dofs;
            Vector<index_t, NumElementDOFs> local_dofs;

            // 1. Compute the Galerkin element matrix A_K
            for (index_t i = 0; i < NumElementDOFs; ++i) {
                for (index_t j = 0; j < NumElementDOFs; ++j) {
                    A_k[i][j] = 0.0;

                    for (index_t k = 0; k < NumElementIntegrationPoints; ++k) {
                        A_K[i][j] += w[k]
                                     * ((DPhiInvT * grad_b_q[k][j]) * (DPhiInvT * grad_b_q[k][i]))
                                     * absDetDPhi;
                    }
                }
            }

            // 2. Compute the contribution to z = Ax with A_K
            for (index_t i = 0; i < NumElementDOFs; ++i) {
                index_t I = global_dofs[i];
                for (index_t j = 0; j < NumElementDOFs; ++j) {
                    index_t J = global_dofs[j];

                    z[i] += A_K[i][j] * x[j];
                }
            }
        }
    }

    template <typename T, unsigned Dim, unsigned Order, unsigned NumElementVertices,
              unsigned NumIntegrationPoints, unsigned NumGlobalDOFs>
    void LagrangeSpace<T, Dim, Order, NumElementVertices, NumIntegrationPoints>::evaluateLoadVector(
        Vector<T, NumGlobalDOFs>& b) const {
        assert(b.dim > 0);  // TODO change assert to be correct
        // TODO implement
    }

    template <typename T, unsigned Dim, unsigned Order, unsigned NumElementVertices,
              unsigned NumIntegrationPoints>
    NDIndex<Dim>
    LagrangeSpace<T, Dim, Order, NumElementVertices, NumIntegrationPoints>::makeNDIndex(
        const Vector<T, Dim>& indices) const {
        // Not sure if this is the best way, but the getVertexPosition function expects an
        // NDIndex, with the vertex index used being the first in the NDIndex. No other index is
        // used, so we can just set the first and the last to the index we actually want.
        NDIndex<Dim> nd_index;
        for (unsigned d = 0; d < Dim; ++d) {
            nd_index[d] = Index(indices[d], indices[d]);
        }
        return nd_index;
    }

    template <typename T, unsigned Dim, unsigned Order, unsigned NumElementVertices,
              unsigned NumIntegrationPoints>
    LagrangeSpace<T, Dim, Order, NumElementVertices, NumIntegrationPoints>::point_t
    LagrangeSpace<T, Dim, Order, NumElementVertices, NumIntegrationPoints>::getCoordinatesForDof(
        const LagrangeSpace<T, Dim, Order, NumElementVertices,
                            NumIntegrationPoints>::global_dof_index_t& dof_index) const {
        return this->mesh_m.getVertexPosition(makeNDIndex(dof_index));
    }

    template <typename T, unsigned Dim, unsigned Order, unsigned NumElementVertices,
              unsigned NumIntegrationPoints>
    LagrangeSpace<T, Dim, Order, NumElementVertices, NumIntegrationPoints>::point_t
    LagrangeSpace<T, Dim, Order, NumElementVertices, NumIntegrationPoints>::getCoordinatesForDof(
        const LagrangeSpace<T, Dim, Order, NumElementVertices,
                            NumIntegrationPoints>::global_dof_index_t& global_vertex_index) const {
        const mesh_vertex_pos_t vertex_indices =
            getMeshVertexPositionFromIndex(global_vertex_index);
        return getCoordinatesForDof(vertex_indices);
    }

    template <typename T, unsigned Dim, unsigned Order, unsigned NumElementVertices,
              unsigned NumIntegrationPoints>
    T LagrangeSpace<T, Dim, Order, NumElementVertices, NumIntegrationPoints>::
        evaluateRefElementBasis(
            const LagrangeSpace<T, Dim, Order, NumElementVertices,
                                NumIntegrationPoints>::local_dof_index_t& local_vertex_index,
            const LagrangeSpace<T, Dim, Order, NumElementVertices,
                                NumIntegrationPoints>::local_point_t& local_coordinates) const {
        // Assert that the local vertex index is valid.
        assert(local_vertex_index < NumElementVertices
               && "The local vertex index is invalid");  // TODO assumes 1st order Lagrange

        assert(ref_element_m.isPointInRefElement(local_coordinates)
               && "Point is not in reference element");

        // Get the local vertex indices for the local vertex index.
        const mesh_vertex_pos_t local_vertex_indices =
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
    LagrangeSpace<T, Dim, Order, NumElementVertices, NumIntegrationPoints>::
        evaluateRefElementBasisGradient(
            const LagrangeSpace<T, Dim, Order, NumElementVertices,
                                NumIntegrationPoints>::local_dof_index_t& local_vertex_index,
            const LagrangeSpace<T, Dim, Order, NumElementVertices,
                                NumIntegrationPoints>::local_point_t& local_coordinates) const {
        // TODO assumes 1st order Lagrange

        // Assert that the local vertex index is valid.
        assert(local_vertex_index < NumElementVertices && "The local vertex index is invalid");

        assert(ref_element_m.isPointInRefElement(local_coordinates)
               && "Point is not in reference element");

        // Get the local vertex indices for the local vertex index.
        const mesh_vertex_pos_t local_vertex_indices =
            this->ref_element_m.getLocalVertices()[local_vertex_index];

        gradient_vec_t gradient(1);

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
